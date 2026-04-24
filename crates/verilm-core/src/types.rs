//! Core types for verifier keys, traces, and verification results.
//!
//! # Security model
//!
//! The protocol has two roles with a strict information boundary:
//!
//! **Prover** — runs inference, records intermediates, builds Merkle
//! commitments, opens challenged traces. Needs only the public model
//! weights. Has no access to the verifier key.
//!
//! **Verifier** — generates secret random vectors r_j, precomputes
//! v_j = r_j^T W_j from public weights, checks opened traces using
//! Freivalds (v · x == r · z), SiLU, chain, and Merkle proofs.
//!
//! The verifier key (`VerifierKey`) is entirely verifier-secret.
//! Since v = r^T W and the prover knows W, leaking any part of the
//! key (r or v) lets the prover forge valid-looking traces.

use serde::{Deserialize, Serialize};

use crate::constants::{MatrixType, ModelConfig};
use crate::field::Fp;
use crate::merkle::MerkleProof;

/// Token-identity acceptance mode for the lm_head decode check.
///
/// The lm_head Freivalds algebraic check (`final_hidden × lm_head == logits`)
/// is always exact. This enum controls only the final step: does the verifier
/// check that the committed token matches `argmax/sample(quantized_logits)`?
///
/// Empirical measurement on Qwen W8A8 showed that the i8→i32 quantized logit
/// path can diverge from the GPU's bf16 logit path by up to ~6500 i32 units
/// (on a ~7000 range), making any tolerance-based acceptance meaningless.
/// Bounded acceptance was tested and rejected — the gap tail is too heavy.
///
/// Future paths to exact token acceptance for currently-unsupported profiles:
/// captured float logit commitment, or deterministic lm_head kernels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DecodeAcceptanceMode {
    /// Exact: `argmax(quantized_logits) == token_id` (greedy) or
    /// `sample(quantized_logits, seed) == token_id` (sampled).
    /// The quantized replay path matches the GPU closely enough for
    /// the top token to be preserved.
    ExactTokenIdentity,

    /// The quantized replay path diverges too far from the GPU's logits
    /// to verify token identity. The lm_head Freivalds matmul check still
    /// runs exactly — only the token-identity step is skipped.
    Unsupported,

    /// Verify token identity via captured LP hidden state (bf16 matmul).
    ///
    /// The prover captures `LogitsProcessor args[1]` (the pruned hidden
    /// state at the decode boundary, shape `(1, hidden_dim)`, dtype bf16)
    /// and commits it in the Merkle leaf. The verifier reconstructs logits
    /// via `lp_hidden_bf16 @ lm_head_bf16.T` and checks argmax/sample.
    ///
    /// ~98.7% pass rate on Llama sampled (300 runs). Not exact — CPU replay
    /// arithmetic ≠ GPU tensor-core arithmetic. Superseded by CapturedLogits.
    LpHiddenBf16,

    /// Verify token identity from the actual GPU logits captured at inference.
    ///
    /// The prover captures the f32 logits that `CanonicalSamplerHook` sampled
    /// from and commits them in the Merkle leaf alongside `lp_hidden_bf16`.
    /// The verifier:
    ///   1. Checks `sample(captured_logits, seed) == token_id` — exact.
    ///   2. Freivalds-binds `captured_logits` to `lp_hidden × lm_head_bf16`
    ///      using a secret ±1 random projection (cheap, probabilistic).
    ///
    /// Solves both correctness (exact sampling from real GPU logits) and
    /// cost (no full lm_head matmul replay, just two dot products).
    CapturedLogits,
}

impl Default for DecodeAcceptanceMode {
    fn default() -> Self {
        DecodeAcceptanceMode::ExactTokenIdentity
    }
}

/// How the verifier checks the attention output `a` for the challenged token.
///
/// This is the phase-selection key in the canonical verifier's `run()`:
/// - `ExactReplay` → `phase_exact_attention` (f64 Q·K^T replay)
/// - `WitnessedScores` → `phase_witnessed_score_attention` (GPU scores)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AttentionVerificationMode {
    /// Replay Q·K^T/√d → softmax → V in f64, compare against committed `a`.
    /// Strongest claim: verifier independently recomputes the full attention.
    /// Requires that f64 replay matches GPU bf16 within the profile tolerance.
    ExactReplay,

    /// Use GPU-captured pre-softmax scores, anchored against canonical Q·K^T/√d.
    /// Replay softmax(witnessed) @ V → requantize, compare against committed `a`.
    /// Used when f64 replay diverges from GPU at later token positions.
    WitnessedScores,

    /// Audit-only: arbitrary-position attention outputs are NOT verified.
    /// The verifier instead audits attention inputs and wiring — score
    /// anchoring, KV provenance, mask/RoPE/GQA checks, and local replay
    /// smoke. The hot path reports status only; individual audits are
    /// wired in subsequent phases (see roadmap items 13–15).
    /// No kernel modification required — works with stock FlashAttention.
    AuditedInputsOnly,

    /// Deterministic attention kernel: exact match between prover and verifier.
    /// Requires custom CUDA kernel replacing FlashAttention.
    /// Future — not yet implemented.
    DeterministicKernel,
}

impl Default for AttentionVerificationMode {
    fn default() -> Self {
        AttentionVerificationMode::ExactReplay
    }
}

/// Verification profile: family-specific validated parameters.
///
/// Each supported model family gets a profile carrying empirically validated
/// tolerances, context limits, and audit policy. The profile is set during
/// keygen (inferred from `config.json` model_type) and stored in the verifier
/// key so the verifier enforces the correct policy without family-specific
/// code paths.
///
/// The core protocol, receipt format, and verifier architecture are identical
/// across profiles — only the numerical parameters differ.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationProfile {
    /// Profile identifier, e.g. "qwen-w8a8", "llama-w8a8".
    pub name: String,

    /// Model family, e.g. "qwen2", "llama".
    pub model_family: String,

    /// Maximum L-inf difference allowed for bridge x_attn check-and-gate
    /// (committed vs derived, per element in i8 space).
    pub bridge_tolerance: u8,

    /// Maximum L-inf difference allowed for attention replay comparison
    /// (claimed a_i8 vs replayed a_i8, per element in i8 space).
    /// Shared cross-family: validated at 10 on Qwen (max 8) and Llama (max 9).
    pub attention_tolerance: u8,

    /// Maximum context length for which the corridor bound has been
    /// empirically validated. Beyond this length, the tolerance may not hold.
    pub max_validated_context: u32,

    /// Whether this profile requires score anchoring for strong verification
    /// claims at long context (roadmap #8).
    pub requires_score_anchoring: bool,

    /// Maximum allowed score anchor gap (L-inf over all heads and positions)
    /// for witnessed-score replay. When `Some(t)`, the verifier computes
    /// canonical Q·K^T/√d from shell Q and committed K, then checks
    /// `|witnessed - canonical| < t`. If the anchor passes, witnessed scores
    /// are used for softmax + V aggregation (tighter corridor).
    /// When `None`, witnessed scores are ignored in verification.
    pub score_anchor_threshold: Option<f32>,

    /// Whether this profile supports exact QKV Freivalds checks.
    /// When false, the verifier skips Wq/Wk/Wv Freivalds (bridge replay
    /// cannot reproduce the GPU's quantized GEMM for this model family).
    /// Default: true (backward compatible — existing keys get Freivalds).
    #[serde(default = "default_true")]
    pub supports_qkv_freivalds: bool,

    /// How the verifier checks attention output for the challenged token.
    ///
    /// Selects between f64 Q·K^T exact replay (`ExactReplay`) and
    /// GPU-witnessed score replay (`WitnessedScores`). Independent of
    /// `score_anchor_threshold`, which parameterizes the witnessed path.
    ///
    /// Default: `ExactReplay` (backward compatible).
    #[serde(default)]
    pub attention_mode: AttentionVerificationMode,

    /// Token-identity acceptance mode for the lm_head decode check.
    ///
    /// Controls whether the verifier checks that `argmax(quantized_logits)`
    /// or `sample(quantized_logits, seed)` matches the committed token.
    /// The lm_head Freivalds matmul check is always exact regardless of
    /// this setting — only the final token-identity step is affected.
    ///
    /// Default: `ExactTokenIdentity` (backward compatible).
    #[serde(default)]
    pub decode_acceptance: DecodeAcceptanceMode,
}

fn default_true() -> bool {
    true
}

impl VerificationProfile {
    /// Qwen2 W8A8 profile, validated on Qwen2.5-7B-W8A8 (A100-80GB).
    ///
    /// Corridor: max L-inf=8, frac≤1 > 99.8%, stable across 6 workloads.
    /// Bridge: 1 bucket tolerance (BF16-vs-f64).
    pub fn qwen_w8a8() -> Self {
        VerificationProfile {
            name: "qwen-w8a8".into(),
            model_family: "qwen2".into(),
            bridge_tolerance: 1,
            attention_tolerance: 10,
            max_validated_context: 1164,
            requires_score_anchoring: false,
            score_anchor_threshold: Some(20.0), // f64 anchor gap ~14; bf16 TBD
            supports_qkv_freivalds: false,
            attention_mode: AttentionVerificationMode::WitnessedScores,
            decode_acceptance: DecodeAcceptanceMode::LpHiddenBf16, // i8→i32 logits diverge (6556 gap), but LP hidden bf16 matmul is 192/192 exact
        }
    }

    /// Llama W8A8 profile, validated on Llama-3.1-8B-W8A8 (H100).
    ///
    /// Corridor: max L-inf=9, frac≤1 > 99.9%, measured after rope_scaling fix.
    /// Slight growth with context (3 at short, 9 at 1165 tokens). Layer 25
    /// is consistently worst.
    /// Bridge: 1 bucket tolerance (BF16-vs-f64).
    /// Score anchoring: bf16 path gives ~0.06 gap (short/medium/long),
    /// validated with cutlass_epilogue_bf16 + f32 RoPE on H100.
    pub fn llama_w8a8() -> Self {
        VerificationProfile {
            name: "llama-w8a8".into(),
            model_family: "llama".into(),
            bridge_tolerance: 1,
            // Small corridor tolerance only. Large FlashAttention/f64 replay
            // gaps at non-zero token positions are not accepted here; they
            // require a kernel-aligned witness or deterministic attention kernel.
            attention_tolerance: 10,
            max_validated_context: 1165,
            requires_score_anchoring: true,
            score_anchor_threshold: Some(0.25), // bf16 anchor gap ~0.06, 4x headroom
            supports_qkv_freivalds: true,
            attention_mode: AttentionVerificationMode::ExactReplay,
            decode_acceptance: DecodeAcceptanceMode::ExactTokenIdentity,
        }
    }

    /// Llama W8A8 with LP-hidden bf16 decode (for sampled decode verification).
    /// Same as llama_w8a8 but uses LpHiddenBf16 instead of ExactTokenIdentity.
    pub fn llama_w8a8_lp_hidden() -> Self {
        VerificationProfile {
            name: "llama-w8a8-lp-hidden".into(),
            decode_acceptance: DecodeAcceptanceMode::LpHiddenBf16,
            ..Self::llama_w8a8()
        }
    }

    /// Llama W8A8 with captured-logits decode (exact sampled decode).
    pub fn llama_w8a8_captured_logits() -> Self {
        VerificationProfile {
            name: "llama-w8a8-captured-logits".into(),
            decode_acceptance: DecodeAcceptanceMode::CapturedLogits,
            ..Self::llama_w8a8()
        }
    }

    /// Qwen W8A8 with captured-logits decode (exact sampled decode).
    pub fn qwen_w8a8_captured_logits() -> Self {
        VerificationProfile {
            name: "qwen-w8a8-captured-logits".into(),
            decode_acceptance: DecodeAcceptanceMode::CapturedLogits,
            ..Self::qwen_w8a8()
        }
    }

    /// Llama W8A8 audit-only attention with captured-logits decode.
    pub fn llama_w8a8_audited() -> Self {
        VerificationProfile {
            name: "llama-w8a8-audited".into(),
            attention_mode: AttentionVerificationMode::AuditedInputsOnly,
            decode_acceptance: DecodeAcceptanceMode::CapturedLogits,
            // Attention tolerance is not used in AuditedInputsOnly mode — set high.
            attention_tolerance: 255,
            ..Self::llama_w8a8()
        }
    }

    /// Qwen W8A8 audit-only attention with captured-logits decode.
    pub fn qwen_w8a8_audited() -> Self {
        VerificationProfile {
            name: "qwen-w8a8-audited".into(),
            attention_mode: AttentionVerificationMode::AuditedInputsOnly,
            decode_acceptance: DecodeAcceptanceMode::CapturedLogits,
            attention_tolerance: 255,
            ..Self::qwen_w8a8()
        }
    }

    /// Detect profile from model family string (from config.json model_type).
    /// Returns None for unrecognized families.
    /// Defaults to audit-only attention profiles (product mode).
    pub fn detect(model_type: &str, quant_family: Option<&str>) -> Option<Self> {
        let is_w8a8 = quant_family.map_or(false, |q| q == "W8A8");
        let family = model_type.to_lowercase();
        if family.contains("qwen2") && is_w8a8 {
            Some(Self::qwen_w8a8_audited())
        } else if family.contains("llama") && is_w8a8 {
            Some(Self::llama_w8a8_audited())
        } else {
            None
        }
    }
}

/// Abstraction for accessing public INT8 weight matrices.
///
/// Used by the prover to compute shell openings at audit time,
/// and optionally by the verifier for debug/oracle replay.
pub trait ShellWeights {
    fn weight(&self, layer: usize, mt: MatrixType) -> &[i8];
}

/// Deployment configuration bound to a batch commitment.
/// Hashed and stored in `BatchCommitment.manifest_hash` to prevent
/// tokenizer substitution, sampling-parameter manipulation, and
/// system-prompt injection attacks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentManifest {
    /// SHA-256 of the full tokenizer identity (vocab + normalizer +
    /// pre-tokenizer + decoder + added tokens). Computed from the
    /// canonical tokenizer.json representation, normalized for determinism.
    pub tokenizer_hash: [u8; 32],
    /// Sampling temperature (0.0 = greedy).
    pub temperature: f32,
    /// Top-k sampling parameter (0 = disabled).
    pub top_k: u32,
    /// Top-p (nucleus) sampling parameter (1.0 = disabled).
    pub top_p: f32,
    /// EOS/stop policy identifier (e.g. "stop", "sample").
    pub eos_policy: String,
    /// Merkle root over all INT8 weight matrices (R_W).
    /// Binds the commitment to a specific quantized checkpoint.
    #[serde(default)]
    pub weight_hash: Option<[u8; 32]>,
    /// Hash of the quantization scheme (e.g. W8A8, Q8_0).
    /// Prevents substitution of a differently-quantized checkpoint.
    #[serde(default)]
    pub quant_hash: Option<[u8; 32]>,
    /// SHA-256 of the system prompt prepended to user input.
    /// Prevents system-prompt injection or substitution.
    #[serde(default)]
    pub system_prompt_hash: Option<[u8; 32]>,

    // --- Logit-modifying parameters (affect token selection) ---
    /// Repetition penalty: multiplicative factor on logits of previously generated tokens.
    /// 1.0 = disabled (no penalty).
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,
    /// Frequency penalty: additive penalty proportional to token frequency in output.
    /// 0.0 = disabled.
    #[serde(default)]
    pub frequency_penalty: f32,
    /// Presence penalty: additive penalty for any token that appeared in output.
    /// 0.0 = disabled.
    #[serde(default)]
    pub presence_penalty: f32,
    /// Logit bias: (token_id, additive_bias) pairs, sorted by token_id.
    /// Empty = disabled.
    #[serde(default)]
    pub logit_bias: Vec<(u32, f32)>,
    /// Bad-word token IDs: tokens that must never be sampled, sorted.
    /// Empty = disabled.
    #[serde(default)]
    pub bad_word_ids: Vec<u32>,
    /// Guided decoding grammar/schema identifier. Empty = disabled.
    /// When non-empty, constrains sampling to tokens valid under this grammar.
    #[serde(default)]
    pub guided_decoding: String,

    // --- Output-level parameters ---
    /// Stop sequences that terminate generation. Empty = rely on eos_policy only.
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    /// Maximum tokens to generate. 0 = no explicit limit.
    #[serde(default)]
    pub max_tokens: u32,

    // --- Four-spec fields (flow through to InputSpec/ModelSpec/DecodeSpec) ---
    /// SHA-256 of the chat template (Jinja2 or equivalent).
    #[serde(default)]
    pub chat_template_hash: Option<[u8; 32]>,
    /// SHA-256 of RoPE / positional encoding configuration.
    #[serde(default)]
    pub rope_config_hash: Option<[u8; 32]>,
    /// RMSNorm epsilon from model config (e.g. 1e-5 for Llama).
    #[serde(default)]
    pub rmsnorm_eps: Option<f64>,
    /// Canonical sampler version identifier (e.g. "chacha20-vi-sample-v1").
    #[serde(default)]
    pub sampler_version: Option<String>,
    /// Decode mode: explicit commitment to "greedy" or "sampled".
    /// When present, the verifier cross-checks against temperature.
    #[serde(default)]
    pub decode_mode: Option<String>,

    // --- Fields that flow through to InputSpec ---
    /// BOS/EOS preprocessing policy (e.g. "add_bos", "none").
    #[serde(default)]
    pub bos_eos_policy: Option<String>,
    /// Truncation policy (e.g. "left", "right", "error").
    #[serde(default)]
    pub truncation_policy: Option<String>,
    /// Special-token handling policy (e.g. "encode", "strip", "pass").
    #[serde(default)]
    pub special_token_policy: Option<String>,
    /// Padding policy (e.g. "right", "left", "none").
    /// Controls how inputs are padded to sequence length.
    #[serde(default)]
    pub padding_policy: Option<String>,

    // --- Fields that flow through to ModelSpec ---
    /// Hash of adapter / LoRA / merged-checkpoint identity.
    #[serde(default)]
    pub adapter_hash: Option<[u8; 32]>,

    // --- Architecture fields that flow through to ModelSpec ---
    /// Number of transformer layers.
    #[serde(default)]
    pub n_layers: Option<u32>,
    /// Hidden dimension (embedding width).
    #[serde(default)]
    pub hidden_dim: Option<u32>,
    /// Vocabulary size (number of logit entries).
    #[serde(default)]
    pub vocab_size: Option<u32>,
    /// Merkle root over embedding table rows.
    #[serde(default)]
    pub embedding_merkle_root: Option<[u8; 32]>,

    // --- Quantization fields that flow through to ModelSpec (#5-7) ---
    /// Quantization family label (e.g. "W8A8", "Q8_0", "AWQ", "GPTQ").
    #[serde(default)]
    pub quant_family: Option<String>,
    /// Scale derivation method (e.g. "absmax", "zeropoint", "group_absmax").
    #[serde(default)]
    pub scale_derivation: Option<String>,
    /// Block size for blockwise quantization schemes (e.g. 32 for Q8_0).
    #[serde(default)]
    pub quant_block_size: Option<u32>,

    // --- Attention runtime fields that flow through to ModelSpec ---
    /// Attention backend (e.g. "sdpa", "eager", "flash_attention_2").
    #[serde(default)]
    pub attn_backend: Option<String>,
    /// Effective dtype for attention computation (e.g. "float16", "bfloat16").
    #[serde(default)]
    pub attn_dtype: Option<String>,

    // --- Remaining architecture fields that flow through to ModelSpec (#8) ---
    /// KV dimension (n_kv_heads * d_head).
    #[serde(default)]
    pub kv_dim: Option<u32>,
    /// FFN intermediate dimension.
    #[serde(default)]
    pub ffn_dim: Option<u32>,
    /// Per-head dimension.
    #[serde(default)]
    pub d_head: Option<u32>,
    /// Number of query heads.
    #[serde(default)]
    pub n_q_heads: Option<u32>,
    /// Number of key-value heads (GQA).
    #[serde(default)]
    pub n_kv_heads: Option<u32>,
    /// RoPE base frequency (theta).
    #[serde(default)]
    pub rope_theta: Option<f64>,

    // --- Fields that flow through to OutputSpec ---
    /// Minimum tokens before EOS is allowed. 0 = no minimum.
    #[serde(default)]
    pub min_tokens: u32,
    /// Whether to ignore EOS tokens during generation.
    #[serde(default)]
    pub ignore_eos: bool,
    /// Detokenization policy (e.g. "default", "strip_special", "raw").
    #[serde(default)]
    pub detokenization_policy: Option<String>,
    /// EOS token ID for the model's tokenizer. Required for min_tokens and
    /// ignore_eos enforcement.
    #[serde(default)]
    pub eos_token_id: Option<u32>,
}

fn default_repetition_penalty() -> f32 {
    1.0
}

impl DeploymentManifest {
    pub const DISABLED_REPETITION_PENALTY: f32 = 1.0;
    pub const DISABLED_FREQUENCY_PENALTY: f32 = 0.0;
    pub const DISABLED_PRESENCE_PENALTY: f32 = 0.0;

    /// Split the flat manifest into the four protocol specs.
    pub fn split(&self) -> (InputSpec, ModelSpec, DecodeSpec, OutputSpec) {
        (
            InputSpec::from(self),
            ModelSpec::from(self),
            DecodeSpec::from(self),
            OutputSpec::from(self),
        )
    }
}

// ===========================================================================
// Four-spec commitment types (paper §3: M = H(H_input || H_model || H_decode || H_output))
// ===========================================================================

/// Preprocessing semantics: tokenizer, chat template, system prompt.
///
/// Binds the input pipeline so that the verifier can confirm the prover
/// used the expected tokenizer and prompt preprocessing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputSpec {
    /// SHA-256 of the full tokenizer identity (vocab + normalizer +
    /// pre-tokenizer + decoder + added tokens). Computed from the
    /// canonical tokenizer.json representation, normalized for determinism.
    pub tokenizer_hash: [u8; 32],
    /// SHA-256 of the system prompt prepended to user input.
    #[serde(default)]
    pub system_prompt_hash: Option<[u8; 32]>,
    /// SHA-256 of the chat template (Jinja2 or equivalent).
    #[serde(default)]
    pub chat_template_hash: Option<[u8; 32]>,
    /// BOS/EOS preprocessing policy (e.g. "add_bos", "none").
    /// Controls whether BOS/EOS tokens are prepended/appended to the input.
    #[serde(default)]
    pub bos_eos_policy: Option<String>,
    /// Truncation policy (e.g. "left", "right", "error").
    /// Controls behavior when input exceeds the model's context window.
    #[serde(default)]
    pub truncation_policy: Option<String>,
    /// Special-token handling policy (e.g. "encode", "strip", "pass").
    /// Controls how special tokens in user input are processed.
    #[serde(default)]
    pub special_token_policy: Option<String>,
    /// Padding policy (e.g. "right", "left", "none").
    /// Controls how inputs are padded to sequence length.
    #[serde(default)]
    pub padding_policy: Option<String>,
}

impl From<&DeploymentManifest> for InputSpec {
    fn from(m: &DeploymentManifest) -> Self {
        InputSpec {
            tokenizer_hash: m.tokenizer_hash,
            system_prompt_hash: m.system_prompt_hash,
            chat_template_hash: m.chat_template_hash,
            bos_eos_policy: m.bos_eos_policy.clone(),
            truncation_policy: m.truncation_policy.clone(),
            special_token_policy: m.special_token_policy.clone(),
            padding_policy: m.padding_policy.clone(),
        }
    }
}

/// Model identity and architecture configuration.
///
/// Binds the checkpoint, quantization scheme, and architecture-affecting
/// knobs so the verifier can confirm the prover used the correct model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    /// Merkle root over all INT8 weight matrices (R_W).
    #[serde(default)]
    pub weight_hash: Option<[u8; 32]>,
    /// Hash of the quantization scheme (e.g. W8A8, Q8_0).
    #[serde(default)]
    pub quant_hash: Option<[u8; 32]>,
    /// Hash of RoPE / positional encoding configuration.
    #[serde(default)]
    pub rope_config_hash: Option<[u8; 32]>,
    /// RMSNorm epsilon (architecture constant, e.g. 1e-5 for Llama).
    #[serde(default)]
    pub rmsnorm_eps: Option<f64>,
    /// Hash of adapter / LoRA / merged-checkpoint identity.
    #[serde(default)]
    pub adapter_hash: Option<[u8; 32]>,
    /// Number of transformer layers.
    #[serde(default)]
    pub n_layers: Option<u32>,
    /// Hidden dimension (embedding width).
    #[serde(default)]
    pub hidden_dim: Option<u32>,
    /// Vocabulary size (number of logit entries).
    #[serde(default)]
    pub vocab_size: Option<u32>,
    /// Merkle root over embedding table rows.
    #[serde(default)]
    pub embedding_merkle_root: Option<[u8; 32]>,

    // --- Quantization identity (#5-7) ---
    /// Quantization family label (e.g. "W8A8", "Q8_0", "AWQ", "GPTQ").
    #[serde(default)]
    pub quant_family: Option<String>,
    /// Scale derivation method (e.g. "absmax", "zeropoint", "group_absmax").
    #[serde(default)]
    pub scale_derivation: Option<String>,
    /// Block size for blockwise quantization schemes (e.g. 32 for Q8_0).
    #[serde(default)]
    pub quant_block_size: Option<u32>,

    // --- Attention runtime semantics ---
    /// Attention backend (e.g. "sdpa", "eager", "flash_attention_2").
    /// Binds the provider's attention implementation so the verifier can
    /// confirm arithmetic-path alignment. W8A8 models MUST use "sdpa"
    /// (eager produces incorrect outputs with compressed_tensors).
    #[serde(default)]
    pub attn_backend: Option<String>,
    /// Effective dtype used for attention computation (e.g. "float16", "bfloat16").
    #[serde(default)]
    pub attn_dtype: Option<String>,

    // --- Remaining architecture knobs (#8) ---
    /// KV dimension (n_kv_heads * d_head).
    #[serde(default)]
    pub kv_dim: Option<u32>,
    /// FFN intermediate dimension.
    #[serde(default)]
    pub ffn_dim: Option<u32>,
    /// Per-head dimension.
    #[serde(default)]
    pub d_head: Option<u32>,
    /// Number of query heads.
    #[serde(default)]
    pub n_q_heads: Option<u32>,
    /// Number of key-value heads (GQA).
    #[serde(default)]
    pub n_kv_heads: Option<u32>,
    /// RoPE base frequency (theta).
    #[serde(default)]
    pub rope_theta: Option<f64>,
}

impl From<&DeploymentManifest> for ModelSpec {
    fn from(m: &DeploymentManifest) -> Self {
        ModelSpec {
            weight_hash: m.weight_hash,
            quant_hash: m.quant_hash,
            rope_config_hash: m.rope_config_hash,
            rmsnorm_eps: m.rmsnorm_eps,
            adapter_hash: m.adapter_hash,
            n_layers: m.n_layers,
            hidden_dim: m.hidden_dim,
            vocab_size: m.vocab_size,
            embedding_merkle_root: m.embedding_merkle_root,
            quant_family: m.quant_family.clone(),
            scale_derivation: m.scale_derivation.clone(),
            quant_block_size: m.quant_block_size,
            attn_backend: m.attn_backend.clone(),
            attn_dtype: m.attn_dtype.clone(),
            kv_dim: m.kv_dim,
            ffn_dim: m.ffn_dim,
            d_head: m.d_head,
            n_q_heads: m.n_q_heads,
            n_kv_heads: m.n_kv_heads,
            rope_theta: m.rope_theta,
        }
    }
}

/// Decode policy: sampling algorithm and parameters.
///
/// Binds the entire sampling pipeline so the verifier can exactly
/// replay token selection from verified logits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodeSpec {
    /// Sampling temperature (0.0 = greedy).
    pub temperature: f32,
    /// Top-k sampling parameter (0 = disabled).
    pub top_k: u32,
    /// Top-p (nucleus) sampling parameter (1.0 = disabled).
    pub top_p: f32,
    /// Repetition penalty: multiplicative factor on logits of previously generated tokens.
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,
    /// Frequency penalty: additive penalty proportional to token frequency.
    #[serde(default)]
    pub frequency_penalty: f32,
    /// Presence penalty: additive penalty for any token that appeared.
    #[serde(default)]
    pub presence_penalty: f32,
    /// Logit bias: (token_id, additive_bias) pairs, sorted by token_id.
    #[serde(default)]
    pub logit_bias: Vec<(u32, f32)>,
    /// Bad-word token IDs: tokens that must never be sampled, sorted.
    #[serde(default)]
    pub bad_word_ids: Vec<u32>,
    /// Guided decoding grammar/schema identifier.
    #[serde(default)]
    pub guided_decoding: String,
    /// Canonical sampler version identifier (e.g. "chacha20-v1").
    #[serde(default)]
    pub sampler_version: Option<String>,
    /// Decode mode: explicit commitment to "greedy" or "sampled".
    #[serde(default)]
    pub decode_mode: Option<String>,
}

impl From<&DeploymentManifest> for DecodeSpec {
    fn from(m: &DeploymentManifest) -> Self {
        DecodeSpec {
            temperature: m.temperature,
            top_k: m.top_k,
            top_p: m.top_p,
            repetition_penalty: m.repetition_penalty,
            frequency_penalty: m.frequency_penalty,
            presence_penalty: m.presence_penalty,
            logit_bias: m.logit_bias.clone(),
            bad_word_ids: m.bad_word_ids.clone(),
            guided_decoding: m.guided_decoding.clone(),
            sampler_version: m.sampler_version.clone(),
            decode_mode: m.decode_mode.clone(),
        }
    }
}

/// Output policy: termination and post-processing rules.
///
/// Binds the output pipeline so the verifier can confirm generation
/// termination conditions and text post-processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputSpec {
    /// EOS/stop policy identifier (e.g. "stop", "sample").
    pub eos_policy: String,
    /// Stop sequences that terminate generation.
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    /// Maximum tokens to generate. 0 = no explicit limit.
    #[serde(default)]
    pub max_tokens: u32,
    /// Minimum tokens before EOS is allowed. 0 = no minimum.
    #[serde(default)]
    pub min_tokens: u32,
    /// Whether to ignore EOS tokens during generation.
    #[serde(default)]
    pub ignore_eos: bool,
    /// Detokenization policy (e.g. "default", "strip_special", "raw").
    /// Controls how token IDs are converted to final output text.
    #[serde(default)]
    pub detokenization_policy: Option<String>,
    /// EOS token ID for the model's tokenizer.
    /// Required for min_tokens and ignore_eos enforcement.
    #[serde(default)]
    pub eos_token_id: Option<u32>,
}

impl From<&DeploymentManifest> for OutputSpec {
    fn from(m: &DeploymentManifest) -> Self {
        OutputSpec {
            eos_policy: m.eos_policy.clone(),
            stop_sequences: m.stop_sequences.clone(),
            max_tokens: m.max_tokens,
            min_tokens: m.min_tokens,
            ignore_eos: m.ignore_eos,
            detokenization_policy: m.detokenization_policy.clone(),
            eos_token_id: m.eos_token_id,
        }
    }
}

/// VERIFIER-SECRET material. Must never be sent to the prover.
///
/// Contains random vectors `r_j` and precomputed `v_j = r_j^T W_j`.
/// The Freivalds check `v · x == r · z` is sound only if the prover
/// does not know `r`. Since the prover knows `W` (open weights),
/// leaking `v = r^T W` is equivalent to leaking `r` (solvable linear
/// system). Therefore the entire key must stay verifier-side.
///
/// The prover needs nothing from the verifier — it runs honest inference,
/// records intermediates, and builds Merkle commitments. The verifier
/// generates this key from its own copy of the public weights.
///
/// # Security invariant
///
/// **If the prover ever sees this key (or any of its fields), the
/// Freivalds verification is broken** — the prover can forge outputs
/// that pass the check without running correct inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifierKey {
    pub version: u8,
    pub config: ModelConfig,
    pub seed: [u8; 32],

    /// Source dtype of the weights used to generate this key.
    /// "I8" = weights were already INT8 (exact).
    /// "BF16" / "F16" = weights were quantized to INT8 via absmax during keygen.
    pub source_dtype: String,

    /// Per-tensor absmax quantization scales, if source was BF16/FP16.
    /// Outer: layer. Inner: MatrixType::PER_LAYER order.
    /// Empty if source was already INT8.
    pub quantization_scales: Vec<Vec<f32>>,

    /// SECRET: Per-matrix-type random vectors r_j.
    /// Index: MatrixType::ALL order (8 entries: Wq..Wd + LmHead).
    /// Must never be revealed to the prover.
    pub r_vectors: Vec<Vec<Fp>>,

    /// SECRET: Precomputed v_j^(i) = r_j^T W_j^(i) for per-layer matrices.
    /// Outer index: layer. Inner index: MatrixType::PER_LAYER order (7 entries).
    /// Leaking v is equivalent to leaking r (prover knows W).
    pub v_vectors: Vec<Vec<Vec<Fp>>>,

    /// Per-layer infinity-norm of W_o (max absolute row sum, scaled).
    pub wo_norms: Vec<f32>,

    /// Max L-infinity norm of V head vectors across all layers and heads.
    pub max_v_norm: f32,

    /// Unembedding matrix (lm_head) for logit verification.
    /// Shape: (vocab_size, hidden_dim), row-major INT8.
    /// When present, the verifier recomputes logits from final_hidden
    /// instead of trusting the margin certificate's self-reported logits.
    pub lm_head: Option<Vec<i8>>,

    /// SECRET: Precomputed v_lm = r_lm^T @ lm_head.
    /// Length = hidden_dim (input dimension of lm_head).
    /// Global (not per-layer). Use `r_for(MatrixType::LmHead)` for the r vector.
    pub v_lm_head: Option<Vec<Fp>>,

    /// SECRET: Precomputed v = r^T @ lm_head_bf16 in f64 for CapturedLogits
    /// Freivalds binding. Length = hidden_dim. The random ±1 vector r is
    /// re-derived from `captured_logits_freivalds_seed` at verify time.
    #[serde(default)]
    pub v_lm_head_f64: Option<Vec<f64>>,

    /// SECRET: Seed for deriving the ±1 random vector used in the
    /// CapturedLogits Freivalds binding. Re-derived at verify time
    /// to compute r · captured_logits for the check.
    #[serde(default)]
    pub captured_logits_freivalds_seed: Option<[u8; 32]>,

    /// SHA-256 hash of the decode artifact containing lm_head bf16 weights.
    /// The decode artifact is a separate binary object (VDEC magic + bincode)
    /// that carries the full unembedding matrix for LP hidden verification.
    /// Keeping it out of the verifier key avoids a 1+ GB key for large models.
    #[serde(default)]
    pub lm_head_bf16_hash: Option<[u8; 32]>,

    /// SHA-256 hash of all INT8 weights in canonical order (weight chain).
    /// Binds the verifier key to a specific quantized checkpoint, ensuring
    /// the Freivalds precomputation corresponds to the published model.
    pub weight_hash: Option<[u8; 32]>,

    /// Per-layer RMSNorm weight vectors for attention input normalization.
    /// `rmsnorm_attn_weights[layer]` has length `hidden_dim`.
    /// Empty for toy model (no RMSNorm in simplified chain).
    pub rmsnorm_attn_weights: Vec<Vec<f32>>,

    /// Per-layer RMSNorm weight vectors for FFN input normalization.
    /// `rmsnorm_ffn_weights[layer]` has length `hidden_dim`.
    /// Empty for toy model (no RMSNorm in simplified chain).
    pub rmsnorm_ffn_weights: Vec<Vec<f32>>,

    /// Per-layer weight scales for each matrix type.
    /// `weight_scales[layer][matrix_type_idx]` is the per-tensor absmax scale.
    /// Used for dequantizing i32 accumulators to f32 in the real requantization
    /// bridge: `output_f32 = acc_i32 * scale_w * scale_x`.
    /// Empty for toy model (unit scale) or native INT8 weights.
    pub weight_scales: Vec<Vec<f32>>,

    /// Per-channel weight scales for native INT8 (W8A8) models.
    /// `per_channel_weight_scales[layer][matrix_type_idx]` is a Vec<f32> of
    /// length `output_dim` for that matrix type. Each channel has its own scale:
    /// `output_f64[i] = acc_i32[i] * scale_w[i] * scale_x`.
    ///
    /// Populated during keygen when `weight_scale` tensors are found alongside
    /// INT8 weight tensors. Empty for keygen-quantized (BF16/FP16→INT8) models
    /// and for toy models.
    #[serde(default)]
    pub per_channel_weight_scales: Vec<Vec<Vec<f32>>>,

    /// Per-layer QKV projection biases (model-dependent, e.g. Qwen2).
    /// `qkv_biases[layer]` = `[q_bias, k_bias, v_bias]` where each is `Vec<f32>`.
    /// Empty if the model has no QKV biases.
    #[serde(default)]
    pub qkv_biases: Vec<[Vec<f32>; 3]>,

    /// RMSNorm epsilon (e.g. 1e-5 for Llama-family models).
    pub rmsnorm_eps: f64,

    /// RoPE configuration hash. When present, cross-checked against the
    /// manifest's `rope_config_hash` to ensure the verifier key was
    /// generated for the same positional encoding configuration.
    #[serde(default)]
    pub rope_config_hash: Option<[u8; 32]>,

    /// Merkle root over embedding table rows.
    /// Each leaf = `hash_embedding_row(row_f32)`. Verifier checks prover's
    /// `initial_residual` against this root via `ShellTokenOpening.embedding_proof`.
    pub embedding_merkle_root: Option<[u8; 32]>,

    /// Final RMSNorm weight vector (`model.norm.weight`), applied after the
    /// last transformer layer and before lm_head. Length = `hidden_dim`.
    /// `None` for toy model (simplified bridge uses clamp-requantize).
    pub final_norm_weights: Option<Vec<f32>>,

    /// Quantization family label (e.g. "W8A8", "Q8_0", "AWQ", "GPTQ").
    /// Cross-checked against the manifest's `quant_family` when both are present.
    #[serde(default)]
    pub quant_family: Option<String>,
    /// Scale derivation method (e.g. "absmax", "zeropoint", "group_absmax").
    /// Cross-checked against the manifest's `scale_derivation` when both are present.
    #[serde(default)]
    pub scale_derivation: Option<String>,
    /// Block size for blockwise quantization schemes (e.g. 32 for Q8_0).
    /// Cross-checked against the manifest's `quant_block_size` when both are present.
    #[serde(default)]
    pub quant_block_size: Option<u32>,
    /// Expected attention backend (e.g. "sdpa", "eager").
    /// Cross-checked against the manifest's `attn_backend` when both are present.
    #[serde(default)]
    pub attn_backend: Option<String>,
    /// Expected attention compute dtype (e.g. "float16", "bfloat16").
    /// Cross-checked against the manifest's `attn_dtype` when both are present.
    #[serde(default)]
    pub attn_dtype: Option<String>,

    /// Per-layer, per-head o_proj sensitivity coefficient for attention certification.
    ///
    /// `o_proj_alpha[layer][head]` = max_j sum_{i in head_slice(h)} |W_o[j,i]|
    ///
    /// This bounds the worst-case residual-stream perturbation from a unit
    /// attention-output perturbation in head h. Used by the adaptive attention
    /// certifier to convert per-head attention-space tail bounds into
    /// residual-space bounds: eps_resid[l,h] = alpha[l,h] * eps_attn[l,h].
    ///
    /// Precomputed during keygen. Empty for toy model.
    #[serde(default)]
    pub o_proj_alpha: Vec<Vec<f64>>,

    /// When true, attention replay dequantizes Q/K accumulators using weight
    /// and activation scales, applies RoPE at each token position, and replays
    /// in f64 space. When false (default), uses raw `requantize` (toy/reference
    /// path where the forward pass does not apply RoPE).
    #[serde(default)]
    pub rope_aware_replay: bool,

    /// Family-specific verification profile with validated tolerances.
    /// Set during keygen from `config.json` model_type.
    /// When `None`, the verifier falls back to hardcoded defaults.
    #[serde(default)]
    pub verification_profile: Option<VerificationProfile>,
}

/// Separate artifact containing the bf16 unembedding matrix for LP hidden
/// decode verification. Kept out of `VerifierKey` because it can be 1+ GB
/// for large vocabularies. Content-addressed by SHA-256 hash stored in
/// `VerifierKey.lm_head_bf16_hash`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodeArtifact {
    /// bf16 bit patterns of the lm_head weight matrix, row-major.
    /// Shape: (vocab_size, hidden_dim). Length = vocab_size * hidden_dim.
    pub lm_head_bf16: Vec<u16>,
    pub vocab_size: usize,
    pub hidden_dim: usize,
}

impl DecodeArtifact {
    /// SHA-256 content hash for binding to the verifier key.
    pub fn content_hash(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(b"vi-decode-artifact-v1");
        hasher.update(self.vocab_size.to_le_bytes());
        hasher.update(self.hidden_dim.to_le_bytes());
        // Hash raw bytes of the u16 array for efficiency.
        let byte_slice = unsafe {
            std::slice::from_raw_parts(
                self.lm_head_bf16.as_ptr() as *const u8,
                self.lm_head_bf16.len() * 2,
            )
        };
        hasher.update(byte_slice);
        hasher.finalize().into()
    }
}

impl VerifierKey {
    /// Bridge x_attn tolerance from the verification profile.
    /// Falls back to 0 for toy models, 1 for production without a profile.
    pub fn bridge_tolerance(&self) -> u8 {
        if let Some(ref profile) = self.verification_profile {
            return profile.bridge_tolerance;
        }
        // Toy model: no rmsnorm weights or weight scales → exact match.
        if self.rmsnorm_attn_weights.is_empty() || self.weight_scales.is_empty() {
            return 0;
        }
        1
    }

    /// Attention replay tolerance from the verification profile.
    /// Falls back to 0 (exact match, appropriate for toy model).
    pub fn attention_tolerance(&self) -> u8 {
        if let Some(ref profile) = self.verification_profile {
            return profile.attention_tolerance;
        }
        0
    }

    /// Score anchor threshold from the verification profile.
    /// Returns `None` when no profile is set or the profile has no threshold
    /// (witnessed scores are not used in verification).
    pub fn score_anchor_threshold(&self) -> Option<f32> {
        self.verification_profile
            .as_ref()
            .and_then(|p| p.score_anchor_threshold)
    }

    /// Attention verification mode from the verification profile.
    /// Defaults to `ExactReplay` when no profile is set.
    pub fn attention_mode(&self) -> AttentionVerificationMode {
        self.verification_profile
            .as_ref()
            .map(|p| p.attention_mode.clone())
            .unwrap_or_default()
    }
    /// Random vector r for any matrix type (including LmHead).
    pub fn r_for(&self, mt: MatrixType) -> &[Fp] {
        let idx = MatrixType::ALL.iter().position(|&m| m == mt).unwrap();
        &self.r_vectors[idx]
    }

    /// Precomputed v = r^T W for a per-layer matrix.
    /// Panics on LmHead — use `v_lm_head` for the global unembedding matrix.
    pub fn v_for(&self, layer: usize, mt: MatrixType) -> &[Fp] {
        let idx = MatrixType::PER_LAYER
            .iter()
            .position(|&m| m == mt)
            .expect("v_for: use v_lm_head for LmHead");
        &self.v_vectors[layer][idx]
    }

    /// Get the per-tensor weight quantization scale for a given layer and matrix type.
    /// Returns 0.0 if weight_scales is empty (native INT8 / toy model).
    ///
    /// For native INT8 (W8A8) models, prefer [`per_channel_scales_for`] which
    /// returns per-channel scales when available.
    pub fn weight_scale_for(&self, layer: usize, mt: MatrixType) -> f32 {
        if self.weight_scales.is_empty() || layer >= self.weight_scales.len() {
            return 0.0;
        }
        let idx = MatrixType::PER_LAYER
            .iter()
            .position(|&m| m == mt)
            .expect("weight_scale_for: not applicable for LmHead");
        if idx >= self.weight_scales[layer].len() {
            return 0.0;
        }
        self.weight_scales[layer][idx]
    }

    /// Get per-channel weight scales for a given layer and matrix type.
    ///
    /// Returns `Some(&[f32])` of length `output_dim` if per-channel scales
    /// are available (W8A8 native INT8 models). Returns `None` if only
    /// per-tensor scales exist (keygen-quantized BF16/FP16) or no scales
    /// exist (toy model).
    pub fn per_channel_scales_for(&self, layer: usize, mt: MatrixType) -> Option<&[f32]> {
        if self.per_channel_weight_scales.is_empty()
            || layer >= self.per_channel_weight_scales.len()
        {
            return None;
        }
        let idx = MatrixType::PER_LAYER
            .iter()
            .position(|&m| m == mt)
            .expect("per_channel_scales_for: not applicable for LmHead");
        if idx >= self.per_channel_weight_scales[layer].len() {
            return None;
        }
        let scales = &self.per_channel_weight_scales[layer][idx];
        if scales.is_empty() {
            None
        } else {
            Some(scales)
        }
    }

    /// Returns true if this key has per-channel weight scales (W8A8 native INT8).
    pub fn has_per_channel_scales(&self) -> bool {
        !self.per_channel_weight_scales.is_empty()
    }

    /// Get the QKV projection bias for a given layer and matrix type.
    /// Returns `None` if the model has no QKV biases or the matrix type has no bias.
    pub fn qkv_bias_for(&self, layer: usize, mt: MatrixType) -> Option<&[f32]> {
        if self.qkv_biases.is_empty() || layer >= self.qkv_biases.len() {
            return None;
        }
        let idx = match mt {
            MatrixType::Wq => 0,
            MatrixType::Wk => 1,
            MatrixType::Wv => 2,
            _ => return None,
        };
        let bias = &self.qkv_biases[layer][idx];
        if bias.is_empty() {
            None
        } else {
            Some(bias)
        }
    }
}

/// One layer's worth of intermediates (full format, used for Merkle hashing).
///
/// Inputs (x_attn, a, x_ffn, h) are INT8 (post-requantization from previous step).
/// Outputs (q, k, v, attn_out, g, u, ffn_out) are i32 accumulators from matmul,
/// because the Freivalds check verifies the exact matrix product, not the
/// requantized result. The requantization step is verified separately.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerTrace {
    pub x_attn: Vec<i8>,    // input to attention block (INT8)
    pub q: Vec<i32>,        // W_q x (i32 accumulator)
    pub k: Vec<i32>,        // W_k x (i32 accumulator)
    pub v: Vec<i32>,        // W_v x (i32 accumulator)
    pub a: Vec<i8>,         // attention vector, input to W_o (INT8)
    pub attn_out: Vec<i32>, // W_o a (i32 accumulator)
    pub x_ffn: Vec<i8>,     // input to FFN block (INT8)
    pub g: Vec<i32>,        // W_g x_ffn (i32 accumulator)
    pub u: Vec<i32>,        // W_u x_ffn (i32 accumulator)
    pub h: Vec<i8>,         // SiLU(g) * u, requantized (INT8)
    pub ffn_out: Vec<i32>,  // W_d h (i32 accumulator)
    /// Requantized K vectors for all tokens in the KV cache at this layer
    /// (Level C only). Each inner Vec has length kv_dim. Empty for Level A/B.
    #[serde(default)]
    pub kv_cache_k: Vec<Vec<i8>>,
    /// Requantized V vectors for all tokens in the KV cache at this layer
    /// (Level C only). Each inner Vec has length kv_dim. Empty for Level A/B.
    #[serde(default)]
    pub kv_cache_v: Vec<Vec<i8>>,
    /// Per-tensor activation scale for x_attn (QKV projection input).
    /// `None` for toy model (unit scale / simplified clamp requantization).
    #[serde(default)]
    pub scale_x_attn: Option<f32>,
    /// Per-tensor activation scale for a (W_o projection input).
    #[serde(default)]
    pub scale_a: Option<f32>,
    /// Per-tensor activation scale for x_ffn (gate_up projection input).
    #[serde(default)]
    pub scale_x_ffn: Option<f32>,
    /// Per-tensor activation scale for h (down projection input).
    #[serde(default)]
    pub scale_h: Option<f32>,
    /// Pre-attention residual stream (f32).
    ///
    /// This is the layer input BEFORE RMSNorm — the actual residual connection
    /// value. For layer 0 it's the embedding output; for layer l>0 it's
    /// `residual[l-1] + dequant(attn_out[l-1]) + dequant(ffn_out[l-1])`.
    ///
    /// Required for paper-correct RMSNorm bridge verification:
    ///   `x_attn = quantize(RMSNorm_attn(residual), scale_x_attn)`
    ///   `x_ffn  = quantize(RMSNorm_ffn(residual + dequant(attn_out)), scale_x_ffn)`
    ///
    /// `None` for toy model (simplified clamp chain, no residual/RMSNorm).
    #[serde(default)]
    pub residual: Option<Vec<f32>>,
}

// ===========================================================================
// Minimal retained-state types (paper-minimal commit path)
// ===========================================================================
//
// The online commit path retains only the non-replayable intermediates.
// Everything else is reconstructed from public weights at audit time.

/// Minimal per-layer state retained for online commitment.
///
/// Per-layer retained state for online commitment and audit replay.
///
/// **Irreducible fields** (`a`, `scale_a`): depend on the full KV prefix
/// via softmax(QK^T/√d)V and cannot be reconstructed from single-token
/// local state + public weights alone.
///
/// Bridge replay scales (`scale_x_ffn`, `scale_h`) remain in
/// [`ShellLayerOpening`] — Freivalds implicitly verifies them at audit time.
///
/// `x_attn_i8` and `scale_x_attn` are committed here as the bridge trust
/// boundary: the verifier check-and-gates against canonical recomputation,
/// then continues from committed values to prevent error propagation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetainedLayerState {
    // --- Irreducible: depends on full KV prefix, not replayable ---
    /// Post-attention INT8 output fed into W_o. Comes from
    /// softmax(QK^T/√d)V which depends on the full KV prefix.
    pub a: Vec<i8>,
    /// Per-tensor activation scale for `a` (W_o projection input).
    /// Paired with the non-derivable `a` vector.
    pub scale_a: f32,

    // --- Bridge trust boundary: committed QKV input ---
    /// Committed quantized attention input: `quantize(rmsnorm(residual), scale_x_attn)`.
    /// When present, the verifier check-and-gates against its canonical derivation
    /// and uses this committed value for downstream QKV Freivalds checks.
    #[serde(default)]
    pub x_attn_i8: Option<Vec<i8>>,
    /// Committed activation scale paired with `x_attn_i8`.
    #[serde(default)]
    pub scale_x_attn: Option<f32>,
}

/// Minimal per-token state retained for online commitment.
///
/// No residuals — they are replayable from `a` + public weights recursively.
/// Prefix binding comes from the retained Merkle tree: the auditor opens
/// prior retained leaves to verify prefix state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetainedTokenState {
    pub layers: Vec<RetainedLayerState>,
}

// ===========================================================================
// KV transcript types (roadmap #3: committed causal KV)
// ===========================================================================

/// Post-RoPE K and dequantized V for one (layer, position) pair.
///
/// These are the exact values consumed by the attention dot product:
///   K_roped = RoPE(dequant(Wk @ x_attn))
///   V_deq   = dequant(Wv @ x_attn)
///
/// Committed per-layer under `kv_roots[layer]` so the verifier can replay
/// attention from committed values without re-deriving from weights.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvEntry {
    /// Post-RoPE K vector, length kv_dim. f64 for deterministic hashing.
    #[serde(with = "f64_hex_vec")]
    pub k_roped: Vec<f64>,
    /// Dequantized V vector (no RoPE), length kv_dim.
    #[serde(with = "f64_hex_vec")]
    pub v_deq: Vec<f64>,
}

/// Serde module that serializes `Vec<f64>` as hex-encoded IEEE 754 LE bytes.
///
/// serde_json's default f64 round-trip (Ryū → decimal → parse) introduces
/// ±1 ULP errors for some values (e.g. products of sin/cos from RoPE).
/// Hex encoding preserves bit-exact f64 values across JSON serialization.
///
/// JSON format: array of hex strings, e.g. `["3ff0000000000000", ...]`.
/// Bincode: transparent (delegates to default Vec<f64> encoding).
mod f64_hex_vec {
    use serde::{self, Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(data: &[f64], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if serializer.is_human_readable() {
            // JSON: encode each f64 as a hex string of its LE bytes
            let hex_strs: Vec<String> = data
                .iter()
                .map(|v| {
                    let bytes = v.to_le_bytes();
                    bytes.iter().map(|b| format!("{:02x}", b)).collect()
                })
                .collect();
            hex_strs.serialize(serializer)
        } else {
            // Bincode: use default Vec<f64> encoding (raw bytes, bit-exact)
            data.serialize(serializer)
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<f64>, D::Error>
    where
        D: Deserializer<'de>,
    {
        if deserializer.is_human_readable() {
            // JSON: decode hex strings back to f64
            let hex_strs: Vec<String> = Vec::deserialize(deserializer)?;
            hex_strs
                .iter()
                .map(|s| {
                    if s.len() != 16 {
                        return Err(serde::de::Error::custom(format!(
                            "expected 16 hex chars for f64, got {}",
                            s.len()
                        )));
                    }
                    let mut bytes = [0u8; 8];
                    for (i, chunk) in s.as_bytes().chunks(2).enumerate() {
                        let hex_str = std::str::from_utf8(chunk)
                            .map_err(serde::de::Error::custom)?;
                        bytes[i] = u8::from_str_radix(hex_str, 16)
                            .map_err(serde::de::Error::custom)?;
                    }
                    Ok(f64::from_le_bytes(bytes))
                })
                .collect()
        } else {
            // Bincode: default Vec<f64> decoding
            Vec::deserialize(deserializer)
        }
    }
}

/// Pre-softmax attention scores witnessed from GPU computation.
///
/// For one layer at one token position, contains the raw Q·K^T/√d scores
/// as computed by the GPU in fp16 (stored as f32 after D2H transfer).
/// Layout: row-major `scores[qh * seq_len + t]` for query head `qh`
/// attending to KV position `t`.
///
/// The verifier uses these instead of replaying Q·K^T, eliminating the
/// f64-vs-fp16 arithmetic mismatch in the score computation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WitnessedScores {
    /// Flat f32 scores, row-major: `scores[qh * seq_len + t]`.
    pub scores: Vec<f32>,
    /// Number of query heads (for reshaping).
    pub n_q_heads: usize,
    /// Sequence length (number of KV positions attended to).
    pub seq_len: usize,
}

/// Per-layer shell intermediates opened by the prover at audit time.
///
/// The prover reconstructs these from the committed retained `a` + weights.
/// The verifier checks each matmul with key-only Freivalds — no weights needed.
///
/// Bridge replay scales (`scale_x_attn`, `scale_x_ffn`, `scale_h`) are provided
/// here rather than in [`RetainedLayerState`]: Freivalds implicitly verifies them
/// (wrong scales -> wrong quantization -> Freivalds fails).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShellLayerOpening {
    /// W_o @ a (i32 accumulators). Input `a` is from retained state.
    pub attn_out: Vec<i32>,
    /// W_g @ x_ffn (i32). x_ffn = requantize(attn_out) — verifier derives this.
    pub g: Vec<i32>,
    /// W_u @ x_ffn (i32).
    pub u: Vec<i32>,
    /// W_d @ h (i32). h = SiLU(requant(g)) * requant(u) — verifier derives this.
    pub ffn_out: Vec<i32>,
    /// W_q @ x_attn (i32). Present for layers > 0 where x_attn is derivable.
    pub q: Option<Vec<i32>>,
    /// W_k @ x_attn (i32). Present for layers > 0.
    pub k: Option<Vec<i32>>,
    /// W_v @ x_attn (i32). Present for layers > 0.
    pub v: Option<Vec<i32>>,
    /// Bridge replay scale: QKV projection input quantization.
    /// Moved from RetainedLayerState — verified implicitly by Freivalds.
    #[serde(default = "default_scale")]
    pub scale_x_attn: f32,
    /// Bridge replay scale: gate_up projection input quantization.
    #[serde(default = "default_scale")]
    pub scale_x_ffn: f32,
    /// Bridge replay scale: down projection input quantization.
    #[serde(default = "default_scale")]
    pub scale_h: f32,
}

fn default_scale() -> f32 {
    1.0
}

/// Shell opening for a complete token: all layers' matmul intermediates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShellTokenOpening {
    pub layers: Vec<ShellLayerOpening>,
    /// Which layer indices are present in `layers`.
    /// `None` means all layers (full audit). When `Some`, `layers[i]` corresponds
    /// to `layer_indices[i]`. Must be a contiguous prefix 0..=L_max for bridge
    /// verification (sequential residual chain).
    #[serde(default)]
    pub layer_indices: Option<Vec<usize>>,
    /// Initial residual stream (embedding[token_id]) for full bridge verification.
    /// When present, enables residual-tracking RMSNorm bridge (paper-correct).
    /// `None` for toy model (simplified clamp bridge).
    #[serde(default)]
    pub initial_residual: Option<Vec<f32>>,
    /// Merkle proof binding `initial_residual` to the embedding table committed
    /// in the verifier key. `leaf_index` = token_id, leaf = `hash_embedding_row(initial_residual)`.
    #[serde(default)]
    pub embedding_proof: Option<MerkleProof>,
    /// Captured pre-final-norm residual stream from the actual GPU inference.
    /// When present, the verifier uses this for exact LM-head/token verification
    /// instead of the shell-replayed final hidden state (which diverges after
    /// many layers of approximate attention).
    #[serde(default)]
    pub final_residual: Option<Vec<f32>>,
    /// Prover-claimed LM-head logits: `lm_head @ quantize(final_norm(final_residual))`.
    /// The verifier checks this claim via Freivalds (v · fh == r · logits_i32)
    /// instead of recomputing the full matmul. Length = vocab_size.
    #[serde(default)]
    pub logits_i32: Option<Vec<i32>>,
    /// Captured LP hidden state at LogitsProcessor input (decode boundary).
    /// Each u16 is a bf16 bit pattern. Length = hidden_dim.
    /// The verifier reconstructs logits via `lp_hidden_bf16 @ lm_head_bf16.T`
    /// and checks argmax/sample for token identity.
    #[serde(default)]
    pub lp_hidden_bf16: Option<Vec<u16>>,
    /// Captured f32 logits from the GPU's actual LogitsProcessor output.
    /// Length = vocab_size. The verifier samples from these directly (exact)
    /// and Freivalds-binds them to `lp_hidden × lm_head_bf16`.
    #[serde(default)]
    pub captured_logits_f32: Option<Vec<f32>>,
}

/// Parameters for the full bridge computation (dequant → residual → RMSNorm → quantize).
///
/// Embedding table lookup for rich prefix opening.
///
/// When provided to the prover's `open_v4`, it loads embedding rows and Merkle
/// proofs for each prefix token, enabling embedding binding verification
/// on prefix tokens (not just the challenged token).
pub trait EmbeddingLookup {
    /// Load the embedding row for the given token ID and its Merkle proof.
    /// Returns `None` if embedding data is unavailable for this token.
    fn embedding_row_and_proof(
        &self,
        token_id: u32,
    ) -> Option<(Vec<f32>, Option<crate::merkle::MerkleProof>)>;
}

/// Contains the RMSNorm weights and initial residual needed for
/// paper-correct bridge verification. Weight scales are passed separately
/// (they're also used by the simplified bridge path).
///
/// When `None`, the simplified bridge is used (toy model / native INT8).
pub struct BridgeParams<'a> {
    /// Per-layer RMSNorm weight vectors for attention input normalization.
    pub rmsnorm_attn_weights: &'a [Vec<f32>],
    /// Per-layer RMSNorm weight vectors for FFN input normalization.
    pub rmsnorm_ffn_weights: &'a [Vec<f32>],
    /// RMSNorm epsilon.
    pub rmsnorm_eps: f64,
    /// Initial residual stream value (embedding[token_id], f32).
    pub initial_residual: &'a [f32],
    /// Embedding Merkle proof for this token. Forwarded into `ShellTokenOpening`.
    pub embedding_proof: Option<MerkleProof>,
}

/// Parameters for computing LM-head logits in the prover's opening.
///
/// When provided, the prover computes `logits_i32 = lm_head @ quantize(final_norm(final_residual))`
/// and includes the result in `ShellTokenOpening.logits_i32` for Freivalds verification.
pub struct TailParams<'a> {
    /// Unembedding matrix (lm_head), row-major INT8. Shape: (vocab_size, hidden_dim).
    pub lm_head: &'a [i8],
    /// Final RMSNorm weight vector (`model.norm.weight`). Length = hidden_dim.
    pub final_norm_weights: &'a [f32],
    /// RMSNorm epsilon.
    pub rmsnorm_eps: f64,
}

/// V4 audit response: opens retained leaves for challenged and prefix tokens,
/// plus prover-computed shell intermediates for the challenged token.
///
/// The verifier checks:
/// - Merkle proofs (retained state is committed)
/// - IO chain (splice resistance)
/// - Prompt/seed/manifest bindings
/// - Shell intermediates via key-only Freivalds + bridge checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct V4AuditResponse {
    /// Challenged token index.
    pub token_index: u32,
    /// Retained state for the challenged token.
    pub retained: RetainedTokenState,
    /// Merkle proof for the challenged token's retained leaf.
    pub merkle_proof: MerkleProof,
    /// IO chain proof for the challenged token.
    pub io_proof: MerkleProof,
    /// Token ID at this position.
    pub token_id: u32,
    /// Previous IO hash (for chain verification).
    pub prev_io_hash: [u8; 32],
    /// Leaf hashes for all prior tokens (prefix), ordered by position.
    /// Each entry is `hash_retained_state_direct(state)` — 32 bytes instead
    /// of the full `RetainedTokenState`, which is the major payload win.
    pub prefix_leaf_hashes: Vec<[u8; 32]>,
    /// Merkle proofs for each prefix token's retained leaf.
    pub prefix_merkle_proofs: Vec<MerkleProof>,
    /// Token IDs for prefix tokens.
    pub prefix_token_ids: Vec<u32>,
    /// Commitment this response was opened from.
    pub commitment: BatchCommitment,
    /// Revealed sampling seed.
    pub revealed_seed: [u8; 32],
    /// Prover-computed shell intermediates for the challenged token.
    /// The protocol verifier requires this; structural-only checks may omit.
    pub shell_opening: Option<ShellTokenOpening>,
    /// Deployment manifest included at audit time. The verifier checks
    /// `hash_manifest(manifest) == commitment.manifest_hash` and extracts
    /// decode parameters for sampling replay.
    #[serde(default)]
    pub manifest: Option<DeploymentManifest>,
    /// Raw prompt bytes for prompt hash verification.
    /// The verifier checks `hash_prompt(prompt) == commitment.prompt_hash`.
    #[serde(default)]
    pub prompt: Option<Vec<u8>>,
    /// Number of prompt tokens (full count including the first token used as
    /// embedding input). Prompt tokens occupy positions 0..n_prompt_tokens-1
    /// in the full token sequence; generated tokens start at n_prompt_tokens-1
    /// in the committed token_ids array (which omits the first token).
    #[serde(default)]
    pub n_prompt_tokens: Option<u32>,
    /// Claimed output text for detokenization verification.
    /// The verifier decodes the committed token IDs and compares against this text
    /// under the committed detokenization_policy.
    #[serde(default)]
    pub output_text: Option<String>,
    // ──── Rich prefix mode (optional) ────
    //
    // When present, the verifier can check embedding binding and (optionally)
    // full shell replay for prefix tokens, not just the challenged token.
    // These fields are populated when the audit challenge requests rich_prefix.
    /// Embedding rows for prefix tokens — `embedding[token_id]` as f32 for each
    /// prefix position. The verifier checks each row against the committed
    /// embedding Merkle root via the corresponding proof.
    #[serde(default)]
    pub prefix_embedding_rows: Option<Vec<Vec<f32>>>,
    /// Merkle proofs binding each prefix embedding row to the embedding table.
    #[serde(default)]
    pub prefix_embedding_proofs: Option<Vec<MerkleProof>>,
    /// Full `RetainedTokenState` for prefix tokens (deep audit).
    /// When present, the verifier can run shell replay + Freivalds on prefix tokens.
    /// Each entry must hash to the corresponding `prefix_leaf_hashes[j]`.
    #[serde(default)]
    pub prefix_retained: Option<Vec<RetainedTokenState>>,
    /// Shell openings for prefix tokens (deep audit).
    /// When present alongside `prefix_retained`, enables Freivalds checks
    /// on prefix tokens. Each entry includes matmul accumulators per layer.
    #[serde(default)]
    pub prefix_shell_openings: Option<Vec<ShellTokenOpening>>,

    // ──── Committed KV transcript (roadmap #3) ────
    /// Per-layer KV entries for the causal prefix of the challenged token.
    /// Outer index: challenged layer index. Inner: positions 0..=token_index.
    /// Only present for challenged layers when `kv_roots` is populated.
    #[serde(default)]
    pub kv_entries: Option<Vec<Vec<KvEntry>>>,

    /// Per-layer Merkle proofs for the KV entries, matching `kv_entries`.
    /// Each inner vec contains proofs for positions 0..=token_index,
    /// verifiable against `commitment.kv_roots[layer]`.
    #[serde(default)]
    pub kv_proofs: Option<Vec<Vec<MerkleProof>>>,

    // ──── Score witnessing (generated-token exact scores) ────
    /// Per-layer witnessed pre-softmax attention scores for the challenged token.
    /// When present, the verifier uses these for softmax + V aggregation instead
    /// of replaying Q·K^T/√d, eliminating the score arithmetic mismatch.
    /// Outer index: layer. Each entry covers all Q-heads × all KV positions.
    #[serde(default)]
    pub witnessed_scores: Option<Vec<WitnessedScores>>,
}

// ===========================================================================
// Q8_0 block-aware types
// ===========================================================================
//
// Q8_0 quantization uses blocks of 32 int8 values, each with its own f16
// scale factor (delta). A Q8_0 matrix-vector multiply computes:
//
//   output_f32[row] = Σ_b (d_w[row,b] · d_x[b] · sumi_b[row])
//
// where:
//   b       ∈ 0..B-1, B = input_dim / 32 (number of blocks)
//   d_w[row,b]  = f16 scale for block b of row `row` in the weight matrix (public)
//   d_x[b]      = f16 scale for block b of the input vector (in the trace)
//   sumi_b[row] = Σ_{j=0}^{31} w_int8[row,b,j] · x_int8[b,j]   (exact i32)
//
// The verification decomposes into two phases:
//
// **Phase A — Blockwise integer correctness (exact, in F_p)**
//
//   For each block b, sumi_b is the result of a (M × 32) matmul of int8 values.
//   This is standard Freivalds, just on smaller sub-matrices.
//
//   Key insight: the existing precomputed v = r^T W (length = input_dim) already
//   encodes per-block information. Slice v into blocks of 32:
//     v_b = v[b*32 .. (b+1)*32]
//
//   Batched check with random coefficients c_b (derived from the verifier's
//   secret key seed — the prover has no information about them):
//
//     Σ_b c_b · (v_b · x_b)  ==  r · (Σ_b c_b · sumi_b)    in F_p
//     ─── LHS (verifier) ───     ──── RHS (from trace) ────
//
//   This is a single Freivalds-style check, same cost as the flat case.
//   Soundness: a cheating prover that corrupts any block's sumi has
//   probability 1/p of passing (random c_b binds each block).
//
// **Phase B — Floating-point assembly (deterministic, from public scales)**
//
//   Given verified sumi_b values and public scales d_w, d_x, the verifier
//   recomputes the f32 output using canonical left-to-right accumulation:
//
//     output_f32[row] = Σ_{b=0}^{B-1} (f32(d_w[row,b]) · f32(d_x[b]) · f32(sumi_b[row]))
//
//   The prover sends the f32 output; the verifier checks it matches.
//   The canonical accumulation order makes this deterministic.
//
// **Chain integration**
//
//   The chain check uses the f32 output: the next layer's input is
//   RMSNorm(residual + output_f32), which the verifier can recompute.
//   For the simplified clamp-requantize chain used in testing:
//     x_ffn_i8 = clamp(round(output_f32), -128, 127)

/// Per-block accumulators for one Q8_0 matrix multiplication.
///
/// For a matrix W of shape (output_dim, input_dim) with B = input_dim / 32 blocks:
/// - `sumi[b]` has length `output_dim`
/// - `sumi[b][row] = Σ_{j=0}^{31} W_int8[row, b*32+j] · x_int8[b*32+j]`
///
/// Each sumi value is bounded: |sumi_b[row]| ≤ 32 · 127 · 127 = 516,382.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Q8BlockAccumulators {
    /// Number of blocks (= input_dim / 32).
    pub n_blocks: usize,
    /// Per-block partial i32 sums. sumi[b] has length output_dim.
    pub sumi: Vec<Vec<i32>>,
}

/// Per-block scale factors for a Q8_0 quantized vector.
///
/// One f32 scale per block of 32 elements. The actual Q8_0 format stores
/// f16 scales, but we convert to f32 for arithmetic (matching ggml's
/// GGML_CPU_FP16_TO_FP32 conversion).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Q8Scales {
    /// One scale factor per block. Length = ceil(vector_len / 32).
    pub scales: Vec<f32>,
}

/// One layer's intermediates for Q8_0 block-aware verification.
///
/// Extends `LayerTrace` with per-block accumulators and input scale factors.
/// The flat i32 outputs in the base `LayerTrace` are replaced by the f32
/// assembled outputs (from Phase B), and the block accumulators (for Phase A)
/// are stored separately per matrix.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Q8LayerTrace {
    // --- Inputs (i8 quantized values + per-block scales) ---
    pub x_attn: Vec<i8>,
    pub x_attn_scales: Q8Scales,

    // --- Attention projections (per-block accumulators) ---
    pub q_blocks: Q8BlockAccumulators,
    pub k_blocks: Q8BlockAccumulators,
    pub v_blocks: Q8BlockAccumulators,

    // --- Attention output ---
    pub a: Vec<i8>,
    pub a_scales: Q8Scales,
    pub attn_out_blocks: Q8BlockAccumulators,

    // --- Assembled f32 outputs (from Phase B, for chain checks) ---
    pub attn_out_f32: Vec<f32>,

    // --- FFN inputs ---
    pub x_ffn: Vec<i8>,
    pub x_ffn_scales: Q8Scales,

    // --- FFN projections (per-block accumulators) ---
    pub g_blocks: Q8BlockAccumulators,
    pub u_blocks: Q8BlockAccumulators,

    // --- SiLU output ---
    pub h: Vec<i8>,
    pub h_scales: Q8Scales,
    pub ffn_out_blocks: Q8BlockAccumulators,

    // --- Assembled f32 FFN output (for chain to next layer) ---
    pub ffn_out_f32: Vec<f32>,
}

/// Protocol version for the IO hash format used in the commitment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommitmentVersion {
    /// V4: Retained-state commitment. Trace tree leaf is
    /// `hash_retained_state_direct(retained)` over per-layer `a_i8 + scale_a`
    /// plus bridge replay scales for exact audit-time RMSNorm replay.
    /// IO leaf chains the retained leaf hash (not ad hoc features):
    /// `H("vi-io-v4" || leaf_hash_t || token_id || prev_io_hash)`.
    /// Prefix binding via retained Merkle tree — auditor opens prior leaves.
    V4 = 4,
}

impl Default for CommitmentVersion {
    fn default() -> Self {
        CommitmentVersion::V4
    }
}

/// Commitment published by the prover after running N tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchCommitment {
    pub merkle_root: [u8; 32],
    /// Root of a second Merkle tree over H(input || output) per token.
    /// Binds each token's first-layer input and last-layer output to the
    /// commitment, enabling cross-token chain verification.
    pub io_root: [u8; 32],
    pub n_tokens: u32,
    /// Four-spec manifest hash: M = H("vi-manifest-v4" || H_input || H_model || H_decode || H_output).
    /// Binds preprocessing, model identity, decode policy, and output policy to the commitment.
    #[serde(default)]
    pub manifest_hash: Option<[u8; 32]>,
    /// H(InputSpec): tokenizer, chat template, system prompt, preprocessing policy.
    #[serde(default)]
    pub input_spec_hash: Option<[u8; 32]>,
    /// H(ModelSpec): R_W, quantization, RoPE, RMSNorm eps, adapter identity.
    #[serde(default)]
    pub model_spec_hash: Option<[u8; 32]>,
    /// H(DecodeSpec): temperature, top-k/p, penalties, sampler version.
    #[serde(default)]
    pub decode_spec_hash: Option<[u8; 32]>,
    /// H(OutputSpec): EOS policy, stop sequences, max/min tokens, detokenization.
    #[serde(default)]
    pub output_spec_hash: Option<[u8; 32]>,
    /// Protocol version for the IO hash format.
    #[serde(default)]
    pub version: CommitmentVersion,
    /// SHA-256 of the canonicalized prompt / request input.
    /// Prevents cross-request replay: a valid proof for prompt A
    /// cannot be reused as proof for prompt B.
    #[serde(default)]
    pub prompt_hash: Option<[u8; 32]>,
    /// SHA-256 of the sampling seed, committed before inference.
    /// The prover reveals the seed after inference; the verifier checks
    /// `SHA-256(revealed_seed) == seed_commitment` and replays sampling.
    /// Prevents re-rolling, cherry-picking, and biased sampling.
    #[serde(default)]
    pub seed_commitment: Option<[u8; 32]>,
    /// Number of prompt tokens (including the first token consumed as embedding
    /// input). Binds the prompt/generation boundary so the verifier knows which
    /// tokens are determined by the tokenizer vs. by sampling.
    #[serde(default)]
    pub n_prompt_tokens: Option<u32>,

    /// Per-layer Merkle root over the KV transcript.
    /// `kv_roots[l]` commits post-RoPE K and dequantized V for all positions
    /// at layer `l`. Empty for legacy/toy commitments without KV binding.
    #[serde(default)]
    pub kv_roots: Vec<[u8; 32]>,
}

/// Audit tier: determines how many layers are opened.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditTier {
    /// ~90% of audits: open a random subset of layers (e.g. 10/80).
    Routine,
    /// ~10% of audits: open all layers.
    Full,
}

/// Challenge issued by the verifier for a stratified audit.
///
/// The verifier picks one token and a set of layer indices to open.
/// For `Full` audits, `layer_indices` contains all layers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditChallenge {
    /// Which token position to audit.
    pub token_index: u32,
    /// Which layer indices the prover must open (sorted).
    pub layer_indices: Vec<usize>,
    /// Audit tier that produced this challenge.
    pub tier: AuditTier,
}
