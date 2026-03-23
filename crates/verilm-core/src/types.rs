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

/// Deployment configuration bound to a batch commitment.
/// Hashed and stored in `BatchCommitment.manifest_hash` to prevent
/// tokenizer substitution, sampling-parameter manipulation, and
/// system-prompt injection attacks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentManifest {
    /// SHA-256 of the tokenizer vocabulary/config files.
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
    /// Outer: layer. Inner: MatrixType::ALL order.
    /// Empty if source was already INT8.
    pub quantization_scales: Vec<Vec<f32>>,

    /// SECRET: Per-matrix-type random vectors r_j.
    /// Index: MatrixType::ALL order (Wq, Wk, Wv, Wo, Wg, Wu, Wd).
    /// Must never be revealed to the prover.
    pub r_vectors: Vec<Vec<Fp>>,

    /// SECRET: Precomputed v_j^(i) = r_j^T W_j^(i).
    /// Outer index: layer. Inner index: MatrixType::ALL order.
    /// v_vectors[layer][matrix_type] has length = input_dim of that matrix.
    /// Leaking v is equivalent to leaking r (prover knows W).
    pub v_vectors: Vec<Vec<Vec<Fp>>>,

    /// Per-layer infinity-norm of W_o (max absolute row sum, scaled).
    /// Used for margin certificate perturbation bounds (Level B).
    /// Empty if not computed (Level A only).
    #[serde(default)]
    pub wo_norms: Vec<f32>,

    /// Max L-infinity norm of V head vectors across all layers and heads.
    /// Used for margin certificate perturbation bounds (Level B).
    /// Defaults to 0.0 if not computed (Level A only).
    #[serde(default)]
    pub max_v_norm: f32,

    /// Unembedding matrix (lm_head) for Level B logit verification.
    /// Shape: (vocab_size, hidden_dim), row-major INT8.
    /// When present, the verifier recomputes logits from final_hidden
    /// instead of trusting the margin certificate's self-reported logits.
    #[serde(default)]
    pub lm_head: Option<Vec<i8>>,

    /// SHA-256 hash of all INT8 weights in canonical order (weight chain).
    /// Binds the verifier key to a specific quantized checkpoint, ensuring
    /// the Freivalds precomputation corresponds to the published model.
    /// `None` for legacy keys generated before the weight-chain feature.
    #[serde(default)]
    pub weight_hash: Option<[u8; 32]>,

    /// Per-layer RMSNorm weight vectors for attention input normalization.
    /// `rmsnorm_attn_weights[layer]` has length `hidden_dim`.
    /// Empty for toy model (no RMSNorm in simplified chain).
    #[serde(default)]
    pub rmsnorm_attn_weights: Vec<Vec<f32>>,

    /// Per-layer RMSNorm weight vectors for FFN input normalization.
    /// `rmsnorm_ffn_weights[layer]` has length `hidden_dim`.
    /// Empty for toy model (no RMSNorm in simplified chain).
    #[serde(default)]
    pub rmsnorm_ffn_weights: Vec<Vec<f32>>,

    /// Per-layer weight scales for each matrix type.
    /// `weight_scales[layer][matrix_type_idx]` is the per-tensor absmax scale.
    /// Used for dequantizing i32 accumulators to f32 in the real requantization
    /// bridge: `output_f32 = acc_i32 * scale_w * scale_x`.
    /// Empty for toy model (unit scale) or native INT8 weights.
    #[serde(default)]
    pub weight_scales: Vec<Vec<f32>>,

    /// RMSNorm epsilon. Default 1e-5 for Llama-family models.
    #[serde(default = "default_rmsnorm_eps")]
    pub rmsnorm_eps: f64,
}

fn default_rmsnorm_eps() -> f64 {
    1e-5
}

impl VerifierKey {
    pub fn r_for(&self, mt: MatrixType) -> &[Fp] {
        let idx = MatrixType::ALL.iter().position(|&m| m == mt).unwrap();
        &self.r_vectors[idx]
    }

    pub fn v_for(&self, layer: usize, mt: MatrixType) -> &[Fp] {
        let idx = MatrixType::ALL.iter().position(|&m| m == mt).unwrap();
        &self.v_vectors[layer][idx]
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
    pub x_attn: Vec<i8>,   // input to attention block (INT8)
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
/// Only the attention boundary (non-replayable) and the dynamic
/// quantization scales needed for exact audit replay.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetainedLayerState {
    /// Post-attention INT8 output fed into W_o. This is the ONLY
    /// non-derivable intermediate: it comes from softmax(QK^T/√d)V
    /// which depends on the full KV prefix and cannot be replayed
    /// from single-token local state + public weights alone.
    pub a: Vec<i8>,
    /// Per-tensor activation scale for `a` (W_o projection input).
    pub scale_a: f32,
    /// Per-tensor activation scale for QKV projection input.
    /// Needed for exact RMSNorm bridge replay at audit time.
    pub scale_x_attn: f32,
    /// Per-tensor activation scale for gate_up projection input.
    pub scale_x_ffn: f32,
    /// Per-tensor activation scale for down projection input.
    pub scale_h: f32,
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
//   Batched check with random coefficients c_b (derived via Fiat-Shamir from
//   the Merkle root, so both sides know them post-commit):
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

/// Compact wire format for one layer's intermediates.
///
/// Omits two fields that the verifier can reconstruct deterministically:
/// - `x_ffn`: equals `requantize(attn_out)` (verified by chain check)
/// - `h`: equals `silu(requant(g), requant(u))` (verified by SiLU check)
///
/// The Merkle hash is still computed over the full `LayerTrace`. The verifier
/// reconstructs the omitted fields before verifying the Merkle proof.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactLayerTrace {
    pub x_attn: Vec<i8>,   // input to attention block (INT8)
    pub q: Vec<i32>,        // W_q x (i32 accumulator)
    pub k: Vec<i32>,        // W_k x (i32 accumulator)
    pub v: Vec<i32>,        // W_v x (i32 accumulator)
    pub a: Vec<i8>,         // attention vector, input to W_o (INT8)
    pub attn_out: Vec<i32>, // W_o a (i32 accumulator)
    // x_ffn omitted: verifier derives as requantize(attn_out)
    pub g: Vec<i32>,        // W_g x_ffn (i32 accumulator)
    pub u: Vec<i32>,        // W_u x_ffn (i32 accumulator)
    // h omitted: verifier derives as silu(requant(g), requant(u))
    pub ffn_out: Vec<i32>,  // W_d h (i32 accumulator)
    /// KV cache K vectors (Level C only). Empty for Level A/B.
    #[serde(default)]
    pub kv_cache_k: Vec<Vec<i8>>,
    /// KV cache V vectors (Level C only). Empty for Level A/B.
    #[serde(default)]
    pub kv_cache_v: Vec<Vec<i8>>,
    #[serde(default)]
    pub scale_x_attn: Option<f32>,
    #[serde(default)]
    pub scale_a: Option<f32>,
    #[serde(default)]
    pub scale_x_ffn: Option<f32>,
    #[serde(default)]
    pub scale_h: Option<f32>,
    #[serde(default)]
    pub residual: Option<Vec<f32>>,
}

impl CompactLayerTrace {
    /// Strip derivable fields from a full trace.
    pub fn from_full(lt: &LayerTrace) -> Self {
        CompactLayerTrace {
            x_attn: lt.x_attn.clone(),
            q: lt.q.clone(),
            k: lt.k.clone(),
            v: lt.v.clone(),
            a: lt.a.clone(),
            attn_out: lt.attn_out.clone(),
            g: lt.g.clone(),
            u: lt.u.clone(),
            ffn_out: lt.ffn_out.clone(),
            kv_cache_k: lt.kv_cache_k.clone(),
            kv_cache_v: lt.kv_cache_v.clone(),
            scale_x_attn: lt.scale_x_attn,
            scale_a: lt.scale_a,
            scale_x_ffn: lt.scale_x_ffn,
            scale_h: lt.scale_h,
            residual: lt.residual.clone(),
        }
    }

    /// Reconstruct the full `LayerTrace` by deriving omitted fields.
    ///
    /// Uses the same deterministic operations the verifier checks:
    /// - `x_ffn = requantize(attn_out)` (clamp i32 to i8)
    /// - `h = SiLU(requant(g)) * requant(u)` via LUT (unit scale)
    ///
    /// Returns `Err` if the compact trace has invalid dimensions (e.g.
    /// empty or mismatched g/u vectors). Malformed compact traces are
    /// rejected with a structured error, never silently accepted.
    pub fn to_full(&self) -> Result<LayerTrace, String> {
        if self.g.len() != self.u.len() {
            return Err(format!(
                "compact trace dimension mismatch: g.len()={} != u.len()={}",
                self.g.len(),
                self.u.len()
            ));
        }
        if self.g.is_empty() {
            return Err("compact trace has empty g/u vectors".into());
        }
        let x_ffn: Vec<i8> = self.attn_out.iter().map(|&v| v.clamp(-128, 127) as i8).collect();
        let h = crate::silu::compute_h_unit_scale(&self.g, &self.u);
        Ok(LayerTrace {
            x_attn: self.x_attn.clone(),
            q: self.q.clone(),
            k: self.k.clone(),
            v: self.v.clone(),
            a: self.a.clone(),
            attn_out: self.attn_out.clone(),
            x_ffn,
            g: self.g.clone(),
            u: self.u.clone(),
            h,
            ffn_out: self.ffn_out.clone(),
            kv_cache_k: self.kv_cache_k.clone(),
            kv_cache_v: self.kv_cache_v.clone(),
            scale_x_attn: self.scale_x_attn,
            scale_a: self.scale_a,
            scale_x_ffn: self.scale_x_ffn,
            scale_h: self.scale_h,
            residual: self.residual.clone(),
        })
    }
}

/// Compact wire format for a complete token trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactTokenTrace {
    pub token_index: u32,
    pub layers: Vec<CompactLayerTrace>,
    pub merkle_root: [u8; 32],
    pub merkle_proof: MerkleProof,
    pub io_proof: MerkleProof,
    #[serde(default)]
    pub final_hidden: Option<Vec<i8>>,
    /// Emitted token ID (mirrored from TokenTrace for compact format).
    #[serde(default)]
    pub token_id: Option<u32>,
    /// Previous token's IO hash for V3 transcript chaining (mirrored from TokenTrace).
    #[serde(default)]
    pub prev_io_hash: Option<[u8; 32]>,
    /// Previous token's KV chain hash (mirrored from TokenTrace).
    #[serde(default)]
    pub prev_kv_hash: Option<[u8; 32]>,
}

impl CompactTokenTrace {
    pub fn from_full(tt: &TokenTrace) -> Self {
        CompactTokenTrace {
            token_index: tt.token_index,
            layers: tt.layers.iter().map(CompactLayerTrace::from_full).collect(),
            merkle_root: tt.merkle_root,
            merkle_proof: tt.merkle_proof.clone(),
            io_proof: tt.io_proof.clone(),
            final_hidden: tt.final_hidden.clone(),
            token_id: tt.token_id,
            prev_io_hash: tt.prev_io_hash,
            prev_kv_hash: tt.prev_kv_hash,
        }
    }

    pub fn to_full(&self) -> Result<TokenTrace, String> {
        let layers: Result<Vec<_>, _> = self
            .layers
            .iter()
            .enumerate()
            .map(|(i, cl)| {
                cl.to_full()
                    .map_err(|e| format!("token {}, layer {}: {}", self.token_index, i, e))
            })
            .collect();
        Ok(TokenTrace {
            token_index: self.token_index,
            layers: layers?,
            merkle_root: self.merkle_root,
            merkle_proof: self.merkle_proof.clone(),
            io_proof: self.io_proof.clone(),
            margin_cert: None,
            final_hidden: self.final_hidden.clone(),
            token_id: self.token_id,
            prev_io_hash: self.prev_io_hash,
            prev_kv_hash: self.prev_kv_hash,
        })
    }
}

/// Compact wire format for a batch proof.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactBatchProof {
    pub commitment: BatchCommitment,
    pub traces: Vec<CompactTokenTrace>,
    #[serde(default)]
    pub revealed_seed: Option<[u8; 32]>,
}

impl CompactBatchProof {
    pub fn from_full(bp: &BatchProof) -> Self {
        CompactBatchProof {
            commitment: bp.commitment.clone(),
            traces: bp.traces.iter().map(CompactTokenTrace::from_full).collect(),
            revealed_seed: bp.revealed_seed,
        }
    }

    pub fn to_full(&self) -> Result<BatchProof, String> {
        let traces: Result<Vec<_>, _> = self
            .traces
            .iter()
            .map(|ct| ct.to_full())
            .collect();
        Ok(BatchProof {
            commitment: self.commitment.clone(),
            traces: traces?,
            revealed_seed: self.revealed_seed,
        })
    }
}

/// Complete trace for one token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenTrace {
    pub token_index: u32,
    pub layers: Vec<LayerTrace>,
    pub merkle_root: [u8; 32],
    pub merkle_proof: MerkleProof,
    /// Proof in the IO tree binding this token's input/output to the commitment.
    pub io_proof: MerkleProof,
    /// Optional margin certificate for Level B verification.
    #[serde(default)]
    pub margin_cert: Option<crate::margin::MarginCertificate>,
    /// Final hidden state (requantized last-layer output) for Level B logit binding.
    /// When present, the verifier recomputes logits = lm_head @ final_hidden and
    /// checks them against the margin certificate's self-reported logits.
    /// Also included in the Merkle leaf hash to bind it to the commitment.
    #[serde(default)]
    pub final_hidden: Option<Vec<i8>>,
    /// Emitted token ID (vocabulary index) for this position.
    /// When present, bound into the IO tree leaf hash (V2 format) so that
    /// the commitment pins the actual sampled token, not just the computation.
    /// `None` for legacy traces that predate token-ID binding.
    #[serde(default)]
    pub token_id: Option<u32>,
    /// Previous token's IO hash for V3 transcript chaining.
    /// Provided by the prover during open so the verifier can reconstruct
    /// the chained IO hash for arbitrary (non-consecutive) challenge sets.
    /// For token 0, this is the genesis zero hash `[0u8; 32]`.
    /// `None` for V1/V2 traces.
    #[serde(default)]
    pub prev_io_hash: Option<[u8; 32]>,
    /// Previous token's KV chain hash for KV provenance chaining.
    /// For token 0, this is the genesis zero hash `[0u8; 32]`.
    /// `None` for traces without KV provenance binding.
    #[serde(default)]
    pub prev_kv_hash: Option<[u8; 32]>,
}

/// Protocol version for the IO hash format used in the commitment.
///
/// Explicit version discriminant so deployments can reject legacy proofs
/// by policy rather than inferring the version from optional fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommitmentVersion {
    /// V1: IO leaf = H(first_input || requant(last_output)). No token-ID binding.
    V1 = 1,
    /// V2: IO leaf = H("vi-io-v2" || first_input || requant(last_output) || token_id).
    /// Binds emitted token ID into the commitment.
    V2 = 2,
    /// V3: IO leaf = H("vi-io-v3" || first_input || requant(last_output) || token_id || prev_io_hash).
    /// Adds transcript chaining: each token's IO leaf depends on the previous,
    /// preventing insertion, deletion, reordering, and retroactive edits.
    V3 = 3,
    /// V4: Retained-state commitment. Trace tree leaf is
    /// `hash_retained_state_direct(retained)` over per-layer `a_i8 + scale_a`
    /// plus transitional replay scales.
    /// IO leaf chains the retained leaf hash (not ad hoc features):
    /// `H("vi-io-v4" || leaf_hash_t || token_id || prev_io_hash)`.
    /// Prefix binding via retained Merkle tree — auditor opens prior leaves.
    V4 = 4,
}

impl Default for CommitmentVersion {
    fn default() -> Self {
        CommitmentVersion::V1
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
    /// Hash of the deployment manifest (Level B). Binds configuration to commitment.
    #[serde(default)]
    pub manifest_hash: Option<[u8; 32]>,
    /// Protocol version for the IO hash format.
    /// V1 = legacy (no token-ID binding), V2 = token-ID bound, V3 = transcript-chained.
    /// Defaults to V1 for backward compatibility with serialized proofs.
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
    /// Root of a Merkle tree over per-token KV chain hashes.
    /// Binds each token's K/V projections into a running provenance chain,
    /// preventing KV cache fabrication under partial opening.
    /// `None` for commitments that predate KV provenance chaining.
    #[serde(default)]
    pub kv_chain_root: Option<[u8; 32]>,
}

/// Opened traces for challenged token indices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchProof {
    pub commitment: BatchCommitment,
    pub traces: Vec<TokenTrace>,
    /// Revealed sampling seed (opened after inference).
    /// Verifier checks `SHA-256(revealed_seed) == commitment.seed_commitment`.
    /// `None` for proofs without sampling binding.
    #[serde(default)]
    pub revealed_seed: Option<[u8; 32]>,
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

/// Prover's response to an audit challenge.
///
/// Contains partial layer traces for the challenged token (only the
/// requested layers) plus requantized KV prefix for attention replay
/// on those layers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditResponse {
    /// The challenged token index (must match `AuditChallenge.token_index`).
    pub token_index: u32,
    /// Layer traces for the requested layers only.
    /// `partial_layers[i]` corresponds to `AuditChallenge.layer_indices[i]`.
    pub partial_layers: Vec<LayerTrace>,
    /// Which layer indices these traces correspond to (must match challenge).
    pub layer_indices: Vec<usize>,
    /// Per-layer KV prefix for attention replay.
    /// `kv_k_prefix[i]` is the K cache up to `token_index` for `layer_indices[i]`.
    /// `kv_v_prefix[i]` is the V cache up to `token_index` for `layer_indices[i]`.
    pub kv_k_prefix: Vec<Vec<Vec<i8>>>,
    pub kv_v_prefix: Vec<Vec<Vec<i8>>>,
    /// Per-layer Merkle proofs for the streaming KV commitment.
    /// `kv_layer_proofs[i]` proves that `layer_indices[i]`'s KV hash
    /// is in the committed KV root for this token.
    pub kv_layer_proofs: Vec<MerkleProof>,
    /// Merkle proof binding this token's full trace to the commitment root.
    pub merkle_root: [u8; 32],
    pub merkle_proof: MerkleProof,
    /// Final hidden state (for margin / Level B checks).
    pub final_hidden: Option<Vec<i8>>,
    /// Token ID produced at this position.
    pub token_id: Option<u32>,
}

/// Compact form of `AuditResponse`: uses `CompactLayerTrace` to omit
/// derivable fields (`x_ffn`, `h`). Same pattern as `CompactBatchProof`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactAuditResponse {
    pub token_index: u32,
    pub partial_layers: Vec<CompactLayerTrace>,
    pub layer_indices: Vec<usize>,
    pub kv_k_prefix: Vec<Vec<Vec<i8>>>,
    pub kv_v_prefix: Vec<Vec<Vec<i8>>>,
    pub kv_layer_proofs: Vec<MerkleProof>,
    pub merkle_root: [u8; 32],
    pub merkle_proof: MerkleProof,
    pub final_hidden: Option<Vec<i8>>,
    pub token_id: Option<u32>,
}

impl CompactAuditResponse {
    pub fn from_full(resp: &AuditResponse) -> Self {
        Self {
            token_index: resp.token_index,
            partial_layers: resp.partial_layers.iter().map(CompactLayerTrace::from_full).collect(),
            layer_indices: resp.layer_indices.clone(),
            kv_k_prefix: resp.kv_k_prefix.clone(),
            kv_v_prefix: resp.kv_v_prefix.clone(),
            kv_layer_proofs: resp.kv_layer_proofs.clone(),
            merkle_root: resp.merkle_root,
            merkle_proof: resp.merkle_proof.clone(),
            final_hidden: resp.final_hidden.clone(),
            token_id: resp.token_id,
        }
    }

    pub fn to_full(&self) -> Result<AuditResponse, String> {
        let partial_layers: Result<Vec<LayerTrace>, String> = self
            .partial_layers
            .iter()
            .map(|cl| cl.to_full())
            .collect();
        Ok(AuditResponse {
            token_index: self.token_index,
            partial_layers: partial_layers?,
            layer_indices: self.layer_indices.clone(),
            kv_k_prefix: self.kv_k_prefix.clone(),
            kv_v_prefix: self.kv_v_prefix.clone(),
            kv_layer_proofs: self.kv_layer_proofs.clone(),
            merkle_root: self.merkle_root,
            merkle_proof: self.merkle_proof.clone(),
            final_hidden: self.final_hidden.clone(),
            token_id: self.token_id,
        })
    }
}

/// Policy constraints the verifier enforces beyond the cryptographic checks.
///
/// By default, all checks are permissive (accept any version, no prompt
/// verification). Strict deployments should set `min_version` and provide
/// `expected_prompt_hash` to close replay and downgrade gaps.
#[derive(Debug, Clone, Default)]
pub struct VerificationPolicy {
    /// Minimum acceptable commitment version. Proofs with a version below
    /// this are rejected outright. `None` = accept any version.
    pub min_version: Option<CommitmentVersion>,
    /// Expected prompt hash. When set, the verifier checks that the
    /// commitment's `prompt_hash` matches this value, binding the proof
    /// to the specific request.
    pub expected_prompt_hash: Option<[u8; 32]>,
    /// Expected manifest hash. When set, the verifier checks that the
    /// commitment's `manifest_hash` matches this value.
    pub expected_manifest_hash: Option<[u8; 32]>,
}

/// Result of verifying one token.
#[derive(Debug)]
pub struct VerifyResult {
    pub passed: bool,
    pub failures: Vec<VerifyFailure>,
}

#[derive(Debug)]
pub struct VerifyFailure {
    pub layer: usize,
    pub matrix: MatrixType,
    pub kind: FailureKind,
}

#[derive(Debug)]
pub enum FailureKind {
    Freivalds,
    SiLu,
    Merkle,
    /// Margin certificate structural check failed (wrong top-1/top-2/delta).
    Margin,
    /// Logit vector doesn't match the trace (Level B).
    LogitMismatch,
}

impl std::fmt::Display for VerifyResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.passed {
            write!(f, "PASS")
        } else {
            writeln!(f, "FAIL ({} failures):", self.failures.len())?;
            for fail in &self.failures {
                writeln!(f, "  layer {}: {:?} - {:?}", fail.layer, fail.matrix, fail.kind)?;
            }
            Ok(())
        }
    }
}
