# Changelog

This changelog tracks the kept canonical VeriLM protocol and its major implementation milestones.

Historical references below to “roadmap #N” refer to the pre-2026-03-30 roadmap numbering. On 2026-03-30 the roadmap was renumbered into a single linear open-items-only sequence.

## 2026-03-31

### Fixed

- **RoPE convention**: `apply_rope_head` used interleaved pairing `[2i, 2i+1]` but vLLM/Qwen/LLaMA use half-rotary `[i, i+half]`. K corridor dropped from L-inf=255 to <0.5 after fix.
- **Q/KV input boundary asymmetry in corridor measurement**: shell opening Q was computed from bridge-derived x_attn while committed K/V used GPU-captured x_attn. Added `use_captured_x_attn` flag to `open_v4` so corridor measurement uses the same authoritative boundary for Q and K/V. Corridor dropped from L-inf=67–117 to L-inf=8.

### Measured

- **Attention corridor on Qwen2.5-7B-W8A8 (A100-80GB)**: 672 measurements across 6 workloads (short through 1164-token long_context), all 28 layers, all decode positions. Global max L-inf = 8. First generated token max L-inf = 5. >92% of elements are exact matches; >99.8% within ±1. No growth with sequence length. Worst layers spread across the stack (not concentrated). The ≤1 target for a formal single-step bound appears achievable — the real BF16-vs-f64 gap is small and stable. Llama-family control measurement pending.

## 2026-03-30

### Added

- **Roadmap #3 closed: committed KV transcript with kv_root covering full causal prefix.** `KvEntry { k_roped, v_deq }` committed per (layer, position) under per-layer Merkle `kv_roots` with domain separator `vi-kv-v1`. Prover derives post-RoPE K/V at commit time via deterministic INT8 matmul on captured `x_attn_i8` + weights (no GPU-side K/V hooks needed). Canonical verifier `phase_kv_transcript` verifies Merkle proofs against `kv_roots`. `replay_deep_prefix_toy/roped` consumes committed KV when available, falling back to shell accumulator reconstruction for legacy. GPU evidence on `neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8` (A100-80GB): kv_roots present on default path (28/28 layers); binary deep-prefix KV Merkle proofs pass (0 failures); binary canonical path has 0 KV failures (only attention replay mismatch remains — tracked in #4). Three non-attention confounders fixed during this milestone: per-channel dequant in prover bridge (Freivalds 0/168 failures, was 168/168), shell QKV using bridge-derived input instead of GPU-captured x_attn, and quant_family/scale_derivation vocabulary alignment between weight provider and manifest. JSON debug path has a known f64 round-trip precision issue on KV entries (not a protocol blocker — binary is the canonical production format).

- x_attn capture enabled by default when `VERILM_CAPTURE=1`. The non-packed commit path (which computes KV transcripts) is now the default for verified inference. The packed path remains available via `VERILM_CAPTURE_X_ATTN=0` but does not produce kv_roots. This ensures a supported verified run always includes committed KV.

### Fixed

- Per-channel dequantization in prover `compute_shell_opening` — previously only the verifier dispatched per-channel, causing W8A8 Freivalds failures on all layers/matrices (per-tensor scale was 0.0 for W8A8).
- Shell QKV now uses bridge-derived x_attn (not GPU-captured) so that QKV accumulators match the verifier's own derivation for Freivalds checks. Captured x_attn is only for KV transcript computation at commit time.
- quant_family/scale_derivation metadata now derived from `WeightProvider` using the same logic as keygen, instead of extracting HuggingFace `quant_method` strings that didn't match key vocabulary.
- `_weight_provider` initialization order in `server.py` — quant fields deferred until after weight provider is created.

### Previously added

- **Roadmap #1 closed: quantization semantics are now first-class protocol data.** Per-channel weight scales flow end-to-end from keygen through verification. Keygen discovers `weight_scale` tensors in W8A8 safetensors (handling both `[N]` and `[N,1]` shapes, BF16/FP16/F32 dtypes), populates `VerifierKey.per_channel_weight_scales`, and sets `quant_family` / `scale_derivation` metadata. The canonical verifier dispatches to per-channel dequant for QKV attention, bridge residual, and SiLU gate paths. Corridor tooling uses key-embedded scales instead of external overrides. Confirmed on real `neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8` via Modal smoke test: 28 layers × 7 matrices of per-channel scales loaded with correct shapes and realistic nonzero ranges. Formal worst-case attention corridor bounds derived in `bounds.rs`. Per-token prefill `scale_a` and grouped-quant `quant_block_size` are real issues tracked separately, not blocking #2.

- **Roadmap #2 closed: bridge trust boundary is now committed and checked with check-and-gate semantics.** `RetainedLayerState` now commits the actual attention-input bridge object (`x_attn_i8`, `scale_x_attn`) alongside `a`, and the retained Merkle hash domain is bumped from `vi-retained-v2` to `vi-retained-v3` with presence markers so missing-versus-present bridge data is hash-distinct. The canonical verifier now gates on the opened bridge boundary before downstream replay: it canonically recomputes `x_attn_i8`, compares against the committed bridge value, and if the check passes continues from the committed value rather than the verifier replay. This makes the bridge the authoritative QKV input boundary and prevents approximation error from propagating across layers. Binary fixtures and golden pins were regenerated; dedicated tests cover honest pass, tampered `x_attn_i8`, tampered `scale_x_attn`, presence-marker hashing, and fail-closed handling when bridge data is missing.

- Production RoPE-aware deep-prefix attention replay landed in the canonical verifier and corridor tooling. `VerifierKey.rope_aware_replay` now dispatches between toy/reference replay and a production path that dequantizes Q/K/V accumulators with activation and weight scales, applies RoPE at each position, and replays attention in f64 post-RoPE space. The deep-prefix path covers both prefix tokens and the opened token, and tests cover honest pass plus fake-`a` rejection on the roped path.

### Changed

- The roadmap was renumbered into a single linear sequence of remaining tasks, and completed items were removed from `roadmap.md` so the roadmap is now strictly an open-items tracker while this changelog carries completed milestones.
- With roadmap `#1` and `#2` closed, the active attention-critical path now starts at roadmap `#3`: full-causal `kv_root`, cleanup of the old confounded gap path, and only then analytical/empirical attention-gap work.

## 2026-03-29

### Added

- Client-side verification wrapper (`verilm_verify::client`) for protocol-level checks that sit outside the canonical verifier's trust boundary. `verify_challenged_binary()` and `verify_challenged_response()` now bind an audit to the verifier-issued `AuditChallenge` by checking the challenged `token_index` and exact opened `layer_indices`, using stable `FailureCode` values (`ChallengeTokenMismatch`, `ChallengeLayerMismatch`) rather than message parsing. Tests cover pass, wrong token, wrong layers, canonical-failure propagation, combined failures, and the baseline no-wrapper path.

### Changed

- The canonical verifier is now the trusted public verification path. Public `verify_v4()` / `verify_v4_full()` entrypoints delegate to the canonical verifier; the old monolithic verifier remains only for differential testing and rollback during the freeze-out period.
- The canonical verifier was cleaned up into a small explicit pipeline: `Ctx::new()` owns precomputed facts, `run()` is the orchestrator, typed phase state (`StructuralState`, `SpecState`, `BridgeState`) wires the phases together, and the bridge path is split into named subchecks instead of transitional branch soup. The trust boundary is now explicit: `canonical` verifies proof correctness, while `client` handles external challenge/receipt protocol checks.
- Retained-state schema trimmed to irreducible fields only. `RetainedLayerState` now commits only `a` and `scale_a`; derivable replay scales (`scale_x_attn`, `scale_x_ffn`, `scale_h`) moved to `ShellLayerOpening`, the retained hash domain bumped to `vi-retained-v2`, prover-side `CapturedLayerScales` carries those values between commit and open, and frozen fixtures / SHA-256 pins were regenerated. Tampering any revealed replay scale is caught by the dependent bridge/Freivalds equations. This closes roadmap `#2`.

## 2026-03-28

### Fixed

- Keygen `rmsnorm_eps` split-brain: key generation and `SafetensorsWeightProvider` both now read `rms_norm_eps` from the model's `config.json` instead of hardcoding `1e-5`. Fixes bridge replay divergence on models like Qwen that use `1e-6`.
- Keygen `rope_theta` now sourced from `config.json` instead of a heuristic based on `hidden_dim`. Fixes mismatch on models with non-standard `rope_theta` values.
- Keygen `vocab_size` early detection from `lm_head.weight` tensor shape in `detect_config()`, before `r` vector generation. Fixes panic when `vocab_size` was 0 at LmHead Freivalds setup.
- Verifier `n_tokens` generation-length semantics: `n_generated = n_tokens - (n_prompt - 1)`, accounting for the committed token_ids array omitting the first embedding token. Fixes off-by-one that allowed `n_tokens_inflate` to pass.
- Verifier LM-head token replay now skipped for prompt-side tokens (where `token_index < gen_start`). Prompt tokens are chosen by the tokenizer, not by argmax/sampling over logits. Fixes false rejection on multi-token audits where the challenged position is inside the prompt.

### Added

- Verification failure taxonomy: every verifier failure is classified into one of six categories (`structural`, `cryptographic_binding`, `spec_mismatch`, `unsupported`, `semantic_violation`, `operational`) via a stable `FailureCode` enum (~48 variants). Each `VerificationFailure` carries `code`, `category`, `message`, and optional `FailureContext` (token index, layer, matrix, field, spec, expected/actual values). Consumers match on codes — no message-text parsing needed. The substring-based `classify_failure()` classifier is removed.
- Structured audit-failure reporting: `V4VerifyReport.failures` is now `Vec<VerificationFailure>` with stable codes and context. `failure_messages()` accessor provides backward-compatible `Vec<&str>`. Python bridge exposes `classified_failures` as `list[dict]` with `code`, `category`, `message`, and optional `context` fields.
- Partial-audit semantics: `V4VerifyReport` now carries `coverage: AuditCoverage` distinguishing `Full` (all layers checked), `Routine` (contiguous prefix), and `Unknown` (no shell opening). Display and JSON serialization include coverage level. Python bridge exposes `coverage` as a dict with `level`, `layers_checked`, and optionally `layers_total`. Consumers can programmatically distinguish routine-audit passes from full-audit passes — partial statistical coverage cannot be mistaken for full exact success.
- Boundary-condition fuzzing suite (`boundary_fuzz.rs`, 55 tests): malformed/truncated binary payloads (empty, magic-only, garbage zstd, wrong bincode, single-bit flips, every-offset truncation), structural field absence/mismatch (seed, prompt hash, n_prompt_tokens bounds), prefix count boundaries, non-contiguous/reversed/empty layer_indices, EOS policy edge cases (stop-not-at-end, unknown policy, ignore_eos, min_tokens without eos_token_id), decode-mode/temperature inconsistency, unsupported decode features, manifest hash binding, unknown version/magic rejection, long/short prompt-output combinations, sampler version rejection, coverage semantics, and failure context metadata. All assert specific `FailureCode` and `FailureCategory` values. This closes roadmap `#8`.
- Cross-version binary format tests (`cross_version.rs`, 19 tests) with frozen on-disk fixtures (`tests/fixtures/`). Compatibility matrix: canonical V4 audit and key fixtures deserialize and verify, byte-stability checks detect format drift, SHA-256 checksums pinned, rejection fixtures for unknown magic, truncated payloads, cross-format misuse, and corrupted bincode. Forward-compatibility: VV5A/VV9Z/VKE2 future magics fail-closed. Cross-format: key-as-audit and audit-as-key rejected. This closes roadmap `#13`.
- Adversarial hardening gate (`hardening_gate.rs` + `make hardening-gate`): explicit named gate that must pass before strong claims or final benchmarks land. Aggregates 8 suites (265+ tests): boundary_fuzz, cross_version, v4_e2e, golden_conformance, weight_chain_adversarial, fiat_shamir_soundness, quantization_parity, plus GPU adversarial existence check. Gate meta-tests verify suite existence, minimum test counts, fixture presence, taxonomy completeness, and coverage semantics. `make gpu-test-adversarial` runs the remote GPU adversarial suite. This closes roadmap `#4`.

### Changed

- Adversarial tamper test (`test_adversarial.py`) hardened:
  - Baselines restored to strict assertions (224/224 checks pass on real W8A8 GPU).
  - Splice tests now use different prompts and audit at generated-token indices past the prompt boundary. Token splice requires verified token divergence before asserting rejection.
  - `final_residual_shift` accepts `merkle` as a valid rejection reason (stronger than a custom "final" reason since it proves retained-leaf hash binding).
  - Multi-token prefix baseline audits at a generated token index, avoiding prompt-side token replay.
  - Freivalds diagnostic script (`diag_freivalds.py`) added for targeted GPU-side failure classification.
- Adversarial suite now achieves 36/36 on the real W8A8 Modal run. Token-splice construction fixed by reading `n_prompt_tokens` from the audit response and auditing past the prompt/template boundary where content tokens actually diverge.

## 2026-03-27

### Added

- Four-spec commitment wiring for `input_spec_hash`, `model_spec_hash`, `decode_spec_hash`, and `output_spec_hash`, with verifier-side recomputation and composed manifest checking.
- Canonical commitment of the full four-spec surface, including chat template, BOS/EOS preprocessing, special-token handling, system-prompt semantics, public model identity `R_W`, adapter identity, RoPE configuration, RMSNorm epsilon, sampler identity, decode knobs, and output stopping rules.
- Explicit model-surface commitments for `n_layers`, `hidden_dim`, `vocab_size`, and `embedding_merkle_root`, with verifier-side cross-checks against the verifier key/config.
- `padding_policy` field in InputSpec and DeploymentManifest, hashed in `hash_input_spec()`, populated by the live server manifest path, and enforced fail-closed in the Python verifier path. Completes truncation/padding binding (roadmap #4).
- `decode_mode` field in DecodeSpec and DeploymentManifest, hashed in `hash_decode_spec()`, with verifier cross-check against temperature: greedy requires temp=0, sampled requires temp>0, unknown modes fail closed. Tests cover all mismatch and pass cases (roadmap #9).
- Quantization identity fields `quant_family`, `scale_derivation`, `quant_block_size` in ModelSpec, DeploymentManifest, and VerifierKey, hashed in `hash_model_spec()`, cross-checked against verifier key, extracted from model quantization config in the live server, and parsed in the Python bridge. Tests cover mismatch rejection and consistent pass (roadmap #5-7).
- Remaining architecture fields in ModelSpec: `kv_dim`, `ffn_dim`, `d_head`, `n_q_heads`, `n_kv_heads`, `rope_theta`, hashed in `hash_model_spec()`, cross-checked against verifier key config, extracted from model config in the live server, parsed in the Python bridge. Tests cover all 6 mismatch rejections plus full-architecture pass (roadmap #8).
- LM-head Freivalds as an explicit eighth matrix family, including verifier-secret key material and canonical verifier checks.
- Exact logits replay after LM-head binding so verified token selection still depends on the committed logits path.
- Exact final-token verification boundary from the captured pre-final-norm residual, with fail-closed behavior when the boundary state is missing.
- Canonical sampled-path replay with fresh per-request `batch_seed`, `seed_commitment`, audit-time seed reveal, and deterministic per-token randomness derivation.
- Request-bound transcript genesis `H("vi-io-genesis-v4" || prompt_hash)` and canonical V4 IO chaining over `leaf_hash`, `token_id`, and prior transcript state.
- Fail-closed prompt binding and prompt/generation-boundary count binding, including `n_prompt_tokens`.
- End-to-end canonical input reconstruction via `PromptTokenizer`, exposed through the Python verifier bridge.
- Full `InputSpec` replay in the Python verifier path, covering tokenizer identity, system prompt, chat template, BOS/EOS preprocessing, special-token policy, and truncation policy.
- Prefix embedding binding for prefix tokens, including per-prefix embedding rows and Merkle proofs in rich-prefix mode.
- Exact deep-prefix audit support, including `prefix_retained`, `prefix_shell_openings`, retained-hash consistency checks, and Freivalds/bridge checks on prefix tokens.
- End-to-end detokenization verification via `Detokenizer`, `output_text` in audit responses, and Python verifier plumbing for detokenizer callbacks.
- End-to-end HTTP coverage for `/chat` and `/audit`, including greedy and sampled JSON audit verification, binary audit verification, routine-tier coverage, and HTTP error-path coverage.
- Tokenizer identity commitment based on canonicalized `tokenizer.json` content via `backend_tokenizer.to_str()`, with deterministic JSON normalization and legacy vocab-only fallback.
- Explicit `weight_hash` / `R_W` cross-checking in the canonical verifier path, in addition to model-spec hash binding.
- Version-locked sampler conformance vectors: golden `derive_token_seed` SHA-256 vectors and golden `sample()` token outputs for temperature, top-k, top-p, combined filtering, and greedy modes. Any silent drift in the `chacha20-vi-sample-v1` pipeline breaks these tests.
- Greedy tie-breaking is fixed to lowest-index selection and pinned as part of the version-locked `chacha20-vi-sample-v1` sampler semantics.
- Golden conformance vectors embedded in the test suite for challenge derivation (`build_audit_challenge` token index and layer indices for 4 seed/token/layer configs), manifest hashing (pinned `hash_manifest` and `hash_model_spec` digests), and end-to-end verification (pinned merkle root, IO root, verdict, and checks_run for deterministic commit→open→verify).
- Challenge protocol specification (`docs/challenge-protocol.md`): exact derivation of token index, layer depth (routine/full), sampling seed, Freivalds block coefficients, all domain separators, and binary wire format.
- Binary format robustness tests: V4 audit response roundtrip through serialize/deserialize, unknown magic rejection, truncated payload rejection, cross-format rejection (key bytes to audit deserializer), version field preservation, and verifier key roundtrip. (Not yet a full cross-version compatibility matrix with frozen binary fixtures.)

### Changed

- Verified serving mode now defaults to sampled decoding (temperature=1.0). Greedy decoding remains available as the explicit `temperature=0` special case. Server API, HTTP endpoint, sampler hook, and Modal endpoint all updated.
- Terminology and documentation framing now consistently describe Freivalds checks as information-theoretically sound / statistically sound rather than cryptographic.
- Runtime-populated input, decode, and output policy fields now flow into the committed four-spec surface instead of relying on static defaults.
- The live canonical sampled path is replayable end to end under the committed decode spec instead of depending on incidental stock serving behavior.
- The key-only retained-state path is now the canonical verifier path for the kept protocol.
- The canonical verifier path now checks reconstructed prompt tokens against the committed input path and the embedding path.
- Output-policy enforcement now covers `eos_policy`, `min_tokens`, `ignore_eos`, stop conditions, special-token stripping, detokenization policy, cleanup behavior, and fail-closed handling for unsupported or unknown policy values.
- Input-policy replay is fail-closed for unknown `bos_eos_policy`, `special_token_policy`, `truncation_policy`, and `detokenization_policy`.
- The verifier now cross-checks committed model-spec values against verifier-key values for `rmsnorm_eps`, `rope_config_hash`, and `weight_hash`.
- The verifier now also cross-checks committed model geometry and embedding commitment values against the verifier key/config: `n_layers`, `hidden_dim`, `vocab_size`, and `embedding_merkle_root`.
- Decode-spec handling is fail-closed for unsupported sampler versions and unsupported non-default decode features; only the canonical sampler version is accepted and supported decode features are replayed.
- Audit retrieval now preserves claimed output text end to end so detokenization checks can run in the canonical verifier flow.
- Bridge documentation and call sites now explicitly treat `bridge_residual_rmsnorm()` as the canonical production W8A8 bridge, while `bridge_requantize()` is demoted to toy-model and last-layer fallback use only.
- The roadmap and protocol framing now treat the kept canonical protocol as structurally complete, with remaining work focused on hardening, conformance, benchmarks, and documentation.
- Removed transitional V4 framing: "dropped in V5" comments and "transitional replay scales" qualifiers replaced with canonical bridge replay scale framing. V4 is the kept protocol, not a stepping stone.
- `RetainedLayerState` documentation now explicitly classifies fields as irreducible (`a`, `scale_a`) vs derivable-with-weight-access (`scale_x_attn`, `scale_x_ffn`, `scale_h`), with derivation paths documented.
- Binary (bincode+zstd with VV4A magic) is now explicitly framed as the sole normative wire format for receipts and audits. JSON verification path demoted to debug/development convenience.
- Audit-time weight loading confirmed bound to `R_W`: server validates WeightProvider hash against manifest at startup, verifier cross-checks manifest weight_hash against key weight_hash.
- `bad_word_ids` field added to DecodeSpec and DeploymentManifest, hashed in `hash_decode_spec()`, fail-closed in verifier (non-empty rejected), parsed in Python bridge, set to empty in live server. Completes decode-surface logit modification binding (roadmap #10).

### Removed

- Legacy V1/V2/V3 protocol framing and code paths from the kept protocol story.
- Dead full-trace artifacts and stale legacy comments that only served the removed pre-V4 protocol paths.
- Dead `requantize_bridge()` and `verify_bridge()` bridge helpers superseded by `bridge_residual_rmsnorm()`.
- Unused `kv_chain_root` commitment field and related dead code.
- Transitional reliance on weight-backed replay as part of the canonical verifier path; weight-backed replay remains debug/oracle-only.
