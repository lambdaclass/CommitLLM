# Changelog

This changelog tracks the kept canonical VeriLM protocol and its major implementation milestones.

## 2026-03-27

### Added

- Four-spec commitment wiring for `input_spec_hash`, `model_spec_hash`, `decode_spec_hash`, and `output_spec_hash`, with verifier-side recomputation and composed manifest checking.
- Canonical commitment of the full four-spec surface, including chat template, BOS/EOS preprocessing, special-token handling, system-prompt semantics, public model identity `R_W`, adapter identity, RoPE configuration, RMSNorm epsilon, sampler identity, decode knobs, and output stopping rules.
- Explicit model-surface commitments for `n_layers`, `hidden_dim`, `vocab_size`, and `embedding_merkle_root`, with verifier-side cross-checks against the verifier key/config.
- `padding_policy` field in InputSpec and DeploymentManifest, hashed in `hash_input_spec()`, populated by the live server manifest path, and enforced fail-closed in the Python verifier path. Completes truncation/padding binding (roadmap #4).
- `decode_mode` field in DecodeSpec and DeploymentManifest, hashed in `hash_decode_spec()`, with verifier cross-check against temperature: greedy requires temp=0, sampled requires temp>0, unknown modes fail closed. Tests cover all mismatch and pass cases (roadmap #9).
- Quantization identity fields in ModelSpec: `quant_family`, `scale_derivation`, `quant_block_size`, hashed in `hash_model_spec()`, extracted from model quantization config in the live server, parsed in the Python bridge (roadmap #5-7).
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

### Removed

- Legacy V1/V2/V3 protocol framing and code paths from the kept protocol story.
- Dead full-trace artifacts and stale legacy comments that only served the removed pre-V4 protocol paths.
- Dead `requantize_bridge()` and `verify_bridge()` bridge helpers superseded by `bridge_residual_rmsnorm()`.
- Unused `kv_chain_root` commitment field and related dead code.
- Transitional reliance on weight-backed replay as part of the canonical verifier path; weight-backed replay remains debug/oracle-only.
