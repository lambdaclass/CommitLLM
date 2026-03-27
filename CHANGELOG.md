# Changelog

This changelog tracks the kept canonical VeriLM protocol and its major implementation milestones.

## 2026-03-27

### Added

- Four-spec commitment wiring for `input_spec_hash`, `model_spec_hash`, `decode_spec_hash`, and `output_spec_hash`, with verifier-side recomputation and composed manifest checking.
- Canonical commitment of the full four-spec surface, including chat template, BOS/EOS preprocessing, special-token handling, system-prompt semantics, public model identity `R_W`, adapter identity, RoPE configuration, RMSNorm epsilon, sampler identity, decode knobs, and output stopping rules.
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
- Tokenizer identity commitment based on canonicalized `tokenizer.json` content via `backend_tokenizer.to_str()`, with deterministic JSON normalization and legacy vocab-only fallback.
- Explicit `weight_hash` / `R_W` cross-checking in the canonical verifier path, in addition to model-spec hash binding.

### Changed

- Terminology and documentation framing now consistently describe Freivalds checks as information-theoretically sound / statistically sound rather than cryptographic.
- Runtime-populated input, decode, and output policy fields now flow into the committed four-spec surface instead of relying on static defaults.
- The live canonical sampled path is replayable end to end under the committed decode spec instead of depending on incidental stock serving behavior.
- The key-only retained-state path is now the canonical verifier path for the kept protocol.
- The canonical verifier path now checks reconstructed prompt tokens against the committed input path and the embedding path.
- Output-policy enforcement now covers `eos_policy`, `min_tokens`, `ignore_eos`, stop conditions, special-token stripping, detokenization policy, cleanup behavior, and fail-closed handling for unsupported or unknown policy values.
- Input-policy replay is fail-closed for unknown `bos_eos_policy`, `special_token_policy`, `truncation_policy`, and `detokenization_policy`.
- The verifier now cross-checks committed model-spec values against verifier-key values for `rmsnorm_eps`, `rope_config_hash`, and `weight_hash`.
- Decode-spec handling is fail-closed for unsupported sampler versions and unsupported non-default decode features; only the canonical sampler version is accepted and supported decode features are replayed.
- Audit retrieval now preserves claimed output text end to end so detokenization checks can run in the canonical verifier flow.
- The roadmap and protocol framing now treat the kept canonical protocol as structurally complete, with remaining work focused on hardening, conformance, benchmarks, and documentation.

### Removed

- Legacy V1/V2/V3 protocol framing and code paths from the kept protocol story.
- Dead full-trace artifacts and stale legacy comments that only served the removed pre-V4 protocol paths.
- Unused `kv_chain_root` commitment field and related dead code.
- Transitional reliance on weight-backed replay as part of the canonical verifier path; weight-backed replay remains debug/oracle-only.
