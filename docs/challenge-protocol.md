# Challenge Protocol Specification

This document specifies the exact timing, derivation, and flow of audit challenges in the VeriLM protocol.

## Overview

The audit protocol is interactive: the verifier generates a challenge **after** seeing the prover's commitment, so the prover cannot know which tokens or layers will be audited at commitment time.

```
Prover                              Verifier
  |                                    |
  |-- BatchCommitment (roots, hashes) ->|
  |                                    |-- generates challenge_seed (32 bytes)
  |<-- AuditChallenge (token, layers) -|
  |                                    |
  |-- V4AuditResponse (opening) ------>|
  |                                    |-- verify_v4()
  |                                    |-- verdict: Pass / Fail
```

## Challenge Seed

The verifier generates a fresh 32-byte `challenge_seed` uniformly at random **after** receiving the `BatchCommitment`. This seed must not be predictable by the prover before commitment.

## Token Index Derivation

```
token_index = u32_le(SHA256("vi-audit-token-v1" || challenge_seed)[0..4]) % n_tokens
```

- Domain separator: `"vi-audit-token-v1"` (17 bytes)
- Input: `challenge_seed` (32 bytes)
- Output: first 4 bytes of SHA-256 digest, interpreted as little-endian u32, reduced mod `n_tokens`

This selects which token the prover must open.

## Layer Index Derivation

### Full Tier

All layers are audited:

```
layer_indices = [0, 1, 2, ..., n_layers-1]
```

### Routine Tier

A contiguous prefix of layers `[0..=L_max]` is audited. The prefix depth is derived from the challenge seed so the prover cannot predict how deep the audit will reach.

```
hash = SHA256("vi-audit-prefix-v1" || challenge_seed || token_index_le32)
raw  = u32_le(hash[0..4])
L_max = max(min_prefix - 1, raw % n_layers)
layer_indices = [0, 1, 2, ..., L_max]
```

Where:
- Domain separator: `"vi-audit-prefix-v1"` (18 bytes)
- `token_index_le32`: the derived token index as 4 little-endian bytes
- `min_prefix = min(10, n_layers)` — guarantees at least 10 layers (or all layers if fewer exist)

The contiguous prefix structure means the verifier always checks the full chain from layer 0 through the audited depth, preventing a prover from selectively hiding early-layer tampering.

## Audit Tiers

| Tier | Layer coverage | Use case |
|------|---------------|----------|
| **Full** | All layers (`0..n_layers`) | Deep audit: complete verification |
| **Routine** | Contiguous prefix (`0..=L_max`) | Routine audit: statistical coverage |

The tier selection is a verifier policy decision, not part of the challenge derivation itself. A typical deployment audits ~90% of requests at Routine tier and ~10% at Full tier.

## Per-Token Sampling Seed Derivation

For sampled (non-greedy) verification, each token gets a deterministic PRNG seed:

```
token_seed = SHA256("vi-sample-v1" || batch_seed || token_index_le32)
```

- Domain separator: `"vi-sample-v1"` (12 bytes)
- `batch_seed`: the per-request sampling seed, committed before inference as `seed_commitment = SHA256("vi-seed-v1" || batch_seed)` and revealed at audit time
- `token_index_le32`: the token position as 4 little-endian bytes

The `token_seed` seeds a ChaCha20Rng instance for categorical sampling.

## Freivalds Block Coefficients

For batched Freivalds checks, per-block random coefficients are derived from the verifier's secret key seed:

```
coeff = Fp(u32_le(SHA256("vi-block-coeff-v2" || key_seed || layer_le32 || matrix_le32 || block_le32)[0..4]))
```

These coefficients are verifier-secret: they are derived from the key seed that the prover never sees.

## Domain Separation

All derivation functions use unique domain separators to prevent cross-function collisions:

| Function | Domain separator | Version |
|----------|-----------------|---------|
| Token index | `vi-audit-token-v1` | v1 |
| Layer depth (routine) | `vi-audit-prefix-v1` | v1 |
| Sampling seed | `vi-sample-v1` | v1 |
| Block coefficients | `vi-block-coeff-v2` | v2 |
| IO chain hash | `vi-io-v4` | v4 |
| IO genesis | `vi-io-genesis-v4` | v4 |
| Manifest hash | `vi-manifest-v4` | v4 |
| Retained-state leaf | `vi-retained-v1` | v1 |
| Final-residual leaf | `vi-retained-fr-v1` | v1 |
| Seed commitment | `vi-seed-v1` | v1 |
| Prompt hash | `vi-prompt-v1` | v1 |
| Embedding row | `vi-embedding-v1` | v1 |
| Weight chain | `vi-weight-chain-v1` | v1 |

## Binary Wire Format

### V4 Audit Response

```
[VV4A]  (4 bytes, magic)
[zstd-compressed bincode(V4AuditResponse)]
```

### Verifier Key

```
[VKEY]  (4 bytes, magic)
[bincode(VerifierKey)]
```

Unknown magic bytes must be rejected (fail-closed).

## Golden Vectors

Conformance vectors are maintained in `crates/verilm-test-vectors/tests/golden_conformance.rs`. These pin:

- `build_audit_challenge` outputs for 4 seed/token/layer configurations
- `hash_manifest` for a canonical deployment manifest with architecture fields
- `hash_model_spec` for the corresponding model spec
- End-to-end `commit_minimal` → `open_v4` → `verify_v4` with pinned roots and verdict
- Binary format roundtrip, truncation rejection, cross-format rejection
