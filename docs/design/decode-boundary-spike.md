# Design Spike: Decode Boundary Capture

This document started as a design spike. It now serves two purposes:

1. record why the old decode path failed
2. record why the kept replacement is `LogitsProcessor`-input hidden

## Problem

The verifier's quantized lm_head replay (`final_residual → f64 RMSNorm → i8 → i32 matmul`)
produces logits that diverge from the GPU's bf16 logit path by up to ~6500 i32 units on a
~7000 range. This makes token-identity verification impossible on the quantized replay path
for Qwen W8A8 (and likely other approximate profiles).

The lm_head Freivalds matmul check is exact and stays. Only the final token-identity step
is currently unsupported.

## Empirical Results

### model.norm is NOT the decode boundary

`model.norm` returns a tuple `(residual, normed)`. Neither element matches what
`LogitsProcessor` actually receives:

- `output[0]` is the residual passthrough (identical to input) — 2/192 exact (1%)
- `output[1]` is the normed hidden — 1/192 exact (0.5%)
- Diff between `model.norm output[1]` and `LogitsProcessor args[1]`: 38–318

vLLM's model runner applies `_prune_hidden_states` (and possibly other transforms)
between `model.norm` and `LogitsProcessor`. The objects are completely different.

### LogitsProcessor input IS the decode boundary

`LogitsProcessor.forward(lm_head, hidden_states, sampling_metadata)` receives
the pruned hidden state as `args[1]`, shape `(1, 3584)`, dtype `bf16`.

Tested on 3 prompts × 64 tokens = 192 generated tokens (greedy, temperature=0):

| Path | Exact | Rate |
|------|-------|------|
| LP output logits argmax == actual token | 192/192 | 100% |
| Recomputed bf16 matmul from LP input | 192/192 | 100% |
| Recomputed fp32 matmul from LP input | 189/192 | 98.4% |

bf16 matmul on CPU reproduces GPU logits with max_diff = 0.125 (rounding noise).
fp32 is slightly worse (189/192) because the GPU computes in bf16 — fp32 introduces
a different rounding path.

### i8→i32 quantized path is dead for token identity

Even with the correct LP-input hidden state, quantizing to i8 and doing i32 matmul
produces 0/192 exact. The quantization error in lm_head (152064 × 3584 matrix)
destroys top-token ranking. This path cannot be rescued with tolerances.

The i8→i32 path remains valid for Freivalds (probabilistic matmul check), but cannot
be used for token-identity verification.

### Hook-level validation after real capture integration (2026-04-13)

After wiring LP-hidden capture into the sidecar, the capture path itself was
validated on the real Qwen W8A8 inference path:

| Check | Result |
|------|--------|
| Count alignment | exact on greedy and sampled |
| Shape / dtype | `(1, 3584)`, `bf16` |
| Reference-hook match | `max_diff = 0.0` |
| Greedy token replay | `192/192` |
| Sampled exact replay | `32/32` |
| Prefill leakage | none |

The earlier capture mismatch was a runtime bug, not a protocol bug:

- CUDA allocator reuse let the producer-side tensor storage be recycled
  before an async D2H copy finished reading it
- the kept fix snapshots on the producer stream before copying

This matters because it upgrades LP-hidden from “interesting candidate” to
“validated runtime boundary,” even though the final promotion gate is still the
real-weight E2E prover/verifier path.

## What LP-hidden solves and does not solve

### Solved: decode boundary

LP-hidden gives a cheap exact boundary for the final decode step:

- Capture point: `LogitsProcessor args[1]`, bf16, shape `(1, hidden_dim)`
- Verification: `captured_hidden @ lm_head_bf16.T → argmax == committed token`
- Cost: 7 KB/token (Qwen 7B), 42× cheaper than full-logit capture
- Evidence: 192/192 exact on greedy decode

This means per-token decode verification is plausible at modest bandwidth:
~7 MB for a 1k-token answer.

### Not solved: layer-internal computation

LP-hidden proves "this token follows from this hidden state." It does NOT prove
"this hidden state came from honest model computation."

Verifying layer internals requires either:
1. Deterministic replay of all ops (including FP non-linear ops) — requires
   pinning RMSNorm, RoPE, SiLU, softmax to exact specs that match GPU execution
2. Captured per-layer boundaries (x_attn at each layer) — ~196 KB/token,
   ~196 MB for a 1k-token answer
3. Hybrid: LP-hidden for every token's decode + deep audit of sampled tokens' layers

### Not solved: attention routing

Attention routing (softmax) is the main unresolved verification gap.

An adversary using the committed weights can manipulate attention scores/softmax
to steer the model's behavior (e.g., ignore safety instructions, bias outputs)
while passing all matmul Freivalds checks and all decode checks.

**W@V Freivalds is a consistency check, not a routing proof.** Verifying
`W @ V == attn_output` constrains the attention weights, but does not pin them.
For one head and one query position:

- W is `[seq_len]`, V is `[seq_len, d_head]`, output is `[d_head]`
- Degrees of freedom = `seq_len - rank(V)`
- When `seq_len > d_head` (most of inference), the nullspace is large
- Non-negativity and sum-to-one constraints help but do not make W unique

W@V Freivalds is useful as a supplementary check but is not a substitute for
verifying softmax itself.

## Options considered

### Option A: Captured float logits

Capture the GPU's logit tensor at the sampler input. Commit it. Verify token identity
against the captured logits + committed randomness.

**Gap in the proof story:**
- lm_head Freivalds checks: `quantized_final_hidden × lm_head == quantized_logits`
- Token identity checks: `sample(captured_float_logits, seed) == token_id`
- These operate on DIFFERENT objects — the two checks do not compose.

**Cost:** 297 KB/token (bf16) — 42× more expensive than LP-hidden.

**Status:** Deprioritized. LP-hidden is cheaper and ties token identity to lm_head
through a single object.

### Option B: Deterministic lm_head kernel

Replace the lm_head matmul with a deterministic kernel that produces exact integer
arithmetic. Both prover and verifier compute the same logits.

**Status:** Backstop. Not needed near-term if LP-hidden holds.

### ~~Option C (original): Captured post-norm hidden~~

**KILLED.** `model.norm` output does not reach the sampler. Diff 38–318.

### Option C (revised): Captured LP-input hidden

Capture the hidden state at `LogitsProcessor args[1]` (bf16, already pruned).
Verify token identity via: `captured_hidden @ lm_head_bf16.T → argmax`.

**Cost:** 7 KB/token (bf16). 42× cheaper than full-logit capture.

| Metric | LP-hidden (bf16) | Full logits (bf16) |
|--------|------------------|--------------------|
| Per token | 7 KB | 297 KB |
| 64 tokens | 0.44 MB | 18.5 MB |
| 1000 tokens | 7 MB | 297 MB |

**Verifier requirement:** bf16 lm_head matmul, not i8→i32.

**Scope of claim:** "Given this LP-hidden, the token choice is correct."
Does NOT claim the LP-hidden came from honest upstream computation.

**Status now:** this option is no longer just a design candidate. The protocol
surface already carries:

- committed `lp_hidden_bf16`
- `DecodeAcceptanceMode::LpHiddenBf16`
- verifier-side `bf16 lm_head` replay
- keygen support for `lm_head_bf16`

The remaining question is not “can this be implemented?” but “does the full
real-weight prover/verifier E2E path validate cleanly, and what is the runtime
overhead of the kept synchronous snapshot?”

**Open questions:**
- Real-weight Qwen E2E (greedy + sampled)
- Longer generations (256+ tokens) — does accuracy hold?
- More diverse prompts — edge cases, code, multilingual
- Verifier-side bf16 arithmetic pinning — Rust `half` crate matches GPU rounding?
- Sync D2H overhead vs a future safe async snapshot

## Remaining verification gaps

### Attention routing (main gap)

Softmax determines which tokens influence each position. An adversary can manipulate
attention routing while using committed weights and passing all matmul checks.

Approaches (all still open):
- Deterministic softmax spec (hard — must match GPU's FlashAttention tiling)
- Captured softmax outputs (expensive — grows with seq_len per head per layer)
- Spot-checked softmax rows (probabilistic)

### Non-linear FP ops

RMSNorm, RoPE, SiLU, and the final norm+prune path are FP operations that may not
be reproducible across GPU/CPU without exact spec pinning. These sit between verified
matmuls but are not independently verified.

Either:
- Pin exact computation specs (formula + precision + accumulation order)
- Capture outputs at boundaries

### Upstream binding of LP-hidden

LP-hidden proves the decode step. For end-to-end verification, the protocol must
also prove the LP-hidden came from honest layer computation. This requires one of
the layer-verification approaches above.

## Decision

**Kept near-term decode path: LP-hidden capture for decode-boundary verification.**

This replaces the old “unsupported” Qwen decode story in code. It does not
claim to verify layer internals or attention routing.

**Future: layer-internal and attention verification is a separate workstream.**

## Next Steps

1. Run the full real-weight Qwen E2E on the `LpHiddenBf16` path
2. Measure capture overhead for the kept synchronous snapshot
3. Test longer generations (256+ tokens) and diverse prompts
4. Benchmark retained-state and audit-payload growth
5. Pin verifier-side bf16 arithmetic (Rust `half` crate)
6. Only after that, update docs/claims to treat Qwen decode as fully shipped

## Status

- [x] Cost benchmark (Modal) — 7 KB/token for LP-hidden, 297 KB/token for logits
- [x] Boundary identification — LogitsProcessor args[1], not model.norm
- [x] Falsification of model.norm — 0.5% exact, killed
- [x] Falsification of i8→i32 path — 0/192 exact, killed
- [x] LP-input bf16 matmul validation — 192/192 exact (greedy)
- [x] Sampled decode validation
- [ ] Longer generation validation
- [x] LP-hidden capture hook implementation
- [x] Verifier bf16 matmul path
- [x] Merkle leaf binding
- [ ] Real-weight full E2E on the integrated protocol path
