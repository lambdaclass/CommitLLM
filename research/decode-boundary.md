# Decode Boundary Findings

Dates:
- initial diagnosis: 2026-04-12
- protocol integration update: 2026-04-13

## Problem

Qwen W8A8 could pass exact `lm_head` Freivalds while still failing final token-identity
checks on honest runs. The original verifier path used:

`final_residual -> RMSNorm replay -> i8 -> i32 lm_head matmul -> argmax/sample`

That path turned out to be wrong for token identity on Qwen. Honest-gap measurement on
the quantized replay logits showed a heavy tail from `0` to `6556` i32 units on a ~7000
range, which killed any tolerance-based acceptance rule.

The open question was whether there existed a cheap captured boundary for the decode step,
analogous to captured `x_attn` on the attention side.

## Result

There is a viable decode boundary:

- **Not** `model.norm` output
- **Yes** `LogitsProcessor` input hidden state (`args[1]`)

This hidden state is already pruned to the sampled row and sits immediately before the
final logits computation used by the sampler.

This is no longer just a hypothesis. The protocol surface now carries LP hidden in code,
and the remaining gate is the full real-weight end-to-end prover/verifier run.

## Diagnostics

### 1. `model.norm` is not the decode boundary

`model.norm` returns a tuple, but neither output is the object that reaches the sampler.

Observed on Qwen W8A8:

- `model.norm output[0]` is residual passthrough
- `model.norm output[1]` is normed hidden
- `model.norm output[1]` vs `LogitsProcessor args[1]` differed by `38–318`

Using `model.norm` output for token replay gave only `1/192` exact matches.

Conclusion:

- `model.norm` is not the decode boundary
- there is a transform/pruning step between `model.norm` and `LogitsProcessor`

### 2. `LogitsProcessor` input is the real decode boundary

Diagnostic on 3 prompts x 64 generated tokens (`192` total), greedy decode:

| Path | Exact | Rate |
|---|---:|---:|
| `LogitsProcessor` output logits argmax | 192/192 | 100% |
| Recomputed `bf16` matmul from `LogitsProcessor args[1]` | 192/192 | 100% |
| Recomputed `fp32` matmul from `LogitsProcessor args[1]` | 189/192 | 98.4% |
| Recomputed `i8 -> i32` matmul from `LogitsProcessor args[1]` | 0/192 | 0% |

Observed max diff between recomputed `bf16` logits and `LogitsProcessor` output:

- `max_diff = 0.125`

This is negligible and consistent with `bf16` rounding noise.

Conclusion:

- the hidden state at `LogitsProcessor args[1]` is the correct decode boundary
- verifier-side `bf16 lm_head` replay from that hidden reproduces the GPU token choice on the measured set
- the quantized `i8 -> i32` surrogate is unsuitable for token identity

### 3. Capture-hook validation after integration

After wiring LP-hidden through the sidecar capture path, the hook itself was validated
on the real Qwen W8A8 inference loop:

| Check | Result |
|---|---|
| Count alignment | exact on greedy and sampled runs |
| Shape / dtype | `(1, 3584)`, `bf16` |
| Reference-hook match | `max_diff = 0.0` |
| Greedy replay | `192/192` |
| Sampled replay | `32/32` exact |
| Prefill leakage | none |

The earlier mismatch was caused by runtime behavior, not protocol semantics:

- CUDA allocator reuse recycled the producer-side tensor storage
- an async D2H copy stream then read from stale storage
- the kept fix snapshots on the producer stream before copy

This matters because it upgrades LP hidden from “right tensor in a diagnostic” to
“correctly capturable protocol object.”

## Cost

For Qwen:

- hidden dim = `3584`
- `bf16` hidden = `3584 * 2 = 7168 bytes ~= 7 KB/token`

For comparison, full logits capture would cost:

- vocab size ~= `152064`
- `bf16` logits = `152064 * 2 = 304128 bytes ~= 297 KB/token`

So captured LP-input hidden is roughly `42x` cheaper than captured `bf16` logits while
still being sufficient to reconstruct logits for token verification.

## Security Interpretation

Captured LP-input hidden is not “free evidence.” It only becomes a strong protocol object
if all of the following hold:

1. the hidden is committed in retained state / Merkle binding
2. the verifier replays `bf16 lm_head` from that exact hidden
3. the token is checked against the committed decode policy and seed

If those conditions are met, LP-input hidden is at least as useful for decode verification
as captured logits, and in one sense cleaner: logits are derived from the hidden via the
committed `lm_head`, rather than being a separate captured witness with a weaker tie to the
linear check.

This does **not** by itself prove the whole upstream model execution. It proves the decode
step given that boundary object.

That distinction remains the central protocol boundary:

- LP hidden is enough to prove the final `hidden -> token` transition
- it is not enough to prove how the hidden was produced
- attention-side and prefix-side verification remain separate work

## Protocol Implication

The decode-strengthening path is now:

1. capture `LogitsProcessor args[1]` for generated tokens only
2. commit it into retained state
3. verify token identity via `captured_lp_hidden -> bf16 lm_head -> canonical sampler`

Not:

- tolerance tuning
- `model.norm` capture
- full-logit capture as the default path

Deterministic `lm_head` kernels remain the clean long-term backstop, but the measured data
now says the first practical local-boundary path is captured LP-input hidden.

As of 2026-04-13, the code path already includes:

- committed `lp_hidden_bf16`
- retained-state / Merkle binding for LP hidden
- `VerifierKey.lm_head_bf16`
- `DecodeAcceptanceMode::LpHiddenBf16`
- verifier-side bf16 replay for greedy and sampled decode

So the protocol question has changed from “should we try this?” to “does the full
real-weight E2E prove cleanly, and what does the kept capture path cost?”

## Open Questions

- real-weight Qwen E2E on the integrated protocol path
- longer generations
- more diverse prompts
- exact verifier-side `bf16` arithmetic pinning
- retained-state / opened-payload benchmark after actual LP-hidden capture lands
- capture overhead of the kept synchronous snapshot vs a future safe async variant

## Bottom Line

The decode problem was not “Qwen is too numerically noisy to verify.” The problem was that
the verifier was using the wrong object. Once the boundary was moved to `LogitsProcessor`
input hidden, `bf16 lm_head` replay matched the GPU token choice exactly on the measured
greedy and sampled runs, at a cost of only about `7 KB/token`. The decode side of the
protocol is now on the right architectural path. The main open verifier problem is no
longer decode; it is attention-side and prefix-side computation.  
