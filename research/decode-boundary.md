# Decode Boundary Research

Dates:
- initial decode-boundary investigation: 2026-04-12
- LP-hidden protocol integration: 2026-04-13
- replay-path A/B diagnostic: 2026-04-17
- captured-logits E2E validation: 2026-04-17

## Executive summary

The decode-side problem is no longer "find a better replay." That work is done.

What we learned:

1. The old quantized replay path (`i8 -> i32 lm_head`) is unusable for sampled decode.
2. `LogitsProcessor` input hidden state (`LP-hidden`) is the correct runtime boundary, and capturing it was a necessary intermediate step.
3. LP-hidden replay is still not exact enough for sampled decode under broad stress because CPU replay arithmetic does not exactly match GPU tensor-core arithmetic.
4. The kept exact sampled-decode path is now **CapturedLogits**:
   - capture the exact GPU logits the sampler used
   - verify the exact sampled token from those logits
   - Freivalds-bind those logits back to `lp_hidden × lm_head_bf16`

Decode-side sampled exactness is therefore closed by boundary capture, not by replay.

## Chronology

### 1. Quantized replay failed

The original verifier path used:

`final_residual -> RMSNorm replay -> i8 -> i32 lm_head -> argmax/sample`

This failed badly on Qwen W8A8 and later on Llama sampled decode:

- the quantized logits diverged enough from the GPU bf16 logits to reorder the sampled distribution
- Llama sampled `ExactTokenIdentity` on the cheap path passed only `239/300 = 79.7%`

Conclusion: replaying approximate logits is not a viable sampled-decode rule.

### 2. LP-hidden identified the right boundary

`model.norm` output was not the decode boundary. The object that actually reaches the sampler is:

`LogitsProcessor.forward(lm_head, hidden_states, sampling_metadata)`

specifically `args[1]`, the pruned hidden state at the sampler boundary.

Capturing LP-hidden was an important protocol result:

- it proved the verifier had been using the wrong object
- it gave a much cheaper decode boundary than full logits
- it enabled real end-to-end decode-boundary experiments

LP-hidden cost:

- Llama 8B: `4096 × 2 bytes ≈ 8 KiB/token`
- Qwen 7B: `3584 × 2 bytes ≈ 7 KiB/token`

### 3. LP-hidden replay improved things, but did not solve sampled exactness

Broad sampled A/B diagnostic on Llama (`300` runs, `T=0.8`, `top_k=50`, `top_p=0.9`):

| Path | Pass rate |
|---|---:|
| i32 ExactTokenIdentity | `239/300 = 79.7%` |
| LP-hidden bf16 replay | `296/300 = 98.7%` |

This proved two things:

- LP-hidden was directionally right
- replay arithmetic was still wrong for the sampling tail

The residual `4/300` failures are not about quantization anymore. They are about CPU replay arithmetic vs GPU tensor-core accumulation order.

Conclusion: LP-hidden remains useful as an intermediate/local boundary and for cheaper greedy paths, but **it is not the kept sampled-decode exactness rule**.

## Kept answer: CapturedLogits

### Proof structure

The kept sampled-decode protocol is:

1. capture the exact GPU f32 logits used by the sampler
2. commit them into the retained Merkle leaf
3. verify exact token sampling from those logits
4. Freivalds-bind them back to `lp_hidden × lm_head_bf16`

This avoids full CPU `lm_head` replay and removes the replay-mismatch class from the token-choice check.

### Why this is better than LP-hidden for sampled decode

LP-hidden is smaller, but it still requires replaying `lm_head`.
Captured logits are larger, but they are the exact sampling object.

For sampled decode, exactness matters more than compactness.

### GPU E2E result

CapturedLogits is now GPU-validated end-to-end on both dense families:

| Model | Greedy | Sampled | Decode checks |
|---|---:|---:|---:|
| Qwen 7B W8A8 | `3/3` | `20/20` | `17,336/17,336` |
| Llama 8B W8A8 | `3/3` | `20/20` | `23,335/23,335` |

Tamper rejection also passes on both models.

Conclusion: sampled decode exactness is now closed on the shipped path.

## Economics

CapturedLogits shifts the problem from verifier CPU to retained-state economics.

Per-token retained bytes:

| Object | Llama 8B | Qwen 7B |
|---|---:|---:|
| LP-hidden bf16 | ~8 KiB | ~7 KiB |
| Captured logits f32 | ~501 KiB | ~594 KiB |

For a 256-token answer:

- Llama: ~125 MiB retained
- Qwen: ~149 MiB retained

Important distinction:

- retained-state cost is per generated answer
- audit bandwidth is per **opened token**, not per answer

Opened-token bandwidth is roughly:

- Llama: ~0.5 MiB per challenged token
- Qwen: ~0.6 MiB per challenged token

Verifier cost is cheap again:

- two dot products per challenged token
- no full `lm_head` CPU matmul

## Current policy

- **Sampled decode**: `CapturedLogits` only
- **Greedy decode**: cheaper validated paths remain acceptable where explicitly supported
- **Attention**: still the main correctness blocker

## Implications

This changes the architecture:

- sampled decode is no longer an open correctness problem
- replay tuning is no longer the right decode work
- the next decode-side question is economics
- the next main correctness problem is arbitrary-position attention

## Open work

1. benchmark captured-logits economics in steady state
2. cache key + decode artifact so keygen is treated as offline
3. define production audit policy around retained-state cost
4. close arbitrary-position attention with either:
   - a strict FlashAttention/kernel-aligned witness, or
   - deterministic attention kernels

## Bottom line

LP-hidden was the correct intermediate discovery because it found the real runtime boundary and killed the wrong replay object. But it was not the final answer. The shipped sampled-decode protocol is now `CapturedLogits`: exact GPU-logit sampling plus algebraic binding back to `lm_head`.
