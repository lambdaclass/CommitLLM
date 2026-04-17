# Design Note: Decode Boundary Evolution

This file is kept as a design-history note. The spike started with LP-hidden.
The shipped sampled-decode path is now `CapturedLogits`.

## What the spike proved

The original decode verifier was using the wrong object.

Bad path:

`final_residual -> RMSNorm replay -> quantized lm_head replay`

Correct runtime boundary:

`LogitsProcessor` input hidden (`args[1]`)

That LP-hidden result mattered because it:

- falsified `model.norm` output as the decode boundary
- proved the verifier had to move to a later runtime boundary
- enabled end-to-end decode-boundary experiments on real weights

## Why LP-hidden was not the final answer

LP-hidden replay is much better than quantized replay, but broad sampled stress
still showed a residual failure tail:

- i32 sampled replay: `239/300 = 79.7%`
- LP-hidden bf16 replay: `296/300 = 98.7%`

That residual gap is not a simple implementation bug. It is the remaining
CPU-replay-vs-GPU-tensor-core arithmetic mismatch in the `lm_head` path.

So LP-hidden remains:

- a correct intermediate decode boundary
- useful for greedy / cheaper replay modes
- not the kept exact sampled-decode rule

## Shipped answer: CapturedLogits

The kept sampled-decode design is:

1. capture the exact GPU logits seen by the sampler
2. commit them in the retained leaf
3. verify exact token sampling from those logits
4. Freivalds-bind them back to `lp_hidden × lm_head_bf16`

This removes full CPU `lm_head` replay from the sampled-token acceptance rule.

### Why this is the right trade

- **Correctness**: exact sampled decode
- **Verifier CPU**: cheap (two dot products)
- **Cost moved to**: retained-state size on the prover

Approximate retained costs:

| Object | Llama 8B | Qwen 7B |
|---|---:|---:|
| LP-hidden bf16 | ~8 KiB/token | ~7 KiB/token |
| Captured logits f32 | ~501 KiB/token | ~594 KiB/token |

So LP-hidden is still the more compact boundary, but captured logits is the
correct exactness boundary for sampled decode.

## Validation state

CapturedLogits decode is GPU-validated end-to-end on both dense families:

- **Qwen**: `3/3` greedy, `20/20` sampled, `17,336/17,336` decode checks passed
- **Llama**: `3/3` greedy, `20/20` sampled, `23,335/23,335` decode checks passed

Tamper rejection passes on both.

## What remains open

This closes decode-side sampled exactness. It does **not** close:

- arbitrary-position attention
- retained-state economics
- key/decode-artifact packaging/distribution

The next protocol decision is therefore:

1. benchmark captured-logits economics in steady state
2. close arbitrary-position attention via:
   - a strict FlashAttention/kernel-aligned witness, or
   - deterministic attention kernels

## Bottom line

LP-hidden was the right discovery. CapturedLogits is the right shipped sampled-decode protocol.
