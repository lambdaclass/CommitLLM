# Exact Arithmetic: Verifiable Attention via Exact Accumulation

Status: research result (2026-10-29). CPU and Apple-GPU simulation only. Not yet a product claim; see [Open questions](#5-open-questions) and the roadmap gates.

## 1. Root cause

Every open dense-model gap in CommitLLM traces back to one seam: the provider computes in GPU floating point, and the verifier replays on the CPU. Floating-point addition is not associative, so any reduction (dot product, softmax sum, `α·V`) depends on accumulation order, and FlashAttention's order depends on tiling, split-KV, batch shape and hardware. The verifier cannot reproduce that order, which is why:

- arbitrary-position attention replay leaves an honest corridor of `L∞ = 8–9` INT8 units ([`attention-gap.md`](./attention-gap.md) §1i, §2 item 13), wide enough for the consistent fake-`a` forgery ([`docs/security/redteam_audit_only.md`](../docs/security/redteam_audit_only.md));
- Qwen `Wq/Wk/Wv` Freivalds is skipped (the bridge cannot match the GPU's quantized QKV path);
- sampled decode needs `CapturedLogits` (~500 KiB/token retained) instead of recomputing logits;
- every new model, FP8, MoE routing and speculative decoding needs its own tolerance study.

[`attention-gap.md`](./attention-gap.md) §1h already measured that every backend is bit-exact run-to-run on the prover; the mismatch is GPU-vs-CPU arithmetic, not randomness.

**Fix:** make every reduction an *exact* integer sum. Integer addition is associative, so the result is independent of order, tiling, batching and hardware. All remaining steps are elementwise and are pinned to single IEEE fp32 ops (mul, add, convert, round-half-even) or integer ops. Those are identical on every device **provided** the implementation honors the spec: no FMA contraction of separate mul/add steps, IEEE round-to-nearest-even division and conversion, and a defined subnormal policy. Lookup tables (RoPE `cos`/`sin`, the `exp` LUT) are part of the spec and must ship as canonical, hash-pinned bytes rather than be regenerated with a host `libm`; in these experiments both devices received tables generated once on the same host. The spec also fixes an **input domain**: per-row maxima are clamped up to `2^-40` and must stay below `2^40` (otherwise fail closed), so every power-of-two scale, and every sum of two of them, stays in the fp32 normal range. Without this, a tiny-but-normal row pushed a score scale below `2^-126` and the CPU (`ldexp`, gradual underflow) and GPU (bit-constructed powers of two) disagreed; with it, zero, subnormal, tiny and large rows are all bit-identical across devices (§3c).

This is not the archived deterministic-attention kernel ([`docs/design/deterministic-attention-spec.md`](../docs/design/deterministic-attention-spec.md)). That kernel stayed in floating point and therefore had to freeze one reduction tree, forbid FMA and emulate `exp`; it measured `+230%` decode (`21.58 → 71.16 ms/token`, CHANGELOG). Exact integer sums need no fixed order, so a kernel keeps normal tiling and int8 tensor cores.

## 2. Spec

### 2a. Integer attention (int16 variant)

Per head, query `i`, keys `t ≤ i + offset`:

```
qi, sq = quant_rows(q, 16)          # per-row scale, applied after the d-reduction
ki_t, sk_t = quant_rows(k_t, 16)    # per-row scale, applied after the d-reduction
vi_t = round(v_t / vs)              # vs static per (layer, head, channel): a scale
                                    # inside the t-reduction would break exactness
S_t = Σ_d qi·ki_t                                  exact int
L_t = f32(S_t) · ((sq · sk_t) · scaling)           fixed fp32 op order
z_t = max_t L_t − L_t                              max is order-independent
e_t = LUT[frac(u_t)] >> int(u_t),  u_t = round(min(z_t, 64) · log2e · 2^LB)
E   = Σ_t e_t,   O_c = Σ_t e_t · vi_tc             exact int64
a_c = (f32(O_c) / f32(E)) · vs_c
```

Static V scales are calibrated on WikiText *train* with 2× headroom; evaluation uses *test*.

### 2b. Option A: bf16 attention with exact accumulation

Keeps floating-point inputs; only the sums are exact (Kulisch / Ozaki style):

- Q/K/V are the served bf16 values. Each row is converted to a 24-bit integer with a **power-of-two** scale, which is exact for every element ≥ `2^-15 ×` the row max (measured inexact fraction: `0.016–0.017%` of elements).
- Dot products and `Σ e_t·v_t` are exact int64; on hardware they are int8-limb tensor-core products (int8×int8→int32 is exact).
- `exp` is fixed point with ~fp32 accuracy (4096-entry LUT on the top fraction bits plus an integer linear term), weights carry 24 bits.
- Result rounds once: `a = (f32(O) / f32(E)) · 2^kv`.
- **Range bound.** `O_c ≤ 2^P · 2^(W−1) · n` must fit int64. With `P = W = 24` this caps context at `2^16` keys (an all-equal-score row at 65,537 keys overflows). Longer contexts need smaller `P` (e.g. `P = 16` → `2^24` keys), split accumulators, or wider sums. The int16 variant (`P = 20`, 16-bit V) is bounded at `2^28` keys. The scripts assert these bounds.

### 2c. Full exact model

[`exact-arithmetic/exact_model.py`](./exact-arithmetic/exact_model.py) extends the spec to every op of a Qwen2 forward pass:

| Op | Exact spec |
|---|---|
| Linear | int8 weights (per-channel scale), activations int16 with per-token power-of-two scale, exact int64 sum, `(f32(acc)·2^k)·w_scale (+ bias)` |
| RMSNorm | 24-bit power-of-two row quantization, exact `Σx²`, integer `isqrt`, integer division |
| RoPE | `x·cos + rot(x)·sin` with a shipped fp32 cos/sin table (no device `sin`/`cos`) |
| Attention | §2a with power-of-two scales, integer final division |
| SiLU | fixed-point `exp`, integer division for the sigmoid |
| Residual | single fp32 add |
| Sampling | integer weights from fixed-point `exp`, exact cumsum, `splitmix64(seed, step)` |

## 3. Results

Reference: the same model with fp32 eager attention. WikiText-2 test, attention swapped only (§3a–3b).

### 3a. Which widths matter (Qwen2.5-0.5B, 4 × 512 tokens, quantized Q/K/V with float softmax, static per-channel V)

| Q-K-V bits | ppl (ref 13.6970) | KL mean | top-1 agree |
|---|---|---|---|
| 8-8-8 | 14.9159 | 5.9e-2 | 89.3% |
| 16-8-16 | 14.7958 | 5.4e-2 | 90.5% |
| 8-16-8 | 13.7110 | 2.5e-3 | 97.1% |
| 16-16-8 | 13.7077 | 7.7e-4 | 98.4% |
| 12-12-12 | 13.7049 | 3.7e-4 | 99.3% |
| 16-16-16 | 13.6977 | 1.0e-6 | 100% |
| 12-12-12, integer softmax | 13.7072 | 3.7e-4 | 99.2% |
| 16-16-16, integer softmax | 13.6976 | 1.2e-6 | 100% |

K needs more than 8 bits (Qwen K outliers), then Q. 8-bit V costs more with a static per-channel scale than with per-token scales (an earlier per-token run gave 99.5% top-1 for 16-16-8). The integer softmax adds only a small error over float softmax at the same widths (here within noise; on Qwen2.5-7B at 4k, KL `1.3e-7 → 6.4e-7` and top-1 `100% → 99.98%`). Artifacts: `results/sweep_q05_v{8,12,16}.json`.

### 3b. Accuracy at scale

| Model, context | Variant | ppl ref → variant | KL mean | top-1 agree |
|---|---|---|---|---|
| Qwen2.5-0.5B, 512 | int16 | 13.6970 → 13.6976 | 1.2e-6 | 100% |
| Qwen2.5-7B, 1k (8 seqs) | int16 | 8.2439 → 8.2441 | 4.0e-6 | 99.96% |
| Qwen2.5-7B, 4k (2 seqs) | int16 | 6.5585 → 6.5585 | 6.0e-7 | 99.98% |
| Llama-3.1-8B, 1k (8 seqs) | int16 | 7.1302 → 7.1302 | 3.0e-7 | 99.98% |
| Llama-3.1-8B, 4k (2 seqs) | int16 | 7.2815 → 7.2810 | 1.2e-6 | 99.96% |
| Qwen2.5-0.5B, 512 | stock bf16 | 13.6970 → 13.6967 | 3.1e-4 | 99.12% |
| Qwen2.5-0.5B, 512 | **Option A** | 13.6970 → 13.6981 | 3.1e-4 | 99.22% |
| Qwen2.5-7B, 1k (8 seqs) | stock bf16 | 8.2439 → 8.2376 | 8.6e-5 | 99.60% |
| Qwen2.5-7B, 1k (8 seqs) | **Option A** | 8.2439 → 8.2377 | 8.2e-5 | 99.58% |
| Qwen2.5-7B, 4k (2 seqs) | stock bf16 | 6.5585 → 6.5580 | 2.2e-4 | 99.49% |
| Qwen2.5-7B, 4k (2 seqs) | **Option A** | 6.5585 → 6.5585 | 2.2e-4 | 99.43% |
| Llama-3.1-8B, 1k (8 seqs) | stock bf16 | 7.1302 → 7.1305 | 2.8e-5 | 99.84% |
| Llama-3.1-8B, 1k (8 seqs) | **Option A** | 7.1302 → 7.1302 | 2.8e-5 | 99.83% |
| Llama-3.1-8B, 4k (2 seqs) | stock bf16 | 7.2815 → 7.2823 | 2.8e-5 | 99.73% |
| Llama-3.1-8B, 4k (2 seqs) | **Option A** | 7.2815 → 7.2814 | 2.9e-5 | 99.72% |

"stock bf16" is a CPU emulation of FlashAttention numerics (bf16 Q/K/V, fp32 softmax, `P` cast to bf16 before `P·V`), not the real kernel and not full bf16 inference; only attention is swapped, the rest of the model runs in fp32. Samples are small (2–8 WikiText sequences, 5 generation prompts): these are near-parity signals, not a quality evaluation. Option A tracks stock bf16 almost exactly; on Qwen-7B and Llama-8B its greedy generations first diverge from fp32 at the same token indices as stock (`[8, 24, 48, 48, 23]` and `[8, 48, 48, 48, 14]` of 48). The int16 variant is *more* accurate than stock bf16 because 16 bits per value is finer than bf16's 8-bit mantissa.

Raw outputs: [`exact-arithmetic/results/`](./exact-arithmetic/results/). The Qwen-7B 1k int16 run used an earlier final step (`f64` division instead of `f32`); all others use the spec as written.

### 3c. Bit-exactness

On real Q/K/V captured from the middle layer of each model, the vectorized spec equals (a) an independent per-row reimplementation in numpy fp32 + Python integers, keys visited one at a time, and (b) a two-pass evaluation over random tiles in random order: **0 mismatches** in every run (int16 and Option A).

Cross-hardware, CPU vs Apple GPU (MPS), [`mps_check.py`](./exact-arithmetic/mps_check.py): the GPU computes integer dot products with 8-bit-limb fp32 matmuls (every partial sum an integer `< 2^24`), the way an int8 tensor-core kernel would; the CPU uses float64 BLAS. Inputs are random with outlier K channels plus edge cases: all-zero rows, subnormal entries, a row below the `2^-40` domain floor, and a large row. Output: `results/mps_check.txt`.

| | bit-identical outputs |
|---|---|
| int16 integer attention | **100%** |
| Option A | **100%** |
| ordinary fp32 attention (control) | 46.8% |

### 3d. Full exact model: regenerate-on-challenge

[`exact_model.py`](./exact-arithmetic/exact_model.py), Qwen2.5-0.5B, chat prompt + 40 generated tokens. CPU uses float64-BLAS exact matmuls; MPS uses 8-bit-limb fp32 matmuls and processes the prompt in chunks `5 / 12 / rest`. Each device then *recomputes* prompt + output in one pass (teacher forcing) and re-derives the tokens.

| Check | greedy | sampled (T = 0.8, seed 42) |
|---|---|---|
| CPU vs MPS tokens | identical | identical |
| CPU vs MPS hidden states (position × layer) | 0 / 1920 differ | 0 / 1920 differ |
| decode vs one-pass recompute, each device | 0 / 1920 differ | 0 / 1920 differ |
| tokens re-derived from recompute | all equal | all equal |
| SHA-256 over all hidden states, first 16 hex digits (4 runs) | `62f95c77…` | `d5894d9c…` |

Recompute took `~0.9 s` vs `~4.4 s` for decode on CPU. Accuracy of the full spec (int8 weights, int16 activations) vs HF fp32 on 512 test tokens: ppl `12.99 → 12.94`, top-1 `95.7%`, KL `0.005`; this gap mixes W8 weights, A16 activations and the fixed-point norm/SiLU approximations; no matched W8A16 float baseline was run, so it is not attributed further. Static V scales clipped 0 values. The hash is one sequential SHA-256 used only to compare runs; it is not the commitment scheme, which needs a Merkle tree with per-`(token, layer)` openings (E4).

## 4. Protocol implications

If the NVIDIA gates below pass:

1. **Attention becomes recomputable, not just audited.** The verifier recomputes any challenged `(token, layer)` attention output exactly on CPU from opened Q/K/V. This proves attention *relative to the opened Q/K/V*. Closing the fake-`a` hole also requires binding those inputs: the challenged Q and every opened K/V row must be tied to committed hidden states (Freivalds on `Wq/Wk/Wv` plus the exact bridge, E3), and those hidden states to the committed prefix. A K/V row whose provenance is only sampled can still be fabricated at unsampled positions; the detection probability of that has to be analyzed, not assumed.
2. **No tolerances.** If the spec is implemented exactly (§1), an honest provider always matches, so any mismatch on a checked value is evidence of cheating.
3. **Regenerate instead of retain.** Because decode and one-pass recompute are bit-identical, the provider can hash per-layer hidden states on GPU during serving, keep only `(prompt, params, seed, tokens, roots)`, and regenerate any trace on challenge. This would replace `CapturedLogits` retention (~500 KiB/token) and most of the commit/finalize cost (`55–100 ms/token`). Pre-challenge roots stop the provider from changing its story; they do not by themselves prove the committed trace is correct. Correctness still comes from the opened checks, so the soundness argument is the same commit-then-open one as today, with the same need to bind every transition a check relies on. A full write-up of this binding is a prerequisite for E4.
4. **Generality.** MoE routing (top-k on exact router logits, with an explicit tie-break such as lowest expert index), speculative-decoding acceptance and FP8 inputs (small integers times powers of two) can be made deterministic by the same construction. None of these is implemented yet.

Answers will not be token-identical to stock FlashAttention (no verifiable kernel can be); the bar is quality parity with bf16, which §3b shows.

## 5. Open questions

1. **NVIDIA bit-exactness.** Only CPU vs Apple GPU is tested. Integer arithmetic makes this very likely, but a CUDA/Triton kernel must be checked against the CPU spec, built without FMA contraction for the fp32 steps.
2. **Performance (go/no-go).** Unmeasured; rough accounting only. Decode attention is memory-bound and reads the same KV bytes (bf16 or int16 cache). Exact softmax needs the row max before `exp`; spilling fp32 scores costs a write plus a read per *query* head, about `6%` of the KV read for Llama-8B GQA (32 query vs 8 KV heads). Int8-limb products cost 4 MMAs per int16×int16 MAC for `Q·K`, and 6 for `e·V` with 20-bit weights (4 if weights are capped at 16 bits); Option A's 24-bit operands cost more (9 and 12). At int8's 2× bf16 rate, prefill `Q·K` and `e·V` cost roughly `2–3×` stock tensor-core time, plus a second `Q·K` pass for the max unless a cheap bounding pass is used, i.e. roughly `2.5–4×` stock attention time. For Llama-8B that is very roughly `+10–30%` prefill at 4k context and much more at 32k. Decode overhead is likely single-digit to low-double-digit percent on attention. These numbers are not reliable enough to plan on; E1 replaces them with measurements. Int8 tensor-core throughput on newer parts (some Blackwell SKUs) must also be checked.
3. **Decode policies and commitments.** The prototype covers greedy and full-vocabulary temperature sampling only. Top-k, top-p, repetition/logit processors and EOS handling need exact integer specs (with defined tie-breaking). The GPU hashing and Merkle commitment behind regenerate-on-challenge are neither implemented nor benchmarked; the prototype copies hidden states to the host and hashes there, so the claimed commit-cost savings are projected, not measured.
4. **Bridge into attention.** Dequant, bias and RoPE between the INT8 GEMM and attention must follow the same elementwise spec (this is where the Qwen QKV gap lives).
5. **KV provenance at scale.** Exact attention is only as good as the opened K/V rows. Checking all rows is cheap (one Freivalds per row), but the opening is ~16 MB per challenged layer at 4k context on Llama-8B; head sampling reduces it.
6. **Accuracy breadth.** Contexts beyond 4k; real bf16 inference with the actual FlashAttention kernel as the baseline; downstream evals, not just perplexity/KL on a handful of sequences.

## 6. Reproduce

Local CPU (and Apple GPU for the cross-hardware checks); needs `torch`, `transformers`, `datasets`.

```bash
cd research/exact-arithmetic
python run.py --model Qwen/Qwen2.5-0.5B-Instruct --seqs 4 --seqlen 512 --variants stockbf16,exactA,int_16-16-16
python run.py --model Qwen/Qwen2.5-7B-Instruct --seqs 8 --seqlen 1024 --gen 48 --variants stockbf16,exactA
python mps_check.py      # CPU vs MPS bit-exactness (needs Apple Silicon)
python exact_model.py    # full exact model: CPU vs MPS, decode vs recompute
```

Variant names: `ref`, `stockbf16`, `exactA`, `qfloat_Q-K-V` (quantized, float softmax), `int_Q-K-V[_lbX_pY]` (integer spec; `lb` = LUT index bits, `p` = weight bits).
