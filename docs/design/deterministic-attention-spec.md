# Deterministic Attention Arithmetic Spec

## Status: FROZEN (2026-04-19)

**Validated**: 10,000/10,000 randomized parity tests + 45/45 targeted edge cases,
GPU (A100, CUDA 12.4) vs CPU (Rust), bit-exact on outputs and softmax weights.

**Any change to the arithmetic below is a protocol-breaking change.**

## Motivation

Stock FlashAttention-2 is not reproducible across GPU↔CPU:
- Tile-order-dependent FP32 accumulation (non-associative addition)
- Hardware `exp2` precision differs between GPU MUFU and CPU libm
- FA2 online-softmax rescaling accumulates rounding errors per tile

Measured gap: L-inf 199–254 (i8 scale) between GPU attention output and
best-effort CPU replay. This is the attention gap.

StockBounded statistical certification (top-k + tail bound) was measured
at 23–28% certification rate (k=16, threshold=0.9, temp=0.8). Too weak
for mainline.

**Goal**: A custom attention kernel where both prover (GPU) and verifier
(CPU) compute identical output, bit-for-bit.

## Scope

This spec covers **single-query decode attention** only:
- One query token attending to a KV cache of length `seq_len`
- GQA (grouped query attention): `n_q_heads` query heads, `n_kv_heads` KV heads
- Head dimension `d_head` (128 for Llama/Qwen)

Prefill (batched Q) is out of scope for v1.

## Inputs

All inputs are the **exact outputs of the QKV projection**, post-RoPE.
For W8A8 models, the QKV projection produces bf16 (Llama) or fp16 (varies
by model) values via `cutlass_scaled_mm`.

```
Q:  bf16[n_q_heads, d_head]              — query for the current token (post-RoPE)
K:  bf16[seq_len, n_kv_heads, d_head]    — KV cache keys (post-RoPE)
V:  bf16[seq_len, n_kv_heads, d_head]    — KV cache values
```

**Input boundary for v1**: Prover captures and commits post-RoPE Q, K, V
in bf16. Verifier receives the same bf16 values. RoPE canonicalization is
a separate concern — v1 sidesteps it by committing post-RoPE inputs.

bf16 → f32 upcast is lossless (bf16 is a subset of f32 — upper 16 bits
of the IEEE 754 representation, zero-extended to 32 bits).

## Frozen Arithmetic Contract

The following arithmetic is the protocol definition. Both prover (GPU CUDA
kernel) and verifier (CPU Rust reference) MUST implement these exact operations.

### Constants

| Name | Value | Hex (f32 bits) | Notes |
|------|-------|----------------|-------|
| LOG2_E | `1.4426950216293335_f32` | `0x3FB8AA3B` | log2(e) rounded to f32 |
| POLY_C1 | `0.693147182_f32` | `0x3F317218` | Minimax coeff (degree 1) |
| POLY_C2 | `0.240226507_f32` | `0x3E75FDF0` | Minimax coeff (degree 2) |
| POLY_C3 | `0.055504109_f32` | `0x3D6354F1` | Minimax coeff (degree 3) |
| POLY_C4 | `0.009618129_f32` | `0x3C1D9533` | Minimax coeff (degree 4) |

### Rule 1: No FMA

All multiply-add sequences use **separate multiply then add** operations.
No fused multiply-add (FMA) at any point.

- GPU: `--fmad=false` compiler flag + `__fmul_rn` / `__fadd_rn` intrinsics
- CPU (Rust): default f32 arithmetic (Rust does not fuse f32 mul+add)

### Rule 2: Rounding — round-to-nearest-even

All rounding in range reduction uses **round-to-nearest-even** (IEEE 754
default, banker's rounding).

- GPU: `rintf()` — round-to-nearest-even
- CPU (Rust): `f32::round_ties_even()` — NOT `f32::round()` which is
  round-half-away-from-zero

**This distinction matters.** The stress test found 3/10,000 failures
before this was corrected. `f32::round()` and `rintf()` disagree on
half-integer inputs (0.5, 1.5, 2.5, ...).

### Rule 3: Precomputed inv_sqrt_d

The softmax scale factor `1/sqrt(d_head)` is computed **once** by the
caller and passed as the same f32 value to both GPU and CPU:

```
inv_sqrt_d: f32 = 1.0_f32 / sqrt(d_head as f32)
```

Both sides receive the same f32 bits. Neither side recomputes this value.

### Rule 4: Frozen parallel reduction tree (v2)

> **v1 (serial)**: Left-to-right sequential accumulation. Replaced in v2.
>
> **v2 (current)**: Fixed binary tree reduction. Both prover and verifier
> use the same tree structure. The tree is deterministic because the
> pair order at each level is fixed.

Reduction contract:

1. **Padding**: Logical length is rounded up to next power of two.
   Padding elements use the operation's identity:
   - sum: `0.0` (positive zero, `0x00000000`)
   - max: `-inf` (`0xFF800000`)
   Padding is appended to the END of the input.

2. **Tree structure**: At each level, adjacent pairs combine:
   ```
   for stride = padded_len/2, padded_len/4, ..., 1:
       for i = 0..stride:
           buf[i] = op(buf[i], buf[i + stride])
   ```
   Left operand always has the lower index.

3. **Pair order**: Within each pair:
   - sum: `result = left + right`  (f32 add, round-to-nearest-even)
   - max: `result = (left >= right) ? left : right`
     (IEEE 754 comparison; left operand wins on ties, including signed zeros)

4. **NaN**: Inputs must not contain NaN. Behavior undefined.

5. **Signed zero**: sum: `0.0 + (-0.0) = 0.0` (positive). max: left wins ties.

6. **Empty**: Returns identity (`0.0` for sum, `-inf` for max).

7. **Length 1**: Returns the single element unchanged.

**Dot products**: For `a · b` of length N, each thread computes `a[i] * b[i]`
(separate multiply, Rule 1), then the products are reduced via `tree_reduce_sum`.
This replaces the sequential `acc = acc + (a[i] * b[i])` loop.

**Note**: The tree sum produces DIFFERENT f32 bits than left-to-right accumulation
(f32 addition is not associative). Both CPU and GPU must use the same tree.
v1 serial kernel results are not comparable to v2 tree kernel results.

Reference implementation: `tree_reduce_sum_f32`, `tree_reduce_max_f32` in
`crates/verilm-core/src/attention.rs`.

CUDA implementation: `kernels/tree_reduce.cu`.

### Rule 5: Bit-level ldexp (no library scalbn/ldexp)

The `p * 2^n` operation in exp uses direct IEEE 754 exponent-field
manipulation, not any library function:

```
ldexp_bitwise(p: f32, n: i32) -> f32:
    bits = p.to_bits()
    biased_exp = ((bits >> 23) & 0xFF) as i32 + n
    if biased_exp <= 0: return 0.0     // underflow
    if biased_exp >= 255: return inf   // overflow (preserving sign)
    return f32::from_bits((bits & 0x807FFFFF) | (biased_exp << 23))
```

### Rule 6: Canonical exp polynomial

```
exp_canonical(x: f32) -> f32:
    t = x * LOG2_E                             // __fmul_rn
    n = rintf(t)                               // round-to-nearest-even
    f = t - n                                  // __fadd_rn(t, -n)

    // Horner form (inside-out):
    p = f * POLY_C4                            // __fmul_rn
    p = p + POLY_C3                            // __fadd_rn
    p = p * f                                  // __fmul_rn
    p = p + POLY_C2                            // __fadd_rn
    p = p * f                                  // __fmul_rn
    p = p + POLY_C1                            // __fadd_rn
    p = p * f                                  // __fmul_rn
    p = p + 1.0                                // __fadd_rn

    return ldexp_bitwise(p, n as i32)
```

### Rule 7: IEEE 754 division for softmax normalization

```
P[t] = exp_scores[t] / sum_exp               // __fdiv_rn (correctly rounded)
```

IEEE 754 guarantees division is correctly rounded for any two f32 inputs.
Both GPU `__fdiv_rn` and Rust `f32::div` produce identical results.

### Rule 8: bf16 → f32 conversion

Lossless upcast via bit shift:

```
bf16_to_f32(bits: u16) -> f32:
    return f32::from_bits((bits as u32) << 16)
```

No library conversion functions.

## Arithmetic Pipeline

### Step 1: Score Computation — S = Q · K^T * inv_sqrt_d

For each query head `qh` and KV position `t`:

```
kv_group = qh / (n_q_heads / n_kv_heads)    // integer division (GQA mapping)

acc: f32 = 0.0
for i in 0..d_head:
    q_val = bf16_to_f32(Q[qh * d_head + i])
    k_val = bf16_to_f32(K[t * n_kv_heads * d_head + kv_group * d_head + i])
    acc = acc + (q_val * k_val)              // Rule 1, Rule 4

S[qh, t] = acc * inv_sqrt_d                 // Rule 3
```

### Step 2: Softmax — P = softmax(S)

For each query head `qh`:

```
// Max score (tree reduce — Rule 4)
m = tree_reduce_max(S[qh, 0..seq_len])

// Exp (embarrassingly parallel — Rule 6)
for t in 0..seq_len:
    e[t] = exp_canonical(S[qh, t] - m)

// Sum (tree reduce — Rule 4)
sum_exp = tree_reduce_sum(e[0..seq_len])

// Normalize (embarrassingly parallel — Rule 7)
for t in 0..seq_len:
    P[qh, t] = e[t] / sum_exp
```

**v3 change**: Max and sum use the frozen tree reduction primitives (Rule 4)
instead of sequential left-to-right accumulation. This produces different f32
bits for `sum_exp` (and therefore different weights) compared to v2. Both CPU
and GPU must use the same tree structure.

### Step 3: V Aggregation — O = P @ V

For each query head `qh` and output dimension `i`:

```
kv_group = qh / (n_q_heads / n_kv_heads)

acc: f32 = 0.0
for t in 0..seq_len:
    v_val = bf16_to_f32(V[t * n_kv_heads * d_head + kv_group * d_head + i])
    acc = acc + (P[qh, t] * v_val)           // Rule 1, Rule 4

O[qh, i] = acc
```

## GPU Kernel Design

One CUDA block per query head. `blockDim.x = max(next_pow2(d_head), next_pow2(seq_len))`.
Step 1: all threads compute score dot products via parallel multiply + tree reduce.
Step 2: all threads compute softmax via parallel tree max, parallel exp,
tree sum, parallel normalize. Step 3: `d_head` threads compute V aggregation
(parallel across `d_head`, sequential across `seq_len`).

Compiled with: `nvcc -O2 --fmad=false -shared -Xcompiler -fPIC`

Source: `kernels/deterministic_attention.cu`

## CPU Reference Implementation

Rust function: `replay_attention_deterministic()` in
`crates/verilm-core/src/attention.rs`

Implements identical arithmetic using default Rust f32 ops (no FMA) +
explicit `round_ties_even()` + `ldexp_bitwise()`.

## Performance Considerations

The sequential f32 accumulation (Rules 1, 4) is not parallelizable across
the reduction dimension. This is the price of determinism.

Parallelism is available across:
- Query heads (`n_q_heads` = 32 for Llama 8B) — fully parallel
- Output dimensions in Step 3 (`d_head` = 128) — fully parallel

For decode (single query token), the overhead is acceptable:
- 128 multiply-adds per score × `seq_len` positions × 32 heads
- At 1K context: ~4M f32 ops — trivial on GPU
- Estimated ~3–10x slower than FlashAttention for attention alone
- ~5–15% overhead on total decode (attention is ~10–15% of decode time)

## Validation

### Parity test suite

| Test | Cases | Result |
|------|-------|--------|
| Randomized sweep (10 GQA configs, seq_len 1–2048) | 10,000 | 10,000/10,000 bit-exact |
| Repeated max scores (uniform softmax) | 6 | 6/6 |
| Near-tie scores | 4 | 4/4 |
| Extreme negative (exp underflow) | 4 | 4/4 |
| All-zero V | 3 | 3/3 |
| One-hot V | 3 | 3/3 |
| Constant V | 4 | 4/4 |
| Minimal context (seq_len=1) | 4 | 4/4 |
| Max-length context (seq_len=2048) | 2 | 2/2 |
| Awkward GQA ratios | 7 | 7/7 |
| Large magnitude inputs | 3 | 3/3 |
| Tiny magnitude inputs | 2 | 2/2 |
| Single dominant position | 3 | 3/3 |
| **Total** | **10,045** | **10,045/10,045** |

GPU: NVIDIA A100-80GB, CUDA 12.4
CPU: Rust (release mode), verilm-core

### Regression gates

- **CI (fast)**: Rust-only unit tests — `cargo test -p verilm-core -- attention`
- **Nightly (full)**: `scripts/modal/stress_deterministic_attention.py` — 10K random + 45 edge cases on A100

## Verification Protocol

### Prover Side (GPU)
1. Run QKV projection (stock `cutlass_scaled_mm`)
2. Apply RoPE
3. Capture post-RoPE Q, K, V in bf16
4. Run deterministic attention kernel (replaces `flash_attn`)
5. Commit attention output to Merkle tree
6. On challenge: open Q, K[:], V[:], attention_output

### Verifier Side (CPU, Rust)
1. Receive Q, K, V, attention_output from opened shell
2. Recompute attention using identical arithmetic (this spec)
3. Compare recomputed output vs committed output: must be **identical**
   (L-inf = 0, not tolerance-based)

## Open Questions

1. **RoPE canonicalization**: v1 sidesteps by committing post-RoPE Q/K.
   Future versions may canonicalize RoPE trig (sin/cos of `theta * position`).

2. **Multi-GPU architectures**: Validated on A100. If IEEE 754 compliance
   varies across SM versions, may need integer dot product fallback.

3. **Causal masking**: For decode (single query), all positions ≤ current
   are valid (no masking needed). Future prefill extension needs causal mask.

4. **KV cache quantization**: v1 assumes bf16 KV cache. fp8/int8 KV cache
   would need both sides to agree on the quantized representation.
