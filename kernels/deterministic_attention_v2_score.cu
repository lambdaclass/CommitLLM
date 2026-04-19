/**
 * Deterministic Attention Kernel v2 — bit-exact against CPU reference.
 *
 * Single-query decode attention with fully specified arithmetic:
 *   Step 1: Score = Q · K^T * inv_sqrt_d  (v2: parallel multiply + tree reduce)
 *   Step 2: Softmax with canonical exp polynomial (v1: thread 0, sequential)
 *   Step 3: Output = P @ V  (v1: sequential per-thread, parallel across d_head)
 *
 * v2 change (score path only):
 *   - Each thread computes one element product q[i]*k[i]
 *   - Products are reduced via frozen binary tree (shared memory)
 *   - Tree structure matches the CPU reference tree_reduce_sum_f32()
 *
 * All library-dependent math is eliminated:
 *   - No scalbnf/ldexp — bit-level exponent manipulation instead
 *   - No sqrtf/rsqrtf — inv_sqrt_d precomputed by caller
 *   - No fma — compiled with --fmad=false, all ops use __fmul_rn/__fadd_rn
 *
 * Compile: nvcc -O2 --fmad=false -shared -Xcompiler -fPIC -o libdet_attn.so deterministic_attention.cu
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

// ─── Bit-level ldexp: p * 2^n without any library call ───────────
// Adds n to the IEEE 754 exponent field. Handles n=0 as identity.
// Assumes p is a normal positive float and |n| < 127.
static __device__ __forceinline__ float ldexp_bitwise(float p, int n) {
    uint32_t bits;
    memcpy(&bits, &p, sizeof(float));
    // Extract biased exponent (bits 30:23)
    int biased_exp = (int)((bits >> 23) & 0xFF);
    biased_exp += n;
    // Clamp to valid range (underflow → 0, overflow → inf)
    if (biased_exp <= 0) return 0.0f;
    if (biased_exp >= 255) {
        // Return +inf with same sign
        bits = (bits & 0x80000000u) | 0x7F800000u;
        float result;
        memcpy(&result, &bits, sizeof(float));
        return result;
    }
    bits = (bits & 0x807FFFFFu) | ((uint32_t)biased_exp << 23);
    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

// ─── Canonical exp: protocol-frozen polynomial ───────────────────
// exp(x) = 2^(x * log2(e)) via minimax polynomial for 2^f on [-0.5, 0.5].
// Coefficients are protocol constants — changing them changes the protocol.
static __device__ __forceinline__ float exp_canonical(float x) {
    float t = __fmul_rn(x, 1.4426950216293335f);  // x * log2(e)
    float n = rintf(t);
    float f = __fadd_rn(t, -n);

    // Horner form: p = 1 + f*(c1 + f*(c2 + f*(c3 + f*c4)))
    float p = __fmul_rn(f, 0.009618129f);
    p = __fadd_rn(p, 0.055504109f);
    p = __fmul_rn(p, f);
    p = __fadd_rn(p, 0.240226507f);
    p = __fmul_rn(p, f);
    p = __fadd_rn(p, 0.693147182f);
    p = __fmul_rn(p, f);
    p = __fadd_rn(p, 1.0f);

    // p * 2^n via bit manipulation (no scalbnf)
    return ldexp_bitwise(p, (int)n);
}

// ─── bf16 → f32: bit shift, no library dependency ───────────────
static __device__ __forceinline__ float bf16_to_f32(uint16_t bits) {
    uint32_t u32 = ((uint32_t)bits) << 16;
    float result;
    memcpy(&result, &u32, sizeof(float));
    return result;
}

// ─── Next power of two (host+device) ──────────────────────────
static __host__ __device__ unsigned int next_pow2(unsigned int v) {
    if (v == 0) return 1;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

/**
 * Deterministic decode attention kernel (v2 — tree-reduced scores).
 *
 * One block per query head. blockDim.x = next_pow2(d_head).
 *
 * Step 1 (v2): All threads compute element-wise products, then tree-reduce
 *   in shared memory to get dot product. Thread 0 applies inv_sqrt_d.
 * Step 2 (v1): Thread 0 computes softmax (sequential).
 * Step 3 (v1): All threads compute V aggregation (parallel across d_head, sequential across seq_len).
 *
 * Shared memory layout:
 *   reduce_buf:  [padded_d_head]  — tree reduction workspace
 *   scores:      [seq_len]        — attention scores
 *   exp_scores:  [seq_len]        — softmax weights
 */
__global__ void deterministic_attention_kernel(
    const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ k,
    const uint16_t* __restrict__ v,
    float* __restrict__ output,
    float* __restrict__ weights,
    int n_q_heads,
    int n_kv_heads,
    int d_head,
    int seq_len,
    float inv_sqrt_d
) {
    int qh = blockIdx.x;
    if (qh >= n_q_heads) return;

    int heads_per_kv = n_q_heads / n_kv_heads;
    int kv_group = qh / heads_per_kv;

    int padded_d = next_pow2(d_head);

    extern __shared__ float smem[];
    float* reduce_buf = smem;                            // [padded_d]
    float* scores     = smem + padded_d;                 // [seq_len]
    float* exp_scores = smem + padded_d + seq_len;       // [seq_len]

    int tid = threadIdx.x;

    // --- Step 1 (v2): Scores via parallel multiply + tree reduce ---
    float max_score = -INFINITY;

    for (int t = 0; t < seq_len; t++) {
        // Phase A: element-wise multiply (each thread handles one dimension)
        if (tid < d_head) {
            float q_val = bf16_to_f32(q[qh * d_head + tid]);
            float k_val = bf16_to_f32(k[t * n_kv_heads * d_head + kv_group * d_head + tid]);
            reduce_buf[tid] = __fmul_rn(q_val, k_val);
        } else if (tid < padded_d) {
            reduce_buf[tid] = 0.0f;  // padding with identity for sum
        }
        __syncthreads();

        // Phase B: fixed binary tree reduction (matches tree_reduce_sum_f32)
        for (int stride = padded_d / 2; stride >= 1; stride >>= 1) {
            if (tid < stride) {
                reduce_buf[tid] = __fadd_rn(reduce_buf[tid], reduce_buf[tid + stride]);
            }
            __syncthreads();
        }

        // Thread 0 has the dot product in reduce_buf[0]
        if (tid == 0) {
            float s = __fmul_rn(reduce_buf[0], inv_sqrt_d);
            scores[t] = s;
            if (s > max_score) max_score = s;
        }
        __syncthreads();
    }

    // --- Step 2: Softmax (thread 0, sequential — unchanged from v1) ---
    if (tid == 0) {
        float sum_exp = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            float e = exp_canonical(__fadd_rn(scores[t], -max_score));
            exp_scores[t] = e;
            sum_exp = __fadd_rn(sum_exp, e);
        }

        for (int t = 0; t < seq_len; t++) {
            float w = __fdiv_rn(exp_scores[t], sum_exp);
            exp_scores[t] = w;
            weights[qh * seq_len + t] = w;
        }
    }

    __syncthreads();

    // --- Step 3: V aggregation (parallel across d_head, sequential across seq_len — v1) ---
    if (tid < d_head) {
        float acc = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            float w = exp_scores[t];
            float v_val = bf16_to_f32(v[t * n_kv_heads * d_head + kv_group * d_head + tid]);
            float prod = __fmul_rn(w, v_val);
            acc = __fadd_rn(acc, prod);
        }
        output[qh * d_head + tid] = acc;
    }
}

// ─── C API ───────────────────────────────────────────────────────

extern "C" {

int deterministic_attention(
    const uint16_t* q_dev,
    const uint16_t* k_dev,
    const uint16_t* v_dev,
    float* output_dev,
    float* weights_dev,
    int n_q_heads,
    int n_kv_heads,
    int d_head,
    int seq_len,
    float inv_sqrt_d
) {
    int padded_d = next_pow2(d_head);
    int threads = padded_d;  // v2: one thread per padded d_head dimension
    int blocks = n_q_heads;
    // smem layout: reduce_buf[padded_d] + scores[seq_len] + exp_scores[seq_len]
    size_t smem_bytes = (padded_d + 2 * seq_len) * sizeof(float);

    deterministic_attention_kernel<<<blocks, threads, smem_bytes>>>(
        q_dev, k_dev, v_dev, output_dev, weights_dev,
        n_q_heads, n_kv_heads, d_head, seq_len, inv_sqrt_d
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA sync error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

int deterministic_attention_host(
    const uint16_t* q_host,
    const uint16_t* k_host,
    const uint16_t* v_host,
    float* output_host,
    float* weights_host,
    int n_q_heads,
    int n_kv_heads,
    int d_head,
    int seq_len,
    float inv_sqrt_d
) {
    size_t q_bytes = n_q_heads * d_head * sizeof(uint16_t);
    size_t kv_bytes = seq_len * n_kv_heads * d_head * sizeof(uint16_t);
    size_t out_bytes = n_q_heads * d_head * sizeof(float);
    size_t w_bytes = n_q_heads * seq_len * sizeof(float);

    uint16_t *q_dev, *k_dev, *v_dev;
    float *out_dev, *w_dev;

    cudaMalloc(&q_dev, q_bytes);
    cudaMalloc(&k_dev, kv_bytes);
    cudaMalloc(&v_dev, kv_bytes);
    cudaMalloc(&out_dev, out_bytes);
    cudaMalloc(&w_dev, w_bytes);

    cudaMemcpy(q_dev, q_host, q_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(k_dev, k_host, kv_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(v_dev, v_host, kv_bytes, cudaMemcpyHostToDevice);

    int rc = deterministic_attention(
        q_dev, k_dev, v_dev, out_dev, w_dev,
        n_q_heads, n_kv_heads, d_head, seq_len, inv_sqrt_d
    );

    if (rc == 0) {
        cudaMemcpy(output_host, out_dev, out_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(weights_host, w_dev, w_bytes, cudaMemcpyDeviceToHost);
    }

    cudaFree(q_dev);
    cudaFree(k_dev);
    cudaFree(v_dev);
    cudaFree(out_dev);
    cudaFree(w_dev);

    return rc;
}

} // extern "C"
