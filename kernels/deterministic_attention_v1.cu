/**
 * Deterministic Attention Kernel v1 — serial score path.
 *
 * KEPT FOR BENCHMARKING ONLY. v2 (deterministic_attention.cu) is the current
 * production kernel with tree-reduced scores.
 *
 * Difference from v2: Step 1 uses thread-0-only sequential dot product
 * instead of parallel multiply + tree reduce.
 *
 * Compile: nvcc -O2 --fmad=false -shared -Xcompiler -fPIC -o libdet_attn_v1.so deterministic_attention_v1.cu
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

// ─── Bit-level ldexp: p * 2^n without any library call ───────────
static __device__ __forceinline__ float ldexp_bitwise(float p, int n) {
    uint32_t bits;
    memcpy(&bits, &p, sizeof(float));
    int biased_exp = (int)((bits >> 23) & 0xFF);
    biased_exp += n;
    if (biased_exp <= 0) return 0.0f;
    if (biased_exp >= 255) {
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
static __device__ __forceinline__ float exp_canonical(float x) {
    float t = __fmul_rn(x, 1.4426950216293335f);
    float n = rintf(t);
    float f = __fadd_rn(t, -n);

    float p = __fmul_rn(f, 0.009618129f);
    p = __fadd_rn(p, 0.055504109f);
    p = __fmul_rn(p, f);
    p = __fadd_rn(p, 0.240226507f);
    p = __fmul_rn(p, f);
    p = __fadd_rn(p, 0.693147182f);
    p = __fmul_rn(p, f);
    p = __fadd_rn(p, 1.0f);

    return ldexp_bitwise(p, (int)n);
}

// ─── bf16 → f32: bit shift, no library dependency ───────────────
static __device__ __forceinline__ float bf16_to_f32(uint16_t bits) {
    uint32_t u32 = ((uint32_t)bits) << 16;
    float result;
    memcpy(&result, &u32, sizeof(float));
    return result;
}

/**
 * v1 kernel: thread 0 does EVERYTHING in Step 1 + Step 2 sequentially.
 * All threads do Step 3 (V aggregation) in parallel across d_head.
 */
__global__ void deterministic_attention_v1_kernel(
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

    extern __shared__ float smem[];
    float* scores = smem;                    // [seq_len]
    float* exp_scores = smem + seq_len;      // [seq_len]

    // --- Step 1: Scores (thread 0, fully sequential) ---
    if (threadIdx.x == 0) {
        float max_score = -INFINITY;

        for (int t = 0; t < seq_len; t++) {
            float acc = 0.0f;
            for (int i = 0; i < d_head; i++) {
                float q_val = bf16_to_f32(q[qh * d_head + i]);
                float k_val = bf16_to_f32(k[t * n_kv_heads * d_head + kv_group * d_head + i]);
                float prod = __fmul_rn(q_val, k_val);
                acc = __fadd_rn(acc, prod);
            }
            float s = __fmul_rn(acc, inv_sqrt_d);
            scores[t] = s;
            if (s > max_score) max_score = s;
        }

        // --- Step 2: Softmax (thread 0, sequential) ---
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

    // --- Step 3: V aggregation (parallel across d_head) ---
    int i = threadIdx.x;
    if (i < d_head) {
        float acc = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            float w = exp_scores[t];
            float v_val = bf16_to_f32(v[t * n_kv_heads * d_head + kv_group * d_head + i]);
            float prod = __fmul_rn(w, v_val);
            acc = __fadd_rn(acc, prod);
        }
        output[qh * d_head + i] = acc;
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
    int threads = d_head;   // v1: d_head threads (only Step 3 uses them)
    int blocks = n_q_heads;
    size_t smem_bytes = 2 * seq_len * sizeof(float);

    deterministic_attention_v1_kernel<<<blocks, threads, smem_bytes>>>(
        q_dev, k_dev, v_dev, output_dev, weights_dev,
        n_q_heads, n_kv_heads, d_head, seq_len, inv_sqrt_d
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA v1 kernel error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA v1 sync error: %s\n", cudaGetErrorString(err));
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
