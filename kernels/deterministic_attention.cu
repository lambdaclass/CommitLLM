/**
 * Deterministic Attention Kernel v4 — bit-exact against CPU reference.
 *
 * Single-query decode attention with fully specified arithmetic:
 *   Step 1: Score = Q · K^T * inv_sqrt_d  (v2: parallel multiply + tree reduce)
 *   Step 2: Softmax (v3: parallel tree max, parallel exp, tree sum, parallel normalize)
 *   Step 3: Output = P @ V  (v4: tiled across seq_len, partials merged via tree reduce)
 *
 * v4 change (V aggregation):
 *   - seq_len is divided into fixed tiles of TILE_SIZE
 *   - Each (head, tile) pair is computed by a separate CUDA block
 *   - Within a tile: sequential accumulation per d_head thread (same as v1)
 *   - Across tiles: partials merged via frozen tree_reduce_sum contract
 *   - Three-kernel architecture: scores+softmax, V-tile partials, V-merge
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

// ═══════════════════════════════════════════════════════════════════
// Kernel A: Scores + Softmax (Steps 1-2, same as v3)
//
// Grid: n_q_heads blocks.  Block: padded_d threads.
// Writes normalized weights to weights[] (global memory).
// ═══════════════════════════════════════════════════════════════════
__global__ void kernel_scores_softmax(
    const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ k,
    float* __restrict__ weights,
    int n_q_heads,
    int n_kv_heads,
    int d_head,
    int seq_len,
    float inv_sqrt_d,
    int padded_d,
    int padded_seq,
    int padded_reduce
) {
    int qh = blockIdx.x;
    if (qh >= n_q_heads) return;

    int heads_per_kv = n_q_heads / n_kv_heads;
    int kv_group = qh / heads_per_kv;

    extern __shared__ float smem[];
    float* reduce_buf = smem;
    float* scores     = smem + padded_reduce;
    float* exp_scores = smem + padded_reduce + seq_len;

    int tid = threadIdx.x;

    // --- Step 1 (v2): Scores via parallel multiply + tree reduce ---
    for (int t = 0; t < seq_len; t++) {
        if (tid < d_head) {
            float q_val = bf16_to_f32(q[qh * d_head + tid]);
            float k_val = bf16_to_f32(k[t * n_kv_heads * d_head + kv_group * d_head + tid]);
            reduce_buf[tid] = __fmul_rn(q_val, k_val);
        } else if (tid < padded_d) {
            reduce_buf[tid] = 0.0f;
        }
        __syncthreads();

        for (int stride = padded_d / 2; stride >= 1; stride >>= 1) {
            if (tid < stride) {
                reduce_buf[tid] = __fadd_rn(reduce_buf[tid], reduce_buf[tid + stride]);
            }
            __syncthreads();
        }

        if (tid == 0) {
            scores[t] = __fmul_rn(reduce_buf[0], inv_sqrt_d);
        }
        __syncthreads();
    }

    // --- Step 2 (v3): Softmax ---

    // 2a. Tree reduce max
    for (int base = tid; base < padded_seq; base += blockDim.x) {
        reduce_buf[base] = (base < seq_len) ? scores[base] : -INFINITY;
    }
    __syncthreads();
    for (int stride = padded_seq / 2; stride >= 1; stride >>= 1) {
        for (int base = tid; base < stride; base += blockDim.x) {
            float left = reduce_buf[base];
            float right = reduce_buf[base + stride];
            reduce_buf[base] = (left >= right) ? left : right;
        }
        __syncthreads();
    }
    float max_score = reduce_buf[0];
    __syncthreads();

    // 2b. Parallel exp
    for (int base = tid; base < seq_len; base += blockDim.x) {
        exp_scores[base] = exp_canonical(__fadd_rn(scores[base], -max_score));
    }
    __syncthreads();

    // 2c. Tree reduce sum
    for (int base = tid; base < padded_seq; base += blockDim.x) {
        reduce_buf[base] = (base < seq_len) ? exp_scores[base] : 0.0f;
    }
    __syncthreads();
    for (int stride = padded_seq / 2; stride >= 1; stride >>= 1) {
        for (int base = tid; base < stride; base += blockDim.x) {
            reduce_buf[base] = __fadd_rn(reduce_buf[base], reduce_buf[base + stride]);
        }
        __syncthreads();
    }
    float sum_exp = reduce_buf[0];
    __syncthreads();

    // 2d. Parallel normalize + write weights to global
    for (int base = tid; base < seq_len; base += blockDim.x) {
        weights[qh * seq_len + base] = __fdiv_rn(exp_scores[base], sum_exp);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel B: Tiled V partial sums
//
// Grid: n_q_heads * n_tiles blocks (1D flattened).
// Block: padded_d threads.
// Each block computes one tile's partial V sum for one head.
// Reads weights from global memory (output of Kernel A).
// Writes partials to partials_dev[qh * n_tiles * d_head + tile * d_head + dim].
// ═══════════════════════════════════════════════════════════════════
__global__ void kernel_v_tiles(
    const uint16_t* __restrict__ v,
    const float* __restrict__ weights,
    float* __restrict__ partials,
    int n_q_heads,
    int n_kv_heads,
    int d_head,
    int seq_len,
    int tile_size,
    int n_tiles
) {
    int block_id = blockIdx.x;
    int qh = block_id / n_tiles;
    int tile_idx = block_id % n_tiles;
    if (qh >= n_q_heads) return;

    int heads_per_kv = n_q_heads / n_kv_heads;
    int kv_group = qh / heads_per_kv;

    int tid = threadIdx.x;
    if (tid >= d_head) return;

    int tile_start = tile_idx * tile_size;
    int tile_end = tile_start + tile_size;
    if (tile_end > seq_len) tile_end = seq_len;

    float acc = 0.0f;
    for (int t = tile_start; t < tile_end; t++) {
        float w = weights[qh * seq_len + t];
        float v_val = bf16_to_f32(v[t * n_kv_heads * d_head + kv_group * d_head + tid]);
        float prod = __fmul_rn(w, v_val);
        acc = __fadd_rn(acc, prod);
    }

    partials[qh * n_tiles * d_head + tile_idx * d_head + tid] = acc;
}

// ═══════════════════════════════════════════════════════════════════
// Kernel C: Merge tile partials via tree reduce
//
// Grid: n_q_heads blocks.  Block: padded_d threads.
// Each thread tree-reduces its n_tiles partials (in registers).
// Writes final output.
// ═══════════════════════════════════════════════════════════════════
__global__ void kernel_v_merge(
    const float* __restrict__ partials,
    float* __restrict__ output,
    int n_q_heads,
    int d_head,
    int n_tiles,
    int padded_tiles
) {
    int qh = blockIdx.x;
    if (qh >= n_q_heads) return;

    int tid = threadIdx.x;
    if (tid >= d_head) return;

    // Load tile partials into register array + identity padding
    // Max supported: 64 tiles (seq_len up to 64 * max_tile_size)
    float buf[64];
    for (int t = 0; t < n_tiles; t++) {
        buf[t] = partials[qh * n_tiles * d_head + t * d_head + tid];
    }
    for (int t = n_tiles; t < padded_tiles; t++) {
        buf[t] = 0.0f;  // identity for sum
    }

    // Frozen binary tree reduction (same contract as tree_reduce_sum_f32)
    for (int stride = padded_tiles / 2; stride >= 1; stride >>= 1) {
        for (int i = 0; i < stride; i++) {
            buf[i] = __fadd_rn(buf[i], buf[i + stride]);
        }
    }

    output[qh * d_head + tid] = buf[0];
}

// ─── C API ───────────────────────────────────────────────────────

extern "C" {

// Default tile size for V aggregation. Protocol constant.
#define DEFAULT_TILE_SIZE 128

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
    int padded_seq = next_pow2(seq_len);
    int padded_reduce = (padded_d > padded_seq) ? padded_d : padded_seq;

    int tile_size = DEFAULT_TILE_SIZE;
    int n_tiles = (seq_len + tile_size - 1) / tile_size;
    int padded_tiles = next_pow2(n_tiles);

    cudaError_t err;

    // --- Kernel A: Scores + Softmax ---
    {
        int threads = padded_d;
        int blocks = n_q_heads;
        size_t smem_bytes = (padded_reduce + 2 * seq_len) * sizeof(float);

        kernel_scores_softmax<<<blocks, threads, smem_bytes>>>(
            q_dev, k_dev, weights_dev,
            n_q_heads, n_kv_heads, d_head, seq_len, inv_sqrt_d,
            padded_d, padded_seq, padded_reduce
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "kernel_scores_softmax error: %s\n", cudaGetErrorString(err));
            return -1;
        }
    }

    // --- Kernel B: Tiled V partial sums ---
    float* partials_dev = nullptr;
    {
        size_t partials_bytes = (size_t)n_q_heads * n_tiles * d_head * sizeof(float);
        err = cudaMalloc(&partials_dev, partials_bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "partials alloc error: %s\n", cudaGetErrorString(err));
            return -1;
        }

        int threads = padded_d;
        int blocks = n_q_heads * n_tiles;

        kernel_v_tiles<<<blocks, threads>>>(
            v_dev, weights_dev, partials_dev,
            n_q_heads, n_kv_heads, d_head, seq_len, tile_size, n_tiles
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "kernel_v_tiles error: %s\n", cudaGetErrorString(err));
            cudaFree(partials_dev);
            return -1;
        }
    }

    // --- Kernel C: Merge tile partials ---
    {
        int threads = padded_d;
        int blocks = n_q_heads;

        kernel_v_merge<<<blocks, threads>>>(
            partials_dev, output_dev,
            n_q_heads, d_head, n_tiles, padded_tiles
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "kernel_v_merge error: %s\n", cudaGetErrorString(err));
            cudaFree(partials_dev);
            return -1;
        }
    }

    cudaFree(partials_dev);

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
