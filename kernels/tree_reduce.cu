/**
 * Frozen parallel reduction tree — CUDA implementation.
 *
 * Must match the Rust reference `tree_reduce_sum_f32` / `tree_reduce_max_f32`
 * in `crates/verilm-core/src/attention.rs` bit-for-bit.
 *
 * CONTRACT (same as Rust — see attention.rs for the full spec):
 *
 * 1. PADDING: Input is logically padded to the next power of two.
 *    Identity: 0.0 for sum, -inf for max. Padding appended to END.
 *
 * 2. TREE STRUCTURE: At each level, stride halves. Pairs:
 *      buf[i] = op(buf[i], buf[i + stride])
 *    Left operand always has lower index.
 *
 * 3. PAIR ORDER:
 *    sum: left + right  (__fadd_rn, round-to-nearest-even)
 *    max: left >= right ? left : right  (IEEE 754 comparison, left wins ties)
 *
 * 4. NaN: Inputs must not contain NaN. Undefined on NaN.
 *
 * 5. SIGNED ZERO: sum: 0.0 + (-0.0) = 0.0. max: left wins on ties.
 *
 * 6. EMPTY: Returns identity (0.0 for sum, -inf for max).
 *
 * 7. LENGTH 1: Returns the single element.
 *
 * 8. WARP/BLOCK STAGING: The kernel uses shared memory for the full tree.
 *    No warp shuffle optimization in this reference — straightforward
 *    shared-memory implementation matching the logical tree exactly.
 *
 * Compile:
 *   nvcc -O2 --fmad=false -shared -Xcompiler -fPIC -o libtree_reduce.so tree_reduce.cu
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <math.h>

// ── Power-of-two ceiling ─────────────────────────────────────────
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

// ── Sum reduction kernel ─────────────────────────────────────────
// One block, blockDim.x >= padded_len/2 threads.
// Input: buf_dev[0..n-1] (n = original length)
// The kernel pads to next power of two with 0.0 and reduces in-place.
// Result in buf_dev[0].
__global__ void tree_reduce_sum_kernel(float* buf, int n, int padded_n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;

    // Load into shared memory with padding
    if (tid < padded_n) {
        smem[tid] = (tid < n) ? buf[tid] : 0.0f;
    }

    // If padded_n > blockDim.x, need a second load
    int tid2 = tid + blockDim.x;
    if (tid2 < padded_n) {
        smem[tid2] = (tid2 < n) ? buf[tid2] : 0.0f;
    }

    __syncthreads();

    // Tree reduction: stride halves each level
    for (int stride = padded_n / 2; stride >= 1; stride >>= 1) {
        if (tid < stride) {
            smem[tid] = __fadd_rn(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }

    // Write result back
    if (tid == 0) {
        buf[0] = smem[0];
    }
}

// ── Max reduction kernel ─────────────────────────────────────────
__global__ void tree_reduce_max_kernel(float* buf, int n, int padded_n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;

    // Load with -inf padding
    if (tid < padded_n) {
        smem[tid] = (tid < n) ? buf[tid] : -INFINITY;
    }

    int tid2 = tid + blockDim.x;
    if (tid2 < padded_n) {
        smem[tid2] = (tid2 < n) ? buf[tid2] : -INFINITY;
    }

    __syncthreads();

    // Tree reduction
    for (int stride = padded_n / 2; stride >= 1; stride >>= 1) {
        if (tid < stride) {
            // left >= right ? left : right (left wins ties)
            float left = smem[tid];
            float right = smem[tid + stride];
            smem[tid] = (left >= right) ? left : right;
        }
        __syncthreads();
    }

    if (tid == 0) {
        buf[0] = smem[0];
    }
}

// ── C API ────────────────────────────────────────────────────────

extern "C" {

/**
 * Tree-reduce sum on device memory. Result in buf_dev[0].
 * buf_dev must have at least `n` elements.
 * Returns 0 on success, -1 on error.
 */
int tree_reduce_sum_f32(float* buf_dev, int n) {
    if (n <= 0) {
        // Write identity to device
        float zero = 0.0f;
        cudaMemcpy(buf_dev, &zero, sizeof(float), cudaMemcpyHostToDevice);
        return 0;
    }
    if (n == 1) {
        return 0; // already in buf_dev[0]
    }

    int padded_n = next_pow2(n);
    // Threads: need at least padded_n for initial load, at least padded_n/2 for reduction
    int threads = padded_n; // simple: one thread per element for load
    if (threads > 1024) threads = 1024;
    size_t smem_bytes = padded_n * sizeof(float);

    tree_reduce_sum_kernel<<<1, threads, smem_bytes>>>(buf_dev, n, padded_n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "tree_reduce_sum kernel error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "tree_reduce_sum sync error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

/**
 * Tree-reduce max on device memory. Result in buf_dev[0].
 */
int tree_reduce_max_f32(float* buf_dev, int n) {
    if (n <= 0) {
        float neg_inf = -INFINITY;
        cudaMemcpy(buf_dev, &neg_inf, sizeof(float), cudaMemcpyHostToDevice);
        return 0;
    }
    if (n == 1) {
        return 0;
    }

    int padded_n = next_pow2(n);
    int threads = padded_n;
    if (threads > 1024) threads = 1024;
    size_t smem_bytes = padded_n * sizeof(float);

    tree_reduce_max_kernel<<<1, threads, smem_bytes>>>(buf_dev, n, padded_n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "tree_reduce_max kernel error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "tree_reduce_max sync error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

/**
 * Host-convenience: copy data to device, reduce, copy result back.
 */
int tree_reduce_sum_f32_host(const float* host_data, int n, float* result) {
    if (n <= 0) {
        *result = 0.0f;
        return 0;
    }
    if (n == 1) {
        *result = host_data[0];
        return 0;
    }

    float* dev;
    cudaMalloc(&dev, n * sizeof(float));
    cudaMemcpy(dev, host_data, n * sizeof(float), cudaMemcpyHostToDevice);

    int rc = tree_reduce_sum_f32(dev, n);
    if (rc == 0) {
        cudaMemcpy(result, dev, sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaFree(dev);
    return rc;
}

int tree_reduce_max_f32_host(const float* host_data, int n, float* result) {
    if (n <= 0) {
        *result = -INFINITY;
        return 0;
    }
    if (n == 1) {
        *result = host_data[0];
        return 0;
    }

    float* dev;
    cudaMalloc(&dev, n * sizeof(float));
    cudaMemcpy(dev, host_data, n * sizeof(float), cudaMemcpyHostToDevice);

    int rc = tree_reduce_max_f32(dev, n);
    if (rc == 0) {
        cudaMemcpy(result, dev, sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaFree(dev);
    return rc;
}

} // extern "C"
