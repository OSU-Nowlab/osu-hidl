#ifndef MACMATCH_FALLBACK_WARPS
#define MACMATCH_FALLBACK_WARPS 8 // 8 warps -> 256 threads
#endif

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>

#include <algorithm>
#include <vector>
#include <stdint.h>
#include <tuple>
#include <limits>
namespace py = pybind11;

using at::Tensor;

// -------------------------------- Utilities ----------------------------------

static inline void checkContigCuda(const at::Tensor &t, const char *name)
{
    TORCH_CHECK(t.is_cuda(), name, " must be on CUDA");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

__device__ __forceinline__ float bf16_to_fp32(const __nv_bfloat16 &x)
{
    return __bfloat162float(x);
}

__device__ __forceinline__ bool better_pair(float val, int idx, float best_val, int best_idx)
{
    return (val < best_val) || ((val == best_val) && (idx < best_idx));
}

// Best-effort L2 prefetch (SM80+; safe no-op elsewhere)
__device__ __forceinline__ void prefetch_L2(const void *ptr)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("prefetch.global.L2 [%0];" ::"l"(ptr));
#endif
}

// Alignment-aware coalesced copy (16B/4B/1B) for update path
__device__ __forceinline__ void copy_row_coalesced(void *__restrict__ dst,
                                                   const void *__restrict__ src,
                                                   size_t nbytes,
                                                   int tid, int nthreads)
{
    uintptr_t da = reinterpret_cast<uintptr_t>(dst);
    uintptr_t sa = reinterpret_cast<uintptr_t>(src);

    // 16B path
    if (((da | sa | nbytes) & 0xF) == 0)
    {
        size_t n16 = nbytes / 16;
        uint4 *d16 = reinterpret_cast<uint4 *>(dst);
        const uint4 *s16 = reinterpret_cast<const uint4 *>(src);
        for (size_t i = tid; i < n16; i += (size_t)nthreads)
            d16[i] = s16[i];
        return;
    }
    // 4B path
    if (((da | sa | nbytes) & 0x3) == 0)
    {
        size_t n4 = nbytes / 4;
        uint32_t *d4 = reinterpret_cast<uint32_t *>(dst);
        const uint32_t *s4 = reinterpret_cast<const uint32_t *>(src);
        for (size_t i = tid; i < n4; i += (size_t)nthreads)
            d4[i] = s4[i];
        return;
    }
    // 1B fallback
    const uint8_t *s8 = reinterpret_cast<const uint8_t *>(src);
    uint8_t *d8 = reinterpret_cast<uint8_t *>(dst);
    for (size_t i = tid; i < nbytes; i += (size_t)nthreads)
        d8[i] = s8[i];
}

// ------------------------ Ring update: row-parallel ---------------------------

__global__ void ring_update_rows_kernel(
    __nv_bfloat16 *__restrict__ q_cache,     // [R,M,H,D]
    __nv_bfloat16 *__restrict__ a_cache,     // [R,M,H,D]
    float *__restrict__ lse_cache,           // [R,M,H]
    int32_t *__restrict__ request_length,    // [R]
    const int32_t *__restrict__ req_ids,     // [B]
    const __nv_bfloat16 *__restrict__ src_q, // [sumN,H,D]
    const __nv_bfloat16 *__restrict__ src_a, // [sumN,H,D]
    const float *__restrict__ src_lse,       // [sumN,H]
    const int32_t *__restrict__ offsets,     // [B+1]
    const int32_t *__restrict__ lens,        // [B]
    int R, int M, int H, int D,
    const int32_t *__restrict__ starts,  // [B]
    const int32_t *__restrict__ keeps,   // [B]
    const int32_t *__restrict__ L_before // [B]
)
{
    int j = blockIdx.x; // row within entry
    int b = blockIdx.y; // batch entry
    int keep = keeps[b];
    if (j >= keep)
        return;

    int req = req_ids[b];
    int start = starts[b];
    int Lb = L_before[b];

    int src_row = offsets[b] + start + j;
    int dest_slot = (Lb + start + j) % M;

    size_t row_bytes_bf16 = (size_t)H * (size_t)D * sizeof(__nv_bfloat16);
    size_t row_bytes_f32 = (size_t)H * sizeof(float);

    // Q/ATTN rows
    {
        const void *s = src_q + (size_t)src_row * (size_t)H * (size_t)D;
        void *d = q_cache + (((size_t)req * M + (size_t)dest_slot) * H) * (size_t)D;
        copy_row_coalesced(d, s, row_bytes_bf16, threadIdx.x, blockDim.x);
    }
    {
        const void *s = src_a + (size_t)src_row * (size_t)H * (size_t)D;
        void *d = a_cache + (((size_t)req * M + (size_t)dest_slot) * H) * (size_t)D;
        copy_row_coalesced(d, s, row_bytes_bf16, threadIdx.x, blockDim.x);
    }
    // LSE row
    {
        const void *s = src_lse + (size_t)src_row * (size_t)H;
        void *d = lse_cache + (((size_t)req * M + (size_t)dest_slot) * H);
        copy_row_coalesced(d, s, row_bytes_f32, threadIdx.x, blockDim.x);
    }
}

// Forward declare update launcher (unchanged)
static inline void ring_update_row_parallel_launcher(
    at::Tensor q_cache, at::Tensor attn_cache, at::Tensor lse_cache, at::Tensor request_length,
    at::Tensor req_ids, at::Tensor src_q, at::Tensor src_attn, at::Tensor src_lse,
    at::Tensor offsets, at::Tensor lens, int M, int H, int D);

// ---------------- Update launcher (unchanged semantics) -----------------------

static inline void ring_update_row_parallel_launcher(
    at::Tensor q_cache, at::Tensor attn_cache, at::Tensor lse_cache, at::Tensor request_length,
    at::Tensor req_ids, at::Tensor src_q, at::Tensor src_attn, at::Tensor src_lse,
    at::Tensor offsets, at::Tensor lens, int M, int H, int D)
{
    // Checks
    checkContigCuda(q_cache, "q_cache");
    checkContigCuda(attn_cache, "attn_cache");
    checkContigCuda(lse_cache, "lse_cache");
    checkContigCuda(request_length, "request_length");
    checkContigCuda(req_ids, "req_ids");
    checkContigCuda(src_q, "src_q");
    checkContigCuda(src_attn, "src_attn");
    checkContigCuda(src_lse, "src_lse");
    checkContigCuda(offsets, "offsets");
    checkContigCuda(lens, "lens");

    TORCH_CHECK(q_cache.scalar_type() == at::kBFloat16, "q_cache must be bf16");
    TORCH_CHECK(attn_cache.scalar_type() == at::kBFloat16, "attn_cache must be bf16");
    TORCH_CHECK(lse_cache.scalar_type() == at::kFloat, "lse_cache must be float32");
    TORCH_CHECK(request_length.scalar_type() == at::kInt, "request_length must be int32");
    TORCH_CHECK(req_ids.scalar_type() == at::kInt, "req_ids must be int32");
    TORCH_CHECK(src_q.scalar_type() == at::kBFloat16 && src_attn.scalar_type() == at::kBFloat16 && src_lse.scalar_type() == at::kFloat,
                "src_q/src_attn must be bf16, src_lse must be float32");
    TORCH_CHECK(offsets.scalar_type() == at::kInt && lens.scalar_type() == at::kInt, "offsets/lens must be int32");

    const int R = (int)q_cache.size(0);
    const int B = (int)req_ids.size(0);
    TORCH_CHECK(offsets.size(0) == B + 1, "offsets must be [B+1]");

    auto stream = c10::cuda::getCurrentCUDAStream();

    // Host precompute (same behavior as your summary)
    std::vector<int32_t> h_req(B);
    C10_CUDA_CHECK(cudaMemcpyAsync(h_req.data(), req_ids.data_ptr<int32_t>(), B * sizeof(int32_t),
                                   cudaMemcpyDeviceToHost, stream.stream()));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
    std::vector<int32_t> h_lens(B), h_offsets(B + 1), h_L_before(R), h_starts(B), h_keeps(B);
    C10_CUDA_CHECK(cudaMemcpyAsync(h_lens.data(), lens.data_ptr<int32_t>(), B * sizeof(int32_t),
                                   cudaMemcpyDeviceToHost, stream.stream()));
    C10_CUDA_CHECK(cudaMemcpyAsync(h_offsets.data(), offsets.data_ptr<int32_t>(), (B + 1) * sizeof(int32_t),
                                   cudaMemcpyDeviceToHost, stream.stream()));
    C10_CUDA_CHECK(cudaMemcpyAsync(h_L_before.data(), request_length.data_ptr<int32_t>(), R * sizeof(int32_t),
                                   cudaMemcpyDeviceToHost, stream.stream()));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));

    for (int b = 0; b < B; ++b)
    {
        const int req = h_req[b];
        const int N_b = h_lens[b];
        const int Lb = h_L_before[req];
        int start = 0, keep = N_b;
        if (N_b > M)
        {
            start = N_b - M;
            keep = M;
        }
        h_starts[b] = start;
        h_keeps[b] = keep;
        h_L_before[b] = Lb;
    }

    auto i32opt = req_ids.options().dtype(at::kInt);
    at::Tensor starts = at::empty({B}, i32opt);
    at::Tensor keeps = at::empty({B}, i32opt);
    at::Tensor Lb = at::empty({B}, i32opt);
    C10_CUDA_CHECK(cudaMemcpyAsync(starts.data_ptr<int32_t>(), h_starts.data(), B * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, stream.stream()));
    C10_CUDA_CHECK(cudaMemcpyAsync(keeps.data_ptr<int32_t>(), h_keeps.data(), B * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, stream.stream()));
    C10_CUDA_CHECK(cudaMemcpyAsync(Lb.data_ptr<int32_t>(), h_L_before.data(), B * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, stream.stream()));

    int keep_max = 0;
    for (int v : h_keeps)
        keep_max = std::max(keep_max, v);
    dim3 grid((unsigned)keep_max, (unsigned)B, 1);
    dim3 block(256);
    ring_update_rows_kernel<<<grid, block, 0, stream.stream()>>>(
        reinterpret_cast<__nv_bfloat16 *>(q_cache.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 *>(attn_cache.data_ptr<at::BFloat16>()),
        lse_cache.data_ptr<float>(),
        request_length.data_ptr<int32_t>(),
        req_ids.data_ptr<int32_t>(),
        reinterpret_cast<const __nv_bfloat16 *>(src_q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16 *>(src_attn.data_ptr<at::BFloat16>()),
        src_lse.data_ptr<float>(),
        offsets.data_ptr<int32_t>(),
        lens.data_ptr<int32_t>(),
        (int)q_cache.size(0), (int)q_cache.size(1), (int)q_cache.size(2), (int)q_cache.size(3),
        starts.data_ptr<int32_t>(), keeps.data_ptr<int32_t>(), Lb.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// --------------------------- Matcher (split-K) --------------------------------
// --------------------------- Matcher (split-K) --------------------------------
//
// Mapping is implicit: request index 'req' == batch index 'n'.
// We read L directly as L = request_length[n].
// Grid mapping (flash-like):
//   - For the tile pass, flatten (n, tile) onto grid.x and use grid.y = H:
//       grid = (N * num_tiles, H, 1)
//     Inside the kernel: tn = blockIdx.x; n = tn / num_tiles; t = tn % num_tiles.
//   - For the merge, use a persistent grid (see section C).

// ----------------------- Tile kernels (adaptive) ------------------------------
//
// Families:
//  - D64:  fully unrolled inner loop (D==64)
//  - D128: fully unrolled inner loop (D==128)
//  - DGEN: generic D (fallback)
//
// Block: BLOCK_THREADS in {256, 320, 384}
// Shared: query vector q_s in fp32 (D*sizeof(float))

#ifndef MACMATCH_PREFETCH_STAGES
#define MACMATCH_PREFETCH_STAGES 2 // set to 2 or 3; can override at compile time
#endif

// ===============================================================
// tile_min_kernel_fast_D  (DVAL known at compile time, e.g., 64/128)
// ===============================================================
template <int BLOCK_THREADS, int DVAL>
__global__ void tile_min_kernel_fast_D(
    const __nv_bfloat16 *__restrict__ q_cache,  // [R,M,H,DVAL]
    const int32_t *__restrict__ request_length, // [R]
    const __nv_bfloat16 *__restrict__ queries,  // [N,H,DVAL]
    float *__restrict__ partial_min,            // [N,H,T]
    int32_t *__restrict__ partial_idx,          // [N,H,T]
    int R, int M, int H, int /*D*/, int N,
    int rows_per_tile, int num_tiles)
{
    // grid = (N * num_tiles, H, 1)
    const int tn = blockIdx.x;
    const int h = blockIdx.y;
    if (h >= H)
        return;

    const int n = tn / num_tiles;
    const int t = tn - n * num_tiles; // tn % num_tiles
    if (n >= N || t >= num_tiles)
        return;

    const int req = n;
    const int L = request_length[req];
    const int V = (L < M) ? L : M;

    const int idxT = ((n * H + h) * num_tiles) + t;

    if (V <= 0)
    {
        if (threadIdx.x == 0)
        {
            partial_min[idxT] = CUDART_INF_F;
            partial_idx[idxT] = 0;
        }
        return;
    }

    const int tile_start = t * rows_per_tile;
    if (tile_start >= V)
    {
        if (threadIdx.x == 0)
        {
            partial_min[idxT] = CUDART_INF_F;
            partial_idx[idxT] = 0;
        }
        return;
    }
    const int tile_end = min(tile_start + rows_per_tile, V);
    const int rows_in_tile = tile_end - tile_start;

    extern __shared__ float q_s[]; // size = DVAL floats (provided by launcher)

    // Load query[n,h,:] to shared
    const __nv_bfloat16 *q_ptr = queries + (((size_t)n * H + h) * (size_t)DVAL);
    const __nv_bfloat162 *q2 = reinterpret_cast<const __nv_bfloat162 *>(q_ptr);
    const int D2 = DVAL >> 1;
    for (int d2 = threadIdx.x; d2 < D2; d2 += BLOCK_THREADS)
    {
        const __nv_bfloat162 v2 = q2[d2];
        const float2 f2 = __bfloat1622float2(v2);
        const int d0 = d2 << 1;
        q_s[d0 + 0] = f2.x;
        q_s[d0 + 1] = f2.y;
    }
    __syncthreads();

    const int warp_id = threadIdx.x >> 5; // /32
    const int lane_id = threadIdx.x & 31; // %32
    const int WARPS = BLOCK_THREADS >> 5;

    float best_val = CUDART_INF_F;
    int best_idx = 0;

    // Each warp evaluates rows rr = warp_id, warp_id + WARPS, ...
    for (int rr = warp_id; rr < rows_in_tile; rr += WARPS)
    {
        const int local_slot = tile_start + rr;
        const __nv_bfloat16 *c_ptr =
            q_cache + (((size_t)req * M + (size_t)local_slot) * H + h) * (size_t)DVAL;

        // ---- Multi-stage L2 prefetch (best-effort) ----
#pragma unroll
        for (int k = 1; k <= MACMATCH_PREFETCH_STAGES; ++k)
        {
            const int rr_pref = rr + k * WARPS;
            if (rr_pref < rows_in_tile)
            {
                const int next_slot = tile_start + rr_pref;
                const __nv_bfloat16 *next_ptr =
                    q_cache + (((size_t)req * M + (size_t)next_slot) * H + h) * (size_t)DVAL;
                prefetch_L2(next_ptr);
            }
        }

        // ---- Vectorized accumulate over DVAL (assumed even) ----
        float acc = 0.f;

        // Main vector path: pairs of bf16 (2-wide)
        const __nv_bfloat162 *c2 = reinterpret_cast<const __nv_bfloat162 *>(c_ptr);
        const int D2 = DVAL >> 1; // number of bf16 pairs

#pragma unroll
        for (int d2 = lane_id; d2 < D2; d2 += 32)
        {
            const __nv_bfloat162 v2 = c2[d2];
            const float2 cf = __bfloat1622float2(v2);
            const int d0 = (d2 << 1);
            const float q0 = q_s[d0 + 0];
            const float q1 = q_s[d0 + 1];
            const float dx0 = q0 - cf.x;
            const float dx1 = q1 - cf.y;
            acc = fmaf(dx0, dx0, acc);
            acc = fmaf(dx1, dx1, acc);
        }

        // Warp reduce sum to lane 0
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
        {
            acc += __shfl_xor_sync(0xffffffff, acc, mask);
        }

        if (lane_id == 0)
        {
            if (better_pair(acc, local_slot, best_val, best_idx))
            {
                best_val = acc;
                best_idx = local_slot;
            }
        }
    }

    // Per-warp winners -> shared
    __shared__ float warp_vals[BLOCK_THREADS / 32];
    __shared__ int warp_idxs[BLOCK_THREADS / 32];
    if (lane_id == 0)
    {
        warp_vals[warp_id] = best_val;
        warp_idxs[warp_id] = best_idx;
    }
    __syncthreads();

    // Reduce per-warp winners using warp 0
    float v = CUDART_INF_F;
    int i = 0;
    if (threadIdx.x < 32)
    {
        int wid = threadIdx.x; // 0..31
        if (wid < WARPS)
        {
            v = warp_vals[wid];
            i = warp_idxs[wid];
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
        {
            const float v2 = __shfl_down_sync(0xffffffff, v, off);
            const int i2 = __shfl_down_sync(0xffffffff, i, off);
            if (better_pair(v2, i2, v, i))
            {
                v = v2;
                i = i2;
            }
        }
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        partial_min[idxT] = v;
        partial_idx[idxT] = i;
    }
}

// ===============================================================
// tile_min_kernel_fast_GEN  (generic D, with vector tail handling)

// ===============================================================
template <int BLOCK_THREADS>
__global__ void tile_min_kernel_fast_GEN(
    const __nv_bfloat16 *__restrict__ q_cache,  // [R,M,H,D]
    const int32_t *__restrict__ request_length, // [R]
    const __nv_bfloat16 *__restrict__ queries,  // [N,H,D]
    float *__restrict__ partial_min,            // [N,H,T]
    int32_t *__restrict__ partial_idx,          // [N,H,T]
    int R, int M, int H, int D, int N,
    int rows_per_tile, int num_tiles)
{
    // grid = (N * num_tiles, H, 1)
    const int tn = blockIdx.x;
    const int h = blockIdx.y;
    if (h >= H)
        return;

    const int n = tn / num_tiles;
    const int t = tn - n * num_tiles; // tn % num_tiles
    if (n >= N || t >= num_tiles)
        return;

    const int req = n;
    const int L = request_length[req];
    const int V = (L < M) ? L : M;

    const int idxT = ((n * H + h) * num_tiles) + t;

    if (V <= 0)
    {
        if (threadIdx.x == 0)
        {
            partial_min[idxT] = CUDART_INF_F;
            partial_idx[idxT] = 0;
        }
        return;
    }

    const int tile_start = t * rows_per_tile;
    if (tile_start >= V)
    {
        if (threadIdx.x == 0)
        {
            partial_min[idxT] = CUDART_INF_F;
            partial_idx[idxT] = 0;
        }
        return;
    }
    const int tile_end = min(tile_start + rows_per_tile, V);
    const int rows_in_tile = tile_end - tile_start;

    extern __shared__ float q_s[]; // size = D floats (provided by launcher)

    // Load query[n,h,:] to shared
    const __nv_bfloat16 *q_ptr = queries + (((size_t)n * H + h) * (size_t)D);
    const __nv_bfloat162 *q2 = reinterpret_cast<const __nv_bfloat162 *>(q_ptr);
    const int D2 = D >> 1;
    for (int d2 = threadIdx.x; d2 < D2; d2 += BLOCK_THREADS)
    {
        const __nv_bfloat162 v2 = q2[d2];
        const float2 f2 = __bfloat1622float2(v2);
        const int d0 = d2 << 1;
        q_s[d0 + 0] = f2.x;
        q_s[d0 + 1] = f2.y;
    }
    __syncthreads();

    const int warp_id = threadIdx.x >> 5; // /32
    const int lane_id = threadIdx.x & 31; // %32
    const int WARPS = BLOCK_THREADS >> 5;

    float best_val = CUDART_INF_F;
    int best_idx = 0;

    // Each warp evaluates rows rr = warp_id, warp_id + WARPS, ...
    for (int rr = warp_id; rr < rows_in_tile; rr += WARPS)
    {
        const int local_slot = tile_start + rr;
        const __nv_bfloat16 *c_ptr =
            q_cache + (((size_t)req * M + (size_t)local_slot) * H + h) * (size_t)D;

        // ---- Multi-stage L2 prefetch (best-effort) ----
#pragma unroll
        for (int k = 1; k <= MACMATCH_PREFETCH_STAGES; ++k)
        {
            const int rr_pref = rr + k * WARPS;
            if (rr_pref < rows_in_tile)
            {
                const int next_slot = tile_start + rr_pref;
                const __nv_bfloat16 *next_ptr =
                    q_cache + (((size_t)req * M + (size_t)next_slot) * H + h) * (size_t)D;
                prefetch_L2(next_ptr);
            }
        }

        // ---- Vectorized + scalar-tail accumulate over D ----
        float acc = 0.f;

        const int D_pairs = D >> 1; // number of bf16 pairs
        const __nv_bfloat162 *c2 = reinterpret_cast<const __nv_bfloat162 *>(c_ptr);

        // Vector part: process pairs (2 elements per lane per iter)
#pragma unroll
        for (int d2 = lane_id; d2 < D_pairs; d2 += 32)
        {
            const __nv_bfloat162 v2 = c2[d2];
            const float2 cf = __bfloat1622float2(v2);
            const int d0 = (d2 << 1);
            const float q0 = q_s[d0 + 0];
            const float q1 = q_s[d0 + 1];
            const float dx0 = q0 - cf.x;
            const float dx1 = q1 - cf.y;
            acc = fmaf(dx0, dx0, acc);
            acc = fmaf(dx1, dx1, acc);
        }

        // Scalar tail if D is odd (at most one extra index per warp step)
        if (D & 1)
        {
            const int last = D - 1;
            // Only the lane that would handle index "last" accumulates it.
            // We align the lane that hits (last >> 1) pair's "odd" tail.
            if (((last >> 1) - lane_id) % 32 == 0)
            {
                const float qf = q_s[last];
                const float cf = bf16_to_fp32(c_ptr[last]);
                const float df = qf - cf;
                acc = fmaf(df, df, acc);
            }
        }

        // Warp reduce sum to lane 0
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
        {
            acc += __shfl_xor_sync(0xffffffff, acc, mask);
        }

        if (lane_id == 0)
        {
            if (better_pair(acc, local_slot, best_val, best_idx))
            {
                best_val = acc;
                best_idx = local_slot;
            }
        }
    }

    // Per-warp winners -> shared
    __shared__ float warp_vals[BLOCK_THREADS / 32];
    __shared__ int warp_idxs[BLOCK_THREADS / 32];
    if (lane_id == 0)
    {
        warp_vals[warp_id] = best_val;
        warp_idxs[warp_id] = best_idx;
    }
    __syncthreads();

    // Reduce per-warp winners using warp 0
    float v = CUDART_INF_F;
    int i = 0;
    if (threadIdx.x < 32)
    {
        int wid = threadIdx.x; // 0..31
        if (wid < WARPS)
        {
            v = warp_vals[wid];
            i = warp_idxs[wid];
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
        {
            const float v2 = __shfl_down_sync(0xffffffff, v, off);
            const int i2 = __shfl_down_sync(0xffffffff, i, off);
            if (better_pair(v2, i2, v, i))
            {
                v = v2;
                i = i2;
            }
        }
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        partial_min[idxT] = v;
        partial_idx[idxT] = i;
    }
}

// --- Warp-only final reducer: one warp handles one (n,h) ---
// No __syncthreads(), no shared-memory tree. Pure warp shuffles.

#ifndef CUDART_INF_F
#define CUDART_INF_F __int_as_float(0x7f800000)
#endif

// Comparator identical to previous "better_pair": lower value wins,
// and on ties, lower index wins.
__device__ __forceinline__ bool better_pair_fi(float v2, int i2, float v1, int i1)
{
    return (v2 < v1) || ((v2 == v1) && (i2 < i1));
}

__global__ void final_reduce_emit_warpwise_kernel(
    const float *__restrict__ partial_min,      // [N,H,T]
    const int32_t *__restrict__ partial_idx,    // [N,H,T]
    const int32_t *__restrict__ request_length, // [R] (R>=N, implicit req=n)
    uint8_t *__restrict__ hit_table,            // [N,H] (u8 bool)
    int32_t *__restrict__ left_start,           // [N,H]
    int32_t *__restrict__ local_out,            // [N,H] (local ring slot 0..M-1)
    int N, int H, int M, int D, int num_tiles,
    float threshold, int my_offset)
{
    const int WARPS = blockDim.x >> 5;    // warps per CTA
    const int warp_id = threadIdx.x >> 5; // 0..WARPS-1
    const int lane = threadIdx.x & 31;    // 0..31

    const int NH = N * H;
    const int nh_lin = blockIdx.x * WARPS + warp_id;
    if (nh_lin >= NH)
        return;

    const int n = nh_lin / H;
    const int h = nh_lin - n * H;

    // Strided over tiles by lane
    const int base = (nh_lin * num_tiles);

    float best_v = CUDART_INF_F;
    int best_i = 0;

    // Each lane scans t = lane, lane+32, ...
    for (int t = lane; t < num_tiles; t += 32)
    {
        const float v = partial_min[base + t];
        const int i = partial_idx[base + t];
        if (better_pair_fi(v, i, best_v, best_i))
        {
            best_v = v;
            best_i = i;
        }
    }

    // Warp reduce (min, idx) with tie-break
    // NOTE: Using XOR/down both work; stick to down for readability.
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
    {
        const float v2 = __shfl_down_sync(0xffffffff, best_v, off);
        const int i2 = __shfl_down_sync(0xffffffff, best_i, off);
        if (better_pair_fi(v2, i2, best_v, best_i))
        {
            best_v = v2;
            best_i = i2;
        }
    }

    // Lane 0 of the warp emits results
    if (lane == 0)
    {
        const int p = best_i; // local ring slot 0..M-1
        local_out[nh_lin] = p;

        // Threshold on unsquared L2: sqrt(min_sq) < sqrt(2D)*(1 - threshold)
        const float one_minus = 1.f - threshold;
        const bool allow = (one_minus > 0.f);
        const float T_sq = 2.f * float(D) * one_minus * one_minus;
        const bool hit = allow && (best_v < T_sq) && (best_v < CUDART_INF_F);

        // Map local slot -> global index (ring)
        const int L = request_length[n];
        int g = 0;
        if (L > 0)
        {
            if (L < M)
            {
                g = p;
            }
            else
            {
                const int base_g = L - M;
                const int tail = L % M;
                int order_idx = p - tail;
                if (order_idx < 0)
                    order_idx += M;
                g = base_g + order_idx;
            }
        }

        // left_start = hit ? max((g+1) - my_offset, 0) : 0
        int32_t left_val = 0;
        if (hit)
        {
            long long tmp = (long long)g + 1LL - (long long)my_offset;
            left_val = (tmp > 0) ? (int32_t)tmp : 0;
        }

        hit_table[nh_lin] = (uint8_t)(hit ? 1 : 0);
        left_start[nh_lin] = left_val;
    }
}

// ------------------------------ Launchers ------------------------------------

// Adaptive rows_per_tile chooser.
// Goal: N * H * num_tiles >= TARGET_WAVES * #SMs  (good latency hiding for small N)
// Keeps manual override if rows_per_stage is in the allowed set.

#ifndef MACMATCH_TARGET_WAVES
#define MACMATCH_TARGET_WAVES 12 // try 8..16 as you like
#endif

#ifndef MACMATCH_MIN_RPT
#define MACMATCH_MIN_RPT 8 // allow small tiles for N≈1
#endif

#ifndef MACMATCH_MAX_RPT
#define MACMATCH_MAX_RPT 256
#endif

static inline int pick_rows_per_stage(int rows_per_stage, int64_t M, int N, int H)
{
    // Accept explicit user choice (common legal values)
    if (rows_per_stage == 8 || rows_per_stage == 16 ||
        rows_per_stage == 32 || rows_per_stage == 64 ||
        rows_per_stage == 128 || rows_per_stage == 256)
    {
        return rows_per_stage;
    }

    // Sanity guards
    if (M <= 0)
        return 64;
    if (N <= 0)
        N = 1;
    if (H <= 0)
        H = 1;

    // Query #SMs (fallback to 132 if runtime query is unavailable)
    int dev = 0, sms = 132;
    if (cudaGetDevice(&dev) == cudaSuccess)
    {
        int tmp = 0;
        if (cudaDeviceGetAttribute(&tmp, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess)
        {
            sms = tmp;
        }
    }

    // How many tiles do we need to reach the target number of CTAs?
    // CTAs = N * H * num_tiles ;  num_tiles = ceil(M / rpt)
    const long long target_ctas = (long long)MACMATCH_TARGET_WAVES * (long long)sms;
    const long long nh = (long long)N * (long long)H;
    long long needed_tiles = (target_ctas + nh - 1) / nh; // ceil
    if (needed_tiles < 1)
        needed_tiles = 1;
    if (needed_tiles > M)
        needed_tiles = M; // cannot exceed M

    // Ideal rows_per_tile to meet the tile target
    // rpt_target = ceil(M / needed_tiles)
    long long rpt_target = (M + needed_tiles - 1) / needed_tiles;

    // Choose from allowed set (prefer largest rpt that still satisfies needed_tiles)
    static const int allowed_desc[] = {256, 128, 64, 32, 16, 8}; // descending
    long long floorRpt = M / needed_tiles;                       // largest rpt that still meets tiles
    if (floorRpt < 1)
        floorRpt = 1;

    int best = MACMATCH_MIN_RPT; // default to smallest (most tiles)
    for (int a : allowed_desc)
    {
        if (a <= MACMATCH_MAX_RPT && a >= MACMATCH_MIN_RPT && a <= floorRpt)
        {
            best = a; // largest allowed ≤ floorRpt
            break;
        }
    }

    // If even the smallest allowed can't meet needed_tiles (rare), best already = MIN_RPT (max tiles).
    if (best > (int)M)
        best = (int)M; // clamp if M < MIN_RPT
    return best;
}

// -----------------------------------------------------------------------------
// Helper: snap to supported warps {8,10,12}, not exceeding 'cap' (rows_per_tile).
// Returns the largest supported value <= cap. Assumes rows_per_tile >= 8.
// -----------------------------------------------------------------------------
static inline int snap_supported_warps_leq(int cap)
{
    // cap is the upper bound (e.g., rows_per_tile), we snap DOWN to {12,10,8}.
    if (cap >= 12)
        return 12;
    if (cap >= 10)
        return 10;
    return 8; // rows_per_tile is always in {8,16,32,64,128,256} in our flow.
}

// -----------------------------------------------------------------------------
// pick_block_threads (UPDATED)
//   - Caps warps by tile height: warps <= rows_per_tile
//   - Snaps to supported {8,10,12} so launcher paths (256/320/384) are used.
//   - Honors MACMATCH_FORCE_WARPS and load_warps_arg, but still caps by tile.
// -----------------------------------------------------------------------------
static inline int pick_block_threads(int D, int rows_per_tile,
                                     int64_t M, int64_t N, int64_t H,
                                     int load_warps_arg)
{
#ifdef MACMATCH_FORCE_WARPS
    // Force, but still cap by tile height and snap to {8,10,12}
    int forced = MACMATCH_FORCE_WARPS;
    if (forced < 4)
        forced = 4;
    if (forced > 12)
        forced = 12;
    int warps = snap_supported_warps_leq(std::min(forced, rows_per_tile));
    return warps * 32;
#else
    // If user hinted warps, accept 8/10/12 exactly; for any other value [4..12],
    // cap by tile height, then snap DOWN to {8,10,12}.
    if (load_warps_arg >= 4 && load_warps_arg <= 12)
    {
        int hinted = std::min(load_warps_arg, rows_per_tile);
        int warps = snap_supported_warps_leq(hinted);
        return warps * 32;
    }

    // Heuristic desired warps BEFORE capping (same policy as before)
    const int NH = (int)(N * H);
    int desired;
    if (D == 128)
    {
        desired = (rows_per_tile >= 128 || M >= 16384 || NH < 256) ? 12 : 10;
    }
    else
    { // D == 64 (or smaller)
        desired = (rows_per_tile >= 128 || M >= 16384 || NH < 256) ? 10 : 8;
    }

    // NEW: cap by tile height, then snap to {8,10,12}
    int warps = snap_supported_warps_leq(std::min(desired, rows_per_tile));
    return warps * 32;
#endif
}

// mac_ring_match launcher (adaptive)

// ------------------------------ Launchers ------------------------------------

// pick_rows_per_stage / pick_block_threads remain unchanged.

// mac_ring_match launcher (adaptive) — now writes outputs in-place
static inline std::tuple<at::Tensor, at::Tensor, at::Tensor> mac_ring_match_launcher(
    at::Tensor q_cache,        // [R,M,H,D] bf16
    at::Tensor request_length, // [R] int32
    at::Tensor query,          // [N,H,D] bf16
    at::Tensor req_ids,        // [N] int32  (ignored; kept for API compatibility)
    double threshold,          // float
    int rows_per_stage,        // tile height
    int load_warps,            // warps/CTA override (optional; 0 keeps heuristic)
    int my_offset,             // int32

    // New: caller-provided output tensors (all on same CUDA device)
    at::Tensor hit_table,  // [N,H] bool
    at::Tensor left_start, // [N,H] int32
    at::Tensor indices_out // [N,H] int32
)
{
    // ---- Input checks (unchanged) ----
    checkContigCuda(q_cache, "q_cache");
    checkContigCuda(request_length, "request_length");
    checkContigCuda(query, "query");
    checkContigCuda(req_ids, "req_ids");
    TORCH_CHECK(q_cache.scalar_type() == at::kBFloat16, "q_cache must be bf16");
    TORCH_CHECK(query.scalar_type() == at::kBFloat16, "query must be bf16");
    TORCH_CHECK(request_length.scalar_type() == at::kInt, "request_length must be int32");
    TORCH_CHECK(req_ids.scalar_type() == at::kInt, "req_ids must be int32"); // API compat

    // ---- Shapes & basic relations ----
    int64_t R = q_cache.size(0), M = q_cache.size(1), H = q_cache.size(2), D = q_cache.size(3);
    int64_t N = query.size(0);
    TORCH_CHECK(query.size(1) == H && query.size(2) == D, "query shape mismatch");
    TORCH_CHECK(req_ids.size(0) == N, "req_ids length mismatch (ignored for mapping)");
    TORCH_CHECK(N <= R, "Implicit mapping requires N <= R (got N=", N, ", R=", R, ")");

    // ---- Output checks (new) ----
    checkContigCuda(hit_table, "hit_table");
    checkContigCuda(left_start, "left_start");
    checkContigCuda(indices_out, "indices_out");

    TORCH_CHECK(hit_table.scalar_type() == at::kBool, "hit_table must be bool");
    TORCH_CHECK(left_start.scalar_type() == at::kInt, "left_start must be int32");
    TORCH_CHECK(indices_out.scalar_type() == at::kInt, "indices_out must be int32");

    // All outputs must be on the same device as inputs
    auto dev = q_cache.device();
    TORCH_CHECK(hit_table.device() == dev, "hit_table must be on same device as q_cache");
    TORCH_CHECK(left_start.device() == dev, "left_start must be on same device as q_cache");
    TORCH_CHECK(indices_out.device() == dev, "indices_out must be on same device as q_cache");
    TORCH_CHECK(request_length.device() == dev &&
                    query.device() == dev &&
                    req_ids.device() == dev,
                "All inputs must be on the same device");

    // Shape checks for outputs
    TORCH_CHECK(hit_table.dim() == 2 && hit_table.size(0) == N && hit_table.size(1) == H,
                "hit_table must have shape [N,H] = [", N, ",", H, "]");
    TORCH_CHECK(left_start.dim() == 2 && left_start.size(0) == N && left_start.size(1) == H,
                "left_start must have shape [N,H] = [", N, ",", H, "]");
    TORCH_CHECK(indices_out.dim() == 2 && indices_out.size(0) == N && indices_out.size(1) == H,
                "indices_out must have shape [N,H] = [", N, ",", H, "]");

    c10::cuda::CUDAGuard device_guard(query.device());
    auto at_stream = at::cuda::getCurrentCUDAStream(query.device().index());
    cudaStream_t cuda_stream = at_stream.stream();

    // ---- Tile plan (unchanged) ----
    const int RPT = pick_rows_per_stage(rows_per_stage, M, (int)N, (int)H);
    int num_tiles = (int)((M + RPT - 1) / RPT);

    // ---- Internal partial buffers [N,H,num_tiles] (unchanged) ----
    auto f32opt = query.options().dtype(at::kFloat);
    auto i32opt = req_ids.options().dtype(at::kInt);
    at::Tensor partial_min = at::empty({N, H, num_tiles}, f32opt);
    at::Tensor partial_idx = at::empty({N, H, num_tiles}, i32opt);

    // ---- Decide block size (unchanged) ----
    const int block_threads = pick_block_threads((int)D, RPT, M, N, H, load_warps);

    // ---- Phase 1: tile pass (unchanged) ----
    {
        const size_t smem_bytes = (size_t)D * sizeof(float);
        dim3 grid((unsigned)(N * (int64_t)num_tiles), (unsigned)H, 1);

        if (block_threads == 256)
        {
            const int BLOCK = 256;
            if (D == 64)
            {
                tile_min_kernel_fast_D<BLOCK, 64><<<grid, BLOCK, smem_bytes, cuda_stream>>>(
                    reinterpret_cast<const __nv_bfloat16 *>(q_cache.data_ptr<at::BFloat16>()),
                    request_length.data_ptr<int32_t>(),
                    reinterpret_cast<const __nv_bfloat16 *>(query.data_ptr<at::BFloat16>()),
                    partial_min.data_ptr<float>(),
                    partial_idx.data_ptr<int32_t>(),
                    (int)R, (int)M, (int)H, (int)D, (int)N, RPT, num_tiles);
            }
            else if (D == 128)
            {
                tile_min_kernel_fast_D<BLOCK, 128><<<grid, BLOCK, smem_bytes, cuda_stream>>>(
                    reinterpret_cast<const __nv_bfloat16 *>(q_cache.data_ptr<at::BFloat16>()),
                    request_length.data_ptr<int32_t>(),
                    reinterpret_cast<const __nv_bfloat16 *>(query.data_ptr<at::BFloat16>()),
                    partial_min.data_ptr<float>(),
                    partial_idx.data_ptr<int32_t>(),
                    (int)R, (int)M, (int)H, (int)D, (int)N, RPT, num_tiles);
            }
            else
            {
                tile_min_kernel_fast_GEN<BLOCK><<<grid, BLOCK, smem_bytes, cuda_stream>>>(
                    reinterpret_cast<const __nv_bfloat16 *>(q_cache.data_ptr<at::BFloat16>()),
                    request_length.data_ptr<int32_t>(),
                    reinterpret_cast<const __nv_bfloat16 *>(query.data_ptr<at::BFloat16>()),
                    partial_min.data_ptr<float>(),
                    partial_idx.data_ptr<int32_t>(),
                    (int)R, (int)M, (int)H, (int)D, (int)N, RPT, num_tiles);
            }
        }
        else if (block_threads == 320)
        {
            const int BLOCK = 320;
            if (D == 64)
            {
                tile_min_kernel_fast_D<BLOCK, 64><<<grid, BLOCK, smem_bytes, cuda_stream>>>(
                    reinterpret_cast<const __nv_bfloat16 *>(q_cache.data_ptr<at::BFloat16>()),
                    request_length.data_ptr<int32_t>(),
                    reinterpret_cast<const __nv_bfloat16 *>(query.data_ptr<at::BFloat16>()),
                    partial_min.data_ptr<float>(),
                    partial_idx.data_ptr<int32_t>(),
                    (int)R, (int)M, (int)H, (int)D, (int)N, RPT, num_tiles);
            }
            else if (D == 128)
            {
                tile_min_kernel_fast_D<BLOCK, 128><<<grid, BLOCK, smem_bytes, cuda_stream>>>(
                    reinterpret_cast<const __nv_bfloat16 *>(q_cache.data_ptr<at::BFloat16>()),
                    request_length.data_ptr<int32_t>(),
                    reinterpret_cast<const __nv_bfloat16 *>(query.data_ptr<at::BFloat16>()),
                    partial_min.data_ptr<float>(),
                    partial_idx.data_ptr<int32_t>(),
                    (int)R, (int)M, (int)H, (int)D, (int)N, RPT, num_tiles);
            }
            else
            {
                tile_min_kernel_fast_GEN<BLOCK><<<grid, BLOCK, smem_bytes, cuda_stream>>>(
                    reinterpret_cast<const __nv_bfloat16 *>(q_cache.data_ptr<at::BFloat16>()),
                    request_length.data_ptr<int32_t>(),
                    reinterpret_cast<const __nv_bfloat16 *>(query.data_ptr<at::BFloat16>()),
                    partial_min.data_ptr<float>(),
                    partial_idx.data_ptr<int32_t>(),
                    (int)R, (int)M, (int)H, (int)D, (int)N, RPT, num_tiles);
            }
        }
        else if (block_threads == 384)
        {
            const int BLOCK = 384;
            if (D == 64)
            {
                tile_min_kernel_fast_D<BLOCK, 64><<<grid, BLOCK, smem_bytes, cuda_stream>>>(
                    reinterpret_cast<const __nv_bfloat16 *>(q_cache.data_ptr<at::BFloat16>()),
                    request_length.data_ptr<int32_t>(),
                    reinterpret_cast<const __nv_bfloat16 *>(query.data_ptr<at::BFloat16>()),
                    partial_min.data_ptr<float>(),
                    partial_idx.data_ptr<int32_t>(),
                    (int)R, (int)M, (int)H, (int)D, (int)N, RPT, num_tiles);
            }
            else if (D == 128)
            {
                tile_min_kernel_fast_D<BLOCK, 128><<<grid, BLOCK, smem_bytes, cuda_stream>>>(
                    reinterpret_cast<const __nv_bfloat16 *>(q_cache.data_ptr<at::BFloat16>()),
                    request_length.data_ptr<int32_t>(),
                    reinterpret_cast<const __nv_bfloat16 *>(query.data_ptr<at::BFloat16>()),
                    partial_min.data_ptr<float>(),
                    partial_idx.data_ptr<int32_t>(),
                    (int)R, (int)M, (int)H, (int)D, (int)N, RPT, num_tiles);
            }
            else
            {
                tile_min_kernel_fast_GEN<BLOCK><<<grid, BLOCK, smem_bytes, cuda_stream>>>(
                    reinterpret_cast<const __nv_bfloat16 *>(q_cache.data_ptr<at::BFloat16>()),
                    request_length.data_ptr<int32_t>(),
                    reinterpret_cast<const __nv_bfloat16 *>(query.data_ptr<at::BFloat16>()),
                    partial_min.data_ptr<float>(),
                    partial_idx.data_ptr<int32_t>(),
                    (int)R, (int)M, (int)H, (int)D, (int)N, RPT, num_tiles);
            }
        }
        else
        {
            // Fallback (shouldn't happen)
            const int BLOCK = MACMATCH_FALLBACK_WARPS * 32;
            tile_min_kernel_fast_GEN<BLOCK><<<grid, BLOCK, smem_bytes, cuda_stream>>>(
                reinterpret_cast<const __nv_bfloat16 *>(q_cache.data_ptr<at::BFloat16>()),
                request_length.data_ptr<int32_t>(),
                reinterpret_cast<const __nv_bfloat16 *>(query.data_ptr<at::BFloat16>()),
                partial_min.data_ptr<float>(),
                partial_idx.data_ptr<int32_t>(),
                (int)R, (int)M, (int)H, (int)D, (int)N, RPT, num_tiles);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // ---- Phase 2: final reducer — writes into provided outputs (unchanged mapping) ----
    {
        const int NH = (int)(N * H);

        int warps_per_cta;
        if (NH < 128)
            warps_per_cta = 1;
        else if (NH < 512)
            warps_per_cta = 2;
        else if (NH < 2048)
            warps_per_cta = 4;
        else
            warps_per_cta = 8;

        const int block_threads2 = warps_per_cta * 32;
        const int grid_x = (NH + warps_per_cta - 1) / warps_per_cta;

        dim3 grid((unsigned)grid_x, 1, 1);
        dim3 block((unsigned)block_threads2, 1, 1);

        final_reduce_emit_warpwise_kernel<<<grid, block, 0, cuda_stream>>>(
            partial_min.data_ptr<float>(),
            partial_idx.data_ptr<int32_t>(),
            request_length.data_ptr<int32_t>(),
            reinterpret_cast<uint8_t *>(hit_table.data_ptr<bool>()),
            left_start.data_ptr<int32_t>(),
            indices_out.data_ptr<int32_t>(),
            (int)N, (int)H, (int)M, (int)D, (int)num_tiles,
            (float)threshold, (int)my_offset);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // ---- Tie provided outputs to current stream (unchanged pattern) ----
    auto streamGuard = at_stream;
    c10::cuda::CUDACachingAllocator::recordStream(hit_table.storage().data_ptr(), streamGuard);
    c10::cuda::CUDACachingAllocator::recordStream(left_start.storage().data_ptr(), streamGuard);
    c10::cuda::CUDACachingAllocator::recordStream(indices_out.storage().data_ptr(), streamGuard);

    // Return them for API convenience
    return {hit_table, left_start, indices_out};
}

// ------------------------------ PyBind exports -------------------------------

static inline int64_t ceil_div_i64(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

// Occupancy helper using the generic-D tile kernel and dynamic shared memory.
// We instantiate for BLOCK_THREADS in {256, 320, 384}.
template <int BLOCK_THREADS>
static inline int occupancy_blocks_per_sm_GEN(int dev, int D)
{
    int blocks_per_sm = 0;
    const size_t smem_bytes = (size_t)D * sizeof(float); // q_s in tile kernel
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, tile_min_kernel_fast_GEN<BLOCK_THREADS>, BLOCK_THREADS, smem_bytes));
    return blocks_per_sm;
}

static inline void get_device_props(int &dev, int &num_sms)
{
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev));
}
// Optional: cap the amount of padding to avoid large temporary buffers when NH is tiny.
#ifndef MACMATCH_MAX_PAD_TILES
#define MACMATCH_MAX_PAD_TILES 8 // increase if you want exact full-wave padding regardless of overhead
#endif

static inline py::dict mac_ring_match_schedule(
    int64_t N,          // batch size (queries)
    int64_t H,          // number of Q heads
    int64_t M,          // ring capacity
    int64_t D,          // head dim
    int load_warps_arg, // 0 => let scheduler search {8,10,12}; else 4..12
    double target_util, // e.g., 0.95
    bool allow_rpt_32   // allow rows_per_stage=32 in search
)
{
    TORCH_CHECK(N >= 0 && H >= 0 && M >= 0 && D > 0, "Invalid (N,H,M,D)");
    TORCH_CHECK(target_util > 0.0 && target_util <= 1.0, "target_util must be in (0,1]");

    // Candidate rows_per_stage (largest first to minimize tiles if saturation is possible).
    std::vector<int> rpt_candidates;
    rpt_candidates.push_back(256);
    rpt_candidates.push_back(128);
    rpt_candidates.push_back(64);
    if (allow_rpt_32)
        rpt_candidates.push_back(32);

    // Candidate warps/CTA (threads = warps * 32).
    std::vector<int> warp_candidates;
    if (load_warps_arg >= 4 && load_warps_arg <= 12)
    {
        warp_candidates.push_back(load_warps_arg);
    }
    else
    {
        // Favor higher warps first for latency hiding on D=128, then 10, then 8.
        warp_candidates = (D == 128) ? std::vector<int>{12, 10, 8}
                                     : std::vector<int>{10, 12, 8};
    }

    // Device resident capacity for each warps/CTA choice.
    int dev = 0, num_sms = 0;
    get_device_props(dev, num_sms);

    struct Cand
    {
        int rpt, warps, block_threads, tiles, blocks_per_sm, max_grid;
        int64_t total_ctas, baseline_ctas;
        double util;
        bool feasible;
    };
    std::vector<Cand> all;

    // Iterate over (warps, rpt) pairs.
    for (int warps : warp_candidates)
    {
        const int block_threads = warps * 32;
        int blocks_per_sm = 0;
        if (block_threads == 256)
            blocks_per_sm = occupancy_blocks_per_sm_GEN<256>(dev, (int)D);
        else if (block_threads == 320)
            blocks_per_sm = occupancy_blocks_per_sm_GEN<320>(dev, (int)D);
        else if (block_threads == 384)
            blocks_per_sm = occupancy_blocks_per_sm_GEN<384>(dev, (int)D);
        else
        {
            // Clamp to supported values; fall back to GEN<256> for occupancy
            blocks_per_sm = occupancy_blocks_per_sm_GEN<256>(dev, (int)D);
        }
        const int max_grid_size = blocks_per_sm * num_sms;
        const int64_t baseline = N * H; // grid.x * grid.y without M-tiling

        for (int rpt : rpt_candidates)
        {
            int tiles = (M == 0) ? 1 : (int)ceil_div_i64(M, rpt);
            if (tiles <= 0)
                tiles = 1;
            const int64_t total = baseline * (int64_t)tiles;
            const double util = (max_grid_size > 0) ? (double)total / (double)max_grid_size : 0.0;
            const bool feasible = (max_grid_size == 0) ? (total > 0) : (util >= target_util);

            all.push_back(Cand{
                /*rpt=*/rpt, /*warps=*/warps, /*block_threads=*/block_threads,
                /*tiles=*/tiles, /*blocks_per_sm=*/blocks_per_sm, /*max_grid=*/max_grid_size,
                /*total_ctas=*/total, /*baseline_ctas=*/baseline, /*util=*/util, /*feasible=*/feasible});
        }
    }

    // Select best candidate (same policy as before).
    auto best_it = all.end();
    {
        // Pass 1: any feasible?
        for (auto it = all.begin(); it != all.end(); ++it)
        {
            if (!it->feasible)
                continue;
            if (best_it == all.end())
            {
                best_it = it;
                continue;
            }
            if (it->rpt > best_it->rpt)
            {
                best_it = it;
                continue;
            }
            if (it->rpt == best_it->rpt)
            {
                // prefer lower total_ctas (closer to exactly saturating)
                if (it->total_ctas < best_it->total_ctas)
                    best_it = it;
            }
        }
        // Pass 2: none feasible -> pick closest util to target, then largest rpt
        if (best_it == all.end())
        {
            double best_gap = std::numeric_limits<double>::infinity();
            for (auto it = all.begin(); it != all.end(); ++it)
            {
                const double gap = std::abs(it->util - std::min(target_util, 1.0));
                if (gap < best_gap)
                {
                    best_gap = gap;
                    best_it = it;
                    continue;
                }
                if (gap == best_gap && it->rpt > best_it->rpt)
                    best_it = it;
            }
        }
    }
    TORCH_CHECK(best_it != all.end(), "Scheduler failed to pick a configuration");

    // ---- NEW: pad to eliminate the partial-wave tail for the selected config ----
    const int blocks_per_sm = best_it->blocks_per_sm;
    const int max_grid_size = best_it->max_grid;     // = blocks_per_sm * num_sms
    const int64_t baseline = best_it->baseline_ctas; // = N * H
    const int tiles = best_it->tiles;

    int padded_tiles = tiles;
    int64_t total_ctas = best_it->total_ctas;
    int64_t padded_total = total_ctas;
    int pad_tiles = 0;
    int64_t pad_ctas = 0;

    if (max_grid_size > 0 && baseline > 0)
    {
        const int64_t quantum = (int64_t)blocks_per_sm * (int64_t)num_sms; // resident CTAs per full wave
        const int64_t rem = total_ctas % quantum;
        if (rem != 0)
        {
            int64_t need_ctas = quantum - rem;
            int64_t add_tiles = (need_ctas + baseline - 1) / baseline; // ceil(need / baseline)
            if (add_tiles > MACMATCH_MAX_PAD_TILES)
                add_tiles = MACMATCH_MAX_PAD_TILES; // safety cap
            if (add_tiles > 0)
            {
                padded_tiles = tiles + (int)add_tiles;
                padded_total = baseline * (int64_t)padded_tiles;
                pad_tiles = (int)add_tiles;
                pad_ctas = baseline * add_tiles;
            }
        }
    }

    // Package result
    py::dict out;
    out["rows_per_stage"] = best_it->rpt;
    out["load_warps"] = best_it->warps;            // pass into mac_ring_match (overrides heuristic)
    out["block_threads"] = best_it->block_threads; // derived

    // Return the PADDED values for launch
    out["num_tiles"] = padded_tiles;  // use this for the z-dimension of the tile pass
    out["total_ctas"] = padded_total; // N * H * padded_tiles

    // Keep the original metrics for insight
    out["baseline_ctas"] = baseline; // N * H
    out["num_tiles_unpadded"] = tiles;
    out["total_ctas_unpadded"] = total_ctas;

    out["padding_tiles"] = pad_tiles;    // how many tiles we added
    out["padding_ctas"] = pad_ctas;      // how many CTAs that equals to
    out["grid_quantum"] = max_grid_size; // resident CTAs per full wave

    out["blocks_per_sm"] = blocks_per_sm;
    out["num_sms"] = num_sms;
    out["max_grid_size"] = max_grid_size; // occupancy-based resident CTAs

    // Utilization reported on the PADDED grid (more accurate for what you'll launch)
    out["utilization"] = (max_grid_size > 0) ? (double)padded_total / (double)max_grid_size : 0.0;
    out["saturates_target"] = best_it->feasible; // util >= target_util (based on unpadded selection)

    out["candidates"] = py::list();
    for (const auto &c : all)
    {
        py::dict row;
        row["rpt"] = c.rpt;
        row["warps"] = c.warps;
        row["tiles"] = c.tiles;
        row["total_ctas"] = c.total_ctas;
        row["util"] = c.util;
        row["feasible"] = c.feasible;
        ((py::list)out["candidates"]).append(row);
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("ring_update_row_parallel",
          &ring_update_row_parallel_launcher,
          "Row-parallel ring-cache update (prefill/decode)");
    m.def("mac_ring_match",
          &mac_ring_match_launcher,
          "L2-argmin matcher over unified ring cache [R,M,H,D]");
    m.def("mac_ring_match_schedule",
          &mac_ring_match_schedule,
          R"doc(
          Occupancy-aware scheduler for mac_ring_match.
          Args:
            N: batch size (#queries)
            H: #Q heads
            M: ring capacity
            D: head dim
            load_warps_override: 0 to auto-search {8,10,12}; else 4..12 to force.
            target_util: target ratio of total CTAs vs. resident capacity (e.g. 0.95)
            allow_rpt_32: include rows_per_stage=32 in the search

          Returns: dict with
            rows_per_stage, load_warps, block_threads, num_tiles,
            total_ctas, baseline_ctas, blocks_per_sm, num_sms, max_grid_size,
            utilization, saturates_target, candidates (for debugging).
        )doc");
}
