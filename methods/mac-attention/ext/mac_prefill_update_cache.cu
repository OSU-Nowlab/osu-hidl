// mac_prefill_update_cache.cu
//
// Cooperative row-update kernel for MAC ring caches.
// - Copies rows from source tensors into (q_cache, attn_cache, lse_cache) at circular positions.
// - Performs a single in-kernel atomic bump to request_length[req] after copies.
// - Chooses a right-sized thread count based on row byte size.

#include <ATen/cuda/CUDAContext.h>   // getCurrentCUDAStream
#include <c10/cuda/CUDAGuard.h>      // c10::cuda::CUDAGuard
#include <c10/cuda/CUDAException.h>  // C10_CUDA_KERNEL_LAUNCH_CHECK
#include <c10/cuda/CUDAStream.h>     // CUDAStreamGuard
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <torch/extension.h>
#include <vector>

using torch::Tensor;

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_DTYPE(x, dt) TORCH_CHECK((x).dtype() == (dt), #x " has wrong dtype")

// --------------------------------- utils ----------------------------------

__device__ __forceinline__ void copy_bytes_16(void* __restrict__ dst_void,
                                              const void* __restrict__ src_void,
                                              size_t n_bytes) {
  if (n_bytes == 0) return;
  uintptr_t du = reinterpret_cast<uintptr_t>(dst_void);
  uintptr_t su = reinterpret_cast<uintptr_t>(src_void);
  uint8_t* __restrict__ d8 = (uint8_t*)dst_void;
  const uint8_t* __restrict__ s8 = (const uint8_t*)src_void;

  if (((du | su) & 0xF) == 0) {
    size_t n16 = n_bytes / 16;
    uint4* __restrict__ d16 = (uint4*)dst_void;
    const uint4* __restrict__ s16 = (const uint4*)src_void;
#pragma unroll 4
    for (size_t i = threadIdx.x; i < n16; i += blockDim.x) {
      uint4 v = s16[i];
      d16[i] = v;
    }
    size_t rem = n_bytes - n16 * 16;
    if (rem) {
      size_t base = n16 * 16;
      for (size_t i = threadIdx.x; i < rem; i += blockDim.x) {
        d8[base + i] = s8[base + i];
      }
    }
  } else if (((du | su) & 0x7) == 0) {
    size_t n8 = n_bytes / 8;
    uint2* __restrict__ d8b = (uint2*)dst_void;
    const uint2* __restrict__ s8b = (const uint2*)src_void;
#pragma unroll 4
    for (size_t i = threadIdx.x; i < n8; i += blockDim.x) {
      uint2 v = s8b[i];
      d8b[i] = v;
    }
    size_t rem = n_bytes - n8 * 8;
    if (rem) {
      size_t base = n8 * 8;
      for (size_t i = threadIdx.x; i < rem; i += blockDim.x) {
        d8[base + i] = s8[base + i];
      }
    }
  } else {
    for (size_t i = threadIdx.x; i < n_bytes; i += blockDim.x) {
      d8[i] = s8[i];
    }
  }
}

struct UpdateParams {
  void* q_cache;        // [R,M,H,D] bf16
  void* attn_cache;     // [R,M,H,D] bf16
  void* lse_cache;      // [R,M,H]    f32
  const void* q_src;    // [sumN,H,D] bf16
  const void* attn_src; // [sumN,H,D] bf16
  const void* lse_src;  // [sumN,H]   f32
  int R, M, H, D;
  const int32_t* req_ids;   // [B]
  int32_t B;
  const int32_t* offsets;   // [B+1] (rows)
  const int32_t* lens;      // [B]
  int32_t* request_length;  // [R]
};

// ------------------------------- helpers ----------------------------------

static void fill_params(UpdateParams& p,
                        Tensor q_cache,
                        Tensor attn_cache,
                        Tensor lse_cache,
                        Tensor request_length,
                        Tensor req_ids,
                        Tensor src_q,
                        Tensor src_attn,
                        Tensor src_lse,
                        Tensor offsets,
                        Tensor lens,
                        int capacity,
                        int num_heads,
                        int head_dim) {
  CHECK_CUDA(q_cache);
  CHECK_CONTIG(q_cache);
  CHECK_DTYPE(q_cache, torch::kBFloat16);
  CHECK_CUDA(attn_cache);
  CHECK_CONTIG(attn_cache);
  CHECK_DTYPE(attn_cache, torch::kBFloat16);
  CHECK_CUDA(lse_cache);
  CHECK_CONTIG(lse_cache);
  CHECK_DTYPE(lse_cache, torch::kFloat32);
  CHECK_CUDA(request_length);
  CHECK_CONTIG(request_length);
  CHECK_DTYPE(request_length, torch::kInt32);

  CHECK_CUDA(req_ids);
  CHECK_CONTIG(req_ids);
  CHECK_DTYPE(req_ids, torch::kInt32);
  CHECK_CUDA(lens);
  CHECK_CONTIG(lens);
  CHECK_DTYPE(lens, torch::kInt32);
  CHECK_CUDA(offsets);
  CHECK_CONTIG(offsets);
  CHECK_DTYPE(offsets, torch::kInt32);

  CHECK_CUDA(src_q);
  CHECK_CONTIG(src_q);
  CHECK_DTYPE(src_q, torch::kBFloat16);
  CHECK_CUDA(src_attn);
  CHECK_CONTIG(src_attn);
  CHECK_DTYPE(src_attn, torch::kBFloat16);
  CHECK_CUDA(src_lse);
  CHECK_CONTIG(src_lse);
  CHECK_DTYPE(src_lse, torch::kFloat32);

  int R = (int)q_cache.size(0);
  TORCH_CHECK(R > 0, "R must be > 0");
  TORCH_CHECK(q_cache.dim() == 4, "q_cache must be [R,M,H,D]");
  TORCH_CHECK(q_cache.size(1) == capacity, "capacity (M) mismatch");
  TORCH_CHECK(q_cache.size(2) == num_heads, "num_heads (H) mismatch");
  TORCH_CHECK(q_cache.size(3) == head_dim, "head_dim (D) mismatch");
  TORCH_CHECK(attn_cache.sizes() == q_cache.sizes(), "attn_cache must match q_cache");
  TORCH_CHECK(
      lse_cache.size(0) == R && lse_cache.size(1) == capacity && lse_cache.size(2) == num_heads,
      "lse_cache shape mismatch");

  TORCH_CHECK(src_q.dim() == 3 && src_q.size(1) == num_heads && src_q.size(2) == head_dim,
              "src_q must be [sumN,H,D]");
  TORCH_CHECK(src_attn.sizes() == src_q.sizes(), "src_attn must match src_q");
  TORCH_CHECK(src_lse.dim() == 2 && src_lse.size(1) == num_heads, "src_lse must be [sumN,H]");
  TORCH_CHECK(offsets.size(0) == (req_ids.size(0) + 1), "offsets must be [B+1]");

  p.q_cache = q_cache.data_ptr();
  p.attn_cache = attn_cache.data_ptr();
  p.lse_cache = lse_cache.data_ptr();
  p.q_src = src_q.data_ptr();
  p.attn_src = src_attn.data_ptr();
  p.lse_src = src_lse.data_ptr();
  p.R = R;
  p.M = capacity;
  p.H = num_heads;
  p.D = head_dim;
  p.req_ids = req_ids.data_ptr<int32_t>();
  p.B = (int)req_ids.size(0);
  p.offsets = offsets.data_ptr<int32_t>();
  p.lens = lens.data_ptr<int32_t>();
  p.request_length = request_length.data_ptr<int32_t>();
}

static inline int next_pow2_clamped(int v) {
  int x = 1;
  while (x < v && x < 256) x <<= 1;
  if (x < 32) x = 32;
  if (x > 256) x = 256;
  return x;
}

// ------------------------------- kernels ----------------------------------

__global__ void mac_prefill_update_cache_kernel(UpdateParams p) {
  const int by = blockIdx.y;
  if (by >= p.B) return;

  const int req = p.req_ids[by];
  const int32_t Ni = p.lens[by];
  if (Ni <= 0 || p.M <= 0) return;

  const int start = (Ni > p.M) ? (Ni - p.M) : 0;
  const int keep = Ni - start;

  const int32_t L_before = p.request_length[req];

  const size_t row_bf16 = (size_t)p.H * (size_t)p.D * sizeof(__nv_bfloat16);
  const size_t row_f32 = (size_t)p.H * sizeof(float);

  uint8_t* __restrict__ q_cache_r = (uint8_t*)p.q_cache + ((size_t)req * p.M) * row_bf16;
  uint8_t* __restrict__ attn_cache_r = (uint8_t*)p.attn_cache + ((size_t)req * p.M) * row_bf16;
  uint8_t* __restrict__ lse_cache_r = (uint8_t*)p.lse_cache + ((size_t)req * p.M) * row_f32;

  const uint8_t* __restrict__ q_src = (const uint8_t*)p.q_src;
  const uint8_t* __restrict__ a_src = (const uint8_t*)p.attn_src;
  const uint8_t* __restrict__ l_src = (const uint8_t*)p.lse_src;

  const int32_t base_off = p.offsets[by];
  const int base_dest = (int)((L_before + start) % p.M);

  for (int bx = blockIdx.x; bx < keep; bx += gridDim.x) {
    int dest_slot = base_dest + bx;
    if (dest_slot >= p.M) dest_slot -= p.M;

    const size_t dst_q = (size_t)dest_slot * row_bf16;
    const size_t dst_a = (size_t)dest_slot * row_bf16;
    const size_t dst_l = (size_t)dest_slot * row_f32;

    const size_t src_row = (size_t)(base_off + start + bx);
    const size_t src_q_off = src_row * row_bf16;
    const size_t src_a_off = src_row * row_bf16;
    const size_t src_l_off = src_row * row_f32;

    copy_bytes_16(q_cache_r + dst_q, q_src + src_q_off, row_bf16);
    copy_bytes_16(attn_cache_r + dst_a, a_src + src_a_off, row_bf16);
    copy_bytes_16(lse_cache_r + dst_l, l_src + src_l_off, row_f32);
  }

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    atomicAdd(&p.request_length[req], Ni);
  }
}

// ------------------------------- launcher ---------------------------------

void mac_prefill_update_cache(Tensor q_cache,
                              Tensor attn_cache,
                              Tensor lse_cache,
                              Tensor request_length,
                              Tensor req_ids,
                              Tensor src_q,
                              Tensor src_attn,
                              Tensor src_lse,
                              Tensor offsets,
                              Tensor lens,
                              int capacity,
                              int num_heads,
                              int head_dim) {
  if (capacity <= 0 || req_ids.size(0) == 0) return;

  UpdateParams p;
  fill_params(p,
              q_cache,
              attn_cache,
              lse_cache,
              request_length,
              req_ids,
              src_q,
              src_attn,
              src_lse,
              offsets,
              lens,
              capacity,
              num_heads,
              head_dim);

  c10::cuda::CUDAGuard dev_guard(q_cache.device());
  c10::cuda::CUDAStreamGuard stream_guard(at::cuda::getCurrentCUDAStream(q_cache.get_device()));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(q_cache.get_device()).stream();

  const size_t row_bf16 = (size_t)num_heads * (size_t)head_dim * sizeof(__nv_bfloat16);
  const size_t row_f32 = (size_t)num_heads * sizeof(float);
  const int row16_q = (int)((row_bf16 + 15) / 16);
  const int row16_l = (int)((row_f32 + 15) / 16);
  const int row16 = (row16_q > row16_l) ? row16_q : row16_l;
  const int threads = next_pow2_clamped(row16);

  const dim3 grid(1, (unsigned)p.B);

  mac_prefill_update_cache_kernel<<<grid, threads, 0, stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// --------------------------------- pybind ---------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mac_prefill_update_cache",
        &mac_prefill_update_cache,
        "Cooperative MAC ring cache update (row-parallel, in-kernel length bump)");
}

