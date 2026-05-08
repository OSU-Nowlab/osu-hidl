/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_ATTENTION_SCHEDULER_CUH_
#define FLASHINFER_ATTENTION_SCHEDULER_CUH_

#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <inttypes.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <vector>
#include <iostream>
// #define USE_LIBDIVIDE
#ifdef USE_LIBDIVIDE
  #include "libdivide.h"   // https://libdivide.com/  (single-header)
#endif
#include "../allocator.h"
#include "../exception.h"
#include "../pos_enc.cuh"
#include "../utils.cuh"
#include "heap.h"
#include <inttypes.h>
#include <cub/cub.cuh>
// ============================================================================
// Profiling helpers (NVTX + synchronized CPU timers)
// Toggle with -DENABLE_NVTX=1 and -DENABLE_TIMING=1
// ============================================================================
#ifndef ENABLE_NVTX
#define ENABLE_NVTX 1
#endif

#ifndef ENABLE_TIMING
#define ENABLE_TIMING 0
#endif

#if ENABLE_NVTX
  // Prefer the nvtx3 header path to avoid duplicate-type definitions when both
  // the top-level nvToolsExt.h and the nested nvtx3 headers are present in
  // the system/cmake include paths. Use the nvtx3 variant which contains the
  // modern NVTX3 API and avoids redefinition collisions on some systems.
  #include <nvtx3/nvToolsExt.h>
  struct NvtxScope {
    explicit NvtxScope(const char* name) { nvtxRangePushA(name); }
    ~NvtxScope() { nvtxRangePop(); }
  };
#else
  struct NvtxScope { explicit NvtxScope(const char*) {} };
#endif

#if ENABLE_TIMING
  #include <chrono>
  #include <cstdio>
  struct ScopedTimer {
    const char* name_;
    cudaStream_t stream_;
    std::chrono::high_resolution_clock::time_point t0_;
    explicit ScopedTimer(const char* name, cudaStream_t s = nullptr)
        : name_(name), stream_(s), t0_(std::chrono::high_resolution_clock::now()) {}
    ~ScopedTimer() {
      if (stream_) {
        // Stream sync to ensure all async ops in this scope are completed
        cudaStreamSynchronize(stream_);
      }
      auto t1 = std::chrono::high_resolution_clock::now();
      double ms = std::chrono::duration<double, std::milli>(t1 - t0_).count();
      std::fprintf(stderr, "[TIMING] %s: %.3f ms\n", name_, ms);
    }
  };
#else
  struct ScopedTimer { explicit ScopedTimer(const char*, cudaStream_t = nullptr) {} };
#endif

// Helpers to mark a scope with both NVTX and CPU timer
#define PROFILE_SCOPE(TAG)                 \
  NvtxScope   _nvtx_scope_##__LINE__{TAG}; \
  ScopedTimer _timer_scope_##__LINE__{TAG}

#define PROFILE_SCOPE_STREAM(TAG, STREAM)  \
  NvtxScope   _nvtx_scope_##__LINE__{TAG}; \
  ScopedTimer _timer_scope_##__LINE__{TAG, STREAM}

#ifndef LIKELY_UNLIKELY_H_
#define LIKELY_UNLIKELY_H_

#if defined(__GNUC__) || defined(__clang__)
  #define LIKELY(x)   __builtin_expect(!!(x), 1)
  #define UNLIKELY(x) __builtin_expect(!!(x), 0)
  #define RESTRICT __restrict__
#else
  #define LIKELY(x)   (x)
  #define UNLIKELY(x) (x)
  #define RESTRICT
#endif

#endif  // LIKELY_UNLIKELY_H_


// ---- Fast divider for many divides by the same positive uint32_t C ----
struct FastDivU32 {
  uint32_t d;
#ifdef USE_LIBDIVIDE
  libdivide::divider_u32 dd;
  explicit FastDivU32(uint32_t denom) : d(denom), dd(denom) {}
  inline uint32_t floor_div(uint32_t x) const { return dd.divide(x); }
  inline uint32_t ceil_div(uint32_t x)  const {
    // x>0 path (we already guard P_i==0 elsewhere), branchless: 1 + floor((x-1)/d)
    return (x == 0u) ? 0u : 1u + dd.divide(x - 1u);
  }
#else
  // Multiply-high method (exact) with one correction; requires __int128 on GCC/Clang/MSVC (x64).
  // m = floor(2^64 / d)
  unsigned long long m;
  explicit FastDivU32(uint32_t denom) : d(denom) {
    const unsigned __int128 one = (unsigned __int128)1;
    m = (unsigned long long)((((one << 64) + denom - 1) / denom)); // ceil(2^64 / d)
  }
  inline uint32_t floor_div(uint32_t x) const {
    // q = floor( (x * m) / 2^64 ), then correct at most once
    unsigned __int128 prod = (unsigned __int128)x * (unsigned __int128)m;
    uint64_t q = (uint64_t)(prod >> 64);
    uint32_t r = x - (uint32_t)(q * d);
    if (UNLIKELY(r >= d)) { ++q; /* r -= d; */ }
    return (uint32_t)q;
  }
  inline uint32_t ceil_div(uint32_t x) const {
    if (x == 0u) return 0u;
    // 1 + floor((x-1)/d) — safe (no overflow), exact
    return 1u + floor_div(x - 1u);
  }
#endif
};

// Simple helper to saturate to uint32_t
static inline uint32_t sat_u32(uint64_t x) {
  return (x > 0xffffffffULL) ? 0xffffffffu : (uint32_t)x;
}


namespace flashinfer {

template <PosEncodingMode POS_ENCODING_MODE, uint32_t num_stages_smem, uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz, typename AttentionVariant,
          typename Params>
__global__ void BatchDecodeWithPagedKVCacheKernel(const __grid_constant__ Params params);

template <PosEncodingMode POS_ENCODING_MODE, uint32_t num_stages_smem, uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz, typename AttentionVariant,
          typename Params>
__global__ void BatchMACAttentionDecodeWithPagedKVCacheKernel(const __grid_constant__ Params params);

template <PosEncodingMode POS_ENCODING_MODE, uint32_t num_stages_smem, uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz, typename AttentionVariant,
          typename Params>
__global__ void MACDecodeWithPagedKVCacheKernel(const __grid_constant__ Params params);

template <PosEncodingMode POS_ENCODING_MODE, uint32_t num_stages_smem, uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz, typename AttentionVariant,
          typename Params>
__global__ void BatchRectificationCacheDecodeWithPagedKVCacheKernel(
    const __grid_constant__ Params params);

template <uint32_t num_stages_smem, uint32_t vec_size_ckv, uint32_t vec_size_kpe, uint32_t bdx,
          uint32_t bdy, uint32_t bdz, uint32_t tile_size_qo_heads, typename AttentionVariant,
          typename Params>
__global__ void BatchDecodeWithPagedKVCacheKernelMLA(Params params);

template <uint32_t HEAD_DIM_CKV, uint32_t HEAD_DIM_KPE, uint32_t QO_TILE_LEN, typename DTypeKV>
std::tuple<uint32_t, uint32_t, uint32_t> LaunchSpecForDecodeKernelMlaCuteSM80(
    const uint32_t num_qo_heads);

template <uint32_t HEAD_DIM_CKV, uint32_t HEAD_DIM_KPE, uint32_t QO_TILE_LEN, typename Params>
__global__ void BatchDecodeWithPagedKVCacheKernelMlaCuteSM80(Params params);

template <typename DType>
inline void CopyToPageLockedBuffer(void* page_locked_int_buffer, int64_t offset,
                                   const std::vector<DType>& vec) {
  DType* ptr = GetPtrFromBaseOffset<DType>(page_locked_int_buffer, offset);
  std::copy(vec.begin(), vec.end(), ptr);
}

// -----------------------------------------------------------------------------
// BuildPAndStartPageFlat
//   - P[i]                 : pages per request i, using kv_indptr_h
//   - start_page_flat[i*H + h] : floor(min(attn_start_pos[i, h*GROUP_SIZE + g]) / page_size)
//     (H = num_kv_heads = num_qo_heads / GROUP_SIZE)
// -----------------------------------------------------------------------------
template <typename IdType>
inline cudaError_t BuildPAndStartPageFlat(
    /*out*/ std::vector<IdType>& P,
    /*out*/ std::vector<IdType>& start_page_flat,
    /*out*/ IdType&              max_pages,
    const IdType* kv_indptr_h,
    uint32_t batch_size,
    uint32_t num_qo_heads,
    uint32_t group_size,
    uint32_t page_size,
    const uint32_t* attn_start_pos_dev,
    cudaStream_t stream,
    /*out*/ uint32_t* attn_start_pos_host_pinned)   // NEW
{
  PROFILE_SCOPE_STREAM("BuildPAndStartPageFlat(pinned)", stream);

  const uint32_t num_kv_heads = num_qo_heads / group_size;

  {
    NvtxScope _n{"BuildPAndStartPageFlat::D2H_attn_start_pos_pinned"};
    ScopedTimer _t{"BuildPAndStartPageFlat::D2H_attn_start_pos_pinned", stream};
    const size_t n = static_cast<size_t>(batch_size) * num_qo_heads;
    FLASHINFER_CUDA_CALL(cudaMemcpyAsync(
        attn_start_pos_host_pinned, attn_start_pos_dev, n * sizeof(uint32_t),
        cudaMemcpyDeviceToHost, stream));
    FLASHINFER_CUDA_CALL(cudaStreamSynchronize(stream));
  }

  {
    // NOTE: Compute P/max_pages after the stream sync above so that a preceding
    // async D2H copy of kv_indptr_h (from Python) is guaranteed to be complete.
    NvtxScope _n{"BuildPAndStartPageFlat::compute_P_and_max"};
    ScopedTimer _t{"BuildPAndStartPageFlat::compute_P_and_max", stream};
    P.resize(batch_size);
    max_pages = 0;
    for (uint32_t i = 0; i < batch_size; ++i) {
      const IdType pages = kv_indptr_h[i + 1] - kv_indptr_h[i];
      P[i] = pages;
      if (pages > max_pages) max_pages = pages;
    }
  }

  {
    NvtxScope _n{"BuildPAndStartPageFlat::build_start_page_flat"};
    ScopedTimer _t{"BuildPAndStartPageFlat::build_start_page_flat", stream};
    start_page_flat.assign(static_cast<size_t>(batch_size) * num_kv_heads, 0);
    for (uint32_t i = 0; i < batch_size; ++i) {
      for (uint32_t h = 0; h < num_kv_heads; ++h) {
        uint32_t min_start = 0xffffffffu;
        const uint32_t base = i * num_qo_heads + h * group_size;
        #pragma unroll
        for (uint32_t g = 0; g < group_size; ++g) {
          const uint32_t v = attn_start_pos_host_pinned[base + g];
          if (v < min_start) min_start = v;
        }
        start_page_flat[i * num_kv_heads + h] =
            static_cast<IdType>(min_start / page_size);
      }
    }
  }
  return cudaSuccess;
}


// -----------------------------------------------------------------------------
// PartitionPagedKVCacheHeadPacked
//   Computes (C, new_batch_size) where:
//     J_i(C) = ceil_div(P[i], C)
//     tile0(i,h,C) = ceil_div(start_page_flat[i*H + h], C)
//     T(i,h,C) = max(0, J_i - tile0)
//     new_batch_size(C) = sum_{i,h} T(i,h,C)
// -----------------------------------------------------------------------------
template <typename IdType>
inline auto PartitionPagedKVCacheHeadPacked(
    const uint32_t max_grid_size,
    const std::vector<IdType>& P,                 // len = B, in PAGES
    const std::vector<IdType>& start_page_flat,   // len = B * H, in PAGES (per-head)
    uint32_t batch_size,
    uint32_t num_kv_heads,
    uint32_t min_pages_per_tile,                  // >= max(128/page_size, 1)
    uint32_t downdate_pages_kept                  // pages covered by the downdate window
) -> std::tuple<uint32_t /*C*/, uint32_t /*X = new_batch_size*/>
{
  if (batch_size == 0 || num_kv_heads == 0) return {std::max(1u, min_pages_per_tile), 0u};

  // 1) Max pages across requests (upper bound for C)
  uint32_t max_pages = 0;
  for (uint32_t i = 0; i < batch_size; ++i) {
    const uint32_t Pi = (uint32_t)P[i];
    if (Pi > max_pages) max_pages = Pi;
  }

  // 2) Precompute per-request downdate start (in pages)
  std::vector<uint32_t> start_dd_pages(batch_size);
  for (uint32_t i = 0; i < batch_size; ++i) {
    const uint32_t Pi = (uint32_t)P[i];
    start_dd_pages[i] = (Pi > downdate_pages_kept) ? (Pi - downdate_pages_kept) : 0u;
  }

  // 3) Fast evaluator for nb(C) with early exit when nb > max_grid_size
  auto total_ctas = [&](uint32_t C, bool allow_early_exit)->uint64_t {
    FastDivU32 divC(C);
    uint64_t nb_total = 0;

    // Pointers for tight, contiguous access
    const IdType* RESTRICT spf = start_page_flat.data();
    const IdType* RESTRICT Pp  = P.data();

    // Optionally parallelize: each i contributes independently to nb_total (pure reduction).
    // Note: early-exit is disabled under OpenMP for correctness.
    #if defined(_OPENMP)
    (void)allow_early_exit;
    #pragma omp parallel for reduction(+:nb_total) schedule(static)
    #endif
    for (int i = 0; i < (int)batch_size; ++i) {
      const uint32_t Pi   = (uint32_t)Pp[i];
      if (Pi == 0u) continue;

      const uint32_t Ji   = divC.ceil_div(Pi);                  // tiles for request i
      if (Ji == 0u) continue;

      const uint32_t tile0_dd = divC.floor_div(start_dd_pages[i]);
      const IdType*   sp_i    = spf + (size_t)i * (size_t)num_kv_heads;

      uint64_t nb_i = 0;

      // Inner loop over heads — division via FastDivU32, simple pointer-walk
      for (uint32_t h = 0; h < num_kv_heads; ++h) {
        const uint32_t sp_main    = (uint32_t)sp_i[h];
        const uint32_t tile0_main = divC.floor_div(sp_main);
        const uint32_t tile0_sched = (tile0_main < tile0_dd) ? tile0_main : tile0_dd;

        // delta = max(0, Ji - tile0_sched)
        const int32_t  delta = (int32_t)Ji - (int32_t)tile0_sched;
        nb_i += (delta > 0) ? (uint32_t)delta : 0u;
      }

      nb_total += nb_i;

      // Single-thread early exit only (OpenMP path omits this)
      #if !defined(_OPENMP)
      if (allow_early_exit && UNLIKELY(nb_total > (uint64_t)max_grid_size)) {
        return nb_total; // stop early when we've already exceeded the cap
      }
      #endif
    }
    return nb_total;
  };

  // 4) Bounds for binary search
  uint32_t low  = std::max<uint32_t>(1u, min_pages_per_tile);
  uint32_t high = std::max<uint32_t>(low, max_pages);

  // Quick check: if minimal tile already fits, return immediately
  if (total_ctas(low, /*early_exit=*/true) <= (uint64_t)max_grid_size) {
    const uint64_t nb = total_ctas(low, /*early_exit=*/false);
    return {low, sat_u32(nb)};
  }

  // 5) Binary search with early-exiting evaluator
  while (low < high) {
    const uint32_t mid = low + ((high - low) >> 1);
    const uint64_t nb  = total_ctas(mid, /*early_exit=*/true);
    if (nb > (uint64_t)max_grid_size) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }

  // 6) Final exact nb at chosen C (no early exit)
  const uint32_t C = low;
  const uint32_t X = sat_u32(total_ctas(C, /*early_exit=*/false));
  return {C, X};
}


template <typename IdType>
inline auto BuildHeadPackedSchedule(
    const std::vector<IdType>& P,                 // len = B
    const std::vector<IdType>& start_page_flat,   // len = B * H
    uint32_t batch_size,
    uint32_t num_kv_heads,
    uint32_t C  // pages per tile
) -> std::tuple<
      std::vector<IdType>, // request_indices
      std::vector<IdType>, // kv_head_indices
      std::vector<IdType>, // kv_tile_indices (absolute j)
      std::vector<IdType>, // o_indptr (len B+1), with J_i per request
      uint32_t             // nnz_total = Σ_i J_i
    >
{
  PROFILE_SCOPE("BuildHeadPackedSchedule");

  std::vector<IdType> request_indices;
  std::vector<IdType> kv_head_indices;
  std::vector<IdType> kv_tile_indices;
  std::vector<IdType> o_indptr(batch_size + 1, 0);

  if (batch_size == 0 || num_kv_heads == 0 || C == 0u) {
    return {request_indices, kv_head_indices, kv_tile_indices, o_indptr, 0u};
  }

  const IdType* RESTRICT Pp   = P.data();
  const IdType* RESTRICT spf0 = start_page_flat.data();
  const size_t  H             = (size_t)num_kv_heads;
  const size_t  BH            = (size_t)batch_size * H;

  // --- Pass 0: compute J_i and o_indptr, and sum nnz_total = Σ_i J_i ---
  uint32_t nnz_total = 0;
  std::vector<uint32_t> J(batch_size, 0);

  {
    NvtxScope   _n{"BuildHeadPackedSchedule::precompute_J_and_prefix"};
    ScopedTimer _t{"BuildHeadPackedSchedule::precompute_J_and_prefix"};

    FastDivU32 divC(C);
    uint64_t nnz_total_u64 = 0;

    for (uint32_t i = 0; i < batch_size; ++i) {
      const uint32_t Pi = (uint32_t)Pp[i];
      const uint32_t Ji = (Pi == 0u) ? 0u : divC.ceil_div(Pi);
      J[i] = Ji;
      nnz_total_u64 += Ji;
      o_indptr[i + 1] = static_cast<IdType>(nnz_total_u64);
    }
    nnz_total = (uint32_t)std::min<uint64_t>(nnz_total_u64, 0xffffffffULL);
  }

  // --- Pass 1: counts per (i,h): T(i,h) = max(0, J_i - floor(start_page_flat/C)) ---
  std::vector<uint32_t> counts(BH, 0);

  {
    NvtxScope   _n{"BuildHeadPackedSchedule::row_counts"};
    ScopedTimer _t{"BuildHeadPackedSchedule::row_counts"};

    FastDivU32 divC(C);

    // Parallelizable; each (i,h) independent
    #if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < (int)batch_size; ++i) {
      const uint32_t Ji = J[(uint32_t)i];
      if (Ji == 0u) {
        // zero this row's heads
        for (uint32_t h = 0; h < num_kv_heads; ++h)
          counts[(size_t)i * H + h] = 0u;
        continue;
      }

      const IdType* RESTRICT sp_i = spf0 + (size_t)i * H;
      for (uint32_t h = 0; h < num_kv_heads; ++h) {
        const uint32_t sp    = (uint32_t)sp_i[h];
        const uint32_t tile0 = divC.floor_div(sp);
        const int32_t  delta = (int32_t)Ji - (int32_t)tile0;
        counts[(size_t)i * H + h] = (delta > 0) ? (uint32_t)delta : 0u;
      }
    }
  }

  // --- Pass 2: exclusive scan of counts -> per-(i,h) output offsets ---
  std::vector<uint64_t> offsets(BH + 1, 0);
  {
    NvtxScope   _n{"BuildHeadPackedSchedule::scan"};
    ScopedTimer _t{"BuildHeadPackedSchedule::scan"};
    for (size_t idx = 0; idx < BH; ++idx) {
      offsets[idx + 1] = offsets[idx] + counts[idx];
    }
  }
  const size_t M_total = (size_t)offsets[BH];

  // Allocate exact sizes, no push_back
  request_indices.resize(M_total);
  kv_head_indices.resize(M_total);
  kv_tile_indices.resize(M_total);

  // --- Pass 3: emit triplets into precomputed slices ---
  {
    NvtxScope   _n{"BuildHeadPackedSchedule::emit_triplets"};
    ScopedTimer _t{"BuildHeadPackedSchedule::emit_triplets"};

    FastDivU32 divC(C);

    IdType* RESTRICT req_out = request_indices.data();
    IdType* RESTRICT hed_out = kv_head_indices.data();
    IdType* RESTRICT til_out = kv_tile_indices.data();

    // Parallel fill: each (i,h) writes to its own disjoint span
    #if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < (int)batch_size; ++i) {
      const uint32_t Ji      = J[(uint32_t)i];
      const IdType*  sp_i    = spf0 + (size_t)i * H;

      for (uint32_t h = 0; h < num_kv_heads; ++h) {
        const size_t   idx     = (size_t)i * H + h;
        const uint32_t T       = counts[idx];
        if (T == 0u) continue;

        const uint32_t tile0   = divC.floor_div((uint32_t)sp_i[h]);
        const uint32_t j_end   = tile0 + T;

        size_t out = (size_t)offsets[idx];

        // Tight contiguous writes; encourage vectorization
        #if defined(__GNUC__) || defined(__clang__)
        #pragma GCC ivdep
        #endif
        for (uint32_t j = tile0; j < j_end; ++j, ++out) {
          req_out[out] = (IdType)i;
          hed_out[out] = (IdType)h;
          til_out[out] = (IdType)j;
        }
      }
    }
  }

  return {std::move(request_indices),
          std::move(kv_head_indices),
          std::move(kv_tile_indices),
          std::move(o_indptr),
          nnz_total};
}



// Fallback schedule (non load-balanced, gdy=1, non-split): X = batch_size
template <typename IdType>
inline void BuildUniformFlattenedSchedule(
    uint32_t batch_size,
    /*out*/ std::vector<IdType>& request_indices,
    /*out*/ std::vector<IdType>& kv_tile_indices,
    /*out*/ std::vector<IdType>& o_indptr) {
  request_indices.resize(batch_size);
  kv_tile_indices.resize(batch_size);
  for (uint32_t i = 0; i < batch_size; ++i) {
    request_indices[i] = static_cast<IdType>(i);
    kv_tile_indices[i] = static_cast<IdType>(0);   // one full tile per request
  }
  // o_indptr is unused when split_kv=false, but keep a valid buffer around
  o_indptr.resize(batch_size + 1);
  for (uint32_t i = 0; i <= batch_size; ++i) o_indptr[i] = static_cast<IdType>(i);
}



/*!
 * \brief Compute the maximum number of pages per batch and the new batch size
 *   after we partition Paged KV-Cache into multiple chunks on KV sequence length
 *   dimension.
 * \tparam IdType A template type indicates the index data type
 * \param max_grid_size The maximum grid size of the kernel
 * \param gdy gridDim.y
 * \param num_pages The number of pages per request in the batch
 * \param max_num_pages_per_batch_lb The pre-set lower bound of maximum number of
 *   pages per batch, default to 1
 * \return (max_num_pages_per_batch, new_batch_size) The number of pages per batch and
 *   the new batch size after the partition.
 */
template <typename IdType>
inline auto PartitionPagedKVCacheBinarySearchMinNumPagePerBatch(
    const uint32_t max_grid_size, const uint32_t gdy, const std::vector<IdType>& num_pages,
    const uint32_t min_num_pages_per_batch = 1) {
  uint32_t low = min_num_pages_per_batch, high = 0;
  for (const IdType& elem : num_pages) {
    high = max(high, elem);
  }
  uint32_t new_batch_size;
  while (low < high) {
    uint32_t mid = (low + high) / 2;
    new_batch_size = 0;
    for (const IdType& elem : num_pages) {
      new_batch_size += ceil_div(elem, mid);
    }
    if (new_batch_size * gdy > max_grid_size) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  new_batch_size = 0;
  for (const IdType& elem : num_pages) {
    new_batch_size += ceil_div(std::max(elem, 1), low);
  }
  return std::make_tuple(low, new_batch_size);
}

inline auto PrefillBinarySearchKVChunkSize(const bool enable_cuda_graph,
                                           const uint32_t max_batch_size_if_split,
                                           const std::vector<int64_t>& packed_qo_len_arr,
                                           const std::vector<int64_t>& kv_len_arr,
                                           const uint32_t qo_chunk_size,
                                           const uint32_t min_kv_chunk_size = 1) {
  const int64_t batch_size = packed_qo_len_arr.size();
  int64_t max_kv_len = 1;
  for (const int64_t& kv_len : kv_len_arr) {
    max_kv_len = std::max(max_kv_len, kv_len);
  }

  int64_t low = min_kv_chunk_size;
  int64_t high = max_kv_len;
  constexpr int64_t min_kv_len = 1;
  while (low < high) {
    const int64_t mid = (low + high) / 2;
    int64_t new_batch_size = 0;
    for (uint32_t i = 0; i < batch_size; ++i) {
      new_batch_size += ceil_div(packed_qo_len_arr[i], qo_chunk_size) *
                        ceil_div(std::max(kv_len_arr[i], min_kv_len), mid);
    }
    if (new_batch_size > max_batch_size_if_split) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return std::make_tuple(enable_cuda_graph || low < max_kv_len, low);
}

/*!
 * \brief Estimate the temporary buffer size and the maximum grid size for the
 *   partition-kv BatchDecodeWithPagedKVCache kernel
 * \tparam DTypeKV A template type indicates the key-value data type
 * \tparam DTypeO A template type indicates the output data type
 * \tparam IdType A template type indicates the index data type
 * \param split_kv Whether to split the KV cache into multiple chunks
 * \param max_grid_size The maximum grid size that can be used in a partiton-kv kernel
 * \param max_num_pages_per_batch The maximum number of pages per batch
 * \param new_batch_size The new batch size after the partition
 * \param paged_kv The paged kv cache data structure
 * \param num_qo_heads A integer indicates the number of heads of query and output
 * \param pos_encoding_mode The positional encoding mode
 * \param stream The cuda stream to launch the kernel
 * \return status Indicates whether CUDA calls are successful
 */
template <uint32_t GROUP_SIZE, uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE,
          typename AttentionVariant, typename Params>
inline cudaError_t BatchDecodeWithPagedKVCacheWorkEstimationDispatched(
    bool& split_kv, uint32_t& max_grid_size, uint32_t& max_num_pages_per_batch,
    uint32_t& new_batch_size, uint32_t& gdy, uint32_t batch_size,
    typename Params::IdType* kv_indptr_h, const uint32_t num_qo_heads, const uint32_t page_size,
    bool enable_cuda_graph, cudaStream_t stream) {
  using DTypeKV = typename Params::DTypeKV;
  using IdType = typename Params::IdType;
  constexpr uint32_t vec_size = std::max(16UL / sizeof(DTypeKV), HEAD_DIM / 32UL);
  auto compute_capacity = GetCudaComputeCapability();
  DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, {
    constexpr uint32_t bdx = HEAD_DIM / vec_size;
    static_assert(bdx <= 32);
    constexpr uint32_t bdy = GROUP_SIZE;
    constexpr uint32_t num_threads = std::max(128U, bdx * bdy);
    constexpr uint32_t bdz = num_threads / (bdx * bdy);
    constexpr uint32_t tile_size_per_bdx = GROUP_SIZE == 1 ? (sizeof(DTypeKV) == 1 ? 2U : 4U) : 1U;
    const uint32_t num_kv_heads = num_qo_heads / GROUP_SIZE;
    gdy = num_kv_heads;
    const uint32_t smem_size =
        2 * NUM_STAGES_SMEM * tile_size_per_bdx * bdy * bdz * HEAD_DIM * sizeof(DTypeKV) +
        std::max(tile_size_per_bdx * num_threads * sizeof(DTypeKV*), 2 * bdy * bdz * sizeof(float));

    auto kernel =
        BatchDecodeWithPagedKVCacheKernel<POS_ENCODING_MODE, NUM_STAGES_SMEM, tile_size_per_bdx,
                                          vec_size, bdx, bdy, bdz, AttentionVariant, Params>;
    int num_blocks_per_sm = 7;
    int num_sm = 112;
    int dev_id = 0;
    FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
    // FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
    // FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel,
                                                                      //  num_threads, smem_size));
    max_grid_size = num_blocks_per_sm * num_sm;
    if (batch_size * gdy >= max_grid_size) {
      split_kv = false;
      max_num_pages_per_batch = 1;
      for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        max_num_pages_per_batch = std::max<uint32_t>(
            max_num_pages_per_batch, kv_indptr_h[batch_idx + 1] - kv_indptr_h[batch_idx]);
      }
      new_batch_size = batch_size;
    } else {
      // compute max_num_pages_per_batch and new_batch_size
      std::vector<IdType> num_pages(batch_size);
      for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        num_pages[batch_idx] = kv_indptr_h[batch_idx + 1] - kv_indptr_h[batch_idx];
      }
      std::tie(max_num_pages_per_batch, new_batch_size) =
          PartitionPagedKVCacheBinarySearchMinNumPagePerBatch(max_grid_size, gdy, num_pages,
                                                              std::max(128 / page_size, 1U));
      if (new_batch_size == batch_size && !enable_cuda_graph) {
        // do not use partition-kv kernel for short sequence, when not using CUDAGraph
        split_kv = false;
      } else {
        // when using CUDAGraph, we always use partition-kv kernel
        split_kv = true;
      }
    }
    return cudaSuccess;
  })
}

template <uint32_t GROUP_SIZE, uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE,
          typename AttentionVariant, typename Params>
inline cudaError_t BatchMACAttentionDecodeWithPagedKVCacheWorkEstimationDispatched(
    bool& split_kv, uint32_t& max_grid_size, uint32_t& max_num_pages_per_batch,
    uint32_t& new_batch_size, uint32_t& gdy, uint32_t batch_size,
    typename Params::IdType* kv_indptr_h, const uint32_t num_qo_heads, const uint32_t page_size,
    bool enable_cuda_graph, cudaStream_t stream) {
  using DTypeKV = typename Params::DTypeKV;
  using IdType = typename Params::IdType;
  constexpr uint32_t vec_size = std::max(16UL / sizeof(DTypeKV), HEAD_DIM / 32UL);
  auto compute_capacity = GetCudaComputeCapability();
  DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, {
    constexpr uint32_t bdx = HEAD_DIM / vec_size;
    static_assert(bdx <= 32);
    constexpr uint32_t bdy = GROUP_SIZE;
    constexpr uint32_t num_threads = std::max(128U, bdx * bdy);
    constexpr uint32_t bdz = num_threads / (bdx * bdy);
    constexpr uint32_t tile_size_per_bdx = GROUP_SIZE == 1 ? (sizeof(DTypeKV) == 1 ? 2U : 4U) : 1U;
    const uint32_t num_kv_heads = num_qo_heads / GROUP_SIZE;
    gdy = num_kv_heads;
    const uint32_t smem_size =
        2 * NUM_STAGES_SMEM * tile_size_per_bdx * bdy * bdz * HEAD_DIM * sizeof(DTypeKV) +
        std::max(tile_size_per_bdx * num_threads * sizeof(DTypeKV*), 2 * bdy * bdz * sizeof(float));

    auto kernel = BatchMACAttentionDecodeWithPagedKVCacheKernel<
        POS_ENCODING_MODE, NUM_STAGES_SMEM, tile_size_per_bdx, vec_size, bdx, bdy, bdz,
        AttentionVariant, Params>;
    int num_blocks_per_sm = 0;
    int num_sm = 0;
    int dev_id = 0;
    FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
    FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
    FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel,
                                                                       num_threads, smem_size));
    max_grid_size = num_blocks_per_sm * num_sm;
    if (batch_size * gdy >= max_grid_size) {
      split_kv = false;
      max_num_pages_per_batch = 1;
      for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        max_num_pages_per_batch = std::max<uint32_t>(
            max_num_pages_per_batch, kv_indptr_h[batch_idx + 1] - kv_indptr_h[batch_idx]);
      }
      new_batch_size = batch_size;
    } else {
      // compute max_num_pages_per_batch and new_batch_size
      std::vector<IdType> num_pages(batch_size);
      for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        num_pages[batch_idx] = kv_indptr_h[batch_idx + 1] - kv_indptr_h[batch_idx];
      }
      std::tie(max_num_pages_per_batch, new_batch_size) =
          PartitionPagedKVCacheBinarySearchMinNumPagePerBatch(max_grid_size, gdy, num_pages,
                                                              std::max(128 / page_size, 1U));
      if (new_batch_size == batch_size && !enable_cuda_graph) {
        // do not use partition-kv kernel for short sequence, when not using CUDAGraph
        split_kv = false;
      } else {
        // when using CUDAGraph, we always use partition-kv kernel
        split_kv = true;
      }
    }
    return cudaSuccess;
  })
}

template <uint32_t GROUP_SIZE, uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE,
          typename AttentionVariant, typename Params>
inline cudaError_t MACDecodeWithPagedKVCacheWorkEstimationDispatched(
    bool& split_kv, uint32_t& max_grid_size, uint32_t& kv_chunk_size_in_pages,
    uint32_t& new_batch_size, uint32_t& gdy, uint32_t batch_size,
    typename Params::IdType* kv_indptr_h, const uint32_t num_qo_heads, const uint32_t page_size,
    bool enable_cuda_graph, const std::vector<typename Params::IdType>& P, const std::vector<typename Params::IdType>& start_page_flat) {

  PROFILE_SCOPE("WorkEstimationDispatched");

  using DTypeKV = typename Params::DTypeKV;
  using IdType  = typename Params::IdType;

  const uint32_t num_kv_heads = num_qo_heads / GROUP_SIZE;

  // === baseline page counts per request ===
  std::vector<IdType> num_pages(batch_size);
  uint32_t max_pages = 0;
  {
    NvtxScope   _n{"WorkEstimation::compute_baseline_pages"};
    ScopedTimer _t{"WorkEstimation::compute_baseline_pages"};
    for (uint32_t i = 0; i < batch_size; ++i) {
      const uint32_t pages = static_cast<uint32_t>(kv_indptr_h[i + 1] - kv_indptr_h[i]);
      num_pages[i] = pages;
      max_pages = std::max(max_pages, pages);
    }
  }

  // === occupancy-based capacity (unchanged) ===
  {
    NvtxScope   _n{"WorkEstimation::occupancy_capacity_query"};
    ScopedTimer _t{"WorkEstimation::occupancy_capacity_query"};

    constexpr uint32_t vec_size = std::max(16U / static_cast<uint32_t>(sizeof(DTypeKV)),
                                           HEAD_DIM / 32U);
    auto compute_capacity = GetCudaComputeCapability();
    DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, {
      constexpr uint32_t bdx = HEAD_DIM / vec_size;
      static_assert(bdx <= 32, "HEAD_DIM / vec_size must be <= 32");
      constexpr uint32_t bdy = GROUP_SIZE;
      constexpr uint32_t num_threads = std::max(128U, bdx * bdy);
      constexpr uint32_t bdz = num_threads / (bdx * bdy);
      constexpr uint32_t tile_size_per_bdx =
          GROUP_SIZE == 1 ? (sizeof(DTypeKV) == 1 ? 2U : 4U) : 1U;

      const uint32_t smem_size =
          2 * NUM_STAGES_SMEM * tile_size_per_bdx * bdy * bdz * HEAD_DIM * sizeof(DTypeKV) +
          std::max(tile_size_per_bdx * num_threads * (uint32_t)sizeof(DTypeKV*),
                   2U * bdy * bdz * (uint32_t)sizeof(float));

      auto kernel =
          MACDecodeWithPagedKVCacheKernel<POS_ENCODING_MODE, NUM_STAGES_SMEM,
                                                   tile_size_per_bdx, vec_size, bdx, bdy, bdz,
                                                   AttentionVariant, Params>;

      int dev_id = 0, num_sm = 0, num_blocks_per_sm = 0;
      FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
      FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
      FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &num_blocks_per_sm, kernel, num_threads, smem_size));

      max_grid_size = static_cast<uint32_t>(num_blocks_per_sm * num_sm);

      // === Key change: we will LAUNCH with gdy=1, but we must CAP total CTAs on x as X * H ===
      const uint32_t gdy_effective = num_kv_heads;  // used only for the binary-search constraint
      gdy = 1;                                      // actual launch y-dimension

      // If even the non-split case (X = batch_size) saturated capacity, don't split.
      const uint64_t base_total_ctas = static_cast<uint64_t>(batch_size) * gdy_effective;
      if (base_total_ctas >= max_grid_size && !enable_cuda_graph) {
        NvtxScope   _n2{"WorkEstimation::non_split_fallback"};
        ScopedTimer _t2{"WorkEstimation::non_split_fallback"};
        split_kv = false;
        kv_chunk_size_in_pages = 1;
        new_batch_size = batch_size;  // X = batch_size, launch will be X*H on x with gdy=1
        return cudaSuccess;
      }

      // === same partitioner, but with gdy_effective (H) ===
      uint32_t max_num_pages_per_batch, new_batch_x;
      {
        NvtxScope   _n2{"WorkEstimation::partition_binary_search"};
        ScopedTimer _t2{"WorkEstimation::partition_binary_search"};
        std::tie(max_num_pages_per_batch, new_batch_x) =
            PartitionPagedKVCacheBinarySearchMinNumPagePerBatch(
                /*max_grid_size=*/max_grid_size,
                /*gdy=*/gdy_effective,
                /*num_pages=*/num_pages,
                /*min_num_pages_per_batch=*/std::max(128U / page_size, 1U));
      }

      // If it collapses back to non-split and not doing graphs, keep non-split.
      if (new_batch_x == batch_size && !enable_cuda_graph) {
        NvtxScope   _n2{"WorkEstimation::non_split_fallback_collapse"};
        ScopedTimer _t2{"WorkEstimation::non_split_fallback_collapse"};
        split_kv = false;
        kv_chunk_size_in_pages = 1;
        new_batch_size = batch_size;
        return cudaSuccess;
      }

      // Else, split as usual.
      split_kv = true;
      kv_chunk_size_in_pages = max_num_pages_per_batch;
      new_batch_size = new_batch_x;  // X (number of tile slots). Launch uses X*H on x with gdy=1.
      return cudaSuccess;
    })
  }
}


template <uint32_t GROUP_SIZE, uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE,
          typename AttentionVariant, typename Params>
inline cudaError_t MACDecodeWithPagedKVCacheWorkEstimationHeadPackedPrepared(
    /*out*/ bool& split_kv,
    /*out*/ uint32_t& max_grid_size,
    /*out*/ uint32_t& kv_chunk_size_in_pages,   // C (pages)
    /*out*/ uint32_t& new_batch_size,           // X (CTAs)
    /*out*/ uint32_t& gdy,                      // = 1
    uint32_t batch_size,
    uint32_t num_qo_heads,
    uint32_t page_size,
    bool enable_cuda_graph,
    const std::vector<typename Params::IdType>& P,                 // in PAGES
    const std::vector<typename Params::IdType>& start_page_flat,   // in PAGES (group-min MAIN)
    uint32_t downdate_range                                        // NEW: in TOKENS
) {

  using DTypeKV = typename Params::DTypeKV;
  using IdType  = typename Params::IdType;

  const uint32_t num_kv_heads = num_qo_heads / GROUP_SIZE;

  // Convert downdate_range (tokens) to pages kept (pagewise lower-bound, safe).
  const uint32_t downdate_pages_kept =
      (downdate_range == 0) ? 0 : ceil_div<uint32_t>(downdate_range + 1, page_size);

  // Occupancy-based capacity
  // auto compute_capacity = GetCudaComputeCapability();
  int dev_id = 0, num_sm = 112, num_blocks_per_sm = 7;
  max_grid_size = static_cast<uint32_t>(num_blocks_per_sm) * static_cast<uint32_t>(num_sm);

  // Always do split in head-packed path (we're balancing)
  split_kv = true;
  gdy = 1;

  // Head-aware binary search (using union with downdate window)
  const uint32_t min_pages_per_tile = std::max(128U / page_size, 1U);
  uint32_t C, X;
  {
    NvtxScope   _n{"MACAttentionDecodePlan::PartitionPagedKVCacheHeadPacked"};
    ScopedTimer _t{"MACAttentionDecodePlan::PartitionPagedKVCacheHeadPacked"};
    std::tie(C, X) = PartitionPagedKVCacheHeadPacked<IdType>(
        max_grid_size, P, start_page_flat, batch_size, num_kv_heads,
        min_pages_per_tile, downdate_pages_kept);
  }
  kv_chunk_size_in_pages = C;
  new_batch_size = X;  // number of CTAs to launch on x (unpadded)
  return cudaSuccess;

  // DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, {
    // constexpr uint32_t vec_size = std::max(16U / static_cast<uint32_t>(sizeof(DTypeKV)),
    //                                        HEAD_DIM / 32U);
    // constexpr uint32_t bdx = HEAD_DIM / vec_size;
    // static_assert(bdx <= 32, "HEAD_DIM/vec_size must be <= 32");
    // constexpr uint32_t bdy = GROUP_SIZE;
    // constexpr uint32_t num_threads = std::max(128U, bdx * bdy);
    // constexpr uint32_t bdz = num_threads / (bdx * bdy);
    // constexpr uint32_t tile_size_per_bdx =
    //     (GROUP_SIZE == 1) ? ((sizeof(DTypeKV) == 1) ? 2U : 4U) : 1U;
    // const uint32_t head_dim = HEAD_DIM;

    // const uint32_t smem_size =
    //     2 * NUM_STAGES_SMEM * tile_size_per_bdx * bdy * bdz * head_dim * sizeof(DTypeKV) +
    //     std::max(tile_size_per_bdx * num_threads * (uint32_t)sizeof(DTypeKV*),
    //              2U * bdy * bdz * (uint32_t)sizeof(float));

    // auto kernel =
    //     MACDecodeWithPagedKVCacheKernel<
    //       POS_ENCODING_MODE, NUM_STAGES_SMEM, tile_size_per_bdx,
    //       vec_size, bdx, bdy, bdz, AttentionVariant, Params>;

    // int dev_id = 0, num_sm = 112, num_blocks_per_sm = 7;
    // FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
    // FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
    // FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    //     &num_blocks_per_sm, kernel, num_threads, smem_size));

  // })
}



// ============================================================================
// Window-based, head-packed planning (tokens) — no attn_start_pos required
// Prefix: RectificationCache...
// ============================================================================

/*!
 * \brief Build per-request KV lengths (in tokens) from a cumsum "indptr" of token lengths.
 * \note This replaces the "P[i] = pages" variant. Here we return L[i] in TOKENS.
 */
template <typename IdType>
inline cudaError_t WindowBuildKVLensFromIndptrTokens(
    /*out*/ std::vector<IdType>& L_tokens,
    /*out*/ IdType&              max_len_tokens,
    const IdType*                len_indptr_h,   // host ptr [batch_size+1], cumsum of TOKENS
    uint32_t                     batch_size) {
  L_tokens.resize(batch_size);
  max_len_tokens = 0;
  for (uint32_t i = 0; i < batch_size; ++i) {
    const IdType Li = len_indptr_h[i + 1] - len_indptr_h[i];
    L_tokens[i] = Li;
    if (Li > max_len_tokens) max_len_tokens = Li;
  }
  return cudaSuccess;
}

/*!
 * \brief Partition (binary search) for minimal token tile size C_tokens such that
 *        total CTAs (head-packed) fit within max_grid_size, when the active window
 *        per request is W_i = min(window_left, L_i).
 *
 * Grid accounting:
 *   J_i(C)     = ceil_div(L_i, C)
 *   start_tok  = max<int>(0, L_i - window_left)
 *   tile0_i    = start_tok / C          // FLOOR: keep first tile that contains the first unmasked token
 *   T_i(C)     = max<int>(0, J_i - tile0_i)
 *   X(C)       = sum_i  T_i(C) * H_kv   // CTAs across all KV heads (identical across heads here)
 */
template <typename IdType>
inline auto WindowPartitionHeadPackedTokens(
    const uint32_t                    max_grid_size,
    const std::vector<IdType>&        L_tokens,        // len = B (TOKENS)
    uint32_t                          batch_size,
    uint32_t                          num_kv_heads,
    uint32_t                          window_left,     // TOKENS
    uint32_t                          min_tokens_per_tile  // e.g., 128
) -> std::tuple<uint32_t /*C_tokens*/, uint32_t /*X = new_batch_size*/> {

  auto total_ctas = [&](uint32_t C_tokens) -> uint64_t {
    uint64_t nb = 0;
    for (uint32_t i = 0; i < batch_size; ++i) {
      const uint32_t L = static_cast<uint32_t>(L_tokens[i]);
      if (L == 0) continue;
      const uint32_t J_i       = ceil_div<uint32_t>(L, C_tokens);
      const uint32_t W_i       = std::min<uint32_t>(L, window_left + 1u);
      const uint32_t start_tok = L - W_i;                     // >= 0
      const uint32_t tile0     = start_tok / C_tokens;        // FLOOR
      if (J_i > tile0) nb += uint64_t(J_i - tile0) * uint64_t(num_kv_heads);
    }
    return nb;
  };

  const uint32_t low0  = std::max<uint32_t>(1u, min_tokens_per_tile);
  // Upper bound for C_tokens: at most max(L_i), but keep >= low0
  uint32_t max_L = 0;
  for (uint32_t i = 0; i < batch_size; ++i) max_L = std::max<uint32_t>(max_L, static_cast<uint32_t>(L_tokens[i]));
  uint32_t low  = low0;
  uint32_t high = std::max<uint32_t>(low, std::max<uint32_t>(1u, max_L));

  // Binary search minimal C_tokens with X(C) <= max_grid_size
  while (low < high) {
    const uint32_t mid = (low + high) / 2;
    const uint64_t nb  = total_ctas(mid);
    if (nb > max_grid_size) low = mid + 1; else high = mid;
  }
  const uint32_t C_tokens = low;
  const uint32_t X        = static_cast<uint32_t>(std::min<uint64_t>(total_ctas(C_tokens), 0xffffffffULL));
  return {C_tokens, X};
}

// ============================================================================
// Compact tail CSR builder for window_left scheduling (TOKENS)
// Emits only active-tail rows per request: A_i = max(0, J_i - tile0_i)
// ============================================================================

template <typename IdType>
inline auto BuildWindowHeadPackedScheduleTokensCompact(
    const std::vector<IdType>&  L_tokens,     // len = B, KV length in TOKENS per request
    uint32_t                    batch_size,
    uint32_t                    num_kv_heads,
    uint32_t                    window_left,  // TOKENS
    uint32_t                    C_tokens      // TOKENS per tile (kv_chunk_size_ptr[0])
) -> std::tuple<
      std::vector<IdType>, // request_indices
      std::vector<IdType>, // kv_head_indices
      std::vector<IdType>, // kv_tile_indices (absolute j)
      std::vector<IdType>, // o_indptr_active (len B+1), rows = A_i per request
      uint32_t             // nnz_total_active = Σ_i A_i
    >
{
  std::vector<IdType> request_indices;
  std::vector<IdType> kv_head_indices;
  std::vector<IdType> kv_tile_indices;
  std::vector<IdType> o_indptr(batch_size + 1, 0);

  // First pass: compute A_i and prefix sum for compact CSR
  std::vector<uint32_t> A(batch_size, 0);
  uint64_t nnz_total_u64 = 0;

  for (uint32_t i = 0; i < batch_size; ++i) {
    const uint32_t L      = static_cast<uint32_t>(L_tokens[i]);
    if (L == 0) { o_indptr[i + 1] = static_cast<IdType>(nnz_total_u64); continue; }

    const uint32_t J_i    = ceil_div<uint32_t>(L, C_tokens);
    const uint32_t W_i      = std::min<uint32_t>(L, window_left + 1u);
    const uint32_t start  = L - W_i;                 // start token index (from request start)
    const uint32_t tile0  = start / C_tokens;        // FLOOR
    const uint32_t A_i    = (J_i > tile0) ? (J_i - tile0) : 0;

    A[i] = A_i;
    nnz_total_u64 += A_i;
    o_indptr[i + 1] = static_cast<IdType>(nnz_total_u64);
  }
  const uint32_t nnz_total = static_cast<uint32_t>(nnz_total_u64);

  // Reserve enough for all emitted CTAs (A_i tiles per KV head)
  request_indices.reserve(static_cast<size_t>(nnz_total) * num_kv_heads);
  kv_head_indices.reserve(static_cast<size_t>(nnz_total) * num_kv_heads);
  kv_tile_indices.reserve(static_cast<size_t>(nnz_total) * num_kv_heads);

  // Emit only active tiles j ∈ [tile0 .. J_i - 1] for each KV head
  for (uint32_t i = 0; i < batch_size; ++i) {
    const uint32_t L      = static_cast<uint32_t>(L_tokens[i]);
    if (L == 0) continue;

    const uint32_t J_i    = ceil_div<uint32_t>(L, C_tokens);
    const uint32_t W_i    = std::min<uint32_t>(L, window_left + 1u);
    const uint32_t start  = L - W_i;
    const uint32_t tile0  = start / C_tokens;        // FLOOR

    if (tile0 >= J_i) continue;                      // no active tiles
    for (uint32_t h = 0; h < num_kv_heads; ++h) {
      for (uint32_t j = tile0; j < J_i; ++j) {
        request_indices.push_back(static_cast<IdType>(i));
        kv_head_indices.push_back(static_cast<IdType>(h));
        kv_tile_indices.push_back(static_cast<IdType>(j));  // ABSOLUTE tile id (important)
      }
    }
  }

  return {request_indices, kv_head_indices, kv_tile_indices, o_indptr, nnz_total};
}

/*!
 * \brief Work estimation wrapper (mirrors the signature/pattern of the original Prepared()
 *        helper) but operates in TOKENS and uses window_left to compute active tiles.
 */
template <uint32_t GROUP_SIZE, uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE,
          typename AttentionVariant, typename Params>
inline cudaError_t RectificationCacheDecodeWorkEstimationHeadPackedPrepared(
    /*out*/ bool&      split_kv,
    /*out*/ uint32_t&  max_grid_size,
    /*out*/ uint32_t&  kv_chunk_size_in_tokens,  // C_tokens
    /*out*/ uint32_t&  new_batch_size,           // X (CTAs)
    /*out*/ uint32_t&  gdy,                      // = 1
    uint32_t           batch_size,
    uint32_t           num_qo_heads,
    uint32_t           /*page_size_unused*/,
    bool               enable_cuda_graph,
    const std::vector<typename Params::IdType>& L_tokens,  // TOKENS
    uint32_t           window_left) {

  using DTypeKV = typename Params::DTypeKV;
  using IdType  = typename Params::IdType;

  const uint32_t num_kv_heads = num_qo_heads / GROUP_SIZE;

  // ---- Occupancy-based capacity (same as original path) ----
  auto compute_capacity = GetCudaComputeCapability();
  DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, {
    constexpr uint32_t vec_size = std::max(16U / static_cast<uint32_t>(sizeof(DTypeKV)),
                                           HEAD_DIM / 32U);
    constexpr uint32_t bdx = HEAD_DIM / vec_size;
    static_assert(bdx <= 32, "HEAD_DIM/vec_size must be <= 32");
    constexpr uint32_t bdy = GROUP_SIZE;
    constexpr uint32_t num_threads = std::max(128U, bdx * bdy);
    constexpr uint32_t bdz = num_threads / (bdx * bdy);
    constexpr uint32_t tile_size_per_bdx =
        (GROUP_SIZE == 1) ? ((sizeof(DTypeKV) == 1) ? 2U : 4U) : 1U;
    const uint32_t head_dim = HEAD_DIM;

    const uint32_t smem_size =
        2 * NUM_STAGES_SMEM * tile_size_per_bdx * bdy * bdz * head_dim * sizeof(DTypeKV) +
        std::max(tile_size_per_bdx * num_threads * (uint32_t)sizeof(DTypeKV*),
                 2U * bdy * bdz * (uint32_t)sizeof(float));

    auto kernel = BatchRectificationCacheDecodeWithPagedKVCacheKernel<
        POS_ENCODING_MODE, NUM_STAGES_SMEM, tile_size_per_bdx, vec_size, bdx, bdy, bdz,
        AttentionVariant, Params>;

    int dev_id = 0, num_sm = 0, num_blocks_per_sm = 0;
    FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
    FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
    FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks_per_sm, kernel, num_threads, smem_size));

    max_grid_size = static_cast<uint32_t>(num_blocks_per_sm) * static_cast<uint32_t>(num_sm);

    // In window-based head-packed path we always split-kv (balanced tail merge)
    split_kv = true;
    gdy = 1;

    // Minimal tile size in TOKENS (match ~128 tokens min from original)
    constexpr uint32_t min_tokens_per_tile = 128U;

    uint32_t C_tokens, X;
    std::tie(C_tokens, X) = WindowPartitionHeadPackedTokens<IdType>(
        max_grid_size, L_tokens, batch_size, num_kv_heads, window_left, min_tokens_per_tile);

    kv_chunk_size_in_tokens = C_tokens;
    new_batch_size = X;
    return cudaSuccess;
  })
}



template <uint32_t HEAD_DIM_CKV, uint32_t HEAD_DIM_KPE, typename AttentionVariant, typename Params>
inline cudaError_t BatchDecodeWithPagedKVCacheWorkEstimationDispatchedMLA(
    bool& split_kv, uint32_t& max_grid_size, uint32_t& max_num_pages_per_batch,
    uint32_t& new_batch_size, uint32_t& gdy, uint32_t batch_size,
    typename Params::IdType* kv_indptr_h, const uint32_t num_qo_heads, const uint32_t page_size,
    bool enable_cuda_graph, cudaStream_t stream) {
  using DTypeKV = typename Params::DTypeKV;
  using IdType = typename Params::IdType;

  auto compute_capacity = GetCudaComputeCapability();
  DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, {
    constexpr uint32_t vec_size_ckv = std::max(16UL / sizeof(DTypeKV), HEAD_DIM_CKV / 32UL);
    constexpr uint32_t bdx = HEAD_DIM_CKV / vec_size_ckv;
    constexpr uint32_t vec_size_kpe = HEAD_DIM_KPE / bdx;

    constexpr uint32_t bdy = 8;
    constexpr uint32_t tile_size_qo_heads = 2;
    constexpr uint32_t qo_heads_per_block = bdy * tile_size_qo_heads;
    constexpr uint32_t num_threads = std::max(128U, bdx * bdy);
    constexpr uint32_t bdz = num_threads / (bdx * bdy);
    gdy = ceil_div(num_qo_heads, qo_heads_per_block);

    const uint32_t smem_size =
        NUM_STAGES_SMEM * bdy * bdz * (HEAD_DIM_CKV + HEAD_DIM_KPE) * sizeof(DTypeKV) +
        std::max(num_threads * sizeof(size_t) * 2, 2 * bdy * bdz * sizeof(float));

    auto kernel =
        BatchDecodeWithPagedKVCacheKernelMLA<NUM_STAGES_SMEM, vec_size_ckv, vec_size_kpe, bdx, bdy,
                                             bdz, tile_size_qo_heads, AttentionVariant, Params>;
    int num_blocks_per_sm = 0;
    int num_sm = 0;
    int dev_id = 0;
    FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
    FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
    FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel,
                                                                       num_threads, smem_size));
    max_grid_size = num_blocks_per_sm * num_sm;
    if (batch_size * gdy >= max_grid_size) {
      split_kv = false;
      max_num_pages_per_batch = 1;
      for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        max_num_pages_per_batch = std::max<uint32_t>(
            max_num_pages_per_batch, kv_indptr_h[batch_idx + 1] - kv_indptr_h[batch_idx]);
      }
      new_batch_size = batch_size;
    } else {
      // compute max_num_pages_per_batch and new_batch_size
      std::vector<IdType> num_pages(batch_size);
      for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        num_pages[batch_idx] = kv_indptr_h[batch_idx + 1] - kv_indptr_h[batch_idx];
      }
      std::tie(max_num_pages_per_batch, new_batch_size) =
          PartitionPagedKVCacheBinarySearchMinNumPagePerBatch(max_grid_size, gdy, num_pages,
                                                              std::max(128 / page_size, 1U));
      if (new_batch_size == batch_size && !enable_cuda_graph) {
        // do not use partition-kv kernel for short sequence, when not using CUDAGraph
        split_kv = false;
      } else {
        // when using CUDAGraph, we always use partition-kv kernel
        split_kv = true;
      }
    }

    return cudaSuccess;
  });
}

template <uint32_t HEAD_DIM_CKV, uint32_t HEAD_DIM_KPE, uint32_t QO_TILE_LEN,
          typename AttentionVariant, typename Params>
inline cudaError_t BatchDecodeWithPagedKVCacheWorkEstimationDispatchedMlaCuteSM80(
    bool& split_kv, uint32_t& max_grid_size, uint32_t& max_num_pages_per_batch,
    uint32_t& new_batch_size, uint32_t& gdy_, uint32_t batch_size,
    typename Params::IdType* kv_indptr_h, const uint32_t num_qo_heads, const uint32_t page_size,
    bool enable_cuda_graph, cudaStream_t stream) {
  using DTypeKV = typename Params::DTypeKV;
  using IdType = typename Params::IdType;

  auto [smem_size, gdy, k_warps] =
      LaunchSpecForDecodeKernelMlaCuteSM80<HEAD_DIM_CKV, HEAD_DIM_KPE, QO_TILE_LEN, DTypeKV>(
          num_qo_heads);
  gdy_ = gdy;
  const uint32_t num_threads = k_warps * 32;
  auto kernel =
      BatchDecodeWithPagedKVCacheKernelMlaCuteSM80<HEAD_DIM_CKV, HEAD_DIM_KPE, QO_TILE_LEN, Params>;
  int num_blocks_per_sm;
  int num_sm = 0;
  int dev_id = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));

  // FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel,
  //                                   num_threads, smem_size));
  // fixme: num_blocks_per_sm is 0 derived from cudaOccupancyMaxActiveBlocksPerMultiprocessor at
  // times, and we fill smem with q-heads as many as possible, so num_blocks_per_sm should be 1
  num_blocks_per_sm = 1;

  max_grid_size = num_blocks_per_sm * num_sm;
  if (batch_size * gdy >= max_grid_size) {
    split_kv = false;
    max_num_pages_per_batch = 1;
    for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      max_num_pages_per_batch = std::max<uint32_t>(
          max_num_pages_per_batch, kv_indptr_h[batch_idx + 1] - kv_indptr_h[batch_idx]);
    }
    new_batch_size = batch_size;
  } else {
    // compute max_num_pages_per_batch and new_batch_size
    std::vector<IdType> num_pages(batch_size);
    for (uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      num_pages[batch_idx] = kv_indptr_h[batch_idx + 1] - kv_indptr_h[batch_idx];
    }
    std::tie(max_num_pages_per_batch, new_batch_size) =
        PartitionPagedKVCacheBinarySearchMinNumPagePerBatch(max_grid_size, gdy, num_pages,
                                                            std::max(128 / page_size, 1U));
    if (new_batch_size == batch_size && !enable_cuda_graph) {
      // do not use partition-kv kernel for short sequence, when not using CUDAGraph
      split_kv = false;
    } else {
      // when using CUDAGraph, we always use partition-kv kernel
      split_kv = true;
    }
  }

  return cudaSuccess;
}

/*!
 * \brief Partition Paged KV-Cache into multiple chunks on KV sequence length
 * \tparam IdType A template type indicates the index data type
 * \param old_batch_size The batch size of the old Paged KV-Cache
 * \param old_page_indptr_h The host-side page indptr of the old Paged KV-Cache
 * \param max_num_pages_per_batch The maximum number of pages per batch
 * \param new_paged_kv_d The device-side new Paged KV-Cache
 * \param stream The cuda stream to launch the kernel
 * \return status Indicates whether CUDA calls are successful
 */
template <typename IdType>
inline auto DecodeSplitKVIndptr(IdType* indptr_h, uint32_t batch_size, uint32_t kv_chunk_size) {
  std::vector<IdType> request_indices, kv_tile_indices, o_indptr;
  o_indptr.push_back(0);

  for (uint32_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    uint32_t num_tiles_kv = ceil_div(
        std::max<uint32_t>(indptr_h[batch_idx + 1] - indptr_h[batch_idx], 1U), kv_chunk_size);
    
    for (uint32_t kv_tile_idx = 0; kv_tile_idx < num_tiles_kv; ++kv_tile_idx) {
      request_indices.push_back(batch_idx);
      kv_tile_indices.push_back(kv_tile_idx);
    }
    o_indptr.push_back(o_indptr.back() + num_tiles_kv);
  }

  return std::make_tuple(request_indices, kv_tile_indices, o_indptr);
}

struct DecodePlanInfo {
  int64_t padded_batch_size;
  int64_t v_offset;
  int64_t s_offset;
  int64_t downdate_v_offset;
  int64_t downdate_s_offset;
  int64_t request_indices_offset;
  int64_t kv_tile_indices_offset;
  int64_t o_indptr_offset;
  int64_t block_valid_mask_offset;
  int64_t kv_chunk_size_ptr_offset;
  bool enable_cuda_graph;
  bool split_kv;

  DecodePlanInfo()
      : padded_batch_size(0),
        v_offset(0),
        s_offset(0),
        downdate_v_offset(0),
        downdate_s_offset(0),
        request_indices_offset(0),
        kv_tile_indices_offset(0),
        o_indptr_offset(0),
        block_valid_mask_offset(0),
        kv_chunk_size_ptr_offset(0),
        enable_cuda_graph(false),
        split_kv(false) {}

  // convert DecodePlanInfo to std::vector<int64_t>
  std::vector<int64_t> ToVector() const {
    return {padded_batch_size,
            v_offset,
            s_offset,
            downdate_v_offset,
            downdate_s_offset,
            request_indices_offset,
            kv_tile_indices_offset,
            o_indptr_offset,
            block_valid_mask_offset,
            kv_chunk_size_ptr_offset,
            enable_cuda_graph,
            split_kv};
  }

  // From std::vector<int64_t> to DecodePlanInfo
  void FromVector(const std::vector<int64_t>& vec) {
    if (vec.size() != 12) {
      std::ostringstream err_msg;
      err_msg << "DecodePlanInfo::FromVector: vec.size() should be 12, but got " << vec.size();
      FLASHINFER_ERROR(err_msg.str());
    }
    padded_batch_size = vec[0];
    v_offset = vec[1];
    s_offset = vec[2];
    downdate_v_offset = vec[3];
    downdate_s_offset = vec[4];
    request_indices_offset = vec[5];
    kv_tile_indices_offset = vec[6];
    o_indptr_offset = vec[7];
    block_valid_mask_offset = vec[8];
    kv_chunk_size_ptr_offset = vec[9];
    enable_cuda_graph = vec[10];
    split_kv = vec[11];
  }
};

template <uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE, typename AttentionVariant,
          typename Params, typename WorkEstimationFunc>
inline cudaError_t DecodePlan(void* float_buffer, size_t float_workspace_size_in_bytes,
                              void* int_buffer, void* page_locked_int_buffer,
                              size_t int_workspace_size_in_bytes, DecodePlanInfo& plan_info,
                              typename Params::IdType* indptr_h, uint32_t batch_size,
                              uint32_t num_qo_heads, uint32_t page_size, bool enable_cuda_graph,
                              cudaStream_t stream, WorkEstimationFunc work_estimation_func) {
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  bool split_kv;
  uint32_t max_grid_size, kv_chunk_size_in_pages, new_batch_size, gdy;
  FLASHINFER_CUDA_CALL(work_estimation_func(split_kv, max_grid_size, kv_chunk_size_in_pages,
                                            new_batch_size, gdy, batch_size, indptr_h, num_qo_heads,
                                            page_size, enable_cuda_graph, stream));
  // printf("max_grid_size: %" PRIu32 ", kv_chunk_size_in_pages(max_num_pages_per_batch): %" PRIu32 ", new_batch_size: %" PRIu32 ", gdy: %" PRIu32 "\n", max_grid_size, kv_chunk_size_in_pages, new_batch_size, gdy);
  size_t padded_batch_size;
  plan_info.enable_cuda_graph = enable_cuda_graph;
  plan_info.split_kv = split_kv;
  padded_batch_size =
      (enable_cuda_graph) ? (split_kv ? max_grid_size / gdy : batch_size) : new_batch_size;
  plan_info.padded_batch_size = padded_batch_size;

  auto [request_indices_vec, kv_tile_indices_vec, o_indptr_vec] =
      DecodeSplitKVIndptr(indptr_h, batch_size, kv_chunk_size_in_pages);

  AlignedAllocator int_allocator(int_buffer, int_workspace_size_in_bytes);
  plan_info.request_indices_offset = int_allocator.aligned_alloc_offset(
      padded_batch_size * sizeof(IdType), 16, "batch_decode_request_indices");
  plan_info.kv_tile_indices_offset = int_allocator.aligned_alloc_offset(
      padded_batch_size * sizeof(IdType), 16, "batch_decode_kv_tile_indices");
  plan_info.o_indptr_offset = int_allocator.aligned_alloc_offset(
      (padded_batch_size + 1) * sizeof(IdType), 16, "batch_decode_o_indptr");
  plan_info.kv_chunk_size_ptr_offset =
      int_allocator.aligned_alloc_offset(sizeof(IdType), 1, "batch_decode_kv_chunk_size_ptr");
  IdType* request_indices_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.request_indices_offset);
  IdType* kv_tile_indices_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_tile_indices_offset);
  IdType* o_indptr_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.o_indptr_offset);
  IdType* kv_chunk_size_ptr_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_chunk_size_ptr_offset);
  std::copy(request_indices_vec.begin(), request_indices_vec.end(), request_indices_h);
  std::copy(kv_tile_indices_vec.begin(), kv_tile_indices_vec.end(), kv_tile_indices_h);
  std::copy(o_indptr_vec.begin(), o_indptr_vec.end(), o_indptr_h);
  kv_chunk_size_ptr_h[0] = kv_chunk_size_in_pages * page_size;

  if (split_kv) {
    AlignedAllocator float_allocator(float_buffer, float_workspace_size_in_bytes);
    plan_info.v_offset = float_allocator.aligned_alloc_offset(
        num_qo_heads * padded_batch_size * HEAD_DIM * sizeof(DTypeO), 16, "batch_decode_tmp_v");
    plan_info.s_offset = float_allocator.aligned_alloc_offset(
        num_qo_heads * padded_batch_size * sizeof(float), 16, "batch_decode_tmp_s");
    plan_info.downdate_v_offset = float_allocator.aligned_alloc_offset(
        num_qo_heads * padded_batch_size * HEAD_DIM * sizeof(DTypeO), 16,
        "batch_decode_tmp_v_downdate");
    plan_info.downdate_s_offset = float_allocator.aligned_alloc_offset(
        num_qo_heads * padded_batch_size * sizeof(float), 16,
        "batch_decode_tmp_s_downdate");

    plan_info.block_valid_mask_offset = int_allocator.aligned_alloc_offset(
        padded_batch_size * sizeof(bool), 16, "batch_decode_block_valid_mask");
    bool* block_valid_mask_h =
        GetPtrFromBaseOffset<bool>(page_locked_int_buffer, plan_info.block_valid_mask_offset);
    for (uint32_t i = 0; i < padded_batch_size; ++i) {
      block_valid_mask_h[i] = i < new_batch_size;
    }
  }

  size_t num_bytes_to_copy = int_allocator.num_allocated_bytes();

  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(int_buffer, page_locked_int_buffer, num_bytes_to_copy,
                                       cudaMemcpyHostToDevice, stream));
  return cudaSuccess;
}



struct MACAttentionDecodePlanInfo {
  int64_t padded_batch_size;
  int64_t v_offset;
  int64_t s_offset;
  int64_t v_dd_offset;
  int64_t s_dd_offset;
  int64_t request_indices_offset;
  int64_t kv_tile_indices_offset;
  int64_t kv_head_indices_offset;      // NEW
  int64_t merge_start_offsets_offset;   // NEW
  int64_t downdate_start_offsets_offset;
  int64_t o_indptr_offset;
  int64_t block_valid_mask_offset;
  int64_t kv_chunk_size_ptr_offset;
  int64_t nnz_total;                // NEW: rows of (tmp_v,tmp_s)
  bool enable_cuda_graph;
  bool split_kv;

  MACAttentionDecodePlanInfo()
      : padded_batch_size(0),
        v_offset(0),
        s_offset(0),
        v_dd_offset(0),
        s_dd_offset(0),
        request_indices_offset(0),
        kv_tile_indices_offset(0),
        kv_head_indices_offset(0),     // NEW
        merge_start_offsets_offset(0),
        downdate_start_offsets_offset(0),
        o_indptr_offset(0),
        block_valid_mask_offset(0),
        kv_chunk_size_ptr_offset(0),
        nnz_total(0),
        enable_cuda_graph(false),
        split_kv(false) {}

  // convert DecodePlanInfo to std::vector<int64_t>
  std::vector<int64_t> ToVector() const {
    return {padded_batch_size,
            v_offset,
            s_offset,
            v_dd_offset,
            s_dd_offset,
            request_indices_offset,
            kv_tile_indices_offset,
            kv_head_indices_offset,        // NEW
            o_indptr_offset,
            block_valid_mask_offset,
            kv_chunk_size_ptr_offset, 
            merge_start_offsets_offset,
            downdate_start_offsets_offset,
            nnz_total,
            enable_cuda_graph,
            split_kv};
  }

  // From std::vector<int64_t> to DecodePlanInfo
  void FromVector(const std::vector<int64_t>& vec) {
    if (vec.size() != 16) {
      std::ostringstream err_msg;
      err_msg << "DecodePlanInfo::FromVector: vec.size() should be 16, but got " << vec.size();
      FLASHINFER_ERROR(err_msg.str());
    }
    padded_batch_size = vec[0];
    v_offset = vec[1];
    s_offset = vec[2];
    v_dd_offset = vec[3];
    s_dd_offset = vec[4];
    request_indices_offset     = vec[5];
    kv_tile_indices_offset     = vec[6];
    kv_head_indices_offset     = vec[7];   // NEW
    o_indptr_offset            = vec[8];
    block_valid_mask_offset    = vec[9];
    kv_chunk_size_ptr_offset   = vec[10];
    merge_start_offsets_offset = vec[11];
    downdate_start_offsets_offset = vec[12];
    nnz_total                  = vec[13];
    enable_cuda_graph          = (vec[14] != 0);
    split_kv                   = (vec[15] != 0);
  }
};


template <uint32_t GROUP_SIZE, uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE, typename AttentionVariant,
          typename Params, typename WorkEstimationFunc>
inline cudaError_t MACAttentionDecodePlan(void* float_buffer, size_t float_workspace_size_in_bytes,
                              void* int_buffer, void* page_locked_int_buffer,
                              size_t int_workspace_size_in_bytes,
                              MACAttentionDecodePlanInfo& plan_info,
                              const std::vector<typename Params::IdType>& P,                 // in PAGES
                              const std::vector<typename Params::IdType>& start_page_flat,   // in PAGES (group-min MAIN)
                              typename Params::IdType* indptr_h, uint32_t batch_size,
                              uint32_t num_qo_heads, uint32_t page_size, bool enable_cuda_graph,
                              cudaStream_t stream, WorkEstimationFunc work_estimation_func,
                              const uint32_t* attn_start_pos_dev, const uint32_t downdate_range,
                              const uint32_t* attn_start_pos_host_pinned) {
  PROFILE_SCOPE_STREAM("MACAttentionDecodePlan", stream);

  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  const uint32_t num_kv_heads = num_qo_heads / GROUP_SIZE;

  // ---- 1) Ask the estimator what to do (pass downdate_range through) ----
  uint32_t max_grid_size = 0, C_pages = 1, new_batch_size = 0, gdy = 1;
  bool split_kv = true;
  {
    NvtxScope   _n{"MACAttentionDecodePlan::work_estimation"};
    ScopedTimer _t{"MACAttentionDecodePlan::work_estimation", stream};
    FLASHINFER_CUDA_CALL(work_estimation_func(split_kv, max_grid_size, C_pages,
                                              new_batch_size, gdy, batch_size,
                                              /*kv_indptr_unused*/ nullptr,
                                              num_qo_heads, page_size, enable_cuda_graph,
                                              downdate_range));  // NEW
  }

  plan_info.enable_cuda_graph = enable_cuda_graph;
  plan_info.split_kv = split_kv;

  // ---- 2) Two branches ----------------------------------------------------
  if (!split_kv) {
    NvtxScope   _n_branch{"MACAttentionDecodePlan::non_split_path"};
    ScopedTimer _t_branch{"MACAttentionDecodePlan::non_split_path", stream};

    // ======================
    // Fallback: NON-SPLIT, gdy=1, uniform flattened layout
    // ======================
    const uint32_t H = num_kv_heads;
    const uint32_t X = batch_size;   // CTAs per x-axis “tile-slot”
    const uint32_t padded_batch_size = X;  // do NOT graph-pad in fallback
    plan_info.padded_batch_size = padded_batch_size;
    plan_info.nnz_total = 0;         // not used in non-split path

    // Build small uniform schedule for X entries
    std::vector<IdType> request_indices_vec, kv_tile_indices_vec, o_indptr_vec;
    {
      NvtxScope   _n2{"MACAttentionDecodePlan::BuildUniformFlattenedSchedule"};
      ScopedTimer _t2{"MACAttentionDecodePlan::BuildUniformFlattenedSchedule", stream};
      BuildUniformFlattenedSchedule<IdType>(batch_size, request_indices_vec,
                                            kv_tile_indices_vec, o_indptr_vec);
    }

    // Allocate integers
    AlignedAllocator int_alloc(int_buffer, int_workspace_size_in_bytes);
    {
      NvtxScope   _n2{"MACAttentionDecodePlan::alloc_integers_non_split"};
      ScopedTimer _t2{"MACAttentionDecodePlan::alloc_integers_non_split", stream};
      plan_info.request_indices_offset =
          int_alloc.aligned_alloc_offset(padded_batch_size * sizeof(IdType), 16, "unif_req_idx");
      plan_info.kv_tile_indices_offset =
          int_alloc.aligned_alloc_offset(padded_batch_size * sizeof(IdType), 16, "unif_kvt_idx");
      plan_info.kv_head_indices_offset = 0; // not used
      plan_info.o_indptr_offset =
          int_alloc.aligned_alloc_offset((batch_size + 1) * sizeof(IdType), 16, "unif_o_indptr");
      plan_info.kv_chunk_size_ptr_offset =
          int_alloc.aligned_alloc_offset(sizeof(IdType), 1, "unif_kv_chunk");
      if (enable_cuda_graph && padded_batch_size > X) {
        plan_info.block_valid_mask_offset =
            int_alloc.aligned_alloc_offset(padded_batch_size * sizeof(bool), 16, "unif_mask");
        bool* mask_h = GetPtrFromBaseOffset<bool>(page_locked_int_buffer, plan_info.block_valid_mask_offset);
        for (uint32_t i = 0; i < padded_batch_size; ++i) mask_h[i] = (i < X);
      } else {
        plan_info.block_valid_mask_offset = 0;
      }
    }

    // No float temporaries in non-split
    plan_info.v_offset = 0;
    plan_info.s_offset = 0;
    plan_info.v_dd_offset = 0;
    plan_info.s_dd_offset = 0;
    plan_info.merge_start_offsets_offset = 0;
    plan_info.downdate_start_offsets_offset = 0;

    // Copy to page-locked host buffers (pad to padded_batch_size)
    {
      NvtxScope   _n2{"MACAttentionDecodePlan::populate_host_buffers_non_split"};
      ScopedTimer _t2{"MACAttentionDecodePlan::populate_host_buffers_non_split", stream};
      IdType* req_h = GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.request_indices_offset);
      IdType* kvt_h = GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_tile_indices_offset);
      IdType* ind_h = GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.o_indptr_offset);
      IdType* csz_h = GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_chunk_size_ptr_offset);

      const size_t n_emit = std::min<size_t>(request_indices_vec.size(), padded_batch_size);
      std::copy(request_indices_vec.begin(), request_indices_vec.begin() + n_emit, req_h);
      std::copy(kv_tile_indices_vec.begin(), kv_tile_indices_vec.begin() + n_emit, kvt_h);
      for (size_t i = n_emit; i < padded_batch_size; ++i) { req_h[i] = 0; kvt_h[i] = 0; }

      std::copy(o_indptr_vec.begin(), o_indptr_vec.end(), ind_h);
      csz_h[0] = static_cast<IdType>(C_pages * page_size); // ignored when partition_kv=false
    }

    {
      NvtxScope   _n2{"MACAttentionDecodePlan::HtoD_int_copy_non_split"};
      ScopedTimer _t2{"MACAttentionDecodePlan::HtoD_int_copy_non_split", stream};
      const size_t bytes = int_alloc.num_allocated_bytes();
      FLASHINFER_CUDA_CALL(cudaMemcpyAsync(int_buffer, page_locked_int_buffer, bytes,
                                          cudaMemcpyHostToDevice, stream));
    }
    return cudaSuccess;
  }

  // ======================
  // Load-balanced head-packed split, gdy=1
  // ======================
  {
    NvtxScope   _n_branch{"MACAttentionDecodePlan::split_head_packed_path"};
    ScopedTimer _t_branch{"MACAttentionDecodePlan::split_head_packed_path", stream};

    // --- Build UNION(start_page) for scheduling: min(main_group, downdate) in PAGES.
    const uint32_t H = num_kv_heads;
    const uint32_t downdate_pages_kept =
        (downdate_range == 0) ? 0 : ceil_div<uint32_t>(downdate_range + 1, page_size);

    std::vector<IdType> start_page_dd(batch_size);  // per-request, in pages
    {
      NvtxScope   _n2{"MACAttentionDecodePlan::compute_start_page_dd"};
      ScopedTimer _t2{"MACAttentionDecodePlan::compute_start_page_dd", stream};
      for (uint32_t i = 0; i < batch_size; ++i) {
        const uint32_t P_i = static_cast<uint32_t>(P[i]);
        start_page_dd[i] = static_cast<IdType>((P_i > downdate_pages_kept) ? (P_i - downdate_pages_kept) : 0u);
      }
    }

    std::vector<IdType> start_page_flat_union(static_cast<size_t>(batch_size) * H);
    {
      NvtxScope   _n2{"MACAttentionDecodePlan::build_union_start_pages"};
      ScopedTimer _t2{"MACAttentionDecodePlan::build_union_start_pages", stream};
      for (uint32_t i = 0; i < batch_size; ++i) {
        for (uint32_t h = 0; h < H; ++h) {
          const IdType sp_main = start_page_flat[i * H + h];
          const IdType sp_dd   = start_page_dd[i];
          start_page_flat_union[i * H + h] = std::min(sp_main, sp_dd);  // union = earlier start
        }
      }
    }

    // Emit schedule & CSR (and get nnz_total) from the UNION starts
    std::vector<IdType> req_vec, kvh_vec, kvt_vec, oind_vec;
    uint32_t nnz_total = 0;
    {
      NvtxScope   _n2{"MACAttentionDecodePlan::BuildHeadPackedSchedule"};
      ScopedTimer _t2{"MACAttentionDecodePlan::BuildHeadPackedSchedule", stream};
      std::tie(req_vec, kvh_vec, kvt_vec, oind_vec, nnz_total) =
          BuildHeadPackedSchedule<IdType>(P, start_page_flat_union, batch_size, num_kv_heads, C_pages);
    }

    const uint32_t padded_batch_size = enable_cuda_graph ? max_grid_size : new_batch_size;
    plan_info.padded_batch_size = padded_batch_size;
    plan_info.nnz_total = nnz_total;

    // Allocate integers
    AlignedAllocator int_alloc(int_buffer, int_workspace_size_in_bytes);
    {
      NvtxScope   _n2{"MACAttentionDecodePlan::alloc_integers_split"};
      ScopedTimer _t2{"MACAttentionDecodePlan::alloc_integers_split", stream};
      plan_info.request_indices_offset =
          int_alloc.aligned_alloc_offset(padded_batch_size * sizeof(IdType), 16, "hp_req_idx");
      plan_info.kv_head_indices_offset =
          int_alloc.aligned_alloc_offset(padded_batch_size * sizeof(IdType), 16, "hp_kvh_idx");
      plan_info.kv_tile_indices_offset =
          int_alloc.aligned_alloc_offset(padded_batch_size * sizeof(IdType), 16, "hp_kvt_idx");
      plan_info.o_indptr_offset =
          int_alloc.aligned_alloc_offset((batch_size + 1) * sizeof(IdType), 16, "hp_o_indptr");
      plan_info.kv_chunk_size_ptr_offset =
          int_alloc.aligned_alloc_offset(sizeof(IdType), 1, "hp_kv_chunk");

      plan_info.merge_start_offsets_offset =
          int_alloc.aligned_alloc_offset(static_cast<size_t>(batch_size) * num_qo_heads * sizeof(IdType),
                                         16, "hp_merge_start_ofs");

      plan_info.downdate_start_offsets_offset =
          int_alloc.aligned_alloc_offset(static_cast<size_t>(batch_size) * num_qo_heads * sizeof(IdType),
                                         16, "hp_downdate_start_ofs");

      if (enable_cuda_graph && padded_batch_size > new_batch_size) {
        plan_info.block_valid_mask_offset =
            int_alloc.aligned_alloc_offset(padded_batch_size * sizeof(bool), 16, "hp_mask");
      } else {
        plan_info.block_valid_mask_offset = 0;
      }
    }

    // Allocate float temporaries by nnz_total
    AlignedAllocator float_alloc(float_buffer, float_workspace_size_in_bytes);
    {
      NvtxScope   _n2{"MACAttentionDecodePlan::alloc_float_temporaries_split"};
      ScopedTimer _t2{"MACAttentionDecodePlan::alloc_float_temporaries_split", stream};
      plan_info.v_offset = float_alloc.aligned_alloc_offset(
          static_cast<size_t>(num_qo_heads) * nnz_total * HEAD_DIM * sizeof(DTypeO), 16, "hp_tmp_v");
      plan_info.s_offset = float_alloc.aligned_alloc_offset(
          static_cast<size_t>(num_qo_heads) * nnz_total * sizeof(float), 16, "hp_tmp_s");
      plan_info.v_dd_offset = float_alloc.aligned_alloc_offset(
          static_cast<size_t>(num_qo_heads) * nnz_total * HEAD_DIM * sizeof(DTypeO), 16, "hp_tmp_v_dd");
      plan_info.s_dd_offset = float_alloc.aligned_alloc_offset(
          static_cast<size_t>(num_qo_heads) * nnz_total * sizeof(float), 16, "hp_tmp_s_dd");
    }

    // Copy to page-locked host buffers
    IdType* req_h  = GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.request_indices_offset);
    IdType* kvh_h  = GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_head_indices_offset);
    IdType* kvt_h  = GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_tile_indices_offset);
    IdType* ind_h  = GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.o_indptr_offset);
    IdType* csz_h  = GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_chunk_size_ptr_offset);
    IdType* ofs_h  = GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.merge_start_offsets_offset);
    IdType* downdate_ofs_h =
        GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.downdate_start_offsets_offset);

    {
      NvtxScope   _n2{"MACAttentionDecodePlan::populate_host_buffers_split"};
      ScopedTimer _t2{"MACAttentionDecodePlan::populate_host_buffers_split", stream};
      const size_t n_emit = std::min<size_t>(req_vec.size(), padded_batch_size);
      std::copy(req_vec.begin(),  req_vec.begin()  + n_emit, req_h);
      std::copy(kvh_vec.begin(),  kvh_vec.begin()  + n_emit, kvh_h);
      std::copy(kvt_vec.begin(),  kvt_vec.begin()  + n_emit, kvt_h);
      for (size_t i = n_emit; i < padded_batch_size; ++i) { req_h[i]=0; kvh_h[i]=0; kvt_h[i]=0; }

      std::copy(oind_vec.begin(), oind_vec.end(), ind_h);
      csz_h[0] = static_cast<IdType>(C_pages * page_size);
    }

    // === Per-(pos, head) merge_start_offsets for MAIN path (in tiles) ===
    {
      NvtxScope   _n2{"MACAttentionDecodePlan::compute_merge_start_offsets_MAIN"};
      ScopedTimer _t2{"MACAttentionDecodePlan::compute_merge_start_offsets_MAIN", stream};

      // Fast constant divisor for all divisions by C_pages
      FastDivU32 divC(C_pages);

      // Precompute Ji = ceil(P[i]/C_pages) and tile0_dd[i] = floor(start_page_dd[i]/C_pages)
      std::vector<uint32_t> J(batch_size);
      std::vector<uint32_t> tile0_dd(batch_size);
      for (uint32_t i = 0; i < batch_size; ++i) {
        const uint32_t Pi = static_cast<uint32_t>(P[i]);
        J[i]        = (Pi == 0u) ? 0u : divC.ceil_div(Pi);
        tile0_dd[i] = divC.floor_div(static_cast<uint32_t>(start_page_dd[i]));
      }

      // Tight pointer aliases (cache-friendly, fewer recomputations)
      const IdType*         spf   = start_page_flat.data();     // [B*Hkv], in pages
      const uint32_t*       attn  = attn_start_pos_host_pinned; // [B*Hqo], in tokens (page_size=1 assumed here)
      IdType*               ofs   = ofs_h;                      // [B*Hqo], out
      const uint32_t        Hkv   = num_kv_heads;
      const uint32_t        Hqo   = num_qo_heads;               // should be Hkv * GROUP_SIZE
      constexpr uint32_t    G     = GROUP_SIZE;

      // Parallelize across (i, hkv). Each iteration writes to a disjoint span in ofs[].
      #if defined(_OPENMP)
      #pragma omp parallel for collapse(2) schedule(static)
      #endif
      for (int i = 0; i < (int)batch_size; ++i) {
        for (int hkv = 0; hkv < (int)Hkv; ++hkv) {
          const uint32_t Ji = J[(uint32_t)i];

          // MAIN group-min start (pages) -> tile
          const uint32_t sp_main_pages = static_cast<uint32_t>(spf[(size_t)i * Hkv + (uint32_t)hkv]);
          const uint32_t tile0_main    = divC.floor_div(sp_main_pages);

          // UNION start for scheduling with downdate
          const uint32_t tile0_sched   = (tile0_main < tile0_dd[(uint32_t)i]) ? tile0_main : tile0_dd[(uint32_t)i];

          const size_t base = (size_t)i * Hqo + (size_t)hkv * G;

          // If the union start already exceeds Ji, the whole head-group clamps to Ji
          if (Ji <= tile0_sched) {
            #pragma unroll
            for (uint32_t g = 0; g < G; ++g) ofs[base + g] = static_cast<IdType>(Ji);
            continue;
          }

          // Otherwise compute per-head start tiles and clamp to Ji
          #if defined(__GNUC__) || defined(__clang__)
          __builtin_prefetch(attn + base, 0, 1);
          #endif

          #pragma unroll
          for (uint32_t g = 0; g < G; ++g) {
            const uint32_t start_pos_tokens = attn[base + g];
            // If tokens != pages, use a composite divisor: divCT = FastDivU32(C_pages * page_size_tokens)
            const uint32_t tile0_head = divC.floor_div(start_pos_tokens); // page_size=1 path
            const uint32_t tmax       = (tile0_head > tile0_sched) ? tile0_head : tile0_sched;
            const uint32_t ofs_val    = (Ji < tmax) ? Ji : tmax;
            ofs[base + g]             = static_cast<IdType>(ofs_val);
          }
        }
      }
    }

    // === Per-(pos, head) downdate_start_offsets (in tiles; replicated across heads) ===
    {
      NvtxScope   _n2{"MACAttentionDecodePlan::compute_downdate_start_offsets_DD"};
      ScopedTimer _t2{"MACAttentionDecodePlan::compute_downdate_start_offsets_DD", stream};
      for (uint32_t i = 0; i < batch_size; ++i) {
        const uint32_t J_i = (P[i] == 0) ? 0 : ceil_div<uint32_t>(static_cast<uint32_t>(P[i]), C_pages);
        const uint32_t tile0_group_dd  = static_cast<uint32_t>(start_page_dd[i]) / C_pages;

        for (uint32_t hkv = 0; hkv < num_kv_heads; ++hkv) {
          const uint32_t tile0_group_main = static_cast<uint32_t>(start_page_flat[i * num_kv_heads + hkv]) / C_pages;
          const uint32_t tile0_sched      = std::min(tile0_group_main, tile0_group_dd);

          // Per-head DD start is identical across heads for a request; replicate:
          const uint32_t tile0_head_dd    = tile0_group_dd;

          const uint32_t ofs_dd = std::min<uint32_t>(J_i, std::max<uint32_t>(tile0_head_dd, tile0_sched));
          for (uint32_t g = 0; g < GROUP_SIZE; ++g) {
            const uint32_t hq = hkv * GROUP_SIZE + g;
            downdate_ofs_h[i * num_qo_heads + hq] = static_cast<IdType>(ofs_dd);
          }
        }
      }
    }

    if (enable_cuda_graph && padded_batch_size > new_batch_size) {
      NvtxScope   _n2{"MACAttentionDecodePlan::populate_block_valid_mask"};
      ScopedTimer _t2{"MACAttentionDecodePlan::populate_block_valid_mask", stream};
      bool* mask_h = GetPtrFromBaseOffset<bool>(page_locked_int_buffer, plan_info.block_valid_mask_offset);
      for (uint32_t i = 0; i < padded_batch_size; ++i) mask_h[i] = (i < new_batch_size);
    }

    {
      NvtxScope   _n2{"MACAttentionDecodePlan::HtoD_int_copy_split"};
      ScopedTimer _t2{"MACAttentionDecodePlan::HtoD_int_copy_split", stream};
      const size_t bytes = int_alloc.num_allocated_bytes();
      FLASHINFER_CUDA_CALL(cudaMemcpyAsync(int_buffer, page_locked_int_buffer, bytes,
                                          cudaMemcpyHostToDevice, stream));
    }
  }
  return cudaSuccess;
}




template <uint32_t GROUP_SIZE, uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE,
          typename AttentionVariant, typename Params, typename WorkEstimationFunc>
inline cudaError_t RectificationCacheDecodePlanCompact(
    void*                          float_buffer,
    size_t                         float_workspace_size_in_bytes,
    void*                          int_buffer,
    void*                          page_locked_int_buffer,
    size_t                         int_workspace_size_in_bytes,
    MACAttentionDecodePlanInfo&    plan_info,
    const std::vector<typename Params::IdType>& L_tokens,     // TOKENS per request
    uint32_t                       batch_size,
    uint32_t                       num_qo_heads,
    uint32_t                       window_left,                // TOKENS
    bool                           enable_cuda_graph,
    cudaStream_t                   stream,
    WorkEstimationFunc             work_estimation_func) {

  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;

  const uint32_t num_kv_heads = num_qo_heads / GROUP_SIZE;

  // 1) Work estimation => token tile size and number of CTAs (unchanged)
  uint32_t max_grid_size = 0, C_tokens = 1, new_batch_size = 0, gdy = 1;
  bool split_kv = true;
  FLASHINFER_CUDA_CALL(work_estimation_func(split_kv, max_grid_size, C_tokens,
                                            new_batch_size, gdy, batch_size,
                                            /*kv_indptr_unused*/ nullptr,
                                            num_qo_heads, /*page_size_unused*/ 1, enable_cuda_graph));

  plan_info.enable_cuda_graph = enable_cuda_graph;
  plan_info.split_kv = split_kv;

  // 2) Build compact head-packed schedule (active tail only)
  std::vector<IdType> req_vec, kvh_vec, kvt_vec, oind_vec;
  uint32_t nnz_total = 0;
  {
    auto tup = BuildWindowHeadPackedScheduleTokensCompact<IdType>(
                  L_tokens, batch_size, num_kv_heads, window_left, C_tokens);
    req_vec   = std::move(std::get<0>(tup));
    kvh_vec   = std::move(std::get<1>(tup));
    kvt_vec   = std::move(std::get<2>(tup));
    oind_vec  = std::move(std::get<3>(tup));
    nnz_total = std::get<4>(tup);
  }

  const uint32_t padded_batch_size = enable_cuda_graph ? max_grid_size : new_batch_size;
  plan_info.padded_batch_size = padded_batch_size;
  plan_info.nnz_total = nnz_total;

  // 3) Integer workspace
  AlignedAllocator int_alloc(int_buffer, int_workspace_size_in_bytes);
  plan_info.request_indices_offset =
      int_alloc.aligned_alloc_offset(padded_batch_size * sizeof(IdType), 16, "winC_req_idx");
  plan_info.kv_head_indices_offset =
      int_alloc.aligned_alloc_offset(padded_batch_size * sizeof(IdType), 16, "winC_kvh_idx");
  plan_info.kv_tile_indices_offset =
      int_alloc.aligned_alloc_offset(padded_batch_size * sizeof(IdType), 16, "winC_kvt_idx");
  plan_info.o_indptr_offset =
      int_alloc.aligned_alloc_offset((batch_size + 1) * sizeof(IdType), 16, "winC_o_indptr");
  plan_info.kv_chunk_size_ptr_offset =
      int_alloc.aligned_alloc_offset(sizeof(IdType), 1, "winC_kv_chunk_tokens");

  // IMPORTANT: no per-head start offsets needed with compact CSR
  plan_info.merge_start_offsets_offset = 0;
  plan_info.downdate_start_offsets_offset = 0;

  if (enable_cuda_graph && padded_batch_size > new_batch_size) {
    plan_info.block_valid_mask_offset =
        int_alloc.aligned_alloc_offset(padded_batch_size * sizeof(bool), 16, "winC_mask");
  } else {
    plan_info.block_valid_mask_offset = 0;
  }

  // 4) Float temporaries sized by nnz_total_active
  AlignedAllocator float_alloc(float_buffer, float_workspace_size_in_bytes);
  plan_info.v_offset = float_alloc.aligned_alloc_offset(
      static_cast<size_t>(num_qo_heads) * nnz_total * HEAD_DIM * sizeof(DTypeO), 16, "winC_tmp_v");
  plan_info.s_offset = float_alloc.aligned_alloc_offset(
      static_cast<size_t>(num_qo_heads) * nnz_total * sizeof(float), 16, "winC_tmp_s");
  plan_info.v_dd_offset = float_alloc.aligned_alloc_offset(
      static_cast<size_t>(num_qo_heads) * nnz_total * HEAD_DIM * sizeof(DTypeO), 16, "winC_tmp_v_dd");
  plan_info.s_dd_offset = float_alloc.aligned_alloc_offset(
      static_cast<size_t>(num_qo_heads) * nnz_total * sizeof(float), 16, "winC_tmp_s_dd");

  // 5) Copy page-locked host buffers
  IdType* req_h = GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.request_indices_offset);
  IdType* kvh_h = GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_head_indices_offset);
  IdType* kvt_h = GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_tile_indices_offset);
  IdType* ind_h = GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.o_indptr_offset);
  IdType* csz_h = GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_chunk_size_ptr_offset);

  const size_t n_emit = std::min<size_t>(req_vec.size(), padded_batch_size);
  std::copy(req_vec.begin(), req_vec.begin() + n_emit, req_h);
  std::copy(kvh_vec.begin(), kvh_vec.begin() + n_emit, kvh_h);
  std::copy(kvt_vec.begin(), kvt_vec.begin() + n_emit, kvt_h);
  for (size_t i = n_emit; i < padded_batch_size; ++i) { req_h[i]=0; kvh_h[i]=0; kvt_h[i]=0; }

  std::copy(oind_vec.begin(), oind_vec.end(), ind_h);
  csz_h[0] = static_cast<IdType>(C_tokens);  // TOKENS per tile

  if (enable_cuda_graph && padded_batch_size > new_batch_size) {
    bool* mask_h = GetPtrFromBaseOffset<bool>(page_locked_int_buffer, plan_info.block_valid_mask_offset);
    for (uint32_t i = 0; i < padded_batch_size; ++i) mask_h[i] = (i < new_batch_size);
  }

  const size_t bytes = int_alloc.num_allocated_bytes();
  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(int_buffer, page_locked_int_buffer, bytes,
                                       cudaMemcpyHostToDevice, stream));
  return cudaSuccess;
}


template <typename T>
__device__ __forceinline__ T ceil_div_u32(T a, T b) { return (a + b - 1) / b; }

template <typename Params>
__global__ void BuildHeadPackedPlanOneKernel(
    const __grid_constant__ Params params,
    uint32_t grid_limit,
    uint32_t cap_schedule,
    uint32_t cap_tiles,
    uint32_t* __restrict__ request_indices,
    uint32_t* __restrict__ kv_head_indices,
    uint32_t* __restrict__ kv_tile_indices,
    uint32_t* __restrict__ merge_start_offsets,
    uint32_t* __restrict__ o_indptr,
    uint32_t* __restrict__ new_batch_size,
    uint32_t* __restrict__ nnz_total) {

  // We intentionally run this with <<<1,1>>> — single CTA, single thread.
  // For tiny N·H (your bottleneck case), this is faster than orchestrating scans/reductions.
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  using IdType = typename Params::IdType;

  const uint32_t N      = params.paged_kv.batch_size;    // requests
  const uint32_t H_kv   = params.paged_kv.num_heads;     // KV heads
  const uint32_t H_qo   = params.num_qo_heads;           // Q/O heads
  const uint32_t GROUP  = H_kv ? (H_qo / H_kv) : 0;      // GQA group size (assumes divisible)
  const uint32_t PAGE   = params.paged_kv.page_size;     // usually 1
  const IdType*  indptr = params.paged_kv.indptr;        // [N+1]
  const uint32_t* attn_start_pos = params.attn_start_pos;// [N*H_qo] (tokens)

  // ---- Precompute P_pages[i] and max ----
  uint32_t Pmax_pages = 0;
  // Also cache P_pages & J_i for the chosen C later
  // We’ll recompute J_i after we finalize C to keep code simple (N is tiny).
  for (uint32_t i = 0; i < N; ++i) {
    const uint32_t P_pages = static_cast<uint32_t>(indptr[i+1] - indptr[i]);
    if (P_pages > Pmax_pages) Pmax_pages = P_pages;
  }
  if (Pmax_pages == 0) {
    // Empty case: nothing to do
    if (o_indptr) {
      o_indptr[0] = 0;
      for (uint32_t i = 0; i < N; ++i) o_indptr[i+1] = 0;
    }
    if (new_batch_size) *new_batch_size = 0;
    if (nnz_total)      *nnz_total      = 0;
    if (params.kv_chunk_size_ptr) params.kv_chunk_size_ptr[0] = 0; // tokens
    return;
  }

  // ---- Pick min_C_pages (host-independent) ----
  // Same lower-bound heuristic you used: at least 128 tokens worth of pages.
  const uint32_t min_C_pages = max(1u, ceil_div_u32(128u, PAGE));

  // ---- Tiny on-device search for C_pages (doubling ladder) ----
  uint32_t C = min_C_pages;
  const uint32_t C_max = max(1u, Pmax_pages);  // J_i ≥ 1 always
  for (;;) {
    // Compute X(C) and nnz_total(C)
    uint64_t X = 0;       // total CTAs (rows in head-packed schedule)
    uint64_t NNZ = 0;     // Σ_i J_i (rows of temporaries)
    for (uint32_t i = 0; i < N; ++i) {
      const uint32_t P_pages = static_cast<uint32_t>(indptr[i+1] - indptr[i]);
      const uint32_t J = ceil_div_u32(P_pages, C);
      NNZ += J;

      // per-KV-head start_page = group-min across GROUP qo-heads
      for (uint32_t hk = 0; hk < H_kv; ++hk) {
        uint32_t min_sp = 0xFFFFFFFFu;
        const uint32_t base_hq = hk * GROUP;
        for (uint32_t g = 0; g < GROUP; ++g) {
          const uint32_t hq = base_hq + g;
          const uint32_t start_pos_tokens = attn_start_pos[i * H_qo + hq];
          const uint32_t sp = start_pos_tokens / PAGE;  // in pages
          min_sp = min(min_sp, sp);
        }
        const uint32_t t0 = min_sp / C;       // FLOOR
        if (J > t0) X += (J - t0);
      }
    }

    const bool fit_schedule = (X <= static_cast<uint64_t>(grid_limit)) && (X <= static_cast<uint64_t>(cap_schedule));
    const bool fit_tiles    = (cap_tiles == 0) || (NNZ <= static_cast<uint64_t>(cap_tiles));
    if (fit_schedule && fit_tiles) break;

    // increase C; stop escalating when reaching C_max
    const uint32_t C_prev = C;
    C = min(C << 1, C_max);
    if (C == C_prev) break; // cannot grow further
  }

  // program chunk size in tokens (C_pages * page_size)
  if (params.kv_chunk_size_ptr) {
    params.kv_chunk_size_ptr[0] = C * PAGE;  // tokens per KV chunk
  }

  // ---- Emit schedule, merge_start_offsets and o_indptr ----
  uint32_t X_final = 0;   // new_batch_size
  uint32_t nnz     = 0;   // Σ J_i
  if (o_indptr) o_indptr[0] = 0;

  for (uint32_t i = 0; i < N; ++i) {
    const uint32_t P_pages = static_cast<uint32_t>(indptr[i+1] - indptr[i]);
    const uint32_t J = ceil_div_u32(P_pages, C);
    nnz += J;
    if (o_indptr) o_indptr[i+1] = nnz;

    // per-QO-head merge start offsets: floor(attn_start_pos / (C*PAGE)), clamp to J
    for (uint32_t hq = 0; hq < H_qo; ++hq) {
      const uint32_t start_pos_tokens = attn_start_pos[i * H_qo + hq];
      uint32_t ofs = start_pos_tokens / (C * PAGE);
      if (ofs > J) ofs = J;
      merge_start_offsets[i * H_qo + hq] = ofs;
    }

    // per-KV-head emission
    // First compute total rows for this request to avoid bounds checks
    uint32_t Tsum_i = 0;
    // (we compute and emit in the same pass to keep it simple; X_final grows monotonically)
    for (uint32_t hk = 0; hk < H_kv; ++hk) {
      uint32_t min_sp = 0xFFFFFFFFu;
      const uint32_t base_hq = hk * GROUP;
      for (uint32_t g = 0; g < GROUP; ++g) {
        const uint32_t hq = base_hq + g;
        const uint32_t start_pos_tokens = attn_start_pos[i * H_qo + hq];
        const uint32_t sp = start_pos_tokens / PAGE;
        min_sp = min(min_sp, sp);
      }
      const uint32_t t0 = min_sp / C;
      const uint32_t t  = (J > t0) ? (J - t0) : 0u;
      Tsum_i += t;

      // emit rows for this head
      for (uint32_t k = 0; k < t; ++k) {
        const uint32_t row = X_final + k;  // contiguous for this head within request i
        request_indices[row]  = i;
        kv_head_indices[row]  = hk;
        kv_tile_indices[row]  = t0 + k;    // absolute tile id (j in [0..J-1])
      }
      X_final += t; // advance after writing this head's rows
    }
  }

  if (new_batch_size) *new_batch_size = X_final;
  if (nnz_total)      *nnz_total      = nnz;
}


template <typename IdType>
inline auto PrefillSplitQOKVIndptr(IdType* qo_indptr_h, IdType* kv_indptr_h,
                                   uint32_t total_num_rows, uint32_t batch_size,
                                   uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t head_dim,
                                   uint32_t page_size, uint32_t max_batch_size_if_split,
                                   bool enable_cuda_graph) {
  std::vector<IdType> request_indices, qo_tile_indices, kv_tile_indices, merge_indptr, o_indptr;
  merge_indptr.push_back(0);
  o_indptr.push_back(0);

  const uint32_t gqa_group_size = num_qo_heads / num_kv_heads;

  // step 1: determine packed_qo_len_arr and verify qo_indptr contents.
  std::vector<int64_t> packed_qo_len_arr(batch_size), kv_len_arr(batch_size);
  for (uint32_t i = 0; i < batch_size; ++i) {
    packed_qo_len_arr[i] = int64_t(qo_indptr_h[i + 1] - qo_indptr_h[i]) * int64_t(gqa_group_size);
    if (packed_qo_len_arr[i] < 0) {
      std::ostringstream err_msg;
      err_msg << "qo_indptr[" << i + 1 << "]" << qo_indptr_h[i + 1] << " - qo_indptr[" << i << "]"
              << qo_indptr_h[i] << " should be non-negative";
      FLASHINFER_ERROR(err_msg.str());
    }
    kv_len_arr[i] = int64_t(kv_indptr_h[i + 1] - kv_indptr_h[i]);
    if (kv_len_arr[i] < 0) {
      std::ostringstream err_msg;
      err_msg << "kv_indptr[" << i + 1 << "]" << kv_indptr_h[i + 1] << " - kv_indptr[" << i << "]"
              << kv_indptr_h[i] << " should be non-negative";
      FLASHINFER_ERROR(err_msg.str());
    }
  }

  // step 2: determine cta_tile_q, kv_chunk_size and total_num_tiles_q
  const uint32_t min_kv_chunk_size = std::max((128 / page_size), 1U);
  uint32_t cta_tile_q;
  uint32_t total_num_tiles_q;
  if (enable_cuda_graph) {
    // When CUDA graphs are enabled, the lengths of sequences determined by
    // qo_indptr_h can vary. We assume that the dummy data based on which
    // the CUDA graph is created fixes the maximum number of tokens.
    const uint64_t max_seq_len = total_num_rows - batch_size + 1;
    uint64_t max_qo_len = uint64_t(max_seq_len) * gqa_group_size;
    cta_tile_q = FA2DetermineCtaTileQ(max_qo_len, head_dim);

    // Find an upper bound for the number of tiles, derived from the total
    // number of rows and the batch size.  The sum of qo lengths rounded
    // up to cta_tile_q will not exceed this number derived from the total
    // number of rows.
    total_num_tiles_q = ceil_div(total_num_rows * gqa_group_size, cta_tile_q) + batch_size - 1;
  } else {
    int64_t sum_packed_qo_len = 0;
    for (uint32_t i = 0; i < batch_size; ++i) {
      sum_packed_qo_len += packed_qo_len_arr[i];
    }
    const int64_t avg_packed_qo_len = sum_packed_qo_len / batch_size;
    cta_tile_q = FA2DetermineCtaTileQ(avg_packed_qo_len, head_dim);

    total_num_tiles_q = 0;
    for (uint32_t i = 0; i < batch_size; ++i) {
      total_num_tiles_q += ceil_div(packed_qo_len_arr[i], cta_tile_q);
    }
  }

  auto [split_kv, kv_chunk_size] =
      PrefillBinarySearchKVChunkSize(enable_cuda_graph, max_batch_size_if_split, packed_qo_len_arr,
                                     kv_len_arr, cta_tile_q, min_kv_chunk_size);

  // step 3: split qo_indptr and kv_indptr
  uint32_t new_batch_size = 0;
  for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    const int64_t packed_qo_len = packed_qo_len_arr[request_idx];
    const int64_t kv_len = std::max(int(kv_len_arr[request_idx]), 1);
    const int64_t num_tiles_q = ceil_div(packed_qo_len, cta_tile_q);
    const int64_t num_tiles_kv = ceil_div(kv_len, kv_chunk_size);

    for (uint32_t q_tile_idx = 0; q_tile_idx < num_tiles_q; ++q_tile_idx) {
      for (uint32_t kv_tile_idx = 0; kv_tile_idx < num_tiles_kv; ++kv_tile_idx) {
        new_batch_size += 1;
        request_indices.push_back(request_idx);
        qo_tile_indices.push_back(q_tile_idx);
        kv_tile_indices.push_back(kv_tile_idx);
      }
    }

    int64_t qo_len = packed_qo_len / gqa_group_size;
    for (uint32_t row = 0; row < qo_len; ++row) {
      merge_indptr.push_back(merge_indptr.back() + num_tiles_kv);
    }
    o_indptr.push_back(o_indptr.back() + qo_len * num_tiles_kv);
  }

  const size_t padded_batch_size =
      enable_cuda_graph ? std::max(max_batch_size_if_split, total_num_tiles_q) : new_batch_size;
  FLASHINFER_CHECK(new_batch_size <= padded_batch_size,
                   "new batch size should not exceed padded batch size");

  // step 4: multiply kv_chunk_size by page_size
  kv_chunk_size *= page_size;

  return std::make_tuple(split_kv, new_batch_size, padded_batch_size, cta_tile_q, kv_chunk_size,
                         std::move(request_indices), std::move(qo_tile_indices),
                         std::move(kv_tile_indices), std::move(merge_indptr), std::move(o_indptr));
}

struct PrefillPlanInfo {
  int64_t padded_batch_size;
  int64_t total_num_rows;
  int64_t total_num_rows_offset;
  int64_t cta_tile_q;
  int64_t request_indices_offset;
  int64_t qo_tile_indices_offset;
  int64_t kv_tile_indices_offset;
  int64_t merge_indptr_offset;
  int64_t o_indptr_offset;
  int64_t kv_chunk_size_ptr_offset;
  int64_t v_offset;
  int64_t s_offset;
  int64_t block_valid_mask_offset;
  bool enable_cuda_graph;
  bool split_kv;

  PrefillPlanInfo()
      : padded_batch_size(0),
        total_num_rows(0),
        total_num_rows_offset(0),
        cta_tile_q(0),
        request_indices_offset(0),
        qo_tile_indices_offset(0),
        kv_tile_indices_offset(0),
        merge_indptr_offset(0),
        o_indptr_offset(0),
        kv_chunk_size_ptr_offset(0),
        v_offset(0),
        s_offset(0),
        block_valid_mask_offset(0),
        enable_cuda_graph(false),
        split_kv(false) {}

  // convert PrefillPlanInfo to std::vector<int64_t>
  std::vector<int64_t> ToVector() const {
    return {padded_batch_size,
            total_num_rows,
            total_num_rows_offset,
            cta_tile_q,
            request_indices_offset,
            qo_tile_indices_offset,
            kv_tile_indices_offset,
            merge_indptr_offset,
            o_indptr_offset,
            kv_chunk_size_ptr_offset,
            v_offset,
            s_offset,
            block_valid_mask_offset,
            enable_cuda_graph,
            split_kv};
  }

  // From std::vector<int64_t> to PrefillPlanInfo
  void FromVector(const std::vector<int64_t>& vec) {
    if (vec.size() != 15) {
      std::ostringstream err_msg;
      err_msg << "PrefillPlanInfo::FromVector: vec.size() should be 15, but got " << vec.size();
      FLASHINFER_ERROR(err_msg.str());
    }
    padded_batch_size = vec[0];
    total_num_rows = vec[1];
    total_num_rows_offset = vec[2];
    cta_tile_q = vec[3];
    request_indices_offset = vec[4];
    qo_tile_indices_offset = vec[5];
    kv_tile_indices_offset = vec[6];
    merge_indptr_offset = vec[7];
    o_indptr_offset = vec[8];
    kv_chunk_size_ptr_offset = vec[9];
    v_offset = vec[10];
    s_offset = vec[11];
    block_valid_mask_offset = vec[12];
    enable_cuda_graph = vec[13];
    split_kv = vec[14];
  }
};

template <typename IdType>
inline cudaError_t PrefillPlan(void* float_buffer, size_t float_workspace_size_in_bytes,
                               void* int_buffer, void* page_locked_int_buffer,
                               size_t int_workspace_size_in_bytes, PrefillPlanInfo& plan_info,
                               IdType* qo_indptr_h, IdType* kv_indptr_h, uint32_t total_num_rows,
                               uint32_t batch_size, uint32_t num_qo_heads, uint32_t num_kv_heads,
                               uint32_t head_dim_qk, uint32_t head_dim_vo, uint32_t page_size,
                               bool enable_cuda_graph, uint32_t sizeof_dtype_o,
                               cudaStream_t stream) {
  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads " << num_qo_heads << " should be divisible by num_kv_heads "
            << num_kv_heads;
    FLASHINFER_ERROR(err_msg.str());
  }

  // step 0: get the number of SMs
  int num_sm = 0;
  int dev_id = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
  int num_blocks_per_sm = 2;
  int max_grid_size = num_blocks_per_sm * num_sm;
  uint32_t max_batch_size_if_split = max_grid_size / num_kv_heads;

  // step 2: determine kv_chunk_size
  auto [split_kv, new_batch_size, padded_batch_size, cta_tile_q, kv_chunk_size, request_indices_vec,
        qo_tile_indices_vec, kv_tile_indices_vec, merge_indptr_vec, o_indptr_vec] =
      PrefillSplitQOKVIndptr(qo_indptr_h, kv_indptr_h, total_num_rows, batch_size, num_qo_heads,
                             num_kv_heads, head_dim_vo, page_size, max_batch_size_if_split,
                             enable_cuda_graph);

  plan_info.cta_tile_q = cta_tile_q;
  plan_info.total_num_rows = total_num_rows;
  plan_info.enable_cuda_graph = enable_cuda_graph;
  plan_info.padded_batch_size = padded_batch_size;
  plan_info.split_kv = split_kv;

  AlignedAllocator int_allocator(int_buffer, int_workspace_size_in_bytes);
  plan_info.request_indices_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * padded_batch_size, 16, "batch_prefill_request_indices");
  plan_info.qo_tile_indices_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * padded_batch_size, 16, "batch_prefill_qo_tile_indices");
  plan_info.kv_tile_indices_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * padded_batch_size, 16, "batch_prefill_kv_tile_indices");
  plan_info.o_indptr_offset = int_allocator.aligned_alloc_offset(sizeof(IdType) * (batch_size + 1),
                                                                 16, "batch_prefill_o_indptr");
  plan_info.kv_chunk_size_ptr_offset =
      int_allocator.aligned_alloc_offset(sizeof(IdType), 1, "batch_prefill_kv_chunk_size_ptr");

  if (plan_info.enable_cuda_graph) {
    plan_info.total_num_rows_offset =
        int_allocator.aligned_alloc_offset(sizeof(uint32_t), 16, "batch_prefill_total_num_rows");
    uint32_t* total_num_rows_h =
        GetPtrFromBaseOffset<uint32_t>(page_locked_int_buffer, plan_info.total_num_rows_offset);
    *total_num_rows_h = qo_indptr_h[batch_size];
  }

  IdType* request_indices_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.request_indices_offset);
  IdType* qo_tile_indices_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.qo_tile_indices_offset);
  IdType* kv_tile_indices_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_tile_indices_offset);
  IdType* o_indptr_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.o_indptr_offset);
  IdType* kv_chunk_size_ptr_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_chunk_size_ptr_offset);
  std::copy(request_indices_vec.begin(), request_indices_vec.end(), request_indices_h);
  std::copy(qo_tile_indices_vec.begin(), qo_tile_indices_vec.end(), qo_tile_indices_h);
  std::copy(kv_tile_indices_vec.begin(), kv_tile_indices_vec.end(), kv_tile_indices_h);
  std::copy(o_indptr_vec.begin(), o_indptr_vec.end(), o_indptr_h);
  kv_chunk_size_ptr_h[0] = kv_chunk_size;

  if (split_kv) {
    AlignedAllocator float_allocator(float_buffer, float_workspace_size_in_bytes);
    plan_info.v_offset = float_allocator.aligned_alloc_offset(
        num_qo_heads * padded_batch_size * cta_tile_q * head_dim_vo * sizeof(float), 16,
        "batch_prefill_tmp_v");
    plan_info.s_offset = float_allocator.aligned_alloc_offset(
        num_qo_heads * padded_batch_size * cta_tile_q * sizeof(float), 16, "batch_prefill_tmp_s");
    plan_info.merge_indptr_offset = int_allocator.aligned_alloc_offset(
        sizeof(IdType) * (plan_info.total_num_rows + 1), 16, "batch_prefill_merge_indptr");
    plan_info.block_valid_mask_offset = int_allocator.aligned_alloc_offset(
        sizeof(bool) * padded_batch_size, 16, "batch_prefill_block_valid_mask");

    IdType* merge_indptr_h =
        GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.merge_indptr_offset);
    bool* block_valid_mask_h =
        GetPtrFromBaseOffset<bool>(page_locked_int_buffer, plan_info.block_valid_mask_offset);
    std::copy(merge_indptr_vec.begin(), merge_indptr_vec.end(), merge_indptr_h);
    for (uint32_t i = 0; i < padded_batch_size; ++i) {
      block_valid_mask_h[i] = i < new_batch_size;
    }
  }

  size_t num_bytes_to_copy = int_allocator.num_allocated_bytes();
  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(int_buffer, page_locked_int_buffer, num_bytes_to_copy,
                                       cudaMemcpyHostToDevice, stream));

  return cudaSuccess;
}

inline float cost_function(int qo_len, int kv_len) { return 2 * float(qo_len) + kv_len; }

template <typename T>
std::vector<T> flatten(const std::vector<std::vector<T>>& vec, int size_after_flatten) {
  std::vector<T> result;
  result.reserve(size_after_flatten);
  for (const auto& inner_vec : vec) {
    result.insert(result.end(), inner_vec.begin(), inner_vec.end());
  }
  return result;
}

inline int packed_causal_kv_end(int qo_len, int kv_len, int qo_tile_idx, int cluster_tile_q,
                                int num_qo_tiles, int group_size) {
  if (qo_tile_idx + 1 == num_qo_tiles) {
    return kv_len;
  }
  int kv_len_init = kv_len - qo_len;  // right aligned
  return min(kv_len_init + ceil_div((qo_tile_idx + 1) * cluster_tile_q, group_size), kv_len);
}

struct PrefillPlanSM90Info {
  int64_t qo_tile_indices_offset;
  int64_t qo_indptr_offset;
  int64_t kv_indptr_offset;
  int64_t qo_len_offset;
  int64_t kv_len_offset;
  int64_t head_indices_offset;
  int64_t work_indptr_offset;
  int64_t batch_indices_offset;
  bool same_schedule_for_all_heads;

  PrefillPlanSM90Info()
      : qo_tile_indices_offset(0),
        qo_indptr_offset(0),
        kv_indptr_offset(0),
        qo_len_offset(0),
        kv_len_offset(0),
        head_indices_offset(0),
        work_indptr_offset(0),
        batch_indices_offset(0),
        same_schedule_for_all_heads(false) {}

  // convert PrefillPlanSM90Info to std::vector<int64_t>
  std::vector<int64_t> ToVector() const {
    return {qo_tile_indices_offset, qo_indptr_offset,     kv_indptr_offset,
            qo_len_offset,          kv_len_offset,        head_indices_offset,
            work_indptr_offset,     batch_indices_offset, same_schedule_for_all_heads};
  }

  // From std::vector<int64_t> to PrefillPlanSM90Info
  void FromVector(const std::vector<int64_t>& vec) {
    if (vec.size() != 9) {
      std::ostringstream err_msg;
      err_msg << "PrefillPlanSM90Info::FromVector: vec.size() should be 9, but got " << vec.size();
      FLASHINFER_ERROR(err_msg.str());
    }
    qo_tile_indices_offset = vec[0];
    qo_indptr_offset = vec[1];
    kv_indptr_offset = vec[2];
    qo_len_offset = vec[3];
    kv_len_offset = vec[4];
    head_indices_offset = vec[5];
    work_indptr_offset = vec[6];
    batch_indices_offset = vec[7];
    same_schedule_for_all_heads = vec[8];
  }
};

template <typename IdType>
inline cudaError_t PrefillSM90Plan(
    void* float_buffer, size_t float_workspace_size_in_bytes, void* int_buffer,
    void* page_locked_int_buffer, size_t int_workspace_size_in_bytes,
    PrefillPlanSM90Info& plan_info, IdType* qo_indptr_h, IdType* kv_indptr_h, IdType* kv_len_arr_h,
    uint32_t total_num_rows, uint32_t batch_size, uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t head_dim_qk, uint32_t head_dim_vo, uint32_t page_size, bool causal,
    bool enable_cuda_graph, uint32_t sizeof_dtype_o, cudaStream_t stream) {
  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads " << num_qo_heads << " should be divisible by num_kv_heads "
            << num_kv_heads;
    FLASHINFER_ERROR(err_msg.str());
  }

  std::vector<std::tuple<int, int, int>> idx_qo_kv_len_vec;
  for (uint32_t i = 0; i < batch_size; ++i) {
    int qo_len = qo_indptr_h[i + 1] - qo_indptr_h[i];
    int kv_len = kv_len_arr_h[i];
    if (kv_len < 0) {
      std::ostringstream err_msg;
      err_msg << "kv_len[" << i << "]" << kv_len << " should be non-negative";
      FLASHINFER_ERROR(err_msg.str());
    }
    if (qo_len < 0) {
      std::ostringstream err_msg;
      err_msg << "qo_indptr[" << i + 1 << "]" << qo_indptr_h[i + 1] << " - qo_indptr[" << i << "]"
              << qo_indptr_h[i] << " should be non-negative";
      FLASHINFER_ERROR(err_msg.str());
    }
    idx_qo_kv_len_vec.push_back({i, qo_len, kv_len});
  }

  std::sort(idx_qo_kv_len_vec.begin(), idx_qo_kv_len_vec.end(),
            [](const auto& a, const auto& b) { return std::get<2>(a) > std::get<2>(b); });
  int cta_tile_q = 128;
  if (head_dim_vo == 64) {
    cta_tile_q = 192;
  }

  int device = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&device));
  int num_sm90_ctas = 0;
  FLASHINFER_CUDA_CALL(
      cudaDeviceGetAttribute(&num_sm90_ctas, cudaDevAttrMultiProcessorCount, device));

  MinHeap cta_cost_heap(num_sm90_ctas);
  std::vector<std::vector<IdType>> cta_qo_tile_indices(num_sm90_ctas, std::vector<IdType>()),
      cta_qo_indptr(num_sm90_ctas, std::vector<IdType>()),
      cta_kv_indptr(num_sm90_ctas, std::vector<IdType>()),
      cta_qo_len(num_sm90_ctas, std::vector<IdType>()),
      cta_kv_len(num_sm90_ctas, std::vector<IdType>()),
      cta_head_indices(num_sm90_ctas, std::vector<IdType>()),
      cta_batch_indices(num_sm90_ctas, std::vector<IdType>());

  int max_num_works_per_head = ceil_div(total_num_rows, cta_tile_q) + batch_size - 1;
  plan_info.same_schedule_for_all_heads = max_num_works_per_head > 4096;

  for (int qo_head_idx = 0;
       qo_head_idx < (plan_info.same_schedule_for_all_heads ? 1 : num_qo_heads); ++qo_head_idx) {
    for (auto& [i, qo_len, kv_len] : idx_qo_kv_len_vec) {
      int num_qo_tiles = ceil_div(qo_len, cta_tile_q);
      for (int qo_tile_idx = num_qo_tiles - 1; qo_tile_idx >= 0; --qo_tile_idx) {
        auto [cta_idx, accum_cost] = cta_cost_heap.pop();
        // NOTE(Zihao): our current FA3 implementation do not fuse query and group heads
        // so the group_size in cost_function is always 1
        int effective_kv_len =
            causal ? packed_causal_kv_end(qo_len, kv_len, qo_tile_idx, cta_tile_q, num_qo_tiles, 1)
                   : kv_len;
        cta_cost_heap.insert({cta_idx, accum_cost + cost_function(cta_tile_q, effective_kv_len)});
        cta_qo_tile_indices[cta_idx].push_back(qo_tile_idx);
        cta_qo_indptr[cta_idx].push_back(qo_indptr_h[i]);
        cta_qo_len[cta_idx].push_back(qo_len);
        cta_kv_indptr[cta_idx].push_back(kv_indptr_h[i]);
        cta_kv_len[cta_idx].push_back(kv_len);
        cta_head_indices[cta_idx].push_back(qo_head_idx);
        cta_batch_indices[cta_idx].push_back(i);
      }
    }
  }

  std::vector<IdType> work_indptr_vec(num_sm90_ctas + 1, 0);
  for (uint32_t i = 0; i < num_sm90_ctas; ++i) {
    work_indptr_vec[i + 1] = work_indptr_vec[i] + cta_qo_tile_indices[i].size();
  }
  int total_num_works = work_indptr_vec.back();
  auto qo_tile_indices_vec = flatten(cta_qo_tile_indices, total_num_works);
  auto qo_indptr_vec = flatten(cta_qo_indptr, total_num_works);
  auto kv_indptr_vec = flatten(cta_kv_indptr, total_num_works);
  auto qo_len_vec = flatten(cta_qo_len, total_num_works);
  auto kv_len_vec = flatten(cta_kv_len, total_num_works);
  auto head_indices_vec = flatten(cta_head_indices, total_num_works);
  auto batch_indices_vec = flatten(cta_batch_indices, total_num_works);

  AlignedAllocator int_allocator(int_buffer, int_workspace_size_in_bytes);
  int max_total_num_works;

  if (enable_cuda_graph) {
    max_total_num_works = plan_info.same_schedule_for_all_heads
                              ? max_num_works_per_head
                              : max_num_works_per_head * num_qo_heads;
  } else {
    max_total_num_works = total_num_works;
  }

  plan_info.qo_tile_indices_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * max_total_num_works, 16, "batch_prefill_sm90_qo_tile_indices");
  plan_info.qo_indptr_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * max_total_num_works, 16, "batch_prefill_sm90_qo_offset");
  plan_info.kv_indptr_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * max_total_num_works, 16, "batch_prefill_sm90_kv_offset");
  plan_info.qo_len_offset = int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works,
                                                               16, "batch_prefill_sm90_qo_len");
  plan_info.kv_len_offset = int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works,
                                                               16, "batch_prefill_sm90_kv_len");
  plan_info.head_indices_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * max_total_num_works, 16, "batch_prefill_sm90_head_indices");
  plan_info.work_indptr_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * (num_sm90_ctas + 1), 16, "batch_prefill_sm90_work_indptr");
  plan_info.batch_indices_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * max_total_num_works, 16, "batch_prefill_sm90_batch_indices");

  IdType* qo_tile_indices_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.qo_tile_indices_offset);
  IdType* qo_offset_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.qo_indptr_offset);
  IdType* kv_offset_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_indptr_offset);
  IdType* qo_len_h = GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.qo_len_offset);
  IdType* kv_len_h = GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_len_offset);
  IdType* head_indices_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.head_indices_offset);
  IdType* work_indptr_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.work_indptr_offset);
  IdType* batch_indices_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.batch_indices_offset);

  std::copy(qo_tile_indices_vec.begin(), qo_tile_indices_vec.end(), qo_tile_indices_h);
  std::copy(qo_indptr_vec.begin(), qo_indptr_vec.end(), qo_offset_h);
  std::copy(kv_indptr_vec.begin(), kv_indptr_vec.end(), kv_offset_h);
  std::copy(qo_len_vec.begin(), qo_len_vec.end(), qo_len_h);
  std::copy(kv_len_vec.begin(), kv_len_vec.end(), kv_len_h);
  std::copy(head_indices_vec.begin(), head_indices_vec.end(), head_indices_h);
  std::copy(work_indptr_vec.begin(), work_indptr_vec.end(), work_indptr_h);
  std::copy(batch_indices_vec.begin(), batch_indices_vec.end(), batch_indices_h);

  size_t num_bytes_to_copy = int_allocator.num_allocated_bytes();
  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(int_buffer, page_locked_int_buffer, num_bytes_to_copy,
                                       cudaMemcpyHostToDevice, stream));
  return cudaSuccess;
}

template <uint32_t NUM_TASKS>
struct HolisticPlanInfo {
  int64_t num_blks_x;
  int64_t num_blks_y;
  struct {
    int64_t q_indptr_offset;
    int64_t kv_indptr_offset;
    int64_t partial_indptr_offset;
    int64_t q_len_offset;
    int64_t kv_len_offset;
    int64_t q_start_offset;
    int64_t kv_start_offset;
    int64_t kv_end_offset;
    int64_t kv_head_idx_offset;
    int64_t work_indptr_offset;
    int64_t len_kv_chunk_offset;
  } tasks[NUM_TASKS];

  int64_t partial_o_offset;
  int64_t partial_lse_offset;
  int64_t merge_indptr_offset;
  int64_t merge_o_indices_offset;
  int64_t num_qo_len_offset;

  static constexpr uint32_t NUM_TASK_ARGS = 11;
  static constexpr uint32_t NUM_SHARED_ARGS = 7;

  std::vector<int64_t> ToVector() const {
    std::vector<int64_t> vec;
    vec.push_back(num_blks_x);
    vec.push_back(num_blks_y);
    for (uint32_t i = 0; i < NUM_TASKS; ++i) {
      vec.push_back(tasks[i].q_indptr_offset);
      vec.push_back(tasks[i].kv_indptr_offset);
      vec.push_back(tasks[i].partial_indptr_offset);
      vec.push_back(tasks[i].q_len_offset);
      vec.push_back(tasks[i].kv_len_offset);
      vec.push_back(tasks[i].q_start_offset);
      vec.push_back(tasks[i].kv_start_offset);
      vec.push_back(tasks[i].kv_end_offset);
      vec.push_back(tasks[i].kv_head_idx_offset);
      vec.push_back(tasks[i].work_indptr_offset);
      vec.push_back(tasks[i].len_kv_chunk_offset);
    }
    vec.push_back(partial_o_offset);
    vec.push_back(partial_lse_offset);
    vec.push_back(merge_indptr_offset);
    vec.push_back(merge_o_indices_offset);
    vec.push_back(num_qo_len_offset);
    return vec;
  }

  void FromVector(const std::vector<int64_t>& vec) {
    if (vec.size() != NUM_SHARED_ARGS + NUM_TASKS * NUM_TASK_ARGS) {
      std::ostringstream err_msg;
      err_msg << "HolisticPlanInfo::FromVector: vec.size() should be "
              << NUM_SHARED_ARGS + NUM_TASKS * NUM_TASK_ARGS << ", but got " << vec.size();
      FLASHINFER_ERROR(err_msg.str());
    }
    num_blks_x = vec[0];
    num_blks_y = vec[1];
    for (uint32_t i = 0; i < NUM_TASKS; ++i) {
      tasks[i].q_indptr_offset = vec[2 + i * NUM_TASK_ARGS + 0];
      tasks[i].kv_indptr_offset = vec[2 + i * NUM_TASK_ARGS + 1];
      tasks[i].partial_indptr_offset = vec[2 + i * NUM_TASK_ARGS + 2];
      tasks[i].q_len_offset = vec[2 + i * NUM_TASK_ARGS + 3];
      tasks[i].kv_len_offset = vec[2 + i * NUM_TASK_ARGS + 4];
      tasks[i].q_start_offset = vec[2 + i * NUM_TASK_ARGS + 5];
      tasks[i].kv_start_offset = vec[2 + i * NUM_TASK_ARGS + 6];
      tasks[i].kv_end_offset = vec[2 + i * NUM_TASK_ARGS + 7];
      tasks[i].kv_head_idx_offset = vec[2 + i * NUM_TASK_ARGS + 8];
      tasks[i].work_indptr_offset = vec[2 + i * NUM_TASK_ARGS + 9];
      tasks[i].len_kv_chunk_offset = vec[2 + i * NUM_TASK_ARGS + 10];
    }
    partial_o_offset = vec[2 + NUM_TASKS * NUM_TASK_ARGS];
    partial_lse_offset = vec[3 + NUM_TASKS * NUM_TASK_ARGS];
    merge_indptr_offset = vec[4 + NUM_TASKS * NUM_TASK_ARGS];
    merge_o_indices_offset = vec[5 + NUM_TASKS * NUM_TASK_ARGS];
    num_qo_len_offset = vec[6 + NUM_TASKS * NUM_TASK_ARGS];
  }
};

template <typename IdType>
inline cudaError_t TwoStageHolisticPlan(void* float_buffer, size_t float_workspace_size_in_bytes,
                                        void* int_buffer, void* page_locked_int_buffer,
                                        size_t int_workspace_size_in_bytes,
                                        HolisticPlanInfo<2>& plan_info, IdType* qo_indptr_h,
                                        IdType* kv_indptr_h, IdType* kv_len_arr_h,
                                        uint32_t batch_size, uint32_t num_qo_heads,
                                        uint32_t num_kv_heads, uint32_t head_dim, bool causal,
                                        cudaStream_t stream) {
  constexpr uint32_t NUM_TASKS = 2;
  const uint32_t CTA_TILE_Q_SIZES[NUM_TASKS] = {128, 16};
  int num_sm = 0;
  int dev_id = 0;

  uint32_t gqa_group_size = num_qo_heads / num_kv_heads;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));

  if (head_dim >= 256) {
    // NOTE (Yilong): optimize this code path
    // constraint gridDim due to cooperative group
    num_sm *= 1;
  } else {
    // NOTE(Zihao): two cta per sm
    num_sm *= 2;
  }

  // step 0. determine the number of blocks in x and y dimensions
  std::vector<std::tuple<int, int, int>> idx_qo_kv_len_vec[NUM_TASKS];
  for (uint32_t i = 0; i < batch_size; ++i) {
    if (qo_indptr_h[i + 1] - qo_indptr_h[i] < 0) {
      std::ostringstream err_msg;
      err_msg << "qo_indptr[" << i + 1 << "]" << qo_indptr_h[i + 1] << " - qo_indptr[" << i << "]"
              << qo_indptr_h[i] << " should be non-negative";
      FLASHINFER_ERROR(err_msg.str());
    }

    int qo_len = qo_indptr_h[i + 1] - qo_indptr_h[i];
    int packed_qo_len = qo_len * gqa_group_size;
    int kv_len = kv_len_arr_h[i];

    if (packed_qo_len > CTA_TILE_Q_SIZES[1]) {
      idx_qo_kv_len_vec[0].push_back({i, qo_len, kv_len});
    } else {
      idx_qo_kv_len_vec[1].push_back({i, qo_len, kv_len});
    }
  }

  int cluster_size = 1;
  int num_clusters = num_sm / cluster_size;
  plan_info.num_blks_x = cluster_size;
  plan_info.num_blks_y = num_clusters;

  auto f = [](int x) {
    if (x <= 128) {
      // This aligns with CTA_TILE_KV in persistent mainloop
      // NOTE (Yilong): Optimize here for smaller batch/seqlen scenarios
      return 128;
    }
    return ceil_div(x, 256) * 256;
  };

  MinHeap cluster_cost_heap(num_clusters);
  AlignedAllocator int_allocator(int_buffer, int_workspace_size_in_bytes);

  // NOTE(Zihao): adjust it later
  const int max_total_num_works = 16384;
  const int max_packed_qo_lens =
      4 * num_clusters * cluster_size * (CTA_TILE_Q_SIZES[0] + CTA_TILE_Q_SIZES[1]);

  // calculate kv_len_limit first, considering all workloads
  int64_t total_kv_lens = 0;
  for (uint32_t task = 0; task < NUM_TASKS; ++task) {
    int cluster_tile_q = CTA_TILE_Q_SIZES[task] * cluster_size;
    for (auto& [_, qo_len, kv_len] : idx_qo_kv_len_vec[task]) {
      int packed_qo_len = qo_len * gqa_group_size;
      int num_qo_tiles = ceil_div(packed_qo_len, cluster_tile_q);
      for (int qo_tile_idx = num_qo_tiles - 1; qo_tile_idx >= 0; --qo_tile_idx) {
        int effective_kv_len =
            causal ? packed_causal_kv_end(qo_len, kv_len, qo_tile_idx, cluster_tile_q, num_qo_tiles,
                                          gqa_group_size)
                   : kv_len;
        total_kv_lens += effective_kv_len;
      }
    }
  }

  // used for remapping the output offsets
  // layout [packed_qo_len x num_kv_tiels, num_kv_heads, head_dim]
  int partial_o_nnz = 0;
  std::vector<IdType> merge_indptr, merge_o_indices, num_expand_qo_len_vec;
  merge_indptr.push_back(partial_o_nnz);
  for (uint32_t task = 0; task < NUM_TASKS; ++task) {
    int cluster_tile_q = CTA_TILE_Q_SIZES[task] * cluster_size;
    int kv_len_limit = 0;
    if (cluster_tile_q >= 64) {
      // chunked-prefill workloads are much more expensive than decode
      // so we use a smaller kv_len_limit for chunked-prefill workloads
      kv_len_limit = f(std::max(ceil_div(total_kv_lens, num_clusters), 1L));
    } else {
      kv_len_limit = f(std::max(ceil_div(total_kv_lens * num_kv_heads, num_clusters), 1L));
    }

    std::vector<std::vector<IdType>> cluster_q_indptr(num_clusters, std::vector<IdType>()),
        cluster_kv_indptr(num_clusters, std::vector<IdType>()),
        cluster_q_len(num_clusters, std::vector<IdType>()),
        cluster_kv_len(num_clusters, std::vector<IdType>()),
        cluster_q_start(num_clusters, std::vector<IdType>()),
        cluster_kv_start(num_clusters, std::vector<IdType>()),
        cluster_kv_end(num_clusters, std::vector<IdType>()),
        cluster_kv_head_idx(num_clusters, std::vector<IdType>()),
        cluster_partial_indptr(num_clusters, std::vector<IdType>()),
        cluster_len_kv_chunk(num_clusters, std::vector<IdType>());

    for (auto& [i, qo_len, kv_len] : idx_qo_kv_len_vec[task]) {
      int packed_qo_len = qo_len * gqa_group_size;
      int num_qo_tiles = ceil_div(packed_qo_len, cluster_tile_q);
      // NOTE (Yilong): this ordering correspoinds to the layout of reduction kernel
      for (int qo_tile_idx = 0; qo_tile_idx < num_qo_tiles; ++qo_tile_idx) {
        int remaining_len = causal
                                ? packed_causal_kv_end(qo_len, kv_len, qo_tile_idx, cluster_tile_q,
                                                       num_qo_tiles, gqa_group_size)
                                : kv_len;
        int kv_start = 0;
        bool split_kv = remaining_len > kv_len_limit;
        int num_kv_tiles = split_kv ? ceil_div(remaining_len, kv_len_limit) : 1;
        int row_tile_size = std::min(cluster_tile_q, packed_qo_len - qo_tile_idx * cluster_tile_q);
        bool zero_kv_len = (remaining_len == 0);
        while (remaining_len > 0 || zero_kv_len) {
          int actual_len = std::min(remaining_len, kv_len_limit);
          for (uint32_t kv_head_idx = 0; kv_head_idx < num_kv_heads; ++kv_head_idx) {
            auto [cluster_idx, accum_cost] = cluster_cost_heap.pop();
            cluster_cost_heap.insert(
                {cluster_idx, accum_cost + cost_function(cluster_tile_q, actual_len)});
            cluster_q_len[cluster_idx].push_back(qo_len);
            cluster_kv_len[cluster_idx].push_back(kv_len);
            cluster_q_indptr[cluster_idx].push_back(qo_indptr_h[i]);
            cluster_kv_indptr[cluster_idx].push_back(kv_indptr_h[i]);

            // use kv_chunk to rematerize num_kv_tiles and kv_tile_idx
            cluster_len_kv_chunk[cluster_idx].push_back(kv_len_limit);
            cluster_partial_indptr[cluster_idx].push_back(partial_o_nnz);

            cluster_q_start[cluster_idx].push_back(qo_tile_idx * cluster_tile_q);
            cluster_kv_start[cluster_idx].push_back(kv_start);
            cluster_kv_end[cluster_idx].push_back(kv_start + actual_len);
            cluster_kv_head_idx[cluster_idx].push_back(kv_head_idx);
          }
          remaining_len -= actual_len;
          zero_kv_len = (remaining_len == 0);
          kv_start += actual_len;
          if (zero_kv_len) {
            break;
          }
        }
        if (split_kv) {
          // non-split kv is directly written through
          for (int row = 0; row < row_tile_size; ++row) {
            merge_indptr.push_back(merge_indptr.back() + num_kv_tiles);
            merge_o_indices.push_back(qo_indptr_h[i] +
                                      (qo_tile_idx * cluster_tile_q + row) / gqa_group_size);
          }
          partial_o_nnz += row_tile_size * num_kv_tiles;
        }
      }
    }

    std::vector<IdType> work_indptr_vec(num_clusters + 1, 0);
    for (uint32_t i = 0; i < num_clusters; ++i) {
      work_indptr_vec[i + 1] = work_indptr_vec[i] + cluster_q_indptr[i].size();
    }
    int total_num_works = work_indptr_vec.back();
    auto q_indptr_vec = flatten(cluster_q_indptr, total_num_works);
    auto kv_indptr_vec = flatten(cluster_kv_indptr, total_num_works);
    auto partial_indptr_vec = flatten(cluster_partial_indptr, total_num_works);
    auto q_len_vec = flatten(cluster_q_len, total_num_works);
    auto kv_len_vec = flatten(cluster_kv_len, total_num_works);
    auto q_start_vec = flatten(cluster_q_start, total_num_works);
    auto kv_start_vec = flatten(cluster_kv_start, total_num_works);
    auto kv_end_vec = flatten(cluster_kv_end, total_num_works);
    auto kv_head_idx_vec = flatten(cluster_kv_head_idx, total_num_works);
    auto len_kv_chunk_vec = flatten(cluster_len_kv_chunk, total_num_works);

    plan_info.tasks[task].q_indptr_offset =
        int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "q_indptr");
    plan_info.tasks[task].kv_indptr_offset =
        int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "kv_indptr");
    plan_info.tasks[task].partial_indptr_offset = int_allocator.aligned_alloc_offset(
        sizeof(IdType) * max_total_num_works, 16, "partial_indptr");
    plan_info.tasks[task].q_len_offset =
        int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "q_len");
    plan_info.tasks[task].kv_len_offset =
        int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "kv_len");
    plan_info.tasks[task].q_start_offset =
        int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "q_start");
    plan_info.tasks[task].kv_start_offset =
        int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "kv_start");
    plan_info.tasks[task].kv_end_offset =
        int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "kv_end");
    plan_info.tasks[task].kv_head_idx_offset =
        int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "kv_head_idx");
    plan_info.tasks[task].work_indptr_offset =
        int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "work_indptr");
    plan_info.tasks[task].len_kv_chunk_offset = int_allocator.aligned_alloc_offset(
        sizeof(IdType) * max_total_num_works, 16, "len_kv_chunk");

    CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.tasks[task].q_indptr_offset,
                           q_indptr_vec);
    CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.tasks[task].kv_indptr_offset,
                           kv_indptr_vec);
    CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.tasks[task].partial_indptr_offset,
                           partial_indptr_vec);
    CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.tasks[task].q_len_offset, q_len_vec);
    CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.tasks[task].kv_len_offset, kv_len_vec);
    CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.tasks[task].q_start_offset,
                           q_start_vec);
    CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.tasks[task].kv_start_offset,
                           kv_start_vec);
    CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.tasks[task].kv_end_offset, kv_end_vec);
    CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.tasks[task].kv_head_idx_offset,
                           kv_head_idx_vec);
    CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.tasks[task].work_indptr_offset,
                           work_indptr_vec);
    CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.tasks[task].len_kv_chunk_offset,
                           len_kv_chunk_vec);
  }

  if (partial_o_nnz > max_packed_qo_lens) {
    std::ostringstream err_msg;
    err_msg << "partial_o_nnz " << partial_o_nnz << " exceeds max_packed_qo_lens "
            << max_packed_qo_lens;
    FLASHINFER_ERROR(err_msg.str());
  }

  // update num_qo_len_vec
  num_expand_qo_len_vec.push_back(merge_indptr.size() - 1);
  // allocate buffer for state merge function
  plan_info.merge_indptr_offset =
      int_allocator.aligned_alloc_offset(sizeof(IdType) * max_packed_qo_lens, 16, "merge_indptr");
  plan_info.merge_o_indices_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * max_packed_qo_lens, 16, "merge_o_indices");
  plan_info.num_qo_len_offset =
      int_allocator.aligned_alloc_offset(sizeof(IdType), 16, "num_qo_len_offset");
  // copy data to paged cpu buffer
  CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.merge_indptr_offset, merge_indptr);
  CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.merge_o_indices_offset, merge_o_indices);
  CopyToPageLockedBuffer(page_locked_int_buffer, plan_info.num_qo_len_offset,
                         num_expand_qo_len_vec);

  size_t num_bytes_to_copy = int_allocator.num_allocated_bytes();
  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(int_buffer, page_locked_int_buffer, num_bytes_to_copy,
                                       cudaMemcpyHostToDevice, stream));
  constexpr size_t sizeof_dtype_o = 2;  // NOTE (Yilong): assume fp16

  // Note(Yilong): adjust it later
  AlignedAllocator float_allocator(float_buffer, float_workspace_size_in_bytes);
  plan_info.partial_o_offset = float_allocator.aligned_alloc_offset(
      2 * max_packed_qo_lens * sizeof_dtype_o * head_dim, 16, "holistic_partial_o");
  plan_info.partial_lse_offset = float_allocator.aligned_alloc_offset(
      2 * max_packed_qo_lens * sizeof(float), 16, "holistic_partial_lse");

  return cudaSuccess;
}

struct MLAPlanInfo {
  int64_t num_blks_x;
  int64_t num_blks_y;
  int64_t q_indptr_offset;
  int64_t kv_indptr_offset;
  int64_t partial_indptr_offset;
  int64_t merge_packed_offset_start_offset;
  int64_t merge_packed_offset_end_offset;
  int64_t merge_partial_packed_offset_start_offset;
  int64_t merge_partial_packed_offset_end_offset;
  int64_t merge_partial_stride_offset;
  int64_t q_len_offset;
  int64_t kv_len_offset;
  int64_t q_start_offset;
  int64_t kv_start_offset;
  int64_t kv_end_offset;
  int64_t work_indptr_offset;
  int64_t partial_o_offset;
  int64_t partial_lse_offset;

  std::vector<int64_t> ToVector() const {
    return {num_blks_x,
            num_blks_y,
            q_indptr_offset,
            kv_indptr_offset,
            partial_indptr_offset,
            merge_packed_offset_start_offset,
            merge_packed_offset_end_offset,
            merge_partial_packed_offset_start_offset,
            merge_partial_packed_offset_end_offset,
            merge_partial_stride_offset,
            q_len_offset,
            kv_len_offset,
            q_start_offset,
            kv_start_offset,
            kv_end_offset,
            work_indptr_offset,
            partial_o_offset,
            partial_lse_offset};
  }

  void FromVector(const std::vector<int64_t>& vec) {
    if (vec.size() != 18) {
      std::ostringstream err_msg;
      err_msg << "MLAPlanInfo::FromVector: vec.size() should be 18, but got " << vec.size();
      FLASHINFER_ERROR(err_msg.str());
    }
    num_blks_x = vec[0];
    num_blks_y = vec[1];
    q_indptr_offset = vec[2];
    kv_indptr_offset = vec[3];
    partial_indptr_offset = vec[4];
    merge_packed_offset_start_offset = vec[5];
    merge_packed_offset_end_offset = vec[6];
    merge_partial_packed_offset_start_offset = vec[7];
    merge_partial_packed_offset_end_offset = vec[8];
    merge_partial_stride_offset = vec[9];
    q_len_offset = vec[10];
    kv_len_offset = vec[11];
    q_start_offset = vec[12];
    kv_start_offset = vec[13];
    kv_end_offset = vec[14];
    work_indptr_offset = vec[15];
    partial_o_offset = vec[16];
    partial_lse_offset = vec[17];
  }
};

template <typename IdType>
inline cudaError_t MLAPlan(void* float_buffer, size_t float_workspace_size_in_bytes,
                           void* int_buffer, void* page_locked_int_buffer,
                           size_t int_workspace_size_in_bytes, MLAPlanInfo& plan_info,
                           IdType* qo_indptr_h, IdType* kv_indptr_h, IdType* kv_len_arr_h,
                           uint32_t batch_size, uint32_t num_heads, uint32_t head_dim_o,
                           bool causal, cudaStream_t stream) {
  int num_sm = 0;
  int dev_id = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));

  // step 0. determine the number of blocks in x and y dimensions
  int accum_packed_qo_len = 0;
  std::vector<std::tuple<int, int, int>> idx_qo_kv_len_vec;
  for (uint32_t i = 0; i < batch_size; ++i) {
    if (qo_indptr_h[i + 1] - qo_indptr_h[i] < 0) {
      std::ostringstream err_msg;
      err_msg << "qo_indptr[" << i + 1 << "]" << qo_indptr_h[i + 1] << " - qo_indptr[" << i << "]"
              << qo_indptr_h[i] << " should be non-negative";
      FLASHINFER_ERROR(err_msg.str());
    }

    int qo_len = qo_indptr_h[i + 1] - qo_indptr_h[i];
    int packed_qo_len = qo_len * num_heads;
    accum_packed_qo_len += packed_qo_len;

    int kv_len = kv_len_arr_h[i];
    idx_qo_kv_len_vec.push_back({i, qo_len, kv_len});
  }
  int avg_packed_qo_len = accum_packed_qo_len / batch_size;

  int cluster_size;
  if (avg_packed_qo_len > 64) {
    cluster_size = 2;  // two ctas in a cluster
  } else {
    cluster_size = 1;  // one cta in a cluster
  }
  uint32_t num_clusters = num_sm / cluster_size;
  plan_info.num_blks_x = cluster_size;
  plan_info.num_blks_y = num_clusters;
  const int cta_tile_q = 64;
  int cluster_tile_q = cluster_size * cta_tile_q;

  int64_t total_kv_lens = 0;
  for (auto& [_, qo_len, kv_len] : idx_qo_kv_len_vec) {
    int packed_qo_len = qo_len * num_heads;
    int num_qo_tiles = ceil_div(packed_qo_len, cluster_tile_q);
    for (int qo_tile_idx = num_qo_tiles - 1; qo_tile_idx >= 0; --qo_tile_idx) {
      int effective_kv_len = causal ? packed_causal_kv_end(qo_len, kv_len, qo_tile_idx,
                                                           cluster_tile_q, num_qo_tiles, num_heads)
                                    : kv_len;
      total_kv_lens += effective_kv_len;
    }
  }

  auto f = [](int x) {
    if (x <= 8) {
      return 32;
    } else if (x <= 16) {
      return 64;
    } else if (x <= 32) {
      return 128;
    } else if (x <= 64) {
      return 192;
    }
    return ceil_div(x, 256) * 256;
  };

  int kv_len_limit = f(std::max(ceil_div(total_kv_lens, num_clusters), 1L));

  // step 1. load-balancing scheduling algorithm
  MinHeap cluster_cost_heap(num_clusters);
  std::vector<std::vector<IdType>> cluster_q_indptr(num_clusters, std::vector<IdType>()),
      cluster_kv_indptr(num_clusters, std::vector<IdType>()),
      cluster_q_len(num_clusters, std::vector<IdType>()),
      cluster_kv_len(num_clusters, std::vector<IdType>()),
      cluster_q_start(num_clusters, std::vector<IdType>()),
      cluster_kv_start(num_clusters, std::vector<IdType>()),
      cluster_kv_end(num_clusters, std::vector<IdType>()),
      cluster_partial_indptr(num_clusters, std::vector<IdType>());

  std::vector<IdType> merge_packed_offset_start(num_sm, 0), merge_packed_offset_end(num_sm, 0),
      merge_partial_packed_offset_start(num_sm, 0), merge_partial_packed_offset_end(num_sm, 0),
      merge_partial_stride(num_sm, 0);

  int merge_cta_counter = 0;
  int partial_o_nnz = 0;

  for (auto& [i, qo_len, kv_len] : idx_qo_kv_len_vec) {
    int packed_qo_len = qo_len * num_heads;
    int num_qo_tiles = ceil_div(packed_qo_len, cluster_tile_q);
    for (int qo_tile_idx = num_qo_tiles - 1; qo_tile_idx >= 0; --qo_tile_idx) {
      int remaining_len = causal ? packed_causal_kv_end(qo_len, kv_len, qo_tile_idx, cluster_tile_q,
                                                        num_qo_tiles, num_heads)
                                 : kv_len;
      int kv_start = 0;
      bool split_kv = remaining_len > kv_len_limit;
      int row_tile_size = std::min(cluster_tile_q, packed_qo_len - qo_tile_idx * cluster_tile_q);
      if (split_kv) {
        /*
         * Proof(Zihao): merge_cta_counter <= num_sm (num_sm == num_clusters * cluster_size)
         *
         * Precondition:
         * 1. kv_len_limit * num_clusters >= total_kv_lens == sum(remaining_len)
         * 2. num_qo_chunks <= max((remaining_len * cluster_size) // kv_len_limit, 1)
         * 3. num_qo_tiles_requires_split <= num_clusters

         * Implication:
         * 1. sum(num_qo_chunks) <= max(sum(remaining_len) * cluster_size / kv_len_limit,
         num_qo_tiles_requires_split)
         * 2. sum(num_qo_chunks) <= max(cluster_size * num_clusters, num_qo_tiles_requires_split)
         */
        int num_qo_chunks = std::max(remaining_len * cluster_size / kv_len_limit, 1);
        // row_chunk_size * num_qo_chunks >= row_tile_size
        int row_chunk_size = ceil_div(row_tile_size, num_qo_chunks);
        int current_q_tile_end =
            std::min(cluster_tile_q, packed_qo_len - qo_tile_idx * cluster_tile_q);
        for (int offset_start = 0; offset_start < row_tile_size; offset_start += row_chunk_size) {
          merge_packed_offset_start[merge_cta_counter] =
              qo_indptr_h[i] * num_heads + qo_tile_idx * cluster_tile_q + offset_start;
          merge_packed_offset_end[merge_cta_counter] =
              qo_indptr_h[i] * num_heads + qo_tile_idx * cluster_tile_q +
              std::min(offset_start + row_chunk_size, current_q_tile_end);
          merge_partial_packed_offset_start[merge_cta_counter] = partial_o_nnz + offset_start;
          merge_partial_packed_offset_end[merge_cta_counter] =
              partial_o_nnz + ceil_div(remaining_len, kv_len_limit) * row_tile_size;
          merge_partial_stride[merge_cta_counter] = row_tile_size;
          merge_cta_counter++;
        }
      }
      bool zero_kv_len = (remaining_len == 0);
      while (remaining_len > 0 || zero_kv_len) {
        auto [cluster_idx, accum_cost] = cluster_cost_heap.pop();
        int actual_len = std::min(remaining_len, kv_len_limit);
        cluster_cost_heap.insert(
            {cluster_idx, accum_cost + cost_function(cluster_tile_q, actual_len)});
        cluster_q_len[cluster_idx].push_back(qo_len);
        cluster_kv_len[cluster_idx].push_back(kv_len);
        cluster_q_indptr[cluster_idx].push_back(qo_indptr_h[i]);
        cluster_kv_indptr[cluster_idx].push_back(kv_indptr_h[i]);
        if (split_kv) {
          cluster_partial_indptr[cluster_idx].push_back(partial_o_nnz);
          partial_o_nnz += row_tile_size;
        } else {
          cluster_partial_indptr[cluster_idx].push_back(-1);
        }
        cluster_q_start[cluster_idx].push_back(qo_tile_idx * cluster_tile_q);
        cluster_kv_start[cluster_idx].push_back(kv_start);
        cluster_kv_end[cluster_idx].push_back(kv_start + actual_len);
        remaining_len -= actual_len;
        kv_start += actual_len;
        if (zero_kv_len) break;
      }
    }
  }

  FLASHINFER_CHECK(merge_cta_counter <= num_sm,
                   "Internal Error: merge_cta_counter should be less than or equal to num_sm, "
                   "please report this bug to the developers");

  int max_total_num_works = 16384;  // NOTE(Zihao): adjust it later

  std::vector<IdType> work_indptr_vec(num_clusters + 1, 0);
  for (uint32_t i = 0; i < num_clusters; ++i) {
    work_indptr_vec[i + 1] = work_indptr_vec[i] + cluster_q_indptr[i].size();
  }
  int total_num_works = work_indptr_vec.back();
  auto q_indptr_vec = flatten(cluster_q_indptr, total_num_works);
  auto kv_indptr_vec = flatten(cluster_kv_indptr, total_num_works);
  auto partial_indptr_vec = flatten(cluster_partial_indptr, total_num_works);
  auto q_len_vec = flatten(cluster_q_len, total_num_works);
  auto kv_len_vec = flatten(cluster_kv_len, total_num_works);
  auto q_start_vec = flatten(cluster_q_start, total_num_works);
  auto kv_start_vec = flatten(cluster_kv_start, total_num_works);
  auto kv_end_vec = flatten(cluster_kv_end, total_num_works);

  AlignedAllocator int_allocator(int_buffer, int_workspace_size_in_bytes);
  plan_info.q_indptr_offset =
      int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "mla_q_indptr");
  plan_info.kv_indptr_offset =
      int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "mla_kv_indptr");
  plan_info.partial_indptr_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * max_total_num_works, 16, "mla_partial_indptr");
  plan_info.merge_packed_offset_start_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * num_sm, 16, "mla_merge_packed_offset_start");
  plan_info.merge_packed_offset_end_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * num_sm, 16, "mla_merge_packed_offset_end");
  plan_info.merge_partial_packed_offset_start_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * num_sm, 16, "mla_merge_partial_packed_offset_start");
  plan_info.merge_partial_packed_offset_end_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * num_sm, 16, "mla_merge_partial_packed_offset_end");
  plan_info.merge_partial_stride_offset =
      int_allocator.aligned_alloc_offset(sizeof(IdType) * num_sm, 16, "mla_merge_partial_stride");
  plan_info.q_len_offset =
      int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "mla_q_len");
  plan_info.kv_len_offset =
      int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "mla_kv_len");
  plan_info.q_start_offset =
      int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "mla_q_start");
  plan_info.kv_start_offset =
      int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "mla_kv_start");
  plan_info.kv_end_offset =
      int_allocator.aligned_alloc_offset(sizeof(IdType) * max_total_num_works, 16, "mla_kv_end");
  plan_info.work_indptr_offset = int_allocator.aligned_alloc_offset(
      sizeof(IdType) * max_total_num_works, 16, "mla_work_indptr");

  IdType* cluster_q_indptr_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.q_indptr_offset);
  IdType* cluster_kv_indptr_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_indptr_offset);
  IdType* cluster_partial_indptr_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.partial_indptr_offset);
  IdType* cluster_merge_packed_offset_start_h = GetPtrFromBaseOffset<IdType>(
      page_locked_int_buffer, plan_info.merge_packed_offset_start_offset);
  IdType* cluster_merge_packed_offset_end_h = GetPtrFromBaseOffset<IdType>(
      page_locked_int_buffer, plan_info.merge_packed_offset_end_offset);
  IdType* cluster_merge_partial_packed_offset_start_h = GetPtrFromBaseOffset<IdType>(
      page_locked_int_buffer, plan_info.merge_partial_packed_offset_start_offset);
  IdType* cluster_merge_partial_packed_offset_end_h = GetPtrFromBaseOffset<IdType>(
      page_locked_int_buffer, plan_info.merge_partial_packed_offset_end_offset);
  IdType* cluster_merge_partial_stride_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.merge_partial_stride_offset);
  IdType* cluster_q_len_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.q_len_offset);
  IdType* cluster_kv_len_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_len_offset);
  IdType* cluster_q_start_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.q_start_offset);
  IdType* cluster_kv_start_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_start_offset);
  IdType* cluster_kv_end_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.kv_end_offset);
  IdType* cluster_work_indptr_h =
      GetPtrFromBaseOffset<IdType>(page_locked_int_buffer, plan_info.work_indptr_offset);

  std::copy(q_indptr_vec.begin(), q_indptr_vec.end(), cluster_q_indptr_h);
  std::copy(kv_indptr_vec.begin(), kv_indptr_vec.end(), cluster_kv_indptr_h);
  std::copy(partial_indptr_vec.begin(), partial_indptr_vec.end(), cluster_partial_indptr_h);
  std::copy(merge_packed_offset_start.begin(), merge_packed_offset_start.end(),
            cluster_merge_packed_offset_start_h);
  std::copy(merge_packed_offset_end.begin(), merge_packed_offset_end.end(),
            cluster_merge_packed_offset_end_h);
  std::copy(merge_partial_packed_offset_start.begin(), merge_partial_packed_offset_start.end(),
            cluster_merge_partial_packed_offset_start_h);
  std::copy(merge_partial_packed_offset_end.begin(), merge_partial_packed_offset_end.end(),
            cluster_merge_partial_packed_offset_end_h);
  std::copy(merge_partial_stride.begin(), merge_partial_stride.end(),
            cluster_merge_partial_stride_h);
  std::copy(q_len_vec.begin(), q_len_vec.end(), cluster_q_len_h);
  std::copy(kv_len_vec.begin(), kv_len_vec.end(), cluster_kv_len_h);
  std::copy(q_start_vec.begin(), q_start_vec.end(), cluster_q_start_h);
  std::copy(kv_start_vec.begin(), kv_start_vec.end(), cluster_kv_start_h);
  std::copy(kv_end_vec.begin(), kv_end_vec.end(), cluster_kv_end_h);
  std::copy(work_indptr_vec.begin(), work_indptr_vec.end(), cluster_work_indptr_h);

  size_t num_bytes_to_copy = int_allocator.num_allocated_bytes();
  FLASHINFER_CUDA_CALL(cudaMemcpyAsync(int_buffer, page_locked_int_buffer, num_bytes_to_copy,
                                       cudaMemcpyHostToDevice, stream));

  constexpr size_t sizeof_dtype_o = 2;
  AlignedAllocator float_allocator(float_buffer, float_workspace_size_in_bytes);
  plan_info.partial_o_offset = float_allocator.aligned_alloc_offset(
      2 * num_clusters * cluster_tile_q * sizeof_dtype_o * head_dim_o, 16, "mla_partial_o");
  plan_info.partial_lse_offset = float_allocator.aligned_alloc_offset(
      2 * num_clusters * cluster_tile_q * sizeof(float), 16, "mla_partial_lse");

  return cudaSuccess;
}

}  // namespace flashinfer
#endif  // FLASHINFER_ATTENTION_SCHEDULER_CUH_
