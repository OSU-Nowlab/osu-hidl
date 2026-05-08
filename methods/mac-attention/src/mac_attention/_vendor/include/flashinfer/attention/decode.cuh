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
#ifndef FLASHINFER_DECODE_CUH_
#define FLASHINFER_DECODE_CUH_
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include <math_constants.h>
#include <iostream>

#include "../cp_async.cuh"
#include "../math.cuh"
#include "../pos_enc.cuh"
#include "../utils.cuh"
#include "../vec_dtypes.cuh"
#include "../logging.h"
#include "cascade.cuh"
#include "state.cuh"

namespace flashinfer {

DEFINE_HAS_MEMBER(decode_maybe_q_rope_offset)

namespace cg = cooperative_groups;
using cp_async::PrefetchMode;
using cp_async::SharedMemFillMode;

namespace {

/*!
 * \brief Load k tile from smem and compute qk
 * \tparam pos_encoding_mode The positional encoding mode used in the kernel
 * \tparam head_dim A template integer indicates the head dimension
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam tile_size A template integer indicates the tile size per (bdx * bdy) threads.
 * \tparam T A template type indicates the input data type
 * \param smem A pointer to the start of shared memory
 * \param q_vec A vector of float indicates the thread-local query vector
 * \param freq A vector of float indicates the thread-local rope frequency
 * \param kv_shared_offset An array of uint32_t indicates the k/v tiles offset
 *   in shared memory of different pipeline stages
 * \param kv_idx A integer indicates the thread-local kv position in kv-cache
 * \param compute_stage_idx A integer indicates the compute stage index in the pipeline
 * \param s A float indicates the thread-local result of qk
 * \param st The self-attention state to be updated
 */
template <PosEncodingMode pos_encoding_mode, uint32_t vec_size, uint32_t bdx, uint32_t tile_size,
          typename AttentionVariant, typename Params, typename T>
__device__ __forceinline__ void compute_qk(
    const Params& params, AttentionVariant variant, const uint32_t batch_idx, const T* smem,
    const vec_t<float, vec_size>& q_vec, const vec_t<float, vec_size>& freq, uint32_t kv_idx_base,
    uint32_t iter_base, uint32_t iter_bound, uint32_t qo_head_idx, uint32_t kv_head_idx, float* s,
    state_t<vec_size>& st, const uint32_t tx, const uint32_t ty, const uint32_t tz) {
  float m_prev = st.m;
#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    vec_t<float, vec_size> k_vec;
    if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
      // apply rotary embedding for all rows in k matrix of kv-cache
      k_vec = vec_apply_llama_rope<vec_size, bdx>(smem + j * bdx * vec_size, freq,
                                                  kv_idx_base + tz * tile_size + j);
    } else {
      // do not apply rotary embedding
      k_vec.cast_load(smem + (j * bdx + tx) * vec_size);
    }
    s[j] = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      s[j] += q_vec[i] * k_vec[i];
    }
#pragma unroll
    for (uint32_t offset = bdx / 2; offset > 0; offset /= 2) {
      s[j] += math::shfl_xor_sync(s[j], offset);
    }
    const uint32_t pos = kv_idx_base + tz * tile_size + j;
    s[j] = variant.LogitsTransform(params, s[j], batch_idx, /*qo_idx=*/0, /*kv_idx=*/pos,
                                   qo_head_idx, kv_head_idx);
    if constexpr (variant.use_softmax) {
      s[j] *= variant.sm_scale_log2;
    }

    bool mask = variant.LogitsMask(params, batch_idx, /*qo_idx=*/0, /*kv_idx=*/pos, qo_head_idx,
                                   kv_head_idx);
    s[j] = (iter_base + tz * tile_size + j < iter_bound && mask) ? s[j] : -math::inf;
    st.m = max(st.m, s[j]);
  }

  if constexpr (variant.use_softmax) {
    float o_scale = math::ptx_exp2(m_prev - st.m);
    st.d *= o_scale;
#pragma unroll
    for (uint32_t j = 0; j < tile_size; ++j) {
      s[j] = math::ptx_exp2(s[j] - st.m);
      st.d += s[j];
    }
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      st.o[i] = st.o[i] * o_scale;
    }
  }
}

/*!
 * \brief Load k tile from smem and compute qk
 * \tparam pos_encoding_mode The positional encoding mode used in the kernel
 * \tparam head_dim A template integer indicates the head dimension
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam tile_size A template integer indicates the tile size per (bdx * bdy) threads.
 * \tparam T A template type indicates the input data type
 * \param smem A pointer to the start of shared memory
 * \param q_vec A vector of float indicates the thread-local query vector
 * \param freq A vector of float indicates the thread-local rope frequency
 * \param kv_shared_offset An array of uint32_t indicates the k/v tiles offset
 *   in shared memory of different pipeline stages
 * \param kv_idx A integer indicates the thread-local kv position in kv-cache
 * \param compute_stage_idx A integer indicates the compute stage index in the pipeline
 * \param s A float indicates the thread-local result of qk
 * \param st The self-attention state to be updated
 */
template <PosEncodingMode pos_encoding_mode, uint32_t vec_size, uint32_t bdx, uint32_t tile_size,
          typename AttentionVariant, typename Params, typename T>
__device__ __forceinline__ void compute_mac_attention_qk(
    const Params& params, AttentionVariant variant, const uint32_t batch_idx, const T* smem,
    const vec_t<float, vec_size>& q_vec, const vec_t<float, vec_size>& freq, uint32_t kv_idx_base,
    uint32_t iter_base, uint32_t iter_bound, uint32_t qo_head_idx, uint32_t kv_head_idx, float* s,
    state_t<vec_size>& st, const uint32_t tx, const uint32_t ty, const uint32_t tz,
    const uint32_t* attn_start_pos, const uint32_t* attn_retrieve_pos, float* s_downdate,
    state_t<vec_size>* st_downdate_ptr, uint32_t cutoff_pos) {
  float m_prev = st.m;
  const bool enable_downdate = st_downdate_ptr != nullptr;
  state_t<vec_size>& st_downdate = enable_downdate ? *st_downdate_ptr : st;
  float m_prev_downdate = enable_downdate ? st_downdate.m : 0.f;
  uint32_t start_pos = attn_start_pos[batch_idx * params.num_qo_heads + qo_head_idx];
  // uint32_t retrieve_pos = attn_retrieve_pos[batch_idx * params.num_qo_heads + qo_head_idx];
#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    vec_t<float, vec_size> k_vec;
    if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
      // apply rotary embedding for all rows in k matrix of kv-cache
      k_vec = vec_apply_llama_rope<vec_size, bdx>(smem + j * bdx * vec_size, freq,
                                                  kv_idx_base + tz * tile_size + j);
    } else {
      // do not apply rotary embedding
      k_vec.cast_load(smem + (j * bdx + tx) * vec_size);
    }
    s[j] = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      s[j] += q_vec[i] * k_vec[i];
    }
#pragma unroll
    for (uint32_t offset = bdx / 2; offset > 0; offset /= 2) {
      s[j] += math::shfl_xor_sync(s[j], offset);
    }
    const uint32_t pos = kv_idx_base + tz * tile_size + j;
    s[j] = variant.LogitsTransform(params, s[j], batch_idx, /*qo_idx=*/0, /*kv_idx=*/pos,
                                   qo_head_idx, kv_head_idx);
    if constexpr (variant.use_softmax) {
      s[j] *= variant.sm_scale_log2;
    }

    bool mask = variant.LogitsMask(params, batch_idx, /*qo_idx=*/0, /*kv_idx=*/pos, qo_head_idx,
                                   kv_head_idx);
    bool start_pos_mask = pos >= start_pos;
    // bool start_pos_mask = (pos >= start_pos) && (pos <= retrieve_pos);

    if (enable_downdate) {
      bool keep = pos >= cutoff_pos;
      if constexpr (variant.use_softmax) {
        s_downdate[j] =
            (keep && iter_base + tz * tile_size + j < iter_bound && mask) ? s[j] : -math::inf;
        st_downdate.m = max(st_downdate.m, s_downdate[j]);
      } else {
        s_downdate[j] = (keep && iter_base + tz * tile_size + j < iter_bound && mask) ? s[j] : 0.f;
      }
    }

    s[j] = (iter_base + tz * tile_size + j < iter_bound && mask && start_pos_mask) ? s[j]
                                                                                    : -math::inf;
    st.m = max(st.m, s[j]);
  }

  if constexpr (variant.use_softmax) {
    float o_scale = math::ptx_exp2(m_prev - st.m);
    st.d *= o_scale;
#pragma unroll
    for (uint32_t j = 0; j < tile_size; ++j) {
      s[j] = math::ptx_exp2(s[j] - st.m);
      st.d += s[j];
    }
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      st.o[i] = st.o[i] * o_scale;
    }
    if (enable_downdate) {
      float o_scale_downdate = math::ptx_exp2(m_prev_downdate - st_downdate.m);
      st_downdate.d *= o_scale_downdate;
#pragma unroll
      for (uint32_t j = 0; j < tile_size; ++j) {
        s_downdate[j] = math::ptx_exp2(s_downdate[j] - st_downdate.m);
        st_downdate.d += s_downdate[j];
      }
#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        st_downdate.o[i] = st_downdate.o[i] * o_scale_downdate;
      }
    }
  }
}

/*!
 * \brief Load k tile from smem and compute qk
 * \tparam pos_encoding_mode The positional encoding mode used in the kernel
 * \tparam head_dim A template integer indicates the head dimension
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam tile_size A template integer indicates the tile size per (bdx * bdy) threads.
 * \tparam T A template type indicates the input data type
 * \param smem A pointer to the start of shared memory
 * \param q_vec A vector of float indicates the thread-local query vector
 * \param freq A vector of float indicates the thread-local rope frequency
 * \param kv_shared_offset An array of uint32_t indicates the k/v tiles offset
 *   in shared memory of different pipeline stages
 * \param kv_idx A integer indicates the thread-local kv position in kv-cache
 * \param compute_stage_idx A integer indicates the compute stage index in the pipeline
 * \param s A float indicates the thread-local result of qk
 * \param st The self-attention state to be updated
 */
template <PosEncodingMode pos_encoding_mode, uint32_t vec_size, uint32_t bdx, uint32_t tile_size,
          typename AttentionVariant, typename Params, typename T>
__device__ __forceinline__ void compute_mac_attention_qk_opt(
    const Params& params, AttentionVariant variant, const uint32_t batch_idx, const T* smem,
    const vec_t<float, vec_size>& q_vec, const vec_t<float, vec_size>& freq, uint32_t kv_idx_base,
    uint32_t iter_base, uint32_t iter_bound, uint32_t qo_head_idx, uint32_t kv_head_idx, float* s,
    state_t<vec_size>& st, const uint32_t tx, const uint32_t ty, const uint32_t tz,
    const uint32_t* attn_start_pos) {
  float m_prev = st.m;
  uint32_t start_pos = attn_start_pos[batch_idx * params.num_qo_heads + qo_head_idx];
#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    // Global KV index for this row in the tile
    const uint32_t pos = kv_idx_base + tz * tile_size + j;

    vec_t<float, vec_size> k_vec;
    if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
      // apply rotary embedding for all rows in k matrix of kv-cache
      k_vec = vec_apply_llama_rope<vec_size, bdx>(smem + j * bdx * vec_size, freq,
                                                  kv_idx_base + tz * tile_size + j);
    } else {
      // do not apply rotary embedding
      k_vec.cast_load(smem + (j * bdx + tx) * vec_size);
    }
    s[j] = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      s[j] += q_vec[i] * k_vec[i];
    }
#pragma unroll
    for (uint32_t offset = bdx / 2; offset > 0; offset /= 2) {
      s[j] += math::shfl_xor_sync(s[j], offset);
    }

    s[j] = variant.LogitsTransform(params, s[j], batch_idx, /*qo_idx=*/0, /*kv_idx=*/pos,
                                   qo_head_idx, kv_head_idx);
    if constexpr (variant.use_softmax) {
      s[j] *= variant.sm_scale_log2;
    }

    bool mask = variant.LogitsMask(params, batch_idx, /*qo_idx=*/0, /*kv_idx=*/pos, qo_head_idx,
                                   kv_head_idx);
    bool start_pos_mask = pos >= start_pos;

    s[j] = (iter_base + tz * tile_size + j < iter_bound && mask && start_pos_mask) ? s[j]
                                                                                    : -math::inf;
    st.m = max(st.m, s[j]);
  }

  if constexpr (variant.use_softmax) {
    float o_scale = math::ptx_exp2(m_prev - st.m);
    st.d *= o_scale;
#pragma unroll
    for (uint32_t j = 0; j < tile_size; ++j) {
      s[j] = math::ptx_exp2(s[j] - st.m);
      st.d += s[j];
    }
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      st.o[i] = st.o[i] * o_scale;
    }
  }
}



/*!
 * \brief Load k tile from smem and compute qk
 * \tparam pos_encoding_mode The positional encoding mode used in the kernel
 * \tparam head_dim A template integer indicates the head dimension
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam tile_size A template integer indicates the tile size per (bdx * bdy) threads.
 * \tparam T A template type indicates the input data type
 * \param smem A pointer to the start of shared memory
 * \param q_vec A vector of float indicates the thread-local query vector
 * \param freq A vector of float indicates the thread-local rope frequency
 * \param kv_shared_offset An array of uint32_t indicates the k/v tiles offset
 *   in shared memory of different pipeline stages
 * \param kv_idx A integer indicates the thread-local kv position in kv-cache
 * \param compute_stage_idx A integer indicates the compute stage index in the pipeline
 * \param s A float indicates the thread-local result of qk
 * \param st The self-attention state to be updated
 */
template <PosEncodingMode pos_encoding_mode,
          uint32_t vec_size,
          uint32_t bdx,
          uint32_t tile_size,
          typename AttentionVariant,
          typename Params,
          typename T>
__device__ __forceinline__ void compute_rectification_cache_qk(
    const Params& params,
    AttentionVariant variant,
    const uint32_t batch_idx,
    const T* __restrict__ smem,                          // K-tile smem base
    const vec_t<float, vec_size>& q_vec,
    const vec_t<float, vec_size>& freq,
    uint32_t kv_idx_base,                                // base index for this K tile (includes rope_offset)
    uint32_t iter_base,                                  // tile-local base iteration (for bounds)
    uint32_t iter_bound,                                 // number of valid rows in this chunk
    uint32_t qo_head_idx,
    uint32_t kv_head_idx,
    float* __restrict__ s,                               // per-row logits buffer (tile_size)
    state_t<vec_size>& st,
    const uint32_t tx, const uint32_t ty, const uint32_t tz,
    const uint32_t start_pos_for_mask) {                 // NEW: already includes rope_offset

  float m_prev = st.m;

#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    // Global KV index for this row in the tile (same coordinate system as start_pos_for_mask)
    const uint32_t pos = kv_idx_base + tz * tile_size + j;

    vec_t<float, vec_size> k_vec;
    if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
      k_vec = vec_apply_llama_rope<vec_size, bdx>(smem + j * bdx * vec_size, freq, pos);
    } else {
      k_vec.cast_load(smem + (j * bdx + tx) * vec_size);
    }

    s[j] = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      s[j] += q_vec[i] * k_vec[i];
    }
#pragma unroll
    for (uint32_t offset = bdx / 2; offset > 0; offset /= 2) {
      s[j] += math::shfl_xor_sync(s[j], offset);
    }

    s[j] = variant.LogitsTransform(params, s[j], batch_idx, /*qo_idx=*/0, /*kv_idx=*/pos,
                                   qo_head_idx, kv_head_idx);
    if constexpr (AttentionVariant::use_softmax) {
      s[j] *= variant.sm_scale_log2;
    }

    const bool mask      = variant.LogitsMask(params, batch_idx, /*qo_idx=*/0, /*kv_idx=*/pos,
                                              qo_head_idx, kv_head_idx);
    // const bool start_pos_mask = (pos >= start_pos_for_mask);

    // Bounds + masks
    s[j] = ( (iter_base + tz * tile_size + j) < iter_bound && mask)
           ? s[j] : -math::inf;

    st.m = max(st.m, s[j]);
  }

  if constexpr (AttentionVariant::use_softmax) {
    const float o_scale = math::ptx_exp2(m_prev - st.m);
    st.d *= o_scale;
#pragma unroll
    for (uint32_t j = 0; j < tile_size; ++j) {
      s[j] = math::ptx_exp2(s[j] - st.m);
      st.d += s[j];
    }
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      st.o[i] = st.o[i] * o_scale;
    }
  }
}


// --- DUAL compute: main (attn_start_pos) + downdate (tail window) ---
// main path code is preserved exactly; downdate mirrors it with dd_start.
template <PosEncodingMode pos_encoding_mode, uint32_t vec_size, uint32_t bdx, uint32_t tile_size,
          typename AttentionVariant, typename Params, typename T>
__device__ __forceinline__ void compute_mac_attention_qk_opt_dual(
    const Params& params, AttentionVariant variant, const uint32_t batch_idx, const T* smem,
    const vec_t<float, vec_size>& q_vec, const vec_t<float, vec_size>& freq, uint32_t kv_idx_base,
    uint32_t iter_base, uint32_t iter_bound, uint32_t qo_head_idx, uint32_t kv_head_idx,
    // OUT: per-tile logits buffers + states for both paths
    float* __restrict__ s_main, state_t<vec_size>& st_main,
    float* __restrict__ s_dd,   state_t<vec_size>& st_dd,
    const uint32_t tx, const uint32_t ty, const uint32_t tz,
    const uint32_t* __restrict__ attn_start_pos,   // [N*H]
    uint32_t start_pos_dd)                          // dd_start (rope_offset + clamp)
{
  // ===== MAIN path (unchanged) =====
  float m_prev_main = st_main.m;
  float m_prev_dd   = st_dd.m;
  const uint32_t start_pos = attn_start_pos[batch_idx * params.num_qo_heads + qo_head_idx];

#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    // Global KV index for this row in the tile
    const uint32_t pos = kv_idx_base + tz * tile_size + j;

    vec_t<float, vec_size> k_vec;
    if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
      // apply rotary embedding for all rows in k matrix of kv-cache
      k_vec = vec_apply_llama_rope<vec_size, bdx>(smem + j * bdx * vec_size, freq,
                                                  kv_idx_base + tz * tile_size + j);
    } else {
      // do not apply rotary embedding
      k_vec.cast_load(smem + (j * bdx + tx) * vec_size);
    }
    float s_val = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      s_val = fmaf(q_vec[i], k_vec[i], s_val);
    }
#pragma unroll
    for (uint32_t offset = bdx / 2; offset > 0; offset /= 2) {
      s_val += math::shfl_xor_sync(s_val, offset);
    }

    s_val = variant.LogitsTransform(params, s_val, batch_idx, /*qo_idx=*/0, /*kv_idx=*/pos,
                                    qo_head_idx, kv_head_idx);
    if constexpr (AttentionVariant::use_softmax) {
      s_val *= variant.sm_scale_log2;
    }

    bool mask      = variant.LogitsMask(params, batch_idx, /*qo_idx=*/0, /*kv_idx=*/pos,
                                        qo_head_idx, kv_head_idx);
    bool start_pos_mask = pos >= start_pos;

    s_main[j] =
        (iter_base + tz * tile_size + j < iter_bound && mask && start_pos_mask) ? s_val : -math::inf;
    st_main.m = max(st_main.m, s_main[j]);

    // ===== DOWNDATE path (mirrors main, but uses start_pos_dd) =====
    bool dd_mask = pos >= start_pos_dd;
    s_dd[j] = (iter_base + tz * tile_size + j < iter_bound && mask && dd_mask) ? s_val : -math::inf;
    st_dd.m = max(st_dd.m, s_dd[j]);
  }

  // After filling s_main[j], s_dd[j] and updating st_main.m, st_dd.m:
  const bool tile_has_main = (st_main.m != -math::inf);
  const bool tile_has_dd   = (st_dd.m   != -math::inf);

  if constexpr (AttentionVariant::use_softmax) {
    // MAIN
    if (tile_has_main) {
      float o_scale = math::ptx_exp2(m_prev_main - st_main.m);
      st_main.d *= o_scale;
      #pragma unroll
      for (uint32_t j = 0; j < tile_size; ++j) {
        s_main[j] = math::ptx_exp2(s_main[j] - st_main.m);
        st_main.d += s_main[j];
      }
      #pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) st_main.o[i] *= o_scale;
    } else {
      #pragma unroll
      for (uint32_t j = 0; j < tile_size; ++j) s_main[j] = 0.f;  // fully masked → no contribution
    }

    // DOWNDATE
    if (tile_has_dd) {
      float o_scale = math::ptx_exp2(m_prev_dd - st_dd.m);
      st_dd.d *= o_scale;
      #pragma unroll
      for (uint32_t j = 0; j < tile_size; ++j) {
        s_dd[j] = math::ptx_exp2(s_dd[j] - st_dd.m);
        st_dd.d += s_dd[j];
      }
      #pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) st_dd.o[i] *= o_scale;
    } else {
      #pragma unroll
      for (uint32_t j = 0; j < tile_size; ++j) s_dd[j] = 0.f;
    }
  }

}



/*!
 * \brief Load v tile from shared memory and update local state
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam tile_size A template integer indicates the tile size per (bdx * bdy) threads.
 * \tparam T A template type indicates the input data type
 * \param smem A pointer to the start of shared memory
 * \param s A float indicates the pre-softmax attention score
 * \param kv_shared_offset An array of uint32_t indicates the k/v tiles offset
 * in shared memory of different pipeline stages
 * \param compute_stage_idx A integer indicates the compute stage index in the pipeline
 * \param st The flashattention state to be updated
 */
template <uint32_t vec_size, uint32_t bdx, uint32_t tile_size, typename T>
__device__ __forceinline__ void update_local_state(const T* smem, const float* s,
                                                   uint32_t compute_stage_idx,
                                                   state_t<vec_size>& st, uint32_t tx) {
#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    vec_t<float, vec_size> v_vec;
    v_vec.cast_load(smem + (j * bdx + tx) * vec_size);
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      st.o[i] = st.o[i] + s[j] * v_vec[i];
    }
  }
}

/*!
 * \brief Synchronize the state of all warps inside a threadblock.
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam bdy A template integer indicates the block size in y dimension
 * \param st The warp local state
 * \param smem The pointer to shared memory buffer for o
 * \param smem_md The pointer to shared memory buffer for m/d
 */
template <uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz, typename AttentionVariant>
__device__ __forceinline__ void sync_state(AttentionVariant variant, state_t<vec_size>& st,
                                           float* smem, float* smem_md, const uint32_t tx,
                                           const uint32_t ty, const uint32_t tz) {
  if constexpr (bdz > 1) {
    constexpr uint32_t head_dim = bdx * vec_size;
    auto block = cg::this_thread_block();
    st.o.store(smem + (tz * bdy + ty) * head_dim + tx * vec_size);
    if constexpr (variant.use_softmax) {
      smem_md[(tz * bdy + ty) * 2] = st.m;
      smem_md[(tz * bdy + ty) * 2 + 1] = st.d;
      block.sync();
      st.init();
#pragma unroll
      for (uint32_t j = 0; j < bdz; ++j) {
        float mz = smem_md[(j * bdy + ty) * 2], dz = smem_md[(j * bdy + ty) * 2 + 1];
        vec_t<float, vec_size> oz;
        oz.load(smem + (j * bdy + ty) * head_dim + tx * vec_size);
        st.merge(oz, mz, dz);
      }
    } else {
      block.sync();
      st.init();
#pragma unroll
      for (uint32_t j = 0; j < bdz; ++j) {
        vec_t<float, vec_size> oz;
        oz.load(smem + (j * bdy + ty) * head_dim + tx * vec_size);
#pragma unroll
        for (uint32_t i = 0; i < vec_size; ++i) {
          st.o[i] += oz[i];
        }
      }
    }
  }
}

}  // namespace

/*!
 * \brief FlashAttention decoding cuda kernel with kv-cache for a single request
 * \tparam pos_encoding_mode The positional encoding mode
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam bdy A template integer indicates the block size in y dimension
 * \tparam DTypeQ A template type indicates the query data type
 * \tparam DTypeKV A template type indicates the key-value data type
 * \tparam DTypeO A template type indicates the output data type
 * \param q [num_qo_heads, head_dim] The query matrix
 * \param k [seq_len, num_kv_heads, head_dim] The key matrix in kv-cache
 * \param v [seq_len, num_kv_heads, head_dim] The value matrix in kv-cache
 * \param o [num_qo_heads, head_dim] The output matrix
 * \param head_dim A integer indicates the head dimension
 * \param rope_rcp_scale A floating number indicate the reciprocal
 *   of scaling ratio used in PI(Position Interpolation) for RoPE (Rotary
 *   Positional Embeddings)
 * \param rope_rcp_theta A floating number indicate the reciprocal
 *   of "theta" used in RoPE (Rotary Positional Embeddings)
 * \param kv_chunk_size A integer indicates the kv-chunk size
 */
template <PosEncodingMode pos_encoding_mode, uint32_t num_stages_smem, uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz, typename AttentionVariant,
          typename Params>
__global__ void SingleDecodeWithKVCacheKernel(const __grid_constant__ Params params) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  const DTypeQ* q = params.q;
  const DTypeKV* k = params.k;
  const DTypeKV* v = params.v;
  const uint32_t q_stride_n = params.q_stride_n;
  const uint32_t q_stride_h = params.q_stride_h;
  const uint32_t kv_stride_n = params.kv_stride_n;
  const uint32_t kv_stride_h = params.kv_stride_h;
  DTypeO* o = params.o;
  float* lse = params.lse;
  uint32_t kv_chunk_size = params.kv_chunk_size;

  auto block = cg::this_thread_block();
  auto grid = cg::this_grid();

  constexpr uint32_t head_dim = bdx * vec_size;
  uint32_t kv_head_idx = blockIdx.y;
  uint32_t qo_head_idx = kv_head_idx * bdy + threadIdx.y;
  uint32_t kv_chunk_idx = blockIdx.x;
  uint32_t num_qo_heads = params.num_qo_heads;

  extern __shared__ uint8_t smem[];
  AttentionVariant variant(params, /*batch_idx=*/0, smem);
  const uint32_t seq_len = variant.kv_len;
  DTypeKV* k_smem = (DTypeKV*)smem;
  DTypeKV* v_smem = (DTypeKV*)(smem + num_stages_smem * bdy * tile_size_per_bdx * bdz * head_dim *
                                          sizeof(DTypeKV));
  float* smem_md = (float*)(smem + 2 * num_stages_smem * bdy * tile_size_per_bdx * bdz * head_dim *
                                       sizeof(DTypeKV));

  uint32_t tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
  vec_t<float, vec_size> q_vec;
  vec_t<float, vec_size> freq;
  if constexpr (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
    const float rope_rcp_scale = params.rope_rcp_scale;
    const float rope_rcp_theta = params.rope_rcp_theta;

#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      freq[i] = rope_rcp_scale *
                __powf(rope_rcp_theta,
                       float(2 * ((tx * vec_size + i) % (head_dim / 2))) / float(head_dim));
    }

    // apply rotary embedding to q matrix
    q_vec = vec_apply_llama_rope<vec_size, bdx>(q + qo_head_idx * q_stride_h, freq, seq_len - 1);
  } else {
    // do not apply rotary embedding to q matrix
    q_vec.cast_load(q + qo_head_idx * q_stride_h + tx * vec_size);
  }
  block.sync();

  uint32_t chunk_start = kv_chunk_idx * kv_chunk_size;
  kv_chunk_size = min(kv_chunk_size, seq_len - chunk_start);
  uint32_t chunk_end = chunk_start + kv_chunk_size;

  // preload k tiles and v tiles
  uint32_t producer_kv_idx_base = chunk_start;
  constexpr uint32_t vec_bits = sizeof(DTypeKV) * vec_size * 8;
#pragma unroll
  for (uint32_t iter = 0; iter < num_stages_smem; ++iter) {
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          k_smem + (((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          k + (producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j) * kv_stride_n +
              kv_head_idx * kv_stride_h + tx * vec_size,
          producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j < chunk_end);
    }
    cp_async::commit_group();
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
          v_smem + (((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          v + (producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j) * kv_stride_n +
              kv_head_idx * kv_stride_h + tx * vec_size,
          producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j < chunk_end);
    }
    cp_async::commit_group();
    producer_kv_idx_base += bdy * bdz * tile_size_per_bdx;
  }

  // pipelining k/v tiles loading and state updating
  uint32_t consumer_kv_idx_base = chunk_start, stage_idx = 0;
  state_t<vec_size> st_local;
  float s[bdy * tile_size_per_bdx];

#pragma unroll 2
  for (uint32_t iter = 0; iter < ceil_div(kv_chunk_size, tile_size_per_bdx * bdy * bdz); ++iter) {
    // compute qk
    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
    compute_qk<pos_encoding_mode, vec_size, bdx, bdy * tile_size_per_bdx>(
        params, variant, /*batch_idx=*/0,
        k_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim, q_vec, freq,
        consumer_kv_idx_base, iter * bdy * tile_size_per_bdx * bdz, kv_chunk_size, qo_head_idx,
        kv_head_idx, s, st_local, tx, ty, tz);
    block.sync();
    // load k
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          k_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          k + (producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j) * kv_stride_n +
              kv_head_idx * kv_stride_h + tx * vec_size,
          producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j < chunk_end);
    }
    cp_async::commit_group();

    // update m/d/o state
    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
    update_local_state<vec_size, bdx, bdy * tile_size_per_bdx>(
        v_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim, s, stage_idx,
        st_local, tx);
    block.sync();

    // load v
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
          v_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          v + (producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j) * kv_stride_n +
              kv_head_idx * kv_stride_h + tx * vec_size,
          producer_kv_idx_base + (tz * bdy + ty) * tile_size_per_bdx + j < chunk_end);
    }
    cp_async::commit_group();

    stage_idx = (stage_idx + 1) % num_stages_smem;
    producer_kv_idx_base += tile_size_per_bdx * bdy * bdz;
    consumer_kv_idx_base += tile_size_per_bdx * bdy * bdz;
  }
  cp_async::wait_group<0>();
  block.sync();

  // sync local state of all warps inside a threadblock
  sync_state<vec_size, bdx, bdy, bdz>(variant, st_local, reinterpret_cast<float*>(smem), smem_md,
                                      tx, ty, tz);
  if constexpr (variant.use_softmax) {
    st_local.normalize();
  }

  st_local.o.cast_store(o + (kv_chunk_idx * num_qo_heads + qo_head_idx) * head_dim + tx * vec_size);
  if (lse != nullptr) {
    lse[kv_chunk_idx * num_qo_heads + qo_head_idx] = st_local.get_lse();
  }
}

/*!
 * \brief FlashAttention decoding cuda kernel with paged kv-cache for multiple requests
 * \tparam pos_encoding_mode The positional encoding mode
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam bdy A template integer indicates the block size in y dimension
 * \tparam bdz A template integer indicates the block size in z dimension
 * \tparam DTypeQ A template type indicates the query data type
 * \tparam DTypeKV A template type indicates the key-value data type
 * \tparam DTypeO A template type indicates the output data type
 * \tparam IdType A template type indicates the index data type
 * \param q [batch_size, num_qo_heads, head_dim] The query matrix
 * \param paged_kv The paged kv-cache data structure
 * \param o [num_qo_heads, head_dim] The output matrix
 * \param tmp Used-allocated temporary buffer
 * \param lse The logsumexp values
 * \param sm_scale A float indicates the scale applied to pre-softmax logits
 * \param rope_rcp_scale A floating number indicate the reciprocal
 *   of scaling ratio used in PI(Position Interpolation) for RoPE (Rotary
 *   Positional Embeddings)
 * \param rope_rcp_theta A floating number indicate the reciprocal
 *   of "theta" used in RoPE (Rotary Positional Embeddings)
 */
template <PosEncodingMode POS_ENCODING_MODE, uint32_t num_stages_smem, uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz, typename AttentionVariant,
          typename Params>
__device__ __inline__ void BatchDecodeWithPagedKVCacheDevice(const Params& params, uint8_t smem[],
                                                             const uint32_t bx = blockIdx.x,
                                                             const uint32_t by = blockIdx.y,
                                                             const uint32_t tx = threadIdx.x,
                                                             const uint32_t ty = threadIdx.y,
                                                             const uint32_t tz = threadIdx.z) {
  auto block = cg::this_thread_block();
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  const DTypeQ* q = params.q;
  DTypeO* o = params.o;
  float* lse = params.lse;
  const auto paged_kv = params.paged_kv;
  const bool* block_valid_mask = params.block_valid_mask;
  const uint32_t padded_batch_size = params.padded_batch_size;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const bool partition_kv = params.partition_kv;

  constexpr uint32_t head_dim = bdx * vec_size;
  const uint32_t batch_idx = params.request_indices[bx];
  const uint32_t kv_tile_idx = params.kv_tile_indices[bx];
  const uint32_t kv_head_idx = by;
  const uint32_t qo_head_idx = kv_head_idx * bdy + ty;
  // printf("bx %u, batch_idx %u, bj %u, q head idx %u, kv head idx %u\n", bx, batch_idx, by,
  // qo_head_idx, kv_head_idx); NOTE(Zihao): when CUDAGraph is enabled, we will launch more blocks
  // than the actual batch size, so we need to check if the current batch is valid
  if (block_valid_mask && !block_valid_mask[bx]) return;
  const uint32_t kv_chunk_size = *(params.kv_chunk_size_ptr);
  const uint32_t kv_len = paged_kv.get_length(batch_idx);
  const uint32_t max_chunk_size = partition_kv ? kv_chunk_size : kv_len;
  const uint32_t chunk_start = partition_kv ? kv_tile_idx * max_chunk_size : 0;
  const uint32_t chunk_end =
      partition_kv ? min((kv_tile_idx + 1) * max_chunk_size, kv_len) : kv_len;
  const uint32_t chunk_size = chunk_end - chunk_start;

  AttentionVariant variant(params, batch_idx, smem);
  DTypeKV* k_smem = (DTypeKV*)smem;
  DTypeKV* v_smem = (DTypeKV*)(smem + num_stages_smem * tile_size_per_bdx * bdy * bdz * head_dim *
                                          sizeof(DTypeKV));
  size_t* kv_offset_smem = (size_t*)(smem + 2 * num_stages_smem * tile_size_per_bdx * bdy * bdz *
                                                head_dim * sizeof(DTypeKV));
  float* smem_md = (float*)(smem + 2 * num_stages_smem * tile_size_per_bdx * bdy * bdz * head_dim *
                                       sizeof(DTypeKV));

  vec_t<float, vec_size> q_vec;
  vec_t<float, vec_size> freq;
  const uint32_t q_stride_n = params.q_stride_n;
  const uint32_t q_stride_h = params.q_stride_h;
  if constexpr (POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
    const IdType* q_rope_offset = nullptr;
    if constexpr (has_decode_maybe_q_rope_offset_v<Params>) {
      q_rope_offset = params.decode_maybe_q_rope_offset;
    }
    int32_t q_rope_offset_val = q_rope_offset == nullptr ? (kv_len - 1) : q_rope_offset[batch_idx];
    const float rope_rcp_scale = params.rope_rcp_scale;
    const float rope_rcp_theta = params.rope_rcp_theta;

#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      freq[i] = rope_rcp_scale *
                __powf(rope_rcp_theta,
                       float(2 * ((tx * vec_size + i) % (head_dim / 2))) / float(head_dim));
    }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif
    // apply rotary embedding to q matrix
    q_vec = vec_apply_llama_rope<vec_size, bdx>(
        q + batch_idx * q_stride_n + qo_head_idx * q_stride_h, freq, q_rope_offset_val);
  } else {
// do not apply rotary embedding to q matrix
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif
    q_vec.cast_load(q + batch_idx * q_stride_n + qo_head_idx * q_stride_h + tx * vec_size);
  }

  // preload k/v tiles
  uint32_t stage_idx = 0;
  constexpr uint32_t vec_bits = sizeof(DTypeKV) * vec_size * 8;
  const IdType last_indptr = paged_kv.indptr[paged_kv.batch_size];

  static_assert(num_stages_smem <= bdx);
  uint32_t packed_page_iter_base = paged_kv.indptr[batch_idx] * paged_kv.page_size + chunk_start;
#pragma unroll
  for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
    uint32_t q, r;
    paged_kv.page_size.divmod(packed_page_iter_base + ((j * bdz + tz) * bdy + ty) * bdx + tx, q, r);
    kv_offset_smem[((j * bdz + tz) * bdy + ty) * bdx + tx] =
        paged_kv.protective_get_kv_offset(q, kv_head_idx, r, 0, last_indptr);
  }
  block.sync();

  size_t kv_offset[tile_size_per_bdx];
#pragma unroll
  for (uint32_t iter = 0; iter < num_stages_smem; ++iter) {
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      kv_offset[j] =
          kv_offset_smem[((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j] + tx * vec_size;
    }
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          k_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          paged_kv.k_data + kv_offset[j],
          ((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size);
    }
    cp_async::commit_group();
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
          v_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          paged_kv.v_data + kv_offset[j],
          ((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size);
    }
    cp_async::commit_group();
    stage_idx = (stage_idx + 1) % num_stages_smem;
  }

  state_t<vec_size> st;
  float s[bdy * tile_size_per_bdx];

#pragma unroll 2
  for (uint32_t iter = 0; iter < ceil_div(chunk_size, tile_size_per_bdx * bdy * bdz); ++iter) {
    if ((iter + num_stages_smem) % bdx == 0) {
#pragma unroll
      for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
        uint32_t q, r;
        paged_kv.page_size.divmod(
            packed_page_iter_base + ((iter + num_stages_smem) * tile_size_per_bdx * bdy * bdz +
                                     ((j * bdz + tz) * bdy + ty) * bdx + tx),
            q, r);
        kv_offset_smem[((j * bdz + tz) * bdy + ty) * bdx + tx] =
            paged_kv.protective_get_kv_offset(q, kv_head_idx, r, 0, last_indptr);
      }
    }
    // compute qk
    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
    compute_qk<POS_ENCODING_MODE, vec_size, bdx, bdy * tile_size_per_bdx>(
        params, variant, batch_idx,
        k_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim, q_vec, freq,
        (paged_kv.rope_pos_offset == nullptr ? 0 : paged_kv.rope_pos_offset[batch_idx]) +
            chunk_start + iter * tile_size_per_bdx * bdy * bdz,
        iter * tile_size_per_bdx * bdy * bdz, chunk_size, qo_head_idx, kv_head_idx, s, st, tx, ty,
        tz);
    block.sync();

#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      kv_offset[j] = kv_offset_smem[((((iter + num_stages_smem) % bdx) * bdz + tz) * bdy + ty) *
                                        tile_size_per_bdx +
                                    j] +
                     tx * vec_size;
    }

    // load k tiles
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          k_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          paged_kv.k_data + kv_offset[j],
          (((iter + num_stages_smem) * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size);
    }
    cp_async::commit_group();

    // update m/d/o states
    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
    update_local_state<vec_size, bdx, bdy * tile_size_per_bdx>(
        v_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim, s, stage_idx, st, tx);
    block.sync();

    // load v tiles
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
          v_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          paged_kv.v_data + kv_offset[j],
          (((iter + num_stages_smem) * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size);
    }
    cp_async::commit_group();
    stage_idx = (stage_idx + 1) % num_stages_smem;
  }
  cp_async::wait_group<0>();
  block.sync();

  // sync local state of all warps inside a threadblock
  sync_state<vec_size, bdx, bdy, bdz>(variant, st, reinterpret_cast<float*>(smem), smem_md, tx, ty,
                                      tz);
  if constexpr (variant.use_softmax) {
    st.normalize();
  }

  if (tz == 0) {
    st.o.cast_store(o + (bx * num_qo_heads + qo_head_idx) * head_dim + tx * vec_size);
    // write lse
    if (lse != nullptr) {
      lse[bx * num_qo_heads + qo_head_idx] = st.get_lse();
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <PosEncodingMode POS_ENCODING_MODE, uint32_t num_stages_smem, uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz, typename AttentionVariant,
          typename Params>
__global__ void BatchDecodeWithPagedKVCacheKernel(const __grid_constant__ Params params) {
  extern __shared__ uint8_t smem[];
  BatchDecodeWithPagedKVCacheDevice<POS_ENCODING_MODE, num_stages_smem, tile_size_per_bdx, vec_size,
                                    bdx, bdy, bdz, AttentionVariant>(params, smem);
}

/*!
 * \brief FlashAttention MACAttention decoding cuda kernel with paged kv-cache for multiple requests
 * \tparam pos_encoding_mode The positional encoding mode
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam bdy A template integer indicates the block size in y dimension
 * \tparam bdz A template integer indicates the block size in z dimension
 * \tparam DTypeQ A template type indicates the query data type
 * \tparam DTypeKV A template type indicates the key-value data type
 * \tparam DTypeO A template type indicates the output data type
 * \tparam IdType A template type indicates the index data type
 * \param q [batch_size, num_qo_heads, head_dim] The query matrix
 * \param paged_kv The paged kv-cache data structure
 * \param o [num_qo_heads, head_dim] The output matrix
 * \param tmp Used-allocated temporary buffer
 * \param lse The logsumexp values
 * \param sm_scale A float indicates the scale applied to pre-softmax logits
 * \param rope_rcp_scale A floating number indicate the reciprocal
 *   of scaling ratio used in PI(Position Interpolation) for RoPE (Rotary
 *   Positional Embeddings)
 * \param rope_rcp_theta A floating number indicate the reciprocal
 *   of "theta" used in RoPE (Rotary Positional Embeddings)
 */
template <PosEncodingMode POS_ENCODING_MODE, uint32_t num_stages_smem, uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz, typename AttentionVariant,
          typename Params>
__device__ __inline__ void BatchMACAttentionDecodeWithPagedKVCacheDevice(
    const Params& params, uint8_t smem[], const uint32_t bx = blockIdx.x,
    const uint32_t by = blockIdx.y, const uint32_t tx = threadIdx.x,
    const uint32_t ty = threadIdx.y, const uint32_t tz = threadIdx.z) {
  auto block = cg::this_thread_block();
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  const DTypeQ* q = params.q;
  DTypeO* o = params.o;
  float* lse = params.lse;

  DTypeO* downdate_o = params.downdate_o;
  float* downdate_lse = params.downdate_lse;

  const auto paged_kv = params.paged_kv;
  const bool* block_valid_mask = params.block_valid_mask;
  const uint32_t padded_batch_size = params.padded_batch_size;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const bool partition_kv = params.partition_kv;

  constexpr uint32_t head_dim = bdx * vec_size;
  const uint32_t batch_idx = params.request_indices[bx];
  const uint32_t kv_tile_idx = params.kv_tile_indices[bx];
  const uint32_t kv_head_idx = by;
  const uint32_t qo_head_idx = kv_head_idx * bdy + ty;
  // printf("bx %u, batch_idx %u, bj %u, q head idx %u, kv head idx %u\n", bx, batch_idx, by,
  // qo_head_idx, kv_head_idx); NOTE(Zihao): when CUDAGraph is enabled, we will launch more blocks
  // than the actual batch size, so we need to check if the current batch is valid
  if (block_valid_mask && !block_valid_mask[bx]) return;
  const uint32_t kv_chunk_size = *(params.kv_chunk_size_ptr);
  const uint32_t kv_len = paged_kv.get_length(batch_idx);
  const uint32_t max_chunk_size = partition_kv ? kv_chunk_size : kv_len;
  const uint32_t chunk_start = partition_kv ? kv_tile_idx * max_chunk_size : 0;
  const uint32_t chunk_end =
      partition_kv ? min((kv_tile_idx + 1) * max_chunk_size, kv_len) : kv_len;
  const uint32_t chunk_size = chunk_end - chunk_start;
  const uint32_t rope_offset =
      paged_kv.rope_pos_offset == nullptr ? 0 : paged_kv.rope_pos_offset[batch_idx];
  const uint32_t cutoff_tokens = kv_len > params.rope_ahead ? kv_len - params.rope_ahead : 0;
  const uint32_t cutoff_pos = rope_offset + cutoff_tokens;
  const bool compute_downdate = downdate_o != nullptr || downdate_lse != nullptr;

  AttentionVariant variant(params, batch_idx, smem);
  DTypeKV* k_smem = (DTypeKV*)smem;
  DTypeKV* v_smem = (DTypeKV*)(smem + num_stages_smem * tile_size_per_bdx * bdy * bdz * head_dim *
                                          sizeof(DTypeKV));
  size_t* kv_offset_smem = (size_t*)(smem + 2 * num_stages_smem * tile_size_per_bdx * bdy * bdz *
                                                head_dim * sizeof(DTypeKV));
  float* smem_md = (float*)(smem + 2 * num_stages_smem * tile_size_per_bdx * bdy * bdz * head_dim *
                                       sizeof(DTypeKV));

  vec_t<float, vec_size> q_vec;
  vec_t<float, vec_size> freq;
  const uint32_t q_stride_n = params.q_stride_n;
  const uint32_t q_stride_h = params.q_stride_h;
  if constexpr (POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
    const IdType* q_rope_offset = nullptr;
    if constexpr (has_decode_maybe_q_rope_offset_v<Params>) {
      q_rope_offset = params.decode_maybe_q_rope_offset;
    }
    int32_t q_rope_offset_val = q_rope_offset == nullptr ? (kv_len - 1) : q_rope_offset[batch_idx];
    const float rope_rcp_scale = params.rope_rcp_scale;
    const float rope_rcp_theta = params.rope_rcp_theta;

#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      freq[i] = rope_rcp_scale *
                __powf(rope_rcp_theta,
                       float(2 * ((tx * vec_size + i) % (head_dim / 2))) / float(head_dim));
    }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif
    // apply rotary embedding to q matrix
    q_vec = vec_apply_llama_rope<vec_size, bdx>(
        q + batch_idx * q_stride_n + qo_head_idx * q_stride_h, freq, q_rope_offset_val);
  } else {
// do not apply rotary embedding to q matrix
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif
    q_vec.cast_load(q + batch_idx * q_stride_n + qo_head_idx * q_stride_h + tx * vec_size);
  }

  // preload k/v tiles
  uint32_t stage_idx = 0;
  constexpr uint32_t vec_bits = sizeof(DTypeKV) * vec_size * 8;
  const IdType last_indptr = paged_kv.indptr[paged_kv.batch_size];

  static_assert(num_stages_smem <= bdx);
  uint32_t packed_page_iter_base = paged_kv.indptr[batch_idx] * paged_kv.page_size + chunk_start;
#pragma unroll
  for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
    uint32_t q, r;
    paged_kv.page_size.divmod(packed_page_iter_base + ((j * bdz + tz) * bdy + ty) * bdx + tx, q, r);
    kv_offset_smem[((j * bdz + tz) * bdy + ty) * bdx + tx] =
        paged_kv.protective_get_kv_offset(q, kv_head_idx, r, 0, last_indptr);
  }
  block.sync();

  size_t kv_offset[tile_size_per_bdx];
#pragma unroll
  for (uint32_t iter = 0; iter < num_stages_smem; ++iter) {
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      kv_offset[j] =
          kv_offset_smem[((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j] + tx * vec_size;
    }
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          k_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          paged_kv.k_data + kv_offset[j],
          ((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size);
    }
    cp_async::commit_group();
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
          v_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          paged_kv.v_data + kv_offset[j],
          ((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size);
    }
    cp_async::commit_group();
    stage_idx = (stage_idx + 1) % num_stages_smem;
  }

  state_t<vec_size> st;
  float s[bdy * tile_size_per_bdx];
  state_t<vec_size> st_downdate;
  float s_downdate[bdy * tile_size_per_bdx];
  if (compute_downdate) {
    st_downdate.init();
  }

#pragma unroll 2
  for (uint32_t iter = 0; iter < ceil_div(chunk_size, tile_size_per_bdx * bdy * bdz); ++iter) {
    if ((iter + num_stages_smem) % bdx == 0) {
#pragma unroll
      for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
        uint32_t q, r;
        paged_kv.page_size.divmod(
            packed_page_iter_base + ((iter + num_stages_smem) * tile_size_per_bdx * bdy * bdz +
                                     ((j * bdz + tz) * bdy + ty) * bdx + tx),
            q, r);
        kv_offset_smem[((j * bdz + tz) * bdy + ty) * bdx + tx] =
            paged_kv.protective_get_kv_offset(q, kv_head_idx, r, 0, last_indptr);
      }
    }
    // compute qk
    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
    compute_mac_attention_qk<POS_ENCODING_MODE, vec_size, bdx, bdy * tile_size_per_bdx>(
        params, variant, batch_idx,
        k_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim, q_vec, freq,
        rope_offset + chunk_start + iter * tile_size_per_bdx * bdy * bdz,
        iter * tile_size_per_bdx * bdy * bdz, chunk_size, qo_head_idx, kv_head_idx, s, st, tx, ty,
        tz, params.attn_start_pos, params.attn_retrieve_pos,
        compute_downdate ? s_downdate : nullptr, compute_downdate ? &st_downdate : nullptr,
        cutoff_pos);
    block.sync();

#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      kv_offset[j] = kv_offset_smem[((((iter + num_stages_smem) % bdx) * bdz + tz) * bdy + ty) *
                                        tile_size_per_bdx +
                                    j] +
                     tx * vec_size;
    }

    // load k tiles
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          k_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          paged_kv.k_data + kv_offset[j],
          (((iter + num_stages_smem) * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size);
    }
    cp_async::commit_group();

    // update m/d/o states
    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
    update_local_state<vec_size, bdx, bdy * tile_size_per_bdx>(
        v_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim, s, stage_idx, st, tx);
    if (compute_downdate) {
      update_local_state<vec_size, bdx, bdy * tile_size_per_bdx>(
          v_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim, s_downdate,
          stage_idx, st_downdate, tx);
    }
    block.sync();

    // load v tiles
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
          v_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          paged_kv.v_data + kv_offset[j],
          (((iter + num_stages_smem) * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size);
    }
    cp_async::commit_group();
    stage_idx = (stage_idx + 1) % num_stages_smem;
  }
  cp_async::wait_group<0>();
  block.sync();

  // sync local state of all warps inside a threadblock
  sync_state<vec_size, bdx, bdy, bdz>(variant, st, reinterpret_cast<float*>(smem), smem_md, tx, ty,
                                      tz);
  if (compute_downdate) {
    sync_state<vec_size, bdx, bdy, bdz>(variant, st_downdate, reinterpret_cast<float*>(smem),
                                        smem_md, tx, ty, tz);
  }
  if constexpr (variant.use_softmax) {
    st.normalize();
    if (compute_downdate) {
      st_downdate.normalize();
    }
  }

  if (tz == 0) {
    st.o.cast_store(o + (bx * num_qo_heads + qo_head_idx) * head_dim + tx * vec_size);
    // write lse
    if (lse != nullptr) {
      lse[bx * num_qo_heads + qo_head_idx] = st.get_lse();
    }
    if (compute_downdate && downdate_o != nullptr) {
      st_downdate.o.cast_store(downdate_o + (bx * num_qo_heads + qo_head_idx) * head_dim +
                               tx * vec_size);
    }
    if (compute_downdate && downdate_lse != nullptr) {
      downdate_lse[bx * num_qo_heads + qo_head_idx] = st_downdate.get_lse();
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <PosEncodingMode POS_ENCODING_MODE, uint32_t num_stages_smem, uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz, typename AttentionVariant,
          typename Params>
__global__ void BatchMACAttentionDecodeWithPagedKVCacheKernel(const __grid_constant__ Params params) {
  extern __shared__ uint8_t smem[];
  BatchMACAttentionDecodeWithPagedKVCacheDevice<POS_ENCODING_MODE, num_stages_smem, tile_size_per_bdx,
                                        vec_size, bdx, bdy, bdz, AttentionVariant>(params, smem);
}

/* final merge with cache */
template <typename T> struct PairTraits;

template <> struct PairTraits<float> {
  using PairT = float2;
  __device__ static inline PairT  load_pair(const float* p, int i) { return reinterpret_cast<const PairT*>(p)[i]; }
  __device__ static inline void   store_pair(float* p, int i, PairT v) { reinterpret_cast<PairT*>(p)[i] = v; }
  __device__ static inline float2 to_f2(PairT v) { return v; }
  __device__ static inline PairT  from_f2(float2 v) { return v; }
  __device__ static inline PairT  zero() { return make_float2(0.f, 0.f); }
};

template <> struct PairTraits<__half> {
  using PairT = __half2;
  __device__ static inline PairT  load_pair(const __half* p, int i) { return reinterpret_cast<const PairT*>(p)[i]; }
  __device__ static inline void   store_pair(__half* p, int i, PairT v) { reinterpret_cast<PairT*>(p)[i] = v; }
  __device__ static inline float2 to_f2(PairT v) { return __half22float2(v); }
  __device__ static inline PairT  from_f2(float2 v) { return __floats2half2_rn(v.x, v.y); }
  __device__ static inline PairT  zero() { return __float2half2_rn(0.f); }
};

template <> struct PairTraits<__nv_bfloat16> {
  using PairT = __nv_bfloat162;
  __device__ static inline PairT  load_pair(const __nv_bfloat16* p, int i) { return reinterpret_cast<const PairT*>(p)[i]; }
  __device__ static inline void   store_pair(__nv_bfloat16* p, int i, PairT v) { reinterpret_cast<PairT*>(p)[i] = v; }
  __device__ static inline float2 to_f2(PairT v) { return __bfloat1622float2(v); }
  __device__ static inline PairT  from_f2(float2 v) { return __floats2bfloat162_rn(v.x, v.y); }
  __device__ static inline PairT  zero() { return __floats2bfloat162_rn(0.f, 0.f); }
};

__device__ __forceinline__ bool isfinite_f(float x) { return isfinite(x); }


/*!
 * \brief FlashAttention MACAttention decoding cuda kernel with paged kv-cache for multiple requests
 * \tparam pos_encoding_mode The positional encoding mode
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam bdy A template integer indicates the block size in y dimension
 * \tparam bdz A template integer indicates the block size in z dimension
 * \tparam DTypeQ A template type indicates the query data type
 * \tparam DTypeKV A template type indicates the key-value data type
 * \tparam DTypeO A template type indicates the output data type
 * \tparam IdType A template type indicates the index data type
 * \param q [batch_size, num_qo_heads, head_dim] The query matrix
 * \param paged_kv The paged kv-cache data structure
 * \param o [num_qo_heads, head_dim] The output matrix
 * \param tmp Used-allocated temporary buffer
 * \param lse The logsumexp values
 * \param sm_scale A float indicates the scale applied to pre-softmax logits
 * \param rope_rcp_scale A floating number indicate the reciprocal
 *   of scaling ratio used in PI(Position Interpolation) for RoPE (Rotary
 *   Positional Embeddings)
 * \param rope_rcp_theta A floating number indicate the reciprocal
 *   of "theta" used in RoPE (Rotary Positional Embeddings)
 */
// -------------------------------------------------------------
// Kernel body: dual-path (MAIN + DOWNDATE) with disjoint scratch
// -------------------------------------------------------------
template <PosEncodingMode POS_ENCODING_MODE, uint32_t num_stages_smem, uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz, typename AttentionVariant,
          typename Params>
__device__ __inline__ void MACDecodeWithPagedKVCacheDevice(
    const Params& params, uint8_t smem[], const uint32_t bx = blockIdx.x,
    const uint32_t by = blockIdx.y, const uint32_t tx = threadIdx.x,
    const uint32_t ty = threadIdx.y, const uint32_t tz = threadIdx.z) {
  auto block = cg::this_thread_block();
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO  = typename Params::DTypeO;
  using IdType  = typename Params::IdType;

  const DTypeQ* q = params.q;
  DTypeO*       o = params.o;
  float*        lse = params.lse;

  // optional downdate outputs (ignored if null)
  DTypeO* downdated_o   = params.downdated_o;
  float*  downdated_lse = params.downdated_lse;

  const auto paged_kv = params.paged_kv;
  const bool* block_valid_mask = params.block_valid_mask;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const bool partition_kv = params.partition_kv;

  constexpr uint32_t head_dim = bdx * vec_size;
  const uint32_t H = params.paged_kv.num_heads;

  // ---------- CTA mapping ----------
  uint32_t batch_idx, kv_head_idx, kv_tile_idx;
  if (params.kv_head_indices != nullptr) {
    const uint32_t cta = blockIdx.x;
    if (params.block_valid_mask && !params.block_valid_mask[cta]) return;
    batch_idx   = params.request_indices[cta];
    kv_head_idx = params.kv_head_indices[cta];
    kv_tile_idx = params.kv_tile_indices[cta];
  } else {
    const uint32_t bx = blockIdx.x;
    const uint32_t base_bx = bx / H;
    kv_head_idx = bx - base_bx * H;
    if (params.block_valid_mask && !params.block_valid_mask[base_bx]) return;
    batch_idx   = params.request_indices[base_bx];
    kv_tile_idx = params.kv_tile_indices[base_bx];
  }
  const uint32_t qo_head_idx = kv_head_idx * bdy + threadIdx.y;

  // ---------- lengths, chunks, RoPE ----------
  const uint32_t kv_chunk_size = *(params.kv_chunk_size_ptr);
  const uint32_t kv_len        = paged_kv.get_length(batch_idx);
  const uint32_t max_chunk_size = partition_kv ? kv_chunk_size : kv_len;
  const uint32_t chunk_start    = partition_kv ? kv_tile_idx * max_chunk_size : 0;
  const uint32_t chunk_end      = partition_kv ? min((kv_tile_idx + 1) * max_chunk_size, kv_len) : kv_len;
  const uint32_t chunk_size     = chunk_end - chunk_start;
  const uint32_t rope_offset    = (paged_kv.rope_pos_offset == nullptr ? 0 : paged_kv.rope_pos_offset[batch_idx]);

  // Robust downdate window start (clamped)
  const uint32_t dd_start = rope_offset +
      ((kv_len > (params.downdate_range + 1)) ? (kv_len - params.downdate_range - 1) : 0u);

  AttentionVariant variant(params, batch_idx, smem);

  // ---------- Shared memory layout ----------
  // Rings: [K ring][V ring][kv_offset slab]
  constexpr size_t RING_ELEMS = size_t(num_stages_smem) * tile_size_per_bdx * bdy * bdz * head_dim;

  DTypeKV* k_smem = reinterpret_cast<DTypeKV*>(smem);
  DTypeKV* v_smem = reinterpret_cast<DTypeKV*>(smem + RING_ELEMS * sizeof(DTypeKV));
  size_t*  kv_offset_smem = reinterpret_cast<size_t*>(
      smem + 2 * RING_ELEMS * sizeof(DTypeKV));

  float*   smem_md = (float*)(smem + 2 * RING_ELEMS * sizeof(DTypeKV));

  // ---------- Load Q (+RoPE if needed) ----------
  vec_t<float, vec_size> q_vec;
  vec_t<float, vec_size> freq;
  const uint32_t q_stride_n = params.q_stride_n;
  const uint32_t q_stride_h = params.q_stride_h;

  if constexpr (POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
    const IdType* q_rope_offset = nullptr;
    if constexpr (has_decode_maybe_q_rope_offset_v<Params>) {
      q_rope_offset = params.decode_maybe_q_rope_offset;
    }
    const int32_t q_rope_offset_val = (q_rope_offset == nullptr ? (kv_len - 1) : q_rope_offset[batch_idx]);
    const float rope_rcp_scale = params.rope_rcp_scale;
    const float rope_rcp_theta = params.rope_rcp_theta;

#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      freq[i] = rope_rcp_scale *
                __powf(rope_rcp_theta,
                       float(2 * ((tx * vec_size + i) % (head_dim / 2))) / float(head_dim));
    }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif
    q_vec = vec_apply_llama_rope<vec_size, bdx>(
        q + batch_idx * q_stride_n + qo_head_idx * q_stride_h, freq, q_rope_offset_val);
  } else {
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif
    q_vec.cast_load(q + batch_idx * q_stride_n + qo_head_idx * q_stride_h + tx * vec_size);
  }

  // ---------- Prefetch first stages ----------
  uint32_t stage_idx = 0;
  constexpr uint32_t vec_bits = sizeof(DTypeKV) * vec_size * 8;
  const IdType last_indptr = paged_kv.indptr[paged_kv.batch_size];

  static_assert(num_stages_smem <= bdx, "num_stages_smem must be <= bdx");
  uint32_t packed_page_iter_base = paged_kv.indptr[batch_idx] * paged_kv.page_size + chunk_start;

#pragma unroll
  for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
    uint32_t qdiv, rmod;
    paged_kv.page_size.divmod(
        packed_page_iter_base + ((j * bdz + tz) * bdy + ty) * bdx + tx, qdiv, rmod);
    kv_offset_smem[((j * bdz + tz) * bdy + ty) * bdx + tx] =
        paged_kv.protective_get_kv_offset(qdiv, kv_head_idx, rmod, 0, last_indptr);
  }
  block.sync();

  size_t kv_offset[tile_size_per_bdx];

  state_t<vec_size> st_main, st_dd;
  float s_main[bdy * tile_size_per_bdx];
  float s_dd  [bdy * tile_size_per_bdx];

#pragma unroll
  for (uint32_t iter = 0; iter < num_stages_smem; ++iter) {
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      kv_offset[j] =
          kv_offset_smem[((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j] + tx * vec_size;
    }
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          k_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim + tx * vec_size,
          paged_kv.k_data + kv_offset[j],
          ((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size);
    }
    cp_async::commit_group();
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
          v_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim + tx * vec_size,
          paged_kv.v_data + kv_offset[j],
          ((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size);
    }
    cp_async::commit_group();
    stage_idx = (stage_idx + 1) % num_stages_smem;
  }

  // ---------- Main loop ----------
#pragma unroll 2
  for (uint32_t iter = 0; iter < ceil_div(chunk_size, tile_size_per_bdx * bdy * bdz); ++iter) {
    if ((iter + num_stages_smem) % bdx == 0) {
#pragma unroll
      for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
        uint32_t qdiv, rmod;
        paged_kv.page_size.divmod(
            packed_page_iter_base + ((iter + num_stages_smem) * tile_size_per_bdx * bdy * bdz +
                                     ((j * bdz + tz) * bdy + ty) * bdx + tx),
            qdiv, rmod);
        kv_offset_smem[((j * bdz + tz) * bdy + ty) * bdx + tx] =
            paged_kv.protective_get_kv_offset(qdiv, kv_head_idx, rmod, 0, last_indptr);
      }
    }

    // Compute QK on resident stage
    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();

    compute_mac_attention_qk_opt_dual<POS_ENCODING_MODE, vec_size, bdx, bdy * tile_size_per_bdx>(
        params, variant, batch_idx,
        k_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim,
        q_vec, freq,
        /*kv_idx_base*/ rope_offset + chunk_start + iter * tile_size_per_bdx * bdy * bdz,
        /*iter_base*/   iter * tile_size_per_bdx * bdy * bdz,
        /*iter_bound*/  chunk_size,
        qo_head_idx, kv_head_idx,
        /*OUT*/ s_main, st_main, s_dd, st_dd,
        tx, ty, tz,
        /*main start*/ params.attn_start_pos,
        /*dd   start*/ dd_start);

    block.sync();

    // Next-stage offsets
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      kv_offset[j] = kv_offset_smem[((((iter + num_stages_smem) % bdx) * bdz + tz) * bdy + ty) *
                                        tile_size_per_bdx + j] + tx * vec_size;
    }

    // Load next-stage K
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          k_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim + tx * vec_size,
          paged_kv.k_data + kv_offset[j],
          (((iter + num_stages_smem) * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size);
    }
    cp_async::commit_group();

    // Update local states (MAIN, then DD) on resident V
    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
    update_local_state<vec_size, bdx, bdy * tile_size_per_bdx>(
        v_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim, s_main, stage_idx, st_main, tx);
    update_local_state<vec_size, bdx, bdy * tile_size_per_bdx>(
        v_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim, s_dd,   stage_idx, st_dd,   tx);
    block.sync();

    // Load next-stage V
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
          v_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim + tx * vec_size,
          paged_kv.v_data + kv_offset[j],
          (((iter + num_stages_smem) * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size);
    }
    cp_async::commit_group();

    stage_idx = (stage_idx + 1) % num_stages_smem;
  }

  cp_async::wait_group<0>();
  block.sync();

  // ---------- Block-wide reductions (disjoint scratch, no barrier in between) ----------
  // MAIN: uses K ring + kv_offset slab
  sync_state<vec_size, bdx, bdy, bdz>(variant, st_main, reinterpret_cast<float*>(smem), smem_md, tx, ty, tz);
  if constexpr (AttentionVariant::use_softmax) st_main.normalize();

  block.sync(); // a must, remove this will destroy everything

  // DOWNDATE: uses V ring + kv_offset slab (now free)
  sync_state<vec_size, bdx, bdy, bdz>(variant, st_dd, reinterpret_cast<float*>(smem), smem_md, tx, ty, tz);
  if constexpr (AttentionVariant::use_softmax) st_dd.normalize();


  // ---------- Final stores (MAIN unchanged) ----------
  if (partition_kv) {
    const uint32_t row = params.o_indptr[batch_idx] + kv_tile_idx;
    if (tz == 0) {
      // MAIN
      st_main.o.cast_store(o + (row * num_qo_heads + qo_head_idx) * head_dim + tx * vec_size);
      if (lse) lse[row * num_qo_heads + qo_head_idx] = st_main.get_lse();

      // DOWNDATE
      if (downdated_o)   st_dd.o.cast_store(downdated_o + (row * num_qo_heads + qo_head_idx) * head_dim + tx * vec_size);
      if (downdated_lse) downdated_lse[row * num_qo_heads + qo_head_idx] = st_dd.get_lse();
    }
  } else {
    if (tz == 0) {
      // MAIN
      st_main.o.cast_store(o + (batch_idx * num_qo_heads + qo_head_idx) * head_dim + tx * vec_size);
      if (lse) lse[batch_idx * num_qo_heads + qo_head_idx] = st_main.get_lse() * CUDART_LN2_F;

      // DOWNDATE
      if (downdated_o)   st_dd.o.cast_store(downdated_o + (batch_idx * num_qo_heads + qo_head_idx) * head_dim + tx * vec_size);
      if (downdated_lse) downdated_lse[batch_idx * num_qo_heads + qo_head_idx] = st_dd.get_lse() * CUDART_LN2_F;
    }
  }

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}



 

template <PosEncodingMode POS_ENCODING_MODE, uint32_t num_stages_smem, uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz, typename AttentionVariant,
          typename Params>
__global__ void MACDecodeWithPagedKVCacheKernel(const __grid_constant__ Params params) {
  extern __shared__ uint8_t smem[];
  MACDecodeWithPagedKVCacheDevice<POS_ENCODING_MODE, num_stages_smem, tile_size_per_bdx,
                                           vec_size, bdx, bdy, bdz, AttentionVariant>(params, smem);
}


// -----------------------------------------------------------------------------
// Compute the masked starting position in the same coordinate system as `pos`
// used inside the QK loop (i.e., including rope_offset if present).
//   - kv_len       : tokens in KV cache for this request
//   - window_left  : tokens of workload to keep from the *right* of the sequence
//   - rope_offset  : optional per-request position offset used for RoPE and pos
// Returns: start_pos_for_mask = rope_offset + max(0, kv_len - min(window_left, kv_len))
// -----------------------------------------------------------------------------
static __device__ __forceinline__
uint32_t WindowStartPosMask(uint32_t kv_len, uint32_t window_left, uint32_t rope_offset) {
  const uint32_t wl_eff = min(kv_len, window_left + 1u);
  const uint32_t start  = kv_len - wl_eff;
  return rope_offset + start;
}

/*!
 * \brief FlashAttention RectificationCache decoding cuda kernel with paged kv-cache for multiple requests
 * \tparam pos_encoding_mode The positional encoding mode
 * \tparam vec_size A template integer indicates the vector size
 * \tparam bdx A template integer indicates the block size in x dimension
 * \tparam bdy A template integer indicates the block size in y dimension
 * \tparam bdz A template integer indicates the block size in z dimension
 * \tparam DTypeQ A template type indicates the query data type
 * \tparam DTypeKV A template type indicates the key-value data type
 * \tparam DTypeO A template type indicates the output data type
 * \tparam IdType A template type indicates the index data type
 * \param q [batch_size, num_qo_heads, head_dim] The query matrix
 * \param paged_kv The paged kv-cache data structure
 * \param o [num_qo_heads, head_dim] The output matrix
 * \param tmp Used-allocated temporary buffer
 * \param lse The logsumexp values
 * \param sm_scale A float indicates the scale applied to pre-softmax logits
 * \param rope_rcp_scale A floating number indicate the reciprocal
 *   of scaling ratio used in PI(Position Interpolation) for RoPE (Rotary
 *   Positional Embeddings)
 * \param rope_rcp_theta A floating number indicate the reciprocal
 *   of "theta" used in RoPE (Rotary Positional Embeddings)
 */
template <PosEncodingMode POS_ENCODING_MODE, uint32_t num_stages_smem, uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz, typename AttentionVariant,
          typename Params>
__device__ __inline__ void BatchRectificationCacheDecodeWithPagedKVCacheDevice(
    const Params& params, uint8_t smem[], const uint32_t bx = blockIdx.x,
    const uint32_t by = blockIdx.y, const uint32_t tx = threadIdx.x,
    const uint32_t ty = threadIdx.y, const uint32_t tz = threadIdx.z) {
  auto block = cg::this_thread_block();
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  const DTypeQ* q = params.q;
  DTypeO* o = params.o;
  float* lse = params.lse;

  const auto paged_kv = params.paged_kv;
  const bool* block_valid_mask = params.block_valid_mask;
  const uint32_t padded_batch_size = params.padded_batch_size;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const bool partition_kv = params.partition_kv;

  constexpr uint32_t head_dim = bdx * vec_size;
  const uint32_t H = params.paged_kv.num_heads;
  const uint32_t cta = blockIdx.x;
  uint32_t batch_idx, kv_head_idx, kv_tile_idx;

  // Head-packed if kv_head_indices != nullptr
  if (params.kv_head_indices != nullptr) {
    const uint32_t cta = blockIdx.x;
    if (params.block_valid_mask && !params.block_valid_mask[cta]) return;
    batch_idx = params.request_indices[cta];
    kv_head_idx = params.kv_head_indices[cta];
    kv_tile_idx = params.kv_tile_indices[cta];
  } else {
    // Uniform flattened fallback (gdy=1, grid.x = X*H)
    const uint32_t bx = blockIdx.x;
    const uint32_t base_bx = bx / H;  // tile-slot
    kv_head_idx = bx - base_bx * H;   // [0..H-1]
    if (params.block_valid_mask && !params.block_valid_mask[base_bx]) return;
    batch_idx = params.request_indices[base_bx];
    kv_tile_idx = params.kv_tile_indices[base_bx];
  }

  const uint32_t qo_head_idx = kv_head_idx * bdy + threadIdx.y;

  // NOTE(Zihao): when CUDAGraph is enabled, we will launch more blocks than
  // the actual batch size, so we need to check if the current batch is valid
  const uint32_t kv_chunk_size = *(params.kv_chunk_size_ptr);
  const uint32_t kv_len = paged_kv.get_length(batch_idx);
  const uint32_t max_chunk_size = partition_kv ? kv_chunk_size : kv_len;
  const uint32_t chunk_start = partition_kv ? kv_tile_idx * max_chunk_size : 0;
  const uint32_t chunk_end =
      partition_kv ? min((kv_tile_idx + 1) * max_chunk_size, kv_len) : kv_len;
  const uint32_t chunk_size = chunk_end - chunk_start;
  const uint32_t rope_offset =
      paged_kv.rope_pos_offset == nullptr ? 0 : paged_kv.rope_pos_offset[batch_idx];
  const uint32_t start_pos_for_mask =
      WindowStartPosMask(/*kv_len=*/kv_len,
                        /*window_left=*/static_cast<uint32_t>(params.window_left),
                        /*rope_offset=*/rope_offset);

  AttentionVariant variant(params, batch_idx, smem);
  DTypeKV* k_smem = (DTypeKV*)smem;
  DTypeKV* v_smem = (DTypeKV*)(smem + num_stages_smem * tile_size_per_bdx * bdy * bdz * head_dim *
                                          sizeof(DTypeKV));
  size_t* kv_offset_smem = (size_t*)(smem + 2 * num_stages_smem * tile_size_per_bdx * bdy * bdz *
                                                head_dim * sizeof(DTypeKV));
  float* smem_md = (float*)(smem + 2 * num_stages_smem * tile_size_per_bdx * bdy * bdz * head_dim *
                                       sizeof(DTypeKV));

  vec_t<float, vec_size> q_vec;
  vec_t<float, vec_size> freq;
  const uint32_t q_stride_n = params.q_stride_n;
  const uint32_t q_stride_h = params.q_stride_h;
  if constexpr (POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
    const IdType* q_rope_offset = nullptr;
    if constexpr (has_decode_maybe_q_rope_offset_v<Params>) {
      q_rope_offset = params.decode_maybe_q_rope_offset;
    }
    int32_t q_rope_offset_val = q_rope_offset == nullptr ? (kv_len - 1) : q_rope_offset[batch_idx];
    const float rope_rcp_scale = params.rope_rcp_scale;
    const float rope_rcp_theta = params.rope_rcp_theta;

#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      freq[i] = rope_rcp_scale *
                __powf(rope_rcp_theta,
                       float(2 * ((tx * vec_size + i) % (head_dim / 2))) / float(head_dim));
    }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif
    // apply rotary embedding to q matrix
    q_vec = vec_apply_llama_rope<vec_size, bdx>(
        q + batch_idx * q_stride_n + qo_head_idx * q_stride_h, freq, q_rope_offset_val);
  } else {
// do not apply rotary embedding to q matrix
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif
    q_vec.cast_load(q + batch_idx * q_stride_n + qo_head_idx * q_stride_h + tx * vec_size);
  }

  // preload k/v tiles
  uint32_t stage_idx = 0;
  constexpr uint32_t vec_bits = sizeof(DTypeKV) * vec_size * 8;
  const IdType last_indptr = paged_kv.indptr[paged_kv.batch_size];

  static_assert(num_stages_smem <= bdx);
  uint32_t packed_page_iter_base = paged_kv.indptr[batch_idx] * paged_kv.page_size + chunk_start;
#pragma unroll
  for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
    uint32_t q, r;
    paged_kv.page_size.divmod(packed_page_iter_base + ((j * bdz + tz) * bdy + ty) * bdx + tx, q, r);
    kv_offset_smem[((j * bdz + tz) * bdy + ty) * bdx + tx] =
        paged_kv.protective_get_kv_offset(q, kv_head_idx, r, 0, last_indptr);
  }
  block.sync();

  size_t kv_offset[tile_size_per_bdx];
#pragma unroll
  for (uint32_t iter = 0; iter < num_stages_smem; ++iter) {
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      kv_offset[j] =
          kv_offset_smem[((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j] + tx * vec_size;
    }
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          k_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          paged_kv.k_data + kv_offset[j],
          ((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size);
    }
    cp_async::commit_group();
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
          v_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          paged_kv.v_data + kv_offset[j],
          ((iter * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size);
    }
    cp_async::commit_group();
    stage_idx = (stage_idx + 1) % num_stages_smem;
  }

  state_t<vec_size> st;
  float s[bdy * tile_size_per_bdx];

#pragma unroll 2
  for (uint32_t iter = 0; iter < ceil_div(chunk_size, tile_size_per_bdx * bdy * bdz); ++iter) {
    if ((iter + num_stages_smem) % bdx == 0) {
#pragma unroll
      for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
        uint32_t q, r;
        paged_kv.page_size.divmod(
            packed_page_iter_base + ((iter + num_stages_smem) * tile_size_per_bdx * bdy * bdz +
                                     ((j * bdz + tz) * bdy + ty) * bdx + tx),
            q, r);
        kv_offset_smem[((j * bdz + tz) * bdy + ty) * bdx + tx] =
            paged_kv.protective_get_kv_offset(q, kv_head_idx, r, 0, last_indptr);
      }
    }
    // compute qk
    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
    compute_rectification_cache_qk<POS_ENCODING_MODE, vec_size, bdx, bdy * tile_size_per_bdx>(
        params, variant, batch_idx,
        k_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim, q_vec, freq,
        rope_offset + chunk_start + iter * tile_size_per_bdx * bdy * bdz,
        iter * tile_size_per_bdx * bdy * bdz, chunk_size, qo_head_idx, kv_head_idx, s, st, tx, ty,
        tz, start_pos_for_mask);
    block.sync();

#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      kv_offset[j] = kv_offset_smem[((((iter + num_stages_smem) % bdx) * bdz + tz) * bdy + ty) *
                                        tile_size_per_bdx +
                                    j] +
                     tx * vec_size;
    }

    // load k tiles
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
          k_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          paged_kv.k_data + kv_offset[j],
          (((iter + num_stages_smem) * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size);
    }
    cp_async::commit_group();

    // update m/d/o states
    cp_async::wait_group<2 * num_stages_smem - 1>();
    block.sync();
    update_local_state<vec_size, bdx, bdy * tile_size_per_bdx>(
        v_smem + (stage_idx * bdz + tz) * bdy * tile_size_per_bdx * head_dim, s, stage_idx, st, tx);
    block.sync();

    // load v tiles
#pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
      cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
          v_smem + (((stage_idx * bdz + tz) * bdy + ty) * tile_size_per_bdx + j) * head_dim +
              tx * vec_size,
          paged_kv.v_data + kv_offset[j],
          (((iter + num_stages_smem) * bdz + tz) * bdy + ty) * tile_size_per_bdx + j < chunk_size);
    }
    cp_async::commit_group();
    stage_idx = (stage_idx + 1) % num_stages_smem;
  }
  cp_async::wait_group<0>();
  block.sync();

  // sync local state of all warps inside a threadblock
  sync_state<vec_size, bdx, bdy, bdz>(variant, st, reinterpret_cast<float*>(smem), smem_md, tx, ty,
                                      tz);
  if constexpr (variant.use_softmax) {
    st.normalize();
  }

  // ---- final store (split vs non-split) ----
  if (partition_kv) {
    const uint32_t C_tokens = *(params.kv_chunk_size_ptr);
    const uint32_t W_i      = min(kv_len, params.window_left + 1u);  // <<< +1 here
    const uint32_t tile0_i  = (kv_len - W_i) / C_tokens;        // FLOOR by integer division

    // kv_tile_idx is ABSOLUTE tile id j
    const uint32_t row = params.o_indptr[batch_idx] + (kv_tile_idx - tile0_i);

    if (tz == 0) {
      st.o.cast_store(o + (row * num_qo_heads + qo_head_idx) * head_dim + tx * vec_size);
      if (lse != nullptr) lse[row * num_qo_heads + qo_head_idx] = st.get_lse();
    }
  } else {
    // unchanged non-split path
    if (tz == 0) {
      st.o.cast_store(o + (batch_idx * num_qo_heads + qo_head_idx) * head_dim + tx * vec_size);
      if (lse != nullptr) lse[batch_idx * num_qo_heads + qo_head_idx] = st.get_lse();
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <PosEncodingMode POS_ENCODING_MODE, uint32_t num_stages_smem, uint32_t tile_size_per_bdx,
          uint32_t vec_size, uint32_t bdx, uint32_t bdy, uint32_t bdz, typename AttentionVariant,
          typename Params>
__global__ void BatchRectificationCacheDecodeWithPagedKVCacheKernel(const __grid_constant__ Params params) {
  extern __shared__ uint8_t smem[];
  BatchRectificationCacheDecodeWithPagedKVCacheDevice<POS_ENCODING_MODE, num_stages_smem, tile_size_per_bdx,
                                           vec_size, bdx, bdy, bdz, AttentionVariant>(params, smem);
}

/*!
 * \brief Get the heuristic number of threads per threadblock
 * \param group_size The number of qo heads that maps to the same kv head in GQA.
 * \param sizeof_dtype The size (in terms of bytes) of the input data type
 */
constexpr uint32_t get_heuristic_num_threads(uint32_t group_size, uint32_t sizeof_dtype) {
  if (group_size == 8U) {
    if (sizeof_dtype == 1U) {
      return 256U;  // not enough registers for 512 threads
    } else {
      return 512U;
    }
  } else {
    return 128U;
  }
}

/*!
 * \brief FlashAttention decoding with kv-cache for a single request
 * \tparam DTypeQ A template type indicates the query data type
 * \tparam DTypeKV A template type indicates the key-value data type
 * \tparam DTypeO A template type indicates the output data type
 * \param q The query matrix, shape: [num_qo_heads, head_dim]
 * \param k The key matrix in kv-cache, shape: [seq_len, num_kv_heads, head_dim]
 *   for NHD layout, [num_kv_heads, seq_len, head_dim] for HND layout
 * \param v The value matrix in kv-cache, shape: [seq_len, num_kv_heads,
 *   head_dim] for NHD layout, [num_kv_heads, seq_len, head_dim] for HND layout
 * \param o The output matrix, shape: [num_qo_heads, head_dim]
 * \param tmp Used-allocated temporary buffer
 * \param num_qo_heads A integer indicates the number of heads of query and output
 * \param num_kv_heads A integer indicates the number of heads of key and value
 * \param seq_len A integer indicates the sequence length
 * \param head_dim A integer indicates the head dimension
 * \param pos_encoding_mode The positional encoding mode
 * \param rope_scale The scaling factor used in RoPE Interpolation
 * \param rope_theta The theta used in RoPE
 * \param stream The cuda stream to launch the kernel
 * \return status Indicates whether CUDA calls are successful
 */
template <uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE, typename AttentionVariant,
          typename Params>
cudaError_t SingleDecodeWithKVCacheDispatched(Params params, typename Params::DTypeO* tmp,
                                              cudaStream_t stream) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const uint32_t num_kv_heads = params.num_kv_heads;
  const uint32_t seq_len = params.kv_len;

  constexpr uint32_t vec_size = std::max(16UL / sizeof(DTypeKV), HEAD_DIM / 32UL);
  constexpr uint32_t bdx = HEAD_DIM / vec_size;
  auto compute_capacity = GetCudaComputeCapability();
  static_assert(bdx <= 32U);
  DISPATCH_GQA_GROUP_SIZE(num_qo_heads / num_kv_heads, GROUP_SIZE, {
    constexpr uint32_t bdy = GROUP_SIZE;
    constexpr uint32_t num_threads =
        std::max(get_heuristic_num_threads(GROUP_SIZE, sizeof(DTypeKV)), bdx * bdy);
    constexpr uint32_t bdz = num_threads / (bdx * bdy);
    constexpr uint32_t tile_size_per_bdx = GROUP_SIZE == 1 ? (sizeof(DTypeKV) == 1 ? 2U : 8U) : 1U;
    DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, {
      const uint32_t smem_size =
          2U * NUM_STAGES_SMEM * bdy * tile_size_per_bdx * bdz * HEAD_DIM * sizeof(DTypeKV) +
          2U * bdy * bdz * sizeof(float);
      auto kernel =
          SingleDecodeWithKVCacheKernel<POS_ENCODING_MODE, NUM_STAGES_SMEM, tile_size_per_bdx,
                                        vec_size, bdx, bdy, bdz, AttentionVariant, Params>;
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

      if (seq_len <= 256 || tmp == nullptr) {
        // no need to use partition-kv kernel
        dim3 nblks = dim3(1, num_kv_heads);
        dim3 nthrs = dim3(bdx, bdy, bdz);
        params.kv_chunk_size = seq_len;
        void* args[] = {(void*)&params};
        FLASHINFER_CUDA_CALL(
            cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
      } else {
        // use partition-kv kernel
        int num_blocks_per_sm = 0;
        int num_sm = 0;
        int dev_id = 0;
        FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
        FLASHINFER_CUDA_CALL(
            cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
        FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &num_blocks_per_sm, kernel, num_threads, smem_size));
        uint32_t max_grid_size = uint32_t(num_blocks_per_sm) * uint32_t(num_sm);
        uint32_t max_num_kv_chunks = max_grid_size / num_kv_heads;
        uint32_t kv_chunk_size = max(ceil_div(seq_len, max_num_kv_chunks), 256);
        uint32_t num_chunks = ceil_div(seq_len, kv_chunk_size);
        dim3 nblks = dim3(num_chunks, num_kv_heads);
        if (nblks.x == 0 || nblks.y == 0) {
          std::ostringstream err_msg;
          err_msg << "Invalid kernel configuration: nblks=(" << nblks.x << "," << nblks.y << ")";
          FLASHINFER_ERROR(err_msg.str());
        }
        dim3 nthrs = dim3(bdx, bdy, bdz);
        float* tmp_lse = (float*)(tmp + num_chunks * num_qo_heads * HEAD_DIM);
        auto o = params.o;
        auto lse = params.lse;
        params.o = tmp;
        params.lse = tmp_lse;
        params.kv_chunk_size = kv_chunk_size;
        void* args[] = {(void*)&params};
        FLASHINFER_CUDA_CALL(
            cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        if constexpr (AttentionVariant::use_softmax) {
          FLASHINFER_CUDA_CALL(
              MergeStates(tmp, tmp_lse, o, lse, num_chunks, 1, num_qo_heads, HEAD_DIM, stream));
        } else {
          FLASHINFER_CUDA_CALL(AttentionSum(tmp, o, num_chunks, 1, num_qo_heads, HEAD_DIM, stream));
        }
      }
    });
  });
  return cudaSuccess;
}

template <uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE, typename AttentionVariant,
          typename Params>
cudaError_t BatchDecodeWithPagedKVCacheDispatched(Params params, typename Params::DTypeO* tmp_v,
                                                  float* tmp_s, bool enable_pdl,
                                                  cudaStream_t stream) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const uint32_t num_kv_heads = params.paged_kv.num_heads;
  const uint32_t padded_batch_size = params.padded_batch_size;

  constexpr uint32_t vec_size = std::max(16UL / sizeof(DTypeKV), HEAD_DIM / 32UL);
  auto compute_capacity = GetCudaComputeCapability();
  constexpr uint32_t bdx = HEAD_DIM / vec_size;
  static_assert(bdx <= 32);
  DISPATCH_GQA_GROUP_SIZE(num_qo_heads / num_kv_heads, GROUP_SIZE, {
    constexpr uint32_t bdy = GROUP_SIZE;
    constexpr uint32_t num_threads = std::max(128U, bdx * bdy);
    constexpr uint32_t bdz = num_threads / (bdx * bdy);
    constexpr uint32_t tile_size_per_bdx = GROUP_SIZE == 1 ? (sizeof(DTypeKV) == 1 ? 2U : 4U) : 1U;
    DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, {
      const uint32_t smem_size =
          2 * NUM_STAGES_SMEM * tile_size_per_bdx * bdy * bdz * HEAD_DIM * sizeof(DTypeKV) +
          std::max(tile_size_per_bdx * num_threads * sizeof(DTypeKV*),
                   2 * bdy * bdz * sizeof(float));
      // printf("bdy %u, bdz %u, tile_size_per_bdx %u, smem size %u\n", bdy, bdz, tile_size_per_bdx,
      // smem_size); printf("BatchDecodeWithPagedKVCacheDispatched...\n");
      auto kernel =
          BatchDecodeWithPagedKVCacheKernel<POS_ENCODING_MODE, NUM_STAGES_SMEM, tile_size_per_bdx,
                                            vec_size, bdx, bdy, bdz, AttentionVariant, Params>;
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      dim3 nblks(padded_batch_size, num_kv_heads);
      dim3 nthrs(bdx, bdy, bdz);
      // PDL launch config
      cudaLaunchAttribute attribute[1];
      cudaLaunchConfig_t config;
      if (enable_pdl) {
        attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute[0].val.programmaticStreamSerializationAllowed = 1;
        config.attrs = attribute;
        config.numAttrs = 1;
        config.gridDim = nblks;
        config.blockDim = nthrs;
        config.dynamicSmemBytes = smem_size;
        config.stream = stream;
      }
      if (tmp_v == nullptr) {
        // do not use partition-kv kernel
        params.partition_kv = false;

        if (enable_pdl) {
          FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, params));
        } else {
          void* args[] = {(void*)&params};
          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        }
      } else {
        // use partition-kv kernel
        params.partition_kv = true;
        auto o = params.o;
        auto lse = params.lse;
        params.o = tmp_v;
        params.lse = tmp_s;
        if (enable_pdl) {
          FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, params));
        } else {
          void* args[] = {(void*)&params};
          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        }
        if constexpr (AttentionVariant::use_softmax) {
          FLASHINFER_CUDA_CALL(VariableLengthMergeStates(
              tmp_v, tmp_s, params.o_indptr, o, lse, params.paged_kv.batch_size, nullptr,
              num_qo_heads, HEAD_DIM, enable_pdl, stream));
        } else {
          FLASHINFER_CUDA_CALL(
              VariableLengthAttentionSum(tmp_v, params.o_indptr, o, params.paged_kv.batch_size,
                                         nullptr, num_qo_heads, HEAD_DIM, enable_pdl, stream));
        }
      }
    });
  });
  return cudaSuccess;
}

template <uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE, typename AttentionVariant,
          typename Params>
cudaError_t BatchMACAttentionDecodeWithPagedKVCacheDispatched(
    Params params, typename Params::DTypeO* tmp_v,
                                                      float* tmp_s,
                                                      typename Params::DTypeO* tmp_v_downdate,
                                                      float* tmp_s_downdate, bool enable_pdl,
                                                      cudaStream_t stream) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const uint32_t num_kv_heads = params.paged_kv.num_heads;
  const uint32_t padded_batch_size = params.padded_batch_size;

  constexpr uint32_t vec_size = std::max(16UL / sizeof(DTypeKV), HEAD_DIM / 32UL);
  auto compute_capacity = GetCudaComputeCapability();
  constexpr uint32_t bdx = HEAD_DIM / vec_size;
  static_assert(bdx <= 32);
  DISPATCH_GQA_GROUP_SIZE(num_qo_heads / num_kv_heads, GROUP_SIZE, {
    constexpr uint32_t bdy = GROUP_SIZE;
    constexpr uint32_t num_threads = std::max(128U, bdx * bdy);
    constexpr uint32_t bdz = num_threads / (bdx * bdy);
    constexpr uint32_t tile_size_per_bdx = GROUP_SIZE == 1 ? (sizeof(DTypeKV) == 1 ? 2U : 4U) : 1U;
    DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, {
      const uint32_t smem_size =
          2 * NUM_STAGES_SMEM * tile_size_per_bdx * bdy * bdz * HEAD_DIM * sizeof(DTypeKV) +
          std::max(tile_size_per_bdx * num_threads * sizeof(DTypeKV*),
                   2 * bdy * bdz * sizeof(float));
      // printf("bdy %u, bdz %u, tile_size_per_bdx %u, smem size %u\n", bdy, bdz, tile_size_per_bdx,
      // smem_size); printf("BatchDecodeWithPagedKVCacheDispatched...\n");
      auto kernel = BatchMACAttentionDecodeWithPagedKVCacheKernel<POS_ENCODING_MODE, NUM_STAGES_SMEM,
                                                          tile_size_per_bdx, vec_size, bdx, bdy,
                                                          bdz, AttentionVariant, Params>;
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      dim3 nblks(padded_batch_size, num_kv_heads);
      dim3 nthrs(bdx, bdy, bdz);
      // printf("gridSize: (%" PRIu32 ", %" PRIu32 ")\n", padded_batch_size, num_kv_heads);
      // printf("blockSize: (%" PRIu32 ", %" PRIu32 ", %" PRIu32 ")\n", bdx, bdy, bdz);

      // PDL launch config
      cudaLaunchAttribute attribute[1];
      cudaLaunchConfig_t config;
      if (enable_pdl) {
        attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute[0].val.programmaticStreamSerializationAllowed = 1;
        config.attrs = attribute;
        config.numAttrs = 1;
        config.gridDim = nblks;
        config.blockDim = nthrs;
        config.dynamicSmemBytes = smem_size;
        config.stream = stream;
      }
      if (tmp_v == nullptr) {
        // do not use partition-kv kernel
        params.partition_kv = false;

        if (enable_pdl) {
          FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, params));
        } else {
          void* args[] = {(void*)&params};
          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        }
      } else {
        // use partition-kv kernel
        params.partition_kv = true;
        auto o = params.o;
        auto lse = params.lse;
        auto downdate_o = params.downdate_o;
        auto downdate_lse = params.downdate_lse;
        params.o = tmp_v;
        params.lse = tmp_s;
        params.downdate_o = tmp_v_downdate;
        params.downdate_lse = tmp_s_downdate;
        if (enable_pdl) {
          FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, params));
        } else {
          void* args[] = {(void*)&params};
          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        }
        if constexpr (AttentionVariant::use_softmax) {
          FLASHINFER_CUDA_CALL(VariableLengthMergeStates(
              tmp_v, tmp_s, params.o_indptr, o, lse, params.paged_kv.batch_size, nullptr,
              num_qo_heads, HEAD_DIM, enable_pdl, stream));
          if (tmp_v_downdate != nullptr && downdate_o != nullptr) {
            FLASHINFER_CUDA_CALL(VariableLengthMergeStates(
                tmp_v_downdate, tmp_s_downdate, params.o_indptr, downdate_o, downdate_lse,
                params.paged_kv.batch_size, nullptr, num_qo_heads, HEAD_DIM, enable_pdl, stream));
          }
        } else {
          FLASHINFER_CUDA_CALL(
              VariableLengthAttentionSum(tmp_v, params.o_indptr, o, params.paged_kv.batch_size,
                                         nullptr, num_qo_heads, HEAD_DIM, enable_pdl, stream));
          if (tmp_v_downdate != nullptr && downdate_o != nullptr) {
            FLASHINFER_CUDA_CALL(VariableLengthAttentionSum(
                tmp_v_downdate, params.o_indptr, downdate_o, params.paged_kv.batch_size, nullptr,
                num_qo_heads, HEAD_DIM, enable_pdl, stream));
          }
        }
      }
    });
  });
  return cudaSuccess;
}

// ---------------- IN-PLACE merge kernel (one warp per (n,h)) ----------------
template <typename OType>
__global__ void MergeWithUnifiedCacheKernelInplace(
    // Shapes
    int N, int H, int D, int M, int R,
    // Decode outputs (to be updated in-place)
    OType* __restrict__ o_rw,    // [N,H,D]
    float* __restrict__ lse_rw,  // [N,H]
    // Unified cache (bf16/f32)
    const __nv_bfloat16* __restrict__ attn_cache,  // [R,M,H,D]
    const float* __restrict__ lse_cache,           // [R,M,H]
    // Routing
    const int32_t* __restrict__ idx,      // [N,H]
    const uint8_t* __restrict__ hit,      // [N,H] (0/1)
    const int32_t* __restrict__ req_ids,  // [N]   (maps batch->request)
    // Optional mask for padded entries
    const uint8_t* __restrict__ batch_valid_mask  // [N] or nullptr
) {
  constexpr int WARP = 32;
  const int warps_per_block = blockDim.x / WARP;
  const int warp_id = threadIdx.x / WARP;
  const int lane    = threadIdx.x & (WARP - 1);

  const int group = blockIdx.x * warps_per_block + warp_id;
  if (group >= N * H) return;

  const int n = group / H;
  const int h = group - n * H;

  if (batch_valid_mask && batch_valid_mask[n] == 0) return;

  if ((D & 1) != 0) return;  // expect even D
  const int pair_count = D >> 1;

  using P  = PairTraits<OType>;
  using OP = typename P::PairT;

  OType* __restrict__ o_row = o_rw + (n * H + h) * D;

  // --- current side: convert log2 -> ln before merging ---
  const float l2_ln = lse_rw[n * H + h];     // log2-domain from compute
  const bool  has2   = isfinite_f(l2_ln);

  // --- cached side (ln domain) only if hit==1 ---
  const bool hhit = (hit ? (hit[n * H + h] != 0) : false);
  float l1 = -CUDART_INF_F; bool has1 = false;
  const __nv_bfloat162* __restrict__ c_row2 = nullptr;

  if (hhit) {
    const int32_t rq = req_ids ? req_ids[n] : n;   // fallback: req=n
    const int32_t s  = idx[n * H + h];             // slot in [0,M)
    const long long lse_off = ((long long)rq * M + (long long)s) * H + h;  // [R,M,H]
    l1   = lse_cache[lse_off];                     // already ln
    has1 = isfinite_f(l1);

    const long long row_off =
      ((((long long)rq * M + (long long)s) * H + h) * (long long)D);       // [R,M,H,D]
    const __nv_bfloat16* c_row = attn_cache + row_off;
    c_row2 = reinterpret_cast<const __nv_bfloat162*>(c_row);
  }

  const bool both_invalid = (!has1 && !has2);
  const bool only_current = (!has1 &&  has2);
  const bool only_cached  = ( has1 && !has2);
  const bool both_valid   = ( has1 &&  has2);

  if (both_invalid) {
    if (lane == 0) lse_rw[n * H + h] = -CUDART_INF_F;     // ln(-inf)
    for (int p = lane; p < pair_count; p += WARP) {
      P::store_pair(o_row, p, P::zero());
    }
    return;
  }
  if (only_current) {
    // No cache contribution; keep output LSE in ln domain
    if (lane == 0) lse_rw[n * H + h] = l2_ln;
    return;
  }
  if (only_cached) {
    if (lane == 0) lse_rw[n * H + h] = l1;                // ln domain
    for (int p = lane; p < pair_count; p += WARP) {
      float2 cf = __bfloat1622float2(c_row2[p]);
      P::store_pair(o_row, p, P::from_f2(cf));
    }
    return;
  }

  // both_valid: stable merge in ln domain (l1, l2_ln)
  float L=0.f, w1=0.f, w2=0.f;
  if (lane == 0) {
    const float m  = fmaxf(l1, l2_ln);
    const float e1 = __expf(l1    - m);
    const float e2 = __expf(l2_ln - m);
    L  = m + __logf(e1 + e2);       // ln domain
    w1 = __expf(l1    - L);
    w2 = __expf(l2_ln - L);
    lse_rw[n * H + h] = L;          // keep LSE in ln after merge
  }
  L  = __shfl_sync(0xffffffffu, L,  0);
  w1 = __shfl_sync(0xffffffffu, w1, 0);
  w2 = __shfl_sync(0xffffffffu, w2, 0);

  for (int p = lane; p < pair_count; p += WARP) {
    OP opin = P::load_pair(o_row, p);
    float2 of = P::to_f2(opin);                 // current output in fp32
    float2 cf = __bfloat1622float2(c_row2[p]);  // cached row (fp32)
    float2 out;
    out.x = fmaf(w1, cf.x, w2 * of.x);
    out.y = fmaf(w1, cf.y, w2 * of.y);
    P::store_pair(o_row, p, P::from_f2(out));
  }
}

template <typename OType>
static inline cudaError_t LaunchMergeWithUnifiedCacheInplace(
    OType* o, float* lse,
    int N, int H, int D,
    const __nv_bfloat16* attn_cache, const float* lse_cache,
    const int32_t* idx, const uint8_t* hit,
    const int32_t* req_ids, const uint8_t* batch_valid_mask,
    int M, int R, bool enable_pdl, cudaStream_t stream) {

  // Robust no-op if cache inputs are absent
  if (!attn_cache || !lse_cache || !idx || !hit || !o || !lse || N==0 || H==0) {
    return cudaSuccess;
  }
  if ((D & 1) != 0) return cudaErrorInvalidValue; // D must be even

  const int warps_per_block = 8;           // 256 threads
  const dim3 nthrs(warps_per_block * 32);
  const dim3 nblks((N * H + warps_per_block - 1) / warps_per_block);

  void* args[] = {
    &N, &H, &D, &M, &R,
    &o, &lse,
    &attn_cache, &lse_cache,
    &idx, &hit, &req_ids,
    &batch_valid_mask
  };

  if (enable_pdl) {
    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg{};
    cfg.attrs = &attr; cfg.numAttrs = 1;
    cfg.gridDim = nblks; cfg.blockDim = nthrs; cfg.dynamicSmemBytes = 0; cfg.stream = stream;
    return cudaLaunchKernelEx(
      &cfg,
      MergeWithUnifiedCacheKernelInplace<OType>,
      N, H, D, M, R,
      o, lse,
      attn_cache, lse_cache,
      idx, hit, req_ids,
      batch_valid_mask);
  } else {
    return cudaLaunchKernel(
      (void*)MergeWithUnifiedCacheKernelInplace<OType>,
      nblks, nthrs, args, /*smem*/0, stream);
  }
}


// Vectorized, one warp per (n,h); D must be even (pair loads).
// ----- helpers for type conversion -----

template <typename OType>
__global__ void DowndateFromMergedKernel(
    int N, int H, int D,
    const OType* __restrict__ full_o, const float* __restrict__ full_lse, // ln
    const OType* __restrict__ dd_o,   const float* __restrict__ dd_lse,   // ln
    OType* __restrict__ out_o,        float* __restrict__ out_lse,        // ln
    const uint8_t* __restrict__ batch_valid_mask) {

  constexpr int WARP = 32;
  constexpr int WARPS_PER_BLOCK = 8;

  const int warp_id = threadIdx.x / WARP;
  const int lane    = threadIdx.x % WARP;
  const int group   = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  const int groups  = N * H;
  if (group >= groups) return;

  const int n = group / H;
  const int h = group % H;
  if (batch_valid_mask && batch_valid_mask[n] == 0) return;

  const int base = (n * H + h) * D;
  const int idx  = (n * H + h);

  // Read LSEs (ln)
  const float l1 = full_lse ? full_lse[idx] : -CUDART_INF_F;
  const float l2 = dd_lse   ? dd_lse[idx]   : -CUDART_INF_F;

  const bool has1 = isfinite(l1);
  const bool has2 = isfinite(l2);

  float out_l = -CUDART_INF_F, t = 0.f, scale = 0.f;

  if (!has1 && !has2) {
    if (lane == 0 && out_lse) out_lse[idx] = -CUDART_INF_F;
    for (int d = lane; d < D; d += WARP) out_o[base + d] = OType(0);
    return;
  }
  if (!has2) {
    if (lane == 0 && out_lse) out_lse[idx] = l1;
    for (int d = lane; d < D; d += WARP) out_o[base + d] = full_o[base + d];
    return;
  }
  if (!has1) {
    if (lane == 0 && out_lse) out_lse[idx] = -CUDART_INF_F;
    for (int d = lane; d < D; d += WARP) out_o[base + d] = OType(0);
    return;
  }

  // Robust ln downdate
  float diff = l2 - l1;
  float em1  = expm1f(diff);
  bool last     = fabsf(em1) <= 1e-7f;
  bool invalid  = (diff > 0.f) && !last;

  if (last) {
    out_l = -CUDART_INF_F; t = 1.f; scale = 0.f;
  } else if (invalid) {
    out_l = l1; t = 0.f; scale = 1.f;
  } else {
    t = em1 + 1.f;
    float one_minus = fmaxf(-em1, 1e-30f);
    out_l = l1 + __logf(one_minus);
    scale = __expf(l1 - out_l);
  }

  unsigned mask = 0xffffffffu;
  float b_out_l = __shfl_sync(mask, out_l, 0);
  float b_t     = __shfl_sync(mask, t,     0);
  float b_scale = __shfl_sync(mask, scale, 0);
  if (lane == 0 && out_lse) out_lse[idx] = b_out_l;

  // pair-vectorized
  const int pairs = D / 2;
  for (int i = lane; i < pairs; i += WARP) {
    if constexpr (std::is_same<OType,float>::value) {
      float2 f2 = reinterpret_cast<const float2*>(full_o + base)[i];
      float2 d2 = reinterpret_cast<const float2*>(dd_o   + base)[i];
      float2 o2;
      o2.x = (f2.x - b_t * d2.x) * b_scale;
      o2.y = (f2.y - b_t * d2.y) * b_scale;
      reinterpret_cast<float2*>(out_o + base)[i] = o2;
    } else if constexpr (std::is_same<OType,__half>::value) {
      __half2 fh2 = reinterpret_cast<const __half2*>(full_o + base)[i];
      __half2 dh2 = reinterpret_cast<const __half2*>(dd_o   + base)[i];
      float2 f2 = __half22float2(fh2), d2 = __half22float2(dh2), o2;
      o2.x = (f2.x - b_t * d2.x) * b_scale;
      o2.y = (f2.y - b_t * d2.y) * b_scale;
      reinterpret_cast<__half2*>(out_o + base)[i] = __floats2half2_rn(o2.x, o2.y);
    } else { // bf16
      __nv_bfloat162 fb2 = reinterpret_cast<const __nv_bfloat162*>(full_o + base)[i];
      __nv_bfloat162 db2 = reinterpret_cast<const __nv_bfloat162*>(dd_o   + base)[i];
      float2 f2 = __bfloat1622float2(fb2), d2 = __bfloat1622float2(db2), o2;
      o2.x = (f2.x - b_t * d2.x) * b_scale;
      o2.y = (f2.y - b_t * d2.y) * b_scale;
      reinterpret_cast<__nv_bfloat162*>(out_o + base)[i] = __floats2bfloat162_rn(o2.x, o2.y);
    }
  }
}

template <typename OType>
cudaError_t LaunchDowndateFromMerged(
    OType* full_o, float* full_lse, OType* dd_o, float* dd_lse,
    OType* out_o,  float* out_lse, int N, int H, int D,
    const uint8_t* batch_valid_mask, bool enable_pdl, cudaStream_t stream) {

  constexpr int WARPS_PER_BLOCK = 8;
  dim3 block(WARPS_PER_BLOCK * 32);
  dim3 grid((N * H + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

  auto kernel = DowndateFromMergedKernel<OType>;
  if (enable_pdl) {
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config;
    config.attrs            = attribute;
    config.numAttrs         = 1;
    config.gridDim          = grid;
    config.blockDim         = block;
    config.dynamicSmemBytes = 0;
    config.stream           = stream;
    FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel,
      N, H, D, full_o, full_lse, dd_o, dd_lse, out_o, out_lse, batch_valid_mask));
  } else {
    void* args[] = { &N, &H, &D, &full_o, &full_lse, &dd_o, &dd_lse, &out_o, &out_lse, &batch_valid_mask };
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, grid, block, args, 0, stream));
  }
  return cudaSuccess;
}



// -----------------------------------------------------------------------------
// Fused in-place MERGE (with unified cache) + DOWNDATE (ln domain)
// One warp handles one (n, h): it merges main with cache in-place, then
// immediately emits the downdated (full ⊖ window) result.
// -----------------------------------------------------------------------------
template <typename OType>
__global__ void MergeWithUnifiedCacheAndDowndateKernelInplace(
    // Shapes
    int N, int H, int D, int M, int R,

    // MAIN (to be merged IN-PLACE with cache; ln domain)
    OType* __restrict__ o_rw,    // [N,H,D], updated in-place with merged (full) output
    float* __restrict__ lse_rw,  // [N,H],   updated in-place with merged ln LSE

    // Unified cache (bf16/f32; ln domain)
    const __nv_bfloat16* __restrict__ attn_cache,  // [R,M,H,D]
    const float* __restrict__ lse_cache,           // [R,M,H]

    // Routing
    const int32_t* __restrict__ idx,      // [N,H]
    const uint8_t* __restrict__ hit,      // [N,H] (0/1)
    const int32_t* __restrict__ req_ids,  // [N]   (maps batch->request)

    // Optional mask for padded entries
    const uint8_t* __restrict__ batch_valid_mask,  // [N] or nullptr

    // --- DOWNDATE stream inputs (already merged across tiles; ln domain) ---
    const OType* __restrict__ dd_o,     // [N,H,D]
    const float* __restrict__ dd_lse,   // [N,H]

    // --- Final DOWNDATED outputs (ln domain) ---
    OType* __restrict__ out_o,          // [N,H,D]
    float* __restrict__ out_lse         // [N,H]
) {
  constexpr int WARP = 32;
  const int warps_per_block = blockDim.x / WARP;
  const int warp_id = threadIdx.x / WARP;
  const int lane    = threadIdx.x & (WARP - 1);

  const int group = blockIdx.x * warps_per_block + warp_id;
  const int groups = N * H;
  if (group >= groups) return;

  const int n = group / H;
  const int h = group - n * H;

  if (batch_valid_mask && batch_valid_mask[n] == 0) {
    // If this (n) is invalid, keep outputs inert
    if (lane == 0) {
      if (out_lse) out_lse[n * H + h] = -CUDART_INF_F;
    }
    return;
  }

  // We require even D for pair-vectorized path
  if ((D & 1) != 0) return;
  const int pair_count = D >> 1;

  // Row pointers
  OType* __restrict__ o_row   = o_rw   + (n * H + h) * D;
  const OType* __restrict__ dd_row = dd_o   ? (dd_o + (n * H + h) * D) : nullptr;
  OType* __restrict__ out_row = out_o  ? (out_o + (n * H + h) * D)     : nullptr;

  // -----------------------------
  // 1) Read current main LSE (ln)
  // -----------------------------
  float l2_ln = lse_rw ? lse_rw[n * H + h] : -CUDART_INF_F;    // ln
  bool  has2  = isfinite(l2_ln);

  // -----------------------------
  // 2) Probe unified cache
  // -----------------------------
  const bool hhit = (hit ? (hit[n * H + h] != 0) : false);
  float l1 = -CUDART_INF_F; bool has1 = false;
  const __nv_bfloat162* __restrict__ c_row2 = nullptr;

  if (hhit) {
    const int32_t rq = req_ids ? req_ids[n] : n;   // fallback: req=n
    const int32_t s  = idx ? idx[n * H + h] : -1;
    if (s >= 0) {
      const long long lse_off = ((long long)rq * M + (long long)s) * H + h;  // [R,M,H]
      l1   = lse_cache ? lse_cache[lse_off] : -CUDART_INF_F;                 // ln
      has1 = isfinite(l1);

      const long long row_off = ((((long long)rq * M + (long long)s) * H + h) * (long long)D);
      const __nv_bfloat16* c_row = attn_cache ? (attn_cache + row_off) : nullptr;
      c_row2 = c_row ? reinterpret_cast<const __nv_bfloat162*>(c_row) : nullptr;
    }
  }

  const bool both_invalid = (!has1 && !has2);
  const bool only_current = (!has1 &&  has2);
  const bool only_cached  = ( has1 && !has2);
  const bool both_valid   = ( has1 &&  has2);

  // -----------------------------
  // 3) Do a numerically stable ln merge for main, and write back in-place
  //     Also produce the final merged LSE (ln) for downdate step
  // -----------------------------
  float L_full = -CUDART_INF_F; // merged ln LSE

  float w1 = 0.f, w2 = 0.f;     // weights used only in both_valid path

  if (both_invalid) {
    // full is invalid -> zeros, -inf
    if (lane == 0 && lse_rw) lse_rw[n * H + h] = -CUDART_INF_F;
    for (int p = lane; p < pair_count; p += WARP) {
      // write zeros to main row
      if constexpr (std::is_same<OType, float>::value) {
        reinterpret_cast<float2*>(o_row)[p] = make_float2(0.f, 0.f);
      } else if constexpr (std::is_same<OType, __half>::value) {
        reinterpret_cast<__half2*>(o_row)[p] = __float2half2_rn(0.f);
      } else { // bf16
        reinterpret_cast<__nv_bfloat162*>(o_row)[p] = __floats2bfloat162_rn(0.f, 0.f);
      }
    }
    L_full = -CUDART_INF_F;
  }
  else if (only_current) {
    if (lane == 0 && lse_rw) lse_rw[n * H + h] = l2_ln; // keep ln
    // o_row already holds the current; leave as-is
    L_full = l2_ln;
  }
  else if (only_cached) {
    if (lane == 0 && lse_rw) lse_rw[n * H + h] = l1;    // keep ln
    for (int p = lane; p < pair_count; p += WARP) {
      float2 cf = __bfloat1622float2(c_row2[p]);
      if constexpr (std::is_same<OType, float>::value) {
        reinterpret_cast<float2*>(o_row)[p] = cf;
      } else if constexpr (std::is_same<OType, __half>::value) {
        reinterpret_cast<__half2*>(o_row)[p] = __floats2half2_rn(cf.x, cf.y);
      } else { // bf16
        reinterpret_cast<__nv_bfloat162*>(o_row)[p] = __floats2bfloat162_rn(cf.x, cf.y);
      }
    }
    L_full = l1;
  }
  else { // both_valid
    float L=0.f;
    if (lane == 0) {
      const float m  = fmaxf(l1, l2_ln);
      const float e1 = __expf(l1    - m);
      const float e2 = __expf(l2_ln - m);
      L  = m + __logf(e1 + e2);   // ln
      w1 = __expf(l1    - L);
      w2 = __expf(l2_ln - L);
      if (lse_rw) lse_rw[n * H + h] = L;
    }
    L_full = __shfl_sync(0xffffffffu, L,  0);
    w1     = __shfl_sync(0xffffffffu, w1, 0);
    w2     = __shfl_sync(0xffffffffu, w2, 0);

    for (int p = lane; p < pair_count; p += WARP) {
      // current pair (float2)
      float2 of;
      if constexpr (std::is_same<OType, float>::value) {
        of = reinterpret_cast<const float2*>(o_row)[p];
      } else if constexpr (std::is_same<OType, __half>::value) {
        of = __half22float2(reinterpret_cast<const __half2*>(o_row)[p]);
      } else { // bf16
        of = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(o_row)[p]);
      }
      // cache pair (float2)
      float2 cf = __bfloat1622float2(c_row2[p]);

      float2 full;
      full.x = fmaf(w1, cf.x, w2 * of.x);
      full.y = fmaf(w1, cf.y, w2 * of.y);

      // write MERGED main back to o_row
      if constexpr (std::is_same<OType, float>::value) {
        reinterpret_cast<float2*>(o_row)[p] = full;
      } else if constexpr (std::is_same<OType, __half>::value) {
        reinterpret_cast<__half2*>(o_row)[p] = __floats2half2_rn(full.x, full.y);
      } else { // bf16
        reinterpret_cast<__nv_bfloat162*>(o_row)[p] = __floats2bfloat162_rn(full.x, full.y);
      }
    }
  }

  // Synchronize L_full across the warp
  L_full = __shfl_sync(0xffffffffu, L_full, 0);

  // -----------------------------
  // 4) DOWNDATE (ln robust) using merged MAIN (o_row,L_full) and DD (dd_row, dd_lse)
  //     out_* receive the final downdated result (may alias dd_* buffers).
  // -----------------------------
  if (!out_row || !out_lse || !dd_row || !dd_lse) return; // graceful no-op

  const float l_dd = dd_lse[n * H + h];
  const bool has_full = isfinite(L_full);
  const bool has_dd   = isfinite(l_dd);

  if (!has_full && !has_dd) {
    if (lane == 0) out_lse[n * H + h] = -CUDART_INF_F;
    for (int p = lane; p < pair_count; p += WARP) {
      if constexpr (std::is_same<OType, float>::value) {
        reinterpret_cast<float2*>(out_row)[p] = make_float2(0.f, 0.f);
      } else if constexpr (std::is_same<OType, __half>::value) {
        reinterpret_cast<__half2*>(out_row)[p] = __float2half2_rn(0.f);
      } else {
        reinterpret_cast<__nv_bfloat162*>(out_row)[p] = __floats2bfloat162_rn(0.f, 0.f);
      }
    }
    return;
  }

  if (!has_dd) {
    // Nothing to subtract; pass through FULL
    if (lane == 0) out_lse[n * H + h] = L_full;
    for (int p = lane; p < pair_count; p += WARP) {
      // read merged FULL from o_row and copy to out_row
      if constexpr (std::is_same<OType, float>::value) {
        reinterpret_cast<float2*>(out_row)[p] = reinterpret_cast<const float2*>(o_row)[p];
      } else if constexpr (std::is_same<OType, __half>::value) {
        reinterpret_cast<__half2*>(out_row)[p] = reinterpret_cast<const __half2*>(o_row)[p];
      } else {
        reinterpret_cast<__nv_bfloat162*>(out_row)[p] =
            reinterpret_cast<const __nv_bfloat162*>(o_row)[p];
      }
    }
    return;
  }

  if (!has_full) {
    // FULL is invalid → result is zero vector with -inf lse
    if (lane == 0) out_lse[n * H + h] = -CUDART_INF_F;
    for (int p = lane; p < pair_count; p += WARP) {
      if constexpr (std::is_same<OType, float>::value) {
        reinterpret_cast<float2*>(out_row)[p] = make_float2(0.f, 0.f);
      } else if constexpr (std::is_same<OType, __half>::value) {
        reinterpret_cast<__half2*>(out_row)[p] = __float2half2_rn(0.f);
      } else {
        reinterpret_cast<__nv_bfloat162*>(out_row)[p] = __floats2bfloat162_rn(0.f, 0.f);
      }
    }
    return;
  }

  // has_full && has_dd: robust ln downdate
  float out_l = -CUDART_INF_F, t = 0.f, scale = 0.f;
  if (lane == 0) {
    float diff = l_dd - L_full;
    float em1  = expm1f(diff);
    bool last     = fabsf(em1) <= 1e-7f;
    bool invalid  = (diff > 0.f) && !last;

    if (last) {
      out_l = -CUDART_INF_F; t = 1.f; scale = 0.f;
    } else if (invalid) {
      out_l = L_full; t = 0.f; scale = 1.f;
    } else {
      t = em1 + 1.f;                      // = exp(diff)
      float one_minus = fmaxf(-em1, 1e-30f);
      out_l = L_full + __logf(one_minus); // ln(1 - exp(diff)) + L_full
      scale = __expf(L_full - out_l);
    }
    out_lse[n * H + h] = out_l;
  }
  out_l = __shfl_sync(0xffffffffu, out_l, 0);
  t     = __shfl_sync(0xffffffffu, t,     0);
  scale = __shfl_sync(0xffffffffu, scale, 0);

  // Pairwise compute: FULL already in o_row (post-merge). Read DD, compute downdated.
  for (int p = lane; p < pair_count; p += WARP) {
    float2 full, ddp;
    // FULL
    if constexpr (std::is_same<OType, float>::value) {
      full = reinterpret_cast<const float2*>(o_row)[p];
    } else if constexpr (std::is_same<OType, __half>::value) {
      full = __half22float2(reinterpret_cast<const __half2*>(o_row)[p]);
    } else {
      full = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(o_row)[p]);
    }
    // DD
    if constexpr (std::is_same<OType, float>::value) {
      ddp = reinterpret_cast<const float2*>(dd_row)[p];
    } else if constexpr (std::is_same<OType, __half>::value) {
      ddp = __half22float2(reinterpret_cast<const __half2*>(dd_row)[p]);
    } else {
      ddp = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(dd_row)[p]);
    }

    float2 outp;
    outp.x = (full.x - t * ddp.x) * scale;
    outp.y = (full.y - t * ddp.y) * scale;

    if constexpr (std::is_same<OType, float>::value) {
      reinterpret_cast<float2*>(out_row)[p] = outp;
    } else if constexpr (std::is_same<OType, __half>::value) {
      reinterpret_cast<__half2*>(out_row)[p] = __floats2half2_rn(outp.x, outp.y);
    } else {
      reinterpret_cast<__nv_bfloat162*>(out_row)[p] = __floats2bfloat162_rn(outp.x, outp.y);
    }
  }
}


template <typename OType>
static inline cudaError_t LaunchMergeWithUnifiedCacheAndDowndateInplace(
    // MAIN (in/out)
    OType* o, float* lse,               // [N,H,D], [N,H] (ln) — merged in-place

    // Shapes
    int N, int H, int D,

    // Unified cache (bf16/f32; ln)
    const __nv_bfloat16* attn_cache, const float* lse_cache,

    // Routing
    const int32_t* idx, const uint8_t* hit,
    const int32_t* req_ids, const uint8_t* batch_valid_mask,

    // Cache ring dims
    int M, int R,

    // DOWNDATE inputs (ln)
    const OType* dd_o, const float* dd_lse,   // [N,H,D], [N,H]

    // DOWNDATED outputs (ln)
    OType* out_o, float* out_lse,             // [N,H,D], [N,H]

    // Launch
    bool enable_pdl, cudaStream_t stream)
{
  // If any of the cache inputs are missing, we cannot do the in-place merge; caller should fall back.
  if (!o || !lse || N == 0 || H == 0) return cudaSuccess;
  if (!attn_cache || !lse_cache || !idx || !hit) return cudaErrorInvalidValue;
  if ((D & 1) != 0)  return cudaErrorInvalidValue;  // must be even

  const int warps_per_block = 8; // 256 threads, matches existing practice
  const dim3 nthrs(warps_per_block * 32);
  const dim3 nblks((N * H + warps_per_block - 1) / warps_per_block);

  void* args[] = {
    // shapes
    &N, &H, &D, &M, &R,
    // main in/out
    &o, &lse,
    // cache
    &attn_cache, &lse_cache,
    // routing
    &idx, &hit, &req_ids,
    &batch_valid_mask,
    // downdate in
    &dd_o, &dd_lse,
    // downdate out
    &out_o, &out_lse
  };

  if (enable_pdl) {
    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg{};
    cfg.attrs = &attr; cfg.numAttrs = 1;
    cfg.gridDim = nblks; cfg.blockDim = nthrs; cfg.dynamicSmemBytes = 0; cfg.stream = stream;
    return cudaLaunchKernelEx(
      &cfg,
      MergeWithUnifiedCacheAndDowndateKernelInplace<OType>,
      // args are passed expanded for cudaLaunchKernelEx functor path
      N, H, D, M, R,
      o, lse,
      attn_cache, lse_cache,
      idx, hit, req_ids,
      batch_valid_mask,
      dd_o, dd_lse,
      out_o, out_lse
    );
  } else {
    return cudaLaunchKernel(
      (void*)MergeWithUnifiedCacheAndDowndateKernelInplace<OType>,
      nblks, nthrs, args, /*smem*/0, stream);
  }
}





template <uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE, typename AttentionVariant,
          typename Params>
cudaError_t MACDecodeWithPagedKVCacheDispatched(Params params,
                                                         typename Params::DTypeO* tmp_v,
                                                         float* tmp_s, typename Params::DTypeO* tmp_v_dd, float* tmp_s_dd, bool enable_pdl,
                                                         cudaStream_t stream) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const uint32_t num_kv_heads = params.paged_kv.num_heads;
  const uint32_t padded_batch_size = params.padded_batch_size;
  const uint32_t N            = params.paged_kv.batch_size;

  DTypeO* o_final = nullptr;     // <<< NEW: final outputs to merge into
  float*  lse_final = nullptr;

  constexpr uint32_t vec_size = std::max(16UL / sizeof(DTypeKV), HEAD_DIM / 32UL);
  auto compute_capacity = GetCudaComputeCapability();
  constexpr uint32_t bdx = HEAD_DIM / vec_size;
  static_assert(bdx <= 32);
  DISPATCH_GQA_GROUP_SIZE(num_qo_heads / num_kv_heads, GROUP_SIZE, {
    constexpr uint32_t bdy = GROUP_SIZE;
    constexpr uint32_t num_threads = std::max(128U, bdx * bdy);
    constexpr uint32_t bdz = num_threads / (bdx * bdy);
    constexpr uint32_t tile_size_per_bdx = GROUP_SIZE == 1 ? (sizeof(DTypeKV) == 1 ? 2U : 4U) : 1U;
    DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, {
      const uint32_t smem_size =
          2 * NUM_STAGES_SMEM * tile_size_per_bdx * bdy * bdz * HEAD_DIM * sizeof(DTypeKV) +
          std::max(tile_size_per_bdx * num_threads * sizeof(DTypeKV*),
                   2 * bdy * bdz * sizeof(float));
      auto kernel = MACDecodeWithPagedKVCacheKernel<POS_ENCODING_MODE, NUM_STAGES_SMEM,
                                                             tile_size_per_bdx, vec_size, bdx, bdy,
                                                             bdz, AttentionVariant, Params>;
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

      const bool head_packed = (params.kv_head_indices != nullptr);
      dim3 nblks(head_packed ? padded_batch_size : padded_batch_size * num_kv_heads, 1);
      dim3 nthrs(bdx, bdy, bdz);


      DTypeO* main_o   = params.o;                 // final main
      float*  main_lse = params.lse;               // ln after merges
      DTypeO* dd_o_out = params.downdated_o;       // merged dd (ln)
      float*  dd_s_out = params.downdated_lse;     // merged dd lse (ln)

      // PDL launch config
      cudaLaunchAttribute attribute[1];
      cudaLaunchConfig_t config;
      if (enable_pdl) {
        attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute[0].val.programmaticStreamSerializationAllowed = 1;
        config.attrs = attribute;
        config.numAttrs = 1;
        config.gridDim = nblks;
        config.blockDim = nthrs;
        config.dynamicSmemBytes = smem_size;
        config.stream = stream;
      }
      if (tmp_v == nullptr) {
        FLASHINFER_LOG_WARN(
            "MACDecodeWithPagedKVCacheDispatched: no merging, result is in log2 base!!!!!");
        // do not use partition-kv kernel
        params.partition_kv = false;

        if (enable_pdl) {
          FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, params));
        } else {
          void* args[] = {(void*)&params};
          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        }
      } else {
        // use partition-kv kernel
        params.partition_kv = true;
        // Remap outputs to temporaries ONLY for the compute
        DTypeO* saved_o   = params.o;   float* saved_s   = params.lse;
        DTypeO* saved_do  = params.downdated_o; float* saved_ds = params.downdated_lse;
        params.o   = tmp_v;     params.lse   = tmp_s;
        params.downdated_o   = tmp_v_dd; params.downdated_lse = tmp_s_dd;

        if (enable_pdl) {
          FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, params));
        } else {
          void* args[] = {(void*)&params};
          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        }
        
        // Restore user pointers for merges/epilogues
        params.o   = saved_o;   params.lse   = saved_s;
        params.downdated_o = saved_do; params.downdated_lse = saved_ds;
        if constexpr (AttentionVariant::use_softmax) {
          // ----------------------------
          // FUSED MAIN + DOWNDATE merge
          // ----------------------------
          FLASHINFER_CUDA_CALL(
              MACAttentionVariableLengthMergeStatesDual(
                /* MAIN */    tmp_v,      tmp_s,    params.merge_start_offsets,
                /* DOWNDATE */tmp_v_dd,   tmp_s_dd, params.downdate_start_offsets,
                /* CSR */     params.o_indptr,
                /* OUT MAIN */main_o,     main_lse,
                /* OUT DD  */ dd_o_out,   dd_s_out,
                /* sizes */  params.paged_kv.batch_size, /*seq_len=*/nullptr,
                             /*num_heads=*/num_qo_heads, /*head_dim=*/HEAD_DIM,
                /* launch */ enable_pdl, stream));
        } else {
          // Non-softmax variant: preserve original behavior
          FLASHINFER_CUDA_CALL(VariableLengthAttentionSum(
              tmp_v, params.o_indptr, main_o,
              params.paged_kv.batch_size, nullptr,
              num_qo_heads, HEAD_DIM, enable_pdl, stream));

          // For DD, mirror the previous fallback behavior (copy MAIN if needed)
          const size_t bytes_o = size_t(N) * num_qo_heads * HEAD_DIM * sizeof(DTypeO);
          if (dd_o_out && main_o) {
            FLASHINFER_CUDA_CALL(cudaMemcpyAsync(dd_o_out, main_o, bytes_o,
                                                 cudaMemcpyDeviceToDevice, stream));
          }
        }
      }

      const bool have_cache = params.attn_cache != nullptr;
      if (have_cache) {
        uint32_t M = params.cache_capacity;
        uint32_t R = params.max_running_requests;
        const uint8_t* batch_mask =
            reinterpret_cast<const uint8_t*>(params.block_valid_mask); // may be nullptr

        const __nv_bfloat16* attn_cache_bf16 =
            reinterpret_cast<const __nv_bfloat16*>(params.attn_cache); // OK iff cache is bf16
        const float* lse_cache_f32 = params.lse_cache;

        const int32_t* idx_int32   = reinterpret_cast<const int32_t*>(params.hit_indices);
        const uint8_t* hit_u8      = reinterpret_cast<const uint8_t*>(params.hit_table);
        const int32_t* req_ids_i32 = reinterpret_cast<const int32_t*>(params.cache_req_ids);

        // Inputs for the downdate phase (the merged downdate stream, ln)
        DTypeO* dd_o_in   = params.downdated_o;     // [N,H,D], ln (from fused variable-length merge)
        float*  dd_lse_in = params.downdated_lse;   // [N,H],   ln

        // Final downdated outputs (we allow aliasing with dd_* to keep allocations unchanged)
        DTypeO* dd_out_o   = params.downdated_o;
        float*  dd_out_lse = params.downdated_lse;

        DISPATCH_HEAD_DIM(HEAD_DIM, HEAD_DIM_CONST, {
          if constexpr (std::is_same<DTypeO, float>::value) {
            FLASHINFER_CUDA_CALL( LaunchMergeWithUnifiedCacheAndDowndateInplace<float>(
                /* main in/out */ params.o, params.lse,
                /* shapes     */ N, /*H=*/num_qo_heads, HEAD_DIM_CONST,
                /* cache      */ attn_cache_bf16, lse_cache_f32,
                /* routing    */ idx_int32, hit_u8, req_ids_i32, batch_mask,
                /* ring dims  */ M, R,
                /* dd in      */ dd_o_in, dd_lse_in,
                /* dd out     */ dd_out_o, dd_out_lse,
                enable_pdl, stream) );
          } else if constexpr (std::is_same<DTypeO, __half>::value) {
            FLASHINFER_CUDA_CALL( LaunchMergeWithUnifiedCacheAndDowndateInplace<__half>(
                params.o, params.lse, N, num_qo_heads, HEAD_DIM_CONST,
                attn_cache_bf16, lse_cache_f32,
                idx_int32, hit_u8, req_ids_i32, batch_mask,
                M, R,
                dd_o_in, dd_lse_in,
                dd_out_o, dd_out_lse,
                enable_pdl, stream) );
          } else { // bf16
            FLASHINFER_CUDA_CALL( LaunchMergeWithUnifiedCacheAndDowndateInplace<__nv_bfloat16>(
                params.o, params.lse, N, num_qo_heads, HEAD_DIM_CONST,
                attn_cache_bf16, lse_cache_f32,
                idx_int32, hit_u8, req_ids_i32, batch_mask,
                M, R,
                dd_o_in, dd_lse_in,
                dd_out_o, dd_out_lse,
                enable_pdl, stream) );
          }
        });
      } else {
        // No cache: we still need the ln-robust downdate on the unmerged MAIN.
        // (This preserves current behavior when unified cache is absent.)
        const uint8_t* batch_mask =
            reinterpret_cast<const uint8_t*>(params.block_valid_mask); // may be nullptr
        DISPATCH_HEAD_DIM(HEAD_DIM, HEAD_DIM_CONST, {
          if constexpr (std::is_same<DTypeO, float>::value) {
            FLASHINFER_CUDA_CALL( LaunchDowndateFromMerged<float>(
                /*full*/ params.o, params.lse,
                /*dd  */ params.downdated_o, params.downdated_lse,
                /*out */ params.downdated_o, params.downdated_lse,
                /*N,H,D*/ N, num_qo_heads, HEAD_DIM_CONST,
                batch_mask, enable_pdl, stream) );
          } else if constexpr (std::is_same<DTypeO, __half>::value) {
            FLASHINFER_CUDA_CALL( LaunchDowndateFromMerged<__half>(
                params.o, params.lse, params.downdated_o, params.downdated_lse,
                params.downdated_o, params.downdated_lse,
                N, num_qo_heads, HEAD_DIM_CONST,
                batch_mask, enable_pdl, stream) );
          } else { // bf16
            FLASHINFER_CUDA_CALL( LaunchDowndateFromMerged<__nv_bfloat16>(
                params.o, params.lse, params.downdated_o, params.downdated_lse,
                params.downdated_o, params.downdated_lse,
                N, num_qo_heads, HEAD_DIM_CONST,
                batch_mask, enable_pdl, stream) );
          }
        });
      }
    });
  });
  return cudaSuccess;
}


// ========= Robust downdate helpers (same math/tolerances as OP2) =========
#ifndef LSE_RTOL
#define LSE_RTOL 1e-2f
#endif
#ifndef LSE_ATOL
#define LSE_ATOL 1e-3f
#endif

__device__ __forceinline__ bool isclosef_relabs(float a, float b, float rtol, float atol) {
  if (isnan(a) || isnan(b)) return false;
  if (isinf(a) || isinf(b)) return a == b;
  return fabsf(a - b) <= (atol + rtol * fabsf(b));
}

// 16B/8B/1B vectorized memcpy (used by the ring copy)
__device__ __forceinline__ void copy_bytes_16_any(void* __restrict__ dst_void,
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
    for (size_t i = threadIdx.x; i < n16; i += blockDim.x) d16[i] = s16[i];
    size_t rem = n_bytes - n16 * 16;
    if (rem) {
      size_t base = n16 * 16;
      for (size_t i = threadIdx.x; i < rem; i += blockDim.x) d8[base + i] = s8[base + i];
    }
  } else if (((du | su) & 0x7) == 0) {
    size_t n8 = n_bytes / 8;
    uint2* __restrict__ d8b = (uint2*)dst_void;
    const uint2* __restrict__ s8b = (const uint2*)src_void;
    for (size_t i = threadIdx.x; i < n8; i += blockDim.x) d8b[i] = s8b[i];
    size_t rem = n_bytes - n8 * 8;
    if (rem) {
      size_t base = n8 * 8;
      for (size_t i = threadIdx.x; i < rem; i += blockDim.x) d8[base + i] = s8[base + i];
    }
  } else {
    for (size_t i = threadIdx.x; i < n_bytes; i += blockDim.x) d8[i] = s8[i];
  }
}

// Base-selectable in-place downdate over [B,H,D] bf16 and [B,H] f32.
// If USE_NATURAL==true, inputs are log2 but the math is done in NATURAL log;
// if USE_NATURAL==false, math is done directly in LOG2 domain.

template <bool USE_NATURAL>
__global__ void downdate_inplace_bf16_kernel_base(
    const __nv_bfloat16* __restrict__ full_out, // [B,H,D]  (bf16)
    const float*         __restrict__ full_lse, // [B,H]    (SEE BELOW)
    __nv_bfloat16*       __restrict__ io_out,   // [B,H,D]  (bf16) downdated in-place
    float*               __restrict__ io_lse,   // [B,H]    (SEE BELOW) downdated in-place
    const uint8_t*       __restrict__ batch_mask, // [B] or nullptr
    int B, int H, int D)
{
  constexpr float LN2 = 0.69314718055994530942f;
  const int m = blockIdx.x;
  if (m >= B * H) return;

  const int b = m / H;
  const int h = m % H;
  if (batch_mask && batch_mask[b] == 0) return;

  const size_t base_vec = ((size_t)b * H + h) * (size_t)D;

  // --- Read LSEs into the WORKING BASE:
  //     USE_NATURAL == true  -> work in NATURAL (ln)
  //     USE_NATURAL == false -> work in log2
  float l1, l2;  // working-base logs
  if constexpr (USE_NATURAL) {
    // ✅ full_lse is ALREADY NATURAL (ln)
    l1 = full_lse[b * H + h];              // ln(full)
    // ✅ io_lse is currently LOG2(window) -> convert to NATURAL
    l2 = io_lse[b * H + h] * LN2;          // ln(window)
  } else {
    // legacy path: both in log2
    l1 = full_lse[b * H + h];              // log2(full)
    l2 = io_lse  [b * H + h];              // log2(window)
  }

  // Tolerances are in the working base
  const float atol_eff = /* choose a sensible absolute tol in this base */ 1e-6f;
  const float rtol_eff = /* relative tol in this base */                   1e-6f;

  // l1, l2 are float32 in NATURAL log (ln)
  constexpr float LAST_EPS_EXPM1 = 1e-7f;  // linear-domain closeness threshold

  float diff = l2 - l1;                    // expect <= 0 in normal downdate
  float em1  = expm1f(diff);               // = exp(diff) - 1, stable near 0

  // "last block" if exp(l2) and exp(l1) are almost equal (|exp(l2)/exp(l1) - 1| small)
  bool last_block = fabsf(em1) <= LAST_EPS_EXPM1;

  // "invalid" if l2 > l1 by more than the 'last_block' tolerance
  bool invalid    = (diff > 0.0f) && !last_block;

  float ret_lse, t_lin, one_minus, scale;  // all in working base
  if (last_block) {
    // window == full -> complement is empty
    ret_lse = -CUDART_INF_F;
    t_lin   = 1.f;            // exp(l2 - l1) == 1
    scale   = 0.f;
  } else if (invalid) {
    // window > full numerically -> leave as full
    ret_lse = l1;
    t_lin   = 0.f;
    scale   = 1.f;
  } else {
    if constexpr (USE_NATURAL) {
      t_lin    = __expf(diff);                                        // exp(l2 - l1)
      one_minus= 1.f - t_lin;
      ret_lse  = l1 + __logf(fmaxf(one_minus, 1e-30f));               // ln(1 - exp(diff))
      scale    = __expf(l1 - ret_lse);                                // exp(l1 - ret)
    } else {
      t_lin    = exp2f(diff);                                         // 2^(l2 - l1)
      one_minus= 1.f - t_lin;
      ret_lse  = l1 + log2f(fmaxf(one_minus, 1e-30f));                // log2(1 - 2^(diff))
      scale    = exp2f(l1 - ret_lse);                                 // 2^(l1 - ret)
    }
  }

  // --- Write back LSE in the WORKING BASE:
  if (threadIdx.x == 0) {
    if constexpr (USE_NATURAL) {
      io_lse[b * H + h] = ret_lse;           // ✅ keep NATURAL (ln) on output
    } else {
      io_lse[b * H + h] = ret_lse;           // keep log2 on output
    }
  }

  // --- Vector update (base-invariant): (full - t * window) * scale
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    const float f_full = __bfloat162float(full_out[base_vec + d]);
    const float f_win  = __bfloat162float(io_out  [base_vec + d]);   // io_out holds windowed vector
    const float f_out  = (f_full - t_lin * f_win) * scale;
    io_out[base_vec + d] = __float2bfloat16(f_out);
  }
}


// ========= Single-token ring write using kv_len from paged_kv =========
// Writes Q row, downdated Attn row, and LSE row at slot:
//   dest = (kv_len - 1) % M    if KV_LEN_IS_L_AFTER == true
//   dest = (kv_len)     % M    if KV_LEN_IS_L_AFTER == false
// LSE is stored as ln if LSE_STORE_NATURAL == true; else log2.
template <typename Params, typename DTypeQ, typename DTypeO,
          bool KV_LEN_IS_L_AFTER>
__global__ void ring_update_single_token_kvlen_kernel(
    Params params,
    const DTypeQ* __restrict__ q_src,        // [B,H,D] (bf16)
    const DTypeO* __restrict__ attn_src,     // [B,H,D] (bf16) (already downdated)
    const float*  __restrict__ lse_src,      // [B,H]   (f32)  (already downdated, log2)
    uint32_t H, uint32_t D)
{
  const int b = blockIdx.x;
  if (b >= (int)params.paged_kv.batch_size) return;
  if (params.block_valid_mask && !params.block_valid_mask[b]) return;

  const uint32_t M = params.cache_capacity;
  const int req = reinterpret_cast<const int32_t*>(params.cache_req_ids)[b];

  // kv_len as in compute kernel
  const uint32_t kv_len = params.paged_kv.get_length(b);
  const uint32_t dest_u = KV_LEN_IS_L_AFTER ? (kv_len + M - 1) % M : (kv_len % M);
  const int dest = (int)dest_u;

  // Row sizes
  const size_t row_bf16 = (size_t)H * (size_t)D * sizeof(__nv_bfloat16);
  const size_t row_f32  = (size_t)H * sizeof(float);

  // Source row bases
  const uint8_t* __restrict__ q_src_b = (const uint8_t*)q_src    + (size_t)b * row_bf16;
  const uint8_t* __restrict__ a_src_b = (const uint8_t*)attn_src + (size_t)b * row_bf16;
  const uint8_t* __restrict__ l_src_b = (const uint8_t*)lse_src  + (size_t)b * row_f32;

  // Destination ring bases
  uint8_t* __restrict__ q_dst_b = (uint8_t*)params.query_cache
                                + ((size_t)req * M + dest) * row_bf16;
  uint8_t* __restrict__ a_dst_b = (uint8_t*)params.attn_cache
                                + ((size_t)req * M + dest) * row_bf16;
  uint8_t* __restrict__ l_dst_b = (uint8_t*)params.lse_cache
                                + ((size_t)req * M + dest) * row_f32;

  // Vectorized copies for Q/Attn rows
  copy_bytes_16_any(q_dst_b, q_src_b, row_bf16);
  copy_bytes_16_any(a_dst_b, a_src_b, row_bf16);

  // LSE: either copy as log2, or convert to ln on the fly
  if (lse_src && params.lse_cache) {
    // Store as log2 (byte-exact)
    copy_bytes_16_any(l_dst_b, l_src_b, row_f32);
  }
}


template <uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE, typename AttentionVariant,
          typename Params>
cudaError_t BatchRectificationCacheDecodeWithPagedKVCacheDispatched(Params params,
                                                         typename Params::DTypeO* tmp_v,
                                                         float* tmp_s, bool enable_pdl,
                                                         cudaStream_t stream) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const uint32_t num_kv_heads = params.paged_kv.num_heads;
  const uint32_t padded_batch_size = params.padded_batch_size;
  const uint32_t N            = params.paged_kv.batch_size;

  DTypeO* o_final = nullptr;     // <<< NEW: final outputs to merge into
  float*  lse_final = nullptr;

  constexpr uint32_t vec_size = std::max(16UL / sizeof(DTypeKV), HEAD_DIM / 32UL);
  auto compute_capacity = GetCudaComputeCapability();
  constexpr uint32_t bdx = HEAD_DIM / vec_size;
  static_assert(bdx <= 32);
  DISPATCH_GQA_GROUP_SIZE(num_qo_heads / num_kv_heads, GROUP_SIZE, {
    constexpr uint32_t bdy = GROUP_SIZE;
    constexpr uint32_t num_threads = std::max(128U, bdx * bdy);
    constexpr uint32_t bdz = num_threads / (bdx * bdy);
    constexpr uint32_t tile_size_per_bdx = GROUP_SIZE == 1 ? (sizeof(DTypeKV) == 1 ? 2U : 4U) : 1U;
    DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, {
      const uint32_t smem_size =
          2 * NUM_STAGES_SMEM * tile_size_per_bdx * bdy * bdz * HEAD_DIM * sizeof(DTypeKV) +
          std::max(tile_size_per_bdx * num_threads * sizeof(DTypeKV*),
                   2 * bdy * bdz * sizeof(float));
      auto kernel = BatchRectificationCacheDecodeWithPagedKVCacheKernel<POS_ENCODING_MODE, NUM_STAGES_SMEM,
                                                             tile_size_per_bdx, vec_size, bdx, bdy,
                                                             bdz, AttentionVariant, Params>;
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

      const bool head_packed = (params.kv_head_indices != nullptr);
      dim3 nblks(head_packed ? padded_batch_size : padded_batch_size * num_kv_heads, 1);
      dim3 nthrs(bdx, bdy, bdz);

      DTypeO* user_o   = params.o;
      float*  user_lse = params.lse;
      // PDL launch config
      cudaLaunchAttribute attribute[1];
      cudaLaunchConfig_t config;
      if (enable_pdl) {
        attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute[0].val.programmaticStreamSerializationAllowed = 1;
        config.attrs = attribute;
        config.numAttrs = 1;
        config.gridDim = nblks;
        config.blockDim = nthrs;
        config.dynamicSmemBytes = smem_size;
        config.stream = stream;
      }
      if (tmp_v == nullptr) {
        // do not use partition-kv kernel
        params.partition_kv = false;

        if (enable_pdl) {
          FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, params));
        } else {
          void* args[] = {(void*)&params};
          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        }
      } else {
        // use partition-kv kernel
        params.partition_kv = true;
        auto o = params.o;
        auto lse = params.lse;
        params.o = tmp_v;
        params.lse = tmp_s;
        if (enable_pdl) {
          FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, params));
        } else {
          void* args[] = {(void*)&params};
          FLASHINFER_CUDA_CALL(
              cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        }
        if constexpr (AttentionVariant::use_softmax) {
          FLASHINFER_CUDA_CALL(RectificationCacheVariableLengthMergeStates(tmp_v, tmp_s, params.o_indptr,
                                    o, lse, params.paged_kv.batch_size, nullptr,
                                    num_qo_heads, HEAD_DIM, enable_pdl, stream));
        } else {
          FLASHINFER_CUDA_CALL(
              VariableLengthAttentionSum(tmp_v, params.o_indptr, o, params.paged_kv.batch_size,
                                         nullptr, num_qo_heads, HEAD_DIM, enable_pdl, stream));
        }
      }
      o_final   = user_o;       // <<< merge into the final, not temps
      lse_final = user_lse;

      const bool have_cache = params.attn_cache != nullptr;

      // Set once, compile-time:
      //   true  -> compute downdate in NATURAL log (ln), convert io/full LSEs as needed.
      //   false -> compute downdate directly in LOG2 (no base conversion).
      constexpr bool kDowndateUseNatural = true;   // <<< flip to false to use log2
      constexpr bool kReturnWindowed     = false;
      // kv_len semantics from paged_kv.get_length(b):
      //   true  -> kv_len is L_after (post-update); write at (kv_len - 1) % M
      //   false -> kv_len is L_before;              write at  kv_len      % M
      constexpr bool kKvLenIsLAfter = true;
          
      // NEW: LSE cache storage base
      //   true  -> write LSE to lse_cache in NATURAL log (ln)
      //   false -> write LSE to lse_cache in LOG2
      // To match your baseline coop path (you multiply by ln2 before the coop update),
      // set this to true.

      DTypeO* o_window_backup   = nullptr;
      float*  lse_window_backup = nullptr;
      const uint32_t B   = params.paged_kv.batch_size;       // active batch
      const uint32_t Hh  = num_qo_heads;                     // heads per row
      const uint32_t Dd  = HEAD_DIM;                         // head dim
      if (have_cache) {
        const uint8_t* batch_mask =
            reinterpret_cast<const uint8_t*>(params.block_valid_mask); // may be nullptr

        const DTypeO* full_attn =
            reinterpret_cast<const DTypeO*>(params.full_attn);
        const float* full_lse = params.full_lse;

        if constexpr (AttentionVariant::use_softmax) {
          const int groups  = (int)(B * Hh);
          const int threads = 256;
          dim3 grid(groups), block(threads);
          // DTypeO must be bf16 to match the downdate kernel used here
          static_assert(std::is_same<DTypeO, __nv_bfloat16>::value,
                        "downdate_inplace_bf16_kernel expects bf16 outputs");
          
          if constexpr (kReturnWindowed) {
            const size_t bytes_o   = size_t(B) * Hh * Dd * sizeof(DTypeO);
            const size_t bytes_lse = size_t(B) * Hh * sizeof(float);
            cudaMallocAsync((void**)&o_window_backup, bytes_o, stream);
            cudaMemcpyAsync(o_window_backup, o_final, bytes_o,
                            cudaMemcpyDeviceToDevice, stream);
            if (lse_final) {
              cudaMallocAsync((void**)&lse_window_backup, bytes_lse, stream);
              cudaMemcpyAsync(lse_window_backup, lse_final, bytes_lse,
                              cudaMemcpyDeviceToDevice, stream);
            }
          }
          if constexpr (kDowndateUseNatural) {
            downdate_inplace_bf16_kernel_base<true><<<grid, block, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(full_attn),
                full_lse,
                reinterpret_cast<__nv_bfloat16*>(o_final),
                lse_final,
                batch_mask,
                (int)B, (int)Hh, (int)Dd);
          } else {
            downdate_inplace_bf16_kernel_base<false><<<grid, block, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(full_attn),
                full_lse,
                reinterpret_cast<__nv_bfloat16*>(o_final),
                lse_final,
                batch_mask,
                (int)B, (int)Hh, (int)Dd);
          }
        } else {
          // No LSE path → skip downdate; we will cache o_final as-is.
        }

        // 4) Cache the downdated results at dest = paged_kv.get_length(b) % M (Ni=1 decoding)
        const DTypeQ* q_src = reinterpret_cast<const DTypeQ*>(params.q); // [B,H,D] bf16 queries
        // Caches accessed inside the kernel via params.{query_cache,attn_cache,lse_cache}
        // NOTE: no request_length32/64 access anymore; we read kv_len from paged_kv on device.
        const int threads = 128;
        dim3 grid(B), block(threads);
        static_assert(std::is_same<DTypeQ, __nv_bfloat16>::value, "q_cache expects bf16");
        static_assert(std::is_same<DTypeO, __nv_bfloat16>::value, "attn_cache expects bf16");
                
        if constexpr (kKvLenIsLAfter) {
          ring_update_single_token_kvlen_kernel<Params, DTypeQ, DTypeO, true>
              <<<grid, block, 0, stream>>>(params,
                reinterpret_cast<const DTypeQ*>(params.q_to_cache),
                reinterpret_cast<const DTypeO*>(o_final),
                (AttentionVariant::use_softmax ? lse_final : nullptr),
                (uint32_t)Hh, (uint32_t)Dd);
        } else {
          ring_update_single_token_kvlen_kernel<Params, DTypeQ, DTypeO, false>
              <<<grid, block, 0, stream>>>(params,
                reinterpret_cast<const DTypeQ*>(params.q_to_cache),
                reinterpret_cast<const DTypeO*>(o_final),
                (AttentionVariant::use_softmax ? lse_final : nullptr),
                (uint32_t)Hh, (uint32_t)Dd);
        }
      }
      const bool can_downdate = AttentionVariant::use_softmax && params.full_attn && params.full_lse;
      if constexpr (kReturnWindowed) {
        if (can_downdate) {
          const size_t bytes_o   = size_t(B) * num_qo_heads * HEAD_DIM * sizeof(DTypeO);
          if (o_window_backup) {
            cudaMemcpyAsync(o_final, o_window_backup, bytes_o,
                            cudaMemcpyDeviceToDevice, stream);
            cudaFreeAsync(o_window_backup, stream);
          }
          if (lse_window_backup) {
            const size_t bytes_lse = size_t(B) * num_qo_heads * sizeof(float);
            cudaMemcpyAsync(lse_final, lse_window_backup, bytes_lse,
                            cudaMemcpyDeviceToDevice, stream);
            cudaFreeAsync(lse_window_backup, stream);
          }
        }
      }
    });
  });
  return cudaSuccess;
}


template <uint32_t vec_size_ckv, uint32_t vec_size_kpe, uint32_t bdx, uint32_t tile_size,
          typename AttentionVariant, typename Params, typename T>
__device__ __forceinline__ void compute_qk_and_update_local_stat_mla(
    const Params& params, AttentionVariant variant, const uint32_t batch_idx, const T* ckv_smem,
    const vec_t<float, vec_size_ckv>& q_nope_vec, const T* kpe_smem,
    const vec_t<float, vec_size_kpe>& q_pe_vec, const vec_t<float, vec_size_kpe>& freq,
    uint32_t kv_idx_base, uint32_t iter_base, uint32_t iter_bound, state_t<vec_size_ckv>& st) {
  uint32_t tx = threadIdx.x, tz = threadIdx.z;
  constexpr uint32_t head_dim_ckv = bdx * vec_size_ckv;
  constexpr uint32_t head_dim_kpe = bdx * vec_size_kpe;
  float s[tile_size];
  float m_prev = st.m;
#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    vec_t<float, vec_size_ckv> ckv_vec;
    ckv_vec.cast_load(ckv_smem + j * head_dim_ckv + tx * vec_size_ckv);

    vec_t<float, vec_size_kpe> kpe_vec;
    kpe_vec.cast_load(kpe_smem + j * head_dim_kpe + tx * vec_size_kpe);

    s[j] = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < vec_size_ckv; ++i) {
      s[j] += q_nope_vec[i] * ckv_vec[i];
    }
#pragma unroll
    for (uint32_t i = 0; i < vec_size_kpe; ++i) {
      s[j] += q_pe_vec[i] * kpe_vec[i];
    }
    s[j] *= params.sm_scale;
#pragma unroll
    for (uint32_t offset = bdx / 2; offset > 0; offset /= 2) {
      s[j] += math::shfl_xor_sync(s[j], offset);
    }
    s[j] = (iter_base + tz * tile_size + j < iter_bound) ? s[j] : -math::inf;
    st.m = max(st.m, s[j]);
  }

  float o_scale = math::ptx_exp2(m_prev - st.m);
  st.d *= o_scale;
#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    s[j] = math::ptx_exp2(s[j] - st.m);
    st.d += s[j];
  }
#pragma unroll
  for (uint32_t i = 0; i < vec_size_ckv; ++i) {
    st.o[i] = st.o[i] * o_scale;
  }

#pragma unroll
  for (uint32_t j = 0; j < tile_size; ++j) {
    vec_t<float, vec_size_ckv> v_vec;
    v_vec.cast_load(ckv_smem + j * head_dim_ckv + tx * vec_size_ckv);
#pragma unroll
    for (uint32_t i = 0; i < vec_size_ckv; ++i) {
      st.o[i] = st.o[i] + s[j] * v_vec[i];
    }
  }
}

template <uint32_t num_stages_smem, uint32_t vec_size_ckv, uint32_t vec_size_kpe, uint32_t bdx,
          uint32_t bdy, uint32_t bdz, uint32_t tile_size_qo_heads, typename AttentionVariant,
          typename Params>
__global__ void BatchDecodeWithPagedKVCacheKernelMLA(Params params) {
  auto block = cg::this_thread_block();
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  const DTypeQ* q_nope = params.q_nope;
  const DTypeQ* q_pe = params.q_pe;
  DTypeO* o = params.o;
  float* lse = params.lse;
  const auto& paged_kv = params.paged_kv;
  const IdType* q_rope_offset = params.q_rope_offset;
  const bool* block_valid_mask = params.block_valid_mask;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const float rope_rcp_scale = params.rope_rcp_scale;
  const float rope_rcp_theta = params.rope_rcp_theta;
  const bool partition_kv = params.partition_kv;
  params.sm_scale *= math::log2e;

  constexpr uint32_t head_dim_ckv = bdx * vec_size_ckv;
  constexpr uint32_t head_dim_kpe = bdx * vec_size_kpe;
  const uint32_t batch_idx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
  const uint32_t t_offset = dim3_offset(bdy, bdx, tz, ty, tx);

  // NOTE(Zihao): when CUDAGraph is enabled, we will launch more blocks than
  // the actual batch size, so we need to check if the current batch is valid
  if (block_valid_mask && !block_valid_mask[batch_idx]) return;
  const uint32_t mapped_batch_idx = params.request_indices[batch_idx];

  const uint32_t orig_seq_len = paged_kv.get_length(mapped_batch_idx);
  int32_t q_rope_offset_val =
      q_rope_offset == nullptr ? (orig_seq_len - 1) : q_rope_offset[mapped_batch_idx];

  const uint32_t kv_chunk_idx_in_orig_mapped_batch = params.kv_tile_indices[batch_idx];
  const uint32_t kv_chunk_size = *(params.kv_chunk_size_ptr);
  const uint32_t cur_chunk_start =
      partition_kv ? kv_chunk_idx_in_orig_mapped_batch * kv_chunk_size : 0;
  const uint32_t cur_chunk_end =
      partition_kv ? min((kv_chunk_idx_in_orig_mapped_batch + 1) * kv_chunk_size, orig_seq_len)
                   : orig_seq_len;
  const uint32_t cur_chunk_len = cur_chunk_end - cur_chunk_start;

  uint32_t packed_page_iter_base =
      paged_kv.indptr[mapped_batch_idx] * paged_kv.page_size + cur_chunk_start;
  const IdType last_indptr = paged_kv.indptr[paged_kv.batch_size];

  constexpr uint32_t kv_iter_len = bdy * bdz;
  constexpr uint32_t compute_qk_tile = bdy;

  extern __attribute__((shared)) uint8_t smem[];
  DTypeKV* ckv_smem = (DTypeKV*)smem;
  DTypeKV* kpe_smem = (DTypeKV*)((uint8_t*)ckv_smem +
                                 num_stages_smem * kv_iter_len * head_dim_ckv * sizeof(DTypeKV));
  size_t* ckv_offset_smem = (size_t*)((uint8_t*)kpe_smem + num_stages_smem * kv_iter_len *
                                                               head_dim_kpe * sizeof(DTypeKV));
  size_t* kpe_offset_smem = (size_t*)((uint8_t*)ckv_offset_smem + bdx * bdy * bdz * sizeof(size_t));
  float* smem_md = (float*)ckv_offset_smem;

  AttentionVariant variant(params, batch_idx, smem);

  vec_t<float, vec_size_ckv> q_nope_vec[tile_size_qo_heads];
  vec_t<float, vec_size_kpe> q_pe_vec[tile_size_qo_heads];
  state_t<vec_size_ckv> st[tile_size_qo_heads];
  uint32_t qo_head_idx[tile_size_qo_heads];

  vec_t<float, vec_size_kpe> freq;

#pragma unroll
  for (uint32_t i = 0; i < vec_size_kpe; ++i) {
    freq[i] = rope_rcp_scale * __powf(rope_rcp_theta, float(2 * ((tx * vec_size_kpe + i) / 2)) /
                                                          float(head_dim_kpe));
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif
  // load q_nope and q_pe tile
#pragma unroll
  for (int i = 0; i < tile_size_qo_heads; ++i) {
    qo_head_idx[i] = dim3_offset(bdy, tile_size_qo_heads, blockIdx.y, threadIdx.y, i);
    if (qo_head_idx[i] < num_qo_heads) {
      q_nope_vec[i].cast_load(q_nope +
                              (mapped_batch_idx * num_qo_heads + qo_head_idx[i]) * head_dim_ckv +
                              tx * vec_size_ckv);
      q_pe_vec[i].cast_load(q_pe +
                            (mapped_batch_idx * num_qo_heads + qo_head_idx[i]) * head_dim_kpe +
                            tx * vec_size_kpe);
    }
  }

  // init paged-cache read offset to be used
  uint32_t q, r;
  paged_kv.page_size.divmod(packed_page_iter_base + t_offset, q, r);
  ckv_offset_smem[t_offset] = paged_kv.protective_get_offset_ckv(q, r, /*feat_idx*/ 0, last_indptr);
  kpe_offset_smem[t_offset] = paged_kv.protective_get_offset_kpe(q, r, /*feat_idx*/ 0, last_indptr);
  block.sync();

  uint32_t stage_idx = 0;
  constexpr uint32_t vec_bits = sizeof(DTypeKV) * vec_size_ckv * 8;
  constexpr uint32_t tx_fold = vec_size_ckv / vec_size_kpe;
  static_assert(num_stages_smem <= bdx);
  size_t offset_bytes;
  bool is_valid_range;
#pragma unroll
  for (uint32_t iter = 0; iter < num_stages_smem; ++iter) {
    is_valid_range = (iter * kv_iter_len + dim2_offset(bdy, tz, ty)) < cur_chunk_len;

    offset_bytes = ckv_offset_smem[dim3_offset(bdz, bdy, iter, tz, ty)] + tx * vec_size_ckv;
    cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
        ckv_smem + (stage_idx * kv_iter_len + dim2_offset(bdy, tz, ty)) * head_dim_ckv +
            tx * vec_size_ckv,
        paged_kv.ckv_data + offset_bytes, is_valid_range);

    offset_bytes =
        kpe_offset_smem[dim3_offset(bdz, bdy, iter, tz, ty)] + tx / tx_fold * vec_size_ckv;
    cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
        kpe_smem + (stage_idx * kv_iter_len + dim2_offset(bdy, tz, ty)) * head_dim_kpe +
            tx / tx_fold * vec_size_ckv,
        paged_kv.kpe_data + offset_bytes, is_valid_range);

    cp_async::commit_group();
    stage_idx = (stage_idx + 1) % num_stages_smem;
  }

#pragma unroll
  for (uint32_t iter = 0; iter < ceil_div(cur_chunk_len, kv_iter_len); ++iter) {
    cp_async::wait_group<1 * num_stages_smem - 1>();
    block.sync();
    const int32_t kv_idx_base =
        (paged_kv.rope_pos_offset == nullptr ? 0 : paged_kv.rope_pos_offset[mapped_batch_idx]) +
        cur_chunk_start + iter * kv_iter_len;
#pragma unroll
    for (int i = 0; i < tile_size_qo_heads; ++i) {
      compute_qk_and_update_local_stat_mla<vec_size_ckv, vec_size_kpe, bdx, compute_qk_tile>(
          params, variant, mapped_batch_idx,
          ckv_smem + (stage_idx * kv_iter_len + tz * compute_qk_tile) * head_dim_ckv, q_nope_vec[i],
          kpe_smem + (stage_idx * kv_iter_len + tz * compute_qk_tile) * head_dim_kpe, q_pe_vec[i],
          freq, kv_idx_base,
          /*iter_base*/ iter * kv_iter_len, /*iter_bound*/ cur_chunk_len, st[i]);
    }

    if ((iter + num_stages_smem) % bdx == 0) {
      uint32_t q, r;
      paged_kv.page_size.divmod(
          packed_page_iter_base + (iter + num_stages_smem) * kv_iter_len + t_offset, q, r);
      ckv_offset_smem[t_offset] =
          paged_kv.protective_get_offset_ckv(q, r, /*feat_idx*/ 0, last_indptr);
      kpe_offset_smem[t_offset] =
          paged_kv.protective_get_offset_kpe(q, r, /*feat_idx*/ 0, last_indptr);
    }
    block.sync();

    is_valid_range =
        ((iter + num_stages_smem) * kv_iter_len + dim2_offset(bdy, tz, ty)) < cur_chunk_len;
    offset_bytes = ckv_offset_smem[dim3_offset(bdz, bdy, (iter + num_stages_smem) % bdx, tz, ty)] +
                   tx * vec_size_ckv;
    cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
        ckv_smem + (stage_idx * kv_iter_len + dim2_offset(bdy, tz, ty)) * head_dim_ckv +
            tx * vec_size_ckv,
        paged_kv.ckv_data + offset_bytes, is_valid_range);

    offset_bytes = kpe_offset_smem[dim3_offset(bdz, bdy, (iter + num_stages_smem) % bdx, tz, ty)] +
                   tx / tx_fold * vec_size_ckv;
    cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kFillZero>(
        kpe_smem + (stage_idx * kv_iter_len + dim2_offset(bdy, tz, ty)) * head_dim_kpe +
            tx / tx_fold * vec_size_ckv,
        paged_kv.kpe_data + offset_bytes, is_valid_range);
    cp_async::commit_group();

    stage_idx = (stage_idx + 1) % num_stages_smem;
  }
  cp_async::wait_group<0>();
  block.sync();

  if (bdz != 1) {
#pragma unroll
    for (int i = 0; i < tile_size_qo_heads; ++i) {
      if (qo_head_idx[i] < num_qo_heads)
        sync_state<vec_size_ckv, bdx, bdy, bdz>(variant, st[i], (float*)smem, smem_md, tx, ty, tz);
    }
  }

  if (tz == 0) {
#pragma unroll
    for (int i = 0; i < tile_size_qo_heads; ++i) {
      if (qo_head_idx[i] < num_qo_heads) {
        st[i].normalize();
        st[i].o.cast_store(o + (batch_idx * num_qo_heads + qo_head_idx[i]) * head_dim_ckv +
                           tx * vec_size_ckv);

        if (lse != nullptr) {
          lse[batch_idx * num_qo_heads + qo_head_idx[i]] = st[i].get_lse();
        }
      }
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <uint32_t HEAD_DIM_CKV, uint32_t HEAD_DIM_KPE, typename AttentionVariant, typename Params>
cudaError_t BatchDecodeWithPagedKVCacheDispatchedMLA(Params params, typename Params::DTypeO* tmp_v,
                                                     float* tmp_s, bool enable_pdl,
                                                     cudaStream_t stream) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  const uint32_t num_qo_heads = params.num_qo_heads;
  const uint32_t padded_batch_size = params.padded_batch_size;

  constexpr uint32_t vec_size_ckv = std::max(16UL / sizeof(DTypeKV), HEAD_DIM_CKV / 32UL);
  constexpr uint32_t bdx = HEAD_DIM_CKV / vec_size_ckv;
  constexpr uint32_t vec_size_kpe = HEAD_DIM_KPE / bdx;

  constexpr uint32_t bdy = 8;
  constexpr uint32_t tile_size_qo_heads = 2;
  constexpr uint32_t qo_heads_per_block = bdy * tile_size_qo_heads;
  constexpr uint32_t num_threads = std::max(128U, bdx * bdy);
  constexpr uint32_t bdz = num_threads / (bdx * bdy);
  const uint32_t gdy = ceil_div(num_qo_heads, qo_heads_per_block);

  auto compute_capacity = GetCudaComputeCapability();
  DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, {
    const uint32_t smem_size =
        NUM_STAGES_SMEM * bdy * bdz * (HEAD_DIM_CKV + HEAD_DIM_KPE) * sizeof(DTypeKV) +
        std::max(num_threads * sizeof(size_t) * 2, 2 * bdy * bdz * sizeof(float));

    auto kernel =
        BatchDecodeWithPagedKVCacheKernelMLA<NUM_STAGES_SMEM, vec_size_ckv, vec_size_kpe, bdx, bdy,
                                             bdz, tile_size_qo_heads, AttentionVariant, Params>;
    FLASHINFER_CUDA_CALL(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    dim3 nblks(padded_batch_size, gdy);
    dim3 nthrs(bdx, bdy, bdz);

    // PDL launch config
    cudaLaunchAttribute attribute[1];
    cudaLaunchConfig_t config;
    if (enable_pdl) {
      attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attribute[0].val.programmaticStreamSerializationAllowed = 1;
      config.attrs = attribute;
      config.numAttrs = 1;
      config.gridDim = nblks;
      config.blockDim = nthrs;
      config.dynamicSmemBytes = smem_size;
      config.stream = stream;
    }

    if (tmp_v == nullptr) {
      // do not use partition-kv kernel
      params.partition_kv = false;
      if (enable_pdl) {
        FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, params));
      } else {
        void* args[] = {(void*)&params};
        FLASHINFER_CUDA_CALL(
            cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
      }
    } else {
      // use partition-kv kernel
      params.partition_kv = true;
      auto o = params.o;
      auto lse = params.lse;
      params.o = tmp_v;
      params.lse = tmp_s;
      if (enable_pdl) {
        FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, params));
      } else {
        void* args[] = {(void*)&params};
        FLASHINFER_CUDA_CALL(
            cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
      }
      FLASHINFER_CUDA_CALL(VariableLengthMergeStates(
          tmp_v, tmp_s, params.o_indptr, o, lse, params.paged_kv.batch_size, nullptr, num_qo_heads,
          HEAD_DIM_CKV, enable_pdl, stream));
    }
  });
  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_DECODE_CUH_
