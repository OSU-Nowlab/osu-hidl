/*
 * This file is derived from FlashInfer and adapted for the standalone mac_attention project.
 *
 * It exposes only the rectification+cache op (windowed decode + downdate + ring-cache update) used by
 * MACRectificationCacheWithPagedKVCacheWrapper, without any dependency on the flashinfer Python
 * package.
 */
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/utils.cuh>
#include <optional>

#include "mac_rectification_cache_config.inc"
#include "pytorch_conversion_utils.h"
#include "pytorch_extension_utils.h"

namespace flashinfer {
template <uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE, typename AttentionVariant,
          typename Params>
cudaError_t BatchRectificationCacheDecodeWithPagedKVCacheDispatched(Params params,
                                                           typename Params::DTypeO* tmp_v,
                                                           float* tmp_s, bool enable_pdl,
                                                           cudaStream_t stream);
}  // namespace flashinfer

using namespace flashinfer;

at::Tensor MACRectificationCacheWithPagedKVCachePlan(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    at::Tensor page_locked_int_workspace_buffer, at::Tensor indptr, int64_t batch_size,
    int64_t num_qo_heads, int64_t num_kv_heads, int64_t page_size, bool enable_cuda_graph,
    int64_t window_left, double logits_soft_cap, int64_t head_dim_qk, int64_t head_dim_vo) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer.size(0) * float_workspace_buffer.element_size();
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer.size(0) * int_workspace_buffer.element_size();

  TORCH_CHECK(page_size == 1,
              "mac_attention v1 requires page_size == 1 (window scheduler expects token==page). "
              "Got ",
              page_size);

  TORCH_CHECK(head_dim_qk == head_dim_vo,
              "CUDA cores template only supports equal head dim for QK and VO");
  TORCH_CHECK(static_cast<uint32_t>(head_dim_qk) == HEAD_DIM_QK,
              "This JIT module was built for head_dim ", HEAD_DIM_QK, ", got ", head_dim_qk);
  CHECK_GQA_HEAD_DIVISIBLE(num_qo_heads, num_kv_heads);

  MACAttentionDecodePlanInfo plan_info;

  const c10::cuda::OptionalCUDAGuard device_guard(float_workspace_buffer.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  DISPATCH_GQA_GROUP_SIZE(num_qo_heads / num_kv_heads, GROUP_SIZE, {
    // Build L_tokens from host indptr (token counts, since page_size==1).
    std::vector<IdType> L_tokens;
    IdType max_len_tokens = 0;
    {
      auto* indptr_h = static_cast<IdType*>(indptr.data_ptr());
      cudaError_t st = WindowBuildKVLensFromIndptrTokens<IdType>(
          L_tokens, max_len_tokens, indptr_h, static_cast<uint32_t>(batch_size));
      TORCH_CHECK(st == cudaSuccess, "WindowBuildKVLensFromIndptrTokens failed: ",
                  cudaGetErrorString(st));
    }

    // Work estimation wrapper in TOKENS.
    auto work_estimation_func =
        [&](bool& split_kv, uint32_t& max_grid_size, uint32_t& kv_chunk_size_in_tokens,
            uint32_t& new_batch_size, uint32_t& gdy, uint32_t /*batch_size_unused*/,
            IdType* /*kv_indptr_unused*/, const uint32_t /*num_qo_heads_unused*/,
            const uint32_t /*page_size_unused*/, bool enable_cuda_graph_) -> cudaError_t {
      return RectificationCacheDecodeWorkEstimationHeadPackedPrepared<GROUP_SIZE, HEAD_DIM_QK,
                                                             POS_ENCODING_MODE, AttentionVariant,
                                                             Params>(
          split_kv, max_grid_size, kv_chunk_size_in_tokens, new_batch_size, gdy,
          static_cast<uint32_t>(batch_size), static_cast<uint32_t>(num_qo_heads),
          /*page_size_unused*/ 1, enable_cuda_graph_, L_tokens,
          static_cast<uint32_t>(window_left));
    };

    // Compact planning.
    cudaError_t status = RectificationCacheDecodePlanCompact<GROUP_SIZE, HEAD_DIM_QK, POS_ENCODING_MODE,
                                                     AttentionVariant, Params>(
        static_cast<void*>(float_workspace_buffer.data_ptr()), float_workspace_size_in_bytes,
        static_cast<void*>(int_workspace_buffer.data_ptr()),
        static_cast<void*>(page_locked_int_workspace_buffer.data_ptr()),
        int_workspace_size_in_bytes, plan_info, L_tokens, static_cast<uint32_t>(batch_size),
        static_cast<uint32_t>(num_qo_heads), static_cast<uint32_t>(window_left),
        enable_cuda_graph, /*stream=*/stream, work_estimation_func);

    TORCH_CHECK(status == cudaSuccess,
                "MACRectificationCache plan failed: ",
                cudaGetErrorString(status));
  });

  return vec_to_tensor(plan_info.ToVector());
}


void MACRectificationCacheWithPagedKVCacheRun(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer, at::Tensor plan_info_vec,
    at::Tensor q, at::Tensor paged_k_cache, at::Tensor paged_v_cache, at::Tensor paged_kv_indptr,
    at::Tensor paged_kv_indices, at::Tensor paged_kv_last_page_len, int64_t max_running_requests,
    int64_t cache_capacity, at::Tensor full_attn, at::Tensor full_lse, at::Tensor q_to_cache,
    at::Tensor query_cache, at::Tensor attn_cache, at::Tensor lse_cache, at::Tensor cache_req_ids,
    bool use_cache, at::Tensor o, std::optional<at::Tensor> maybe_lse, int64_t kv_layout_code,
    int64_t window_left, bool enable_pdl, at::Tensor maybe_alibi_slopes, double logits_soft_cap,
    double sm_scale, double rope_scale, double rope_theta) {
  MACAttentionDecodePlanInfo plan_info;
  plan_info.FromVector(tensor_to_vec(plan_info_vec));

  QKVLayout kv_layout = static_cast<QKVLayout>(kv_layout_code);
  auto device = q.device();
  int64_t batch_size = q.size(0);
  int64_t num_qo_heads = q.size(1);

  int64_t num_kv_heads, page_size;
  if (kv_layout == QKVLayout::kHND) {
    num_kv_heads = paged_k_cache.size(1);
    page_size = paged_k_cache.size(2);
  } else {
    page_size = paged_k_cache.size(1);
    num_kv_heads = paged_k_cache.size(2);
  }

  uint32_t head_dim_qk = q.size(2);
  uint32_t head_dim_vo = paged_v_cache.size(3);

  TORCH_CHECK(page_size == 1,
              "mac_attention v1 requires page_size == 1 (window scheduler expects token==page). "
              "Got ",
              page_size);
  TORCH_CHECK(head_dim_qk == head_dim_vo,
              "CUDA cores template only supports equal head dim for QK and VO");
  TORCH_CHECK(head_dim_qk == HEAD_DIM_QK,
              "This JIT module was built for head_dim ", HEAD_DIM_QK, ", got ", head_dim_qk);
  CHECK_GQA_HEAD_DIVISIBLE(num_qo_heads, num_kv_heads);

  if (maybe_lse) {
    const auto& lse = *maybe_lse;
    TORCH_CHECK(lse.size(0) == batch_size);
    TORCH_CHECK(lse.size(1) == num_qo_heads);
  }

  void* float_buffer = static_cast<void*>(float_workspace_buffer.data_ptr());
  void* int_buffer = static_cast<void*>(int_workspace_buffer.data_ptr());

  // get q_stride_n and q_stride_h
  const auto q_stride_n = q.stride(0);
  const auto q_stride_h = q.stride(1);

  // get kv_cache_strides
  const int64_t* kv_cache_strides = nullptr;
  auto k_strides = paged_k_cache.strides();
  auto v_strides = paged_v_cache.strides();
  TORCH_CHECK(k_strides == v_strides, "k/v strides must be identical");
  kv_cache_strides = k_strides.data();

  const c10::cuda::OptionalCUDAGuard device_guard(device);
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  paged_kv_t<DTypeKV, IdType> paged_kv(
      static_cast<uint32_t>(num_kv_heads), static_cast<uint32_t>(page_size), HEAD_DIM_QK,
      static_cast<uint32_t>(batch_size), kv_layout,
      static_cast<DTypeKV*>(paged_k_cache.data_ptr()),
      static_cast<DTypeKV*>(paged_v_cache.data_ptr()), kv_cache_strides,
      static_cast<IdType*>(paged_kv_indices.data_ptr()),
      static_cast<IdType*>(paged_kv_indptr.data_ptr()),
      static_cast<IdType*>(paged_kv_last_page_len.data_ptr()));

  Params params;
  params.q = static_cast<DTypeQ*>(q.data_ptr());
  params.paged_kv = paged_kv;
  params.max_running_requests = static_cast<uint32_t>(max_running_requests);
  params.cache_capacity = static_cast<uint32_t>(cache_capacity);

  params.full_attn = use_cache ? static_cast<DTypeO*>(full_attn.data_ptr()) : nullptr;
  params.full_lse = use_cache ? static_cast<float*>(full_lse.data_ptr()) : nullptr;
  params.q_to_cache = use_cache ? static_cast<DTypeQ*>(q_to_cache.data_ptr()) : nullptr;
  params.query_cache = use_cache ? static_cast<DTypeQ*>(query_cache.data_ptr()) : nullptr;
  params.attn_cache = use_cache ? static_cast<DTypeO*>(attn_cache.data_ptr()) : nullptr;
  params.lse_cache = use_cache ? static_cast<float*>(lse_cache.data_ptr()) : nullptr;
  params.cache_req_ids = use_cache ? static_cast<uint32_t*>(cache_req_ids.data_ptr()) : nullptr;
  params.use_cache = use_cache;

  params.o = static_cast<DTypeO*>(o.data_ptr());
  params.lse = maybe_lse ? static_cast<float*>(maybe_lse->data_ptr()) : nullptr;
  params.padded_batch_size = 0;
  params.num_qo_heads = static_cast<uint32_t>(num_qo_heads);
  params.q_stride_n = q_stride_n;
  params.q_stride_h = q_stride_h;
  params.window_left = static_cast<int32_t>(window_left);

  params.maybe_alibi_slopes =
      maybe_alibi_slopes.numel() ? static_cast<float*>(maybe_alibi_slopes.data_ptr()) : nullptr;
  params.logits_soft_cap = static_cast<float>(logits_soft_cap);
  params.sm_scale = static_cast<float>(sm_scale);
  params.rope_rcp_scale = static_cast<float>(1.0 / rope_scale);
  params.rope_rcp_theta = static_cast<float>(1.0 / rope_theta);

  params.request_indices = nullptr;
  params.kv_tile_indices = nullptr;
  params.kv_head_indices = nullptr;
  params.merge_start_offsets = nullptr;
  params.o_indptr = nullptr;
  params.kv_chunk_size_ptr = nullptr;
  params.block_valid_mask = nullptr;
  params.partition_kv = plan_info.split_kv;

  DTypeO* tmp_v = nullptr;
  float* tmp_s = nullptr;
  params.request_indices =
      GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.request_indices_offset);
  params.kv_tile_indices =
      GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.kv_tile_indices_offset);
  params.kv_head_indices = plan_info.split_kv
                               ? GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.kv_head_indices_offset)
                               : nullptr;
  params.o_indptr = GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.o_indptr_offset);
  params.kv_chunk_size_ptr =
      GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.kv_chunk_size_ptr_offset);

  if (plan_info.split_kv) {
    tmp_v = GetPtrFromBaseOffset<DTypeO>(float_buffer, plan_info.v_offset);
    tmp_s = GetPtrFromBaseOffset<float>(float_buffer, plan_info.s_offset);
    if (plan_info.enable_cuda_graph) {
      params.block_valid_mask = plan_info.block_valid_mask_offset
                                    ? GetPtrFromBaseOffset<bool>(int_buffer, plan_info.block_valid_mask_offset)
                                    : nullptr;
    }
  }
  params.padded_batch_size = plan_info.padded_batch_size;

  cudaError_t status = flashinfer::BatchRectificationCacheDecodeWithPagedKVCacheDispatched<
      HEAD_DIM_QK, POS_ENCODING_MODE, AttentionVariant, Params>(params, tmp_v, tmp_s, enable_pdl,
                                                               /*stream=*/stream);
  TORCH_CHECK(status == cudaSuccess,
              "MACRectificationCache run failed with error ",
              cudaGetErrorString(status));
}
