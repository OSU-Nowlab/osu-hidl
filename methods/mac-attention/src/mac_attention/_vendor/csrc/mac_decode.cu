/*
 * This file is derived from FlashInfer and adapted for the standalone mac_attention project.
 *
 * The goal is to expose only the MACDecodeWithPagedKVCacheWrapper operator with
 * the tuned MACAttention extensions (downdate_range + pinned host buffer planning).
 */
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/utils.cuh>
#include <optional>

#include "mac_decode_config.inc"
#include "pytorch_conversion_utils.h"
#include "pytorch_extension_utils.h"

namespace flashinfer {
template <uint32_t HEAD_DIM, PosEncodingMode POS_ENCODING_MODE, typename AttentionVariant,
          typename Params>
cudaError_t MACDecodeWithPagedKVCacheDispatched(
    Params params, typename Params::DTypeO* tmp_v, float* tmp_s, typename Params::DTypeO* tmp_v_dd,
    float* tmp_s_dd, bool enable_pdl, cudaStream_t stream);
}  // namespace flashinfer

using namespace flashinfer;

at::Tensor MACDecodeWithPagedKVCachePlan(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    at::Tensor page_locked_int_workspace_buffer, at::Tensor indptr, int64_t batch_size,
    int64_t num_qo_heads, int64_t num_kv_heads, int64_t page_size, bool enable_cuda_graph,
    int64_t window_left, double logits_soft_cap, int64_t head_dim_qk, int64_t head_dim_vo,
    at::Tensor attn_start_pos, int64_t downdate_range,
    at::Tensor attn_start_pos_host_pinned) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer.size(0) * float_workspace_buffer.element_size();
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer.size(0) * int_workspace_buffer.element_size();

  TORCH_CHECK(page_size == 1,
              "mac_attention v1 requires page_size == 1 (scheduler assumes token==page). Got ",
              page_size);

  TORCH_CHECK(attn_start_pos.scalar_type() == at::kInt,
              "attn_start_pos must be torch.int32 tensor");
  TORCH_CHECK(attn_start_pos.is_cuda(),
              "attn_start_pos tensor must reside on CUDA device during planning");
  TORCH_CHECK(attn_start_pos.numel() == batch_size * num_qo_heads,
              "attn_start_pos must have shape [batch_size, num_qo_heads]");
  auto attn_start_pos_cuda_contig = attn_start_pos.contiguous();

  TORCH_CHECK(attn_start_pos_host_pinned.device().is_cpu(),
              "attn_start_pos_host_pinned must be a CPU tensor");
  TORCH_CHECK(attn_start_pos_host_pinned.is_pinned(),
              "attn_start_pos_host_pinned must be pinned (page-locked) CPU memory");
  TORCH_CHECK(attn_start_pos_host_pinned.scalar_type() == at::kInt,
              "attn_start_pos_host_pinned must be torch.int32 tensor");
  TORCH_CHECK(attn_start_pos_host_pinned.numel() >= batch_size * num_qo_heads,
              "attn_start_pos_host_pinned must have numel >= batch_size * num_qo_heads");

  TORCH_CHECK(head_dim_qk == head_dim_vo,
              "CUDA cores template only supports equal head dim for QK and VO");
  TORCH_CHECK(static_cast<uint32_t>(head_dim_qk) == HEAD_DIM_QK,
              "This JIT module was built for head_dim ", HEAD_DIM_QK, ", got ", head_dim_qk);

  CHECK_GQA_HEAD_DIVISIBLE(num_qo_heads, num_kv_heads);

  MACAttentionDecodePlanInfo plan_info;

  const c10::cuda::OptionalCUDAGuard device_guard(float_workspace_buffer.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  DISPATCH_GQA_GROUP_SIZE(num_qo_heads / num_kv_heads, GROUP_SIZE, {
    std::vector<IdType> P, start_page_flat;
    IdType max_pages = 0;
    {
      cudaError_t build_status = BuildPAndStartPageFlat<IdType>(
          P, start_page_flat, max_pages,
          static_cast<IdType*>(indptr.data_ptr()),  // kv_indptr_h (CPU)
          static_cast<uint32_t>(batch_size),
          static_cast<uint32_t>(num_qo_heads),
          GROUP_SIZE,
          static_cast<uint32_t>(page_size),
          reinterpret_cast<const uint32_t*>(attn_start_pos_cuda_contig.data_ptr<int32_t>()),
          stream,
          reinterpret_cast<uint32_t*>(attn_start_pos_host_pinned.data_ptr<int32_t>()));
      TORCH_CHECK(build_status == cudaSuccess, "BuildPAndStartPageFlat failed with error ",
                  cudaGetErrorString(build_status));
    }

    auto work_estimation_func =
        [&](bool& split_kv, uint32_t& max_grid_size, uint32_t& kv_chunk_size_in_pages,
            uint32_t& new_batch_size, uint32_t& gdy, uint32_t /*batch_size_unused*/,
            IdType* /*kv_indptr_unused*/, const uint32_t /*num_qo_heads_unused*/,
            const uint32_t /*page_size_unused*/, bool enable_cuda_graph_,
            uint32_t downdate_range_) -> cudaError_t {
      return MACDecodeWithPagedKVCacheWorkEstimationHeadPackedPrepared<
          GROUP_SIZE, HEAD_DIM_QK, POS_ENCODING_MODE, AttentionVariant, Params>(
          split_kv, max_grid_size, kv_chunk_size_in_pages, new_batch_size, gdy,
          static_cast<uint32_t>(batch_size),
          static_cast<uint32_t>(num_qo_heads),
          static_cast<uint32_t>(page_size),
          enable_cuda_graph_, P, start_page_flat, downdate_range_);
    };

    cudaError_t status =
        MACAttentionDecodePlan<GROUP_SIZE, HEAD_DIM_QK, POS_ENCODING_MODE, AttentionVariant, Params>(
            static_cast<void*>(float_workspace_buffer.data_ptr()), float_workspace_size_in_bytes,
            static_cast<void*>(int_workspace_buffer.data_ptr()),
            static_cast<void*>(page_locked_int_workspace_buffer.data_ptr()),
            int_workspace_size_in_bytes, plan_info, P, start_page_flat,
            static_cast<IdType*>(indptr.data_ptr()), static_cast<uint32_t>(batch_size),
            static_cast<uint32_t>(num_qo_heads), static_cast<uint32_t>(page_size), enable_cuda_graph,
            /*stream=*/stream, work_estimation_func,
            reinterpret_cast<const uint32_t*>(attn_start_pos_cuda_contig.data_ptr<int32_t>()),
            static_cast<uint32_t>(downdate_range),
            reinterpret_cast<const uint32_t*>(attn_start_pos_host_pinned.data_ptr<int32_t>()));

    TORCH_CHECK(status == cudaSuccess, "MACAttentionDecodePlan failed with error ",
                cudaGetErrorString(status));
  });

  return vec_to_tensor(plan_info.ToVector());
}


void MACDecodeWithPagedKVCacheRun(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer, at::Tensor plan_info_vec,
    at::Tensor q, at::Tensor paged_k_cache, at::Tensor paged_v_cache, at::Tensor paged_kv_indptr,
    at::Tensor paged_kv_indices, at::Tensor paged_kv_last_page_len, int64_t max_running_requests,
    int64_t cache_capacity, at::Tensor attn_cache, at::Tensor lse_cache, at::Tensor hit_indices,
    at::Tensor hit_table, at::Tensor cache_req_ids, bool use_cache, at::Tensor attn_start_pos,
    int64_t downdate_range, at::Tensor o, std::optional<at::Tensor> maybe_lse,
    at::Tensor downdated_o, std::optional<at::Tensor> maybe_downdated_lse, int64_t kv_layout_code,
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
              "mac_attention v1 requires page_size == 1 (scheduler assumes token==page). Got ",
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
  if (maybe_downdated_lse) {
    const auto& lse = *maybe_downdated_lse;
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

  // This module is compiled for fixed dtypes; just map to the fixed C++ types.
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
  params.attn_cache = use_cache ? static_cast<DTypeO*>(attn_cache.data_ptr()) : nullptr;
  params.lse_cache = use_cache ? static_cast<float*>(lse_cache.data_ptr()) : nullptr;
  params.hit_indices = use_cache ? static_cast<uint32_t*>(hit_indices.data_ptr()) : nullptr;
  params.hit_table = use_cache ? static_cast<bool*>(hit_table.data_ptr()) : nullptr;
  params.cache_req_ids = use_cache ? static_cast<uint32_t*>(cache_req_ids.data_ptr()) : nullptr;
  params.attn_start_pos = static_cast<uint32_t*>(attn_start_pos.data_ptr());
  params.downdate_range = static_cast<int32_t>(downdate_range);

  params.o = static_cast<DTypeO*>(o.data_ptr());
  params.lse = maybe_lse ? static_cast<float*>(maybe_lse->data_ptr()) : nullptr;
  params.downdated_o = static_cast<DTypeO*>(downdated_o.data_ptr());
  params.downdated_lse =
      maybe_downdated_lse ? static_cast<float*>(maybe_downdated_lse->data_ptr()) : nullptr;

  params.maybe_alibi_slopes =
      maybe_alibi_slopes.numel() ? static_cast<float*>(maybe_alibi_slopes.data_ptr()) : nullptr;
  params.num_qo_heads = static_cast<uint32_t>(num_qo_heads);
  params.q_stride_n = static_cast<IdType>(q_stride_n);
  params.q_stride_h = static_cast<IdType>(q_stride_h);
  params.window_left = static_cast<int32_t>(window_left);
  params.logits_soft_cap = static_cast<float>(logits_soft_cap);
  params.sm_scale = static_cast<float>(sm_scale);
  params.rope_rcp_scale = static_cast<float>(1.0 / rope_scale);
  params.rope_rcp_theta = static_cast<float>(1.0 / rope_theta);

  // Schedule pointers (produced by plan)
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
  params.merge_start_offsets =
      plan_info.merge_start_offsets_offset
          ? GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.merge_start_offsets_offset)
          : nullptr;
  params.downdate_start_offsets =
      plan_info.downdate_start_offsets_offset
          ? GetPtrFromBaseOffset<IdType>(int_buffer, plan_info.downdate_start_offsets_offset)
          : nullptr;
  params.block_valid_mask = (plan_info.enable_cuda_graph && plan_info.block_valid_mask_offset)
                                ? GetPtrFromBaseOffset<bool>(int_buffer, plan_info.block_valid_mask_offset)
                                : nullptr;
  params.partition_kv = plan_info.split_kv;
  params.padded_batch_size = static_cast<uint32_t>(plan_info.padded_batch_size);

  // Temp buffers (split-kv path). The tuned kernel requires both MAIN and DD temporaries.
  DTypeO* tmp_v = nullptr;
  float* tmp_s = nullptr;
  DTypeO* tmp_v_dd = nullptr;
  float* tmp_s_dd = nullptr;
  if (plan_info.split_kv) {
    tmp_v = GetPtrFromBaseOffset<DTypeO>(float_buffer, plan_info.v_offset);
    tmp_s = GetPtrFromBaseOffset<float>(float_buffer, plan_info.s_offset);
    tmp_v_dd = GetPtrFromBaseOffset<DTypeO>(float_buffer, plan_info.v_dd_offset);
    tmp_s_dd = GetPtrFromBaseOffset<float>(float_buffer, plan_info.s_dd_offset);
  }

  cudaError_t status = flashinfer::MACDecodeWithPagedKVCacheDispatched<
      HEAD_DIM_QK, POS_ENCODING_MODE, AttentionVariant, Params>(params, tmp_v, tmp_s, tmp_v_dd,
                                                                tmp_s_dd, enable_pdl,
                                                                /*stream=*/stream);
  TORCH_CHECK(status == cudaSuccess, "MACDecodeWithPagedKVCacheDispatched failed: ",
              cudaGetErrorString(status));
}
