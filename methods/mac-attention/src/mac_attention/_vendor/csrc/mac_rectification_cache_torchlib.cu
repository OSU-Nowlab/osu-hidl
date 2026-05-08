/*
 * Torch operator registrations for the mac_attention MAC rectification+cache op.
 */
#include "mac_rectification_cache_config.inc"
#include "pytorch_extension_utils.h"

at::Tensor MACRectificationCacheWithPagedKVCachePlan(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
    at::Tensor page_locked_int_workspace_buffer, at::Tensor indptr, int64_t batch_size,
    int64_t num_qo_heads, int64_t num_kv_heads, int64_t page_size, bool enable_cuda_graph,
    int64_t window_left, double logits_soft_cap, int64_t head_dim_qk, int64_t head_dim_vo);

void MACRectificationCacheWithPagedKVCacheRun(
    at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer, at::Tensor plan_info_vec,
    at::Tensor q, at::Tensor paged_k_cache, at::Tensor paged_v_cache, at::Tensor paged_kv_indptr,
    at::Tensor paged_kv_indices, at::Tensor paged_kv_last_page_len, int64_t max_running_requests,
    int64_t cache_capacity, at::Tensor full_attn, at::Tensor full_lse, at::Tensor q_to_cache,
    at::Tensor query_cache, at::Tensor attn_cache, at::Tensor lse_cache, at::Tensor cache_req_ids,
    bool use_cache, at::Tensor o, std::optional<at::Tensor> maybe_lse, int64_t kv_layout_code,
    int64_t window_left, bool enable_pdl, at::Tensor maybe_alibi_slopes, double logits_soft_cap,
    double sm_scale, double rope_scale, double rope_theta);

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("plan", MACRectificationCacheWithPagedKVCachePlan);
  m.def("run", MACRectificationCacheWithPagedKVCacheRun);
}
