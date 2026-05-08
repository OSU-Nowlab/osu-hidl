import math

import pytest
import torch

from mac_attention import MACRectificationCacheWithPagedKVCacheWrapper


def _ref_decode_attention(
    q: torch.Tensor,  # [B,Hq,D]
    k: torch.Tensor,  # [B,L,Hkv,D]
    v: torch.Tensor,  # [B,L,Hkv,D]
    *,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, Hq, D = q.shape
    _, L, Hkv, _ = k.shape
    group_size = Hq // Hkv
    kv_head = (torch.arange(Hq, device=q.device) // group_size).to(torch.long)  # [Hq]

    k_hq = k[:, :, kv_head, :]  # [B,L,Hq,D]
    v_hq = v[:, :, kv_head, :]  # [B,L,Hq,D]

    logits = (q.unsqueeze(1).to(torch.float32) * k_hq.to(torch.float32)).sum(-1) * float(
        sm_scale
    )  # [B,L,Hq]
    lse = torch.logsumexp(logits, dim=1)  # [B,Hq] (ln)
    w = torch.exp(logits - lse[:, None, :])
    out = (w.unsqueeze(-1) * v_hq.to(torch.float32)).sum(dim=1)  # [B,Hq,D] fp32
    return out.to(q.dtype), lse


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mac_rectification_cache_window_only_matches_reference_no_cache():
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    B = 2
    Hkv = 2
    group_size = 4
    Hq = Hkv * group_size
    D = 128
    kv_len = 16
    page_size = 1

    window_left = 7  # last 8 tokens
    start = kv_len - (window_left + 1)
    assert start > 0

    total_pages = B * kv_len
    k_pages = torch.randn(total_pages, page_size, Hkv, D, device=device, dtype=dtype)
    v_pages = torch.randn(total_pages, page_size, Hkv, D, device=device, dtype=dtype)

    indptr = torch.tensor([0, kv_len, 2 * kv_len], device=device, dtype=torch.int32)
    indices = torch.arange(total_pages, device=device, dtype=torch.int32)
    last_page_len = torch.ones((B,), device=device, dtype=torch.int32)

    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = MACRectificationCacheWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        Hq,
        Hkv,
        D,
        page_size,
        pos_encoding_mode="NONE",
        q_data_type=dtype,
        data_type=dtype,
        window_left=window_left,
    )

    q = torch.randn(B, Hq, D, device=device, dtype=dtype)

    # Dummy tensors for cache path (ignored when use_cache=False)
    full_out = torch.empty_like(q)
    full_lse = torch.empty((B, Hq), device=device, dtype=torch.float32)
    q_to_cache = torch.empty_like(q)
    query_cache = torch.empty((B, 32, Hq, D), device=device, dtype=dtype)
    attn_cache = torch.empty_like(query_cache)
    lse_cache = torch.empty((B, 32, Hq), device=device, dtype=torch.float32)
    req_ids = torch.arange(B, device=device, dtype=torch.int32)

    out, lse = wrapper.forward_return_lse(
        q,
        (k_pages, v_pages),
        B,
        32,
        full_out,
        full_lse,
        q_to_cache,
        query_cache,
        attn_cache,
        lse_cache,
        req_ids,
        False,
        sm_scale=1.0 / math.sqrt(D),
    )

    # Reference window-only attention (no downdate).
    k_full = k_pages.squeeze(1).reshape(B, kv_len, Hkv, D)
    v_full = v_pages.squeeze(1).reshape(B, kv_len, Hkv, D)
    ref_out, ref_lse = _ref_decode_attention(
        q, k_full[:, start:], v_full[:, start:], sm_scale=1.0 / math.sqrt(D)
    )

    torch.testing.assert_close(out, ref_out, rtol=2e-2, atol=2e-2)
    # When use_cache=False, the underlying decode kernel returns LSE in log2 domain.
    ref_lse_log2 = ref_lse / math.log(2.0)
    torch.testing.assert_close(lse, ref_lse_log2, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mac_rectification_cache_downdate_and_cache_update_match_reference():
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    B = 2
    Hkv = 2
    group_size = 4
    Hq = Hkv * group_size
    D = 128
    kv_len = 16
    page_size = 1

    window_left = 7  # last 8 tokens
    start = kv_len - (window_left + 1)
    assert start > 0

    total_pages = B * kv_len
    k_pages = torch.randn(total_pages, page_size, Hkv, D, device=device, dtype=dtype)
    v_pages = torch.randn(total_pages, page_size, Hkv, D, device=device, dtype=dtype)

    indptr = torch.tensor([0, kv_len, 2 * kv_len], device=device, dtype=torch.int32)
    indices = torch.arange(total_pages, device=device, dtype=torch.int32)
    last_page_len = torch.ones((B,), device=device, dtype=torch.int32)

    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = MACRectificationCacheWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        Hq,
        Hkv,
        D,
        page_size,
        pos_encoding_mode="NONE",
        q_data_type=dtype,
        data_type=dtype,
        window_left=window_left,
    )

    q = torch.randn(B, Hq, D, device=device, dtype=dtype)
    q_to_cache = q.clone()

    # Reference full/window (ln domain), then compute complement (rest) in ln domain.
    k_full = k_pages.squeeze(1).reshape(B, kv_len, Hkv, D)
    v_full = v_pages.squeeze(1).reshape(B, kv_len, Hkv, D)
    full_out_ref, full_lse_ref = _ref_decode_attention(q, k_full, v_full, sm_scale=1.0 / math.sqrt(D))
    win_out_ref, win_lse_ref = _ref_decode_attention(
        q, k_full[:, start:], v_full[:, start:], sm_scale=1.0 / math.sqrt(D)
    )

    z_full = torch.exp(full_lse_ref)
    z_win = torch.exp(win_lse_ref)
    z_rest = z_full - z_win
    rest_lse_ref = torch.log(z_rest)
    rest_out_ref = ((full_out_ref.to(torch.float32) * z_full[:, :, None]) - (win_out_ref.to(torch.float32) * z_win[:, :, None])) / z_rest[:, :, None]
    rest_out_ref = rest_out_ref.to(dtype)

    # Cache tensors.
    max_running_requests = B
    cache_capacity = 32
    query_cache = torch.zeros((max_running_requests, cache_capacity, Hq, D), device=device, dtype=dtype)
    attn_cache = torch.zeros_like(query_cache)
    lse_cache = torch.full(
        (max_running_requests, cache_capacity, Hq), float("-inf"), device=device, dtype=torch.float32
    )
    req_ids = torch.arange(B, device=device, dtype=torch.int32)

    # Run op: expects full_lse in ln domain.
    out, lse = wrapper.forward_return_lse(
        q,
        (k_pages, v_pages),
        max_running_requests,
        cache_capacity,
        full_out_ref,
        full_lse_ref,
        q_to_cache,
        query_cache,
        attn_cache,
        lse_cache,
        req_ids,
        True,
        sm_scale=1.0 / math.sqrt(D),
    )

    torch.testing.assert_close(out, rest_out_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(lse, rest_lse_ref, rtol=2e-2, atol=2e-2)

    # Cache update: writes to slot (kv_len - 1) % M when page_size==1 and kv_len<M.
    slot = kv_len - 1
    torch.testing.assert_close(query_cache[0, slot], q_to_cache[0], rtol=0, atol=0)
    torch.testing.assert_close(attn_cache[0, slot], rest_out_ref[0], rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(lse_cache[0, slot], rest_lse_ref[0], rtol=2e-2, atol=2e-2)
