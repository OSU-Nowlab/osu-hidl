import math

import pytest
import torch

from mac_attention import MACDecodeWithPagedKVCacheWrapper


def _ref_decode_attention(
    q: torch.Tensor,  # [B,Hq,D]
    k: torch.Tensor,  # [B,L,Hkv,D]
    v: torch.Tensor,  # [B,L,Hkv,D]
    attn_start_pos: torch.Tensor,  # [B,Hq] int32
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
    pos = torch.arange(L, device=q.device, dtype=torch.int32)[None, :, None]
    mask = pos >= attn_start_pos[:, None, :].to(torch.int32)
    logits = logits.masked_fill(~mask, float("-inf"))

    lse = torch.logsumexp(logits, dim=1)  # [B,Hq]
    w = torch.exp(logits - lse[:, None, :])
    out = (w.unsqueeze(-1) * v_hq.to(torch.float32)).sum(dim=1).to(q.dtype)  # [B,Hq,D]
    return out, lse


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mac_decode_matches_reference_no_cache():
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    B = 2
    Hkv = 2
    group_size = 4
    Hq = Hkv * group_size
    D = 128
    kv_len = 8
    page_size = 1

    # Paged KV cache (NHD): [max_pages, page_size, Hkv, D]
    total_pages = B * kv_len
    k_pages = torch.randn(total_pages, page_size, Hkv, D, device=device, dtype=dtype)
    v_pages = torch.randn(total_pages, page_size, Hkv, D, device=device, dtype=dtype)

    indptr = torch.tensor([0, kv_len, 2 * kv_len], device=device, dtype=torch.int32)
    indices = torch.arange(total_pages, device=device, dtype=torch.int32)
    last_page_len = torch.ones((B,), device=device, dtype=torch.int32)

    attn_start_pos = torch.zeros((B, Hq), device=device, dtype=torch.int32)

    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = MACDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
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
        attn_start_pos=attn_start_pos,
        downdate_range=0,
    )

    q = torch.randn(B, Hq, D, device=device, dtype=dtype)

    # Placeholders for cache inputs (ignored when use_cache=False)
    dummy_cache = torch.empty(1, device=device, dtype=dtype)
    dummy_f32 = torch.empty(1, device=device, dtype=torch.float32)
    dummy_i32 = torch.empty(1, device=device, dtype=torch.int32)
    dummy_bool = torch.empty(1, device=device, dtype=torch.bool)

    out, lse = wrapper.forward_return_lse(
        q,
        (k_pages, v_pages),
        0,
        0,
        dummy_cache,
        dummy_f32,
        dummy_i32,
        dummy_bool,
        dummy_i32,
        False,
        attn_start_pos,
        sm_scale=1.0 / math.sqrt(D),
    )

    # Build contiguous full KV sequences for reference.
    k_full = k_pages.squeeze(1).reshape(B, kv_len, Hkv, D)
    v_full = v_pages.squeeze(1).reshape(B, kv_len, Hkv, D)
    ref_out, ref_lse = _ref_decode_attention(q, k_full, v_full, attn_start_pos, 1.0 / math.sqrt(D))

    torch.testing.assert_close(out, ref_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(lse, ref_lse, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mac_decode_downdated_outputs_match_prefix_reference():
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    B = 2
    Hkv = 2
    group_size = 4
    Hq = Hkv * group_size
    D = 128
    kv_len = 8
    page_size = 1
    downdate_range = 2
    dd_start = kv_len - downdate_range - 1
    assert dd_start > 0

    total_pages = B * kv_len
    k_pages = torch.randn(total_pages, page_size, Hkv, D, device=device, dtype=dtype)
    v_pages = torch.randn(total_pages, page_size, Hkv, D, device=device, dtype=dtype)

    indptr = torch.tensor([0, kv_len, 2 * kv_len], device=device, dtype=torch.int32)
    indices = torch.arange(total_pages, device=device, dtype=torch.int32)
    last_page_len = torch.ones((B,), device=device, dtype=torch.int32)

    attn_start_pos = torch.zeros((B, Hq), device=device, dtype=torch.int32)

    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = MACDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
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
        attn_start_pos=attn_start_pos,
        downdate_range=downdate_range,
    )

    q = torch.randn(B, Hq, D, device=device, dtype=dtype)

    dummy_cache = torch.empty(1, device=device, dtype=dtype)
    dummy_f32 = torch.empty(1, device=device, dtype=torch.float32)
    dummy_i32 = torch.empty(1, device=device, dtype=torch.int32)
    dummy_bool = torch.empty(1, device=device, dtype=torch.bool)

    out, lse, downdated_out, downdated_lse = wrapper.forward_return_lse(
        q,
        (k_pages, v_pages),
        0,
        0,
        dummy_cache,
        dummy_f32,
        dummy_i32,
        dummy_bool,
        dummy_i32,
        False,
        attn_start_pos,
        downdate_range,  # positional compat
        sm_scale=1.0 / math.sqrt(D),
    )

    k_full = k_pages.squeeze(1).reshape(B, kv_len, Hkv, D)
    v_full = v_pages.squeeze(1).reshape(B, kv_len, Hkv, D)

    # Prefix reference: [0, dd_start) (since attn_start_pos is 0).
    ref_out, ref_lse = _ref_decode_attention(
        q, k_full[:, :dd_start], v_full[:, :dd_start], attn_start_pos, 1.0 / math.sqrt(D)
    )

    torch.testing.assert_close(downdated_out, ref_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(downdated_lse, ref_lse, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_plan_refreshes_when_indptr_changes():
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    B = 2
    Hkv = 2
    group_size = 4
    Hq = Hkv * group_size
    D = 128
    page_size = 1

    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = MACDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD")

    q = torch.randn(B, Hq, D, device=device, dtype=dtype)
    attn_start_pos = torch.zeros((B, Hq), device=device, dtype=torch.int32)

    dummy_cache = torch.empty(1, device=device, dtype=dtype)
    dummy_f32 = torch.empty(1, device=device, dtype=torch.float32)
    dummy_i32 = torch.empty(1, device=device, dtype=torch.int32)
    dummy_bool = torch.empty(1, device=device, dtype=torch.bool)

    for kv_len in (8, 6):
        total_pages = B * kv_len
        k_pages = torch.randn(total_pages, page_size, Hkv, D, device=device, dtype=dtype)
        v_pages = torch.randn(total_pages, page_size, Hkv, D, device=device, dtype=dtype)

        indptr = torch.tensor([0, kv_len, 2 * kv_len], device=device, dtype=torch.int32)
        indices = torch.arange(total_pages, device=device, dtype=torch.int32)
        last_page_len = torch.ones((B,), device=device, dtype=torch.int32)

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
            attn_start_pos=attn_start_pos,
            downdate_range=0,
        )

        out, _lse = wrapper.forward_return_lse(
            q,
            (k_pages, v_pages),
            0,
            0,
            dummy_cache,
            dummy_f32,
            dummy_i32,
            dummy_bool,
            dummy_i32,
            False,
            attn_start_pos,
            sm_scale=1.0 / math.sqrt(D),
        )

        k_full = k_pages.squeeze(1).reshape(B, kv_len, Hkv, D)
        v_full = v_pages.squeeze(1).reshape(B, kv_len, Hkv, D)
        ref_out, _ref_lse = _ref_decode_attention(
            q, k_full, v_full, attn_start_pos, 1.0 / math.sqrt(D)
        )
        torch.testing.assert_close(out, ref_out, rtol=2e-2, atol=2e-2)
