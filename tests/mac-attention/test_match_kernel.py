import pytest
import torch

from mac_attention import load_mac_match_extension


def _global_index(local_slot: int, length: int, capacity: int) -> int:
    if length <= 0:
        return 0
    if length < capacity:
        return local_slot
    base = length - capacity
    tail = length % capacity
    order_idx = local_slot - tail
    if order_idx < 0:
        order_idx += capacity
    return base + order_idx


def _reference_match(
    q_cache: torch.Tensor,
    request_length: torch.Tensor,
    queries: torch.Tensor,
    *,
    threshold: float,
    my_offset: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_queries, n_heads, head_dim = queries.shape
    capacity = q_cache.size(1)
    hit = torch.zeros((n_queries, n_heads), dtype=torch.bool)
    left = torch.zeros((n_queries, n_heads), dtype=torch.int32)
    idx = torch.zeros((n_queries, n_heads), dtype=torch.int32)

    one_minus = 1.0 - float(threshold)
    allow = one_minus > 0.0
    threshold_sq = 2.0 * float(head_dim) * one_minus * one_minus

    for n in range(n_queries):
        length = int(request_length[n].item())
        valid = min(length, capacity)
        for h in range(n_heads):
            best_val = float("inf")
            best_idx = 0
            for local_slot in range(valid):
                row = q_cache[n, local_slot, h].to(torch.float32)
                query = queries[n, h].to(torch.float32)
                dist = float(((query - row) * (query - row)).sum().item())
                if dist < best_val or (dist == best_val and local_slot < best_idx):
                    best_val = dist
                    best_idx = local_slot
            idx[n, h] = best_idx
            is_hit = allow and best_val < threshold_sq and best_val < float("inf")
            hit[n, h] = is_hit
            if is_hit:
                global_idx = _global_index(best_idx, length, capacity)
                left[n, h] = max(global_idx + 1 - my_offset, 0)
    return hit, left, idx


def _call_match(
    ext,
    q_cache: torch.Tensor,
    request_length: torch.Tensor,
    queries: torch.Tensor,
    req_ids: torch.Tensor,
    *,
    threshold: float,
    rows_per_stage: int,
    load_warps: int,
    my_offset: int,
    use_prealloc: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if use_prealloc:
        hit = torch.empty((queries.size(0), queries.size(1)), device=queries.device, dtype=torch.bool)
        left = torch.empty((queries.size(0), queries.size(1)), device=queries.device, dtype=torch.int32)
        idx = torch.empty((queries.size(0), queries.size(1)), device=queries.device, dtype=torch.int32)
        result = ext.mac_ring_match(
            q_cache,
            request_length,
            queries,
            req_ids,
            float(threshold),
            int(rows_per_stage),
            int(load_warps),
            int(my_offset),
            hit,
            left,
            idx,
        )
        if result is None:
            return hit, left, idx
        return result

    return ext.mac_ring_match(
        q_cache,
        request_length,
        queries,
        req_ids,
        float(threshold),
        int(rows_per_stage),
        int(load_warps),
        int(my_offset),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mac_match_matches_reference_with_wrapped_ring():
    device = torch.device("cuda")
    dtype = torch.bfloat16

    request_length = torch.tensor([3, 4, 6], device=device, dtype=torch.int32)
    n_queries = int(request_length.numel())
    capacity = 4
    n_heads = 2
    head_dim = 8
    threshold = 0.95
    my_offset = 2

    q_cache = torch.empty((n_queries, capacity, n_heads, head_dim), device=device, dtype=dtype)
    for n in range(n_queries):
        for slot in range(capacity):
            for h in range(n_heads):
                value = n * 100 + slot * 10 + h
                q_cache[n, slot, h].fill_(value)

    queries = torch.empty((n_queries, n_heads, head_dim), device=device, dtype=dtype)
    queries[0, 0].copy_(q_cache[0, 2, 0])
    queries[0, 1].copy_(q_cache[0, 1, 1])
    queries[1, 0].copy_(q_cache[1, 3, 0])
    queries[1, 1].fill_(999)
    queries[2, 0].fill_(999)
    queries[2, 1].copy_(q_cache[2, 0, 1])

    req_ids = torch.arange(n_queries, device=device, dtype=torch.int32)
    ext = load_mac_match_extension(verbose=False)
    plan = ext.mac_ring_match_schedule(n_queries, n_heads, capacity, head_dim, 0, 0.95, False)

    ref_hit, ref_left, ref_idx = _reference_match(
        q_cache,
        request_length,
        queries,
        threshold=threshold,
        my_offset=my_offset,
    )

    out_hit, out_left, out_idx = _call_match(
        ext,
        q_cache,
        request_length,
        queries,
        req_ids,
        threshold=threshold,
        rows_per_stage=int(plan["rows_per_stage"]),
        load_warps=int(plan["load_warps"]),
        my_offset=my_offset,
        use_prealloc=True,
    )

    torch.testing.assert_close(out_hit.cpu(), ref_hit)
    torch.testing.assert_close(out_left.cpu(), ref_left)
    torch.testing.assert_close(out_idx.cpu(), ref_idx)

    out_hit_2, out_left_2, out_idx_2 = _call_match(
        ext,
        q_cache,
        request_length,
        queries,
        req_ids,
        threshold=threshold,
        rows_per_stage=int(plan["rows_per_stage"]),
        load_warps=int(plan["load_warps"]),
        my_offset=my_offset,
        use_prealloc=False,
    )

    torch.testing.assert_close(out_hit_2.cpu(), ref_hit)
    torch.testing.assert_close(out_left_2.cpu(), ref_left)
    torch.testing.assert_close(out_idx_2.cpu(), ref_idx)
