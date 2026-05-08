#!/usr/bin/env python3

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple

import torch

from mac_attention import MACDecodeWithPagedKVCacheWrapper, load_mac_match_extension


def measure_cuda_time_ms(func_to_measure, num_runs: int = 100, num_warmup_runs: int = 10) -> float:
    for _ in range(num_warmup_runs):
        func_to_measure()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_runs):
        func_to_measure()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / float(num_runs)


def _attn_start_pos_val(kv_len: int, kv_access: float) -> int:
    # window_size_left = int(kv_access * (kv_len - 1))
    # attn_start_pos = (kv_len - 1) - window_size_left
    window = int(float(kv_access) * float(kv_len - 1))
    return int(kv_len - 1 - window)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    device = torch.device("cuda")
    dtype = torch.bfloat16

    _ = load_mac_match_extension(verbose=True)

    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    kv_lens = [32768, 65536, 131072, 262144]
    kv_accesses = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]

    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128
    page_size = 1

    max_running_requests = 128
    cache_capacity = 512
    downdate_range = 256

    # k/v: [num_blocks, page_size, Hkv, D] for kv_layout="NHD"
    num_blocks = 2621440
    k_cache = torch.empty((num_blocks, page_size, num_kv_heads, head_dim), device=device, dtype=dtype)
    v_cache = torch.empty_like(k_cache)

    # Unified cache tensors used when use_cache=True in forward_return_lse
    attn_cache = torch.empty(
        (max_running_requests, cache_capacity, num_qo_heads, head_dim), device=device, dtype=dtype
    )
    lse_cache = torch.empty(
        (max_running_requests, cache_capacity, num_qo_heads), device=device, dtype=torch.float32
    )

    # Wrapper workspace
    float_workspace = torch.empty(2048 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = MACDecodeWithPagedKVCacheWrapper(float_workspace, kv_layout="NHD")

    # Pinned host buffer for the tuned scheduler.
    attn_host_pinned = torch.empty(65536, dtype=torch.int32, device="cpu", pin_memory=True)

    # Pre-generate routing tensors per batch size.
    routing_by_b: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for b in batch_sizes:
        indices = torch.randint(0, cache_capacity, (b, num_qo_heads), device=device, dtype=torch.int32)
        hit_table = (torch.rand((b, num_qo_heads), device=device) < 0.7)
        req_ids = torch.arange(b, device=device, dtype=torch.int32)
        routing_by_b[b] = (indices, hit_table, req_ids)

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "bench_plan_attention_results.csv"

    fieldnames = [
        "batch_size",
        "context_length",
        "KV_Access",
        "standalone_plan_time_us",
        "standalone_attn_time_us",
    ]

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for b in batch_sizes:
            indices, hit_table, req_ids = routing_by_b[b]

            # q: [B, Hq, D]
            q = torch.empty((b, num_qo_heads, head_dim), device=device, dtype=dtype)

            # last_page_len is always 1 when page_size == 1
            last_page_len = torch.ones((b,), device=device, dtype=torch.int32)

            for kv_len in kv_lens:
                # indptr: [B+1], indices: [sum(pages_per_sample)] == [B*kv_len] when page_size==1
                indptr = torch.arange(0, (b + 1) * kv_len, step=kv_len, device=device, dtype=torch.int32)
                page_indices = torch.randint(0, num_blocks, (b * kv_len,), device=device, dtype=torch.int32)

                for kv_access in kv_accesses:
                    start_pos = _attn_start_pos_val(kv_len, kv_access)
                    attn_start_pos = torch.full(
                        (b, num_qo_heads), start_pos, device=device, dtype=torch.int32
                    )

                    def plan_once():
                        wrapper.plan(
                            indptr,
                            page_indices,
                            last_page_len,
                            num_qo_heads,
                            num_kv_heads,
                            head_dim,
                            page_size,
                            pos_encoding_mode="NONE",
                            q_data_type=dtype,
                            data_type=dtype,
                            attn_start_pos=attn_start_pos,
                            downdate_range=downdate_range,
                            attn_start_pos_host_pinned_opt=attn_host_pinned,
                        )

                    plan_once()

                    def run_once():
                        wrapper.forward_return_lse(
                            q,
                            (k_cache, v_cache),
                            max_running_requests,
                            cache_capacity,
                            attn_cache,
                            lse_cache,
                            indices,
                            hit_table,
                            req_ids,
                            True,
                            attn_start_pos,
                            downdate_range,
                        )

                    attn_ms = measure_cuda_time_ms(run_once)
                    plan_ms = measure_cuda_time_ms(plan_once)

                    writer.writerow(
                        {
                            "batch_size": b,
                            "context_length": kv_len,
                            "KV_Access": kv_access,
                            "standalone_plan_time_us": round(plan_ms * 1000.0, 2),
                            "standalone_attn_time_us": round(attn_ms * 1000.0, 2),
                        }
                    )
                    f.flush()

    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
