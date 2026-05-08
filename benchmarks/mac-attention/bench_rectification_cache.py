#!/usr/bin/env python3

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Tuple

import torch

from mac_attention import MACRectificationCacheWithPagedKVCacheWrapper


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


def _window_left_val(kv_len: int, kv_access: int) -> int:
    return min(int(kv_access), int(kv_len - 1))


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    device = torch.device("cuda")
    dtype = torch.bfloat16

    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    kv_lens = [32768, 65536, 131072, 262144]
    kv_accesses = [64, 128, 256, 512, 1024, 2048]

    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128
    page_size = 1

    max_running_requests = 128
    cache_capacity = 512

    num_blocks = 2621440
    k_cache = torch.empty((num_blocks, page_size, num_kv_heads, head_dim), device=device, dtype=dtype)
    v_cache = torch.empty_like(k_cache)

    # Unified ring caches.
    query_cache = torch.empty(
        (max_running_requests, cache_capacity, num_qo_heads, head_dim), device=device, dtype=dtype
    )
    attn_cache = torch.empty_like(query_cache)
    lse_cache = torch.empty(
        (max_running_requests, cache_capacity, num_qo_heads), device=device, dtype=torch.float32
    )

    float_workspace = torch.empty(2048 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = MACRectificationCacheWithPagedKVCacheWrapper(float_workspace, kv_layout="NHD")

    # Dummy request ids per batch size.
    req_ids_by_b: Dict[int, torch.Tensor] = {}
    for b in batch_sizes:
        req_ids_by_b[b] = torch.arange(b, device=device, dtype=torch.int32)

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "bench_rectification_cache_results.csv"

    fieldnames = [
        "batch_size",
        "context_length",
        "KV_Access",
        "standalone_plan_time_us",
        "standalone_macRectificationCache_time_us",
    ]

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for b in batch_sizes:
            req_ids = req_ids_by_b[b]

            q = torch.empty((b, num_qo_heads, head_dim), device=device, dtype=dtype)
            q_to_cache = torch.empty_like(q)

            # Placeholders: full attention output/LSE (inputs for downdate).
            full_attn = torch.empty_like(q)
            full_lse = torch.empty((b, num_qo_heads), device=device, dtype=torch.float32)

            last_page_len = torch.ones((b,), device=device, dtype=torch.int32)

            for kv_len in kv_lens:
                indptr = torch.arange(0, (b + 1) * kv_len, step=kv_len, device=device, dtype=torch.int32)
                page_indices = torch.randint(0, num_blocks, (b * kv_len,), device=device, dtype=torch.int32)

                full_lse.fill_(math.log(float(kv_len)))

                for kv_access in kv_accesses:
                    window_left = _window_left_val(kv_len, kv_access)

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
                            window_left=window_left,
                        )

                    plan_once()

                    def run_once():
                        wrapper.forward_return_lse(
                            q,
                            (k_cache, v_cache),
                            max_running_requests,
                            cache_capacity,
                            full_attn,
                            full_lse,
                            q_to_cache,
                            query_cache,
                            attn_cache,
                            lse_cache,
                            req_ids,
                            True,
                            window_left=window_left,
                        )

                    attn_ms = measure_cuda_time_ms(run_once)
                    plan_ms = measure_cuda_time_ms(plan_once)

                    writer.writerow(
                        {
                            "batch_size": b,
                            "context_length": kv_len,
                            "KV_Access": kv_access,
                            "standalone_plan_time_us": round(plan_ms * 1000.0, 2),
                            "standalone_macRectificationCache_time_us": round(attn_ms * 1000.0, 2),
                        }
                    )
                    f.flush()

    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
