#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from typing import Any, Dict, Tuple

import torch

from mac_attention import (
    MACDecodeWithPagedKVCacheWrapper,
    MACRectificationCacheWithPagedKVCacheWrapper,
    load_mac_match_extension,
)


def _match_schedule(
    ext: Any, *, N: int, H: int, M: int, D: int, target_util: float = 0.95, allow_rpt_32: bool = True
) -> Tuple[int, int, Dict[str, Any]]:
    plan = ext.mac_ring_match_schedule(int(N), int(H), int(M), int(D), int(0), float(target_util), bool(allow_rpt_32))
    return int(plan["rows_per_stage"]), int(plan["load_warps"]), plan


def main() -> None:
    ap = argparse.ArgumentParser("End-to-end MAC-Attention workflow example")
    ap.add_argument("--batch", type=int, default=4, help="B (#queries)")
    ap.add_argument("--Hq", type=int, default=8, help="#Q heads")
    ap.add_argument("--Hkv", type=int, default=2, help="#KV heads")
    ap.add_argument("--D", type=int, default=128, help="head_dim")
    ap.add_argument("--cache_capacity", type=int, default=512, help="Ring cache capacity (M)")
    ap.add_argument("--max_running_requests", type=int, default=64, help="Max requests (R)")
    ap.add_argument("--kv_len", type=int, default=1024, help="Context length for paged KV (page_size==1)")
    ap.add_argument(
        "--window_left",
        type=int,
        default=256,
        help="Fixed window_left for macRectificationCache",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Match threshold (higher is stricter; hit if L2 < sqrt(2D)*(1-threshold))",
    )
    ap.add_argument("--my_offset", type=int, default=256, help="Match offset (feeds attn_start_pos via left)")
    ap.add_argument("--steps", type=int, default=2, help="Run 2 steps to show miss->hit transition")
    ap.add_argument("--workspace_mb", type=int, default=512, help="Workspace size per wrapper (MiB)")
    ap.add_argument("--verbose_build", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this example")

    device = torch.device("cuda")
    dtype = torch.bfloat16

    B = int(args.batch)
    Hq = int(args.Hq)
    Hkv = int(args.Hkv)
    D = int(args.D)
    M = int(args.cache_capacity)
    R = int(args.max_running_requests)
    kv_len = int(args.kv_len)
    page_size = 1

    if B > R:
        raise ValueError(f"batch must be <= max_running_requests (got batch={B}, R={R})")
    if kv_len <= 0:
        raise ValueError("kv_len must be > 0")
    if int(args.my_offset) < 0:
        raise ValueError("my_offset must be >= 0")

    torch.manual_seed(0)

    # -------------------------------------------------------------------------
    # Setup: build/load the match extension and create all persistent buffers.
    # -------------------------------------------------------------------------
    mac_match = load_mac_match_extension(verbose=args.verbose_build)

    # Paged KV cache (page_size == 1): keep it tiny for a runnable example.
    # page_indices are in [0, num_blocks).
    num_blocks = B * kv_len
    k_cache = torch.randn((num_blocks, page_size, Hkv, D), device=device, dtype=dtype)
    v_cache = torch.randn_like(k_cache)

    # Page tables: each request uses a contiguous slice of length kv_len.
    indptr = torch.arange(0, (B + 1) * kv_len, step=kv_len, device=device, dtype=torch.int32)
    page_indices = torch.arange(0, B * kv_len, device=device, dtype=torch.int32)
    last_page_len = torch.ones((B,), device=device, dtype=torch.int32)

    # Unified ring caches (query_cache is the matcher input).
    query_cache = torch.zeros((R, M, Hq, D), device=device, dtype=dtype)
    attn_cache = torch.zeros_like(query_cache)
    lse_cache = torch.full((R, M, Hq), -1.0e9, device=device, dtype=torch.float32)

    # request_length drives match's left_start computation and ring semantics.
    request_length = torch.full((R,), kv_len, device=device, dtype=torch.int32)

    # Wrappers (JIT compilation happens on first plan()).
    ws_bytes = int(args.workspace_mb) * 1024 * 1024
    attn_ws = torch.empty((ws_bytes,), dtype=torch.uint8, device=device)
    cache_ws = torch.empty((ws_bytes,), dtype=torch.uint8, device=device)
    mac_attn = MACDecodeWithPagedKVCacheWrapper(attn_ws, kv_layout="NHD")
    rectification_cache = MACRectificationCacheWithPagedKVCacheWrapper(cache_ws, kv_layout="NHD")

    # macRectificationCache uses a fixed window_left, independent of match/attention outputs.
    rectification_cache.plan(
        indptr,
        page_indices,
        last_page_len,
        Hq,
        Hkv,
        D,
        page_size,
        pos_encoding_mode="NONE",
        q_data_type=dtype,
        data_type=dtype,
        window_left=int(args.window_left),
    )

    # Match scheduler (rows_per_stage/load_warps).
    rows_per_stage, load_warps, match_plan = _match_schedule(mac_match, N=B, H=Hq, M=M, D=D)
    print(
        "Match plan:",
        f"rows_per_stage={rows_per_stage}",
        f"load_warps={load_warps}",
        f"block_threads={int(match_plan['block_threads'])}",
        f"num_tiles={int(match_plan['num_tiles'])}",
        f"util={float(match_plan['utilization']):.3f}",
    )

    # -------------------------------------------------------------------------
    # Step loop: (1) match -> (2) attention -> (3) macRectificationCache.
    # -------------------------------------------------------------------------
    req_ids = torch.arange(B, device=device, dtype=torch.int32)

    q_prev = None
    for step in range(int(args.steps)):
        # Use the same q on step 1 to demonstrate a hit after caches are updated.
        if step == 1 and q_prev is not None:
            q = q_prev
        else:
            q = torch.randn((B, Hq, D), device=device, dtype=dtype)
            q_prev = q

        hit = torch.empty((B, Hq), device=device, dtype=torch.bool)
        left = torch.empty((B, Hq), device=device, dtype=torch.int32)
        idx = torch.empty((B, Hq), device=device, dtype=torch.int32)

        # 1) mac match kernel: queries + query_cache -> hit, left, idx.
        mac_match.mac_ring_match(
            query_cache,
            request_length,
            q,
            req_ids,
            float(args.threshold),
            int(rows_per_stage),
            int(load_warps),
            int(args.my_offset),
            hit,
            left,
            idx,
        )

        hit_rate = float(hit.float().mean().item())
        left_min = int(left.min().item())
        left_max = int(left.max().item())
        print(f"\n[step {step}] match: hit_rate={hit_rate:.3f} left_start=[{left_min},{left_max}]")

        # 2) mac attention kernel: queries + paged KV + caches + hit/left/idx -> o, lse for the actual attention output of this layer
        attn_start_pos = left
        attn_start_pos_host_pinned = torch.empty((B * Hq,), dtype=torch.int32, device="cpu", pin_memory=True)

        mac_attn.plan(
            indptr,
            page_indices,
            last_page_len,
            Hq,
            Hkv,
            D,
            page_size,
            pos_encoding_mode="NONE",
            q_data_type=dtype,
            data_type=dtype,
            attn_start_pos=attn_start_pos,
            downdate_range=int(args.my_offset),
            attn_start_pos_host_pinned_opt=attn_start_pos_host_pinned,
        )

        full_o, full_lse = mac_attn.forward_return_lse(
            q,
            (k_cache, v_cache),
            R,
            M,
            attn_cache,
            lse_cache,
            idx,
            hit,
            req_ids,
            True,
            attn_start_pos,
        )

        # 3) macRectificationCache: consumes (full_o, full_lse), updates caches, and returns windowed output (unused).
        rect_o, rect_lse = rectification_cache.forward_return_lse(
            q,
            (k_cache, v_cache),
            R,
            M,
            full_o,
            full_lse,
            q,
            query_cache,
            attn_cache,
            lse_cache,
            req_ids,
            True,
            window_left=int(args.window_left),
        )

        print(
            f"[step {step}] attention(full): o={tuple(full_o.shape)} lse={tuple(full_lse.shape)} | "
            f"macRectificationCache(window): o={tuple(rect_o.shape)} lse={tuple(rect_lse.shape)}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
