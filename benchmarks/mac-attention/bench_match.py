#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from mac_attention import load_mac_match_extension, load_standalone_extension


def build_ext(*, name: str, cu_path: str, verbose: bool = False) -> Any:
    return load_standalone_extension(
        name=name,
        sources=[str(cu_path)],
        verbose=verbose,
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-Xptxas",
            "-dlcm=cg",
        ],
    )


def bench(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1_000_000.0 / float(iters)


def choose_iters(M: int) -> Tuple[int, int]:
    if M <= 4096:
        return 300, 60
    if M <= 8192:
        return 200, 50
    if M <= 16384:
        return 120, 30
    if M <= 32768:
        return 60, 20
    return 30, 10


def measure_schedule_us(
    ext: Any,
    *,
    N: int,
    H: int,
    M: int,
    D: int,
    load_warps_override: int,
    target_util: float,
    allow_rpt_32: bool,
    num_runs: int = 50,
    num_warmup_runs: int = 10,
) -> Tuple[Dict[str, Any], float]:
    def schedule_once() -> Dict[str, Any]:
        return ext.mac_ring_match_schedule(
            int(N),
            int(H),
            int(M),
            int(D),
            int(load_warps_override),
            float(target_util),
            bool(allow_rpt_32),
        )

    # Avoid capturing unrelated async GPU work in the timing.
    torch.cuda.synchronize()
    plan: Dict[str, Any] = {}
    for _ in range(num_warmup_runs):
        plan = schedule_once()

    t0 = time.perf_counter()
    for _ in range(num_runs):
        plan = schedule_once()
    t1 = time.perf_counter()
    return plan, (t1 - t0) * 1_000_000.0 / float(num_runs)


@torch.inference_mode()
def l2_sq_at_winner_indices(
    q_cache: torch.Tensor, queries: torch.Tensor, req_ids: torch.Tensor, indices_out: torch.Tensor
) -> torch.Tensor:
    """
    Compute squared L2 between queries[n,h,:] and q_cache[req_ids[n], indices_out[n,h], h, :].
    Returns: [N,H] float32.
    """
    device = q_cache.device
    N, H, _D = queries.shape
    r_idx = req_ids.view(N, 1).expand(N, H)
    h_idx = torch.arange(H, device=device).view(1, H).expand(N, H)
    rows = q_cache[r_idx, indices_out, h_idx, :]  # [N,H,D] bf16
    diff = queries.float() - rows.float()
    return (diff * diff).sum(dim=-1)


def check_close_by_l2(
    q_cache: torch.Tensor,
    queries: torch.Tensor,
    req_ids: torch.Tensor,
    ref: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    out: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    rtol: float = 5e-3,
    atol: float = 1e-2,
) -> Tuple[bool, Dict[str, float]]:
    _hit_ref, _left_ref, ref_idx = ref
    _hit_out, _left_out, out_idx = out
    d_ref = l2_sq_at_winner_indices(q_cache, queries, req_ids, ref_idx)
    d_out = l2_sq_at_winner_indices(q_cache, queries, req_ids, out_idx)

    diff = (d_ref - d_out).abs()
    tol = atol + rtol * torch.maximum(d_ref.abs(), d_out.abs())
    mask = diff > tol

    stats = {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "median_abs": float(diff.median().item()),
        "violations": float(mask.sum().item()),
    }
    return (not bool(mask.any())), stats


def check_outputs_equal(
    ref: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    out: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> bool:
    hit_ref, left_ref, idx_ref = ref
    hit_out, left_out, idx_out = out
    return bool(
        torch.equal(hit_ref, hit_out) and torch.equal(left_ref, left_out) and torch.equal(idx_ref, idx_out)
    )


def call_mac_ring_match(
    ext: Any,
    *,
    q_cache: torch.Tensor,
    request_length: torch.Tensor,
    queries: torch.Tensor,
    req_ids: torch.Tensor,
    threshold: float,
    rows_per_stage: int,
    load_warps: int,
    my_offset: int,
    hit: Optional[torch.Tensor] = None,
    left: Optional[torch.Tensor] = None,
    idx: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    support_attr = "_mac_match_supports_prealloc"

    base_args = (
        q_cache,
        request_length,
        queries,
        req_ids,
        float(threshold),
        int(rows_per_stage),
        int(load_warps),
        int(my_offset),
    )

    result = None

    def _fallback_call():
        return ext.mac_ring_match(*base_args)

    def _inplace_call():
        return ext.mac_ring_match(*base_args, hit, left, idx)

    have_buffers = hit is not None and left is not None and idx is not None
    cached_support = getattr(ext, support_attr, None)

    if not have_buffers:
        result = _fallback_call()
    elif cached_support is True:
        result = _inplace_call()
    elif cached_support is False:
        result = _fallback_call()
    else:
        try:
            result = _inplace_call()
            setattr(ext, support_attr, True)
        except TypeError as err:
            if "incompatible function arguments" in str(err):
                setattr(ext, support_attr, False)
                result = _fallback_call()
            else:
                raise

    if result is None:
        if hit is None or left is None or idx is None:
            raise RuntimeError("ring_match returned None without in-place output tensors")
        return hit, left, idx

    hit_table, left_start, indices_out = result
    if hit_table.dtype != torch.bool:
        hit_table = hit_table.to(torch.bool)
    return hit_table, left_start, indices_out


def resolve_kernel_path(path: str, script_dir: Path) -> str:
    if not path:
        raise ValueError("Kernel path must be a non-empty string")
    if os.path.isabs(path):
        return path
    return str((script_dir / path).resolve())


def main() -> None:
    ap = argparse.ArgumentParser("MAC-Attention match-kernel benchmark")
    ap.add_argument(
        "--baseline",
        type=str,
        default="",
        help="Baseline .cu to compare against (expects mac_ring_match(...) binding). Empty disables baseline.",
    )
    ap.add_argument("--R", type=int, default=64)
    ap.add_argument("--M", type=int, nargs="+", default=[256, 512, 1024, 2048])
    ap.add_argument("--H", type=int, nargs="+", default=[8, 32])
    ap.add_argument("--D", type=int, nargs="+", default=[128])
    ap.add_argument("--N", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--rows_per_stage", type=int, default=0, help="0 = use scheduler")
    ap.add_argument("--load_warps", type=int, default=0, help="0 = use scheduler")
    ap.add_argument("--target_util", type=float, default=0.95)
    ap.add_argument("--allow_rpt_32", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--my_offset", type=int, default=256)
    ap.add_argument("--rtol", type=float, default=5e-3)
    ap.add_argument("--atol", type=float, default=1e-2)
    ap.add_argument(
        "--csv",
        type=str,
        default=str((Path(__file__).resolve().parent / "results" / "bench_match_results.csv")),
    )
    ap.add_argument("--ncu_profile", action="store_true", help="Single launch; skip correctness/CSV")
    ap.add_argument("--verbose_build", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    script_dir = Path(__file__).resolve().parent
    baseline_path = resolve_kernel_path(args.baseline, script_dir) if args.baseline else ""

    print("Compiling/loading extensions...")
    mac_ext = load_mac_match_extension(verbose=args.verbose_build)

    baseline_ext = None
    baseline_key = ""
    if baseline_path:
        baseline_key = Path(baseline_path).stem
        baseline_ext = build_ext(name=f"{baseline_key}_ext", cu_path=baseline_path, verbose=args.verbose_build)

    out_csv = Path(args.csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "R",
        "M",
        "H",
        "D",
        "N",
        "threshold",
        "my_offset",
        "baseline",
        "baseline_rows_per_stage",
        "baseline_load_warps",
        "baseline_block_threads",
        "baseline_num_tiles",
        "baseline_utilization",
        "baseline_schedule_us",
        "baseline_kernel_us",
        "baseline_us",
        "macMatch_rows_per_stage",
        "macMatch_load_warps",
        "macMatch_block_threads",
        "macMatch_num_tiles",
        "macMatch_utilization",
        "macMatch_schedule_us",
        "macMatch_kernel_us",
        "macMatch_us",
        "speedup",
        "output_correctness",
        "l2_correctness",
    ]

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        device = torch.device("cuda")

        R = int(args.R)
        Ms = [int(x) for x in args.M]
        Hs = [int(x) for x in args.H]
        Ds = [int(x) for x in args.D]
        Ns = [int(x) for x in args.N]

        cache_by_mh: Dict[Tuple[int, int, int], Tuple[torch.Tensor, torch.Tensor]] = {}

        for M in Ms:
            for H in Hs:
                for D in Ds:
                    q_cache = torch.randn((R, M, H, D), dtype=torch.bfloat16, device=device)
                    request_length = torch.randint(
                        low=0, high=2 * M, size=(R,), dtype=torch.int32, device=device
                    )
                    cache_by_mh[(M, H, D)] = (q_cache, request_length)

        for M in Ms:
            for H in Hs:
                for D in Ds:
                    q_cache, request_length = cache_by_mh[(M, H, D)]
                    iters, warmup = choose_iters(M)

                    for N in Ns:
                        if N > R:
                            continue

                        load_warps_override = int(args.load_warps) if args.load_warps > 0 else 0

                        baseline_plan: Dict[str, Any] = {}
                        baseline_schedule_us = ""
                        if baseline_ext is not None:
                            baseline_plan, baseline_schedule_us = measure_schedule_us(
                                baseline_ext,
                                N=N,
                                H=H,
                                M=M,
                                D=D,
                                load_warps_override=load_warps_override,
                                target_util=float(args.target_util),
                                allow_rpt_32=bool(args.allow_rpt_32),
                            )

                        mac_plan, mac_schedule_us = measure_schedule_us(
                            mac_ext,
                            N=N,
                            H=H,
                            M=M,
                            D=D,
                            load_warps_override=load_warps_override,
                            target_util=float(args.target_util),
                            allow_rpt_32=bool(args.allow_rpt_32),
                        )

                        baseline_rows_per_stage = (
                            int(args.rows_per_stage)
                            if args.rows_per_stage > 0
                            else (int(baseline_plan["rows_per_stage"]) if baseline_ext is not None else "")
                        )
                        baseline_load_warps = (
                            int(args.load_warps)
                            if args.load_warps > 0
                            else (int(baseline_plan["load_warps"]) if baseline_ext is not None else "")
                        )

                        mac_rows_per_stage = (
                            int(args.rows_per_stage) if args.rows_per_stage > 0 else int(mac_plan["rows_per_stage"])
                        )
                        mac_load_warps = (
                            int(args.load_warps) if args.load_warps > 0 else int(mac_plan["load_warps"])
                        )

                        queries = torch.randn((N, H, D), dtype=torch.bfloat16, device=device)
                        req_ids = torch.arange(N, dtype=torch.int32, device=device)

                        hit_base = torch.empty((N, H), device=device, dtype=torch.bool)
                        left_base = torch.empty((N, H), device=device, dtype=torch.int32)
                        idx_base = torch.empty((N, H), device=device, dtype=torch.int32)

                        hit_mac = torch.empty((N, H), device=device, dtype=torch.bool)
                        left_mac = torch.empty((N, H), device=device, dtype=torch.int32)
                        idx_mac = torch.empty((N, H), device=device, dtype=torch.int32)

                        def run_baseline_once():
                            if baseline_ext is None:
                                raise RuntimeError("baseline disabled")
                            return call_mac_ring_match(
                                baseline_ext,
                                q_cache=q_cache,
                                request_length=request_length,
                                queries=queries,
                                req_ids=req_ids,
                                threshold=float(args.threshold),
                                rows_per_stage=int(baseline_rows_per_stage),
                                load_warps=int(baseline_load_warps),
                                my_offset=int(args.my_offset),
                                hit=hit_base,
                                left=left_base,
                                idx=idx_base,
                            )

                        def run_mac_once():
                            return call_mac_ring_match(
                                mac_ext,
                                q_cache=q_cache,
                                request_length=request_length,
                                queries=queries,
                                req_ids=req_ids,
                                threshold=float(args.threshold),
                                rows_per_stage=mac_rows_per_stage,
                                load_warps=mac_load_warps,
                                my_offset=int(args.my_offset),
                                hit=hit_mac,
                                left=left_mac,
                                idx=idx_mac,
                            )

                        if args.ncu_profile:
                            torch.cuda.synchronize()
                            if baseline_ext is not None:
                                _ = run_baseline_once()
                            _ = run_mac_once()
                            torch.cuda.synchronize()
                            print(
                                f"NCU mode: ran once for R={R} M={M} H={H} D={D} N={N} "
                                f"base_rps={baseline_rows_per_stage} base_warps={baseline_load_warps} "
                                f"mac_rps={mac_rows_per_stage} mac_warps={mac_load_warps}"
                            )
                            continue

                        out_ref = None
                        baseline_kernel_us = ""
                        if baseline_ext is not None:
                            out_ref = run_baseline_once()
                            baseline_kernel_us = bench(run_baseline_once, iters, warmup)

                        out_mac = run_mac_once()
                        mac_kernel_us = bench(run_mac_once, iters, warmup)

                        output_correctness = ""
                        l2_correctness = ""
                        speedup = ""
                        if out_ref is not None:
                            output_ok = check_outputs_equal(out_ref, out_mac)
                            output_correctness = "OK" if output_ok else "FAIL"

                            ok, stats = check_close_by_l2(
                                q_cache,
                                queries,
                                req_ids,
                                out_ref,
                                out_mac,
                                rtol=float(args.rtol),
                                atol=float(args.atol),
                            )
                            l2_correctness = "OK" if ok else "FAIL"
                            baseline_us = float(baseline_schedule_us) + float(baseline_kernel_us)
                            mac_us = float(mac_schedule_us) + float(mac_kernel_us)
                            speedup = (baseline_us / mac_us) if mac_us > 0 else float("inf")
                            print(
                                f"R={R} M={M} H={H} D={D} N={N} | "
                                f"base_sched_us={float(baseline_schedule_us):.2f} base_us={baseline_us:.2f} "
                                f"mac_sched_us={mac_schedule_us:.2f} mac_us={mac_us:.2f} "
                                f"speedup={speedup:.4f}x | out={output_correctness} l2={l2_correctness} "
                                f"(max|Δ|={stats['max_abs']:.3e})"
                            )
                        else:
                            mac_us = mac_schedule_us + mac_kernel_us
                            print(
                                f"R={R} M={M} H={H} D={D} N={N} | "
                                f"mac_sched_us={mac_schedule_us:.2f} mac_us={mac_us:.2f}"
                            )

                        baseline_us = (
                            (float(baseline_schedule_us) + float(baseline_kernel_us))
                            if baseline_ext is not None
                            else ""
                        )
                        mac_us = float(mac_schedule_us) + float(mac_kernel_us)

                        row: Dict[str, object] = {
                            "R": R,
                            "M": M,
                            "H": H,
                            "D": D,
                            "N": N,
                            "threshold": float(args.threshold),
                            "my_offset": int(args.my_offset),
                            "baseline": baseline_key or "",
                            "baseline_rows_per_stage": baseline_rows_per_stage,
                            "baseline_load_warps": baseline_load_warps,
                            "baseline_block_threads": (
                                int(baseline_plan["block_threads"]) if baseline_ext is not None else ""
                            ),
                            "baseline_num_tiles": int(baseline_plan["num_tiles"]) if baseline_ext is not None else "",
                            "baseline_utilization": (
                                float(baseline_plan["utilization"]) if baseline_ext is not None else ""
                            ),
                            "baseline_schedule_us": baseline_schedule_us,
                            "baseline_kernel_us": baseline_kernel_us,
                            "baseline_us": baseline_us,
                            "macMatch_rows_per_stage": mac_rows_per_stage,
                            "macMatch_load_warps": mac_load_warps,
                            "macMatch_block_threads": int(mac_plan["block_threads"]),
                            "macMatch_num_tiles": int(mac_plan["num_tiles"]),
                            "macMatch_utilization": float(mac_plan["utilization"]),
                            "macMatch_schedule_us": mac_schedule_us,
                            "macMatch_kernel_us": mac_kernel_us,
                            "macMatch_us": mac_us,
                            "speedup": speedup,
                            "output_correctness": output_correctness,
                            "l2_correctness": l2_correctness,
                        }
                        writer.writerow(row)
                        f.flush()

    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
