#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from mac_attention import load_mac_prefill_update_cache_extension


def measure_cuda_time_us(func_to_measure, *, num_runs: int = 200, num_warmup_runs: int = 50) -> float:
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
    return float(start.elapsed_time(end)) * 1000.0 / float(num_runs)


def _next_pow2_clamped(v: int) -> int:
    x = 1
    while x < v and x < 256:
        x <<= 1
    if x < 32:
        x = 32
    if x > 256:
        x = 256
    return x


def _threads_for_row_bytes(*, H: int, D: int) -> int:
    row_bf16 = H * D * 2
    row_f32 = H * 4
    row16 = max((row_bf16 + 15) // 16, (row_f32 + 15) // 16)
    return _next_pow2_clamped(int(row16))


def est_bytes(*, R: int, M: int, H: int, D: int, B: int, tokens_per_req: int) -> int:
    # Cache tensors
    q_bytes = R * M * H * D * 2
    a_bytes = q_bytes
    lse_bytes = R * M * H * 4
    rl_bytes = R * 4

    # Per-request metadata
    meta_bytes = (B + (B + 1) + B) * 4  # req_ids + offsets + lens

    # Sources
    sumN = B * tokens_per_req
    src_q_bytes = sumN * H * D * 2
    src_a_bytes = src_q_bytes
    src_l_bytes = sumN * H * 4
    return int(q_bytes + a_bytes + lse_bytes + rl_bytes + meta_bytes + src_q_bytes + src_a_bytes + src_l_bytes)


def _reference_update_cpu(
    *,
    q_cache: torch.Tensor,
    attn_cache: torch.Tensor,
    lse_cache: torch.Tensor,
    request_length: torch.Tensor,
    req_ids: torch.Tensor,
    src_q: torch.Tensor,
    src_attn: torch.Tensor,
    src_lse: torch.Tensor,
    offsets: torch.Tensor,
    lens: torch.Tensor,
    capacity: int,
) -> None:
    B = int(req_ids.numel())
    for by in range(B):
        req = int(req_ids[by].item())
        Ni = int(lens[by].item())
        if Ni <= 0 or capacity <= 0:
            continue

        start = (Ni - capacity) if (Ni > capacity) else 0
        keep = Ni - start
        L_before = int(request_length[req].item())
        base_off = int(offsets[by].item())
        base_dest = (L_before + start) % capacity

        for bx in range(keep):
            dest_slot = base_dest + bx
            if dest_slot >= capacity:
                dest_slot -= capacity
            src_row = base_off + start + bx
            q_cache[req, dest_slot].copy_(src_q[src_row])
            attn_cache[req, dest_slot].copy_(src_attn[src_row])
            lse_cache[req, dest_slot].copy_(src_lse[src_row])

        request_length[req] += Ni


def correctness_check(ext: Any) -> None:
    device = torch.device("cuda")

    R, M, H, D = 8, 8, 4, 16
    B = 4

    # Make lens cover both <=M and >M cases.
    lens_vals = [1, 2, 9, 16]
    lens = torch.tensor(lens_vals, device=device, dtype=torch.int32)
    offsets_vals = [0]
    for n in lens_vals:
        offsets_vals.append(offsets_vals[-1] + int(n))
    offsets = torch.tensor(offsets_vals, device=device, dtype=torch.int32)
    sumN = int(offsets_vals[-1])

    req_ids = torch.arange(B, device=device, dtype=torch.int32)

    q_init = torch.randn((R, M, H, D), device=device, dtype=torch.bfloat16)
    a_init = torch.randn_like(q_init)
    l_init = torch.randn((R, M, H), device=device, dtype=torch.float32)
    rl_init = torch.randint(0, 3 * M, (R,), device=device, dtype=torch.int32)

    src_q = torch.randn((sumN, H, D), device=device, dtype=torch.bfloat16)
    src_a = torch.randn_like(src_q)
    src_l = torch.randn((sumN, H), device=device, dtype=torch.float32)

    q_ref = q_init.cpu().clone()
    a_ref = a_init.cpu().clone()
    l_ref = l_init.cpu().clone()
    rl_ref = rl_init.cpu().clone()

    _reference_update_cpu(
        q_cache=q_ref,
        attn_cache=a_ref,
        lse_cache=l_ref,
        request_length=rl_ref,
        req_ids=req_ids.cpu(),
        src_q=src_q.cpu(),
        src_attn=src_a.cpu(),
        src_lse=src_l.cpu(),
        offsets=offsets.cpu(),
        lens=lens.cpu(),
        capacity=M,
    )

    q_out = q_init.clone()
    a_out = a_init.clone()
    l_out = l_init.clone()
    rl_out = rl_init.clone()

    ext.mac_prefill_update_cache(
        q_out,
        a_out,
        l_out,
        rl_out,
        req_ids,
        src_q,
        src_a,
        src_l,
        offsets,
        lens,
        M,
        H,
        D,
    )
    torch.cuda.synchronize()

    if not torch.equal(rl_out.cpu(), rl_ref):
        raise AssertionError("request_length mismatch")
    if not torch.equal(q_out.cpu(), q_ref):
        raise AssertionError("q_cache mismatch")
    if not torch.equal(a_out.cpu(), a_ref):
        raise AssertionError("attn_cache mismatch")
    if not torch.equal(l_out.cpu(), l_ref):
        raise AssertionError("lse_cache mismatch")


def _parse_int_list(xs: Sequence[int]) -> List[int]:
    return [int(x) for x in xs]


def main() -> None:
    ap = argparse.ArgumentParser("MAC-Attention prefill cache-update benchmark")
    ap.add_argument("--R", type=int, default=64, help="max running requests (ring buffer batch)")
    ap.add_argument("--M", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192])
    ap.add_argument("--H", type=int, nargs="+", default=[8, 32])
    ap.add_argument("--D", type=int, nargs="+", default=[128])
    ap.add_argument("--B", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64], help="batch size")
    ap.add_argument("--tokens_per_req", type=int, default=1, help="number of new rows per request (lens)")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--verbose_build", action="store_true")
    ap.add_argument(
        "--csv",
        type=str,
        default=str((Path(__file__).resolve().parent / "results" / "bench_prefill_update_cache_results.csv")),
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    total_mem = int(props.total_memory)
    print(f"GPU: {props.name} ({total_mem / (1024**3):.2f} GiB)")

    print("Compiling/loading extension...")
    ext = load_mac_prefill_update_cache_extension(verbose=bool(args.verbose_build))

    print("Running correctness check...")
    correctness_check(ext)
    print("Correctness: OK")

    R = int(args.R)
    Ms = _parse_int_list(args.M)
    Hs = _parse_int_list(args.H)
    Ds = _parse_int_list(args.D)
    Bs = _parse_int_list(args.B)
    tokens_per_req = int(args.tokens_per_req)
    warmup = int(args.warmup)
    iters = int(args.iters)

    out_csv = Path(args.csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "gpu",
        "R",
        "M",
        "H",
        "D",
        "B",
        "tokens_per_req",
        "threads",
        "avg_us",
    ]

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for M in Ms:
            for H in Hs:
                for D in Ds:
                    threads = _threads_for_row_bytes(H=H, D=D)
                    for B in Bs:
                        if B > R:
                            print(f"[SKIP] B={B} > R={R}")
                            continue

                        need = est_bytes(R=R, M=M, H=H, D=D, B=B, tokens_per_req=tokens_per_req)
                        if need > int(0.65 * total_mem):
                            print(
                                f"[SKIP] R={R} M={M} H={H} D={D} B={B}: "
                                f"estimated {need / (1024**3):.2f} GiB > 65% of GPU memory"
                            )
                            continue

                        q_cache = torch.empty((R, M, H, D), device=device, dtype=torch.bfloat16)
                        attn_cache = torch.empty_like(q_cache)
                        lse_cache = torch.empty((R, M, H), device=device, dtype=torch.float32)
                        request_length = torch.zeros((R,), device=device, dtype=torch.int32)

                        req_ids = torch.arange(B, device=device, dtype=torch.int32)
                        lens = torch.full((B,), tokens_per_req, device=device, dtype=torch.int32)
                        offsets = torch.arange(B + 1, device=device, dtype=torch.int32) * int(tokens_per_req)
                        sumN = int(B * tokens_per_req)

                        src_q = torch.randn((sumN, H, D), device=device, dtype=torch.bfloat16)
                        src_attn = torch.randn_like(src_q)
                        src_lse = torch.randn((sumN, H), device=device, dtype=torch.float32)

                        def run_once() -> None:
                            ext.mac_prefill_update_cache(
                                q_cache,
                                attn_cache,
                                lse_cache,
                                request_length,
                                req_ids,
                                src_q,
                                src_attn,
                                src_lse,
                                offsets,
                                lens,
                                M,
                                H,
                                D,
                            )

                        avg_us = measure_cuda_time_us(run_once, num_runs=iters, num_warmup_runs=warmup)

                        row: Dict[str, object] = {
                            "gpu": props.name,
                            "R": R,
                            "M": M,
                            "H": H,
                            "D": D,
                            "B": B,
                            "tokens_per_req": tokens_per_req,
                            "threads": threads,
                            "avg_us": f"{avg_us:.4f}",
                        }
                        writer.writerow(row)
                        print(f"[OK] R={R} M={M} H={H} D={D} B={B} T={tokens_per_req}: {avg_us:.3f} us")

    print(f"Saved CSV to {out_csv}")


if __name__ == "__main__":
    main()
