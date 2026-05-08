#!/usr/bin/env python
"""
Barrier communication benchmark.

Measures latency of barrier synchronization across all ranks.
"""
import time
import torch
import torch.distributed as dist
from .utils import (benchmark_parser, init_torch_distributed, print_header,
                    print_results)


def run_barrier_benchmark(args):
    """Run barrier benchmark."""
    rank, world_size = init_torch_distributed(args.backend)

    # Print header
    print_header(rank, "Barrier", world_size, "N/A", args.backend)

    # Warmup
    for _ in range(args.warmups):
        dist.barrier()

    # Timed trials
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(args.trials):
        dist.barrier()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()
    avg_latency = (end_time - start_time) / args.trials
    latency_ms = avg_latency * 1000

    # Print results (barrier has no data size or bandwidth)
    if rank == 0:
        if args.raw:
            print(f"0,{latency_ms:.6f}")
        else:
            print(f"Barrier Latency: {latency_ms:.3f} ms")


def main():
    parser = benchmark_parser()
    args = parser.parse_args()

    run_barrier_benchmark(args)


if __name__ == '__main__':
    main()
