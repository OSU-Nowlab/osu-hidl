#!/usr/bin/env python
"""
All-Reduce communication benchmark.

Measures latency and bandwidth of all_reduce operations across all ranks.
"""
import torch
import torch.distributed as dist
from .utils import (benchmark_parser, init_torch_distributed, get_dtype,
                    get_bandwidth, get_algbw, max_numel, benchmark_loop,
                    print_header, print_results, get_scan_sizes)


def run_all_reduce_single(size, dtype, args):
    """Run all_reduce benchmark for a single message size."""
    rank, world_size = init_torch_distributed(args.backend)

    # Create tensor
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    tensor = torch.ones(size, dtype=dtype, device=device)

    # Communication function
    def comm_fn(t):
        dist.all_reduce(t, op=dist.ReduceOp.SUM)

    # Run benchmark
    avg_latency = benchmark_loop(tensor,
                                 comm_fn,
                                 warmups=args.warmups,
                                 trials=args.trials)

    # Calculate metrics
    latency_ms = avg_latency * 1000
    size_bytes = size * tensor.element_size()

    # All-reduce algorithm bandwidth factor: 2*(N-1)/N for ring algorithm
    algo_factor = 2 * (world_size - 1) / world_size
    bandwidth = get_algbw(size_bytes, avg_latency, algo_factor, args.bw_unit)

    # Print results
    print_results(rank, size, latency_ms, bandwidth, args.bw_unit, args.raw)

    return avg_latency


def run_all_reduce_scan(args):
    """Run all_reduce benchmark scanning across message sizes."""
    rank, world_size = init_torch_distributed(args.backend)
    dtype = get_dtype(args.dtype)

    # Print header
    print_header(rank, "All-Reduce", world_size, args.dtype, args.backend)

    # Get sizes to scan
    sizes = get_scan_sizes(args.maxsize)

    for size in sizes:
        run_all_reduce_single(size, dtype, args)


def main():
    parser = benchmark_parser()
    args = parser.parse_args()

    rank, world_size = init_torch_distributed(args.backend)
    dtype = get_dtype(args.dtype)

    if args.scan:
        # Scan across message sizes
        run_all_reduce_scan(args)
    else:
        # Single large message size
        print_header(rank, "All-Reduce", world_size, args.dtype, args.backend)
        max_size = max_numel(dtype, args.mem_factor)
        run_all_reduce_single(max_size, dtype, args)


if __name__ == '__main__':
    main()
