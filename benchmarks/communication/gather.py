#!/usr/bin/env python
"""
Gather communication benchmark.

Measures latency and bandwidth of gather operations from all ranks to root.
"""
import torch
import torch.distributed as dist
from .utils import (benchmark_parser, init_torch_distributed, get_dtype,
                    get_bandwidth, max_numel, benchmark_loop, print_header,
                    print_results, get_scan_sizes)


def run_gather_single(size, dtype, args):
    """Run gather benchmark for a single message size."""
    rank, world_size = init_torch_distributed(args.backend)

    # Create tensors
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Each rank sends 'size' elements
    send_tensor = torch.ones(size, dtype=dtype, device=device) * rank

    # Root receives from all ranks
    if rank == 0:
        recv_tensors = [
            torch.zeros(size, dtype=dtype, device=device)
            for _ in range(world_size)
        ]
    else:
        recv_tensors = None

    # Communication function
    def comm_fn(t):
        dist.gather(t, recv_tensors if rank == 0 else None, dst=0)

    # Run benchmark
    avg_latency = benchmark_loop(send_tensor,
                                 comm_fn,
                                 warmups=args.warmups,
                                 trials=args.trials)

    # Calculate metrics
    latency_ms = avg_latency * 1000
    size_bytes = size * send_tensor.element_size()
    bandwidth = get_bandwidth(size_bytes, avg_latency, args.bw_unit)

    # Print results
    print_results(rank, size, latency_ms, bandwidth, args.bw_unit, args.raw)

    return avg_latency


def run_gather_scan(args):
    """Run gather benchmark scanning across message sizes."""
    rank, world_size = init_torch_distributed(args.backend)
    dtype = get_dtype(args.dtype)

    # Print header
    print_header(rank, "Gather", world_size, args.dtype, args.backend)

    # Get sizes to scan
    sizes = get_scan_sizes(args.maxsize)

    for size in sizes:
        run_gather_single(size, dtype, args)


def main():
    parser = benchmark_parser()
    args = parser.parse_args()

    rank, world_size = init_torch_distributed(args.backend)
    dtype = get_dtype(args.dtype)

    if args.scan:
        # Scan across message sizes
        run_gather_scan(args)
    else:
        # Single large message size
        print_header(rank, "Gather", world_size, args.dtype, args.backend)
        # Divide by world_size since root gathers from all ranks
        max_size = max_numel(dtype, args.mem_factor) // world_size
        run_gather_single(max_size, dtype, args)


if __name__ == '__main__':
    main()
