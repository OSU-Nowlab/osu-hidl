#!/usr/bin/env python3
"""
MPI Communication Validation

Quick validation of MPI communication correctness and basic performance.

Usage:
    mpirun -np 2 python validate.py
    mpirun -np 4 python validate.py --performance
"""

import argparse
import os
import time
import torch
import torch.distributed as dist


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='MPI Communication Validation')
    parser.add_argument(
        '--performance',
        action='store_true',
        help='Run performance tests in addition to correctness')
    parser.add_argument('--backend',
                        type=str,
                        default='mpi',
                        choices=['mpi', 'gloo'],
                        help='Distributed backend to use')
    parser.add_argument('--message_size',
                        type=int,
                        default=1000000,
                        help='Message size in elements for performance tests')
    return parser.parse_args()


def init_distributed(backend):
    """Initialize distributed process group."""
    # For Gloo with torchrun, use environment variables
    if backend == 'gloo':
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend=backend)
    else:
        # MPI backend
        dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size


def print_rank0(msg, rank=None):
    """Print only from rank 0."""
    if rank is None:
        rank = dist.get_rank()
    if rank == 0:
        print(msg)


def test_all_reduce_sum(rank, world_size, device):
    """Test all-reduce with sum operation."""
    tensor = torch.ones(100, device=device) * rank
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected = sum(range(world_size))
    return torch.allclose(tensor, torch.ones(100, device=device) * expected)


def test_all_reduce_max(rank, world_size, device):
    """Test all-reduce with max operation."""
    tensor = torch.ones(100, device=device) * rank
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    expected = world_size - 1
    return torch.allclose(tensor, torch.ones(100, device=device) * expected)


def test_broadcast(rank, world_size, device):
    """Test broadcast from rank 0."""
    if rank == 0:
        tensor = torch.ones(100, device=device) * 42
    else:
        tensor = torch.zeros(100, device=device)

    dist.broadcast(tensor, src=0)
    return torch.allclose(tensor, torch.ones(100, device=device) * 42)


def test_all_gather(rank, world_size, device):
    """Test all-gather operation."""
    tensor = torch.ones(100, device=device) * rank
    gather_list = [torch.zeros(100, device=device) for _ in range(world_size)]

    dist.all_gather(gather_list, tensor)

    for i, t in enumerate(gather_list):
        if not torch.allclose(t, torch.ones(100, device=device) * i):
            return False
    return True


def test_reduce_scatter(rank, world_size, device):
    """Test reduce-scatter operation."""
    # Each rank contributes data that will be scattered after reduction
    input_tensor = torch.ones(100 * world_size, device=device) * rank
    output_tensor = torch.zeros(100, device=device)

    input_list = list(input_tensor.chunk(world_size))
    dist.reduce_scatter(output_tensor, input_list, op=dist.ReduceOp.SUM)

    expected = sum(range(world_size))
    return torch.allclose(output_tensor,
                          torch.ones(100, device=device) * expected)


def test_barrier(rank, world_size, device):
    """Test barrier synchronization."""
    try:
        dist.barrier()
        return True
    except Exception:
        return False


def run_correctness_tests(rank, world_size, device):
    """Run all correctness tests."""
    tests = [
        ("All-reduce (sum)",
         lambda: test_all_reduce_sum(rank, world_size, device)),
        ("All-reduce (max)",
         lambda: test_all_reduce_max(rank, world_size, device)),
        ("Broadcast", lambda: test_broadcast(rank, world_size, device)),
        ("All-gather", lambda: test_all_gather(rank, world_size, device)),
        ("Reduce-scatter",
         lambda: test_reduce_scatter(rank, world_size, device)),
        ("Barrier", lambda: test_barrier(rank, world_size, device)),
    ]

    print_rank0("\nRunning correctness tests...", rank)

    passed = 0
    for name, test_fn in tests:
        dist.barrier()
        try:
            result = test_fn()
            status = "PASS" if result else "FAIL"
            if result:
                passed += 1
        except Exception as e:
            status = f"ERROR: {e}"

        print_rank0(f"  {name:.<30} {status}", rank)

    return passed, len(tests)


def run_performance_tests(rank, world_size, device, message_size):
    """Run basic performance tests."""
    print_rank0("\nRunning performance tests...", rank)

    # All-reduce latency
    tensor = torch.randn(message_size, device=device)

    # Warmup
    for _ in range(5):
        dist.all_reduce(tensor.clone(), op=dist.ReduceOp.SUM)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    dist.barrier()

    # Measure
    trials = 20
    start = time.perf_counter()
    for _ in range(trials):
        dist.all_reduce(tensor.clone(), op=dist.ReduceOp.SUM)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    dist.barrier()

    elapsed = time.perf_counter() - start
    avg_latency = (elapsed / trials) * 1000  # ms

    # Calculate bandwidth
    # All-reduce: 2*(N-1)/N * message_size for ring algorithm
    size_bytes = message_size * tensor.element_size()
    algo_factor = 2 * (world_size - 1) / world_size
    bandwidth = (size_bytes * algo_factor) / (elapsed / trials) / 1e9  # GB/s

    print_rank0(
        f"  All-reduce latency: {avg_latency:.2f} ms ({size_bytes/1e6:.1f}MB message)",
        rank)
    print_rank0(f"  All-reduce bandwidth: {bandwidth:.1f} GB/s", rank)

    # Broadcast latency
    tensor = torch.randn(message_size, device=device)

    for _ in range(5):
        dist.broadcast(tensor.clone(), src=0)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    dist.barrier()

    start = time.perf_counter()
    for _ in range(trials):
        dist.broadcast(tensor.clone(), src=0)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    dist.barrier()

    elapsed = time.perf_counter() - start
    avg_latency = (elapsed / trials) * 1000

    print_rank0(
        f"  Broadcast latency: {avg_latency:.2f} ms ({size_bytes/1e6:.1f}MB message)",
        rank)


def main():
    """Main entry point."""
    args = get_args()

    # Initialize distributed
    rank, world_size = init_distributed(args.backend)

    # Determine device using LOCAL_RANK for proper GPU assignment
    local_rank = int(
        os.environ.get('LOCAL_RANK', rank % max(1, torch.cuda.device_count())))
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    # Print header
    print_rank0("=" * 60, rank)
    print_rank0("MPI Communication Validation", rank)
    print_rank0("=" * 60, rank)
    print_rank0(f"Backend: {args.backend.upper()}", rank)
    print_rank0(f"World size: {world_size}", rank)
    print_rank0(f"CUDA available: {torch.cuda.is_available()}", rank)
    print_rank0(f"Device: {device}", rank)

    # Run correctness tests
    passed, total = run_correctness_tests(rank, world_size, device)

    # Run performance tests if requested
    if args.performance:
        run_performance_tests(rank, world_size, device, args.message_size)

    # Print summary
    print_rank0("", rank)
    print_rank0("=" * 60, rank)
    print_rank0("Validation Summary", rank)
    print_rank0("=" * 60, rank)
    print_rank0(f"Correctness: {passed}/{total} tests passed", rank)

    if passed == total:
        print_rank0("\nStatus: PASS", rank)
    else:
        print_rank0(f"\nStatus: FAIL ({total - passed} tests failed)", rank)

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
