"""
Utility functions for communication benchmarks.
"""
import argparse
import os
import time
import torch
import torch.distributed as dist


def benchmark_parser():
    """Common argument parser for all communication benchmarks."""
    parser = argparse.ArgumentParser(description='Communication Benchmark')
    parser.add_argument('--local_rank',
                        type=int,
                        default=0,
                        help='Local rank passed from distributed launcher')
    parser.add_argument('--trials',
                        type=int,
                        default=100,
                        help='Number of timed iterations')
    parser.add_argument('--warmups',
                        type=int,
                        default=50,
                        help='Number of warmup (non-timed) iterations')
    parser.add_argument('--maxsize',
                        type=int,
                        default=24,
                        help='Max message size as a power of 2')
    parser.add_argument('--bw-unit',
                        type=str,
                        default='Gbps',
                        choices=['Gbps', 'GBps'],
                        help='Bandwidth unit')
    parser.add_argument('--backend',
                        type=str,
                        default='mpi',
                        choices=['mpi', 'gloo'],
                        help='Communication backend to use')
    parser.add_argument('--scan',
                        action='store_true',
                        help='Enables scanning all message sizes')
    parser.add_argument('--raw',
                        action='store_true',
                        help='Print raw data without units')
    parser.add_argument('--dtype',
                        type=str,
                        default='float32',
                        help='PyTorch tensor dtype')
    parser.add_argument('--mem-factor',
                        type=float,
                        default=0.5,
                        help='Proportion of max available GPU memory to use')

    # Add operation-specific flags (will be ignored by individual benchmarks)
    parser.add_argument('--all-reduce',
                        action='store_true',
                        help='Run all_reduce')
    parser.add_argument('--broadcast',
                        action='store_true',
                        help='Run broadcast')
    parser.add_argument('--all-gather',
                        action='store_true',
                        help='Run all_gather')
    parser.add_argument('--gather', action='store_true', help='Run gather')
    parser.add_argument('--scatter', action='store_true', help='Run scatter')
    parser.add_argument('--barrier', action='store_true', help='Run barrier')

    return parser


def env2int(env_list, default=-1):
    """Get integer from environment variable list."""
    for e in env_list:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return default


def init_torch_distributed(backend='mpi'):
    """
    Initialize PyTorch distributed with proper environment setup.
    Handles MPI, SLURM, and other launchers.
    """
    import os

    if not dist.is_initialized():
        # Set MASTER_PORT if not set
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '29500'

        # Set MASTER_ADDR if not set (default to localhost for single-node)
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = 'localhost'

        # Get local rank from various MPI/SLURM environment variables
        local_rank = env2int([
            'LOCAL_RANK', 'MPI_LOCALRANKID', 'OMPI_COMM_WORLD_LOCAL_RANK',
            'MV2_COMM_WORLD_LOCAL_RANK', 'SLURM_LOCALID'
        ],
                             default=0)

        if 'LOCAL_RANK' not in os.environ:
            os.environ['LOCAL_RANK'] = str(local_rank)

        # Get global rank
        rank = env2int([
            'RANK', 'MPI_RANKID', 'OMPI_COMM_WORLD_RANK',
            'MV2_COMM_WORLD_RANK', 'SLURM_PROCID'
        ],
                       default=0)

        if 'RANK' not in os.environ:
            os.environ['RANK'] = str(rank)

        # Get world size
        world_size = env2int([
            'WORLD_SIZE', 'OMPI_COMM_WORLD_SIZE', 'MV2_COMM_WORLD_SIZE',
            'SLURM_NPROCS'
        ],
                             default=1)

        if 'WORLD_SIZE' not in os.environ:
            os.environ['WORLD_SIZE'] = str(world_size)

        # Initialize distributed
        dist.init_process_group(backend=backend)

        # Set CUDA device
        if torch.cuda.is_available():
            local_rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(local_rank)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    return rank, world_size


def get_dtype(dtype_str):
    """Convert dtype string to torch dtype."""
    dtype_map = {
        'float32': torch.float32,
        'float': torch.float32,
        'float16': torch.float16,
        'half': torch.float16,
        'bfloat16': torch.bfloat16,
        'int32': torch.int32,
        'int': torch.int32,
        'int64': torch.int64,
        'long': torch.int64,
    }
    return dtype_map.get(dtype_str.lower(), torch.float32)


def get_bandwidth(size_bytes, duration_sec, bw_unit='Gbps'):
    """
    Calculate bandwidth from size and duration.

    Args:
        size_bytes: Size in bytes
        duration_sec: Duration in seconds
        bw_unit: 'Gbps' (gigabits per second) or 'GBps' (gigabytes per second)

    Returns:
        Bandwidth in specified units
    """
    if bw_unit == 'Gbps':
        return (size_bytes * 8) / (duration_sec * 1e9)
    elif bw_unit == 'GBps':
        return size_bytes / (duration_sec * 1e9)
    else:
        raise ValueError(f"Unknown bandwidth unit: {bw_unit}")


def get_algbw(size_bytes, duration_sec, algo_factor, bw_unit='Gbps'):
    """
    Calculate algorithm bandwidth (accounts for actual data movement).

    For collective operations, the actual amount of data transferred differs
    from the message size. For example:
    - all_reduce: 2*(N-1)/N factor (ring algorithm)
    - broadcast: 1 factor (tree algorithm)
    - all_gather: (N-1)/N factor

    Args:
        size_bytes: Message size in bytes
        duration_sec: Duration in seconds
        algo_factor: Algorithm-specific factor
        bw_unit: 'Gbps' or 'GBps'

    Returns:
        Algorithm bandwidth in specified units
    """
    effective_bytes = size_bytes * algo_factor
    return get_bandwidth(effective_bytes, duration_sec, bw_unit)


def max_numel(dtype, mem_factor=0.5):
    """
    Calculate maximum number of elements that fit in GPU memory.

    Args:
        dtype: PyTorch dtype
        mem_factor: Fraction of available memory to use (default 0.5)

    Returns:
        Maximum number of elements
    """
    if not torch.cuda.is_available():
        # Default to reasonable size for CPU-only
        return 2**24  # 16M elements

    dtype_size = torch.tensor([], dtype=dtype).element_size()
    available_mem = torch.cuda.get_device_properties(0).total_memory
    usable_mem = available_mem * mem_factor
    return int(usable_mem / dtype_size)


def benchmark_loop(tensor, comm_fn, warmups=50, trials=100, sync=True):
    """
    Run benchmark loop with warmup and timing.

    Args:
        tensor: Input tensor for communication
        comm_fn: Communication function to benchmark
        warmups: Number of warmup iterations
        trials: Number of timed iterations
        sync: Whether to synchronize before timing

    Returns:
        Average latency in seconds
    """
    # Warmup
    for _ in range(warmups):
        comm_fn(tensor)

    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    if sync:
        dist.barrier()

    # Timed trials
    start_time = time.time()
    for _ in range(trials):
        comm_fn(tensor)

    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    if sync:
        dist.barrier()

    end_time = time.time()
    avg_latency = (end_time - start_time) / trials

    return avg_latency


def print_header(rank, op_name, world_size, dtype, backend):
    """Print benchmark header."""
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"{op_name} Benchmark")
        print(f"{'='*60}")
        print(f"World Size: {world_size}")
        print(f"Backend: {backend}")
        print(f"Dtype: {dtype}")
        print(f"{'='*60}\n")


def print_results(rank, size, latency_ms, bandwidth, bw_unit, raw=False):
    """Print benchmark results."""
    if rank == 0:
        if raw:
            print(f"{size},{latency_ms:.6f}")
        else:
            print(f"Size: {size:>12,} elements | "
                  f"Latency: {latency_ms:>8.3f} ms | "
                  f"Bandwidth: {bandwidth:>8.2f} {bw_unit}")


def get_scan_sizes(maxsize):
    """
    Generate message sizes for scanning.

    Args:
        maxsize: Maximum size as power of 2

    Returns:
        List of message sizes (powers of 2)
    """
    return [2**i for i in range(0, maxsize + 1)]
