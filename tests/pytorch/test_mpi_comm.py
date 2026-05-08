#!/usr/bin/env python3
"""
PyTorch MPI Communication Tests

Basic tests for verifying MPI communication in PyTorch distributed training.
These tests verify that the NOWLAB PyTorch fork correctly supports MPI backend.

Usage:
    # Run with 2 processes
    mpirun -np 2 python test_mpi_comm.py

    # Run with 4 processes
    mpirun -np 4 python test_mpi_comm.py
"""

import torch
import torch.distributed as dist
import sys


def test_mpi_initialization():
    """Test that MPI backend can be initialized."""
    try:
        dist.init_process_group(backend='mpi')
        rank = dist.get_rank()
        size = dist.get_world_size()
        print(f"[Rank {rank}/{size}] MPI initialization successful")
        return True
    except Exception as e:
        print(f"MPI initialization failed: {e}")
        return False


def test_all_reduce():
    """Test all_reduce operation."""
    rank = dist.get_rank()
    size = dist.get_world_size()

    # Create tensor with rank as value
    tensor = torch.ones(10) * rank

    # All-reduce sum
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # Expected result: sum of all ranks
    expected = sum(range(size))

    if torch.allclose(tensor, torch.ones(10) * expected):
        print(f"[Rank {rank}] all_reduce test passed (sum = {expected})")
        return True
    else:
        print(f"[Rank {rank}] all_reduce test failed")
        print(f"  Expected: {expected}, Got: {tensor[0].item()}")
        return False


def test_broadcast():
    """Test broadcast operation."""
    rank = dist.get_rank()

    # Root sends value 42
    if rank == 0:
        tensor = torch.ones(10) * 42
    else:
        tensor = torch.zeros(10)

    # Broadcast from rank 0
    dist.broadcast(tensor, src=0)

    if torch.allclose(tensor, torch.ones(10) * 42):
        print(f"[Rank {rank}] broadcast test passed")
        return True
    else:
        print(f"[Rank {rank}] broadcast test failed")
        return False


def test_gather():
    """Test gather operation."""
    rank = dist.get_rank()
    size = dist.get_world_size()

    # Each rank sends its rank number
    send_tensor = torch.ones(5) * rank

    if rank == 0:
        gather_list = [torch.zeros(5) for _ in range(size)]
        dist.gather(send_tensor, gather_list=gather_list, dst=0)

        # Verify gathered data
        success = True
        for i, tensor in enumerate(gather_list):
            if not torch.allclose(tensor, torch.ones(5) * i):
                success = False
                break

        if success:
            print(f"[Rank {rank}] gather test passed")
            return True
        else:
            print(f"[Rank {rank}] gather test failed")
            return False
    else:
        dist.gather(send_tensor, dst=0)
        print(f"[Rank {rank}] gather send complete")
        return True


def test_scatter():
    """Test scatter operation."""
    rank = dist.get_rank()
    size = dist.get_world_size()

    recv_tensor = torch.zeros(5)

    if rank == 0:
        # Root sends different values to each rank
        scatter_list = [torch.ones(5) * i for i in range(size)]
        dist.scatter(recv_tensor, scatter_list=scatter_list, src=0)
    else:
        dist.scatter(recv_tensor, src=0)

    # Each rank should receive tensor with its rank value
    if torch.allclose(recv_tensor, torch.ones(5) * rank):
        print(f"[Rank {rank}] scatter test passed")
        return True
    else:
        print(f"[Rank {rank}] scatter test failed")
        print(f"  Expected: {rank}, Got: {recv_tensor[0].item()}")
        return False


def test_all_gather():
    """Test all_gather operation."""
    rank = dist.get_rank()
    size = dist.get_world_size()

    # Each rank sends its rank number
    send_tensor = torch.ones(5) * rank
    gather_list = [torch.zeros(5) for _ in range(size)]

    dist.all_gather(gather_list, send_tensor)

    # Verify all ranks received correct data
    success = True
    for i, tensor in enumerate(gather_list):
        if not torch.allclose(tensor, torch.ones(5) * i):
            success = False
            break

    if success:
        print(f"[Rank {rank}] all_gather test passed")
        return True
    else:
        print(f"[Rank {rank}] all_gather test failed")
        return False


def test_barrier():
    """Test barrier synchronization."""
    rank = dist.get_rank()

    try:
        dist.barrier()
        print(f"[Rank {rank}] barrier test passed")
        return True
    except Exception as e:
        print(f"[Rank {rank}] barrier test failed: {e}")
        return False


def test_cuda_aware_mpi():
    """Test CUDA-aware MPI communication (if CUDA is available)."""
    if not torch.cuda.is_available():
        print("[INFO] CUDA not available, skipping CUDA-aware MPI test")
        return True

    rank = dist.get_rank()
    size = dist.get_world_size()

    try:
        # Create tensor on GPU
        tensor = torch.ones(10, device='cuda') * rank

        # All-reduce on GPU
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        # Expected result
        expected = sum(range(size))

        if torch.allclose(tensor, torch.ones(10, device='cuda') * expected):
            print(f"[Rank {rank}] CUDA-aware MPI test passed")
            return True
        else:
            print(f"[Rank {rank}] CUDA-aware MPI test failed")
            return False
    except Exception as e:
        print(f"[Rank {rank}] CUDA-aware MPI test error: {e}")
        return False


def run_all_tests():
    """Run all MPI communication tests."""
    print("=" * 60)
    print("PyTorch MPI Communication Tests")
    print("=" * 60)

    # Initialize MPI
    if not test_mpi_initialization():
        print("\nMPI initialization failed. Aborting tests.")
        sys.exit(1)

    rank = dist.get_rank()
    if rank == 0:
        print(f"\nRunning tests with {dist.get_world_size()} processes...")
        print("-" * 60)

    # Run tests
    tests = [
        ("All-Reduce", test_all_reduce),
        ("Broadcast", test_broadcast),
        ("Gather", test_gather),
        ("Scatter", test_scatter),
        ("All-Gather", test_all_gather),
        ("Barrier", test_barrier),
        ("CUDA-aware MPI", test_cuda_aware_mpi),
    ]

    results = []
    for test_name, test_func in tests:
        if rank == 0:
            print(f"\nRunning: {test_name}")
        dist.barrier()  # Synchronize before each test

        result = test_func()
        results.append((test_name, result))

        dist.barrier()  # Synchronize after each test

    # Print summary
    if rank == 0:
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)

        passed = sum(1 for _, result in results if result)
        total = len(results)

        for test_name, result in results:
            status = "PASS" if result else "FAIL"
            print(f"{status}: {test_name}")

        print("-" * 60)
        print(f"Total: {passed}/{total} tests passed")

        if passed == total:
            print("\nAll tests passed!")
            sys.exit(0)
        else:
            print(f"\n{total - passed} test(s) failed")
            sys.exit(1)

    # Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    run_all_tests()
