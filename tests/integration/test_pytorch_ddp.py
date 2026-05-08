#!/usr/bin/env python3
"""
Integration test for PyTorch DDP with MPI backend.

This test verifies that PyTorch Distributed Data Parallel works correctly
with the MVAPICH-based MPI backend.

Usage:
    mpirun -np 2 python test_pytorch_ddp.py

Requirements:
    - PyTorch with MPI backend support
    - At least 2 MPI ranks
    - GPU (optional, will use CPU if not available)
"""

import sys
import torch
import torch.nn as nn
import torch.distributed as dist


class SimpleModel(nn.Module):
    """A simple model for testing DDP."""

    def __init__(self, input_size=100, hidden_size=50, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def test_ddp_training():
    """Test a simple DDP training loop."""
    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group(backend="mpi")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Determine device
    if torch.cuda.is_available():
        local_rank = rank % torch.cuda.device_count()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if rank == 0:
        print(f"Running DDP test with {world_size} ranks on {device.type}")

    # Create model and wrap with DDP
    model = SimpleModel().to(device)
    ddp_model = nn.parallel.DistributedDataParallel(model)

    # Create optimizer and loss function
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Run a few training iterations
    num_iterations = 5
    batch_size = 32

    for i in range(num_iterations):
        # Create random data (different per rank)
        x = torch.randn(batch_size, 100).to(device)
        y = torch.randn(batch_size, 10).to(device)

        # Forward pass
        optimizer.zero_grad()
        output = ddp_model(x)
        loss = criterion(output, y)

        # Backward pass
        loss.backward()
        optimizer.step()

        if rank == 0:
            print(f"  Iteration {i + 1}/{num_iterations}, Loss: {loss.item():.4f}")

    # Verify gradients are synchronized across ranks
    dist.barrier()

    # Check that all ranks have the same model parameters
    for name, param in ddp_model.named_parameters():
        param_tensor = param.data.clone()
        dist.all_reduce(param_tensor, op=dist.ReduceOp.SUM)
        param_tensor /= world_size

        # Parameters should be identical across ranks after averaging
        diff = (param.data - param_tensor).abs().max().item()
        assert diff < 1e-5, f"Parameter {name} differs across ranks: {diff}"

    if rank == 0:
        print("  Parameter sync verified across all ranks")

    dist.barrier()
    return True


def main():
    """Run the DDP integration test."""
    print("=" * 60)
    print("PyTorch DDP Integration Test")
    print("=" * 60)

    try:
        success = test_ddp_training()

        if dist.is_initialized():
            rank = dist.get_rank()
            if rank == 0:
                print("\n" + "=" * 60)
                print("PASSED: PyTorch DDP integration test")
                print("=" * 60)
            dist.destroy_process_group()

        return 0 if success else 1

    except Exception as e:
        print(f"\nFAILED: {e}")
        if dist.is_initialized():
            dist.destroy_process_group()
        return 1


if __name__ == "__main__":
    sys.exit(main())
