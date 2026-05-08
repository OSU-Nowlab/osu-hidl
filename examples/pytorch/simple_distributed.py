#!/usr/bin/env python3
"""
Simple Distributed Training Example with PyTorch + MPI

This example demonstrates basic distributed training using PyTorch's
MPI backend with MVAPICH-Plus.

Usage:
    mpiexec -n 4 python simple_distributed.py

Requirements:
    - PyTorch with MPI support
    - MVAPICH-Plus
    - CUDA-capable GPUs
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def print_rank0(msg):
    """Print only from rank 0."""
    if dist.get_rank() == 0:
        print(msg)


def setup():
    """Initialize distributed training."""
    # Initialize MPI backend
    dist.init_process_group(backend='mpi')

    # Get rank and world size
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Set device for this process
    n_gpus = torch.cuda.device_count()
    device_id = rank % n_gpus
    torch.cuda.set_device(device_id)

    print_rank0(f"Initialized {world_size} processes")
    print_rank0(f"Using {n_gpus} GPUs per node")

    return rank, world_size, device_id


def cleanup():
    """Clean up distributed training."""
    dist.destroy_process_group()


class SimpleModel(nn.Module):
    """Simple neural network for demonstration."""

    def __init__(self, input_size=1000, hidden_size=500, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def train_epoch(model, optimizer, criterion, device, num_batches=100):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for i in range(num_batches):
        # Generate random data
        inputs = torch.randn(32, 1000, device=device)
        labels = torch.randint(0, 10, (32, ), device=device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    """Main training loop."""
    # Setup distributed training
    rank, world_size, device_id = setup()
    device = torch.device(f'cuda:{device_id}')

    print_rank0("=" * 50)
    print_rank0("PyTorch Distributed Training Example")
    print_rank0("=" * 50)
    print_rank0(f"PyTorch version: {torch.__version__}")
    print_rank0(f"CUDA available: {torch.cuda.is_available()}")
    print_rank0(f"CUDA version: {torch.version.cuda}")
    print_rank0(f"MPI available: {dist.is_mpi_available()}")
    print_rank0(f"Gloo available: {dist.is_gloo_available()}")
    print_rank0("")

    # Create model
    print_rank0("Creating model...")
    model = SimpleModel().to(device)

    # Wrap with DDP
    ddp_model = DDP(model, device_ids=[device_id])

    # Create optimizer and loss function
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    num_epochs = 5
    print_rank0(f"Training for {num_epochs} epochs...")
    print_rank0("")

    for epoch in range(num_epochs):
        avg_loss = train_epoch(ddp_model, optimizer, criterion, device)

        # Gather loss from all ranks
        loss_tensor = torch.tensor([avg_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        global_avg_loss = loss_tensor.item()

        print_rank0(
            f"Epoch {epoch+1}/{num_epochs} - Loss: {global_avg_loss:.4f}")

    print_rank0("")
    print_rank0("=" * 50)
    print_rank0("Training complete!")
    print_rank0("=" * 50)

    # Cleanup
    cleanup()


if __name__ == "__main__":
    main()
