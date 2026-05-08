#!/usr/bin/env python3
"""
Simple DeepSpeed ZeRO Training Example

Demonstrates distributed training with DeepSpeed ZeRO optimization
using MVAPICH-Plus MPI backend.

Usage:
    deepspeed --num_gpus=4 simple_zero_training.py --deepspeed_config=ds_config_zero2.json

    Or with MPI directly:
    mpiexec -n 4 python simple_zero_training.py --deepspeed_config=ds_config_zero2.json

Requirements:
    - DeepSpeed installed
    - PyTorch with MPI support
    - MVAPICH-Plus
"""

import argparse
import torch
import torch.nn as nn
import deepspeed
from torch.utils.data import Dataset, DataLoader


class SimpleDataset(Dataset):
    """Synthetic dataset for demonstration."""

    def __init__(self, size=10000, input_dim=1000, num_classes=10):
        self.size = size
        self.input_dim = input_dim
        self.num_classes = num_classes

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.randn(self.input_dim)
        y = torch.randint(0, self.num_classes, (1, )).item()
        return x, y


class SimpleModel(nn.Module):
    """Simple neural network for demonstration."""

    def __init__(self, input_size=1000, hidden_size=2000, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='DeepSpeed ZeRO Training Example')

    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='Local rank passed from distributed launcher')
    parser.add_argument('--epochs',
                        type=int,
                        default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Batch size per GPU')

    # DeepSpeed arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()
    return args


def main():
    """Main training loop."""
    args = get_args()

    # Initialize DeepSpeed
    deepspeed.init_distributed()

    # Only print from rank 0
    def print_rank0(msg):
        if deepspeed.comm.get_rank() == 0:
            print(msg)

    print_rank0("=" * 60)
    print_rank0("DeepSpeed ZeRO Training Example")
    print_rank0("=" * 60)
    print_rank0(f"DeepSpeed version: {deepspeed.__version__}")
    print_rank0(f"PyTorch version: {torch.__version__}")
    print_rank0(f"CUDA available: {torch.cuda.is_available()}")
    print_rank0(f"World size: {deepspeed.comm.get_world_size()}")
    print_rank0("")

    # Create model
    model = SimpleModel()

    # Create dataset and dataloader
    dataset = SimpleDataset()

    # DeepSpeed engine setup
    print_rank0("Initializing DeepSpeed engine...")

    model_engine, optimizer, train_loader, _ = deepspeed.initialize(
        args=args,
        model=model,
        training_data=dataset,
        model_parameters=model.parameters())

    print_rank0(f"DeepSpeed config: {args.deepspeed_config}")
    print_rank0(
        f"Batch size per GPU: {model_engine.train_micro_batch_size_per_gpu()}")
    print_rank0(
        f"Gradient accumulation steps: {model_engine.gradient_accumulation_steps()}"
    )
    print_rank0("")

    # Training loop
    criterion = nn.CrossEntropyLoss()

    print_rank0(f"Training for {args.epochs} epochs...")
    print_rank0("")

    for epoch in range(args.epochs):
        model_engine.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(model_engine.local_rank)
            target = target.to(model_engine.local_rank)

            # Cast to FP16 if enabled in DeepSpeed config
            if model_engine.fp16_enabled():
                data = data.half()

            # Forward pass
            outputs = model_engine(data)
            loss = criterion(outputs, target)

            # Backward pass (DeepSpeed handles gradient sync)
            model_engine.backward(loss)

            # Optimizer step (DeepSpeed handles ZeRO optimization)
            model_engine.step()

            total_loss += loss.item()
            num_batches += 1

            # Limit batches for demo
            if num_batches >= 100:
                break

        avg_loss = total_loss / num_batches
        print_rank0(
            f"Epoch {epoch + 1}/{args.epochs} - Average Loss: {avg_loss:.4f}")

    print_rank0("")
    print_rank0("=" * 60)
    print_rank0("Training complete!")
    print_rank0("=" * 60)

    # Save checkpoint
    if args.local_rank == 0 or args.local_rank == -1:
        print_rank0("Saving checkpoint...")
        model_engine.save_checkpoint('./checkpoints', tag='final')
        print_rank0("Checkpoint saved to ./checkpoints/")


if __name__ == "__main__":
    main()
