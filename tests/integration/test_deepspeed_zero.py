#!/usr/bin/env python3
"""
Integration test for DeepSpeed ZeRO with MPI backend.

This test verifies that DeepSpeed ZeRO optimization works correctly
with the MVAPICH-based MPI backend.

Usage:
    mpirun -np 2 python test_deepspeed_zero.py

Requirements:
    - DeepSpeed with MPI backend support
    - At least 1 GPU
    - Pydantic v1 (pydantic<2.0) - DeepSpeed is not compatible with Pydantic v2
"""

import sys
import argparse

# Try to import deepspeed, gracefully skip if Pydantic v2 incompatibility
try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
    SKIP_REASON = None
except (ImportError, AttributeError) as e:
    error_msg = str(e)
    if "FieldInfo" in error_msg or "required" in error_msg or "pydantic" in error_msg.lower():
        DEEPSPEED_AVAILABLE = False
        SKIP_REASON = "DeepSpeed incompatible with Pydantic v2. Install pydantic<2.0 to run this test."
    else:
        # Re-raise if it's a different error
        raise

import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """A simple model for testing DeepSpeed."""

    def __init__(self, input_size=1000, hidden_size=500, output_size=100):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def get_ds_config(stage=2):
    """Get DeepSpeed configuration for ZeRO."""
    return {
        "train_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-4
            }
        },
        "fp16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": stage,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e7,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e7,
            "overlap_comm": True
        }
    }


def test_zero_training(stage=2):
    """Test a simple DeepSpeed ZeRO training loop."""
    # Parse DeepSpeed arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--stage", type=int, default=stage)
    parser = deepspeed.add_config_arguments(parser)
    args, _ = parser.parse_known_args()

    # Initialize DeepSpeed
    deepspeed.init_distributed()

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    if rank == 0:
        print(f"Running DeepSpeed ZeRO-{args.stage} test with {world_size} ranks")

    # Create model
    model = SimpleModel()

    # Get config
    ds_config = get_ds_config(stage=args.stage)

    # Initialize DeepSpeed engine
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )

    # Run a few training iterations
    num_iterations = 5
    batch_size = ds_config["train_batch_size"] // world_size

    criterion = nn.MSELoss()

    for i in range(num_iterations):
        # Create random data
        x = torch.randn(batch_size, 1000).to(model_engine.device)
        y = torch.randn(batch_size, 100).to(model_engine.device)

        # Forward pass
        output = model_engine(x)
        loss = criterion(output, y)

        # Backward pass
        model_engine.backward(loss)
        model_engine.step()

        if rank == 0:
            print(f"  Iteration {i + 1}/{num_iterations}, Loss: {loss.item():.4f}")

    # Synchronize
    torch.distributed.barrier()

    if rank == 0:
        print("  ZeRO training completed successfully")

    return True


def main():
    """Run the DeepSpeed ZeRO integration test."""
    print("=" * 60)
    print("DeepSpeed ZeRO Integration Test")
    print("=" * 60)

    # Check if DeepSpeed is available
    if not DEEPSPEED_AVAILABLE:
        print(f"\nSKIPPED: {SKIP_REASON}")
        print("=" * 60)
        return 0  # Exit successfully - this is an expected skip

    try:
        success = test_zero_training()

        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            if rank == 0:
                print("\n" + "=" * 60)
                print("PASSED: DeepSpeed ZeRO integration test")
                print("=" * 60)

        return 0 if success else 1

    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
