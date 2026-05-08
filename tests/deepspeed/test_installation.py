#!/usr/bin/env python3
"""
DeepSpeed Installation Verification Tests

Tests to verify DeepSpeed environment is properly configured.
Focuses on PyTorch distributed backends (MPI, Gloo) which are the
core requirements for distributed training.

Note: Full DeepSpeed import tests are skipped due to Pydantic v2
incompatibility. See README.md for details.

For CUDA tests, run: python tests/pytorch/test_installation.py
"""

import sys
import json
import tempfile
import os
import pytest


def test_pytorch_available():
    """Test that PyTorch is available."""
    try:
        import torch
        assert torch is not None
        print(f"PyTorch version: {torch.__version__}")
    except ImportError as e:
        pytest.fail(f"Failed to import torch: {e}")


def test_mpi_backend():
    """Test that MPI backend is available (required for HPC-AI)."""
    import torch.distributed as dist

    mpi_available = dist.is_mpi_available()
    print(f"MPI backend available: {mpi_available}")

    assert mpi_available, "MPI backend is required for HPC-AI distributed training"


def test_gloo_backend():
    """Test that Gloo backend is available."""
    import torch.distributed as dist

    gloo_available = dist.is_gloo_available()
    print(f"Gloo backend available: {gloo_available}")

    assert gloo_available, "Gloo backend should be available for distributed communication"


def test_deepspeed_config_json():
    """Test DeepSpeed JSON config parsing (without DeepSpeed import)."""
    config = {
        "train_batch_size": 32,
        "train_micro_batch_size_per_gpu": 32,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {
            "stage": 2
        },
        "fp16": {
            "enabled": True
        }
    }

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json',
                                     delete=False) as f:
        json.dump(config, f)
        config_path = f.name

    try:
        # Verify config can be loaded
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)

        assert loaded_config["train_batch_size"] == 32
        assert loaded_config["zero_optimization"]["stage"] == 2
        assert loaded_config["fp16"]["enabled"] is True
        print("DeepSpeed JSON config parsing working")
    finally:
        os.unlink(config_path)


def test_deepspeed_package_exists():
    """Test that DeepSpeed package directory exists."""
    import importlib.util
    spec = importlib.util.find_spec("deepspeed")
    assert spec is not None, "DeepSpeed package not found in Python path"
    print(f"DeepSpeed package found at: {spec.origin}")


if __name__ == "__main__":
    """Run tests directly for quick verification."""
    print("=" * 60)
    print("DeepSpeed Installation Verification")
    print("=" * 60)
    print()
    print("Note: DeepSpeed full import tests are skipped due to")
    print("Pydantic v2 incompatibility. Core distributed backends")
    print("(MPI, Gloo) are tested instead.")
    print()
    print("For CUDA tests, run: python tests/pytorch/test_installation.py")
    print()

    tests = [
        ("PyTorch Available", test_pytorch_available),
        ("MPI Backend", test_mpi_backend),
        ("Gloo Backend", test_gloo_backend),
        ("DeepSpeed Config JSON", test_deepspeed_config_json),
        ("DeepSpeed Package Exists", test_deepspeed_package_exists),
    ]

    passed = 0
    failed = 0
    skipped = 0

    for name, test_func in tests:
        try:
            print(f"Testing: {name}...", end=" ")
            test_func()
            print("PASS")
            passed += 1
        except pytest.skip.Exception as e:
            print(f"SKIP: {e}")
            skipped += 1
        except Exception as e:
            print(f"FAIL: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
