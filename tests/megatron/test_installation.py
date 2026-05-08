#!/usr/bin/env python3
"""
Megatron-LM installation checks for the OSU HPC-AI stack.

These tests are intentionally lightweight. They verify that the Python package
surface needed by Megatron-LM is discoverable without launching a training job.
Set HPC_AI_STRICT_MEGATRON=1 on MRI when Megatron-LM is expected to be installed.
"""

import importlib.util
import os

import pytest


STRICT = os.environ.get("HPC_AI_STRICT_MEGATRON", "0") == "1"


def require_module(module_name):
    """Return a module spec or skip/fail depending on strict mode."""
    try:
        spec = importlib.util.find_spec(module_name)
    except ModuleNotFoundError:
        spec = None
    if spec is not None:
        return spec

    message = f"{module_name} is not available"
    if STRICT:
        pytest.fail(message)
    pytest.skip(message)


def test_torch_available():
    """Verify PyTorch is importable before checking Megatron-LM."""
    torch = pytest.importorskip("torch")
    major, minor = torch.__version__.split(".")[:2]
    assert int(major) >= 2
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch major.minor: {major}.{minor}")


def test_megatron_core_available():
    """Megatron Core is the package surface used by modern Megatron-LM."""
    spec = require_module("megatron.core")
    print(f"Megatron Core found at: {spec.origin}")


def test_megatron_training_package_available():
    """A source checkout of Megatron-LM exposes training scripts and helpers."""
    spec = importlib.util.find_spec("megatron.training")
    if spec is None:
        pytest.skip("megatron.training is only present for some Megatron-LM installs")
    print(f"Megatron training package found at: {spec.origin}")


def test_distributed_backends_visible():
    """Verify distributed backend visibility without initializing a process group."""
    torch = pytest.importorskip("torch")
    dist = pytest.importorskip("torch.distributed")

    print(f"MPI available: {dist.is_mpi_available()}")
    print(f"NCCL available: {dist.is_nccl_available()}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if STRICT:
        assert dist.is_mpi_available(), "MPI backend is required on the MRI stack"
        assert dist.is_nccl_available(), "NCCL backend should be available on GPU nodes"


def test_cuda_visible_when_required():
    """Require CUDA only when the strict MRI validation mode is enabled."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        if STRICT:
            pytest.fail("CUDA is required for strict Megatron-LM validation")
        pytest.skip("CUDA is not available on this machine")

    print(f"GPU count: {torch.cuda.device_count()}")
    assert torch.cuda.device_count() > 0
