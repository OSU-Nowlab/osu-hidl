#!/usr/bin/env python3
"""
verl installation checks for the OSU HPC-AI stack.

Set HPC_AI_STRICT_VERL=1 on MRI when verl, Megatron-LM, and vLLM are expected to
be installed together.
"""

import importlib.util
import os

import pytest


STRICT = os.environ.get("HPC_AI_STRICT_VERL", "0") == "1"


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


def test_verl_available():
    """Verify the main verl package can be imported."""
    spec = require_module("verl")
    print(f"verl found at: {spec.origin}")


def test_verl_trainer_entrypoint_available():
    """Verify the PPO trainer entry point exists when verl is installed."""
    spec = require_module("verl.trainer.main_ppo")
    print(f"verl PPO trainer found at: {spec.origin}")


def test_vllm_rollout_dependency_visible():
    """vLLM is the target inference engine for the first MRI recipe."""
    spec = require_module("vllm")
    print(f"vLLM found at: {spec.origin}")


def test_megatron_training_dependency_visible():
    """Megatron Core is required for the Megatron trainer backend."""
    spec = require_module("megatron.core")
    print(f"Megatron Core found at: {spec.origin}")


def test_torch_stack_not_overwritten():
    """Check PyTorch visibility and distributed backend status."""
    torch = pytest.importorskip("torch")
    dist = pytest.importorskip("torch.distributed")

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPI available: {dist.is_mpi_available()}")
    print(f"NCCL available: {dist.is_nccl_available()}")

    if STRICT:
        assert torch.cuda.is_available(), "CUDA should be available on an MRI GPU node"
        assert dist.is_mpi_available(), "Custom PyTorch should keep the MPI backend"
