#!/usr/bin/env python3
"""
Unified test runner for OSU HPC-AI stack.

This script provides a simple interface to run tests for specific frameworks
or all frameworks at once. It handles the different execution requirements
for each framework (e.g., MPI for PyTorch, DeepSpeed launcher for DeepSpeed).

Usage:
    python run_tests.py --pytorch      # Run PyTorch tests
    python run_tests.py --deepspeed    # Run DeepSpeed tests
    python run_tests.py --vllm         # Run vLLM tests
    python run_tests.py --sglang       # Run SGLang tests
    python run_tests.py --megatron     # Run Megatron-LM tests
    python run_tests.py --verl         # Run verl tests
    python run_tests.py --mac-attention  # Run MAC-Attention tests
    python run_tests.py --integration  # Run integration tests
    python run_tests.py --all          # Run all tests
"""

import argparse
import subprocess
import sys
from pathlib import Path


# Get the tests directory
TESTS_DIR = Path(__file__).parent.resolve()
ROOT_DIR = TESTS_DIR.parent


def run_command(cmd, description, cwd=None):
    """Run a command and return success/failure."""
    print(f"\n{'=' * 60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or TESTS_DIR,
            capture_output=False,
        )
        return result.returncode == 0
    except FileNotFoundError as e:
        print(f"Error: Command not found - {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def run_pytorch_tests():
    """Run PyTorch tests."""
    results = []

    # Installation test
    results.append(run_command(
        ["python", "pytorch/test_installation.py"],
        "PyTorch Installation Test"
    ))

    # MPI communication test (requires mpirun)
    print("\nNote: MPI tests require running with mpirun:")
    print("  mpirun -np 2 python tests/pytorch/test_mpi_comm.py")

    return all(results)


def run_deepspeed_tests():
    """Run DeepSpeed tests."""
    results = []

    # Installation test
    results.append(run_command(
        ["python", "deepspeed/test_installation.py"],
        "DeepSpeed Installation Test"
    ))

    return all(results)


def run_vllm_tests():
    """Run vLLM tests."""
    results = []

    # Installation test
    results.append(run_command(
        ["python", "vllm/test_installation.py"],
        "vLLM Installation Test"
    ))

    # Integration test
    results.append(run_command(
        ["python", "integration/test_vllm_inference.py"],
        "vLLM Integration Test"
    ))

    return all(results)


def run_sglang_tests():
    """Run SGLang tests."""
    results = []

    # Installation test
    results.append(run_command(
        ["python", "sglang/test_installation.py"],
        "SGLang Installation Test"
    ))

    # Integration test
    results.append(run_command(
        ["python", "integration/test_sglang_serving.py"],
        "SGLang Integration Test"
    ))

    return all(results)


def run_mac_attention_tests():
    """Run MAC-Attention tests."""
    return run_command(
        ["python", "-m", "pytest", "mac-attention", "-q"],
        "MAC-Attention Correctness Tests"
    )


def run_megatron_tests():
    """Run Megatron-LM tests."""
    return run_command(
        ["python", "-m", "pytest", "megatron", "-q"],
        "Megatron-LM Installation Tests"
    )


def run_verl_tests():
    """Run verl tests."""
    return run_command(
        ["python", "-m", "pytest", "verl", "-q"],
        "verl Installation Tests"
    )


def run_integration_tests():
    """Run integration tests (requires GPU and proper launchers)."""
    print("\n" + "=" * 60)
    print("Integration Tests")
    print("=" * 60)
    print("\nIntegration tests require specific launchers:")
    print("  - PyTorch DDP:    mpirun -np 2 python tests/integration/test_pytorch_ddp.py")  # noqa: E501
    print("  - DeepSpeed ZeRO: deepspeed --num_gpus=2 tests/integration/test_deepspeed_zero.py")  # noqa: E501
    print("  - vLLM:           python tests/integration/test_vllm_inference.py")
    print("  - SGLang:         python tests/integration/test_sglang_serving.py")

    results = []

    # Run the ones that can run without special launchers
    results.append(run_command(
        ["python", "integration/test_vllm_inference.py"],
        "vLLM Integration Test"
    ))

    results.append(run_command(
        ["python", "integration/test_sglang_serving.py"],
        "SGLang Integration Test"
    ))

    return all(results)


def main():
    parser = argparse.ArgumentParser(
        description="OSU HPC-AI Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_tests.py --pytorch      # Run PyTorch tests only
    python run_tests.py --all          # Run all tests
    python run_tests.py --integration  # Run integration tests

Note: Some tests require specific launchers (mpirun, deepspeed).
See the output for instructions on running those tests.
        """
    )

    parser.add_argument("--pytorch", action="store_true",
                        help="Run PyTorch tests")
    parser.add_argument("--deepspeed", action="store_true",
                        help="Run DeepSpeed tests")
    parser.add_argument("--vllm", action="store_true",
                        help="Run vLLM tests")
    parser.add_argument("--sglang", action="store_true",
                        help="Run SGLang tests")
    parser.add_argument("--megatron", action="store_true",
                        help="Run Megatron-LM tests")
    parser.add_argument("--verl", action="store_true",
                        help="Run verl tests")
    parser.add_argument("--mac-attention", dest="mac_attention", action="store_true",
                        help="Run MAC-Attention tests")
    parser.add_argument("--integration", action="store_true",
                        help="Run integration tests")
    parser.add_argument("--all", action="store_true",
                        help="Run all tests")

    args = parser.parse_args()

    # If no args specified, show help
    if not any([args.pytorch, args.deepspeed, args.vllm, args.sglang,
                args.megatron, args.verl, args.mac_attention,
                args.integration, args.all]):
        parser.print_help()
        return 0

    print("=" * 60)
    print("OSU HPC-AI Test Runner")
    print("=" * 60)

    results = {}

    if args.pytorch or args.all:
        results["PyTorch"] = run_pytorch_tests()

    if args.deepspeed or args.all:
        results["DeepSpeed"] = run_deepspeed_tests()

    if args.vllm or args.all:
        results["vLLM"] = run_vllm_tests()

    if args.sglang or args.all:
        results["SGLang"] = run_sglang_tests()

    if args.megatron or args.all:
        results["Megatron-LM"] = run_megatron_tests()

    if args.verl or args.all:
        results["verl"] = run_verl_tests()

    if args.mac_attention or args.all:
        results["MAC-Attention"] = run_mac_attention_tests()

    if args.integration:
        results["Integration"] = run_integration_tests()

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("All tests PASSED")
        return 0
    else:
        print("Some tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
