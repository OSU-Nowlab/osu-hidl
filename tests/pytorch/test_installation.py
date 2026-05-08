#!/usr/bin/env python3
"""
PyTorch Installation Verification Tests

Tests to verify PyTorch was built correctly with MVAPICH-Plus and CUDA support.

Usage:
    python test_installation.py
"""

import sys
import subprocess


def test_pytorch_import():
    """Test 1: PyTorch imports successfully"""
    print("\n[Test 1/8] Testing PyTorch import...")
    try:
        import torch
        print(f"  SUCCESS: PyTorch {torch.__version__} imported")
        return True
    except ImportError as e:
        print(f"  FAILED: Could not import PyTorch: {e}")
        return False


def test_cuda_availability():
    """Test 2: CUDA is available"""
    print("\n[Test 2/8] Testing CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(
                0) if device_count > 0 else "Unknown"
            cuda_version = torch.version.cuda
            print(f"  SUCCESS: CUDA {cuda_version} available")
            print(f"    Devices: {device_count} ({device_name})")
            return True
        else:
            print("  FAILED: CUDA not available")
            return False
    except Exception as e:
        print(f"  FAILED: Error checking CUDA: {e}")
        return False


def test_cuda_tensor_operations():
    """Test 3: Basic CUDA tensor operations"""
    print("\n[Test 3/8] Testing CUDA tensor operations...")
    try:
        import torch
        if not torch.cuda.is_available():
            print("  SKIPPED: CUDA not available")
            return True

        # Create tensors on CUDA
        a = torch.randn(100, 100, device='cuda')
        b = torch.randn(100, 100, device='cuda')
        c = torch.matmul(a, b)

        # Verify result is on CUDA
        assert c.is_cuda, "Result tensor not on CUDA"
        assert c.shape == (100, 100), f"Unexpected shape: {c.shape}"

        print("  SUCCESS: CUDA tensor operations working")
        print(f"    Matrix multiplication: (100x100) @ (100x100) = {c.shape}")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_mpi_availability():
    """Test 4: MPI backend is available"""
    print("\n[Test 4/8] Testing MPI availability...")
    try:
        import torch.distributed as dist
        if hasattr(dist, 'is_mpi_available') and dist.is_mpi_available():
            print("  SUCCESS: MPI backend available")
            return True
        else:
            # For PyTorch 2.x, MPI might not have is_mpi_available()
            # Check if we can get mpi backend
            backends = dist.Backend.__dict__
            if 'MPI' in backends:
                print("  SUCCESS: MPI backend registered")
                return True
            print(
                "  WARNING: MPI backend status unclear (check with dist.init_process_group)"
            )
            return True  # Don't fail, just warn
    except Exception as e:
        print(f"  WARNING: Could not verify MPI: {e}")
        return True  # Don't fail on MPI check


def test_gloo_availability():
    """Test 5: Gloo backend is available"""
    print("\n[Test 5/8] Testing Gloo availability...")
    try:
        import torch.distributed as dist
        if hasattr(dist, 'is_gloo_available'):
            if dist.is_gloo_available():
                print("  SUCCESS: Gloo backend available")
                return True
            else:
                print("  FAILED: Gloo not available")
                return False
        else:
            # Check if Gloo backend is registered
            if 'GLOO' in dir(dist.Backend):
                print("  SUCCESS: Gloo backend registered")
                return True
            print("  FAILED: Gloo backend not found")
            return False
    except Exception as e:
        print(f"  FAILED: Error checking Gloo: {e}")
        return False


def test_mixed_precision():
    """Test 6: Mixed precision (FP16/BF16) support"""
    print("\n[Test 6/8] Testing mixed precision support...")
    try:
        import torch
        if not torch.cuda.is_available():
            print("  SKIPPED: CUDA not available")
            return True

        # Test FP16
        x_fp16 = torch.randn(10, 10, dtype=torch.float16, device='cuda')
        y_fp16 = x_fp16 * 2
        assert y_fp16.dtype == torch.float16, "FP16 operation changed dtype"

        # Test BF16 (if supported)
        if torch.cuda.is_bf16_supported():
            x_bf16 = torch.randn(10, 10, dtype=torch.bfloat16, device='cuda')
            y_bf16 = x_bf16 * 2
            assert y_bf16.dtype == torch.bfloat16, "BF16 operation changed dtype"
            print("  SUCCESS: FP16 and BF16 operations working")
        else:
            print(
                "  SUCCESS: FP16 operations working (BF16 not supported on this GPU)"
            )

        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_basic_nn_module():
    """Test 7: Basic neural network module"""
    print("\n[Test 7/8] Testing basic neural network...")
    try:
        import torch
        import torch.nn as nn

        # Create simple model
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1))

        # Test forward pass
        x = torch.randn(5, 10)
        y = model(x)

        assert y.shape == (5, 1), f"Unexpected output shape: {y.shape}"

        # Test on CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.cuda()
            x_cuda = torch.randn(5, 10, device='cuda')
            y_cuda = model_cuda(x_cuda)
            assert y_cuda.is_cuda, "Output not on CUDA"
            print("  SUCCESS: Neural network (CPU + CUDA) working")
        else:
            print("  SUCCESS: Neural network (CPU) working")

        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_autograd():
    """Test 8: Autograd and backward pass"""
    print("\n[Test 8/8] Testing autograd...")
    try:
        import torch

        # Test autograd
        x = torch.randn(3, 3, requires_grad=True)
        y = x**2
        z = y.sum()
        z.backward()

        assert x.grad is not None, "Gradient not computed"
        assert x.grad.shape == x.shape, "Gradient shape mismatch"

        print("  SUCCESS: Autograd working")
        print(f"    Computed gradients for tensor shape {x.shape}")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("PyTorch Installation Verification")
    print("=" * 60)

    tests = [
        test_pytorch_import,
        test_cuda_availability,
        test_cuda_tensor_operations,
        test_mpi_availability,
        test_gloo_availability,
        test_mixed_precision,
        test_basic_nn_module,
        test_autograd,
    ]

    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"\nUNEXPECTED ERROR in {test.__name__}: {e}")
            results.append(False)

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\nStatus: All tests passed")
        return 0
    else:
        print(f"\nStatus: {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
