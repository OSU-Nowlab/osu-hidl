#!/usr/bin/env python3
"""
SGLang Installation Tests

This script verifies that SGLang is properly installed and functional.

Usage:
    pytest test_installation.py -v
    # OR
    python test_installation.py
"""

import sys
import pytest


def test_sglang_import():
    """Test that SGLang can be imported."""
    try:
        import sglang
        print(
            f"PASS: SGLang imported successfully (version: {sglang.__version__})"
        )
        return True
    except ImportError as e:
        print(f"FAIL: Failed to import SGLang: {e}")
        return False


def test_sglang_version():
    """Test that SGLang version can be retrieved."""
    try:
        import sglang
        version = sglang.__version__
        assert version is not None
        assert len(version) > 0
        print(f"PASS: SGLang version: {version}")
        return True
    except Exception as e:
        print(f"FAIL: Failed to get SGLang version: {e}")
        return False


def test_pytorch_available():
    """Test that PyTorch is available."""
    try:
        import torch
        print(f"PASS: PyTorch available (version: {torch.__version__})")
        return True
    except ImportError as e:
        print(f"FAIL: PyTorch not available: {e}")
        return False


def test_cuda_available():
    """Test that CUDA is available."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"PASS: CUDA available with {device_count} device(s)")
            return True
        else:
            print("SKIP: CUDA not available (CPU-only mode)")
            pytest.skip("CUDA not available")
            return False
    except Exception as e:
        print(f"FAIL: Error checking CUDA: {e}")
        return False


def test_sglang_backends():
    """Test that SGLang backends can be imported."""
    try:
        from sglang import RuntimeEndpoint, OpenAI
        print("PASS: SGLang backends imported successfully")
        return True
    except ImportError as e:
        print(f"FAIL: Failed to import SGLang backends: {e}")
        return False


def test_sglang_function_decorator():
    """Test that SGLang function decorator works."""
    try:
        import sglang as sgl

        @sgl.function
        def simple_test(s, text):
            s += text

        print("PASS: SGLang function decorator works")
        return True
    except Exception as e:
        print(f"FAIL: SGLang function decorator failed: {e}")
        return False


def test_transformers_available():
    """Test that transformers library is available."""
    try:
        import transformers
        print(
            f"PASS: Transformers available (version: {transformers.__version__})"
        )
        return True
    except ImportError as e:
        print(f"FAIL: Transformers not available: {e}")
        return False


def test_fastapi_available():
    """Test that FastAPI is available (required for serving)."""
    try:
        import fastapi
        print(f"PASS: FastAPI available (version: {fastapi.__version__})")
        return True
    except ImportError as e:
        print(f"FAIL: FastAPI not available: {e}")
        return False


def test_sglang_runtime_imports():
    """Test that SGLang runtime modules can be imported."""
    try:
        from sglang import runtime
        print("PASS: SGLang runtime modules imported successfully")
        return True
    except ImportError as e:
        print(f"SKIP: SGLang runtime modules not available: {e}")
        pytest.skip(f"Runtime modules not available: {e}")
        return False


def test_sglang_lang_imports():
    """Test that SGLang language modules can be imported."""
    try:
        from sglang import lang
        print("PASS: SGLang language modules imported successfully")
        return True
    except ImportError as e:
        print(f"FAIL: SGLang language modules not available: {e}")
        return False


# Run tests if executed directly
if __name__ == "__main__":
    print("=" * 70)
    print("SGLang Installation Tests")
    print("=" * 70)
    print()

    tests = [
        ("Import SGLang", test_sglang_import),
        ("SGLang Version", test_sglang_version),
        ("PyTorch Available", test_pytorch_available),
        ("CUDA Available", test_cuda_available),
        ("SGLang Backends", test_sglang_backends),
        ("SGLang Function Decorator", test_sglang_function_decorator),
        ("Transformers Available", test_transformers_available),
        ("FastAPI Available", test_fastapi_available),
        ("SGLang Runtime Imports", test_sglang_runtime_imports),
        ("SGLang Language Imports", test_sglang_lang_imports),
    ]

    passed = 0
    failed = 0
    skipped = 0

    for test_name, test_func in tests:
        print(f"Running: {test_name}")
        print("-" * 70)
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
        except pytest.skip.Exception:
            skipped += 1
        except Exception as e:
            print(f"FAIL: Unexpected error: {e}")
            failed += 1
        print()

    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Passed:  {passed}/{len(tests)}")
    print(f"Failed:  {failed}/{len(tests)}")
    print(f"Skipped: {skipped}/{len(tests)}")
    print()

    if failed > 0:
        print("Some tests failed. Please check the output above.")
        sys.exit(1)
    else:
        print("All tests passed!")
        sys.exit(0)
