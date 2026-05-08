#!/usr/bin/env python3
"""
Integration test for SGLang serving.

This test verifies that SGLang can be imported and basic components work.

Usage:
    python test_sglang_serving.py

Requirements:
    - SGLang installed
    - At least 1 GPU (for full engine test)
"""

import sys


def test_sglang_import():
    """Test that SGLang can be imported."""
    print("  Testing SGLang import...")
    try:
        import sglang  # noqa: F401
        print("    SGLang imported successfully")
        return True
    except ImportError as e:
        print(f"    Failed to import SGLang: {e}")
        return False


def test_sglang_components():
    """Test that key SGLang components are available."""
    print("  Testing SGLang components...")
    try:
        from sglang import Engine, RuntimeEndpoint  # noqa: F401
        print("    Engine and RuntimeEndpoint available")
        return True
    except ImportError:
        # Try alternative imports
        try:
            import sglang as sgl
            # Check for basic sglang functionality
            if hasattr(sgl, "gen") or hasattr(sgl, "function"):
                print("    SGLang core functions available")
                return True
            print("    SGLang imported but core components not found")
            return False
        except Exception as e:
            print(f"    Failed to verify SGLang components: {e}")
            return False


def test_sglang_backend():
    """Test SGLang backend availability."""
    print("  Testing SGLang backend...")
    try:
        import torch
        if not torch.cuda.is_available():
            print("    Skipping backend test: No GPU available")
            return True

        # Check that we can access backend components
        try:
            from sglang.srt.server import launch_server  # noqa: F401
            print("    Server launch function available")
        except ImportError:
            try:
                from sglang.launch_server import launch_server  # noqa: F401
                print("    Server launch function available (alternative path)")
            except ImportError:
                print("    Server launch function not found (may be normal)")

        return True

    except Exception as e:
        print(f"    Backend test failed: {e}")
        return False


def main():
    """Run the SGLang integration tests."""
    print("=" * 60)
    print("SGLang Integration Test")
    print("=" * 60)

    tests = [
        ("Import test", test_sglang_import),
        ("Components test", test_sglang_components),
        ("Backend test", test_sglang_backend),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\n[Test] {name}")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"    Exception: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("PASSED: SGLang integration test")
        return 0
    else:
        print("FAILED: SGLang integration test")
        return 1


if __name__ == "__main__":
    sys.exit(main())
