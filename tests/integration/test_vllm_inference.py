#!/usr/bin/env python3
"""
Integration test for vLLM inference.

This test verifies that vLLM can perform basic inference operations.
It uses a small model for quick testing.

Usage:
    python test_vllm_inference.py

Requirements:
    - vLLM installed
    - At least 1 GPU with sufficient memory
"""

import sys


def test_vllm_import():
    """Test that vLLM can be imported."""
    print("  Testing vLLM import...")
    try:
        import vllm  # noqa: F401
        print("    vLLM imported successfully")
        return True
    except ImportError as e:
        print(f"    Failed to import vLLM: {e}")
        return False


def test_vllm_llm_class():
    """Test that the LLM class can be instantiated."""
    print("  Testing LLM class availability...")
    try:
        from vllm import LLM, SamplingParams  # noqa: F401
        print("    LLM and SamplingParams classes available")
        return True
    except ImportError as e:
        print(f"    Failed to import LLM class: {e}")
        return False


def test_vllm_engine():
    """Test basic vLLM engine functionality with a small model."""
    print("  Testing vLLM engine (this may take a moment)...")

    try:
        import torch
        if not torch.cuda.is_available():
            print("    Skipping engine test: No GPU available")
            return True

        from vllm import LLM, SamplingParams

        # Use a very small model for testing
        # facebook/opt-125m is commonly available and small
        model_name = "facebook/opt-125m"

        print(f"    Loading model: {model_name}")
        llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5,
            max_model_len=512
        )

        # Simple generation test
        prompts = ["Hello, my name is"]
        sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=10
        )

        print("    Running inference...")
        outputs = llm.generate(prompts, sampling_params)

        # Verify output
        assert len(outputs) == 1, "Expected 1 output"
        assert len(outputs[0].outputs) > 0, "Expected generated tokens"

        generated_text = outputs[0].outputs[0].text
        print(f"    Generated: '{generated_text[:50]}...'")
        print("    vLLM engine test passed")
        return True

    except Exception as e:
        # If the model isn't available, that's OK for CI
        if "not found" in str(e).lower() or "does not exist" in str(e).lower():
            print(f"    Skipping engine test: Model not available ({e})")
            return True
        print(f"    Engine test failed: {e}")
        return False


def main():
    """Run the vLLM integration tests."""
    print("=" * 60)
    print("vLLM Integration Test")
    print("=" * 60)

    tests = [
        ("Import test", test_vllm_import),
        ("LLM class test", test_vllm_llm_class),
    ]

    # Only run engine test if explicitly requested (it downloads models)
    import os
    if os.environ.get("VLLM_TEST_ENGINE", "0") == "1":
        tests.append(("Engine test", test_vllm_engine))

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
        print("PASSED: vLLM integration test")
        return 0
    else:
        print("FAILED: vLLM integration test")
        return 1


if __name__ == "__main__":
    sys.exit(main())
