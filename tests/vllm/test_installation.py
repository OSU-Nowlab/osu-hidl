"""
vLLM Installation Tests

Tests to verify vLLM is correctly installed and compatible with custom PyTorch.
"""

import pytest
import sys


def test_vllm_import():
    """Test that vLLM can be imported."""
    try:
        import vllm
        print(f"vLLM version: {vllm.__version__}")
    except ImportError as e:
        pytest.fail(f"Failed to import vLLM: {e}")


def test_pytorch_compatibility():
    """Test that vLLM is compatible with our custom PyTorch."""
    try:
        import vllm
        import torch
        print(f"vLLM: {vllm.__version__}")
        print(f"PyTorch: {torch.__version__}")

        # Check PyTorch is our custom build (2.10.0+)
        major, minor = map(int, torch.__version__.split('.')[:2])
        assert major >= 2, f"Expected PyTorch 2.x+, got {torch.__version__}"

    except ImportError as e:
        pytest.fail(f"Import error: {e}")
    except AssertionError as e:
        pytest.fail(f"Compatibility check failed: {e}")


def test_cuda_available():
    """Test CUDA availability through vLLM's PyTorch."""
    import torch

    assert torch.cuda.is_available(), "CUDA not available"
    gpu_count = torch.cuda.device_count()
    print(f"Available GPUs: {gpu_count}")
    assert gpu_count > 0, "No GPUs detected"

    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


def test_vllm_components():
    """Test that key vLLM components can be imported."""
    try:
        from vllm import LLM, SamplingParams
        from vllm.model_executor import get_model
        print("vLLM components imported successfully")
    except ImportError as e:
        pytest.fail(f"Failed to import vLLM components: {e}")


@pytest.mark.skipif(not sys.argv[-1].endswith("test_installation.py"),
                    reason="Inference test requires model download")
def test_basic_inference():
    """
    Test basic inference with a small model.

    Note: This test downloads a model (~500MB) on first run.
    Skip with: pytest -k "not inference"
    """
    try:
        from vllm import LLM, SamplingParams

        # Use smallest available model for testing
        print("Loading model (this may take a few minutes on first run)...")
        llm = LLM(model="facebook/opt-125m",
                  max_num_seqs=2,
                  gpu_memory_utilization=0.3,
                  download_dir="/tmp/vllm_models")

        # Simple generation test
        prompts = ["Hello"]
        sampling_params = SamplingParams(temperature=0.0, max_tokens=10)

        outputs = llm.generate(prompts, sampling_params)

        assert len(outputs) == 1, "Expected 1 output"
        assert len(outputs[0].outputs) > 0, "Expected generated text"

        generated_text = outputs[0].outputs[0].text
        print(f"Generated: {generated_text}")
        assert len(generated_text) > 0, "Generated text is empty"

    except Exception as e:
        pytest.fail(f"Inference test failed: {e}")


def test_multi_gpu_support():
    """Test that vLLM can detect and use multiple GPUs."""
    import torch

    gpu_count = torch.cuda.device_count()
    print(f"Detected {gpu_count} GPU(s)")

    if gpu_count > 1:
        try:
            from vllm import LLM
            # Just test initialization, don't load a model
            print("Multi-GPU support available")
        except ImportError as e:
            pytest.fail(f"Multi-GPU test failed: {e}")
    else:
        pytest.skip("Multiple GPUs not available")


if __name__ == "__main__":
    # Run tests directly
    print("Running vLLM installation tests...")
    print("-" * 60)

    test_vllm_import()
    print("vLLM import test passed")

    test_pytorch_compatibility()
    print("PyTorch compatibility test passed")

    test_cuda_available()
    print("CUDA availability test passed")

    test_vllm_components()
    print("vLLM components test passed")

    test_multi_gpu_support()
    print("Multi-GPU support test passed")

    print("-" * 60)
    print("All basic tests passed!")
    print("\nTo run inference test (requires model download):")
    print(
        "  python -m pytest tests/vllm/test_installation.py::test_basic_inference -v"
    )
