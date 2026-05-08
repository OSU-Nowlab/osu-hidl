# vLLM Installation Tests

Test suite to verify vLLM is correctly installed and compatible with our custom PyTorch build.

---

## Running Tests

### Quick Test (No Model Download)

```bash
cd ~/osu-hpc-ai
python tests/vllm/test_installation.py
```

### Full Test Suite with Pytest

```bash
cd ~/osu-hpc-ai
python -m pytest tests/vllm/ -v
```

### Run Specific Tests

```bash
# Skip inference test (no model download)
python -m pytest tests/vllm/ -v -k "not inference"

# Only run inference test
python -m pytest tests/vllm/test_installation.py::test_basic_inference -v
```

---

## Test Coverage

| Test | Purpose | Requirements |
|------|---------|--------------|
| `test_vllm_import` | Basic import check | vLLM installed |
| `test_pytorch_compatibility` | Verify PyTorch version | Custom PyTorch 2.x+ |
| `test_cuda_available` | CUDA and GPU detection | CUDA 12.6, GPU node |
| `test_vllm_components` | Component imports | vLLM installed |
| `test_basic_inference` | End-to-end inference | Model download (~500MB) |
| `test_multi_gpu_support` | Multi-GPU detection | 2+ GPUs |

---

## Expected Output

```
Running vLLM installation tests...
------------------------------------------------------------
vLLM version: 0.x.x
vLLM import test passed
vLLM: 0.x.x
PyTorch: 2.10.0
PyTorch compatibility test passed
Available GPUs: 2
GPU 0: NVIDIA A100-SXM4-40GB
GPU 1: NVIDIA A100-SXM4-40GB
CUDA availability test passed
vLLM components imported successfully
vLLM components test passed
Detected 2 GPU(s)
Multi-GPU support available
Multi-GPU support test passed
------------------------------------------------------------
All basic tests passed!
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'vllm'"

**Solution:**
```bash
conda activate hpc-ai-build
pip install vllm
```

### "CUDA not available"

**Solution:** Run tests on GPU node:
```bash
salloc -p devel -t 1:00:00 -C cuda -N 1 --gpus-per-node=2
conda activate hpc-ai-build
python tests/vllm/test_installation.py
```

### Inference Test Fails

**Solution:** The inference test downloads a model on first run. Ensure:
- Internet connectivity
- Sufficient disk space (~1GB)
- GPU memory available

Skip with: `pytest -k "not inference"`
