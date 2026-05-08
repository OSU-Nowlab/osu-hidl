# SGLang Installation Tests

Test suite to verify SGLang is correctly installed with custom PyTorch support.

---

## Running Tests

### Quick Test

```bash
cd ~/osu-hpc-ai
source setup_runtime_env.sh
python tests/sglang/test_installation.py
```

### Full Test Suite with Pytest

```bash
cd ~/osu-hpc-ai
python -m pytest tests/sglang/ -v
```

### Prerequisites

1. Activate conda environment: `conda activate hpc-ai-build`
2. Set runtime environment: `source ~/osu-hpc-ai/setup_runtime_env.sh`
3. Allocate GPU node (for CUDA tests): `salloc -p devel -t 1:00:00 -C cuda -N 1 --gpus-per-node=1`

---

## Test Coverage

| Test | Purpose | Requirements |
|------|---------|--------------|
| `test_sglang_import` | Basic import check | SGLang installed |
| `test_sglang_version` | Version verification | SGLang installed |
| `test_pytorch_available` | PyTorch integration | Custom PyTorch |
| `test_cuda_available` | CUDA availability | GPU node |
| `test_sglang_backends` | Backend imports | SGLang installed |
| `test_sglang_function_decorator` | Function decorator | SGLang installed |
| `test_transformers_available` | Transformers library | transformers installed |
| `test_fastapi_available` | FastAPI for serving | fastapi installed |
| `test_sglang_runtime_imports` | Runtime modules | SGLang installed |
| `test_sglang_lang_imports` | Language modules | SGLang installed |

---

## Expected Output

```
SGLang Installation Tests
============================================================

Running: Import SGLang
PASS: SGLang imported successfully (version: 0.4.9)

Running: SGLang Version
PASS: SGLang version: 0.4.9

Running: PyTorch Available
PASS: PyTorch available (version: 2.10.0)

Running: CUDA Available
PASS: CUDA available with 2 device(s)

Running: SGLang Backends
PASS: SGLang backends imported successfully

Running: SGLang Function Decorator
PASS: SGLang function decorator works

Running: Transformers Available
PASS: Transformers available

Running: FastAPI Available
PASS: FastAPI available

Running: SGLang Runtime Imports
PASS: SGLang runtime modules imported successfully

Running: SGLang Language Imports
PASS: SGLang language modules imported successfully

Test Summary
============================================================
Passed:  10/10
Failed:  0/10
Skipped: 0/10

All tests passed!
```

---

## Troubleshooting

### ImportError: No module named 'sglang'

**Solution:**
```bash
conda activate hpc-ai-build
cd ~/osu-hpc-ai/frameworks/sglang/python
pip install -e . --no-deps
```

### CUDA Tests Failing

**Solution:** Run tests on GPU node:
```bash
salloc -p devel -t 1:00:00 -C cuda -N 1 --gpus-per-node=1
python tests/sglang/test_installation.py
```

### Missing Dependencies

**Solution:** Install SGLang dependencies:
```bash
pip install --no-deps fastapi uvicorn transformers huggingface_hub
```
