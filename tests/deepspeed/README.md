# DeepSpeed Installation Tests

Test suite to verify DeepSpeed environment is correctly configured with PyTorch and MVAPICH-Plus support.

---

## Known Limitations

### Pydantic v2 Incompatibility

DeepSpeed uses Pydantic v1 APIs internally for configuration validation. If you have Pydantic v2 installed, you will see errors like:

```
AttributeError: 'FieldInfo' object has no attribute 'required'
```

**Workaround:** Install Pydantic v1:
```bash
pip install "pydantic<2.0"
```

**Note:** This may affect other packages (e.g., SGLang) that require Pydantic v2. The core distributed training functionality (MPI/Gloo backends) works regardless of this issue.

---

## Running Tests

### Quick Test

```bash
cd ~/osu-hpc-ai
source setup_runtime_env.sh
python tests/deepspeed/test_installation.py
```

### Full Test Suite with Pytest

```bash
cd ~/osu-hpc-ai
python -m pytest tests/deepspeed/ -v
```

### Prerequisites

1. Activate conda environment: `conda activate hpc-ai-build`
2. Set runtime environment: `source ~/osu-hpc-ai/setup_runtime_env.sh`
3. Allocate GPU node (for GPU tests): `salloc -p devel -t 1:00:00 -C cuda -N 1 --gpus-per-node=1`

---

## Test Coverage

| Test | Purpose | Requirements |
|------|---------|--------------|
| `test_pytorch_available` | PyTorch integration | Custom PyTorch |
| `test_mpi_backend` | MPI backend availability | MVAPICH-Plus |
| `test_gloo_backend` | Gloo backend availability | Gloo library |
| `test_cuda_available` | CUDA availability | GPU node |
| `test_basic_cuda_operation` | Basic CUDA ops | GPU node |
| `test_deepspeed_config_json` | JSON config parsing | None |
| `test_deepspeed_package_exists` | Package installed | DeepSpeed |

---

## Expected Output

```
DeepSpeed Installation Verification
============================================================

Note: DeepSpeed full import tests are skipped due to
Pydantic v2 incompatibility. Core distributed backends
(MPI, Gloo) are tested instead.

Testing: PyTorch Available... PASS
Testing: MPI Backend... PASS
Testing: Gloo Backend... PASS
Testing: DeepSpeed Config JSON... PASS
Testing: DeepSpeed Package Exists... PASS

============================================================
Results: 5 passed, 0 failed, 0 skipped
============================================================
```

---

## Troubleshooting

### ModuleNotFoundError: No module named 'deepspeed'

**Solution:**
```bash
conda activate hpc-ai-build
cd ~/osu-hpc-ai/frameworks/deepspeed
pip install -e . --no-deps
```

### CUDA Out of Memory

**Solution:** Reduce batch size or use smaller model in tests.

### MPI Backend Not Available

**Solution:** Ensure PyTorch was built with MPI support:
```bash
python -c "import torch.distributed as dist; print(dist.is_mpi_available())"
```

### Pydantic v2 Errors

**Solution:** See "Known Limitations" section above.
