# PyTorch Installation Tests

Test suite to verify PyTorch is correctly installed with MVAPICH-Plus and CUDA support.

---

## Running Tests

### Quick Test

```bash
cd ~/osu-hpc-ai
source setup_runtime_env.sh
python tests/pytorch/test_installation.py
```

### Prerequisites

1. Activate conda environment: `conda activate hpc-ai-build`
2. Set runtime environment: `source ~/osu-hpc-ai/setup_runtime_env.sh`
3. Allocate GPU node (for CUDA tests): `salloc -p devel -t 1:00:00 -C cuda -N 1 --gpus-per-node=1`

---

## Test Coverage

| Test | Purpose | Requirements |
|------|---------|--------------|
| `test_import_torch` | Basic import check | PyTorch installed |
| `test_cuda_available` | CUDA availability | GPU node |
| `test_cuda_operations` | CUDA tensor operations | GPU node |
| `test_mpi_available` | MPI backend | MVAPICH-Plus |
| `test_gloo_available` | Gloo backend | Gloo library |
| `test_mixed_precision` | FP16/BF16 support | GPU node |
| `test_basic_nn` | Neural network ops | PyTorch installed |
| `test_autograd` | Automatic differentiation | PyTorch installed |

---

## Expected Output

```
============================================================
PyTorch Installation Verification
============================================================

[Test 1/8] Testing PyTorch import...
  SUCCESS: PyTorch 2.10.0 imported

[Test 2/8] Testing CUDA availability...
  SUCCESS: CUDA 12.6 available
    Devices: 2 (NVIDIA A100-SXM4-40GB)

[Test 3/8] Testing CUDA tensor operations...
  SUCCESS: CUDA tensor operations working

[Test 4/8] Testing MPI availability...
  SUCCESS: MPI backend available

[Test 5/8] Testing Gloo availability...
  SUCCESS: Gloo backend available

[Test 6/8] Testing mixed precision support...
  SUCCESS: FP16 and BF16 operations working

[Test 7/8] Testing basic neural network...
  SUCCESS: Neural network working

[Test 8/8] Testing autograd...
  SUCCESS: Autograd working

============================================================
Test Summary
============================================================
Passed: 8/8

Status: All tests passed
```

---

## Troubleshooting

### ImportError: GLIBCXX_3.4.30 not found

**Solution:**
```bash
source ~/osu-hpc-ai/setup_runtime_env.sh
```

### CUDA not available

**Solution:** Run tests on GPU node:
```bash
salloc -p devel -t 1:00:00 -C cuda -N 1 --gpus-per-node=1
python tests/pytorch/test_installation.py
```

### MPI backend not available

**Solution:** Ensure MVAPICH-Plus is in environment:
```bash
export PATH=~/osu-hpc-ai-dev/install/mvapich-plus-4.1-cuda12.6/bin:$PATH
export LD_LIBRARY_PATH=~/osu-hpc-ai-dev/install/mvapich-plus-4.1-cuda12.6/lib:$LD_LIBRARY_PATH
```
