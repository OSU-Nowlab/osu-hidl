# Troubleshooting Guide

This guide covers common issues encountered when using the OSU HPC-AI stack, organized by category. For framework-specific issues, see the troubleshooting sections in each user guide.

## Table of Contents

- [Quick Diagnostics Checklist](#quick-diagnostics-checklist)
- [MPI and MVAPICH-Plus Issues](#mpi-and-mvapich-plus-issues)
- [GPU and CUDA Issues](#gpu-and-cuda-issues)
- [Framework-Specific Issues](#framework-specific-issues)
  - [PyTorch DDP](#pytorch-ddp)
  - [DeepSpeed](#deepspeed)
  - [vLLM](#vllm)
  - [SGLang](#sglang)
- [Environment and Module Issues](#environment-and-module-issues)
- [Debug Workflows](#debug-workflows)
- [Reporting Issues](#reporting-issues)

---

## Quick Diagnostics Checklist

Run this environment check before digging into specific errors:

```bash
#!/bin/bash
# Save as check_env.sh and run: bash check_env.sh

echo "=== Python ==="
python --version

echo "=== CUDA ==="
nvcc --version 2>/dev/null || echo "nvcc not found — is cuda module loaded?"
nvidia-smi 2>/dev/null || echo "nvidia-smi not found — are you on a GPU node?"

echo "=== MPI ==="
which mpirun && mpirun --version || echo "mpirun not found"
mpicc --version 2>/dev/null | head -1 || echo "mpicc not found"

echo "=== PyTorch ==="
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('GPU count:', torch.cuda.device_count())"
python -c "import torch.distributed as dist; print('MPI backend:', dist.is_mpi_available())"

echo "=== Framework Imports ==="
python -c "import deepspeed; print('DeepSpeed:', deepspeed.__version__)" 2>/dev/null || echo "DeepSpeed: not found"
python -c "import vllm; print('vLLM:', vllm.__version__)" 2>/dev/null || echo "vLLM: not found"
python -c "import sglang; print('SGLang:', sglang.__version__)" 2>/dev/null || echo "SGLang: not found"
python -c "import mcr_dl; print('MCR-DL: installed')" 2>/dev/null || echo "MCR-DL: not found"

echo "=== Key Paths ==="
echo "MVAPICH_HOME: ${MVAPICH_HOME:-not set}"
echo "CUDA_HOME: ${CUDA_HOME:-not set}"
echo "LD_PRELOAD: ${LD_PRELOAD:-not set}"
```

---

## MPI and MVAPICH-Plus Issues

### MPI backend not available in PyTorch

**Error**: `RuntimeError: MPI backend not available` or `dist.is_mpi_available()` returns `False`

**Cause**: PyTorch was not built with `USE_MPI=1` and `USE_CUDA_MPI=1`.

**Solution**: Rebuild PyTorch from the HPC-AI fork with MPI enabled:

```bash
USE_MPI=1 USE_CUDA_MPI=1 python setup.py install
```

See [docs/build-guides/pytorch.md](build-guides/pytorch.md) for the full build process.

### `mpirun: command not found`

**Cause**: MVAPICH-Plus is not in `PATH`.

**Solution**:

```bash
export MVAPICH_HOME="${HOME}/osu-hpc-ai-dev/install/mvapich-plus-4.1-cuda12.6"
export PATH="${MVAPICH_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MVAPICH_HOME}/lib:${LD_LIBRARY_PATH}"
```

On MRI, load modules first: `module load gcc/13.3.0 cuda/12.6`

### `init_process_group` hangs

**Cause**: Ranks cannot reach each other (firewall, wrong hostname, mismatched world size).

**Diagnosis**:

```bash
# Test connectivity between nodes
mpirun -np 2 -host node1,node2 hostname

# Run a minimal MPI communication test
mpirun -np 2 python -c "
import torch, torch.distributed as dist
dist.init_process_group(backend='mpi')
print(f'Rank {dist.get_rank()} initialized')
dist.destroy_process_group()
"
```

**Solutions**:
1. Verify all nodes can ping each other: `mpirun -np 2 -host node1,node2 ping -c 1 node2`
2. Check your hostfile matches the actual SLURM allocation: `scontrol show hostnames $SLURM_JOB_NODELIST`
3. Ensure the same conda environment is activated on all nodes

### Collective hangs mid-training

**Cause**: One rank diverged from the collective call sequence (e.g., exception on one rank, conditional barrier).

**Diagnosis**: Set a short timeout and enable verbose logging:

```bash
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO    # If using NCCL
```

```python
dist.init_process_group(backend='mpi', timeout=datetime.timedelta(seconds=30))
```

The rank that diverges will raise a timeout error with a stack trace.

**Solution**: Ensure all ranks call the same collectives in the same order. Wrap rank-0-only operations so they don't include collective calls:

```python
# Wrong
if rank == 0:
    dist.broadcast(tensor, src=0)   # Only rank 0 calls this → deadlock

# Correct
dist.broadcast(tensor, src=0)       # All ranks call this
if rank == 0:
    save_checkpoint(tensor)         # Only rank 0 saves
```

### GPUDirect RDMA not working

**Symptom**: Poor bandwidth despite InfiniBand setup; bandwidth comparable to CPU-staged transfers (~10 Gbps vs expected ~200 Gbps).

**Diagnosis**:

```bash
# Check GPUDirect availability
python -c "import torch; print(torch.cuda.is_available())"
mpirun -np 2 $MVAPICH_HOME/libexec/mvapich-plus/osu_bw  # C-level baseline
mpirun -np 2 python benchmarks/communication/all_reduce.py  # Python-level
```

**Solution**:

```bash
export MV2_USE_CUDA=1
export MV2_USE_GPUDIRECT_RDMA=1
export MV2_CUDA_USE_NAIVE=0
```

---

## GPU and CUDA Issues

### CUDA not available (`torch.cuda.is_available()` returns `False`)

**Cause 1**: Not on a GPU node.

```bash
# Allocate a GPU node first
salloc -p devel -t 1:00:00 -C cuda -N 1 --gpus-per-node=1
```

**Cause 2**: CUDA module not loaded.

```bash
module load cuda/12.6
export CUDA_HOME=/opt/cuda/12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

**Cause 3**: Wrong Python environment (system Python instead of conda).

```bash
which python   # Should show conda env path
conda activate hpc-ai-build
```

### `CUDA_HOME not found` during build

```bash
export CUDA_HOME=/opt/cuda/12.6
export CUDACXX=$CUDA_HOME/bin/nvcc
export CMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc
export CUDAToolkit_ROOT=$CUDA_HOME
```

### Expected tensors on same device

**Error**: `RuntimeError: Expected all tensors to be on the same device`

**Cause**: Model parameters and input tensors are on different GPUs.

**Solution**: Explicitly assign each rank to its GPU based on local rank:

```python
local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', 0))
device = torch.device(f'cuda:{local_rank}')
torch.cuda.set_device(device)
model = model.to(device)
batch = batch.to(device)
```

### CUDA out of memory

**Error**: `torch.cuda.OutOfMemoryError: CUDA out of memory`

**Solutions** (try in order):
1. Reduce batch size per GPU
2. Enable gradient accumulation (same effective batch, less memory per step)
3. Use `torch.cuda.empty_cache()` and `gc.collect()` between iterations
4. Enable mixed precision (`torch.cuda.amp.autocast()`)
5. Use DeepSpeed ZeRO-2 or ZeRO-3 to partition optimizer states
6. Enable activation checkpointing

### `libstdc++.so.6: version 'GLIBCXX_3.4.30' not found`

**Cause**: Conda's bundled libstdc++ is older than what the compiled extensions require.

**Solution**:

```bash
export LD_PRELOAD=/opt/gcc/13.3.0/lib64/libstdc++.so.6:$LD_PRELOAD
```

Add this to your environment setup script so it applies to every session.

### cuDNN not found

```bash
# Install via pip
pip install nvidia-cudnn-cu12

# Or set paths manually
export CUDNN_HOME=/path/to/cudnn
export LD_LIBRARY_PATH=$CUDNN_HOME/lib:$LD_LIBRARY_PATH
```

---

## Framework-Specific Issues

### PyTorch DDP

#### DDP training slower than expected

1. Verify MPI backend is active: `python -c "import torch.distributed as dist; dist.init_process_group('mpi'); print(dist.get_backend())"`
2. Enable GPUDirect RDMA (see [MPI Issues](#gpudirect-rdma-not-working))
3. Check gradient bucket size: increase `bucket_cap_mb` in `DistributedDataParallel()` for large models
4. Profile with `torch.profiler` to identify the bottleneck

#### Unused parameters error

**Error**: `Expected to have finished reduction in the prior iteration before starting a new one`

**Cause**: Some model parameters are not used in every forward pass (e.g., conditional modules).

**Solution**:

```python
model = DDP(model, find_unused_parameters=True)
```

For more context, see [docs/user-guides/pytorch-ddp.md](user-guides/pytorch-ddp.md).

---

### DeepSpeed

#### Batch size assertion

**Error**: `AssertionError: train_batch_size is not equal to micro_batch_per_gpu * gradient_acc_step * world_size`

```python
# Correct formula:
train_batch_size = train_micro_batch_size_per_gpu * gradient_accumulation_steps * num_gpus
```

#### FP16 dtype mismatch

**Error**: `RuntimeError: mat1 and mat2 must have the same dtype`

```python
if model_engine.fp16_enabled():
    batch = {k: v.half() if v.is_floating_point() else v for k, v in batch.items()}
```

#### `ds_report` shows missing custom ops

Custom CUDA kernels (FusedAdam, transformer) were not compiled. Rebuild DeepSpeed:

```bash
DS_BUILD_FUSED_ADAM=1 DS_BUILD_TRANSFORMER=1 pip install -e . --no-build-isolation
```

For more context, see [docs/user-guides/deepspeed-zero.md](user-guides/deepspeed-zero.md).

---

### vLLM

#### PyTorch ABI mismatch

**Error**: `undefined symbol: _ZN3c104cuda9SetDeviceEab`

**Cause**: vLLM was compiled against a different PyTorch version than what is installed.

**Solution**: Rebuild vLLM from source following [docs/build-guides/vllm.md](build-guides/vllm.md). The custom PyTorch 2.10.0 build must be installed before building vLLM.

#### `tensor_parallel_size` validation error

**Error**: `num_attention_heads must be divisible by tensor_parallel_size`

Reduce `tensor_parallel_size` to a divisor of the model's attention head count. For Llama-2 (32 heads): valid values are 1, 2, 4, 8.

#### Build killed during CUDA compilation

**Cause**: Insufficient RAM for parallel CUDA kernel compilation.

**Solution**:

```bash
export MAX_JOBS=1
export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR
salloc --mem=128G   # Request sufficient memory
```

For more context, see [docs/user-guides/vllm-serving.md](user-guides/vllm-serving.md).

---

### SGLang

#### Server fails to start

```bash
# Check the port is not already in use
lsof -i :30000

# Verify model can be loaded
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')"
```

#### `setuptools_scm` missing during SGLang build

```bash
pip install setuptools_scm
```

#### PyTorch overwritten during SGLang install

**Cause**: `pip install -e .` without `--no-deps` installs stock PyTorch.

**Solution**: Always build SGLang with `--no-deps` and install its dependencies manually. See [docs/build-guides/sglang.md](build-guides/sglang.md).

For more context, see [docs/user-guides/sglang-structured.md](user-guides/sglang-structured.md).

---

## Environment and Module Issues

### Conflicting Python environments

**Symptom**: Imports succeed from command line but fail inside SLURM job.

**Cause**: SLURM job inherits a different environment.

**Solution**: Explicitly activate conda environment inside the SLURM script:

```bash
#!/bin/bash
#SBATCH ...

module reset
module load gcc/13.3.0 cuda/12.6

# Explicitly activate conda — don't rely on inherited environment
source ${HOME}/miniconda3/bin/activate
conda activate hpc-ai-build
```

### Module load order matters

Load GCC before CUDA to avoid linker conflicts:

```bash
module reset            # Clear all modules first
module load gcc/13.3.0  # GCC must come first
module load cuda/12.6   # Then CUDA
```

### `_CONDA_PYTHON_SYSCONFIGDATA_NAME` causes compiler issues

This variable set by conda can confuse build systems:

```bash
unset _CONDA_PYTHON_SYSCONFIGDATA_NAME
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v "compiler_compat" | paste -sd:)
```

### Conda environment path issues

If `which python` shows the wrong Python:

```bash
eval "$(conda shell.bash hook)"
conda activate hpc-ai-build
which python   # Should now show the conda env path
```

---

## Debug Workflows

### Verify MPI communication step by step

```bash
# Step 1: Basic MPI launch
mpirun -np 2 python -c "
import os
print(f'Rank {os.environ.get(\"OMPI_COMM_WORLD_RANK\")}: hello from {os.uname()[1]}')
"

# Step 2: PyTorch distributed init
mpirun -np 2 python -c "
import torch.distributed as dist
dist.init_process_group(backend='mpi')
print(f'Rank {dist.get_rank()}/{dist.get_world_size()}: init OK')
dist.destroy_process_group()
"

# Step 3: GPU tensor communication
mpirun -np 2 python -c "
import torch, torch.distributed as dist
dist.init_process_group(backend='mpi')
rank = dist.get_rank()
t = torch.tensor([float(rank)], device=f'cuda:{rank}')
dist.all_reduce(t)
print(f'Rank {rank}: all_reduce = {t.item()}')   # Should be 1.0 (0+1) on both ranks
dist.destroy_process_group()
"
```

### Check GPU visibility per rank

```bash
mpirun -np 4 python -c "
import os, torch
local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', 0))
print(f'Local rank {local_rank}: GPU = {torch.cuda.get_device_name(local_rank)}')
"
```

### Minimal repro for OOM

```python
import torch

# Find the maximum batch size that fits
for batch_size in [512, 256, 128, 64, 32, 16]:
    try:
        x = torch.randn(batch_size, 1024, device='cuda')
        y = x @ x.T
        torch.cuda.synchronize()
        print(f"Batch size {batch_size}: OK")
        break
    except torch.cuda.OutOfMemoryError:
        print(f"Batch size {batch_size}: OOM")
        torch.cuda.empty_cache()
```

### Identify which node/rank is failing

Set a per-rank log file:

```bash
mpirun -np 4 python -c "
import os, sys
rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', 0))
sys.stdout = open(f'rank_{rank}.log', 'w')
sys.stderr = sys.stdout
# ... your code
"
```

Then inspect: `cat rank_*.log`

---

## Reporting Issues

When opening a GitHub issue, include:

```bash
# Collect system info
uname -a
nvcc --version
gcc --version
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
mpirun --version
echo $MVAPICH_HOME
```

- **GitHub Issues**: [https://github.com/OSU-Nowlab/osu-hpc-ai/issues](https://github.com/OSU-Nowlab/osu-hpc-ai/issues)
- **Official Support**: [https://hpc-ai.engineering.osu.edu](https://hpc-ai.engineering.osu.edu)

Attach the full error traceback and, if building from source, the build log.
