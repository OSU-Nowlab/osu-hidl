# vLLM Build Guide - MRI Cluster

This guide covers building vLLM from source with custom PyTorch (MVAPICH-Plus enabled) on the MRI cluster.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Manual Build Steps](#manual-build-steps)
- [Troubleshooting](#troubleshooting)
- [Verification](#verification)
- [Known Issues](#known-issues)

## Overview

**What is vLLM?**

vLLM is a high-throughput LLM inference engine featuring:
- PagedAttention for efficient KV cache management
- Continuous batching for maximum throughput
- Multi-GPU tensor parallelism
- OpenAI-compatible API
- Memory-efficient inference (2-24x faster than HuggingFace)

**Why build from source?**

Building from source ensures compatibility with our custom PyTorch 2.10.0 build that includes MVAPICH-Plus integration. Pre-compiled wheels may have ABI incompatibilities with custom PyTorch.

**Key Features of This Build:**
- vLLM (latest from OSU-Nowlab fork)
- Compatible with PyTorch 2.10.0 (custom MVAPICH-Plus build)
- CUDA 12.6 support
- Python 3.11
- Target A100 GPUs (compute capability 8.0)

## Prerequisites

### System Requirements

1. **Custom PyTorch Installation:**
   vLLM requires PyTorch to be installed first. See [PyTorch build guide](pytorch.md).

2. **GPU Node Access:**
   ```bash
   salloc -p devel -t 3:00:00 -C cuda -N 1 --gpus-per-node=2 --mem=128G
   ```

3. **Software Dependencies:**
   - GCC 13.3.0
   - CUDA 12.6
   - Python 3.11 (conda environment)
   - Custom PyTorch 2.10.0 (from HPC-AI stack)

4. **Build Resources:**
   - Time: 30-60 minutes (serial build)
   - Disk: ~10GB for build artifacts
   - Memory: 128GB RAM (recommended for source build)

### Installation Paths

```bash
PYTORCH_SRC=~/osu-hpc-ai/frameworks/pytorch
VLLM_SRC=~/osu-hpc-ai/frameworks/vllm
```

## Quick Start

### Automated Build (Recommended)

```bash
# Request GPU node with sufficient memory
salloc -p devel -t 3:00:00 -C cuda -N 1 --gpus-per-node=2 --mem=128G

# SSH to allocated node
srun --pty bash

# Run build script
cd ~/osu-hpc-ai
bash build_scripts/mri/build_vllm_mri.sh
```

This script automatically:
- Loads required modules (GCC 13.3.0, CUDA 12.6)
- Activates conda environment
- Sets up all environment variables including PyTorch lib paths
- Installs dependencies without overwriting custom PyTorch
- Builds vLLM from source
- Runs verification tests

### Verification

```bash
conda activate hpc-ai-build
source ~/osu-hpc-ai/setup_runtime_env.sh

python -c "import vllm; print('vLLM version:', vllm.__version__)"

# Run full test suite
cd ~/osu-hpc-ai/tests/vllm
python test_installation.py
```

## Manual Build Steps

If you need to build manually or customize the build:

### Step 1: Allocate GPU Node

```bash
salloc -p devel -t 3:00:00 -C cuda -N 1 --gpus-per-node=2 --mem=128G
srun --pty bash
```

### Step 2: Load Modules

```bash
module reset
module load gcc/13.3.0
module load cuda/12.6
```

### Step 3: Activate Conda Environment

```bash
eval "$(conda shell.bash hook)"
conda activate hpc-ai-build
```

### Step 4: Set Environment Variables

```bash
# Paths
export MVAPICH_HOME=~/osu-hpc-ai-dev/install/mvapich-plus-4.1-cuda12.6
export CUDA_HOME=/opt/cuda/12.6
export NCCL_HOME=~/osu-hpc-ai-dev/libext/nccl/build

# Update PATH and library paths
export PATH=$MVAPICH_HOME/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MVAPICH_HOME/lib:$CUDA_HOME/lib64:$NCCL_HOME/lib:$LD_LIBRARY_PATH
export CPATH=$CUDA_HOME/include:$MVAPICH_HOME/include:$NCCL_HOME/include:$CPATH

# CUDA toolchain
export CUDACXX=$CUDA_HOME/bin/nvcc
export CMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc
export CUDAToolkit_ROOT=$CUDA_HOME
export LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$LIBRARY_PATH

# Clean conda toolchain
unset _CONDA_PYTHON_SYSCONFIGDATA_NAME
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v "compiler_compat" | paste -sd:)

# Force correct libstdc++ version (GCC 13.3.0)
export LD_PRELOAD=/opt/gcc/13.3.0/lib64/libstdc++.so.6:$LD_PRELOAD

# Use disk-backed tmp directory (critical for memory stability)
export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR

# Add PyTorch lib directory to LD_LIBRARY_PATH (critical for ABI compatibility)
export TORCH_LIB_DIR=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH=$TORCH_LIB_DIR:$LD_LIBRARY_PATH
```

### Step 5: Clean Previous Build

```bash
cd ~/osu-hpc-ai/frameworks/vllm

rm -rf build/ dist/ vllm.egg-info/ _skbuild/
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.so" -delete 2>/dev/null || true
```

### Step 6: Install Dependencies

**Important:** All dependencies must be installed with `--no-deps` to prevent overwriting custom PyTorch.

```bash
pip install --quiet --no-deps \
    ray psutil sentencepiece numpy transformers \
    fastapi uvicorn pydantic aioprometheus prometheus-client \
    tiktoken lm-format-enforcer outlines typing-extensions \
    filelock pyzmq msgspec pandas mistral-common gguf \
    aiohttp requests pillow scipy packaging huggingface_hub
```

### Step 7: Build vLLM from Source

```bash
# Enable verbose build output
export MAX_JOBS=1
export CMAKE_ARGS="-DENABLE_FP8=OFF"
export TORCH_CUDA_ARCH_LIST="8.0"
export USE_ROCM=0
export SKBUILD_VERBOSE=1
export CMAKE_VERBOSE_MAKEFILE=ON
export NINJA_STATUS="[%f/%t] "

# Build vLLM
pip install -e . --no-deps --no-build-isolation -v 2>&1 | tee ~/vllm_build.log
```

**Build flags explained:**
- `MAX_JOBS=1`: Serial build to avoid OOM
- `CMAKE_ARGS="-DENABLE_FP8=OFF"`: Disable FP8 for this build configuration
- `TORCH_CUDA_ARCH_LIST="8.0"`: Target A100 GPUs only
- `USE_ROCM=0`: Explicitly disable ROCm
- `SKBUILD_VERBOSE=1`: Enable verbose CMake output
- `--no-deps`: Prevent pip from overwriting custom PyTorch
- `--no-build-isolation`: Use current environment (don't create isolated build env)

**Note:** This takes 30-60 minutes. Monitor progress:
```bash
# In another terminal
tail -f ~/vllm_build.log
```

### Step 8: Cleanup Temporary Files

After successful build:
```bash
rm -rf $HOME/tmp/*
```

## Troubleshooting

### Build Errors

#### 1. Out of Memory during CUDA compilation

**Error:**
```
nvcc error: 'cudafe++' died due to signal 9 (Kill signal)
```

**Cause:** CUDA kernel compilation exceeds available memory.

**Solution:** Request more RAM:
```bash
salloc --mem=128G
```

And ensure `TMPDIR` is set to disk-backed directory:
```bash
export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR
```

#### 2. PyTorch ABI Mismatch

**Error:**
```
undefined symbol: _ZN3c104cuda9SetDeviceEab
```

**Cause:** vLLM compiled against different PyTorch version.

**Solution:** Ensure `TORCH_LIB_DIR` is in `LD_LIBRARY_PATH`:
```bash
export TORCH_LIB_DIR=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH=$TORCH_LIB_DIR:$LD_LIBRARY_PATH
```

#### 3. FP8 Header Missing

**Error:**
```
fatal error: c10/util/Float8_e4m3fnuz.h: No such file or directory
```

**Cause:** FP8 support requires PyTorch 2.2+.

**Solution:** Build with `CMAKE_ARGS="-DENABLE_FP8=OFF"` (already in our script).

#### 4. CUDA_HOME Not Set

**Error:**
```
OSError: CUDA_HOME environment variable is not set
```

**Cause:** CMake cannot find CUDA toolkit.

**Solution:**
```bash
export CUDA_HOME=/opt/cuda/12.6
export CUDACXX=/opt/cuda/12.6/bin/nvcc
export CMAKE_CUDA_COMPILER=/opt/cuda/12.6/bin/nvcc
```

#### 5. Build Appears Stuck

**Symptom:** No output for several minutes during CUDA kernel compilation.

**Cause:** Large CUDA files take time to compile.

**Solution:** This is normal. Monitor with verbose output:
```bash
tail -f ~/vllm_build.log
```

Look for `[X/313]` progress counter.

### Runtime Errors

#### 1. Import Error

**Error:**
```
ModuleNotFoundError: No module named 'vllm'
```

**Solution:** Ensure conda environment is activated and vLLM is installed:
```bash
conda activate hpc-ai-build
python -c "import vllm; print(vllm.__version__)"
```

#### 2. CUDA Out of Memory

**Error:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solution:** Reduce model size or batch size:
```python
llm = LLM(
    model="facebook/opt-125m",  # Use smaller model
    max_num_seqs=4,  # Reduce batch size
    gpu_memory_utilization=0.8  # Reduce memory usage
)
```

#### 3. libstdc++ Version Mismatch

**Error:**
```
version 'GLIBCXX_3.4.30' not found
```

**Solution:**
```bash
source ~/osu-hpc-ai/setup_runtime_env.sh
```

## Verification

### Quick Verification

```bash
source ~/osu-hpc-ai/setup_runtime_env.sh

# Check vLLM import
python -c "import vllm; print('vLLM:', vllm.__version__)"

# Check PyTorch compatibility
python -c "import vllm, torch; print(f'PyTorch: {torch.__version__}')"

# Check CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Full Test Suite

```bash
cd ~/osu-hpc-ai/tests/vllm
source ~/osu-hpc-ai/setup_runtime_env.sh
python test_installation.py
```

Expected output:
```
vLLM Installation Tests
============================================================
Import Test... PASS
PyTorch Compatibility... PASS
CUDA Availability... PASS
Basic Inference... PASS

Results: 4 passed, 0 failed
```

### Example Usage

```bash
cd ~/osu-hpc-ai/examples/inference
source ~/osu-hpc-ai/setup_runtime_env.sh

# Basic inference
python vllm_basic.py

# Multi-GPU inference (requires 2 GPUs)
python vllm_multi_gpu.py
```

## Known Issues

### 1. Memory-Intensive Build

**Issue:** Build requires substantial RAM (128GB recommended).

**Cause:** CUDA kernel compilation creates large temporary files.

**Workaround:** Use `MAX_JOBS=1` and `TMPDIR=$HOME/tmp` to reduce memory pressure.

### 2. Long Build Time

**Issue:** Serial build takes 30-60 minutes.

**Cause:** `MAX_JOBS=1` to avoid OOM.

**Workaround:** Start build and monitor progress with `tail -f ~/vllm_build.log`.

### 3. Large Disk Usage

**Issue:** Build creates ~10GB of temporary files.

**Cause:** CUDA intermediate artifacts stored in `$TMPDIR`.

**Solution:** Clean up after successful build:
```bash
rm -rf $HOME/tmp/*
```

### 4. Model Download on First Use

**Issue:** First inference run downloads models (can be slow).

**Cause:** Models downloaded from HuggingFace on demand.

**Workaround:** Pre-download models or use local model paths.

---
