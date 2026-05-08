# PyTorch Build Guide - MRI Cluster

This guide covers building PyTorch from source with MVAPICH-Plus (CUDA-aware MPI) and NCCL support on the MRI cluster.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Manual Build Steps](#manual-build-steps)
- [Troubleshooting](#troubleshooting)
- [Verification](#verification)
- [Known Issues](#known-issues)

## Overview

**What is PyTorch?**

PyTorch is a deep learning framework that provides:
- Flexible tensor computation with GPU acceleration
- Automatic differentiation for building neural networks
- Distributed training with multiple GPU/node support
- Extensive ecosystem of tools and libraries

**Why build from source with MVAPICH-Plus?**

Building PyTorch with MVAPICH-Plus integration enables:
- CUDA-aware MPI: Direct GPU-to-GPU communication via GPUDirect RDMA
- Better scalability: Reduced latency for multi-node training
- NCCL integration: Optimized collective operations for NVIDIA GPUs
- Custom optimizations: Target specific hardware (A100 GPUs)

**Key Features of This Build:**
- PyTorch 2.10.0 from the OSU-Nowlab fork
- CUDA 12.6 support with compute capability 6.0-9.0
- MVAPICH-Plus 4.1 integration
- NCCL 2.26.4 for multi-GPU communication
- GCC 13.3.0 toolchain

**Note:** This guide now targets the HPC-AI PyTorch 2.10.0 integration. Check your local `frameworks/pytorch` checkout for the exact branch or commit you are building.

## Prerequisites

### System Requirements

1. **GPU Node Access:**
   ```bash
   # Option A: Interactive (devel partition, 6-hour max)
   salloc -p devel -t 6:00:00 -C cuda -N 1 --gpus-per-node=1 --mem=64G

   # Option B: Batch job (compute partition, longer builds)
   sbatch build_scripts/mri/submit_pytorch_build.slurm
   ```

2. **Software Dependencies:**
   - GCC 13.3.0 (module: `gcc/13.3.0`)
   - CUDA 12.6 (module: `cuda/12.6`)
   - MVAPICH-Plus 4.1 with CUDA support
   - NCCL 2.26.4
   - Python 3.11 (conda environment)

3. **Build Resources:**
   - Time: 4-6 hours
   - Disk: ~25GB for build artifacts
   - Memory: 64GB+ RAM (MAX_JOBS=4)

### Installation Paths

```bash
MVAPICH_HOME=~/osu-hpc-ai-dev/install/mvapich-plus-4.1-cuda12.6
NCCL_HOME=~/osu-hpc-ai-dev/libext/nccl/build
PYTORCH_SRC=~/osu-hpc-ai/frameworks/pytorch   # branch: hpc_ai_v1.0
```

## Quick Start

### Automated Build (Recommended)

```bash
# Option A: Interactive build (devel partition)
salloc -p devel -t 6:00:00 -C cuda -N 1 --gpus-per-node=1 --mem=64G
cd ~/osu-hpc-ai
bash build_scripts/mri/build_pytorch_mri.sh

# Option B: Batch job (compute partition, recommended for long builds)
cd ~/osu-hpc-ai
sbatch build_scripts/mri/submit_pytorch_build.slurm

# For a clean rebuild (wipes previous build artifacts):
bash build_scripts/mri/build_pytorch_mri.sh --clean
```

This script automatically:
- Loads required modules (GCC 13.3.0, CUDA 12.6)
- Activates conda environment
- Sets up all environment variables
- Builds and installs PyTorch with MVAPICH-Plus
- Runs verification tests

### Verification

After build completes:

```bash
# Set runtime environment
source ~/osu-hpc-ai/setup_runtime_env.sh

# Test PyTorch
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch.distributed as dist; print('MPI available:', dist.is_mpi_available())"

# Run full test suite
cd ~/osu-hpc-ai/tests/pytorch
python test_installation.py
```

## Manual Build Steps

If you need to build manually or customize the build:

### Step 1: Allocate GPU Node

```bash
# Interactive session (devel partition, 6-hour max)
salloc -p devel -t 6:00:00 -C cuda -N 1 --gpus-per-node=1 --mem=64G

# Note: The compute partition only allows batch jobs, not interactive sessions.
# For long builds, use: sbatch build_scripts/mri/submit_pytorch_build.slurm
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

# Add CUDA static library path
export LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$LIBRARY_PATH

# Clean conda toolchain issues
unset _CONDA_PYTHON_SYSCONFIGDATA_NAME
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v "compiler_compat" | paste -sd:)

# Force correct libstdc++ version (GCC 13.3.0)
export LD_PRELOAD=/opt/gcc/13.3.0/lib64/libstdc++.so.6:$LD_PRELOAD
```

### Step 5: Install Python Dependencies

```bash
pip install pyyaml typing-extensions numpy ninja cmake
```

### Step 6: Prepare PyTorch Source

```bash
# Clone the OSU-Nowlab PyTorch fork (if not already present)
git clone -b hpc_ai_v1.0 https://github.com/OSU-Nowlab/pytorch.git \
  ~/osu-hpc-ai/frameworks/pytorch

cd ~/osu-hpc-ai/frameworks/pytorch
git submodule sync
git submodule update --init --recursive

# For incremental builds (default), no cleaning needed
# The build system will reuse existing artifacts

# For a clean rebuild only (if you need to start fresh):
git clean -fdx
make clean 2>/dev/null || true
python setup.py clean 2>/dev/null || true

# Or use the build script with --clean flag:
# bash build_scripts/mri/build_pytorch_mri.sh --clean
```

### Step 7: Configure Build

```bash
export _GLIBCXX_USE_CXX11_ABI=1
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"

MAX_JOBS=4 \
USE_XNNPACK=0 \
USE_CUDA=1 \
USE_MPI=1 \
USE_CUDA_MPI=1 \
USE_DISTRIBUTED=1 \
BUILD_TEST=0 \
TORCH_CUDA_ARCH_LIST="6.0;7.0;8.0;9.0" \
USE_NCCL=1 \
USE_SYSTEM_NCCL=1 \
USE_ROCM=0 \
NCCL_INCLUDE_DIR=$NCCL_HOME/include \
NCCL_LIB_DIR=$NCCL_HOME/lib \
python setup.py install --cmake-only
```

**Build flags explained:**
- `MAX_JOBS=4`: Parallel compilation (4 cores)
- `USE_XNNPACK=0`: Disable XNNPACK (AVX512-FP16 assembler compatibility)
- `USE_CUDA=1`: Enable CUDA support
- `USE_MPI=1`: Enable MPI backend
- `USE_CUDA_MPI=1`: Enable CUDA-aware MPI
- `TORCH_CUDA_ARCH_LIST="6.0;7.0;8.0;9.0"`: Target multiple GPU architectures
- `USE_NCCL=1`: Enable NCCL backend
- `USE_SYSTEM_NCCL=1`: Use system NCCL instead of bundled
- `USE_ROCM=0`: Explicitly disable ROCm (prevent conflicts)

### Step 8: Build and Install

```bash
MAX_JOBS=4 \
USE_XNNPACK=0 \
USE_CUDA=1 \
USE_MPI=1 \
USE_CUDA_MPI=1 \
USE_DISTRIBUTED=1 \
BUILD_TEST=0 \
TORCH_CUDA_ARCH_LIST="6.0;7.0;8.0;9.0" \
USE_NCCL=1 \
USE_SYSTEM_NCCL=1 \
USE_ROCM=0 \
NCCL_INCLUDE_DIR=$NCCL_HOME/include \
NCCL_LIB_DIR=$NCCL_HOME/lib \
python setup.py develop
```

**Note:** This takes 4-6 hours.

## Troubleshooting

### Build Errors

#### 1. XNNPACK AVX512-FP16 Assembler Error

**Error:**
```
Error: no such instruction: 'vcvtne2ps2bf16 %zmm1,%zmm0,%zmm2'
```

**Cause:** System assembler doesn't support AVX512-FP16 instructions.

**Solution:** Build with `USE_XNNPACK=0` (already in our build script).

#### 2. ROCm Conflict

**Error:**
```
Both CUDA and ROCm are enabled and found. Please turn off one of them.
```

**Cause:** MRI compute nodes have both NVIDIA CUDA and AMD ROCm installed.

**Solution:** Build with `USE_ROCM=0` (already in our build script).

#### 3. libcudadevrt.a not found

**Error:**
```
/usr/bin/ld: cannot find -lcudadevrt
```

**Cause:** CUDA static libraries not in linker search path.

**Solution:**
```bash
export LIBRARY_PATH=/opt/cuda/12.6/targets/x86_64-linux/lib:$LIBRARY_PATH
```

#### 4. Out of Memory during compilation

**Error:**
```
nvcc error: 'cudafe++' died due to signal 9 (Kill signal)
```

**Cause:** Compilation requires more memory than available.

**Solution:** Request node with more RAM:
```bash
salloc -p devel -t 6:00:00 --mem=128G
# Or use batch submission for longer builds with more memory
```

#### 5. Time Limit Exceeded

**Error:**
```
salloc: Job has exceeded its time limit
```

**Cause:** Build takes longer than allocated time.

**Solution:** Use compute partition with 12-hour limit instead of devel (3-hour limit).

### Runtime Errors

#### 1. GLIBCXX_3.4.30 not found

**Error:**
```
ImportError: libstdc++.so.6: version 'GLIBCXX_3.4.30' not found
```

**Cause:** PyTorch was built with GCC 13.3.0, but conda's libstdc++ is older.

**Solution:**
```bash
# For quick testing:
export LD_PRELOAD=/opt/gcc/13.3.0/lib64/libstdc++.so.6:$LD_PRELOAD

# For permanent use:
source ~/osu-hpc-ai/setup_runtime_env.sh
```

#### 2. CUDA not available

**Error:**
```python
>>> torch.cuda.is_available()
False
```

**Cause:** Not running on a GPU node or CUDA not properly loaded.

**Solution:**
```bash
# Allocate GPU node
salloc -p devel -t 1:00:00 -C cuda -N 1 --gpus-per-node=1

# Verify CUDA
nvidia-smi
```

#### 3. MPI backend not available

**Error:**
```python
>>> dist.is_mpi_available()
False
```

**Cause:** MVAPICH-Plus not in PATH or not built with MPI support.

**Solution:**
```bash
# Verify MVAPICH paths
which mpicc
export PATH=~/osu-hpc-ai-dev/install/mvapich-plus-4.1-cuda12.6/bin:$PATH
```

## Verification

### Quick Verification

```bash
# Set runtime environment
source ~/osu-hpc-ai/setup_runtime_env.sh

# Check version
python -c "import torch; print('PyTorch:', torch.__version__)"

# Check CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Check MPI
python -c "import torch.distributed as dist; print('MPI:', dist.is_mpi_available())"

# Check Gloo
python -c "import torch.distributed as dist; print('Gloo:', dist.is_gloo_available())"
```

### Full Test Suite

```bash
cd ~/osu-hpc-ai/tests/pytorch
source ~/osu-hpc-ai/setup_runtime_env.sh
python test_installation.py
```

Expected output:
```
PyTorch Installation Verification
============================================================
Passed: 8/8

Status: All tests passed
```

### Integration Tests

Verify PyTorch works with other frameworks:

```bash
# Test DeepSpeed
cd ~/osu-hpc-ai/tests/deepspeed
source ~/osu-hpc-ai/setup_runtime_env.sh
python test_installation.py

# Test SGLang
cd ~/osu-hpc-ai/tests/sglang
python test_installation.py
```

## Known Issues

### 1. Long Build Time

**Issue:** Build takes 4-6 hours on compute nodes.

**Cause:** PyTorch is a large codebase with extensive CUDA kernel compilation.

**Workaround:** Use automated build script and let it run overnight.

### 2. pynvml Deprecation Warning

**Issue:** Deprecation warning on import.

**Impact:** Cosmetic only, doesn't affect functionality.

**Solution:** Can be safely ignored.

### 3. Memory Requirements

**Issue:** Build may fail with OOM on standard nodes.

**Cause:** CUDA kernel compilation is memory-intensive.

**Solution:** Request high-memory nodes (64-128GB) for build.

---
