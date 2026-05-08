# SGLang Build Guide - MRI Cluster

This guide covers building SGLang from source with custom PyTorch (MVAPICH-Plus enabled) on the MRI cluster.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Manual Build Steps](#manual-build-steps)
- [Troubleshooting](#troubleshooting)
- [Verification](#verification)

## Overview

**What is SGLang?**

SGLang is a fast serving framework for large language models and vision language models, featuring:
- RadixAttention for efficient KV cache management
- Structured generation (JSON, regex)
- High-throughput inference
- Multi-modal support

**Why build from source?**

Building from source ensures compatibility with our custom PyTorch build that includes MVAPICH-Plus integration.

**Key Features of This Build:**
- SGLang 0.4.9
- Compatible with PyTorch 2.10.0 (custom MVAPICH-Plus build)
- CUDA 12.6 support
- Python 3.11

## Prerequisites

### System Requirements

1. **Custom PyTorch Installation:**
   SGLang requires PyTorch to be installed first. See [PyTorch build guide](pytorch.md).

2. **GPU Node Access:**
   ```bash
   salloc -p devel -t 1:00:00 -C cuda -N 1 --gpus-per-node=1
   ```

3. **Software Dependencies:**
   - GCC 13.3.0
   - CUDA 12.6
   - Python 3.11 (conda environment)
   - Custom PyTorch 2.10.0 (from HPC-AI stack)

4. **Build Resources:**
   - Time: ~10-15 minutes
   - Disk: ~2GB
   - Memory: Standard (no special requirements)

## Quick Start

### Automated Build (Recommended)

```bash
cd ~/osu-hpc-ai
bash build_scripts/mri/build_sglang_mri.sh
```

This script automatically:
- Loads required modules
- Activates conda environment
- Installs dependencies without overwriting custom PyTorch
- Builds SGLang from source
- Runs verification tests

### Verification

```bash
conda activate hpc-ai-build
export LD_PRELOAD=/opt/gcc/13.3.0/lib64/libstdc++.so.6:$LD_PRELOAD

python -c "import sglang; print('SGLang version:', sglang.__version__)"

# Run full test suite
cd ~/osu-hpc-ai/tests/sglang
python test_installation.py
```

## Manual Build Steps

If you need to build manually or customize the build:

### Step 1: Allocate GPU Node

```bash
salloc -p devel -t 1:00:00 -C cuda -N 1 --gpus-per-node=1
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
# Required for runtime
export LD_PRELOAD=/opt/gcc/13.3.0/lib64/libstdc++.so.6:$LD_PRELOAD
```

### Step 5: Install Dependencies (Without Overwriting PyTorch)

**Critical:** Use `--no-deps` to prevent pip from installing public PyTorch

```bash
cd ~/osu-hpc-ai/frameworks/sglang

# Install SGLang dependencies manually (excluding torch)
pip install --no-deps aiohttp anthropic async-timeout cloudpickle compressed-tensors \
    diskcache fastapi interegular lark multiprocess numpy outlines outlines-core \
    openai packaging pillow prometheus_client psutil pydantic ray[default] \
    regex requests tiktoken transformers uvicorn uvloop

# Install setuptools_scm if needed
pip install setuptools_scm
```

### Step 6: Build SGLang

```bash
cd ~/osu-hpc-ai/frameworks/sglang

# Clean previous builds
rm -rf build/ dist/ *.egg-info

# Build with --no-deps to prevent PyTorch overwrite
pip install -e . --no-deps
```

## Troubleshooting

### Build Errors

#### 1. setuptools_scm missing

**Error:**
```
ModuleNotFoundError: No module named 'setuptools_scm'
```

**Solution:**
```bash
pip install setuptools_scm
```

#### 2. PyTorch overwrite

**Error:**
```
Installing collected packages: torch-2.10.0
Attempting uninstall: torch-2.10.0
```

**Cause:** `pip install -e .` without `--no-deps` installs dependencies including public PyTorch.

**Solution:** Always use `--no-deps` and install dependencies manually (see Step 5).

### Runtime Errors

#### 1. Import Error: libstdc++

**Error:**
```
ImportError: libstdc++.so.6: version 'GLIBCXX_3.4.30' not found
```

**Solution:**
```bash
export LD_PRELOAD=/opt/gcc/13.3.0/lib64/libstdc++.so.6:$LD_PRELOAD
```

#### 2. CUDA not available

**Error:**
```python
>>> import torch; torch.cuda.is_available()
False
```

**Cause:** Not on GPU node or CUDA not properly loaded.

**Solution:**
```bash
# Allocate GPU node
salloc -p devel -t 1:00:00 -C cuda -N 1 --gpus-per-node=1

# Verify CUDA
nvidia-smi
```

## Verification

### Quick Verification

```bash
export LD_PRELOAD=/opt/gcc/13.3.0/lib64/libstdc++.so.6:$LD_PRELOAD

# Check version
python -c "import sglang; print('SGLang:', sglang.__version__)"

# Check PyTorch (should be custom 2.10.0)
python -c "import torch; print('PyTorch:', torch.__version__)"

# Check CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Full Test Suite

```bash
cd ~/osu-hpc-ai/tests/sglang
export LD_PRELOAD=/opt/gcc/13.3.0/lib64/libstdc++.so.6:$LD_PRELOAD
python test_installation.py
```

Expected output:
```
============================================================
SGLang Installation Verification
============================================================
...
============================================================
Test Summary
============================================================
Passed: 6/6

Status: All tests passed
```

### Run Examples

```bash
cd ~/osu-hpc-ai/examples/sglang
export LD_PRELOAD=/opt/gcc/13.3.0/lib64/libstdc++.so.6:$LD_PRELOAD

# Simple text generation
python simple_generation.py
```

---
