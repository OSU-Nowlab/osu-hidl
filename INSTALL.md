# OSU HPC-AI Installation Guide

This guide provides detailed instructions for installing the OSU HPC-AI stack on HPC systems.

## Table of Contents

- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Installation Methods](#installation-methods)
- [Post-Installation](#post-installation)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Option 1: Pre-built Wheels (Recommended)

The fastest way to get started is using our pre-built wheels:

```bash
# Install PyTorch (wheel URL provided after registration)
# pip install <wheel-url>/torch-2.x.x+cu126-cp312-cp312-linux_x86_64.whl

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

Pre-built wheels are available for registered users. Visit [hpc-ai.engineering.osu.edu](https://hpc-ai.engineering.osu.edu) for download instructions.

### Option 2: Build from Source

For custom builds or unsupported configurations:

```bash
# Clone the monorepo
git clone --recursive https://github.com/OSU-Nowlab/osu-hpc-ai.git
cd osu-hpc-ai

# Build PyTorch
cd frameworks/pytorch
git submodule update --init --recursive
export USE_CUDA=1 USE_MPI=1
MAX_JOBS=4 python setup.py install
```

---

## Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Linux (kernel 3.10+) | Linux (kernel 5.x+) |
| **GPU** | NVIDIA (Compute ≥ 6.0) | NVIDIA A100/H100 |
| **CUDA** | 11.0 | 12.6 |
| **GCC** | 9.0 | 13.3.0 |
| **Python** | 3.8 | 3.11-3.12 |
| **RAM** | 32 GB | 128+ GB |
| **Storage** | 50 GB | 200+ GB |

### Software Dependencies

#### Required

1. **CUDA Toolkit**
   ```bash
   # Verify CUDA installation
   nvcc --version

   # If not installed, download from:
   # https://developer.nvidia.com/cuda-downloads
   ```

2. **GCC Compiler**
   ```bash
   # Verify GCC version
   gcc --version

   # Must be compatible with your CUDA version
   # CUDA 12.x requires GCC ≤ 12
   ```

3. **Python**
   ```bash
   # Verify Python version
   python3 --version

   # Install via system package manager or conda
   ```

4. **CMake**
   ```bash
   cmake --version  # Should be ≥ 3.18

   # Install if needed:
   pip install cmake
   ```

#### Optional (but Recommended)

1. **MVAPICH-Plus** (for MPI distributed training)
   - Download from: http://mvapich.cse.ohio-state.edu/

2. **cuDNN** (for accelerated neural networks)
   ```bash
   pip install nvidia-pyindex
   pip install nvidia-cudnn
   ```

---

## Installation Methods

### Method 1: Using Pre-built Wheels

#### Step 1: Register and Get Access

Register at [hpc-ai.engineering.osu.edu](https://hpc-ai.engineering.osu.edu) to receive download links for pre-built wheels matching your:
- CUDA version (e.g., cu126 for CUDA 12.6)
- Python version (e.g., cp312 for Python 3.12)
- Architecture (x86_64 or aarch64)

#### Step 2: Install

```bash
# Create virtual environment (recommended)
python3 -m venv ~/hpc-ai-env
source ~/hpc-ai-env/bin/activate

# Install PyTorch wheel (URL provided after registration)
# pip install <wheel-url>/torch-<version>-<platform>.whl

# Install additional dependencies
pip install numpy scipy scikit-learn
```

#### Step 3: Verify

```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Method 2: Building from Source

#### Step 1: Clone Repository

```bash
# Clone with submodules
git clone --recursive https://github.com/OSU-Nowlab/osu-hpc-ai.git
cd osu-hpc-ai

# Or clone and initialize submodules separately
git clone https://github.com/OSU-Nowlab/osu-hpc-ai.git
cd osu-hpc-ai
git submodule update --init --recursive
```

#### Step 2: Set Up Environment

```bash
# Create conda environment
conda create -n hpc-ai-build python=3.11
conda activate hpc-ai-build

# Install build dependencies
pip install numpy ninja pyyaml cmake setuptools typing-extensions
```

#### Step 3: Build MVAPICH-Plus (Optional)

```bash
# See http://mvapich.cse.ohio-state.edu/ for detailed instructions

# Quick build:
cd /path/to/mvapich-plus
./autogen.sh
./configure --prefix=$HOME/mvapich-install --with-cuda=$CUDA_HOME --enable-cuda
make -j$(nproc) install

# Set environment
export MVAPICH_HOME=$HOME/mvapich-install
export PATH=$MVAPICH_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MVAPICH_HOME/lib:$LD_LIBRARY_PATH
```

#### Step 4: Build PyTorch

```bash
cd frameworks/pytorch

# Configure build
export USE_CUDA=1
export USE_MPI=1  # If MVAPICH-Plus is installed
export USE_CUDNN=1
export USE_NCCL=1
export TORCH_CUDA_ARCH_LIST="6.0;7.0;8.0;9.0"  # Adjust for your GPUs
export MAX_JOBS=4  # Adjust based on available RAM

# Build
python setup.py develop  # For development
# OR
python setup.py install  # For production

# This will take 1-3 hours depending on your system
```

For detailed build instructions, see [docs/build-guides/pytorch.md](docs/build-guides/pytorch.md).

For running distributed training with the MPI backend, see [docs/user-guides/pytorch-ddp.md](docs/user-guides/pytorch-ddp.md).

### Method 3: Automated Build Script (MRI Cluster)

For users on the MRI cluster at OSU:

```bash
# Clone repository
git clone --recursive https://github.com/OSU-Nowlab/osu-hpc-ai.git
cd osu-hpc-ai

# Build frameworks in order (see docs/build-guides/ for details)
bash build_scripts/mri/build_pytorch_mri.sh
bash build_scripts/mri/build_deepspeed_mri.sh
bash build_scripts/mri/build_sglang_mri.sh

# vLLM requires special allocation
salloc -p devel -t 3:00:00 -C cuda -N 1 --gpus-per-node=2 --mem=128G
bash build_scripts/mri/build_vllm_mri.sh
```

---

## Post-Installation

### Environment Configuration

Create a setup script for easy environment loading:

```bash
# Create ~/setup_hpc_ai.sh
cat > ~/setup_hpc_ai.sh << 'EOF'
#!/bin/bash

# Load modules (adjust for your system)
module load gcc/13.3.0
module load cuda/12.6

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hpc-ai-build

# Set MPI paths (if using MVAPICH-Plus)
export MVAPICH_HOME=$HOME/mvapich-install
export PATH=$MVAPICH_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MVAPICH_HOME/lib:$LD_LIBRARY_PATH

# Fix libstdc++ issues (if needed)
export LD_PRELOAD=/opt/gcc/13.3.0/lib64/libstdc++.so.6

echo "HPC-AI environment loaded"
EOF

chmod +x ~/setup_hpc_ai.sh
```

Usage:
```bash
source ~/setup_hpc_ai.sh
```

### Optional Components

Additional frameworks may be available in the `frameworks/` directory. Refer to each framework's README for installation instructions.

---

## Verification

### Basic Tests

```bash
# Test PyTorch import
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Test CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

# Test MPI backend
python -c "import torch.distributed as dist; print(f'MPI available: {dist.is_mpi_available()}')"

# Test Gloo backend
python -c "import torch.distributed as dist; print(f'Gloo available: {dist.is_gloo_available()}')"
```

### MPI Communication Tests

```bash
# Run MPI tests (requires 2+ processes)
cd tests/pytorch
mpirun -np 2 python test_mpi_comm.py

# Or on SLURM cluster
srun -N 2 --ntasks-per-node=2 python test_mpi_comm.py
```

Output:
```
============================================================
PyTorch MPI Communication Tests
============================================================
[Rank 0/2] MPI initialization successful
[Rank 1/2] MPI initialization successful
...
Total: 7/7 tests passed

All tests passed!
```

### GPU Computation Test

```bash
python << 'EOF'
import torch

# Create tensors on GPU
x = torch.randn(1000, 1000, device='cuda')
y = torch.randn(1000, 1000, device='cuda')

# Matrix multiplication
z = torch.matmul(x, y)

print(f"GPU computation successful")
print(f"  Result shape: {z.shape}")
print(f"  Device: {z.device}")
EOF
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Not Found

**Symptom**: `RuntimeError: CUDA not found`

**Solution**:
```bash
export CUDA_HOME=/usr/local/cuda-12.6  # Adjust path
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

#### 2. Library Version Mismatch

**Symptom**: `version 'GLIBCXX_X.X.XX' not found`

**Solution**:
```bash
# Preload correct libstdc++
export LD_PRELOAD=/opt/gcc/13.3.0/lib64/libstdc++.so.6
```

#### 3. Out of Memory During Build

**Symptom**: Build killed with no error

**Solution**:
```bash
# Reduce parallel jobs
export MAX_JOBS=2  # Or even 1
python setup.py install
```

#### 4. MPI Not Available

**Symptom**: `RuntimeError: MPI backend not available`

**Solution**:
```bash
# Ensure MPI is in PATH
which mpicc

# Rebuild PyTorch with MPI
export USE_MPI=1
python setup.py develop
```

#### 5. cuDNN Not Found

**Symptom**: `Could not find cuDNN library`

**Solution**:
```bash
# Install via pip
pip install nvidia-pyindex
pip install nvidia-cudnn

# Or set paths manually
export CUDNN_HOME=/path/to/cudnn
export LD_LIBRARY_PATH=$CUDNN_HOME/lib:$LD_LIBRARY_PATH
```

### Getting Help

- **Documentation**: [https://hpc-ai.engineering.osu.edu](https://hpc-ai.engineering.osu.edu)
- **Issues**: [GitHub Issues](https://github.com/OSU-Nowlab/osu-hpc-ai/issues)
- **HPC-AI Project**: Visit the official HPC-AI page for contact information and support

When reporting issues, please include:
- System information (`uname -a`)
- CUDA version (`nvcc --version`)
- GCC version (`gcc --version`)
- Python version (`python --version`)
- Full error message
- Build logs (if building from source)

---

## Next Steps

After installation:

1. **Run Examples**: See [examples/](examples/) for usage examples
2. **Read User Guides**: Get started with framework-specific documentation:
   - [PyTorch DDP with MPI](docs/user-guides/pytorch-ddp.md)
   - [DeepSpeed ZeRO](docs/user-guides/deepspeed-zero.md)
   - [vLLM Serving](docs/user-guides/vllm-serving.md)
   - [SGLang Structured Output](docs/user-guides/sglang-structured.md)
   - [MCR-DL Backend Selection](docs/user-guides/mcr-dl-backend.md)
3. **Run Benchmarks**: Test performance with [benchmarks/](benchmarks/)
4. **Troubleshooting**: See [docs/troubleshooting.md](docs/troubleshooting.md) for common issues
5. **Cluster Setup**: Cluster-specific guides in [docs/cluster-guides/](docs/cluster-guides/)

---

**Maintained by**: NOWLAB Team, The Ohio State University
