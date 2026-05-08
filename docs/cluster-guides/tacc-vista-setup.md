# TACC Vista Setup Guide

This guide covers setting up and using the OSU HPC-AI stack on TACC Vista.

## Cluster Overview

| Component | Details |
|-----------|---------|
| Architecture | ARM64 (NVIDIA Grace CPU) |
| GPUs | NVIDIA H200 / GH200 (96 GB HBM3, 1 per node) |
| CPU | NVIDIA Grace (72 cores @ 3.1 GHz) |
| CPU Memory | 116 GB DDR5 (CPU-GPU unified via NVLink-C2C) |
| Interconnect | Mellanox NDR InfiniBand (400 Gb/s) |
| Job Scheduler | SLURM |
| Status | Supported and tested |

## ARM64 Architecture Considerations

Vista uses NVIDIA Grace CPUs (ARM64/AArch64). This affects:

- **Compiler flags**: Use `-march=armv8.2-a+fp16` instead of x86 flags
- **Wheel compatibility**: x86_64 wheels do not work on ARM64; build from source or use ARM64 wheels
- **ABI**: ARM64 uses a different ABI — do not copy binaries from x86 clusters
- **GH200 NVLink-C2C**: CPU and GPU share a unified memory space; GPU memory is accessible from CPU and vice versa without PCIe overhead

> The HPC-AI stack provides ARM64-compatible builds. Follow the standard build guides but on an ARM64 Vista node.

## Module Loading

```bash
module reset
module load gcc/13.3.0    # Or the latest GCC available on Vista
module load cuda/12.5     # Vista uses CUDA 12.5
```

Check available modules:

```bash
module avail gcc
module avail cuda
```

## Environment Setup

```bash
#!/bin/bash
# save as ~/setup_hpc_ai_vista.sh

module reset
module load gcc/13.3.0
module load cuda/12.5

# MVAPICH-Plus (ARM64 build)
export MVAPICH_HOME="${HOME}/osu-hpc-ai-dev/install/mvapich-plus-4.1-cuda12.5-arm64"
export PATH="${MVAPICH_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MVAPICH_HOME}/lib:${LD_LIBRARY_PATH}"
export CPATH="${MVAPICH_HOME}/include:${CPATH}"

# CUDA
export CUDA_HOME=/opt/cuda/12.5
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# NDR InfiniBand (400 Gb/s)
export MV2_USE_CUDA=1
export MV2_USE_GPUDIRECT_RDMA=1
export MV2_CUDA_USE_NAIVE=0

eval "$(conda shell.bash hook)"
conda activate hpc-ai-build
```

## Building from Source on Vista

Because Vista is ARM64, you must build all frameworks on a Vista compute node (not cross-compile):

```bash
# Allocate a compute node for building
salloc -p development -t 2:00:00 -N 1 --gpus-per-node=1

# After SSH to compute node, set up environment
source ~/setup_hpc_ai_vista.sh

# Build PyTorch (ARM64 — no x86-specific flags)
cd ~/osu-hpc-ai
bash build_scripts/mri/build_pytorch_mri.sh   # Adjust paths for Vista
```

### ARM64-Specific Build Flags

When building PyTorch or extensions, ensure no x86-specific flags are passed:

```bash
# Remove x86 flags from CFLAGS/CXXFLAGS if present
unset CFLAGS
unset CXXFLAGS

# Set ARM64 flags
export CFLAGS="-march=armv8.2-a+fp16 -O3"
export CXXFLAGS="-march=armv8.2-a+fp16 -O3"

# Target Grace CPU + H200 GPU (Hopper architecture)
export TORCH_CUDA_ARCH_LIST="9.0"     # H100/H200 = sm_90
```

## GH200 NVLink-C2C Memory

Vista's GH200 uses NVLink-C2C to connect the Grace CPU and H200 GPU. This enables:
- **Unified memory**: GPU code can access CPU memory directly without explicit copies
- **High bandwidth**: ~900 GB/s CPU-GPU bandwidth (vs. ~64 GB/s on PCIe)

To take advantage in PyTorch:

```python
# Unified memory allocation (accessible from both CPU and GPU)
tensor = torch.zeros(1024, device='cuda')
cpu_view = tensor.cpu()   # Zero-copy on GH200

# Pinned memory is especially efficient on GH200
tensor = torch.zeros(1024, pin_memory=True)
```

## SLURM Partitions on Vista

| Partition | Time Limit | Nodes | Use Case |
|-----------|------------|-------|----------|
| `development` | 2:00:00 | Up to 4 | Interactive testing |
| `gpu-h100` | 48:00:00 | Varies | Production training |
| `large` | 24:00:00 | Varies | Large-scale jobs |

```bash
# Interactive session
salloc -p development -t 2:00:00 -N 1 --gpus-per-node=1

# Check available partitions
sinfo -o "%P %a %l %D %t"
```

## Running Training Jobs on Vista

### For latest PyTorch DDP instructions and tuning on Vista, please refer to [this section](docs/user-guides/pytorch-ddp.md)

## Common Vista-Specific Issues

### Wrong architecture binary

**Error**: `Exec format error` or `cannot execute binary file`

**Cause**: Trying to run an x86_64 binary on ARM64.

**Solution**: Rebuild all binaries on a Vista compute node.

### Missing ARM64 Python wheels

Many pip packages provide ARM64 wheels for modern Python versions. If a wheel is missing:

```bash
# Build from source
pip install --no-binary :all: package_name

# Or use conda which handles ARM64
conda install package_name
```

### CUDA arch mismatch (H200 = sm_90)

```bash
export TORCH_CUDA_ARCH_LIST="9.0"
```

If you see `no kernel image available for execution` you compiled for the wrong CUDA architecture.

### Grace CPU binding

With 1 GPU per node, each MPI process should own the full node:

```bash
mpirun -np 4 --bind-to none python train.py
```

## Additional Resources

- **Troubleshooting Guide**: [../troubleshooting.md](../troubleshooting.md)
- **TACC Vista User Guide**: [https://docs.tacc.utexas.edu/hpc/vista/](https://docs.tacc.utexas.edu/hpc/vista/)
- **Official HPC-AI Userguide**: [https://hpc-ai.engineering.osu.edu/userguide/](https://hpc-ai.engineering.osu.edu/userguide/)
- **MVAPICH-Plus Downloads**: [https://mvapich.cse.ohio-state.edu/downloads/](https://mvapich.cse.ohio-state.edu/downloads/)
