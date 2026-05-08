# OLCF Frontier Setup Guide

This guide covers setting up and using the OSU HPC-AI stack on OLCF Frontier.

## Cluster Overview

| Component | Details |
|-----------|---------|
| Architecture | x86_64 (AMD EPYC) |
| CPUs | AMD EPYC 7A53 (64 cores @ 2 GHz, 2 sockets per node) |
| CPU Memory | 512 GB DDR4 |
| GPUs | AMD MI250X (4 per node, 2 GCDs each = 8 logical GPUs, 128 GB HBM2e total) |
| Interconnect | HPE Slingshot (200 Gb/s) |
| Job Scheduler | SLURM |
| Status | Supported and tested |

## AMD GPU Architecture Notes

### MI250X GCD Layout

Each MI250X has **2 Graphics Compute Dies (GCDs)**. SLURM and ROCm expose each GCD as an independent GPU:

- `--gpus-per-node=8` = 4 physical MI250X cards × 2 GCDs
- Each GCD has 64 GB HBM2e
- Inter-GCD communication within a card is faster (~200 GB/s) than cross-card

This means `torch.cuda.device_count()` returns 8 on a full Frontier node.

### ROCm Instead of CUDA

Frontier uses AMD ROCm instead of NVIDIA CUDA. PyTorch is compiled with ROCm support. Key differences:
- `import torch; torch.cuda.is_available()` — still returns `True` on ROCm (PyTorch maps CUDA APIs to HIP)
- CUDA kernels (FP8, custom CUDA ops in vLLM) may not be available on ROCm
- `TORCH_CUDA_ARCH_LIST` is replaced by `PYTORCH_ROCM_ARCH` (target: `gfx90a` for MI250X)

## Module Loading

```bash
module reset
module load PrgEnv-gnu       # GNU programming environment
module load gcc/12.2.0       # Or available GCC version
module load rocm/5.7.0       # ROCm for AMD GPUs
module load cray-python/3.11.5
```

Check available modules:

```bash
module avail rocm
module avail gcc
```

## Environment Setup

```bash
#!/bin/bash
# save as ~/setup_hpc_ai_frontier.sh

module reset
module load PrgEnv-gnu
module load gcc/12.2.0
module load rocm/5.7.0

# MVAPICH-Plus (ROCm-aware, Slingshot)
export MVAPICH_HOME="${HOME}/osu-hpc-ai-dev/install/mvapich-plus-4.1-rocm5.7-slingshot"
export PATH="${MVAPICH_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MVAPICH_HOME}/lib:${LD_LIBRARY_PATH}"
export CPATH="${MVAPICH_HOME}/include:${CPATH}"

# ROCm
export ROCM_HOME=/opt/rocm-5.7.0
export PATH="${ROCM_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_HOME}/lib:${LD_LIBRARY_PATH}"
export HIP_PATH=${ROCM_HOME}

# Slingshot network (libfabric)
export FI_CXI_RX_MATCH_MODE=hybrid
export MPICH_GPU_SUPPORT_ENABLED=1

eval "$(conda shell.bash hook)"
conda activate hpc-ai-build
```

## Building from Source on Frontier

Frontier requires ROCm-aware builds. The key difference from CUDA builds is specifying the MI250X target:

```bash
# ROCm GPU architecture for MI250X
export PYTORCH_ROCM_ARCH="gfx90a"

# Do NOT set TORCH_CUDA_ARCH_LIST on Frontier
unset TORCH_CUDA_ARCH_LIST

# Build PyTorch with ROCm support
USE_ROCM=1 USE_MPI=1 python setup.py install
```

## MI250X GPU Binding

With 8 GCDs per node, proper GPU-to-rank assignment is critical:

```bash
# 8 MPI ranks per node, one per GCD
mpirun -np 8 -ppn 8 python train.py
```

Inside your script, set the GPU based on local rank:

```python
import os
import torch

local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', 0))
device = torch.device(f'cuda:{local_rank}')
torch.cuda.set_device(device)
```

### Checking GCD Topology

```bash
rocm-smi --showtoponuma   # NUMA topology per GCD
rocm-smi --showbw          # Bandwidth between GCDs
```

## SLURM Partitions on Frontier

| Partition | Time Limit | Use Case |
|-----------|------------|----------|
| `batch` | 24:00:00 | Production training |
| `debug` | 0:30:00 | Quick testing |
| `extended` | 120:00:00 | Long runs (allocation required) |

```bash
# Interactive session (1 node, all 8 GCDs)
salloc -A project_id -p debug -t 0:30:00 -N 1

# Full node allocation
salloc -A project_id -p batch -t 4:00:00 -N 4
```

Note: Replace `project_id` with your OLCF project allocation.

## Running Training Jobs on Frontier

### For latest PyTorch DDP instructions and tuning on Frontier, please refer to [this section](docs/user-guides/pytorch-ddp.md)

## Common Frontier-Specific Issues

### `torch.cuda.is_available()` returns False

Check ROCm is loaded and visible:

```bash
rocm-smi                          # Should list MI250X GPUs
echo $ROCM_HOME                   # Should be set
python -c "import torch; print(torch.version.hip)"  # Should show ROCm version
```

### CUDA kernel not available on ROCm

Some CUDA-specific extensions (FP8, custom CUDA ops) do not work on ROCm. Use the standard PyTorch operations instead, or check if a HIP-ported version is available.

### Wrong GCD count

If `torch.cuda.device_count()` returns 4 instead of 8:

```bash
export ROCM_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   # Expose all GCDs
```

Or unset `HIP_VISIBLE_DEVICES` if it was set elsewhere.

### Slingshot connection errors

```bash
# Check libfabric provider
fi_info -p cxi   # Should list Slingshot endpoints

# Verify MPICH GPU support
echo $MPICH_GPU_SUPPORT_ENABLED   # Should be 1
```

### AMD-specific memory issues

MI250X does not support NVIDIA Unified Memory (`cudaMallocManaged`). Use explicit memory allocation:

```python
# Instead of torch.zeros(..., device='cuda')  — both work
# Ensure no CUDA-specific unified memory calls in custom extensions
tensor = torch.zeros(1024, device='cuda')   # OK on ROCm
```

## Additional Resources

- **Troubleshooting Guide**: [../troubleshooting.md](../troubleshooting.md)
- **OLCF Frontier User Guide**: [https://docs.olcf.ornl.gov/systems/frontier_user_guide.html](https://docs.olcf.ornl.gov/systems/frontier_user_guide.html)
- **ROCm Documentation**: [https://rocm.docs.amd.com/](https://rocm.docs.amd.com/)
- **Official HPC-AI Userguide**: [https://hpc-ai.engineering.osu.edu/userguide/](https://hpc-ai.engineering.osu.edu/userguide/)
- **MVAPICH-Plus Downloads**: [https://hpc-ai.engineering.osu.edu/userguide/](https://mvapich.cse.ohio-state.edu/downloads/)
