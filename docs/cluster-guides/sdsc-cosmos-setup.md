# SDSC Cosmos Setup Guide

This guide covers setting up and using the OSU HPC-AI stack on SDSC Cosmos.

## Cluster Overview

| Component | Details |
|-----------|---------|
| Architecture | x86_64 (AMD EPYC) |
| CPUs | AMD EPYC (96 cores @ 3.7 GHz) |
| Memory | 512 GB HBM3 (unified CPU+GPU) |
| GPUs | Integrated AMD CDNA3 APU (MI310X, 128 GB HBM3 per APU) |
| Interconnect | HPE Cray Slingshot-11 (200 Gb/s) |
| Status | Validated |

## Integrated GPU Architecture

Cosmos uses AMD CDNA3 APUs where the CPU and GPU share HBM3 memory. This is different from discrete GPU setups:

- **No PCIe**: CPU and GPU access the same physical HBM3 memory without a PCIe bus
- **Unified addressing**: GPU tensors can be directly accessed from CPU without explicit copies
- **No separate GPU memory**: All 128 GB is shared between compute and GPU workloads

This means `cudaMalloc` allocates from the shared HBM3 pool — there is no separate "GPU VRAM" vs "CPU RAM".

## Module Loading

```bash
module reset
module load PrgEnv-gnu       # Or PrgEnv-cray / PrgEnv-amd depending on Cosmos config
module load rocm             # ROCm for CDNA3
module load cray-python
```

Check available modules:

```bash
module avail
module avail rocm
```

## Environment Setup

```bash
#!/bin/bash
# save as ~/setup_hpc_ai_cosmos.sh

module reset
module load PrgEnv-gnu
module load rocm

# MVAPICH-Plus (ROCm-aware, Slingshot)
export MVAPICH_HOME="${HOME}/osu-hpc-ai-dev/install/mvapich-plus-4.1-rocm-slingshot"
export PATH="${MVAPICH_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MVAPICH_HOME}/lib:${LD_LIBRARY_PATH}"
export CPATH="${MVAPICH_HOME}/include:${CPATH}"

# ROCm
export ROCM_HOME=/opt/rocm
export PATH="${ROCM_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_HOME}/lib:${LD_LIBRARY_PATH}"

# ROCm-aware MPI + Slingshot
export FI_CXI_RX_MATCH_MODE=hybrid
export MPICH_GPU_SUPPORT_ENABLED=1

eval "$(conda shell.bash hook)"
conda activate hpc-ai-build
```

## Building from Source on Cosmos

Target the CDNA3 (MI310X) architecture:

```bash
# CDNA3 target
export PYTORCH_ROCM_ARCH="gfx942"   # MI300/CDNA3

USE_ROCM=1 USE_MPI=1 python setup.py install
```

## Running Training Jobs

### For latest PyTorch DDP instructions and tuning on Cosmos, please refer to [this section](docs/user-guides/pytorch-ddp.md)

## Unified Memory Usage

With HBM3 unified memory, you can leverage larger effective memory:

```python
import torch

# Tensors allocated on the GPU draw from the shared HBM3 pool
model = YourModel().cuda()

# On APU, pinned memory is less meaningful since there's no PCIe bottleneck
# Standard allocation works well
tensor = torch.zeros(large_size, device='cuda')
```

For workloads with large CPU-side preprocessing, the unified memory eliminates the usual CPU→GPU transfer bottleneck.

## Common Cosmos-Specific Issues

### GPU count unexpected

Cosmos APU configurations vary. Always check at runtime:

```bash
python -c "import torch; print(torch.cuda.device_count())"
rocm-smi --showallinfo
```

### ROCm version mismatch

Ensure PyTorch and ROCm module versions are compatible:

```bash
python -c "import torch; print(torch.version.hip)"
ls /opt/rocm*/   # Check available ROCm versions
```

### Slingshot libfabric issues

```bash
fi_info -p cxi   # Verify Slingshot provider
echo $FI_CXI_RX_MATCH_MODE   # Should be set
```

## Additional Resources

- **Troubleshooting Guide**: [../troubleshooting.md](../troubleshooting.md)
- **SDSC Cosmos User Guide**: [https://www.sdsc.edu/support/user_guides/cosmos.html](https://www.sdsc.edu/support/user_guides/cosmos.html)
- **ROCm Documentation**: [https://rocm.docs.amd.com/](https://rocm.docs.amd.com/)
- **Official HPC-AI Userguide**: [https://hpc-ai.engineering.osu.edu/userguide/](https://hpc-ai.engineering.osu.edu/userguide/)
