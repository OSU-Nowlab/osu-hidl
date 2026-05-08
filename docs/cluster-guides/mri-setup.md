# MRI Cluster Setup Guide

This guide covers setting up and using the OSU HPC-AI stack on the OSU MRI (Massive RAM and I/O) cluster.

## Cluster Overview

| Component | Details |
|-----------|---------|
| Architecture | x86_64 |
| GPUs | NVIDIA A100 (40GB SXM4) |
| CUDA | 12.6 |
| Interconnect | InfiniBand HDR |
| Job Scheduler | SLURM |
| Status | Production, fully supported |

## Module Loading

Always start with `module reset` to clear any inherited modules before loading the correct stack:

```bash
module reset
module load gcc/13.3.0
module load cuda/12.6
```

> **Important**: Load GCC before CUDA. Loading in the wrong order can cause linker conflicts.

Verify:

```bash
which gcc     # Should show /opt/gcc/13.3.0/bin/gcc
nvcc --version  # Should show CUDA 12.6
```

## Environment Setup

Add these to your `~/.bashrc` or create an activation script:

```bash
#!/bin/bash
# save as ~/setup_hpc_ai.sh

# Modules
module reset
module load gcc/13.3.0
module load cuda/12.6

# MVAPICH-Plus
export MVAPICH_HOME="${HOME}/osu-hpc-ai-dev/install/mvapich-plus-4.1-cuda12.6"
export PATH="${MVAPICH_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MVAPICH_HOME}/lib:${LD_LIBRARY_PATH}"
export CPATH="${MVAPICH_HOME}/include:${CPATH}"

# CUDA
export CUDA_HOME=/opt/cuda/12.6
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Fix libstdc++ version mismatch
export LD_PRELOAD=/opt/gcc/13.3.0/lib64/libstdc++.so.6:$LD_PRELOAD

# GPUDirect RDMA (InfiniBand)
export MV2_USE_CUDA=1
export MV2_USE_GPUDIRECT_RDMA=1
export MV2_CUDA_USE_NAIVE=0

# Conda
eval "$(conda shell.bash hook)"
conda activate hpc-ai-build
```

Source before working: `source ~/setup_hpc_ai.sh`

## SLURM Partitions

| Partition | Time Limit | GPUs | Use Case |
|-----------|------------|------|----------|
| `devel` | 1:00:00 | Up to 4 per node | Interactive testing, short jobs |
| `gpu` | 24:00:00+ | Up to 8 per node | Production training jobs |
| `long` | 72:00:00 | Varies | Long-running jobs |

Check available partitions:

```bash
sinfo -o "%P %a %l %D %t %N"
```

## Interactive GPU Sessions

```bash
# Single GPU (quick tests, debugging)
salloc -p devel -t 1:00:00 -C cuda -N 1 --gpus-per-node=1

# Multi-GPU (distributed testing)
salloc -p devel -t 3:00:00 -C cuda -N 1 --gpus-per-node=4

# Multi-GPU with extra RAM (vLLM build, large models)
salloc -p devel -t 3:00:00 -C cuda -N 1 --gpus-per-node=2 --mem=128G

# Multi-node
salloc -p gpu -t 4:00:00 -C cuda -N 2 --gpus-per-node=4
```

After allocation, SSH to the compute node or use `srun`:

```bash
srun --pty bash
```

## Running Training Jobs

### PyTorch DDP (mpirun)

```bash
source ~/setup_hpc_ai.sh

mpirun -np 4 -ppn 4 python examples/pytorch/simple_distributed.py
```

Multi-node:

```bash
# Get node list from SLURM
NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ',')

mpirun -np 8 -ppn 4 -host ${NODES%,} \
    -x LD_PRELOAD -x LD_LIBRARY_PATH -x PATH \
    -x MV2_USE_CUDA=1 -x MV2_USE_GPUDIRECT_RDMA=1 \
    python train.py
```

### DeepSpeed

```bash
deepspeed --num_gpus=4 examples/deepspeed/simple_zero_training.py \
    --deepspeed_config=examples/deepspeed/ds_config_zero2.json
```

### SLURM Batch Job Template

```bash
#!/bin/bash
#SBATCH -J hpc-ai-train
#SBATCH -N 2
#SBATCH --gpus-per-node=4
#SBATCH -t 4:00:00
#SBATCH -p gpu
#SBATCH -o slurm-%j.out

# Environment
module reset
module load gcc/13.3.0
module load cuda/12.6

export MVAPICH_HOME="${HOME}/osu-hpc-ai-dev/install/mvapich-plus-4.1-cuda12.6"
export PATH="${MVAPICH_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MVAPICH_HOME}/lib:${LD_LIBRARY_PATH}"
export LD_PRELOAD=/opt/gcc/13.3.0/lib64/libstdc++.so.6:$LD_PRELOAD
export MV2_USE_CUDA=1
export MV2_USE_GPUDIRECT_RDMA=1

eval "$(conda shell.bash hook)"
conda activate hpc-ai-build

# Launch
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)

mpirun -np 8 -ppn 4 \
    -x MASTER_ADDR=$MASTER_ADDR \
    -x MASTER_PORT=29500 \
    -x LD_PRELOAD -x LD_LIBRARY_PATH -x PATH \
    python train.py
```

Submit: `sbatch train.slurm`

Monitor: `squeue -u $USER`

## Conda Environment Path

The shared HPC-AI conda environment is `hpc-ai-build`. It is located in your home directory under `miniconda3/envs/hpc-ai-build/`.

```bash
# Activate
conda activate hpc-ai-build

# List installed packages
conda list | grep -E "torch|deepspeed|vllm|sglang"

# Check Python path
which python
```

## Build Script Paths

All MRI-specific automated build scripts are in `build_scripts/mri/`:

| Script | Purpose | Time |
|--------|---------|------|
| `build_pytorch_mri.sh` | Build PyTorch 2.10.0 | 4–6 hours |
| `build_deepspeed_mri.sh` | Build DeepSpeed | 20–30 min |
| `build_sglang_mri.sh` | Build SGLang | 10–15 min |
| `build_vllm_mri.sh` | Build vLLM (needs 128GB RAM) | 30–60 min |
| `build_megatron_mri.sh` | Set up Megatron-LM for trainer validation | 10–20 min |
| `build_verl_mri.sh` | Set up verl after PyTorch, vLLM, and Megatron-LM | 10–20 min |
| `submit_pytorch_build.slurm` | Submit PyTorch build as batch job | not applicable |

Run PyTorch build as a batch job (avoids interactive timeout):

```bash
sbatch build_scripts/mri/submit_pytorch_build.slurm
```

## Common MRI-Specific Errors

### `libstdc++.so.6: version 'GLIBCXX_3.4.30' not found`

```bash
export LD_PRELOAD=/opt/gcc/13.3.0/lib64/libstdc++.so.6:$LD_PRELOAD
```

### Build killed silently (OOM)

Allocate more RAM for the build:

```bash
salloc --mem=128G
```

And set `MAX_JOBS=1` in the build script environment.

### `nvidia-smi: command not found` on login node

Login nodes do not have GPUs. Allocate a compute node first:

```bash
salloc -p devel -t 1:00:00 -C cuda -N 1 --gpus-per-node=1
```

### Module conflicts

```bash
module reset   # Always start fresh
module list    # Show what is currently loaded
```

## Additional Resources

- **Troubleshooting Guide**: [../troubleshooting.md](../troubleshooting.md)
- **Build Guides**: [../build-guides/](../build-guides/)
- **Official HPC-AI Userguide**: [https://hpc-ai.engineering.osu.edu/userguide/](https://hpc-ai.engineering.osu.edu/userguide/)
- **MVAPICH-Plus Downloads**: [https://mvapich.cse.ohio-state.edu/downloads/](https://mvapich.cse.ohio-state.edu/downloads/)
