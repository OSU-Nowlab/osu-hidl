# PyTorch Distributed Data Parallel with MPI

This guide covers running PyTorch Distributed Data Parallel (DDP) training using the MPI backend with MVAPICH-Plus.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Initializing the MPI Backend](#initializing-the-mpi-backend)
- [Running DDP Training](#running-ddp-training)
- [MVAPICH-Plus Tuning](#mvapich-plus-tuning)
- [Common Patterns](#common-patterns)
- [Troubleshooting](#troubleshooting)

## Overview

PyTorch's Distributed Data Parallel (DDP) enables training across multiple GPUs and nodes. The MPI backend, powered by MVAPICH-Plus, provides:

- **CUDA-aware MPI**: Direct GPU-to-GPU communication without CPU staging
- **GPUDirect RDMA**: Low-latency transfers over InfiniBand/Slingshot
- **Vendor-neutral**: Works across NVIDIA and AMD GPUs
- **HPC integration**: Native compatibility with SLURM, PBS, and other job schedulers

> **Official Documentation**: For system-specific build scripts and optimized MVAPICH-Plus configurations for TACC Vista, OLCF Frontier, and SDSC Cosmos, see the official HPC-AI userguide:
> [https://hpc-ai.engineering.osu.edu/userguide/pytorch-ddp/](https://hpc-ai.engineering.osu.edu/userguide/pytorch-ddp/)

## Prerequisites

### MVAPICH-Plus

MVAPICH-Plus is the preferred MPI runtime for DDP training with PyTorch on GPUs and CPUs.

- **Download**: [https://mvapich.cse.ohio-state.edu/downloads/](https://mvapich.cse.ohio-state.edu/downloads/)
- **User Guide**: [https://mvapich-docs.readthedocs.io/en/mvapich-plus/](https://mvapich-docs.readthedocs.io/en/mvapich-plus/)

### PyTorch with GPU-Aware MPI

We provide an open-source PyTorch branch with enhanced GPU-Aware MPI support:

- **Repository**: [https://github.com/OSU-Nowlab/pytorch](https://github.com/OSU-Nowlab/pytorch)
- **Official PyTorch Build Guide**: [https://github.com/pytorch/pytorch#installation](https://github.com/pytorch/pytorch#installation)

To enable GPU-Aware MPI when building, add `USE_CUDA_MPI=1` to your setup command:

```bash
USE_CUDA_MPI=1 USE_MPI=1 python setup.py install
```

## Quick Start

### Basic DDP Script

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize MPI backend
dist.init_process_group(backend='mpi')

# Get rank and device
rank = dist.get_rank()
device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
torch.cuda.set_device(device)

# Create model and wrap with DDP
model = YourModel().to(device)
model = DDP(model)

# Training loop
for batch in dataloader:
    outputs = model(batch.to(device))
    loss = criterion(outputs, labels.to(device))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Cleanup
dist.destroy_process_group()
```

### Launch with mpirun

```bash
# Single node, 4 GPUs
mpirun -np 4 python train.py

# Multi-node (2 nodes, 4 GPUs each)
mpirun -np 8 -ppn 4 -hostfile hosts.txt python train.py
```

### Launch with srun (SLURM)

```bash
# Single node
srun -n 4 --gpus-per-task=1 python train.py

# Multi-node
srun -N 2 -n 8 --gpus-per-node=4 python train.py
```

## Initializing the MPI Backend

### Standard Initialization

```python
import torch.distributed as dist

# MPI backend handles rank/world_size automatically from mpirun/srun
dist.init_process_group(backend='mpi')

# Get process information
rank = dist.get_rank()
world_size = dist.get_world_size()
```

### With Explicit Configuration

```python
import os
import torch.distributed as dist

# MPI provides these automatically, but you can access them
rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', 0))
world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', 1))
local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', 0))

dist.init_process_group(backend='mpi')
```

### Verifying MPI Availability

```python
import torch.distributed as dist

# Check if MPI backend is available
if dist.is_mpi_available():
    print("MPI backend is available")
else:
    print("MPI backend not available - was PyTorch built with USE_MPI=1?")
```

## Running DDP Training

### Single Node Multi-GPU

```bash
# Launch with mpirun
mpirun -np 4 python train.py

# Or with srun
srun -n 4 --gpus-per-task=1 python train.py
```

### Multi-Node Training

Create a hostfile listing your nodes:

```text
# hosts.txt
node1 slots=4
node2 slots=4
```

Launch:

```bash
# With mpirun
mpirun -np 8 -ppn 4 -hostfile hosts.txt python train.py

# With srun (SLURM handles node allocation)
srun -N 2 -n 8 --ntasks-per-node=4 python train.py
```

### SLURM Batch Script Example

```bash
#!/bin/bash
#SBATCH -N 2                      # Number of nodes
#SBATCH -n 8                      # Total tasks (GPUs)
#SBATCH --ntasks-per-node=4       # GPUs per node
#SBATCH --gpus-per-node=4         # Request GPUs
#SBATCH -t 4:00:00                # Time limit
#SBATCH -p gpu                    # GPU partition

# Load environment
source ~/setup_hpc_ai.sh

# Run training
srun python train.py
```

## MVAPICH-Plus Tuning

MVAPICH-Plus provides environment variables (CVARs) to optimize collective operations.

### General Recommendations

```bash
# Enable GPU-aware MPI
export MPIR_CVAR_ENABLE_GPU=1
```

### NVIDIA TACC-Vista InfiniBand Networks (NDR/HDR)

```bash
# Note that the osu_gpu_direct algorithm is optimized for Vista's 1GPU per node setup
export MPIR_CVAR_ENABLE_GPU=1
export MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM=osu_gpu_direct
export MPIR_CVAR_ALLREDUCE_THROTTLE=4
export MPIR_CVAR_ALLREDUCE_COMPOSITION=2
export MPIR_CVAR_ALLREDUCE_IPC_MSG_SIZE_THRESHOLD=65536
```

### AMD Slingshot Networks (Frontier, Cosmos)

```bash
export MPIR_CVAR_ENABLE_GPU=1
export MPIR_CVAR_PMI_VERSION=2
export MPIR_CVAR_CH4_OFI_ENABLE_HMEM=1
export MPIR_CVAR_CH4_OFI_ENABLE_MR_HMEM=1
export MPIR_CVAR_ALLREDUCE_COMPOSITION=8
export MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM=osu_gpu_rsa
```

### Setting CVARs in Job Scripts

With mpirun:
```bash
mpirun -np 4 \
    -genv MPIR_CVAR_ENABLE_GPU=1 \
    -genv MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM=osu_gpu_direct \
    python train.py
```

With srun:
```bash
srun -n 4 --export=ALL,MPIR_CVAR_ENABLE_GPU=1,MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM=osu_gpu_direct \
    python train.py
```

## Common Patterns

### Gradient Accumulation

```python
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    outputs = model(batch)
    loss = criterion(outputs, labels) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    with autocast():
        outputs = model(batch)
        loss = criterion(outputs, labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Checkpointing

```python
# Save checkpoint (only on rank 0)
if dist.get_rank() == 0:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),  # Note: .module for DDP
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, 'checkpoint.pt')

# Synchronize before continuing
dist.barrier()
```

### Data Loading

```python
from torch.utils.data import DataLoader, DistributedSampler

# Create distributed sampler
sampler = DistributedSampler(
    dataset,
    num_replicas=dist.get_world_size(),
    rank=dist.get_rank(),
    shuffle=True
)

# Create dataloader with sampler
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=4,
    pin_memory=True
)

# Update sampler epoch for proper shuffling
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)
    for batch in dataloader:
        # training loop
```

## Troubleshooting

### MPI Backend Not Available

**Error**: `RuntimeError: MPI backend not available`

**Cause**: PyTorch was not built with MPI support.

**Solution**: Verify build flags and rebuild if needed:
```bash
python -c "import torch.distributed as dist; print(dist.is_mpi_available())"
# Should print: True
```

### CUDA Device Mismatch

**Error**: `RuntimeError: Expected all tensors to be on the same device`

**Cause**: Model and data on different GPUs.

**Solution**: Set device based on local rank:
```python
local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', 0))
device = torch.device(f'cuda:{local_rank}')
torch.cuda.set_device(device)
```

### Hanging on init_process_group

**Cause**: Network connectivity issues or mismatched world sizes.

**Solution**:
1. Verify all nodes can communicate
2. Check hostfile matches SLURM allocation
3. Ensure MPI environment is consistent across nodes

### Out of Memory

**Cause**: Batch size too large for GPU memory.

**Solution**:
1. Reduce batch size per GPU
2. Use gradient accumulation
3. Enable mixed precision training

## Additional Resources

- [HPC-AI PyTorch DDP Userguide](https://hpc-ai.engineering.osu.edu/userguide/pytorch-ddp/) - Official guide with system-specific examples
- [MVAPICH-Plus Documentation](https://mvapich-docs.readthedocs.io/en/mvapich-plus/)
- [MVAPICH-Plus Downloads](https://mvapich.cse.ohio-state.edu/downloads/)
- [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [HPC-AI Project Homepage](https://hpc-ai.engineering.osu.edu/)
- [OSU-Nowlab PyTorch Fork](https://github.com/OSU-Nowlab/pytorch)
