# Building MCR-DL with MVAPICH-Plus

This guide explains how to build MCR-DL with MVAPICH-Plus support for optimized distributed communication on HPC systems.

## Overview

[MCR-DL](https://github.com/OSU-Nowlab/MCR-DL) (Modular Communication Runtime for Deep Learning) is a NOWLAB communication library that provides efficient collective operations for distributed deep learning. It integrates directly with PyTorch and MVAPICH-Plus to deliver high-performance all-reduce, all-gather, and point-to-point communication primitives.

Key features:

- **MPI-based collectives** via MVAPICH-Plus for CUDA-aware GPU communication
- **PyTorch integration** through a drop-in `torch.distributed` backend
- **Modular design** — swap communication backends without changing training code
- **GPUDirect RDMA** support for low-latency GPU-to-GPU transfers
- **Optimized for InfiniBand and Slingshot** interconnects

## Prerequisites

### System Requirements

- **GPU**: NVIDIA GPUs with Compute Capability ≥ 6.0 (Pascal or newer)
- **CUDA**: Version 11.0+ (12.x recommended)
- **GCC**: Version 9.0+ (13.x recommended)
- **Python**: Version 3.8-3.12

### Required Dependencies

1. **PyTorch with MPI support**
   - Must be built with MVAPICH-Plus integration
   - See [PyTorch Build Guide](pytorch.md)

2. **MVAPICH-Plus**
   - CUDA-aware MPI library, Version 4.1+ recommended
   - Download: http://mvapich.cse.ohio-state.edu/

3. **CUDA Toolkit**
   - Includes nvcc compiler
   - cuDNN 8.0+ required

## Installation

### Option 1: Build from Source (Recommended)

#### Step 1: Set Up Environment

```bash
# Load CUDA and GCC (on MRI, use module system)
export CUDA_HOME=/path/to/cuda
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Set up MVAPICH-Plus
export MVAPICH_HOME=/path/to/mvapich-plus
export PATH="${MVAPICH_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MVAPICH_HOME}/lib:${LD_LIBRARY_PATH}"
export CPATH="${MVAPICH_HOME}/include:${CPATH}"

export CC=$(which mpicc)
export CXX=$(which mpicxx)
```

#### Step 2: Activate Python Environment

```bash
# Using the shared hpc-ai-build conda environment
conda activate hpc-ai-build

# Or create a dedicated environment
conda create -n mcr-dl python=3.12
conda activate mcr-dl
```

#### Step 3: Verify PyTorch Installation

```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch.distributed as dist; print(f'MPI available: {dist.is_mpi_available()}')"
```

If MPI is unavailable, rebuild PyTorch with MVAPICH-Plus support first (see [pytorch.md](pytorch.md)).

#### Step 4: Build and Install MCR-DL

```bash
# From the repo root (submodule already initialized)
cd frameworks/mcr-dl

pip install -e .
```

#### Step 5: Verify Installation

```bash
python -c "import mcr_dl; print('MCR-DL installed successfully')"
```

### Option 2: Quick Test with MPI

```bash
# Single-node test (4 GPUs)
mpirun -np 4 python -c "
import torch
import mcr_dl
import torch.distributed as dist

dist.init_process_group(backend='mpi')
rank = dist.get_rank()
tensor = torch.ones(1).cuda(rank)
dist.all_reduce(tensor)
print(f'Rank {rank}: all_reduce result = {tensor.item()}')
"
```

## Usage with PyTorch

MCR-DL provides a communication backend compatible with `torch.distributed`. Replace the standard backend initialization with MCR-DL:

```python
import torch
import torch.distributed as dist
import mcr_dl

# Initialize with MCR-DL backend
dist.init_process_group(backend='mpi')

# Use standard torch.distributed API — no other code changes needed
dist.all_reduce(tensor)
dist.all_gather(tensor_list, tensor)
```

### Integration with DeepSpeed

MCR-DL can be used alongside DeepSpeed for communication-optimized ZeRO training:

```python
import deepspeed

ds_config = {
    "zero_optimization": {"stage": 2},
    "communication_data_type": "fp16",
}

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config,
)
```

## Troubleshooting

### `ImportError: No module named 'mcr_dl'`

The package was not installed. From the repo root:

```bash
cd frameworks/mcr-dl && pip install -e .
```

If the submodule is not initialized:

```bash
git submodule update --init frameworks/mcr-dl
```

### MPI initialization errors

Ensure MVAPICH-Plus is loaded and PyTorch was built with MPI support:

```bash
python -c "import torch.distributed as dist; assert dist.is_mpi_available()"
```

### CUDA device errors in multi-GPU runs

Each MPI rank must be assigned to its own GPU. Set `CUDA_VISIBLE_DEVICES` or use:

```bash
mpirun -np 4 --bind-to none python train.py
```

## Further Reading

- [MCR-DL GitHub](https://github.com/OSU-Nowlab/MCR-DL)
- [PyTorch Build Guide](pytorch.md)
- [PyTorch DDP with MPI](../user-guides/pytorch-ddp.md)
- [DeepSpeed Build Guide](deepspeed.md)
