# MCR-DL Backend Selection and Configuration

This guide covers using MCR-DL as the communication runtime for distributed deep learning, including when to use it versus NCCL, how to configure backends, and how to run communication benchmarks.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [MCR-DL vs NCCL](#mcr-dl-vs-nccl)
- [Quick Start](#quick-start)
- [Backend Configuration](#backend-configuration)
- [Integration with DeepSpeed](#integration-with-deepspeed)
- [Running Communication Benchmarks](#running-communication-benchmarks)
- [Tuning MVAPICH-Plus CVARs](#tuning-mvapich-plus-cvars)
- [Troubleshooting](#troubleshooting)
- [Additional Resources](#additional-resources)

## Overview

MCR-DL (Modular Communication Runtime for Deep Learning) is a communication library from the NOWLAB that provides efficient collective operations for distributed training. It integrates with PyTorch's `torch.distributed` API as a drop-in MPI backend, delivering GPU-aware collectives through MVAPICH-Plus.

**Key capabilities**:
- **MPI-based collectives** via MVAPICH-Plus (AllReduce, AllGather, Scatter, Broadcast, Barrier)
- **CUDA-aware communication**: GPU-to-GPU transfers via GPUDirect RDMA without CPU staging
- **Modular design**: Swap communication backends without changing training code
- **Optimized for HPC interconnects**: InfiniBand and Slingshot (HPE Cassini)
- **Vendor-neutral**: NVIDIA and AMD GPUs

> **Official Documentation**: For cluster-specific configuration, see the official HPC-AI userguide:
> [https://hpc-ai.engineering.osu.edu/userguide](https://hpc-ai.engineering.osu.edu/userguide)

## Prerequisites

### MCR-DL with MVAPICH-Plus

MCR-DL requires PyTorch built with MVAPICH-Plus MPI support.

- **MCR-DL Build Guide**: [docs/build-guides/mcr-dl.md](../build-guides/mcr-dl.md)
- **PyTorch Build Guide**: [docs/build-guides/pytorch.md](../build-guides/pytorch.md)
- **Repository**: [https://github.com/OSU-Nowlab/MCR-DL](https://github.com/OSU-Nowlab/MCR-DL)

### Verify Installation

```bash
python -c "import mcr_dl; print('MCR-DL installed')"
python -c "import torch.distributed as dist; print('MPI available:', dist.is_mpi_available())"
mpirun --version
```

## MCR-DL vs NCCL

Both MCR-DL (via MPI) and NCCL are available backends for GPU collectives. Choose based on your hardware and network:

| Factor | MCR-DL (MPI) | NCCL |
|--------|--------------|------|
| **Network** | InfiniBand, Slingshot | InfiniBand, Ethernet, NVLink |
| **GPU vendor** | NVIDIA, AMD | NVIDIA only |
| **Multi-node** | Native — designed for HPC clusters | Supported but less HPC-optimized |
| **SLURM integration** | Native | Via `nccl-rdma-sharp` or plugins |
| **Communication control** | Fine-grained CVAR tuning | Limited tuning knobs |
| **RDMA support** | GPUDirect RDMA via MVAPICH-Plus | GPUDirect RDMA (NVIDIA only) |
| **AMD support** | Yes (via ROCm-aware MVAPICH) | No |

**Use MCR-DL when**:
- Running on HPC clusters with InfiniBand or Slingshot networks
- Using AMD GPUs (MI250X, MI310X)
- You need fine-grained control over communication parameters
- Running across many nodes where MPI process management is already in use

**Use NCCL when**:
- Running on a single node or small cluster with NVLink
- Using NVIDIA GPUs on Ethernet or simple IB configurations
- Integration with frameworks that default to NCCL (e.g., stock PyTorch DDP)

**On TACC Vista and OLCF Frontier**, MCR-DL with MVAPICH-Plus over Slingshot is the recommended backend. On MRI with A100s and InfiniBand, both work but MCR-DL provides more tuning options.

## Quick Start

### Initialize MCR-DL Backend

```python
import torch
import torch.distributed as dist
import mcr_dl

# Initialize using MPI backend (powered by MCR-DL + MVAPICH-Plus)
dist.init_process_group(backend='mpi')

rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
torch.cuda.set_device(device)

# Standard collective operations — no code changes from NCCL
tensor = torch.ones(1024, device=device)
dist.all_reduce(tensor)
dist.barrier()

print(f"Rank {rank}/{world_size}: all_reduce result = {tensor[0].item()}")
dist.destroy_process_group()
```

```bash
# Launch with mpirun
mpirun -np 4 python train.py

# Launch with srun (SLURM)
srun -n 4 python train.py
```

### Verify Communication Works

```bash
mpirun -np 2 python -c "
import torch, torch.distributed as dist, mcr_dl

dist.init_process_group(backend='mpi')
rank = dist.get_rank()
t = torch.tensor([float(rank)], device=f'cuda:{rank}')
dist.all_reduce(t)
expected = sum(range(dist.get_world_size()))
assert t.item() == expected, f'Rank {rank}: expected {expected}, got {t.item()}'
print(f'Rank {rank}: AllReduce passed')
dist.destroy_process_group()
"
```

## Backend Configuration

### Environment Variables

Set these before launching your training script:

```bash
# Enable CUDA-aware MPI (required for GPU-to-GPU transfers)
export MV2_USE_CUDA=1

# Enable GPUDirect RDMA (InfiniBand only)
export MV2_USE_GPUDIRECT_RDMA=1
export MV2_CUDA_USE_NAIVE=0

# MVAPICH-Plus home
export MVAPICH_HOME="${HOME}/osu-hpc-ai-dev/install/mvapich-plus-4.1-cuda12.6"
export PATH="${MVAPICH_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MVAPICH_HOME}/lib:${LD_LIBRARY_PATH}"
```

### Complete Environment Setup (MRI Cluster)

```bash
module reset
module load gcc/13.3.0
module load cuda/12.6

export MVAPICH_HOME="${HOME}/osu-hpc-ai-dev/install/mvapich-plus-4.1-cuda12.6"
export PATH="${MVAPICH_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MVAPICH_HOME}/lib:${LD_LIBRARY_PATH}"
export CPATH="${MVAPICH_HOME}/include:${CPATH}"

export MV2_USE_CUDA=1
export MV2_USE_GPUDIRECT_RDMA=1
export MV2_CUDA_USE_NAIVE=0

export LD_PRELOAD=/opt/gcc/13.3.0/lib64/libstdc++.so.6:$LD_PRELOAD

conda activate hpc-ai-build
```

### NCCL Fallback

When running on a single node without InfiniBand (e.g., NVLink-only), NCCL may provide better bandwidth. You can use NCCL alongside MCR-DL by switching the process group backend:

```python
# Use NCCL for single-node, MPI/MCR-DL for multi-node
import os
backend = 'nccl' if os.environ.get('SLURM_NNODES', '1') == '1' else 'mpi'
dist.init_process_group(backend=backend)
```

## Integration with DeepSpeed

DeepSpeed uses `torch.distributed` internally. Initializing with the MPI backend automatically routes DeepSpeed's collectives through MCR-DL.

```python
import deepspeed
import torch.distributed as dist
import mcr_dl

# Initialize MPI backend before deepspeed.initialize
dist.init_process_group(backend='mpi')

model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config={
        "train_batch_size": 64,
        "train_micro_batch_size_per_gpu": 32,
        "zero_optimization": {"stage": 2},
        "communication_data_type": "fp16",
        "fp16": {"enabled": True}
    }
)
```

Launch via `mpirun` rather than the `deepspeed` CLI when using MCR-DL:

```bash
mpirun -np 8 -ppn 4 \
    -x MV2_USE_CUDA=1 \
    -x MV2_USE_GPUDIRECT_RDMA=1 \
    -x LD_LIBRARY_PATH \
    python train.py
```

## Running Communication Benchmarks

The `benchmarks/communication/` directory contains benchmarks measuring Python-layer collective latency and bandwidth. Use these to:
- Establish performance baselines on your cluster
- Detect regressions between configuration changes
- Compare MCR-DL vs NCCL performance

### Single Operation

```bash
cd ~/osu-hpc-ai/benchmarks/communication

# AllReduce at a single large message size
mpirun -np 4 python all_reduce.py

# Scan across message sizes (latency vs size curve)
mpirun -np 4 python all_reduce.py --scan

# Other operations
mpirun -np 4 python all_gather.py --scan
mpirun -np 4 python broadcast.py --scan
mpirun -np 4 python scatter.py --scan
mpirun -np 4 python barrier.py
```

### All Operations

```bash
# Run all benchmarks
mpirun -np 4 python run_all.py --scan

# Select specific operations
mpirun -np 4 python run_all.py --scan --all-reduce --broadcast --all-gather
```

### Benchmark Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--trials` | 100 | Number of timed iterations |
| `--warmups` | 50 | Warmup iterations (not timed) |
| `--maxsize` | 24 | Max message size as power of 2 (2^24 = 16M elements) |
| `--bw-unit` | Gbps | Bandwidth unit (`Gbps` or `GBps`) |
| `--backend` | mpi | Communication backend (`mpi` or `gloo`) |
| `--scan` | off | Scan all message sizes |
| `--dtype` | float32 | PyTorch tensor dtype |

### On SLURM

```bash
srun -n 4 python all_reduce.py --scan
```

### Expected Output

```
AllReduce | Latency (us) | Bandwidth (Gbps)
--------------------------------------------
      1024 |         25.3 |            0.32
      4096 |         26.1 |            1.26
     16384 |         28.4 |            4.63
     65536 |         35.2 |           14.91
    262144 |         57.8 |           36.28
   1048576 |        148.3 |           56.62
   4194304 |        523.1 |           64.11
  16777216 |       2012.4 |           66.74
```

Compare your results against theoretical InfiniBand peak (HDR: 200 Gbps) or Slingshot peak to assess efficiency.

## Tuning MVAPICH-Plus CVARs

MVAPICH-Plus exposes configuration variables (CVARs) via environment variables. Key tuning knobs:

### GPUDirect RDMA

```bash
export MV2_USE_CUDA=1                    # Enable CUDA-aware MPI
export MV2_USE_GPUDIRECT_RDMA=1         # Enable GPUDirect RDMA transfers
export MV2_CUDA_USE_NAIVE=0             # Disable naive (CPU-copy) fallback
```

### AllReduce Algorithm

```bash
# Force ring allreduce (good for large messages)
export MV2_ALLREDUCE_MV2_RING=1

# Use recursive halving for small messages
export MV2_ALLREDUCE_SMALL_MSG=1
```

### Message Threshold

```bash
# Tune the large-message threshold (bytes)
export MV2_ALLREDUCE_THRESHOLD=262144   # Default: 256KB
```

### InfiniBand-Specific

```bash
export MV2_IB_HCA=mlx5_0               # Specify HCA device
export MV2_NUM_HCAS=1                   # Number of HCAs to use
```

### Slingshot-Specific (Vista, Frontier)

```bash
export MV2_SLINGSHOT_EAGER_THRESHOLD=65536  # Eager protocol threshold
export FI_CXI_RX_MATCH_MODE=hybrid          # Slingshot match mode
```

## Troubleshooting

### `ImportError: No module named 'mcr_dl'`

```bash
cd ~/osu-hpc-ai/frameworks/mcr-dl
pip install -e .
```

If the submodule is not initialized:

```bash
git submodule update --init frameworks/mcr-dl
```

### MPI not available in PyTorch

```bash
python -c "import torch.distributed as dist; print(dist.is_mpi_available())"
```

If `False`, rebuild PyTorch with `USE_MPI=1 USE_CUDA_MPI=1`. See [docs/build-guides/pytorch.md](../build-guides/pytorch.md).

### CUDA device assignment error

Each MPI rank needs its own GPU. Ensure one rank per GPU:

```bash
# Let MVAPICH-Plus handle binding
mpirun -np 4 --bind-to none python train.py
```

Or set `CUDA_VISIBLE_DEVICES` per rank:

```python
import os
rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))
os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
```

### Low benchmark bandwidth

1. Confirm `MV2_USE_GPUDIRECT_RDMA=1` is set
2. Verify you are on a GPU node (not login node)
3. Check that `nvidia-smi` shows the correct GPU count
4. Run `mpirun -np 2 $MVAPICH_HOME/libexec/mvapich-plus/osu_bw` to measure raw MPI bandwidth (C level) for comparison

### Rank communication hangs

Ensure all ranks call the same collective in the same order. A deadlock typically means one rank skipped a collective (e.g., due to a conditional on rank 0 only):

```python
# Wrong: only rank 0 calls all_reduce
if rank == 0:
    dist.all_reduce(tensor)

# Correct: all ranks call all_reduce
dist.all_reduce(tensor)
```

## Additional Resources

- **MCR-DL Repository**: [https://github.com/OSU-Nowlab/MCR-DL](https://github.com/OSU-Nowlab/MCR-DL)
- **MVAPICH-Plus User Guide**: [https://mvapich-docs.readthedocs.io/en/mvapich-plus/](https://mvapich-docs.readthedocs.io/en/mvapich-plus/)
- **OSU Micro-Benchmarks**: [https://mvapich.cse.ohio-state.edu/benchmarks/](https://mvapich.cse.ohio-state.edu/benchmarks/)
- **Official HPC-AI Userguide**: [https://hpc-ai.engineering.osu.edu/userguide/](https://hpc-ai.engineering.osu.edu/userguide/)
- **MCR-DL Build Guide**: [../build-guides/mcr-dl.md](../build-guides/mcr-dl.md)
- **PyTorch DDP Guide**: [pytorch-ddp.md](pytorch-ddp.md)
- **Communication Benchmarks**: [../../benchmarks/communication/](../../benchmarks/communication/)
