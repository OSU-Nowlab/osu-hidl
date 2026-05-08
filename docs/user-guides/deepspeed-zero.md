# DeepSpeed ZeRO Optimization

This guide covers using DeepSpeed ZeRO (Zero Redundancy Optimizer) for memory-efficient distributed training with the MVAPICH-Plus MPI backend on HPC clusters.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Choosing a ZeRO Stage](#choosing-a-zero-stage)
- [Configuration Reference](#configuration-reference)
- [Activation Checkpointing](#activation-checkpointing)
- [CPU Offloading](#cpu-offloading)
- [Running on SLURM](#running-on-slurm)
- [Common Patterns](#common-patterns)
- [Troubleshooting](#troubleshooting)
- [Additional Resources](#additional-resources)

## Overview

DeepSpeed ZeRO eliminates memory redundancy in data-parallel training by partitioning optimizer states, gradients, and parameters across GPUs. This allows training models that would otherwise not fit in GPU memory.

**ZeRO provides**:
- **ZeRO-1**: Optimizer state partitioning — 4x memory reduction
- **ZeRO-2**: Optimizer state + gradient partitioning — 8x memory reduction
- **ZeRO-3**: Optimizer state + gradient + parameter partitioning — linear memory scaling with GPU count

> **Official Documentation**: For cluster-specific build scripts and configurations, see the official HPC-AI userguide:
> [https://hpc-ai.engineering.osu.edu/userguide](https://hpc-ai.engineering.osu.edu/userguide)

## Prerequisites

### DeepSpeed with MVAPICH-Plus

This stack provides DeepSpeed built against our custom PyTorch 2.10.0 with MVAPICH-Plus MPI support.

- **Build Guide**: [docs/build-guides/deepspeed.md](../build-guides/deepspeed.md)
- **Repository**: [https://github.com/OSU-Nowlab/DeepSpeed](https://github.com/OSU-Nowlab/DeepSpeed)

### Verify Installation

```bash
python -c "import deepspeed; print('DeepSpeed:', deepspeed.__version__)"
ds_report
python -c "import torch.distributed as dist; print('MPI available:', dist.is_mpi_available())"
```

All three checks must pass before proceeding.

## Quick Start

### Basic Training Script

```python
import torch
import deepspeed

def create_model_and_optimizer():
    model = YourModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer

model, optimizer = create_model_and_optimizer()

# Initialize DeepSpeed engine
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config="ds_config_zero2.json"
)

# Training loop
for batch in dataloader:
    outputs = model_engine(batch["input"].to(model_engine.device))
    loss = criterion(outputs, batch["labels"].to(model_engine.device))
    model_engine.backward(loss)
    model_engine.step()
```

### Launch

```bash
# Single node, 2 GPUs
deepspeed --num_gpus=2 train.py --deepspeed_config=ds_config_zero2.json

# Multi-node via mpirun (2 nodes, 4 GPUs each)
mpirun -np 8 -ppn 4 \
    -x MASTER_ADDR=$(hostname) -x MASTER_PORT=29500 \
    python train.py --deepspeed_config=ds_config_zero2.json
```

## Choosing a ZeRO Stage

| Scenario | Recommended Stage | Why |
|----------|-------------------|-----|
| Model fits comfortably in GPU memory | ZeRO-2 | Best throughput, ~8x memory savings |
| Model barely fits or OOM with ZeRO-2 | ZeRO-3 | Partitions parameters, linear scaling |
| Need large batch size on limited GPU count | ZeRO-2 + gradient accumulation | Maintain effective batch without extra GPUs |
| Very large model (70B+) on few nodes | ZeRO-3 + CPU offloading | Maximum memory capacity at throughput cost |

**Practical rule**: Start with ZeRO-2. Only move to ZeRO-3 if you encounter OOM.

ZeRO-3 adds communication overhead for parameter gathering during forward/backward passes. For most training scenarios where the model fits in memory with ZeRO-2, ZeRO-3 will be slower without providing benefit.

## Configuration Reference

### ZeRO-2 (`ds_config_zero2.json`)

```json
{
  "train_batch_size": 64,
  "train_micro_batch_size_per_gpu": 32,
  "gradient_accumulation_steps": 1,
  "steps_per_print": 10,
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "zero_optimization": {
    "stage": 2,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 2e8,
    "allgather_bucket_size": 2e8,
    "reduce_scatter": true,
    "allgather_partitions": true
  },
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.001,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.0
    }
  },
  "gradient_clipping": 1.0
}
```

### ZeRO-3 (`ds_config_zero3.json`)

```json
{
  "train_batch_size": 128,
  "train_micro_batch_size_per_gpu": 32,
  "gradient_accumulation_steps": 1,
  "fp16": {"enabled": true},
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "gradient_clipping": 1.0
}
```

### Key Parameters

**Batch size constraint**: `train_batch_size` must equal `train_micro_batch_size_per_gpu × gradient_accumulation_steps × num_gpus`.

```bash
# Example: 4 GPUs, micro_batch=32, grad_accum=2
# train_batch_size = 32 * 2 * 4 = 256
```

**Bucket sizes**: Larger buckets reduce the number of collective operations but use more memory.

```json
"reduce_bucket_size": 5e8,      // Increase for fewer, larger AllReduce calls
"allgather_bucket_size": 5e8    // Increase for fewer, larger AllGather calls
```

**Communication overlap**: Always keep `overlap_comm: true` — this pipelines communication with computation and is critical for performance on InfiniBand/Slingshot networks.

## Activation Checkpointing

Activation checkpointing trades recomputation for memory by not storing intermediate activations during the forward pass. Use this when GPU memory is the bottleneck.

### Enable in Config

```json
{
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": false,
    "contiguous_memory_optimization": true,
    "number_checkpoints": null,
    "synchronize_checkpoint_boundary": false,
    "profile": false
  }
}
```

### Enable in Model Code

```python
import deepspeed
from deepspeed.runtime.activation_checkpointing import checkpointing

# Wrap transformer layers for checkpointing
deepspeed.checkpointing.configure(
    mpu=None,
    partition_activations=True,
    contiguous_checkpointing=True
)

# Use in forward pass (replaces torch.utils.checkpoint.checkpoint)
output = deepspeed.checkpointing.checkpoint(layer, input)
```

**Memory impact**: Reduces activation memory by the number of checkpointed layers. Increases compute by ~33% (one extra forward pass per checkpointed segment).

## CPU Offloading

CPU offloading moves optimizer states and/or parameters from GPU to CPU memory, enabling training of very large models at the cost of PCIe bandwidth.

> **Note**: CPU offloading requires ZeRO-3. It is not available in ZeRO-2.

### Optimizer Offload

Offloads optimizer states (Adam moments) to CPU. Useful when optimizer states are the memory bottleneck.

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    }
  }
}
```

### Parameter Offload

Offloads model parameters to CPU. Use only when GPU memory is severely constrained, as this incurs significant PCIe transfer overhead.

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    }
  }
}
```

`pin_memory: true` uses pinned (page-locked) memory for faster CPU-GPU transfers. Enable this when the host has sufficient RAM.

### NVMe Offload

For extremely large models, parameters can be offloaded to NVMe storage:

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_param": {
      "device": "nvme",
      "nvme_path": "/local/scratch"
    }
  }
}
```

Use a local NVMe path (not network storage) for acceptable bandwidth.

## Running on SLURM

### Interactive Session

```bash
# Allocate nodes with GPUs
salloc -p devel -t 3:00:00 -C cuda -N 2 --gpus-per-node=4

# Set up environment (MRI cluster)
module reset
module load gcc/13.3.0
module load cuda/12.6
export MVAPICH_HOME="${HOME}/osu-hpc-ai-dev/install/mvapich-plus-4.1-cuda12.6"
export PATH="${MVAPICH_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MVAPICH_HOME}/lib:${LD_LIBRARY_PATH}"
export LD_PRELOAD=/opt/gcc/13.3.0/lib64/libstdc++.so.6:$LD_PRELOAD
source ${HOME}/miniconda3/bin/activate hpc-ai-build

# Launch via deepspeed launcher (single node)
deepspeed --num_gpus=4 train.py --deepspeed_config=ds_config_zero2.json

# Launch via mpirun (multi-node)
mpirun -np 8 -ppn 4 \
    -x MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1) \
    -x MASTER_PORT=29500 \
    -x LD_PRELOAD -x LD_LIBRARY_PATH \
    python train.py --deepspeed_config=ds_config_zero2.json
```

### Batch Job

```bash
#!/bin/bash
#SBATCH -J deepspeed-zero
#SBATCH -N 2
#SBATCH --gpus-per-node=4
#SBATCH -t 4:00:00
#SBATCH -p gpu
#SBATCH -o slurm-%j.out

module reset
module load gcc/13.3.0
module load cuda/12.6

export MVAPICH_HOME="${HOME}/osu-hpc-ai-dev/install/mvapich-plus-4.1-cuda12.6"
export PATH="${MVAPICH_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MVAPICH_HOME}/lib:${LD_LIBRARY_PATH}"
export LD_PRELOAD=/opt/gcc/13.3.0/lib64/libstdc++.so.6:$LD_PRELOAD
export MV2_USE_CUDA=1
export MV2_USE_GPUDIRECT_RDMA=1

source ${HOME}/miniconda3/bin/activate hpc-ai-build

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)

mpirun -np 8 -ppn 4 \
    -x MASTER_ADDR=$MASTER_ADDR \
    -x MASTER_PORT=29500 \
    -x LD_PRELOAD -x LD_LIBRARY_PATH -x PATH \
    python train.py --deepspeed_config=ds_config_zero2.json
```

```bash
sbatch deepspeed_job.slurm
```

## Common Patterns

### Gradient Accumulation

Simulate a larger batch size without increasing GPU memory:

```json
{
  "train_batch_size": 256,
  "train_micro_batch_size_per_gpu": 32,
  "gradient_accumulation_steps": 4
}
```

`deepspeed.initialize` handles the accumulation steps automatically — `model_engine.step()` only updates weights every `gradient_accumulation_steps` micro-batches.

### Mixed Precision with BF16

BF16 avoids the dynamic loss scaling required by FP16 and is preferred on Ampere GPUs (A100) and later:

```json
{
  "bf16": {
    "enabled": true
  },
  "fp16": {
    "enabled": false
  }
}
```

### Checkpointing

```python
# Save checkpoint
model_engine.save_checkpoint("./checkpoints", tag="epoch_5")

# Load checkpoint
model_engine.load_checkpoint("./checkpoints", tag="epoch_5")
```

ZeRO-3 checkpoints are sharded across ranks. Use `stage3_gather_16bit_weights_on_model_save: true` in your config to gather weights to rank 0 for a single-file checkpoint.

### Monitoring with `wall_clock_breakdown`

```json
{
  "steps_per_print": 10,
  "wall_clock_breakdown": true
}
```

This prints per-step timing for forward, backward, optimizer, and communication phases — useful for identifying bottlenecks.

## Troubleshooting

### Batch size assertion error

**Error**: `train_batch_size is not equal to micro_batch_per_gpu * gradient_acc_step * world_size`

**Solution**: Recalculate `train_batch_size` for your GPU count:

```python
train_batch_size = train_micro_batch_size_per_gpu * gradient_accumulation_steps * num_gpus
```

### FP16 dtype mismatch

**Error**: `RuntimeError: mat1 and mat2 must have the same dtype`

**Cause**: Input tensors not cast to FP16 when `fp16.enabled: true`.

**Solution**:

```python
if model_engine.fp16_enabled():
    batch = {k: v.half() if v.is_floating_point() else v for k, v in batch.items()}
```

### CUDA out of memory

Try these in order:
1. Reduce `train_micro_batch_size_per_gpu`
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Switch from ZeRO-2 to ZeRO-3
4. Enable activation checkpointing
5. Enable CPU offloading

### Slow throughput

1. Verify `overlap_comm: true` in config
2. Enable GPUDirect RDMA:
   ```bash
   export MV2_USE_CUDA=1
   export MV2_USE_GPUDIRECT_RDMA=1
   ```
3. Increase bucket sizes to reduce collective call frequency
4. Check network with `osu_bw` from MVAPICH-Plus libexec

### libstdc++ version error

```bash
export LD_PRELOAD=/opt/gcc/13.3.0/lib64/libstdc++.so.6:$LD_PRELOAD
```

## Additional Resources

- **DeepSpeed Documentation**: [https://www.deepspeed.ai/](https://www.deepspeed.ai/)
- **ZeRO Paper**: [https://arxiv.org/abs/1910.02054](https://arxiv.org/abs/1910.02054)
- **MVAPICH-Plus User Guide**: [https://mvapich-docs.readthedocs.io/en/mvapich-plus/](https://mvapich-docs.readthedocs.io/en/mvapich-plus/)
- **Official HPC-AI Userguide**: [https://hpc-ai.engineering.osu.edu/userguide/](https://hpc-ai.engineering.osu.edu/userguide/)
- **DeepSpeed Build Guide**: [../build-guides/deepspeed.md](../build-guides/deepspeed.md)
- **Examples**: [../../examples/deepspeed/](../../examples/deepspeed/)
