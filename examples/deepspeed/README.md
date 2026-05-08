# DeepSpeed Examples

Comprehensive examples demonstrating DeepSpeed ZeRO optimization with MVAPICH-Plus MPI backend on HPC systems.

## Table of Contents

- [Prerequisites](#prerequisites)
- [GPU Allocation](#gpu-allocation)
- [Quick Start](#quick-start)
- [Examples](#examples)
- [Configuration Guide](#configuration-guide)
- [Testing & Validation](#testing--validation)
- [Troubleshooting](#troubleshooting)
- [Performance Tips](#performance-tips)

---

## Prerequisites

### 1. Verify Installation

Before running examples, ensure all components are properly installed:

```bash
# Check CUDA
nvidia-smi

# Check PyTorch with CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Check MPI backend
python -c "import torch.distributed as dist; print(f'MPI available: {dist.is_mpi_available()}')"

# Check DeepSpeed
python -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')"
ds_report
```

**All checks must pass** before proceeding. If any fail, see [../../docs/build-guides/](../../docs/build-guides/) for build instructions.

### 2. Set Up Environment

#### On MRI Cluster (OSU):

```bash
# Load required modules
module reset
module load gcc/13.3.0
module load cuda/12.6

# Set MVAPICH-Plus paths
export MVAPICH_HOME="${HOME}/osu-hpc-ai-dev/install/mvapich-plus-4.1-cuda12.6"
export PATH="${MVAPICH_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MVAPICH_HOME}/lib:${LD_LIBRARY_PATH}"
export CPATH="${MVAPICH_HOME}/include:${CPATH}"

# Fix libstdc++ compatibility
export LD_PRELOAD=/opt/gcc/13.3.0/lib64/libstdc++.so.6:$LD_PRELOAD

# Activate conda environment
source ${HOME}/miniconda3/bin/activate
conda activate hpc-ai-build
```

#### On Generic Systems:

```bash
# Load system modules (adjust for your system)
module load gcc cuda python

# Set MPI paths
export PATH="/path/to/mvapich/bin:${PATH}"
export LD_LIBRARY_PATH="/path/to/mvapich/lib:${LD_LIBRARY_PATH}"

# Activate virtual environment
source /path/to/venv/bin/activate
```

---

## GPU Allocation

### Interactive Session on MRI

```bash
# Single GPU (for quick testing)
salloc -p devel -t 3:00:00 -C cuda -N 1 --gpus-per-node=1

# Multiple GPUs (for distributed training)
salloc -p devel -t 3:00:00 -C cuda -N 1 --gpus-per-node=2

# After allocation completes, navigate to examples
cd /home/$USER/osu-hpc-ai/examples/deepspeed
```

### Batch Job on SLURM Systems

Create `run_example.slurm`:

```bash
#!/bin/bash
#SBATCH -J deepspeed-test
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -t 1:00:00
#SBATCH -p gpu

# Load environment
module load gcc/13.3.0 cuda/12.6
source ~/miniconda3/bin/activate hpc-ai-build

# Navigate and run
cd ~/osu-hpc-ai/examples/deepspeed
deepspeed --num_gpus=4 simple_zero_training.py --deepspeed_config=ds_config_zero2.json
```

Submit with:
```bash
sbatch run_example.slurm
```

---

## Quick Start

### Test 1: Single GPU Basic Training

**Purpose**: Verify DeepSpeed installation and basic functionality.

```bash
# Navigate to examples
cd ~/osu-hpc-ai/examples/deepspeed

# Run with 1 GPU
deepspeed --num_gpus=1 simple_zero_training.py --deepspeed_config=ds_config_zero2.json
```

**Expected output**:
```
DeepSpeed version: 0.10.2+502c2e46
PyTorch version: 2.10.0
CUDA available: True
World size: 1

Initializing DeepSpeed engine...
[Rank 0] DeepSpeed info: version=0.10.2+502c2e46
[Rank 0] Creating torch.float16 ZeRO stage 2 optimizer
...
Epoch 1/5 - Average Loss: 2.3136
Epoch 2/5 - Average Loss: 2.3062
...
Training complete!
Checkpoint saved to ./checkpoints/
```

**Expected time**: ~2-3 minutes
**Memory usage**: ~0.9 GB GPU memory

### Test 2: Multi-GPU Distributed Training

**Purpose**: Verify ZeRO optimization and multi-GPU communication.

```bash
# Run with 2 GPUs (adjust train_batch_size if needed)
deepspeed --num_gpus=2 simple_zero_training.py --deepspeed_config=ds_config_zero2.json
```

**Expected output**: Similar to single GPU but with:
- `World size: 2`
- Faster training (higher samples/sec)
- Better GPU utilization

### Test 3: ZeRO Stage 3 (Maximum Memory Efficiency)

**Purpose**: Test parameter partitioning for large models.

```bash
deepspeed --num_gpus=2 simple_zero_training.py --deepspeed_config=ds_config_zero3.json
```

**Expected output**:
- Lower per-GPU memory usage
- Slightly slower due to more communication
- `ZeRO stage 3 optimizer` in logs

---

## Examples

### `simple_zero_training.py`

**Description**: Basic distributed training example demonstrating DeepSpeed ZeRO optimization.

**Features**:
- Simple 4-layer MLP (10M parameters)
- Synthetic dataset generation
- ZeRO Stage 2/3 optimization
- FP16 mixed precision training
- Checkpoint saving
- Distributed training with MPI backend

**Model Architecture**:
```
Input (1000) → FC1 (2000) → ReLU →
FC2 (2000) → ReLU →
FC3 (2000) → ReLU →
Output (10)
```

**Training Configuration**:
- 5 epochs, 100 batches per epoch
- Batch size: 32 per GPU
- Optimizer: FusedAdam (custom CUDA kernel)
- Learning rate: 0.001 with warmup
- Loss: CrossEntropyLoss

**Usage**:
```bash
# Basic usage
deepspeed --num_gpus=N simple_zero_training.py --deepspeed_config=CONFIG.json

# Custom options
deepspeed --num_gpus=4 simple_zero_training.py \
    --deepspeed_config=ds_config_zero2.json \
    --epochs=10 \
    --batch_size=64
```

**Command-line Arguments**:
- `--epochs`: Number of training epochs (default: 5)
- `--batch_size`: Batch size per GPU (default: 32)
- `--deepspeed_config`: Path to DeepSpeed config file (required)
- `--local_rank`: Local process rank (set automatically by launcher)

---

## Configuration Guide

### Configuration Files

#### `ds_config_zero2.json` - ZeRO Stage 2

**Best for**: Most distributed training scenarios

**Features**:
- Optimizer state + gradient partitioning
- ~8x memory reduction per GPU
- Good balance of speed and memory savings

**Key settings**:
```json
{
  "train_batch_size": 64,                    // Must = micro_batch * grad_accum * GPUs
  "train_micro_batch_size_per_gpu": 32,      // Batch size per GPU
  "gradient_accumulation_steps": 1,          // Accumulate N batches before update
  "zero_optimization": {
    "stage": 2,                              // ZeRO-2 optimization
    "overlap_comm": true,                    // Overlap communication with computation
    "contiguous_gradients": true,            // Reduce memory fragmentation
    "reduce_bucket_size": 2e8,               // Gradient reduction bucket size
    "allgather_bucket_size": 2e8             // Parameter gathering bucket size
  },
  "fp16": {
    "enabled": true,                         // Enable FP16 training
    "loss_scale": 0,                         // Dynamic loss scaling
    "initial_scale_power": 16,
    "loss_scale_window": 1000
  },
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.001,
      "betas": [0.9, 0.999],
      "eps": 1e-08
    }
  }
}
```

#### `ds_config_zero3.json` - ZeRO Stage 3

**Best for**: Very large models that don't fit in GPU memory

**Features**:
- Optimizer + gradient + parameter partitioning
- Linear memory scaling with number of GPUs
- Maximum memory efficiency

**Additional settings over ZeRO-2**:
```json
{
  "zero_optimization": {
    "stage": 3,
    "stage3_prefetch_bucket_size": 5e7,      // Parameter prefetch size
    "stage3_param_persistence_threshold": 1e5 // Keep small params in memory
  }
}
```

### Customization Guide

#### Adjusting for GPU Memory

**If you encounter OOM (Out of Memory)**:

1. **Reduce micro batch size**:
   ```json
   "train_micro_batch_size_per_gpu": 16,  // Was 32
   ```

2. **Increase gradient accumulation** (maintains effective batch size):
   ```json
   "gradient_accumulation_steps": 2,  // Was 1
   ```

3. **Use ZeRO-3 instead of ZeRO-2**:
   ```json
   "zero_optimization": {"stage": 3}
   ```

4. **Enable CPU offloading** (ZeRO-3 only):
   ```json
   "zero_optimization": {
     "stage": 3,
     "offload_optimizer": {"device": "cpu"},
     "offload_param": {"device": "cpu"}
   }
   ```

#### Adjusting for Different GPU Counts

**Important**: `train_batch_size` must equal `micro_batch_per_gpu * gradient_accum_steps * num_gpus`

**Examples**:
```json
// 1 GPU
{"train_batch_size": 32, "train_micro_batch_size_per_gpu": 32, "gradient_accumulation_steps": 1}

// 2 GPUs
{"train_batch_size": 64, "train_micro_batch_size_per_gpu": 32, "gradient_accumulation_steps": 1}

// 4 GPUs
{"train_batch_size": 128, "train_micro_batch_size_per_gpu": 32, "gradient_accumulation_steps": 1}

// 4 GPUs with gradient accumulation
{"train_batch_size": 256, "train_micro_batch_size_per_gpu": 32, "gradient_accumulation_steps": 2}
```

#### Performance Optimization

**For maximum throughput**:
```json
{
  "zero_optimization": {
    "overlap_comm": true,              // Essential for good performance
    "allgather_bucket_size": 5e8,      // Larger = fewer comm calls
    "reduce_bucket_size": 5e8,         // Larger = fewer comm calls
    "contiguous_gradients": true       // Reduce memory fragmentation
  }
}
```

**For gradient clipping**:
```json
{
  "gradient_clipping": 1.0              // Clip gradients to prevent instability
}
```

**For monitoring**:
```json
{
  "steps_per_print": 10,                // Print every N steps
  "wall_clock_breakdown": true          // Detailed timing breakdown
}
```

---

## Testing & Validation

### Validation Checklist

After running examples, verify:

**Training completes without errors**
- [ ] No CUDA errors
- [ ] No MPI errors
- [ ] No memory errors

**Loss decreases over epochs**
- [ ] Loss shows downward trend
- [ ] No NaN or Inf values

**GPU utilization is high**
- [ ] Check with `nvidia-smi` during training
- [ ] Should be >80% utilization

**Checkpoint saved successfully**
- [ ] `./checkpoints/final/` directory created
- [ ] Contains `mp_rank_00_model_states.pt`
- [ ] Contains `zero_pp_rank_0_mp_rank_00_optim_states.pt`

**Performance metrics**
- [ ] Samples/sec reported
- [ ] Throughput increases with more GPUs

### Expected Performance

**Single GPU (A100 40GB)**:
- Throughput: ~2,400 samples/sec
- Memory: ~0.9 GB
- Time per epoch: ~13 seconds

**2 GPUs (A100 40GB)**:
- Throughput: ~4,800 samples/sec (near-linear scaling)
- Memory: ~0.5 GB per GPU (ZeRO-2 reduces memory)
- Time per epoch: ~7 seconds

**4 GPUs (A100 40GB)**:
- Throughput: ~9,600 samples/sec
- Memory: ~0.3 GB per GPU
- Time per epoch: ~3.5 seconds

### Validation Commands

```bash
# Check checkpoint integrity
python -c "
import torch
ckpt = torch.load('./checkpoints/final/mp_rank_00_model_states.pt')
print(f'Checkpoint keys: {list(ckpt.keys())}')
"

# Load and verify model
cd ../..
python tests/deepspeed/test_checkpoint_loading.py

# Benchmark performance
cd examples/deepspeed
python -m pytest ../../tests/deepspeed/test_throughput.py -v
```

---

## Troubleshooting

### Common Issues

#### 1. Batch Size Mismatch

**Error**:
```
AssertionError: train_batch_size is not equal to micro_batch_per_gpu * gradient_acc_step * world_size
```

**Solution**: Adjust `train_batch_size` in config:
```bash
# For 2 GPUs with micro_batch=32, gradient_accum=1:
# train_batch_size must be: 32 * 1 * 2 = 64

# Quick fix:
python -c "import json; config = json.load(open('ds_config_zero2.json')); config['train_batch_size'] = 64; json.dump(config, open('ds_config_zero2.json', 'w'), indent=2)"
```

#### 2. FP16 Dtype Mismatch

**Error**:
```
RuntimeError: mat1 and mat2 must have the same dtype
```

**Cause**: Input data not cast to FP16 when FP16 training is enabled.

**Solution**: This is handled in `simple_zero_training.py`. If you're writing custom code:
```python
data = data.to(device)
if model_engine.fp16_enabled():
    data = data.half()
```

#### 3. libstdc++ Version Error

**Error**:
```
ImportError: libstdc++.so.6: version 'GLIBCXX_3.4.30' not found
```

**Solution**: Preload correct library:
```bash
export LD_PRELOAD=/opt/gcc/13.3.0/lib64/libstdc++.so.6:$LD_PRELOAD
```

#### 4. MPI Not Found

**Error**: `deepspeed: command not found` or `MPI backend not available`

**Solution**:
```bash
# Verify DeepSpeed is installed
which deepspeed

# Verify MPI backend
python -c "import torch.distributed as dist; print(dist.is_mpi_available())"

# If False, rebuild PyTorch with MPI support
```

#### 5. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions** (try in order):
1. Reduce micro batch size in config: `train_micro_batch_size_per_gpu: 16`
2. Use ZeRO-3: `ds_config_zero3.json`
3. Enable CPU offloading (see Configuration Guide above)

#### 6. Slow Performance

**Symptoms**: Much lower samples/sec than expected

**Solutions**:
1. Verify `overlap_comm: true` in config
2. Check network with: `mpirun -np 2 $MVAPICH_HOME/libexec/mvapich-plus/osu_bw`
3. Enable GPUDirect RDMA:
   ```bash
   export MV2_USE_CUDA=1
   export MV2_USE_GPUDIRECT_RDMA=1
   ```
4. Increase bucket sizes in config

---

## Performance Tips

### 1. Choose Right ZeRO Stage

- **ZeRO-2**: Default choice for most models
- **ZeRO-3**: Only if model doesn't fit in memory with ZeRO-2

### 2. Enable FP16

Always enable FP16 unless you have numerical stability issues:
```json
{"fp16": {"enabled": true}}
```

### 3. Optimize Communication

```json
{
  "zero_optimization": {
    "overlap_comm": true,
    "allgather_bucket_size": 5e8,
    "reduce_bucket_size": 5e8
  }
}
```

### 4. Tune Batch Sizes

- Start with largest `train_micro_batch_size_per_gpu` that fits in memory
- Use `gradient_accumulation_steps` to reach desired effective batch size
- Monitor GPU utilization with `nvidia-smi`

### 5. Use CUDA-aware MPI

Ensure GPUDirect RDMA is enabled:
```bash
export MV2_USE_CUDA=1
export MV2_USE_GPUDIRECT_RDMA=1
export MV2_CUDA_USE_NAIVE=0
```

### 6. Monitor Training

```bash
# In one terminal: run training
deepspeed --num_gpus=4 simple_zero_training.py --deepspeed_config=ds_config_zero2.json

# In another terminal: monitor GPUs
watch -n 1 nvidia-smi

# View logs
tail -f train.log
```

---

## Next Steps

### 1. Try Real Models

- Adapt example for your model architecture
- Test with HuggingFace Transformers
- Benchmark against baseline PyTorch

### 2. Multi-Node Training

```bash
# On MRI with SLURM
salloc -N 2 --gpus-per-node=4 -t 2:00:00

# Run across nodes
mpiexec -n 8 python simple_zero_training.py --deepspeed_config=ds_config_zero2.json
```

### 3. Advanced Features

- Try CPU/NVMe offloading for very large models
- Experiment with activation checkpointing
- Use mixed batch sizes for better throughput
- Enable compression for reduced communication

### 4. Benchmarking

```bash
# Run included benchmarks
cd ../../benchmarks
python -m pytest communication/ -v

# Profile your training
deepspeed --num_gpus=4 simple_zero_training.py \
    --deepspeed_config=ds_config_zero2.json \
    --profile
```

---

## Resources

- **DeepSpeed Documentation**: https://www.deepspeed.ai/
- **ZeRO Paper**: https://arxiv.org/abs/1910.02054
- **OSU HPC-AI Docs**: [../../docs/build-guides/](../../docs/build-guides/)
- **Build Instructions**: [../../docs/build-guides/deepspeed.md](../../docs/build-guides/deepspeed.md)
- **PyTorch Build Guide**: [../../docs/build-guides/pytorch.md](../../docs/build-guides/pytorch.md)

---

## Support

- **Issues**: https://github.com/OSU-Nowlab/osu-hpc-ai/issues
- **DeepSpeed Issues**: https://github.com/OSU-Nowlab/DeepSpeed/issues

---

**Maintained by**: Network-Based Computing Laboratory, The Ohio State University
