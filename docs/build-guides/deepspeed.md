# Building DeepSpeed with MVAPICH-Plus

This guide explains how to build DeepSpeed with MVAPICH-Plus support for optimized distributed training on HPC systems.

## Overview

This NOWLAB fork of DeepSpeed integrates with CUDA-aware MPI (MVAPICH-Plus) for high-performance distributed training. Key features:

- **ZeRO Optimization** with MPI backend
- **CUDA-aware collective operations** via MVAPICH-Plus
- **Custom CUDA kernels** (fused Adam, transformer ops, etc.)
- **GPUDirect RDMA** support for efficient GPU-to-GPU communication

## Prerequisites

### System Requirements

- **GPU**: NVIDIA GPUs with Compute Capability ≥ 6.0 (Pascal or newer)
- **CUDA**: Version 11.0+ (12.x recommended)
- **GCC**: Version 11.0+ (13.x recommended)
- **Python**: Version 3.8-3.12

### Required Dependencies

1. **PyTorch with MPI support**
   - Must be built with MVAPICH-Plus integration
   - See [PyTorch Build Guide](pytorch.md)

2. **MVAPICH-Plus**
   - CUDA-aware MPI library
   - Version 4.1+ recommended
   - Download: http://mvapich.cse.ohio-state.edu/

3. **CUDA Toolkit**
   - Includes nvcc compiler
   - cuDNN 8.0+ required for some operations

## Quick Start

### Option 1: Using OSU HPC-AI Build Scripts (Recommended for MRI)

If you're using the OSU HPC-AI monorepo on MRI cluster:

```bash
# From osu-hpc-ai root
bash build_scripts/mri/build_deepspeed_mri.sh
```

This handles all configuration automatically.

### Option 2: Manual Build (Generic Systems)

#### Step 1: Set Up Environment

```bash
# Load CUDA and GCC
export CUDA_HOME=/path/to/cuda
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Set up MVAPICH-Plus
export MVAPICH_HOME=/path/to/mvapich-plus
export PATH="${MVAPICH_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MVAPICH_HOME}/lib:${LD_LIBRARY_PATH}"
export CPATH="${MVAPICH_HOME}/include:${CPATH}"

# Set MPI compilers
export CC=$(which mpicc)
export CXX=$(which mpicxx)
```

#### Step 2: Activate Python Environment

```bash
# Using conda (recommended)
conda create -n deepspeed python=3.12
conda activate deepspeed

# Or using venv
python -m venv deepspeed-env
source deepspeed-env/bin/activate
```

#### Step 3: Verify PyTorch Installation

```bash
# Check PyTorch with MPI backend
python -c "import torch; print(torch.__version__)"
python -c "import torch.distributed as dist; print(f'MPI available: {dist.is_mpi_available()}')"
```

If MPI backend is not available, PyTorch must be rebuilt with MVAPICH-Plus support first.

#### Step 4: Build DeepSpeed

```bash
# Clone DeepSpeed (if not already done)
git clone -b nowlab-main https://github.com/OSU-Nowlab/DeepSpeed.git
cd DeepSpeed

# Configure build options
export DS_BUILD_OPS=1                     # Build custom CUDA ops
export DS_BUILD_FUSED_ADAM=1             # Fused Adam optimizer
export DS_BUILD_CPU_ADAM=1               # CPU Adam (for offloading)
export DS_BUILD_UTILS=1                  # Utility ops
export DS_BUILD_FUSED_LAMB=1             # Fused LAMB optimizer
export DS_BUILD_TRANSFORMER=1            # Transformer kernels
export DS_BUILD_STOCHASTIC_TRANSFORMER=1 # Stochastic transformer
export DS_BUILD_SPARSE_ATTN=0            # Sparse attention (disable if issues)

# Set CUDA architectures (adjust for your GPUs)
# 6.0=Pascal, 7.0=Volta, 8.0=Ampere, 9.0=Hopper
export TORCH_CUDA_ARCH_LIST="8.0;9.0"

# Build and install
pip install -e .

# This takes approximately 20-30 minutes
```

#### Step 5: Verify Installation

```bash
# Check DeepSpeed version
python -c "import deepspeed; print(deepspeed.__version__)"

# Run environment report
ds_report
```

## Build Configuration Options

### Essential Build Flags

| Flag | Description | Recommended |
|------|-------------|-------------|
| `DS_BUILD_OPS` | Build all custom ops | 1 |
| `DS_BUILD_FUSED_ADAM` | Fused Adam optimizer | 1 |
| `DS_BUILD_CPU_ADAM` | CPU Adam for offloading | 1 |
| `DS_BUILD_UTILS` | Utility operations | 1 |
| `DS_BUILD_TRANSFORMER` | Transformer kernels | 1 |

### Optional Build Flags

| Flag | Description | When to Enable |
|------|-------------|----------------|
| `DS_BUILD_FUSED_LAMB` | Fused LAMB optimizer | If using LAMB |
| `DS_BUILD_SPARSE_ATTN` | Sparse attention | For sparse models |
| `DS_BUILD_STOCHASTIC_TRANSFORMER` | Stochastic ops | Advanced use cases |

### CUDA Architecture Targeting

Set `TORCH_CUDA_ARCH_LIST` to match your GPUs:

```bash
# Single architecture (faster build)
export TORCH_CUDA_ARCH_LIST="8.0"  # A100 only

# Multiple architectures (portable)
export TORCH_CUDA_ARCH_LIST="7.0;8.0;9.0"  # V100, A100, H100

# All modern GPUs
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;9.0"
```

## System-Specific Notes

### HPC Clusters with Module System

```bash
# Typical module loading
module reset
module load gcc/13.3.0
module load cuda/12.6
module load python/3.12

# Then proceed with build
```

### SLURM-Based Systems

Build on a compute node with GPU access:

```bash
# Request interactive node
salloc -N 1 --gres=gpu:1 --time=2:00:00

# Proceed with build
```

### Without CUDA-aware MPI

If MVAPICH-Plus is not available, DeepSpeed will fall back to NCCL for GPU communication. Performance may be reduced on some systems.

## Using Pre-built Wheels

For common configurations, we provide pre-built wheels:

```bash
# Check available wheels
https://hpc-ai.engineering.osu.edu/wheels/

# Install (example)
pip install https://hpc-ai.engineering.osu.edu/wheels/deepspeed-X.Y.Z+cu126-cp312-cp312-linux_x86_64.whl
```

**Note**: Pre-built wheels still require MVAPICH-Plus at runtime for MPI backend.

## Usage

### Basic Training Script

```python
import deepspeed

# Initialize DeepSpeed
deepspeed.init_distributed()

# Create model and config
model = YourModel()
config = {
    "train_batch_size": 128,
    "zero_optimization": {"stage": 2},
    "fp16": {"enabled": True}
}

# Initialize engine
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=config
)

# Training loop
for batch in dataloader:
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

### Running with MPI

```bash
# Using DeepSpeed launcher
deepspeed --num_gpus=4 train.py --deepspeed_config=config.json

# Using MPI directly
mpiexec -n 4 python train.py --deepspeed_config=config.json

# Multi-node
mpiexec -n 8 -hosts node1,node2 python train.py --deepspeed_config=config.json
```

## ZeRO Optimization Stages

DeepSpeed's ZeRO optimization partitions model states across GPUs:

| Stage | Partitions | Memory Reduction | Communication Overhead |
|-------|-----------|------------------|----------------------|
| ZeRO-1 | Optimizer states | ~4x | Low |
| ZeRO-2 | + Gradients | ~8x | Medium |
| ZeRO-3 | + Parameters | Linear with GPUs | Higher |

### Choosing a Stage

- **ZeRO-1**: Minimal overhead, moderate memory savings
- **ZeRO-2**: **Recommended for most cases** - good balance
- **ZeRO-3**: Maximum memory efficiency for very large models

## Performance Tuning

### Enable GPUDirect RDMA

For MVAPICH-Plus:
```bash
export MV2_USE_CUDA=1
export MV2_USE_GPUDIRECT_RDMA=1
```

### Optimize Communication

In DeepSpeed config:
```json
{
  "zero_optimization": {
    "overlap_comm": true,
    "allgather_bucket_size": 2e8,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true
  }
}
```

### FP16 Training

```json
{
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000
  }
}
```

## Troubleshooting

### Build Issues

#### 1. nvcc Not Found

**Symptom**: `nvcc: command not found` during build

**Solution**:
```bash
export CUDA_HOME=/path/to/cuda
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
```

#### 2. Out of Memory During Build

**Symptom**: Build process killed, system hanging

**Solution**:
```bash
export MAX_JOBS=4  # Reduce parallel compilation jobs
# Or for very limited memory:
export MAX_JOBS=1
```

#### 3. MPI Compiler Not Found

**Symptom**: `mpicc: command not found`

**Solution**:
```bash
which mpicc  # Should point to MVAPICH-Plus
export PATH="${MVAPICH_HOME}/bin:${PATH}"
export CC=$(which mpicc)
export CXX=$(which mpicxx)
```

#### 4. CUDA 12.6 Transformer Inference Compilation Error

**Symptom**:
```
csrc/transformer/inference/csrc/gelu.cu(16): error: class "__nv_bfloat162" has already been defined
```

**Root Cause**: Incompatibility between DeepSpeed transformer inference kernels and CUDA 12.6

**Solution**: Disable transformer inference ops (not needed for training):
```bash
export DS_BUILD_TRANSFORMER_INFERENCE=0
pip install -e .
```

**Note**: This only affects inference-specific ops. Training ops work fine.

#### 5. Missing Spatial Directory Error

**Symptom**:
```
error: could not create 'deepspeed/ops/spatial/spatial_inference_op...so': No such file or directory
```

**Root Cause**: Build process tries to copy compiled ops before directory exists

**Solution**: Create the directory manually before building:
```bash
mkdir -p deepspeed/ops/spatial
pip install -e .
```

**Permanent Fix**: This is fixed in our MRI build script automatically.

#### 6. libstdc++ Version Mismatch

**Symptom**:
```
ImportError: libstdc++.so.6: version 'GLIBCXX_3.4.30' not found
```

**Root Cause**: DeepSpeed/PyTorch built with newer GCC than conda environment provides

**Solution**: Preload the correct libstdc++ library:
```bash
export LD_PRELOAD=/opt/gcc/13.3.0/lib64/libstdc++.so.6:$LD_PRELOAD
```

Add this to your environment activation script:
```bash
# Add to ~/.bashrc or conda environment activation
echo 'export LD_PRELOAD=/opt/gcc/13.3.0/lib64/libstdc++.so.6:$LD_PRELOAD' >> ~/.bashrc
```

### Runtime Issues

#### 1. MPI Backend Not Available

**Symptom**: `RuntimeError: MPI backend not available`

**Solution**:
```bash
# Check PyTorch MPI support
python -c "import torch.distributed as dist; print(f'MPI available: {dist.is_mpi_available()}')"

# If False, rebuild PyTorch with MPI support
# See pytorch.md in this directory
```

#### 2. Batch Size Configuration Error

**Symptom**:
```
AssertionError: Check batch related parameters. train_batch_size is not equal to
micro_batch_per_gpu * gradient_acc_step * world_size
```

**Root Cause**: `train_batch_size` in config doesn't match: `micro_batch_per_gpu * gradient_accumulation_steps * num_gpus`

**Solution**: Adjust your config file:
```json
{
  "train_batch_size": 64,              // Must equal: 32 * 1 * 2
  "train_micro_batch_size_per_gpu": 32,
  "gradient_accumulation_steps": 1,
  // When running with 2 GPUs
}
```

**Formula**: `train_batch_size = micro_batch_per_gpu * gradient_accumulation_steps * world_size`

#### 3. FP16 Dtype Mismatch Error

**Symptom**:
```
RuntimeError: mat1 and mat2 must have the same dtype
```

**Root Cause**: When FP16 is enabled, model weights are in FP16 but input data remains in FP32

**Solution**: Cast input tensors to match model dtype:
```python
# In your training loop
data = data.to(device)
if model_engine.fp16_enabled():
    data = data.half()  # Cast to FP16

outputs = model_engine(data)
```

**Note**: This is handled automatically in our example scripts.

#### 4. CUDA Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions** (try in order):
1. Reduce `train_micro_batch_size_per_gpu` (e.g., 32 → 16)
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Use higher ZeRO stage (ZeRO-2 → ZeRO-3)
4. Enable CPU/NVMe offloading:
   ```json
   {
     "zero_optimization": {
       "stage": 3,
       "offload_optimizer": {"device": "cpu"},
       "offload_param": {"device": "cpu"}
     }
   }
   ```

#### 5. Slow Training Performance

**Symptoms**: Lower than expected throughput

**Solutions**:
1. Enable communication overlap:
   ```json
   {"zero_optimization": {"overlap_comm": true}}
   ```

2. Increase bucket sizes for fewer communication calls:
   ```json
   {
     "zero_optimization": {
       "allgather_bucket_size": 5e8,
       "reduce_bucket_size": 5e8
     }
   }
   ```

3. Verify GPUDirect RDMA is enabled:
   ```bash
   export MV2_USE_CUDA=1
   export MV2_USE_GPUDIRECT_RDMA=1
   ```

4. Check network bandwidth:
   ```bash
   # Test MPI bandwidth
   mpirun -np 2 $MVAPICH_HOME/libexec/mvapich-plus/osu_bw
   ```

#### 6. MIG Partition Warnings

**Symptom**:
```
Detected CUDA_VISIBLE_DEVICES=MIG-xxx but ignoring it because --num_gpus was used
```

**Root Cause**: SLURM sets CUDA_VISIBLE_DEVICES to MIG UUIDs, but DeepSpeed overrides it

**Solution**: This is expected behavior. DeepSpeed properly manages GPU visibility. You can safely ignore this warning.

**Alternative**: Use MPI directly instead of DeepSpeed launcher:
```bash
mpiexec -n 2 python train.py --deepspeed_config=config.json
```

## Testing

### Quick Test

```bash
# Test import
python -c "import deepspeed; print(deepspeed.__version__)"

# Run environment report
ds_report

# Test MPI
mpiexec -n 2 python -c "import deepspeed; deepspeed.init_distributed(); print('Success')"
```

### Run Example

```bash
# Get example from osu-hpc-ai
git clone https://github.com/OSU-Nowlab/osu-hpc-ai.git
cd osu-hpc-ai/examples/deepspeed

# Run ZeRO-2 training
deepspeed --num_gpus=4 simple_zero_training.py --deepspeed_config=ds_config_zero2.json
```

## Resources

- **DeepSpeed Documentation**: https://www.deepspeed.ai/
- **ZeRO Paper**: https://arxiv.org/abs/1910.02054
- **MVAPICH-Plus**: http://mvapich.cse.ohio-state.edu/

## Support

- **Issues**: https://github.com/OSU-Nowlab/DeepSpeed/issues
- **Upstream DeepSpeed**: https://github.com/microsoft/DeepSpeed

---

**Maintained by**: NOWLAB, The Ohio State University
