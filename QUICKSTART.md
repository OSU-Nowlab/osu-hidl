# OSU HPC-AI Quick Start Guide

Get up and running with the OSU HPC-AI stack in minutes.

## Prerequisites

- HPC cluster with NVIDIA GPUs
- Access to module system (gcc, cuda)
- SSH access to build nodes

## Option 1: Using Pre-built Wheels (Fastest)

```bash
# Register at hpc-ai.engineering.osu.edu to get wheel download links
# Example installation (URL provided after registration):
# pip install <wheel-url>/torch-2.x.x+cu126-cp312-linux_x86_64.whl

# Verify
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch.distributed as dist; print(f'MPI: {dist.is_mpi_available()}')"
```

**Note**: Pre-built wheels still require MVAPICH-Plus at runtime.

## Option 2: Build from Source (MRI Cluster)

### Step 1: Clone the Repository

```bash
cd "${HOME}"
git clone --recursive https://github.com/OSU-Nowlab/osu-hpc-ai.git
cd osu-hpc-ai
```

### Step 2: Build Frameworks

Build in dependency order:

```bash
# 1. Build PyTorch (~4-6 hours)
bash build_scripts/mri/build_pytorch_mri.sh

# 2. Build DeepSpeed (~20-30 min)
bash build_scripts/mri/build_deepspeed_mri.sh

# 3. Build SGLang (~5-10 min)
bash build_scripts/mri/build_sglang_mri.sh

# 4. Build vLLM (~30-60 min, requires 128GB RAM)
salloc -p devel -t 3:00:00 -C cuda -N 1 --gpus-per-node=2 --mem=128G
srun --pty bash
bash build_scripts/mri/build_vllm_mri.sh
```

### Step 3: Verify Installation

```bash
cd tests/pytorch
mpirun -n 2 python test_mpi_comm.py
```

## Option 3: Manual Build (Full Control)

### Step 1: Set Up Dependencies

```bash
cd ~/osu-hpc-ai
bash build_scripts/mri/setup_dependencies_mri.sh
```

**Manual cuDNN Download**:
1. Visit: https://developer.nvidia.com/rdp/cudnn-archive
2. Download: cuDNN v9.8.0 for CUDA 12.x
3. Extract:
   ```bash
   cd ~/osu-hpc-ai-dev/libext/cudnn
   tar -xf cudnn-linux-x86_64-9.8.0.87_cuda12-archive.tar.xz
   ```

### Step 2: Build MVAPICH-Plus

```bash
bash build_scripts/mri/build_mvapich_mri.sh
```

### Step 3: Build PyTorch

```bash
bash build_scripts/mri/build_pytorch_mri.sh
```

### Step 4: Build DeepSpeed (Optional)

```bash
# Allocate GPU for building
salloc -p devel -t 2:00:00 -C cuda -N 1 --gpus-per-node=1

# Build DeepSpeed (~30 minutes)
bash build_scripts/mri/build_deepspeed_mri.sh
```

### Step 5: Verify

```bash
# Test PyTorch
cd tests/pytorch
mpirun -n 2 python test_mpi_comm.py

# Test DeepSpeed (if built)
cd ../deepspeed
python test_installation.py
```

## Using the HPC-AI Stack

### Environment Setup

Create an activation script for convenience:

```bash
cat > ~/activate_hpc_ai.sh << 'EOF'
#!/bin/bash
# Activate HPC-AI environment

# Load modules
module reset
module load gcc/13.3.0
module load cuda/12.6

# Set MVAPICH-Plus paths
export MVAPICH_HOME="${HOME}/osu-hpc-ai-dev/install/mvapich-plus-4.1-cuda12.6"
export PATH="${MVAPICH_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MVAPICH_HOME}/lib:${LD_LIBRARY_PATH}"
export CPATH="${MVAPICH_HOME}/include:${CPATH}"

# Activate conda
source ${HOME}/osu-hpc-ai-dev/libext/miniconda3/bin/activate
conda activate hpc-ai-build

echo "HPC-AI environment activated"
EOF

chmod +x ~/activate_hpc_ai.sh
```

Use it:

```bash
source ~/activate_hpc_ai.sh
```

### Run PyTorch Example

```bash
cd ~/osu-hpc-ai/examples/pytorch

# Single node, 4 GPUs
mpiexec -n 4 python simple_distributed.py

# Multi-node, 8 GPUs (2 nodes x 4 GPUs)
mpiexec -np 8 -hosts node1,node2 python simple_distributed.py
```

### Run DeepSpeed Example

```bash
cd ~/osu-hpc-ai/examples/deepspeed

# Single GPU (testing)
deepspeed --num_gpus=1 simple_zero_training.py --deepspeed_config=ds_config_zero2.json

# Multi-GPU (production)
deepspeed --num_gpus=4 simple_zero_training.py --deepspeed_config=ds_config_zero2.json
```

**Expected output**:
```
DeepSpeed ZeRO Training Example
Creating torch.float16 ZeRO stage 2 optimizer
Training for 5 epochs...
Epoch 1/5 - Average Loss: 2.3136
Epoch 2/5 - Average Loss: 2.3062
...
Training complete!
```

### Run Tests

```bash
# PyTorch tests
cd ~/osu-hpc-ai/tests/pytorch
mpirun -n 2 python test_mpi_comm.py

# DeepSpeed tests
cd ~/osu-hpc-ai/tests/deepspeed
python test_installation.py
```

## Common Workflows

### Interactive Development

#### PyTorch Development

```bash
# Request interactive node
salloc -N 1 --ntasks-per-node=4 --gres=gpu:4 --time=2:00:00

# Activate environment
source ~/activate_hpc_ai.sh

# Test your code
mpiexec -n 4 python your_script.py

# Exit when done
exit
```

#### DeepSpeed Development

```bash
# Request GPUs (MRI cluster)
salloc -p devel -t 3:00:00 -C cuda -N 1 --gpus-per-node=2

# Activate environment
source ~/activate_hpc_ai.sh

# Quick test
cd ~/osu-hpc-ai/examples/deepspeed
deepspeed --num_gpus=2 simple_zero_training.py --deepspeed_config=ds_config_zero2.json

# Develop your model
deepspeed --num_gpus=2 my_training.py --deepspeed_config=my_config.json
```

### Batch Job Submission

#### PyTorch Job

Create `pytorch_train.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=pytorch-train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=10:00:00
#SBATCH --output=train-%j.out

# Activate environment
source ~/activate_hpc_ai.sh

# Run training
mpiexec -n 8 python train.py --config config.yaml
```

#### DeepSpeed Job

Create `deepspeed_train.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=deepspeed-train
#SBATCH -N 2
#SBATCH --gpus-per-node=4
#SBATCH --time=10:00:00
#SBATCH -p gpu
#SBATCH --output=deepspeed-%j.out

# Setup
module load gcc/13.3.0 cuda/12.6
source ~/miniconda3/bin/activate hpc-ai-build
export LD_PRELOAD=/opt/gcc/13.3.0/lib64/libstdc++.so.6

# Run with DeepSpeed
cd ~/osu-hpc-ai/examples/deepspeed
deepspeed --num_gpus=8 --num_nodes=2 train.py --deepspeed_config=ds_config_zero2.json
```

Submit:

```bash
sbatch pytorch_train.slurm
# or
sbatch deepspeed_train.slurm
```

### Debugging

```bash
# Enable verbose MPI output
export MV2_DEBUG_SHOW_BACKTRACE=1
export MV2_DEBUG_CORESIZE=unlimited

# Run with debugging
mpiexec -n 2 python your_script.py
```

## Next Steps

### Learn More

- **Documentation**: [`docs/`](docs/)
- **Examples**: [`examples/`](examples/)

### Try Advanced Features

1. **DeepSpeed ZeRO Optimization**:
   - See [`examples/deepspeed/`](examples/deepspeed/)
   - Try ZeRO Stage 2/3 for memory-efficient training
   - Use FP16 for 2x speedup

2. **Distributed Training**:
   - Multi-GPU: Scale to 4-8 GPUs per node
   - Multi-Node: Scale across multiple nodes
   - See [`docs/build-guides/`](docs/build-guides/)

3. **Performance Tuning**:
   - Enable GPUDirect RDMA for faster communication
   - Tune batch sizes for your model
   - Use communication overlap
   - See [`benchmarks/`](benchmarks/)

4. **LLM Inference**:
   - SGLang: Fast serving for LLMs with structured generation
   - See [`examples/sglang/`](examples/sglang/)

### Get Help

- **Issues**: [GitHub Issues](https://github.com/OSU-Nowlab/osu-hpc-ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/OSU-Nowlab/osu-hpc-ai/discussions)

## Troubleshooting

### Build Issues

**Problem**: Out of memory during PyTorch build

**Solution**:
```bash
# Edit build_pytorch_mri.sh
export MAX_JOBS=4  # Reduce from 16
```

**Problem**: CUDA not found

**Solution**:
```bash
module load cuda/12.6
echo $CUDA_HOME  # Verify CUDA path is set
```

### Runtime Issues

**Problem**: MPI not found

**Solution**:
```bash
source ~/activate_hpc_ai.sh
which mpicc  # Verify MVAPICH-Plus is loaded
```

**Problem**: Import errors

**Solution**:
```bash
conda activate hpc-ai-build
python -c "import torch; print(torch.__file__)"
```

**Problem**: libstdc++ version mismatch

**Symptom**: `GLIBCXX_3.4.30 not found`

**Solution**:
```bash
export LD_PRELOAD=/opt/gcc/13.3.0/lib64/libstdc++.so.6
```

### DeepSpeed Issues

**Problem**: Batch size mismatch

**Symptom**: `AssertionError: train_batch_size is not equal to...`

**Solution**:
```bash
# Adjust config for your GPU count
# train_batch_size = micro_batch_per_gpu * gradient_accum_steps * num_gpus
# Example: 32 * 1 * 2 = 64
```

**Problem**: CUDA out of memory

**Solution**:
```bash
# Reduce micro batch size in config
"train_micro_batch_size_per_gpu": 16  # was 32

# Or use ZeRO-3 for better memory efficiency
--deepspeed_config=ds_config_zero3.json
```

**Problem**: FP16 dtype mismatch

**Symptom**: `mat1 and mat2 must have same dtype`

**Solution**: This is fixed in our examples. If writing custom code:
```python
if model_engine.fp16_enabled():
    data = data.half()
```

### Performance Issues

**Problem**: Slow training

**Solution**:
```bash
# Enable GPUDirect RDMA
export MV2_USE_CUDA=1
export MV2_USE_GPUDIRECT_RDMA=1

# Enable CUDA-aware MPI
export MV2_USE_GDR=1
```

See [INSTALL.md](INSTALL.md#troubleshooting) for more help.

## Reference

### Key Directories

| Path | Description |
|------|-------------|
| `~/osu-hpc-ai/` | Monorepo (submodules, docs, examples) |
| `~/osu-hpc-ai-dev/` | Build workspace |
| `~/osu-hpc-ai-dev/install/` | Installed binaries |
| `~/osu-hpc-ai-dev/logs/` | Build logs |
| `~/osu-hpc-ai-dev/libext/` | Dependencies |

### Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `MVAPICH_HOME` | `~/osu-hpc-ai-dev/install/mvapich-plus-4.1-cuda12.6` | MVAPICH install |
| `CUDA_HOME` | `/opt/cuda/12.6` | CUDA toolkit |
| `PATH` | `$MVAPICH_HOME/bin:...` | Find MPI commands |
| `LD_LIBRARY_PATH` | `$MVAPICH_HOME/lib:...` | Find MPI libraries |

### Useful Commands

```bash
# Check MPI
mpichversion
which mpicc

# Check PyTorch
python -c "import torch; print(torch.__version__}"
python -c "import torch.distributed as dist; print(f'MPI: {dist.is_mpi_available()}')"

# Check DeepSpeed
python -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')"
ds_report

# Test MPI
mpiexec -n 2 hostname

# Test PyTorch + MPI
mpiexec -n 2 python -c "import torch.distributed as dist; dist.init_process_group('mpi')"

# Check GPUs
nvidia-smi

# Quick DeepSpeed test
cd ~/osu-hpc-ai/tests/deepspeed
python test_installation.py
```

---

For detailed documentation, see [README.md](README.md) and [`docs/`](docs/).
