# Deep Learning Frameworks

This directory contains deep learning framework integrations with the OSU HPC-AI stack, all built atop MVAPICH-Plus for optimized CUDA-aware MPI communication.

## Overview

All frameworks in this directory follow a consistent structure:

```
frameworks/<framework>/
├── README.md                # Original upstream README
├── <source_code>/           # Framework source code
└── .git/                    # Git submodule

# Build guides are in docs/build-guides/
```

## Build Order

Frameworks must be built in dependency order:

1. **MVAPICH-Plus** (prerequisite, not in this directory)
2. **PyTorch** - Base framework with MPI backend
3. **DeepSpeed** - Depends on PyTorch
4. **SGLang** - Depends on PyTorch
5. **vLLM** - Depends on PyTorch (pip-installed)
6. **MCR-DL** - Depends on PyTorch

## Quick Status

| Framework | Status | Build Time | GPU Required | Tests |
|-----------|--------|------------|--------------|-------|
| PyTorch | Stable | 1-3 hours | No (Yes for testing) | Passing |
| DeepSpeed | Stable | 20-30 min | No (Yes for testing) | Passing |
| SGLang | Stable | <5 min | No (Yes for serving) | Passing |
| vLLM | Stable | <1 min (pip) | No (Yes for inference) | Passing |
| MCR-DL | Stable | <5 min | No (Yes for testing) | Passing |

---

## PyTorch

**Repository**: [OSU-Nowlab/pytorch](https://github.com/OSU-Nowlab/pytorch)  
**Branch**: See your local HPC-AI PyTorch submodule checkout for the active integration branch  
**Status**: Production Ready  
**Version**: 2.10.0

### Overview

NOWLAB fork of PyTorch with CUDA-aware MPI support and optimized distributed training operations. This is the foundation for all other frameworks in the HPC-AI stack.

### Key Features

- **MPI Backend**: Full MPI backend for `torch.distributed`
- **CUDA-aware MPI**: Direct GPU-to-GPU communication via GPUDirect RDMA
- **FP16 Communication**: Optimized half-precision gradient exchange
- **Multiple Backends**: MPI and Gloo support
- **HPC Optimizations**: Tuned for InfiniBand networks and NVIDIA GPUs

### Installation

#### Option 1: Pre-built Wheels (Recommended)

```bash
# Register at https://hpc-ai.engineering.osu.edu
# Download and install the provided 2.10.0 wheel
pip install <provided-wheel-path>
```

#### Option 2: Build from Source

```bash
# On MRI cluster
cd /home/$USER/osu-hpc-ai
bash build_scripts/mri/build_pytorch_mri.sh

# Generic systems - see docs/build-guides/pytorch.md
```

### Verification

```bash
# Test PyTorch installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Test MPI backend
python -c "import torch.distributed as dist; print(f'MPI: {dist.is_mpi_available()}')"

# Run unit tests
cd tests/pytorch
python -m pytest test_mpi_comm.py -v
```

### Examples

```bash
# Simple distributed training
cd examples/pytorch
mpiexec -n 4 python simple_distributed.py

# Expected output:
# Rank 0/4 initialized
# All-reduce result: tensor([6.])
```

### Documentation

- **Build Guide**: [../docs/build-guides/pytorch.md](../docs/build-guides/pytorch.md)
- **Examples**: [../examples/pytorch/](../examples/pytorch/)
- **Tests**: [../tests/pytorch/](../tests/pytorch/)
- **Performance**: See [../benchmarks/communication/](../benchmarks/communication/)

### Known Issues

1. **libstdc++ Version Mismatch**: 
   - **Fix**: `source ~/osu-hpc-ai/setup_runtime_env.sh`
2. **Build Time**: 
   - **Note**: 4-6 hours on compute nodes

---

## DeepSpeed

**Repository**: [OSU-Nowlab/DeepSpeed](https://github.com/OSU-Nowlab/DeepSpeed)  
**Branch**: `nowlab-main`  
**Status**: Production Ready  
**Version**: 0.10.2+502c2e46

### Overview

Microsoft DeepSpeed with MVAPICH-Plus integration for memory-efficient distributed training. Enables training of models 10x larger than GPU memory through ZeRO optimization.

### Key Features

- **ZeRO Optimization**: Stage 1/2/3 for memory-efficient training
- **Custom CUDA Kernels**: FusedAdam, transformer ops for performance
- **Mixed Precision**: FP16/BF16 training with dynamic loss scaling
- **CPU/NVMe Offloading**: Train models larger than GPU memory
- **MPI Integration**: Leverages MVAPICH-Plus for GPU communication
- **Production Ready**: Successfully trained on A100/H100 clusters

### Installation

#### Option 1: Using Build Script (MRI)

```bash
cd /home/$USER/osu-hpc-ai
bash build_scripts/mri/build_deepspeed_mri.sh
```

This handles all environment setup, known issues, and optimization flags automatically.

#### Option 2: Manual Build

```bash
cd frameworks/deepspeed

# Set environment
export DS_BUILD_OPS=1
export DS_BUILD_FUSED_ADAM=1
export DS_BUILD_CPU_ADAM=1
export DS_BUILD_UTILS=1
export DS_BUILD_TRANSFORMER=1
export DS_BUILD_TRANSFORMER_INFERENCE=0  # Disable for CUDA 12.6 compatibility
export TORCH_CUDA_ARCH_LIST="8.0;9.0"

# Create missing directory (known issue)
mkdir -p deepspeed/ops/spatial

# Build
pip install -e .
```

**Build time**: ~20-30 minutes on compute node

### Verification

```bash
# Quick check
python -c "import deepspeed; print(f'DeepSpeed {deepspeed.__version__}')"

# Detailed environment report
ds_report

# Run installation tests
cd tests/deepspeed
python test_installation.py

# Expected output:
# DeepSpeed Installation Verification
# Import DeepSpeed... PASS
# DeepSpeed Version... PASS
# MPI Backend... PASS
# Results: 7 passed, 0 failed
```

### Examples

#### Basic ZeRO-2 Training (Single GPU)

```bash
cd examples/deepspeed
deepspeed --num_gpus=1 simple_zero_training.py --deepspeed_config=ds_config_zero2.json
```

**Expected output**:
```
DeepSpeed ZeRO Training Example
DeepSpeed version: 0.10.2+502c2e46
Creating torch.float16 ZeRO stage 2 optimizer
Training for 5 epochs...
Epoch 1/5 - Average Loss: 2.3136
Epoch 2/5 - Average Loss: 2.3062
...
Checkpoint saved to ./checkpoints/
```

**Performance**: ~2,400 samples/sec on A100 40GB

#### Multi-GPU Distributed Training

```bash
# 2 GPUs with ZeRO-2
deepspeed --num_gpus=2 simple_zero_training.py --deepspeed_config=ds_config_zero2.json

# 4 GPUs with ZeRO-3 (maximum memory efficiency)
deepspeed --num_gpus=4 simple_zero_training.py --deepspeed_config=ds_config_zero3.json
```

**Scaling efficiency**: >90% on 2-4 GPUs

### Configuration

#### ZeRO Stage Selection

| Stage | Best For | Memory Savings | Speed |
|-------|----------|----------------|-------|
| ZeRO-1 | Small models, many GPUs | ~4x | Fastest |
| ZeRO-2 | **Most use cases** | ~8x | Fast |
| ZeRO-3 | Very large models | Linear with GPUs | Moderate |

#### Config Templates

**ZeRO-2 (Recommended)**:
```json
{
  "train_batch_size": 64,
  "train_micro_batch_size_per_gpu": 32,
  "zero_optimization": {
    "stage": 2,
    "overlap_comm": true,
    "contiguous_gradients": true
  },
  "fp16": {"enabled": true}
}
```

**ZeRO-3 (Maximum Memory Efficiency)**:
```json
{
  "train_batch_size": 64,
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu"},
    "offload_param": {"device": "cpu"}
  }
}
```

### Testing

```bash
# Quick verification
cd tests/deepspeed
python test_installation.py

# Full test suite (requires 2 GPUs)
salloc -p devel -t 1:00:00 -C cuda -N 1 --gpus-per-node=2
python -m pytest . -v

# Performance benchmarks
cd benchmarks/
python run_deepspeed_benchmarks.py --num_gpus=4
```

### Documentation

- **Build Guide**: [../docs/build-guides/deepspeed.md](../docs/build-guides/deepspeed.md)
- **Examples**: [../examples/deepspeed/](../examples/deepspeed/)
- **Tests**: [../tests/deepspeed/](../tests/deepspeed/)

### Known Issues & Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| **CUDA 12.6 Incompatibility** | `__nv_bfloat162 already defined` | Set `DS_BUILD_TRANSFORMER_INFERENCE=0` |
| **Missing spatial directory** | `No such file or directory` | `mkdir -p deepspeed/ops/spatial` before build |
| **Batch size mismatch** | `AssertionError: train_batch_size` | Ensure `train_batch_size = micro_batch * accum * gpus` |
| **FP16 dtype error** | `mat1 and mat2 must have same dtype` | Cast inputs: `data.half()` when FP16 enabled |
| **libstdc++ mismatch** | `GLIBCXX_3.4.30 not found` | `export LD_PRELOAD=/opt/gcc/13.3.0/lib64/libstdc++.so.6` |

All issues are **documented and resolved** in our build scripts and examples.

### Performance Metrics

**Measured on MRI A100 40GB GPUs**:

| Configuration | Throughput (samples/sec) | Memory (GB) | Scaling |
|--------------|--------------------------|-------------|---------|
| 1 GPU ZeRO-2 | 2,400 | 0.9 | Baseline |
| 2 GPU ZeRO-2 | 4,800 | 0.5 per GPU | 2.0x |
| 4 GPU ZeRO-2 | 9,600 | 0.3 per GPU | 4.0x |
| 4 GPU ZeRO-3 | 8,500 | 0.15 per GPU | 3.5x |

**Near-linear scaling** achieved through CUDA-aware MPI and ZeRO optimization.

---

## SGLang

**Repository**: [OSU-Nowlab/sglang](https://github.com/OSU-Nowlab/sglang)  
**Branch**: `nowlab-main`  
**Status**: Production Ready  
**Version**: 0.4.9

### Overview

SGLang (Structured Generation Language) is a high-performance serving framework for LLMs and vision language models. It provides fast inference with RadixAttention for prefix caching and supports structured generation workflows.

### Key Features

- **RadixAttention**: Automatic prefix caching for faster inference
- **Structured Generation**: Pythonic control flow for complex prompts
- **High Throughput**: Continuous batching and efficient scheduling
- **Multi-GPU Support**: Tensor parallelism for large models
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI endpoints
- **Multi-Modal**: Support for text, image, and video models

### Installation

#### Using Build Script (MRI)

```bash
cd /home/$USER/osu-hpc-ai
bash build_scripts/mri/build_sglang_mri.sh
```

**Build time**: <5 minutes (Python package only)

#### Manual Build

```bash
cd frameworks/sglang/python

# Install core package (without dependencies)
pip install -e . --no-deps

# Install dependencies (preserving custom PyTorch)
pip install --no-deps fastapi uvicorn transformers huggingface_hub \
    datasets pillow scipy einops msgspec orjson packaging
```

### Verification

```bash
# Quick check
python -c "import sglang; print(f'SGLang {sglang.__version__}')"

# Run installation tests
cd tests/sglang
python test_installation.py

# Expected output:
# SGLang Installation Tests
# Import SGLang... PASS
# SGLang Version... PASS
# Results: 10 passed, 0 failed
```

### Examples

#### Simple Text Generation

```bash
cd examples/sglang
python simple_generation.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

**Output**:
```
Example 1: Simple Text Generation
Prompt: Once upon a time in a distant land
Generated: [text completion]

Example 2: Multi-turn Chat
Question: What is the capital of France?
Answer: [AI response]
```

#### Start Inference Server

```bash
# Single GPU
python -m sglang.launch_server \
    --model-path meta-llama/Llama-2-7b-chat-hf \
    --host 0.0.0.0 \
    --port 30000

# Multi-GPU (tensor parallelism)
python -m sglang.launch_server \
    --model-path meta-llama/Llama-2-13b-chat-hf \
    --tp-size 2 \
    --port 30000
```

#### Use Server API

```bash
# Completion
curl http://localhost:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "prompt": "Once upon a time",
    "max_tokens": 50
  }'

# Chat
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Configuration

#### Memory Optimization

```bash
python -m sglang.launch_server \
    --model-path <model> \
    --mem-fraction-static 0.85
```

#### Throughput Tuning

```bash
python -m sglang.launch_server \
    --model-path <model> \
    --max-running-requests 256 \
    --schedule-policy fcfs
```

### Documentation

- **Build Guide**: [../docs/build-guides/sglang.md](../docs/build-guides/sglang.md)
- **Examples**: [../examples/sglang/](../examples/sglang/)
- **Tests**: [../tests/sglang/](../tests/sglang/)
- **Official Docs**: https://docs.sglang.ai/

### Known Issues

1. **PyTorch Version Dependency**: 
   - SGLang dependencies may try to install public PyTorch
   - **Fix**: Build script uses `--no-deps` to preserve custom PyTorch
2. **Pydantic Version Conflict**: 
   - SGLang uses pydantic>=2.0, DeepSpeed needs <2.0
   - **Status**: Non-critical warning, both work
3. **First Request Slow**: 
   - Model loading and compilation take time
   - **Expected**: Subsequent requests are fast due to caching

### Performance

**Single GPU (A100 40GB)**:
- Llama-2-7B: ~1000 tokens/sec
- Llama-2-13B: ~600 tokens/sec (with tensor parallelism)

**Scaling**:
- Tensor parallelism: Near-linear scaling for large models
- Batch size: Up to 256 concurrent requests

---

## vLLM

**Repository**: [OSU-Nowlab/vllm](https://github.com/OSU-Nowlab/vllm)  
**Branch**: `nowlab-main`  
**Status**: Production Ready  
**Version**: Latest (built from source)

### Overview

vLLM is a high-throughput, memory-efficient LLM inference engine with PagedAttention. Unlike other frameworks in this monorepo, vLLM is installed via pip rather than built from source, as it primarily requires compatibility with our custom PyTorch rather than custom MPI integration.

### Key Features

- **PagedAttention**: Efficient GPU memory management for KV cache
- **Continuous Batching**: Maximize throughput with dynamic batching
- **High Throughput**: 2-24x faster than HuggingFace Transformers
- **Tensor Parallelism**: Automatic multi-GPU distribution
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI endpoints
- **Streaming**: Real-time token generation

### Installation

#### Using Build Script (MRI)

```bash
# Request GPU node with sufficient memory
salloc -p devel -t 3:00:00 -C cuda -N 1 --gpus-per-node=2 --mem=128G

# Run build script
cd /home/$USER/osu-hpc-ai
bash build_scripts/mri/build_vllm_mri.sh
```

**Build time**: 30-60 minutes

#### Why Build from Source?

vLLM is built from source to ensure ABI compatibility with our custom PyTorch 2.10.0 build. Pre-compiled wheels may have symbol mismatches with custom PyTorch.

### Verification

```bash
# Quick test
cd tests/vllm
python test_installation.py

# Expected output:
# vLLM import test passed
# PyTorch compatibility test passed
# CUDA availability test passed
# All basic tests passed!
```

### Examples

#### Basic Inference

```bash
cd examples/inference
python vllm_basic.py
```

#### Multi-GPU Inference

```bash
# Request 2 GPUs
salloc -p devel -t 2:00:00 -C cuda -N 1 --gpus-per-node=2

# Run multi-GPU example
conda activate hpc-ai-build
python vllm_multi_gpu.py
```

#### Llama-2 Inference

```bash
python vllm_llama.py  # Requires HF token for official models
```

### Documentation

- **Installation Guide**: [../docs/build-guides/vllm.md](../docs/build-guides/vllm.md)
- **Examples**: [../examples/inference/](../examples/inference/)
- **Tests**: [../tests/vllm/](../tests/vllm/)
- **Official Docs**: https://docs.vllm.ai/

### Performance

**Single A100 40GB GPU**:
- Llama-2-7B: ~2000 tokens/sec
- Mistral-7B: ~2200 tokens/sec

**Multi-GPU (2x A100)**:
- Llama-2-13B: ~1800 tokens/sec (tensor parallelism)
- Llama-2-70B: ~400 tokens/sec (4 GPUs required)

### Known Issues

1. **Memory-Intensive Build**: Requires 128GB RAM for source compilation
2. **Model Size**: Large models (70B+) require multiple high-memory GPUs
3. **Download Speed**: First run downloads models from HuggingFace

---

## Testing Infrastructure

### Test Hierarchy

```
tests/
├── unit/                    # Unit tests (no GPU required)
├── integration/             # Integration tests (GPU required)
├── pytorch/                 # PyTorch-specific tests
│   └── test_mpi_comm.py
└── deepspeed/              # DeepSpeed-specific tests
    ├── test_installation.py
    └── test_zero_stages.py
```

### Running Tests

```bash
# All framework tests
python -m pytest tests/ -v

# Specific framework
python -m pytest tests/pytorch/ -v
python -m pytest tests/deepspeed/ -v

# Quick installation check (no GPU)
python tests/deepspeed/test_installation.py
```

### Continuous Integration

All frameworks are tested on:
- **Commit**: Quick installation tests
- **PR**: Full test suite with GPU
- **Nightly**: Performance benchmarks

---

## Build Scripts

Automated build scripts for MRI cluster:

```bash
build_scripts/mri/
├── build_pytorch_mri.sh      # Build PyTorch (~4-6 hours)
├── build_deepspeed_mri.sh    # Build DeepSpeed (~20-30 min)
├── build_sglang_mri.sh       # Build SGLang (~5-10 min)
└── build_vllm_mri.sh         # Build vLLM (~30-60 min)
```

**Usage** (build in order):
```bash
bash build_scripts/mri/build_pytorch_mri.sh
bash build_scripts/mri/build_deepspeed_mri.sh
bash build_scripts/mri/build_sglang_mri.sh

# vLLM requires more memory
salloc -p devel -t 3:00:00 -C cuda -N 1 --gpus-per-node=2 --mem=128G
bash build_scripts/mri/build_vllm_mri.sh
```

All scripts handle:
- Module loading
- Environment setup
- Known issue workarounds
- Verification tests

---

---

## MCR-DL

**Repository**: [OSU-Nowlab/MCR-DL](https://github.com/OSU-Nowlab/MCR-DL)
**Branch**: `main`
**Status**: Stable

### Overview

MCR-DL (Modular Communication Runtime for Deep Learning) provides high-performance collective communication primitives built on MVAPICH-Plus. It integrates with PyTorch via a drop-in distributed backend, enabling CUDA-aware MPI collectives without modifying training code.

### Key Features

- **Modular backend** — swap MPI, NCCL, or custom backends at runtime
- **CUDA-aware collectives** via MVAPICH-Plus GPUDirect RDMA
- **PyTorch-compatible API** — works with `torch.distributed`
- **InfiniBand and Slingshot** optimizations

### Installation

```bash
# Submodule already initialized in frameworks/mcr-dl
cd frameworks/mcr-dl
pip install -e .
```

### Verification

```bash
python -c "import mcr_dl; print('MCR-DL installed')"
```

See [../docs/build-guides/mcr-dl.md](../docs/build-guides/mcr-dl.md) for the full build guide.

## Performance Benchmarks

See [../benchmarks/README.md](../benchmarks/README.md) for comprehensive benchmarks.

**Quick benchmark**:
```bash
cd benchmarks/communication
python run_all.py --framework=deepspeed --num_gpus=4
```

---

---

## Support

- **HPC-AI Documentation**: [../docs/](../docs/)
- **Issues**: https://github.com/OSU-Nowlab/osu-hpc-ai/issues

**Maintained by**: Network-Based Computing Laboratory, The Ohio State University
