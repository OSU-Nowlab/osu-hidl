# vLLM Inference and Serving

This guide covers running LLM inference and serving with vLLM using our custom PyTorch 2.10.0 build on HPC clusters.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Tensor Parallelism](#tensor-parallelism)
- [Throughput Tuning](#throughput-tuning)
- [OpenAI-Compatible API Server](#openai-compatible-api-server)
- [Running on SLURM](#running-on-slurm)
- [Common Patterns](#common-patterns)
- [Troubleshooting](#troubleshooting)
- [Additional Resources](#additional-resources)

## Overview

vLLM provides high-throughput LLM inference with:

- **PagedAttention**: Efficiently manages KV cache memory using paging, eliminating fragmentation and enabling dynamic batch sizes
- **Continuous batching**: Processes new requests as GPU capacity becomes available, rather than waiting for a full batch
- **Tensor parallelism**: Distributes model layers across multiple GPUs for large models
- **OpenAI-compatible API**: Drop-in replacement for OpenAI API clients

> **Official Documentation**: For cluster-specific build scripts and configurations, see the official HPC-AI userguide:
> [https://hpc-ai.engineering.osu.edu/userguide](https://hpc-ai.engineering.osu.edu/userguide)

## Prerequisites

### vLLM with Custom PyTorch

This stack provides vLLM built against our custom PyTorch 2.10.0 with MVAPICH-Plus. Using stock vLLM wheels may cause ABI incompatibilities.

- **Build Guide**: [docs/build-guides/vllm.md](../build-guides/vllm.md)
- **Repository**: [https://github.com/OSU-Nowlab/vllm](https://github.com/OSU-Nowlab/vllm)

### Verify Installation

```bash
python -c "import vllm; print('vLLM:', vllm.__version__)"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count())"
```

### GPU Allocation (MRI Cluster)

```bash
# Single GPU
salloc -p devel -t 2:00:00 -C cuda -N 1 --gpus-per-node=1

# Multiple GPUs for tensor parallelism
salloc -p devel -t 2:00:00 -C cuda -N 1 --gpus-per-node=4 --mem=128G
```

## Quick Start

### Basic Text Generation

```python
from vllm import LLM, SamplingParams

# Load model (downloads on first run)
llm = LLM(
    model="facebook/opt-125m",
    gpu_memory_utilization=0.85,
    max_num_seqs=8
)

# Configure generation
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100
)

# Generate (batched)
prompts = [
    "The key to distributed training is",
    "MVAPICH-Plus enables",
]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
    print("-" * 60)
```

```bash
# Set up environment (MRI cluster)
module load gcc/13.3.0 cuda/12.6
conda activate hpc-ai-build
export LD_PRELOAD=/opt/gcc/13.3.0/lib64/libstdc++.so.6:$LD_PRELOAD

python basic_inference.py
```

### Running the Included Examples

```bash
cd ~/osu-hpc-ai

# Basic inference (single GPU, small model)
python examples/inference/vllm_basic.py

# Multi-GPU inference
python examples/inference/vllm_multi_gpu.py
```

## Tensor Parallelism

Tensor parallelism shards model weight matrices across GPUs, enabling inference of models too large for a single GPU and increasing throughput via parallel computation.

### Selecting `tensor_parallel_size`

`tensor_parallel_size` must divide evenly into the number of attention heads in the model. Common values:

| Model size | Recommended TP | Minimum GPU memory per GPU |
|-----------|----------------|---------------------------|
| 7B | 1–2 | ~14GB → ~7GB with TP=2 |
| 13B | 2–4 | ~26GB → ~7GB with TP=4 |
| 70B | 4–8 | ~140GB → ~18GB with TP=8 |

### Usage

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=2,          # Use 2 GPUs
    dtype="float16",
    gpu_memory_utilization=0.85,
    max_num_seqs=16
)

outputs = llm.generate(
    ["Explain distributed training in one paragraph."],
    SamplingParams(max_tokens=200)
)
```

vLLM manages GPU placement automatically — no additional configuration is required when running on a single node.

### Pipeline Parallelism

For very large models across many GPUs, combine tensor and pipeline parallelism:

```python
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,
    pipeline_parallel_size=2,    # 4 TP * 2 PP = 8 GPUs total
    dtype="float16"
)
```

Pipeline parallelism splits model layers across stages. It reduces per-GPU memory further but adds inter-stage communication latency.

## Throughput Tuning

### Key Parameters

| Parameter | Effect | Starting value |
|-----------|--------|----------------|
| `max_num_seqs` | Max concurrent sequences in flight | 256 |
| `gpu_memory_utilization` | Fraction of GPU memory for KV cache | 0.85 |
| `max_model_len` | Maximum context length | Model default |
| `max_num_batched_tokens` | Max tokens processed per iteration | `max_num_seqs × max_model_len` |

### Maximize Throughput

```python
llm = LLM(
    model="facebook/opt-1.3b",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.90,    # More memory for KV cache
    max_num_seqs=256,               # Higher concurrency
    max_model_len=2048,             # Limit context to free KV cache space
    dtype="float16"
)
```

### Benchmark Throughput

```bash
# vLLM's built-in benchmark
python -m vllm.entrypoints.benchmark_throughput \
    --model facebook/opt-1.3b \
    --dataset ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 \
    --tensor-parallel-size 2
```

## OpenAI-Compatible API Server

vLLM can serve models via an HTTP API compatible with OpenAI clients.

### Start the Server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model facebook/opt-1.3b \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 \
    --host 0.0.0.0 \
    --port 8000
```

### Query the Server

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"   # vLLM doesn't require a real key
)

response = client.chat.completions.create(
    model="facebook/opt-1.3b",
    messages=[{"role": "user", "content": "What is MPI?"}],
    max_tokens=200,
    temperature=0.7
)

print(response.choices[0].message.content)
```

```bash
# Or with curl
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "facebook/opt-1.3b",
        "prompt": "What is MPI?",
        "max_tokens": 100
    }'
```

### Serving Behind a Compute Node

On HPC clusters, the API server runs on a compute node. Forward the port to your login session:

```bash
# On login node: forward port from compute node
ssh -L 8000:NODE_HOSTNAME:8000 login_node
```

## Running on SLURM

```bash
#!/bin/bash
#SBATCH -J vllm-serve
#SBATCH -N 1
#SBATCH --gpus-per-node=2
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -p gpu
#SBATCH -o slurm-%j.out

module reset
module load gcc/13.3.0
module load cuda/12.6

export CUDA_HOME=/opt/cuda/12.6
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_PRELOAD=/opt/gcc/13.3.0/lib64/libstdc++.so.6:$LD_PRELOAD

source ${HOME}/miniconda3/bin/activate hpc-ai-build

python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 \
    --host 0.0.0.0 \
    --port 8000
```

```bash
sbatch vllm_serve.slurm
```

## Common Patterns

### Batch Inference from File

```python
from vllm import LLM, SamplingParams
import json

llm = LLM(model="facebook/opt-1.3b", tensor_parallel_size=2)
sampling_params = SamplingParams(temperature=0.0, max_tokens=200)

with open("prompts.jsonl") as f:
    prompts = [json.loads(line)["prompt"] for line in f]

outputs = llm.generate(prompts, sampling_params)

with open("outputs.jsonl", "w") as f:
    for output in outputs:
        json.dump({
            "prompt": output.prompt,
            "generated": output.outputs[0].text
        }, f)
        f.write("\n")
```

### Greedy Decoding

```python
# temperature=0.0 gives greedy (deterministic) decoding
sampling_params = SamplingParams(temperature=0.0, max_tokens=200)
```

### Using a Local Model

```python
llm = LLM(
    model="/path/to/local/model",   # Local HuggingFace model directory
    tokenizer="/path/to/local/model"
)
```

### Pre-downloading Models

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="meta-llama/Llama-2-7b-hf",
    local_dir="/scratch/models/llama-2-7b",
    token="your_hf_token"
)
```

## Troubleshooting

### CUDA out of memory

1. Reduce `gpu_memory_utilization` (try 0.7 or 0.6)
2. Reduce `max_num_seqs`
3. Add more GPUs and increase `tensor_parallel_size`
4. Reduce `max_model_len` to limit KV cache size

### libstdc++ version mismatch

```bash
export LD_PRELOAD=/opt/gcc/13.3.0/lib64/libstdc++.so.6:$LD_PRELOAD
```

### HuggingFace authentication required

```bash
export HF_TOKEN=your_token_here
# Or: huggingface-cli login
```

### `tensor_parallel_size` error

**Error**: `ValueError: num_attention_heads must be divisible by tensor_parallel_size`

Reduce `tensor_parallel_size` to a value that divides the model's attention head count. For most Llama models: 32 heads → valid values are 1, 2, 4, 8.

### PyTorch ABI mismatch

If you installed vLLM from a stock wheel rather than building from source, you may see undefined symbol errors. Rebuild vLLM from source against the custom PyTorch 2.10.0 build following [docs/build-guides/vllm.md](../build-guides/vllm.md).

### Server not accepting connections

Verify the server is bound to `0.0.0.0` (not `127.0.0.1`) and that no firewall rules block the port between login and compute nodes.

## Additional Resources

- **vLLM Documentation**: [https://docs.vllm.ai/](https://docs.vllm.ai/)
- **PagedAttention Paper**: [https://arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)
- **MVAPICH-Plus User Guide**: [https://mvapich-docs.readthedocs.io/en/mvapich-plus/](https://mvapich-docs.readthedocs.io/en/mvapich-plus/)
- **Official HPC-AI Userguide**: [https://hpc-ai.engineering.osu.edu/userguide/](https://hpc-ai.engineering.osu.edu/userguide/)
- **vLLM Build Guide**: [../build-guides/vllm.md](../build-guides/vllm.md)
- **Examples**: [../../examples/inference/](../../examples/inference/)
