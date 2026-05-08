# vLLM vs SGLang Serving Comparison

Compare throughput and latency between vLLM and SGLang inference engines.

## Overview

This recipe benchmarks two popular LLM serving frameworks:
- **vLLM**: PagedAttention for efficient KV cache management
- **SGLang**: RadixAttention for prefix caching

## Requirements

- 1+ NVIDIA GPU with 16GB+ VRAM
- vLLM and SGLang installed with HPC-AI stack
- ~5GB disk space for test model

## Quick Start

```bash
# From the recipe directory
cd recipe/serving-comparison

# Run comparison with default model (OPT-1.3B)
bash run.sh

# Run with custom model
bash run.sh --model facebook/opt-6.7b
```

## Files

| File | Description |
|------|-------------|
| `benchmark_vllm.py` | vLLM throughput benchmark |
| `benchmark_sglang.py` | SGLang throughput benchmark |
| `compare.py` | Run both and compare results |
| `run.sh` | Launcher script |

## Expected Output

```
============================================================
LLM Serving Framework Comparison
============================================================
Model: facebook/opt-1.3b
Prompts: 50
Max tokens: 128

Running vLLM benchmark...
  Throughput: 1,234.5 tokens/sec
  Avg latency: 103.2 ms/request

Running SGLang benchmark...
  Throughput: 1,456.7 tokens/sec
  Avg latency: 87.6 ms/request

============================================================
Results Summary
============================================================
| Metric          | vLLM      | SGLang    | Winner  |
|-----------------|-----------|-----------|---------|
| Throughput      | 1,234 t/s | 1,457 t/s | SGLang  |
| Avg Latency     | 103.2 ms  | 87.6 ms   | SGLang  |
| Memory Usage    | 12.3 GB   | 11.8 GB   | SGLang  |
```

## Configuration

### Benchmark Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | `facebook/opt-1.3b` | Model to benchmark |
| `--num_prompts` | 50 | Number of prompts to process |
| `--max_tokens` | 128 | Max tokens per generation |
| `--warmup` | 5 | Warmup iterations |

### Multi-GPU

Both frameworks support tensor parallelism:

```bash
# vLLM with 2 GPUs
python benchmark_vllm.py --tensor_parallel_size 2

# SGLang with 2 GPUs
python benchmark_sglang.py --tp_size 2
```

## Interpreting Results

- **Throughput**: Higher is better (tokens generated per second)
- **Latency**: Lower is better (time per request)
- **Memory**: Lower is better (GPU memory usage)

Results vary based on:
- Model size
- Prompt length
- Batch size
- Hardware configuration

## Troubleshooting

**vLLM import error**: Ensure vLLM is installed:
```bash
pip install vllm
```

**SGLang import error**: Ensure SGLang is installed:
```bash
pip install sglang
```

**Out of memory**: Use a smaller model or reduce `--num_prompts`

---

**Maintained by**: NOWLAB Team, The Ohio State University
