# SGLang Structured Generation and Serving

This guide covers using SGLang for structured LLM generation, efficient serving, and multi-GPU inference on HPC clusters.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Structured Generation](#structured-generation)
- [Grammar Constraints](#grammar-constraints)
- [Efficient Sampling](#efficient-sampling)
- [Launching the Server](#launching-the-server)
- [Multi-GPU Serving](#multi-gpu-serving)
- [Running on SLURM](#running-on-slurm)
- [Common Patterns](#common-patterns)
- [Troubleshooting](#troubleshooting)
- [Additional Resources](#additional-resources)

## Overview

SGLang provides a Pythonic interface for LLM generation with:

- **`@sgl.function` decorator**: Express multi-step prompts as regular Python functions with variables, loops, and branches
- **RadixAttention**: Efficient KV cache reuse across requests sharing a common prefix
- **Structured generation**: Constrain outputs to JSON schemas, regex patterns, or choice sets
- **OpenAI-compatible API**: Drop-in replacement for OpenAI API clients
- **Multi-GPU tensor parallelism**: Scale large models across GPUs

> **Official Documentation**: For cluster-specific build scripts and configurations, see the official HPC-AI userguide:
> [https://hpc-ai.engineering.osu.edu/userguide/](https://hpc-ai.engineering.osu.edu/userguide)

## Prerequisites

### SGLang with Custom PyTorch

This stack provides SGLang built against our custom PyTorch 2.10.0 with MVAPICH-Plus.

- **Build Guide**: [docs/build-guides/sglang.md](../build-guides/sglang.md)
- **Repository**: [https://github.com/OSU-Nowlab/sglang](https://github.com/OSU-Nowlab/sglang)

### Verify Installation

```bash
python -c "import sglang; print('SGLang:', sglang.__version__)"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### GPU Allocation (MRI Cluster)

```bash
salloc -p devel -t 2:00:00 -C cuda -N 1 --gpus-per-node=1
```

Set up environment:

```bash
module reset
module load gcc/13.3.0
module load cuda/12.6
export LD_PRELOAD=/opt/gcc/13.3.0/lib64/libstdc++.so.6:$LD_PRELOAD
conda activate hpc-ai-build
```

## Quick Start

### Runtime Backend (No Server Required)

```python
import sglang as sgl

@sgl.function
def simple_generation(s, prompt):
    s += prompt
    s += sgl.gen("output", max_tokens=100, temperature=0.8)

# Set backend with model
sgl.set_default_backend(sgl.RuntimeEndpoint("TinyLlama/TinyLlama-1.1B-Chat-v1.0"))

state = simple_generation.run(prompt="Distributed computing enables")
print(state["output"])
```

### Run the Included Example

```bash
cd ~/osu-hpc-ai
python examples/sglang/simple_generation.py \
    --backend runtime \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

## Structured Generation

SGLang's `@sgl.function` decorator lets you write multi-step prompts as Python functions. Generation calls (`sgl.gen(...)`) produce named outputs that you can access after execution.

### Multi-Step Prompts

```python
import sglang as sgl

@sgl.function
def structured_report(s, topic):
    s += f"Generate a technical report about {topic}.\n\n"

    s += "Title: "
    s += sgl.gen("title", max_tokens=20, stop="\n")
    s += "\n\n"

    s += "Summary:\n"
    s += sgl.gen("summary", max_tokens=150)

    s += "\n\nKey Findings:\n"
    for i in range(3):
        s += f"{i + 1}. "
        s += sgl.gen(f"finding_{i}", max_tokens=50, stop="\n")
        s += "\n"

state = structured_report.run(topic="CUDA-aware MPI")
print("Title:", state["title"])
print("Summary:", state["summary"])
for i in range(3):
    print(f"Finding {i+1}:", state[f"finding_{i}"])
```

### Multi-Turn Chat

```python
@sgl.function
def multi_turn_chat(s, question):
    s += sgl.system("You are a helpful HPC assistant.")
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=200))

state = multi_turn_chat.run(question="What is MVAPICH-Plus?")
print(state["answer"])
```

### Conditional Generation

```python
@sgl.function
def classify_and_explain(s, text):
    s += f"Classify the sentiment of: '{text}'\n"
    s += "Sentiment (positive/negative/neutral): "
    s += sgl.gen("sentiment", max_tokens=5, stop="\n")

    if "positive" in state["sentiment"].lower():
        s += "\nExplain what is positive about it: "
        s += sgl.gen("explanation", max_tokens=100)
```

> **Note**: Conditional logic on generated values requires the server backend (not runtime) because values must be available before branching.

### Batch Processing

```python
@sgl.function
def summarize(s, text):
    s += f"Summarize in one sentence:\n{text}\nSummary: "
    s += sgl.gen("summary", max_tokens=80, stop="\n")

documents = [
    {"text": "MPI enables message passing across distributed processes..."},
    {"text": "GPUDirect RDMA bypasses the CPU during GPU-to-GPU transfers..."},
]

states = summarize.run_batch(documents, num_threads=4)
for doc, state in zip(documents, states):
    print(state["summary"])
```

`run_batch` parallelizes requests using `num_threads` threads — useful for high-throughput offline processing.

## Grammar Constraints

SGLang can constrain model outputs to structured formats using `regex`, `choices`, or JSON schema.

### Choice Constraints

Force the model to output exactly one of a set of options:

```python
@sgl.function
def classify(s, text):
    s += f"Text: {text}\nSentiment: "
    s += sgl.gen("label", choices=["positive", "negative", "neutral"])

state = classify.run(text="This cluster is incredibly fast!")
print(state["label"])   # Always one of: positive, negative, neutral
```

### Regex Constraints

Constrain output to match a regular expression pattern:

```python
@sgl.function
def extract_number(s, text):
    s += f"Extract the GPU count from: '{text}'\nGPU count: "
    s += sgl.gen("count", regex=r"\d+")

state = extract_number.run(text="We used 8 A100 GPUs for this experiment.")
print(state["count"])   # A string of digits
```

### JSON Schema Constraints

Constrain output to a valid JSON object matching a schema:

```python
import json

schema = json.dumps({
    "type": "object",
    "properties": {
        "model": {"type": "string"},
        "gpus": {"type": "integer"},
        "throughput_tps": {"type": "number"}
    },
    "required": ["model", "gpus", "throughput_tps"]
})

@sgl.function
def extract_benchmark(s, text):
    s += f"Extract benchmark results from: '{text}'\n"
    s += "JSON: "
    s += sgl.gen("result", json_schema=schema)

state = extract_benchmark.run(
    text="Llama-2-7B achieved 1200 tokens/sec on 2 A100 GPUs."
)
print(json.loads(state["result"]))
```

## Efficient Sampling

### Temperature and Top-P

```python
# Greedy (deterministic)
sgl.gen("output", max_tokens=100, temperature=0.0)

# Sampling (creative)
sgl.gen("output", max_tokens=100, temperature=0.8, top_p=0.95)

# Conservative
sgl.gen("output", max_tokens=100, temperature=0.3, top_p=0.9)
```

### Stop Tokens

Stop generation at specific strings rather than hitting `max_tokens`:

```python
# Stop at newline
sgl.gen("line", max_tokens=100, stop="\n")

# Stop at any of multiple strings
sgl.gen("section", max_tokens=500, stop=["###", "\n\n", "END"])
```

### Fork and Merge

Generate multiple completions from a shared prefix in parallel:

```python
@sgl.function
def generate_variants(s, prompt):
    s += prompt
    forks = s.fork(3)                     # 3 parallel continuations
    for fork in forks:
        fork += sgl.gen("variant", max_tokens=100, temperature=0.9)
    s.join(forks)

state = generate_variants.run(prompt="The future of HPC is")
for i, fork in enumerate(state.forks):
    print(f"Variant {i+1}:", fork["variant"])
```

## Launching the Server

For production serving and to enable conditional branching, start the SGLang server separately.

### Start the Server

```bash
python -m sglang.launch_server \
    --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --host 0.0.0.0 \
    --port 30000 \
    --mem-fraction-static 0.85 \
    --max-running-requests 256
```

### Connect a Client

```python
import sglang as sgl

sgl.set_default_backend(sgl.OpenAI("http://localhost:30000/v1"))

@sgl.function
def generate(s, prompt):
    s += prompt
    s += sgl.gen("output", max_tokens=100)

state = generate.run(prompt="Distributed training with MPI enables")
print(state["output"])
```

### OpenAI-Compatible API Requests

```bash
# Completion
curl http://localhost:30000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "prompt": "Hello", "max_tokens": 50}'

# Chat
curl http://localhost:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "messages": [{"role": "user", "content": "What is MPI?"}]}'
```

## Multi-GPU Serving

Scale large models across multiple GPUs using tensor parallelism.

### Single Node

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-2-13b-chat-hf \
    --tp-size 2 \
    --host 0.0.0.0 \
    --port 30000 \
    --mem-fraction-static 0.85
```

### Multi-Node

```bash
# Node 0 (master)
python -m sglang.launch_server \
    --model-path meta-llama/Llama-2-70b-hf \
    --tp-size 4 \
    --nnodes 2 \
    --node-rank 0 \
    --master-addr $(hostname) \
    --master-port 12345 \
    --port 30000

# Node 1
python -m sglang.launch_server \
    --model-path meta-llama/Llama-2-70b-hf \
    --tp-size 4 \
    --nnodes 2 \
    --node-rank 1 \
    --master-addr NODE0_HOSTNAME \
    --master-port 12345 \
    --port 30000
```

After both nodes start, send requests to node 0's port 30000.

## Running on SLURM

### Batch Job

```bash
#!/bin/bash
#SBATCH -J sglang-serve
#SBATCH -N 1
#SBATCH --gpus-per-node=2
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -p gpu
#SBATCH -o slurm-%j.out

module reset
module load gcc/13.3.0
module load cuda/12.6
export LD_PRELOAD=/opt/gcc/13.3.0/lib64/libstdc++.so.6:$LD_PRELOAD

source ${HOME}/miniconda3/bin/activate hpc-ai-build

python -m sglang.launch_server \
    --model-path meta-llama/Llama-2-7b-chat-hf \
    --tp-size 2 \
    --host 0.0.0.0 \
    --port 30000 \
    --mem-fraction-static 0.85 \
    --max-running-requests 256
```

```bash
sbatch sglang_serve.slurm
```

## Common Patterns

### High-Throughput Offline Inference

```python
import sglang as sgl

sgl.set_default_backend(sgl.RuntimeEndpoint("facebook/opt-1.3b"))

@sgl.function
def process(s, text):
    s += f"Summarize: {text}\nSummary: "
    s += sgl.gen("summary", max_tokens=100, stop="\n")

with open("inputs.txt") as f:
    items = [{"text": line.strip()} for line in f]

states = process.run_batch(items, num_threads=8)
summaries = [s["summary"] for s in states]
```

### RadixAttention for Shared Prefixes

SGLang automatically reuses KV cache for prompts that share a common prefix. Structure your prompts to maximize prefix sharing:

```python
@sgl.function
def qa_with_context(s, context, question):
    # This context prefix is shared across many questions
    s += f"Context:\n{context}\n\nQuestion: {question}\nAnswer: "
    s += sgl.gen("answer", max_tokens=200)

context = "MVAPICH-Plus is a CUDA-aware MPI implementation..."
questions = ["What is MVAPICH-Plus?", "What is CUDA-aware MPI?", ...]

states = qa_with_context.run_batch(
    [{"context": context, "question": q} for q in questions]
)
```

When multiple requests share the same `context`, RadixAttention serves subsequent requests without recomputing the shared prefix.

## Troubleshooting

### Server fails to start

Check that the port is free and the model path is correct:

```bash
lsof -i :30000    # Check if port is in use
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')"
```

### CUDA out of memory

Reduce `--mem-fraction-static` (try 0.7) or reduce `--max-running-requests`.

### Slow first request

Expected — model loading and CUDA kernel compilation happen on the first request. Subsequent requests use cached kernels and are much faster.

### libstdc++ import error

```bash
export LD_PRELOAD=/opt/gcc/13.3.0/lib64/libstdc++.so.6:$LD_PRELOAD
```

### `choices` constraint not working

Ensure the server is running with a version of SGLang that supports constrained decoding. Check: `python -c "import sglang; print(sglang.__version__)"` and verify against [docs/build-guides/sglang.md](../build-guides/sglang.md).

### Conditional branch does not execute

Conditional logic on generated values (`if state["label"] == "positive"`) requires the **server backend** (`sgl.OpenAI(...)`), not the runtime backend, because the runtime evaluates `@sgl.function` eagerly.

## Additional Resources

- **SGLang Documentation**: [https://docs.sglang.ai/](https://docs.sglang.ai/)
- **RadixAttention Paper**: [https://arxiv.org/abs/2312.07104](https://arxiv.org/abs/2312.07104)
- **Official HPC-AI Userguide**: [https://hpc-ai.engineering.osu.edu/userguide/](https://hpc-ai.engineering.osu.edu/userguide/)
- **SGLang Build Guide**: [../build-guides/sglang.md](../build-guides/sglang.md)
- **Examples**: [../../examples/sglang/](../../examples/sglang/)
