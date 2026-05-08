# SGLang Examples

This directory contains examples demonstrating how to use SGLang for efficient LLM inference and serving.

## Overview

SGLang (Structured Generation Language) provides a pythonic interface for programming LLMs with features like:
- Structured generation with control flow
- Efficient prefix caching (RadixAttention)
- Multi-GPU tensor parallelism
- OpenAI-compatible API server
- Multi-modal support

## Quick Start

### Prerequisites

```bash
# Ensure SGLang is installed
pip install sglang

# Or use the custom build
cd /path/to/osu-hpc-ai/frameworks/sglang/python
pip install -e .
```

### Running Examples

#### Simple Generation

```bash
python simple_generation.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

#### With Server Backend

```bash
# Terminal 1: Start SGLang server
python -m sglang.launch_server \
    --model-path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --host 127.0.0.1 \
    --port 30000

# Terminal 2: Run examples
python simple_generation.py --backend openai --port 30000
```

## Example Scripts

### 1. simple_generation.py

Demonstrates basic SGLang functionality:
- Simple text completion
- Multi-turn chat
- Structured generation with control flow

**Usage:**
```bash
python simple_generation.py [OPTIONS]

Options:
  --backend {runtime,openai}  Backend to use (default: runtime)
  --model MODEL               Model path or name
  --host HOST                 Server host for OpenAI backend
  --port PORT                 Server port for OpenAI backend
```

**Examples:**
```bash
# Use runtime backend with specific model
python simple_generation.py \
    --backend runtime \
    --model meta-llama/Llama-2-7b-chat-hf

# Use OpenAI-compatible server
python simple_generation.py \
    --backend openai \
    --host localhost \
    --port 30000
```

## Advanced Usage

### Multi-GPU Inference

For larger models, use tensor parallelism:

```bash
# Start server with 2 GPUs
python -m sglang.launch_server \
    --model-path meta-llama/Llama-2-13b-chat-hf \
    --tp-size 2 \
    --host 0.0.0.0 \
    --port 30000
```

### Structured Output

SGLang excels at structured generation:

```python
import sglang as sgl

@sgl.function
def generate_json(s, topic):
    s += f"Generate a JSON object about {topic}:\n"
    s += "{\n"
    s += '  "title": "'
    s += sgl.gen("title", max_tokens=20, stop='"')
    s += '",\n  "description": "'
    s += sgl.gen("description", max_tokens=100, stop='"')
    s += '"\n}'

state = generate_json.run(topic="machine learning")
print(state.text())
```

### Batch Processing

Process multiple requests efficiently:

```python
import sglang as sgl

@sgl.function
def classify_sentiment(s, text):
    s += f"Classify the sentiment of: '{text}'\n"
    s += "Sentiment: "
    s += sgl.gen("sentiment", max_tokens=5)

texts = [
    "I love this product!",
    "This is terrible.",
    "It's okay, nothing special."
]

# Batch processing
states = classify_sentiment.run_batch([{"text": t} for t in texts])
for text, state in zip(texts, states):
    print(f"{text} -> {state['sentiment']}")
```

## Server Configuration

### Basic Server Setup

```bash
python -m sglang.launch_server \
    --model-path <model-name-or-path> \
    --host 0.0.0.0 \
    --port 30000
```

### Production Settings

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-2-7b-chat-hf \
    --host 0.0.0.0 \
    --port 30000 \
    --tp-size 1 \
    --mem-fraction-static 0.85 \
    --max-running-requests 256 \
    --schedule-policy fcfs
```

### Multi-Node Setup

```bash
# Node 0
python -m sglang.launch_server \
    --model-path <model> \
    --tp-size 4 \
    --nnodes 2 \
    --node-rank 0 \
    --master-addr <node0-ip> \
    --master-port 12345

# Node 1
python -m sglang.launch_server \
    --model-path <model> \
    --tp-size 4 \
    --nnodes 2 \
    --node-rank 1 \
    --master-addr <node0-ip> \
    --master-port 12345
```

## API Usage

### Python Client

```python
import requests

# Completion
response = requests.post(
    "http://localhost:30000/v1/completions",
    json={
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "prompt": "Once upon a time",
        "max_tokens": 50
    }
)
print(response.json()["choices"][0]["text"])

# Chat
response = requests.post(
    "http://localhost:30000/v1/chat/completions",
    json={
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "messages": [
            {"role": "user", "content": "What is AI?"}
        ]
    }
)
print(response.json()["choices"][0]["message"]["content"])
```

### cURL

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

## Performance Tips

1. **Use appropriate batch size**: Increase `--max-running-requests` for higher throughput
2. **Tune memory**: Adjust `--mem-fraction-static` based on GPU memory
3. **Enable chunked prefill**: For mixed workloads with varying prompt lengths
4. **Use tensor parallelism**: Split large models across multiple GPUs
5. **Monitor with profiler**: Use `--enable-torch-profiler` for performance analysis

## Troubleshooting

### Out of Memory

```bash
# Reduce memory usage
python -m sglang.launch_server \
    --model-path <model> \
    --mem-fraction-static 0.7
```

### Slow First Request

This is normal - model loading and compilation take time. Subsequent requests are much faster due to caching.

### CUDA Errors

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Set visible devices
export CUDA_VISIBLE_DEVICES=0,1
```

## Additional Resources

- **SGLang Documentation**: https://docs.sglang.ai/
- **Backend Guide**: https://docs.sglang.ai/backend/
- **Frontend Guide**: https://docs.sglang.ai/frontend/
- **Build Guide**: `../../docs/build-guides/sglang.md`
- **Tests**: `../../tests/sglang/`

## Notes

- SGLang works with any PyTorch backend including custom MVAPICH builds
- The examples use small models by default for quick testing
- For production, use larger models with appropriate hardware
- SGLang's structured generation is particularly powerful for complex workflows
