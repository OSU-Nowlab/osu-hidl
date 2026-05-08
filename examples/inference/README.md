# Inference Examples

Example scripts demonstrating LLM inference with vLLM and SGLang.

---

## vLLM Examples

### Basic Inference

Simple text generation with a small model:

```bash
cd ~/osu-hpc-ai
python examples/inference/vllm_basic.py
```

**Features:**
- Uses `facebook/opt-125m` (small, fast download)
- Demonstrates basic prompt-response workflow
- Good for testing installation

---

### Llama-2 Inference

Text generation with Llama-2 models:

```bash
# Request GPU node
salloc -p devel -t 2:00:00 -C cuda -N 1 --gpus-per-node=2

# Run example
conda activate hpc-ai-build
python examples/inference/vllm_llama.py
```

**Requirements:**
- GPU with 14GB+ memory (for 7B model)
- HuggingFace token (for official Meta models)
- ~13GB download on first run

**Models:**
- `meta-llama/Llama-2-7b-hf` (requires HF token)
- `NousResearch/Llama-2-7b-hf` (open alternative)

---

### Multi-GPU Inference

Tensor parallelism across multiple GPUs:

```bash
# Request 2+ GPUs
salloc -p devel -t 2:00:00 -C cuda -N 1 --gpus-per-node=2

# Run example
conda activate hpc-ai-build
python examples/inference/vllm_multi_gpu.py
```

**Features:**
- Automatic tensor parallelism
- Higher throughput with batching
- Demonstrates multi-GPU efficiency

---

## Environment Setup

### On MRI Cluster

```bash
# Load modules
module load gcc/13.3.0
module load cuda/12.6

# Activate environment
conda activate hpc-ai-build

# Set CUDA paths
export CUDA_HOME=/opt/cuda/12.6
export LD_LIBRARY_PATH=/opt/cuda/12.6/lib64:$LD_LIBRARY_PATH
```

---

## Model Downloads

Models are downloaded from HuggingFace on first run:

| Model | Size | Memory | Download Time |
|-------|------|--------|---------------|
| `facebook/opt-125m` | 500MB | ~1GB | 1-2 min |
| `facebook/opt-1.3b` | 2.5GB | ~3GB | 3-5 min |
| `meta-llama/Llama-2-7b-hf` | 13GB | ~14GB | 10-20 min |

**Tip:** Download models beforehand:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "facebook/opt-125m"
AutoTokenizer.from_pretrained(model_name)
AutoModelForCausalLM.from_pretrained(model_name)
```

---

## Performance Tips

### Single GPU
- Use smaller models (125M - 1.3B)
- Increase `gpu_memory_utilization` (0.8-0.95)
- Reduce `max_num_seqs` if OOM

### Multi-GPU
- Use larger models (7B+) to benefit from parallelism
- Set `tensor_parallel_size` to number of GPUs
- Increase batch size (`max_num_seqs`)

### Memory Issues
```python
llm = LLM(
    model="facebook/opt-125m",
    gpu_memory_utilization=0.5,  # Reduce memory usage
    max_num_seqs=4  # Smaller batch size
)
```

---

## Troubleshooting

### "CUDA out of memory"
**Solution:** Use smaller model or reduce memory utilization

### "HuggingFace authentication required"
**Solution:** Set HF token:
```bash
export HF_TOKEN=your_token_here
```

### Model download slow/fails
**Solution:** Pre-download or use local model path:
```python
llm = LLM(model="/path/to/local/model")
```
