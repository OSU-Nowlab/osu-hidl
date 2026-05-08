# Llama-2 LoRA Fine-tuning with DeepSpeed

Fine-tune Llama-2 using LoRA (Low-Rank Adaptation) with DeepSpeed ZeRO optimization.

## Overview

This recipe demonstrates memory-efficient fine-tuning of large language models using:
- **LoRA**: Reduces trainable parameters by 99%+
- **DeepSpeed ZeRO-2**: Optimizer state partitioning across GPUs
- **FP16 Training**: Mixed precision for 2x memory savings

## Requirements

- 1+ NVIDIA GPU with 16GB+ VRAM (A100 recommended)
- DeepSpeed installed with HPC-AI stack
- ~10GB disk space for model weights

## Quick Start

```bash
# From the recipe directory
cd recipe/llama2-lora-finetune

# Run with defaults (uses TinyLlama for testing)
bash run.sh

# Run with Llama-2-7B (requires HuggingFace token)
bash run.sh --model meta-llama/Llama-2-7b-hf
```

## Files

| File | Description |
|------|-------------|
| `finetune.py` | Main training script |
| `ds_config.json` | DeepSpeed ZeRO-2 configuration |
| `run.sh` | Launcher script |

## Configuration

### Model Options

| Model | VRAM Required | Training Time |
|-------|---------------|---------------|
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | ~4GB | ~5 min |
| `meta-llama/Llama-2-7b-hf` | ~16GB | ~30 min |
| `meta-llama/Llama-2-13b-hf` | ~32GB (2 GPUs) | ~1 hour |

### LoRA Parameters

Default configuration in `finetune.py`:
- `lora_r=8`: Rank of LoRA matrices
- `lora_alpha=16`: Scaling factor
- `lora_dropout=0.05`: Dropout for regularization

## Expected Output

```
============================================================
Llama-2 LoRA Fine-tuning with DeepSpeed
============================================================
Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
LoRA rank: 8
DeepSpeed ZeRO stage: 2

Loading model...
Trainable parameters: 4,194,304 / 1,100,048,384 (0.38%)

Training...
Epoch 1/3 - Loss: 2.4521
Epoch 2/3 - Loss: 2.1893
Epoch 3/3 - Loss: 1.9764

Training complete!
Adapter saved to: ./output/lora_adapter
```

## Customization

### Use Your Own Dataset

Modify `finetune.py` to load your dataset:

```python
# Replace the synthetic dataset with your data
from datasets import load_dataset
dataset = load_dataset("your-dataset")
```

### Scale to Multiple GPUs

```bash
# 4 GPUs
deepspeed --num_gpus=4 finetune.py --deepspeed_config=ds_config.json
```

### Use ZeRO-3 for Larger Models

For 70B+ models, use ZeRO-3 with CPU offloading. Modify `ds_config.json`:

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu"},
    "offload_param": {"device": "cpu"}
  }
}
```

## Troubleshooting

**Out of memory**: Reduce `per_device_train_batch_size` or use ZeRO-3

**Slow training**: Ensure GPUDirect RDMA is enabled:
```bash
export MV2_USE_CUDA=1
export MV2_USE_GPUDIRECT_RDMA=1
```

---

**Maintained by**: NOWLAB Team, The Ohio State University
