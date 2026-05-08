#!/bin/bash
#
# Llama-2 LoRA Fine-tuning Launcher
#
# Usage:
#   bash run.sh                                    # Test with TinyLlama
#   bash run.sh --model meta-llama/Llama-2-7b-hf  # Use Llama-2-7B
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default to 1 GPU, or use all available
NUM_GPUS=${NUM_GPUS:-1}

echo "============================================================"
echo "Llama-2 LoRA Fine-tuning Recipe"
echo "============================================================"
echo "Using $NUM_GPUS GPU(s)"
echo ""

# Check for required packages
python -c "import transformers, peft, deepspeed" 2>/dev/null || {
    echo "Missing required packages. Install with:"
    echo "  pip install transformers peft datasets"
    exit 1
}

# Run training
deepspeed --num_gpus="$NUM_GPUS" finetune.py \
    --deepspeed_config=ds_config.json \
    "$@"
