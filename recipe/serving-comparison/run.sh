#!/bin/bash
#
# vLLM vs SGLang Comparison Launcher
#
# Usage:
#   bash run.sh                              # Run comparison
#   bash run.sh --model facebook/opt-6.7b   # Use larger model
#   bash run.sh --vllm_only                 # Only test vLLM
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "vLLM vs SGLang Serving Comparison Recipe"
echo "============================================================"
echo ""

# Check for at least one framework
VLLM_INSTALLED=$(python -c "import vllm; print('yes')" 2>/dev/null || echo "no")
SGLANG_INSTALLED=$(python -c "import sglang; print('yes')" 2>/dev/null || echo "no")

if [ "$VLLM_INSTALLED" = "no" ] && [ "$SGLANG_INSTALLED" = "no" ]; then
    echo "Neither vLLM nor SGLang is installed."
    echo "Install at least one with:"
    echo "  pip install vllm"
    echo "  pip install sglang"
    exit 1
fi

echo "vLLM installed: $VLLM_INSTALLED"
echo "SGLang installed: $SGLANG_INSTALLED"
echo ""

# Run comparison
python compare.py "$@"
