#!/bin/bash
#
# SGLang Build Script for MRI Cluster
# Builds SGLang with PyTorch 2.10.0 + MVAPICH-Plus + NCCL
#
# Prerequisites:
#   - GPU allocation: salloc -p devel -t 1:00:00 -C cuda -N 1 --gpus-per-node=1
#   - Build time: 5-10 minutes
#
#
set -e

echo "========================================"
echo "SGLang Build Script for MRI"
echo "========================================"
echo ""

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SGLANG_DIR="$REPO_ROOT/frameworks/sglang"

# System modules
echo "Step 1/7: Loading system modules..."
module load gcc/13.3.0 2>/dev/null || true
module load cuda/12.6 2>/dev/null || true

# Environment paths
export CUDA_HOME=/opt/cuda/12.6
export MVAPICH_HOME=/home/fritz.299/osu-hpc-ai-dev/install/mvapich-plus-4.1-cuda12.6
export PATH=$MVAPICH_HOME/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MVAPICH_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDA_HOME/include:$MVAPICH_HOME/include:$CPATH

echo "  GCC: $(gcc --version | head -n1)"
echo "  CUDA: $CUDA_HOME"
echo "  MVAPICH: $MVAPICH_HOME"
echo ""

# Activate conda environment
echo "Step 2/7: Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate hpc-ai-build

# Clean up Conda toolchain conflicts
echo "Step 3/7: Cleaning up conda compiler conflicts..."
unset _CONDA_PYTHON_SYSCONFIGDATA_NAME
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v "compiler_compat" | paste -sd:)
export LD_PRELOAD=/opt/gcc/13.3.0/lib64/libstdc++.so.6:$LD_PRELOAD

echo "  Python: $(python --version)"
echo "  Conda env: $CONDA_DEFAULT_ENV"
echo ""

# Install dependencies
echo "Step 4/7: Installing Python dependencies..."
cd "$SGLANG_DIR/python"
pip install -U pip setuptools wheel

# Check if torch is available
if ! python -c "import torch" 2>/dev/null; then
    echo "ERROR: PyTorch not found in hpc-ai-build environment"
    echo "Please install PyTorch with MVAPICH support first"
    exit 1
fi

echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Clean previous build
echo "Step 5/7: Cleaning previous build artifacts..."
rm -rf build dist *.egg-info
echo "  Build directory cleaned"
echo ""

# Build SGLang
echo "Step 6/7: Building SGLang..."
echo "  Note: Installing SGLang WITHOUT overwriting custom PyTorch"
echo ""

# Install core SGLang package without dependencies
pip install -e . --no-deps

# Install runtime dependencies one by one, SKIPPING torch/torchvision/triton
# These would overwrite our custom PyTorch with MVAPICH support
echo "  Installing runtime dependencies (excluding PyTorch/torchvision/triton)..."
pip install --no-deps blobfile build datasets fastapi hf_transfer huggingface_hub \
    interegular modelscope msgspec orjson packaging partial_json_parser pillow \
    prometheus-client pynvml python-multipart pyzmq uvicorn uvloop einops \
    transformers timm soundfile

# Install dependencies that have torch as a dependency, but skip torch itself
pip install --no-deps scipy compressed-tensors llguidance

# Install outlines carefully (it pulls in torch by default)
# We skip its torch dependency since we have custom PyTorch
pip install --no-deps outlines==0.1.11

# Install other transitive dependencies
pip install --no-deps aiohttp requests tqdm numpy ipython setproctitle \
    fsspec pyarrow dill httpx pandas lxml regex safetensors cuda-python \
    annotated-types pydantic-core typing-inspection jsonschema referencing \
    airportsdata cloudpickle diskcache lark nest-asyncio pycountry

echo "  Dependencies installed successfully"

echo ""
echo "========================================"
echo "Build Complete!"
echo "========================================"
echo ""
echo "Verifying installation..."
python -c "import sglang; print(f'  SGLang: {sglang.__version__}')"
echo ""
echo "Next steps:"
echo "  1. Run tests: python tests/sglang/test_installation.py"
echo "  2. Try examples: cd examples/sglang"
echo ""
