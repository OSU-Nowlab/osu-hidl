#!/bin/bash
#
# vLLM Build Script for MRI Cluster
# Builds vLLM from source with PyTorch 2.10.0 + MVAPICH-Plus + NCCL
#
# Requirements:
#   - At least 128GB RAM (use: salloc --mem=128G)
#   - Single-threaded build (MAX_JOBS=1) to avoid OOM
#
set -e

echo "========================================"
echo "vLLM Build Script for MRI"
echo "========================================"
echo ""

# Step 1/7: Load modules
echo "Step 1/7: Loading modules..."
module reset 2>/dev/null || true
module load gcc/13.3.0 2>/dev/null || true
module load cuda/12.6 2>/dev/null || true
echo "  Loaded: gcc/13.3.0, cuda/12.6"
echo ""

# Step 2/7: Activate conda environment
echo "Step 2/7: Activating conda environment..."
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "hpc-ai-build" ]; then
    echo "  Activating hpc-ai-build environment..."
    eval "$(conda shell.bash hook)"
    conda activate hpc-ai-build
else
    echo "  Already in hpc-ai-build environment"
fi
echo ""

# Step 3/7: Set up environment variables
echo "Step 3/7: Setting up environment variables..."
export MVAPICH_HOME=~/osu-hpc-ai-dev/install/mvapich-plus-4.1-cuda12.6
export CUDA_HOME=/opt/cuda/12.6
export NCCL_HOME=~/osu-hpc-ai-dev/libext/nccl/build

export PATH=$MVAPICH_HOME/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MVAPICH_HOME/lib:$CUDA_HOME/lib64:$NCCL_HOME/lib:$LD_LIBRARY_PATH
export CPATH=$CUDA_HOME/include:$MVAPICH_HOME/include:$NCCL_HOME/include:$CPATH

# CUDA toolchain
export CUDACXX=$CUDA_HOME/bin/nvcc
export CMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc
export CUDAToolkit_ROOT=$CUDA_HOME
export LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$LIBRARY_PATH

# Clean conda toolchain
unset _CONDA_PYTHON_SYSCONFIGDATA_NAME
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v "compiler_compat" | paste -sd:)

# Force correct libstdc++ (GCC 13.3.0)
export LD_PRELOAD=/opt/gcc/13.3.0/lib64/libstdc++.so.6:$LD_PRELOAD

# Use disk-backed tmp directory (critical for memory stability)
# /tmp is tmpfs (RAM) on MRI - NVCC writes multi-GB intermediate files
export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR

# Disable precompiled wheels (force source build)
unset VLLM_USE_PRECOMPILED
unset VLLM_PRECOMPILED_WHEEL_LOCATION

echo "  MVAPICH_HOME: $MVAPICH_HOME"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  NCCL_HOME: $NCCL_HOME"
echo "  TMPDIR: $TMPDIR (disk-backed, not RAM)"
echo ""

# Step 4/7: Verify prerequisites
echo "Step 4/7: Verifying prerequisites..."
if [ ! -d "$MVAPICH_HOME" ]; then
    echo "  ERROR: MVAPICH-Plus not found at $MVAPICH_HOME"
    exit 1
fi

if [ ! -d "$NCCL_HOME" ]; then
    echo "  ERROR: NCCL not found at $NCCL_HOME"
    exit 1
fi

PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null) || {
    echo "  ERROR: PyTorch not found. Build PyTorch first!"
    exit 1
}
echo "  PyTorch: $PYTORCH_VERSION"
echo "  MVAPICH-Plus: $(ls -d $MVAPICH_HOME 2>/dev/null | head -1)"
echo "  NCCL: $(ls $NCCL_HOME/lib/libnccl.so* 2>/dev/null | head -1)"
echo ""

# Step 4.5/7: Add PyTorch libs to LD_LIBRARY_PATH (critical for ABI compatibility)
echo "Step 4.5/7: Adding PyTorch lib directory to LD_LIBRARY_PATH..."
export TORCH_LIB_DIR=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH=$TORCH_LIB_DIR:$LD_LIBRARY_PATH
echo "  TORCH_LIB_DIR: $TORCH_LIB_DIR"
echo ""

# Step 5/7: Clean previous build
echo "Step 5/7: Cleaning previous build..."
cd ~/osu-hpc-ai/frameworks/vllm

rm -rf build/ dist/ vllm.egg-info/ _skbuild/ || true
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.so" -delete 2>/dev/null || true
echo "  Build directory cleaned"
echo ""

# Step 6/7: Install dependencies (with --no-deps to protect PyTorch)
echo "Step 6/7: Installing vLLM dependencies..."
pip install --quiet --no-deps \
    ray psutil sentencepiece numpy transformers \
    fastapi uvicorn pydantic aioprometheus prometheus-client \
    tiktoken lm-format-enforcer outlines typing-extensions \
    filelock pyzmq msgspec pandas mistral-common gguf \
    aiohttp requests pillow scipy packaging huggingface_hub
echo "  Dependencies installed"
echo ""

# Step 7/7: Build vLLM from source
echo "Step 7/7: Building vLLM from source..."
echo "  Configuration:"
echo "    MAX_JOBS: 1 (serial build to avoid OOM)"
echo "    TORCH_CUDA_ARCH_LIST: 8.0 (A100 only)"
echo "    ATTENTION_BACKEND: Default (will auto-select)"
echo "    FP8: Disabled for this build configuration"
echo "    VERBOSE: Enabled (CMake + Ninja output)"
echo ""
echo "  This will take 30-60 minutes (serial build)..."
echo "  Watch real-time: tail -f ~/vllm_build.log"
echo ""

# Enable verbose build output
export MAX_JOBS=1
export CMAKE_ARGS="-DENABLE_FP8=OFF"
export TORCH_CUDA_ARCH_LIST="8.0"
export USE_ROCM=0
export SKBUILD_VERBOSE=1
export CMAKE_VERBOSE_MAKEFILE=ON
export NINJA_STATUS="[%f/%t] "

pip install -e . --no-deps --no-build-isolation -v 2>&1 | tee ~/vllm_build.log

echo ""
echo "========================================"
echo "Build Complete!"
echo "========================================"
echo ""
echo "Verifying installation..."
python -c "import vllm; print(f'  vLLM: {vllm.__version__}')"
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
echo ""
echo "Next steps:"
echo "  1. Run tests: python tests/vllm/test_installation.py"
echo "  2. Try examples: cd examples/inference"
echo "  3. Cleanup: rm -rf \$HOME/tmp/*"
echo ""
