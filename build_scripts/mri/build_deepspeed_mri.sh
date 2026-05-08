#!/bin/bash
#
# DeepSpeed Build Script for MRI Cluster
# Builds DeepSpeed with PyTorch 2.10.0 + MVAPICH-Plus + NCCL
#
# Prerequisites:
#   - GPU allocation: salloc -p devel -t 2:00:00 -C cuda -N 1 --gpus-per-node=1
#   - Build time: 20-30 minutes
#
set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

WORK_DIR="${HOME}/osu-hpc-ai-dev"
CUDA_VERSION="12.6"
GCC_VERSION="13.3.0"
CONDA_ENV="hpc-ai-build"

# Paths
HPC_AI_ROOT="${HOME}/osu-hpc-ai"
SRC_DIR="${HPC_AI_ROOT}/frameworks/deepspeed"
LOG_DIR="${WORK_DIR}/logs/deepspeed"
MVAPICH_HOME="${WORK_DIR}/install/mvapich-plus-4.1-cuda${CUDA_VERSION}"

# Build configuration
MAX_JOBS=16

# ============================================================================
# SETUP
# ============================================================================

echo "========================================"
echo "DeepSpeed Build Script for MRI"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Source Directory:  ${SRC_DIR}"
echo "  CUDA Version:      ${CUDA_VERSION}"
echo "  GCC Version:       ${GCC_VERSION}"
echo "  Conda Environment: ${CONDA_ENV}"
echo "  MVAPICH Home:      ${MVAPICH_HOME}"
echo ""

# Create log directory
mkdir -p "${LOG_DIR}"

# ============================================================================
# MODULE LOADING
# ============================================================================

echo "[1/5] Loading modules..."
module reset
module load gcc/${GCC_VERSION}
module load cuda/${CUDA_VERSION}

echo "  GCC:  $(which gcc)"
echo "  CUDA: ${CUDA_HOME}"
echo ""

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

echo "[2/5] Setting up environment..."

# CUDA paths
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# MVAPICH-Plus paths
if [ -d "${MVAPICH_HOME}" ]; then
    export PATH="${MVAPICH_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${MVAPICH_HOME}/lib:${LD_LIBRARY_PATH}"
    export CPATH="${MVAPICH_HOME}/include:${CPATH}"
    echo "  MVAPICH: ${MVAPICH_HOME}"
else
    echo "  ERROR: MVAPICH-Plus not found at ${MVAPICH_HOME}"
    echo "  Build it first using build_mvapich_mri.sh"
    exit 1
fi

# Set MPI compiler wrappers
export CC="$(which mpicc)"
export CXX="$(which mpicxx)"

echo "  MPI CC:  ${CC}"
echo "  MPI CXX: ${CXX}"
echo ""

# ============================================================================
# CONDA ENVIRONMENT
# ============================================================================

echo "[3/5] Activating Conda environment..."

MINICONDA_DIR="${WORK_DIR}/libext/miniconda3"

if [ -f "${MINICONDA_DIR}/bin/activate" ]; then
    source "${MINICONDA_DIR}/bin/activate"
    conda activate "${CONDA_ENV}"
    echo "  Conda env: ${CONDA_ENV}"
else
    echo "  ERROR: Miniconda not found"
    exit 1
fi

# Verify PyTorch is installed
if ! python -c "import torch" 2>/dev/null; then
    echo "  ERROR: PyTorch not installed"
    echo "  Build PyTorch first using build_pytorch_mri.sh"
    exit 1
fi

TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
echo "  PyTorch: ${TORCH_VERSION}"

# Verify MPI backend
MPI_AVAILABLE=$(python -c "import torch.distributed as dist; print(dist.is_mpi_available())" 2>/dev/null || echo "False")
if [ "$MPI_AVAILABLE" != "True" ]; then
    echo "  WARNING: PyTorch MPI backend not available"
    echo "  DeepSpeed may fall back to NCCL"
fi

echo ""

# ============================================================================
# BUILD DEEPSPEED
# ============================================================================

echo "[4/5] Building DeepSpeed..."
echo "  This will take ~20-30 minutes..."
echo ""

cd "${SRC_DIR}"

# Clean previous builds
echo "  Cleaning previous builds..."
rm -rf build dist *.egg-info
python setup.py clean || true

# DeepSpeed build options
export DS_BUILD_OPS=1                    # Build custom CUDA ops
export DS_BUILD_FUSED_ADAM=1            # Fused Adam optimizer
export DS_BUILD_CPU_ADAM=1              # CPU Adam (for offload)
export DS_BUILD_UTILS=1                 # Utility ops
export DS_BUILD_FUSED_LAMB=1            # Fused LAMB optimizer
export DS_BUILD_TRANSFORMER=1           # Transformer kernels
export DS_BUILD_STOCHASTIC_TRANSFORMER=1
export DS_BUILD_SPARSE_ATTN=0           # Disable if build issues

# CUDA architecture
export TORCH_CUDA_ARCH_LIST="8.0;9.0"   # A100 and H100

# Parallel build
export MAX_JOBS=${MAX_JOBS}

echo "  Build Configuration:"
echo "    DS_BUILD_OPS:                    ${DS_BUILD_OPS}"
echo "    DS_BUILD_FUSED_ADAM:             ${DS_BUILD_FUSED_ADAM}"
echo "    DS_BUILD_CPU_ADAM:               ${DS_BUILD_CPU_ADAM}"
echo "    DS_BUILD_TRANSFORMER:            ${DS_BUILD_TRANSFORMER}"
echo "    TORCH_CUDA_ARCH_LIST:            ${TORCH_CUDA_ARCH_LIST}"
echo "    MAX_JOBS:                        ${MAX_JOBS}"
echo ""

START_TIME=$(date +%s)

# Install DeepSpeed in editable mode (--no-deps to preserve custom PyTorch)
echo "  Installing DeepSpeed (logs: ${LOG_DIR}/install.log)..."
pip install -e . --no-deps > "${LOG_DIR}/install.log" 2>&1

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))

echo "  Build complete in ${ELAPSED_MIN} minutes"
echo "  Install log: ${LOG_DIR}/install.log"
echo ""

# ============================================================================
# VERIFICATION
# ============================================================================

echo "[5/5] Verifying installation..."

# Check DeepSpeed import
if python -c "import deepspeed" 2>/dev/null; then
    DS_VERSION=$(python -c "import deepspeed; print(deepspeed.__version__)")
    echo "DeepSpeed installed: ${DS_VERSION}"
else
    echo "DeepSpeed import failed"
    exit 1
fi

# Check ds_report
if command -v ds_report &> /dev/null; then
    echo "ds_report available"
    echo ""
    echo "DeepSpeed Environment Report:"
    echo "----------------------------"
    ds_report | head -20
else
    echo "Warning: ds_report command not found"
fi

echo ""

# ============================================================================
# SUMMARY
# ============================================================================

echo "========================================"
echo "Build Complete!"
echo "========================================"
echo ""
echo "Verifying installation..."
python -c "import deepspeed; print(f'  DeepSpeed: {deepspeed.__version__}')"
echo ""
echo "Next steps:"
echo "  1. Run tests: python tests/deepspeed/test_installation.py"
echo "  2. Try examples: cd examples/deepspeed"
echo ""
