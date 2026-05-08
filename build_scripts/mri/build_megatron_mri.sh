#!/bin/bash
#
# Megatron-LM Setup Script for MRI Cluster
# Installs Megatron-LM on top of the custom PyTorch + MVAPICH-Plus stack.
#
# Prerequisites:
#   - PyTorch built with MVAPICH-Plus support
#   - Conda environment: hpc-ai-build
#   - Recommended allocation:
#     salloc -p devel -t 1:00:00 -C cuda -N 1 --gpus-per-node=1 --mem=64G
#
set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

WORK_DIR="${WORK_DIR:-${HOME}/osu-hpc-ai-dev}"
HPC_AI_ROOT="${HPC_AI_ROOT:-${HOME}/osu-hpc-ai}"
CUDA_VERSION="${CUDA_VERSION:-12.6}"
GCC_VERSION="${GCC_VERSION:-13.3.0}"
CONDA_ENV="${CONDA_ENV:-hpc-ai-build}"

MEGATRON_REPO="${MEGATRON_REPO:-https://github.com/NVIDIA/Megatron-LM.git}"
MEGATRON_REF="${MEGATRON_REF:-main}"
MEGATRON_SRC="${MEGATRON_SRC:-${WORK_DIR}/src/Megatron-LM}"
MVAPICH_HOME="${MVAPICH_HOME:-${WORK_DIR}/install/mvapich-plus-4.1-cuda${CUDA_VERSION}}"
NCCL_HOME="${NCCL_HOME:-${WORK_DIR}/libext/nccl/build}"

echo "========================================"
echo "Megatron-LM Setup Script for MRI"
echo "========================================"
echo ""
echo "Configuration:"
echo "  HPC-AI root:       ${HPC_AI_ROOT}"
echo "  Source directory:  ${MEGATRON_SRC}"
echo "  Megatron ref:      ${MEGATRON_REF}"
echo "  CUDA version:      ${CUDA_VERSION}"
echo "  GCC version:       ${GCC_VERSION}"
echo "  Conda environment: ${CONDA_ENV}"
echo ""

# ============================================================================
# MODULES AND ENVIRONMENT
# ============================================================================

echo "[1/6] Loading modules..."
module reset
module load "gcc/${GCC_VERSION}"
module load "cuda/${CUDA_VERSION}"
echo "  GCC:  $(which gcc)"
echo "  CUDA: ${CUDA_HOME:-/opt/cuda/${CUDA_VERSION}}"
echo ""

echo "[2/6] Activating Conda environment..."
MINICONDA_DIR="${WORK_DIR}/libext/miniconda3"
if [ -f "${MINICONDA_DIR}/bin/activate" ]; then
    source "${MINICONDA_DIR}/bin/activate"
    conda activate "${CONDA_ENV}"
else
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
fi
echo "  Python: $(python --version)"
echo "  Conda env: ${CONDA_DEFAULT_ENV}"
echo ""

echo "[3/6] Setting up runtime paths..."
export CUDA_HOME="${CUDA_HOME:-/opt/cuda/${CUDA_VERSION}}"
export PATH="${MVAPICH_HOME}/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MVAPICH_HOME}/lib:${CUDA_HOME}/lib64:${NCCL_HOME}/lib:${LD_LIBRARY_PATH:-}"
export CPATH="${CUDA_HOME}/include:${MVAPICH_HOME}/include:${NCCL_HOME}/include:${CPATH:-}"
export LIBRARY_PATH="${CUDA_HOME}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_PRELOAD="/opt/gcc/${GCC_VERSION}/lib64/libstdc++.so.6:${LD_PRELOAD:-}"
export MV2_USE_CUDA=1
export MV2_USE_GPUDIRECT_RDMA=1
export MV2_CUDA_USE_NAIVE=0

unset _CONDA_PYTHON_SYSCONFIGDATA_NAME
export PATH="$(echo "${PATH}" | tr ':' '\n' | grep -v "compiler_compat" | paste -sd:)"

echo "  MVAPICH_HOME: ${MVAPICH_HOME}"
echo "  NCCL_HOME:    ${NCCL_HOME}"
echo "  CUDA_HOME:    ${CUDA_HOME}"
echo ""

# ============================================================================
# PREREQUISITES
# ============================================================================

echo "[4/6] Verifying PyTorch stack..."
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"
python -c "import torch.distributed as dist; print(f'  MPI available: {dist.is_mpi_available()}')"
python -c "import torch.distributed as dist; print(f'  NCCL available: {dist.is_nccl_available()}')"
echo ""

# ============================================================================
# SOURCE CHECKOUT
# ============================================================================

echo "[5/6] Preparing Megatron-LM source..."
mkdir -p "$(dirname "${MEGATRON_SRC}")"
if [ -d "${MEGATRON_SRC}/.git" ]; then
    echo "  Existing checkout found. Updating ${MEGATRON_SRC}"
    git -C "${MEGATRON_SRC}" fetch origin "${MEGATRON_REF}" || git -C "${MEGATRON_SRC}" fetch origin
    git -C "${MEGATRON_SRC}" checkout "${MEGATRON_REF}"
    git -C "${MEGATRON_SRC}" pull --ff-only origin "${MEGATRON_REF}" || true
else
    echo "  Cloning ${MEGATRON_REPO}"
    git clone --recursive --branch "${MEGATRON_REF}" "${MEGATRON_REPO}" "${MEGATRON_SRC}"
fi
git -C "${MEGATRON_SRC}" submodule update --init --recursive
echo "  Source ready: ${MEGATRON_SRC}"
echo ""

# ============================================================================
# INSTALL
# ============================================================================

echo "[6/6] Installing Megatron-LM without replacing custom PyTorch..."
MEGATRON_RUNTIME_DEPS=(
    packaging
    setuptools
    wheel
    pybind11
    numpy
    pyyaml
    regex
    sentencepiece
    transformers
    tokenizers
    huggingface_hub
    safetensors
    tqdm
)
pip install --quiet --no-deps "${MEGATRON_RUNTIME_DEPS[@]}"
pip install -e "${MEGATRON_SRC}" --no-deps --no-build-isolation
echo ""

echo "Verifying Megatron-LM installation..."
python -c "import megatron.core; print('  Megatron Core import: OK')"
python -c "import importlib.util; spec = importlib.util.find_spec('megatron.training'); print('  Megatron training package:', 'OK' if spec else 'not present')"

echo ""
echo "========================================"
echo "Megatron-LM Setup Complete"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Run tests: HPC_AI_STRICT_MEGATRON=1 python -m pytest tests/megatron -q"
echo "  2. Install verl after vLLM and Megatron-LM are both available"
echo ""
