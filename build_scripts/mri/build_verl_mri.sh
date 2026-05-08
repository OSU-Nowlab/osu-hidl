#!/bin/bash
#
# verl Setup Script for MRI Cluster
# Installs verl on top of custom PyTorch, Megatron-LM, and vLLM.
#
# Prerequisites:
#   - PyTorch built with MVAPICH-Plus support
#   - vLLM built against the custom PyTorch environment
#   - Megatron-LM installed with build_megatron_mri.sh
#   - Conda environment: hpc-ai-build
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

VERL_REPO="${VERL_REPO:-https://github.com/verl-project/verl.git}"
VERL_REF="${VERL_REF:-main}"
VERL_SRC="${VERL_SRC:-${WORK_DIR}/src/verl}"
MVAPICH_HOME="${MVAPICH_HOME:-${WORK_DIR}/install/mvapich-plus-4.1-cuda${CUDA_VERSION}}"
NCCL_HOME="${NCCL_HOME:-${WORK_DIR}/libext/nccl/build}"
INSTALL_VERL_RUNTIME_DEPS="${INSTALL_VERL_RUNTIME_DEPS:-1}"

echo "========================================"
echo "verl Setup Script for MRI"
echo "========================================"
echo ""
echo "Configuration:"
echo "  HPC-AI root:       ${HPC_AI_ROOT}"
echo "  Source directory:  ${VERL_SRC}"
echo "  verl ref:          ${VERL_REF}"
echo "  CUDA version:      ${CUDA_VERSION}"
echo "  GCC version:       ${GCC_VERSION}"
echo "  Conda environment: ${CONDA_ENV}"
echo ""
echo "Note: Current upstream verl documentation recommends CUDA 12.8 for the"
echo "latest setup. This script preserves MRI's CUDA ${CUDA_VERSION} stack and"
echo "is intended as the first compatibility-validation path."
echo ""

# ============================================================================
# MODULES AND ENVIRONMENT
# ============================================================================

echo "[1/7] Loading modules..."
module reset
module load "gcc/${GCC_VERSION}"
module load "cuda/${CUDA_VERSION}"
echo "  GCC:  $(which gcc)"
echo "  CUDA: ${CUDA_HOME:-/opt/cuda/${CUDA_VERSION}}"
echo ""

echo "[2/7] Activating Conda environment..."
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

echo "[3/7] Setting up runtime paths..."
export CUDA_HOME="${CUDA_HOME:-/opt/cuda/${CUDA_VERSION}}"
export PATH="${MVAPICH_HOME}/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MVAPICH_HOME}/lib:${CUDA_HOME}/lib64:${NCCL_HOME}/lib:${LD_LIBRARY_PATH:-}"
export CPATH="${CUDA_HOME}/include:${MVAPICH_HOME}/include:${NCCL_HOME}/include:${CPATH:-}"
export LIBRARY_PATH="${CUDA_HOME}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_PRELOAD="/opt/gcc/${GCC_VERSION}/lib64/libstdc++.so.6:${LD_PRELOAD:-}"
export MV2_USE_CUDA=1
export MV2_USE_GPUDIRECT_RDMA=1
export MV2_CUDA_USE_NAIVE=0
export VLLM_USE_V1="${VLLM_USE_V1:-1}"

unset _CONDA_PYTHON_SYSCONFIGDATA_NAME
export PATH="$(echo "${PATH}" | tr ':' '\n' | grep -v "compiler_compat" | paste -sd:)"

echo "  MVAPICH_HOME: ${MVAPICH_HOME}"
echo "  NCCL_HOME:    ${NCCL_HOME}"
echo "  CUDA_HOME:    ${CUDA_HOME}"
echo "  VLLM_USE_V1:  ${VLLM_USE_V1}"
echo ""

# ============================================================================
# PREREQUISITES
# ============================================================================

echo "[4/7] Verifying base stack..."
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import torch.distributed as dist; print(f'  MPI available: {dist.is_mpi_available()}')"
python -c "import vllm; print(f'  vLLM: {vllm.__version__}')"
python -c "import megatron.core; print('  Megatron Core: available')"
echo ""

# ============================================================================
# SOURCE CHECKOUT
# ============================================================================

echo "[5/7] Preparing verl source..."
mkdir -p "$(dirname "${VERL_SRC}")"
if [ -d "${VERL_SRC}/.git" ]; then
    echo "  Existing checkout found. Updating ${VERL_SRC}"
    git -C "${VERL_SRC}" fetch origin "${VERL_REF}" || git -C "${VERL_SRC}" fetch origin
    git -C "${VERL_SRC}" checkout "${VERL_REF}"
    git -C "${VERL_SRC}" pull --ff-only origin "${VERL_REF}" || true
else
    echo "  Cloning ${VERL_REPO}"
    git clone --recursive --branch "${VERL_REF}" "${VERL_REPO}" "${VERL_SRC}"
fi
git -C "${VERL_SRC}" submodule update --init --recursive
echo "  Source ready: ${VERL_SRC}"
echo ""

# ============================================================================
# INSTALL
# ============================================================================

echo "[6/7] Installing verl without replacing custom PyTorch or vLLM..."
if [ "${INSTALL_VERL_RUNTIME_DEPS}" = "1" ]; then
    VERL_RUNTIME_DEPS=(
        accelerate
        codetiming
        datasets
        dill
        hydra-core
        numpy
        pandas
        peft
        "pyarrow>=15.0.0"
        pylatexenc
        "ray[data,train,tune,serve]"
        torchdata
        transformers
        wandb
        orjson
        pybind11
        tensordict
        omegaconf
    )
    pip install --quiet --no-deps "${VERL_RUNTIME_DEPS[@]}"
else
    echo "  Skipping runtime dependency install because INSTALL_VERL_RUNTIME_DEPS=0"
fi
pip install -e "${VERL_SRC}" --no-deps
echo ""

# ============================================================================
# VERIFICATION
# ============================================================================

echo "[7/7] Verifying verl integration surface..."
python -c "import verl; print('verl', verl.__version__)"
python -c "import vllm; print('vllm', vllm.__version__)"
python -c "import megatron; print('megatron OK')"

echo ""
echo "========================================"
echo "verl Setup Complete"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Run tests: HPC_AI_STRICT_VERL=1 python -m pytest tests/verl -q"
echo "  2. Run the dummy recipe after model and data paths are configured"
echo ""
