#!/bin/bash
################################################################################
# PyTorch Build Script for MRI Cluster
#
# Builds PyTorch with MVAPICH-Plus (CUDA-aware MPI) and NCCL support
#
# Prerequisites:
#   - GPU node allocation (choose one):
#     Interactive: salloc -p devel -t 6:00:00 -C cuda -N 1 --gpus-per-node=1 --mem=64G
#     Batch job:   sbatch build_scripts/mri/submit_pytorch_build.slurm
#   - MVAPICH-Plus installed at: ~/osu-hpc-ai-dev/install/mvapich-plus-4.1-cuda12.6
#   - NCCL built at: ~/osu-hpc-ai-dev/libext/nccl/build
#   - Conda environment: hpc-ai-build
#
# Build time: 4-6 hours (MAX_JOBS=4)
# Memory required: 64GB+ RAM
#
# Usage:
#   bash build_scripts/mri/build_pytorch_mri.sh           # Incremental build
#   bash build_scripts/mri/build_pytorch_mri.sh --clean   # Clean build (wipes previous)
#
################################################################################

set -e  # Exit on error

# Parse arguments
CLEAN_BUILD=false
for arg in "$@"; do
    case $arg in
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
    esac
done

echo "========================================"
echo "PyTorch Build Script for MRI"
echo "========================================"
echo "Clean build: $CLEAN_BUILD"
echo ""

# Step 1/8: Load modules
echo "Step 1/8: Loading modules..."
module reset 2>/dev/null || true
module load gcc/13.3.0 2>/dev/null || true
module load cuda/12.6 2>/dev/null || true
echo "  Loaded: gcc/13.3.0, cuda/12.6"
echo ""

# Step 2/8: Activate conda environment
echo "Step 2/8: Activating conda environment..."
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "hpc-ai-build" ]; then
    echo "  Activating hpc-ai-build environment..."
    eval "$(conda shell.bash hook)"
    conda activate hpc-ai-build
else
    echo "  Already in hpc-ai-build environment"
fi
echo ""

# Step 3/8: Set up environment variables
echo "Step 3/8: Setting up environment variables..."
export MVAPICH_HOME=~/osu-hpc-ai-dev/install/mvapich-plus-4.1-cuda12.6
export CUDA_HOME=/opt/cuda/12.6
export NCCL_HOME=~/osu-hpc-ai-dev/libext/nccl/build

export PATH=$MVAPICH_HOME/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MVAPICH_HOME/lib:$CUDA_HOME/lib64:$NCCL_HOME/lib:$LD_LIBRARY_PATH
export CPATH=$CUDA_HOME/include:$MVAPICH_HOME/include:$NCCL_HOME/include:$CPATH

# Add CUDA static library path
export LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$LIBRARY_PATH

# Clean conda toolchain issues
unset _CONDA_PYTHON_SYSCONFIGDATA_NAME
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v "compiler_compat" | paste -sd:)

# Force correct libstdc++ version
export LD_PRELOAD=/opt/gcc/13.3.0/lib64/libstdc++.so.6:$LD_PRELOAD

# Explicitly set MPI compilers for CMake detection
export MPI_HOME=$MVAPICH_HOME
export MPI_C_COMPILER=$MVAPICH_HOME/bin/mpicc
export MPI_CXX_COMPILER=$MVAPICH_HOME/bin/mpicxx
export MPICC=$MVAPICH_HOME/bin/mpicc
export MPICXX=$MVAPICH_HOME/bin/mpicxx

echo "  MVAPICH_HOME: $MVAPICH_HOME"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  NCCL_HOME: $NCCL_HOME"
echo "  MPI_C_COMPILER: $MPI_C_COMPILER"
echo ""

# Step 4/8: Verify prerequisites
echo "Step 4/8: Verifying prerequisites..."
if [ ! -d "$MVAPICH_HOME" ]; then
    echo "ERROR: MVAPICH-Plus not found at $MVAPICH_HOME"
    exit 1
fi
if [ ! -d "$NCCL_HOME" ]; then
    echo "ERROR: NCCL not found at $NCCL_HOME"
    exit 1
fi
if [ ! -x "$MPI_C_COMPILER" ]; then
    echo "ERROR: mpicc not found at $MPI_C_COMPILER"
    exit 1
fi

echo "  MVAPICH-Plus: $(which mpicc)"
echo "  mpicc version: $(mpicc --version 2>&1 | head -1)"
echo "  NCCL: $NCCL_HOME/lib/libnccl.so"
echo "  GCC: $(gcc --version | head -1)"
echo "  CUDA: $(nvcc --version | grep release)"
echo ""

# Step 5/8: Install Python dependencies
echo "Step 5/8: Installing Python dependencies..."
pip install --quiet pyyaml typing-extensions cmake ninja
echo "  Dependencies installed"
echo ""

# Step 6/8: Prepare PyTorch source
echo "Step 6/8: Preparing PyTorch source directory..."
cd ~/osu-hpc-ai/frameworks/pytorch
echo "  Current directory: $PWD"

if [ "$CLEAN_BUILD" = true ]; then
    echo "  --clean specified: Performing full clean..."
    echo "  Syncing and updating submodules..."
    git submodule sync
    git submodule update --init --recursive
    echo "  Cleaning build artifacts..."
    git clean -fdx
    make clean 2>/dev/null || true
    python setup.py clean 2>/dev/null || true
    echo "  Clean complete"
else
    echo "  Incremental build: Keeping existing build artifacts"
    echo "  (Use --clean for a fresh build)"
fi
echo ""

# Step 7/8: Configure build (CMake stage)
echo "Step 7/8: Configuring PyTorch build (CMake)..."
echo "  Build flags:"
echo "    MAX_JOBS=4"
echo "    USE_CUDA=1, USE_MPI=1, USE_CUDA_MPI=1"
echo "    USE_XNNPACK=0, USE_FBGEMM=0 (disabled for stability)"
echo "    TORCH_CUDA_ARCH_LIST=8.0 (A100)"
echo "    USE_NCCL=1, USE_SYSTEM_NCCL=1"
echo ""

export _GLIBCXX_USE_CXX11_ABI=1
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"

# Build configuration - all flags set as environment variables
export MAX_JOBS=4
export USE_XNNPACK=0
export USE_FBGEMM=0
export USE_CUDA=1
export USE_MPI=1
export USE_CUDA_MPI=1
export USE_DISTRIBUTED=1
export BUILD_TEST=0
export BUILD_MOBILE_BENCHMARK=0
export BUILD_MOBILE_TEST=0
export TORCH_CUDA_ARCH_LIST="8.0"
export USE_NCCL=1
export USE_SYSTEM_NCCL=1
export NCCL_INCLUDE_DIR=$NCCL_HOME/include
export NCCL_LIB_DIR=$NCCL_HOME/lib

# Disable Intel XPU/SYCL support (system has oneAPI but not PTI library)
export USE_XPU=0
export USE_KINETO_SYCL=0

echo "Running CMake configuration..."
python setup.py build --cmake-only 2>&1 | tee ~/pytorch_cmake.log

# Verify MPI was detected
if grep -q "USE_MPI.*ON" build/CMakeCache.txt 2>/dev/null; then
    echo "  CMake detected MPI: YES"
else
    echo "  WARNING: CMake may not have detected MPI!"
    echo "  Check ~/pytorch_cmake.log for details"
fi
echo ""

# Step 8/8: Build and install PyTorch
echo "Step 8/8: Building and installing PyTorch..."
echo "  This will take 4-6 hours..."
echo "  Monitor progress: tail -f ~/pytorch_build.log"
echo ""

python setup.py develop 2>&1 | tee ~/pytorch_build.log

echo ""
echo "========================================"
echo "PyTorch Build Complete!"
echo "========================================"
echo ""

# Verify installation
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch.distributed as dist; print(f'MPI available: {dist.is_mpi_available()}')"
python -c "import torch.distributed as dist; print(f'NCCL available: {dist.is_nccl_available()}')"

# Check if MPI is actually enabled
MPI_STATUS=$(python -c "import torch.distributed as dist; print(dist.is_mpi_available())")
if [ "$MPI_STATUS" = "True" ]; then
    echo ""
    echo "SUCCESS: MPI backend is available!"
else
    echo ""
    echo "WARNING: MPI backend is NOT available!"
    echo "Check ~/pytorch_cmake.log and ~/pytorch_build.log for errors"
fi

echo ""
echo "Build logs saved to:"
echo "  ~/pytorch_cmake.log"
echo "  ~/pytorch_build.log"
echo ""
echo "Next steps:"
echo "  1. Test MPI: mpirun -np 2 python -c \"import torch.distributed as dist; dist.init_process_group('mpi')\""
echo "  2. Run examples: cd ~/osu-hpc-ai/examples/pytorch"
echo "  3. Run tests: cd ~/osu-hpc-ai/tests/pytorch"
echo ""
