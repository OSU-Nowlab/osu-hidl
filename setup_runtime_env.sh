#!/bin/bash
#
# Runtime Environment Setup for OSU-HPC-AI
#
# Source this file before running any PyTorch/DeepSpeed/SGLang/vLLM code:
#   source ~/osu-hpc-ai/setup_runtime_env.sh
#

# Load correct GCC libstdc++ (PyTorch built with GCC 13.3.0)
export LD_PRELOAD=/opt/gcc/13.3.0/lib64/libstdc++.so.6

# MVAPICH-Plus paths (if using MPI)
export MVAPICH_HOME=~/osu-hpc-ai-dev/install/mvapich-plus-4.1-cuda12.6
export PATH=$MVAPICH_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MVAPICH_HOME/lib:$LD_LIBRARY_PATH

# CUDA paths (targets lib first for libnvJitLink, then lib64)
export CUDA_HOME=/opt/cuda/12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# NCCL paths
export NCCL_HOME=~/osu-hpc-ai-dev/libext/nccl/build
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH

echo "Runtime environment configured:"
echo "  GCC libstdc++: /opt/gcc/13.3.0/lib64/libstdc++.so.6"
echo "  CUDA: $CUDA_HOME"
echo "  MVAPICH: $MVAPICH_HOME"
echo "  NCCL: $NCCL_HOME"
