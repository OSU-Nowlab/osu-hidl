# Megatron-LM Build Guide - MRI Cluster

This guide covers setting up Megatron-LM on MRI using the OSU HPC-AI PyTorch build with MVAPICH-Plus support.

## Overview

Megatron-LM is NVIDIA's reference implementation for training large transformer models at scale. It is built on Megatron Core, a lower-level library that provides tensor parallelism, pipeline parallelism, sequence parallelism, distributed optimizers, and model components for large language model training.

In this repository, Megatron-LM is used as the training backend for verl experiments. The dummy workflow is not meant to train a useful model. It verifies that Megatron-LM can import, see the custom PyTorch distributed stack, and run a tiny mock-data configuration on MRI.

The official Megatron Core installation guide is available at [NVIDIA Megatron Core installation](https://docs.nvidia.com/megatron-core/developer-guide/0.17.0/get-started/install.html). The Megatron-LM source repository is [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM).

## Prerequisites

Build and activate the base HPC-AI stack first:

```bash
cd ~/osu-hpc-ai
bash build_scripts/mri/build_pytorch_mri.sh
```

Use a GPU allocation for validation:

```bash
salloc -p devel -t 1:00:00 -C cuda -N 1 --gpus-per-node=1 --mem=64G
srun --pty bash
```

Load the MRI runtime environment:

```bash
module reset
module load gcc/13.3.0
module load cuda/12.6
source ~/osu-hpc-ai/setup_runtime_env.sh
conda activate hpc-ai-build
```

## Automated Setup

Run the MRI setup script:

```bash
cd ~/osu-hpc-ai
bash build_scripts/mri/build_megatron_mri.sh
```

The script clones Megatron-LM into `~/osu-hpc-ai-dev/src/Megatron-LM`, installs it in editable mode, and uses `--no-deps` so pip does not replace the custom PyTorch build.

You can pin a different upstream ref:

```bash
MEGATRON_REF=<branch-or-tag> bash build_scripts/mri/build_megatron_mri.sh
```

## Verification

Run the lightweight test suite:

```bash
HPC_AI_STRICT_MEGATRON=1 python -m pytest tests/megatron -q
```

Check the core imports directly:

```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch.distributed as dist; print(dist.is_mpi_available())"
python -c "import megatron.core; print('Megatron Core OK')"
```

The strict test mode expects the MRI stack to expose CUDA and MPI. On a login node or local laptop, run without strict mode and missing GPU-only pieces will be skipped.

After the import checks pass, run the mock-data training example:

```bash
HPC_AI_RUN_MEGATRON_DUMMY=1 bash examples/megatron-lm/run_simple_mcore_dummy.sh
```

This wraps Megatron Core's official simple training loop, which uses generated data and does not need dataset preprocessing.

## Troubleshooting

### Megatron Core import fails

Confirm that the editable install points to the intended checkout:

```bash
python -c "import importlib.util; print(importlib.util.find_spec('megatron.core'))"
```

If it is missing, rerun the setup script and inspect the pip output.

### PyTorch loses MPI support

If `dist.is_mpi_available()` returns `False`, the active environment is not using the custom HPC-AI PyTorch build. Reactivate the `hpc-ai-build` environment and source the runtime setup script.

### pip tries to replace torch

Do not install Megatron-LM with unconstrained dependency resolution inside the HPC-AI environment. Use the provided setup script or pass `--no-deps` when installing editable source.

## Next Step

After Megatron-LM imports cleanly, install vLLM and verl to complete the stack.
