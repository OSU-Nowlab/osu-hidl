#!/bin/bash
#
# Run Megatron Core's simple mock-data training loop from a Megatron-LM checkout.
#
set -euo pipefail

if [ "${HPC_AI_RUN_MEGATRON_DUMMY:-0}" != "1" ]; then
    echo "Set HPC_AI_RUN_MEGATRON_DUMMY=1 to launch the Megatron-LM dummy run."
    exit 2
fi

WORK_DIR="${WORK_DIR:-${HOME}/osu-hpc-ai-dev}"
MEGATRON_SRC="${MEGATRON_SRC:-${WORK_DIR}/src/Megatron-LM}"
MEGATRON_DUMMY_GPUS="${MEGATRON_DUMMY_GPUS:-2}"
MEGATRON_DUMMY_SCRIPT="${MEGATRON_DUMMY_SCRIPT:-${MEGATRON_SRC}/examples/run_simple_mcore_train_loop.py}"

if [ ! -f "${MEGATRON_DUMMY_SCRIPT}" ]; then
    echo "Megatron-LM dummy script not found: ${MEGATRON_DUMMY_SCRIPT}"
    echo "Run build_scripts/mri/build_megatron_mri.sh or set MEGATRON_SRC."
    exit 1
fi

python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch.distributed as dist; print(f'MPI available: {dist.is_mpi_available()}')"
python -c "import megatron.core; print('Megatron Core import: OK')"

echo "Launching Megatron-LM dummy training loop"
echo "  source: ${MEGATRON_SRC}"
echo "  script: ${MEGATRON_DUMMY_SCRIPT}"
echo "  local processes: ${MEGATRON_DUMMY_GPUS}"

cd "${MEGATRON_SRC}"
torchrun --nproc_per_node="${MEGATRON_DUMMY_GPUS}" "${MEGATRON_DUMMY_SCRIPT}" "$@"
