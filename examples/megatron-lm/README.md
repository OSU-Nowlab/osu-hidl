# Megatron-LM Dummy Training Example

This example launches Megatron Core's simple mock-data training loop from a Megatron-LM source checkout. It verifies that Megatron-LM can run on top of the custom OSU HPC-AI PyTorch stack before using Megatron-LM inside verl.

The official Megatron Core training examples describe `examples/run_simple_mcore_train_loop.py` as the basic mock-data check. This wrapper keeps the MRI paths and safety checks consistent with the rest of this repository.

## Prerequisites

Install PyTorch and Megatron-LM first:

```bash
cd ~/osu-hpc-ai
bash build_scripts/mri/build_pytorch_mri.sh
bash build_scripts/mri/build_megatron_mri.sh
```

Request a GPU allocation. The upstream example is designed for two GPUs:

```bash
salloc -p devel -t 1:00:00 -C cuda -N 1 --gpus-per-node=2 --mem=64G
srun --pty bash
```

## Run

The launcher is gated so it does not start a distributed job by accident:

```bash
HPC_AI_RUN_MEGATRON_DUMMY=1 bash examples/megatron-lm/run_simple_mcore_dummy.sh
```

By default, the script expects Megatron-LM at `~/osu-hpc-ai-dev/src/Megatron-LM` and uses two local processes. Override those values if needed:

```bash
HPC_AI_RUN_MEGATRON_DUMMY=1 \
MEGATRON_SRC=/path/to/Megatron-LM \
MEGATRON_DUMMY_GPUS=1 \
bash examples/megatron-lm/run_simple_mcore_dummy.sh
```

Use one GPU only if the installed Megatron-LM revision supports the simple loop with `--nproc_per_node=1`.

## What Success Looks Like

A successful run should initialize torch distributed, build a tiny model, use generated mock data, and print training-step logs. This confirms the Megatron-LM package and custom PyTorch runtime are compatible enough for the next verl validation step.

If this example fails, fix it before proceeding with further verl integration.
