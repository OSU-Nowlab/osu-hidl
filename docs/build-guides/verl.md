# verl Build Guide - MRI Cluster

This guide covers installing verl on MRI after PyTorch, vLLM, and Megatron-LM are already available in the HPC-AI environment.

## Overview

verl is a reinforcement learning training framework for large language model post-training. In the workflow planned here, Megatron-LM owns the training path and vLLM owns rollout generation. The trainer updates the policy model, the rollout engine generates responses, and verl coordinates the data movement, rewards, and optimization loop.

The current upstream verl documentation recommends Python 3.10 or newer and CUDA 12.8 or newer for the latest setup. The current MRI build scripts target CUDA 12.6. The MRI scripts in this repository preserve the existing HPC-AI CUDA 12.6 stack and should be treated as the reproducible setup path for this project.

Official references:

- [verl installation](https://verl.readthedocs.io/en/latest/start/install.html)
- [verl Megatron-LM backend](https://verl.readthedocs.io/en/latest/workers/megatron_workers.html)
- [verl engine workers](https://verl.readthedocs.io/en/latest/workers/engine_workers.html)

## Build Order

Run setup in this order:

```bash
cd ~/osu-hpc-ai
bash build_scripts/mri/build_pytorch_mri.sh
bash build_scripts/mri/build_vllm_mri.sh
bash build_scripts/mri/build_megatron_mri.sh
bash build_scripts/mri/build_verl_mri.sh
```

This order protects the custom PyTorch build. vLLM must be built against that PyTorch. Megatron-LM must be visible before using the Megatron backend in verl.

## Automated Setup

Use the MRI script:

```bash
cd ~/osu-hpc-ai
bash build_scripts/mri/build_verl_mri.sh
```

The script clones verl into `~/osu-hpc-ai-dev/src/verl`, installs runtime packages with `--no-deps`, then installs verl in editable mode with `--no-deps`. This is deliberate because upstream RL and inference packages can otherwise replace `torch` or `vllm`.

You can pin a specific revision:

```bash
VERL_REF=<branch-or-tag> bash build_scripts/mri/build_verl_mri.sh
```

If dependency installation has already been handled manually, skip the curated runtime dependency step:

```bash
INSTALL_VERL_RUNTIME_DEPS=0 bash build_scripts/mri/build_verl_mri.sh
```

## Verification

Run the environment checker:

```bash
python -c "import verl; print('verl', verl.__version__)"
python -c "import vllm; print('vllm', vllm.__version__)"
python -c "import megatron; print('megatron OK')"
```

Run the verl tests in strict mode:

```bash
HPC_AI_STRICT_VERL=1 python -m pytest tests/verl -q
```

The tests verify package discovery and confirm the custom PyTorch stack still exposes CUDA and MPI.

## Troubleshooting

### vLLM import fails after installing verl

Check whether pip replaced vLLM or PyTorch:

```bash
python -c "import torch; print(torch.__version__, torch.__file__)"
python -c "import vllm; print(vllm.__version__, vllm.__file__)"
```

If either package points to an unexpected environment, reactivate `hpc-ai-build` and reinstall using the MRI scripts with `--no-deps`.

### CUDA version mismatch

Upstream verl currently recommends CUDA 12.8 for the latest setup. MRI currently uses CUDA 12.6 in this repository. If a future verl revision fails during import or runtime, pin `VERL_REF` to a revision known to support CUDA 12.6, then record the exact revision in the user guide.

### Missing Ray or Hydra dependencies

The setup script installs a conservative runtime dependency set without allowing dependency resolution to replace core packages. If an import is still missing, install the missing package with care and re-run the environment checker before launching a job.

## Next Step

After the install checks pass, verify that verl, vLLM, and Megatron-LM all import without errors.
