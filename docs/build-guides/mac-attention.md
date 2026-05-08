# MAC-Attention Build Guide

This guide covers installing and verifying the imported MAC-Attention method
package in the OSU HPC-AI monorepo.

## Overview

MAC-Attention is a method for accelerating long-context decoding by reusing
attention work from semantically similar recent queries. In this repository it
is packaged as a reusable source tree under `methods/mac-attention/`, with
examples, benchmarks, and tests in the same top-level locations used by the
rest of the monorepo.

The current imported source includes two kinds of CUDA code:

- Standalone extensions under `methods/mac-attention/ext/`
- JIT-built wrapper kernels inside `methods/mac-attention/src/mac_attention/_vendor/`

## Requirements

- NVIDIA GPU with CUDA support
- Python environment with PyTorch already installed
- `filelock`
- A compiler toolchain that can build PyTorch CUDA extensions

The imported wrapper source currently hardcodes `sm_90` and `sm_90a` nvcc
targets. Based on that source configuration, the full wrapper path should be
treated as Hopper-targeted until other GPU architectures are validated in this
repo.

## Installation

Install the local package in editable mode:

```bash
python -m pip install -e methods/mac-attention --no-deps
```

Use `--no-deps` inside this monorepo so pip does not replace the existing
PyTorch installation.

## Verification

Check that the package imports and that the standalone kernel sources resolve
from the installed package:

```bash
python - <<'PY'
from mac_attention import extension_source_path, method_root
print(method_root())
print(extension_source_path("macMatch.cu"))
print(extension_source_path("mac_prefill_update_cache.cu"))
PY
```

Run the example:

```bash
python examples/mac-attention/e2e_workflow.py --steps 1
```

Run the correctness tests:

```bash
python -m pytest tests/mac-attention -q
```

## JIT Compilation

MAC-Attention compiles CUDA extensions on first use. Build artifacts are cached
under:

```text
${MAC_WORKSPACE_BASE:-$HOME}/.cache/mac
```

The standalone kernels use:

```text
${MAC_WORKSPACE_BASE:-$HOME}/.cache/mac/torch_extensions
```

Set `MAC_JIT_VERBOSE=1` to print detailed build output.

## Troubleshooting

If pip tries to install a different PyTorch build, reinstall with `--no-deps`.

If the wrapper kernels fail with a kernel image or architecture error, treat
that as a GPU compatibility issue first. The imported source currently targets
Hopper-class GPUs for the JIT wrapper path.
