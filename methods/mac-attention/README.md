# MAC-Attention

Reusable MAC-Attention source package for the OSU HPC-AI monorepo.

This directory contains the Python package and standalone CUDA kernels used by
the MAC-Attention examples, benchmarks, and tests elsewhere in the repository.

The imported source tree comes from Jinghan Yao's original MAC-Attention
release. Repo-local changes are limited to packaging, path resolution helpers,
and small integration utilities so the method can be exercised from this
monorepo's examples, benchmarks, and tests.

## Install

```bash
python -m pip install -e methods/mac-attention --no-deps
```

Use `--no-deps` inside the HPC-AI stack so pip does not replace the existing
PyTorch installation.

## Quick Verification

```bash
python - <<'PY'
from mac_attention import extension_source_path, method_root
print(method_root())
print(extension_source_path("macMatch.cu"))
PY
```
