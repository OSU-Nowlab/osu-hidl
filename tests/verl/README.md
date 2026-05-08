# verl Tests

Smoke tests for a verl installation that uses Megatron-LM for training and vLLM for rollout generation.

## Quick Start

```bash
python -m pytest tests/verl -q
```

By default, tests skip missing packages so they can run before the MRI install is complete. On MRI, use strict mode after installing the full stack:

```bash
HPC_AI_STRICT_VERL=1 python -m pytest tests/verl -q
```

These tests verify package discovery and PyTorch backend visibility. The full verl dummy run is intentionally separate because it needs GPUs, model assets, and Ray runtime state.
