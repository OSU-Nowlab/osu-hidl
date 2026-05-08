# Megatron-LM Tests

Smoke tests for the Megatron-LM installation path on the OSU HPC-AI stack.

## Quick Start

```bash
python -m pytest tests/megatron -q
```

By default, tests skip missing Megatron-LM packages so the suite can run on local machines before the MRI install is complete. On MRI, use strict mode after installation:

```bash
HPC_AI_STRICT_MEGATRON=1 python -m pytest tests/megatron -q
```

These tests do not launch a training job. The dummy training path belongs in the integration tests and recipe once the MRI install is validated.
