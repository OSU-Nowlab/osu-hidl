# Integration Tests

End-to-end integration tests for the OSU HPC-AI stack.

## Purpose

Integration tests verify that the full stack works correctly - from PyTorch/DeepSpeed through the MPI backend to multi-GPU/multi-node execution.

## Available Tests

| Test | Description | Requirements |
|------|-------------|--------------|
| `test_pytorch_ddp.py` | PyTorch DDP with MPI backend | 2+ MPI ranks, GPU optional |
| `test_deepspeed_zero.py` | DeepSpeed ZeRO training | GPU required, DeepSpeed |
| `test_vllm_inference.py` | vLLM inference smoke test | GPU required, vLLM |
| `test_sglang_serving.py` | SGLang serving smoke test | GPU required, SGLang |
| `test_verl_megatron_vllm.py` | verl Megatron-LM vLLM recipe guard and optional dummy run | MRI GPU node for full run |

## Running Tests

### PyTorch DDP Test

```bash
# Run with 2 MPI ranks
mpirun -np 2 python test_pytorch_ddp.py

# Or with 4 ranks on multiple GPUs
mpirun -np 4 python test_pytorch_ddp.py
```

### DeepSpeed ZeRO Test

```bash
# Run with DeepSpeed launcher
deepspeed --num_gpus=2 test_deepspeed_zero.py

# Test different ZeRO stages
deepspeed --num_gpus=2 test_deepspeed_zero.py --stage 2
deepspeed --num_gpus=2 test_deepspeed_zero.py --stage 3
```

### vLLM Inference Test

```bash
# Basic import/component test
python test_vllm_inference.py

# Full engine test (downloads model)
VLLM_TEST_ENGINE=1 python test_vllm_inference.py
```

### SGLang Serving Test

```bash
python test_sglang_serving.py
```

### verl Megatron-LM vLLM Recipe Test

```bash
# Local guard checks only
python -m pytest test_verl_megatron_vllm.py -q

# Full dummy recipe on MRI
VERL_TEST_DUMMY_RUN=1 python -m pytest test_verl_megatron_vllm.py -q
```

## Running All Integration Tests

Use the test runner from the parent directory:

```bash
cd /path/to/osu-hpc-ai
python tests/run_tests.py --integration
```

## What These Tests Verify

- **PyTorch DDP**: Process group initialization, gradient synchronization, parameter consistency across ranks
- **DeepSpeed ZeRO**: ZeRO optimizer partitioning, communication with MPI backend, training loop completion
- **vLLM**: Import success, LLM class availability, optional inference execution
- **SGLang**: Import success, core component availability, backend accessibility
- **verl Megatron-LM vLLM**: Recipe files, launch guard, optional full dummy run

## Adding New Tests

When adding integration tests:

1. Test should complete in < 60 seconds
2. Should work with minimal hardware (1-2 GPUs)
3. Should have clear pass/fail output
4. Should handle missing dependencies gracefully (skip vs fail)

---

**Maintained by**: NOWLAB Team, The Ohio State University
