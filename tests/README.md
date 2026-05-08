# Tests

Correctness tests for the OSU HPC-AI stack.

## Quick Start

```bash
# Run all tests
python tests/run_tests.py --all

# Run specific framework tests
python tests/run_tests.py --pytorch
python tests/run_tests.py --deepspeed
python tests/run_tests.py --vllm
python tests/run_tests.py --sglang
python tests/run_tests.py --megatron
python tests/run_tests.py --verl
python tests/run_tests.py --mac-attention
```

## Test Manifest

### Installation Tests

Verify each framework is properly installed and configured.

| Test | Framework | Command | What it verifies |
|------|-----------|---------|------------------|
| `pytorch/test_installation.py` | PyTorch | `python tests/pytorch/test_installation.py` | PyTorch import, MPI backend, CUDA support |
| `deepspeed/test_installation.py` | DeepSpeed | `python tests/deepspeed/test_installation.py` | DeepSpeed import, version, MPI backend |
| `vllm/test_installation.py` | vLLM | `python tests/vllm/test_installation.py` | vLLM import, components |
| `sglang/test_installation.py` | SGLang | `python tests/sglang/test_installation.py` | SGLang import, components |
| `megatron/test_installation.py` | Megatron-LM | `python -m pytest tests/megatron -q` | Megatron Core import, PyTorch backend visibility |
| `verl/test_installation.py` | verl | `python -m pytest tests/verl -q` | verl import, vLLM and Megatron dependency visibility |

### MPI Communication Tests

Verify MPI operations work correctly with the MVAPICH backend.

| Test | Command | What it verifies |
|------|---------|------------------|
| `pytorch/test_mpi_comm.py` | `mpirun -np 2 python tests/pytorch/test_mpi_comm.py` | All-reduce, broadcast, gather, scatter, barrier |

### Integration Tests

End-to-end tests that verify the full stack works correctly.

| Test | Command | What it verifies |
|------|---------|------------------|
| `integration/test_pytorch_ddp.py` | `mpirun -np 2 python tests/integration/test_pytorch_ddp.py` | DDP training, gradient sync, parameter consistency |
| `integration/test_deepspeed_zero.py` | `deepspeed --num_gpus=2 tests/integration/test_deepspeed_zero.py` | ZeRO optimizer, MPI communication, training loop |
| `integration/test_vllm_inference.py` | `python tests/integration/test_vllm_inference.py` | vLLM import, LLM class, optional inference |
| `integration/test_sglang_serving.py` | `python tests/integration/test_sglang_serving.py` | SGLang import, components, backend |
| `integration/test_verl_megatron_vllm.py` | `python -m pytest tests/integration/test_verl_megatron_vllm.py -q` | verl Megatron-LM vLLM recipe files, launch guard, optional dummy run |

### MAC-Attention Tests

Correctness tests for the imported MAC-Attention method package.

| Test | Command | What it verifies |
|------|---------|------------------|
| `mac-attention/test_mac_decode.py` | `python -m pytest tests/mac-attention/test_mac_decode.py -q` | Decode wrapper output against a reference implementation |
| `mac-attention/test_mac_rectification_cache.py` | `python -m pytest tests/mac-attention/test_mac_rectification_cache.py -q` | Rectification and cache update behavior against a reference implementation |
| `mac-attention/test_match_kernel.py` | `python -m pytest tests/mac-attention/test_match_kernel.py -q` | Match-kernel slot, hit, and left-start semantics |

## Directory Structure

```
tests/
├── run_tests.py          # Unified test runner
├── README.md             # This file (test manifest)
├── pytorch/              # PyTorch-specific tests
│   ├── test_installation.py
│   └── test_mpi_comm.py
├── deepspeed/            # DeepSpeed-specific tests
│   └── test_installation.py
├── vllm/                 # vLLM-specific tests
│   └── test_installation.py
├── sglang/               # SGLang-specific tests
│   └── test_installation.py
├── megatron/             # Megatron-LM-specific tests
│   └── test_installation.py
├── verl/                 # verl-specific tests
│   └── test_installation.py
├── mac-attention/        # MAC-Attention correctness tests
│   ├── test_mac_decode.py
│   ├── test_mac_rectification_cache.py
│   └── test_match_kernel.py
└── integration/          # End-to-end integration tests
    ├── test_pytorch_ddp.py
    ├── test_deepspeed_zero.py
    ├── test_vllm_inference.py
    ├── test_sglang_serving.py
    └── test_verl_megatron_vllm.py
```

## Test vs Benchmark

- **Tests** (here): Verify correctness - do we get expected results?
- **Benchmarks** (`../benchmarks/`): Measure performance - is it fast enough?

## Running on HPC Clusters

```bash
# Allocate GPU node
salloc -p gpu -t 1:00:00 --gpus-per-node=2

# Setup environment
module load gcc/13.3.0 cuda/12.6
source ~/miniconda3/bin/activate hpc-ai-build

# Run tests
cd /path/to/osu-hpc-ai
python tests/run_tests.py --all
```

## Adding New Tests

1. Create test file in appropriate directory
2. Follow naming convention: `test_*.py`
3. Add entry to this manifest
4. Ensure test completes in < 60 seconds
5. Handle missing dependencies gracefully (skip vs fail)

See [docs/testing_guide.md](../docs/testing_guide.md) for detailed testing guide.

---

**Maintained by**: NOWLAB Team, The Ohio State University
