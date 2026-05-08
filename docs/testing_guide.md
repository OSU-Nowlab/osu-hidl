# OSU HPC-AI Testing Guide

Comprehensive testing guide for the OSU HPC-AI stack. This document covers testing strategy, test organization, running tests, and contributing new tests.

## Table of Contents

- [Overview](#overview)
- [Test Organization](#test-organization)
- [Quick Start](#quick-start)
- [Test Categories](#test-categories)
- [Running Tests](#running-tests)
- [Writing Tests](#writing-tests)
- [Benchmarking](#benchmarking)
- [Continuous Integration](#continuous-integration)
- [Troubleshooting](#troubleshooting)

---

## Overview

The HPC-AI testing infrastructure ensures:
- **Correctness**: All components work as expected
- **Performance**: Meet performance targets
- **Compatibility**: Works across different systems
- **Reliability**: Catches regressions early

### Testing Philosophy

1. **Test at multiple levels**: Installation, integration, end-to-end
2. **Test realistic scenarios**: Use real-world workloads when possible
3. **Automate everything**: All tests can run unattended
4. **Clear pass/fail criteria**: No ambiguous test results
5. **Fast feedback**: Critical tests run quickly

---

## Test Organization

```
osu-hpc-ai/
├── tests/                       # Test suite
│   ├── integration/             # End-to-end integration tests
│   │   ├── test_pytorch_ddp.py
│   │   ├── test_deepspeed_zero.py
│   │   ├── test_vllm_inference.py
│   │   └── test_sglang_serving.py
│   ├── pytorch/                 # PyTorch-specific tests
│   │   ├── test_installation.py
│   │   └── test_mpi_comm.py
│   ├── deepspeed/               # DeepSpeed-specific tests
│   │   └── test_installation.py
│   ├── vllm/                    # vLLM-specific tests
│   │   └── test_installation.py
│   ├── sglang/                  # SGLang-specific tests
│   │   └── test_installation.py
│   ├── megatron/                # Megatron-LM-specific tests
│   │   └── test_installation.py
│   ├── verl/                    # verl-specific tests
│   │   └── test_installation.py
│   ├── run_tests.py             # Unified test runner
│   └── README.md                # Test manifest
├── benchmarks/                  # Performance benchmarks
│   ├── communication/           # MPI communication benchmarks
│   └── README.md
└── examples/                    # Working examples (also serve as tests)
    ├── pytorch/
    ├── deepspeed/
    └── sglang/
```

### Test Levels

| Level | Purpose | GPU Required | Run Frequency |
|-------|---------|--------------|---------------|
| **Installation** | Verify components installed | No | Every commit |
| **Integration** | Test component interactions | Yes | Every PR |
| **Benchmarks** | Measure performance | Yes | Weekly |

---

## Quick Start

### Prerequisites

```bash
# Install pytest
pip install pytest pytest-timeout pytest-xdist

# Verify installation
pytest --version
```

### Run All Tests

```bash
cd /path/to/osu-hpc-ai

# Run test runner for specific framework
python tests/run_tests.py --pytorch
python tests/run_tests.py --deepspeed
python tests/run_tests.py --vllm
python tests/run_tests.py --sglang
python tests/run_tests.py --megatron
python tests/run_tests.py --verl

# Or run all
python tests/run_tests.py --all
```

### Quick Validation

```bash
# Verify installation for each framework
python tests/pytorch/test_installation.py
python tests/deepspeed/test_installation.py
python tests/vllm/test_installation.py
python tests/sglang/test_installation.py

# Test PyTorch MPI communication
mpirun -np 2 python tests/pytorch/test_mpi_comm.py

# Run integration tests
mpirun -np 2 python tests/integration/test_pytorch_ddp.py
deepspeed --num_gpus=2 tests/integration/test_deepspeed_zero.py
```

---

## Test Categories

### 1. Installation Tests

**Purpose**: Verify components are properly installed

**Location**: `tests/*/test_installation.py`

**Example**:
```bash
# DeepSpeed installation
python tests/deepspeed/test_installation.py

# Expected output:
# Import DeepSpeed... PASS
# DeepSpeed Version... PASS
# MPI Backend... PASS
# Results: 7 passed, 0 failed
```

**What they check**:
- Libraries can be imported
- Versions are correct
- Dependencies are available
- CUDA/MPI backends work

### 2. Integration Tests

**Purpose**: Test end-to-end workflows

**Location**: `tests/integration/`

**Available Tests**:

| Test | Description | Command |
|------|-------------|---------|
| `test_pytorch_ddp.py` | DDP training with MPI | `mpirun -np 2 python test_pytorch_ddp.py` |
| `test_deepspeed_zero.py` | ZeRO training | `deepspeed --num_gpus=2 test_deepspeed_zero.py` |
| `test_vllm_inference.py` | vLLM inference | `python test_vllm_inference.py` |
| `test_sglang_serving.py` | SGLang serving | `python test_sglang_serving.py` |
| `test_verl_megatron_vllm.py` | verl Megatron-LM vLLM recipe | `python -m pytest test_verl_megatron_vllm.py -q` |

**What they test**:
- Multi-GPU communication
- Framework interoperability
- End-to-end workflows

### 3. Framework-Specific Tests

#### PyTorch Tests

**Location**: `tests/pytorch/`

```bash
# Test MPI communication
mpirun -np 2 python tests/pytorch/test_mpi_comm.py
```

**Tests**:
- MPI backend initialization
- Distributed operations (all-reduce, broadcast, etc.)
- GPU-to-GPU communication
- Multi-node setup

#### DeepSpeed Tests

**Location**: `tests/deepspeed/`

```bash
# Installation check
python tests/deepspeed/test_installation.py
```

**Tests**:
- Import and version verification
- MPI backend availability
- CUDA support

### 4. Performance Benchmarks

**Purpose**: Measure and track performance

**Location**: `benchmarks/`

```bash
cd benchmarks/communication
mpirun -np 4 python run_all.py

# Expected output:
# All-Reduce: 12.3 GB/s
# All-Gather: 11.8 GB/s
# Broadcast: 13.1 GB/s
```

See [../benchmarks/README.md](../benchmarks/README.md) for details.

---

## Running Tests

### Using the Test Runner

```bash
cd /path/to/osu-hpc-ai

# Run specific framework tests
python tests/run_tests.py --pytorch
python tests/run_tests.py --deepspeed
python tests/run_tests.py --vllm
python tests/run_tests.py --sglang

# Run integration tests
python tests/run_tests.py --integration

# Run all tests
python tests/run_tests.py --all
```

### On HPC Clusters (e.g., MRI)

#### Interactive Session

```bash
# Allocate GPU node
salloc -p devel -t 2:00:00 -C cuda -N 1 --gpus-per-node=2

# Setup environment
module reset
module load gcc/13.3.0 cuda/12.6
source ~/miniconda3/bin/activate hpc-ai-build
export LD_PRELOAD=/opt/gcc/13.3.0/lib64/libstdc++.so.6

# Navigate and run tests
cd /home/$USER/osu-hpc-ai
python tests/run_tests.py --all
```

#### Batch Job

Create `run_tests.slurm`:
```bash
#!/bin/bash
#SBATCH -J hpc-ai-tests
#SBATCH -N 1
#SBATCH --gpus-per-node=2
#SBATCH -t 1:00:00
#SBATCH -p gpu
#SBATCH -o test_output_%j.log

# Setup
module load gcc/13.3.0 cuda/12.6
source ~/miniconda3/bin/activate hpc-ai-build
export LD_PRELOAD=/opt/gcc/13.3.0/lib64/libstdc++.so.6

# Run tests
cd $HOME/osu-hpc-ai
python tests/run_tests.py --all
```

Submit:
```bash
sbatch run_tests.slurm
```

### Test Selection with pytest

```bash
# Run specific test file
python -m pytest tests/deepspeed/test_installation.py -v

# Run specific test function
python -m pytest tests/deepspeed/test_installation.py::test_import_deepspeed -v

# Run tests matching pattern
python -m pytest tests/ -k "installation" -v
```

---

## Writing Tests

### Test Template

```python
"""
Module: test_my_feature.py
Description: Tests for my new feature
"""

import pytest
import torch


class TestMyFeature:
    """Test suite for my feature."""

    def setup_method(self):
        """Setup before each test."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def teardown_method(self):
        """Cleanup after each test."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @pytest.mark.gpu
    def test_gpu_functionality(self):
        """Test GPU-specific functionality."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        data = torch.randn(100, 100).cuda()
        result = my_gpu_function(data)
        assert result.is_cuda

    @pytest.mark.integration
    def test_integration(self):
        """Test integration with other components."""
        # Test implementation
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Test Markers

Use pytest markers to categorize tests:

```python
@pytest.mark.integration   # Integration test (may need GPU)
@pytest.mark.gpu           # Requires GPU
@pytest.mark.slow          # Takes > 1 minute
@pytest.mark.mpi           # Requires MPI
```

Run tests by marker:
```bash
# Only GPU tests
python -m pytest tests/ -m gpu -v

# Skip slow tests
python -m pytest tests/ -m "not slow" -v
```

### Assertions

```python
# Basic assertions
assert value == expected
assert value > 0
assert not torch.isnan(tensor).any()

# Approximate equality (for floats)
assert abs(actual - expected) < 1e-5
torch.testing.assert_close(tensor1, tensor2, rtol=1e-5, atol=1e-8)

# Exception testing
with pytest.raises(RuntimeError):
    function_that_should_fail()

# Skip tests conditionally
if not torch.cuda.is_available():
    pytest.skip("CUDA not available")
```

---

## Benchmarking

### Running Benchmarks

```bash
cd benchmarks/communication

# Run all benchmarks
mpirun -np 4 python run_all.py

# Run specific benchmark with size scan
mpirun -np 4 python all_reduce.py --scan
```

### Adding Benchmarks

```python
# benchmarks/my_benchmark.py
import time
import torch

def benchmark_operation(size, num_iterations=100):
    """Benchmark an operation."""
    data = torch.randn(size, size).cuda()

    # Warmup
    for _ in range(10):
        result = torch.matmul(data, data)

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()

    for _ in range(num_iterations):
        result = torch.matmul(data, data)

    torch.cuda.synchronize()
    elapsed = time.time() - start

    throughput = (size * size * 2) * num_iterations / elapsed / 1e9
    return throughput

if __name__ == "__main__":
    result = benchmark_operation(1000)
    print(f"Throughput: {result:.2f} GFLOPS")
```

---

## Continuous Integration

### GitHub Actions

Tests run automatically on:
- **Pull Requests**: Full test suite
- **Commits to main**: Quick tests
- **Nightly**: Full tests + benchmarks

### Local Pre-commit Checks

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

---

## Troubleshooting

### Common Issues

#### Tests Fail to Import Modules

```bash
# Ensure Python path is set
export PYTHONPATH="/path/to/osu-hpc-ai:$PYTHONPATH"

# Or install in development mode
cd /path/to/osu-hpc-ai
pip install -e .
```

#### CUDA Out of Memory

```bash
# Reduce batch sizes in test configs
# Or run tests sequentially
python -m pytest tests/ -v --maxfail=1
```

#### MPI Tests Hang

```bash
# Check MPI is properly initialized
which mpicc

# Verify MPI backend
python -c "import torch.distributed as dist; print(dist.is_mpi_available())"

# Run with timeout
python -m pytest tests/pytorch/test_mpi_comm.py --timeout=60
```

### Debug Mode

```bash
# Enable verbose logging
export DEEPSPEED_LOG_LEVEL=DEBUG
export TORCH_DISTRIBUTED_DEBUG=INFO

# Run with Python debugger
python -m pdb -m pytest tests/my_test.py
```

### Getting Help

If tests fail:

1. Check test output for error messages
2. Look in test logs: `test_output_*.log`
3. Check GitHub issues: https://github.com/OSU-Nowlab/osu-hpc-ai/issues

---

## Resources

- **pytest Documentation**: https://docs.pytest.org/
- **PyTorch Testing**: https://github.com/pytorch/pytorch/wiki/Running-and-writing-tests
- **DeepSpeed Testing**: https://github.com/microsoft/DeepSpeed/tree/master/tests
- **Our Examples**: [examples/](../examples/)

---

**Maintained by**: Network-Based Computing Laboratory, The Ohio State University
