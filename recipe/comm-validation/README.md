# Communication Validation Recipe

Validate MPI communication performance and compare backends.

## Overview

This recipe validates that your HPC-AI stack communication is working correctly and measures performance. It uses the benchmarks from `benchmarks/communication/` with recommended configurations.

## Requirements

- 2+ MPI processes (can be GPUs on same node or across nodes)
- PyTorch with MPI backend
- MVAPICH-Plus (recommended) or other MPI implementation

## Quick Start

```bash
# From the recipe directory
cd recipe/comm-validation

# Run basic validation (2 processes)
bash run.sh

# Run across 4 processes
bash run.sh --np 4

# Run on multiple nodes
bash run.sh --np 8 --hosts node1,node2
```

## Files

| File | Description |
|------|-------------|
| `validate.py` | Quick validation of MPI operations |
| `run.sh` | Launcher script |

## What It Tests

1. **Correctness**: Verifies all-reduce, broadcast, etc. produce correct results
2. **Basic Performance**: Measures latency and bandwidth for key operations
3. **GPU-awareness**: Tests CUDA-aware MPI if GPUs are available

## Expected Output

```
============================================================
MPI Communication Validation
============================================================
Backend: MPI (MVAPICH-Plus)
World size: 4
CUDA available: True

Running correctness tests...
  All-reduce (sum).......... PASS
  All-reduce (max).......... PASS
  Broadcast................. PASS
  All-gather................ PASS
  Reduce-scatter............ PASS
  Barrier................... PASS

Running performance tests...
  All-reduce latency: 0.15 ms (1MB message)
  All-reduce bandwidth: 12.5 GB/s
  Broadcast latency: 0.08 ms (1MB message)

============================================================
Validation Summary
============================================================
Correctness: 6/6 tests passed
Performance: Within expected range

Status: PASS
```

## Performance Expectations

Expected performance on InfiniBand networks with A100 GPUs:

| Operation | Latency (1MB) | Bandwidth |
|-----------|---------------|-----------|
| All-reduce | 0.1-0.3 ms | 10-15 GB/s |
| Broadcast | 0.05-0.2 ms | 15-20 GB/s |
| All-gather | 0.1-0.3 ms | 10-15 GB/s |

Lower performance may indicate:
- Missing GPUDirect RDMA support
- Network configuration issues
- Incorrect environment variables

## Full Benchmark Suite

For comprehensive benchmarks, use the full benchmark suite:

```bash
cd ../../benchmarks/communication
mpirun -np 4 python run_all.py --scan
```

## Troubleshooting

**MPI not available**: Ensure MVAPICH-Plus is in your PATH:
```bash
export PATH=$MVAPICH_HOME/bin:$PATH
```

**Low performance**: Enable GPUDirect RDMA:
```bash
export MV2_USE_CUDA=1
export MV2_USE_GPUDIRECT_RDMA=1
```

**Processes hang**: Check firewall and network connectivity between nodes.

---

**Maintained by**: NOWLAB Team, The Ohio State University
