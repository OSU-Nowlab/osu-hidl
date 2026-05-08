# Communication Benchmarks

The intent of these benchmarks is to measure communication latency/bandwidth of PyTorch distributed (with MVAPICH-Plus backend) communication operations at the Python layer. These benchmarks are complementary to C-level comms benchmarks like [OSU Micro-Benchmarks](https://mvapich.cse.ohio-state.edu/benchmarks/) in that users can:
- Easily debug which layer of the communication software stack hangs or performance degradations originate from.
- Measure the expected communication performance of PyTorch distributed with MPI backend.
- Detect performance regressions between releases.

## Running Benchmarks

### Single Communication Operation

Run with a single large message size (calculated to fit within GPU memory):
```bash
mpirun -np 4 python all_reduce.py
```

Scan across message sizes:
```bash
mpirun -np 4 python all_reduce.py --scan
```

With SLURM:
```bash
srun -n 4 python all_reduce.py --scan
```

### All Communication Benchmarks

Run all available communication benchmarks:
```bash
mpirun -np 4 python run_all.py
```

Like the individual benchmarks, `run_all.py` supports scanning arguments for the max message size, bandwidth-unit, etc. Simply pass the desired arguments to `run_all.py` and they'll be propagated to each comm op.

Users can choose specific communication operations to run in `run_all.py` by passing them as arguments (all operations are run by default):
```bash
mpirun -np 4 python run_all.py --scan --all-reduce --broadcast --all-gather
```

## Available Arguments

```
usage: <benchmark>.py [-h] [--local_rank LOCAL_RANK] [--trials TRIALS] [--warmups WARMUPS]
                      [--maxsize MAXSIZE] [--bw-unit {Gbps,GBps}] [--backend {mpi,gloo}]
                      [--scan] [--raw] [--dtype DTYPE] [--mem-factor MEM_FACTOR]

options:
  -h, --help            show this help message and exit
  --local_rank LOCAL_RANK
  --trials TRIALS       Number of timed iterations (default: 100)
  --warmups WARMUPS     Number of warmup (non-timed) iterations (default: 50)
  --maxsize MAXSIZE     Max message size as a power of 2 (default: 24 = 16M elements)
  --bw-unit {Gbps,GBps} Bandwidth unit to report (default: Gbps)
  --backend {mpi,gloo}
                        Communication library to use (default: mpi for OSU HPC-AI)
  --scan                Enables scanning all message sizes
  --raw                 Print the message size and latency without units
  --dtype DTYPE         PyTorch tensor dtype (default: float32)
  --mem-factor MEM_FACTOR
                        Proportion of max available GPU memory to use for single-size evals
```

## Benchmark Output

Each benchmark outputs:
- Message size (in number of elements or bytes)
- Latency (in microseconds or milliseconds)
- Bandwidth (in Gbps or GBps)
- Number of processes used

This data can be used to:
- Establish performance baselines
- Detect regressions between releases
- Compare with theoretical peak performance
- Identify performance bottlenecks

---

**Maintained by**: NOWLAB Team, The Ohio State University
