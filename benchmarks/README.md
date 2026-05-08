# Benchmarks

Performance benchmarks for the OSU HPC-AI stack.

## Purpose

Benchmarks measure **performance** to detect regressions between releases. They complement correctness tests by answering: "Given that we have correctness, does performance drop dramatically in this release vs the last one?"

## Structure

```
benchmarks/
├── communication/       # Communication operation benchmarks
│   ├── all_reduce.py
│   ├── broadcast.py
│   ├── all_gather.py
│   ├── scatter.py
│   ├── gather.py
│   ├── barrier.py
│   ├── utils.py
│   ├── run_all.py
│   └── README.md
├── mac-attention/       # MAC-Attention method benchmarks
│   ├── bench_match.py
│   ├── bench_prefill_update_cache.py
│   ├── bench_plan_attention.py
│   ├── bench_rectification_cache.py
│   └── README.md
└── README.md
```

## Running Benchmarks

### Individual Communication Operation

Run a single communication benchmark with default message size:
```bash
cd benchmarks/communication
mpirun -np 4 python all_reduce.py
```

Scan across message sizes:
```bash
mpirun -np 4 python all_reduce.py --scan
```

### All Communication Benchmarks

Run all communication benchmarks:
```bash
cd benchmarks/communication
mpirun -np 4 python run_all.py
```

Run specific operations with scanning:
```bash
mpirun -np 4 python run_all.py --scan --all-reduce --broadcast
```

### MAC-Attention Benchmarks

Run the standalone MAC-Attention performance scripts from the repository root:

```bash
python benchmarks/mac-attention/bench_match.py
python benchmarks/mac-attention/bench_prefill_update_cache.py
python benchmarks/mac-attention/bench_plan_attention.py
python benchmarks/mac-attention/bench_rectification_cache.py
```

## Benchmark vs Test

- **Tests** (in `tests/`): Verify correctness - do we get expected results within tolerance?
- **Benchmarks** (in `benchmarks/`): Measure performance - is performance acceptable compared to baseline?

## Adding New Benchmarks

See [communication/README.md](communication/README.md) for details on adding new communication benchmarks.

---

**Maintained by**: NOWLAB Team, The Ohio State University
