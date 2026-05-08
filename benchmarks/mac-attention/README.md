# MAC-Attention Benchmarks

Performance benchmarks for the standalone MAC-Attention kernels and wrappers.

These scripts measure different parts of the MAC-Attention workflow. The match
kernel and decode attention path are on the critical path during decoding. The
prefill cache update and rectification cache update are designed to be overlapped.

## Prerequisites

Install the local MAC-Attention package without replacing the existing PyTorch
environment:

```bash
python -m pip install -e methods/mac-attention --no-deps
```

Run these benchmarks on a CUDA system. The imported wrapper kernels are
currently documented as Hopper-targeted because the source hardcodes `sm_90`
and `sm_90a` build targets.

## Available Benchmarks

| Script | Purpose | Output |
|------|------|------|
| `bench_match.py` | Measures the query-match kernel and its scheduler | `results/bench_match_results.csv` |
| `bench_prefill_update_cache.py` | Measures prefill cache population | `results/bench_prefill_update_cache_results.csv` |
| `bench_plan_attention.py` | Measures decode planning and MAC attention latency | `results/bench_plan_attention_results.csv` |
| `bench_rectification_cache.py` | Measures rectification and cache update latency | `results/bench_rectification_cache_results.csv` |

## Running Benchmarks

Run a single benchmark from the repository root:

```bash
python benchmarks/mac-attention/bench_match.py
python benchmarks/mac-attention/bench_prefill_update_cache.py
python benchmarks/mac-attention/bench_plan_attention.py
python benchmarks/mac-attention/bench_rectification_cache.py
```

All CSV output is written under `benchmarks/mac-attention/results/`.
The repository keeps that directory empty by default. Benchmark runs generate
their CSV files locally.
