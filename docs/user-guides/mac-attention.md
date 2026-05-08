# MAC-Attention User Guide

MAC-Attention is a method for accelerating long-context decoding by reusing
attention work from semantically similar recent queries. It is not a serving
framework by itself. It is designed to replace the attention path inside a
framework such as SGLang or vLLM.

Three parameters control its behavior. `tau` is the similarity threshold for
declaring a match. `K` is the number of recent queries kept in the search
window. `r` is the rectification band width, which is the number of tokens
recomputed near the reuse boundary.

## How It Works

MAC-Attention breaks decode-time reuse into three steps:

1. Match compares the current pre-RoPE query against a short ring buffer of
   recent pre-RoPE queries.
2. Amend rectifies the reused result by recomputing a short band near the match
   boundary.
3. Complete computes attention on the remaining tail and merges it with the
   reused prefix summary in a numerically stable way.

When a good match is found, the cost of the reused prefix becomes effectively
constant with respect to context length. When no match is found, the method
falls back to normal attention behavior.

## Current Repo Layout

The monorepo currently exposes MAC-Attention in four places:

- `methods/mac-attention/` contains the reusable source package and standalone kernels.
- `examples/mac-attention/` contains a compact workflow example.
- `benchmarks/mac-attention/` contains standalone performance scripts.
- `tests/mac-attention/` contains correctness tests.

This repository state does not yet wire MAC-Attention directly into
`frameworks/sglang` or `frameworks/vllm`. It provides the method package and the
supporting validation assets needed for that later integration step.

## Quick Start

Install the local package:

```bash
python -m pip install -e methods/mac-attention --no-deps
```

Run the example:

```bash
python examples/mac-attention/e2e_workflow.py --steps 1
```

Run the tests:

```bash
python -m pytest tests/mac-attention -q
```

## Parameter Guidance

The MLSys 2026 paper reports using `K = 1024` and `r = 256` across benchmarks,
with `tau = 0.45` for context-heavy tasks and `tau = 0.75` for generation-heavy
tasks. Those are good starting points for this repo as well.

Lower `tau` usually produces more matches but looser reuse. Higher `tau` is
stricter and usually preserves fidelity better. Larger `K` searches farther back
in time but increases match work. Larger `r` recomputes more tokens near the
boundary and usually improves accuracy at the cost of some saved work.

## Hardware Note

The paper evaluates MAC-Attention with SGLang 0.4.9 and FlashInfer 0.2.7 on
NVIDIA H100 SXM5 with CUDA 12.8.1.

Separately, based on the imported source in this repo, the full JIT wrapper path
should currently be treated as Hopper-targeted because it hardcodes `sm_90` and
`sm_90a` nvcc targets. That is a source-level observation, not a claim from the
paper itself.
