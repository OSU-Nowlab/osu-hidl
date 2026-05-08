# MAC-Attention Example

This directory contains a compact end-to-end MAC-Attention workflow example.

The example shows the three main stages used during decoding:

1. Query matching against a short ring buffer of recent queries.
2. MAC decode attention using the match result to skip part of the prefix.
3. Rectification plus cache update for future reuse.

## Prerequisites

Install the local MAC-Attention package without replacing the existing PyTorch
environment:

```bash
python -m pip install -e methods/mac-attention --no-deps
```

Run the example on a CUDA system:

```bash
python examples/mac-attention/e2e_workflow.py --steps 1
```
