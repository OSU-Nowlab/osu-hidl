# MAC-Attention Tests

Correctness tests for the imported MAC-Attention method package.

This test suite covers the decode wrapper, the rectification cache wrapper, and
the standalone match kernel. The tests are written to skip cleanly when CUDA is
not available.

## Prerequisites

Install the local MAC-Attention package without replacing the existing PyTorch
environment:

```bash
python -m pip install -e methods/mac-attention --no-deps
```

## Running Tests

From the repository root:

```bash
python -m pytest tests/mac-attention -q
```

Or through the shared test runner:

```bash
python tests/run_tests.py --mac-attention
```
