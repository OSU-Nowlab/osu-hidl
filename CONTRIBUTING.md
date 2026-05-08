# Contributing to OSU HPC-AI

OSU HPC-AI welcomes your contributions! This document provides guidelines for contributing to the project.

## Prerequisites

Before contributing, ensure you have:

1. **Python 3.9+** installed
2. **Git** configured with your name and email
3. A working build of PyTorch with MPI support (see [INSTALL.md](INSTALL.md))

### Development Setup

```bash
# Clone the repository
git clone --recursive https://github.com/OSU-Nowlab/osu-hpc-ai.git
cd osu-hpc-ai

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -e ".[dev]"
```

### Pre-commit Setup

We use [pre-commit](https://pre-commit.com/) for code quality checks. Install it before making commits:

```bash
pip install pre-commit
pre-commit install
```

Pre-commit will now run automatically on every commit. To run manually on all files:

```bash
pre-commit run --all-files
```

If a check fails, pre-commit will either fix the issue automatically or show you what needs to be fixed. Stage the corrected files and retry your commit.

## Code Style

We follow standard Python conventions:

- Use 4 spaces for indentation (no tabs)
- Maximum line length of 100 characters
- Use descriptive variable and function names
- Add docstrings to all public functions and classes
- Type hints are encouraged but not required

## Testing

OSU HPC-AI uses [pytest](https://docs.pytest.org/) for testing. Tests are organized by framework:

```
tests/
├── pytorch/      # PyTorch installation and MPI tests
├── deepspeed/    # DeepSpeed integration tests
├── vllm/         # vLLM inference tests
├── sglang/       # SGLang serving tests
├── unit/         # Unit tests
└── integration/  # Integration tests
```

### Running Tests

```bash
# Run all tests (requires proper environment setup)
pytest tests/

# Run specific framework tests
pytest tests/pytorch/

# Run a single test file
pytest tests/pytorch/test_installation.py

# Run with verbose output
pytest -v tests/pytorch/test_installation.py
```

### MPI Tests

For tests that require MPI:

```bash
mpirun -np 2 python tests/pytorch/test_mpi_comm.py
```

### Writing Tests

When adding new features, include appropriate tests:

1. **Integration tests** for end-to-end workflows in `tests/integration/`
2. **Framework tests** for framework-specific functionality in `tests/<framework>/`

See [docs/testing_guide.md](docs/testing_guide.md) for the full testing guide.

## Submitting Changes

### Pull Request Process

1. **Fork the repository** and create a new branch from `main`
2. **Make your changes** with clear, descriptive commits
3. **Add tests** for new functionality
4. **Update documentation** if needed
5. **Submit a pull request** with a clear description

### Commit Messages

Write clear, concise commit messages:

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Fix bug" not "Fixes bug")
- Keep the subject line under 72 characters
- Reference issues when applicable ("Fix #123")

Good examples:
```
Add all-reduce benchmark for MPI backend
Fix memory leak in distributed validation script
Update PyTorch build guide for CUDA 12.6
```

### Pull Request Guidelines

Your PR should include:

1. **Summary**: Brief description of changes
2. **Motivation**: Why the change is needed
3. **Test plan**: How to verify the changes work
4. **Documentation**: Updates to relevant docs

## Feature Contributions

For significant new features, we recommend:

1. **Open an issue first** to discuss the feature
2. **Describe the motivation** and use cases
3. **Outline the implementation** approach
4. **Plan for testing** and documentation

### Feature Requirements

New features should:

- Support MPI backend (our primary focus)
- Include unit tests with reasonable coverage
- Have documentation for users
- Follow existing code patterns

## Reporting Issues

When reporting bugs, include:

1. **Environment details**: OS, Python version, CUDA version, MPI implementation
2. **Steps to reproduce**: Minimal example that triggers the issue
3. **Expected behavior**: What should happen
4. **Actual behavior**: What happens instead
5. **Error messages**: Full stack traces if applicable

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).

## Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion

## Acknowledgments

Thank you to all contributors who help improve OSU HPC-AI!

---

**Maintained by**: Network-Based Computing Laboratory (NOWLAB), The Ohio State University
