from setuptools import setup, find_packages

setup(
    name="osu-hpc-ai",
    version="0.1.0",
    description=
    "OSU HPC-AI benchmarks, recipes, and tests for distributed DL on HPC",
    packages=find_packages(exclude=("tests", "tests.*")),
    python_requires=">=3.9",
    include_package_data=True,
    install_requires=[],
    extras_require={
        "dev": ["pytest", "packaging"],
        "deepspeed": ["deepspeed"],
        "vllm": ["vllm"],
        "sglang": ["sglang"],
    },
    entry_points={
        "console_scripts": [
            "hpc-ai-bench-all-reduce=benchmarks.communication.all_reduce:main",
            "hpc-ai-bench-broadcast=benchmarks.communication.broadcast:main",
            "hpc-ai-bench-all-gather=benchmarks.communication.all_gather:main",
            "hpc-ai-bench-gather=benchmarks.communication.gather:main",
            "hpc-ai-bench-scatter=benchmarks.communication.scatter:main",
            "hpc-ai-bench-barrier=benchmarks.communication.barrier:main",
            "hpc-ai-bench-suite=benchmarks.communication.run_all:main",
        ]
    },
)
