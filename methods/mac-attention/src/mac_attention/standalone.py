from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

from torch.utils.cpp_extension import load as load_cpp_ext

from ._repo import extension_source_path


def torch_extension_build_dir() -> str:
    base = os.environ.get("MAC_WORKSPACE_BASE") or os.environ.get("MAC_ATTENTION_WORKSPACE_BASE")
    base_path = Path(base).expanduser() if base else Path.home()
    build_dir = base_path / ".cache" / "mac" / "torch_extensions"
    build_dir.mkdir(parents=True, exist_ok=True)
    return str(build_dir)


def load_standalone_extension(
    *,
    name: str,
    sources: Iterable[str],
    verbose: bool = False,
    extra_cuda_cflags: Iterable[str] | None = None,
    extra_cflags: Iterable[str] | None = None,
) -> Any:
    cuda_flags = list(extra_cuda_cflags or [])
    cflags = list(extra_cflags or [])
    return load_cpp_ext(
        name=name,
        sources=list(sources),
        verbose=verbose,
        extra_cuda_cflags=cuda_flags,
        extra_cflags=cflags,
        build_directory=torch_extension_build_dir(),
    )


def load_mac_match_extension(*, verbose: bool = False) -> Any:
    return load_standalone_extension(
        name="macMatch_ext",
        sources=[str(extension_source_path("macMatch.cu"))],
        verbose=verbose,
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-Xptxas",
            "-dlcm=cg",
        ],
    )


def load_mac_prefill_update_cache_extension(*, verbose: bool = False) -> Any:
    return load_standalone_extension(
        name="mac_prefill_update_cache_ext",
        sources=[str(extension_source_path("mac_prefill_update_cache.cu"))],
        verbose=verbose,
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-Xptxas",
            "-dlcm=cg",
        ],
    )
