from __future__ import annotations

import functools
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple

import torch
from filelock import FileLock
from torch.utils.cpp_extension import load as load_cpp_ext

from .utils import cache_root, write_text_if_different


_DTYPE_TAG: Dict[torch.dtype, str] = {
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float32: "f32",
}


def _dtype_tag(dtype: torch.dtype) -> str:
    if dtype not in _DTYPE_TAG:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return _DTYPE_TAG[dtype]


def _posenc_literal(pos_encoding_mode: int) -> str:
    if pos_encoding_mode == 0:
        return "flashinfer::PosEncodingMode::kNone"
    if pos_encoding_mode == 1:
        return "flashinfer::PosEncodingMode::kRoPELlama"
    if pos_encoding_mode == 2:
        return "flashinfer::PosEncodingMode::kALiBi"
    raise ValueError(f"Unsupported pos_encoding_mode: {pos_encoding_mode}")


def _ctype_for_dtype(dtype: torch.dtype) -> str:
    if dtype is torch.float16:
        return "nv_half"
    if dtype is torch.bfloat16:
        return "nv_bfloat16"
    if dtype is torch.float32:
        return "float"
    raise ValueError(f"Unsupported dtype: {dtype}")


def _uri(
    *,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    head_dim: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
) -> str:
    return (
        "mac_decode_"
        f"q_{_dtype_tag(q_dtype)}_"
        f"kv_{_dtype_tag(kv_dtype)}_"
        f"hd_{int(head_dim)}_"
        f"pos_{int(pos_encoding_mode)}_"
        f"swa_{1 if use_sliding_window else 0}_"
        f"cap_{1 if use_logits_soft_cap else 0}"
    )


def _package_root() -> Path:
    # src/mac_attention/_jit/mac_decode.py -> src/mac_attention
    return Path(__file__).resolve().parents[1]


def _vendor_dir() -> Path:
    return _package_root() / "_vendor"


def _vendor_include_dir() -> Path:
    return _vendor_dir() / "include"


def _vendor_csrc_dir() -> Path:
    return _vendor_dir() / "csrc"


def _generated_config_inc(
    *,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    head_dim: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
) -> str:
    dtype_q = _ctype_for_dtype(q_dtype)
    dtype_kv = _ctype_for_dtype(kv_dtype)
    posenc = _posenc_literal(pos_encoding_mode)
    use_swa = "true" if use_sliding_window else "false"
    use_cap = "true" if use_logits_soft_cap else "false"
    return f"""#pragma once
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/page.cuh>
#include <flashinfer/pos_enc.cuh>

#include "pytorch_extension_utils.h"

using IdType = int32_t;
using DTypeQ = {dtype_q};
using DTypeKV = {dtype_kv};
using DTypeO = DTypeQ;

static constexpr uint32_t HEAD_DIM_QK = {int(head_dim)};
static constexpr flashinfer::PosEncodingMode POS_ENCODING_MODE = {posenc};
static constexpr bool USE_SLIDING_WINDOW = {use_swa};
static constexpr bool USE_LOGITS_SOFT_CAP = {use_cap};

using AttentionVariant = flashinfer::MACAttention</*use_custom_mask=*/false, USE_SLIDING_WINDOW,
                                                   USE_LOGITS_SOFT_CAP, /*use_alibi_bias=*/false>;
using Params = flashinfer::MACDecodeParams<DTypeQ, DTypeKV, DTypeO, IdType>;
"""


def _generated_kernel_inst(
    *,
    q_dtype: torch.dtype,
    head_dim: int,
    pos_encoding_mode: int,
) -> str:
    dtype_o = _ctype_for_dtype(q_dtype)
    posenc = _posenc_literal(pos_encoding_mode)
    return f"""#include <flashinfer/attention/decode.cuh>
#include "mac_decode_config.inc"

using namespace flashinfer;

namespace flashinfer {{
template cudaError_t MACDecodeWithPagedKVCacheDispatched<
    {int(head_dim)}, {posenc}, AttentionVariant, Params>(
    Params params, {dtype_o}* tmp_v, float* tmp_s, {dtype_o}* tmp_v_dd, float* tmp_s_dd,
    bool enable_pdl, cudaStream_t stream);
}}
"""


@functools.lru_cache(maxsize=None)
def get_mac_decode_op(
    *,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    head_dim: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
) -> SimpleNamespace:
    if not torch.cuda.is_available():
        raise RuntimeError("mac_attention requires CUDA; torch.cuda.is_available() is False")

    name = _uri(
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        head_dim=head_dim,
        pos_encoding_mode=pos_encoding_mode,
        use_sliding_window=use_sliding_window,
        use_logits_soft_cap=use_logits_soft_cap,
    )

    root = cache_root()
    gen_dir = root / "generated" / name
    build_dir = root / "built" / name

    lock_path = root / "locks" / f"{name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path), thread_local=False):
        gen_dir.mkdir(parents=True, exist_ok=True)
        build_dir.mkdir(parents=True, exist_ok=True)

        # Generate per-module config + explicit instantiations.
        write_text_if_different(
            gen_dir / "mac_decode_config.inc",
            _generated_config_inc(
                q_dtype=q_dtype,
                kv_dtype=kv_dtype,
                head_dim=head_dim,
                pos_encoding_mode=pos_encoding_mode,
                use_sliding_window=use_sliding_window,
                use_logits_soft_cap=use_logits_soft_cap,
            ),
        )
        write_text_if_different(
            gen_dir / "mac_decode_kernel_inst.cu",
            _generated_kernel_inst(
                q_dtype=q_dtype,
                head_dim=head_dim,
                pos_encoding_mode=pos_encoding_mode,
            ),
        )

        verbose = (
            os.environ.get("MAC_JIT_VERBOSE", os.environ.get("MAC_ATTENTION_JIT_VERBOSE", "0"))
            == "1"
        )

        extra_cuda_cflags = [
            "-O3",
            "--use_fast_math",
            "-lineinfo",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-DENABLE_NVTX=0",
            "-DENABLE_TIMING=0",
            "-DFLASHINFER_ENABLE_F16",
            "-DFLASHINFER_ENABLE_BF16",
            "-DFLASHINFER_ENABLE_FP8_E4M3",
            "-DFLASHINFER_ENABLE_FP8_E5M2",
            "-gencode=arch=compute_90,code=sm_90",
            "-gencode=arch=compute_90a,code=sm_90a",
        ]
        extra_cflags = ["-O3", "-std=c++17", "-Wno-switch-bool"]

        # Compile + load. load_cpp_ext is idempotent when name+build_directory are stable.
        load_cpp_ext(
            name=name,
            sources=[
                str(_vendor_csrc_dir() / "mac_decode.cu"),
                str(_vendor_csrc_dir() / "mac_decode_torchlib.cu"),
                str(gen_dir / "mac_decode_kernel_inst.cu"),
            ],
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            with_cuda=True,
            extra_include_paths=[
                str(gen_dir),
                str(_vendor_include_dir()),
                str(_vendor_csrc_dir()),
            ],
            build_directory=str(build_dir),
            verbose=verbose,
        )

    ns = getattr(torch.ops, name)
    # Cache the default overload callables to minimize per-call dispatcher overhead.
    return SimpleNamespace(plan=ns.plan.default, run=ns.run.default)
