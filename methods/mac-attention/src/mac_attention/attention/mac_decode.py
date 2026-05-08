from __future__ import annotations

import math
from typing import Any, Optional, Tuple, Union

import torch

from .._jit.mac_decode import get_mac_decode_op

PagedKVCache = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


_ALIBI_CACHE = {}


def _canonicalize_dtype(dtype: Optional[Union[str, torch.dtype]]) -> Optional[torch.dtype]:
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        key = dtype.strip()
        # Common aliases
        if key in {"fp16", "f16", "float16", "half"}:
            return torch.float16
        if key in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if key in {"fp32", "f32", "float32"}:
            return torch.float32
        try:
            return getattr(torch, key)
        except AttributeError as exc:
            raise ValueError(f"Unknown dtype string: {dtype}") from exc
    raise TypeError(f"dtype must be torch.dtype or str, got {type(dtype)}")


def _pos_encoding_mode_to_int(mode: str) -> int:
    m = (mode or "NONE").strip().upper()
    if m in {"NONE", "NOPE"}:
        return 0
    if m in {"ROPE_LLAMA", "LLAMA"}:
        return 1
    if m in {"ALIBI"}:
        return 2
    raise ValueError(f"Unsupported pos_encoding_mode: {mode!r}")


def _kv_layout_to_code(kv_layout: str) -> int:
    layout = kv_layout.strip().upper()
    if layout == "NHD":
        return 0
    if layout == "HND":
        return 1
    raise ValueError(f"kv_layout must be 'NHD' or 'HND', got {kv_layout!r}")


def _device_support_pdl(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    try:
        major, _minor = torch.cuda.get_device_capability(device)
        return major >= 9
    except Exception:
        return False


def _unpack_paged_kv_cache(
    paged_kv_cache: PagedKVCache, kv_layout: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(paged_kv_cache, (tuple, list)):
        if len(paged_kv_cache) != 2:
            raise ValueError("paged_kv_cache tuple must be (k_cache, v_cache)")
        return paged_kv_cache[0], paged_kv_cache[1]
    if not torch.is_tensor(paged_kv_cache):
        raise TypeError(
            f"paged_kv_cache must be a Tensor or (Tensor,Tensor), got {type(paged_kv_cache)}"
        )
    if paged_kv_cache.dim() != 5:
        raise ValueError("paged_kv_cache Tensor must be 5D: [max_pages,2,...]")
    # Both NHD and HND store (k,v) on dim=1.
    k_cache = paged_kv_cache[:, 0]
    v_cache = paged_kv_cache[:, 1]
    return k_cache, v_cache


def _get_alibi_slopes(num_qo_heads: int, device: torch.device) -> torch.Tensor:
    # Match flashinfer.utils.get_alibi_slopes
    key = (num_qo_heads, device)
    buf = _ALIBI_CACHE.get(key)
    if buf is not None:
        return buf
    n = 2 ** math.floor(math.log2(num_qo_heads))
    m0 = 2.0 ** (-8.0 / n)
    m = torch.pow(m0, torch.arange(1, 1 + n, device=device, dtype=torch.float32))
    if n < num_qo_heads:
        mhat0 = 2.0 ** (-4.0 / n)
        mhat = torch.pow(
            mhat0,
            torch.arange(1, 1 + 2 * (num_qo_heads - n), 2, device=device, dtype=torch.float32),
        )
        m = torch.cat([m, mhat])
    buf = m.float()
    _ALIBI_CACHE[key] = buf
    return buf


class MACDecodeWithPagedKVCacheWrapper:
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        use_cuda_graph: bool = False,
        use_tensor_cores: bool = False,
    ) -> None:
        if use_cuda_graph:
            raise ValueError("mac_attention v1 does not support use_cuda_graph=True")
        if use_tensor_cores:
            raise ValueError("mac_attention v1 does not support use_tensor_cores=True")
        if not torch.is_tensor(float_workspace_buffer) or not float_workspace_buffer.is_cuda:
            raise ValueError("float_workspace_buffer must be a CUDA tensor")
        if float_workspace_buffer.dtype != torch.uint8:
            raise ValueError("float_workspace_buffer must be a torch.uint8 CUDA tensor")

        self._kv_layout = kv_layout.strip().upper()
        _kv_layout_to_code(self._kv_layout)

        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device

        self._int_workspace_buffer = torch.empty(
            (8 * 1024 * 1024,), dtype=torch.uint8, device=self.device
        )
        self._pin_memory_int_workspace_buffer = torch.empty(
            (8 * 1024 * 1024,), dtype=torch.uint8, device="cpu", pin_memory=True
        )

        self._attn_start_pos_host_pinned: Optional[torch.Tensor] = None

        self._paged_kv_indptr_buf: Optional[torch.Tensor] = None
        self._paged_kv_indices_buf: Optional[torch.Tensor] = None
        self._paged_kv_last_page_len_buf: Optional[torch.Tensor] = None
        self._indptr_host_buf: Optional[torch.Tensor] = None

        self._plan_info: Optional[torch.Tensor] = None
        self._cached_module = None
        self._cached_op_key: Optional[Tuple[object, ...]] = None

        self._cached_q_data_type: Optional[torch.dtype] = None
        self._cached_kv_data_type: Optional[torch.dtype] = None

        self._pos_encoding_mode: str = "NONE"
        self._window_left: int = -1
        self._logits_soft_cap: float = 0.0
        self._sm_scale: Optional[float] = None
        self._rope_scale: float = 1.0
        self._rope_theta: float = 1e4

        self._planned_downdate_range: int = 0
        self._planned_posenc_i: int = 0
        self._planned_use_sliding_window: bool = False
        self._planned_use_logits_cap: bool = False
        self._planned_head_dim: int = 0

    begin_forward = plan = None  # overwritten below

    def plan(
        self,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        *,
        pos_encoding_mode: str = "NONE",
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        q_data_type: Optional[Union[str, torch.dtype]] = "float16",
        kv_data_type: Optional[Union[str, torch.dtype]] = None,
        data_type: Optional[Union[str, torch.dtype]] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
        non_blocking: bool = True,
        attn_start_pos: Optional[torch.Tensor] = None,
        downdate_range: int = 0,
        attn_start_pos_host_pinned_opt: Optional[torch.Tensor] = None,
    ) -> None:
        batch_size = int(last_page_len.numel())
        if page_size != 1:
            raise ValueError(
                f"mac_attention v1 requires page_size == 1, got page_size={page_size}"
            )
        if attn_start_pos is None:
            raise ValueError("attn_start_pos is required for MACDecode planning")
        if attn_start_pos.dtype != torch.int32:
            raise ValueError("attn_start_pos must be torch.int32")
        if not attn_start_pos.is_cuda or attn_start_pos.device != self.device:
            raise ValueError("attn_start_pos must be on the same CUDA device as the wrapper")
        if attn_start_pos.numel() != batch_size * num_qo_heads:
            raise ValueError("attn_start_pos must have shape [batch_size, num_qo_heads]")

        if data_type is not None:
            if q_data_type is None:
                q_data_type = data_type
            if kv_data_type is None:
                kv_data_type = data_type

        q_dtype = _canonicalize_dtype(q_data_type)
        if q_dtype is None:
            raise ValueError("q_data_type must be provided")
        kv_dtype = _canonicalize_dtype(kv_data_type) if kv_data_type is not None else q_dtype
        if kv_dtype is None:
            kv_dtype = q_dtype

        if logits_soft_cap is None:
            logits_soft_cap = 0.0

        posenc_i = _pos_encoding_mode_to_int(pos_encoding_mode)
        use_sliding_window = window_left != -1
        use_logits_cap = float(logits_soft_cap) > 0.0

        self._cached_q_data_type = q_dtype
        self._cached_kv_data_type = kv_dtype

        # Cache runtime params for run()
        self._pos_encoding_mode = pos_encoding_mode
        self._window_left = int(window_left)
        self._logits_soft_cap = float(logits_soft_cap)
        self._sm_scale = sm_scale
        if rope_scale is not None:
            self._rope_scale = float(rope_scale)
        if rope_theta is not None:
            self._rope_theta = float(rope_theta)

        self._planned_downdate_range = int(downdate_range)
        self._planned_posenc_i = int(posenc_i)
        self._planned_use_sliding_window = bool(use_sliding_window)
        self._planned_use_logits_cap = bool(use_logits_cap)
        self._planned_head_dim = int(head_dim)

        # Store page tables on device for run().
        if indptr.device == self.device:
            self._paged_kv_indptr_buf = indptr
        else:
            self._paged_kv_indptr_buf = indptr.to(self.device, non_blocking=non_blocking)
        if indices.device == self.device:
            self._paged_kv_indices_buf = indices
        else:
            self._paged_kv_indices_buf = indices.to(self.device, non_blocking=non_blocking)
        if last_page_len.device == self.device:
            self._paged_kv_last_page_len_buf = last_page_len
        else:
            self._paged_kv_last_page_len_buf = last_page_len.to(
                self.device, non_blocking=non_blocking
            )

        # Host indptr for planning (reuse a small CPU buffer to avoid per-call allocations).
        if indptr.device.type == "cpu":
            indptr_host = indptr.contiguous()
        else:
            if (
                self._indptr_host_buf is None
                or self._indptr_host_buf.dtype != indptr.dtype
                or self._indptr_host_buf.shape != indptr.shape
            ):
                self._indptr_host_buf = torch.empty(
                    indptr.shape, dtype=indptr.dtype, device="cpu", pin_memory=True
                )
            # Copy indptr from device to pinned host memory every plan() call.
            # Use non_blocking=True and rely on the scheduler's stream sync to ensure
            # the data is visible before the CPU reads it.
            self._indptr_host_buf.copy_(indptr, non_blocking=True)
            indptr_host = self._indptr_host_buf

        # Pinned host buffer for attn_start_pos D2H copy used by the tuned scheduler.
        needed = batch_size * num_qo_heads
        if attn_start_pos_host_pinned_opt is not None:
            buf = attn_start_pos_host_pinned_opt
            if buf.device.type != "cpu" or not buf.is_pinned():
                raise ValueError("attn_start_pos_host_pinned_opt must be a pinned CPU tensor")
            if buf.dtype != torch.int32:
                raise ValueError("attn_start_pos_host_pinned_opt must be torch.int32")
            if buf.numel() < needed:
                raise ValueError(
                    "attn_start_pos_host_pinned_opt must have numel >= batch_size*num_qo_heads"
                )
            self._attn_start_pos_host_pinned = buf.contiguous()
        else:
            if (
                self._attn_start_pos_host_pinned is None
                or self._attn_start_pos_host_pinned.numel() < needed
            ):
                # Over-allocate a bit to reduce reallocs.
                cap = 1
                while cap < needed:
                    cap *= 2
                self._attn_start_pos_host_pinned = torch.empty(
                    (cap,), dtype=torch.int32, device="cpu", pin_memory=True
                )

        assert self._attn_start_pos_host_pinned is not None

        op_key = (
            q_dtype,
            kv_dtype,
            int(head_dim),
            int(posenc_i),
            bool(use_sliding_window),
            bool(use_logits_cap),
        )
        if self._cached_module is None or self._cached_op_key != op_key:
            self._cached_module = get_mac_decode_op(
                q_dtype=q_dtype,
                kv_dtype=kv_dtype,
                head_dim=int(head_dim),
                pos_encoding_mode=posenc_i,
                use_sliding_window=use_sliding_window,
                use_logits_soft_cap=use_logits_cap,
            )
            self._cached_op_key = op_key

        self._plan_info = self._cached_module.plan(
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._pin_memory_int_workspace_buffer,
            indptr_host,
            batch_size,
            int(num_qo_heads),
            int(num_kv_heads),
            int(page_size),
            False,  # enable_cuda_graph
            int(window_left),
            float(logits_soft_cap),
            int(head_dim),
            int(head_dim),
            attn_start_pos,
            int(downdate_range),
            self._attn_start_pos_host_pinned,
        )

    begin_forward = plan

    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: PagedKVCache,
        max_running_requests: int,
        cache_capacity: int,
        attn_cache: torch.Tensor,
        lse_cache: torch.Tensor,
        hit_indices: torch.Tensor,
        hit_table: torch.Tensor,
        cache_req_ids: torch.Tensor,
        use_cache: bool,
        attn_start_pos: torch.Tensor,
        *args: Any,
        q_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        enable_pdl: Optional[bool] = None,
        downdate_range: Optional[int] = None,
        return_downdated: bool = False,
    ):
        if self._plan_info is None or self._cached_module is None:
            raise RuntimeError("plan(...) must be called before run(...)")

        # Interpret a final positional int as downdate_range for compatibility.
        if args:
            if len(args) == 1 and isinstance(args[0], int) and downdate_range is None:
                downdate_range = int(args[0])
            else:
                raise TypeError(
                    "Unsupported extra positional args for mac_attention MACDecode. "
                    "Only an optional final positional int (downdate_range) is supported."
                )

        if enable_pdl is None:
            enable_pdl = _device_support_pdl(q.device)

        k_cache, v_cache = _unpack_paged_kv_cache(paged_kv_cache, self._kv_layout)

        if q.dtype != self._cached_q_data_type:
            raise ValueError(
                f"q dtype {q.dtype} does not match planned q_data_type {self._cached_q_data_type}"
            )
        if k_cache.dtype != self._cached_kv_data_type:
            raise ValueError(
                f"k dtype {k_cache.dtype} does not match planned kv_data_type {self._cached_kv_data_type}"
            )

        if q.device != self.device:
            raise ValueError("q must be on the same CUDA device as the wrapper")

        if downdate_range is None:
            downdate_range = self._planned_downdate_range
        else:
            dr = int(downdate_range)
            if dr > self._planned_downdate_range:
                raise ValueError(
                    "downdate_range at run-time must be <= the value used in plan(). "
                    f"planned={self._planned_downdate_range}, got={dr}"
                )
            downdate_range = dr

        logits_soft_cap = float(self._logits_soft_cap)
        sm_scale = self._sm_scale
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(q.shape[-1])
        if q_scale is not None:
            sm_scale *= float(q_scale)
        if k_scale is not None:
            sm_scale *= float(k_scale)

        rope_scale = float(self._rope_scale) if self._rope_scale is not None else 1.0
        rope_theta = float(self._rope_theta) if self._rope_theta is not None else 1e4

        if out is None:
            out = torch.empty_like(q)
        if return_lse and lse is None:
            lse = torch.empty((q.size(0), q.size(1)), dtype=torch.float32, device=q.device)

        # Always allocate downdated outputs (kernel expects pointers).
        downdated_out = torch.empty_like(q)
        downdated_lse = torch.empty((q.size(0), q.size(1)), dtype=torch.float32, device=q.device)

        # Optional alibi slopes.
        posenc_i = _pos_encoding_mode_to_int(self._pos_encoding_mode)
        if posenc_i == 2:
            alibi_slopes = _get_alibi_slopes(q.size(1), q.device)
        else:
            alibi_slopes = torch.empty((0,), dtype=torch.float32, device=q.device)

        self._cached_module.run(
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._plan_info,
            q,
            k_cache,
            v_cache,
            self._paged_kv_indptr_buf,
            self._paged_kv_indices_buf,
            self._paged_kv_last_page_len_buf,
            int(max_running_requests),
            int(cache_capacity),
            attn_cache,
            lse_cache,
            hit_indices,
            hit_table,
            cache_req_ids,
            bool(use_cache),
            attn_start_pos,
            int(downdate_range),
            out,
            lse if return_lse else None,
            downdated_out,
            downdated_lse,
            _kv_layout_to_code(self._kv_layout),
            int(self._window_left),
            bool(enable_pdl),
            alibi_slopes,
            float(logits_soft_cap),
            float(sm_scale),
            float(rope_scale),
            float(rope_theta),
        )

        if v_scale is not None:
            scale = float(v_scale)
            out *= scale
            downdated_out *= scale

        if return_lse:
            if return_downdated:
                return out, lse, downdated_out, downdated_lse
            return out, lse
        return out

    def forward_return_lse(
        self,
        q: torch.Tensor,
        paged_kv_cache: PagedKVCache,
        max_running_requests: int,
        cache_capacity: int,
        attn_cache: torch.Tensor,
        lse_cache: torch.Tensor,
        hit_indices: torch.Tensor,
        hit_table: torch.Tensor,
        cache_req_ids: torch.Tensor,
        use_cache: bool,
        attn_start_pos: torch.Tensor,
        *args: Any,
        q_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
        pos_encoding_mode: Optional[str] = None,
        downdate_range: Optional[int] = None,
        return_downdated: bool = False,
        enable_pdl: Optional[bool] = None,
    ):
        # Allow overriding runtime knobs via this call, but keep compile-time flags consistent with
        # what was selected during plan().
        if pos_encoding_mode is not None:
            posenc_i = _pos_encoding_mode_to_int(pos_encoding_mode)
            if posenc_i != self._planned_posenc_i:
                raise ValueError(
                    "pos_encoding_mode differs from the value used in plan(); re-plan is required. "
                    f"planned={self._pos_encoding_mode!r}, got={pos_encoding_mode!r}"
                )

        use_swa = int(window_left) != -1
        if use_swa != self._planned_use_sliding_window:
            raise ValueError(
                "window_left toggles sliding-window mode vs the plan(); re-plan is required. "
                f"planned_window_left={self._window_left}, got_window_left={window_left}"
            )
        if logits_soft_cap is not None:
            use_cap = float(logits_soft_cap) > 0.0
            if use_cap != self._planned_use_logits_cap:
                raise ValueError(
                    "logits_soft_cap toggles logits-capping vs the plan(); re-plan is required. "
                    f"planned_logits_soft_cap={self._logits_soft_cap}, got_logits_soft_cap={logits_soft_cap}"
                )

        self._window_left = int(window_left)
        if logits_soft_cap is not None:
            self._logits_soft_cap = float(logits_soft_cap)
        self._sm_scale = sm_scale
        if rope_scale is not None:
            self._rope_scale = float(rope_scale)
        if rope_theta is not None:
            self._rope_theta = float(rope_theta)

        # If a final positional int is provided, treat it as downdate_range and request 4 outputs.
        if args:
            if len(args) == 1 and isinstance(args[0], int) and downdate_range is None:
                downdate_range = int(args[0])
                return_downdated = True
                args = ()
            else:
                raise TypeError(
                    "Unsupported extra positional args for forward_return_lse. "
                    "Only an optional final positional int (downdate_range) is supported."
                )

        if downdate_range is not None:
            return_downdated = True

        return self.run(
            q,
            paged_kv_cache,
            max_running_requests,
            cache_capacity,
            attn_cache,
            lse_cache,
            hit_indices,
            hit_table,
            cache_req_ids,
            use_cache,
            attn_start_pos,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            return_lse=True,
            enable_pdl=enable_pdl,
            downdate_range=downdate_range,
            return_downdated=return_downdated,
        )

    # Alias used by some integrations.
    run_return_lse = forward_return_lse

    def end_forward(self) -> None:
        # No-op placeholder for compatibility.
        return
