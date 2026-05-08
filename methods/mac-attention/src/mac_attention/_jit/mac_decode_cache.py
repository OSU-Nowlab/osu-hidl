from __future__ import annotations

# Backward-compat shim (renamed to rectification+cache).

from .mac_rectification_cache import (  # noqa: F401
    get_mac_rectification_cache_op,
)

get_mac_decode_cache_op = get_mac_rectification_cache_op

__all__ = [
    "get_mac_rectification_cache_op",
    "get_mac_decode_cache_op",
]

