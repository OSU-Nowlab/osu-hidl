from __future__ import annotations

from pathlib import Path


def method_root() -> Path:
    """Return the root of the vendored MAC-Attention method tree."""
    return Path(__file__).resolve().parents[2]


def extension_source_path(name: str) -> Path:
    """Resolve a standalone CUDA source shipped with the method tree."""
    path = method_root() / "ext" / name
    if not path.is_file():
        raise FileNotFoundError(f"MAC-Attention extension source not found: {path}")
    return path
