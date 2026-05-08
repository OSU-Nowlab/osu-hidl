from __future__ import annotations

import os
import shutil
from pathlib import Path


def _arch_tag() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return f"sm{major}{minor}"
    except Exception:
        pass
    return "no_cuda"


def cache_root() -> Path:
    base = os.environ.get("MAC_WORKSPACE_BASE") or os.environ.get("MAC_ATTENTION_WORKSPACE_BASE")
    if base:
        base_path = Path(base).expanduser()
    else:
        base_path = Path.home()
    return base_path / ".cache" / "mac" / _arch_tag()


def write_text_if_different(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        existing = None
    if existing == content:
        return
    path.write_text(content, encoding="utf-8")


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
