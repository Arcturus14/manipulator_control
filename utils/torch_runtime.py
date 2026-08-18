"""Persistent workspace-local caches for Torch compilation."""
from __future__ import annotations

import os
from pathlib import Path


_CACHE_ROOT = Path(__file__).resolve().parents[1] / ".torch_cache"


def configure_compilation_cache() -> Path:
    """Keep Inductor/Triton kernels across simulation processes.

    Torch keys compiled artifacts by the graph and runtime configuration, so
    stale entries are naturally bypassed after a controller or environment
    change.  Keeping this directory inside the workspace also makes its
    lifecycle explicit and lets ``.gitignore`` exclude it from source control.
    """
    inductor_path = _CACHE_ROOT / "inductor"
    triton_path = _CACHE_ROOT / "triton"
    inductor_path.mkdir(parents=True, exist_ok=True)
    triton_path.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_path)
    os.environ["TRITON_CACHE_DIR"] = str(triton_path)
    return _CACHE_ROOT
