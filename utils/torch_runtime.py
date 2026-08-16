"""Process-local runtime configuration for Torch compilation."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


_INDUCTOR_CACHE: tempfile.TemporaryDirectory[str] | None = None


def configure_temporary_inductor_cache() -> Path:
    """Use an ephemeral Inductor cache that is removed at normal shutdown."""
    global _INDUCTOR_CACHE
    if _INDUCTOR_CACHE is None:
        _INDUCTOR_CACHE = tempfile.TemporaryDirectory(
            prefix="manipulator_torch_inductor_"
        )
    cache_path = Path(_INDUCTOR_CACHE.name)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_path)
    return cache_path
