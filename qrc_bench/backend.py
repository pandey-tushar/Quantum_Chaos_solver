"""Array backend selection: numpy (CPU) or cupy (GPU)."""
from __future__ import annotations

import numpy as np


def get_xp(name: str = "numpy"):
    if name == "numpy":
        return np
    if name == "cupy":
        import cupy
        return cupy
    raise ValueError(f"unknown backend {name!r}; expected 'numpy' or 'cupy'")


def to_numpy(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)
