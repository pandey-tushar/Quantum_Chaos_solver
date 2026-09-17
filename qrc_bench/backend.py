"""Array backend selection: numpy (CPU), cupy (GPU), or auto."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

BACKENDS = ("auto", "numpy", "cupy")
GPU_MIN_QUBITS = 10     # measured crossover on an RTX 2050: below this the per-kernel overhead wins


@lru_cache(maxsize=1)
def gpu_available() -> bool:
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def resolve_backend(name: str, q: int) -> str:
    """'auto' = cupy when a GPU is present and q >= GPU_MIN_QUBITS, numpy otherwise."""
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; expected one of {BACKENDS}")
    if name != "auto":
        return name
    return "cupy" if q >= GPU_MIN_QUBITS and gpu_available() else "numpy"


def get_xp(name: str = "numpy"):
    if name == "numpy":
        return np
    if name == "cupy":
        import cupy
        return cupy
    raise ValueError(f"unknown backend {name!r}; expected 'numpy' or 'cupy'")


def to_numpy(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)
