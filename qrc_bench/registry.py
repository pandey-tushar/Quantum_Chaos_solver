"""Name -> implementation lookup for every swappable pipeline stage."""
from __future__ import annotations

KINDS = ("task", "reservoir", "baseline")
_REGISTRY: dict[str, dict[str, object]] = {k: {} for k in KINDS}


def register(kind: str, name: str):
    """Decorator: ``@register("reservoir", "ising_xx")``."""
    if kind not in _REGISTRY:
        raise KeyError(f"unknown kind {kind!r}; expected one of {KINDS}")

    def deco(obj):
        if name in _REGISTRY[kind]:
            raise KeyError(f"{kind} {name!r} registered twice")
        _REGISTRY[kind][name] = obj
        return obj
    return deco


def get(kind: str, name: str):
    _load_all()
    try:
        return _REGISTRY[kind][name]
    except KeyError:
        raise KeyError(f"unknown {kind} {name!r}; available: {sorted(_REGISTRY[kind])}") from None


def names(kind: str) -> list[str]:
    _load_all()
    return sorted(_REGISTRY[kind])


def _load_all():
    # importing the subpackages runs their @register decorators
    import qrc_bench.tasks  # noqa: F401
    import qrc_bench.reservoirs  # noqa: F401
    import qrc_bench.baselines  # noqa: F401
