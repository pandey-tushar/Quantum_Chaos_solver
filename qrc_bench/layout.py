"""Qubit layout derived from the task: input qubits from the encoding, memory qubits from a rule."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from qrc_bench.encoders import ENCODINGS, n_encoded_qubits
from qrc_bench.readouts import readout_width

Q_MAX = None        # no default cap; pass q_max to guard against runs that are too large
MEM_RULES = ("fixed", "ratio", "match-features")


@dataclass(frozen=True)
class Layout:
    n_series: int
    encoding: str
    n_in: int
    n_mem: int
    q: int
    readout: str
    n_taus: int
    qrc_window: int
    qrc_features: int
    poly2_window: int
    poly2_features: int

    def as_dict(self):
        return asdict(self)


def poly2_width(n_series: int, window: int) -> int:
    n = n_series * window
    return n + n * (n + 1) // 2


def derive(n_series: int, encoding: str = "per_series", mem_rule: str = "fixed",
           n_mem: int | None = None, mem_ratio: float | None = None, readout: str = "ZZ",
           n_taus: int = 1, qrc_window: int = 1, poly2_window: int = 1, q_max: int | None = Q_MAX) -> Layout:
    """Examples: Case I = derive(5, n_mem=3, n_taus=2, poly2_window=3) -> q 8, 72 vs 135 features.
    Case II = derive(1, n_mem=4, qrc_window=5, poly2_window=5) -> q 5, 75 vs 20 features."""
    n_in = n_encoded_qubits(n_series, encoding)

    def width(m):
        return readout_width(readout, n_in + m) * n_taus * qrc_window

    p2 = poly2_width(n_series, poly2_window)
    if mem_rule == "fixed":
        if n_mem is None:
            raise ValueError("mem_rule 'fixed' needs n_mem")
    elif mem_rule == "ratio":
        if mem_ratio is None:
            raise ValueError("mem_rule 'ratio' needs mem_ratio")
        n_mem = math.ceil(mem_ratio * n_in)
    elif mem_rule == "match-features":
        limit = (q_max - n_in) if q_max is not None else max(p2, 1)   # width grows with n_mem, so this ends
        n_mem = next((m for m in range(0, limit + 1) if width(m) >= p2), None)
        if n_mem is None:
            raise ValueError(f"no n_mem with q <= {q_max} reaches the Poly2 width {p2}")
    else:
        raise ValueError(f"unknown mem_rule {mem_rule!r}; expected one of {MEM_RULES}")
    q = n_in + n_mem
    if q_max is not None and q > q_max:
        raise ValueError(f"layout needs q = {n_in} + {n_mem} = {q} > q_max = {q_max}")
    return Layout(n_series, encoding, n_in, n_mem, q, readout, n_taus, qrc_window,
                  width(n_mem), poly2_window, p2)
