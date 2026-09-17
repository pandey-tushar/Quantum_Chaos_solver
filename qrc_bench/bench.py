"""Timing benchmark for the reservoir simulators (per-step wall time + agreement with a reference)."""
from __future__ import annotations

import time

import numpy as np

from qrc_bench.qrc import qrc_features
from qrc_bench.reservoirs.ising_xx import IsingXX


def _sync(backend):
    if backend == "cupy":
        import cupy
        cupy.cuda.Stream.null.synchronize()


def benchmark(layouts=((1, 4), (5, 3), (5, 6)), steps=200, methods=("branch", "batched"),
              backends=("numpy", "cupy"), precisions=("double", "single"), memories=("reset", "recurrent"),
              mem_depth=3, repeats=2, seed=0, log=lambda s: None) -> list[dict]:
    """One row per (layout, memory, method, backend, precision). The reference for agreement is the
    batched numpy double-precision result, itself checked against 'branch' where both apply."""
    rows = []
    for n_in, n_mem in layouts:
        res = IsingXX(n_in + n_mem, seed)
        ang = np.random.default_rng(seed).uniform(-np.pi / 2, np.pi / 2, (steps, n_in))
        for memory in memories:
            ref = None
            for method in methods:
                for backend in backends:
                    for precision in precisions:
                        if method != "batched" and (memory != "reset" or backend != "numpy" or precision != "double"):
                            continue
                        kw = dict(taus=(1.0,), mem_depth=mem_depth, memory=memory, method=method,
                                  backend=backend, precision=precision)
                        qrc_features(ang[:2], n_in, n_mem, res, **kw)          # warm-up (kernels, eigh cache)
                        best = np.inf
                        for _ in range(repeats):
                            _sync(backend)
                            t0 = time.perf_counter()
                            F = qrc_features(ang, n_in, n_mem, res, **kw)
                            _sync(backend)
                            best = min(best, time.perf_counter() - t0)
                        if ref is None:
                            ref = F
                        row = {"n_in": n_in, "n_mem": n_mem, "q": n_in + n_mem, "memory": memory, "method": method,
                               "backend": backend, "precision": precision, "steps": steps,
                               "ms_per_step": 1000 * best / steps,
                               "max_abs_diff_vs_reference": float(np.max(np.abs(F - ref)))}
                        rows.append(row)
                        log(f"q={row['q']:2d} ({n_in}+{n_mem}) {memory:9s} {method:7s} {backend:5s} {precision:6s} "
                            f"{row['ms_per_step']:9.3f} ms/step  |diff| {row['max_abs_diff_vs_reference']:.1e}")
    return rows
