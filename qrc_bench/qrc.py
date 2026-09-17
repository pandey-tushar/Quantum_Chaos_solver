"""Quantum reservoir feature matrix: encode -> evolve (reset or recurrent memory, optional feedback) -> read out."""
from __future__ import annotations

import numpy as np

from qrc_bench.backend import get_xp, resolve_backend, to_numpy
from qrc_bench.encoders import shift_ry
from qrc_bench.readouts import readout_matrix
from qrc_bench.simulate import batched, branch, dense
from qrc_bench.simulate.ops import product_state, sign_tables

METHODS = ("batched", "branch", "dense")
PRECISIONS = {"double": np.complex128, "single": np.complex64}
LOOP_SIMULATORS = {"branch": branch.final_diag, "dense": dense.final_diag}


def resolve_precision(precision: str, backend: str) -> str:
    """'auto' = single precision on the GPU (much faster there, ~1e-6 feature error), double on the CPU."""
    if precision == "auto":
        return "single" if backend == "cupy" else "double"
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision {precision!r}; expected 'auto' or one of {tuple(PRECISIONS)}")
    return precision


def resolve_budget_mb(budget, backend: str) -> float:
    """'auto' = half of the GPU memory that is free (device + CuPy pool), 512 MB on the CPU."""
    if budget != "auto":
        return float(budget)
    if backend != "cupy":
        return 512
    import cupy
    free = cupy.cuda.Device().mem_info[0] + cupy.get_default_memory_pool().free_bytes()
    return max(32.0, 0.5 * free / 2 ** 20)


def auto_chunk(n_in: int, n_mem: int, precision: str, mem_budget_mb: float) -> int:
    """Batch elements (streams x steps) so the largest intermediates (~4 x (B, 2^q, 2^n_mem)) fit the budget."""
    itemsize = 8 if precision == "single" else 16
    per_element = 4 * 2 ** (n_in + n_mem) * 2 ** n_mem * itemsize
    return max(1, int(mem_budget_mb * 2 ** 20 // per_element))


def _check_angles(angles, n_in: int, streams: bool):
    base = angles.ndim - (1 if streams else 0)             # 2: (T, n_in); 3: (T, n_in, 2)
    ok = (base == 2 and angles.shape[-1] == n_in) or (base == 3 and angles.shape[-2:] == (n_in, 2))
    if not ok:
        lead = "(S, T" if streams else "(T"
        raise ValueError(f"angles {angles.shape} must be {lead}, {n_in}) or {lead}, {n_in}, 2)")


def qrc_features(angles: np.ndarray, n_in: int, n_mem: int, reservoir, taus=(1.0,), readout: str = "ZZ",
                 mem_depth: int = 3, k_fb: float = 0.0, memory: str = "reset", method: str = "batched",
                 backend: str = "numpy", precision: str = "auto", chunk_size: int | None = None,
                 mem_budget_mb="auto", streams: bool = False) -> np.ndarray:
    """Features (T, len(taus) * width) from angles (T, n_in) [R_Y] or (T, n_in, 2) [R_Y, R_Z].
    streams=True: angles carry a leading stream axis (S, T, ...); all streams share the reservoir
    and run as one batch, returning (S, T, F).

    memory='reset' (the cartography paper's reservoir): at step t the memory qubits start in
    |0...0> and the last ``mem_depth`` inputs are driven in order, tracing out the input qubits
    between drives. memory='recurrent': the memory state is carried forward, one drive per step.
    Feedback shifts the current drive by k_fb * tanh(mean <Z> of step t-1); k_fb = 0 is the open loop.
    One block per tau, concatenated (two taus = the paper's ZZ_QR2 readout).
    method='batched' is the fast path (numpy or cupy); 'branch' and 'dense' are per-step
    references for tests (reset memory, double precision, numpy, one stream).
    backend 'auto' picks the GPU from q >= 10 when one is present; precision / mem_budget_mb 'auto'
    follow the backend (see resolve_backend, resolve_precision, resolve_budget_mb).
    """
    q = n_in + n_mem
    if reservoir.q != q:
        raise ValueError(f"reservoir has q={reservoir.q}, layout needs {n_in} + {n_mem}")
    _check_angles(np.asarray(angles), n_in, streams)
    stacked = streams
    if memory not in batched.MEMORY_MODES:
        raise ValueError(f"unknown memory mode {memory!r}; expected one of {batched.MEMORY_MODES}")
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    backend = resolve_backend(backend, q)
    precision = resolve_precision(precision, backend)
    if method != "batched":
        if memory != "reset" or backend != "numpy" or precision != "double" or stacked:
            raise ValueError("reference methods support only reset memory, numpy, double precision, one stream")
        return _loop_features(angles, n_in, n_mem, reservoir, taus, readout, mem_depth, k_fb, method)

    xp = get_xp(backend)
    dtype = PRECISIONS[precision]
    real = np.float32 if precision == "single" else np.float64
    z_np, zz_np = sign_tables(q)
    z_signs, zz_signs = xp.asarray(z_np, dtype=real), xp.asarray(zz_np, dtype=real)
    zbar = z_signs.mean(axis=0)

    def readout_fn(P):
        return readout_matrix(P, readout, z_signs, zz_signs, xp)

    budget = resolve_budget_mb(mem_budget_mb, backend)
    chunk = chunk_size or auto_chunk(n_in, n_mem, precision, budget)
    ang = xp.asarray(angles, dtype=real)
    if not stacked:
        ang = ang[None]
    blocks = [batched.simulate(ang, reservoir.unitary(tau, xp=xp, dtype=dtype), n_in, n_mem, readout_fn, xp,
                               dtype=dtype, memory=memory, mem_depth=mem_depth, k_fb=k_fb, zbar=zbar,
                               chunk_size=chunk)
              for tau in taus]
    F = to_numpy(xp.concatenate(blocks, axis=2)).astype(np.float64)
    return F if stacked else F[0]


def _loop_features(angles, n_in, n_mem, reservoir, taus, readout, mem_depth, k_fb, method):
    simulate = LOOP_SIMULATORS[method]
    q = n_in + n_mem
    z_signs, zz_signs = sign_tables(q)
    T = len(angles)
    blocks = []
    for tau in taus:
        U = reservoir.unitary(tau)
        D = np.zeros((T, 2 ** q))
        prev_mbar = 0.0
        for t in range(T):
            fb = k_fb * np.tanh(prev_mbar)
            psis = [product_state(np.clip(shift_ry(angles[tt:tt + 1], fb if tt == t else 0.0)[0], -np.pi, np.pi))
                    for tt in range(max(0, t - mem_depth + 1), t + 1)]
            D[t] = simulate(psis, U, n_in, n_mem, np)
            prev_mbar = float(np.mean(z_signs @ D[t]))
        blocks.append(readout_matrix(D, readout, z_signs, zz_signs))
    return np.concatenate(blocks, axis=1)
