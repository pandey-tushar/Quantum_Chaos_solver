"""Quantum reservoir feature matrix: encode -> evolve (reset or recurrent memory, optional feedback) -> read out."""
from __future__ import annotations

import numpy as np

from qrc_bench.backend import get_xp, resolve_backend, to_numpy
from qrc_bench.encoders import shift_ry
from qrc_bench.readouts import readout_bases, readout_matrix
from qrc_bench.shots import sample_probs
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
                 mem_budget_mb="auto", streams: bool = False, shots: int | None = None,
                 shot_seed: int = 0) -> np.ndarray:
    """Features (T, len(taus) * width) from angles (T, n_in) [R_Y] or (T, n_in, 2) [R_Y, R_Z].
    streams=True: angles carry a leading stream axis (S, T, ...); all streams share the reservoir
    and run as one batch, returning (S, T, F).

    ``reservoir`` may also be a sequence of R reservoirs of the same q (streams=True only). Then
    angles are (R, s, T, ...) and the result is (R, s, T, F): all R x s streams run as one batch,
    so a tuning trial's reservoir seeds cost one simulate call instead of R. That pays while the
    drive is launch-bound rather than bound by streaming the unitaries: measured on an RTX 2050,
    one recurrent step of 9 streams costs 1.9 ms as 3 reservoirs x 3 streams at q = 10 (3 separate
    calls: 5.2 ms) but 8.4 ms at q = 12 (3 separate calls: 9.0 ms), because each reservoir's U is
    re-read per drive and U is 134 MB at q = 12 in single precision.
    The R unitaries of a tau are stacked into one (R, D, D) device array, each reservoir's own
    device copy being released as it is stacked. Device memory for the unitaries then peaks at
    about 2 x R x (one unitary): the stacked array plus the Driver's U4 block.

    memory='reset' (the cartography paper's reservoir): at step t the memory qubits start in
    |0...0> and the last ``mem_depth`` inputs are driven in order, tracing out the input qubits
    between drives. memory='recurrent': the memory state is carried forward, one drive per step.
    Feedback shifts the current drive by k_fb * tanh(mean <Z> of step t-1); k_fb = 0 is the open loop.
    One block per tau, concatenated (two taus = the paper's ZZ_QR2 readout).
    readout='ZX' measures a second, non-commuting basis beside the Z one and appends
    [<X_i>, <X_i X_j>] to the ZZ block, doubling the width. It costs a second drive per step and a
    second unitary stack per reservoir on the device (134 MB at q = 12 in single precision);
    method='batched' only, as the reference simulators know only the Z diagonal.
    method='batched' is the fast path (numpy or cupy); 'branch' and 'dense' are per-step
    references for tests (reset memory, double precision, numpy, one stream).
    backend 'auto' picks the GPU from q >= 10 when one is present; precision / mem_budget_mb 'auto'
    follow the backend (see resolve_backend, resolve_precision, resolve_budget_mb).

    ``shots``: read every step from ``shots`` projective measurements instead of exact
    expectations (method='batched' only; the reference methods refuse it), split equally across
    the readout's measurement bases, so 'ZX' measures shots // 2 per basis and the total budget is
    the same as a single-basis readout's. This is the restarting
    protocol: each step's measurement is a fresh preparation of the same circuit, so simulating it
    is exactly "sample from the unmeasured state at every step" -- the memory state carried into the
    next step stays exact and is never collapsed. The Z and ZZ features and the feedback signal come
    from the same empirical distribution, so their shot noise is correlated as on hardware.
    One call is one noise realisation, seeded by ``shot_seed`` alone: a batched multi-reservoir call
    and the same reservoirs run in separate calls give different, equally valid realisations.
    ``shots=None`` (the default) keeps exact expectations, which no hardware can measure.
    """
    q = n_in + n_mem
    multi = isinstance(reservoir, (list, tuple))
    reservoirs = list(reservoir) if multi else [reservoir]
    for res in reservoirs:
        if res.q != q:
            raise ValueError(f"reservoir has q={res.q}, layout needs {n_in} + {n_mem}")
    ang_np = np.asarray(angles)
    if multi:
        if not streams:
            raise ValueError("a sequence of reservoirs needs streams=True and angles (R, s, T, ...)")
        if ang_np.ndim < 4 or ang_np.shape[0] != len(reservoirs):
            raise ValueError(f"angles {ang_np.shape} must start with the {len(reservoirs)} reservoirs")
    _check_angles(ang_np[0] if multi else ang_np, n_in, streams)
    stacked = streams
    if memory not in batched.MEMORY_MODES:
        raise ValueError(f"unknown memory mode {memory!r}; expected one of {batched.MEMORY_MODES}")
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if shots is not None and method != "batched":
        raise ValueError(f"shots require method='batched', got {method!r}")
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
    bases = readout_bases(readout)

    def readout_fn(P):
        return readout_matrix(P, readout, z_signs, zz_signs, xp)

    shot_rng = xp.random.default_rng(shot_seed) if shots is not None else None
    per_basis = None if shots is None else shots // len(bases)

    def sample(P):                                  # one generator = one realisation per call
        return sample_probs(P, per_basis, shot_rng, xp)

    budget = resolve_budget_mb(mem_budget_mb, backend)
    chunk = chunk_size or auto_chunk(n_in, n_mem, precision, budget)
    ang = xp.asarray(angles, dtype=real)
    if not stacked:
        ang = ang[None]
    lead = ang.shape[:2] if multi else ()
    if multi:
        ang = ang.reshape(-1, *ang.shape[2:])          # reservoir-major streams: r * s + i
    blocks = [batched.simulate(ang, _unitaries(reservoirs, tau, xp, dtype, multi), n_in, n_mem, readout_fn, xp,
                               dtype=dtype, memory=memory, mem_depth=mem_depth, k_fb=k_fb, zbar=zbar,
                               chunk_size=chunk, sample=None if shots is None else sample,
                               x_basis="X" in bases)
              for tau in taus]
    F = to_numpy(xp.concatenate(blocks, axis=2)).astype(np.float64)
    if multi:
        return F.reshape(*lead, *F.shape[1:])
    return F if stacked else F[0]


def _unitaries(reservoirs, tau, xp, dtype, multi):
    """One U (D, D), or the R reservoirs' unitaries stacked into one (R, D, D) device array.
    Each reservoir's own device copy is dropped once it has been written into the stack."""
    if not multi:
        return reservoirs[0].unitary(tau, xp=xp, dtype=dtype)
    D = 2 ** reservoirs[0].q
    U = xp.empty((len(reservoirs), D, D), dtype=dtype)
    for r, res in enumerate(reservoirs):
        U[r] = res.unitary(tau, xp=xp, dtype=dtype)
        getattr(res, "_u", {}).clear()
    return U


def _loop_features(angles, n_in, n_mem, reservoir, taus, readout, mem_depth, k_fb, method):
    if "X" in readout_bases(readout):
        raise ValueError(f"multi-basis readout requires method='batched', got {method!r}")
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
