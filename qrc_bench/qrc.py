"""Quantum reservoir feature matrix: encode -> evolve (sliding window, optional feedback) -> read out."""
from __future__ import annotations

import numpy as np

from qrc_bench.backend import get_xp, to_numpy
from qrc_bench.readouts import readout_matrix
from qrc_bench.simulate import branch, dense
from qrc_bench.simulate.ops import product_state, sign_tables

SIMULATORS = {"branch": branch.final_diag, "dense": dense.final_diag}


def qrc_features(angles: np.ndarray, n_in: int, n_mem: int, reservoir, taus=(1.0,),
                 readout: str = "ZZ", mem_depth: int = 3, k_fb: float = 0.0,
                 method: str = "branch", backend: str = "numpy") -> np.ndarray:
    """Features (T, len(taus) * width) from angles (T, n_in).

    Memory: at step t the memory qubits start in |0...0> and the last ``mem_depth``
    inputs are driven in order, tracing out the input qubits between drives.
    Feedback: the current drive is shifted by k_fb * tanh(mean <Z> of step t-1);
    k_fb = 0 is the open loop. One block per tau, concatenated (two taus = the
    paper's ZZ_QR2 readout).
    """
    q = n_in + n_mem
    if reservoir.q != q:
        raise ValueError(f"reservoir has q={reservoir.q}, layout needs {n_in} + {n_mem}")
    if angles.shape[1] != n_in:
        raise ValueError(f"angles have {angles.shape[1]} columns, layout has n_in={n_in}")
    xp = get_xp(backend)
    simulate = SIMULATORS[method]
    z_signs, zz_signs = sign_tables(q)
    T = len(angles)
    blocks = []
    for tau in taus:
        U = xp.asarray(reservoir.unitary(tau))
        D = np.zeros((T, 2 ** q))
        prev_mbar = 0.0
        for t in range(T):
            fb = k_fb * np.tanh(prev_mbar)
            psis = [product_state(np.clip(angles[tt] + (fb if tt == t else 0.0), -np.pi, np.pi), xp)
                    for tt in range(max(0, t - mem_depth + 1), t + 1)]
            D[t] = to_numpy(simulate(psis, U, n_in, n_mem, xp))
            prev_mbar = float(np.mean(z_signs @ D[t]))
        blocks.append(readout_matrix(D, readout, z_signs, zz_signs))
    return np.concatenate(blocks, axis=1)
