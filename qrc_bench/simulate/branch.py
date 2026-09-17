"""Exact fast simulator: propagate the <= 2^n_mem pure branches of the memory ensemble.

The memory register is at most 2^n_mem dimensional, so the mixed state is an ensemble
of at most 2^n_mem pure states. Propagating those branches gives diag(rho) exactly
(no approximation beyond dropping eigenvalues below ``tol``), at a fraction of the
dense cost. Validated against the dense path and the paper's q = 8 JSON to ~1e-15.
"""
from __future__ import annotations


def final_diag(psi_ins, U, n_in: int, n_mem: int, xp, tol: float = 1e-13):
    d1, d2 = 2 ** n_in, 2 ** n_mem
    M = xp.zeros((d2, 1), dtype=xp.complex128)
    M[0, 0] = 1.0                                           # memory |0...0>
    p = xp.ones(1)
    for k, psi in enumerate(psi_ins):
        Phi = U @ xp.kron(psi[:, None], M)                  # (d1*d2, rank)
        if k == len(psi_ins) - 1:
            return xp.real((xp.abs(Phi) ** 2) @ p)
        A = Phi.reshape(d1, d2, -1) * xp.sqrt(p)[None, None, :]
        rho_mem = xp.einsum("iak,ibk->ab", A, A.conj())      # trace out the input qubits
        w, V = xp.linalg.eigh(rho_mem)
        keep = w > tol
        M, p = V[:, keep], w[keep]
