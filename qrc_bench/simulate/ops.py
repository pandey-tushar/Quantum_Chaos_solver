"""Qubit operators and diagonal-observable sign tables (qubit 0 = most significant bit)."""
from __future__ import annotations

import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def pauli_string(ops: dict[int, np.ndarray], q: int) -> np.ndarray:
    """Tensor product with ``ops[site]`` on the given sites and identity elsewhere."""
    out = np.array([[1.0]], dtype=complex)
    for k in range(q):
        out = np.kron(out, ops.get(k, I2))
    return out


def sign_tables(q: int):
    """(z_signs (q, 2^q), zz_signs (q(q-1)/2, 2^q)) so that <Z_i> = z_signs @ diag(rho)."""
    idx = np.arange(2 ** q)
    bits = (idx[:, None] >> np.arange(q - 1, -1, -1)) & 1
    z_signs = ((-1.0) ** bits).T
    zz = [(-1.0) ** (bits[:, i] + bits[:, k]) for i in range(q) for k in range(i + 1, q)]
    zz_signs = np.array(zz) if zz else np.zeros((0, 2 ** q))
    return z_signs, zz_signs


def product_state(angles, xp=np):
    """Per qubit R_Y(a)|0> = [cos(a/2), sin(a/2)] for angles (n,), or R_Z(b) R_Y(a)|0> =
    [e^{-ib/2} cos(a/2), e^{ib/2} sin(a/2)] for angles (n, 2); kron over qubits."""
    angles = np.asarray(angles, dtype=float)
    psi = xp.ones(1, dtype=xp.complex128)
    for a in angles:
        if np.ndim(a) == 0:
            pair = [np.cos(a / 2.0), np.sin(a / 2.0)]
        else:
            pair = [np.cos(a[0] / 2.0) * np.exp(-0.5j * a[1]), np.sin(a[0] / 2.0) * np.exp(0.5j * a[1])]
        psi = xp.kron(psi, xp.asarray(pair, dtype=xp.complex128))
    return psi


def partial_trace_first(rho, n_first: int, n_keep: int):
    """Trace out the first ``n_first`` qubits of a (n_first + n_keep)-qubit density matrix."""
    d1, d2 = 2 ** n_first, 2 ** n_keep
    return np.einsum("iaib->ab", rho.reshape(d1, d2, d1, d2))
