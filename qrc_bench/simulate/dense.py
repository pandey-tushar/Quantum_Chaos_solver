"""Reference simulator: full 2^q x 2^q density matrices (slow, used to check the fast path)."""
from __future__ import annotations

from qrc_bench.simulate.ops import partial_trace_first


def final_diag(psi_ins, U, n_in: int, n_mem: int, xp):
    """Drive the window ``psi_ins`` (oldest first) through U, tracing out the input
    qubits between steps; memory starts in |0...0>. Returns diag(rho) after the last step."""
    rho_mem = xp.zeros((2 ** n_mem, 2 ** n_mem), dtype=xp.complex128)
    rho_mem[0, 0] = 1.0
    Ud = U.conj().T
    rho = None
    for k, psi in enumerate(psi_ins):
        rho = U @ xp.kron(xp.outer(psi, psi.conj()), rho_mem) @ Ud
        if k < len(psi_ins) - 1:
            rho_mem = partial_trace_first(rho, n_in, n_mem)
    return xp.real(xp.diag(rho))
