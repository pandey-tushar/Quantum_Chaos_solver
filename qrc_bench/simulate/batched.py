"""Batched exact simulator for numpy or cupy, over S parallel streams (data series) and time.

Memory state of the n_mem qubits, per batch element, in one of two exact forms:
  factor  M (B, d2, r), rho_mem = M M^dagger, used while r <= d2 (cheap early drives)
  rho     R (B, d2, d2), the density matrix itself, once the factor would exceed d2 columns
One drive of a product input state psi (B, d1), with W = sum_i psi_i U[:, i, :] of shape (B, D, d2):
  factor: Phi = W @ M;  diag(rho_full) = sum_k |Phi_jk|^2;  tracing the inputs gives the factor
          Phi reshaped (B, d2, d1 * r) (the input index becomes a column index)
  rho:    X = W @ R;    diag(rho_full) = Re sum_a X_ja conj(W_ja);  Tr_in = sum_i X_i @ W_i^dagger
No eigendecompositions are needed. Everything stays on the device; only the final features
are copied to the host.

Memory modes:
  reset      every step restarts memory in |0...0> and re-drives the last mem_depth inputs
             (the cartography paper's reservoir; a fixed-window feature map when k_fb = 0).
             All (stream, step) windows of a chunk run as one batch.
  recurrent  the memory state is carried forward, one drive per step (a true recurrent reservoir);
             all streams advance together; its trace is renormalised each step against round-off.
Feedback shifts each stream's R_Y drive by k_fb * tanh(mean <Z> of that stream's previous step).
"""
from __future__ import annotations

import numpy as np

from qrc_bench.encoders import shift_ry

MEMORY_MODES = ("reset", "recurrent")
READOUT_BLOCK = 64          # recurrent: steps of probabilities buffered before one batched readout


def product_states(angles, xp, dtype):
    """(B, n) R_Y angles or (B, n, 2) (R_Y, R_Z) angles -> (B, 2^n) product states, qubit 0 most significant."""
    B, n = angles.shape[:2]
    if angles.ndim == 2:
        c, s = xp.cos(angles / 2.0).astype(dtype), xp.sin(angles / 2.0).astype(dtype)
    else:
        ph = xp.exp(0.5j * angles[..., 1])
        c = (xp.cos(angles[..., 0] / 2.0) * ph.conj()).astype(dtype)
        s = (xp.sin(angles[..., 0] / 2.0) * ph).astype(dtype)
    psi = xp.ones((B, 1), dtype=dtype)
    for a in range(n):
        pair = xp.stack([c[:, a], s[:, a]], axis=1)
        psi = (psi[:, :, None] * pair[:, None, :]).reshape(B, -1)
    return psi


class Driver:
    """Precomputed pieces for one unitary U on n_in input + n_mem memory qubits."""

    def __init__(self, U, n_in, n_mem, xp, dtype):
        self.xp, self.dtype = xp, dtype
        self.d1, self.d2 = 2 ** n_in, 2 ** n_mem
        self.D = self.d1 * self.d2
        self.U4 = xp.ascontiguousarray(U.reshape(self.D, self.d1, self.d2).transpose(1, 0, 2))  # (d1, D, d2)

    def fresh_memory(self, B):
        M = self.xp.zeros((B, self.d2, 1), dtype=self.dtype)
        M[:, 0, 0] = 1.0
        return ("factor", M)

    def _W(self, psi):
        return self.xp.tensordot(psi, self.U4, axes=(1, 0))      # (B, D, d2)

    def final(self, state, psi):
        """diag(rho_full) after driving psi, (B, D) real."""
        kind, S = state
        W = self._W(psi)
        X = W @ S
        if kind == "factor":
            return (X.real ** 2 + X.imag ** 2).sum(axis=2)
        return (X * W.conj()).real.sum(axis=2)

    def step(self, state, psi, renormalise=False):
        """Drive psi and trace out the inputs: returns (next memory state, diag(rho_full))."""
        xp = self.xp
        kind, S = state
        W = self._W(psi)
        X = W @ S
        B = X.shape[0]
        if kind == "factor":
            P = (X.real ** 2 + X.imag ** 2).sum(axis=2)
            r = X.shape[2]
            M = X.reshape(B, self.d1, self.d2, r).transpose(0, 2, 1, 3).reshape(B, self.d2, self.d1 * r)
            if M.shape[2] <= self.d2:
                nxt = ("factor", xp.ascontiguousarray(M))
            else:
                nxt = ("rho", M @ M.conj().transpose(0, 2, 1))
        else:
            P = (X * W.conj()).real.sum(axis=2)
            X4 = X.reshape(B, self.d1, self.d2, self.d2)
            W4 = W.reshape(B, self.d1, self.d2, self.d2)
            nxt = ("rho", (X4 @ W4.conj().transpose(0, 1, 3, 2)).sum(axis=1))
        if renormalise:
            kind2, S2 = nxt
            if kind2 == "rho":
                nxt = ("rho", S2 / xp.trace(S2, axis1=1, axis2=2).real[:, None, None])
            else:
                nxt = ("factor", S2 / xp.sqrt((S2.real ** 2 + S2.imag ** 2).sum(axis=(1, 2)))[:, None, None])
        return nxt, P

    @staticmethod
    def take(state, idx):
        kind, S = state
        return (kind, S[idx])


def simulate(angles, U, n_in, n_mem, readout_fn, xp, dtype=np.complex128, memory="reset",
             mem_depth=3, k_fb=0.0, zbar=None, chunk_size=256):
    """Features for every stream and step. angles: (S, T, n_in[, 2]) device array;
    readout_fn: (B, 2^q) probabilities -> (B, F); zbar: (2^q,) mean-<Z> weights, needed for
    feedback. ``chunk_size`` bounds the batch (streams x steps). Returns a device array (S, T, F)."""
    if memory not in MEMORY_MODES:
        raise ValueError(f"unknown memory mode {memory!r}; expected one of {MEMORY_MODES}")
    drv = Driver(U, n_in, n_mem, xp, dtype)
    S, T = angles.shape[:2]
    tail = angles.shape[2:]
    feedback = k_fb != 0.0
    mbar = xp.zeros(S, dtype=xp.float32 if dtype == np.complex64 else xp.float64)

    def psi_of(ang):
        return product_states(xp.clip(ang.reshape(-1, *tail), -np.pi, np.pi), xp, dtype)

    def with_feedback(ang):                                        # ang (S, n_in[, 2])
        return shift_ry(ang, (k_fb * xp.tanh(mbar))[:, None])

    rows = []
    if memory == "recurrent":
        state = drv.fresh_memory(S)
        for b0 in range(0, T, READOUT_BLOCK):
            b1 = min(T, b0 + READOUT_BLOCK)
            psis = None if feedback else psi_of(angles[:, b0:b1].swapaxes(0, 1))
            buf = []
            for k, t in enumerate(range(b0, b1)):
                psi = psi_of(with_feedback(angles[:, t])) if feedback else psis[k * S:(k + 1) * S]
                state, P = drv.step(state, psi, renormalise=True)
                if feedback:
                    mbar = P @ zbar
                buf.append(P)
            Pb = xp.stack(buf, axis=1)                             # (S, k, D)
            rows.append(readout_fn(Pb.reshape(S * (b1 - b0), -1)).reshape(S, b1 - b0, -1))
        return xp.concatenate(rows, axis=1)

    steps = max(1, chunk_size // S)
    for c0 in range(0, T, steps):
        ts = np.arange(c0, min(T, c0 + steps))
        L = np.minimum(ts + 1, mem_depth)
        for length in np.unique(L):                                 # increasing length = time order
            sel = ts[L == length]
            n = len(sel)
            state = drv.fresh_memory(S * n)
            for k in range(length - 1):
                state, _ = drv.step(state, psi_of(angles[:, xp.asarray(sel - (length - 1) + k)]))
            if not feedback:
                P = drv.final(state, psi_of(angles[:, xp.asarray(sel)]))
                rows.append(readout_fn(P).reshape(S, n, -1))
                continue
            buf = []
            for j, t in enumerate(sel):
                P = drv.final(drv.take(state, xp.arange(S) * n + j), psi_of(with_feedback(angles[:, t])))
                mbar = P @ zbar
                buf.append(P)
            rows.append(readout_fn(xp.stack(buf, axis=1).reshape(S * n, -1)).reshape(S, n, -1))
    return xp.concatenate(rows, axis=1)
