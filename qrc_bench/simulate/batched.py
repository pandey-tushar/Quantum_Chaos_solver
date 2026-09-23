"""Batched exact simulator for numpy or cupy, over S parallel streams (data series) and time.

Memory state of the n_mem qubits, per batch element, in one of two exact forms:
  factor  M (B, d2, r), rho_mem = M M^dagger, used while r <= d2 (cheap early drives)
  rho     R (B, d2, d2), the density matrix itself, once the factor would exceed d2 columns
One drive of a product input state psi (B, d1), with W = sum_i psi_i U[:, i, :] of shape (B, D, d2):
  factor: Phi = W @ M;  diag(rho_full) = sum_k |Phi_jk|^2;  tracing the inputs gives the factor
          Phi reshaped (B, d2, d1 * r) (the input index becomes a column index)
  rho:    X = W @ R;    diag(rho_full) = Re sum_a X_ja conj(W_ja);  Tr_in = sum_i X_i @ W_i^dagger
An X-basis readout reuses both formulas with W_X = sum_i psi_i U_X[:, i, :], U_X = H^(x)q U, which
gives diag(H^(x)q rho_full H^(x)q); the memory state always comes from the Z-frame W (see Driver).
A drive of a memory still in |0...0> is the same formula with M = e_0, so only the d1 columns
U[:, i * d2] are touched: ``Driver.drive_fresh`` does that one in O(D d1) instead of O(D d1 d2).
No eigendecompositions are needed. Everything stays on the device; only the final features
are copied to the host.

Memory modes:
  reset      every step restarts memory in |0...0> and re-drives the last mem_depth inputs
             (the cartography paper's reservoir; a fixed-window feature map when k_fb = 0).
             All (stream, step) windows of a chunk run as one batch.
  recurrent  the memory state is carried forward, one drive per step (a true recurrent reservoir);
             all streams advance together; its trace is renormalised each step against round-off.
Feedback shifts each stream's R_Y drive by k_fb * tanh(mean <Z> of that stream's previous step).

On cupy three fused kernels carry the elementwise work: the product state (with the feedback shift
and the clip folded in) and the two diag(rho_full) reductions, which as cupy expressions would each
cost several passes over the (B, D, d2) intermediates. The numpy path is the plain formulation.
"""
from __future__ import annotations

import numpy as np

from qrc_bench.encoders import shift_ry

MEMORY_MODES = ("reset", "recurrent")
READOUT_BLOCK = 64          # recurrent: steps of probabilities buffered before one batched readout

# amplitude of basis state j = prod_a [bit_a(j) ? sin(ry_a / 2) : cos(ry_a / 2)] * exp(-+ i rz_a / 2),
# with the feedback shift k_fb * tanh(mbar_b) on the R_Y angles and every angle then clipped.
# ``ang`` is indexed by its logical C-order position, which cupy resolves through the array's own
# strides, so a strided view of the angles (one time step of many) needs no copy first.
_PRODUCT_CODE = """
const int dim = 1 << n;
const int b = i / dim;
const int j = i - b * dim;
const double PI = 3.14159265358979323846;
const double sh = shifted ? k_fb * tanh((double)mbar[b]) : 0.0;
C amp = C(1, 0);
for (int a = 0; a < n; ++a) {
    const int bit = (j >> (n - 1 - a)) & 1;
    double ry = (double)ang[(b * n + a) * w] + sh;
    ry = ry > PI ? PI : (ry < -PI ? -PI : ry);
    const double re = bit ? sin(0.5 * ry) : cos(0.5 * ry);
    C f = C(re, 0);
    if (w == 2) {
        double rz = (double)ang[(b * n + a) * w + 1];
        rz = rz > PI ? PI : (rz < -PI ? -PI : rz);
        const double ph = (bit ? 0.5 : -0.5) * rz;
        f = C(re * cos(ph), re * sin(ph));
    }
    amp = amp * f;
}
psi = amp;
"""
_PRODUCT_KERNEL = None      # one cupy ElementwiseKernel; it resolves T (real) and C (complex) per call
_REDUCERS = {}              # real dtype name -> (sum |x|^2, Re sum x conj(y)) cupy ReductionKernels


def _is_cupy(xp) -> bool:
    return xp.__name__ == "cupy"


def _reducers(xp, dtype):
    """The two last-axis reduction kernels for one complex dtype, compiled on first use."""
    name = np.empty(0, dtype=dtype).real.dtype.name
    if name not in _REDUCERS:
        _REDUCERS[name] = (
            xp.ReductionKernel('T x', f'{name} p', 'x.real() * x.real() + x.imag() * x.imag()',
                               'a + b', 'p = a', '0', f'qrc_sum_abs2_{name}'),
            xp.ReductionKernel('T x, T y', f'{name} p', 'x.real() * y.real() + x.imag() * y.imag()',
                               'a + b', 'p = a', '0', f'qrc_sum_redot_{name}'),
        )
    return _REDUCERS[name]


def sum_abs2(X, xp):
    """sum_k |X[..., k]|^2 over the last axis, real."""
    if _is_cupy(xp):
        return _reducers(xp, X.dtype)[0](X, axis=X.ndim - 1)
    return (X.real ** 2 + X.imag ** 2).sum(axis=X.ndim - 1)


def sum_redot(X, W, xp):
    """Re sum_k X[..., k] conj(W[..., k]) over the last axis, real."""
    if _is_cupy(xp):
        return _reducers(xp, X.dtype)[1](X, W, axis=X.ndim - 1)
    return (X * W.conj()).real.sum(axis=X.ndim - 1)


def hadamard_rows(A, axis: int, xp):
    """H^(x)q applied to ``axis`` of a C-contiguous ``A`` whose length there is 2^q, in place.

    A fast Walsh-Hadamard transform: q butterfly passes over A instead of a 2^q x 2^q matmul.
    Which qubit each pass handles does not matter, as H is applied to all of them. Returns A."""
    D = A.shape[axis]
    q = int(D).bit_length() - 1
    lead = int(np.prod(A.shape[:axis], dtype=np.int64))
    trail = int(np.prod(A.shape[axis + 1:], dtype=np.int64))
    V = A.reshape(lead, D, trail)
    h = 1
    while h < D:
        B = V.reshape(lead, D // (2 * h), 2, h, trail)
        x = B[:, :, 0].copy()
        B[:, :, 0] = x + B[:, :, 1]
        B[:, :, 1] = x - B[:, :, 1]
        h *= 2
    A *= A.dtype.type(2.0 ** (-0.5 * q))
    return A


def _each(fn, P):
    """Apply a per-distribution function to one probability batch, or to one per basis."""
    return tuple(fn(p) for p in P) if isinstance(P, tuple) else fn(P)


def _z_basis(P):
    """The Z-basis distribution of a probability batch: the feedback signal never leaves it."""
    return P[0] if isinstance(P, tuple) else P


def _stack_steps(buf, rows, xp):
    """Buffered per-step batches -> (rows, 2^q), per basis, with the step axis folded into rows."""
    if isinstance(buf[0], tuple):
        return tuple(xp.stack([b[k] for b in buf], axis=1).reshape(rows, -1)
                     for k in range(len(buf[0])))
    return xp.stack(buf, axis=1).reshape(rows, -1)


def product_states(angles, xp, dtype, k_fb=0.0, mbar=None):
    """(B, n) R_Y angles or (B, n, 2) (R_Y, R_Z) angles -> (B, 2^n) product states, qubit 0 most
    significant. With ``mbar`` (B,) the R_Y angles are first shifted by the feedback
    k_fb * tanh(mbar); the angles are then clipped to [-pi, pi]."""
    global _PRODUCT_KERNEL
    B, n = angles.shape[:2]
    if _is_cupy(xp):
        if _PRODUCT_KERNEL is None:
            _PRODUCT_KERNEL = xp.ElementwiseKernel(
                'raw T ang, raw T mbar, float64 k_fb, int32 n, int32 w, int32 shifted',
                'C psi', _PRODUCT_CODE, 'qrc_product_states')
        mb = angles if mbar is None else mbar.astype(angles.dtype, copy=False)
        psi = xp.empty((B, 2 ** n), dtype=dtype)
        _PRODUCT_KERNEL(angles, mb, float(k_fb), np.int32(n),
                        np.int32(1 if angles.ndim == 2 else 2), np.int32(mbar is not None), psi)
        return psi
    if mbar is not None:
        angles = shift_ry(angles, (k_fb * xp.tanh(mbar))[:, None])
    angles = xp.clip(angles, -np.pi, np.pi)
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
    """Precomputed pieces for one unitary U (D, D) on n_in input + n_mem memory qubits, or for
    R unitaries (R, D, D) driven side by side.

    With R > 1 every batch is reservoir-major: B is a multiple of R and its first B // R elements
    belong to reservoir 0, the next B // R to reservoir 1, and so on. ``simulate`` keeps that
    ordering in all three of its paths as long as the streams themselves are ordered
    reservoir-major (stream index = r * s + i).

    ``x_basis`` adds the X-frame unitary U_X = H^(x)q U, i.e. H on every qubit applied to U's row
    index, so that P_X = diag(H^(x)q rho_full H^(x)q) comes out of exactly the same two formulas
    with W_X in place of W. ``final`` and ``step`` then return (P_Z, P_X) instead of P. It costs
    one more unitary-sized device array per reservoir (134 MB at q = 12 in single precision) and
    a second drive per probability batch; the memory state always comes from the Z-frame W."""

    def __init__(self, U, n_in, n_mem, xp, dtype, x_basis: bool = False):
        self.xp, self.dtype = xp, dtype
        self.d1, self.d2 = 2 ** n_in, 2 ** n_mem
        self.D = self.d1 * self.d2
        self.R = 1 if U.ndim == 2 else U.shape[0]
        self.x_basis = x_basis
        self.U4 = xp.ascontiguousarray(                                    # (R, d1, D, d2)
            U.reshape(self.R, self.D, self.d1, self.d2).transpose(0, 2, 1, 3))
        self._Uw = self.U4.reshape(self.R, self.d1, self.D * self.d2)      # view, one gemm per reservoir
        self._U0 = xp.ascontiguousarray(self.U4[:, :, :, 0])               # (R, d1, D), memory index 0
        if x_basis:
            # axis 2 of U4 is U's row index, so the transform needs no second copy of U
            self.U4_X = hadamard_rows(self.U4.copy(), 2, xp)
            self._UwX = self.U4_X.reshape(self.R, self.d1, self.D * self.d2)

    def fresh_memory(self, B):
        M = self.xp.zeros((B, self.d2, 1), dtype=self.dtype)
        M[:, 0, 0] = 1.0
        return ("factor", M)

    def drive_fresh(self, psi):
        """State after driving psi into a memory that is still |0...0>. Same numbers as
        ``step(fresh_memory(B), psi)[0]``, but with M = e_0 only the columns U[:, i * d2] enter,
        so it reads d2 times less of U and never forms the (B, D, d2) drive."""
        B = psi.shape[0]
        assert B % self.R == 0, f"batch of {B} is not reservoir-major over {self.R} reservoirs"
        X = (psi.reshape(self.R, B // self.R, self.d1) @ self._U0).reshape(B, self.d1, self.d2)
        M = self.xp.ascontiguousarray(X.transpose(0, 2, 1))                # (B, d2, d1)
        if self.d1 <= self.d2:
            return ("factor", M)
        return ("rho", M @ M.conj().transpose(0, 2, 1))                    # as step does once r > d2

    def _W(self, psi, Uw=None):
        B = psi.shape[0]
        assert B % self.R == 0, f"batch of {B} is not reservoir-major over {self.R} reservoirs"
        Uw = self._Uw if Uw is None else Uw
        return (psi.reshape(self.R, B // self.R, self.d1) @ Uw).reshape(B, self.D, self.d2)

    def _diag_x(self, state, psi):
        """diag(H^(x)q rho_full H^(x)q) after driving psi: the Z formulas with W_X = H^(x)q W."""
        kind, S = state
        WX = self._W(psi, self._UwX)
        X = WX @ S
        if kind == "factor":
            return sum_abs2(X, self.xp)
        return sum_redot(X, WX, self.xp)

    def final(self, state, psi):
        """diag(rho_full) after driving psi, (B, D) real; with ``x_basis``, (P_Z, P_X)."""
        kind, S = state
        W = self._W(psi)
        X = W @ S
        P = sum_abs2(X, self.xp) if kind == "factor" else sum_redot(X, W, self.xp)
        return (P, self._diag_x(state, psi)) if self.x_basis else P

    def step(self, state, psi, renormalise=False, x_probs=True):
        """Drive psi and trace out the inputs: returns (next memory state, diag(rho_full)).
        ``x_probs=False`` skips the X-basis drive for a step whose probabilities are discarded."""
        xp = self.xp
        kind, S = state
        W = self._W(psi)
        X = W @ S
        B = X.shape[0]
        PX = self._diag_x(state, psi) if self.x_basis and x_probs else None
        if kind == "factor":
            P = sum_abs2(X, xp)
            r = X.shape[2]
            M = X.reshape(B, self.d1, self.d2, r).transpose(0, 2, 1, 3).reshape(B, self.d2, self.d1 * r)
            if M.shape[2] <= self.d2:
                nxt = ("factor", xp.ascontiguousarray(M))
            else:
                nxt = ("rho", M @ M.conj().transpose(0, 2, 1))
        else:
            P = sum_redot(X, W, xp)
            X4 = X.reshape(B, self.d1, self.d2, self.d2)
            W4 = W.reshape(B, self.d1, self.d2, self.d2)
            nxt = ("rho", (X4 @ W4.conj().transpose(0, 1, 3, 2)).sum(axis=1))
        if renormalise:
            kind2, S2 = nxt
            if kind2 == "rho":
                nxt = ("rho", S2 / xp.trace(S2, axis1=1, axis2=2).real[:, None, None])
            else:
                nxt = ("factor", S2 / xp.sqrt((S2.real ** 2 + S2.imag ** 2).sum(axis=(1, 2)))[:, None, None])
        return nxt, (P if PX is None else (P, PX))

    @staticmethod
    def take(state, idx):
        kind, S = state
        return (kind, S[idx])


def simulate(angles, U, n_in, n_mem, readout_fn, xp, dtype=np.complex128, memory="reset",
             mem_depth=3, k_fb=0.0, zbar=None, chunk_size=256, sample=None, x_basis=False):
    """Features for every stream and step. angles: (S, T, n_in[, 2]) device array;
    U: one unitary (D, D) or R of them (R, D, D), in which case the S streams must be ordered
    reservoir-major (stream = r * S/R + i) so every batch reaching the Driver is too;
    readout_fn: (B, 2^q) probabilities -> (B, F); zbar: (2^q,) mean-<Z> weights, needed for
    feedback. ``chunk_size`` bounds the batch (streams x steps). Returns a device array (S, T, F).

    ``sample``: (B, 2^q) probabilities -> the same shape, the finite-shot estimate of them. It is
    applied wherever a probability batch is produced, so the readout and the feedback signal read
    the same measured distribution; the memory state itself is never touched (see qrc_features).

    ``x_basis``: also measure in the X basis, so every probability batch is a (P_Z, P_X) pair that
    ``readout_fn`` and ``sample`` see one basis at a time. The feedback signal stays on the Z
    basis, and so does the memory state; the Driver then holds a second unitary stack."""
    if memory not in MEMORY_MODES:
        raise ValueError(f"unknown memory mode {memory!r}; expected one of {MEMORY_MODES}")
    drv = Driver(U, n_in, n_mem, xp, dtype, x_basis=x_basis)
    S, T = angles.shape[:2]
    tail = angles.shape[2:]
    feedback = k_fb != 0.0
    mbar = xp.zeros(S, dtype=xp.float32 if dtype == np.complex64 else xp.float64)

    def psi_of(ang, mb=None):                                  # mb: feed the previous step's mbar
        return product_states(ang.reshape(-1, *tail), xp, dtype, k_fb, mb)

    rows = []
    if memory == "recurrent":
        state = drv.fresh_memory(S)
        for b0 in range(0, T, READOUT_BLOCK):
            b1 = min(T, b0 + READOUT_BLOCK)
            psis = None if feedback else psi_of(angles[:, b0:b1].swapaxes(0, 1))
            buf = []
            for k, t in enumerate(range(b0, b1)):
                psi = psi_of(angles[:, t], mbar) if feedback else psis[k * S:(k + 1) * S]
                state, P = drv.step(state, psi, renormalise=True)
                if feedback:
                    if sample is not None:
                        P = _each(sample, P)
                    mbar = _z_basis(P) @ zbar
                buf.append(P)
            Pb = _stack_steps(buf, S * (b1 - b0), xp)              # (S, k, D) flattened
            if sample is not None and not feedback:
                Pb = _each(sample, Pb)                             # whole block at once: fewer launches
            rows.append(readout_fn(Pb).reshape(S, b1 - b0, -1))
        return xp.concatenate(rows, axis=1)

    steps = max(1, chunk_size // S)
    for c0 in range(0, T, steps):
        ts = np.arange(c0, min(T, c0 + steps))
        L = np.minimum(ts + 1, mem_depth)
        for length in np.unique(L):                                 # increasing length = time order
            lo, n = int(ts[L == length][0]), int((L == length).sum())   # L is sorted: sel is lo .. lo + n
            # the first of the ``length`` drives starts from |0...0>, which drive_fresh does cheaply
            state = (drv.fresh_memory(S * n) if length == 1 else
                     drv.drive_fresh(psi_of(angles[:, lo - length + 1:lo + n - length + 1])))
            for k in range(1, length - 1):
                state, _ = drv.step(state, psi_of(angles[:, lo - length + 1 + k:lo + n - length + 1 + k]),
                                    x_probs=False)
            if not feedback:
                P = drv.final(state, psi_of(angles[:, lo:lo + n]))
                if sample is not None:
                    P = _each(sample, P)
                rows.append(readout_fn(P).reshape(S, n, -1))
                continue
            base = xp.arange(S) * n                                 # one device index array per group
            buf = []
            for j, t in enumerate(range(lo, lo + n)):
                P = drv.final(drv.take(state, base + j), psi_of(angles[:, t], mbar))
                if sample is not None:
                    P = _each(sample, P)
                mbar = _z_basis(P) @ zbar
                buf.append(P)
            rows.append(readout_fn(_stack_steps(buf, S * n, xp)).reshape(S, n, -1))
    return xp.concatenate(rows, axis=1)
