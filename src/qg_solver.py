"""
Single-layer barotropic quasi-geostrophic (QG) model on a beta-plane,
pseudo-spectral, doubly periodic.

Governing equation (vorticity form):

    dq/dt + J(psi, q) + beta * d(psi)/dx + r * q + nu * (-Laplacian)^p q = F

with
    q   = Laplacian(psi)              (relative vorticity, single-layer)
    psi : streamfunction
    u   = -d(psi)/dy
    v   =  d(psi)/dx
    J   : Jacobian (flux form, dealiased by 2/3 rule)
    r   : linear bottom drag                       [s^-1]
    nu  : hyperviscosity coefficient               [m^(2p)/s]
    p   : hyperviscosity order (default p=4 -> -nu * Laplacian^4)
    beta: beta-plane parameter                     [m^-1 s^-1]
    F   : optional stochastic forcing in vorticity space

The simulation is in non-dimensional or dimensional form depending on the
caller's choice of length, time, and beta. Defaults below are dimensional
(SI), tuned to produce mid-latitude-like Rossby-wave dynamics on a
~1000 km square domain.

Numerical scheme:
    - Field representation: real-valued q on a (Nx, Ny) periodic grid.
    - Spatial: pseudo-spectral. Linear terms exact in spectral space.
      Nonlinear Jacobian computed in physical space, transformed back,
      then truncated by the 2/3 rule for dealiasing.
    - Time: classical RK4 with fixed step. Caller is responsible for CFL.

This is a deliberately minimal solver. It is correct enough for the
SWE+QRC -> QG+QRC handoff (the same residual-learning architecture
applies), and is easy to swap for `pyqg` later if conda/Linux is
available. The optional `linear=True` flag drops the Jacobian entirely
to give a "physics core" whose residual against the full nonlinear
truth is the target of QRC correction.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, Dict, Optional


# ---------------------------------------------------------------------------
# Grid + spectral helpers
# ---------------------------------------------------------------------------

def make_qg_grid(Nx: int = 128, Ny: int = 128,
                 Lx: float = 1.0e6, Ly: float = 1.0e6) -> Dict[str, object]:
    """Build a uniform doubly-periodic grid + the spectral wavenumbers."""
    dx = Lx / Nx
    dy = Ly / Ny
    # 2*pi periodic wavenumbers, scaled to the physical domain
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)        # shape (Nx,)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=dy)        # shape (Ny,)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")          # both (Nx, Ny)
    K2 = KX ** 2 + KY ** 2
    K2[0, 0] = 1.0  # avoid div-by-zero on the zero-mean mode

    # 2/3 dealias mask
    kx_max = (2.0 / 3.0) * np.max(np.abs(kx))
    ky_max = (2.0 / 3.0) * np.max(np.abs(ky))
    dealias = (np.abs(KX) <= kx_max) & (np.abs(KY) <= ky_max)

    return {
        "Nx": Nx, "Ny": Ny, "Lx": Lx, "Ly": Ly,
        "dx": dx, "dy": dy,
        "x": (np.arange(Nx) + 0.5) * dx,
        "y": (np.arange(Ny) + 0.5) * dy,
        "KX": KX, "KY": KY, "K2": K2,
        "dealias": dealias,
    }


def vorticity_to_streamfunction(q: np.ndarray, grid: Dict[str, object]) -> np.ndarray:
    """Solve psi from Laplacian(psi) = q, with zero domain mean."""
    K2 = grid["K2"]
    q_hat = np.fft.fft2(q)
    psi_hat = -q_hat / K2
    psi_hat[0, 0] = 0.0
    return np.real(np.fft.ifft2(psi_hat))


def velocity_from_psi(psi: np.ndarray, grid: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray]:
    """u = -d(psi)/dy, v = d(psi)/dx; spectral derivatives."""
    KX = grid["KX"]; KY = grid["KY"]
    psi_hat = np.fft.fft2(psi)
    u = np.real(np.fft.ifft2(-1j * KY * psi_hat))
    v = np.real(np.fft.ifft2( 1j * KX * psi_hat))
    return u, v


def _ddx(f: np.ndarray, grid: Dict[str, object]) -> np.ndarray:
    return np.real(np.fft.ifft2(1j * grid["KX"] * np.fft.fft2(f)))


def _ddy(f: np.ndarray, grid: Dict[str, object]) -> np.ndarray:
    return np.real(np.fft.ifft2(1j * grid["KY"] * np.fft.fft2(f)))


# ---------------------------------------------------------------------------
# Right-hand side
# ---------------------------------------------------------------------------

def qg_rhs(q: np.ndarray, grid: Dict[str, object],
           beta: float = 1.6e-11,         # m^-1 s^-1, mid-latitude
           r: float = 0.0,                # s^-1, bottom drag
           nu: float = 0.0,               # hyperviscosity coefficient
           p_hyper: int = 4,              # order of (-Laplacian)^p
           linear: bool = False,
           forcing: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute dq/dt for the barotropic QG equation.

    If `linear=True`, the Jacobian J(psi, q) is dropped entirely. Useful
    as a deliberately-crude "physics core" whose residual against the
    full nonlinear truth is a clean target for a learned correction.
    """
    psi = vorticity_to_streamfunction(q, grid)
    u, v = velocity_from_psi(psi, grid)

    # Linear terms (always present)
    rhs = -beta * _ddx(psi, grid)

    if r > 0.0:
        rhs -= r * q

    if nu > 0.0:
        # Spectral hyperviscosity: (-Laplacian)^p applied in spectral space
        # = K2^p multiplier. Use raw K2 here (no zero-mean fix needed; the
        # zero-mean mode just carries the constant which we want to leave alone).
        K2 = grid["KX"] ** 2 + grid["KY"] ** 2
        hyp = -nu * K2 ** p_hyper
        rhs += np.real(np.fft.ifft2(hyp * np.fft.fft2(q)))

    # Nonlinear Jacobian (flux form), with dealiasing
    if not linear:
        # J(psi, q) = (u * dq/dx + v * dq/dy) on a divergence-free flow
        # is equivalent to div(u q). We compute via flux form for accuracy.
        Uq = u * q
        Vq = v * q
        Uq_hat = np.fft.fft2(Uq) * grid["dealias"]
        Vq_hat = np.fft.fft2(Vq) * grid["dealias"]
        Jac = np.real(np.fft.ifft2(1j * grid["KX"] * Uq_hat
                                     + 1j * grid["KY"] * Vq_hat))
        rhs -= Jac

    if forcing is not None:
        rhs = rhs + forcing

    return rhs


# ---------------------------------------------------------------------------
# Time integration
# ---------------------------------------------------------------------------

def qg_rk4_step(q: np.ndarray, dt: float, grid: Dict[str, object],
                **rhs_kwargs) -> np.ndarray:
    k1 = qg_rhs(q, grid, **rhs_kwargs)
    k2 = qg_rhs(q + 0.5 * dt * k1, grid, **rhs_kwargs)
    k3 = qg_rhs(q + 0.5 * dt * k2, grid, **rhs_kwargs)
    k4 = qg_rhs(q + dt * k3, grid, **rhs_kwargs)
    return q + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def integrate_qg(q0: np.ndarray, grid: Dict[str, object],
                  dt: float, n_steps: int, save_every: int = 1,
                  **rhs_kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate forward and return (times, vorticity_snapshots)."""
    q = q0.copy()
    n_save = n_steps // save_every + 1
    times = np.zeros(n_save)
    snapshots = np.zeros((n_save,) + q.shape, dtype=q.dtype)
    snapshots[0] = q
    save_idx = 1
    for k in range(1, n_steps + 1):
        q = qg_rk4_step(q, dt, grid, **rhs_kwargs)
        if k % save_every == 0 and save_idx < n_save:
            times[save_idx] = k * dt
            snapshots[save_idx] = q
            save_idx += 1
    return times[:save_idx], snapshots[:save_idx]


# ---------------------------------------------------------------------------
# Diagnostics + IC factories
# ---------------------------------------------------------------------------

def kinetic_energy(q: np.ndarray, grid: Dict[str, object]) -> float:
    """Domain-integrated kinetic energy = 0.5 * integral( |grad psi|^2 ) dA."""
    psi = vorticity_to_streamfunction(q, grid)
    u, v = velocity_from_psi(psi, grid)
    return float(0.5 * np.sum(u ** 2 + v ** 2) * grid["dx"] * grid["dy"])


def enstrophy(q: np.ndarray, grid: Dict[str, object]) -> float:
    """Domain-integrated enstrophy = 0.5 * integral(q^2) dA."""
    return float(0.5 * np.sum(q ** 2) * grid["dx"] * grid["dy"])


def random_vortex_ic(grid: Dict[str, object], amp: float = 1.0e-5,
                      k_peak: int = 6, k_width: int = 2,
                      seed: int = 0) -> np.ndarray:
    """
    Random vorticity field with energy concentrated at wavenumber ~k_peak.
    `amp` sets the RMS vorticity. Used as a typical Rossby-wave / 2D-turbulence IC.
    """
    Nx, Ny = grid["Nx"], grid["Ny"]
    rng = np.random.default_rng(seed)
    KX, KY = grid["KX"], grid["KY"]
    K = np.sqrt(KX ** 2 + KY ** 2) * (grid["Lx"] / (2.0 * np.pi))   # in cycles per domain
    # Energy spectrum peaked at k_peak with width k_width
    Espec = np.exp(-((K - k_peak) ** 2) / (2.0 * k_width ** 2))
    # Random Fourier amplitudes consistent with Hermitian symmetry of real q
    real_part = rng.standard_normal((Nx, Ny))
    imag_part = rng.standard_normal((Nx, Ny))
    q_hat = (real_part + 1j * imag_part) * np.sqrt(Espec)
    # Make sure the inverse FFT gives a real field by symmetrising
    q_hat[0, 0] = 0.0
    q = np.real(np.fft.ifft2(q_hat))
    rms = np.sqrt(np.mean(q ** 2)) + 1e-30
    return amp * q / rms


def cfl_dt(grid: Dict[str, object], u_max: float = 1.0,
            safety: float = 0.4) -> float:
    """Suggested time step from advective CFL given a reference u_max."""
    return safety * min(grid["dx"], grid["dy"]) / max(u_max, 1.0e-6)
