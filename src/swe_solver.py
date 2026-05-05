"""
Shallow Water Equations on a doubly-periodic Arakawa C-grid.

Vector-invariant nonlinear formulation (Sadourny enstrophy-conserving scheme):

    du/dt = +q * V_at_u - d(K + g h)/dx
    dv/dt = -q * U_at_v - d(K + g h)/dy
    dh/dt = -dU/dx - dV/dy

with mass fluxes U = h_u u, V = h_v v; kinetic energy K = (u^2 + v^2)/2 at
h-points; relative vorticity zeta = dv/dx - du/dy at corners; potential
vorticity q = (f + zeta)/h_corner at corners. A small biharmonic
hyperviscosity is added for numerical stability.

Grid layout (lower-left convention, periodic in both directions):
    h[i, j]   at (x_i,           y_j)
    u[i, j]   at (x_i + dx/2,    y_j)        -- i+1/2 face in x
    v[i, j]   at (x_i,           y_j + dy/2) -- j+1/2 face in y
    corner    at (x_i + dx/2,    y_j + dy/2)

All fields are stored as (Nx, Ny) arrays. Periodic neighbours are obtained
with np.roll: shift -1 means "to the right/up", +1 means "to the left/down".

Time integration: classical RK4 with a fixed step. The user is responsible
for picking dt that satisfies the gravity-wave CFL condition,
    dt < min(dx, dy) / sqrt(g * H_max).
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, Dict


# --- Grid ---------------------------------------------------------------------

def make_grid(Nx: int = 64, Ny: int = 64,
              Lx: float = 1.0e6, Ly: float = 1.0e6) -> Dict[str, float]:
    """Build a uniform doubly-periodic grid descriptor."""
    dx = Lx / Nx
    dy = Ly / Ny
    return {
        "Nx": Nx, "Ny": Ny, "Lx": Lx, "Ly": Ly, "dx": dx, "dy": dy,
        "x_h": (np.arange(Nx) + 0.5) * dx,
        "y_h": (np.arange(Ny) + 0.5) * dy,
    }


# --- Derivative / interpolation helpers (periodic) ----------------------------

def _ddx_centred(f: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(f, -1, axis=0) - np.roll(f, +1, axis=0)) / (2.0 * dx)


def _ddy_centred(f: np.ndarray, dy: float) -> np.ndarray:
    return (np.roll(f, -1, axis=1) - np.roll(f, +1, axis=1)) / (2.0 * dy)


def _laplacian(f: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return ((np.roll(f, -1, axis=0) - 2.0 * f + np.roll(f, +1, axis=0)) / dx**2
            + (np.roll(f, -1, axis=1) - 2.0 * f + np.roll(f, +1, axis=1)) / dy**2)


def _biharmonic(f: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return _laplacian(_laplacian(f, dx, dy), dx, dy)


# --- Right-hand side ---------------------------------------------------------

def swe_rhs(state: np.ndarray, grid: Dict[str, float],
            g: float = 9.81, f0: float = 1.0e-4, H: float = 100.0,
            nu: float = 0.0, linear: bool = False) -> np.ndarray:
    """
    Compute du/dt, dv/dt, dh/dt for the rotating shallow water equations.

    Parameters
    ----------
    state : ndarray, shape (3, Nx, Ny)
        Packed (u, v, h). h is total layer thickness (not perturbation).
    grid  : dict
        From `make_grid`.
    g, f0, H, nu : floats
        Gravity, Coriolis, mean depth, biharmonic hyperviscosity coefficient.
        H is also used as the linearisation reference height when linear=True.
    linear : bool
        If True, return the *linearised* rotating SWE about the rest state
        (h=H, u=v=0):
            du/dt = +f0 * v_at_u  - g * dh/dx
            dv/dt = -f0 * u_at_v  - g * dh/dy
            dh/dt = -H * (du/dx + dv/dy)
        This drops the vorticity-velocity coupling (q*V), the kinetic-energy
        term in the Bernoulli potential, and the layer-thickness advection in
        the mass flux. Useful as a deliberately-crude "physics core" whose
        residual against the full nonlinear truth is a clean target for a
        learned subgrid correction.

    Returns
    -------
    rhs : ndarray, shape (3, Nx, Ny)
    """
    dx, dy = grid["dx"], grid["dy"]
    u, v, h = state[0], state[1], state[2]

    if linear:
        # --- linearised rotating SWE ---
        v_at_u = 0.25 * (v
                          + np.roll(v, -1, axis=0)
                          + np.roll(v, +1, axis=1)
                          + np.roll(np.roll(v, +1, axis=1), -1, axis=0))
        u_at_v = 0.25 * (u
                          + np.roll(u, -1, axis=1)
                          + np.roll(u, +1, axis=0)
                          + np.roll(np.roll(u, +1, axis=0), -1, axis=1))
        dh_dx_u = (np.roll(h, -1, axis=0) - h) / dx
        dh_dy_v = (np.roll(h, -1, axis=1) - h) / dy

        du_dt = f0 * v_at_u - g * dh_dx_u
        dv_dt = -f0 * u_at_v - g * dh_dy_v

        # Continuity in linear form: dh/dt = -H * div(u)
        # divergence at h-point uses backward-staggered u, v
        div_u = (u - np.roll(u, +1, axis=0)) / dx + (v - np.roll(v, +1, axis=1)) / dy
        dh_dt = -H * div_u

        if nu > 0.0:
            du_dt -= nu * _biharmonic(u, dx, dy)
            dv_dt -= nu * _biharmonic(v, dx, dy)
            dh_dt -= nu * _biharmonic(h, dx, dy)
        return np.stack([du_dt, dv_dt, dh_dt], axis=0)

    # Interpolations onto u/v/corner points -----------------------------------
    h_u = 0.5 * (h + np.roll(h, -1, axis=0))                       # at (i+1/2, j)
    h_v = 0.5 * (h + np.roll(h, -1, axis=1))                       # at (i, j+1/2)
    h_corner = 0.25 * (h
                       + np.roll(h, -1, axis=0)
                       + np.roll(h, -1, axis=1)
                       + np.roll(np.roll(h, -1, axis=0), -1, axis=1))

    U = h_u * u                                                     # mass flux x
    V = h_v * v                                                     # mass flux y

    # Continuity: dh/dt = -dU/dx - dV/dy at h-points (i, j)
    # dU/dx at (i,j) uses U[i,j] - U[i-1,j].
    dh_dt = (-(U - np.roll(U, +1, axis=0)) / dx
             - (V - np.roll(V, +1, axis=1)) / dy)

    # Kinetic energy at h-points: average of squared u/v from neighbour faces
    K = 0.5 * (0.5 * (u**2 + np.roll(u, +1, axis=0)**2)
               + 0.5 * (v**2 + np.roll(v, +1, axis=1)**2))

    # Bernoulli potential at h-points
    Phi = g * h + K

    # Relative vorticity at corners (i+1/2, j+1/2)
    zeta = ((np.roll(v, -1, axis=0) - v) / dx
            - (np.roll(u, -1, axis=1) - u) / dy)

    # Potential vorticity at corners. Cap denominator to avoid blow-up when
    # h_corner approaches zero (should not happen in well-posed runs).
    q = (f0 + zeta) / np.maximum(h_corner, 1.0e-3 * H)

    # u-equation at u-point (i+1/2, j)
    # PV at u-point = average of two corners (j-1/2, j+1/2) below/above:
    q_at_u = 0.5 * (np.roll(q, +1, axis=1) + q)
    # V averaged onto u-point: 4 surrounding V-points
    V_at_u = 0.25 * (V
                     + np.roll(V, -1, axis=0)
                     + np.roll(V, +1, axis=1)
                     + np.roll(np.roll(V, +1, axis=1), -1, axis=0))
    dPhi_dx_u = (np.roll(Phi, -1, axis=0) - Phi) / dx
    du_dt = q_at_u * V_at_u - dPhi_dx_u

    # v-equation at v-point (i, j+1/2)
    q_at_v = 0.5 * (np.roll(q, +1, axis=0) + q)
    U_at_v = 0.25 * (U
                     + np.roll(U, -1, axis=1)
                     + np.roll(U, +1, axis=0)
                     + np.roll(np.roll(U, +1, axis=0), -1, axis=1))
    dPhi_dy_v = (np.roll(Phi, -1, axis=1) - Phi) / dy
    dv_dt = -q_at_v * U_at_v - dPhi_dy_v

    # Biharmonic hyperviscosity (small, just for grid-scale noise control)
    if nu > 0.0:
        du_dt -= nu * _biharmonic(u, dx, dy)
        dv_dt -= nu * _biharmonic(v, dx, dy)
        dh_dt -= nu * _biharmonic(h, dx, dy)

    return np.stack([du_dt, dv_dt, dh_dt], axis=0)


# --- Time integration --------------------------------------------------------

def rk4_step(state: np.ndarray, dt: float, grid, **rhs_kwargs) -> np.ndarray:
    k1 = swe_rhs(state, grid, **rhs_kwargs)
    k2 = swe_rhs(state + 0.5 * dt * k1, grid, **rhs_kwargs)
    k3 = swe_rhs(state + 0.5 * dt * k2, grid, **rhs_kwargs)
    k4 = swe_rhs(state + dt * k3, grid, **rhs_kwargs)
    return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def integrate_swe(state0: np.ndarray, grid, dt: float, n_steps: int,
                  save_every: int = 1, **rhs_kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrate the SWE forward and return (times, snapshots).

    Parameters
    ----------
    state0 : ndarray, shape (3, Nx, Ny)
    dt     : float
    n_steps : int
    save_every : int
        Save a snapshot every `save_every` steps (always includes t=0 and final).

    Returns
    -------
    times     : ndarray, shape (n_save,)
    snapshots : ndarray, shape (n_save, 3, Nx, Ny)
    """
    state = state0.copy()
    n_save = n_steps // save_every + 1
    times = np.zeros(n_save)
    snapshots = np.zeros((n_save,) + state0.shape, dtype=state0.dtype)
    snapshots[0] = state
    save_idx = 1
    for k in range(1, n_steps + 1):
        state = rk4_step(state, dt, grid, **rhs_kwargs)
        if k % save_every == 0 and save_idx < n_save:
            times[save_idx] = k * dt
            snapshots[save_idx] = state
            save_idx += 1
    return times[:save_idx], snapshots[:save_idx]


# --- Diagnostics -------------------------------------------------------------

def total_energy(state: np.ndarray, grid, g: float = 9.81, H_ref: float = 0.0) -> float:
    """Domain-integrated total energy (kinetic + available potential)."""
    dx, dy = grid["dx"], grid["dy"]
    u, v, h = state[0], state[1], state[2]
    KE = 0.5 * (h * (0.5 * (u**2 + np.roll(u, +1, axis=0)**2)
                     + 0.5 * (v**2 + np.roll(v, +1, axis=1)**2)))
    PE = 0.5 * g * (h - H_ref)**2
    return float(np.sum(KE + PE) * dx * dy)


def total_mass(state: np.ndarray, grid) -> float:
    return float(np.sum(state[2]) * grid["dx"] * grid["dy"])


# --- Initial condition factories --------------------------------------------

def gaussian_bump_ic(grid, H: float = 100.0, amp: float = 1.0,
                     x0_frac: float = 0.5, y0_frac: float = 0.5,
                     sigma_frac: float = 0.08) -> np.ndarray:
    """h = H + amp * Gaussian, u=v=0. Triggers radial gravity-wave propagation."""
    Nx, Ny, Lx, Ly = grid["Nx"], grid["Ny"], grid["Lx"], grid["Ly"]
    x = grid["x_h"][:, None]
    y = grid["y_h"][None, :]
    x0 = x0_frac * Lx
    y0 = y0_frac * Ly
    sigma = sigma_frac * min(Lx, Ly)
    bump = amp * np.exp(-((x - x0)**2 + (y - y0)**2) / (2.0 * sigma**2))
    h = H + bump
    u = np.zeros((Nx, Ny))
    v = np.zeros((Nx, Ny))
    return np.stack([u, v, h], axis=0)


def random_balanced_ic(grid, H: float = 100.0, amp: float = 1.0,
                       k_max: int = 4, f0: float = 1.0e-4, g: float = 9.81,
                       seed: int = 0) -> np.ndarray:
    """
    Geostrophically-balanced random initial condition with energy at low modes.
    Builds h as a sum of low-wavenumber sinusoids, then sets (u, v) from the
    geostrophic relation -f v = -g dh/dx, f u = -g dh/dy. (continuous form;
    interpolation onto the C-grid is approximate but adequate for an IC.)
    """
    Nx, Ny, Lx, Ly = grid["Nx"], grid["Ny"], grid["Lx"], grid["Ly"]
    rng = np.random.default_rng(seed)
    x = grid["x_h"][:, None]
    y = grid["y_h"][None, :]
    h_pert = np.zeros((Nx, Ny))
    for kx in range(1, k_max + 1):
        for ky in range(1, k_max + 1):
            phase_x = rng.uniform(0, 2 * np.pi)
            phase_y = rng.uniform(0, 2 * np.pi)
            weight = rng.standard_normal() / (kx**2 + ky**2)
            h_pert += weight * np.cos(2 * np.pi * kx * x / Lx + phase_x) \
                            * np.cos(2 * np.pi * ky * y / Ly + phase_y)
    h_pert *= amp / (np.std(h_pert) + 1e-12)
    h = H + h_pert

    # Geostrophic velocities from h gradient (centred differences at h-points,
    # then nearest-neighbour shift onto u/v faces; good enough as an IC).
    dh_dx = (np.roll(h, -1, axis=0) - np.roll(h, +1, axis=0)) / (2.0 * grid["dx"])
    dh_dy = (np.roll(h, -1, axis=1) - np.roll(h, +1, axis=1)) / (2.0 * grid["dy"])
    u = -g * dh_dy / f0
    v = +g * dh_dx / f0
    return np.stack([u, v, h], axis=0)


def cfl_dt(grid, g: float = 9.81, H: float = 100.0, safety: float = 0.5) -> float:
    """Suggested time step from gravity-wave CFL."""
    c = np.sqrt(g * H)
    return safety * min(grid["dx"], grid["dy"]) / c
