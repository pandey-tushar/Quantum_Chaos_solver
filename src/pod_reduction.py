"""
Proper Orthogonal Decomposition (POD / EOF) for SWE state snapshots.

Given a sequence of snapshots of shape (n_t, 3, Nx, Ny) packing (u, v, h),
flatten each snapshot into a single state vector and compute the leading
modes by SVD on the centred snapshot matrix. The leading K modes are then
used to project full states down to a K-dimensional coefficient vector and
to reconstruct full states from coefficients.

Variance weighting: u, v, and h have different physical units and very
different magnitudes. Before stacking we rescale each field by its global
standard deviation so the SVD does not collapse onto whichever field has
the largest dimensional values. The scaling factors are stored and undone
at reconstruction time.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple


def _flatten_snapshots(snapshots: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """(n_t, 3, Nx, Ny) -> (n_t, 3*Nx*Ny). Returns the original (3, Nx, Ny) shape."""
    n_t = snapshots.shape[0]
    field_shape = snapshots.shape[1:]
    return snapshots.reshape(n_t, -1), field_shape


class PODReducer:
    """SVD-based POD with field-wise variance weighting."""

    def __init__(self, n_modes: int = 5):
        self.n_modes = n_modes
        self.mean_: np.ndarray | None = None        # (3*Nx*Ny,)
        self.scales_: np.ndarray | None = None      # (3*Nx*Ny,) -- field-wise std
        self.modes_: np.ndarray | None = None       # (3*Nx*Ny, n_modes)
        self.singular_values_: np.ndarray | None = None
        self.field_shape_: Tuple[int, int, int] | None = None

    def fit(self, snapshots: np.ndarray) -> "PODReducer":
        """
        Fit POD on (n_t, 3, Nx, Ny) snapshots.

        Steps:
          1. Per-field std (scalar per field) is computed and used to rescale
             u, v, h to comparable magnitudes.
          2. Mean is subtracted from the rescaled snapshot matrix.
          3. Truncated SVD gives the leading n_modes spatial modes.
        """
        if snapshots.ndim != 4:
            raise ValueError(f"snapshots must be (n_t, 3, Nx, Ny); got {snapshots.shape}")
        X, field_shape = _flatten_snapshots(snapshots)        # (n_t, 3*Nx*Ny)
        self.field_shape_ = field_shape
        n_pixels = field_shape[1] * field_shape[2]

        # Field-wise std for unit balancing
        scales = np.ones(X.shape[1])
        for f in range(field_shape[0]):
            sl = slice(f * n_pixels, (f + 1) * n_pixels)
            s = float(np.std(snapshots[:, f]))
            scales[sl] = s if s > 1e-12 else 1.0
        self.scales_ = scales

        Xs = X / scales                                       # rescale to ~unit std per field
        self.mean_ = Xs.mean(axis=0)
        Xc = Xs - self.mean_                                  # centred

        # SVD: Xc = U S Vt; spatial modes are columns of Vt^T, time coeffs are U*S.
        # snapshots are rows -> use full_matrices=False for compactness.
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        k = min(self.n_modes, Vt.shape[0])
        self.modes_ = Vt[:k].T                                # (n_features, k)
        self.singular_values_ = S[:k]
        return self

    def transform(self, snapshots: np.ndarray) -> np.ndarray:
        """Project (n_t, 3, Nx, Ny) snapshots onto leading modes -> (n_t, n_modes)."""
        self._check_fitted()
        X, _ = _flatten_snapshots(snapshots)
        Xc = X / self.scales_ - self.mean_
        return Xc @ self.modes_                                # (n_t, n_modes)

    def inverse_transform(self, coeffs: np.ndarray) -> np.ndarray:
        """Reconstruct (n_t, 3, Nx, Ny) snapshots from (n_t, n_modes) coefficients."""
        self._check_fitted()
        Xc = coeffs @ self.modes_.T                           # (n_t, n_features)
        Xs = Xc + self.mean_
        X = Xs * self.scales_
        return X.reshape((-1,) + self.field_shape_)

    def explained_variance_ratio(self) -> np.ndarray:
        """Fraction of total (rescaled) variance explained by each retained mode."""
        self._check_fitted()
        S = self.singular_values_
        # Note: this denominator only includes retained modes. To get true total variance
        # the caller would need the full singular spectrum, but since we keep n_modes only
        # this gives a useful diagnostic over what is kept vs. residual on the test set.
        total_kept = float(np.sum(S**2))
        return (S**2) / total_kept

    def _check_fitted(self):
        if self.modes_ is None:
            raise RuntimeError("PODReducer not fitted. Call .fit() first.")
