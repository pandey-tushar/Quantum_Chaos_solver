"""Finite-shot measurement of a Z-basis outcome distribution.

Hardware never reports an expectation value: it reports counts. One step's readout is M projective
measurements of the same prepared state, so the exact distribution P over the 2^q computational
outcomes is replaced by the empirical P_hat = counts / M with counts ~ Multinomial(M, P). Z and ZZ
commute, so one set of shots gives every Z and ZZ feature at once, with the correlations the shot
noise really has; the feedback signal is read from the same P_hat.

The sampler is exact (no Gaussian approximation) and vectorised over the rows of a batch, on numpy
and on cupy alike: per row a cumulative distribution and M uniforms, turned into outcome indices by
one searchsorted over the whole batch. Shifting row r's cdf and uniforms by r keeps every search
inside its own row, so the batch costs one searchsorted and one bincount rather than one per row.
"""
from __future__ import annotations

import numpy as np

MAX_UNIFORMS = 1 << 24      # uniforms per launch; rows are sub-batched to stay under it


def sample_probs(P, shots, rng, xp, max_uniforms: int = MAX_UNIFORMS):
    """(B, 2^q) exact probabilities -> the empirical distribution of ``shots`` measurements per row.

    Same shape and float dtype as ``P``. ``rng`` is an ``xp.random.default_rng`` generator (numpy or
    cupy); one generator used across calls makes the whole run one noise realisation.
    """
    if isinstance(shots, bool) or not isinstance(shots, (int, np.integer)) or shots <= 0:
        raise ValueError(f"shots must be a positive int, got {shots!r}")
    shots = int(shots)
    rows, k = P.shape
    block = max(1, int(max_uniforms) // shots)
    out = []
    for lo in range(0, rows, block):
        hi = min(rows, lo + block)
        n = hi - lo
        # float64 cdf normalised by its own last column: a single-precision P does not sum to 1
        cdf = xp.cumsum(P[lo:hi], axis=1, dtype=np.float64)
        cdf = cdf / cdf[:, -1:]
        offset = xp.arange(n, dtype=np.float64)[:, None]
        u = rng.random((n, shots), dtype=np.float64)
        flat = xp.searchsorted((cdf + offset).ravel(), (u + offset).ravel(), side="right")
        # the offsets keep every search inside its own row; the clip guards the two boundaries,
        # where adding a large offset to a float64 cdf can round a hit into the neighbouring row
        base = (xp.arange(n) * k)[:, None]
        flat = xp.clip(flat.reshape(n, shots), base, base + (k - 1)).ravel()
        out.append(_counts(flat, n, k, xp).reshape(n, k).astype(P.dtype) / P.dtype.type(shots))
    return out[0] if len(out) == 1 else xp.concatenate(out, axis=0)


def _counts(flat, n, k, xp):
    """Occurrences of each of the n * k flat indices. cupy.bincount validates its input with two
    device-to-host synchronisations, which cost more than the count itself here, so on cupy the
    same count is scattered into a preallocated buffer instead."""
    if xp.__name__ != "cupy":
        return xp.bincount(flat, minlength=n * k)
    counts = xp.zeros(n * k, dtype=np.int32)
    xp.add.at(counts, flat, np.int32(1))
    return counts
