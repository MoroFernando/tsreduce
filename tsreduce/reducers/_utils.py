"""Small, decision-free helpers shared by reducers.

These are pure, presentation/math utilities with no per-method behaviour baked
in, so sharing them creates no coupling: a reducer uses one only if it wants
exactly this behaviour, and a fix here is meant to apply to everyone. Anything
that encodes how a specific method behaves (its loss, optimiser, training loop,
projection, …) stays inside that method, not here.
"""
import numpy as np


def pad_or_trim(Z, w):
    """Return the 2-D matrix Z with exactly w columns (zero-pad or trim)."""
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    if Z.shape[1] == w:
        return Z
    if Z.shape[1] > w:
        return Z[:, :w]
    pad = np.zeros((Z.shape[0], w - Z.shape[1]))
    return np.hstack([Z, pad])


def rank_capped_components(w, X_channel):
    """Component count capped by the matrix rank: max(1, min(w, rows, cols))."""
    return max(1, min(w, X_channel.shape[0], X_channel.shape[1]))


