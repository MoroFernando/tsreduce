"""Uniform Downsampling"""
import numpy as np

from ..base import BaseReducer


class UniformDownsampling(BaseReducer):
    """Uniform index sampling along the time axis.

    Picks *n_timepoints_out_* linearly-spaced indices from each series. No
    smoothing or averaging — just index selection. Stateless — fit is a no-op.
    """

    def __init__(self, *, target_len=None, retention_rate=None):
        super().__init__(target_len=target_len, retention_rate=retention_rate)

    def _fit(self, X: np.ndarray, y=None) -> None:
        pass

    def _transform(self, X: np.ndarray) -> np.ndarray:
        indices = np.linspace(0, X.shape[2] - 1, self.n_timepoints_out_).astype(int)
        return X[:, :, indices]
