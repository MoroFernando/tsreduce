"""Uniform Downsampling"""
import numpy as np

from ..base import BaseReducer


class UniformDownsampling(BaseReducer):

    def __init__(self, *, target_length=None, retention_rate=None):
        super().__init__(target_length=target_length, retention_rate=retention_rate)

    def _fit(self, X: np.ndarray) -> None:
        pass

    def _transform(self, X: np.ndarray) -> np.ndarray:
        indices = np.linspace(0, X.shape[2] - 1, self.n_timepoints_out_).astype(int)
        return X[:, :, indices]
