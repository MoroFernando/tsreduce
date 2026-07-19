"""Piecewise Aggregate Approximation (PAA)"""
import numpy as np
from pyts.approximation import PiecewiseAggregateApproximation

from ..base import BaseReducer


class PAA(BaseReducer):

    def __init__(self, *, target_length=None, retention_rate=None):
        super().__init__(target_length=target_length, retention_rate=retention_rate)

    def _fit(self, X: np.ndarray, y=None) -> None:
        pass  # stateless — no parameters learned

    def _transform(self, X: np.ndarray) -> np.ndarray:
        paa = PiecewiseAggregateApproximation(
            window_size=None,
            output_size=self.n_timepoints_out_,
            overlapping=False,
        )

        n_samples, n_channels, n_timepoints = X.shape
        w = self.n_timepoints_out_

        # pyts operates on 2-D (n_series, n_timepoints); collapse the sample and
        # channel axes into one, transform in a single batch, then restore them.
        flat = X.reshape(n_samples * n_channels, n_timepoints)
        reduced = np.asarray(paa.transform(flat))
        return reduced.reshape(n_samples, n_channels, w)
