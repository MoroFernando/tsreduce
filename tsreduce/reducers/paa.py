"""Piecewise Aggregate Approximation (PAA)"""
import numpy as np
from pyts.approximation import PiecewiseAggregateApproximation

from ..base import BaseReducer


class PAA(BaseReducer):
    """Piecewise Aggregate Approximation (PAA).

    Divides each time series into ``n_timepoints_out_`` equal-width,
    non-overlapping windows and replaces each window with its mean value.
    This is one of the most widely used time-series dimensionality reduction
    methods due to its simplicity and effectiveness as a preprocessing step.

    Stateless — :meth:`fit` is a no-op; the transformation depends only on
    the target output length.

    Uses :class:`pyts.approximation.PiecewiseAggregateApproximation` internally.

    Parameters
    ----------
    target_len : int, optional
        Number of segments (output timepoints). Mutually exclusive with
        ``retention_rate``.
    retention_rate : float, optional
        Fraction of timepoints to retain. Mutually exclusive with
        ``target_len``.

    Examples
    --------
    >>> import numpy as np
    >>> from tsreduce import PAA
    >>> X = np.random.randn(10, 200)
    >>> paa = PAA(target_len=50)
    >>> paa.fit_transform(X).shape
    (10, 50)
    """

    def __init__(self, *, target_len=None, retention_rate=None):
        super().__init__(target_len=target_len, retention_rate=retention_rate)

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
