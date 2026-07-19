"""Uniform Downsampling"""
import numpy as np

from ..base import BaseReducer


class UniformDownsampling(BaseReducer):
    """Uniform index sampling along the time axis.

    Selects ``n_timepoints_out_`` linearly-spaced indices from each series
    using :func:`numpy.linspace`. No smoothing or averaging is applied —
    this is pure index selection. Stateless — :meth:`fit` is a no-op.

    Parameters
    ----------
    target_len : int, optional
        Number of samples to keep. Mutually exclusive with ``retention_rate``.
    retention_rate : float, optional
        Fraction of samples to keep. Mutually exclusive with ``target_len``.

    Examples
    --------
    >>> import numpy as np
    >>> from tsreduce import UniformDownsampling
    >>> X = np.random.randn(10, 200)
    >>> UniformDownsampling(target_len=50).fit_transform(X).shape
    (10, 50)
    """

    def __init__(self, *, target_len=None, retention_rate=None):
        super().__init__(target_len=target_len, retention_rate=retention_rate)

    def _fit(self, X: np.ndarray, y=None) -> None:
        pass

    def _transform(self, X: np.ndarray) -> np.ndarray:
        indices = np.linspace(0, X.shape[2] - 1, self.n_timepoints_out_).astype(int)
        return X[:, :, indices]
