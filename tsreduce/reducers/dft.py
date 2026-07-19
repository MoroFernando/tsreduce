"""Discrete Fourier Transform (DFT)"""
import numpy as np

from ..base import BaseReducer


class DFT(BaseReducer):
    """Discrete Fourier Transform (DFT) compression.

    Computes the real FFT of each series and retains the ``n_timepoints_out_``
    lowest-frequency coefficients as a feature vector. The DC term is kept as
    a single real value; remaining coefficients are stored as interleaved
    (real, imaginary) pairs. Stateless — :meth:`fit` is a no-op.

    Parameters
    ----------
    target_len : int, optional
        Number of output features. Mutually exclusive with ``retention_rate``.
    retention_rate : float, optional
        Fraction of timepoints to retain. Mutually exclusive with ``target_len``.

    Examples
    --------
    >>> from tsreduce import DFT
    >>> X = np.random.randn(10, 200)
    >>> DFT(target_len=50).fit_transform(X).shape
    (10, 50)
    """

    def __init__(self, *, target_len=None, retention_rate=None):
        super().__init__(target_len=target_len, retention_rate=retention_rate)

    def _fit(self, X: np.ndarray, y=None) -> None:
        pass

    def _transform(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_channels, n_timepoints = X.shape
        w = self.n_timepoints_out_

        flat = X.reshape(n_samples * n_channels, n_timepoints)
        coeffs = np.fft.rfft(flat, axis=1)

        # Real feature vector: DC term (real) then interleaved (real, imag) of the
        # remaining low-frequency coefficients.
        dc = coeffs[:, :1].real
        rest = coeffs[:, 1:]
        interleaved = np.empty((rest.shape[0], 2 * rest.shape[1]), dtype=float)
        interleaved[:, 0::2] = rest.real
        interleaved[:, 1::2] = rest.imag
        features = np.concatenate([dc, interleaved], axis=1)

        if features.shape[1] < w:
            pad = np.zeros((features.shape[0], w - features.shape[1]))
            features = np.concatenate([features, pad], axis=1)

        return features[:, :w].reshape(n_samples, n_channels, w)
