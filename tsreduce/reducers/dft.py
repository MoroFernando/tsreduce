"""Discrete Fourier Transform (DFT)"""
import numpy as np

from ..base import BaseReducer


class DFT(BaseReducer):

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
