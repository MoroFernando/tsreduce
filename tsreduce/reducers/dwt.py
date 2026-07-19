"""Discrete Wavelet Transform (DWT)"""
import numpy as np
import pywt

from ..base import BaseReducer


class DWT(BaseReducer):
    """Discrete Wavelet Transform with energy-ranked coefficient selection.

    Decomposes each channel of the training set and keeps the *n_timepoints_out_*
    coefficients with the highest average energy. The same indices are applied at
    transform time. Defaults to Daubechies-6 wavelets; *level* is inferred from
    the series length when not supplied.
    """

    def __init__(self, *, target_len=None, retention_rate=None, wavelet="db6", level=None):
        super().__init__(target_len=target_len, retention_rate=retention_rate)
        self.wavelet = wavelet
        self.level = level

    def _fit(self, X: np.ndarray, y=None) -> None:
        w = self.n_timepoints_out_
        self.channel_state_ = [
            self._fit_channel(X[:, c, :], w) for c in range(X.shape[1])
        ]

    def _transform(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_channels, _ = X.shape
        w = self.n_timepoints_out_
        reduced = np.empty((n_samples, n_channels, w), dtype=float)
        for c in range(n_channels):
            reduced[:, c, :] = self._transform_channel(X[:, c, :], self.channel_state_[c])
        return reduced

    def _decompose_matrix(self, X_channel, state):
        if state["degenerate"]:
            return np.asarray(X_channel, dtype=float)
        rows = [
            pywt.coeffs_to_array(
                pywt.wavedec(series, wavelet=self.wavelet, level=state["level"], mode="periodization")
            )[0]
            for series in X_channel
        ]
        return np.asarray(rows, dtype=float)

    def _fit_channel(self, X_channel, w):
        X_channel = np.asarray(X_channel, dtype=float)
        max_level = pywt.dwt_max_level(X_channel.shape[1], pywt.Wavelet(self.wavelet).dec_len)

        if max_level <= 0:
            state = {"degenerate": True, "level": None}
            coeff_matrix = X_channel
        else:
            use_level = max_level if self.level is None else min(self.level, max_level)
            state = {"degenerate": False, "level": use_level}
            coeff_matrix = self._decompose_matrix(X_channel, state)

        energy = np.mean(np.square(coeff_matrix), axis=0)
        n_components = max(1, min(w, coeff_matrix.shape[1]))
        # Stable sort keeps ties deterministic while ranking by decreasing energy.
        state["indices"] = np.argsort(-energy, kind="mergesort")[:n_components]
        return state

    def _transform_channel(self, X_channel, state):
        coeff_matrix = self._decompose_matrix(X_channel, state)
        selected = coeff_matrix[:, state["indices"]]

        w = self.n_timepoints_out_
        if selected.shape[1] < w:
            pad = np.zeros((selected.shape[0], w - selected.shape[1]))
            selected = np.concatenate([selected, pad], axis=1)
        return selected
