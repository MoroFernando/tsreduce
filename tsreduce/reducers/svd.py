"""Singular Value Decomposition (SVD)"""
import numpy as np

from ..base import BaseReducer
from ._utils import pad_or_trim, rank_capped_components


class _DatasetSVDEstimator:
    """Dataset-level SVD projection for one channel.

    The training set is treated as a matrix whose rows are time series and whose
    columns are time points. The right singular vectors define a common basis for
    all series; each series is represented by its projection coefficients on the
    first components. The same fitted basis is reused for the test data.
    """

    def __init__(self, n_components, center=False):
        self.n_components = n_components
        self.center = center

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0) if self.center else np.zeros(X.shape[1])
        _, _, vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        self.components_ = vt[: self.n_components]
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_.T


class SVD(BaseReducer):
    """Per-channel SVD projection.

    Learns a shared basis of right singular vectors from the training set and
    projects each series onto it. Equivalent to PCA without centering by default;
    set *center=True* to subtract the channel mean before decomposition.
    """

    def __init__(self, *, target_len=None, retention_rate=None, center=False):
        super().__init__(target_len=target_len, retention_rate=retention_rate)
        self.center = center

    def _fit(self, X: np.ndarray, y=None) -> None:
        w = self.n_timepoints_out_
        self.estimators_ = []
        for c in range(X.shape[1]):
            X_channel = np.asarray(X[:, c, :], dtype=float)
            n_components = rank_capped_components(w, X_channel)
            self.estimators_.append(
                _DatasetSVDEstimator(n_components, center=self.center).fit(X_channel)
            )

    def _transform(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_channels, _ = X.shape
        w = self.n_timepoints_out_
        reduced = np.empty((n_samples, n_channels, w), dtype=float)
        for c, estimator in enumerate(self.estimators_):
            Z = estimator.transform(np.asarray(X[:, c, :], dtype=float))
            reduced[:, c, :] = pad_or_trim(Z, w)
        return reduced
