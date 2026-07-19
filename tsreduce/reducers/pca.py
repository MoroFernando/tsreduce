"""Principal Component Analysis (PCA)"""
import numpy as np
from sklearn.decomposition import PCA as _SklearnPCA

from ..base import BaseReducer
from ._utils import pad_or_trim, rank_capped_components


class PCA(BaseReducer):

    def __init__(self, *, target_len=None, retention_rate=None, random_state=None):
        super().__init__(target_len=target_len, retention_rate=retention_rate)
        self.random_state = random_state

    def _fit(self, X: np.ndarray, y=None) -> None:
        w = self.n_timepoints_out_
        self.estimators_ = []
        for c in range(X.shape[1]):
            X_channel = np.asarray(X[:, c, :], dtype=float)
            n_components = rank_capped_components(w, X_channel)
            estimator = _SklearnPCA(
                n_components=n_components, random_state=self.random_state
            ).fit(X_channel)
            self.estimators_.append(estimator)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_channels, _ = X.shape
        w = self.n_timepoints_out_
        reduced = np.empty((n_samples, n_channels, w), dtype=float)
        for c, estimator in enumerate(self.estimators_):
            Z = estimator.transform(np.asarray(X[:, c, :], dtype=float))
            reduced[:, c, :] = pad_or_trim(Z, w)
        return reduced
