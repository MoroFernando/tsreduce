"""Isomap"""
import numpy as np
from sklearn.manifold import Isomap as _SklearnIsomap

from ..base import BaseReducer
from ._utils import pad_or_trim


class Isomap(BaseReducer):
    """Per-channel geodesic manifold embedding (Isomap).

    Fits :class:`sklearn.manifold.Isomap` independently on each channel.
    ``n_neighbors`` defaults to ``sqrt(n_samples)`` when not supplied. The
    component count is backed off automatically if the geodesic distance
    matrix yields significant negative eigenvalues, ensuring a valid embedding.

    Parameters
    ----------
    target_len : int, optional
        Number of embedding dimensions per channel. Mutually exclusive with
        ``retention_rate``.
    retention_rate : float, optional
        Fraction of dimensions to retain. Mutually exclusive with
        ``target_len``.
    n_neighbors : int or None, default=None
        Number of neighbours for the geodesic graph. Defaults to
        ``max(2, int(sqrt(n_samples)))`` when ``None``.


    Examples
    --------
    >>> from tsreduce import Isomap
    >>> X = np.random.randn(50, 200)
    >>> Isomap(target_len=10).fit_transform(X).shape
    (50, 10)
    """

    def __init__(self, *, target_len=None, retention_rate=None, n_neighbors=None):
        super().__init__(target_len=target_len, retention_rate=retention_rate)
        self.n_neighbors = n_neighbors

    def _fit(self, X: np.ndarray, y=None) -> None:
        w = self.n_timepoints_out_
        self.estimators_ = [
            self._fit_channel(w, X[:, c, :]) for c in range(X.shape[1])
        ]

    def _transform(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_channels, _ = X.shape
        w = self.n_timepoints_out_
        reduced = np.empty((n_samples, n_channels, w), dtype=float)
        for c, estimator in enumerate(self.estimators_):
            Z = estimator.transform(np.asarray(X[:, c, :], dtype=float))
            reduced[:, c, :] = pad_or_trim(Z, w)
        return reduced

    def _fit_channel(self, w, X_channel):
        X_channel = np.asarray(X_channel, dtype=float)
        n_samples, n_features = X_channel.shape
        if n_samples < 3:
            raise ValueError("Isomap requires at least three training samples.")

        n_neighbors = self.n_neighbors
        if n_neighbors is None:
            n_neighbors = max(2, int(np.sqrt(n_samples)))
        n_neighbors = min(n_neighbors, n_samples - 1)

        # The embedding cannot have more dimensions than the fitted graph rank;
        # padding restores the requested output width afterwards.
        n_components = max(1, min(w, n_samples - 1, n_features))

        # Isomap's centered geodesic-distance matrix is often not PSD: asking for
        # many components reaches into significantly negative eigenvalues, which
        # sklearn's internal KernelPCA rejects. Back the component count off until
        # the kept spectrum is PSD-valid (the top eigenvalue is always positive,
        # so this terminates, at worst at 1). The output is padded back to w.
        while True:
            estimator = _SklearnIsomap(n_neighbors=n_neighbors, n_components=n_components)
            try:
                estimator.fit(X_channel)
                return estimator
            except ValueError as err:
                if n_components > 1 and "negative eigenvalues" in str(err):
                    n_components = max(1, n_components // 2)
                    continue
                raise
