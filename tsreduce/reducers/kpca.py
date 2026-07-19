"""Kernel PCA (KPCA)"""
import numpy as np
from sklearn.decomposition import KernelPCA

from ..base import BaseReducer
from ._utils import pad_or_trim, rank_capped_components


class KPCA(BaseReducer):
    """Per-channel Kernel PCA (KPCA).

    Applies a non-linear manifold embedding independently to each channel using
    :class:`sklearn.decomposition.KernelPCA`. Defaults to the RBF kernel;
    ``gamma`` is inferred by sklearn when not supplied. The component count is
    capped at the channel matrix rank.

    Parameters
    ----------
    target_len : int, optional
        Number of kernel components to retain per channel. Mutually exclusive
        with ``retention_rate``.
    retention_rate : float, optional
        Fraction of components to retain. Mutually exclusive with
        ``target_len``.
    kernel : str, default='rbf'
        Kernel function, as accepted by
        :class:`sklearn.decomposition.KernelPCA`.
    gamma : float or None, default=None
        Kernel coefficient for ``'rbf'``, ``'poly'``, and ``'sigmoid'``
        kernels. Inferred by sklearn when ``None``.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    estimators_ : list of sklearn.decomposition.KernelPCA
        Fitted KPCA estimator for each channel.

    Examples
    --------
    >>> import numpy as np
    >>> from tsreduce import KPCA
    >>> X = np.random.randn(50, 200)
    >>> KPCA(target_len=20).fit_transform(X).shape
    (50, 20)
    """

    def __init__(self, *, target_len=None, retention_rate=None,
                 kernel="rbf", gamma=None, random_state=None):
        super().__init__(target_len=target_len, retention_rate=retention_rate)
        self.kernel = kernel
        self.gamma = gamma
        self.random_state = random_state

    def _fit(self, X: np.ndarray, y=None) -> None:
        w = self.n_timepoints_out_
        self.estimators_ = []
        for c in range(X.shape[1]):
            X_channel = np.asarray(X[:, c, :], dtype=float)
            n_components = rank_capped_components(w, X_channel)
            self.estimators_.append(
                KernelPCA(
                    n_components=n_components,
                    kernel=self.kernel,
                    gamma=self.gamma,
                    eigen_solver="auto",
                    random_state=self.random_state,
                ).fit(X_channel)
            )

    def _transform(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_channels, _ = X.shape
        w = self.n_timepoints_out_
        reduced = np.empty((n_samples, n_channels, w), dtype=float)
        for c, estimator in enumerate(self.estimators_):
            Z = estimator.transform(np.asarray(X[:, c, :], dtype=float))
            reduced[:, c, :] = pad_or_trim(Z, w)
        return reduced
