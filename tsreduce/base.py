"""Base contract for all tsreduce dimensionality-reduction methods."""
from abc import abstractmethod
from time import perf_counter

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class BaseReducer(BaseEstimator, TransformerMixin):
    """Abstract base class for time-series dimensionality reducers.

    Provides a unified sklearn-compatible interface for all reducers in the
    library. Subclasses only need to implement :meth:`_fit` and
    :meth:`_transform`; input validation, shape normalisation, and timing are
    handled here.

    Exactly one of ``target_len`` or ``retention_rate`` must be supplied at
    construction time.

    Parameters
    ----------
    target_len : int, optional
        Desired length of the output series (number of timepoints to keep).
        Must be a positive integer and cannot exceed the input series length.
        Mutually exclusive with ``retention_rate``.
    retention_rate : float, optional
        Fraction of timepoints to retain, in the range ``(0, 1]``.
        The effective output length is ``max(1, round(n_timepoints * retention_rate))``.
        Mutually exclusive with ``target_len``.
    verbose : bool, default=False
        Whether to display a progress bar during fitting. Only relevant for
        iterative reducers (e.g. deep learning models); stateless reducers
        ignore this parameter.

    Attributes
    ----------
    n_timepoints_in_ : int
        Number of timepoints in the series seen during :meth:`fit`.
    n_channels_in_ : int
        Number of channels in the data seen during :meth:`fit`.
    n_timepoints_out_ : int
        Number of timepoints in the reduced output series.
    retention_rate_ : float
        Fraction of timepoints actually retained after fitting, computed as
        ``n_timepoints_out_ / n_timepoints_in_``. Always available after fit,
        regardless of which constructor parameter was used.
    fit_time_ : float
        Wall-clock time (in seconds) spent in :meth:`_fit`.
    transform_time_ : float
        Wall-clock time (in seconds) spent in the last call to :meth:`_transform`.

    Examples
    --------
    Subclasses are instantiated like any sklearn transformer:

    >>> from tsreduce import PAA
    >>> X = np.random.randn(10, 200)  # 10 series of length 200
    >>> paa = PAA(target_len=50)
    >>> X_reduced = paa.fit_transform(X)
    >>> X_reduced.shape
    (10, 50)
    """

    def __init__(self, *, target_len=None, retention_rate=None, verbose: bool = False):
        if target_len is None and retention_rate is None:
            raise ValueError(
                "Provide exactly one of 'target_len' or 'retention_rate'."
            )
        if target_len is not None and retention_rate is not None:
            raise ValueError(
                "Provide exactly one of 'target_len' or 'retention_rate', not both."
            )
        if retention_rate is not None and not (0 < retention_rate <= 1):
            raise ValueError(
                f"'retention_rate' must be in (0, 1], got {retention_rate!r}."
            )
        if target_len is not None and (
            not isinstance(target_len, (int, np.integer)) or target_len < 1
        ):
            raise ValueError(
                f"'target_len' must be a positive integer, got {target_len!r}."
            )
        self.target_len = target_len
        self.retention_rate = retention_rate
        self.verbose = verbose

    # ------------------------------------------------------------------
    # sklearn-compatible fit / transform
    # ------------------------------------------------------------------

    def fit(self, X, y=None):
        """Fit the reducer to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timepoints) or \
(n_samples, n_channels, n_timepoints)
            Input time series. 2-D arrays are treated as single-channel.
        y : array-like of shape (n_samples,), optional
            Target labels. Ignored by unsupervised reducers; forwarded to
            supervised subclasses.

        Returns
        -------
        self : BaseReducer
            Fitted reducer.
        """
        X_3d = self._validate_and_reshape(X)
        n_samples, n_channels, n_timepoints = X_3d.shape

        if self.target_len is not None:
            n_timepoints_out = int(self.target_len)
        else:
            n_timepoints_out = max(1, round(n_timepoints * self.retention_rate))

        if n_timepoints_out > n_timepoints:
            raise ValueError(
                f"Computed n_timepoints_out ({n_timepoints_out}) exceeds the "
                f"series length ({n_timepoints})."
            )

        self.n_timepoints_in_ = n_timepoints
        self.n_channels_in_ = n_channels
        self.n_timepoints_out_ = n_timepoints_out
        self.retention_rate_ = n_timepoints_out / n_timepoints

        t0 = perf_counter()
        self._fit(X_3d, y)
        self.fit_time_ = perf_counter() - t0

        return self

    def transform(self, X):
        """Reduce the time series using the fitted reducer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timepoints) or \
(n_samples, n_channels, n_timepoints)
            Input time series. Must have the same number of channels and
            timepoints as the data passed to :meth:`fit`.

        Returns
        -------
        X_reduced : ndarray of shape (n_samples, ``n_timepoints_out_``) or \
(n_samples, n_channels, ``n_timepoints_out_``)
            Reduced time series. Returns 2-D if the input was 2-D, 3-D otherwise.
        """
        check_is_fitted(self, "n_timepoints_out_")

        was_2d = X.ndim == 2
        X_3d = self._validate_and_reshape(X)
        n_samples = X_3d.shape[0]

        t0 = perf_counter()
        result = self._transform(X_3d)
        self.transform_time_ = perf_counter() - t0

        expected = (n_samples, X_3d.shape[1], self.n_timepoints_out_)
        if result.shape != expected:
            raise ValueError(
                f"_transform must return shape {expected}, got {result.shape}."
            )

        if was_2d:
            return result[:, 0, :]  # squeeze channel axis
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_and_reshape(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 2:
            return X.reshape(X.shape[0], 1, X.shape[1])
        if X.ndim == 3:
            return X
        raise ValueError(
            f"X must be 2-D (n_samples, n_timepoints) or 3-D "
            f"(n_samples, n_channels, n_timepoints), got ndim={X.ndim}."
        )

    # ------------------------------------------------------------------
    # Abstract interface for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _fit(self, X: np.ndarray, y=None) -> None:
        """Fit on 3-D X (n_samples, n_channels, n_timepoints). No-op if stateless.

        y is forwarded from fit() and may be used by supervised subclasses.
        Unsupervised methods can ignore it.
        """

    @abstractmethod
    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Transform 3-D X. Must return (n_samples, n_channels, n_timepoints_out_)."""
