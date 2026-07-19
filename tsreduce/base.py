"""Base contract for all tsreduce dimensionality-reduction methods."""
from abc import abstractmethod
from time import perf_counter

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class BaseReducer(BaseEstimator, TransformerMixin):
    """Abstract base for time-series dimensionality reducers.

    Exactly one of *target_length* or *retention_rate* must be supplied.
    Subclasses only need to implement :meth:`_fit` and :meth:`_transform`;
    shape normalisation, timing, and validation are handled here.
    """

    def __init__(self, *, target_length=None, retention_rate=None):
        if target_length is None and retention_rate is None:
            raise ValueError(
                "Provide exactly one of 'target_length' or 'retention_rate'."
            )
        if target_length is not None and retention_rate is not None:
            raise ValueError(
                "Provide exactly one of 'target_length' or 'retention_rate', not both."
            )
        if retention_rate is not None and not (0 < retention_rate <= 1):
            raise ValueError(
                f"'retention_rate' must be in (0, 1], got {retention_rate!r}."
            )
        if target_length is not None and (
            not isinstance(target_length, (int, np.integer)) or target_length < 1
        ):
            raise ValueError(
                f"'target_length' must be a positive integer, got {target_length!r}."
            )
        self.target_length = target_length
        self.retention_rate = retention_rate

    # ------------------------------------------------------------------
    # sklearn-compatible fit / transform
    # ------------------------------------------------------------------

    def fit(self, X, y=None):
        X_3d = self._validate_and_reshape(X)
        n_samples, n_channels, n_timepoints = X_3d.shape

        if self.target_length is not None:
            n_timepoints_out = int(self.target_length)
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

        t0 = perf_counter()
        self._fit(X_3d)
        self.fit_time_ = perf_counter() - t0

        return self

    def transform(self, X):
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
    def _fit(self, X: np.ndarray) -> None:
        """Fit on 3-D X (n_samples, n_channels, n_timepoints). No-op if stateless."""

    @abstractmethod
    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Transform 3-D X. Must return (n_samples, n_channels, n_timepoints_out_)."""
