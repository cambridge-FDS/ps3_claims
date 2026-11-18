
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip numerical values using fitted quantile boundaries."""

    def __init__(self, lower_quantile=0.05, upper_quantile=0.95):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        """Calculate clipping boundaries from the training data."""
        if not 0 <= self.lower_quantile <= self.upper_quantile <= 1:
            raise ValueError(
                "Quantiles must satisfy 0 <= lower <= upper <= 1."
            )

        X_array = np.asarray(X, dtype=float)

        if X_array.ndim not in (1, 2) or X_array.size == 0:
            raise ValueError("X must be a non-empty 1D or 2D array.")

        self.n_features_in_ = (
            1 if X_array.ndim == 1 else X_array.shape[1]
        )

        self.lower_quantile_ = np.nanquantile(
            X_array, self.lower_quantile, axis=0
        )
        self.upper_quantile_ = np.nanquantile(
            X_array, self.upper_quantile, axis=0
        )

        return self

    def transform(self, X):
        """Clip new data using the boundaries learned during fit."""
        check_is_fitted(
            self, ["lower_quantile_", "upper_quantile_"]
        )

        X_array = np.asarray(X, dtype=float)

        if X_array.ndim not in (1, 2):
            raise ValueError("X must be a 1D or 2D array.")

        n_features = (
            1 if X_array.ndim == 1 else X_array.shape[1]
        )

        if n_features != self.n_features_in_:
            raise ValueError("Input feature count does not match fit.")

        return np.clip(
            X_array, self.lower_quantile_, self.upper_quantile_
        )