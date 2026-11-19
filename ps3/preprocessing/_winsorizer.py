import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# TODO: Write a simple Winsorizer transformer which takes a lower and upper quantile and cuts the
# data accordingly
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile: float, upper_quantile: float) -> None:
        """Store the quantile levels used to calculate the clipping limits."""
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X: np.ndarray) -> "Winsorizer":
        """Learn the lower and upper cutoffs from the training data."""
        self.lower_quantile_ = np.quantile(X, self.lower_quantile)
        self.upper_quantile_ = np.quantile(X, self.upper_quantile)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Clip values using the cutoffs learned during fit."""
        return np.clip(X, self.lower_quantile_, self.upper_quantile_)
