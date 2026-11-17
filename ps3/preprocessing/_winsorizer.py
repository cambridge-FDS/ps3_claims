import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class Winsorizer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Winsorize numeric features at quantiles learned from the training data.

    Values below the lower quantile are set to the lower quantile, values above
    the upper quantile are set to the upper quantile. The cut-offs are computed
    per column in ``fit`` and reused in ``transform``, so test data is clipped
    with the cut-offs learned on the training data (no leakage).

    """

    def __init__(
        self, lower_quantile: float = 0.05, upper_quantile: float = 0.95
    ) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "Winsorizer":
        """Compute the lower and upper cut-offs on the (training) data.

        Parameters
        ----------
        X : np.ndarray
            Training data, shape (n_samples,) or (n_samples, n_features)
        y : np.ndarray or None, optional
            Ignored. Only here so the class works inside a scikit-learn Pipeline,
            by default None

        Returns
        -------
        Winsorizer
            The fitted transformer itself (``self``).
        """
        self.lower_quantile_ = np.nanquantile(X, self.lower_quantile, axis=0)
        self.upper_quantile_ = np.nanquantile(X, self.upper_quantile, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Clip the data at the cut-offs computed in ``fit``.

        Parameters
        ----------
        X : np.ndarray
            Data to winsorize (train or test), same number of columns as in ``fit``

        Returns
        -------
        np.ndarray
            Clipped copy of ``X``; the input itself is not modified.
        """
        check_is_fitted(self)
        X_clipped = np.clip(X, self.lower_quantile_, self.upper_quantile_)
        return X_clipped  # type: ignore[no-any-return]
