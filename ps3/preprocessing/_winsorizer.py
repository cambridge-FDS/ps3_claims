import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# TODO: Write a simple Winsorizer transformer which takes a lower and upper quantile and cuts the
# data accordingly
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile=0.05, upper_quantile=0.95):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_quantile_ = None
        self.upper_quantile_ = None
        pass

    def fit(self, X, y=None):
        """
        Computes the quantiles of the class given a dataset X and stores them in the attributes
        Args:
            X: np.array of shape (n_samples, n_features)
            y: np.array of shape (n_samples,)
        Returns:
            self
        """
        # Calulate the quantiles 
        self.lower_quantile_ = np.quantile(X, self.lower_quantile, axis=0)
        self.upper_quantile_ = np.quantile(X, self.upper_quantile, axis=0)
        return self

    def transform(self, X):
        """
        Cuts the given array at the quantiles (saved as an attribute of the class) and returns the modified array.
        Args:
            X: np.array of shape (n_samples, n_features)
        Returns:
            np.array of shape (n_samples, n_features)
        """
        # Check if the model has been fitted
        if not (check_is_fitted(self, 'lower_quantile_') and check_is_fitted(self, 'upper_quantile_')):
            # Fit the model
            self.fit(X)

        # Clip the data
        X = np.clip(X, self.lower_quantile_, self.upper_quantile_)
        return X
