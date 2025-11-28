import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# TODO: Write a simple Winsorizer transformer which takes a lower and upper quantile and cuts the
# data accordingly
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile=0.05, upper_quantile=0.95):
        """
        Initialize the Winsorizer transformer.
        
        Parameters
        ----------
        lower_quantile : float, default=0.05
            The lower quantile value (between 0 and 1) for clipping.
        upper_quantile : float, default=0.95
            The upper quantile value (between 0 and 1) for clipping.
        """
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        """
        Compute the quantile values from the training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, n_features)
            Training data.
        y : None
            Ignored. Present for API consistency by convention.
            
        Returns
        -------
        self : object
            Fitted transformer.
        """
        X = np.asarray(X)
        
        # Compute the actual quantile values from the data
        self.lower_quantile_ = np.quantile(X, self.lower_quantile)
        self.upper_quantile_ = np.quantile(X, self.upper_quantile)
        
        return self

    def transform(self, X):
        """
        Clip the data at the computed quantile values.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, n_features)
            Data to transform.
            
        Returns
        -------
        X_transformed : ndarray of shape (n_samples,) or (n_samples, n_features)
            Transformed data with values clipped at the quantiles.
        """
        # Check if fit has been called
        check_is_fitted(self, ['lower_quantile_', 'upper_quantile_'])
        
        X = np.asarray(X)
        
        # Clip the array at the computed quantile values
        X_transformed = np.clip(X, self.lower_quantile_, self.upper_quantile_)
        
        return X_transformed
