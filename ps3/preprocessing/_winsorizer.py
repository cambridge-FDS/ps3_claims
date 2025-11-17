import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# TODO: Write a simple Winsorizer transformer which takes a lower and upper quantile and cuts the
# data accordingly
class Winsorizer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer to cap outliers using quantiles.

    This transformer learns the lower and upper quantile values from the
    training data (during 'fit') and then uses these values to 'clip'
    data (during 'transform').

    Parameters
    ----------
    lower_quantile : float
        The lower percentile to use as the "floor". 
        Must be between 0 and 1.
    
    upper_quantile : float
        The upper percentile to use as the "ceiling". 
        Must be between 0 and 1.
    """

    def __init__(self, lower_quantile: float, upper_quantile: float):
        self.lower_quantile = lower_quantile 
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):

        self.lower_quantile_ = np.quantile(X, self.lower_quantile, axis=0)
        self.upper_quantile_ = np.quantile(X, self.upper_quantile, axis=0)

        return self

    def transform(self, X):
        X_clipped = np.clip(X, self.lower_quantile_, self.upper_quantile_)
        
        return X_clipped
