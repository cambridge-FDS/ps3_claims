import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


# TODO: Write a simple Winsorizer transformer which takes a lower and upper quantile and cuts the
# data accordingly
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, lower_quantile, upper_quantile):
        self.columns = columns
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        cols = [c for c in self.columns if pd.api.types.is_numeric_dtype(X[c])]
        self.lower_quantile_ = X[cols].quantile(
            self.lower_quantile
        )  # Series keyed by column
        self.upper_quantile_ = X[cols].quantile(self.upper_quantile)
        return self

    def transform(self, X):
        check_is_fitted(self)
        X_transformed = X.copy()
        # Only clip the columns that were fitted (the numeric ones in `columns`)
        for col in self.lower_quantile_.index:
            X_transformed[col] = np.clip(
                X_transformed[col],
                self.lower_quantile_[col],
                self.upper_quantile_[col],
            )
        return X_transformed
