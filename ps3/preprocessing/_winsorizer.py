import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# TODO: Write a simple Winsorizer transformer which takes a lower and upper quantile and cuts the
# data accordingly
class Winsorizer(BaseEstimator, TransformerMixin):

    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):

        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):

        X = np.asarray(X)

        # 1次元が来たら (n_samples, 1) にして扱いやすくする
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # 列ごと (axis=0) に分位点を計算
        self.lower_quantile_ = np.quantile(X, self.lower_quantile, axis=0)
        self.upper_quantile_ = np.quantile(X, self.upper_quantile, axis=0)

        return self

    def transform(self, X):
        check_is_fitted(self, ["lower_quantile_", "upper_quantile_"])

        X = np.asarray(X)
        squeeze_back = False

        if X.ndim == 1:
            X = X.reshape(-1, 1)
            squeeze_back = True

        # 列ごとに下側・上側でクリップ
        X_clipped = np.clip(X, self.lower_quantile_, self.upper_quantile_)

        if squeeze_back:
            return X_clipped.ravel()
        return X_clipped
