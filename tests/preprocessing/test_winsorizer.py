
"""Tests for the Winsorizer transformer."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ps3.preprocessing import Winsorizer


@pytest.mark.parametrize(
    "lower_quantile, upper_quantile",
    [(0.0, 1.0), (0.05, 0.95), (0.5, 0.5)],
)
def test_winsorizer(lower_quantile, upper_quantile):
    """Check clipping for different quantile settings."""
    X = np.arange(101, dtype=float)

    transformer = Winsorizer(lower_quantile, upper_quantile)
    X_transformed = transformer.fit_transform(X)

    lower = np.quantile(X, lower_quantile)
    upper = np.quantile(X, upper_quantile)

    np.testing.assert_allclose(
        X_transformed, np.clip(X, lower, upper)
    )
    np.testing.assert_allclose(transformer.lower_quantile_, lower)
    np.testing.assert_allclose(transformer.upper_quantile_, upper)
    assert X_transformed.shape == X.shape


def test_transform_uses_training_quantiles():
    """Check that test data does not change fitted boundaries."""
    X_train = np.array([[0.0], [10.0], [20.0], [30.0], [100.0]])
    X_test = np.array([[-100.0], [50.0], [1000.0]])

    transformer = Winsorizer(0.25, 0.75).fit(X_train)
    result = transformer.transform(X_test)

    np.testing.assert_allclose(result, [[10.0], [30.0], [30.0]])
    np.testing.assert_allclose(
        X_test, [[-100.0], [50.0], [1000.0]]
    )


def test_two_features_and_pipeline():
    """Check multiple features and sklearn Pipeline compatibility."""
    X = np.array([
        [0.0, 100.0],
        [10.0, 200.0],
        [20.0, 300.0],
        [30.0, 400.0],
        [100.0, 1000.0],
    ])

    transformer = Winsorizer(0.25, 0.75).fit(X)

    expected = np.array([
        [10.0, 200.0],
        [10.0, 200.0],
        [20.0, 300.0],
        [30.0, 400.0],
        [30.0, 400.0],
    ])

    np.testing.assert_allclose(transformer.transform(X), expected)

    pipeline = Pipeline([
        ("winsorizer", Winsorizer(0.25, 0.75)),
        ("scaler", StandardScaler()),
    ])

    result = pipeline.fit_transform(X)

    assert result.shape == X.shape
    assert np.isfinite(result).all()


def test_transform_before_fit():
    """Check that transform requires a fitted transformer."""
    with pytest.raises(NotFittedError):
        Winsorizer().transform(np.array([0.0, 10.0, 20.0]))


@pytest.mark.parametrize(
    "lower_quantile, upper_quantile",
    [(-0.1, 0.9), (0.2, 1.1), (0.8, 0.2)],
)
def test_invalid_quantiles(lower_quantile, upper_quantile):
    """Check invalid quantile settings."""
    with pytest.raises(ValueError):
        Winsorizer(lower_quantile, upper_quantile).fit(
            np.arange(5, dtype=float)
        )


def test_wrong_feature_count():
    """Check that transform uses the fitted number of features."""
    transformer = Winsorizer().fit(
        np.array([[0.0, 10.0], [1.0, 11.0]])
    )

    with pytest.raises(ValueError):
        transformer.transform(np.array([[2.0]]))