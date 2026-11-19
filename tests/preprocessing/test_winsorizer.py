import numpy as np
import pytest

from ps3.preprocessing import Winsorizer

# TODO: Test your implementation of a simple Winsorizer
@pytest.mark.parametrize(
    "lower_quantile, upper_quantile", [(0, 1), (0.05, 0.95), (0.5, 0.5)]
)
def test_winsorizer(lower_quantile: float, upper_quantile: float) -> None:
    """Test that the winsorizer learns and applies both quantile cutoffs."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, 1000)

    winsorizer = Winsorizer(lower_quantile, upper_quantile)
    X_transformed = winsorizer.fit_transform(X)

    # Check the cutoff values learned during fit.
    lower = np.quantile(X, lower_quantile)
    upper = np.quantile(X, upper_quantile)
    assert np.isclose(winsorizer.lower_quantile_, lower)
    assert np.isclose(winsorizer.upper_quantile_, upper)

    # Every output must lie between the two cutoffs.
    assert np.all(X_transformed >= lower)
    assert np.all(X_transformed <= upper)

    # Check what happened to each group of input values.
    inside = (X >= lower) & (X <= upper)
    assert np.allclose(X_transformed[inside], X[inside])
    assert np.allclose(X_transformed[X < lower], lower)
    assert np.allclose(X_transformed[X > upper], upper)
