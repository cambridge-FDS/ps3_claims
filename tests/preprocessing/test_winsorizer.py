import numpy as np
import pytest

from ps3.preprocessing import Winsorizer

# TODO: Test your implementation of a simple Winsorizer
@pytest.mark.parametrize(
    "lower_quantile, upper_quantile", [(0, 1), (0.05, 0.95), (0.5, 0.5)]
)
def test_winsorizer(lower_quantile, upper_quantile):
    
    np.random.seed(42)  # For reproducibility
    X = np.random.normal(0, 1, 1000)
    
    # Create and fit the winsorizer
    winsorizer = Winsorizer(lower_quantile=lower_quantile, upper_quantile=upper_quantile)
    winsorizer.fit(X)
    
    # Transform the data
    X_transformed = winsorizer.transform(X)
    
    # Test 1: Check that the quantile values were computed correctly
    expected_lower = np.quantile(X, lower_quantile)
    expected_upper = np.quantile(X, upper_quantile)
    
    assert np.isclose(winsorizer.lower_quantile_, expected_lower), \
        f"Lower quantile mismatch: expected {expected_lower}, got {winsorizer.lower_quantile_}"
    assert np.isclose(winsorizer.upper_quantile_, expected_upper), \
        f"Upper quantile mismatch: expected {expected_upper}, got {winsorizer.upper_quantile_}"
    
    # Test 2: Check that all values are within the clipped range
    assert np.all(X_transformed >= winsorizer.lower_quantile_), \
        "Some values are below the lower quantile"
    assert np.all(X_transformed <= winsorizer.upper_quantile_), \
        "Some values are above the upper quantile"
    
    # Test 3: Check that the transformation matches np.clip behavior
    expected_transformed = np.clip(X, expected_lower, expected_upper)
    assert np.allclose(X_transformed, expected_transformed), \
        "Transformed data does not match expected clipped values"
    
    # Test 4: Check output shape
    assert X_transformed.shape == X.shape, \
        f"Shape mismatch: input shape {X.shape}, output shape {X_transformed.shape}"
    
    # Test 5: Special case - when lower_quantile == upper_quantile
    if lower_quantile == upper_quantile:
        # All values should be equal to the median
        assert np.allclose(X_transformed, winsorizer.lower_quantile_), \
            "When quantiles are equal, all values should be clipped to that quantile value"