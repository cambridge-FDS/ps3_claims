import numpy as np
import pytest

from ps3.preprocessing import Winsorizer

# Test your implementation of a simple Winsorizer
@pytest.mark.parametrize(
    "lower_quantile, upper_quantile", [(0, 1), (0.05, 0.95), (0.5, 0.5), (0.01, 0.99)]
)
def test_winsorizer(lower_quantile, upper_quantile):

    X = np.random.normal(0, 1, 1000)

    # Initialize the Winsorizer
    winsorizer = Winsorizer(lower_quantile=lower_quantile, upper_quantile=upper_quantile)

    # Fit the Winsorizer
    winsorizer.fit(X)

    # Transform the data
    X_winsorized = winsorizer.transform(X)

    # Check that the data is winsorized correctly
    lower_bound = np.quantile(X, lower_quantile)
    upper_bound = np.quantile(X, upper_quantile)
    
    assert np.all(X_winsorized >= lower_bound)
    assert np.all(X_winsorized <= upper_bound)
