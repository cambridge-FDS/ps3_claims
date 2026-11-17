import numpy as np
import pytest

from ps3.preprocessing import Winsorizer


# Done: Test your implementation of a simple Winsorizer
@pytest.mark.parametrize(  # type: ignore[misc]
    "lower_quantile, upper_quantile", [(0, 1), (0.05, 0.95), (0.5, 0.5)]
)
def test_winsorizer(lower_quantile: float, upper_quantile: float) -> None:
    # Arrange
    X = np.random.normal(0, 1, 1000)

    # Act
    X_winsorized = Winsorizer(lower_quantile, upper_quantile).fit_transform(X)

    # Assert
    assert np.isclose(X_winsorized.min(), np.nanquantile(X, lower_quantile))
    assert np.isclose(X_winsorized.max(), np.nanquantile(X, upper_quantile))
