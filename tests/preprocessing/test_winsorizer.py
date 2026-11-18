import numpy as np
import pandas as pd
import pytest

from ps3.preprocessing import Winsorizer


@pytest.mark.parametrize(
    "lower_quantile, upper_quantile", [(0, 1), (0.05, 0.95), (0.5, 0.5)]
)
def test_winsorizer(lower_quantile, upper_quantile):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(0, 1, 1000)})

    result = Winsorizer(["x"], lower_quantile, upper_quantile).fit_transform(df)

    lower = np.quantile(df["x"], lower_quantile)
    upper = np.quantile(df["x"], upper_quantile)

    # No value lies outside the cutoffs, and the extremes sit exactly on them
    assert result["x"].min() == pytest.approx(lower)
    assert result["x"].max() == pytest.approx(upper)

    # Values already inside the cutoffs are left unchanged
    inside = (df["x"] >= lower) & (df["x"] <= upper)
    pd.testing.assert_series_equal(result.loc[inside, "x"], df.loc[inside, "x"])
