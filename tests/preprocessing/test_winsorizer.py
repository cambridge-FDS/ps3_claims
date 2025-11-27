import numpy as np
import pytest

from ps3.preprocessing import Winsorizer

# TODO: Test your implementation of a simple Winsorizer
@pytest.mark.parametrize(
    "lower_quantile, upper_quantile", [(0, 1), (0.05, 0.95), (0.5, 0.5)]
)
def test_winsorizer(lower_quantile, upper_quantile):

    X = np.random.normal(0, 1, 1000)

    wz = Winsorizer(lower_quantile=lower_quantile, upper_quantile=upper_quantile)
    wz.fit(X)
    X_trans = wz.transform(X)

    # ① 形が同じであること
    assert X_trans.shape == X.shape

    # ② lower=0, upper=1 のときは、クリップしても何も変わらないはず
    if lower_quantile == 0 and upper_quantile == 1:
        assert np.allclose(X_trans, X)
    else:
        # ③ それ以外のときは、ちゃんと分位点以内に収まっていること
        q_low = np.quantile(X, lower_quantile)
        q_high = np.quantile(X, upper_quantile)

        assert np.all(X_trans >= q_low - 1e-8)
        assert np.all(X_trans <= q_high + 1e-8)
