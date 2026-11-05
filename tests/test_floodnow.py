import numpy as np
import pandas as pd

from floodnow.models import conformal_q, make_xy, regime_quantiles, regimes


def _q(n=400):
    rng = np.random.default_rng(0)
    return pd.Series(
        np.exp(rng.normal(3, 1, n)), index=pd.date_range("2000-01-01", periods=n, freq="D")
    )


def test_features_do_not_use_the_target_day():
    q = _q()
    X, y = make_xy(q)
    q2 = q.copy()
    q2.iloc[200] *= 100  # change the value on day 200
    X2, _ = make_xy(q2)
    # features of day 199 (which predict day 200) must be unchanged
    assert X.iloc[199].equals(X2.iloc[199])
    assert abs(y.iloc[199] - np.log1p(q.iloc[200])) < 1e-12


def test_regimes_and_quantiles():
    lev = np.array([1.0, 2.0, 3.0, 4.0])
    assert list(regimes(lev, (2.5, 3.5))) == [0, 0, 1, 2]
    assert conformal_q(np.arange(1.0, 100.0), 0.1) == 90.0
    res = np.concatenate([np.ones(100), 10 * np.ones(100)])
    reg = np.concatenate([np.zeros(100, int), np.ones(100, int)])
    qs = regime_quantiles(res, reg, 0.1, n_reg=3)
    assert qs[0] == 1.0 and qs[1] == 10.0 and np.isinf(qs[2])
