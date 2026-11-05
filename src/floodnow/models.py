from __future__ import annotations

import numpy as np
import pandas as pd


def make_xy(q: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Features known on day t; target = log1p(Q) on day t+1."""
    z = np.log1p(q)
    X = pd.DataFrame(index=q.index)
    for k in (0, 1, 2, 3, 6):
        X[f"lag{k}"] = z.shift(k)
    X["d1"] = z - z.shift(1)
    X["mean7"] = z.rolling(7, min_periods=5).mean()
    X["mean30"] = z.rolling(30, min_periods=20).mean()
    X["max7"] = z.rolling(7, min_periods=5).max()
    doy = q.index.dayofyear
    X["sin"] = np.sin(2 * np.pi * doy / 365.25)
    X["cos"] = np.cos(2 * np.pi * doy / 365.25)
    y = z.shift(-1)
    return X, y


def regimes(level: np.ndarray, edges: tuple[float, float]) -> np.ndarray:
    """0 = low, 1 = medium, 2 = high, from today's log-flow and train quantile edges."""
    return np.digitize(level, edges)


def conformal_q(res: np.ndarray, alpha: float) -> float:
    n = len(res)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    s = np.sort(res)
    return float(s[k - 1]) if k <= n else float("inf")


def regime_quantiles(res: np.ndarray, reg: np.ndarray, alpha: float, n_reg: int = 3) -> np.ndarray:
    return np.array(
        [
            conformal_q(res[reg == r], alpha) if (reg == r).sum() > 20 else np.inf
            for r in range(n_reg)
        ]
    )
