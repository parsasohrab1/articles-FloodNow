"""USGS daily discharge, 10 gauges (urban and mixed), 1995-2024: day-ahead forecast + conformal intervals.

usage: python research/run_study.py <dir with <site>.json USGS daily-value files>
train 1995-2009; calibration 2010-2016; test 2017-2024 (includes Hurricane Harvey, Aug 2017).
Flood day = next-day flow above the 99th percentile of the training years.
"""

import glob
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from floodnow.data import load_dv
from floodnow.models import conformal_q, make_xy, regime_quantiles, regimes

warnings.filterwarnings("ignore")
ALPHA = 0.1


def one_site(path: str):
    q = load_dv(path)
    X, y = make_xy(q)
    ok = X.notna().all(axis=1) & y.notna()
    X, y, q = X[ok], y[ok], q[ok]
    tr = X.index < "2010-01-01"
    ca = (X.index >= "2010-01-01") & (X.index < "2017-01-01")
    te = X.index >= "2017-01-01"
    if tr.sum() < 2000 or ca.sum() < 1000 or te.sum() < 1000:
        return None
    models = {
        "ridge": Ridge(1.0).fit(X[tr], y[tr]),
        "boosting": HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(X[tr], y[tr]),
    }
    lev = X["lag0"].to_numpy()
    edges = (np.quantile(lev[tr], 0.5), np.quantile(lev[tr], 0.9))
    reg = regimes(lev, edges)
    flood_thr = np.quantile(y[tr], 0.99)  # log1p units
    flood = (y[te] >= flood_thr).to_numpy()
    high = reg[te] == 2
    out = {
        "n_test": int(te.sum()),
        "n_flood_days_test": int(flood.sum()),
        "n_high_regime_test": int(high.sum()),
    }
    persist = X["lag0"].to_numpy()
    out["mae_log"] = {"persistence": float(np.abs(y[te] - persist[te]).mean())}
    for name, m in models.items():
        p = m.predict(X)
        res = np.abs(y.to_numpy() - p)
        out["mae_log"][name] = float(res[te].mean())
        qm = conformal_q(res[ca], ALPHA)
        qr = regime_quantiles(res[ca], reg[ca], ALPHA)
        up_m = p + qm
        up_r = p + qr[reg]
        cov_m = res[te] <= qm
        cov_r = res[te] <= qr[reg[te]]
        d = {
            "coverage_marginal": float(cov_m.mean()),
            "coverage_regime": float(cov_r.mean()),
            "coverage_high_regime_marginal": float(cov_m[high].mean()) if high.any() else None,
            "coverage_high_regime_regime": float(cov_r[high].mean()) if high.any() else None,
            "coverage_flood_days_marginal": float(cov_m[flood].mean()) if flood.any() else None,
            "coverage_flood_days_regime": float(cov_r[flood].mean()) if flood.any() else None,
            "width_log_marginal": float(2 * qm),
            "width_log_regime_high": float(2 * qr[2]),
        }
        for lab, score in (("point", p), ("upper_marginal", up_m), ("upper_regime", up_r)):
            alarm = score[te] >= flood_thr
            d[f"flood_hit_{lab}"] = float(alarm[flood].mean()) if flood.any() else None
            d[f"false_alarm_rate_{lab}"] = float(alarm[~flood].mean())
        out[name] = d
    return out


def main(folder: str):
    res = {
        "note": "real USGS daily discharge; chronological split; 10 gauges; alpha=0.1",
        "sites": {},
    }
    for f in sorted(glob.glob(f"{folder}/*.json")):
        site = os.path.basename(f)[:-5]
        r = one_site(f)
        if r:
            res["sites"][site] = r
    agg = {}
    for mname in ("ridge", "boosting"):
        keys = [k for k in next(iter(res["sites"].values()))[mname]]
        agg[mname] = {
            k: float(
                np.nanmean(
                    [
                        s[mname][k] if s[mname][k] is not None else np.nan
                        for s in res["sites"].values()
                    ]
                )
            )
            for k in keys
        }
    agg["mae_log"] = {
        k: float(np.mean([s["mae_log"][k] for s in res["sites"].values()]))
        for k in ("persistence", "ridge", "boosting")
    }
    res["mean_over_sites"] = agg
    res["n_sites"] = len(res["sites"])
    Path(__file__).with_name("results.json").write_text(json.dumps(res, indent=1))
    print(res["n_sites"], agg["mae_log"])
    for m in ("ridge", "boosting"):
        print(m, {k: round(v, 3) for k, v in agg[m].items()})


if __name__ == "__main__":
    main(sys.argv[1])
