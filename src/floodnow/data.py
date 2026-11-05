from __future__ import annotations

import json

import pandas as pd


def load_dv(path: str) -> pd.Series:
    """USGS NWIS daily-value JSON (parameter 00060, mean discharge, cfs) -> daily Series."""
    d = json.load(open(path))  # noqa: SIM115
    ts = d["value"]["timeSeries"][0]
    nodata = float(ts["variable"]["noDataValue"])
    v = ts["values"][0]["value"]
    s = pd.Series([float(x["value"]) for x in v], index=pd.to_datetime([x["dateTime"] for x in v]))
    s = s[s != nodata]
    s = s[s >= 0]
    return s.asfreq("D")
