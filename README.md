# articles-FloodNow

Day-ahead streamflow forecasts for urban and mixed USGS gauges with regime-aware conformal intervals and a flood-day
alert rule, on real data (USGS NWIS daily discharge, 10 gauges, 1995–2024). TRL 4 (public real data, laboratory validation only).

```
pip install -e ".[dev]" && python -m pytest
# data: USGS NWIS daily values, parameter 00060 (see docs/TRL.md for the gauge list); one JSON per site
python research/run_study.py <dir with site JSON files>
```
