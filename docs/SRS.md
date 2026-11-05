# Software Requirements Specification – FloodNow

- SR-1 `data.load_dv`: USGS daily-value JSON to a daily series (no-data values removed).
- SR-2 `models.make_xy`: features from day t only (log1p flow at lags 0,1,2,3,6; 1-day change; 7- and 30-day means; 7-day max; annual harmonics); target log1p flow on day t+1. A unit test checks that changing day t+1 does not alter the features of day t.
- SR-3 `models.regimes` (low <= train median, medium to the 90th percentile, high above) and `regime_quantiles`: split-conformal half-width per regime from calibration residuals.
- SR-4 `research/run_study.py`: ridge and gradient boosting; train 1995–2009, calibration 2010–2016, test 2017–2024 (includes Hurricane Harvey); flood day = next-day flow above the 99th percentile of the training years; alpha 0.1.
- SR-5 Two unit tests.
