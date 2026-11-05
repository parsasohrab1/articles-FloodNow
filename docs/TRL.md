# TRL assessment – FloodNow

**Claimed level: TRL 4** (laboratory validation on public real data). Not TRL 5: daily means from 10 gauges, no rainfall input, no operational setting.

Data: USGS NWIS daily mean discharge for gauges 08075000 (Brays Bayou, Houston), 08074500 (Buffalo Bayou, Houston), 01589440 (Jones Falls, Baltimore), 01651000 (NE Branch Anacostia, DC), 02336300 (Peachtree Creek, Atlanta), 05531500 (Salt Creek, Illinois), 01646500 (Potomac, Little Falls), 01585200 (Herring Run, Baltimore), 08158000 (Colorado River, Austin), 03339000 (Vermilion River, Illinois). Test 2017–2024: 28790 gauge-days, of which 293 flood days (next-day flow above the training 99th percentile). Mean over 10 gauges, alpha = 0.1.

| Model | Test MAE (log flow) | Coverage marginal | Coverage regime-aware | High-flow regime: marginal | High-flow regime: regime-aware | Flood days: marginal | Flood days: regime-aware |
|---|---|---|---|---|---|---|---|
| persistence | 0.334 | | | | | | |
| ridge | 0.325 | 0.885 | 0.889 | 0.798 | 0.891 | 0.245 | 0.308 |
| boosting | 0.350 | 0.889 | 0.894 | 0.835 | 0.902 | 0.274 | 0.306 |

Flood-day alert (hit rate / false-alarm rate): alarm when the point forecast, or the upper bound of the interval, reaches the flood threshold.

| Model | Point forecast | Upper bound (marginal) | Upper bound (regime-aware) |
|---|---|---|---|
| ridge | 0.175 / 0.001 | 0.376 / 0.008 | 0.428 / 0.013 |
| boosting | 0.221 / 0.001 | 0.368 / 0.009 | 0.435 / 0.015 |

## Findings and caveats
- Day-ahead forecasts from daily flow alone barely beat persistence (ridge 0.325 vs 0.334 log MAE); gradient boosting is worse than persistence (0.350). Without rainfall input there is little to gain.
- Regime-aware calibration repairs coverage in the high-flow regime (about 0.80–0.84 to 0.89–0.90) at the cost of wider intervals there (log width 1.8–2.1 against 1.4–1.5).
- Coverage on true flood days stays low (0.25–0.31): flood days sit at the edge of the high regime and the intervals are not conditionally valid there. A guarantee averaged over a regime is not a guarantee on the rare extreme days.
- Using the upper bound as an alert roughly doubles the flood-day hit rate against the point forecast (0.38–0.44 against 0.18–0.22) with false-alarm rates around 1 %, but still misses more than half of the flood days.
- Daily means cannot resolve flash floods on small urban catchments; a 1-day-ahead daily forecast is not a warning system.
- Only 293 of the 28 790 test gauge-days (1.0 %) are flood days, so per-gauge flood statistics rest on a handful of events (including Hurricane Harvey, 2017); the averages above are over gauges and are noisy.
