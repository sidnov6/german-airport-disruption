# Daily airport dataset — de_airports_daily_v1

52,596 airport-days · 12 airports · 2014-01-01 to 2025-12-31 · 37 predictors · config `503e5a902822`

Target `disrupted_day`: share_arrivals_delayed_15 >= 0.02 (EUROCONTROL ATFM airport arrival delay).

## Checks

| check | result |
|---|---|
| row_count_airports_x_days | PASS |
| keys_unique | PASS |
| every_row_has_split | PASS |
| targets_consistent | PASS |
| no_same_day_outcome_in_predictors | PASS |
| lag1_uses_previous_day | PASS |
| each_split_has_both_classes | PASS |
| stations_at_airports | PASS |

## By split

| group | rows | labeled | disrupted | rate |
|---|---:|---:|---:|---:|
| calibrate | 4,380 | 4,380 | 364 | 0.083 |
| model_select | 4,380 | 4,371 | 302 | 0.069 |
| test | 8,772 | 8,771 | 533 | 0.061 |
| train | 35,064 | 35,027 | 2,051 | 0.059 |

## By airport

| group | rows | labeled | disrupted | rate |
|---|---:|---:|---:|---:|
| EDDC | 4,383 | 4,368 | 4 | 0.001 |
| EDDF | 4,383 | 4,383 | 850 | 0.194 |
| EDDG | 4,383 | 4,370 | 2 | 0.000 |
| EDDH | 4,383 | 4,383 | 286 | 0.065 |
| EDDK | 4,383 | 4,383 | 596 | 0.136 |
| EDDL | 4,383 | 4,383 | 523 | 0.119 |
| EDDM | 4,383 | 4,383 | 451 | 0.103 |
| EDDN | 4,383 | 4,383 | 7 | 0.002 |
| EDDP | 4,383 | 4,383 | 229 | 0.052 |
| EDDS | 4,383 | 4,367 | 172 | 0.039 |
| EDDV | 4,383 | 4,383 | 34 | 0.008 |
| EDDW | 4,383 | 4,380 | 96 | 0.022 |

## By year

| group | rows | labeled | disrupted | rate |
|---|---:|---:|---:|---:|
| 2014 | 4,380 | 4,380 | 251 | 0.057 |
| 2015 | 4,380 | 4,380 | 230 | 0.053 |
| 2016 | 4,392 | 4,392 | 271 | 0.062 |
| 2017 | 4,380 | 4,380 | 340 | 0.078 |
| 2018 | 4,380 | 4,380 | 408 | 0.093 |
| 2019 | 4,380 | 4,380 | 381 | 0.087 |
| 2020 | 4,392 | 4,358 | 37 | 0.008 |
| 2021 | 4,380 | 4,377 | 133 | 0.030 |
| 2022 | 4,380 | 4,371 | 302 | 0.069 |
| 2023 | 4,380 | 4,380 | 364 | 0.083 |
| 2024 | 4,392 | 4,391 | 278 | 0.063 |
| 2025 | 4,380 | 4,380 | 255 | 0.058 |

## Stations

| airport | station | name | distance km |
|---|---|---|---:|
| EDDF | 01420 | Frankfurt/Main | 3.62 |
| EDDM | 01262 | München-Flughafen | 2.13 |
| EDDL | 01078 | Düsseldorf | 0.73 |
| EDDH | 01975 | Hamburg-Fuhlsbüttel | 0.31 |
| EDDK | 02667 | Köln/Bonn | 1.05 |
| EDDS | 04931 | Stuttgart-Echterdingen | 0.21 |
| EDDP | 02932 | Leipzig/Halle | 0.29 |
| EDDV | 02014 | Hannover | 0.61 |
| EDDN | 03668 | Nürnberg | 1.74 |
| EDDW | 00691 | Bremen | 0.81 |
| EDDC | 01048 | Dresden-Klotzsche | 1.06 |
| EDDG | 01766 | Münster/Osnabrück | 0.83 |

## Missing predictor values

| predictor | nulls |
|---|---:|
| temp_mean_c | 185 |
| temp_min_c | 185 |
| temp_max_c | 185 |
| frost_day | 185 |
| rh_mean_pct | 197 |
| wind_mean_ms | 304 |
| wind_max_ms | 304 |
| gust_max_ms | 321 |
| gust_hours_ge15 | 321 |
| precip_sum_mm | 192 |
| precip_max_1h_mm | 192 |
| precip_hours | 192 |
| cold_precip_hours | 192 |
| vis_min_m | 754 |
| low_vis_hours_lt1000 | 754 |
| vis_hours_lt5000 | 754 |
| fog_hours | 826 |
| thunder_hours | 826 |
| snow_hours | 826 |
| freezing_hours | 826 |
| arrivals_lag1 | 50 |
| arrivals_lag7 | 122 |
| arrivals_mean_7d | 92 |
| delay_per_arrival_lag1 | 59 |
| share_delayed15_lag1 | 59 |
| disrupted_lag1 | 59 |
| delay_per_arrival_mean_7d | 99 |
| delay_per_arrival_mean_28d | 281 |
| disrupted_rate_28d | 281 |
| weather_delay_per_arrival_mean_28d | 281 |

## Assumptions

- Blank EUROCONTROL delay cells (before 2023) are treated as zero ATFM delay; from 2023 the source reports explicit zeros.
- Weather predictors are same-day observations used as a stand-in for a forecast; results are an upper bound.
- Weather-code features rely on reported present-weather codes; `-1` (not reported) counts as no event.
- 2020-2021 traffic collapse (COVID) is inside the training period; `covid_traffic_period` is kept as an audit column.
- ATFM delay is network-regulation delay, not airline schedule punctuality.
