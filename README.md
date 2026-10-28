# Forecasting ATFM arrival disruption at German airports

Introduction to Data Analytics, project repository.

We combine public EUROCONTROL air traffic flow management (ATFM) delay data with hourly weather observations from the German Weather Service (DWD) to study disrupted operating days at 12 German airports from 2014 to 2025. The repository contains the data pipeline, the processed dataset, two descriptive (BI) notebooks, four model notebooks and the micropublication.

**Question:** how well can simple linear models and gradient-boosted trees estimate the likelihood and severity of daily ATFM arrival disruption at German airports?

A **disrupted day** is an airport-day on which at least 2% of arriving flights received more than 15 minutes of ATFM delay (our own threshold; 6.2% of all airport-days, 19.4% at Frankfurt).

## Main results (test years 2024 to 2025, 8,771 airport-days, evaluated once)

| Classification: disrupted day | Average precision | ROC-AUC | Precision / recall at 0.20 |
|---|---:|---:|---:|
| LightGBM classifier | 0.389 | 0.895 | 0.39 / 0.50 |
| Logistic regression | 0.376 | 0.889 | 0.34 / 0.56 |
| Previous 28-day disruption rate | 0.242 | 0.848 | 0.25 / 0.62 (threshold 0.15) |
| Base rate | 0.061 | 0.500 | |

| Regression: ATFM delay minutes per arrival | MAE | RMSE | R² |
|---|---:|---:|---:|
| LightGBM regressor (Tweedie) | 0.200 | 0.947 | 0.16 |
| Ridge regression | 0.290 | 0.957 | 0.14 |
| Always predict zero | 0.207 | 1.054 | |
| Previous 28-day average | 0.298 | 0.989 | 0.08 |

- Both classifiers clearly beat the history benchmark. Most of LightGBM's lead comes from Frankfurt (AP 0.63 vs 0.55).
- Regression adds little: 89% of airport-days have no ATFM delay, so predicting zero is hard to beat on MAE.
- Thunderstorms are the strongest weather signal: on thunderstorm days, Frankfurt, Munich and Stuttgart were 3.9 to 5.5 times as likely to be disrupted as on other days.
- Same-day observed weather is used as if it were a perfect forecast, so these scores are an upper bound for real day-ahead use.

## Data

| Source | Content | Link |
|---|---|---|
| EUROCONTROL, Aviation Intelligence Portal | Daily airport arrival ATFM delay by cause, arrivals, arrivals delayed > 15 min (2014 to 2025) | [ansperformance.eu](https://ansperformance.eu/capacity/atfm_apt), files `apt_dly_<year>.csv.bz2` |
| Deutscher Wetterdienst (DWD), Climate Data Center | Hourly station observations at the 12 airports: temperature, humidity, wind, gusts, precipitation, visibility, present-weather codes | [opendata.dwd.de](https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/) (CC BY 4.0) |
| Natural Earth | Country and state boundaries for the map figure only | [naturalearthdata.com](https://www.naturalearthdata.com) (public domain) |

[`docs/DATA.md`](docs/DATA.md) describes every processing step: file selection, encoding fixes, cleaning rules, the hourly-to-daily aggregation, the weather-code groups, target and feature definitions, missing values and known caveats. All 84 downloaded files are listed with URL and SHA-256 hash in [`data/source_manifest.json`](data/source_manifest.json).

The processed dataset is included: `data/processed/airport_day.csv` (and `.parquet`), 52,596 rows (12 airports × 4,383 days) and 79 columns. `feature_contract.json` lists the 37 model inputs and the same-day outcome columns that must not be used as inputs; `data_quality.md` shows the build checks.

## Repository structure

```
pipeline/             download raw data and build the dataset
  config.yaml         airports, DWD stations, thresholds, split dates
  download.py         downloads with retries, integrity checks and hashing
  build_dataset.py    cleaning, aggregation, features, checks
data/
  source_manifest.json
  geo/                Germany and neighbouring boundaries for the map (Natural Earth, public domain)
  processed/          airport_day.csv/.parquet, feature_contract.json, data_quality.*
notebooks/
  01_bi_overview.ipynb              traffic, delay and disruption KPIs
  02_bi_drivers.ipynb               delay causes, weather risk ratios, traffic x weather
  03_logistic_regression.ipynb      classification, linear model
  04_lightgbm_classification.ipynb  classification, gradient boosting + comparison
  05_ridge_regression.ipynb         delay per arrival, linear model
  06_lightgbm_regression.ipynb      delay per arrival, gradient boosting + comparison
outputs/              saved models, test predictions and metrics per model notebook
figures/              micropublication panels A to E (+ one extra) and make_figures.py
docs/DATA.md          data sources and processing
paper/                micropublication (LaTeX)
```

## Method in short

- **Split by time:** train 2014 to 2021, choose settings on 2022, choose calibration and alert threshold on 2023, test on 2024 to 2025 once.
- **Predictors (37):** airport, calendar, delay and traffic history from earlier days only, and same-day observed weather.
- **Benchmarks:** constant training rate, previous 28-day disruption rate, always zero, training mean, previous 28-day average delay.
- **Metrics:** average precision (main), ROC-AUC, Brier score, precision and recall at a fixed threshold for classification; MAE (main), RMSE and R² for regression.

## Reproduce

```bash
pip install -r requirements.txt

# optional: rebuild the dataset from the raw sources (about 134 MB download)
python -m pipeline download
python -m pipeline build

# run the notebooks in order (03 before 04, 05 before 06)
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 0*.ipynb

# figure panels for the micropublication
cd .. && python figures/make_figures.py
```

We checked that a rebuild from the raw files gives a dataset identical to the one in `data/processed/`, and that re-running the notebooks reproduces every file in `outputs/` exactly (Python 3.13, versions in `requirements.txt`).

## Limitations

- Observed weather stands in for a forecast, and weather is measured at one station per airport.
- ATFM delay measures flow-management regulations, not airline punctuality.
- The capacity side (staffing, runway closures, regulations planned in advance) is not in the data.
- Four airports (Dresden, Münster-Osnabrück, Nuremberg, Hannover) have almost no disrupted days, so per-airport results there are not meaningful.
- 2020 and 2021 (COVID) are part of the training period.

## Team

Siddharth Jain, Manasvi Japi, Rong Qin
