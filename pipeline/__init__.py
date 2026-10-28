"""Code overview: daily ATFM arrival disruption at 12 German airports, 2014-2025.

This repository turns two public data sources into one analysis table, analyses
it with BI notebooks, trains four models on it and produces the figures for the
micropublication. Everything runs from the repository root.

1. Data pipeline (this package, ``pipeline/``)
-----------------------------------------------
``python -m pipeline download`` then ``python -m pipeline build``.

``config.yaml``
    The 12 airports with their DWD weather station, the date range, the
    disruption thresholds (2% and 10% of arrivals delayed over 15 min) and the
    year-based split. ``build_dataset.DailyConfig`` reads and validates it.

``download.py``
    Generic download helpers: retries, a check that the file is the expected
    archive type and not an HTML error page, SHA-256 hashing, and the
    ``Manifest`` that records every downloaded file in
    ``data/source_manifest.json``.

``build_dataset.py``
    The actual data processing, in the order it runs:

    * ``acquire`` downloads the 12 yearly EUROCONTROL files and, for each
      airport, the 6 DWD hourly products (temperature, wind, precipitation,
      visibility, gusts, present-weather codes): 84 files.
    * ``load_eurocontrol`` reads each yearly file (``read_apt_dly`` fixes the
      mixed UTF-8 / Latin-1 encoding of 2025), keeps the 12 airports and
      removes exact duplicates.
    * ``load_weather`` parses every DWD archive (``parse_dwd``: header check,
      UTC check, -999 -> missing) and aggregates hours to UTC days
      (``daily_weather``: a daily value needs at least 20 valid hours).
    * ``build`` puts both on a complete airport x day grid (52,596 rows),
      treats a blank EUROCONTROL delay cell as zero, computes the targets
      (delay per arrival, share delayed over 15 min, ``disrupted_day``), the
      calendar features, the history features from earlier days only
      (``history_features``) and the split.
    * ``feature_contract`` lists the 37 allowed model inputs and the same-day
      outcome columns that must never be inputs; ``checks`` and ``quality``
      verify the table and summarise it; ``run`` writes everything to
      ``data/processed/``.

``__main__.py``
    The command line entry point for the two commands above.

Every step is described in plain language, with the exact rules, in
``docs/DATA.md``.

2. Notebooks (``notebooks/``)
-----------------------------
Run in order; each reads ``data/processed/airport_day.parquet``.

* ``01_bi_overview`` and ``02_bi_drivers``: descriptive KPIs, delay causes,
  weather risk ratios and traffic x weather tables.
* ``03_logistic_regression`` and ``04_lightgbm_classification``: will an
  airport-day be disrupted? (average precision, ROC-AUC, Brier score)
* ``05_ridge_regression`` and ``06_lightgbm_regression``: how many ATFM delay
  minutes per arrival? (MAE, RMSE, R2)

All four model notebooks train on 2014-2021, choose settings on 2022, choose
calibration and the alert threshold on 2023, and score 2024-2025 once. They
save the model, the test predictions and the metrics to ``outputs/``.

3. Figures (``figures/make_figures.py``)
----------------------------------------
Reads ``outputs/`` and the processed data and draws the five micropublication
panels (map, precision-recall curves, weather risk ratios, traffic x weather,
pipeline diagram).

Reproducibility
---------------
A rebuild from the raw downloads gives a dataset identical to the one in
``data/processed/``, and re-running the notebooks reproduces every file in
``outputs/`` exactly (library versions in ``requirements.txt``).
"""
