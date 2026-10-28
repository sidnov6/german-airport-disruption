"""Daily airport-level dataset: EUROCONTROL public daily ATFM arrival data + DWD hourly weather.

One row = one airport-day (UTC). All inputs are public and downloaded anonymously.

Availability assumptions (documented in the feature contract):
  * history features (lags, rolling means) use only days strictly before the row's day;
  * same-day weather is the OBSERVED weather, used as a stand-in for a day-ahead forecast
    ("perfect forecast" assumption), so scores are an upper bound for a real forecast;
  * the same day's arrival count is only a denominator for targets, never a predictor.
"""

from __future__ import annotations

import bz2
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

from .download import AcquisitionError, Manifest, download, sha256_file, utc_now_iso

CODE_VERSION = "1.0.0"
EC_URL = "https://www.eurocontrol.int/performance/data/download/csv/apt_dly_{year}.csv.bz2"
DWD_BASE = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly"

DWD = {
    # product: (directory, expected header, {raw var: name}, text columns)
    "TU": ("air_temperature", ["STATIONS_ID", "MESS_DATUM", "QN_9", "TT_TU", "RF_TU", "eor"], {"TT_TU": "temp_c", "RF_TU": "rh_pct"}, []),
    "FF": ("wind", ["STATIONS_ID", "MESS_DATUM", "QN_3", "F", "D", "eor"], {"F": "wind_ms", "D": "wind_dir_deg"}, []),
    "RR": ("precipitation", ["STATIONS_ID", "MESS_DATUM", "QN_8", "R1", "RS_IND", "WRTR", "eor"], {"R1": "precip_mm"}, []),
    "VV": ("visibility", ["STATIONS_ID", "MESS_DATUM", "QN_8", "V_VV_I", "V_VV", "eor"], {"V_VV": "vis_m"}, ["V_VV_I"]),
    "FX": ("extreme_wind", ["STATIONS_ID", "MESS_DATUM", "QN_8", "FX_911", "eor"], {"FX_911": "gust_ms"}, []),
    "WW": ("weather_phenomena", ["STATIONS_ID", "MESS_DATUM", "QN_8", "WW", "WW_Text", "eor"], {"WW": "ww"}, ["WW_Text"]),
}

# Present-weather code groups, WMO code tables 4677 (ww 00-99, observer) and 4680 (wawa, stored as 100-199, automatic).
def _r(a, b):
    """Inclusive integer range as a set, used to write the WMO weather-code groups."""
    return set(range(a, b + 1))

WW_GROUPS = {
    "fog": {28} | _r(40, 49) | {120} | _r(130, 135),
    "thunder": {17, 29} | _r(91, 99) | {126} | _r(190, 196),
    "snow": {22, 23, 26} | _r(36, 39) | _r(68, 79) | _r(83, 88) | {124, 127} | {167, 168} | _r(170, 178) | _r(185, 187),
    "freezing": {24, 56, 57, 66, 67, 125} | _r(154, 156) | _r(164, 166),
}

CAUSES = {"A": "accident", "C": "atc_capacity", "D": "deicing", "E": "equipment_non_atc", "G": "aerodrome_capacity",
          "I": "atc_industrial", "M": "airspace_mgmt", "N": "industrial_non_atc", "O": "other", "P": "special_event",
          "R": "atc_routeing", "S": "atc_staffing", "T": "equipment_atc", "V": "environment", "W": "weather", "NA": "not_specified"}


EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km between two latitude/longitude points (used for station-airport distance)."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float)) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def check_metadata_utc(param_text: str, variables: list[str], start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Verify DWD parameter metadata says UTC for every variable over [start, end)."""
    lines = [ln for ln in param_text.splitlines() if ln.strip()]
    rows = [ln.split(";") for ln in lines[1:]]
    result = {}
    for var in variables:
        covering = []
        for r in rows:
            if len(r) < 6 or not r[0].strip().isdigit() or r[4].strip() != var:
                continue
            von = pd.Timestamp(r[1].strip()).tz_localize("UTC")
            bis_s = r[2].strip()
            bis = pd.Timestamp(bis_s).tz_localize("UTC") + pd.Timedelta(days=1) if bis_s else pd.Timestamp.max.tz_localize("UTC")
            if von < end and bis > start:
                covering.append(";".join(r))
        ok = bool(covering) and all(("UTC" in c and "MEZ" not in c) for c in covering)
        result[var] = {"utc_verified": ok, "metadata_rows": covering}
    return result


class DailyConfig:
    """The validated contents of pipeline/config.yaml plus resolved paths, years and a config hash."""
    def __init__(self, path: str | Path):
        """Read the YAML file and reject missing keys, duplicate airports and unknown DWD products."""
        self.path = Path(path).resolve()
        self.root = self.path.parent.parent  # pipeline/config.yaml -> repository root
        self.raw = yaml.safe_load(self.path.read_text())
        req = ["dataset_version", "date_start", "date_end_exclusive", "airports", "dwd_products", "min_valid_hours", "targets", "splits", "paths"]
        miss = [k for k in req if k not in self.raw]
        if miss:
            raise ValueError(f"daily config missing keys: {miss}")
        icaos = [a["icao"] for a in self.raw["airports"]]
        if len(set(icaos)) != len(icaos):
            raise ValueError("duplicate airport in config")
        unknown = set(self.raw["dwd_products"]) - set(DWD)
        if unknown:
            raise ValueError(f"unsupported DWD products {unknown}")

    def __getitem__(self, k):
        """Access a raw config value, e.g. cfg['airports']."""
        return self.raw[k]

    def p(self, key) -> Path:
        """Resolve a path from the config's `paths` section relative to the repository root."""
        v = Path(self.raw["paths"][key])
        return v if v.is_absolute() else self.root / v

    @property
    def start(self):
        """First day of the dataset (UTC)."""
        return pd.Timestamp(self.raw["date_start"], tz="UTC")

    @property
    def end(self):
        """Day after the last day of the dataset (UTC, exclusive)."""
        return pd.Timestamp(self.raw["date_end_exclusive"], tz="UTC")

    @property
    def years(self):
        """Calendar years covered, e.g. 2014..2025."""
        return list(range(self.start.year, (self.end - pd.Timedelta(days=1)).year + 1))

    @property
    def config_hash(self):
        """SHA-256 of the config without paths, written to the outputs to tie them to the settings used."""
        sem = {k: v for k, v in self.raw.items() if k != "paths"}
        return hashlib.sha256(json.dumps(sem, sort_keys=True).encode()).hexdigest()

    @property
    def out_dir(self) -> Path:
        """Folder the processed dataset is written to (data/processed)."""
        return self.p("processed")


# ----------------------------------------------------------------------------- acquisition

def _validate_bz2(path: Path) -> None:
    """Decompress a whole bz2 file once to prove it is complete and not corrupted."""
    try:
        with bz2.open(path) as f:
            while f.read(1 << 20):
                pass
    except (OSError, EOFError) as e:
        raise AcquisitionError(f"bzip2 integrity failure: {path} ({e})") from e


def _listing(url: str) -> str:
    """Fetch a DWD directory listing (HTML) to find each station's exact archive name."""
    r = requests.get(url, timeout=(15, 60))
    if r.status_code != 200:
        raise AcquisitionError(f"HTTP {r.status_code} listing {url}")
    return r.text


def acquire(cfg: DailyConfig) -> list[dict]:
    """Download all raw inputs (12 EUROCONTROL years, 6 DWD products x 12 stations) and record them in the manifest.

    A cached file whose hash differs from the manifest stops the run, so a silently revised source is noticed.
    """
    manifest = Manifest(cfg.p("metadata") / "source_manifest.json")
    recs = []
    for y in cfg.years:
        url = EC_URL.format(year=y)
        dest = cfg.p("raw_eurocontrol_public") / f"apt_dly_{y}.csv.bz2"
        old = manifest.records.get(f"ec_apt_dly_{y}", {})
        if dest.exists() and old.get("sha256") and sha256_file(dest) != old["sha256"]:
            raise AcquisitionError(f"{dest.name} changed since it was registered; investigate the revision first")
        rec = download(url, dest, None, None, expect="bzip2")
        _validate_bz2(dest)
        manifest.upsert({"source_id": f"ec_apt_dly_{y}", "provider": "EUROCONTROL (Aviation Intelligence Portal, public)",
                         "url": url, "local_path": str(dest.relative_to(cfg.root)), "bytes": rec["bytes"], "sha256": rec["sha256"],
                         "http_status": rec["http_status"], "retrieval_utc": rec["retrieval_utc"], "cache": rec["cache"],
                         "coverage": f"{y} daily airport arrival ATFM delay, all reporting airports", "format": "csv.bz2"})
        recs.append(rec)
    listings = {}
    for prod in cfg["dwd_products"]:
        folder = DWD[prod][0]
        listings[prod] = _listing(f"{DWD_BASE}/{folder}/historical/")
        for a in cfg["airports"]:
            st = a["dwd_station"]
            names = sorted(set(re.findall(rf"stundenwerte_{prod}_{st}_\d{{8}}_\d{{8}}_hist\.zip", listings[prod])))
            if len(names) != 1:
                raise AcquisitionError(f"DWD {prod} station {st}: expected exactly one historical file, found {names}")
            url = f"{DWD_BASE}/{folder}/historical/{names[0]}"
            dest = cfg.p("raw_dwd") / prod / names[0]
            rec = download(url, dest, None, None, expect="zip")
            manifest.upsert({"source_id": f"dwd_{prod}_{st}", "provider": "Deutscher Wetterdienst (CDC open data)", "url": url,
                             "local_path": str(dest.relative_to(cfg.root)), "bytes": rec["bytes"], "sha256": rec["sha256"],
                             "http_status": rec["http_status"], "retrieval_utc": rec["retrieval_utc"], "cache": rec["cache"],
                             "product": prod, "station_id": st, "airport": a["icao"], "encoding": "latin-1", "format": "zip"})
            recs.append(rec)
    manifest.save()
    return recs


# ----------------------------------------------------------------------------- EUROCONTROL daily

def read_apt_dly(path: Path) -> tuple[pd.DataFrame, int]:
    """Decode line by line: UTF-8, falling back to Latin-1 (the 2025 file mixes both, in names only)."""
    raw = bz2.open(path).read()
    lines, fallback = [], 0
    for ln in raw.split(b"\n"):
        try:
            lines.append(ln.decode("utf-8"))
        except UnicodeDecodeError:
            lines.append(ln.decode("latin-1"))
            fallback += 1
    df = pd.read_csv(io.StringIO("\n".join(lines)), dtype={"APT_ICAO": str, "APT_NAME": str, "STATE_NAME": str, "ATFM_VERSION": str})
    return df, fallback


EC_COLUMNS = ["YEAR", "MONTH_NUM", "MONTH_MON", "FLT_DATE", "APT_ICAO", "APT_NAME", "STATE_NAME", "FLT_ARR_1", "DLY_APT_ARR_1"] + \
    [f"DLY_APT_ARR_{c}_1" for c in CAUSES] + ["FLT_ARR_1_DLY", "FLT_ARR_1_DLY_15", "ATFM_VERSION"]


def load_eurocontrol(cfg: DailyConfig) -> tuple[pd.DataFrame, dict]:
    """Read the yearly EUROCONTROL files, keep the 12 airports and parse dates.

    Returns the rows and a per-year audit (rows read, rows kept, encoding fallbacks, ATFM versions).
    """
    icaos = [a["icao"] for a in cfg["airports"]]
    parts, audit = [], {}
    for y in cfg.years:
        df, fb = read_apt_dly(cfg.p("raw_eurocontrol_public") / f"apt_dly_{y}.csv.bz2")
        if list(df.columns) != EC_COLUMNS:
            raise ValueError(f"apt_dly_{y}: unexpected columns {list(df.columns)}")
        sub = df[df["APT_ICAO"].isin(icaos)].copy()
        audit[str(y)] = {"rows_all_airports": int(len(df)), "rows_selected": int(len(sub)), "latin1_fallback_lines": fb,
                         "atfm_versions": sorted(sub["ATFM_VERSION"].dropna().unique().tolist())}
        parts.append(sub)
    ec = pd.concat(parts, ignore_index=True)
    ec["date"] = pd.to_datetime(ec["FLT_DATE"], utc=True, format="ISO8601").dt.normalize()
    key = ["APT_ICAO", "date"]
    dup = ec.duplicated(key, keep=False)
    if dup.any():
        if ec[dup].drop_duplicates().duplicated(key).any():
            raise ValueError("conflicting duplicate airport-day rows in EUROCONTROL data")
        ec = ec.drop_duplicates(key)
    return ec, audit


# ----------------------------------------------------------------------------- DWD hourly -> daily

def parse_dwd(zip_path: Path, prod: str, station: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Parse one DWD hourly archive into a UTC-indexed frame of the variables we use.

    Checks the header, the station id and that the metadata states UTC; -999 becomes missing.
    """
    folder, header, vars_, text_cols = DWD[prod]
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.startswith(f"produkt_{prod.lower()}_stunde_")]
        if len(members) != 1:
            raise ValueError(f"{zip_path.name}: expected one produkt member, got {members}")
        text = zf.read(members[0]).decode("latin-1")
        meta = zf.read(f"Metadaten_Parameter_{prod.lower()}_stunde_{station}.txt").decode("latin-1")
    first = [c.strip() for c in text.split("\n", 1)[0].split(";")]
    if first != header:
        raise ValueError(f"{zip_path.name}: header {first} != {header}")
    utc = check_metadata_utc(meta, list(vars_), start, end)
    if not all(v["utc_verified"] for v in utc.values()):
        raise ValueError(f"{zip_path.name}: metadata does not confirm UTC for {start.date()}..{end.date()}")
    use = ["STATIONS_ID", "MESS_DATUM", "eor"] + list(vars_)
    df = pd.read_csv(io.StringIO(text), sep=";", dtype=str, usecols=lambda c: c.strip() in use, keep_default_na=False,
                     on_bad_lines="error", engine="c")
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].str.strip()
    if not (df["eor"] == "eor").all():
        raise ValueError(f"{zip_path.name}: bad eor marker")
    if set(df["STATIONS_ID"].str.zfill(5)) != {station}:
        raise ValueError(f"{zip_path.name}: unexpected station ids")
    lo, hi = start.strftime("%Y%m%d%H"), end.strftime("%Y%m%d%H")
    df = df[(df["MESS_DATUM"] >= lo) & (df["MESS_DATUM"] < hi)]
    out = pd.DataFrame({"ts": pd.to_datetime(df["MESS_DATUM"], format="%Y%m%d%H", utc=True)})
    for raw, name in vars_.items():
        v = pd.to_numeric(df[raw], errors="raise")
        out[name] = v.where(v != -999).astype("float64")
    dup = out.duplicated("ts", keep=False)
    if dup.any():
        if out[dup].drop_duplicates().duplicated("ts").any():
            raise ValueError(f"{zip_path.name}: conflicting duplicate hours")
        out = out.drop_duplicates("ts")
    return out.set_index("ts")


def daily_weather(hourly: dict[str, pd.DataFrame], start, end, min_valid: int) -> pd.DataFrame:
    """Aggregate one station's hourly products to UTC days. Nothing is filled or interpolated."""
    grid = pd.date_range(start, end, freq="h", inclusive="left", tz="UTC")
    h = pd.DataFrame(index=grid)
    for prod, df in hourly.items():
        h = h.join(df, how="left")
    day = h.index.floor("D")
    g = lambda s: s.groupby(day)  # noqa: E731
    out = pd.DataFrame(index=pd.DatetimeIndex(sorted(set(day)), name="date"))

    def valid_mask(col):
        """Number of valid (non-missing) hours of a variable per day."""
        return g(h[col].notna()).sum()

    def agg(col, fn, name, need=True):
        """Daily aggregate of a variable (mean, min, max or sum), missing when fewer than min_valid hours exist."""
        v = getattr(g(h[col]), fn)()
        n = valid_mask(col)
        out[name] = v.where(n >= min_valid) if need else v
        return n

    def count(cond, col, name):
        """Daily count of hours meeting a condition, among valid hours only."""
        n = valid_mask(col)
        out[name] = g(cond.astype(float).where(h[col].notna())).sum().where(n >= min_valid)

    if "temp_c" in h:
        n = agg("temp_c", "mean", "temp_mean_c"); agg("temp_c", "min", "temp_min_c"); agg("temp_c", "max", "temp_max_c")
        out["temp_valid_hours"] = n
        out["frost_day"] = (out["temp_min_c"] <= 0).astype("boolean").where(out["temp_min_c"].notna())
        agg("rh_pct", "mean", "rh_mean_pct")
    if "wind_ms" in h:
        out["wind_valid_hours"] = agg("wind_ms", "mean", "wind_mean_ms"); agg("wind_ms", "max", "wind_max_ms")
    if "gust_ms" in h:
        out["gust_valid_hours"] = agg("gust_ms", "max", "gust_max_ms")
        count(h["gust_ms"] >= 15, "gust_ms", "gust_hours_ge15")
    if "precip_mm" in h:
        out["precip_valid_hours"] = agg("precip_mm", "sum", "precip_sum_mm"); agg("precip_mm", "max", "precip_max_1h_mm")
        count(h["precip_mm"] >= 0.1, "precip_mm", "precip_hours")
        if "temp_c" in h:
            cold = (h["precip_mm"] >= 0.1) & (h["temp_c"] <= 1.0)
            out["cold_precip_hours"] = g(cold.astype(float).where(h["precip_mm"].notna() & h["temp_c"].notna())).sum() \
                .where(valid_mask("precip_mm") >= min_valid)
    if "vis_m" in h:
        out["vis_valid_hours"] = agg("vis_m", "min", "vis_min_m")
        count(h["vis_m"] < 1000, "vis_m", "low_vis_hours_lt1000")
        count(h["vis_m"] < 5000, "vis_m", "vis_hours_lt5000")
    if "ww" in h:
        ww = h["ww"]
        out["ww_reported_hours"] = g((ww >= 0).astype(float).where(ww.notna())).sum()
        for grp, codes in WW_GROUPS.items():
            count(ww.isin(list(codes)), "ww", f"{grp}_hours")
    out.index = out.index.rename("date")
    return out


def load_weather(cfg: DailyConfig) -> tuple[pd.DataFrame, dict]:
    """Parse all six DWD products for every airport and aggregate them to daily values."""
    start, end = cfg.start, cfg.end
    frames, audit = [], {}
    for a in cfg["airports"]:
        st = a["dwd_station"]
        hourly = {}
        for prod in cfg["dwd_products"]:
            files = sorted((cfg.p("raw_dwd") / prod).glob(f"stundenwerte_{prod}_{st}_*_hist.zip"))
            if len(files) != 1:
                raise FileNotFoundError(f"DWD {prod} {st}: expected one file, found {files}. Run `python -m pipeline download` first.")
            hourly[prod] = parse_dwd(files[0], prod, st, start, end)
        d = daily_weather(hourly, start, end, cfg["min_valid_hours"])
        d.insert(0, "airport_icao", a["icao"])
        d.insert(1, "dwd_station", st)
        frames.append(d.reset_index())
        audit[a["icao"]] = {"station": st, **{f"{p}_hours": int(len(hourly[p])) for p in hourly}}
    return pd.concat(frames, ignore_index=True), audit


# ----------------------------------------------------------------------------- calendar

def easter(y: int) -> pd.Timestamp:
    """Date of Easter Sunday for a year (Gregorian computus), used for the movable holidays."""
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * m + 114) // 31
    day = (h + l_ - 7 * m + 114) % 31 + 1
    return pd.Timestamp(year=y, month=month, day=day, tz="UTC")


def german_national_holidays(years) -> set:
    """All German national public holidays for the given years (fixed dates plus Easter-based ones)."""
    out = set()
    for y in years:
        e = easter(y)
        for d in [pd.Timestamp(f"{y}-01-01", tz="UTC"), e - pd.Timedelta(days=2), e + pd.Timedelta(days=1),
                  pd.Timestamp(f"{y}-05-01", tz="UTC"), e + pd.Timedelta(days=39), e + pd.Timedelta(days=50),
                  pd.Timestamp(f"{y}-10-03", tz="UTC"), pd.Timestamp(f"{y}-12-25", tz="UTC"), pd.Timestamp(f"{y}-12-26", tz="UTC")]:
            out.add(d)
    return out


# ----------------------------------------------------------------------------- dataset

def assign_split(dates: pd.Series, splits: dict) -> pd.Series:
    """Label each date train / model_select / calibrate / test from the config; overlapping ranges are an error."""
    out = pd.Series(pd.NA, index=dates.index, dtype="string")
    hits = pd.Series(0, index=dates.index)
    for name, (a, b) in splits.items():
        m = (dates >= pd.Timestamp(a, tz="UTC")) & (dates < pd.Timestamp(b, tz="UTC"))
        out[m] = name
        hits += m.astype(int)
    if (hits > 1).any():
        raise AssertionError("overlapping splits")
    return out


def history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lags and trailing windows per airport, using days strictly before each row (shift(1))."""
    df = df.sort_values(["airport_icao", "date"]).copy()
    for c in ("arrivals", "atfm_delay_per_arrival", "share_arrivals_delayed_15", "weather_delay_per_arrival", "disrupted_day"):
        df[c + "__f"] = pd.array(df[c], dtype="Float64").to_numpy(dtype="float64", na_value=np.nan)
    g = df.groupby("airport_icao", sort=False)
    prev = lambda col, k=1: g[col + "__f"].shift(k)  # noqa: E731
    df["arrivals_lag1"] = prev("arrivals")
    df["arrivals_lag7"] = prev("arrivals", 7)
    df["delay_per_arrival_lag1"] = prev("atfm_delay_per_arrival")
    df["share_delayed15_lag1"] = prev("share_arrivals_delayed_15")
    df["disrupted_lag1"] = prev("disrupted_day")

    def trailing(col, w, minp):
        """Mean of a column over the previous w days (excluding today), requiring at least minp values."""
        return g[col + "__f"].transform(lambda s: s.shift(1).rolling(w, min_periods=minp).mean())

    df["arrivals_mean_7d"] = trailing("arrivals", 7, 5)
    df["delay_per_arrival_mean_7d"] = trailing("atfm_delay_per_arrival", 7, 5)
    df["delay_per_arrival_mean_28d"] = trailing("atfm_delay_per_arrival", 28, 20)
    df["disrupted_rate_28d"] = trailing("disrupted_day", 28, 20)
    df["weather_delay_per_arrival_mean_28d"] = trailing("weather_delay_per_arrival", 28, 20)
    return df.drop(columns=[c for c in df.columns if c.endswith("__f")])


WEATHER_PREDICTORS = ["temp_mean_c", "temp_min_c", "temp_max_c", "frost_day", "rh_mean_pct", "wind_mean_ms", "wind_max_ms",
                      "gust_max_ms", "gust_hours_ge15", "precip_sum_mm", "precip_max_1h_mm", "precip_hours", "cold_precip_hours",
                      "vis_min_m", "low_vis_hours_lt1000", "vis_hours_lt5000", "fog_hours", "thunder_hours", "snow_hours",
                      "freezing_hours"]
HISTORY_PREDICTORS = ["arrivals_lag1", "arrivals_lag7", "arrivals_mean_7d", "delay_per_arrival_lag1", "share_delayed15_lag1",
                      "disrupted_lag1", "delay_per_arrival_mean_7d", "delay_per_arrival_mean_28d", "disrupted_rate_28d",
                      "weather_delay_per_arrival_mean_28d"]
CALENDAR_PREDICTORS = ["weekday", "month", "doy_sin", "doy_cos", "is_weekend", "is_public_holiday"]
TARGETS = ["disrupted_day", "severe_disrupted_day", "any_atfm_delay_day", "weather_atfm_day", "atfm_delay_per_arrival",
           "share_arrivals_delayed_15", "atfm_delay_min", "weather_delay_per_arrival"]


def build(cfg: DailyConfig) -> dict:
    """Assemble the airport-day table: grid, EUROCONTROL join, weather join, targets, calendar, history, split.

    A blank EUROCONTROL delay cell is treated as 0 (no delay recorded); rows with no arrivals get no label.
    """
    ec, ec_audit = load_eurocontrol(cfg)
    wx, wx_audit = load_weather(cfg)
    days = pd.date_range(cfg.start, cfg.end, freq="D", inclusive="left", tz="UTC")
    base = pd.MultiIndex.from_product([[a["icao"] for a in cfg["airports"]], days], names=["airport_icao", "date"]).to_frame(index=False)
    meta = pd.DataFrame(cfg["airports"]).rename(columns={"icao": "airport_icao", "name": "airport_name"})
    base = base.merge(meta[["airport_icao", "airport_name", "dwd_station"]], on="airport_icao", how="left", validate="many_to_one")

    e = ec.rename(columns={"APT_ICAO": "airport_icao"})
    blank = e["DLY_APT_ARR_1"].isna()
    cols = {"FLT_ARR_1": "arrivals", "DLY_APT_ARR_1": "atfm_delay_min", "FLT_ARR_1_DLY": "arrivals_atfm_delayed",
            "FLT_ARR_1_DLY_15": "arrivals_atfm_delayed_15", "ATFM_VERSION": "atfm_version"}
    cols.update({f"DLY_APT_ARR_{c}_1": f"delay_{n}_min" for c, n in CAUSES.items()})
    e = e[["airport_icao", "date"] + list(cols)].rename(columns=cols)
    num = [c for c in cols.values() if c not in ("arrivals", "atfm_version")]
    e["delay_blank_in_source"] = blank.to_numpy()
    e[num] = e[num].fillna(0.0)  # blank = no ATFM delay recorded that day (explicit zeros from 2023 onward)
    e["eurocontrol_row_present"] = True
    df = base.merge(e, on=["airport_icao", "date"], how="left", validate="one_to_one")
    df["eurocontrol_row_present"] = df["eurocontrol_row_present"].fillna(False).astype(bool)
    df = df.merge(wx.drop(columns=["dwd_station"]), on=["airport_icao", "date"], how="left", validate="one_to_one")
    assert len(df) == len(base)

    ok = df["arrivals"].notna() & (df["arrivals"] > 0)
    arr = df["arrivals"].where(ok)
    df["atfm_delay_per_arrival"] = (df["atfm_delay_min"] / arr).astype("Float64")
    df["share_arrivals_delayed_15"] = (df["arrivals_atfm_delayed_15"] / arr).astype("Float64")
    df["weather_delay_per_arrival"] = (df["delay_weather_min"] / arr).astype("Float64")
    t = cfg["targets"]

    def lab(cond):
        """Turn a boolean condition into a nullable 0/1 label that is missing where arrivals are missing or zero."""
        s = pd.Series(pd.NA, index=df.index, dtype="Int8")
        s[ok] = cond[ok].astype("int8")
        return s

    df["disrupted_day"] = lab(df["share_arrivals_delayed_15"] >= t["disrupted_share_15"])
    df["severe_disrupted_day"] = lab(df["share_arrivals_delayed_15"] >= t["severe_share_15"])
    df["any_atfm_delay_day"] = lab(df["atfm_delay_min"] > 0)
    df["weather_atfm_day"] = lab(df["delay_weather_min"] > 0)

    d = df["date"]
    df["weekday"] = d.dt.weekday.astype("Int8")
    df["month"] = d.dt.month.astype("Int8")
    doy = d.dt.dayofyear.astype(float)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["is_weekend"] = (d.dt.weekday >= 5)
    df["is_public_holiday"] = d.isin(german_national_holidays(cfg.years))
    df["year"] = d.dt.year.astype("Int16")
    df["covid_traffic_period"] = (d >= pd.Timestamp("2020-03-15", tz="UTC")) & (d < pd.Timestamp("2022-01-01", tz="UTC"))

    df = history_features(df)
    df["split"] = assign_split(df["date"], cfg["splits"])
    df = df.sort_values(["date", "airport_icao"], kind="mergesort").reset_index(drop=True)
    df["airport_icao"] = df["airport_icao"].astype("string")
    for c in ("frost_day",):
        df[c] = df[c].astype("boolean")
    return {"df": df, "ec_audit": ec_audit, "wx_audit": wx_audit}


def feature_contract(cfg: DailyConfig) -> dict:
    """Describe the 37 allowed predictors (with when each is known) and the forbidden same-day outcome columns."""
    wx_av = "Same-day OBSERVED weather standing in for a day-ahead forecast (perfect-forecast assumption; scores are an upper bound)"
    hist_av = "Computed from days strictly before the row's day (known at the start of the day)"
    cal_av = "Known in advance"
    desc = {
        "airport_icao": ("categorical", "ICAO airport code", cal_av),
        "weekday": ("ordinal", "UTC weekday, Monday=0", cal_av), "month": ("ordinal", "1-12", cal_av),
        "doy_sin": ("numeric", "sin(2*pi*day_of_year/365.25)", cal_av), "doy_cos": ("numeric", "cos(2*pi*day_of_year/365.25)", cal_av),
        "is_weekend": ("flag", "Saturday or Sunday", cal_av), "is_public_holiday": ("flag", "German national public holiday", cal_av),
        "temp_mean_c": ("numeric", "mean hourly air temperature, degC", wx_av), "temp_min_c": ("numeric", "min hourly temperature, degC", wx_av),
        "temp_max_c": ("numeric", "max hourly temperature, degC", wx_av), "frost_day": ("flag", "temp_min_c <= 0", wx_av),
        "rh_mean_pct": ("numeric", "mean relative humidity, %", wx_av), "wind_mean_ms": ("numeric", "mean hourly wind speed, m/s", wx_av),
        "wind_max_ms": ("numeric", "max hourly-mean wind speed, m/s", wx_av), "gust_max_ms": ("numeric", "max hourly gust (FX_911), m/s", wx_av),
        "gust_hours_ge15": ("count", "hours with gust >= 15 m/s", wx_av), "precip_sum_mm": ("numeric", "daily precipitation, mm", wx_av),
        "precip_max_1h_mm": ("numeric", "max hourly precipitation, mm", wx_av), "precip_hours": ("count", "hours with >= 0.1 mm", wx_av),
        "cold_precip_hours": ("count", "hours with >= 0.1 mm and temp <= 1 degC (snow/ice proxy)", wx_av),
        "vis_min_m": ("numeric", "minimum hourly visibility, m", wx_av), "low_vis_hours_lt1000": ("count", "hours with visibility < 1000 m", wx_av),
        "vis_hours_lt5000": ("count", "hours with visibility < 5000 m", wx_av),
        "fog_hours": ("count", "hours with a fog weather code (WMO 4677/4680)", wx_av),
        "thunder_hours": ("count", "hours with a thunderstorm weather code", wx_av), "snow_hours": ("count", "hours with a snow weather code", wx_av),
        "freezing_hours": ("count", "hours with freezing drizzle/rain code", wx_av),
        "arrivals_lag1": ("numeric", "IFR arrivals previous day", hist_av), "arrivals_lag7": ("numeric", "IFR arrivals same weekday last week", hist_av),
        "arrivals_mean_7d": ("numeric", "mean arrivals over previous 7 days", hist_av),
        "delay_per_arrival_lag1": ("numeric", "ATFM arrival delay per arrival previous day, min", hist_av),
        "share_delayed15_lag1": ("numeric", "share of arrivals ATFM-delayed >15 min previous day", hist_av),
        "disrupted_lag1": ("flag", "previous day was a disrupted day", hist_av),
        "delay_per_arrival_mean_7d": ("numeric", "mean delay per arrival previous 7 days", hist_av),
        "delay_per_arrival_mean_28d": ("numeric", "mean delay per arrival previous 28 days", hist_av),
        "disrupted_rate_28d": ("numeric", "share of disrupted days in previous 28 days", hist_av),
        "weather_delay_per_arrival_mean_28d": ("numeric", "mean weather-coded delay per arrival previous 28 days", hist_av),
    }
    preds = [{"name": k, "kind": v[0], "definition": v[1], "availability_assumption": v[2], "allowed_for_model_input": True,
              "missingness": "null when fewer than min_valid_hours valid hours (weather) or insufficient history (lags)"} for k, v in desc.items()]
    return {"dataset_version": cfg["dataset_version"], "config_hash": cfg.config_hash, "grain": "airport_icao x UTC date",
            "predictor_allowlist": [p["name"] for p in preds], "predictors": preds,
            "targets": {"disrupted_day": f"share_arrivals_delayed_15 >= {cfg['targets']['disrupted_share_15']}",
                        "severe_disrupted_day": f"share_arrivals_delayed_15 >= {cfg['targets']['severe_share_15']}",
                        "any_atfm_delay_day": "atfm_delay_min > 0", "weather_atfm_day": "delay_weather_min > 0",
                        "atfm_delay_per_arrival": "atfm_delay_min / arrivals", "share_arrivals_delayed_15": "arrivals_atfm_delayed_15 / arrivals"},
            "forbidden_same_day_outcomes": ["arrivals", "atfm_delay_min", "arrivals_atfm_delayed", "arrivals_atfm_delayed_15"] +
                                           [f"delay_{n}_min" for n in CAUSES.values()] + TARGETS,
            "keys": ["airport_icao", "date", "split"],
            "note": "Target definition is EUROCONTROL ATFM airport arrival delay (network regulations), not gate/schedule punctuality."}


def checks(df: pd.DataFrame, contract: dict, cfg: DailyConfig) -> dict:
    """Run the build checks: row count, unique keys, splits, label consistency, no leakage, lag correctness."""
    n_exp = len(cfg["airports"]) * len(pd.date_range(cfg.start, cfg.end, freq="D", inclusive="left"))
    allow = contract["predictor_allowlist"]
    c = {
        "row_count_airports_x_days": len(df) == n_exp,
        "keys_unique": not df.duplicated(["airport_icao", "date"]).any(),
        "every_row_has_split": bool(df["split"].notna().all()),
        "targets_consistent": bool(((df["share_arrivals_delayed_15"] >= cfg["targets"]["disrupted_share_15"]).astype("Int8")
                                    .where(df["disrupted_day"].notna()) == df["disrupted_day"]).dropna().all()),
        "no_same_day_outcome_in_predictors": not (set(contract["forbidden_same_day_outcomes"]) & set(allow)),
    }
    # history features must equal a recomputation from strictly earlier days
    s = df[df["airport_icao"] == df["airport_icao"].iloc[0]].sort_values("date")
    c["lag1_uses_previous_day"] = bool((s["arrivals_lag1"].iloc[1:].to_numpy() == s["arrivals"].iloc[:-1].to_numpy())[
        s["arrivals"].iloc[:-1].notna().to_numpy() & s["arrivals_lag1"].iloc[1:].notna().to_numpy()].all())
    both = {}
    for sp in cfg["splits"]:
        y = df.loc[df["split"] == sp, "disrupted_day"].dropna()
        both[sp] = bool((y == 0).any() and (y == 1).any())
    c["each_split_has_both_classes"] = all(both.values())
    return c


def quality(df: pd.DataFrame, contract: dict, cfg: DailyConfig, ec_audit, wx_audit, chk) -> dict:
    """Summarise the table: label counts by split, airport and year, missing values, source audits and checks."""
    allow = contract["predictor_allowlist"]
    def prev(g):
        """Rows, labelled rows, disrupted rows and disruption rate for one group."""
        y = g["disrupted_day"].dropna()
        return {"rows": int(len(g)), "labeled": int(len(y)), "disrupted": int(y.sum()), "rate": float(y.mean()) if len(y) else None}
    station_dist = {}
    inv = {a["icao"]: a for a in cfg["airports"]}
    return {
        "dataset_version": cfg["dataset_version"], "config_hash": cfg.config_hash, "code_version": CODE_VERSION,
        "built_utc": utc_now_iso(), "rows": int(len(df)), "airports": len(cfg["airports"]),
        "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
        "by_split": {s: prev(g) for s, g in df.groupby("split")},
        "by_airport": {a: prev(g) for a, g in df.groupby("airport_icao")},
        "by_year": {str(y): prev(g) for y, g in df.groupby("year")},
        "missing_predictors": {c: int(df[c].isna().sum()) for c in allow if int(df[c].isna().sum())},
        "missing_targets": int(df["disrupted_day"].isna().sum()),
        "eurocontrol_missing_airport_days": int((~df["eurocontrol_row_present"]).sum()),
        "eurocontrol": ec_audit, "weather": wx_audit, "checks": chk, "station_distance_km": station_dist,
    }


def station_distances(cfg: DailyConfig) -> dict:
    """Distance between each airport reference point and its DWD station (from the station inventory files)."""
    out = {}
    for a in cfg["airports"]:
        z = sorted((cfg.p("raw_dwd") / "TU").glob(f"stundenwerte_TU_{a['dwd_station']}_*_hist.zip"))[0]
        with zipfile.ZipFile(z) as zf:
            geo = zf.read(f"Metadaten_Geographie_{a['dwd_station']}.txt").decode("latin-1").strip().splitlines()
        last = geo[-1].split(";")
        lat, lon = float(last[2]), float(last[3])
        out[a["icao"]] = {"station": a["dwd_station"], "station_name": last[-1].strip(), "lat": lat, "lon": lon,
                          "distance_km": round(float(haversine_km(a["lat"], a["lon"], lat, lon)), 2)}
    return out


def run(cfg: DailyConfig) -> dict:
    """Build the dataset, check it, and write the parquet/CSV files, contract and quality report."""
    res = build(cfg)
    df = res["df"]
    contract = feature_contract(cfg)
    chk = checks(df, contract, cfg)
    q = quality(df, contract, cfg, res["ec_audit"], res["wx_audit"], chk)
    q["station_distance_km"] = station_distances(cfg)
    far = {k: v for k, v in q["station_distance_km"].items() if v["distance_km"] > cfg.raw.get("max_station_distance_km", 10)}
    q["checks"]["stations_at_airports"] = not far
    out = cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "airport_day.parquet", index=False)
    feats = df[list(dict.fromkeys(["airport_icao", "date", "split"] + contract["predictor_allowlist"]))]
    feats.to_parquet(out / "airport_day_features.parquet", index=False)
    df.assign(date=df["date"].dt.strftime("%Y-%m-%d")).to_csv(out / "airport_day.csv", index=False)
    (out / "feature_contract.json").write_text(json.dumps(contract, indent=2))
    (out / "data_quality.json").write_text(json.dumps(q, indent=2, default=str))
    (out / "data_quality.md").write_text(render_md(q, contract))
    return q


def render_md(q: dict, contract: dict) -> str:
    """Format the quality summary as the Markdown report data/processed/data_quality.md."""
    f = lambda v: "—" if v is None else (f"{v:.3f}" if isinstance(v, float) else f"{v:,}" if isinstance(v, int) else str(v))  # noqa: E731
    L = [f"# Daily airport dataset — {q['dataset_version']}", "",
         f"{q['rows']:,} airport-days · {q['airports']} airports · {q['date_range'][0]} to {q['date_range'][1]} · "
         f"{len(contract['predictor_allowlist'])} predictors · config `{q['config_hash'][:12]}`", "",
         "Target `disrupted_day`: " + contract["targets"]["disrupted_day"] + " (EUROCONTROL ATFM airport arrival delay).", "",
         "## Checks", "", "| check | result |", "|---|---|"]
    L += [f"| {k} | {'PASS' if v else 'FAIL'} |" for k, v in q["checks"].items()]
    for title, key in (("By split", "by_split"), ("By airport", "by_airport"), ("By year", "by_year")):
        L += ["", f"## {title}", "", "| group | rows | labeled | disrupted | rate |", "|---|---:|---:|---:|---:|"]
        L += [f"| {k} | {f(v['rows'])} | {f(v['labeled'])} | {f(v['disrupted'])} | {f(v['rate'])} |" for k, v in q[key].items()]
    L += ["", "## Stations", "", "| airport | station | name | distance km |", "|---|---|---|---:|"]
    L += [f"| {k} | {v['station']} | {v['station_name']} | {v['distance_km']} |" for k, v in q["station_distance_km"].items()]
    L += ["", "## Missing predictor values", "", "| predictor | nulls |", "|---|---:|"]
    L += [f"| {k} | {v:,} |" for k, v in q["missing_predictors"].items()] or ["| none | 0 |"]
    L += ["", "## Assumptions", "",
          "- Blank EUROCONTROL delay cells (before 2023) are treated as zero ATFM delay; from 2023 the source reports explicit zeros.",
          "- Weather predictors are same-day observations used as a stand-in for a forecast; results are an upper bound.",
          "- Weather-code features rely on reported present-weather codes; `-1` (not reported) counts as no event.",
          "- 2020-2021 traffic collapse (COVID) is inside the training period; `covid_traffic_period` is kept as an audit column.",
          "- ATFM delay is network-regulation delay, not airline schedule punctuality."]
    return "\n".join(L) + "\n"
