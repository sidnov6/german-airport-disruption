"""Command line entry point.

    python -m pipeline download   # EUROCONTROL yearly files + DWD hourly station archives -> data/raw/
    python -m pipeline build      # raw files -> data/processed/airport_day.{parquet,csv} + contract + quality report
"""

from __future__ import annotations

import argparse
import sys

from . import build_dataset as bd
from .download import AcquisitionError


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline")
    ap.add_argument("command", choices=["download", "build"])
    ap.add_argument("--config", default="pipeline/config.yaml")
    a = ap.parse_args(argv)
    cfg = bd.DailyConfig(a.config)
    try:
        if a.command == "download":
            recs = bd.acquire(cfg)
            print(f"{len(recs)} files ready ({sum(r['cache'] == 'downloaded' for r in recs)} downloaded, "
                  f"{sum(r['cache'] == 'reused' for r in recs)} reused). Manifest: data/source_manifest.json")
            return 0
        q = bd.run(cfg)
    except (AcquisitionError, FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"{q['rows']:,} airport-days, {q['airports']} airports, {q['date_range'][0]}..{q['date_range'][1]}")
    for s_, v in q["by_split"].items():
        print(f"  {s_:13s} rows={v['rows']:>6,} labeled={v['labeled']:>6,} disrupted={v['disrupted']:>5,} rate={v['rate']:.3f}")
    failed = [k for k, v in q["checks"].items() if not v]
    print("checks:", "all pass" if not failed else f"FAILED {failed}")
    print(f"Outputs: {cfg.out_dir}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
