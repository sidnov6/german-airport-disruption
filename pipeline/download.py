"""Downloading with retries, integrity checks, SHA-256 hashing and the source manifest."""

from __future__ import annotations

import hashlib
import json
import os
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import requests

TIMEOUT = (15, 120)  # connect, read seconds
MAX_ATTEMPTS = 4


class AcquisitionError(RuntimeError):
    pass


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sniff_format(path: Path) -> str:
    with open(path, "rb") as f:
        head = f.read(8)
    if head.startswith(b"PK\x03\x04") or head.startswith(b"PK\x05\x06"):
        return "zip"
    if head.startswith(b"\x1f\x8b"):
        return "gzip"
    if head.startswith(b"BZh"):
        return "bzip2"
    if head.startswith(b"\xfd7zXZ"):
        return "xz"
    if head.startswith(b"7z\xbc\xaf"):
        return "7z"
    lower = head.lower()
    if lower.startswith(b"<!doc") or lower.startswith(b"<html") or lower.startswith(b"<?xml"):
        return "html"
    return "text"


def validate_zip(path: Path) -> list[str]:
    """CRC-check every member and reject unsafe member paths. Returns member names."""
    try:
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
            if bad is not None:
                raise AcquisitionError(f"ZIP CRC failure in {path.name}: member {bad}")
            names = zf.namelist()
    except zipfile.BadZipFile as e:
        raise AcquisitionError(f"Not a valid ZIP: {path} ({e})") from e
    for n in names:
        if not is_safe_member(n):
            raise AcquisitionError(f"Unsafe archive member path in {path.name}: {n!r}")
    return names


def is_safe_member(name: str) -> bool:
    p = PurePosixPath(name.replace("\\", "/"))
    return not (p.is_absolute() or ".." in p.parts or (p.parts and ":" in p.parts[0]))


def download(url: str, dest: Path, expected_sha256: str | None = None,
             expected_bytes: int | None = None, expect: str = "zip") -> dict:
    """Download with bounded retries, atomic rename, integrity checks and cache reuse.

    Returns a manifest record. A cached file is reused if it matches the expected hash
    (or, with no pinned hash, passes the archive integrity check).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        digest = sha256_file(dest)
        if expected_sha256 is None or digest == expected_sha256:
            if expect == "zip":
                validate_zip(dest)
            return {"local_path": str(dest), "bytes": dest.stat().st_size, "sha256": digest,
                    "http_status": None, "retrieval_utc": None, "cache": "reused"}
        # A different hash on cached data means the pinned version changed; do not overwrite silently.
        raise AcquisitionError(
            f"Cached {dest.name} has sha256 {digest}, expected {expected_sha256}. "
            "Investigate a source revision before replacing it."
        )

    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            retrieval = utc_now_iso()
            with requests.get(url, stream=True, timeout=TIMEOUT, allow_redirects=True) as r:
                status = r.status_code
                if status in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"transient HTTP {status}")
                if status != 200:
                    raise AcquisitionError(f"HTTP {status} for {url}")
                ctype = r.headers.get("content-type", "")
                if "text/html" in ctype.lower():
                    raise AcquisitionError(f"Got HTML (login/error page?) instead of data from {url}")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            fmt = sniff_format(tmp)
            if fmt == "html":
                raise AcquisitionError(f"Downloaded content is HTML, not data: {url}")
            if expect == "zip":
                if fmt != "zip":
                    raise AcquisitionError(f"Expected ZIP from {url}, got {fmt}")
                validate_zip(tmp)
            size = tmp.stat().st_size
            digest = sha256_file(tmp)
            if expected_bytes is not None and size != expected_bytes:
                raise AcquisitionError(f"{dest.name}: {size} bytes, expected {expected_bytes}. Possible source revision.")
            if expected_sha256 is not None and digest != expected_sha256:
                raise AcquisitionError(f"{dest.name}: sha256 {digest} != pinned {expected_sha256}. Possible source revision.")
            os.replace(tmp, dest)
            return {"local_path": str(dest), "bytes": size, "sha256": digest,
                    "http_status": status, "retrieval_utc": retrieval, "cache": "downloaded"}
        except AcquisitionError:
            tmp.unlink(missing_ok=True)
            raise
        except (requests.RequestException, OSError) as e:
            tmp.unlink(missing_ok=True)
            last_err = e
            if attempt < MAX_ATTEMPTS:
                time.sleep(2 ** attempt)
    raise AcquisitionError(f"Download failed after {MAX_ATTEMPTS} attempts: {url} ({last_err})")


class Manifest:
    """data/source_manifest.json: one record per downloaded file, keyed by source_id."""

    def __init__(self, path: Path):
        self.path = path
        self.records: dict[str, dict] = {}
        if path.exists():
            data = json.loads(path.read_text())
            self.records = {r["source_id"]: r for r in data.get("sources", [])}

    def upsert(self, record: dict) -> None:
        old = self.records.get(record["source_id"], {})
        merged = {**old, **{k: v for k, v in record.items() if v is not None or k not in old}}
        # Keep the original retrieval time/status when a cached file is merely reused.
        if record.get("cache") == "reused" and old.get("sha256") == record.get("sha256"):
            for k in ("retrieval_utc", "http_status"):
                if old.get(k) is not None:
                    merged[k] = old[k]
        self.records[record["source_id"]] = merged

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        out = {"sources": [self.records[k] for k in sorted(self.records)]}
        self.path.write_text(json.dumps(out, indent=2, sort_keys=True, default=str) + "\n")
