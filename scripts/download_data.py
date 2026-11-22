"""Download LogHub datasets from the official Zenodo record and verify checksums.

Usage: python scripts/download_data.py HDFS_v1 [BGL] [Thunderbird]
"""

from __future__ import annotations

import hashlib
import os
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

ZENODO = "https://zenodo.org/records/8196385/files/{name}?download=1"
# Loghub-2.0: full-size logs annotated with a ground-truth template per line.
LOGHUB2 = "https://zenodo.org/records/8275861/files/{name}?download=1"

DATASETS = {
    "HDFS_v1": {"file": "HDFS_v1.zip", "md5": "76a24b4d9a6164d543fb275f89773260"},
    "BGL": {"file": "BGL.zip", "md5": "4452953c470f2d95fcb32d5f6e733f7a"},
    "Thunderbird": {"file": "Thunderbird.tar.gz", "md5": "0891b048df2919dc78c99c4428686b44", "size": 2_016_100_298},
    "BGL_loghub2": {"file": "BGL.zip", "md5": "521ee643375bb6e9d0dd88636ff7ece1", "source": LOGHUB2, "archive": "BGL_loghub2.zip"},
}

# Same root as the pipeline: LOGLENS_DATA when set (the Thunderbird volume), else ./data.
RAW_DIR = Path(os.environ.get("LOGLENS_DATA", Path(__file__).resolve().parents[1] / "data")) / "raw"


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, size: int | None = None, attempts: int = 30) -> None:
    """Download to dest, resuming a partial .part file with HTTP Range requests.

    Large archives over a slow link can drop mid-transfer, and a dropped
    connection can end the stream without an error; so the transfer only
    counts as finished once the expected size is on disk.
    """
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, attempts + 1):
        have = tmp.stat().st_size if tmp.exists() else 0
        if size and have >= size:
            break
        headers = {"User-Agent": "loglens-downloader"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as resp:
                if have and resp.status != 206:  # server ignored the range: start over
                    have = 0
                    tmp.unlink(missing_ok=True)
                total = size or have + int(resp.headers.get("Content-Length", 0))
                last = -1
                with tmp.open("ab") as out:
                    while chunk := resp.read(1 << 20):
                        out.write(chunk)
                        have += len(chunk)
                        pct = int(have * 100 / total) if total else 0
                        if pct // 10 != last:
                            last = pct // 10
                            print(f"  {dest.name}: {have / 1e6:,.0f} MB ({pct}%)", flush=True)
            if not size or have >= size:
                break
            print(f"  {dest.name}: connection ended at {have / 1e6:,.0f} MB, resuming", flush=True)
        except OSError as e:
            print(f"  {dest.name}: attempt {attempt} failed ({e}), resuming", flush=True)
            time.sleep(min(60, 5 * attempt))
    else:
        raise SystemExit(f"{dest.name}: gave up after {attempts} attempts")
    tmp.replace(dest)


def extract(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(target)
    else:
        with tarfile.open(archive) as t:
            t.extractall(target, filter="data")


def fetch(name: str) -> None:
    spec = DATASETS[name]
    archive = RAW_DIR / spec.get("archive", spec["file"])
    target = RAW_DIR / name
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if archive.exists() and md5sum(archive) == spec["md5"]:
        print(f"{name}: archive present, checksum ok")
    else:
        print(f"{name}: downloading {spec['file']}")
        if archive.exists():  # a truncated earlier download: keep its bytes and resume
            archive.replace(archive.with_suffix(archive.suffix + ".part"))
        download(spec.get("source", ZENODO).format(name=spec["file"]), archive, spec.get("size"))
        got = md5sum(archive)
        if got != spec["md5"]:
            raise SystemExit(f"{name}: checksum mismatch ({got} != {spec['md5']})")
        print(f"{name}: checksum ok")

    if target.exists() and any(target.iterdir()):
        print(f"{name}: already extracted to {target}")
    else:
        print(f"{name}: extracting")
        extract(archive, target)
    print(f"{name}: done")


if __name__ == "__main__":
    names = sys.argv[1:] or ["HDFS_v1"]
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        raise SystemExit(f"unknown dataset(s): {unknown}; choose from {list(DATASETS)}")
    for n in names:
        fetch(n)
