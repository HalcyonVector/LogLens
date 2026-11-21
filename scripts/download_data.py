"""Download LogHub datasets from the official Zenodo record and verify checksums.

Usage: python scripts/download_data.py HDFS_v1 [BGL] [Thunderbird]
"""

from __future__ import annotations

import hashlib
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ZENODO = "https://zenodo.org/records/8196385/files/{name}?download=1"
# Loghub-2.0: full-size logs annotated with a ground-truth template per line.
LOGHUB2 = "https://zenodo.org/records/8275861/files/{name}?download=1"

DATASETS = {
    "HDFS_v1": {"file": "HDFS_v1.zip", "md5": "76a24b4d9a6164d543fb275f89773260"},
    "BGL": {"file": "BGL.zip", "md5": "4452953c470f2d95fcb32d5f6e733f7a"},
    "Thunderbird": {"file": "Thunderbird.tar.gz", "md5": "0891b048df2919dc78c99c4428686b44"},
    "BGL_loghub2": {"file": "BGL.zip", "md5": "521ee643375bb6e9d0dd88636ff7ece1", "source": LOGHUB2, "archive": "BGL_loghub2.zip"},
}

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "loglens-downloader"})
    with urllib.request.urlopen(req) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        last = -1
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            pct = int(done * 100 / total) if total else 0
            if pct // 10 != last:
                last = pct // 10
                print(f"  {dest.name}: {done / 1e6:,.0f} MB ({pct}%)", flush=True)
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
        download(spec.get("source", ZENODO).format(name=spec["file"]), archive)
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
