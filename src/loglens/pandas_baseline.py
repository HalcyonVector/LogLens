"""The HDFS chain in plain pandas, one process, for comparison with Spark.

Same work as bronze -> silver -> gold, with the templates already learned:
split each line with the same regex, mask it with the same rules, tag it with
its learned template (a lookup on the masked text, as Spark's dictionary
matcher does), explode block IDs, then group into per-block event sequences
and count vectors. The file is read in chunks, because 11M lines of Python
strings do not fit comfortably in memory at once.

Usage: python -m loglens.pandas_baseline hdfs [fraction]   (fraction of the file, default 1.0)
"""

from __future__ import annotations

import json
import os
import re
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from loglens import config
from loglens.masking import BLOCK_RE, masks_for
from loglens.parse import UNK, load_miner

CHUNK = 1_000_000


def run(dataset: str, fraction: float = 1.0) -> dict:
    cfg = config.settings(dataset)
    templates = json.loads((config.SILVER / f"{dataset}_templates.json").read_text())
    ids = {t["cluster_id"]: t["event_id"] for t in templates}
    miner = load_miner((config.SILVER / f"{dataset}_drain.pkl").read_bytes(), cfg["drain_sim_th"], dataset)
    masks = [(f"<{n}>", re.compile(p)) for n, p in masks_for(dataset)]
    line_rx = re.compile(cfg["regex"])
    block_rx = re.compile(BLOCK_RE)
    total = int(cfg["expected_lines"] * fraction)
    # Read the same input file the Spark scale run used, from the same storage.
    src = Path(os.environ.get("LOGLENS_SCALE", "/scale")) / "inputs" / f"{dataset}_{int(round(fraction * 100))}pct.log"
    if not src.exists():
        src = cfg["raw"]

    timings = {"read_and_split_s": 0.0, "mask_s": 0.0, "tag_s": 0.0, "explode_s": 0.0, "sessionize_s": 0.0}
    lookup: dict[str, str] = {}
    parts = []
    t_all = time.perf_counter()
    with open(src, encoding="utf-8", errors="replace") as f:
        line_no = 0
        while line_no < total:
            t = time.perf_counter()
            lines = []
            for _ in range(min(CHUNK, total - line_no)):
                ln = f.readline()
                if not ln:
                    break
                lines.append(ln.rstrip("\n"))
            if not lines:
                break
            df = pd.Series(lines).str.extract(line_rx)
            df.columns = cfg["fields"]
            df["ts"] = pd.to_datetime(df["date"] + df["time"], format="%y%m%d%H%M%S")
            df["line_id"] = np.arange(line_no, line_no + len(df))
            line_no += len(df)
            timings["read_and_split_s"] += time.perf_counter() - t

            t = time.perf_counter()
            masked = df["message"]
            for token, rx in masks:
                masked = masked.str.replace(rx, token, regex=True)
            masked = masked.str.split().str.join(" ")
            timings["mask_s"] += time.perf_counter() - t

            t = time.perf_counter()
            for m in masked.unique():
                if m not in lookup:
                    c = miner.match(m, full_search_strategy="fallback")
                    lookup[m] = ids.get(c.cluster_id, UNK) if c is not None else UNK
            df["event_id"] = masked.map(lookup)
            timings["tag_s"] += time.perf_counter() - t

            t = time.perf_counter()
            df["block_id"] = df["message"].map(lambda s: sorted(set(block_rx.findall(s))))
            df = df[["line_id", "ts", "event_id", "block_id"]].explode("block_id")
            parts.append(df)
            timings["explode_s"] += time.perf_counter() - t

    t = time.perf_counter()
    events = pd.concat(parts, ignore_index=True).sort_values(["block_id", "ts", "line_id"])
    vocab = [t["event_id"] for t in templates] + [UNK]
    sessions = events.groupby("block_id", sort=False)["event_id"].agg(list)
    counts = pd.crosstab(events["block_id"], events["event_id"]).reindex(columns=vocab, fill_value=0)
    timings["sessionize_s"] = time.perf_counter() - t
    timings = {k: round(v, 1) for k, v in timings.items()}
    timings["total_s"] = round(time.perf_counter() - t_all, 1)
    peak_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    out = {
        "engine": "pandas, one process, 1M-line chunks",
        "input": str(src),
        "dataset": dataset,
        "fraction": fraction,
        "lines": line_no,
        "sessions": int(len(sessions)),
        "events": int(len(events)),
        "count_matrix": list(counts.shape),
        "distinct_masked": len(lookup),
        "timings": timings,
        "lines_per_s": round(line_no / timings["total_s"]),
        "peak_memory_mb": peak_mb,
    }
    print(json.dumps(out, indent=2), flush=True)
    path = config.ROOT / "data" / "dashboard" / f"{dataset}_pandas.json"
    previous = json.loads(path.read_text())["runs"] if path.exists() else []
    runs = [r for r in previous if r["fraction"] != fraction] + [out]
    path.write_text(json.dumps({"runs": sorted(runs, key=lambda r: r["fraction"])}, indent=2))
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run(args[0] if args else "hdfs", float(args[1]) if len(args) > 1 else 1.0)
