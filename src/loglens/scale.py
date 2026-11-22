"""Scale benchmark: the same Spark chain on growing inputs, and where the time goes.

Chain, each stage written to Parquet so every stage is really computed:
  bronze    read text, split each line with the dataset's regex, parse timestamps
  silver    mask each line and tag it with a template: distinct masked lines are
            matched once with Drain3, in parallel across the cores (the template
            tree is broadcast), then joined back to every line
  gold      HDFS: one row per block (event sequence and count map);
            Thunderbird: one row per hour (alert label and a sparse count map)

For Thunderbird the templates are learned inside the run (Drain3 on the first
60% of the input's lines); for HDFS the existing templates are reused, so the
timing matches the pandas comparison, which also reuses them.

Inputs are prefixes of the raw log (the first 1%, 10% and 100% of its lines),
copied to the scale volume once. Everything runs from /scale, a Docker named
volume (native Linux storage), unless --bind is given, which reads and writes
through the Windows bind mount instead, to measure what that costs.

Usage: python -m loglens.scale hdfs|thunderbird [--fractions 0.01,0.1,1] [--bind]
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
from drain3 import TemplateMiner
from pyspark.sql import functions as F

from loglens import config
from loglens.masking import block_ids, mask_col
from loglens.parse import UNK, load_miner, miner_config
from loglens.spark import get_spark

SCALE = Path(os.environ.get("LOGLENS_SCALE", "/scale"))
EXPORT = config.ROOT / "data" / "dashboard"


def du(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def prefix_file(src: Path, dst: Path, n_lines: int) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        for i, line in enumerate(fi):
            if i >= n_lines:
                break
            fo.write(line)
    tmp.replace(dst)


def timed(timings: dict, key: str, fn):
    t = time.perf_counter()
    out = fn()
    timings[key] = round(time.perf_counter() - t, 1)
    return out


def run_one(dataset: str, path: Path, work: Path, n_expected: int) -> dict:
    cfg = config.settings(dataset)
    spark = get_spark(f"loglens-scale-{dataset}")
    if work.exists():
        shutil.rmtree(work)
    timings: dict = {}

    # ---- bronze
    def bronze():
        raw = spark.read.text(str(path)).withColumn("line_id", F.monotonically_increasing_id())
        pattern = cfg["regex"]
        cols = [F.regexp_extract("value", pattern, i + 1).alias(n) for i, n in enumerate(cfg["fields"])]
        if cfg["ts"]["format"] == "unix":
            ts = F.col(cfg["ts"]["cols"][0]).cast("long").cast("timestamp")
        else:
            ts = F.to_timestamp(F.concat(*cfg["ts"]["cols"]), cfg["ts"]["format"])
        df = raw.select("line_id", *cols).withColumn("ts", ts)
        keep = ["line_id", "ts", "message"] + [c for c in ("alert",) if c in cfg["fields"]]
        df.select(*keep).write.parquet(str(work / "bronze"))
        return spark.read.parquet(str(work / "bronze"))

    b = timed(timings, "bronze_s", bronze)
    lines = b.count()

    # ---- templates
    def learn():
        if cfg.get("scale_fit", False):
            cut = b.approxQuantile("line_id", [0.6], 0.001)[0]
            train = b.filter(F.col("line_id") < cut).select("line_id", mask_col("message", dataset).alias("m"))
            distinct = train.groupBy("m").agg(F.min("line_id").alias("first")).orderBy("first").collect()
            miner = TemplateMiner(config=miner_config(cfg["drain_sim_th"], dataset))
            for r in distinct:
                miner.add_log_message(r["m"])
            return miner, {c.cluster_id: f"T{i:04d}" for i, c in enumerate(miner.drain.clusters, start=1)}, len(distinct)
        templates = json.loads((config.SILVER / f"{dataset}_templates.json").read_text())
        miner = load_miner((config.SILVER / f"{dataset}_drain.pkl").read_bytes(), cfg["drain_sim_th"], dataset)
        return miner, {t["cluster_id"]: t["event_id"] for t in templates}, None

    miner, ids, train_distinct = timed(timings, "fit_s", learn)

    # ---- silver
    def silver():
        # Match each distinct masked message once, in parallel on every core: the template
        # tree is broadcast and each worker matches its share of the distinct messages.
        # (A driver-side loop was fine for HDFS's 38 distinct messages and took 71% of the
        # run at 10% of Thunderbird, with 209K of them.)
        masked = b.withColumn("m", mask_col("message", dataset))
        distinct = masked.select("m").distinct()
        state = spark.sparkContext.broadcast(pickle.dumps(miner.drain))
        id_map = spark.sparkContext.broadcast(ids)
        sim_th, ds = cfg["drain_sim_th"], dataset
        # "fallback" scans every template of the same length when the tree lookup misses:
        # cheap with HDFS's 50 templates, slow with Thunderbird's thousands.
        strategy = os.environ.get("LOGLENS_MATCH", "fallback")

        def match(batches):
            m = load_miner(state.value, sim_th, ds)
            for pdf in batches:
                out = []
                for msg in pdf["m"]:
                    c = m.match(msg, full_search_strategy=strategy)
                    out.append(id_map.value.get(c.cluster_id, UNK) if c is not None else UNK)
                yield pd.DataFrame({"m": pdf["m"], "event_id": out})

        # The distinct messages are small (tens of MB), so adaptive execution would merge them
        # into one or two partitions and leave one Python worker doing all the matching;
        # an explicit repartition spreads them across every core.
        parts = 4 * (os.cpu_count() or 4)
        lookup = distinct.repartition(parts).mapInPandas(match, "m string, event_id string")
        tagged = masked.join(lookup, "m", "left").drop("m")
        tagged.write.parquet(str(work / "silver"))
        return spark.read.parquet(str(work / "silver")), distinct.count()

    s, n_distinct = timed(timings, "silver_s", silver)

    # ---- gold
    def gold():
        if cfg["session"]["kind"] == "block":
            ev = s.select("line_id", "ts", "event_id", F.explode(block_ids("message")).alias("key"))
            g = ev.groupBy("key").agg(
                F.sort_array(F.collect_list(F.struct("ts", "line_id", "event_id"))).alias("seq"),
                F.count("*").alias("n"),
            ).select("key", F.col("seq.event_id").alias("events"), "n")
        else:
            ev = s.withColumn("key", F.date_format("ts", "yyyy-MM-dd HH"))
            g = ev.groupBy("key").agg(
                F.count("*").alias("n"),
                F.max((F.col("alert") != "-").cast("int")).alias("label"),
                F.sum((F.col("alert") != "-").cast("int")).alias("alert_lines"),
            )
        counts = ev.groupBy("key", "event_id").count().groupBy("key").agg(
            F.map_from_entries(F.collect_list(F.struct("event_id", "count"))).alias("counts")
        )
        g.join(counts, "key").write.parquet(str(work / "gold"))
        return spark.read.parquet(str(work / "gold"))

    g = timed(timings, "gold_s", gold)
    sessions = g.count()
    extra = {}
    if "alert" in cfg["fields"]:
        extra = g.agg(F.sum("label").alias("anomalous_windows"), F.sum("alert_lines").alias("alert_lines")).first().asDict()
    unk = s.filter(F.col("event_id") == UNK).count()
    timings["total_s"] = round(sum(v for k, v in timings.items() if k.endswith("_s")), 1)
    out = {
        "lines": lines,
        "lines_expected": n_expected,
        "input_bytes": path.stat().st_size,
        "sessions": sessions,
        "distinct_masked": n_distinct,
        "train_distinct_masked": train_distinct,
        "templates": len(ids),
        "match_strategy": os.environ.get("LOGLENS_MATCH", "fallback"),
        "unk_lines": unk,
        "bytes": {k: du(work / k) for k in ("bronze", "silver", "gold")},
        "timings": timings,
        "lines_per_s": round(lines / timings["total_s"]),
        **extra,
    }
    spark.stop()
    shutil.rmtree(work, ignore_errors=True)
    return out


def run(dataset: str, fractions: list[float], bind: bool) -> None:
    cfg = config.settings(dataset)
    src = cfg["raw"]
    root = (config.ROOT / "data" / "scale_bind") if bind else SCALE
    results = []
    for f in fractions:
        n = int(cfg["expected_lines"] * f)
        tag = f"{dataset}_{int(round(f * 100))}pct"
        # Read from the storage being measured: copy (a prefix of) the log there, unless the
        # full log already lives there (Thunderbird's 30 GB is not copied twice).
        if f == 1.0 and (bind or str(src).startswith(str(root))):
            path = src
        else:
            path = root / "inputs" / f"{tag}.log"
            prefix_file(src, path, n)
        r = run_one(dataset, path, root / "work" / tag, n)
        r.update({"dataset": dataset, "fraction": f, "storage": "bind mount" if bind else "docker volume"})
        results.append(r)
        print(json.dumps(r), flush=True)
    name = f"{dataset}_scale{'_bind' if bind else ''}"
    path = EXPORT / f"{name}.json"
    previous = json.loads(path.read_text())["runs"] if path.exists() else []
    keep = [p for p in previous if p["fraction"] not in fractions]
    path.write_text(json.dumps({"runs": sorted(keep + results, key=lambda r: r["fraction"])}, indent=2))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    fr = [0.01, 0.1, 1.0]
    for a in sys.argv:
        if a.startswith("--fractions="):
            fr = [float(x) for x in a.split("=", 1)[1].split(",")]
    run(args[0] if args else "hdfs", fr, "--bind" in sys.argv)
