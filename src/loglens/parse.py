"""Silver layer: learn log templates on the training period, then tag every line.

Fitting: Drain3 only sees lines before the training cutoff. Lines are masked
in Spark first and Drain3 is fed each distinct masked message once, in order
of first appearance, so it covers every training line without a Python loop
over millions of rows.

Matching every line, three ways (benchmarked with --benchmark):
  udf   Drain3's own match() inside a pandas UDF, miner state broadcast
  regex each template compiled to a Java regex, matched natively in Spark SQL
  dict  distinct masked messages matched once on the driver, then a broadcast join
Lines that match no template get event ID "UNK".

Usage: python -m loglens.parse hdfs [--benchmark]
"""

from __future__ import annotations

import json
import pickle
import sys
import time

import pandas as pd
from drain3 import TemplateMiner
from drain3.masking import MaskingInstruction
from drain3.template_miner_config import TemplateMinerConfig
from pyspark.sql import Column, DataFrame, SparkSession, functions as F

from loglens import config, runlog
from loglens.masking import block_ids, mask_col, masks_for
from loglens.spark import get_spark
from loglens.split import before, load_cutoffs, period

UNK = "UNK"
METHODS = ("udf", "regex", "dict")
DEFAULT_METHOD = "dict"


# ---------------------------------------------------------------- Drain3 ----


def miner_config(sim_th: float = 0.4, dataset: str = "hdfs") -> TemplateMinerConfig:
    cfg = TemplateMinerConfig()
    cfg.profiling_enabled = False
    cfg.drain_sim_th = sim_th
    cfg.drain_depth = 4
    cfg.masking_instructions = [MaskingInstruction(p, name) for name, p in masks_for(dataset)]
    return cfg


def load_miner(drain_state: bytes, sim_th: float, dataset: str = "hdfs") -> TemplateMiner:
    miner = TemplateMiner(config=miner_config(sim_th, dataset))
    miner.drain = pickle.loads(drain_state)
    return miner


def train_period(bronze: DataFrame, cutoff: dict) -> DataFrame:
    """Lines strictly before the first validation session's first event."""
    return bronze.filter(before(cutoff))


def fit(bronze: DataFrame, cutoff: dict, sim_th: float, dataset: str = "hdfs") -> tuple[TemplateMiner, list[dict], dict]:
    train = train_period(bronze, cutoff).select("line_id", mask_col("message", dataset).alias("masked"))
    distinct = (
        train.groupBy("masked")
        .agg(F.min("line_id").alias("first"), F.count("*").alias("n"))
        .orderBy("first")
        .collect()
    )
    miner = TemplateMiner(config=miner_config(sim_th, dataset))
    for row in distinct:
        miner.add_log_message(row["masked"])

    # Count training lines per final cluster (clusters can generalise after
    # a message was added, so match again against the finished tree).
    counts: dict[int, int] = {}
    for row in distinct:
        c = miner.match(row["masked"], full_search_strategy="fallback")
        if c is not None:
            counts[c.cluster_id] = counts.get(c.cluster_id, 0) + row["n"]

    clusters = sorted(miner.drain.clusters, key=lambda c: -counts.get(c.cluster_id, 0))
    templates = []
    width = max(2, len(str(len(clusters))))
    for rank, c in enumerate(clusters, start=1):
        templates.append(
            {
                "event_id": f"T{rank:0{width}d}",
                "cluster_id": c.cluster_id,
                "template": c.get_template(),
                "train_lines": counts.get(c.cluster_id, 0),
            }
        )
    stats = {
        "train_lines": sum(r["n"] for r in distinct),
        "train_distinct_masked": len(distinct),
        "templates": len(templates),
        "templates_with_train_lines": sum(1 for t in templates if t["train_lines"] > 0),
    }
    return miner, templates, stats


# -------------------------------------------------------------- matchers ----


def template_regex(template: str) -> str:
    """Drain3 templates are whitespace-separated tokens; <*> is one token."""
    parts = [r"\S+" if tok == "<*>" else r"\Q" + tok + r"\E" for tok in template.split()]
    return "^" + r"\s".join(parts) + "$"


def match_regex(df: DataFrame, templates: list[dict], dataset: str = "hdfs") -> Column:
    # Fewest wildcards first, so the most specific template wins a tie.
    ordered = sorted(templates, key=lambda t: (t["template"].count("<*>"), t["event_id"]))
    masked = mask_col("message", dataset)
    expr = None
    for t in ordered:
        cond = masked.rlike(template_regex(t["template"]))
        expr = F.when(cond, t["event_id"]) if expr is None else expr.when(cond, t["event_id"])
    return expr.otherwise(UNK)


_MINER_CACHE: dict = {}


def match_udf(spark: SparkSession, miner: TemplateMiner, templates: list[dict], dataset: str = "hdfs"):
    state = spark.sparkContext.broadcast(pickle.dumps(miner.drain))
    sim_th = miner.drain.sim_th
    ids = {t["cluster_id"]: t["event_id"] for t in templates}

    @F.pandas_udf("string")
    def tag(messages: pd.Series) -> pd.Series:
        m = _MINER_CACHE.get("miner")
        if m is None:
            m = _MINER_CACHE["miner"] = load_miner(state.value, sim_th, dataset)
        out = []
        for msg in messages:
            c = m.match(msg, full_search_strategy="fallback")
            out.append(ids.get(c.cluster_id, UNK) if c is not None else UNK)
        return pd.Series(out)

    return tag("message")


def match_dict(
    spark: SparkSession, bronze: DataFrame, miner: TemplateMiner, templates: list[dict], dataset: str = "hdfs"
) -> tuple[DataFrame, int]:
    ids = {t["cluster_id"]: t["event_id"] for t in templates}
    with_masked = bronze.withColumn("masked", mask_col("message", dataset))
    distinct = [r["masked"] for r in with_masked.select("masked").distinct().collect()]
    rows = []
    for msg in distinct:
        c = miner.match(msg, full_search_strategy="fallback")
        rows.append((msg, ids.get(c.cluster_id, UNK) if c is not None else UNK))
    lookup = spark.createDataFrame(rows, "masked string, event_id string")
    tagged = with_masked.join(F.broadcast(lookup), "masked", "left").drop("masked")
    return tagged.withColumn("event_id", F.coalesce("event_id", F.lit(UNK))), len(distinct)


def tag_lines(spark, bronze, miner, templates, method: str, dataset: str = "hdfs") -> DataFrame:
    if method == "udf":
        return bronze.withColumn("event_id", match_udf(spark, miner, templates, dataset))
    if method == "regex":
        return bronze.withColumn("event_id", match_regex(bronze, templates, dataset))
    if method == "dict":
        return match_dict(spark, bronze, miner, templates, dataset)[0]
    raise ValueError(method)


# ------------------------------------------------------------------- run ----


def benchmark(spark, bronze, miner, templates, n_lines: int, dataset: str = "hdfs") -> dict:
    """Time each matcher end to end (read + mask + match) with a no-op sink."""
    bronze.select("line_id").write.format("noop").mode("overwrite").save()  # warm the file cache
    results = {}
    for method in METHODS:
        start = time.perf_counter()
        tag_lines(spark, bronze, miner, templates, method, dataset).select(
            "line_id", "event_id"
        ).write.format("noop").mode("overwrite").save()
        secs = time.perf_counter() - start
        results[method] = {"seconds": round(secs, 1), "lines_per_s": round(n_lines / secs)}
        print(f"benchmark {method}: {secs:.1f}s", flush=True)

    tagged = {m: tag_lines(spark, bronze, miner, templates, m, dataset).select("line_id", F.col("event_id").alias(m)) for m in METHODS}
    both = tagged["dict"].join(tagged["regex"], "line_id").join(tagged["udf"], "line_id")
    agree = both.agg(
        F.sum((F.col("dict") != F.col("udf")).cast("int")).alias("dict_vs_udf"),
        F.sum((F.col("dict") != F.col("regex")).cast("int")).alias("dict_vs_regex"),
        F.sum((F.col("udf") != F.col("regex")).cast("int")).alias("udf_vs_regex"),
    ).first().asDict()
    results["disagreements"] = agree
    return results


def run(dataset: str, do_benchmark: bool) -> None:
    spark = get_spark(f"loglens-parse-{dataset}")
    timings: dict = {}
    cutoffs = load_cutoffs(dataset)
    cfg = config.settings(dataset)
    bronze = spark.read.parquet(str(config.BRONZE / config.parent(dataset))).drop(cfg["partition"])
    n_lines = bronze.count()

    with runlog.timer(timings, "fit_s"):
        sim_th = cfg["drain_sim_th"]
        # Normally templates are learned on the training period only; "all" fits them on the
        # whole log, the benchmark habit this project measures rather than follows.
        fit_cut = {"ts": "9999-12-31T00:00:00", "line_id": 0} if cfg.get("parser_fit") == "all" else cutoffs["val_start"]
        miner, templates, fit_stats = fit(bronze, fit_cut, sim_th, dataset)
        fit_stats["parser_fit"] = cfg.get("parser_fit", "train")
        fit_stats["drain_sim_th"] = sim_th
    print(f"fitted {fit_stats['templates']} templates from {fit_stats['train_lines']:,} training lines", flush=True)

    config.SILVER.mkdir(parents=True, exist_ok=True)
    (config.SILVER / f"{dataset}_drain.pkl").write_bytes(pickle.dumps(miner.drain))
    (config.SILVER / f"{dataset}_templates.json").write_text(json.dumps(templates, indent=2))

    bench = None
    if do_benchmark:
        with runlog.timer(timings, "benchmark_s"):
            bench = benchmark(spark, bronze, miner, templates, n_lines, dataset)
        method = min(METHODS, key=lambda m: bench[m]["seconds"])
    else:
        method = DEFAULT_METHOD

    with runlog.timer(timings, "write_s"):
        tagged = tag_lines(spark, bronze, miner, templates, method, dataset)
        base_cols = ["line_id", "ts", "level", "component", "event_id"]
        if cfg.get("session", {}).get("kind") == "block":
            # HDFS: one row per (line, block it names); a line repeating its block counts once.
            events = tagged.select(*base_cols, F.explode(block_ids("message")).alias("block_id"))
        else:
            # Windowed logs: one row per line; windows are cut in the gold stage.
            events = tagged.select(*base_cols, *[c for c in cfg.get("keep", []) if c != "pid"])
        out = config.SILVER / f"{dataset}_events"
        events.write.mode("overwrite").parquet(str(out))

    silver = spark.read.parquet(str(out))
    lines = silver.dropDuplicates(["line_id"])
    unk = (F.col("event_id") == UNK).cast("int")

    by_period = {
        r["period"]: {"lines": r["lines"], "unk": r["unk"], "unk_rate": round(r["unk"] / r["lines"], 6)}
        for r in lines.groupBy(period(cutoffs).alias("period")).agg(F.count("*").alias("lines"), F.sum(unk).alias("unk")).collect()
    }
    per_hour = [
        r.asDict()
        for r in lines.groupBy(F.date_format("ts", "yyyy-MM-dd-HH").alias("hour"))
        .agg(F.count("*").alias("lines"), F.sum(unk).alias("unk"))
        .orderBy("hour")
        .collect()
    ]
    per_template = {
        r["event_id"]: r["lines"] for r in lines.groupBy("event_id").agg(F.count("*").alias("lines")).collect()
    }
    silver_rows = silver.count()
    for t in templates:
        t["all_lines"] = per_template.get(t["event_id"], 0)

    checks = {
        "every_line_tagged": lines.count() == n_lines,
        "train_unk_is_zero": by_period.get("train", {}).get("unk", 1) == 0,
    }
    runlog.write(
        dataset,
        "parse",
        {
            "method": method,
            "fit": fit_stats,
            "lines": n_lines,
            "silver_rows": silver_rows,
            "unk_by_period": by_period,
            "benchmark": bench,
            "checks": checks,
            "timings": timings,
        },
    )
    config.DASHBOARD.mkdir(parents=True, exist_ok=True)
    (config.DASHBOARD / f"{dataset}_parsing.json").write_text(
        json.dumps(
            {
                "templates": templates,
                "unk_by_period": by_period,
                "unk_per_hour": per_hour,
                "benchmark": bench,
                "method": method,
                "fit": fit_stats,
            },
            indent=2,
        )
    )
    spark.stop()
    if not all(checks.values()):
        raise SystemExit(f"parse checks failed: {checks}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run(args[0] if args else "hdfs", "--benchmark" in sys.argv)
