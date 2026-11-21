"""Bronze layer: raw text -> structured Parquet, with a quarantine table for unparsed lines.

Usage: python -m loglens.ingest hdfs
"""

from __future__ import annotations

import sys

from pyspark.sql import DataFrame, functions as F

from loglens import config, runlog
from loglens.spark import get_spark


def parse_lines(raw: DataFrame, cfg: dict) -> DataFrame:
    """Split each raw line into fields. `parsed` is False when the regex does not match."""
    pattern = cfg["regex"]
    cols = [
        F.regexp_extract("value", pattern, i + 1).alias(name)
        for i, name in enumerate(cfg["fields"])
    ]
    df = raw.select("line_id", "value", *cols).withColumn(
        "parsed", F.col("value").rlike(pattern)
    )
    ts = F.to_timestamp(F.concat(*cfg["ts"]["cols"]), cfg["ts"]["format"])
    part = {"hour": "yyyy-MM-dd-HH", "day": "yyyy-MM-dd"}[cfg["partition"]]
    df = df.withColumn("ts", ts).withColumn(cfg["partition"], F.date_format("ts", part))
    if "pid" in cfg["fields"]:
        df = df.withColumn("pid", F.col("pid").cast("int"))
    return df


def run(dataset: str) -> None:
    cfg = config.DATASETS[dataset]
    spark = get_spark(f"loglens-ingest-{dataset}")
    timings: dict = {}

    out = config.BRONZE / dataset
    quarantine = config.BRONZE / f"{dataset}_quarantine"

    with runlog.timer(timings, "total_s"):
        # line_id keeps the original file order, which breaks ties between
        # events that share a timestamp (HDFS timestamps are per second).
        # It increases with file order but is not contiguous across partitions.
        raw = spark.read.text(str(cfg["raw"])).withColumn(
            "line_id", F.monotonically_increasing_id()
        )
        parsed = parse_lines(raw, cfg).cache()

        good = parsed.filter(F.col("parsed") & F.col("ts").isNotNull())
        bad = parsed.filter(~(F.col("parsed") & F.col("ts").isNotNull()))

        keep = ["line_id", "ts", "level", "component", "message", *cfg.get("keep", []), cfg["partition"]]
        (
            good.select(*keep)
            .repartition(cfg["partition"])
            .sortWithinPartitions("line_id")
            .write.mode("overwrite")
            .partitionBy(cfg["partition"])
            .parquet(str(out))
        )
        bad.select("line_id", "value").write.mode("overwrite").parquet(str(quarantine))

        raw_n = parsed.count()
        bronze_n = spark.read.parquet(str(out)).count()
        quarantine_n = spark.read.parquet(str(quarantine)).count()
        span = good.agg(F.min("ts").alias("first"), F.max("ts").alias("last")).first()
        alerts = None
        if "alert" in cfg["fields"]:
            alerts = spark.read.parquet(str(out)).filter(F.col("alert") != "-").count()

    checks = {
        "raw_matches_expected": raw_n == cfg["expected_lines"],
        "reconciles": raw_n == bronze_n + quarantine_n,
    }
    if "expected_alerts" in cfg:
        checks["alerts_match_expected"] = alerts == cfg["expected_alerts"]
    runlog.write(
        dataset,
        "bronze",
        {
            "raw_lines": raw_n,
            "expected_lines": cfg["expected_lines"],
            "bronze_rows": bronze_n,
            "quarantine_rows": quarantine_n,
            "quarantine_rate": round(quarantine_n / raw_n, 6) if raw_n else None,
            "alert_lines": alerts,
            "first_ts": span["first"].isoformat() if span["first"] else None,
            "last_ts": span["last"].isoformat() if span["last"] else None,
            "span_hours": round((span["last"] - span["first"]).total_seconds() / 3600, 2)
            if span["first"]
            else None,
            "checks": checks,
            "timings": timings,
        },
    )
    spark.stop()
    if not all(checks.values()):
        raise SystemExit(f"bronze quality checks failed: {checks}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "hdfs")
