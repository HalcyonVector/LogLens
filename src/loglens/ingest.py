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
    ts = F.to_timestamp(F.concat("date", "time"), cfg["ts_format"])
    return (
        df.withColumn("ts", ts)
        .withColumn("pid", F.col("pid").cast("int"))
        .withColumn("hour", F.date_format("ts", "yyyy-MM-dd-HH"))
    )


def run(dataset: str) -> None:
    cfg = config.DATASETS[dataset]
    spark = get_spark(f"loglens-ingest-{dataset}")
    timings: dict = {}

    out = config.BRONZE / dataset
    quarantine = config.BRONZE / f"{dataset}_quarantine"

    with runlog.timer(timings, "total_s"):
        # line_id keeps the original file order, which breaks ties between
        # events that share a timestamp (HDFS timestamps are per second).
        raw = spark.read.text(str(cfg["raw"])).withColumn(
            "line_id", F.monotonically_increasing_id()
        )
        parsed = parse_lines(raw, cfg).cache()

        good = parsed.filter(F.col("parsed") & F.col("ts").isNotNull())
        bad = parsed.filter(~(F.col("parsed") & F.col("ts").isNotNull()))

        keep = ["line_id", "ts", "pid", "level", "component", "message", "hour"]
        (
            good.select(*keep)
            .repartition("hour")
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

    checks = {
        "raw_matches_expected": raw_n == cfg["expected_lines"],
        "reconciles": raw_n == bronze_n + quarantine_n,
    }
    runlog.write(
        dataset,
        "bronze",
        {
            "raw_lines": raw_n,
            "expected_lines": cfg["expected_lines"],
            "bronze_rows": bronze_n,
            "quarantine_rows": quarantine_n,
            "quarantine_rate": round(quarantine_n / raw_n, 6) if raw_n else None,
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
