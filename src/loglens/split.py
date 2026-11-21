"""Chronological train/validation/test split by session start.

Sessions are ordered by their first event, using (ts, line_id) so events in the
same second keep file order. The first 60% are train, the next 10% validation
and the last 30% test. The first event of the first validation session is the
training cutoff: anything fitted on "training data" (Drain3 templates,
scalers, thresholds) may only see lines before it.

What a session is depends on the log:
  block  HDFS: every block ID a line names (labels from LogHub's file)
  time   fixed, non-overlapping time windows (a window is anomalous if any
         line in it carries an alert tag)
  count  fixed, non-overlapping windows of N consecutive lines

For a log with several windowing schemes (BGL), each scheme gets its own
split and its own parser fitted on exactly that scheme's training period:
60% of 1-hour windows and 60% of 20-line windows end months apart, because the
log's volume varies so much from hour to hour.

Usage: python -m loglens.split hdfs | bgl_1h | bgl_20ev
"""

from __future__ import annotations

import json
import sys

from pyspark.sql import Column, DataFrame, Window, functions as F

from loglens import config, runlog
from loglens.masking import block_ids
from loglens.spark import get_spark

FRACTIONS = {"train": 0.6, "val": 0.1, "test": 0.3}


def load_cutoffs(dataset: str) -> dict:
    return json.loads((config.RUNLOG / f"{dataset}_split.json").read_text())["cutoffs"]


def before(cutoff: dict) -> Column:
    """True for lines strictly before a cutoff, ordered by (ts, line_id)."""
    ts = F.lit(cutoff["ts"]).cast("timestamp")
    return (F.col("ts") < ts) | ((F.col("ts") == ts) & (F.col("line_id") < F.lit(cutoff["line_id"])))


def period(cutoffs: dict) -> Column:
    """Which period of the chronological split a line falls in."""
    return (
        F.when(before(cutoffs["val_start"]), "train")
        .when(before(cutoffs["test_start"]), "val")
        .otherwise("test")
    )


def assign_sessions(lines: DataFrame, name: str) -> DataFrame:
    """Add the session column for `name` (one row per line, or per line and block for HDFS)."""
    cfg = config.settings(name)
    col = cfg["session_col"]
    kind = cfg["session"]["kind"]
    if kind == "block":
        return lines.withColumn(col, F.explode(block_ids("message")))
    if kind == "time":
        t0 = lines.agg(F.min(F.col("ts").cast("double"))).first()[0]
        width = float(cfg["session"]["seconds"])
        return lines.withColumn(col, F.floor((F.col("ts").cast("double") - F.lit(t0)) / width).cast("long"))
    if kind == "count":
        # One global ordering of 4.7M lines; Spark runs it in a single partition, which is fine at this size.
        rank = F.row_number().over(Window.orderBy("ts", "line_id")) - 1
        return lines.withColumn(col, F.floor(rank / F.lit(cfg["session"]["size"])).cast("long"))
    raise ValueError(kind)


def session_labels(spark, name: str, events: DataFrame) -> DataFrame:
    cfg = config.settings(name)
    col = cfg["session_col"]
    if cfg["session"]["kind"] == "block":
        return spark.read.csv(str(cfg["labels"]), header=True).select(
            F.col("BlockId").alias(col), (F.col("Label") == "Anomaly").cast("int").alias("label")
        )
    return events.groupBy(col).agg(F.max((F.col("alert") != "-").cast("int")).alias("label"))


def run(dataset: str) -> None:
    cfg = config.settings(dataset)
    col = cfg["session_col"]
    spark = get_spark(f"loglens-split-{dataset}")
    timings: dict = {}

    with runlog.timer(timings, "total_s"):
        bronze = spark.read.parquet(str(config.BRONZE / config.parent(dataset)))
        cols = ["line_id", "ts", "message"] + (["alert"] if "alert" in cfg["fields"] else [])
        events = assign_sessions(bronze.select(*cols), dataset).cache()
        first = events.groupBy(col).agg(
            F.min(F.struct("ts", "line_id")).alias("f"), F.count("*").alias("n_events")
        )
        first = first.select(
            col,
            F.col("f.ts").alias("first_ts"),
            F.col("f.line_id").alias("first_line_id"),
            "n_events",
        )
        n = first.count()
        n_train = int(n * FRACTIONS["train"])
        n_val = int(n * FRACTIONS["val"])

        order = Window.orderBy("first_ts", "first_line_id")
        ranked = first.withColumn("order", F.row_number().over(order))
        ranked = ranked.withColumn(
            "split",
            F.when(F.col("order") <= n_train, "train")
            .when(F.col("order") <= n_train + n_val, "val")
            .otherwise("test"),
        )

        joined = ranked.join(session_labels(spark, dataset, events), col, "full_outer")
        out = config.GOLD / f"{dataset}_split"
        joined.write.mode("overwrite").parquet(str(out))

        split = spark.read.parquet(str(out))
        missing_label = split.filter(F.col("label").isNull()).count()
        missing_block = split.filter(F.col("split").isNull()).count()
        per_split = {
            r["split"]: {
                "sessions": r["sessions"],
                "anomalies": r["anomalies"],
                "anomaly_rate": round(r["anomalies"] / r["sessions"], 5),
                "first_start": r["first_start"].isoformat(),
                "last_start": r["last_start"].isoformat(),
                "mean_events": round(r["mean_events"], 1),
            }
            for r in split.filter(F.col("split").isNotNull())
            .groupBy("split")
            .agg(
                F.count("*").alias("sessions"),
                F.sum("label").alias("anomalies"),
                F.min("first_ts").alias("first_start"),
                F.max("first_ts").alias("last_start"),
                F.mean("n_events").alias("mean_events"),
            )
            .collect()
        }

        def boundary(at: int) -> dict:
            r = split.filter(F.col("order") == at).first()
            return {"ts": r["first_ts"].isoformat(), "line_id": r["first_line_id"]}

        cutoffs = {"val_start": boundary(n_train + 1), "test_start": boundary(n_train + n_val + 1)}
        lines_in_sessions = split.agg(F.sum("n_events")).first()[0]
        bronze_lines = bronze.count()

    checks = {
        "every_session_labelled": missing_label == 0,
        "every_label_has_session": missing_block == 0,
    }
    if dataset == "hdfs":
        checks["session_count_matches"] = n == 575_061
    if cfg["session"]["kind"] != "block":
        # Windows partition the log: every bronze line lands in exactly one window.
        checks["every_line_in_one_window"] = lines_in_sessions == bronze_lines
    runlog.write(
        dataset,
        "split",
        {
            "sessions": n,
            "session": cfg["session"],
            "fractions": FRACTIONS,
            "per_split": per_split,
            "cutoffs": cutoffs,
            "sessions_without_label": missing_label,
            "labels_without_session": missing_block,
            "lines_in_sessions": lines_in_sessions,
            "checks": checks,
            "timings": timings,
        },
    )
    spark.stop()
    if not all(checks.values()):
        raise SystemExit(f"split checks failed: {checks}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "hdfs")
