"""Chronological train/validation/test split by session start.

Sessions (HDFS blocks) are ordered by their first event, using (ts, line_id) so
events in the same second keep file order. The first 60% are train, the next
10% validation and the last 30% test. The first event of the first validation
session is the training cutoff: anything fitted on "training data" (Drain3
templates, scalers, thresholds) may only see lines before it.

Usage: python -m loglens.split hdfs
"""

from __future__ import annotations

import json
import sys

from pyspark.sql import Column, Window, functions as F

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


def run(dataset: str) -> None:
    cfg = config.DATASETS[dataset]
    spark = get_spark(f"loglens-split-{dataset}")
    timings: dict = {}

    with runlog.timer(timings, "total_s"):
        bronze = spark.read.parquet(str(config.BRONZE / dataset))
        events = bronze.select("line_id", "ts", F.explode(block_ids("message")).alias("block_id"))
        first = events.groupBy("block_id").agg(
            F.min(F.struct("ts", "line_id")).alias("f"), F.count("*").alias("n_events")
        )
        first = first.select(
            "block_id",
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

        labels = (
            spark.read.csv(str(cfg["labels"]), header=True)
            .select(
                F.col("BlockId").alias("block_id"),
                (F.col("Label") == "Anomaly").cast("int").alias("label"),
            )
        )
        joined = ranked.join(labels, "block_id", "full_outer")
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
            }
            for r in split.filter(F.col("split").isNotNull())
            .groupBy("split")
            .agg(
                F.count("*").alias("sessions"),
                F.sum("label").alias("anomalies"),
                F.min("first_ts").alias("first_start"),
                F.max("first_ts").alias("last_start"),
            )
            .collect()
        }

        def boundary(at: int) -> dict:
            r = split.filter(F.col("order") == at).first()
            return {"ts": r["first_ts"].isoformat(), "line_id": r["first_line_id"]}

        cutoffs = {"val_start": boundary(n_train + 1), "test_start": boundary(n_train + n_val + 1)}

    checks = {
        "every_block_labelled": missing_label == 0,
        "every_label_has_block": missing_block == 0,
        "session_count_matches": n == 575_061 if dataset == "hdfs" else True,
    }
    runlog.write(
        dataset,
        "split",
        {
            "sessions": n,
            "fractions": FRACTIONS,
            "per_split": per_split,
            "cutoffs": cutoffs,
            "blocks_without_label": missing_label,
            "labels_without_block": missing_block,
            "checks": checks,
            "timings": timings,
        },
    )
    spark.stop()
    if not all(checks.values()):
        raise SystemExit(f"split checks failed: {checks}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "hdfs")
