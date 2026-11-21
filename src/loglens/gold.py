"""Gold layer: one row per session (HDFS block), ready for modelling.

Each session gets its event sequence (ordered by timestamp, then file
position), an event count vector over the learned vocabulary (templates in
training-frequency order, then UNK), simple session stats, hashes of the
sequence and of the count vector (for the duplicate analysis), and its
chronological split and label.

Usage: python -m loglens.gold hdfs
"""

from __future__ import annotations

import json
import sys

from pyspark.sql import functions as F

from loglens import config, runlog
from loglens.parse import UNK
from loglens.spark import get_spark


def vocabulary(dataset: str) -> list[str]:
    templates = json.loads((config.SILVER / f"{dataset}_templates.json").read_text())
    return [t["event_id"] for t in templates] + [UNK]


def run(dataset: str) -> None:
    spark = get_spark(f"loglens-gold-{dataset}")
    timings: dict = {}
    vocab = vocabulary(dataset)

    with runlog.timer(timings, "total_s"):
        silver = spark.read.parquet(str(config.SILVER / f"{dataset}_events"))
        sessions = silver.groupBy("block_id").agg(
            F.sort_array(F.collect_list(F.struct("ts", "line_id", "event_id"))).alias("s"),
            F.count("*").alias("n_events"),
            F.sum((F.col("level") == "WARN").cast("int")).alias("n_warn"),
            F.min("ts").alias("start_ts"),
            F.max("ts").alias("end_ts"),
        )
        sessions = sessions.select(
            "block_id",
            F.col("s.event_id").alias("events"),
            # Seconds from the session's first event to each event, for detection latency.
            F.transform(
                "s", lambda e: (e["ts"].cast("long") - F.col("start_ts").cast("long")).cast("int")
            ).alias("offsets_s"),
            "n_events",
            "n_warn",
            "start_ts",
            "end_ts",
            (F.col("end_ts").cast("long") - F.col("start_ts").cast("long")).alias("duration_s"),
        )
        # Spark reads a lambda's arity (x vs x, index), so bind v in a factory, not a default arg.
        def is_event(v: str):
            return lambda x: x == F.lit(v)

        counts = F.array(*[F.size(F.filter("events", is_event(v))) for v in vocab])
        sessions = (
            sessions.withColumn("counts", counts)
            .withColumn("seq_hash", F.xxhash64(F.concat_ws(" ", "events")))
            .withColumn("count_hash", F.xxhash64(F.concat_ws(" ", F.col("counts").cast("array<string>"))))
        )
        split = spark.read.parquet(str(config.GOLD / f"{dataset}_split")).select(
            "block_id", "split", "label", "order"
        )
        out = config.GOLD / f"{dataset}_sessions"
        sessions.join(split, "block_id", "inner").write.mode("overwrite").parquet(str(out))

        gold = spark.read.parquet(str(out))
        stats = gold.agg(
            F.count("*").alias("sessions"),
            F.sum("n_events").alias("events"),
            F.sum(F.aggregate("counts", F.lit(0), lambda a, x: a + x)).alias("counted"),
            F.countDistinct("seq_hash").alias("distinct_sequences"),
            F.countDistinct("count_hash").alias("distinct_count_vectors"),
            F.max("n_events").alias("max_len"),
            F.expr("percentile_approx(n_events, 0.5)").alias("median_len"),
        ).first().asDict()
        silver_rows = silver.count()
        silver_lines = silver.select("line_id").distinct().count()

    checks = {
        "session_count_matches": stats["sessions"] == 575_061 if dataset == "hdfs" else True,
        "every_silver_row_in_a_session": stats["events"] == silver_rows,
        "count_vectors_cover_every_event": stats["counted"] == stats["events"],
        "one_block_per_line": silver_rows == silver_lines,
    }
    runlog.write(
        dataset,
        "gold",
        {
            **stats,
            "silver_rows": silver_rows,
            "silver_distinct_lines": silver_lines,
            "vocabulary_size": len(vocab),
            "checks": checks,
            "timings": timings,
        },
    )
    spark.stop()
    if not all(checks.values()):
        raise SystemExit(f"gold checks failed: {checks}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "hdfs")
