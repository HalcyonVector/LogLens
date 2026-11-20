"""Week 1 EDA on the bronze layer. Writes a small JSON aggregate for the dashboard.

Usage: python -m loglens.eda hdfs
"""

from __future__ import annotations

import json
import sys

from pyspark.sql import functions as F

from loglens import config
from loglens.spark import get_spark

# HDFS block IDs can be negative, e.g. blk_-1608999687919862906.
BLOCK_RE = r"blk_-?\d+"


def rows(df) -> list[dict]:
    return [r.asDict() for r in df.collect()]


def run(dataset: str) -> None:
    cfg = config.DATASETS[dataset]
    spark = get_spark(f"loglens-eda-{dataset}")
    bronze = spark.read.parquet(str(config.BRONZE / dataset)).cache()
    total = bronze.count()

    per_hour = (
        bronze.groupBy("hour")
        .pivot("level")
        .count()
        .na.fill(0)
        .orderBy("hour")
    )
    levels = bronze.groupBy("level").count().orderBy(F.desc("count"))
    components = (
        bronze.groupBy("component")
        .agg(
            F.count("*").alias("lines"),
            F.sum((F.col("level") != "INFO").cast("int")).alias("non_info"),
        )
        .orderBy(F.desc("lines"))
    )
    # How many distinct blocks a line mentions: motivates the explode in sessionization.
    blocks_per_line = (
        bronze.select(
            F.size(F.regexp_extract_all("message", F.lit(BLOCK_RE), F.lit(0))).alias("n")
        )
        .groupBy("n")
        .count()
        .orderBy("n")
    )

    out = {
        "dataset": dataset,
        "total_lines": total,
        "lines_per_hour": rows(per_hour),
        "levels": rows(levels),
        "components": rows(components),
        "blocks_per_line": rows(blocks_per_line),
    }

    labels_path = cfg.get("labels")
    if labels_path and labels_path.exists():
        labels = spark.read.csv(str(labels_path), header=True)
        lab = rows(labels.groupBy("Label").count())
        out["labels"] = lab
        out["sessions"] = sum(r["count"] for r in lab)

    config.DASHBOARD.mkdir(parents=True, exist_ok=True)
    path = config.DASHBOARD / f"{dataset}_eda.json"
    path.write_text(json.dumps(out, indent=2, default=str))

    print(f"total lines: {total:,}")
    print("levels:", out["levels"])
    print("top components:", out["components"][:8])
    print("blocks per line:", out["blocks_per_line"])
    print("labels:", out.get("labels"))
    print("hours:", len(out["lines_per_hour"]))
    spark.stop()


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "hdfs")
