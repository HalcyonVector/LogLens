"""How a log without sessions changes over time: alert categories and vocabulary by period.

For each windowing scheme of a log with alert tags (BGL): alert lines per
category in the training, validation and test periods of that scheme, the
share of test alert lines whose category never occurs in training, and the
share of lines the scheme's own parser could not name (UNK) in each period.

Usage: python -m loglens.drift bgl
"""

from __future__ import annotations

import json
import sys

from pyspark.sql import functions as F

from loglens import config, runlog
from loglens.spark import get_spark
from loglens.split import load_cutoffs, period


def run(dataset: str) -> None:
    cfg = config.settings(dataset)
    spark = get_spark(f"loglens-drift-{dataset}")
    bronze = spark.read.parquet(str(config.BRONZE / dataset)).select("line_id", "ts", "alert")
    out = {}
    for scheme in cfg["schemes"]:
        cut = load_cutoffs(scheme)
        lines = bronze.withColumn("p", period(cut))
        totals = {r["p"]: r["n"] for r in lines.groupBy("p").agg(F.count("*").alias("n")).collect()}
        rows = lines.filter(F.col("alert") != "-").groupBy("p", "alert").count().collect()
        by: dict = {}
        for r in rows:
            by.setdefault(r["alert"], {"train": 0, "val": 0, "test": 0})[r["p"]] = r["count"]
        alerts = {p: sum(v[p] for v in by.values()) for p in ("train", "val", "test")}
        new_in_test = sum(v["test"] for v in by.values() if v["train"] == 0)
        parse = json.loads((config.RUNLOG / f"{scheme}_parse.json").read_text())
        out[scheme] = {
            "cutoffs": cut,
            "lines": totals,
            "alert_lines": alerts,
            "categories": sorted(
                ({"alert": a, **v} for a, v in by.items()), key=lambda d: -(d["train"] + d["val"] + d["test"])
            ),
            "test_alert_lines_in_categories_unseen_in_train": new_in_test,
            "test_alert_share_unseen_categories": round(new_in_test / max(1, alerts["test"]), 4),
            "unk_rate_by_period": {k: v["unk_rate"] for k, v in parse["unk_by_period"].items()},
            "parser_fit": parse["fit"].get("parser_fit", "train"),
        }
        print(scheme, alerts, out[scheme]["test_alert_share_unseen_categories"], out[scheme]["unk_rate_by_period"], flush=True)
    runlog.write(dataset, "drift", out)
    (config.DASHBOARD / f"{dataset}_drift.json").write_text(json.dumps(out, indent=2))
    spark.stop()


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "bgl")
