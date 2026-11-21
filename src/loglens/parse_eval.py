"""Score the learned templates against LogHub's ground truth.

LogHub ships 29 ground-truth templates for HDFS_v1 (HDFS.log_templates.csv) and
the ground-truth event sequence of every block (Event_traces.csv). Each line's
ground-truth event comes from matching those templates, and the result is
validated against the published traces before it is used as a reference.

Metric: grouping accuracy (GA, Zhu et al. 2019). A line counts as correctly
parsed when the set of lines sharing its learned template is exactly the set
sharing its ground-truth template. UNK lines are never correct.

Sweep: --sweep refits Drain3 at several similarity thresholds and reports GA
per period. The threshold is chosen on the validation period; the test period
is reported but never used to choose.

Usage: python -m loglens.parse_eval hdfs [--sweep]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

from pyspark.sql import Column, DataFrame, Window, functions as F

from drain3 import TemplateMiner

from loglens import config, runlog
from loglens.masking import mask_col
from loglens.parse import miner_config
from loglens.spark import get_spark
from loglens.split import load_cutoffs, period

GT_DIR = config.RAW / "HDFS_v1" / "preprocessed"


def gt_templates(spark, dataset: str) -> list[tuple[str, str]]:
    if config.parent(dataset) == "hdfs":
        path = GT_DIR / "HDFS.log_templates.csv"
    else:
        path = config.settings(dataset)["ground_truth"] / "BGL" / "BGL_full.log_templates.csv"
    return [(r["EventId"], r["EventTemplate"]) for r in spark.read.csv(str(path), header=True, escape='"').collect()]


def gt_lines(spark, dataset: str) -> DataFrame:
    """(line_id, ts, gt) for every bronze line that has a ground-truth event.

    HDFS: each line is tagged with LogHub's 29 template patterns.
    BGL: Loghub-2.0 annotates its own copy of the log, which leaves out 2.5% of the
    raw lines (empty messages, NULL nodes, register dumps). Its lines are matched to
    bronze on (timestamp to the microsecond, node, message); unmatched lines are left out.
    """
    cfg = config.settings(dataset)
    bronze = spark.read.parquet(str(config.BRONZE / config.parent(dataset)))
    if config.parent(dataset) == "hdfs":
        return bronze.select("line_id", "ts", gt_column(gt_templates(spark, dataset)).alias("gt"))
    gt_dir = cfg["ground_truth"] / "BGL"
    pattern, fields = cfg["regex"], cfg["fields"]
    full = spark.read.text(str(gt_dir / "BGL_full.log")).withColumn("_m", F.monotonically_increasing_id())
    full = full.withColumn("LineId", F.row_number().over(Window.orderBy("_m")))

    def get(name: str) -> Column:
        return F.regexp_extract("value", pattern, fields.index(name) + 1)

    full = full.select(
        "LineId",
        F.to_timestamp(get("time"), cfg["ts"]["format"]).alias("ts"),
        get("node").alias("node"),
        get("message").alias("message"),
    )
    structured = spark.read.csv(str(gt_dir / "BGL_full.log_structured.csv"), header=True, escape='"').select(
        F.col("LineId").cast("int").alias("LineId"), F.col("EventId").alias("gt")
    )
    key = full.join(structured, "LineId").select("ts", "node", "message", "gt").dropDuplicates(["ts", "node", "message"])
    lines = bronze.select("line_id", "ts", "node", "message")
    return lines.join(key, ["ts", "node", "message"]).select("line_id", "ts", "gt")


def gt_regex(template: str) -> str:
    parts = template.split("[*]")
    return "^" + ".*".join(r"\Q" + p + r"\E" if p else "" for p in parts) + "$"


def gt_column(templates: list[tuple[str, str]]) -> Column:
    # Longest literal text first, so a specific template beats a general one.
    ordered = sorted(templates, key=lambda t: -len(t[1].replace("[*]", "")))
    expr = None
    for event_id, template in ordered:
        cond = F.col("message").rlike(gt_regex(template))
        expr = F.when(cond, event_id) if expr is None else expr.when(cond, event_id)
    return expr.otherwise("NONE")


def grouping_accuracy(cells: list[tuple[str, str, int]], unk: str = "UNK") -> dict:
    """cells: (predicted, ground_truth, count). Returns GA and merge/split counts."""
    pred_to_gt: dict[str, set] = defaultdict(set)
    gt_to_pred: dict[str, set] = defaultdict(set)
    total = 0
    for p, g, n in cells:
        pred_to_gt[p].add(g)
        gt_to_pred[g].add(p)
        total += n
    correct = sum(
        n
        for p, g, n in cells
        if p != unk and len(pred_to_gt[p]) == 1 and len(gt_to_pred[g]) == 1
    )
    return {
        "lines": total,
        "correct": correct,
        "grouping_accuracy": round(correct / total, 6) if total else None,
        "predicted_templates": len([p for p in pred_to_gt if p != unk]),
        "ground_truth_templates": len(gt_to_pred),
        "gt_split_across_templates": sum(1 for g, ps in gt_to_pred.items() if len(ps - {unk}) > 1),
        "templates_mixing_gt": sum(1 for p, gs in pred_to_gt.items() if p != unk and len(gs) > 1),
    }


def run(dataset: str) -> None:
    spark = get_spark(f"loglens-parse-eval-{dataset}")
    timings: dict = {}

    with runlog.timer(timings, "total_s"):
        templates_gt = gt_templates(spark, dataset)
        gt = gt_lines(spark, dataset).select("line_id", "gt").cache()
        bronze_lines = spark.read.parquet(str(config.BRONZE / config.parent(dataset))).count()
        gt_coverage = gt.count() / bronze_lines

        silver = spark.read.parquet(str(config.SILVER / f"{dataset}_events"))
        cutoffs = load_cutoffs(dataset)
        lines = (
            silver.dropDuplicates(["line_id"])
            .select("line_id", "event_id", period(cutoffs).alias("period"))
            .join(gt, "line_id")
        )
        contingency = [
            (r["event_id"], r["gt"], r["period"], r["n"])
            for r in lines.groupBy("event_id", "gt", "period").agg(F.count("*").alias("n")).collect()
        ]

        trace_check = None
        if dataset == "hdfs":
            # Validate the ground-truth tagging against LogHub's published traces.
            traces = spark.read.csv(str(GT_DIR / "Event_traces.csv"), header=True).select(
                F.col("BlockId").alias("block_id"),
                F.split(F.regexp_replace("Features", r"[\[\]]", ""), ",").alias("published"),
            )
            ours = (
                silver.join(gt, "line_id")
                .groupBy("block_id")
                .agg(F.sort_array(F.collect_list(F.struct("ts", "line_id", "gt"))).alias("s"))
                .select("block_id", F.col("s.gt").alias("ours"))
            )
            compared = traces.join(ours, "block_id", "full_outer")
            trace_check = compared.agg(
                F.count("*").alias("blocks"),
                F.sum((F.col("ours") == F.col("published")).cast("int")).alias("exact"),
                F.sum((F.sort_array("ours") == F.sort_array("published")).cast("int")).alias("same_events"),
                F.sum(F.col("ours").isNull().cast("int")).alias("missing_ours"),
                F.sum(F.col("published").isNull().cast("int")).alias("missing_published"),
            ).first().asDict()
        unmatched_gt = sum(n for _, g, _, n in contingency if g == "NONE")

    overall = grouping_accuracy([(p, g, n) for p, g, _, n in contingency])
    by_period = {
        per: grouping_accuracy([(p, g, n) for p, g, pp, n in contingency if pp == per])
        for per in ("train", "val", "test")
    }
    # Collapse periods for the dashboard's template explorer.
    cells: dict = defaultdict(int)
    for p, g, _, n in contingency:
        cells[(p, g)] += n

    result = {
        "overall": overall,
        "by_period": by_period,
        "lines_without_gt_template": unmatched_gt,
        "gt_coverage": round(gt_coverage, 6),
    }
    checks = {"gt_tags_every_line": unmatched_gt == 0}
    if trace_check is not None:
        result["trace_check"] = trace_check
        result["trace_exact_share"] = round(trace_check["exact"] / trace_check["blocks"], 6)
        result["trace_same_events_share"] = round(trace_check["same_events"] / trace_check["blocks"], 6)
        checks["gt_reproduces_published_events"] = result["trace_same_events_share"] > 0.999
    else:
        # Loghub-2.0 leaves out 2.5% of BGL's raw lines; every one of its own lines should match.
        checks["gt_covers_expected_share"] = gt_coverage > 0.97
    runlog.write(dataset, "parse_eval", {**result, "checks": checks, "timings": timings})
    (config.DASHBOARD / f"{dataset}_parse_eval.json").write_text(
        json.dumps(
            {
                **result,
                "gt_templates": dict(templates_gt),
                "contingency": [{"event_id": p, "gt": g, "lines": n} for (p, g), n in sorted(cells.items())],
            },
            indent=2,
        )
    )
    spark.stop()
    if not all(checks.values()):
        print(f"WARNING: parse_eval checks failed: {checks}")


SWEEP = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def sweep(dataset: str) -> None:
    spark = get_spark(f"loglens-parse-sweep-{dataset}")
    cutoffs = load_cutoffs(dataset)
    bronze = spark.read.parquet(str(config.BRONZE / config.parent(dataset))).select("line_id", "message")
    tagged = gt_lines(spark, dataset).select("line_id", "ts", "gt").join(bronze, "line_id")
    cells = (
        tagged.select(
            "line_id",
            period(cutoffs).alias("period"),
            mask_col("message", dataset).alias("masked"),
            "gt",
        )
        .groupBy("period", "masked", "gt")
        .agg(F.min("line_id").alias("first"), F.count("*").alias("n"))
        .collect()
    )
    spark.stop()

    first_seen: dict[str, int] = {}
    for r in cells:
        if r["period"] == "train":
            first_seen[r["masked"]] = min(first_seen.get(r["masked"], r["first"]), r["first"])
    train_msgs = sorted(first_seen, key=first_seen.get)

    results = []
    for th in SWEEP:
        miner = TemplateMiner(config=miner_config(th, dataset))
        for msg in train_msgs:
            miner.add_log_message(msg)
        assign: dict[str, str] = {}
        for msg in {r["masked"] for r in cells}:
            c = miner.match(msg, full_search_strategy="fallback")
            assign[msg] = f"C{c.cluster_id}" if c is not None else "UNK"
        row = {"drain_sim_th": th, "templates": len(miner.drain.clusters)}
        for per in ("train", "val", "test"):
            ga = grouping_accuracy([(assign[r["masked"]], r["gt"], r["n"]) for r in cells if r["period"] == per])
            row[per] = {k: ga[k] for k in ("grouping_accuracy", "gt_split_across_templates", "templates_mixing_gt")}
        results.append(row)
        print(
            f"sim_th={th}: templates={row['templates']} "
            + " ".join(f"{p}_GA={row[p]['grouping_accuracy']:.4f}" for p in ("train", "val", "test")),
            flush=True,
        )

    best = max(results, key=lambda r: (r["val"]["grouping_accuracy"], -r["templates"]))
    runlog.write(dataset, "parse_sweep", {"chosen_drain_sim_th": best["drain_sim_th"], "chosen_by": "val grouping accuracy, then fewer templates", "results": results})
    (config.DASHBOARD / f"{dataset}_parse_sweep.json").write_text(
        json.dumps({"chosen": best["drain_sim_th"], "results": results}, indent=2)
    )


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ds = args[0] if args else "hdfs"
    sweep(ds) if "--sweep" in sys.argv else run(ds)
