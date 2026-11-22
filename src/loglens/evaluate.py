"""Deeper evaluation on the chronological test period, and the dashboard's inputs.

For each final model (the LSTM and LogBERT at seed 0, whose per-event scores are saved):
  intervals    95% bootstrap intervals (1,000 resamples of test sessions) for
               precision, recall, F1 and PR-AUC, plus paired intervals for the
               differences that matter (PCA vs the best LSTM, LightGBM vs PCA)
  operations   alerts per hour, false alarms per hour and per 1,000 normal
               sessions, and flagged vs labelled anomalies per hour
  timeout      a live system cannot know a session has ended; the idle time
               after which it would assume so is set from normal training
               sessions, and every detector that needs a finished session
               (count models, the END event) has to wait for it
  latency      for each caught anomaly, how long after its first event the
               alert could fire, and after how many events
  errors       templates over-represented in false alarms and in misses
  drill-down   example sessions with the event that made the LSTM fire

Also exports pipeline health from the run logs, so the dashboard only ever
reads data/dashboard/.

Usage: python -m loglens.evaluate hdfs
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from loglens import config, runlog
from loglens.metrics import prf

B = 1000
SEED = 7
TIMEOUT_QUANTILE = 0.99
BEST_LSTM = "deeplog_end_tuned_logprob"

# name -> (source, variant or column, score kind)
MODELS = {
    "warn_rule": ("baseline", "warn_rule", None),
    "seq_lookup": ("baseline", "seq_lookup", None),
    "pca": ("baseline", "pca", None),
    "iforest": ("baseline", "iforest", None),
    "lightgbm": ("baseline", "lightgbm", None),
    "deeplog_topg": ("lstm", "deeplog", "rank"),
    "deeplog_end_topg": ("lstm", "deeplog_end", "rank"),
    "deeplog_end_logprob": ("lstm", "deeplog_end", "nll"),
    "deeplog_end_tuned_logprob": ("lstm", "deeplog_end_tuned", "nll"),
    "logbert_topg": ("logbert", "logbert", "rank"),
    "logbert_end_logprob": ("logbert", "logbert_end", "nll"),
}
# Detectors that judge a whole session can only decide once it is over. LogBERT reads the
# events on both sides of the one it judges, so it waits for the end of the session too.
NEEDS_FINISHED_SESSION = {"warn_rule", "seq_lookup", "pca", "iforest", "lightgbm", "logbert_topg", "logbert_end_logprob"}


# ---------------------------------------------------------------- loading ----


def load(dataset: str) -> dict:
    gold = pd.read_parquet(
        config.GOLD / f"{dataset}_sessions",
        columns=["block_id", "events", "offsets_s", "n_events", "duration_s", "start_ts", "seq_hash", "split", "label", "order"],
    ).sort_values("order").reset_index(drop=True)
    base_scores = pd.read_parquet(config.GOLD / f"{dataset}_baseline_scores.parquet")
    gold = gold.merge(base_scores, on="block_id", how="left")
    baselines = json.loads((config.DASHBOARD / f"{dataset}_baselines.json").read_text())
    lstm = json.loads((config.DASHBOARD / f"{dataset}_lstm.json").read_text())
    logbert = json.loads((config.DASHBOARD / f"{dataset}_logbert.json").read_text())
    ranks = {
        v: json.loads((config.GOLD / f"{dataset}_{v}_ranks.json").read_text())
        for v in {m[1] for m in MODELS.values() if m[0] != "baseline"}
    }
    templates = json.loads((config.SILVER / f"{dataset}_templates.json").read_text())
    return {"gold": gold, "baselines": baselines, "lstm": lstm, "logbert": logbert, "ranks": ranks, "templates": templates}


def model_scores(data: dict) -> dict:
    """Per model: session scores, the validation-chosen threshold, and per-event scores for LSTMs."""
    gold, out = data["gold"], {}
    hashes = gold["seq_hash"].astype(str)
    for name, (source, key, kind) in MODELS.items():
        if source == "baseline":
            thr = data["baselines"]["results"]["chrono"][key]["threshold"]
            out[name] = {"scores": gold[key].to_numpy(), "threshold": thr, "events": None}
        else:
            seed0 = data[source][key]["results"]["chrono"]["seeds"]["0"]
            thr = seed0["threshold"] if kind == "rank" else seed0["nll_score"]["threshold"]
            per_seq = {h: np.asarray(v["ranks" if kind == "rank" else "nll"], dtype=float) for h, v in data["ranks"][key].items()}
            session_max = {h: float(v.max()) for h, v in per_seq.items()}
            out[name] = {
                "scores": hashes.map(session_max).to_numpy(),
                "threshold": thr,
                "events": per_seq,
                "end_event": data[source][key]["config"]["end"],
            }
    return out


# -------------------------------------------------------------- intervals ----


def bootstrap(y: np.ndarray, scores: dict, thresholds: dict, pairs: list[tuple[str, str]]) -> tuple[dict, dict]:
    rng = np.random.default_rng(SEED)
    n = len(y)
    flags = {m: (s >= thresholds[m]).astype(np.int8) for m, s in scores.items()}
    samples = {m: {"precision": [], "recall": [], "f1": [], "pr_auc": []} for m in scores}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        for m in scores:
            r = prf(yb, flags[m][idx])
            samples[m]["precision"].append(r["precision"])
            samples[m]["recall"].append(r["recall"])
            samples[m]["f1"].append(r["f1"])
            samples[m]["pr_auc"].append(average_precision_score(yb, scores[m][idx]))

    def ci(values) -> list[float]:
        lo, hi = np.percentile(values, [2.5, 97.5])
        return [round(float(lo), 4), round(float(hi), 4)]

    intervals = {m: {k: ci(v) for k, v in s.items()} for m, s in samples.items()}
    paired = {}
    for a, b in pairs:
        for k in ("f1", "pr_auc"):
            diff = np.array(samples[a][k]) - np.array(samples[b][k])
            paired[f"{a}_minus_{b}_{k}"] = {
                "mean": round(float(diff.mean()), 4),
                "ci": ci(diff),
                "share_of_resamples_above_zero": round(float((diff > 0).mean()), 4),
            }
    return intervals, paired


# ----------------------------------------------------------- operations ----


def timeout_analysis(gold: pd.DataFrame) -> dict:
    """Idle time after which a live system would treat a session as finished."""
    def max_gap(offsets) -> int:
        o = np.asarray(offsets)
        return int(np.diff(o).max()) if len(o) > 1 else 0

    normal_train = gold[(gold["split"] == "train") & (gold["label"] == 0)]
    gaps = normal_train["offsets_s"].map(max_gap)
    timeout = int(np.quantile(gaps, TIMEOUT_QUANTILE))
    test = gold[gold["split"] == "test"]
    test_gaps = test["offsets_s"].map(max_gap)
    durations = gold.loc[gold["label"] == 0, "duration_s"]
    return {
        "timeout_s": timeout,
        "chosen_as": f"{int(TIMEOUT_QUANTILE * 100)}th percentile of the longest idle gap inside normal training sessions",
        "max_gap_percentiles_s": {f"p{q}": int(np.percentile(gaps, q)) for q in (50, 90, 99)},
        "normal_duration_percentiles_s": {f"p{q}": int(np.percentile(durations, q)) for q in (50, 90, 99)},
        "test_sessions_cut_early": int((test_gaps > timeout).sum()),
        "test_sessions_cut_early_share": round(float((test_gaps > timeout).mean()), 6),
    }


def first_trigger(event_scores: np.ndarray, threshold: float) -> int | None:
    hits = np.nonzero(event_scores >= threshold)[0]
    return int(hits[0]) if len(hits) else None


def latency(gold: pd.DataFrame, te: np.ndarray, ms: dict, timeout: int) -> dict:
    """Seconds and events from a caught anomaly's first event to the earliest possible alert."""
    y = gold["label"].to_numpy()
    out = {}
    for name, m in ms.items():
        caught = te & (y == 1) & (m["scores"] >= m["threshold"])
        rows = gold.loc[caught]
        secs, events = [], []
        for _, r in rows.iterrows():
            if name in NEEDS_FINISHED_SESSION:
                secs.append(r["duration_s"] + timeout)
                events.append(r["n_events"])
                continue
            pos = first_trigger(m["events"][str(r["seq_hash"])], m["threshold"])
            if pos is None:  # cannot happen for a caught session; guard anyway
                continue
            if pos >= r["n_events"]:  # fired on the END event: needs the timeout
                secs.append(r["duration_s"] + timeout)
                events.append(r["n_events"])
            else:
                secs.append(int(r["offsets_s"][pos]))
                events.append(pos + 1)
        if not secs:
            continue
        secs, events = np.asarray(secs), np.asarray(events)
        out[name] = {
            "caught": int(len(secs)),
            "seconds": {f"p{q}": int(np.percentile(secs, q)) for q in (10, 50, 90)},
            "events": {f"p{q}": int(np.percentile(events, q)) for q in (10, 50, 90)},
            "within_1_minute": round(float((secs <= 60).mean()), 4),
            "within_1_hour": round(float((secs <= 3600).mean()), 4),
            "histogram_minutes": np.histogram(np.minimum(secs / 60, 600), bins=[0, 1, 5, 15, 60, 180, 360, 601])[0].tolist(),
        }
    return out


def operations(gold: pd.DataFrame, te: np.ndarray, ms: dict) -> dict:
    y = gold["label"].to_numpy()
    start = gold.loc[te, "start_ts"]
    hours = (start.max() - start.min()).total_seconds() / 3600
    hour_bin = gold["start_ts"].dt.floor("h").dt.strftime("%Y-%m-%dT%H:00")
    out = {"test_hours": round(hours, 2), "per_model": {}, "per_hour": {}}
    labelled = pd.Series(y[te]).groupby(hour_bin[te].to_numpy()).sum()
    sessions = pd.Series(1, index=range(int(te.sum()))).groupby(hour_bin[te].to_numpy()).sum()
    out["per_hour"]["_sessions"] = {k: int(v) for k, v in sessions.items()}
    out["per_hour"]["_labelled_anomalies"] = {k: int(v) for k, v in labelled.items()}
    for name, m in ms.items():
        flag = m["scores"] >= m["threshold"]
        fp = te & flag & (y == 0)
        tp = te & flag & (y == 1)
        out["per_model"][name] = {
            "alerts": int((te & flag).sum()),
            "alerts_per_hour": round(float((te & flag).sum() / hours), 1),
            "false_alarms_per_hour": round(float(fp.sum() / hours), 1),
            "false_alarms_per_1000_normal": round(1000 * float(fp.sum() / (te & (y == 0)).sum()), 2),
        }
        out["per_hour"][name] = {
            "tp": {k: int(v) for k, v in pd.Series(tp[te]).groupby(hour_bin[te].to_numpy()).sum().items()},
            "fp": {k: int(v) for k, v in pd.Series(fp[te]).groupby(hour_bin[te].to_numpy()).sum().items()},
        }
    return out


# ----------------------------------------------------------------- errors ----


def error_lift(gold: pd.DataFrame, te: np.ndarray, m: dict, names: dict, top: int = 8) -> dict:
    """Templates over-represented in false alarms (vs normal sessions) and misses (vs anomalies)."""
    y = gold["label"].to_numpy()
    flag = m["scores"] >= m["threshold"]
    has = gold["events"].map(lambda ev: set(ev))

    def lift(group: np.ndarray, reference: np.ndarray) -> list[dict]:
        if group.sum() == 0:
            return []
        g_sets, r_sets = has[group], has[reference]
        rows = []
        for e in set().union(*g_sets):
            g = sum(e in s for s in g_sets)
            r = sum(e in s for s in r_sets)
            if g < 5:
                continue
            g_share, r_share = g / len(g_sets), r / len(r_sets)
            rows.append({"event_id": e, "template": names.get(e, e), "sessions": g,
                         "share": round(g_share, 4), "reference_share": round(r_share, 4),
                         "lift": round(g_share / r_share, 2) if r_share else None})
        # Rank by how much more often the template appears (share minus reference share), so a
        # template in 1% of these sessions and 0% of the reference cannot top the list on lift alone.
        rows.sort(key=lambda d: -(d["share"] - d["reference_share"]))
        return [r for r in rows if r["share"] > r["reference_share"]][:top]

    return {
        "false_alarms": int((te & flag & (y == 0)).sum()),
        "misses": int((te & ~flag & (y == 1)).sum()),
        "in_false_alarms": lift(te & flag & (y == 0), te & (y == 0)),
        "in_misses": lift(te & ~flag & (y == 1), te & (y == 1)),
    }


# -------------------------------------------------------------- drill-down ----


def drilldown(gold: pd.DataFrame, te: np.ndarray, ms: dict, per_group: int = 10) -> list[dict]:
    y = gold["label"].to_numpy()
    lstm, pca = ms[BEST_LSTM], ms["pca"]
    f_lstm = lstm["scores"] >= lstm["threshold"]
    f_pca = pca["scores"] >= pca["threshold"]
    groups = {
        "caught_by_lstm": te & (y == 1) & f_lstm,
        "lstm_missed_pca_caught": te & (y == 1) & ~f_lstm & f_pca,
        "pca_missed_lstm_caught": te & (y == 1) & f_lstm & ~f_pca,
        "lstm_false_alarm": te & (y == 0) & f_lstm,
    }
    out = []
    for group, mask in groups.items():
        rows = gold.loc[mask].drop_duplicates("seq_hash")
        # Prefer variety: the most common sequences in the group first, then shorter ones.
        counts = gold.loc[mask, "seq_hash"].value_counts()
        rows = rows.assign(_n=rows["seq_hash"].map(counts)).sort_values(["_n", "n_events"], ascending=[False, True])
        for _, r in rows.head(per_group).iterrows():
            h = str(r["seq_hash"])
            ev_scores = lstm["events"][h]
            events = list(r["events"]) + (["END"] if lstm["end_event"] else [])
            trigger = first_trigger(ev_scores, lstm["threshold"])
            flagged_by = [m for m in ("pca", "lightgbm", "iforest", BEST_LSTM) if ms[m]["scores"][r.name] >= ms[m]["threshold"]]
            out.append({
                "group": group,
                "block_id": r["block_id"],
                "label": int(r["label"]),
                "sessions_like_this": int(r["_n"]),
                "n_events": int(r["n_events"]),
                "duration_s": int(r["duration_s"]),
                "events": events,
                "surprise": [round(float(s), 2) for s in ev_scores],
                "trigger": trigger,
                "flagged_by": flagged_by,
            })
    return out


# ------------------------------------------------- overlap and templates ----


def overlap(gold: pd.DataFrame, te: np.ndarray, ms: dict, a: str, b: str) -> dict:
    """Which test anomalies each of two models catches: both, only one, or neither."""
    y = gold["label"].to_numpy()
    an = te & (y == 1)
    fa = ms[a]["scores"] >= ms[a]["threshold"]
    fb = ms[b]["scores"] >= ms[b]["threshold"]
    return {
        "a": a, "b": b, "anomalies": int(an.sum()),
        "both": int((an & fa & fb).sum()),
        "only_a": int((an & fa & ~fb).sum()),
        "only_b": int((an & ~fa & fb).sum()),
        "neither": int((an & ~fa & ~fb).sum()),
    }


def template_stats(gold: pd.DataFrame, names: dict) -> list[dict]:
    """Per event: sessions that contain it and how many of those are anomalous, by period."""
    # explode keeps each session's index, so (index, event) is one session containing that event.
    rows = gold[["events", "label", "split"]].explode("events").reset_index().drop_duplicates(["index", "events"])
    g = rows.groupby("events").agg(sessions=("label", "size"), anomalous=("label", "sum"))
    by_split = rows.groupby(["events", "split"]).size().unstack(fill_value=0)
    out = []
    for e, r in g.iterrows():
        out.append({
            "event_id": e,
            "template": names.get(e, e),
            "sessions": int(r["sessions"]),
            "anomalous_sessions": int(r["anomalous"]),
            "anomaly_rate": round(float(r["anomalous"] / r["sessions"]), 4),
            "train_sessions": int(by_split.loc[e].get("train", 0)),
            "test_sessions": int(by_split.loc[e].get("test", 0)),
        })
    return sorted(out, key=lambda d: -d["sessions"])


# ------------------------------------------------------- pipeline health ----


def pipeline_health(dataset: str) -> dict:
    def read(stage: str) -> dict:
        p = config.RUNLOG / f"{dataset}_{stage}.json"
        return json.loads(p.read_text()) if p.exists() else {}

    bronze, split, parse, pe, gold = (read(s) for s in ("bronze", "split", "parse", "parse_eval", "gold"))
    checks = {}
    for stage, d in (("bronze", bronze), ("split", split), ("parse", parse), ("parse_eval", pe), ("gold", gold)):
        for k, v in (d.get("checks") or {}).items():
            checks[f"{stage}.{k}"] = v
    return {
        "raw_lines": bronze.get("raw_lines"),
        "quarantine_rows": bronze.get("quarantine_rows"),
        "span_hours": bronze.get("span_hours"),
        "first_ts": bronze.get("first_ts"),
        "last_ts": bronze.get("last_ts"),
        "templates": parse.get("fit", {}).get("templates"),
        "drain_sim_th": parse.get("fit", {}).get("drain_sim_th"),
        "train_distinct_masked": parse.get("fit", {}).get("train_distinct_masked"),
        "unk_by_period": parse.get("unk_by_period"),
        "matching_benchmark": parse.get("benchmark"),
        "grouping_accuracy": pe.get("overall", {}).get("grouping_accuracy"),
        "grouping_accuracy_test": pe.get("by_period", {}).get("test", {}).get("grouping_accuracy"),
        "trace_exact_share": pe.get("trace_exact_share"),
        "sessions": gold.get("sessions"),
        "events": gold.get("events"),
        "distinct_sequences": gold.get("distinct_sequences"),
        "distinct_count_vectors": gold.get("distinct_count_vectors"),
        "per_split": split.get("per_split"),
        "checks": checks,
        "checks_passed": sum(bool(v) for v in checks.values()),
        "checks_total": len(checks),
        "run_date": gold.get("finished_utc"),
    }


# -------------------------------------------------------------------- run ----


def run(dataset: str) -> None:
    data = load(dataset)
    gold = data["gold"]
    te = (gold["split"] == "test").to_numpy()
    y = gold["label"].to_numpy()
    ms = model_scores(data)
    names = {t["event_id"]: t["template"] for t in data["templates"]} | {"END": "(end of session)", "UNK": "(unknown template)"}

    point = {}
    for name, m in ms.items():
        r = prf(y[te], (m["scores"][te] >= m["threshold"]).astype(int))
        r["pr_auc"] = round(float(average_precision_score(y[te], m["scores"][te])), 6)
        point[name] = r
        print(f"{name:28s} F1={r['f1']:.3f} PR-AUC={r['pr_auc']:.3f} FP={r['fp']}", flush=True)
        # The saved scores must reproduce the training run's own result, or everything below drifts.
        source, key, kind = MODELS[name]
        recorded = (
            data["baselines"]["results"]["chrono"][key]["f1"] if source == "baseline"
            else (lambda s: s["f1"] if kind == "rank" else s["nll_score"]["f1"])(data[source][key]["results"]["chrono"]["seeds"]["0"])
        )
        if r["f1"] != recorded:
            raise SystemExit(f"{name}: F1 from saved scores {r['f1']} != recorded {recorded}")

    intervals, paired = bootstrap(
        y[te],
        {k: m["scores"][te] for k, m in ms.items()},
        {k: m["threshold"] for k, m in ms.items()},
        [("pca", BEST_LSTM), ("lightgbm", "pca"), ("deeplog_end_logprob", "deeplog_topg"),
         ("logbert_end_logprob", BEST_LSTM), ("pca", "logbert_end_logprob"), ("logbert_topg", "deeplog_topg")],
    )
    print("bootstrap done", json.dumps(paired), flush=True)
    tout = timeout_analysis(gold)
    print("timeout", json.dumps(tout), flush=True)
    lat = latency(gold, te, ms, tout["timeout_s"])
    ops = operations(gold, te, ms)
    errors = {m: error_lift(gold, te, ms[m], names) for m in ("pca", BEST_LSTM)}
    drill = drilldown(gold, te, ms)

    result = {
        "models": {m: {**point[m], "ci": intervals[m]} for m in ms},
        "paired_differences": paired,
        "bootstrap": {"resamples": B, "seed": SEED, "note": "LSTM rows use seed 0, whose per-event scores are saved"},
        "timeout": tout,
        "latency": lat,
        "operations": ops,
        "errors": errors,
        "overlap": {
            "pca_vs_best_lstm": overlap(gold, te, ms, "pca", BEST_LSTM),
            "pca_vs_deeplog": overlap(gold, te, ms, "pca", "deeplog_topg"),
            "pca_vs_logbert": overlap(gold, te, ms, "pca", "logbert_end_logprob"),
            "lstm_vs_logbert": overlap(gold, te, ms, BEST_LSTM, "logbert_end_logprob"),
        },
        "templates": template_stats(gold, names),
        "best_lstm": BEST_LSTM,
    }
    runlog.write(dataset, "evaluate", {k: v for k, v in result.items() if k != "operations"})
    (config.DASHBOARD / f"{dataset}_evaluation.json").write_text(json.dumps(result, indent=2))
    (config.DASHBOARD / f"{dataset}_drilldown.json").write_text(json.dumps({"model": BEST_LSTM, "templates": names, "sessions": drill}, indent=1))
    (config.DASHBOARD / f"{dataset}_pipeline.json").write_text(json.dumps(pipeline_health(dataset), indent=2))


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "hdfs")
