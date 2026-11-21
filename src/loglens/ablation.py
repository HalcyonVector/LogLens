"""Training-size ablation: how much training data does each model need?

Each model is refitted on a random 1%, 10%, 50% and 100% of the training
sessions (the rest are set aside, not moved to another split) and scored on
the same chronological test set, with thresholds still chosen on validation.

The baselines are cheap, so each fraction uses 5 different random draws and
reports the mean and the range; a single draw can mislead (see the README).
The LSTM uses the published settings with the END event and the
log-probability score, one draw per fraction. 1% of the training period is
about 3,450 sessions, close to the 4,855 normal sessions the DeepLog paper
trained on.

LSTM runs are checkpointed like the main LSTM runs, so an interrupted
ablation resumes. --reuse-lstm keeps the LSTM rows of the previous run (same
draw, same settings) and refits only the baselines.

Usage: python -m loglens.ablation hdfs [--reuse-lstm]
"""

from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.metrics import average_precision_score

from loglens import config, runlog
from loglens.baselines import SCORERS, load
from loglens.lstm import (
    PUBLISHED, SEEDS, build_ctx, checkpoint_clear, checkpoint_load, checkpoint_save, fit_and_score, sequences,
)
from loglens.metrics import evaluate

FRACTIONS = (0.01, 0.1, 0.5, 1.0)
BASELINES = ("pca", "iforest", "lightgbm")
SEED = 11
BASELINE_DRAWS = (11, 12, 13, 14, 15)


def subsample(split: np.ndarray, fraction: float, seed: int = SEED) -> np.ndarray:
    """Keep a random fraction of training sessions; mark the rest as unused."""
    # object dtype: a fixed-width string array ("<U5") would silently truncate "unused".
    out = split.astype(object)
    train_idx = np.flatnonzero(split == "train")
    keep = np.random.default_rng(seed).choice(train_idx, size=max(1, int(len(train_idx) * fraction)), replace=False)
    out[train_idx] = "unused"
    out[keep] = "train"
    return out


def score(y, s, split) -> dict:
    va, te = split == "val", split == "test"
    m = evaluate(y[va], s[va], y[te], s[te])
    return {"f1": m["f1"], "precision": m["precision"], "recall": m["recall"],
            "pr_auc": round(float(average_precision_score(y[te], s[te])), 6), "fpr": m["fpr"]}


def run(dataset: str, reuse_lstm: bool = False) -> None:
    df = load(dataset)
    X = np.stack(df["counts"].to_numpy()).astype(np.float32)
    y = df["label"].to_numpy().astype(int)
    base = {"X": X, "y": y, "n_warn": df["n_warn"].to_numpy(), "seq_hash": df["seq_hash"].to_numpy(),
            "count_hash": df["count_hash"].to_numpy()}
    chrono = df["split"].to_numpy()

    previous = {}
    path = config.DASHBOARD / f"{dataset}_ablation.json"
    if reuse_lstm and path.exists():
        previous = {r["fraction"]: r for r in json.loads(path.read_text())["rows"] if r["model"] == "deeplog_end_logprob"}

    rows = []
    for frac in FRACTIONS:
        n_train = int((subsample(chrono, frac) == "train").sum())
        for name in BASELINES:
            draws = []
            for seed in (BASELINE_DRAWS if frac < 1.0 else BASELINE_DRAWS[:1]):  # 100% is the same set every time
                split = subsample(chrono, frac, seed)
                s, _ = SCORERS[name]({**base, "split": split})
                draws.append(score(y, s, split))
            f1s = [d["f1"] for d in draws]
            r = {"model": name, "fraction": frac, "train_sessions": n_train,
                 "f1": round(float(np.mean(f1s)), 6), "f1_min": min(f1s), "f1_max": max(f1s),
                 "pr_auc": round(float(np.mean([d["pr_auc"] for d in draws])), 6), "draws": draws}
            rows.append(r)
            print(f"{frac:>5.0%} {name:9s} F1={r['f1']:.3f} (range {r['f1_min']:.3f} to {r['f1_max']:.3f}) "
                  f"PR-AUC={r['pr_auc']:.3f}", flush=True)

        if frac in previous:
            r = previous[frac]
        else:
            ctx = build_ctx(dataset)
            assert (ctx["y"] == y).all()  # same session order as `load` (both sort by order)
            key = f"ablation_{dataset}_deeplog_end_frac{frac}"
            saved = checkpoint_load(key)
            if saved is None:
                split = subsample(chrono, frac)
                res = fit_and_score(ctx, sequences(ctx, end=True), split, {"end": True, **PUBLISHED}, SEEDS[0])
                saved = {**score(y, res["s_nll"], split), "train_windows": res["train_windows"],
                         "epochs": res["info"]["epochs"], "train_s": res["train_s"]}
                checkpoint_save(key, saved)
            r = {"model": "deeplog_end_logprob", "fraction": frac, "train_sessions": n_train, **saved}
        rows.append(r)
        print(f"{frac:>5.0%} {'lstm':9s} F1={r['f1']:.3f} PR-AUC={r['pr_auc']:.3f} ({r['epochs']} epochs)", flush=True)

    out = {"fractions": list(FRACTIONS), "baseline_draws": len(BASELINE_DRAWS), "rows": rows}
    runlog.write(dataset, "ablation", out)
    path.write_text(json.dumps(out, indent=2))
    checkpoint_clear(f"ablation_{dataset}_")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run(args[0] if args else "hdfs", reuse_lstm="--reuse-lstm" in sys.argv)
