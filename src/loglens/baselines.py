"""Baselines on gold sessions, under a chronological and a random split.

Models (all score sessions; higher = more anomalous):
  warn_rule   number of WARN lines in the session (no fitting)
  seq_lookup  anomaly rate of the identical event sequence in training;
              a sequence never seen in training scores 1.0 (novelty = suspicious)
  pca         Xu et al. 2009: TF-IDF weighted counts, PCA fitted on training
              sessions, score = squared residual off the principal subspace
  iforest     Isolation Forest on log(1 + counts), fitted on training sessions
  lightgbm    supervised gradient boosting on counts (a reference: it sees labels)

Protocols:
  chrono  the gold split (sessions ordered by first event, 60/10/30)
  random  the same proportions after a seeded shuffle, for the leakage gap

Every threshold and hyperparameter is chosen on validation; test is only scored.

Usage: python -m loglens.baselines hdfs
"""

from __future__ import annotations

import json
import sys
import time

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

from loglens import config, runlog
from loglens.metrics import evaluate, f1_at_prevalence, pr_curve_points, prf

SEED = 42
FRACTIONS = (0.6, 0.1, 0.3)
MODELS = ("warn_rule", "seq_lookup", "pca", "iforest", "lightgbm")


# ------------------------------------------------------------------ data ----


def load(dataset: str) -> pd.DataFrame:
    df = pd.read_parquet(
        config.GOLD / f"{dataset}_sessions",
        columns=[config.session_col(dataset), "counts", "n_events", "n_warn", "seq_hash", "count_hash", "split", "label", "order"],
    )
    return df.sort_values("order").reset_index(drop=True)


def random_split(n: int, seed: int = SEED) -> np.ndarray:
    rank = np.random.default_rng(seed).permutation(n)
    n_train, n_val = int(n * FRACTIONS[0]), int(n * FRACTIONS[1])
    return np.where(rank < n_train, "train", np.where(rank < n_train + n_val, "val", "test"))


# ---------------------------------------------------------------- models ----


def score_warn_rule(d: dict) -> tuple[np.ndarray, dict]:
    return d["n_warn"].astype(float), {}


def score_seq_lookup(d: dict) -> tuple[np.ndarray, dict]:
    tr = d["split"] == "train"
    rate = pd.Series(d["y"][tr]).groupby(d["seq_hash"][tr]).mean()
    scores = pd.Series(d["seq_hash"]).map(rate).fillna(1.0).to_numpy()
    return scores, {"train_distinct_sequences": int(rate.size)}


def score_pca(d: dict) -> tuple[np.ndarray, dict]:
    X, tr, va = d["X"], d["split"] == "train", d["split"] == "val"
    df_ = (X[tr] > 0).sum(axis=0)
    idf = np.log((1 + tr.sum()) / (1 + df_)) + 1.0  # smoothed, as in scikit-learn
    Z = X * idf
    mu = Z[tr].mean(axis=0)
    best = None
    for var in (0.9, 0.95, 0.99):
        pca = PCA(n_components=var, svd_solver="full", random_state=SEED).fit(Z[tr] - mu)
        P = pca.components_
        Zc = Z - mu
        resid = Zc - (Zc @ P.T) @ P
        s = np.einsum("ij,ij->i", resid, resid)
        ap = average_precision_score(d["y"][va], s[va])
        if best is None or ap > best[0]:
            best = (ap, s, {"variance_kept": var, "components": int(P.shape[0])})
    return best[1], best[2]


def score_iforest(d: dict) -> tuple[np.ndarray, dict]:
    X, tr = np.log1p(d["X"]), d["split"] == "train"
    model = IsolationForest(n_estimators=300, random_state=SEED, n_jobs=-1).fit(X[tr])
    return -model.score_samples(X), {"n_estimators": 300}


def score_lightgbm(d: dict) -> tuple[np.ndarray, dict]:
    X, y, tr, va = d["X"], d["y"], d["split"] == "train", d["split"] == "val"
    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        random_state=SEED,
        verbose=-1,
    )
    model.fit(
        X[tr], y[tr],
        eval_set=[(X[va], y[va])],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model.predict_proba(X)[:, 1], {"best_iteration": int(model.best_iteration_ or 0)}


SCORERS = {
    "warn_rule": score_warn_rule,
    "seq_lookup": score_seq_lookup,
    "pca": score_pca,
    "iforest": score_iforest,
    "lightgbm": score_lightgbm,
}


# -------------------------------------------------------------- analysis ----


def duplication(d: dict) -> dict:
    tr, te = d["split"] == "train", d["split"] == "test"
    seen_seq = np.isin(d["seq_hash"][te], np.unique(d["seq_hash"][tr]))
    seen_cnt = np.isin(d["count_hash"][te], np.unique(d["count_hash"][tr]))
    y_te = d["y"][te]

    def share(mask: np.ndarray, sel: np.ndarray) -> float:
        return round(float(mask[sel].mean()), 6) if sel.any() else None

    return {
        "test_sessions": int(te.sum()),
        "test_seq_seen_in_train": share(seen_seq, np.ones_like(y_te, bool)),
        "test_normal_seq_seen_in_train": share(seen_seq, y_te == 0),
        "test_anomalous_seq_seen_in_train": share(seen_seq, y_te == 1),
        "test_counts_seen_in_train": share(seen_cnt, np.ones_like(y_te, bool)),
        "test_anomalous_counts_seen_in_train": share(seen_cnt, y_te == 1),
        "unseen_test_sessions": int((~seen_seq).sum()),
        "unseen_test_anomalies": int(y_te[~seen_seq].sum()),
    }


def leakage_gap(chrono: dict, rand: dict) -> dict:
    """Random minus chronological F1, raw and at the chronological test's anomaly rate.

    The two test sets have different anomaly rates, which moves precision (and F1)
    even for an identical detector. The adjusted gap rescales the random-split
    result to the chronological prevalence, so what is left is the detector itself.
    """
    adjusted = f1_at_prevalence(rand["recall"], rand["fpr"], chrono["prevalence"])
    return {
        "chrono_f1": chrono["f1"],
        "random_f1": rand["f1"],
        "gap_points": round(100 * (rand["f1"] - chrono["f1"]), 2),
        "random_f1_at_chrono_prevalence": adjusted,
        "adjusted_gap_points": round(100 * (adjusted - chrono["f1"]), 2),
        "chrono_recall": chrono["recall"],
        "random_recall": rand["recall"],
        "chrono_fpr": chrono["fpr"],
        "random_fpr": rand["fpr"],
        "chrono_prevalence": chrono["prevalence"],
        "random_prevalence": rand["prevalence"],
    }


def label_conflicts(df: pd.DataFrame) -> dict:
    """Identical event sequences that carry both labels: no sequence model can get these all right."""
    g = df.groupby("seq_hash")["label"].agg(["nunique", "size"])
    conflicted = g[g["nunique"] > 1]
    in_conflict = df["seq_hash"].isin(conflicted.index)
    return {
        "distinct_sequences": int(g.shape[0]),
        "conflicting_sequences": int(conflicted.shape[0]),
        "sessions_in_conflicting_sequences": int(in_conflict.sum()),
        "anomalies_in_conflicting_sequences": int(df.loc[in_conflict, "label"].sum()),
    }


# ------------------------------------------------------------------- run ----


def run(dataset: str) -> None:
    timings: dict = {}
    df = load(dataset)
    X = np.stack(df["counts"].to_numpy()).astype(np.float32)
    base = {
        "X": X,
        "y": df["label"].to_numpy().astype(int),
        "n_warn": df["n_warn"].to_numpy(),
        "seq_hash": df["seq_hash"].to_numpy(),
        "count_hash": df["count_hash"].to_numpy(),
    }
    protocols = {"chrono": df["split"].to_numpy(), "random": random_split(len(df))}

    mlflow.set_tracking_uri(f"file:{config.ROOT / 'mlruns'}")
    mlflow.set_experiment(f"{dataset}-baselines")

    results: dict = {}
    curves: dict = {}
    dup: dict = {}
    chrono_scores: dict = {}
    for proto, split in protocols.items():
        d = {**base, "split": split}
        va, te = split == "val", split == "test"
        dup[proto] = duplication(d)
        unseen = ~np.isin(d["seq_hash"], np.unique(d["seq_hash"][split == "train"]))
        results[proto] = {}
        for name in MODELS:
            start = time.perf_counter()
            scores, params = SCORERS[name](d)
            secs = time.perf_counter() - start
            m = evaluate(d["y"][va], scores[va], d["y"][te], scores[te])
            u = te & unseen
            m["unseen_only"] = {
                **prf(d["y"][u], (scores[u] >= m["threshold"]).astype(int)),
                "pr_auc": round(float(average_precision_score(d["y"][u], scores[u])), 6) if d["y"][u].sum() else None,
            }
            m["fit_and_score_s"] = round(secs, 2)
            m["params"] = params
            results[proto][name] = m
            if proto == "chrono":
                curves[name] = pr_curve_points(d["y"][te], scores[te])
                chrono_scores[name] = scores
            print(
                f"{proto:6s} {name:10s} P={m['precision']:.3f} R={m['recall']:.3f} "
                f"F1={m['f1']:.3f} PR-AUC={m['pr_auc']:.3f} | unseen F1={m['unseen_only']['f1']:.3f} "
                f"({secs:.1f}s)",
                flush=True,
            )
            with mlflow.start_run(run_name=f"{name}-{proto}"):
                mlflow.set_tags({"model": name, "protocol": proto, "dataset": dataset})
                mlflow.log_params({"model": name, "protocol": proto, **params})
                mlflow.log_metrics(
                    {
                        "test_precision": m["precision"],
                        "test_recall": m["recall"],
                        "test_f1": m["f1"],
                        "test_pr_auc": m["pr_auc"] or 0.0,
                        "val_f1": m["val"]["f1"],
                        "unseen_test_f1": m["unseen_only"]["f1"],
                        "fit_and_score_s": m["fit_and_score_s"],
                    }
                )
        timings[proto] = {n: results[proto][n]["fit_and_score_s"] for n in MODELS}

    gap = {n: leakage_gap(results["chrono"][n], results["random"][n]) for n in MODELS}
    conflicts = label_conflicts(df)
    summary = {"results": results, "leakage_gap": gap, "duplication": dup, "label_conflicts": conflicts}
    runlog.write(dataset, "baselines", {**summary, "timings": timings})
    (config.DASHBOARD / f"{dataset}_baselines.json").write_text(
        json.dumps({**summary, "pr_curves_chrono_test": curves}, indent=2)
    )
    # Per-session scores under the chronological protocol, for intervals and operational metrics.
    col = config.session_col(dataset)
    pd.DataFrame({col: df[col], **chrono_scores}).to_parquet(
        config.GOLD / f"{dataset}_baseline_scores.parquet", index=False
    )


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "hdfs")
