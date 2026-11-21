"""DeepLog-style LSTM (Du et al. 2017), trained on normal training sessions only.

For every event in a session the model sees the previous h=10 events (padded
at the start, so the first events are predicted too) and ranks all possible
next events. A session is anomalous at top-g if any real next event is not
among the g most likely. That makes the session score its worst rank:
flagged at g  <=>  max_rank > g, so choosing g on validation is choosing a
threshold on max_rank. An event the parser could not match (UNK) is always
outside the top-g. A second, continuous score is the session's largest
negative log-probability, which gives the PR curve more resolution.

Shortcuts that do not change what is being estimated:
  training  identical (window, next event) pairs are merged and counted. Each
            batch draws pairs with probability proportional to their count,
            which is the same distribution as drawing from every training
            window (7.1M for HDFS), at a fraction of the cost. (Weighting the
            loss instead gives the same expectation but very noisy batches,
            because one pair can stand for 345K windows.)
  scoring   each distinct event sequence is scored once and mapped back to
            all sessions that share it

Tuning: --tune searches history length and hidden size for the END variant on
the chronological validation period only; the chosen setting then runs as an
extra variant next to the published one.

Usage: python -m loglens.lstm hdfs [--tune]
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter

import mlflow
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import average_precision_score

from loglens import config, runlog
from loglens.baselines import leakage_gap, random_split
from loglens.gold import vocabulary
from loglens.metrics import best_f1_threshold, evaluate, pr_curve_points, prf

H = 10  # history window, as published in DeepLog
PAD = 0
UNK_RANK = 999  # an UNK event is outside every top-g
SEEDS = (0, 1, 2)
EMBED, HIDDEN, LAYERS = 32, 64, 2  # hidden size and layers as published
# Early stopping ends almost every run well before the cap; it is only a safety net.
BATCH, LR, MAX_EPOCHS, PATIENCE = 1024, 1e-3, 60, 4
SAMPLES_PER_EPOCH = 128 * BATCH  # windows drawn per epoch, proportional to their counts


# -------------------------------------------------------------- encoding ----


def encoder(vocab: list[str]) -> dict[str, int]:
    """Event ID -> class index; 0 is padding, so classes are 1..len(vocab)."""
    return {e: i + 1 for i, e in enumerate(vocab)}


def windows(seq: list[int], h: int = H) -> tuple[np.ndarray, np.ndarray]:
    """Every (previous h events, next event) pair, left-padded so every event is a target."""
    padded = [PAD] * h + list(seq)
    X = np.array([padded[t : t + h] for t in range(len(seq))], dtype=np.int64).reshape(-1, h)
    y = np.array(seq, dtype=np.int64)
    return X, y


def weighted_windows(
    seqs: list[list[int]], weights: list[int], h: int = H
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge identical (window, next) pairs, weighting each by the windows it stands for."""
    counter: Counter = Counter()
    for seq, w in zip(seqs, weights):
        X, y = windows(seq, h)
        for xi, yi in zip(map(tuple, X), y):
            counter[(xi, int(yi))] += w
    keys = list(counter)
    X = np.array([k[0] for k in keys], dtype=np.int64).reshape(-1, h)
    y = np.array([k[1] for k in keys], dtype=np.int64)
    w = np.array([counter[k] for k in keys], dtype=np.float64)
    return X, y, w


# ----------------------------------------------------------------- model ----


class DeepLog(nn.Module):
    def __init__(self, n_classes: int, hidden: int = HIDDEN):
        super().__init__()
        self.embed = nn.Embedding(n_classes, EMBED, padding_idx=PAD)
        self.lstm = nn.LSTM(EMBED, hidden, num_layers=LAYERS, batch_first=True)
        self.out = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(self.embed(x))
        logits = self.out(h[:, -1, :])
        logits[:, PAD] = float("-inf")  # padding is never a prediction
        return logits


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def weighted_loss(model, X, y, w) -> float:
    model.eval()
    total, wsum = 0.0, 0.0
    with torch.no_grad():
        for i in range(0, len(X), 8192):
            xb, yb, wb = (torch.from_numpy(a[i : i + 8192]) for a in (X, y, w))
            ce = nn.functional.cross_entropy(model(xb), yb, reduction="none")
            total += float((ce.double() * wb).sum())
            wsum += float(wb.sum())
    return total / wsum


def train(train_w, val_w, n_classes: int, seed: int, hidden: int = HIDDEN) -> tuple[DeepLog, dict]:
    seed_everything(seed)
    X, y, w = train_w
    model = DeepLog(n_classes, hidden)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    p = w / w.sum()
    rng = np.random.default_rng(seed)
    best, best_state, bad, history = float("inf"), None, 0, []
    for epoch in range(MAX_EPOCHS):
        model.train()
        draws = rng.choice(len(X), size=SAMPLES_PER_EPOCH, p=p)
        for i in range(0, len(draws), BATCH):
            idx = draws[i : i + BATCH]
            xb, yb = torch.from_numpy(X[idx]), torch.from_numpy(y[idx])
            loss = nn.functional.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        val = weighted_loss(model, *val_w)
        history.append(round(val, 5))
        if val < best - 1e-5:
            best, bad = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, {"epochs": len(history), "best_val_loss": round(best, 5), "val_loss_history": history}


def rank_sequences(model: DeepLog, seqs: list[list[int]], unk_class: int, h: int = H) -> list[dict]:
    """Per sequence: rank (1 = most likely) and -log p of every real next event."""
    model.eval()
    lens = [len(s) for s in seqs]
    X = np.concatenate([windows(s, h)[0] for s in seqs])
    y = np.concatenate([windows(s, h)[1] for s in seqs])
    ranks = np.empty(len(y), dtype=np.int64)
    nll = np.empty(len(y), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(X), 16384):
            logits = model(torch.from_numpy(X[i : i + 16384]))
            yb = torch.from_numpy(y[i : i + 16384])
            target = logits.gather(1, yb[:, None])
            ranks[i : i + 16384] = (logits > target).sum(1).numpy() + 1
            nll[i : i + 16384] = (-torch.log_softmax(logits, 1).gather(1, yb[:, None])[:, 0]).numpy()
    ranks[y == unk_class] = UNK_RANK
    nll[y == unk_class] = 50.0
    out, start = [], 0
    for n in lens:
        out.append({"ranks": ranks[start : start + n], "nll": nll[start : start + n]})
        start += n
    return out


# ------------------------------------------------------------------- run ----

PUBLISHED = {"h": H, "hidden": HIDDEN}
VARIANTS = {
    # DeepLog as published: only the events present in a session are judged.
    "deeplog": {"end": False, **PUBLISHED},
    # Plus an END event after the last real event, so a session that stops
    # where END is unlikely (a truncated session) can be flagged too.
    "deeplog_end": {"end": True, **PUBLISHED},
}
# Searched for the END variant on the chronological validation period only.
TUNE_GRID = [{"h": h, "hidden": hidden} for h in (5, 10, 20) for hidden in (64, 128)]


def stat(values: list[float]) -> dict:
    return {"mean": round(float(np.mean(values)), 6), "sd": round(float(np.std(values, ddof=1)), 6)}


def build_ctx(dataset: str) -> dict:
    vocab = vocabulary(dataset)
    enc = encoder(vocab + ["END"])
    df = pd.read_parquet(
        config.GOLD / f"{dataset}_sessions",
        columns=["block_id", "events", "seq_hash", "split", "label", "order"],
    ).sort_values("order").reset_index(drop=True)
    first = df.drop_duplicates("seq_hash")  # distinct sequences, encoded once
    return {
        "dataset": dataset,
        "y": df["label"].to_numpy().astype(int),
        "seq_hash": df["seq_hash"].to_numpy(),
        "protocols": {"chrono": df["split"].to_numpy(), "random": random_split(len(df))},
        "distinct": {h: [enc[e] for e in ev] for h, ev in zip(first["seq_hash"], first["events"])},
        "n_classes": len(enc) + 1,
        "unk_class": enc["UNK"],
        "end_class": enc["END"],
    }


def sequences(ctx: dict, end: bool) -> dict:
    return {h: s + [ctx["end_class"]] if end else s for h, s in ctx["distinct"].items()}


def fit_and_score(ctx: dict, distinct: dict, split: np.ndarray, cfg: dict, seed: int) -> dict:
    """Train on normal training sessions, then score every session two ways."""
    seq_hash, normal = ctx["seq_hash"], ctx["y"] == 0

    def weighted(mask: np.ndarray):
        counts = pd.Series(seq_hash[mask]).value_counts()
        return weighted_windows([distinct[h] for h in counts.index], counts.tolist(), cfg["h"])

    train_w = weighted((split == "train") & normal)
    val_w = weighted((split == "val") & normal)
    t0 = time.perf_counter()
    model, info = train(train_w, val_w, ctx["n_classes"], seed, cfg["hidden"])
    t_train = time.perf_counter() - t0

    hashes = list(distinct)
    seqs = [distinct[h] for h in hashes]
    t1 = time.perf_counter()
    ranked = rank_sequences(model, seqs, ctx["unk_class"], cfg["h"])
    t_score = time.perf_counter() - t1

    by_hash = dict(zip(hashes, ranked))
    max_rank = pd.Series({h: int(r["ranks"].max()) for h, r in by_hash.items()})
    max_nll = pd.Series({h: float(r["nll"].max()) for h, r in by_hash.items()})
    return {
        "s_rank": pd.Series(seq_hash).map(max_rank).to_numpy().astype(float),
        "s_nll": pd.Series(seq_hash).map(max_nll).to_numpy(),
        "by_hash": by_hash,
        "info": info,
        "train_s": round(t_train, 1),
        "score_s": round(t_score, 2),
        "windows_scored_per_s": round(sum(len(s) for s in seqs) / t_score),
        "train_unique_windows": int(len(train_w[0])),
        "train_windows": int(train_w[2].sum()),
    }


# Checkpoints: each finished tuning setting and each finished (variant, protocol,
# seed) is saved as it completes, so an interrupted run resumes where it stopped.
# They are deleted once a run completes, so a later rerun always starts fresh;
# --fresh clears them by hand (needed if the gold layer changed mid-run).
CHECKPOINTS = config.RUNLOG / "lstm_checkpoints"


def checkpoint_load(key: str) -> dict | None:
    path = CHECKPOINTS / f"{key}.json"
    return json.loads(path.read_text()) if path.exists() else None


def checkpoint_save(key: str, obj: dict) -> None:
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINTS / f"{key}.json.tmp"
    tmp.write_text(json.dumps(obj))
    tmp.replace(CHECKPOINTS / f"{key}.json")  # atomic, so a crash never leaves half a file


def checkpoint_clear(prefix: str = "") -> None:
    if CHECKPOINTS.exists():
        for path in CHECKPOINTS.glob(f"{prefix}*.json"):
            path.unlink()


def tune(dataset: str) -> None:
    """Pick history length and hidden size for the END variant on validation only."""
    ctx = build_ctx(dataset)
    split = ctx["protocols"]["chrono"]
    distinct = sequences(ctx, end=True)
    y, va = ctx["y"], split == "val"
    rows = []
    for cfg in TUNE_GRID:
        key = f"tune_{dataset}_h{cfg['h']}_hidden{cfg['hidden']}"
        row = checkpoint_load(key)
        if row is not None:
            rows.append(row)
            print(f"tune h={cfg['h']:2d} hidden={cfg['hidden']:3d}: resumed from checkpoint", flush=True)
            continue
        r = fit_and_score(ctx, distinct, split, cfg, SEEDS[0])
        row = {**cfg, "epochs": r["info"]["epochs"], "best_val_loss": r["info"]["best_val_loss"], "train_s": r["train_s"]}
        for name, s in (("nll", r["s_nll"]), ("rank", r["s_rank"])):
            thr = best_f1_threshold(y[va], s[va])
            row[f"val_{name}_pr_auc"] = round(float(average_precision_score(y[va], s[va])), 6)
            row[f"val_{name}_f1"] = prf(y[va], (s[va] >= thr).astype(int))["f1"]
        checkpoint_save(key, row)
        rows.append(row)
        print(
            f"tune h={cfg['h']:2d} hidden={cfg['hidden']:3d}: val PR-AUC (log-prob)={row['val_nll_pr_auc']:.4f} "
            f"val F1 (log-prob)={row['val_nll_f1']:.4f} val F1 (top-g)={row['val_rank_f1']:.4f} "
            f"loss={row['best_val_loss']:.4f} epochs={row['epochs']} {row['train_s']:.0f}s",
            flush=True,
        )
    # Highest validation PR-AUC wins; on a tie, the smaller model.
    best = max(rows, key=lambda r: (r["val_nll_pr_auc"], -r["hidden"], -r["h"]))
    chosen = {"h": best["h"], "hidden": best["hidden"]}
    runlog.write(dataset, "lstm_tune", {"chosen": chosen, "chosen_by": "validation PR-AUC of the log-probability score (END variant, chronological split, seed 0)", "grid": rows})
    (config.DASHBOARD / f"{dataset}_lstm_tune.json").write_text(json.dumps({"chosen": chosen, "grid": rows}, indent=2))
    checkpoint_clear(f"tune_{dataset}_")


def tuned_config(dataset: str) -> dict | None:
    path = config.RUNLOG / f"{dataset}_lstm_tune.json"
    return json.loads(path.read_text())["chosen"] if path.exists() else None


def run_variant(variant: str, cfg: dict, ctx: dict) -> dict:
    y_all, seq_hash = ctx["y"], ctx["seq_hash"]
    distinct = sequences(ctx, cfg["end"])

    results: dict = {}
    curves: dict = {}
    per_sequence: dict = {}
    for proto, split in ctx["protocols"].items():
        va, te = split == "val", split == "test"
        unseen = ~np.isin(seq_hash, np.unique(seq_hash[split == "train"]))
        results[proto] = {"seeds": {}}

        for seed in SEEDS:
            key = f"run_{ctx['dataset']}_{variant}_{proto}_seed{seed}_h{cfg['h']}_hidden{cfg['hidden']}_end{int(cfg['end'])}"
            saved = checkpoint_load(key)
            if saved is not None:
                results[proto]["seeds"][seed] = saved["m"]
                results[proto]["train_unique_windows"] = saved["train_unique_windows"]
                results[proto]["train_windows"] = saved["train_windows"]
                if saved.get("curves"):
                    curves.update(saved["curves"])
                    per_sequence = saved["per_sequence"]
                print(f"{variant} {proto} seed={seed}: resumed from checkpoint", flush=True)
                continue

            r = fit_and_score(ctx, distinct, split, cfg, seed)
            s_rank, s_nll, info = r["s_rank"], r["s_nll"], r["info"]
            results[proto]["train_unique_windows"] = r["train_unique_windows"]
            results[proto]["train_windows"] = r["train_windows"]
            new_curves: dict = {}

            m = evaluate(y_all[va], s_rank[va], y_all[te], s_rank[te])
            m["g"] = int(m["threshold"]) - 1  # flagged when max_rank >= threshold, i.e. > g
            m_nll = evaluate(y_all[va], s_nll[va], y_all[te], s_nll[te])
            u = te & unseen
            m["unseen_only"] = {
                **prf(y_all[u], (s_rank[u] >= m["threshold"]).astype(int)),
                "pr_auc": round(float(average_precision_score(y_all[u], s_rank[u])), 6),
            }
            m["nll_score"] = {k: m_nll[k] for k in ("precision", "recall", "f1", "fpr", "pr_auc", "prevalence", "threshold")}
            m["nll_score"]["unseen_f1"] = prf(y_all[u], (s_nll[u] >= m_nll["threshold"]).astype(int))["f1"]
            m["train"] = info
            m["train_s"] = r["train_s"]
            m["score_s"] = r["score_s"]
            m["windows_scored_per_s"] = r["windows_scored_per_s"]
            results[proto]["seeds"][seed] = m
            print(
                f"{variant} {proto} seed={seed} g={m['g']} P={m['precision']:.3f} R={m['recall']:.3f} "
                f"F1={m['f1']:.3f} PR-AUC={m['pr_auc']:.3f} | log-prob F1={m_nll['f1']:.3f} "
                f"PR-AUC={m_nll['pr_auc']:.3f} | unseen F1={m['unseen_only']['f1']:.3f} | "
                f"epochs={info['epochs']} train={r['train_s']:.0f}s",
                flush=True,
            )
            if proto == "chrono" and seed == SEEDS[0]:
                new_curves = {
                    variant: pr_curve_points(y_all[te], s_rank[te]),
                    f"{variant}_nll": pr_curve_points(y_all[te], s_nll[te]),
                }
                curves.update(new_curves)
                per_sequence = {
                    str(h): {"ranks": v["ranks"].tolist(), "nll": np.round(v["nll"], 3).tolist()}
                    for h, v in r["by_hash"].items()
                }
            with mlflow.start_run(run_name=f"{variant}-{proto}-seed{seed}"):
                mlflow.set_tags({"model": variant, "protocol": proto, "dataset": ctx["dataset"]})
                mlflow.log_params(
                    {"model": variant, "protocol": proto, "seed": seed, "h": cfg["h"], "hidden": cfg["hidden"],
                     "g": m["g"], "end_event": cfg["end"], "embed": EMBED, "layers": LAYERS,
                     "epochs": info["epochs"]}
                )
                mlflow.log_metrics(
                    {"test_precision": m["precision"], "test_recall": m["recall"], "test_f1": m["f1"],
                     "test_pr_auc": m["pr_auc"], "test_fpr": m["fpr"], "val_f1": m["val"]["f1"],
                     "unseen_test_f1": m["unseen_only"]["f1"], "nll_test_f1": m_nll["f1"],
                     "nll_test_pr_auc": m_nll["pr_auc"], "train_s": m["train_s"]}
                )
            checkpoint_save(
                key,
                {
                    "m": m,
                    "train_unique_windows": r["train_unique_windows"],
                    "train_windows": r["train_windows"],
                    "curves": new_curves,
                    "per_sequence": per_sequence if new_curves else None,
                },
            )

        seeds = list(results[proto]["seeds"].values())
        summary = {k: stat([s[k] for s in seeds]) for k in ("precision", "recall", "f1", "pr_auc", "fpr")}
        summary["unseen_f1"] = stat([s["unseen_only"]["f1"] for s in seeds])
        for k in ("precision", "recall", "f1", "pr_auc", "fpr", "unseen_f1"):
            summary[f"nll_{k}"] = stat([s["nll_score"][k] for s in seeds])
        summary["g"] = [s["g"] for s in seeds]
        summary["epochs"] = [s["train"]["epochs"] for s in seeds]
        results[proto]["summary"] = summary

    # Paired by seed: same seed under each protocol, for both scores.
    keys = ("chrono_f1", "random_f1", "gap_points", "random_f1_at_chrono_prevalence", "adjusted_gap_points")
    per_seed = [leakage_gap(results["chrono"]["seeds"][s], results["random"]["seeds"][s]) for s in SEEDS]
    per_seed_nll = [
        leakage_gap(results["chrono"]["seeds"][s]["nll_score"], results["random"]["seeds"][s]["nll_score"])
        for s in SEEDS
    ]
    gap = {k: round(float(np.mean([g[k] for g in per_seed])), 6) for k in keys}
    gap["nll"] = {k: round(float(np.mean([g[k] for g in per_seed_nll])), 6) for k in keys}
    gap["per_seed"] = per_seed
    return {"results": results, "leakage_gap": gap, "curves": curves, "per_sequence": per_sequence, "config": cfg}


def run(dataset: str) -> None:
    # PyTorch defaults to one thread per physical core, which beats using hyperthreads too.
    ctx = build_ctx(dataset)
    variants = dict(VARIANTS)
    tuned = tuned_config(dataset)
    if tuned and tuned != PUBLISHED:
        variants["deeplog_end_tuned"] = {"end": True, **tuned}

    mlflow.set_tracking_uri(f"file:{config.ROOT / 'mlruns'}")
    mlflow.set_experiment(f"{dataset}-baselines")

    out = {v: run_variant(v, cfg, ctx) for v, cfg in variants.items()}

    log = {v: {"config": o["config"], "results": o["results"], "leakage_gap": o["leakage_gap"]} for v, o in out.items()}
    runlog.write(dataset, "lstm", {**log, "seeds": list(SEEDS), "tuned": tuned})
    export = {
        v: {
            "config": o["config"],
            "results": o["results"],
            "leakage_gap": o["leakage_gap"],
            "pr_curves_chrono_test": o["curves"],
        }
        for v, o in out.items()
    }
    (config.DASHBOARD / f"{dataset}_lstm.json").write_text(json.dumps(export, indent=2))
    # Per-sequence ranks (chronological protocol, first seed) for latency analysis and the drill-down.
    for v, o in out.items():
        (config.GOLD / f"{dataset}_{v}_ranks.json").write_text(json.dumps(o["per_sequence"]))
    checkpoint_clear(f"run_{dataset}_")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ds = args[0] if args else "hdfs"
    if "--fresh" in sys.argv:
        checkpoint_clear()
    tune(ds) if "--tune" in sys.argv else run(ds)
