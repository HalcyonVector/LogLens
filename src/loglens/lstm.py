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

Usage: python -m loglens.lstm hdfs
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
from loglens.metrics import evaluate, pr_curve_points, prf

H = 10  # history window
PAD = 0
UNK_RANK = 999  # an UNK event is outside every top-g
SEEDS = (0, 1, 2)
EMBED, HIDDEN, LAYERS = 32, 64, 2
BATCH, LR, MAX_EPOCHS, PATIENCE = 1024, 1e-3, 30, 4
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


def weighted_windows(seqs: list[list[int]], weights: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge identical (window, next) pairs, weighting each by the windows it stands for."""
    counter: Counter = Counter()
    for seq, w in zip(seqs, weights):
        X, y = windows(seq)
        for xi, yi in zip(map(tuple, X), y):
            counter[(xi, int(yi))] += w
    keys = list(counter)
    X = np.array([k[0] for k in keys], dtype=np.int64).reshape(-1, H)
    y = np.array([k[1] for k in keys], dtype=np.int64)
    w = np.array([counter[k] for k in keys], dtype=np.float64)
    return X, y, w


# ----------------------------------------------------------------- model ----


class DeepLog(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.embed = nn.Embedding(n_classes, EMBED, padding_idx=PAD)
        self.lstm = nn.LSTM(EMBED, HIDDEN, num_layers=LAYERS, batch_first=True)
        self.out = nn.Linear(HIDDEN, n_classes)

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


def train(train_w, val_w, n_classes: int, seed: int) -> tuple[DeepLog, dict]:
    seed_everything(seed)
    X, y, w = train_w
    model = DeepLog(n_classes)
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


def rank_sequences(model: DeepLog, seqs: list[list[int]], unk_class: int) -> list[dict]:
    """Per sequence: rank (1 = most likely) and -log p of every real next event."""
    model.eval()
    lens = [len(s) for s in seqs]
    X = np.concatenate([windows(s)[0] for s in seqs])
    y = np.concatenate([windows(s)[1] for s in seqs])
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

VARIANTS = {
    # DeepLog as published: only the events present in a session are judged.
    "deeplog": False,
    # Plus an END event after the last real event, so a session that stops
    # where END is unlikely (a truncated session) can be flagged too.
    "deeplog_end": True,
}


def stat(values: list[float]) -> dict:
    return {"mean": round(float(np.mean(values)), 6), "sd": round(float(np.std(values, ddof=1)), 6)}


def run_variant(variant: str, end: bool, ctx: dict) -> dict:
    y_all, seq_hash, protocols = ctx["y"], ctx["seq_hash"], ctx["protocols"]
    end_class = ctx["end_class"]
    distinct = {h: s + [end_class] if end else s for h, s in ctx["distinct"].items()}
    hashes = list(distinct)
    seqs = [distinct[h] for h in hashes]

    results: dict = {}
    curves: dict = {}
    per_sequence: dict = {}
    for proto, split in protocols.items():
        normal = y_all == 0

        def weighted(mask: np.ndarray):
            counts = pd.Series(seq_hash[mask]).value_counts()
            return weighted_windows([distinct[h] for h in counts.index], counts.tolist())

        train_w = weighted((split == "train") & normal)
        val_w = weighted((split == "val") & normal)
        va, te = split == "val", split == "test"
        unseen = ~np.isin(seq_hash, np.unique(seq_hash[split == "train"]))
        results[proto] = {
            "seeds": {},
            "train_unique_windows": int(len(train_w[0])),
            "train_windows": int(train_w[2].sum()),
        }
        print(
            f"{variant} {proto}: {len(train_w[0]):,} unique training windows "
            f"standing for {int(train_w[2].sum()):,}",
            flush=True,
        )

        for seed in SEEDS:
            t0 = time.perf_counter()
            model, info = train(train_w, val_w, ctx["n_classes"], seed)
            t_train = time.perf_counter() - t0
            t1 = time.perf_counter()
            ranked = rank_sequences(model, seqs, ctx["unk_class"])
            t_score = time.perf_counter() - t1

            by_hash = {h: r for h, r in zip(hashes, ranked)}
            max_rank = pd.Series({h: int(r["ranks"].max()) for h, r in by_hash.items()})
            max_nll = pd.Series({h: float(r["nll"].max()) for h, r in by_hash.items()})
            s_rank = pd.Series(seq_hash).map(max_rank).to_numpy().astype(float)
            s_nll = pd.Series(seq_hash).map(max_nll).to_numpy()

            m = evaluate(y_all[va], s_rank[va], y_all[te], s_rank[te])
            m["g"] = int(m["threshold"]) - 1  # flagged when max_rank >= threshold, i.e. > g
            m_nll = evaluate(y_all[va], s_nll[va], y_all[te], s_nll[te])
            u = te & unseen
            m["unseen_only"] = {
                **prf(y_all[u], (s_rank[u] >= m["threshold"]).astype(int)),
                "pr_auc": round(float(average_precision_score(y_all[u], s_rank[u])), 6),
            }
            m["nll_score"] = {k: m_nll[k] for k in ("precision", "recall", "f1", "fpr", "pr_auc")}
            m["train"] = info
            m["train_s"] = round(t_train, 1)
            m["score_s"] = round(t_score, 2)
            m["windows_scored_per_s"] = round(sum(len(s) for s in seqs) / t_score)
            results[proto]["seeds"][seed] = m
            print(
                f"{variant} {proto} seed={seed} g={m['g']} P={m['precision']:.3f} R={m['recall']:.3f} "
                f"F1={m['f1']:.3f} PR-AUC={m['pr_auc']:.3f} | nll F1={m_nll['f1']:.3f} "
                f"PR-AUC={m_nll['pr_auc']:.3f} | unseen F1={m['unseen_only']['f1']:.3f} | "
                f"epochs={info['epochs']} train={t_train:.0f}s",
                flush=True,
            )
            if proto == "chrono" and seed == SEEDS[0]:
                curves[variant] = pr_curve_points(y_all[te], s_rank[te])
                curves[f"{variant}_nll"] = pr_curve_points(y_all[te], s_nll[te])
                per_sequence = {
                    str(h): {"ranks": r["ranks"].tolist(), "nll": np.round(r["nll"], 3).tolist()}
                    for h, r in by_hash.items()
                }
            with mlflow.start_run(run_name=f"{variant}-{proto}-seed{seed}"):
                mlflow.set_tags({"model": variant, "protocol": proto, "dataset": ctx["dataset"]})
                mlflow.log_params(
                    {"model": variant, "protocol": proto, "seed": seed, "h": H, "g": m["g"], "end_event": end,
                     "embed": EMBED, "hidden": HIDDEN, "layers": LAYERS, "epochs": info["epochs"]}
                )
                mlflow.log_metrics(
                    {"test_precision": m["precision"], "test_recall": m["recall"], "test_f1": m["f1"],
                     "test_pr_auc": m["pr_auc"], "test_fpr": m["fpr"], "val_f1": m["val"]["f1"],
                     "unseen_test_f1": m["unseen_only"]["f1"], "nll_test_f1": m_nll["f1"],
                     "nll_test_pr_auc": m_nll["pr_auc"], "train_s": m["train_s"]}
                )

        seeds = list(results[proto]["seeds"].values())
        summary = {k: stat([s[k] for s in seeds]) for k in ("precision", "recall", "f1", "pr_auc", "fpr")}
        summary["unseen_f1"] = stat([s["unseen_only"]["f1"] for s in seeds])
        summary["nll_f1"] = stat([s["nll_score"]["f1"] for s in seeds])
        summary["nll_pr_auc"] = stat([s["nll_score"]["pr_auc"] for s in seeds])
        summary["g"] = [s["g"] for s in seeds]
        results[proto]["summary"] = summary

    # Paired by seed: same seed under each protocol.
    per_seed = [leakage_gap(results["chrono"]["seeds"][s], results["random"]["seeds"][s]) for s in SEEDS]
    keys = ("chrono_f1", "random_f1", "gap_points", "random_f1_at_chrono_prevalence", "adjusted_gap_points")
    gap = {k: round(float(np.mean([g[k] for g in per_seed])), 6) for k in keys}
    gap["per_seed"] = per_seed
    return {"results": results, "leakage_gap": gap, "curves": curves, "per_sequence": per_sequence, "end_event": end}


def run(dataset: str) -> None:
    # PyTorch defaults to one thread per physical core, which beats using hyperthreads too.
    vocab = vocabulary(dataset)
    enc = encoder(vocab + ["END"])

    df = pd.read_parquet(
        config.GOLD / f"{dataset}_sessions",
        columns=["block_id", "events", "seq_hash", "split", "label", "order"],
    ).sort_values("order").reset_index(drop=True)

    # Distinct sequences, encoded once.
    first = df.drop_duplicates("seq_hash")
    ctx = {
        "dataset": dataset,
        "y": df["label"].to_numpy().astype(int),
        "seq_hash": df["seq_hash"].to_numpy(),
        "protocols": {"chrono": df["split"].to_numpy(), "random": random_split(len(df))},
        "distinct": {h: [enc[e] for e in ev] for h, ev in zip(first["seq_hash"], first["events"])},
        "n_classes": len(enc) + 1,
        "unk_class": enc["UNK"],
        "end_class": enc["END"],
    }

    mlflow.set_tracking_uri(f"file:{config.ROOT / 'mlruns'}")
    mlflow.set_experiment(f"{dataset}-baselines")

    out = {v: run_variant(v, end, ctx) for v, end in VARIANTS.items()}

    log = {v: {"results": o["results"], "leakage_gap": o["leakage_gap"]} for v, o in out.items()}
    runlog.write(dataset, "lstm", {**log, "h": H, "seeds": list(SEEDS)})
    export = {
        v: {
            "results": o["results"],
            "leakage_gap": o["leakage_gap"],
            "pr_curves_chrono_test": o["curves"],
            "end_event": o["end_event"],
        }
        for v, o in out.items()
    }
    (config.DASHBOARD / f"{dataset}_lstm.json").write_text(json.dumps(export, indent=2))
    # Per-sequence ranks (chronological protocol, first seed) for latency analysis and the drill-down.
    for v, o in out.items():
        (config.GOLD / f"{dataset}_{v}_ranks.json").write_text(json.dumps(o["per_sequence"]))


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "hdfs")
