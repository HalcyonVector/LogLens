"""LogBERT-lite: a small masked transformer over event sequences (after Guo et al. 2021).

LogBERT reads a whole session at once. During training some events are hidden
behind a MASK token and the model predicts them from every other event, before
and after, on normal training sessions only (masked log-key prediction). A DIST
token at the front of each session is pulled towards the centre of all normal
sessions (the hypersphere loss), which LogBERT uses as a regulariser.

Scoring hides each event on its own and asks the model to predict it from the
rest, which gives the same two per-event numbers the LSTM produces: the rank of
the real event among all predictions, and its -log p. So the session scores are
the same as the LSTM's (worst rank, used as top-g; least likely event), and the
whole evaluation is shared with it (lstm.run_variant).

Because the model looks at events after the one it judges, it can only decide
once a session is over, like the count models.

This is a scaled-down LogBERT (2 layers of width 64, against 4 of width 256),
small enough to train on a laptop CPU in minutes.

Usage: python -m loglens.logbert hdfs
"""

from __future__ import annotations

import json
import sys
import time

import mlflow
import numpy as np
import pandas as pd
import torch
from torch import nn

from loglens import config, runlog
from loglens.lstm import PAD, SEEDS, UNK_RANK, build_ctx, checkpoint_clear, run_variant, seed_everything, sequences

D_MODEL, HEADS, LAYERS, FF, DROPOUT = 64, 4, 2, 128, 0.1
MAX_LEN = 512
MASK_RATE = 0.15  # share of events hidden per training session, as in BERT
VHM_WEIGHT = 0.1  # weight of the hypersphere loss on the DIST token
BATCH, LR, MAX_EPOCHS, PATIENCE = 128, 1e-3, 60, 4
SESSIONS_PER_EPOCH = 64 * BATCH  # drawn in proportion to how often each sequence occurs
SCORE_TOKENS = 1 << 16  # tokens per scoring batch

VARIANTS = {
    "logbert": {"end": False, "d_model": D_MODEL, "layers": LAYERS},
    "logbert_end": {"end": True, "d_model": D_MODEL, "layers": LAYERS},
}


class LogBERT(nn.Module):
    """Token ids: 0 padding, 1..n_classes-1 events, then MASK and DIST."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes
        self.mask_id, self.dist_id = n_classes, n_classes + 1
        self.tok = nn.Embedding(n_classes + 2, D_MODEL, padding_idx=PAD)
        self.pos = nn.Embedding(MAX_LEN, D_MODEL)
        layer = nn.TransformerEncoderLayer(D_MODEL, HEADS, FF, DROPOUT, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(D_MODEL)
        self.out = nn.Linear(D_MODEL, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(x.shape[1], device=x.device)
        h = self.tok(x) + self.pos(positions)[None]
        h = self.norm(self.enc(h, src_key_padding_mask=x == PAD))
        logits = self.out(h)
        logits[..., PAD] = float("-inf")  # padding is never a prediction
        return logits, h[:, 0]


def batch_tensor(seqs: list[list[int]], dist_id: int) -> torch.Tensor:
    """DIST token, then the events, right-padded to the longest sequence in the batch."""
    width = 1 + max(len(s) for s in seqs)
    x = np.full((len(seqs), width), PAD, dtype=np.int64)
    for i, s in enumerate(seqs):
        x[i, 0] = dist_id
        x[i, 1 : 1 + len(s)] = s
    return torch.from_numpy(x)


def mask_batch(x: torch.Tensor, mask_id: int, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Hide MASK_RATE of the real events (at least one per session); targets elsewhere are -100."""
    real = (x != PAD).numpy()
    real[:, 0] = False  # never the DIST token
    hide = (rng.random(real.shape) < MASK_RATE) & real
    for i in np.nonzero(~hide.any(axis=1))[0]:  # every session predicts something
        hide[i, rng.choice(np.nonzero(real[i])[0])] = True
    hide_t = torch.from_numpy(hide)
    target = torch.where(hide_t, x, torch.full_like(x, -100))
    return torch.where(hide_t, torch.full_like(x, mask_id), x), target


def masked_loss(model: LogBERT, x: torch.Tensor, target: torch.Tensor, center: torch.Tensor | None):
    logits, dist = model(x)
    ce = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=-100)
    if center is None:
        return ce
    return ce + VHM_WEIGHT * ((dist - center) ** 2).sum(1).mean()


def length_batches(idx: np.ndarray, lengths: np.ndarray, size: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Batches of similar length (less padding), in random order."""
    order = idx[np.argsort(lengths[idx], kind="stable")]
    batches = [order[i : i + size] for i in range(0, len(order), size)]
    rng.shuffle(batches)
    return batches


def dist_center(model: LogBERT, seqs: list[list[int]], lengths: np.ndarray, p: np.ndarray, rng) -> torch.Tensor:
    model.eval()
    draws = rng.choice(len(seqs), size=min(4096, 8 * len(seqs)), p=p)
    outs = []
    with torch.no_grad():
        for b in length_batches(draws, lengths, 512, rng):
            outs.append(model(batch_tensor([seqs[i] for i in b], model.dist_id))[1])
    return torch.cat(outs).mean(0)


def val_loss(model: LogBERT, seqs: list[list[int]], weights: np.ndarray, center: torch.Tensor) -> float:
    """Masked loss over normal validation sequences with a fixed mask, weighted by their counts."""
    model.eval()
    rng = np.random.default_rng(12345)  # the same hidden events every epoch
    lengths = np.array([len(s) for s in seqs])
    total = 0.0
    with torch.no_grad():
        for b in [np.argsort(lengths)[i : i + 512] for i in range(0, len(seqs), 512)]:
            x, target = mask_batch(batch_tensor([seqs[i] for i in b], model.dist_id), model.mask_id, rng)
            logits, _ = model(x)
            ce = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=-100, reduction="none"
            ).reshape(target.shape)
            per_seq = ce.sum(1) / (target != -100).sum(1)
            total += float((per_seq.double() * torch.from_numpy(weights[b])).sum())
    return total / float(weights.sum())


def train(train_seqs, train_w, val_seqs, val_w, n_classes: int, seed: int) -> tuple[LogBERT, dict]:
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    model = LogBERT(n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lengths = np.array([len(s) for s in train_seqs])
    p = train_w / train_w.sum()
    best, best_state, bad, history = float("inf"), None, 0, []
    for _epoch in range(MAX_EPOCHS):
        center = dist_center(model, train_seqs, lengths, p, rng)
        model.train()
        draws = rng.choice(len(train_seqs), size=SESSIONS_PER_EPOCH, p=p)
        for b in length_batches(draws, lengths, BATCH, rng):
            x, target = mask_batch(batch_tensor([train_seqs[i] for i in b], model.dist_id), model.mask_id, rng)
            loss = masked_loss(model, x, target, center)
            opt.zero_grad()
            loss.backward()
            opt.step()
        val = val_loss(model, val_seqs, val_w, center)
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


def score_sequences(model: LogBERT, seqs: list[list[int]], unk_class: int) -> list[dict]:
    """Hide each event on its own and predict it from all the others: its rank and -log p."""
    model.eval()
    pairs = [(i, j) for i, s in enumerate(seqs) for j in range(len(s))]
    lengths = np.array([len(seqs[i]) for i, _ in pairs])
    order = np.argsort(lengths, kind="stable")
    ranks = [np.empty(len(s), dtype=np.int64) for s in seqs]
    nll = [np.empty(len(s), dtype=np.float64) for s in seqs]
    with torch.no_grad():
        # Copies of one length go together, as many per batch as fit in SCORE_TOKENS.
        for length in np.unique(lengths):
            same = order[lengths[order] == length]
            per_batch = max(1, SCORE_TOKENS // (int(length) + 1))
            for chunk in (same[i : i + per_batch] for i in range(0, len(same), per_batch)):
                x = batch_tensor([seqs[pairs[k][0]] for k in chunk], model.dist_id)
                pos = torch.tensor([pairs[k][1] + 1 for k in chunk])
                rows = torch.arange(len(chunk))
                target = x[rows, pos].clone()
                x[rows, pos] = model.mask_id
                logits = model(x)[0][rows, pos]
                r = ((logits > logits.gather(1, target[:, None])).sum(1) + 1).numpy()
                q = (-torch.log_softmax(logits, 1).gather(1, target[:, None])[:, 0]).numpy()
                for n_, k in enumerate(chunk):
                    i_, j_ = pairs[k]
                    ranks[i_][j_], nll[i_][j_] = r[n_], q[n_]
    out = []
    for s, r, q in zip(seqs, ranks, nll):
        unk = np.asarray(s) == unk_class
        r[unk], q[unk] = UNK_RANK, 50.0  # an unknown event is always outside the top-g, as for the LSTM
        out.append({"ranks": r, "nll": q})
    return out


def fit_and_score(ctx: dict, distinct: dict, split: np.ndarray, cfg: dict, seed: int) -> dict:
    """Train on normal training sessions, then score every distinct sequence (the lstm.run_variant contract)."""
    seq_hash, normal = ctx["seq_hash"], ctx["y"] == 0

    def weighted(mask: np.ndarray):
        counts = pd.Series(seq_hash[mask]).value_counts()
        return [distinct[h] for h in counts.index], counts.to_numpy().astype(np.float64)

    train_seqs, train_w = weighted((split == "train") & normal)
    val_seqs, val_w = weighted((split == "val") & normal)
    t0 = time.perf_counter()
    model, info = train(train_seqs, train_w, val_seqs, val_w, ctx["n_classes"], seed)
    t_train = time.perf_counter() - t0

    hashes = list(distinct)
    seqs = [distinct[h] for h in hashes]
    t1 = time.perf_counter()
    scored = score_sequences(model, seqs, ctx["unk_class"])
    t_score = time.perf_counter() - t1

    by_hash = dict(zip(hashes, scored))
    max_rank = pd.Series({h: int(r["ranks"].max()) for h, r in by_hash.items()})
    max_nll = pd.Series({h: float(r["nll"].max()) for h, r in by_hash.items()})
    return {
        "s_rank": pd.Series(seq_hash).map(max_rank).to_numpy().astype(float),
        "s_nll": pd.Series(seq_hash).map(max_nll).to_numpy(),
        "by_hash": by_hash,
        "info": info,
        "model": model,
        "train_s": round(t_train, 1),
        "score_s": round(t_score, 2),
        "windows_scored_per_s": round(sum(len(s) for s in seqs) / t_score),
        "train_unique_windows": len(train_seqs),  # distinct training sequences
        "train_windows": int(train_w.sum()),  # training sessions they stand for
        "params": {"heads": HEADS, "ff": FF, "mask_rate": MASK_RATE, "vhm_weight": VHM_WEIGHT},
    }


def run(dataset: str) -> None:
    ctx = build_ctx(dataset)
    mlflow.set_tracking_uri(f"file:{config.ROOT / 'mlruns'}")
    mlflow.set_experiment(f"{dataset}-baselines")
    out = {v: run_variant(v, cfg, ctx, fit=fit_and_score) for v, cfg in VARIANTS.items()}
    runlog.write(dataset, "logbert", {**{v: {"config": o["config"], "results": o["results"], "leakage_gap": o["leakage_gap"]} for v, o in out.items()}, "seeds": list(SEEDS)})
    export = {
        v: {"config": o["config"], "results": o["results"], "leakage_gap": o["leakage_gap"], "pr_curves_chrono_test": o["curves"]}
        for v, o in out.items()
    }
    (config.DASHBOARD / f"{dataset}_logbert.json").write_text(json.dumps(export, indent=2))
    for v, o in out.items():
        (config.GOLD / f"{dataset}_{v}_ranks.json").write_text(json.dumps(o["per_sequence"]))
        checkpoint_clear(f"run_{dataset}_{v}_")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run(args[0] if args else "hdfs")
