"""Streaming replay of the test period with Spark Structured Streaming.

The offline evaluation judges finished sessions. A live system sees one line at
a time and never knows a session is over; it can only assume so after a spell of
silence (an idle timeout). This stage replays the raw HDFS log from the start of
the test period to its end as a stream and runs the detectors the way a live
system would:

  source    raw lines with their file position, one Parquet file per 5 minutes
            of log time, read one file per micro-batch
  parse     the batch pipeline's own line regex, masking and Spark SQL template
            matcher, an event-time watermark, and a join to the test blocks
  state     applyInPandasWithState keeps one state per block: its last events,
            its event counts and what has been decided so far
  decide    DeepLog (published, top-g) judges every event as it arrives; the
            tuned END LSTM judges every event, plus END when the block closes;
            PCA judges the block's count vector when it closes. A block closes
            after 5 minutes, 30 minutes, 2 hours or 12.3 hours of silence (each
            timeout tracked separately), and finally at the end of the log

Two sentinel lines after the end of the log move the watermark far enough for
every block to close. They name a block (blk_0) that is not in the test set: Spark
pushes the "has a block" filter below the watermark, so a sentinel without one
would be dropped before the watermark ever saw it. Each decision is written with the log time it was made, so
latency is measured in log time, as a live system would experience it.

Checks against the offline run: every test-session event arrives once, and the
verdicts made at the end of the log are the offline verdicts.

Usage: python -m loglens.stream hdfs [--prepare-only] [--fresh] [--analyse-only]
"""

from __future__ import annotations

import os

# One thread per Python worker for every maths library: Spark already runs one worker per
# task, and 16 workers each starting 16 BLAS threads starved the driver of CPU.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import io
import json
import pickle
import shutil
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from loglens import config, runlog

DATASET = "hdfs"
BIN_S = 300  # log seconds per replay file, and so per micro-batch
SHORT_TIMEOUTS_S = (300, 1800, 7200)  # plus the evaluation's 12.3 h timeout and the end of the log
END_OF_LOG = -1
UNK_RANK, UNK_NLL = 999, 50.0
TIE_TOLERANCE = 1e-4  # a session scoring this close to its threshold may flip on rounding alone
STREAM = config.DATA / "stream"
# Each Python worker holds two LSTMs and PyTorch (about 300 MB), so the stateful stage runs on
# fewer partitions than the batch stages, to fit the laptop's memory.
STATE_PARTITIONS = 8

STATE_FIELDS = ("first_ms", "last_ms", "n", "hist", "counts", "dl_ms", "end_max", "end_ms", "closed")
STATE_SCHEMA = (
    "first_ms long, last_ms long, n int, hist string, counts string, dl_ms long, end_max double, end_ms long, closed int"
)
OUT_SCHEMA = "block_id string, model string, timeout_s long, flagged int, decided_ms long, first_ms long, n_events int"


# --------------------------------------------------------- decision logic ----
# Pure Python, so it is tested without Spark (tests/test_stream.py).


class Models:
    """The three detectors, their validation-chosen thresholds and the event encoding."""

    def __init__(self, rank, nll, pca, thr_rank, thr_nll, thr_pca, h, unk, end, n_vocab):
        self.rank, self.nll, self.pca = rank, nll, pca  # rank(windows, targets), nll(windows, targets), pca(counts)
        self.thr_rank, self.thr_nll, self.thr_pca = thr_rank, thr_nll, thr_pca
        self.h, self.unk, self.end, self.n_vocab = h, unk, end, n_vocab


def new_state(n_vocab: int) -> dict:
    return {"first_ms": -1, "last_ms": -1, "n": 0, "hist": "", "counts": ",".join(["0"] * n_vocab),
            "dl_ms": -1, "end_max": -1.0, "end_ms": -1, "closed": 0}


def window(hist: list[int], h: int) -> list[int]:
    last = hist[-h:]
    return [0] * (h - len(last)) + last  # left-padded, as in training


def advance(st: dict, ts_ms, classes, slots, watermark_ms: int, flush_ms: int, log_end_ms: int,
            timeouts_ms: tuple, M: Models) -> list[tuple]:
    """Apply new events (in log order) and the watermark to one block; return its new decisions.

    A decision is (model, timeout_s, flagged, decided_ms, first_ms, n_events). For each finite
    timeout the block closes at last event + timeout, once the next event is later than that or
    the watermark passes it; events after a close no longer count for that timeout. The
    END_OF_LOG close happens when the watermark reaches `flush_ms`, and is dated at the end of
    the log.
    """
    hist = [int(c) for c in st["hist"].split(",") if c]
    counts = np.array([int(c) for c in st["counts"].split(",")], dtype=np.float64)
    out: list[tuple] = []
    n_t = len(timeouts_ms)

    def close(k: int, at_ms: int) -> None:
        nll_end = float(M.nll(np.array([window(hist, M.h)]), np.array([M.end]))[0])
        end_flag = st["end_max"] >= M.thr_nll or nll_end >= M.thr_nll
        end_at = st["end_ms"] if st["end_ms"] >= 0 else at_ms  # fired on an event, or on END now
        t_s = END_OF_LOG if k == n_t else timeouts_ms[k] // 1000
        out.append(("deeplog_end", t_s, int(end_flag), end_at if end_flag else at_ms, st["first_ms"], st["n"]))
        out.append(("pca", t_s, int(M.pca(counts) >= M.thr_pca), at_ms, st["first_ms"], st["n"]))
        if k == n_t and st["dl_ms"] < 0:  # DeepLog never fired: its verdict is "normal"
            out.append(("deeplog", 0, 0, at_ms, st["first_ms"], st["n"]))
        st["closed"] |= 1 << k

    def close_due(now_ms: int, strict: bool) -> None:
        for k, t in enumerate(timeouts_ms):
            due = st["last_ms"] + t
            if not st["closed"] >> k & 1 and (due < now_ms if strict else due <= now_ms):
                close(k, due)

    if len(classes):
        # Every new event's window is known up front, so both LSTMs score them in one call each.
        wins, h_ = [], list(hist)
        for c in classes:
            wins.append(window(h_, M.h))
            h_.append(int(c))
        wins, tg = np.array(wins), np.array(classes)
        ranks = M.rank(wins, tg).astype(float)
        nlls = M.nll(wins, tg).astype(float)
        ranks[tg == M.unk], nlls[tg == M.unk] = UNK_RANK, UNK_NLL
        for t, c, slot, r, q in zip(ts_ms, classes, slots, ranks, nlls):
            t = int(t)
            if st["n"]:
                close_due(t, strict=True)  # the silence before this event already closed the block
            else:
                st["first_ms"] = t
            if st["dl_ms"] < 0 and r >= M.thr_rank:
                st["dl_ms"] = t
                out.append(("deeplog", 0, 1, t, st["first_ms"], st["n"] + 1))
            st["end_max"] = max(st["end_max"], q)
            if st["end_ms"] < 0 and q >= M.thr_nll:
                st["end_ms"] = t
            counts[int(slot)] += 1
            hist.append(int(c))
            st["last_ms"], st["n"] = t, st["n"] + 1
    if st["n"]:
        close_due(watermark_ms, strict=False)
        if watermark_ms >= flush_ms and not st["closed"] >> n_t & 1:
            close(n_t, log_end_ms)
    st["hist"] = ",".join(map(str, hist[-M.h:]))
    st["counts"] = ",".join(str(int(c)) for c in counts)
    return out


def next_timeout(st: dict, timeouts_ms: tuple, flush_ms: int) -> int | None:
    """When the block next needs a look without new data, or None once everything is decided."""
    pending = [st["last_ms"] + t for k, t in enumerate(timeouts_ms) if not st["closed"] >> k & 1]
    if not st["closed"] >> len(timeouts_ms) & 1:
        pending.append(flush_ms)
    return min(pending) if pending else None


# ------------------------------------------------------------ the models ----


def fit_pca(dataset: str) -> dict:
    """Refit PCA exactly as the baseline did (training sessions, its chosen variance) and check it."""
    from sklearn.decomposition import PCA

    from loglens.baselines import SEED, load

    base = json.loads((config.DASHBOARD / f"{dataset}_baselines.json").read_text())["results"]["chrono"]["pca"]
    df = load(dataset)
    X = np.stack(df["counts"].to_numpy()).astype(np.float64)
    tr = (df["split"] == "train").to_numpy()
    idf = np.log((1 + tr.sum()) / (1 + (X[tr] > 0).sum(axis=0))) + 1.0
    Z = X * idf
    mu = Z[tr].mean(axis=0)
    P = PCA(n_components=base["params"]["variance_kept"], svd_solver="full", random_state=SEED).fit(Z[tr] - mu).components_
    Zc = Z - mu
    resid = Zc - (Zc @ P.T) @ P
    s = np.einsum("ij,ij->i", resid, resid)
    saved = pd.read_parquet(config.GOLD / f"{dataset}_baseline_scores.parquet", columns=[config.session_col(dataset), "pca"])
    saved = saved.set_index(config.session_col(dataset))["pca"].reindex(df[config.session_col(dataset)]).to_numpy()
    if not np.allclose(s, saved, rtol=1e-9, atol=1e-9):
        raise SystemExit("refitted PCA does not reproduce the baseline's scores")
    return {"idf": idf, "mu": mu, "P": P, "threshold": base["threshold"]}


_WORKER: dict = {}


def worker_models(blob: bytes) -> Models:
    """Built once per Python worker from the broadcast weights."""
    if "m" in _WORKER:
        return _WORKER["m"]
    import torch

    from loglens.lstm import DeepLog

    torch.set_num_threads(1)  # one worker per core already
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:  # already set in this worker
        pass
    spec = pickle.loads(blob)

    def lstm(name):
        ck = torch.load(io.BytesIO(spec[name]), weights_only=False)
        net = DeepLog(ck["n_classes"], ck["config"]["hidden"])
        net.load_state_dict(ck["state"])
        net.eval()
        return net, ck

    dl, dl_ck = lstm("deeplog")
    en, en_ck = lstm("deeplog_end_tuned")

    def rank(w, y):
        with torch.no_grad():
            logits = dl(torch.from_numpy(np.asarray(w, dtype=np.int64)))
            t = logits.gather(1, torch.from_numpy(np.asarray(y, dtype=np.int64))[:, None])
            return ((logits > t).sum(1) + 1).numpy()

    def nll(w, y):
        with torch.no_grad():
            logits = en(torch.from_numpy(np.asarray(w, dtype=np.int64)))
            yt = torch.from_numpy(np.asarray(y, dtype=np.int64))[:, None]
            return (-torch.log_softmax(logits, 1).gather(1, yt)[:, 0]).numpy()

    pca = spec["pca"]

    def pca_score(counts):
        z = counts * pca["idf"] - pca["mu"]
        r = z - (z @ pca["P"].T) @ pca["P"]
        return float(r @ r)

    if dl_ck["config"]["h"] != en_ck["config"]["h"]:
        raise SystemExit("both LSTMs must share a history length")
    _WORKER["m"] = Models(rank, nll, pca_score, dl_ck["threshold_rank"], en_ck["threshold_nll"], pca["threshold"],
                          dl_ck["config"]["h"], spec["unk"], spec["end"], spec["n_vocab"])
    return _WORKER["m"]


# --------------------------------------------------------------- prepare ----


_TS: dict[str, datetime] = {}


def hdfs_ts(key: str) -> datetime | None:
    """'081111 060046' -> datetime, cached (many lines share a second)."""
    if key not in _TS:
        try:
            _TS[key] = datetime.strptime(key, "%y%m%d %H%M%S")
        except ValueError:
            _TS[key] = None
    return _TS[key]


def prepare(dataset: str) -> dict:
    """Cut the raw log from the test cutoff onwards into 5-minute Parquet files, plus two sentinels."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    cut = json.loads((config.RUNLOG / f"{dataset}_split.json").read_text())["cutoffs"]["test_start"]["ts"]
    start = datetime.fromisoformat(cut)
    src = STREAM / "replay"
    shutil.rmtree(src, ignore_errors=True)
    src.mkdir(parents=True)
    bins: dict[int, list] = {}
    last = start
    cut_key = start.strftime("%y%m%d %H%M%S")  # the prefix sorts like the time within one century
    with open(config.settings(dataset)["raw"], encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if line[:13] < cut_key:
                continue
            ts = hdfs_ts(line[:13])
            if ts is None:
                continue
            last = max(last, ts)
            bins.setdefault(int((ts - start).total_seconds()) // BIN_S, []).append((i, line.rstrip("\n")))
    timeouts = timeouts_s(dataset)
    sentinels = [last + timedelta(seconds=max(timeouts) + 3600 * k) for k in (1, 2)]
    files = [(b, sorted(rows)) for b, rows in sorted(bins.items())]
    files += [(10**6 + k, [(10**9 + k, s.strftime("%y%m%d %H%M%S") + " 0 INFO loglens.Replay: end of replay blk_0")])
              for k, s in enumerate(sentinels)]
    base = time.time() - len(files) - 10
    for n_, (b, rows) in enumerate(files):
        path = src / f"part-{n_:05d}.parquet"
        pq.write_table(pa.table({"line_id": [r[0] for r in rows], "value": [r[1] for r in rows]}), path)
        os.utime(path, (base + n_, base + n_))  # the file source reads in modification order
    meta = {
        "test_start": start.isoformat(),
        "log_end": last.isoformat(),
        "flush": (sentinels[0] - timedelta(seconds=1)).isoformat(),
        "lines": sum(len(r) for _, r in files) - 2,
        "files": len(files),
        "timeouts_s": list(timeouts),
    }
    (STREAM / "replay_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def timeouts_s(dataset: str) -> tuple:
    main = json.loads((config.DASHBOARD / f"{dataset}_evaluation.json").read_text())["timeout"]["timeout_s"]
    return (*SHORT_TIMEOUTS_S, int(main))


def ms(iso: str) -> int:
    """Log time (naive, treated as UTC like the Spark session) to epoch milliseconds."""
    return int((datetime.fromisoformat(iso) - datetime(1970, 1, 1)).total_seconds() * 1000)


# ------------------------------------------------------------------- run ----


def run_stream(dataset: str, meta: dict) -> dict:
    import torch  # noqa: F401  (fails early if the image lacks it)
    from pyspark.sql import functions as F
    from pyspark.sql.streaming.state import GroupStateTimeout

    from loglens.gold import vocabulary
    from loglens.ingest import parse_lines
    from loglens.lstm import MODELS_DIR, encoder
    from loglens.masking import block_ids
    from loglens.parse import match_regex
    from loglens.spark import get_spark

    cfg = config.settings(dataset)
    vocab = vocabulary(dataset)
    enc = encoder(vocab + ["END"])
    slot = {e: i for i, e in enumerate(vocab)}  # count-vector position, as in the gold layer
    spec = {
        "deeplog": (MODELS_DIR / f"{dataset}_deeplog_seed0.pt").read_bytes(),
        "deeplog_end_tuned": (MODELS_DIR / f"{dataset}_deeplog_end_tuned_seed0.pt").read_bytes(),
        "pca": fit_pca(dataset),
        "unk": enc["UNK"], "end": enc["END"], "n_vocab": len(vocab),
    }
    timeouts_ms = tuple(t * 1000 for t in meta["timeouts_s"])
    flush_ms, log_end_ms = ms(meta["flush"]), ms(meta["log_end"])

    spark = get_spark(f"loglens-stream-{dataset}")
    spark.conf.set("spark.sql.streaming.numRecentProgressUpdates", "10000")
    spark.conf.set("spark.sql.shuffle.partitions", str(STATE_PARTITIONS))
    blob = spark.sparkContext.broadcast(pickle.dumps(spec))
    enc_b, slot_b = spark.sparkContext.broadcast(enc), spark.sparkContext.broadcast(slot)

    def per_block(key, pdfs, state):
        M = worker_models(blob.value)
        st = dict(zip(STATE_FIELDS, state.get)) if state.exists else new_state(M.n_vocab)
        if state.hasTimedOut:
            ts, cls, slots = [], [], []
        else:
            frames = list(pdfs)
            pdf = pd.concat(frames).sort_values(["ts_ms", "line_id"]) if frames else pd.DataFrame(columns=["ts_ms", "event_id"])
            ts = pdf["ts_ms"].tolist()
            cls = [enc_b.value[e] for e in pdf["event_id"]]
            slots = [slot_b.value[e] for e in pdf["event_id"]]
        decisions = advance(st, ts, cls, slots, state.getCurrentWatermarkMs(), flush_ms, log_end_ms, timeouts_ms, M)
        nxt = next_timeout(st, timeouts_ms, flush_ms)
        if nxt is None:
            state.remove()
        else:
            state.update(tuple(st[f] for f in STATE_FIELDS))
            state.setTimeoutTimestamp(nxt)
        if decisions:
            d = pd.DataFrame(decisions, columns=["model", "timeout_s", "flagged", "decided_ms", "first_ms", "n_events"])
            d.insert(0, "block_id", key[0])
            yield d

    templates = json.loads((config.SILVER / f"{dataset}_templates.json").read_text())
    test_blocks = (
        spark.read.parquet(str(config.GOLD / f"{dataset}_split")).filter(F.col("split") == "test").select("block_id")
    )
    raw = spark.readStream.schema("line_id long, value string").option("maxFilesPerTrigger", 1).parquet(str(STREAM / "replay"))
    lines = parse_lines(raw, cfg).withWatermark("ts", "0 seconds")  # before the join, so the sentinels count
    events = (
        lines.withColumn("event_id", match_regex(lines, templates, dataset))
        .withColumn("block_id", F.element_at(block_ids("message"), 1))
        .join(F.broadcast(test_blocks), "block_id")
        .select("block_id", "ts", (F.col("ts").cast("long") * 1000).alias("ts_ms"), "line_id", "event_id")
    )
    decided = events.groupBy("block_id").applyInPandasWithState(
        per_block, OUT_SCHEMA, STATE_SCHEMA, "append", GroupStateTimeout.EventTimeTimeout
    )
    out, ckpt = STREAM / "decisions", STREAM / "checkpoint"
    shutil.rmtree(out, ignore_errors=True)
    shutil.rmtree(ckpt, ignore_errors=True)
    t0 = time.perf_counter()
    q = (decided.writeStream.format("parquet").option("path", str(out)).option("checkpointLocation", str(ckpt))
         .outputMode("append").trigger(availableNow=True).start())
    q.awaitTermination()
    wall = time.perf_counter() - t0
    progress = [p for p in q.recentProgress if p.get("numInputRows") is not None]
    spark.stop()
    batches = [{"rows": int(p["numInputRows"]), "ms": int(p["durationMs"].get("triggerExecution", 0))} for p in progress]
    return {"wall_s": round(wall, 1), "batches": batches}


# --------------------------------------------------------------- analyse ----


def analyse(dataset: str, meta: dict, processing: dict) -> dict:
    d = pd.read_parquet(STREAM / "decisions")
    gold = pd.read_parquet(config.GOLD / f"{dataset}_sessions", columns=["block_id", "seq_hash", "n_events", "split", "label", "start_ts"])
    test = gold[gold["split"] == "test"].set_index("block_id")
    d = d.join(test[["label"]], on="block_id")
    log_end_ms, start_ms = ms(meta["log_end"]), ms(meta["test_start"])
    hours = (log_end_ms - start_ms) / 3.6e6

    # Offline verdicts for the same sessions, from the saved per-sequence scores.
    lstm = json.loads((config.DASHBOARD / f"{dataset}_lstm.json").read_text())
    base = json.loads((config.DASHBOARD / f"{dataset}_baselines.json").read_text())["results"]["chrono"]

    def offline(variant: str, kind: str) -> pd.Series:
        per = json.loads((config.GOLD / f"{dataset}_{variant}_ranks.json").read_text())
        top = {int(h): max(v[kind]) for h, v in per.items()}
        return test["seq_hash"].map(top)

    score = {
        "deeplog": (offline("deeplog", "ranks"), lstm["deeplog"]["results"]["chrono"]["seeds"]["0"]["threshold"]),
        "deeplog_end": (offline("deeplog_end_tuned", "nll"), lstm["deeplog_end_tuned"]["results"]["chrono"]["seeds"]["0"]["nll_score"]["threshold"]),
        "pca": (pd.read_parquet(config.GOLD / f"{dataset}_baseline_scores.parquet", columns=["block_id", "pca"])
                .set_index("block_id")["pca"].reindex(test.index), base["pca"]["threshold"]),
    }
    off = {m: (s >= thr).astype(int) for m, (s, thr) in score.items()}

    final = d[(d["timeout_s"] == END_OF_LOG) | (d["model"] == "deeplog")]
    final = final.sort_values("decided_ms").drop_duplicates(["block_id", "model"])  # DeepLog: its first verdict
    agree = {}
    for m in ("deeplog", "deeplog_end", "pca"):
        s = final[final["model"] == m].set_index("block_id")["flagged"].reindex(test.index)
        bad = s.fillna(-1).astype(int) != off[m]
        # The stream scores one block at a time, so its floating point can differ from the batch
        # run's in the last digits; that only matters for a session sitting exactly on the threshold.
        tie = (score[m][0] - score[m][1]).abs() < TIE_TOLERANCE
        agree[m] = {"sessions": int(s.notna().sum()), "disagree": int(bad.sum()), "disagree_off_threshold": int((bad & ~tie).sum())}
    pca_final = d[(d["model"] == "pca") & (d["timeout_s"] == END_OF_LOG)]
    checks = {
        "every_test_session_decided": all(a["sessions"] == len(test) for a in agree.values()),
        "every_test_event_streamed_once": int(pca_final["n_events"].sum()) == int(test["n_events"].sum()),
        "end_of_log_verdicts_match_offline": all(a["disagree_off_threshold"] == 0 for a in agree.values()),
    }

    rows = []
    for (m, t), g in d.groupby(["model", "timeout_s"]):
        if m == "deeplog":
            g = g.sort_values("decided_ms").drop_duplicates("block_id")
        y, f = g["label"].to_numpy(), g["flagged"].to_numpy()
        tp, fp, fn = int(((y == 1) & (f == 1)).sum()), int(((y == 0) & (f == 1)).sum()), int(((y == 1) & (f == 0)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        caught = g[(g["label"] == 1) & (g["flagged"] == 1)]
        lat = (caught["decided_ms"] - caught["first_ms"]) / 1000
        alerts = g[g["flagged"] == 1]
        rows.append({
            "model": m, "timeout_s": int(t), "sessions": int(len(g)),
            "precision": round(p, 6), "recall": round(r, 6), "f1": round(2 * p * r / (p + r) if p + r else 0.0, 6),
            "tp": tp, "fp": fp, "fn": fn,
            "alerts_before_log_end": int((alerts["decided_ms"] <= log_end_ms).sum()),
            "caught_before_log_end": int((caught["decided_ms"] <= log_end_ms).sum()),
            "false_alarms_per_hour": round(float(((alerts["label"] == 0) & (alerts["decided_ms"] <= log_end_ms)).sum()) / hours, 2),
            "latency_s": {f"p{q}": round(float(np.percentile(lat, q)), 1) for q in (10, 50, 90)} if len(lat) else None,
            "within_1_minute": round(float((lat <= 60).mean()), 4) if len(lat) else None,
        })
    return {"meta": meta, "processing": processing, "results": rows, "agreement_with_offline": agree, "checks": checks}


def summarise_processing(meta: dict, proc: dict) -> dict:
    hours = (ms(meta["log_end"]) - ms(meta["test_start"])) / 3.6e6
    b = pd.DataFrame(proc["batches"])
    data_batches = b[b["rows"] > 0]
    return {
        "wall_s": proc["wall_s"],
        "lines_replayed": meta["lines"],
        "log_hours": round(hours, 2),
        "lines_per_s": round(meta["lines"] / proc["wall_s"]),
        "speedup_vs_real_time": round(hours * 3600 / proc["wall_s"], 1),
        "batches": int(len(b)),
        "batch_ms": {f"p{q}": int(np.percentile(data_batches["ms"], q)) for q in (50, 90, 99)},
        "log_seconds_per_batch": BIN_S,
    }


def run(dataset: str, prepare_only: bool = False, fresh: bool = False, analyse_only: bool = False) -> None:
    t = time.perf_counter()
    meta_path = STREAM / "replay_meta.json"
    if not fresh and meta_path.exists() and json.loads(meta_path.read_text())["timeouts_s"] == list(timeouts_s(dataset)):
        meta = json.loads(meta_path.read_text())  # the replay files only depend on the raw log and the cutoff
    else:
        meta = prepare(dataset)
    print("prepared", json.dumps(meta), f"{time.perf_counter() - t:.0f}s", flush=True)
    if prepare_only:
        return
    proc_path = STREAM / "processing.json"
    if analyse_only:  # reuse the last replay's decisions and timings
        processing = json.loads(proc_path.read_text())
    else:
        proc = run_stream(dataset, meta)
        print("streamed", proc["wall_s"], "s,", len(proc["batches"]), "batches", flush=True)
        processing = summarise_processing(meta, proc)
        proc_path.write_text(json.dumps(processing, indent=2))
    result = analyse(dataset, meta, processing)
    print(json.dumps({"checks": result["checks"], "agreement": result["agreement_with_offline"], "processing": result["processing"]}, indent=1), flush=True)
    for r in result["results"]:
        print(f"{r['model']:12s} T={r['timeout_s']:>6} F1={r['f1']:.3f} P={r['precision']:.3f} R={r['recall']:.3f} "
              f"caught before end={r['caught_before_log_end']} median latency={r['latency_s'] and r['latency_s']['p50']}", flush=True)
    runlog.write(dataset, "stream", result)
    (config.DASHBOARD / f"{dataset}_stream.json").write_text(json.dumps(result, indent=2))
    if not all(result["checks"].values()):
        raise SystemExit(f"stream checks failed: {result['checks']}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run(args[0] if args else DATASET, "--prepare-only" in sys.argv, "--fresh" in sys.argv, "--analyse-only" in sys.argv)
