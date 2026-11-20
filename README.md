# LogLens: Do Sequence Models Really Catch Anomalous Logs?

Published log anomaly detectors report near-perfect scores on the HDFS benchmark, but most of those numbers come from random train/test splits that let a model see the future. LogLens asks what is left once the evaluation is honest about time. It takes the 11.2 million raw lines of the LogHub HDFS_v1 log, parses them in PySpark into bronze, silver and gold Parquet layers, groups them into 575,061 block-level sessions, and compares a DeepLog-style LSTM against PCA, Isolation Forest and a supervised LightGBM on a strict chronological split. It also measures how much a random split inflates each score, how many test sequences the model has already seen verbatim in training, and how many alerts per hour each model would raise. The method is then repeated on BGL (no natural sessions) and the pipeline is scaled to all 211 million Thunderbird lines.

Built with Python, PySpark (Parquet medallion layers), Drain3, scikit-learn, LightGBM and MLflow on the public LogHub datasets, run in Docker. PyTorch joins with the LSTM stage.

> **What it's for:** checking whether a log anomaly detector's headline score survives a time-ordered evaluation, and seeing, session by session, which event in a sequence made the model raise an alert.

**Live demo:** not published yet. The results dashboard (a single static page) will be linked here once the first models are evaluated.

**Status:** in progress. Ingest, the chronological split, template parsing (validated against LogHub's ground truth), sessionization and five baselines are complete; the LSTM, the deeper evaluation and the dashboard are next. Screenshots will be added with the dashboard.

---

## Disclaimer

This is a personal research and learning project on public benchmark data. It is **not a production intrusion or anomaly detection system** and makes no claims to that effect. The HDFS logs are from 2008 and their labels come from the dataset's original authors; any result here describes this benchmark, not how a model would behave on your systems.

---

## Findings

Real results from the stages built so far. Model results will be added as each stage lands.

### The data

- **Every line parses, and the totals reconcile exactly.** One regex splits all **11,175,629** raw lines into date, time, pid, level, component and message, with **0** sent to quarantine. Raw = bronze + quarantine holds to the line, and the log spans **38.69 hours** (9 Nov 2008 20:35 to 11 Nov 2008 11:16), matching LogHub's published 38.7.
- **One line in eight names its block twice, and a naive count would double them.** **1,402,056** lines (12.5%) contain two block IDs, but always the same block (as in "Deleting block blk_X file .../blk_X"). Exploding every match would invent 1.4M duplicate events; de-duplicating IDs within each line gives exactly one block per line, so the 11,175,629 lines become exactly 11,175,629 session events. (An earlier version of this README read these as lines spanning two sessions; the gold-layer checks caught it.)
- **The levels say almost nothing on their own.** There are **no ERROR lines at all**: 96.8% INFO and 3.2% WARN (362,793 lines), and **98.2%** of those WARN lines come from a single component, `dfs.DataNode$DataXceiver`. Four components produce 98.8% of all lines.
- **Accuracy would be meaningless here.** Of 575,061 block sessions, **16,838** are labelled anomalous (2.93%), so a model that calls everything normal scores 97.07% accuracy. LogLens reports precision, recall, F1 and PR-AUC instead.

### The chronological split

- **Time is very uneven across the split.** Ordering sessions by their first event, the first 60% (train, **345,036** sessions) start over **31.7 hours**, while the last 30% (test, **172,519**) start within the final **5.1 hours**. Line volume swings 34x from hour to hour, so the split is made by session order, not clock time.
- **The test period has a third fewer anomalies.** The anomaly rate is **3.23%** in train, **3.40%** in validation and **2.16%** in test. Any threshold tuned on earlier data meets a different base rate later, which a random split would hide.
- **The most common warning disappears after training.** All **356,207** "Got exception while serving" lines (the bulk of the WARN lines) occur before the training cutoff; none appear in validation or test. A model that leans on warning volume would see a shift that a random split hides.

### Parsing (silver layer)

- **Masking collapses the training period to 51 distinct messages.** After block IDs, IPs, paths and numbers are masked in Spark, the **6,856,030** training-period lines reduce to **51** distinct messages, so Drain3 learns its templates from every training line in about 25 seconds instead of looping over millions of rows in Python.
- **Drain3's default threshold hides an anomaly signal.** At the default similarity threshold (0.4), Drain3 merged "PacketResponder N for block X **terminating**" (1.7M lines, normal) with "... **Interrupted**" (49 lines, a failure) into one template. That single merge cut grouping accuracy against LogHub's ground truth to **84.6%**. Sweeping the threshold and choosing on the validation period gives **0.9**: **50 templates, zero merges, 99.84% grouping accuracy** (99.80% on validation, 99.78% on test). The only remaining differences are harmless splits, such as one template per exception text where LogHub uses one for all.
- **Template drift is tiny but real.** Only **15** lines in validation and test match no learned template (4 and 11). Ten of them are "Adding an already existing block", an event that never occurs in the training period. These get an `UNK` event rather than being forced into the nearest template.
- **Spark-native matching is fastest, and all three methods agree on every line.** Tagging all 11.2M lines took **34.0 s** with templates compiled to regexes in Spark SQL (about 329,000 lines per second), **49.7 s** with Drain3 inside a pandas UDF and **64.7 s** with a distinct-message lookup join, and all three produced the same event for all 11,175,629 lines.
- **The session logic reproduces LogHub's published traces exactly.** Tagging each line with LogHub's 29 ground-truth templates and grouping by block reproduces the published event sequence, in order, for **all 575,061 blocks**. This validates the block explode and the (timestamp, line) ordering before any model is trained on them.

### Sessions and baselines

- **575,061 sessions, but only 18,375 distinct event sequences and 597 distinct count vectors.** The median session has 19 events and the longest 298. HDFS block behaviour is extremely repetitive.
- **Labels are almost a function of the sequence.** Only **10** sequences occur with both labels (46 sessions), so a model that sees the whole sequence has almost no irreducible error.
- **A time-ordered split does not remove duplication here.** **99.0%** of chronological test sessions have an event sequence already seen in training, against **97.7%** under a random split (for anomalous test sessions, 80.6% against 78.9%). On HDFS the repetition comes from the system itself, not from shuffling.

Chronological test set (172,519 sessions, 3,723 anomalous). Each threshold is chosen on the validation period; single run, confidence intervals not yet computed.

| Model | Precision | Recall | F1 | PR-AUC | F1 on never-seen sequences |
|-------|-----------|--------|----|--------|----------------------------|
| WARN rule (any warning line) | 1.000 | 0.446 | 0.617 | 0.458 | 0.690 |
| Sequence lookup + novelty | 0.792 | 1.000 | 0.884 | 0.792 | 0.598 |
| PCA, Xu et al. (5 components) | 0.998 | 0.932 | 0.964 | 0.930 | 0.844 |
| Isolation Forest | 0.806 | 0.568 | 0.666 | 0.751 | 0.678 |
| LightGBM (supervised reference) | 0.979 | 0.998 | 0.989 | 0.9999 | 0.944 |

- **A random split barely changes the count-based models.** F1 moves by **-0.03** points for PCA, **+0.9** for LightGBM and **+2.2** for Isolation Forest between the chronological and random protocols. With duplication this high under both splits, there is little extra for a random split to leak. Whether sequence models such as the LSTM behave the same is still open.
- **The largest gap runs the other way, and it is distribution shift.** The WARN rule scores F1 **0.617** on the chronological test but **0.107** on a random one. The 356,207 "Got exception while serving" warnings, mostly in normal sessions, only occur early in the log: a random test set inherits them as 38,064 false alarms, while the chronological test period never sees them.
- **Novelty is the strongest single signal.** **1,699** chronological test sessions have a sequence never seen in training; **42.6%** of them are anomalous (against 2.2% overall), and they hold **19.4%** of all test anomalies. Looking up the training label of seen sequences and flagging every unseen one already reaches F1 **0.884**, beating Isolation Forest and the WARN rule.
- **The honest comparison is on never-seen sequences.** On those 1,699 sessions F1 falls from 0.964 to **0.844** for PCA and from 0.989 to **0.944** for LightGBM. On repeated sequences the models are largely recognising what they were shown in training.

---

## Features

### Ingest (bronze layer)

- Spark reads the raw text, splits each line with one regex and parses the `yyMMdd HHmmss` timestamp.
- Lines that fail the regex or the timestamp go to a **quarantine table** rather than being dropped, so nothing disappears silently.
- A `line_id` keeps the original file order, which breaks ties between events in the same second (HDFS timestamps have one-second resolution).
- Output is Parquet partitioned by hour (40 partitions for HDFS).

### Chronological split

- Sessions are ordered by their first event (timestamp, then file position) and split 60% train, 10% validation and 30% test, with every block joined to its label.
- The first event of the first validation session is the **training cutoff**: everything fitted on "training data" (templates now, models and thresholds later) only sees lines before it.

### Template parsing (silver layer)

- Shared masking rules (IPs, paths, block IDs, numbers) are applied identically in Spark and in Drain3.
- Drain3 is fitted only on training-period lines, with its similarity threshold chosen by a sweep scored on the validation period.
- Every line gets an event ID by one of three interchangeable matchers (Spark SQL regex, pandas UDF, lookup join); the fastest is used and all three are checked to agree.
- Lines matching no learned template get `UNK`, and the UNK rate is reported per period and per hour.
- Output is one row per (line, block) with its event ID, ready to group into sessions.

### Parser evaluation

- Grouping accuracy against LogHub's ground-truth templates, overall and per period, with counts of merged and split templates.
- A check that the ground-truth tagging reproduces LogHub's published per-block event sequences.

### Sessions (gold layer)

- One row per block: the ordered event sequence, a 51-dimensional count vector (50 templates plus UNK), WARN count, duration, hashes of the sequence and of the count vector, and the block's split and label.

### Baselines and evaluation

- Five scorers: a WARN-line rule, a sequence lookup with novelty, PCA on TF-IDF weighted counts (Xu et al. 2009), Isolation Forest and supervised LightGBM.
- Two protocols on the same sessions: the chronological split and a seeded random split with the same proportions.
- Every threshold and hyperparameter (PCA's variance kept, LightGBM's early stopping) is chosen on validation; test is only scored.
- Precision, recall, F1 and PR-AUC on test, plus the same metrics on test sessions whose sequence never appears in training.
- Duplication analysis (share of test sequences and count vectors already seen in training, per protocol) and a count of sequences that carry both labels.
- Every run is logged to MLflow for side-by-side comparison.

### Data quality checks

- Each stage writes a JSON run log with its row counts, timings and pass/fail checks, and fails loudly when a check fails: raw = bronze + quarantine, every block has exactly one label, every line gets an event ID, no training-period line is UNK, every event lands in exactly one session and the count vectors add up to the event totals.

### Exploratory analysis

- Lines per hour by level, level mix, lines and non-INFO lines per component, blocks named per line, and the label balance, written to a small JSON aggregate that the dashboard will read.

### Planned stages

- **Sequence model:** a DeepLog-style LSTM trained on normal training sequences, tuned on the validation period.
- **Evaluation:** bootstrap confidence intervals, a training-size ablation, alerts per hour and detection latency.
- **Generalization and scale:** BGL with time windows, and the full Thunderbird log.
- **Dashboard:** one static page with inline SVG charts, built only from the exported aggregates.

---

## Data Source

| Source | Type | What It Provides |
|--------|------|------------------|
| [LogHub HDFS_v1](https://github.com/logpai/loghub) | Raw log, labels and ground truth ([Zenodo](https://zenodo.org/records/8196385)) | 11,175,629 lines from a 200+ node Hadoop cluster, a Normal/Anomaly label for each of 575,061 blocks, 29 ground-truth templates and the published event sequence of every block |
| [LogHub BGL](https://github.com/logpai/loghub) | Raw log, labelled per line | 4,747,963 lines from a Blue Gene/L supercomputer (generalization run, planned) |
| [LogHub Thunderbird](https://github.com/logpai/loghub) | Raw log, labelled per line | 211,212,192 lines (scale run, planned) |
| [Loghub-2.0](https://github.com/logpai/loghub-2.0) | Annotated templates | Ground-truth templates for the BGL and Thunderbird runs (planned) |

**How the data is cleaned and joined:**

1. **Download:** `scripts/download_data.py` fetches each archive from LogHub's Zenodo record and verifies its MD5 checksum before extracting.
2. **Split fields:** a regex splits each line into date, time, pid, level, component and message; non-matching lines are quarantined.
3. **Order:** each line keeps its original file position (`line_id`) alongside its timestamp.
4. **Sessions and labels:** block IDs are extracted from each message and de-duplicated within the line (a line can repeat its block ID), each block's first event fixes its place in the chronological split, and labels are joined on block ID.
5. **Mask and template:** IPs, paths, block IDs and numbers are masked, and each line is tagged with a template learned on the training period only.
6. **Sessionize:** events are grouped by block and ordered by timestamp, then file position, into a sequence and a count vector per session.
7. **Reconcile:** row counts are checked against the raw file and LogHub's published totals at every stage.

---

## Tech Stack

| Layer | Technology | Details |
|-------|-----------|---------|
| **Processing** | PySpark 3.5 (local mode) | Regex parsing, explode, window functions, partitioned Parquet |
| **Storage** | Parquet | Bronze/silver/gold layers on local disk |
| **Parsing** | Drain3 | Template mining on the training period; templates compiled to Spark SQL regexes for matching |
| **Models** | scikit-learn, LightGBM | PCA, Isolation Forest, gradient boosting; metrics and PR curves |
| **Tracking** | MLflow (local file store) | One run per model and protocol, compared in the MLflow UI |
| **Environment** | Docker (Python 3.12, Java 17) | Avoids Spark's Windows dependencies; code and data are bind-mounted |
| **Testing** | pytest | Unit tests for the line regex, masking, template regexes and grouping accuracy |

---

## Prerequisites

- **Docker Desktop** (or Docker Engine on Linux): [Download here](https://www.docker.com/products/docker-desktop/)
- About **3 GB of free disk** for HDFS_v1 and its Parquet layers

No API keys and no accounts.

---

## Quick Start

```bash
docker compose build
docker compose run --rm loglens make download      # HDFS_v1 from Zenodo, checksum verified
docker compose run --rm loglens make bronze        # raw text -> bronze Parquet + run log
docker compose run --rm loglens make eda           # aggregates -> data/dashboard/hdfs_eda.json
docker compose run --rm loglens make split         # chronological train/val/test by session start
docker compose run --rm loglens make parse-sweep   # choose the Drain3 threshold on validation
docker compose run --rm loglens make parse         # templates, three-matcher benchmark, silver layer
docker compose run --rm loglens make parse-eval    # grouping accuracy vs LogHub ground truth
docker compose run --rm loglens make gold          # one row per session: sequence, counts, label
docker compose run --rm loglens make baselines     # five baselines, two protocols, logged to MLflow
docker compose run --rm loglens make test
docker compose run --rm --service-ports loglens make mlflow-ui   # http://localhost:5000
```

`make all` runs every stage in order. Spark's driver memory defaults to 5 GB; set `LOGLENS_DRIVER_MEMORY` to change it.

---

## Project Structure

```
LogLens/
├── README.md
├── LICENSE
├── Makefile                      # Pipeline targets, run inside the container
├── requirements.txt
├── docker-compose.yml            # One service, repo bind-mounted at /app
├── docker/
│   ├── Dockerfile                # Python 3.12 + Java 17 + PySpark
│   └── Dockerfile.dockerignore   # Keeps data/ out of the build context
│
├── scripts/
│   └── download_data.py          # Zenodo download with MD5 verification
│
├── src/loglens/
│   ├── config.py                 # Paths, per-dataset regex, expected counts, Drain3 threshold
│   ├── spark.py                  # Laptop-tuned Spark session
│   ├── runlog.py                 # Per-stage JSON run log and timers
│   ├── ingest.py                 # Bronze layer with quarantine and reconciliation
│   ├── eda.py                    # Exploratory aggregates for the dashboard
│   ├── split.py                  # Chronological split by session start, training cutoff
│   ├── masking.py                # Masking rules shared by Spark and Drain3
│   ├── parse.py                  # Drain3 fit, three matchers, silver layer
│   ├── parse_eval.py             # Grouping accuracy, trace check, threshold sweep
│   ├── gold.py                   # Sessions: sequences, count vectors, hashes, labels
│   ├── metrics.py                # Validation-chosen thresholds, P/R/F1, PR-AUC, PR curves
│   └── baselines.py              # Five baselines, two protocols, duplication analysis, MLflow
│
├── tests/
│   ├── test_ingest_regex.py
│   ├── test_parsing.py
│   └── test_metrics.py
│
└── data/
    ├── raw/                      # Downloaded archives (not tracked)
    ├── bronze/ silver/ gold/     # Parquet layers (not tracked)
    ├── runlog/                   # Per-stage run logs (not tracked)
    └── dashboard/                # Small aggregates the dashboard reads (tracked)
```

---

## Architecture

```
raw text ──► bronze (Parquet by hour) ──► split (session start) ──► silver (event ID per line) ──► gold (sessions) ──► baselines ──► data/dashboard/*.json ──► static page
               │ quarantine                 │ training cutoff ─────► Drain3 fit                                       │ MLflow           (planned)
               ▼                                                         │
           run log (counts, checks, timings)                     parse eval vs LogHub ground truth
```

- **Medallion layers.** Bronze holds every line as structured fields; silver adds an event ID and the line's block; gold holds one row per session.
- **Spark for volume, pandas for models.** Everything up to gold runs in Spark. The gold table (575,061 rows, 51 counts each) fits comfortably in memory, so the models read it into pandas and NumPy. Each layer is rebuilt from the one before it, so every stage is reproducible from the raw log.
- **The training cutoff is computed once.** The split stage records the first event of the first validation and test sessions; every later stage that fits anything reads those cutoffs from the run log, so no stage can quietly fit on the future.
- **Matching runs in the JVM.** Templates are compiled to Java regexes and matched in a single Spark SQL `CASE WHEN`, which avoids moving 11M rows through Python. The pandas UDF and lookup-join versions stay in the code as the benchmark and as a cross-check.
- **Run log.** Every stage writes `data/runlog/<dataset>_<stage>.json`. These feed the dashboard's pipeline-health panel.
- **The dashboard never reads raw data.** It reads only the small aggregates in `data/dashboard/`, so the page stays a single file.
- **Execution.** Spark runs in local mode inside one container. Bronze ingest takes about 4 minutes, the split about 1.3 minutes, the parse about 1.5 minutes (plus 4.7 for the benchmark) and the gold layer about 2 minutes, mostly spent reading through the Windows bind mount. All five baselines under both protocols fit and score in under a minute.

---

## Testing and Validation

**Unit tests (16):** the HDFS line regex (a typical line, a message containing colons, a line that must not parse); masking (IP with port, a path that contains a block ID, numbers but not digits inside words, idempotence); template regexes (a wildcard is exactly one token, ground-truth wildcards allow anything); grouping accuracy (a perfect one-to-one case, and merges, splits and UNK lines all counted as wrong); and evaluation (precision/recall/F1 counts, threshold selection, a threshold taken from validation and applied unchanged to test, the random split's proportions and determinism, and label-conflict counting).

**Pipeline checks, on the full dataset:** the raw line count equals LogHub's published 11,175,629; raw = bronze + quarantine; the time span matches LogHub's 38.7 hours; every one of the 575,061 blocks has exactly one label; every line gets an event ID and no training-period line is UNK; the three matchers agree on every line; ground-truth tagging reproduces LogHub's published event sequence for every block; and every event lands in exactly one session, with count vectors that sum to the event totals.

---

## Honest Limitations

- **Only half the main question is answered.** The baselines are in; the sequence model the project is built around is not, so nothing yet says whether a sequence model beats PCA under this protocol.
- **Single runs, no intervals yet.** Each score comes from one fit with a fixed seed. Small differences (such as LightGBM's 0.9 point gap) may be within noise until bootstrap intervals are added.
- **Validation thresholds maximise F1.** A real deployment would more likely fix an alert budget; alerts per hour are not reported yet.
- **The random protocol reuses the chronological parser.** Templates were fitted on the chronological training period for both protocols. With only 15 unseen lines, this has a negligible effect, but it is not a fully separate pipeline.
- **LightGBM sees labels; the other models do not.** It is shown as a supervised reference, not as a like-for-like competitor.
- **One benchmark, one era.** HDFS_v1 is a 2008 Hadoop log with block-level labels from its original authors; results may not carry over to modern systems, which is why BGL is planned as a second dataset.
- **Labels are taken as given.** No relabelling or label-noise analysis is done.
- **The parser threshold was chosen with ground-truth templates.** The sweep uses LogHub's template labels (not the anomaly labels) on the validation period. A real deployment would not have those, so the 0.4 result shows what an untuned parser would have done.
- **Ground-truth tags come from LogHub's template patterns.** Each line's reference event is assigned by matching LogHub's 29 templates; this is validated against LogHub's published traces (an exact match for every block), but it is not a separate human annotation.
- **Timings are single runs on one laptop.** They show relative speed, not a controlled benchmark, and include reading through a Windows bind mount.

---

## Suggested Future Features

- **A semantic event model** (sentence embeddings of templates) so unseen templates in later periods are handled without retraining.
- **A streaming replay** of the test period with Spark Structured Streaming, to measure real time-to-alert.
- **More LogHub systems**, such as Hadoop and OpenStack, to widen the generalization check.

---

## Author

**Sagnik**

GitHub: [@HalcyonVector](https://github.com/HalcyonVector)

---

## License

MIT License. See [LICENSE](./LICENSE). Free to use, modify, and distribute, with no warranty.
