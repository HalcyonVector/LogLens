# LogLens: Do Sequence Models Really Catch Anomalous Logs?

Published log anomaly detectors report near-perfect scores on the HDFS benchmark, but most of those numbers come from random train/test splits that let a model see the future. LogLens asks what is left once the evaluation is honest about time. It takes the 11.2 million raw lines of the LogHub HDFS_v1 log, parses them in PySpark into bronze, silver and gold Parquet layers, groups them into 575,061 block-level sessions, and compares a DeepLog-style LSTM against PCA, Isolation Forest and a supervised LightGBM on a strict chronological split. It also measures how much a random split inflates each score, how many test sequences the model has already seen verbatim in training, and how many alerts per hour each model would raise. The method is then repeated on BGL (no natural sessions) and the pipeline is scaled to all 211 million Thunderbird lines.

Built with Python, PySpark (Parquet medallion layers) and Drain3 on the public LogHub datasets, run in Docker. scikit-learn, LightGBM and PyTorch join with the modelling stage.

> **What it's for:** checking whether a log anomaly detector's headline score survives a time-ordered evaluation, and seeing, session by session, which event in a sequence made the model raise an alert.

**Live demo:** not published yet. The results dashboard (a single static page) will be linked here once the first models are evaluated.

**Status:** in progress. Ingest, the chronological split and template parsing are complete and validated against LogHub's ground truth; sessionization, models and the dashboard are next. Screenshots will be added with the dashboard.

---

## Disclaimer

This is a personal research and learning project on public benchmark data. It is **not a production intrusion or anomaly detection system** and makes no claims to that effect. The HDFS logs are from 2008 and their labels come from the dataset's original authors; any result here describes this benchmark, not how a model would behave on your systems.

---

## Findings

Real results from the stages built so far. Model results will be added as each stage lands.

### The data

- **Every line parses, and the totals reconcile exactly.** One regex splits all **11,175,629** raw lines into date, time, pid, level, component and message, with **0** sent to quarantine. Raw = bronze + quarantine holds to the line, and the log spans **38.69 hours** (9 Nov 2008 20:35 to 11 Nov 2008 11:16), matching LogHub's published 38.7.
- **One line in eight belongs to two sessions.** **1,402,056** lines (12.5%) name two HDFS blocks, 9,773,573 name one and none name zero. Grouping lines by block without exploding them first would drop 1.4M events from one of the two sessions; exploding them gives **12,577,685** session events.
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

### Data quality checks

- Each stage writes a JSON run log with its row counts, timings and pass/fail checks, and fails loudly when a check fails: raw = bronze + quarantine, every block has exactly one label, every line gets an event ID, and no training-period line is UNK.

### Exploratory analysis

- Lines per hour by level, level mix, lines and non-INFO lines per component, blocks named per line, and the label balance, written to a small JSON aggregate that the dashboard will read.

### Planned stages

- **Sessionization (gold):** block-level event sequences and count vectors, joined to labels.
- **Models:** PCA, Isolation Forest, LightGBM and a DeepLog-style LSTM, tuned on the validation period.
- **Evaluation:** bootstrap confidence intervals, the random vs chronological gap, a duplicate-sequence analysis, alerts per hour and detection latency.
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
4. **Sessions and labels:** block IDs are extracted from each message (a line can name two), each block's first event fixes its place in the chronological split, and labels are joined on block ID.
5. **Mask and template:** IPs, paths, block IDs and numbers are masked, and each line is tagged with a template learned on the training period only.
6. **Reconcile:** row counts are checked against the raw file and LogHub's published totals at every stage.

---

## Tech Stack

| Layer | Technology | Details |
|-------|-----------|---------|
| **Processing** | PySpark 3.5 (local mode) | Regex parsing, explode, window functions, partitioned Parquet |
| **Storage** | Parquet | Bronze/silver/gold layers on local disk |
| **Parsing** | Drain3 | Template mining on the training period; templates compiled to Spark SQL regexes for matching |
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
docker compose run --rm loglens make test
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
│   └── parse_eval.py             # Grouping accuracy, trace check, threshold sweep
│
├── tests/
│   ├── test_ingest_regex.py
│   └── test_parsing.py
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
raw text ──► bronze (Parquet by hour) ──► split (session start) ──► silver (event ID per line, per block) ──► gold (sessions) ──► models ──► data/dashboard/*.json ──► static page
               │ quarantine                 │ training cutoff ─────► Drain3 fit                                 planned            planned
               ▼                                                         │
           run log (counts, checks, timings)                     parse eval vs LogHub ground truth
```

- **Medallion layers.** Bronze holds every line as structured fields; silver adds an event ID and explodes lines to one row per block they name; gold will hold one row per session. Each layer is rebuilt from the one before it, so every stage is reproducible from the raw log.
- **The training cutoff is computed once.** The split stage records the first event of the first validation and test sessions; every later stage that fits anything reads those cutoffs from the run log, so no stage can quietly fit on the future.
- **Matching runs in the JVM.** Templates are compiled to Java regexes and matched in a single Spark SQL `CASE WHEN`, which avoids moving 11M rows through Python. The pandas UDF and lookup-join versions stay in the code as the benchmark and as a cross-check.
- **Run log.** Every stage writes `data/runlog/<dataset>_<stage>.json`. These feed the dashboard's pipeline-health panel.
- **The dashboard never reads raw data.** It reads only the small aggregates in `data/dashboard/`, so the page stays a single file.
- **Execution.** Spark runs in local mode inside one container. Bronze ingest takes about 4 minutes, the split about 1.3 minutes and the parse about 1.5 minutes (plus 4.7 for the benchmark), mostly spent reading through the Windows bind mount.

---

## Testing and Validation

**Unit tests (11):** the HDFS line regex (a typical line, a message containing colons, a line that must not parse); masking (IP with port, a path that contains a block ID, numbers but not digits inside words, idempotence); template regexes (a wildcard is exactly one token, ground-truth wildcards allow anything); and grouping accuracy (a perfect one-to-one case, and merges, splits and UNK lines all counted as wrong).

**Pipeline checks, on the full dataset:** the raw line count equals LogHub's published 11,175,629; raw = bronze + quarantine; the time span matches LogHub's 38.7 hours; every one of the 575,061 blocks has exactly one label; every line gets an event ID and no training-period line is UNK; the three matchers agree on every line; and ground-truth tagging reproduces LogHub's published event sequence for every block.

---

## Honest Limitations

- **No model results yet.** Nothing here yet answers the project's main question.
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
