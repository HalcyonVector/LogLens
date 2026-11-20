# LogLens: Do Sequence Models Really Catch Anomalous Logs?

Published log anomaly detectors report near-perfect scores on the HDFS benchmark, but most of those numbers come from random train/test splits that let a model see the future. LogLens asks what is left once the evaluation is honest about time. It takes the 11.2 million raw lines of the LogHub HDFS_v1 log, parses them in PySpark into bronze, silver and gold Parquet layers, groups them into 575,061 block-level sessions, and compares a DeepLog-style LSTM against PCA, Isolation Forest and a supervised LightGBM on a strict chronological split. It also measures how much a random split inflates each score, how many test sequences the model has already seen verbatim in training, and how many alerts per hour each model would raise. The method is then repeated on BGL (no natural sessions) and the pipeline is scaled to all 211 million Thunderbird lines.

Built with Python and PySpark (Parquet medallion layers) on the public LogHub datasets, run in Docker. Drain3, scikit-learn, LightGBM and PyTorch join as the parsing and modelling stages land.

> **What it's for:** checking whether a log anomaly detector's headline score survives a time-ordered evaluation, and seeing, session by session, which event in a sequence made the model raise an alert.

**Live demo:** not published yet. The results dashboard (a single static page) will be linked here once the first models are evaluated.

**Status:** in progress. The ingest stage and its data checks are complete; parsing, sessionization, models and the dashboard are next. Screenshots will be added with the dashboard.

---

## Disclaimer

This is a personal research and learning project on public benchmark data. It is **not a production intrusion or anomaly detection system** and makes no claims to that effect. The HDFS logs are from 2008 and their labels come from the dataset's original authors; any result here describes this benchmark, not how a model would behave on your systems.

---

## Findings

Real results from the stages built so far. Model results will be added as each stage lands.

- **Every line parses, and the totals reconcile exactly.** One regex splits all **11,175,629** raw lines into date, time, pid, level, component and message, with **0** sent to quarantine. Raw = bronze + quarantine holds to the line, and the log spans **38.69 hours** (9 Nov 2008 20:35 to 11 Nov 2008 11:16), matching LogHub's published 38.7.
- **One line in eight belongs to two sessions.** **1,402,056** lines (12.5%) name two HDFS blocks, 9,773,573 name one and none name zero. Grouping lines by block without exploding them first would drop 1.4M events from one of the two sessions; exploding them gives **12,577,685** session events.
- **The levels say almost nothing on their own.** There are **no ERROR lines at all**: 96.8% INFO and 3.2% WARN (362,793 lines), and **98.2%** of those WARN lines come from a single component, `dfs.DataNode$DataXceiver`. Four components produce 98.8% of all lines.
- **Accuracy would be meaningless here.** Of 575,061 block sessions, **16,838** are labelled anomalous (2.93%), so a model that calls everything normal scores 97.07% accuracy. LogLens reports precision, recall, F1 and PR-AUC instead.
- **Volume swings 34x from hour to hour**, from about 24,500 lines in the quietest hours to about 845,000 in the busiest. The chronological split is therefore made by session order, not by clock time, so each part gets the intended share of sessions.

---

## Features

### Ingest (bronze layer)

- Spark reads the raw text, splits each line with one regex and parses the `yyMMdd HHmmss` timestamp.
- Lines that fail the regex or the timestamp go to a **quarantine table** rather than being dropped, so nothing disappears silently.
- A `line_id` keeps the original file order, which breaks ties between events in the same second (HDFS timestamps have one-second resolution).
- Output is Parquet partitioned by hour (40 partitions for HDFS).

### Data quality checks

- Each stage writes a JSON run log with its row counts, timings and pass/fail checks.
- Bronze checks that the raw line count matches LogHub's published count and that raw = bronze + quarantine. The stage fails loudly if either check fails.

### Exploratory analysis

- Lines per hour by level, level mix, lines and non-INFO lines per component, blocks named per line, and the label balance, written to a small JSON aggregate that the dashboard will read.

### Planned stages

- **Parsing (silver):** Drain3 templates learned on the training period only, matched at scale with Spark-native regexes, and scored against Loghub-2.0's ground-truth templates.
- **Sessionization (gold):** block-level event sequences and count vectors, joined to labels.
- **Models:** PCA, Isolation Forest, LightGBM and a DeepLog-style LSTM, tuned on a validation period.
- **Evaluation:** bootstrap confidence intervals, the random vs chronological gap, a duplicate-sequence analysis, alerts per hour and detection latency.
- **Generalization and scale:** BGL with time windows, and the full Thunderbird log.
- **Dashboard:** one static page with inline SVG charts, built only from the exported aggregates.

---

## Data Source

| Source | Type | What It Provides |
|--------|------|------------------|
| [LogHub HDFS_v1](https://github.com/logpai/loghub) | Raw log + label file ([Zenodo](https://zenodo.org/records/8196385)) | 11,175,629 lines from a 200+ node Hadoop cluster, and a Normal/Anomaly label for each of 575,061 blocks |
| [LogHub BGL](https://github.com/logpai/loghub) | Raw log, labelled per line | 4,747,963 lines from a Blue Gene/L supercomputer (generalization run, planned) |
| [LogHub Thunderbird](https://github.com/logpai/loghub) | Raw log, labelled per line | 211,212,192 lines (scale run, planned) |
| [Loghub-2.0](https://github.com/logpai/loghub-2.0) | Annotated templates | Ground-truth template for each line, to measure parser accuracy (planned) |

**How the data is cleaned and joined:**

1. **Download:** `scripts/download_data.py` fetches each archive from LogHub's Zenodo record and verifies its MD5 checksum before extracting.
2. **Split:** a regex splits each line into date, time, pid, level, component and message; non-matching lines are quarantined.
3. **Order:** each line keeps its original file position (`line_id`) alongside its timestamp.
4. **Partition:** bronze is written as Parquet partitioned by hour.
5. **Reconcile:** row counts are checked against the raw file and LogHub's published total.

---

## Tech Stack

| Layer | Technology | Details |
|-------|-----------|---------|
| **Processing** | PySpark 3.5 (local mode) | Regex parsing, explode, window functions, partitioned Parquet |
| **Storage** | Parquet | Bronze/silver/gold layers on local disk |
| **Environment** | Docker (Python 3.12, Java 17) | Avoids Spark's Windows dependencies; code and data are bind-mounted |
| **Parsing** | Drain3 | Log template mining (next stage) |
| **Testing** | pytest | Unit tests for the line regex |

---

## Prerequisites

- **Docker Desktop** (or Docker Engine on Linux): [Download here](https://www.docker.com/products/docker-desktop/)
- About **3 GB of free disk** for HDFS_v1 and its Parquet layers

No API keys and no accounts.

---

## Quick Start

```bash
docker compose build
docker compose run --rm loglens make download   # HDFS_v1 from Zenodo, checksum verified
docker compose run --rm loglens make bronze     # raw text -> bronze Parquet + run log
docker compose run --rm loglens make eda        # aggregates -> data/dashboard/hdfs_eda.json
docker compose run --rm loglens make test
```

Spark's driver memory defaults to 5 GB; set `LOGLENS_DRIVER_MEMORY` to change it.

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
│   ├── config.py                 # Paths, per-dataset regex and expected counts
│   ├── spark.py                  # Laptop-tuned Spark session
│   ├── runlog.py                 # Per-stage JSON run log and timers
│   ├── ingest.py                 # Bronze layer with quarantine and reconciliation
│   └── eda.py                    # Exploratory aggregates for the dashboard
│
├── tests/
│   └── test_ingest_regex.py
│
└── data/
    ├── raw/                      # Downloaded archives (not tracked)
    ├── bronze/                   # Parquet layers (not tracked)
    ├── runlog/                   # Per-stage run logs (not tracked)
    └── dashboard/                # Small aggregates the dashboard reads (tracked)
```

---

## Architecture

```
raw text ──► bronze (Parquet by hour) ──► silver (event IDs) ──► gold (sessions) ──► models ──► data/dashboard/*.json ──► static page
               │ quarantine                 │ planned               │ planned           │ planned
               ▼
           run log (counts, checks, timings)
```

- **Medallion layers.** Bronze holds every line as structured fields; silver will add a template ID per line; gold will hold one row per session. Each layer is rebuilt from the one before it, so every stage is reproducible from the raw log.
- **Run log.** Every stage writes `data/runlog/<dataset>_<stage>.json`. These feed the dashboard's pipeline-health panel.
- **The dashboard never reads raw data.** It reads only the small aggregates in `data/dashboard/`, so the page stays a single file.
- **Execution.** Spark runs in local mode inside one container. Bronze ingest of HDFS_v1 takes about 4 minutes (about 48,000 lines per second), most of it reading through the Windows bind mount.

---

## Testing and Validation

**Unit tests (3):** the HDFS line regex on a typical line, a message that itself contains colons (the component must stop at the first one), and a line that must not parse.

**Pipeline checks, on the full dataset:** the raw line count equals LogHub's published 11,175,629; raw = bronze + quarantine; the time span matches LogHub's 38.7 hours; and the label file gives 575,061 sessions with 16,838 anomalies, matching the published figures.

---

## Honest Limitations

- **Only the ingest stage exists so far.** There are no model results yet, so nothing here yet answers the project's main question.
- **One benchmark, one era.** HDFS_v1 is a 2008 Hadoop log with block-level labels from its original authors; results may not carry over to modern systems, which is why BGL is planned as a second dataset.
- **Labels are taken as given.** No relabelling or label-noise analysis is done.
- **Local-mode Spark.** Everything runs on one machine; the timings describe a laptop, not a cluster.

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
