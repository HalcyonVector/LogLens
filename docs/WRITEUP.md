# What is left of log anomaly detection once the test set comes from the future

A short write-up of the LogLens results. The full numbers, method and caveats are in the [README](../README.md), and every chart is on the [dashboard](https://halcyonvector.github.io/LogLens/).

## The question

Log anomaly detectors such as DeepLog report F1 scores close to 1.0 on the LogHub HDFS benchmark. Most of those scores come from random train/test splits, so the model trains on sessions from the same hours it is tested on. A detector in production only ever sees the past. LogLens asks how much of the published performance survives a split that respects time, and why.

## The data

- **HDFS_v1:** 11,175,629 lines from a Hadoop cluster over 38.7 hours, grouped by block ID into 575,061 sessions, 2.93% labelled anomalous by the dataset's authors.
- **BGL:** 4,747,963 lines from a Blue Gene/L supercomputer over 215 days, with an alert tag on every line, plus Loghub-2.0's ground-truth template for each line.
- **Thunderbird:** 211,212,192 lines (31.8 GB) from the Thunderbird supercomputer, used to test whether the pipeline holds up at scale.

Every line is parsed in PySpark into bronze (structured fields), silver (a Drain3 template per line) and gold (one row per session) Parquet layers. Row counts reconcile with LogHub's published totals at every stage, and the HDFS session logic reproduces LogHub's published event sequence for every one of the 575,061 blocks.

## The method

Sessions are ordered by their first event and split 60/10/30 into train, validation and test. The first validation event is the training cutoff: the template parser, every model and every threshold are fitted only on data before it, and thresholds are chosen on validation and applied unchanged to test. The same models are then run on a seeded random split with the same proportions, and the difference is corrected for the two test sets' different anomaly rates.

Nine detectors are compared: a warning-line rule, a sequence lookup, PCA on event counts (Xu et al. 2009), Isolation Forest, supervised LightGBM, and four variants of a DeepLog-style LSTM, including one with an END event and a tuned one. Scores carry 95% bootstrap intervals, LSTM results are averaged over 3 seeds, and every run is logged to MLflow.

## What the data shows

**1. On later data, a 2009 count model beats the LSTM.** PCA reaches F1 **0.964** [0.960, 0.968] against **0.870** [0.862, 0.877] for the best LSTM, a paired difference of **+9.4 points** [+8.7, +10.1]. The two rank sessions equally well (PR-AUC is tied); the LSTM loses at the alert threshold, with **902** false alarms against PCA's **7**. Published DeepLog, without an END event, reaches only 0.610.

**2. On HDFS the random split is not the main problem; repetition is.** 99.0% of chronological test sessions replay an event sequence already seen in training, so even an honest split mostly tests recognition. A random split adds 3 to 8 F1 points for the sequence models and under 1 for the count models. On the 1,699 test sessions with never-seen sequences, PCA's F1 falls to 0.844.

**3. The models fail differently.** One anomaly in five simply stops early: 683 test anomalies are exact prefixes of normal sessions, invisible to a next-event model. Published DeepLog and PCA miss almost disjoint sets of anomalies, so they are complements rather than rivals.

**4. Speed is the LSTM's real advantage.** A count model must wait until a session is over, and HDFS blocks can sit idle for hours, so a live PCA needs a 12.3-hour idle timeout and alerts a median **12.5 hours** after an anomaly starts. Published DeepLog fires on the surprising event itself, a median **6 seconds** in.

**5. The time-ordered test set is also the end of the log.** Only 59.6% of test sessions reach their delete events, against 92.3% in training, because the log stops. Part of what every model is judged on is that cut-off, not behaviour.

**6. On BGL, a one-line rule beats every model.** Flagging any window with a FATAL or FAILURE line scores F1 **0.854** on 1-hour windows; the best model reaches 0.657. The failures themselves change: 41.7% of test-period alert lines belong to categories never seen in training, and a parser fitted on the training period cannot name 69% of test lines. The common benchmark recipe (a random split plus a parser fitted on the whole log) turns LightGBM's honest **0.114** into **0.993**.

**7. At scale, parsing is the pipeline.** All 211M Thunderbird lines go from raw text to hourly windows in **3.6 hours** on one laptop, but ten times the lines of the 10% run cost 27 times the time: the template vocabulary keeps growing (9,037 templates at 10%, 69,831 on the full log), and learning and matching templates take 94% of the run. On HDFS, Spark only beats a single pandas process past about a million lines.

## What it means

- Report detectors on a time-ordered split, with thresholds fixed on an earlier validation period. The gap to a random split is small on HDFS and tens of points on BGL, so a random-split score says little about either.
- Report duplication. When almost every test sequence was seen in training, a high score measures memory, not detection.
- Judge detectors on alerts per hour and time to alert, not F1 alone. PCA is more accurate; DeepLog is hours faster. A practical system would use both.
- Treat parsing as part of the model. A parser's untuned default threshold hid a failure signal on HDFS, an unseen vocabulary broke every model on BGL, and the parser's growing vocabulary set the cost on Thunderbird.

## Caveats

These are public benchmark logs from 2005 to 2008 with labels from their original authors. The LSTM search was small, BGL results have no bootstrap intervals, the timings are single runs on one laptop, and the Thunderbird runs measure the data pipeline only, with no detector trained on them. The README's Honest Limitations section lists all of these in full.
