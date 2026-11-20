import numpy as np

from loglens.baselines import label_conflicts, random_split
from loglens.metrics import best_f1_threshold, evaluate, prf

import pandas as pd


def test_prf_counts():
    y = np.array([1, 1, 0, 0, 1])
    pred = np.array([1, 0, 1, 0, 1])
    r = prf(y, pred)
    assert (r["tp"], r["fp"], r["fn"]) == (2, 1, 1)
    assert abs(r["f1"] - 2 / 3) < 1e-6


def test_best_threshold_separates_perfectly_separable_scores():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    thr = best_f1_threshold(y, s)
    assert 0.3 < thr <= 0.8


def test_threshold_comes_from_validation_not_test():
    y_val, s_val = np.array([0, 1]), np.array([0.2, 0.6])
    # On test the anomaly scores 0.5, below the validation threshold: it must be missed.
    y_te, s_te = np.array([0, 1]), np.array([0.1, 0.5])
    r = evaluate(y_val, s_val, y_te, s_te)
    assert r["threshold"] == 0.6
    assert r["recall"] == 0.0


def test_random_split_proportions_and_determinism():
    a, b = random_split(1000, seed=1), random_split(1000, seed=1)
    assert (a == b).all()
    assert (a == "train").sum() == 600 and (a == "val").sum() == 100 and (a == "test").sum() == 300


def test_label_conflicts_counts_sequences_with_both_labels():
    df = pd.DataFrame({"seq_hash": [1, 1, 2, 2, 3], "label": [0, 1, 0, 0, 1]})
    r = label_conflicts(df)
    assert r["distinct_sequences"] == 3
    assert r["conflicting_sequences"] == 1
    assert r["sessions_in_conflicting_sequences"] == 2
    assert r["anomalies_in_conflicting_sequences"] == 1


def test_prf_reports_false_positive_rate_and_prevalence():
    y = np.array([0, 0, 0, 0, 1])
    r = prf(y, np.array([1, 0, 0, 0, 1]))
    assert r["fpr"] == 0.25
    assert r["prevalence"] == 0.2


def test_f1_at_prevalence_matches_direct_computation():
    from loglens.metrics import f1_at_prevalence

    # 100 anomalies, 900 normals; recall 0.8 and fpr 0.1 -> tp 80, fp 90.
    p = 80 / 170
    expected = 2 * p * 0.8 / (p + 0.8)
    assert abs(f1_at_prevalence(0.8, 0.1, 0.1) - expected) < 1e-6


def test_same_detector_scores_higher_f1_when_anomalies_are_commoner():
    from loglens.metrics import f1_at_prevalence

    assert f1_at_prevalence(0.8, 0.01, 0.03) > f1_at_prevalence(0.8, 0.01, 0.02)
