import numpy as np

from loglens.stream import END_OF_LOG, UNK_RANK, Models, advance, new_state, next_timeout, window

END, UNK = 8, 7
T = (300_000,)  # one 5-minute timeout, in ms
FLUSH, LOG_END = 10_000_000, 9_000_000


def models(pca_thr: float = 2.5) -> Models:
    # rank = the event's class (so class >= 5 is "surprising"); END is unlikely (nll 10).
    rank = lambda w, y: np.asarray(y, dtype=float)
    nll = lambda w, y: np.where(np.asarray(y) == END, 10.0, 0.0)
    pca = lambda counts: float(counts.sum())  # flags a block with 3 or more events
    return Models(rank, nll, pca, thr_rank=5, thr_nll=5, thr_pca=pca_thr, h=2, unk=UNK, end=END, n_vocab=3)


def feed(st, ts, classes, wm, M=None):
    return advance(st, ts, classes, [0] * len(classes), wm, FLUSH, LOG_END, T, M or models())


def test_window_is_left_padded_to_the_history_length():
    assert window([3], 3) == [0, 0, 3]
    assert window([1, 2, 3, 4], 2) == [3, 4]


def test_deeplog_fires_on_the_surprising_event_itself():
    st = new_state(3)
    out = feed(st, [0, 1000, 2000], [1, 6, 1], wm=0)
    assert out == [("deeplog", 0, 1, 1000, 0, 2)]
    assert st["dl_ms"] == 1000


def test_silence_closes_the_block_at_last_event_plus_timeout():
    st = new_state(3)
    feed(st, [0, 1000], [1, 1], wm=0)
    out = feed(st, [], [], wm=301_000)
    # Closed at 1000 + 300000 ms: END is unlikely, so the END LSTM flags it; PCA sees 2 events.
    assert ("deeplog_end", 300, 1, 301_000, 0, 2) in out
    assert ("pca", 300, 0, 301_000, 0, 2) in out
    assert next_timeout(st, T, FLUSH) == FLUSH  # only the end-of-log verdict is left


def test_events_after_a_close_do_not_count_for_that_timeout():
    st = new_state(3)
    out = feed(st, [0, 1000, 900_000, 901_000], [1, 1, 1, 1], wm=0)
    pca = [d for d in out if d[0] == "pca"]
    assert pca == [("pca", 300, 0, 301_000, 0, 2)]   # closed by the gap, judged on 2 events
    out = feed(st, [], [], wm=FLUSH)
    assert ("pca", END_OF_LOG, 1, LOG_END, 0, 4) in out  # the end-of-log verdict sees all 4


def test_end_of_log_closes_everything_and_records_deeplog_negatives():
    st = new_state(3)
    feed(st, [0], [1], wm=0)
    out = feed(st, [], [], wm=FLUSH)
    models_seen = {(d[0], d[1]) for d in out}
    assert ("deeplog", 0) in models_seen and ("pca", END_OF_LOG) in models_seen and ("pca", 300) in models_seen
    assert [d for d in out if d[0] == "deeplog"][0][2] == 0
    assert next_timeout(st, T, FLUSH) is None


def test_unknown_events_are_always_surprising():
    st = new_state(3)
    out = feed(st, [0], [UNK], wm=0, M=models())
    assert out[0][:3] == ("deeplog", 0, 1)
    assert UNK_RANK > 5
