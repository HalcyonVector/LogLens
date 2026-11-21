import numpy as np
import torch

from loglens.lstm import H, PAD, UNK_RANK, DeepLog, encoder, rank_sequences, weighted_windows, windows


def test_windows_left_pad_so_every_event_is_a_target():
    X, y = windows([5, 6, 7])
    assert X.shape == (3, H)
    assert list(y) == [5, 6, 7]
    assert (X[0] == PAD).all()                  # first event predicted from padding only
    assert list(X[2][-2:]) == [5, 6]            # third event sees the two before it


def test_weighted_windows_merge_identical_pairs():
    # Two sequences share their first window/target; weights add up.
    X, y, w = weighted_windows([[1, 2], [1, 3]], [10, 5])
    pairs = {(tuple(x), int(t)): float(c) for x, t, c in zip(X, y, w)}
    assert pairs[((PAD,) * H, 1)] == 15.0
    assert w.sum() == 2 * 10 + 2 * 5


def test_encoder_reserves_zero_for_padding():
    enc = encoder(["T01", "T02", "UNK"])
    assert enc == {"T01": 1, "T02": 2, "UNK": 3}


def test_unk_event_always_ranks_outside_top_g():
    torch.manual_seed(0)
    model = DeepLog(n_classes=4)
    out = rank_sequences(model, [[1, 3, 2]], unk_class=3)
    assert out[0]["ranks"][1] == UNK_RANK
    assert 1 <= out[0]["ranks"][0] <= 3 and 1 <= out[0]["ranks"][2] <= 3


def test_padding_is_never_predicted():
    model = DeepLog(n_classes=4)
    logits = model(torch.zeros((2, H), dtype=torch.long))
    assert torch.isinf(logits[:, PAD]).all()


def test_history_length_and_hidden_size_are_configurable():
    X, y, w = weighted_windows([[1, 2, 3]], [4], h=5)
    assert X.shape == (3, 5)
    torch.manual_seed(0)
    model = DeepLog(n_classes=4, hidden=16)
    out = rank_sequences(model, [[1, 2, 3]], unk_class=99, h=5)
    assert len(out[0]["ranks"]) == 3


def test_checkpoints_round_trip_and_clear(tmp_path, monkeypatch):
    import loglens.lstm as lstm

    monkeypatch.setattr(lstm, "CHECKPOINTS", tmp_path / "ckpt")
    assert lstm.checkpoint_load("run_x") is None
    lstm.checkpoint_save("run_x", {"m": {"f1": 0.5}})
    lstm.checkpoint_save("tune_y", {"h": 5})
    assert lstm.checkpoint_load("run_x") == {"m": {"f1": 0.5}}
    lstm.checkpoint_clear("run_")
    assert lstm.checkpoint_load("run_x") is None
    assert lstm.checkpoint_load("tune_y") == {"h": 5}   # other prefixes are kept
