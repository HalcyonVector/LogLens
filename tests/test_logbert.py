import numpy as np
import torch

from loglens.logbert import LogBERT, batch_tensor, mask_batch, score_sequences
from loglens.lstm import PAD, UNK_RANK


def model(n_classes: int = 6) -> LogBERT:
    torch.manual_seed(0)
    return LogBERT(n_classes).eval()


def test_masking_hides_real_events_only_and_at_least_one_each():
    m = model()
    x = batch_tensor([[1, 2], [3, 4, 5, 1, 2, 3, 4, 5]], m.dist_id)
    masked, target = mask_batch(x, m.mask_id, np.random.default_rng(0))
    hidden = masked == m.mask_id
    assert hidden.any(dim=1).all()                     # every session predicts something
    assert not hidden[:, 0].any()                      # never the DIST token
    assert not hidden[x == PAD].any()                  # never padding
    assert (target[hidden] == x[hidden]).all() and (target[~hidden] == -100).all()


def test_scores_one_rank_and_nll_per_event_with_unk_outside_top_g():
    out = score_sequences(model(), [[1, 5, 2], [3]], unk_class=5)
    assert [len(o["ranks"]) for o in out] == [3, 1]
    assert out[0]["ranks"][1] == UNK_RANK and out[0]["nll"][1] == 50.0
    assert all(1 <= r <= 5 for r in (out[0]["ranks"][0], out[0]["ranks"][2], out[1]["ranks"][0]))


def test_a_sessions_scores_do_not_depend_on_its_batch():
    # Padding must be invisible to attention: scored alone or next to longer sessions,
    # the same session gets the same numbers.
    m = model()
    alone = score_sequences(m, [[1, 2, 3]], unk_class=99)[0]
    together = score_sequences(m, [[4, 4, 4, 4, 4, 4, 4], [1, 2, 3], [2]], unk_class=99)[1]
    assert np.array_equal(alone["ranks"], together["ranks"])
    assert np.allclose(alone["nll"], together["nll"], atol=1e-5)


def test_the_hidden_event_cannot_be_seen():
    m = model()
    a = batch_tensor([[1, 2, 3]], m.dist_id)
    b = batch_tensor([[1, 4, 3]], m.dist_id)
    a[0, 2] = b[0, 2] = m.mask_id                      # hide the second event in both
    with torch.no_grad():
        assert torch.allclose(m(a)[0][0, 2], m(b)[0][0, 2])


def test_padding_is_never_predicted():
    m = model()
    with torch.no_grad():
        logits, _ = m(batch_tensor([[1, 2], [3]], m.dist_id))
    assert torch.isinf(logits[..., PAD]).all()
