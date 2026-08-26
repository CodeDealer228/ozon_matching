"""Real tests of StreamingAPLoss, imported from the exact copy in
generated/streaming_ap_loss.py (see test_extraction_matches_kernel.py for the
guarantee that copy is byte-identical to the kernel)."""
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from generated.streaming_ap_loss import StreamingAPLoss


def _toy_batch(n_pos=4, n_neg=4, seed=0):
    torch.manual_seed(seed)
    n = n_pos + n_neg
    labels = torch.zeros(n)
    labels[:n_pos] = 1.0
    scores = torch.rand(n, requires_grad=True)
    rows = torch.arange(n)
    anchors = torch.arange(n_pos)  # positive rows are always first -- matches labels[:n_pos]==1
    return scores, labels, rows, anchors


def test_forward_returns_finite_fp32_scalar_with_finite_gradient():
    scores, labels, rows, anchors = _toy_batch()
    ap_loss = StreamingAPLoss(len(scores))
    loss = ap_loss(scores, labels, rows, anchors)
    loss.backward()
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()


def test_rejects_non_fp32_state():
    scores, labels, rows, anchors = _toy_batch()
    ap_loss = StreamingAPLoss(len(scores))
    ap_loss.u_all = ap_loss.u_all.half()  # corrupt the FP32 contract on purpose
    try:
        ap_loss(scores, labels, rows, anchors)
    except RuntimeError as error:
        assert "FP32" in str(error)
    else:
        raise AssertionError("expected StreamingAPLoss to reject non-FP32 state")


def test_ema_state_only_updates_at_anchor_rows():
    scores, labels, rows, anchors = _toy_batch(n_pos=3, n_neg=5)
    ap_loss = StreamingAPLoss(len(scores))
    assert torch.all(ap_loss.u_all == 0) and torch.all(ap_loss.u_pos == 0)
    ap_loss(scores, labels, rows, anchors)
    anchor_rows = rows[anchors]
    touched = torch.zeros(len(scores), dtype=torch.bool)
    touched[anchor_rows] = True
    assert torch.all(ap_loss.u_all[touched] != 0), "anchor rows should have a nonzero EMA after one forward"
    assert torch.all(ap_loss.u_all[~touched] == 0), "non-anchor rows must stay untouched"
    assert torch.all(ap_loss.u_pos[~touched] == 0)


def test_reset_zeroes_state():
    scores, labels, rows, anchors = _toy_batch()
    ap_loss = StreamingAPLoss(len(scores))
    ap_loss(scores, labels, rows, anchors)
    assert torch.any(ap_loss.u_all != 0)
    ap_loss.reset_()
    assert torch.all(ap_loss.u_all == 0)
    assert torch.all(ap_loss.u_pos == 0)


def test_gradient_pushes_positives_up_and_negatives_down():
    """A positive scored BELOW a negative is the case AP-optimization exists to
    fix. The surrogate's gradient (w.r.t. this positive/negative pair, judged
    against a positive anchor) must point the way plain SGD would need it to:
    descent should raise the positive's score and lower the negative's."""
    scores = torch.tensor([0.2, 0.8, 0.5, 0.5], requires_grad=True)  # [pos, pos, pos, neg]
    labels = torch.tensor([1.0, 1.0, 1.0, 0.0])
    rows = torch.arange(4)
    anchors = torch.tensor([1])  # the confidently-correct positive is the anchor
    ap_loss = StreamingAPLoss(4, gamma=0.0)  # gamma=0: no EMA lag, immediate signal
    loss = ap_loss(scores, labels, rows, anchors)
    loss.backward()
    # SGD moves opposite the gradient: for the misranked positive (index 0) we need
    # grad < 0 (so score rises); for the negative (index 3) we need grad > 0 (so it falls).
    assert scores.grad[0] < 0, "misranked positive should get pushed up by descent"
    assert scores.grad[3] > 0, "the negative should get pushed down by descent"


def test_optimizing_the_surrogate_improves_real_average_precision():
    """End-to-end behavioral check: gradient-descending on nothing but this
    surrogate's gradient, starting from a near-random ranking, should raise
    the actual sklearn average_precision_score -- the metric this project
    reports and that the surrogate exists to approximate."""
    torch.manual_seed(3)
    n_pos, n_neg = 10, 30
    n = n_pos + n_neg
    labels = torch.zeros(n)
    labels[:n_pos] = 1.0
    scores = torch.rand(n, requires_grad=True)
    rows = torch.arange(n)
    anchors = torch.arange(n_pos)
    ap_before = average_precision_score(labels.numpy(), scores.detach().numpy())

    ap_loss = StreamingAPLoss(n, gamma=0.0)
    optimizer = torch.optim.SGD([scores], lr=3.0)
    for _ in range(60):
        optimizer.zero_grad()
        loss = ap_loss(scores, labels, rows, anchors)
        loss.backward()
        optimizer.step()
    ap_after = average_precision_score(labels.numpy(), scores.detach().numpy())

    assert ap_after > ap_before, f"AP should improve: before={ap_before:.4f} after={ap_after:.4f}"
    assert ap_after > 0.98, f"60 SGD steps on 40 points should nearly separate them: got {ap_after:.4f}"
