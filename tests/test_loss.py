import torch

from lakun.vendor.laya.common import QTYPES
from lakun.train import decision_loss


def test_laya_proper_score_loss_has_gradients_for_all_three_types() -> None:
    logits = torch.tensor([[1.0, 0.2, -1e4], [0.1, 0.8, -0.3], [0.4, -0.2, -1e4]],
                          requires_grad=True)
    batch = {
        "target": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
        "marker_mask": torch.tensor([[True, True, False], [True, True, True], [True, True, False]]),
        "qtype": torch.tensor([QTYPES["choice"], QTYPES["score"], QTYPES["noul"]]),
    }
    loss = decision_loss(logits, batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad[:, :2].abs().sum(dim=1) > 0).all()
