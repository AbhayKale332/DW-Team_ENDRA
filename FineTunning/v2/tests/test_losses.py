import math

import torch

from models.losses import (
    bin_ce_loss,
    compute_losses,
    regression_loss,
    seg_ce_loss,
    silog_loss,
)


class _Cfg:
    w_l1 = w_silog = 1.0
    w_grad = 0.5
    silog_lambda = 0.85
    silog_shift = 1.0
    w_head_a = w_head_b = w_fused = 1.0
    w_bin_ce = 0.5
    w_seg = 0.3


def test_regression_finite_and_zero_on_match():
    t = torch.rand(2, 1, 16, 16) * 20
    v = torch.ones_like(t, dtype=torch.bool)
    total, s = regression_loss(t.clone(), t, v, _Cfg)
    assert total.item() < 1e-3
    assert not any(math.isnan(x) for x in s.values())

    p = torch.rand(2, 1, 16, 16) * 20
    total2, _ = regression_loss(p, t, v, _Cfg)
    assert torch.isfinite(total2)


def test_silog_zero_on_match():
    t = torch.rand(4, 1, 8, 8) * 10 + 1
    v = torch.ones_like(t, dtype=torch.bool)
    assert silog_loss(t.clone(), t, v, 0.85, 1.0).item() < 1e-4


def test_bin_ce_finite():
    probs = torch.softmax(torch.randn(2, 16, 8, 8), dim=1)
    centres = torch.linspace(1, 100, 16).expand(2, 16)
    tgt = torch.rand(2, 1, 8, 8) * 50
    v = torch.ones_like(tgt, dtype=torch.bool)
    assert torch.isfinite(bin_ce_loss(probs, centres, tgt, v))


def test_seg_ce_respects_has_seg():
    logits = torch.randn(2, 7, 8, 8)
    cls = torch.randint(0, 7, (2, 8, 8))
    assert seg_ce_loss(logits, cls, torch.tensor([False, False])).item() == 0.0
    assert seg_ce_loss(logits, cls, torch.tensor([True, False])).item() != 0.0


def test_compute_losses_end_to_end():
    B, H, W = 2, 16, 16
    out = {
        "a": torch.rand(B, 1, H, W) * 10,
        "b": torch.rand(B, 1, H, W) * 10,
        "fused": torch.rand(B, 1, H, W) * 10,
        "b_probs": torch.softmax(torch.randn(B, 16, H, W), dim=1),
        "b_centres": torch.linspace(1, 100, 16).expand(B, 16),
        "seg": torch.randn(B, 7, H, W),
    }
    batch = {
        "target": torch.rand(B, 1, H, W) * 10,
        "valid": torch.ones(B, 1, H, W, dtype=torch.bool),
        "cls": torch.randint(0, 7, (B, H, W)),
        "has_seg": torch.tensor([True, True]),
    }
    total, stats = compute_losses(out, batch, _Cfg)
    assert torch.isfinite(total) and total.item() > 0
    assert {"l1_fused", "bin_ce", "seg_ce"} <= set(stats)
