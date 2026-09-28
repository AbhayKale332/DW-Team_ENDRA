import numpy as np
import torch

from config import Config
from models.losses import (
    StratumBalancer, bin_ce_loss, compute_losses, flatness_loss, gradient_loss,
    l1_loss, normal_loss, seg_ce_loss, silog_loss,
)


def _scene(B=2, S=32):
    tgt = torch.zeros(B, 1, S, S)
    tgt[:, :, 8:20, 8:20] = 18.0            # a tall block on flat ground
    valid = torch.ones(B, 1, S, S, dtype=torch.bool)
    return tgt, valid


def test_l1_and_silog_are_zero_at_the_optimum():
    tgt, valid = _scene()
    assert float(l1_loss(tgt, tgt, valid)) == 0.0
    assert float(silog_loss(tgt, tgt, valid, 0.85, 1.0)) < 1e-3


def test_gradient_and_normal_zero_at_optimum():
    tgt, valid = _scene()
    assert float(gradient_loss(tgt, tgt, valid)) < 1e-6
    assert float(normal_loss(tgt, tgt, valid)) < 1e-6


def test_stratum_weights_upweight_the_tail():
    """The fix for v2's -4.5 m bias above 15 m: rare heights must weigh more."""
    tgt, valid = _scene()
    b = StratumBalancer(beta=0.5, clip=5.0)
    w = b.weights(tgt, valid)
    ground = w[tgt < 1.0].mean()
    tall = w[tgt > 15.0].mean()
    assert tall > 2.0 * ground, (float(ground), float(tall))
    assert abs(float(w[valid].mean()) - 1.0) < 1e-4     # normalised, mean 1


def test_stratum_weights_disabled_when_beta_zero():
    tgt, valid = _scene()
    w = StratumBalancer(beta=0.0).weights(tgt, valid)
    assert torch.allclose(w, torch.ones_like(w))


def test_normal_loss_uses_physical_slope():
    """Same surface, different GSD -> the loss must see a different slope."""
    S = 32
    ramp = torch.linspace(0, 5, S).view(1, 1, 1, S).expand(1, 1, S, S).contiguous()
    flat = torch.zeros_like(ramp)
    valid = torch.ones_like(ramp, dtype=torch.bool)
    coarse = float(normal_loss(ramp, flat, valid, torch.tensor([2.0])))
    fine = float(normal_loss(ramp, flat, valid, torch.tensor([0.25])))
    assert fine > coarse


def test_flatness_penalises_invented_structure_on_flat_ground():
    tgt = torch.zeros(1, 1, 32, 32)
    valid = torch.ones_like(tgt, dtype=torch.bool)
    smooth = torch.zeros_like(tgt)
    bumpy = torch.from_numpy(
        (np.indices((32, 32)).sum(0) % 2).astype(np.float32) * 3.0
    ).view(1, 1, 32, 32)
    assert float(flatness_loss(smooth, tgt, valid)) < 1e-6
    assert float(flatness_loss(bumpy, tgt, valid)) > 1.0


def test_bin_ce_decreases_when_logits_point_at_the_right_bin():
    B, K, S = 2, 8, 16
    tgt = torch.full((B, 1, S, S), 10.0)
    valid = torch.ones(B, 1, S, S, dtype=torch.bool)
    centres = torch.linspace(0, 70, K).unsqueeze(0).repeat(B, 1)
    right = torch.searchsorted(
        (0.5 * (centres[:, 1:] + centres[:, :-1])).contiguous(),
        tgt.reshape(B, -1).contiguous())[0, 0].item()
    good = torch.zeros(B, K, S // 2, S // 2)
    good[:, right] = 12.0
    bad = torch.zeros(B, K, S // 2, S // 2)
    bad[:, (right + 4) % K] = 12.0
    assert float(bin_ce_loss(good, centres, tgt, valid)) < \
        float(bin_ce_loss(bad, centres, tgt, valid))


def test_seg_ce_ignores_unlabelled_only_batch():
    from config import SEG_IGNORE_INDEX, N_SEG_CLASSES

    logits = torch.randn(1, N_SEG_CLASSES, 8, 8)
    cls = torch.full((1, 8, 8), SEG_IGNORE_INDEX, dtype=torch.long)
    assert float(seg_ce_loss(logits, cls)) == 0.0


def test_compute_losses_is_finite_and_backprops():
    cfg = Config()
    B, S, K = 2, 32, 8
    tgt, valid = _scene(B, S)
    feat = torch.randn(B, 1, S, S, requires_grad=True)
    out = {
        "a": torch.relu(feat), "b": torch.relu(feat) * 1.1,
        "fused": torch.relu(feat), "alpha": torch.rand(B, 1, S, S),
        "seg": torch.randn(B, 8, S, S, requires_grad=True),
        "b_logits": torch.randn(B, K, S // 2, S // 2, requires_grad=True),
        "b_centres": torch.linspace(0, 100, K).unsqueeze(0).repeat(B, 1),
    }
    batch = {"target": tgt, "valid": valid,
             "cls": torch.zeros(B, S, S, dtype=torch.long),
             "gsd_m": torch.full((B,), 0.5)}
    loss, stats = compute_losses(out, batch, cfg, StratumBalancer(cfg.stratum_balance_beta))
    assert torch.isfinite(loss)
    loss.backward()
    assert feat.grad is not None and torch.isfinite(feat.grad).all()
    # stats are detached 0-d tensors, not floats: the trainer syncs them only on
    # the steps it logs.
    for k in ("l1", "silog", "grad", "normal", "flat", "bin", "seg"):
        assert torch.is_tensor(stats[k]) and not stats[k].requires_grad
        assert np.isfinite(float(stats[k]))


def test_all_invalid_batch_is_zero_not_nan():
    cfg = Config()
    B, S, K = 1, 16, 4
    valid = torch.zeros(B, 1, S, S, dtype=torch.bool)
    out = {
        "a": torch.zeros(B, 1, S, S), "b": torch.zeros(B, 1, S, S),
        "fused": torch.zeros(B, 1, S, S, requires_grad=True),
        "alpha": torch.zeros(B, 1, S, S),
        "seg": torch.randn(B, 8, S, S),
        "b_logits": torch.randn(B, K, S // 2, S // 2),
        "b_centres": torch.linspace(0, 100, K).unsqueeze(0),
    }
    batch = {"target": torch.zeros(B, 1, S, S), "valid": valid,
             "cls": torch.full((B, S, S), 7, dtype=torch.long),
             "gsd_m": torch.full((B,), 0.5)}
    loss, _ = compute_losses(out, batch, cfg, StratumBalancer())
    assert torch.isfinite(loss)
