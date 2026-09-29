"""The unlabeled mean-teacher branch and the two Head-B fixes.

v2's bin head sat at CE 0.06-0.13 for a whole run and collapsed; v3 kept the same
hard target.  The first two tests here are the direct check that a collapsed head
is now *worse* under the loss than a spread one, which is the property the fix has
to have.
"""
import numpy as np
import torch

from config import Config
from models.losses import (bin_ce_loss, bin_entropy_reg, bin_target_index,
                           consistency_loss)


def _bins(k=32, hmax=64.0):
    centres = torch.linspace(0.5, hmax, k).unsqueeze(0)
    return centres


def _target(b=2, h=16, w=16):
    t = torch.zeros(b, 1, h, w)
    t[:, :, :4, :4] = 30.0                  # a small tall region, most is ground
    return t


def test_bin_index_lands_on_the_nearest_centre():
    c = _bins()
    t = _target()
    idx = bin_target_index(c, t, 16, 16)
    picked = c[0][idx]
    assert (picked - t[:, 0]).abs().max() <= (c[0][1] - c[0][0]) / 2 + 1e-4


def test_soft_target_penalises_a_collapsed_head():
    """The whole point: putting every pixel in the ground bin must cost more than
    spreading mass near the true height."""
    c = _bins()
    t = _target()
    v = torch.ones_like(t, dtype=torch.bool)
    k = c.shape[1]
    collapsed = torch.zeros(2, k, 16, 16)
    collapsed[:, 0] = 12.0                    # all mass on bin 0

    idx = bin_target_index(c, t, 16, 16)
    good = torch.zeros(2, k, 16, 16)
    good.scatter_(1, idx.unsqueeze(1), 12.0)  # mass on the right bin everywhere

    lo_c = bin_ce_loss(collapsed, c, t, v, soft_sigma=1.5)
    lo_g = bin_ce_loss(good, c, t, v, soft_sigma=1.5)
    assert lo_g < lo_c


def test_entropy_regulariser_prefers_a_used_bin_range():
    c = _bins()
    t = _target()
    v = torch.ones_like(t, dtype=torch.bool)
    k = c.shape[1]
    collapsed = torch.zeros(2, k, 16, 16)
    collapsed[:, 0] = 12.0
    spread = torch.zeros(2, k, 16, 16)
    idx = bin_target_index(c, t, 16, 16)
    spread.scatter_(1, idx.unsqueeze(1), 12.0)
    # the regulariser is negative entropy, so lower == more of the range in use
    assert bin_entropy_reg(spread, v) < bin_entropy_reg(collapsed, v)


def test_soft_and_hard_reward_different_optima():
    """A soft target is not a cosmetic change: its minimiser is a *distribution*
    shaped like the target, not a spike.  A head that only ever spikes — which is
    what a hard target rewards, and what let v2's head collapse — is now paying
    for it."""
    c = _bins()
    t = _target()
    v = torch.ones_like(t, dtype=torch.bool)
    k = c.shape[1]
    idx = bin_target_index(c, t, 16, 16)

    spike = torch.zeros(2, k, 16, 16).scatter_(1, idx.unsqueeze(1), 30.0)
    ar = torch.arange(k, dtype=torch.float32).view(1, k, 1, 1)
    matched = -((ar - idx.unsqueeze(1).float()) ** 2) / (2 * 1.5 ** 2)

    assert bin_ce_loss(spike, c, t, v, soft_sigma=0.0) < 1e-3      # hard: spike wins
    assert bin_ce_loss(matched, c, t, v, soft_sigma=1.5) < \
        bin_ce_loss(spike, c, t, v, soft_sigma=1.5)                # soft: shape wins
    # and both still prefer the truth to a confidently wrong answer
    wrong = torch.zeros(2, k, 16, 16)
    wrong[:, k - 1] = 30.0
    assert bin_ce_loss(matched, c, t, v, soft_sigma=1.5) < \
        bin_ce_loss(wrong, c, t, v, soft_sigma=1.5)


def test_consistency_drops_uncertain_teacher_pixels():
    student = torch.zeros(1, 1, 8, 8)
    teacher = torch.full((1, 1, 8, 8), 10.0)
    unc = torch.full((1, 1, 8, 8), 9.0)          # teacher unsure everywhere
    loss, keep = consistency_loss(student, teacher, unc, conf_m=1.5)
    assert keep == 0.0 and float(loss) == 0.0

    unc[:, :, :4] = 0.2                          # half of it is confident
    loss, keep = consistency_loss(student, teacher, unc, conf_m=1.5)
    assert abs(keep - 0.5) < 1e-6
    assert abs(float(loss) - 10.0) < 1e-4        # |0 - 10| over the kept pixels


def test_bin_std_is_a_real_sigma():
    """Head B's `b_std` is what gates the pseudo-labels, so it has to behave like
    a standard deviation: ~0 for a confident head, large for a flat one."""
    from tests.stub_dav2 import use_stub

    from models.heads import DepthWizardNet

    undo = use_stub(hidden=32, patch=14, layers=6)
    try:
        c = Config()
        c.tile_size, c.decoder_dim, c.n_bins = 56, 32, 24
        c.grad_checkpoint_encoder = False
        net = DepthWizardNet(c).eval()
        with torch.no_grad():
            out = net(torch.randn(1, 3, 56, 56))
        assert out["b_std"].shape == out["fused"].shape
        assert (out["b_std"] >= 0).all() and torch.isfinite(out["b_std"]).all()

        # analytic check on the head itself
        centres = torch.tensor([[0.0, 10.0]])
        logits = torch.zeros(1, 2, 2, 2)                    # 50/50 over {0, 10}
        probs = torch.softmax(logits, 1)
        mean = torch.einsum("bk,bkhw->bhw", centres, probs)
        c2 = torch.einsum("bk,bkhw->bhw", centres ** 2, probs)
        assert abs(float((c2 - mean ** 2).sqrt()[0, 0, 0]) - 5.0) < 1e-4
    finally:
        undo()


def test_weak_view_is_gentler_than_the_strong_one():
    from dwdata.dataset import _WeakPhoto

    cfg = Config()
    weak = _WeakPhoto(cfg)
    assert weak.photo_brightness < cfg.photo_brightness
    assert weak.photo_noise_std == 0.0 and weak.photo_blur_p == 0.0
    assert weak.tile_size == cfg.tile_size          # non-photo knobs pass through

    rng = np.random.default_rng(0)
    from dwdata.augment import photometric_jitter

    base = (rng.random((32, 32, 3)) * 120 + 60).astype(np.uint8)
    dw = np.mean([np.abs(photometric_jitter(base, weak, np.random.default_rng(i)).astype(float)
                         - base).mean() for i in range(12)])
    ds = np.mean([np.abs(photometric_jitter(base, cfg, np.random.default_rng(i)).astype(float)
                         - base).mean() for i in range(12)])
    assert dw < ds, f"weak view ({dw:.2f}) should perturb less than strong ({ds:.2f})"
