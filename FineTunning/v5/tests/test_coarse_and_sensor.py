"""v5 training-side changes: coarse-label supervision and the Cartosat augmentations."""

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from models.losses import StratumBalancer, coarse_flags, coarse_terms, compute_losses


def _out(pred, B, S, K=8):
    return {"a": pred, "b": pred, "fused": pred, "alpha": torch.rand(B, 1, S, S),
            "seg": torch.randn(B, 8, S, S),
            "b_logits": torch.randn(B, K, S // 2, S // 2),
            "b_centres": torch.linspace(0, 100, K).unsqueeze(0).repeat(B, 1)}


def _blocky_scene(S=32):
    """Sharp truth, and a label that only knows its 4x4 block means — the
    information content of ~2 m stereo on 0.5 m pixels."""
    t = torch.zeros(1, 1, S, S)
    t[..., 9:23, 5:19] = 15.0
    blocky = F.avg_pool2d(t, 4).repeat_interleave(4, -1).repeat_interleave(4, -2)
    return t, blocky


def test_coarse_terms_do_not_punish_detail_inside_a_block():
    sharp, blurry = _blocky_scene()
    cfg = Config()
    v = torch.ones_like(sharp, dtype=torch.bool)
    full_px = compute_l1(sharp, blurry, v)
    pooled, _ = coarse_terms(sharp, F.avg_pool2d(sharp, 4).repeat_interleave(4, -1)
                             .repeat_interleave(4, -2), v, cfg)
    assert float(pooled) < 1e-3 < full_px      # the sharp map matches the 4x4 means


def compute_l1(p, t, v):
    return float((p - t).abs()[v].mean())


def test_coarse_flags_follow_the_source_prefix():
    cfg = Config()
    cfg.coarse_label_sources = "dfc23,india_labeled"
    f = coarse_flags({"src": ["gamus", "dfc23_g050", "india_labeled"]}, cfg, "cpu")
    assert f.view(-1).tolist() == [False, True, True]
    assert coarse_flags({"src": ["gamus", "synrs3d_g05"]}, cfg, "cpu") is None
    cfg.coarse_label_sources = ""
    assert coarse_flags({"src": ["dfc23_g050"]}, cfg, "cpu") is None


def test_coarse_samples_leave_the_edge_terms():
    """A blurry DFC23 label must not drive the normal / flatness / bin terms."""
    cfg = Config()
    cfg.coarse_label_sources = "dfc23"
    B, S = 2, 32
    sharp, blurry = _blocky_scene(S)
    tgt = torch.cat([sharp, blurry])
    val = torch.ones(B, 1, S, S, dtype=torch.bool)
    pred = torch.cat([sharp, sharp]).requires_grad_(True)
    batch = {"target": tgt, "valid": val, "cls": torch.full((B, S, S), 7),
             "gsd_m": torch.full((B,), 0.5), "src": ["gamus", "dfc23_g050"]}
    loss, st = compute_losses(_out(pred, B, S), batch, cfg, StratumBalancer(0.0))
    assert float(st["normal"]) < 1e-5 and float(st["flat"]) < 1e-5   # sample 0 is exact
    assert float(st["coarse"]) < 1e-2                                 # pooled: forgiving
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    # the same batch treated the v4 way does penalise the sharp prediction
    cfg.coarse_label_sources = ""
    _, st4 = compute_losses(_out(pred.detach(), B, S), batch, cfg, StratumBalancer(0.0))
    assert float(st4["normal"]) > float(st["normal"]) + 1e-3


def test_veg_mask_drops_zeroed_trees_from_coarse_samples():
    from dwdata.dataset import exg_mask

    rgb = np.full((8, 8, 3), 120, np.uint8)
    rgb[:4] = (40, 140, 40)                    # green canopy
    m = exg_mask(rgb, 0.05)
    assert m[:4].all() and not m[4:].any()


def test_pansharpen_sim_blurs_colour_not_luminance():
    from dwdata.gpu_aug import sensor_augment

    cfg = Config()
    cfg.aug_pansharp_p, cfg.aug_pansharp_lo, cfg.aug_pansharp_hi = 1.0, 3.0, 3.0
    x = 0.3 + 0.4 * torch.rand(2, 3, 48, 48)          # away from the [0,1] clamp
    y = sensor_augment(x.clone(), cfg)
    lum = lambda t: 0.299 * t[:, 0] + 0.587 * t[:, 1] + 0.114 * t[:, 2]  # noqa: E731
    assert (lum(y) - lum(x)).abs().max() < 1e-4
    chroma = lambda t: t[:, 0] - lum(t)                    # noqa: E731
    hf = lambda c: (c - F.avg_pool2d(c[:, None], 3, 1, 1)[:, 0]).abs().mean()  # noqa: E731
    assert hf(chroma(y)) < 0.5 * hf(chroma(x))


def test_grayscale_aug_and_defaults_off():
    from dwdata.gpu_aug import sensor_augment

    cfg = Config()
    x = torch.rand(2, 3, 16, 16)
    assert sensor_augment(x, cfg) is x                     # v4 config: untouched
    cfg.aug_gray_p = 1.0
    y = sensor_augment(x.clone(), cfg)
    assert torch.allclose(y[:, 0], y[:, 1]) and torch.allclose(y[:, 1], y[:, 2])
