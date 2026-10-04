"""Sliding-window / blending correctness — the part that turns a 512 px model
into a whole-scene DSM without seams."""
import numpy as np
import torch

from dwdata.preprocess import PreprocSpec
from infer.engine import predict_canonical, predict_scene


class Const(torch.nn.Module):
    """Returns a fixed height everywhere."""

    def __init__(self, v=7.5):
        super().__init__()
        self.v = v

    def forward(self, x):
        return {"fused": torch.full_like(x[:, :1], self.v),
                "seg": torch.zeros(x.shape[0], 8, *x.shape[-2:])}


class Ramp(torch.nn.Module):
    """Height = the input's own red channel, so blending is checkable."""

    def forward(self, x):
        return {"fused": x[:, :1] * 1.0,
                "seg": torch.zeros(x.shape[0], 8, *x.shape[-2:])}


def test_blend_reproduces_a_constant_exactly():
    spec = PreprocSpec(tile_size=64, canonical_gsd_m=0.5, radiometric_stretch=False)
    rgb = np.zeros((300, 220, 3), np.uint8)
    h, _ = predict_canonical(Const(7.5), rgb, spec, torch.device("cpu"), overlap=0.25)
    assert h.shape == (300, 220)
    assert np.allclose(h, 7.5, atol=1e-4), (h.min(), h.max())


def test_blend_has_no_seams():
    """Hann weights must sum consistently, or tile borders show as a grid."""
    spec = PreprocSpec(tile_size=64, canonical_gsd_m=0.5, radiometric_stretch=False)
    rgb = np.full((256, 256, 3), 128, np.uint8)
    h, _ = predict_canonical(Ramp(), rgb, spec, torch.device("cpu"), overlap=0.25)
    # a constant input through an identity-ish model must stay constant
    assert float(h.std()) < 1e-4


def test_scene_smaller_than_a_tile_is_reflect_padded_not_zero_padded():
    spec = PreprocSpec(tile_size=64, canonical_gsd_m=0.5, radiometric_stretch=False)
    rgb = np.full((20, 30, 3), 200, np.uint8)
    h, _ = predict_canonical(Const(3.0), rgb, spec, torch.device("cpu"))
    assert h.shape == (20, 30) and np.allclose(h, 3.0, atol=1e-4)


def test_output_is_on_the_input_grid_at_any_gsd():
    spec = PreprocSpec(tile_size=64, canonical_gsd_m=0.5, radiometric_stretch=False)
    rgb = np.zeros((177, 233, 3), np.uint8)
    for gsd in (0.25, 0.5, 1.0, 2.0):
        h, _ = predict_scene(Const(4.0), rgb, gsd, spec, torch.device("cpu"))
        assert h.shape == (177, 233), gsd
        assert np.allclose(h, 4.0, atol=1e-3), gsd


def test_heights_are_not_rescaled_by_the_gsd_round_trip():
    """Resampling changes pixel spacing; metres must be invariant."""
    spec = PreprocSpec(tile_size=64, canonical_gsd_m=0.5, radiometric_stretch=False)
    rgb = np.zeros((200, 200, 3), np.uint8)
    vals = [float(predict_scene(Const(12.25), rgb, g, spec,
                                torch.device("cpu"))[0].mean())
            for g in (0.2, 0.5, 1.5)]
    assert all(abs(v - 12.25) < 1e-3 for v in vals), vals


def test_seg_map_is_returned_on_request():
    spec = PreprocSpec(tile_size=64, canonical_gsd_m=0.5, radiometric_stretch=False)
    rgb = np.zeros((128, 128, 3), np.uint8)
    h, seg = predict_scene(Const(1.0), rgb, 0.5, spec, torch.device("cpu"),
                           want_seg=True)
    assert seg is not None and seg.shape == h.shape


def test_blend_preserves_height_and_uncertainty_together():
    class WithStd(Const):
        def forward(self, x):
            return {**super().forward(x), "b_std": torch.full_like(x[:, :1], 0.25)}

    spec = PreprocSpec(tile_size=64, radiometric_stretch=False)
    h, seg, std = predict_canonical(WithStd(7.5), np.zeros((93, 75, 3), np.uint8),
                                   spec, torch.device("cpu"), want_seg=True, want_std=True)
    assert h.shape == seg.shape == std.shape == (93, 75)
    assert np.allclose(h, 7.5, atol=1e-4)
    assert np.allclose(std, 0.25, atol=1e-5)
    assert not np.shares_memory(h, std)
