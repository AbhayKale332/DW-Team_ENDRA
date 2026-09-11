import numpy as np

from config import Config
from dwdata.augment import (
    achievable_gsd_range, center_window, crop_and_scale, dihedral,
    photometric_jitter, sample_window,
)


def test_window_never_exceeds_source():
    """The v2 regression: a requested GSD must never produce a padded tile."""
    rng = np.random.default_rng(0)
    for (H, W, gsd) in ((1024, 1024, 0.33), (512, 512, 0.3), (1000, 1000, 1.0)):
        lo, hi = achievable_gsd_range(H, W, gsd, 512, 0.25, 2.0)
        for _ in range(500):
            t, l, win, eff = sample_window(H, W, gsd, 512, rng.uniform(lo, hi), rng)
            assert 0 <= t and t + win <= H
            assert 0 <= l and l + win <= W
            assert win <= min(H, W)


def test_achievable_range_is_clamped_to_source_extent():
    # 1024 px at 0.33 m can fill a 512 tile at up to 1024*0.33/512 = 0.66 m
    lo, hi = achievable_gsd_range(1024, 1024, 0.33, 512, 0.25, 2.0)
    assert abs(hi - 0.66) < 1e-6 and lo >= 0.25


def test_crop_and_scale_shapes_and_dtypes():
    rgb = np.zeros((300, 300, 3), np.uint8)
    h = np.zeros((300, 300), np.float32)
    s = np.zeros((300, 300), np.int64)
    v = np.ones((300, 300), bool)
    r, hh, ss, vv = crop_and_scale(rgb, h, s, v, 10, 10, 200, 128)
    assert r.shape == (128, 128, 3) and hh.shape == (128, 128)
    assert hh.dtype == np.float32 and ss.dtype == np.int64 and vv.dtype == bool


def test_crop_and_scale_preserves_metric_height():
    """Resampling changes pixel spacing, never the metre values."""
    h = np.full((256, 256), 17.5, np.float32)
    _, hh, _, _ = crop_and_scale(np.zeros((256, 256, 3), np.uint8), h,
                                 np.zeros((256, 256), np.int64),
                                 np.ones((256, 256), bool), 0, 0, 256, 512)
    assert np.allclose(hh, 17.5, atol=1e-3)


def test_invalid_never_becomes_valid_after_resample():
    v = np.zeros((64, 64), bool)
    v[:32] = True
    _, _, _, vv = crop_and_scale(np.zeros((64, 64, 3), np.uint8),
                                 np.zeros((64, 64), np.float32),
                                 np.zeros((64, 64), np.int64), v, 0, 0, 64, 128)
    assert vv[:64].all() and not vv[64:].any()


def test_center_window_is_deterministic():
    a = center_window(1024, 1024, 0.33, 512, 0.5)
    assert a == center_window(1024, 1024, 0.33, 512, 0.5)


def test_dihedral_is_invertible():
    a = np.arange(16).reshape(4, 4)
    for k in range(4):
        assert np.array_equal(dihedral(dihedral(a, k, False), (4 - k) % 4, False), a)


def test_photometric_jitter_changes_rgb_within_range():
    rng = np.random.default_rng(0)
    c = Config()
    rgb = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    out = photometric_jitter(rgb, c, rng)
    assert out.dtype == np.uint8 and out.shape == rgb.shape
    assert not np.array_equal(out, rgb)
