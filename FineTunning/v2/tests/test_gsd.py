import numpy as np

from dwdata.gsd import center_crop_or_pad, jitter_gsd, rescale_to_gsd, roundtrip_predict


def test_rescale_changes_spacing_not_heights(synthetic_tile):
    rgb, h, s = synthetic_tile
    r2, h2, s2 = rescale_to_gsd(rgb, h, s, src_gsd_m=1.0, dst_gsd_m=0.5)
    assert r2.shape[0] == rgb.shape[0] * 2
    # tall region peak height is preserved (bilinear keeps the plateau max)
    assert abs(h2.max() - h.max()) < 1.0


def test_center_crop_or_pad_shapes():
    a = np.arange(100 * 100, dtype=np.float32).reshape(100, 100)
    assert center_crop_or_pad(a, 64).shape == (64, 64)
    assert center_crop_or_pad(a, 128).shape == (128, 128)


def test_jitter_within_range(synthetic_tile):
    rgb, h, s = synthetic_tile
    rng = np.random.default_rng(1)
    _, _, _, g = jitter_gsd(rgb, h, s, 0.33, 0.25, 2.0, rng)
    assert 0.25 <= g <= 2.0


def test_roundtrip_preserves_geometry(synthetic_tile):
    rgb, h, _ = synthetic_tile

    def fake_predict(rgb_canon):
        # "model" returns a constant field == mean input intensity / 10
        return np.full(rgb_canon.shape[:2], rgb_canon.mean() / 10.0, np.float32)

    out = roundtrip_predict(fake_predict, rgb, src_gsd_m=0.33, canonical_gsd_m=0.5)
    assert out.shape == h.shape
    assert np.isfinite(out).all()
