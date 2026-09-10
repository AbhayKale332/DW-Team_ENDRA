import numpy as np

from dwdata.preprocess import (
    PreprocSpec, apply_stretch, gsd_to_shape, hann2d, resize,
    scene_stretch_bounds, tile_origins, to_canonical,
)


def test_spec_roundtrip():
    s = PreprocSpec(canonical_gsd_m=0.4, tile_size=384)
    assert PreprocSpec.from_dict(s.to_dict()) == s


def test_normalise_matches_manual():
    s = PreprocSpec()
    rgb = np.full((4, 5, 3), 128, np.uint8)
    x = s.normalise(rgb)
    exp = (128 / 255 - np.asarray(s.mean)) / np.asarray(s.std)
    assert x.shape == (3, 4, 5)
    assert np.allclose(x[:, 0, 0], exp, atol=1e-5)


def test_denormalise_inverts():
    s = PreprocSpec()
    rgb = (np.random.rand(8, 8, 3) * 255).astype(np.uint8)
    assert np.abs(s.denormalise(s.normalise(rgb)).astype(int) - rgb.astype(int)).max() <= 1


def test_stretch_is_full_range_and_idempotent():
    rgb = (np.random.rand(64, 64, 3) * 40 + 100).astype(np.uint8)
    lo, hi = scene_stretch_bounds(rgb)
    a = apply_stretch(rgb, lo, hi)
    assert a.min() == 0 and a.max() == 255
    lo2, hi2 = scene_stretch_bounds(a)
    assert np.abs(apply_stretch(a, lo2, hi2).astype(int) - a.astype(int)).mean() < 3


def test_gsd_resample_preserves_ground_extent():
    # 1024 px @ 0.33 m == 337.9 m on the ground; at 0.5 m that must be ~676 px
    h, w = gsd_to_shape(1024, 1024, 0.33, 0.5)
    assert abs(h * 0.5 - 1024 * 0.33) < 0.5
    rgb = np.zeros((1024, 1024, 3), np.uint8)
    assert to_canonical(rgb, 0.33, PreprocSpec()).shape[:2] == (h, w)


def test_tiles_cover_extent_exactly():
    for ext in (512, 513, 1000, 3000):
        xs = tile_origins(ext, 512, 0.25)
        assert xs[0] == 0
        assert xs[-1] + 512 >= ext
        assert all(x + 512 <= max(ext, 512) for x in xs)


def test_hann_window_is_positive():
    w = hann2d(64)
    assert w.shape == (64, 64) and w.min() > 0


def test_resize_preserves_int_labels():
    seg = np.random.randint(0, 7, (32, 32)).astype(np.int64)
    out = resize(seg, (64, 64), "nearest")
    assert set(np.unique(out)).issubset(set(np.unique(seg)))
