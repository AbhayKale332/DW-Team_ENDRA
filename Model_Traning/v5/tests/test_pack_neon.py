"""tools/pack_neon.py: resampling, the 1 m label grid, and the location split."""

import numpy as np

from tools.pack_neon import OFFSETS, TILE, block_mean_rgb, cut, label_2x, split_site


def test_block_mean_rgb_marks_any_nodata_source_pixel():
    rgb = np.full((3, 10, 10), 100, np.uint8)
    rgb[:, 0, 0] = 0                               # one no-data pixel in block (0, 0)
    rgb[:, 5:, 5:] = 200
    img, ok = block_mean_rgb(rgb)
    assert img.shape == (2, 2, 3) and ok.tolist() == [[False, True], [True, True]]
    assert img[1, 1].tolist() == [200, 200, 200] and img[0, 1].tolist() == [100, 100, 100]


def test_label_2x_blocks_are_single_lidar_cells():
    chm = np.array([[1.0, -9999.0], [30.0, 0.0]], np.float32)
    h, ok = label_2x(chm, -9999.0)
    assert h.shape == (4, 4) and not ok[0:2, 2:4].any() and ok.sum() == 12
    assert (h[2:4, 0:2] == 30).all()


def test_cut_grid_covers_the_km_tile_on_even_offsets():
    assert all(o % 2 == 0 for o in OFFSETS) and OFFSETS[-1] + TILE == 2000
    rgb = np.full((2000, 2000, 3), 50, np.uint8)
    h = np.zeros((2000, 2000), np.float32)
    h[:1000] = 200.0                               # above max_h -> invalid
    ok = np.ones((2000, 2000), bool)
    got = list(cut(rgb, ok, h, ok, min_valid=0.5, max_h=150.0))
    assert len(got) == 8 and all(r >= 2 for r, *_ in got)
    assert all(v.shape == (TILE, TILE) and hh[v].max() <= 150 for *_, hh, v in got)


def test_split_is_by_location_and_train_never_touches_heldout():
    rng = np.random.default_rng(1)
    stats = {(e, n): {"valid": 1.0, "cover2": float(rng.random()), "mean": 5.0, "p95": 10.0}
             for e in range(250000, 262000, 1000) for n in range(4100000, 4110000, 1000)}
    sp = split_site(stats, n=24, n_val=1, n_test=2, seed=0)
    assert len(sp["test"]) == 2 and len(sp["val"]) == 1 and len(sp["train"]) == 21
    held = {(t["e"], t["n"]) for k in ("val", "test") for t in sp[k]}
    for t in sp["train"]:
        assert all(abs(t["e"] - e) > 1000 or abs(t["n"] - n) > 1000 for e, n in held)
    assert split_site(stats, 24, 1, 2, seed=0) == sp


def test_object_mask_finds_dropped_buildings_and_kept_wires():
    from tools.pack_neon import object_mask

    chm = np.zeros((20, 20), np.float32)
    dtm = np.full((20, 20), 100.0, np.float32)
    dsm = dtm.copy()
    dsm[2:6, 2:6] += 6.0                           # barn: in the DSM, 0 in the CHM
    chm[15, :] = 14.0                              # wire: in the CHM only
    chm[10:13, 10:13] = dsm[10:13, 10:13] = 8.0    # tree: in both
    dsm[10:13, 10:13] += 100.0
    bad = object_mask(chm, dsm, dtm, -9999.0)
    assert bad[2:6, 2:6].all() and bad[15].all()
    assert not bad[11, 11] and not bad[8, 8]
    chm[10:13, 10:13] = 8.0
    dsm[10:13, 10:13] = 100.0                      # crown the DSM missed: still a tree
    assert not object_mask(chm, dsm, dtm, -9999.0)[11, 11]
