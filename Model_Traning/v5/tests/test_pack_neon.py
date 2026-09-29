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
    h[:1200] = 200.0                               # above max_h -> invalid
    ok = np.ones((2000, 2000), bool)
    got = list(cut(rgb, ok, h, ok, min_valid=0.5, max_h=150.0))
    assert len(got) == 2 and all(r == 1 for r, *_ in got)
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


def test_bounded_yields_every_result_and_errors():
    from concurrent.futures import ThreadPoolExecutor

    from tools.pack_neon import _bounded

    def f(x):
        if x == 3:
            raise ValueError("bad")
        return x * 2

    with ThreadPoolExecutor(2) as ex:
        got = list(_bounded(ex, f, list(range(10)), 3))
    assert sorted(g for g in got if not isinstance(g, Exception)) == [0, 2, 4, 8, 10, 12, 14, 16, 18]
    assert sum(isinstance(g, ValueError) for g in got) == 1


def test_site_caps_keep_tall_forest_and_cut_scrub_spikes():
    from tools.pack_neon import site_height_caps

    caps = site_height_caps({"WREF": [56.0, 57.0, 60.0], "MOAB": [5.0, 5.6, 64.0], "EMPTY": []})
    assert caps == {"MOAB": 18.4, "WREF": 95.5}              # 1.5 x median + 10


def test_cap_mask_grows_around_spikes_and_stays_valid_only():
    from tools.pack_neon import cap_mask

    h = np.zeros((20, 20), np.float32)
    h[10, 10] = 100.0
    v = np.ones((20, 20), bool)
    v[10, 11] = False
    bad = cap_mask(h, v, 30.0)
    assert bad[8:13, 8:13].sum() == 24 and bad.sum() == 24     # 5 x 5 minus the invalid px
    assert not cap_mask(h, v, 200.0).any()


def test_clean_is_idempotent(tmp_path):
    import json

    from dwdata.packed import PackedStore, ShardWriter
    from tools.pack_neon import clean

    root = tmp_path / "neon"
    for sp in ("train", "val", "test"):
        w = ShardWriter(root / sp, tile_px=16, gsd_m=0.5, shard_tiles=4, has_seg=False)
        for i in range(3):
            h = np.full((16, 16), 5.0, np.float32)
            if i == 0:
                h[8, 8] = 90.0                                   # a spike in a 5 m scrub
            w.add(f"MOAB_2023_{i}_0_r0_c0", np.zeros((16, 16, 3), np.uint8), h, None, np.ones((16, 16), bool))
        w.finalise()
        (root / sp / "landscape_v1.npy").write_bytes(b"stale")
    (root / "build_info.json").write_text("{}")
    clean(tmp_path)
    info = json.loads((root / "build_info.json").read_text())
    assert info["height_caps"]["MOAB"] < 90 and info["height_capped_px"]["val"] == {"MOAB": 25}
    assert not PackedStore(root / "val").get(0)[3][6:11, 6:11].any()
    assert not list(root.glob("*/landscape_v1*.npy"))
    clean(tmp_path)                                              # second run: same caps, same counts
    assert json.loads((root / "build_info.json").read_text()) == info
