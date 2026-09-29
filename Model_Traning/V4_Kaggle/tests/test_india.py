"""Indian-imagery ingest.

The single most important assertion here is that an unlabeled tile's zeros can
never be read as 0 m labels — that exact confusion is what put 41 % black
padding-at-zero into v2's training set.
"""
import numpy as np
from PIL import Image

from dwdata.india import (INDIA_AOIS, _deg2tile, ndsm_from_absolute, pack_labeled,
                          pack_unlabeled, pair_rasters, tile_gsd_m)
from dwdata.packed import PackedStore, store_exists


def _write(d, stem, h=600, w=600, height=None, seg=False):
    rng = np.random.default_rng(abs(hash(stem)) % 2 ** 31)
    Image.fromarray((rng.random((h, w, 3)) * 200 + 30).astype(np.uint8)) \
        .save(d / f"{stem}_rgb.png")
    if height is not None:
        np.save(d / "_tmp.npy", height)          # keep tifffile out of the test
        Image.fromarray(height.astype(np.float32), mode="F").save(d / f"{stem}_ndsm.tif")
    if seg:
        Image.fromarray(np.zeros((h, w), np.uint8), mode="L").save(d / f"{stem}_seg.png")


def test_pairing_by_filename(tmp_path):
    hh = np.zeros((600, 600), np.float32)
    hh[100:200, 100:200] = 12.0
    _write(tmp_path, "delhi_001", height=hh, seg=True)
    _write(tmp_path, "mumbai_002")               # rgb only
    recs = {r["stem"]: r for r in pair_rasters(tmp_path)}
    assert set(recs) == {"delhi_001", "mumbai_002"}
    assert recs["delhi_001"]["hgt"] is not None
    assert recs["delhi_001"]["seg"] is not None
    assert recs["mumbai_002"]["hgt"] is None


def test_pack_labeled_produces_a_usable_store(tmp_path):
    hh = np.zeros((600, 600), np.float32)
    hh[100:300, 100:300] = 12.0
    _write(tmp_path, "delhi_001", height=hh)
    out = tmp_path / "packed"
    pack_labeled(tmp_path, out, tile_px=256, gsd_m=0.5)
    assert store_exists(out)
    st = PackedStore(out)
    assert len(st) == 4                          # 600 px -> 2x2 whole 256 tiles
    rgb, h, cls, val = st.get(0)
    assert rgb.shape == (256, 256, 3) and h.shape == (256, 256)
    assert val.all(), "a fully-valid source tile must stay fully valid"
    assert abs(h[150, 150] - 12.0) < 1e-3


def test_unlabeled_tiles_carry_no_supervision(tmp_path):
    _write(tmp_path, "bhuvan_kolkata", h=520, w=520)
    out = tmp_path / "unl"
    pack_unlabeled(tmp_path, out, tile_px=256, gsd_m=0.56)
    st = PackedStore(out)
    assert len(st) == 4
    _rgb, h, _cls, val = st.get(0)
    assert not val.any(), "unlabeled tiles must have an all-False valid mask"
    assert (h == 0).all()                        # zeros, but never *valid* zeros


def test_flat_tiles_are_dropped(tmp_path):
    Image.fromarray(np.full((512, 512, 3), 12, np.uint8)).save(tmp_path / "ocean_rgb.png")
    out = tmp_path / "unl2"
    pack_unlabeled(tmp_path, out, tile_px=256, gsd_m=0.5)
    assert not store_exists(out) or len(PackedStore(out)) == 0


def test_absolute_dsm_becomes_height_above_ground():
    y, x = np.mgrid[0:400, 0:400]
    dem = (200 + 0.05 * x + 0.03 * y).astype(np.float32)
    dem[150:250, 150:250] += 20.0
    n = ndsm_from_absolute(dem, win=90)
    assert n.min() >= 0
    assert n[10, 10] < 2.0, "bare terrain should come out near zero"
    assert n[200, 200] > 12.0, "the block should survive as height"


def test_aoi_table_is_sane():
    assert len(INDIA_AOIS) >= 12
    kinds = {v[4] for v in INDIA_AOIS.values()}
    assert kinds == {"urban", "sparse", "hilly", "forested"}, \
        "the unlabeled set must span the four landscapes the rubric grades"
    for name, (lon0, lat0, lon1, lat1, _k) in INDIA_AOIS.items():
        assert 68 < lon0 < lon1 < 98, name          # inside India's longitudes
        assert 6 < lat0 < lat1 < 36, name           # ...and its latitudes
    x, y = _deg2tile(72.8, 19.0, 18)
    assert x > 0 and y > 0
    assert 0.4 < tile_gsd_m(18, 20.0) < 0.7          # z18 is roughly the working GSD
