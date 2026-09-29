"""DFC23 Track 2 ingest.

Two things here are load-bearing and neither is obvious from the packer.

First, DFC23 discriminates the optical raster from the height raster by **parent
directory** and gives both the identical filename (`rgb/P_0199.tif` and
`dsm/P_0199.tif`).  `dwdata.india.pair_rasters` keys on filename *suffix*, so
handed that tree directly it claims whichever file it walks first as the RGB and
every scene comes out unlabelled — silently, as a store full of zero heights.

Second, the train/val split has to be by **scene**, not by tile.  Tiles cut from
one scene are near-duplicates of each other, so a tile-level split leaks the val
set into training and the reported number stops measuring anything.  DFC23 is
the only source in the mix that is the real deployment domain, which makes its
val number the one worth protecting.
"""
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from dwdata.packed import PackedStore, store_exists  # noqa: E402
from prepare_data import _dfc23_pairs, _dfc23_stage, prepare_dfc23  # noqa: E402


def _tif(path, arr, gsd=0.5):
    arr = np.asarray(arr)
    bands = 1 if arr.ndim == 2 else arr.shape[2]
    data = arr[None] if arr.ndim == 2 else np.transpose(arr, (2, 0, 1))
    with rasterio.open(
            path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
            count=bands, dtype=data.dtype,
            crs="EPSG:32643",
            transform=rasterio.transform.from_origin(0, 0, gsd, gsd)) as ds:
        ds.write(data)


def _scene(root, stem, h=600, w=600, tall=12.0, gsd=0.5):
    """One DFC23-shaped scene: rgb/<stem>.tif + dsm/<stem>.tif, same name."""
    (root / "rgb").mkdir(parents=True, exist_ok=True)
    (root / "dsm").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(stem)) % 2 ** 31)
    _tif(root / "rgb" / f"{stem}.tif",
         (rng.random((h, w, 3)) * 200 + 30).astype(np.uint8), gsd)
    ndsm = np.zeros((h, w), np.float32)
    ndsm[100:300, 100:300] = tall
    _tif(root / "dsm" / f"{stem}.tif", ndsm, gsd)
    return ndsm


class _Args:
    """The argparse namespace prepare_dfc23 reads."""
    def __init__(self, **kw):
        self.dfc23_dir = ""
        self.dfc23_tile = 128
        self.dfc23_max = 0
        self.dfc23_val_frac = 0.25
        self.dfc23_gsd = 0.5
        self.dfc23_max_height_m = 0.0
        self.dfc23_absolute_dsm = False
        self.force = True
        self.__dict__.update(kw)


# --- pairing -------------------------------------------------------------
def test_pairs_rgb_and_dsm_across_sibling_directories(tmp_path):
    _scene(tmp_path, "P_0199")
    _scene(tmp_path, "P_0309")
    recs = {r["stem"]: r for r in _dfc23_pairs(tmp_path)}
    assert set(recs) == {"P_0199", "P_0309"}
    for r in recs.values():
        assert r["rgb"].parent.name == "rgb"
        assert r["hgt"].parent.name == "dsm"


def test_an_rgb_with_no_matching_dsm_is_dropped(tmp_path):
    _scene(tmp_path, "P_0199")
    rng = np.random.default_rng(0)
    _tif(tmp_path / "rgb" / "P_9999.tif",
         (rng.random((64, 64, 3)) * 255).astype(np.uint8))
    assert [r["stem"] for r in _dfc23_pairs(tmp_path)] == ["P_0199"]


def test_missing_subdirectory_returns_nothing_rather_than_guessing(tmp_path):
    (tmp_path / "rgb").mkdir()
    _tif(tmp_path / "rgb" / "P_0199.tif", np.zeros((64, 64, 3), np.uint8))
    assert _dfc23_pairs(tmp_path) == []


def test_staging_renames_into_the_suffix_convention(tmp_path):
    _scene(tmp_path, "P_0199")
    recs = _dfc23_pairs(tmp_path)
    stage = tmp_path / "stage"
    _dfc23_stage(recs, stage)
    names = sorted(p.name for p in stage.iterdir())
    assert names == ["P_0199_ndsm.tif", "P_0199_rgb.tif"]
    # and pair_rasters — which could not read the original tree — now can
    from dwdata.india import pair_rasters
    got = pair_rasters(stage)
    assert len(got) == 1 and got[0]["hgt"] is not None


# --- packing -------------------------------------------------------------
def test_pack_produces_a_usable_store_with_heights_intact(tmp_path):
    src = tmp_path / "track2" / "train"
    for i in range(6):
        _scene(src, f"P_{i:04d}", tall=12.0)
    out = tmp_path / "out"
    prepare_dfc23(out, _Args(dfc23_dir=str(src)))

    stores = sorted(p.parent.parent.name for p in out.glob("dfc23_*/train/index.json"))
    assert stores, f"nothing packed: {sorted(out.rglob('*'))}"
    st = PackedStore(out / stores[0] / "train")
    assert len(st) > 0
    rgb, h, cls, val = st.get(0)
    assert rgb.shape == (128, 128, 3) and rgb.dtype == np.uint8
    assert h.shape == (128, 128)
    assert val.any()
    # the planted 12 m block survives packing somewhere in the store
    assert max(float(np.asarray(st.get(i)[1]).max()) for i in range(len(st))) \
        == pytest.approx(12.0, abs=0.1)


def test_split_is_by_scene_so_no_stem_appears_in_both(tmp_path):
    src = tmp_path / "track2" / "train"
    for i in range(8):
        _scene(src, f"P_{i:04d}")
    out = tmp_path / "out"
    prepare_dfc23(out, _Args(dfc23_dir=str(src), dfc23_val_frac=0.25))

    fam = sorted(out.glob("dfc23_*"))
    assert len(fam) == 1, f"one GSD family expected, got {fam}"
    tr, va = fam[0] / "train", fam[0] / "val"
    assert store_exists(tr) and store_exists(va)
    # tile stems are "<scene>_<y>_<x>"; the scene is what must not cross over
    scene = lambda s: s.rsplit("_", 2)[0]  # noqa: E731
    s_tr = {scene(s) for s in PackedStore(tr).stems}
    s_va = {scene(s) for s in PackedStore(va).stems}
    assert s_tr and s_va
    assert not (s_tr & s_va), f"scene leak across the split: {s_tr & s_va}"


def test_differing_gsds_are_packed_into_separate_stores(tmp_path):
    """0.5 m SuperView and 0.8 m Gaofen-2 scenes must not share one nominal.

    `index.json` carries a single `gsd_m` and `dataset.py` turns it straight into
    the crop's effective GSD, which is what ties apparent object size to metric
    height — so one store for both would be a 1.6x scale lie about half of it.
    """
    src = tmp_path / "track2" / "train"
    for i in range(5):
        _scene(src, f"SV_{i:04d}", gsd=0.5)
    for i in range(5):
        _scene(src, f"GF_{i:04d}", gsd=0.8)
    out = tmp_path / "out"
    prepare_dfc23(out, _Args(dfc23_dir=str(src), dfc23_val_frac=0.0))

    got = {p.parent.parent.name: PackedStore(p.parent).gsd_m
           for p in out.glob("dfc23_*/train/index.json")}
    assert len(got) == 2, got
    assert sorted(round(g, 2) for g in got.values()) == [0.5, 0.8]


def test_absolute_dsm_is_not_silently_accepted_as_ndsm(tmp_path, capsys):
    """A DSM in metres above sea level has a median far from 0.  Packing it as
    an nDSM poisons the target, and the only cheap defence is to say so."""
    src = tmp_path / "track2" / "train"
    (src / "rgb").mkdir(parents=True)
    (src / "dsm").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(5):
        _tif(src / "rgb" / f"P_{i:04d}.tif",
             (rng.random((600, 600, 3)) * 200 + 30).astype(np.uint8))
        _tif(src / "dsm" / f"P_{i:04d}.tif",
             np.full((600, 600), 214.0, np.float32))    # elevation ASL
    prepare_dfc23(tmp_path / "out", _Args(dfc23_dir=str(src), dfc23_val_frac=0.0,
                                          dfc23_tile=512))
    out = capsys.readouterr().out
    assert "HEIGHT CHECK" in out
    assert "ABSOLUTE DSM" in out, out


def test_a_store_with_no_class_rasters_does_not_claim_to_have_them(tmp_path):
    """`has_seg` is what `loaders.py` prints and what its `w_seg > 0` guard
    tests, so a store advertising labels it does not carry defeats the only
    check standing between a live `w_seg` and a loss term that is always zero —
    which is the exact bug that made every v4 run carry an inert seg head.
    DFC23 packs `cls=None` for every tile; `ShardWriter` defaults `has_seg` to
    True, and `pack_labeled` used to take that default."""
    src = tmp_path / "track2" / "train"
    for i in range(6):
        _scene(src, f"P_{i:04d}")
    out = tmp_path / "out"
    prepare_dfc23(out, _Args(dfc23_dir=str(src), dfc23_val_frac=0.0))
    st = PackedStore(out / "dfc23_g050" / "train")
    assert st.has_seg is False
    _, _, cls, _ = st.get(0)
    assert (np.asarray(cls) == 255).all()      # NO_LABEL everywhere


def _scene_with_border_blunder(root, stem, h=512, w=512):
    """A real DFC23 failure mode, reproduced: ordinary city under a ribbon of
    100 m+ nDSM glued to the top edge.

    Measured on `GF2_NewDelhi_28.5557_77.1194`: every pixel above 100 m in the
    scene — 2,615 of them, up to 183.2 m — lay in rows 0-31, over RGB whose
    luminance (119.5) and texture (std 33.6) match the rest of the tile
    (133.5 / 38.8). Stereo blunders, not buildings.
    """
    (root / "rgb").mkdir(parents=True, exist_ok=True)
    (root / "dsm").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(stem)) % 2 ** 31)
    _tif(root / "rgb" / f"{stem}.tif",
         (rng.random((h, w, 3)) * 200 + 30).astype(np.uint8))
    nd = np.zeros((h, w), np.float32)
    nd[200:300, 200:300] = 14.0          # a real building, interior
    nd[0:24, 0:288] = 150.0              # the artefact ribbon, on the border
    _tif(root / "dsm" / f"{stem}.tif", nd)


def test_border_concentrated_tall_mass_is_reported(tmp_path, capsys):
    src = tmp_path / "track2" / "train"
    for i in range(5):
        _scene_with_border_blunder(src, f"P_{i:04d}")
    prepare_dfc23(tmp_path / "out", _Args(dfc23_dir=str(src), dfc23_val_frac=0.0))
    out = capsys.readouterr().out
    assert ">100 m:" in out, out
    assert "concentrated at the tile border" in out, out


def test_max_height_invalidates_rather_than_clips(tmp_path):
    """A blunder is an *unknown*, not a building of the ceiling height — clipping
    to the ceiling would train 100 m as fact on an ordinary rooftop."""
    src = tmp_path / "track2" / "train"
    for i in range(5):
        _scene_with_border_blunder(src, f"P_{i:04d}")
    out = tmp_path / "out"
    prepare_dfc23(out, _Args(dfc23_dir=str(src), dfc23_val_frac=0.0,
                             dfc23_tile=512, dfc23_max_height_m=100.0))
    st = PackedStore(out / "dfc23_g050" / "train")
    _, h, _, val = st.get(0)
    h, val = np.asarray(h, np.float32), np.asarray(val, bool)
    assert not (h[val] > 100.0).any(), "a pixel above the ceiling survived as valid"
    assert h.shape == (512, 512), "these indices assume one tile per scene"
    assert not val[0:24, 0:288].any(), "the ribbon should be invalid, not clipped"
    assert val[200:300, 200:300].all(), "the real 14 m building must survive"
    assert float(h[val].max()) == pytest.approx(14.0, abs=0.1)


def test_max_height_off_by_default_leaves_the_target_untouched(tmp_path):
    src = tmp_path / "track2" / "train"
    for i in range(5):
        _scene_with_border_blunder(src, f"P_{i:04d}")
    out = tmp_path / "out"
    prepare_dfc23(out, _Args(dfc23_dir=str(src), dfc23_val_frac=0.0,
                             dfc23_tile=512))
    st = PackedStore(out / "dfc23_g050" / "train")
    _, h, _, val = st.get(0)
    assert float(np.asarray(h)[np.asarray(val, bool)].max()) == pytest.approx(150.0, abs=0.1)


def _scene_with_black_padding(root, stem, black_frac=0.6, h=512, w=512):
    """A real DFC23 scene shape: part of the tile is outside the optical
    footprint, so the RGB is pure black there — and the nDSM under it reads
    exactly 0.0, indistinguishable from flat ground.

    Measured over all 1773 train scenes: 12 carry all-black RGB, worst
    `SV_Berlin_52.4902_13.5090` at 53.09 % of the tile, 73.4 % of whose black
    pixels are labelled exactly 0 m.  `valid = isfinite & > -2 & < 500` calls
    every one of them a supervised 0 m ground sample.  This is the failure that
    put 41 % black-padding-at-zero into v2 — see `dwdata/india.py`'s docstring.
    """
    (root / "rgb").mkdir(parents=True, exist_ok=True)
    (root / "dsm").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(stem)) % 2 ** 31)
    rgb = (rng.random((h, w, 3)) * 200 + 30).astype(np.uint8)
    cut = int(w * black_frac)
    rgb[:, :cut] = 0                       # outside the optical footprint
    _tif(root / "rgb" / f"{stem}.tif", rgb)
    nd = np.zeros((h, w), np.float32)
    nd[200:300, 400:500] = 14.0            # a real building, in the imaged part
    _tif(root / "dsm" / f"{stem}.tif", nd)


def test_black_rgb_is_not_supervised_as_flat_ground(tmp_path):
    """No image means no supervision.  A pixel with no optical data cannot be
    a 0 m ground sample, and `valid` is the only thing that says so."""
    src = tmp_path / "track2" / "train"
    for i in range(5):
        _scene_with_black_padding(src, f"P_{i:04d}", black_frac=0.3)
    out = tmp_path / "out"
    prepare_dfc23(out, _Args(dfc23_dir=str(src), dfc23_val_frac=0.0,
                             dfc23_tile=512))
    st = PackedStore(out / "dfc23_g050" / "train")
    rgb, _, _, val = st.get(0)
    rgb, val = np.asarray(rgb), np.asarray(val, bool)
    black = (rgb == 0).all(-1)
    assert black.any(), "the fixture should have produced black padding"
    assert not val[black].any(), "black RGB was supervised as 0 m ground"
    assert val[:, 400:].all(), "the imaged part must stay supervised"


def test_a_mostly_black_scene_is_dropped_entirely(tmp_path):
    """`pack_labeled` already drops a tile under 50 % valid.  That gate is dead
    on DFC23 — no NaN, no nodata tag, min 0.0 over all 1773 scenes, so
    `valid.mean()` is exactly 1.000 everywhere.  Counting black RGB as invalid
    is what gives it something to fire on."""
    src = tmp_path / "track2" / "train"
    for i in range(5):
        _scene_with_black_padding(src, f"P_{i:04d}", black_frac=0.6)
    out = tmp_path / "out"
    prepare_dfc23(out, _Args(dfc23_dir=str(src), dfc23_val_frac=0.0,
                             dfc23_tile=512))
    assert len(PackedStore(out / "dfc23_g050" / "train")) == 0, \
        "a 60 %-black scene was packed as 60 % supervised ground"


def test_a_resized_height_raster_says_so(tmp_path, capsys):
    """`pack_labeled` resizes a height raster that does not match the RGB grid
    onto it, silently.  Over all 1773 DFC23 train scenes rgb/ and dsm/ share a
    geotransform exactly, so a mismatch here means the pairing is wrong, not
    that the product is coarser — and a silent bilinear resize turns that into
    plausible-looking targets.  Survivable, but it has to be visible."""
    src = tmp_path / "track2" / "train"
    _scene(src, "P_0000", h=512, w=512)
    (src / "dsm" / "P_0000.tif").unlink()
    _tif(src / "dsm" / "P_0000.tif", np.zeros((256, 256), np.float32), 1.0)
    prepare_dfc23(tmp_path / "out", _Args(dfc23_dir=str(src),
                                          dfc23_val_frac=0.0, dfc23_tile=256))
    out = capsys.readouterr().out
    assert "resiz" in out.lower(), out
    assert "256" in out and "512" in out, out


def test_the_height_check_samples_the_whole_store_not_just_the_front(tmp_path,
                                                                    capsys):
    """The border/interior verdict is a prevalence claim, so it has to be read
    off the whole store.  It used to read the first 64 tiles: on the real split
    that sample caught `GF2_NewDelhi_28.5557_77.1194` and almost none of the
    Rio/New York high-rise scenes, and printed `[!] concentrated at the tile
    border` — the opposite of the whole-split answer, where >100 m mass is 0.64x
    as dense in the border band as in the interior over all 1773 scenes.  A
    wrong verdict here is what would put `--dfc23_max_height_m 100` on by
    default and delete 214,540 px of genuine high-rise target."""
    src = tmp_path / "track2" / "train"
    for i in range(5):                       # blunder scenes, sorted first
        _scene_with_border_blunder(src, f"A_{i:04d}")
    for i in range(75):                      # ordinary scenes, sorted after
        _scene(src, f"B_{i:04d}", h=512, w=512)
    prepare_dfc23(tmp_path / "out", _Args(dfc23_dir=str(src),
                                          dfc23_val_frac=0.0, dfc23_tile=512))
    out = capsys.readouterr().out
    line = [ln for ln in out.splitlines() if "HEIGHT CHECK" in ln
            and "/train" in ln][0]
    # 80 scenes, one of which prepare_dfc23 always holds out for val
    assert "(over 79 tiles)" in line, line
