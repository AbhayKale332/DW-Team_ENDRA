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
    prepare_dfc23(tmp_path / "out", _Args(dfc23_dir=str(src), dfc23_val_frac=0.0))
    out = capsys.readouterr().out
    assert "HEIGHT CHECK" in out
    assert "ABSOLUTE DSM" in out, out
