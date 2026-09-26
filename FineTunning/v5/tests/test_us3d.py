"""US3D (DFC2019 Track 1) ingest.

Each tile location has many views (`JAX_004_006` is location JAX_004, view 006),
so the val split must hold out whole locations.  A view-level split would put
other dates of every val tile into training.
"""
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from config import CLASS_NAMES, SEG_IGNORE_INDEX  # noqa: E402
from dwdata.dataset import seg_ids  # noqa: E402
from dwdata.packed import PackedStore  # noqa: E402
from prepare_data import US3D_TO_GAMUS, _us3d_pairs, prepare_us3d  # noqa: E402


def _tif(path, arr):
    arr = np.asarray(arr)
    data = arr[None] if arr.ndim == 2 else np.transpose(arr, (2, 0, 1))
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0],
                       width=arr.shape[1], count=data.shape[0], dtype=data.dtype) as ds:
        ds.write(data)


def _tree(root, locs=("JAX_004", "JAX_068", "OMA_212", "OMA_247", "JAX_214", "OMA_300"),
          views=3, t=64):
    for d in ("rgb", "agl", "cls"):
        (root / d).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for loc in locs:
        for v in range(views):
            k = f"{loc}_{v:03d}"
            _tif(root / "rgb" / f"{k}_RGB.tif", (rng.random((t, t, 3)) * 200 + 30).astype(np.uint8))
            agl = np.zeros((t, t), np.float32)
            agl[:16] = 9.0                                     # a tree the label keeps
            _tif(root / "agl" / f"{k}_AGL.tif", agl)
            cls = np.full((t, t), 2, np.uint8)
            cls[:16] = 5
            cls[-4:] = 65
            _tif(root / "cls" / f"{k}_CLS.tif", cls)
    _tif(root / "rgb" / "JAX_999_000_RGB.tif", np.zeros((t, t, 3), np.uint8))   # no AGL


class _Args:
    def __init__(self, d):
        self.us3d_dir, self.us3d_val_frac, self.us3d_gsd = str(d), 0.34, 0.3
        self.us3d_max_height_m, self.force = 150.0, False


def test_pairs_need_an_agl_and_key_on_location(tmp_path):
    _tree(tmp_path)
    recs = _us3d_pairs(tmp_path)
    assert len(recs) == 18 and {r["loc"] for r in recs} >= {"JAX_004", "OMA_247"}
    assert all(r["cls"] is not None for r in recs)


def test_split_holds_out_whole_locations_and_keeps_trees(tmp_path):
    _tree(tmp_path / "src")
    prepare_us3d(tmp_path / "out", _Args(tmp_path / "src"))
    tr = PackedStore(tmp_path / "out" / "us3d" / "train")
    va = PackedStore(tmp_path / "out" / "us3d" / "val")
    loc = lambda s: "_".join(s.split("_")[:2])  # noqa: E731
    lt = {loc(tr.stems[i]) for i in range(len(tr))}
    lv = {loc(va.stems[i]) for i in range(len(va))}
    assert lv and not (lt & lv)                       # no location on both sides
    assert len(tr) + len(va) == 18 and len(va) % 3 == 0   # every view of a val location
    assert abs(tr.gsd_m - 0.3) < 1e-6                 # no CRS -> the fallback
    _, h, cls, v = tr.get(0)
    assert float(h[:16].mean()) == pytest.approx(9.0) and v.all()
    seg = seg_ids(cls, "us3d")                        # what the seg loss sees
    assert (seg[:16] == CLASS_NAMES.index("tree")).all()
    assert (seg[16:-4] == CLASS_NAMES.index("ground")).all()
    assert (seg[-4:] == SEG_IGNORE_INDEX).all()       # 65 = unlabelled
    assert US3D_TO_GAMUS[6] == CLASS_NAMES.index("building")
