"""A real GeoTIFF through the whole georeferenced path.

This is the deliverable the problem statement names: "Accepts single-view optical
satellite imagery (PNG, JPG, or TIFF) and outputs a high-fidelity DSM in a
standard geospatial format."  So the test writes a GeoTIFF, runs the CLI's own
functions over it, and checks the products open again with the *same* grid — a
DSM on a shifted transform is not a DSM.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from config import Config  # noqa: E402
from dwdata.preprocess import PreprocSpec, read_scene  # noqa: E402
from tests.stub_encoder import use_stub  # noqa: E402

GSD = 0.6
CRS = "EPSG:32643"        # UTM 43N — covers western India, metres


def _write_geotiff(path, n=192):
    rng = np.random.default_rng(0)
    rgb = (rng.random((3, n, n)) * 200 + 30).astype(np.uint8)
    tr = from_origin(300000.0, 2100000.0, GSD, GSD)
    with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=3,
                       dtype="uint8", crs=CRS, transform=tr) as d:
        d.write(rgb)
    return tr


def _write_dem(path, n=16, base=120.0):
    """A coarse 30 m DEM covering the scene — and, like GLO-30, a *surface* model."""
    tr = from_origin(299900.0, 2100100.0, 30.0, 30.0)
    y, x = np.mgrid[0:n, 0:n]
    dem = (base + 0.4 * x + 0.2 * y).astype(np.float32)
    with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=1,
                       dtype="float32", crs=CRS, transform=tr) as d:
        d.write(dem, 1)
    return dem


def _net():
    from models.heads import DepthWizardNet

    c = Config()
    c.tile_size, c.decoder_dim, c.n_bins = 64, 32, 8
    c.grad_checkpoint_encoder = False
    return DepthWizardNet(c).eval(), PreprocSpec(tile_size=64, canonical_gsd_m=0.5)


def test_gsd_comes_from_the_transform(tmp_path):
    p = tmp_path / "scene.tif"
    _write_geotiff(p)
    rgb, meta = read_scene(p)
    assert meta.georeferenced
    assert meta.gsd_source == "geotiff"
    assert abs(meta.gsd_m - GSD) < 1e-6, "metric scale must come from the CRS"
    assert rgb.shape == (192, 192, 3)


def test_outputs_land_on_the_source_grid(tmp_path):
    undo = use_stub(hidden=32, patch=16, layers=8)
    try:
        from infer.engine import predict_scene
        from infer.predict import write_outputs

        src = tmp_path / "scene.tif"
        tr = _write_geotiff(src)
        net, spec = _net()
        rgb, meta = read_scene(src)
        h, seg = predict_scene(net, rgb, meta.gsd_m, spec, torch.device("cpu"),
                               want_seg=True, batch_tiles=2)
        assert h.shape == rgb.shape[:2]

        out = tmp_path / "out"
        payload = write_outputs(out, "scene", rgb, h, meta, spec, seg=seg)
        assert "ndsm_m.tif" in payload["files"]
        assert "terrain.glb" in payload["files"]

        with rasterio.open(out / "ndsm_m.tif") as d:
            assert d.crs.to_string() == CRS
            assert d.transform.almost_equals(tr), "the DSM moved off the source grid"
            assert d.count == 1 and d.dtypes[0] == "float32"
            back = d.read(1)
        assert np.allclose(back, h, atol=1e-5)
        assert d.width == 192 and d.height == 192
    finally:
        undo()


def test_absolute_dsm_does_not_double_count(tmp_path):
    """The v3 bug, on real rasters: DEM + nDSM counts structures twice."""
    undo = use_stub(hidden=32, patch=16, layers=8)
    try:
        from geo.dem import fetch_dem
        from infer.predict import to_absolute

        src = tmp_path / "scene.tif"
        _write_geotiff(src)
        dem_path = tmp_path / "dem.tif"
        _write_dem(dem_path)
        _rgb, meta = read_scene(src)

        res = fetch_dem(meta.transform, meta.crs, meta.width, meta.height,
                        source="local", local_path=str(dem_path))
        assert res.coverage > 0.99, "the coarse DEM did not reproject onto the grid"

        # a synthetic prediction with a known 25 m tower on flat ground
        ndsm = np.zeros((meta.height, meta.width), np.float32)
        ndsm[60:120, 60:120] = 25.0
        seg = np.where(ndsm > 1, 2, 0).astype(np.int32)
        dsm, info = to_absolute(ndsm, seg, meta, dem_source="local",
                                dem_path=str(dem_path))
        assert dsm is not None
        terrain_at_tower = res.array[90, 90]
        assert abs(dsm[90, 90] - (terrain_at_tower + 25.0)) < 2.0, \
            f"expected ~{terrain_at_tower + 25:.1f} m, got {dsm[90, 90]:.1f} m"
        assert abs(dsm[5, 5] - res.array[5, 5]) < 2.0, "flat ground drifted"
        assert "calibration" in info and "dtm" in info
    finally:
        undo()


def test_predict_cli_report_runs(tmp_path):
    """`--report` must print the contract without a GPU and without predicting."""
    undo = use_stub(hidden=32, patch=16, layers=8)
    try:
        from config import safe_config_dict
        from models.heads import DepthWizardNet

        c = Config()
        c.tile_size, c.decoder_dim, c.n_bins = 64, 32, 8
        c.grad_checkpoint_encoder = False
        net = DepthWizardNet(c).eval()
        spec = PreprocSpec(tile_size=64, canonical_gsd_m=0.5)
        ck = tmp_path / "best.pt"
        torch.save({"model": net.state_dict(), "preproc": spec.to_dict(),
                    "config": safe_config_dict(c), "epoch": 1,
                    "encoder_included": True}, ck)

        src = tmp_path / "scene.tif"
        _write_geotiff(src)
        root = Path(__file__).resolve().parents[1]
        code = (
            "import sys;sys.path.insert(0, %r);"
            "from tests.stub_encoder import use_stub;use_stub(32,16,8);"
            "import dwdata.preprocess as pp;"
            "pp.resolve_encoder_stats=lambda *a,**k:(pp.DINOV3_SAT_MEAN, pp.DINOV3_SAT_STD);"
            "sys.argv=['predict', %r, '--ckpt', %r, '--report'];"
            "import infer.predict as ip;ip.main()"
        ) % (str(root), str(src), str(ck))
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert "gsd=0.6000 m (geotiff)" in r.stdout
        assert "contract:" in r.stdout and "0.5 m/px" in r.stdout
    finally:
        undo()
