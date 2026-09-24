"""v5 viewer back-end: the uncertainty layer (C5), reference validation (C2) and
full-resolution AOIs out of a windowed scene (C6)."""

import io
import json

import numpy as np
import pytest
import torch

rasterio = pytest.importorskip("rasterio")
from affine import Affine  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

from dwdata.preprocess import PreprocSpec, SceneMeta  # noqa: E402
from tests.test_scene_io import TR, H, W, _write_product  # noqa: E402


class _PerPixelStd(torch.nn.Module):
    """Tiling-invariant heights plus a Head-B-like spread that grows with height."""

    def forward(self, x):
        h = x[:, :1] * 3.0 + 10.0
        return {"fused": h, "seg": torch.zeros(x.shape[0], 8, *x.shape[-2:]),
                "b_std": 0.2 + 0.05 * h}


def _spec():
    return PreprocSpec(tile_size=32, canonical_gsd_m=0.6)


# ---------------------------------------------------------------------
# C5 — uncertainty
# ---------------------------------------------------------------------
def test_std_flows_through_predict_scene_and_write_outputs(tmp_path):
    from dwdata.preprocess import read_scene
    from infer.engine import predict_scene
    from infer.predict import write_outputs

    _write_product(tmp_path / "p")
    rgb, meta = read_scene(tmp_path / "p")
    h, seg, std = predict_scene(_PerPixelStd(), rgb, meta.gsd_m, _spec(), torch.device("cpu"),
                                want_seg=True, valid=meta.valid, return_std=True)
    assert std is not None and std.shape == h.shape
    fin = np.isfinite(h)
    assert (np.isfinite(std) == fin).all()                     # NoData stays NoData
    assert np.allclose(std[fin], 0.2 + 0.05 * h[fin], atol=1e-3)
    out = write_outputs(tmp_path / "o", "s", rgb, h, meta, _spec(), seg=seg, mesh=False,
                        std=std)
    assert (tmp_path / "o" / "ndsm_std_m.npy").is_file()
    assert (tmp_path / "o" / "ndsm_std_m.tif").is_file()
    u = out["uncertainty"]
    assert u["confident_threshold_m"] >= 1.0 and 0 < u["confident_frac"] <= 1


def test_models_without_b_std_still_return_three_values():
    from infer.engine import predict_scene

    class NoStd(torch.nn.Module):
        def forward(self, x):
            return {"fused": torch.ones(x.shape[0], 1, *x.shape[-2:])}

    rgb = np.full((40, 40, 3), 120, np.uint8)
    h, seg, std = predict_scene(NoStd(), rgb, 0.6, _spec(), torch.device("cpu"),
                                return_std=True)
    assert std is None and seg is None and h.shape == (40, 40)


# ---------------------------------------------------------------------
# C2 — reference validation
# ---------------------------------------------------------------------
def _job(tmp_path, n=160, gsd=0.6):
    """A georeferenced job dir: a sloping DSM (terrain + one 12 m block)."""
    from infer.predict import write_outputs

    tr = from_origin(500000.0, 2000000.0, gsd, gsd)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    ndsm = np.zeros((n, n), np.float32)
    ndsm[60:100, 60:100] = 12.0
    dtm = 300.0 + 0.05 * xx + 0.02 * yy
    std = np.where(ndsm > 1, 3.0, 0.5).astype(np.float32)
    meta = SceneMeta(gsd_m=gsd, gsd_source="geotiff", georeferenced=True, transform=tr,
                     crs=rasterio.crs.CRS.from_epsg(32645), width=n, height=n, path="x.tif")
    rgb = np.full((n, n, 3), 128, np.uint8)
    write_outputs(tmp_path / "job", "x", rgb, ndsm, meta, _spec(), dsm_abs=ndsm + dtm,
                  dtm=dtm, mesh=False, datum="EGM2008", std=std, geotiff=False)
    return tmp_path / "job", tr, ndsm, dtm


def _write_ref(path, arr, tr, epsg, nodata=None):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype="float32", crs=f"EPSG:{epsg}", transform=tr,
                       nodata=nodata) as d:
        d.write(arr.astype(np.float32), 1)


def test_reference_dsm_is_reprojected_classified_and_scored(tmp_path):
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from serve.validate import validate_reference

    job, tr, ndsm, dtm = _job(tmp_path)
    truth = ndsm + dtm + 2.0                    # the reference sits 2 m higher
    # hand it over in lon/lat at a different resolution, as a real DEM would be
    src_crs, n = "EPSG:32645", truth.shape[0]
    dtr, w, h = calculate_default_transform(src_crs, "EPSG:4326", n, n,
                                            *rasterio.transform.array_bounds(n, n, tr),
                                            resolution=(0.6 / 111000 * 0.8,) * 2)
    ll = np.full((h, w), -9999, np.float32)
    reproject(truth, ll, src_transform=tr, src_crs=src_crs, dst_transform=dtr,
              dst_crs="EPSG:4326", resampling=Resampling.bilinear, dst_nodata=-9999)
    _write_ref(tmp_path / "ref.tif", ll, dtr, 4326, nodata=-9999)

    r = validate_reference(job, str(tmp_path / "ref.tif"), ref_datum="EGM2008")
    assert r["kind"] == "dsm" and r["placement"] == "reprojected"
    assert r["datum_conversion"]["note"] == "same datum"
    # smooth terrain: two bilinear resamplings cost centimetres away from the block
    assert r["median_err_on_ground_m"] == pytest.approx(-2.0, abs=0.05)
    assert r["per_pixel"]["bias_m"] == pytest.approx(-2.0, abs=0.25)
    assert "per_pixel_confident" in r and "per_cell" in r
    # the block's walls are where resampling hurts; they are not "confident"
    def spread(m):
        return (m["rmse_m"] ** 2 - m["bias_m"] ** 2) ** 0.5
    assert spread(r["per_pixel_confident"]) < spread(r["per_pixel"])
    assert (job / "gt_dsm_m.npy").is_file() and (job / "validation.json").is_file()


def test_reference_ndsm_and_the_kind_guard(tmp_path):
    from serve.validate import validate_reference

    job, tr, ndsm, _ = _job(tmp_path)
    _write_ref(tmp_path / "ref.tif", ndsm, tr, 32645)
    r = validate_reference(job, str(tmp_path / "ref.tif"))
    assert r["kind"] == "ndsm" and r["per_pixel"]["rmse_m"] < 1e-3
    (job / "dsm_m.npy").unlink()
    with pytest.raises(ValueError, match="absolute DSM"):
        validate_reference(job, str(tmp_path / "ref.tif"), kind="dsm")


# ---------------------------------------------------------------------
# C6 — full-resolution AOIs
# ---------------------------------------------------------------------
def test_aoi_out_of_a_windowed_scene_is_full_resolution_and_registered(tmp_path):
    from dwdata.preprocess import _meta_from_source, open_scene
    from infer.predict import run_windowed
    from serve.validate import extract_aoi

    _write_product(tmp_path / "p")
    src = open_scene(tmp_path / "p")
    meta = _meta_from_source(src, tmp_path / "p")
    out = tmp_path / "job"
    payload = run_windowed(_PerPixelStd(), _spec(), src, meta, torch.device("cpu"), out,
                           absolute=False, dem_source="", band_rows=24, mesh=False,
                           overview_max=40)
    step = payload["windowed"]["overview_step"]
    assert step == 3 and payload["uncertainty"]["valid_px"] > 0

    r = extract_aoi(out, row=4, col=5, h=20, w=20)
    assert r["decimation"] == 1 and r["window_full_res"] == [12, 15, 60, 60]
    a = out / r["dir"]
    got = np.load(a / "ndsm_m.npy")
    with rasterio.open(out / "ndsm_m.tif") as d:
        want = d.read(1)[12:72, 15:75]
    assert got.shape == (60, 60)
    assert np.allclose(got, want, equal_nan=True)
    m = json.loads((a / "meta.json").read_text())
    t = Affine(*m["scene"]["transform"])
    assert t.almost_equals(TR * Affine.translation(15, 12), precision=1e-6)
    assert m["scene"]["gsd_m"] == pytest.approx(0.6)
    assert (a / "ndsm_std_m.npy").is_file() and (a / "rgb.png").is_file()


def test_aoi_is_decimated_to_the_cap(tmp_path):
    from serve.validate import extract_aoi

    job, *_ = _job(tmp_path)
    r = extract_aoi(job, 20, 10, 120, 150, max_px=64)
    # col 10 snaps down to 9, the decimated source grid (0, 3, 6, ...)
    assert r["decimation"] == 3 and r["shape"] == [40, 51]
    m = json.loads((job / r["dir"] / "meta.json").read_text())
    assert m["scene"]["gsd_m"] == pytest.approx(1.8)
    t = Affine(*m["scene"]["transform"])
    assert t.almost_equals(from_origin(500000.0, 2000000.0, 0.6, 0.6)
                           * Affine.translation(9, 20) * Affine.scale(3), precision=1e-6)
    got = np.load(job / r["dir"] / "ndsm_m.npy")
    full = np.load(job / "ndsm_m.npy")
    assert np.array_equal(got, full[20:140:3, 9:160:3])
    with pytest.raises(ValueError, match="too small"):
        extract_aoi(job, 0, 0, 4, 4)


# ---------------------------------------------------------------------
# the service endpoints
# ---------------------------------------------------------------------
fastapi = pytest.importorskip("fastapi")


def test_reference_and_aoi_endpoints(tmp_path):
    import serve.app as sa
    from fastapi.testclient import TestClient

    sa._state.update(runtime="torch", model=_PerPixelStd().eval(), spec=_spec(),
                     device=torch.device("cpu"), ckpt="stub")
    sa._jobs.clear()
    old = sa.JOBS
    sa.JOBS = tmp_path / "jobs"
    try:
        c = TestClient(sa.create_app())
        d = tmp_path / "prod"
        _write_product(d)
        files = [("file", (f.name, f.read_bytes(), "application/octet-stream"))
                 for f in sorted(d.iterdir())]
        job = c.post("/api/predict", files=files).json()["job"]
        st = c.get(f"/api/job/{job}").json()
        assert st["stage"] == "done", st.get("error")
        assert c.get(f"/api/result/{job}/ndsm_std_m.npy").status_code == 200

        # a reference nDSM on the same grid: the prediction itself + 1 m
        pred = np.load(io.BytesIO(c.get(f"/api/result/{job}/ndsm_m.npy").content))
        _write_ref(tmp_path / "ref.tif", np.nan_to_num(pred, nan=-9999) + 1.0, TR, 32644,
                   nodata=-9999)
        with open(tmp_path / "ref.tif", "rb") as fh:
            r = c.post(f"/api/reference/{job}", files={"file": ("ref.tif", fh.read())},
                       data={"kind": "ndsm"})
        assert r.status_code == 200, r.text
        v = r.json()
        assert v["per_pixel"]["bias_m"] == pytest.approx(-1.0, abs=1e-3)
        assert c.get(f"/api/result/{job}/gt_ndsm_m.npy").status_code == 200
        assert c.get(f"/api/result/{job}/validation.json").json()["kind"] == "ndsm"
        bad = c.post(f"/api/reference/{job}", files={"file": ("r.tif", b"x")},
                     data={"kind": "dsm"})
        assert bad.status_code == 422
        assert c.post("/api/reference/nojob", files={"file": ("r.tif", b"x")}
                      ).status_code == 404

        a = c.post(f"/api/aoi/{job}", data={"row": 10, "col": 12, "h": 40, "w": 30})
        assert a.status_code == 200, a.text
        base = a.json()["base"]
        arr = np.load(io.BytesIO(c.get(f"{base}/ndsm_m.npy").content))
        assert arr.shape == (40, 30)
        assert c.get(f"{base}/meta.json").json()["aoi"]["overview_window"] == [10, 12, 40, 30]
    finally:
        sa.JOBS = old
