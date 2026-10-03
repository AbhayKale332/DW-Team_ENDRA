"""Coarse scoring rejects unknown datums and missing reference coverage."""
from types import SimpleNamespace
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from eval.judge_proxy import score_scene
from geo import dem


@pytest.fixture
def scene(tmp_path):
    tr = from_origin(500000, 2200000, 1, 1)
    profile = dict(driver="GTiff", height=8, width=8, count=1, dtype="float32",
                   crs="EPSG:32644", transform=tr, nodata=np.nan)
    for name, value in (("dsm_m.tif", 12), ("ndsm_m.tif", .5)):
        with rasterio.open(tmp_path / name, "w", **profile) as f:
            f.write(np.full((8, 8), value, np.float32), 1)
            f.update_tags(1, VERTICAL_DATUM="EGM96")
    return tmp_path, SimpleNamespace(transform=tr, crs="EPSG:32644")


def test_known_offset(scene, monkeypatch):
    monkeypatch.setattr(dem, "fetch_dem", lambda *a, **k:
        dem.DemResult(np.full((2, 2), 10, np.float32), "test", [], 1, datum="EGM96"))
    out = score_scene(*scene, 4, ["test"], band_rows=3)["test"]
    assert out["per_pixel"]["rmse_m"] == pytest.approx(2)
    assert out["per_30m_cell"]["rmse_m"] == pytest.approx(2)


def test_unknown_datum_not_scored(scene, monkeypatch):
    monkeypatch.setattr(dem, "fetch_dem", lambda *a, **k:
        dem.DemResult(np.full((2, 2), 10, np.float32), "test", [], 1, datum=""))
    out = score_scene(*scene, 4, ["test"])["test"]
    assert "error" in out and "per_pixel" not in out


def test_missing_reference_not_filled_into_scores(scene, monkeypatch):
    arr = np.full((2, 2), 10, np.float32)
    arr[:, 1] = np.nan
    monkeypatch.setattr(dem, "fetch_dem", lambda *a, **k:
        dem.DemResult(arr, "test", [], .5, datum="EGM96"))
    out = score_scene(*scene, 4, ["test"])["test"]
    assert 0 < out["per_pixel"]["n"] < 64
    assert out["per_30m_cell"]["n"] == 2
    assert out["per_pixel"]["rmse_m"] == pytest.approx(2)
