"""v5 DEM stack: per-source datums, conversion, compound CRS, offline cache."""

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from geo import dem as D  # noqa: E402


def test_every_source_declares_a_datum():
    assert D.SOURCE_INFO["copernicus30"]["datum"] == "EGM2008"
    assert D.SOURCE_INFO["srtmgl1"]["datum"] == "EGM96"
    assert D.SOURCE_INFO["cartodem"]["datum"] == "WGS84"          # ellipsoidal per NRSC
    assert D.canonical_source("srtm30") == "terrain_tiles"        # v4 name, honest now
    assert "not native SRTM" in D.SOURCE_INFO["terrain_tiles"]["note"]
    with pytest.raises(ValueError):
        D.canonical_source("nope")


def test_opentopo_without_key_says_so(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENTOPO_KEY", raising=False)
    r = D.fetch_dem(from_origin(372093.6, 2253942.6, 30, 30), "EPSG:32645", 4, 4,
                    source="srtmgl1", cache_dir=str(tmp_path))
    assert r.coverage == 0 and "OPENTOPO_KEY" in r.note and r.datum == "EGM96"


def test_cache_hit_needs_no_network(tmp_path, monkeypatch):
    """A primed cache file is used as-is — the offline path."""
    tr = from_origin(372093.6, 2253942.6, 30, 30)
    bbox = D._pad_bbox(D._bounds_lonlat(tr, rasterio.crs.CRS.from_epsg(32645), 8, 8))
    f = D._cache_path(tmp_path, "copernicus30", bbox)
    w, s, e, n = bbox
    with rasterio.open(f, "w", driver="GTiff", height=10, width=10, count=1,
                       dtype="float32", crs="EPSG:4326",
                       transform=from_origin(w, n, (e - w) / 10, (n - s) / 10)) as ds:
        ds.write(np.full((10, 10), 42.0, np.float32), 1)
    monkeypatch.setattr(D, "_merge_to", lambda *a, **k: pytest.fail("went online"))
    r = D.fetch_dem(tr, "EPSG:32645", 8, 8, source="copernicus30", cache_dir=str(tmp_path))
    assert r.coverage == 1.0 and np.allclose(r.array, 42.0) and r.datum == "EGM2008"


def test_local_dem_carries_the_declared_datum(tmp_path):
    tr = from_origin(0, 100, 1, 1)
    p = tmp_path / "d.tif"
    with rasterio.open(p, "w", driver="GTiff", height=5, width=5, count=1,
                       dtype="float32", crs="EPSG:32645", transform=tr) as ds:
        ds.write(np.ones((5, 5), np.float32), 1)
    r = D.fetch_dem(tr, "EPSG:32645", 5, 5, source="local", local_path=str(p),
                    local_datum="EGM96")
    assert r.datum == "EGM96" and r.coverage == 1.0


def test_same_datum_is_a_no_op():
    a = np.arange(4, dtype=np.float32).reshape(2, 2)
    out, info = D.to_datum(a, from_origin(0, 0, 1, 1), "EPSG:32645", "EGM96", "EGM96")
    assert out is a and info["applied"]


def test_geoid_at_bhubaneswar_matches_geographiclib():
    """-62.02 m (EGM2008) / -62.41 m (EGM96) at 20.325 N 85.834 E (GeoidEval).
    Needs a PROJ geoid grid (cached, or fetched from cdn.proj.org)."""
    n08 = D.geoid_undulation(np.array([85.834]), np.array([20.325]), "EGM2008")
    n96 = D.geoid_undulation(np.array([85.834]), np.array([20.325]), "EGM96")
    if n08 is None or n96 is None:
        pytest.skip("no PROJ geoid grid reachable")
    assert n08[0] == pytest.approx(-62.02, abs=0.05)
    assert n96[0] == pytest.approx(-62.41, abs=0.05)


def test_compound_crs_has_a_vertical_part():
    wkt = D.vertical_crs_wkt("EPSG:32645", "EGM2008")
    assert wkt and "EGM2008" in wkt and "UTM zone 45N" in wkt
    assert D.vertical_crs_wkt("EPSG:32645", "WGS84") is None
