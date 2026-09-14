"""Coarse terrain sources: Copernicus GLO-30, SRTM, CartoDEM, or a local file.

The plan's decomposition is

    DSM_absolute = DTM_coarse  +  nDSM_predicted
                   ^ geometry     ^ the network

so this module's whole job is producing `DTM_coarse` on the prediction's own
grid.  Nothing here fits a scale to relative depth — the network already outputs
metres, which is the point of training on metric nDSM in the first place.

Sources, in the order you should prefer them:

| id | resolution | auth | notes |
|---|---|---|---|
| `copernicus30` | 30 m | none | public AWS bucket `copernicus-dem-30m`, COG per 1x1 degree tile. Default. |
| `srtm30` | 30 m | none | `s3://elevation-tiles-prod` Terrain Tiles (SRTM+ merged), also anonymous |
| `cartodem` | 30 m / 10 m | Bhoonidhi login | ISRO's own; **cannot** be fetched anonymously, so it is a local-file path here. Worth naming to an ISRO panel, and worth using when the team has the files. |
| `local` | whatever you have | — | any GeoTIFF/VRT |

Every one of them is a *surface* model (Copernicus GLO-30 and SRTM both include
buildings and canopy).  Adding a predicted nDSM to one directly double-counts
every structure — which is why `geo/calibrate.py` fits the terrain only on ground
pixels rather than using the DEM as-is.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

COP30_URL = ("https://copernicus-dem-30m.s3.amazonaws.com/"
             "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/"
             "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif")
# Terrain Tiles "geotiff" pyramid — SRTM/NED/etc merged, anonymous HTTP.
TERRAIN_TILES_URL = ("https://s3.amazonaws.com/elevation-tiles-prod/"
                     "geotiff/{z}/{x}/{y}.tif")

SOURCES = ("copernicus30", "srtm30", "cartodem", "local")


@dataclass
class DemResult:
    array: np.ndarray            # on the caller's grid, metres above the DEM's datum
    source: str
    files: list                  # what was actually read
    coverage: float              # fraction of the grid that got real data
    note: str = ""

    def summary(self) -> dict:
        return {"source": self.source, "files": [str(f) for f in self.files],
                "coverage": self.coverage, "note": self.note,
                "min_m": float(np.nanmin(self.array)) if self.array.size else None,
                "max_m": float(np.nanmax(self.array)) if self.array.size else None}


# ---------------------------------------------------------------------
def _bounds_lonlat(transform, crs, width: int, height: int):
    """(min_lon, min_lat, max_lon, max_lat) of a raster grid."""
    from rasterio.warp import transform_bounds

    left, top = transform * (0, 0)
    right, bottom = transform * (width, height)
    b = (min(left, right), min(top, bottom), max(left, right), max(top, bottom))
    return transform_bounds(crs, "EPSG:4326", *b, densify_pts=21)


def _cop30_tiles(lon0, lat0, lon1, lat1) -> list[str]:
    urls = []
    for lat in range(math.floor(lat0), math.floor(lat1) + 1):
        for lon in range(math.floor(lon0), math.floor(lon1) + 1):
            urls.append(COP30_URL.format(
                ns="N" if lat >= 0 else "S", lat=abs(lat),
                ew="E" if lon >= 0 else "W", lon=abs(lon)))
    return urls


def _terrain_tiles(lon0, lat0, lon1, lat1, z: int = 11) -> list[str]:
    def d2t(lon, lat):
        n = 2 ** z
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
        return max(0, min(n - 1, x)), max(0, min(n - 1, y))

    x0, y0 = d2t(lon0, lat1)
    x1, y1 = d2t(lon1, lat0)
    return [TERRAIN_TILES_URL.format(z=z, x=x, y=y)
            for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]


def fetch_dem(transform, crs, width: int, height: int, *, source: str = "copernicus30",
              local_path: str = "", cache_dir: str = "", timeout: float = 60.0) -> DemResult:
    """Coarse elevation reprojected onto the (transform, crs, width, height) grid.

    Reads remote COGs through GDAL's `/vsicurl/` so only the windows that overlap
    the scene are transferred — a 1x1 degree Copernicus tile is ~50 MB on disk but
    a 2 km scene pulls a few hundred KB of it.
    """
    import rasterio
    from rasterio.warp import Resampling, reproject

    dst = np.full((height, width), np.nan, np.float32)
    used, note = [], ""

    if source == "local" or (source == "cartodem" and local_path):
        if not local_path:
            return DemResult(dst, source, [], 0.0, "no --dem path given")
        candidates = [local_path]
        if source == "cartodem":
            note = ("CartoDEM supplied as a local file — Bhoonidhi has no anonymous "
                    "API, so the team downloads it once and passes --dem")
    elif source == "copernicus30":
        candidates = [f"/vsicurl/{u}" for u in _cop30_tiles(
            *_bounds_lonlat(transform, crs, width, height))]
    elif source == "srtm30":
        candidates = [f"/vsicurl/{u}" for u in _terrain_tiles(
            *_bounds_lonlat(transform, crs, width, height))]
    else:
        raise ValueError(f"unknown DEM source {source!r}; expected one of {SOURCES}")

    if cache_dir:
        os.environ.setdefault("GDAL_HTTP_TIMEOUT", str(int(timeout)))
        os.environ.setdefault("CPL_VSIL_CURL_CACHE_SIZE", "200000000")

    for path in candidates:
        try:
            with rasterio.open(path) as src:
                tmp = np.full((height, width), np.nan, np.float32)
                reproject(
                    source=rasterio.band(src, 1), destination=tmp,
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=transform, dst_crs=crs,
                    src_nodata=src.nodata, dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                )
            take = np.isnan(dst) & np.isfinite(tmp)
            if take.any():
                dst[take] = tmp[take]
                used.append(path)
        except Exception as e:  # noqa: BLE001
            print(f"[dem] {path}: {e}")

    cov = float(np.isfinite(dst).mean())
    if cov == 0:
        note = note or "no DEM coverage — check network access and the scene CRS"
    print(f"[dem] {source}: {len(used)} tile(s), {cov * 100:.1f}% coverage")
    return DemResult(dst, source, used, cov, note)


def fill_nan(a: np.ndarray) -> np.ndarray:
    """Nearest-finite fill, so a partly covered DEM is still usable."""
    a = np.asarray(a, np.float32)
    bad = ~np.isfinite(a)
    if not bad.any() or bad.all():
        return np.nan_to_num(a, nan=0.0)
    try:
        from scipy.ndimage import distance_transform_edt

        _, idx = distance_transform_edt(bad, return_distances=True, return_indices=True)
        return a[tuple(idx)]
    except Exception:  # noqa: BLE001
        return np.nan_to_num(a, nan=float(np.nanmedian(a)))


def load_dem_for_scene(meta, height_shape, *, source: str = "copernicus30",
                       local_path: str = "") -> DemResult | None:
    """Convenience wrapper around `fetch_dem` for a `preprocess.SceneMeta`."""
    if not meta.georeferenced and source != "local":
        print("[dem] scene is not georeferenced — a remote DEM cannot be located")
        return None
    h, w = height_shape
    return fetch_dem(meta.transform, meta.crs, w, h,
                     source=source, local_path=local_path)
