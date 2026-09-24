"""Coarse elevation sources, with their vertical datum made explicit.

The judges score the absolute DSM against **SRTM or Copernicus 30 m**
(`CompetitionContext/FAQs.md`), so in v5 this module is on the scoring path, not
a convenience.  Three things matter and v4 got two of them wrong:

1. **Which DEM.**  v4's `srtm30` was AWS *Terrain Tiles* — a Web-Mercator mosaic
   of SRTM plus other sources, resampled into 512 px tiles — not native SRTM GL1.
   It survives here under its honest name (`terrain_tiles`; `srtm30` is kept as
   an alias so old commands still run).  Native SRTM GL1 and NASADEM come from
   the OpenTopography API (free key in `OPENTOPO_KEY`); Copernicus GLO-30 stays
   on the anonymous AWS bucket.

2. **Which vertical datum.**  SRTM GL1 / NASADEM heights are above EGM96,
   Copernicus GLO-30 above EGM2008, CartoDEM above the **WGS84 ellipsoid**
   (NRSC spec).  At Bhubaneswar the geoid sits at -62 m, at Kurnool -86 m, so an
   ellipsoidal CartoDEM used as if orthometric puts the whole DSM 60-90 m low; and
   EGM96 vs EGM2008 alone differs by -0.4 m (Bhubaneswar) to +4.7 m (Gangtok).
   Every `DemResult` now carries `datum`, and `load_dem_for_scene` converts to a
   requested target datum through PROJ geoid grids (`to_datum`).

3. **Offline.**  A judge's laptop may have no internet.  v4 then produced no
   absolute DSM at all.  Every remote fetch is now written to a bbox-keyed cache
   (`DW_DEM_CACHE`, default `~/.cache/depthwizard/dem`) and read back from there
   first, so priming the cache once per demo AOI (`python -m geo.dem --prime`)
   makes the whole path work offline.  A user-supplied file (`local`) always
   works.

| id | datum | auth | notes |
|---|---|---|---|
| `copernicus30` (`cop30`) | EGM2008 | none | AWS `copernicus-dem-30m` COGs. Default. |
| `srtmgl1` | EGM96 | `OPENTOPO_KEY` | native SRTM GL1 via OpenTopography |
| `nasadem` | EGM96 | `OPENTOPO_KEY` | NASADEM via OpenTopography |
| `terrain_tiles` (`srtm30`) | EGM96 (mixed) | none | AWS Terrain Tiles mosaic — **not** native SRTM |
| `cartodem` | WGS84 ellipsoid | Bhoonidhi login | local file only |
| `local` | as declared (`local_datum`) | — | any GeoTIFF/VRT |

All of them are *surface* models (they already hold part of the buildings and
canopy), which is what `geo/calibrate.py`'s `dem_anchored` mode is built around.
"""

from __future__ import annotations

import math
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

COP30_URL = ("https://copernicus-dem-30m.s3.amazonaws.com/"
             "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/"
             "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif")
# Terrain Tiles "geotiff" pyramid — SRTM/NED/ETOPO merged, anonymous HTTP.
TERRAIN_TILES_URL = ("https://s3.amazonaws.com/elevation-tiles-prod/"
                     "geotiff/{z}/{x}/{y}.tif")
OPENTOPO_URL = ("https://portal.opentopography.org/API/globaldem?demtype={demtype}"
                "&south={s}&north={n}&west={w}&east={e}&outputFormat=GTiff"
                "&API_Key={key}")

# Vertical datums by name.  `None` EPSG = ellipsoidal (no geoid involved).
DATUM_EPSG: dict[str, int | None] = {"EGM96": 5773, "EGM2008": 3855, "WGS84": None}

SOURCE_INFO: dict[str, dict] = {
    "copernicus30": {"datum": "EGM2008", "kind": "cop30"},
    "srtmgl1": {"datum": "EGM96", "kind": "opentopo", "demtype": "SRTMGL1"},
    "nasadem": {"datum": "EGM96", "kind": "opentopo", "demtype": "NASADEM"},
    "terrain_tiles": {"datum": "EGM96", "kind": "terrain_tiles",
                      "note": "AWS Terrain Tiles mosaic (SRTM + other sources, "
                              "Web-Mercator resampled) — not native SRTM GL1"},
    "cartodem": {"datum": "WGS84", "kind": "local"},
    "local": {"datum": "", "kind": "local"},
}
ALIASES = {"cop30": "copernicus30", "srtm30": "terrain_tiles", "srtm": "srtmgl1"}
SOURCES = tuple(SOURCE_INFO) + tuple(ALIASES)


def canonical_source(source: str) -> str:
    s = ALIASES.get(source, source)
    if s not in SOURCE_INFO:
        raise ValueError(f"unknown DEM source {source!r}; expected one of {SOURCES}")
    return s


def default_cache_dir() -> Path:
    return Path(os.environ.get("DW_DEM_CACHE",
                               Path.home() / ".cache" / "depthwizard" / "dem"))


@dataclass
class DemResult:
    array: np.ndarray            # on the caller's grid, metres above `datum`
    source: str
    files: list                  # what was actually read
    coverage: float              # fraction of the grid that got real data
    note: str = ""
    datum: str = ""              # "EGM96" | "EGM2008" | "WGS84" (ellipsoid) | ""

    def summary(self) -> dict:
        fin = np.isfinite(self.array)
        return {"source": self.source, "files": [str(f) for f in self.files],
                "coverage": self.coverage, "note": self.note, "datum": self.datum,
                "min_m": float(np.nanmin(self.array)) if fin.any() else None,
                "max_m": float(np.nanmax(self.array)) if fin.any() else None}


# ---------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------
def _bounds_lonlat(transform, crs, width: int, height: int):
    """(min_lon, min_lat, max_lon, max_lat) of a raster grid."""
    from rasterio.warp import transform_bounds

    left, top = transform * (0, 0)
    right, bottom = transform * (width, height)
    b = (min(left, right), min(top, bottom), max(left, right), max(top, bottom))
    return transform_bounds(crs, "EPSG:4326", *b, densify_pts=21)


def _pad_bbox(b, pad_deg: float = 0.01):
    """Round outward to 0.01 deg so nearby scenes share one cache file."""
    w, s, e, n = b
    f = 100.0
    return (math.floor((w - pad_deg) * f) / f, math.floor((s - pad_deg) * f) / f,
            math.ceil((e + pad_deg) * f) / f, math.ceil((n + pad_deg) * f) / f)


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


# ---------------------------------------------------------------------
# fetching into the cache
# ---------------------------------------------------------------------
def _cache_path(cache_dir: Path, source: str, bbox) -> Path:
    w, s, e, n = bbox
    return cache_dir / f"{source}_{w:.2f}_{s:.2f}_{e:.2f}_{n:.2f}.tif"


def _merge_to(paths: list[str], bounds, dst_crs: str, out: Path) -> bool:
    """Windowed read of remote COGs over `bounds` (in `dst_crs`), merged to `out`."""
    import rasterio
    from rasterio.merge import merge

    srcs = []
    for p in paths:
        try:
            srcs.append(rasterio.open(p))
        except Exception as e:  # noqa: BLE001 — a missing ocean tile is normal
            print(f"[dem] {p}: {e}")
    if not srcs:
        return False
    try:
        arr, tr = merge(srcs, bounds=bounds, nodata=np.nan, dtype="float32")
        prof = dict(driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=1,
                    dtype="float32", crs=srcs[0].crs, transform=tr, nodata=np.nan,
                    compress="deflate", predictor=3, tiled=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".part.tif")
        with rasterio.open(tmp, "w", **prof) as d:
            d.write(arr[0], 1)
        tmp.replace(out)
        return True
    finally:
        for s in srcs:
            s.close()


def _fetch_to_cache(source: str, bbox, cache_dir: Path, timeout: float) -> tuple[Path | None, str]:
    """Make sure a GeoTIFF covering `bbox` for `source` exists in the cache."""
    out = _cache_path(cache_dir, source, bbox)
    if out.is_file():
        return out, "cache hit"
    info = SOURCE_INFO[source]
    os.environ.setdefault("GDAL_HTTP_TIMEOUT", str(int(timeout)))
    w, s, e, n = bbox
    try:
        if info["kind"] == "cop30":
            ok = _merge_to([f"/vsicurl/{u}" for u in _cop30_tiles(w, s, e, n)],
                           (w, s, e, n), "EPSG:4326", out)
        elif info["kind"] == "terrain_tiles":
            from rasterio.warp import transform_bounds

            b3857 = transform_bounds("EPSG:4326", "EPSG:3857", w, s, e, n)
            ok = _merge_to([f"/vsicurl/{u}" for u in _terrain_tiles(w, s, e, n)],
                           b3857, "EPSG:3857", out)
        elif info["kind"] == "opentopo":
            key = os.environ.get("OPENTOPO_KEY", "")
            if not key:
                return None, (f"{source} needs a free OpenTopography API key in "
                              "OPENTOPO_KEY (https://portal.opentopography.org)")
            url = OPENTOPO_URL.format(demtype=info["demtype"], s=s, n=n, w=w, e=e, key=key)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".part.tif")
            with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
                f.write(r.read())
            tmp.replace(out)
            ok = True
        else:
            return None, f"{source} is local-only"
    except Exception as ex:  # noqa: BLE001
        return None, f"fetch failed ({ex}) — offline? prime the cache with --prime"
    return (out if ok and out.is_file() else None), ("fetched" if ok else "no tiles")


# ---------------------------------------------------------------------
# datum conversion
# ---------------------------------------------------------------------
def geoid_undulation(lon: np.ndarray, lat: np.ndarray, datum: str) -> np.ndarray | None:
    """N = ellipsoidal height of the geoid `datum` at (lon, lat), metres.

    Uses PROJ's geoid grids (us_nga_egm96_15 / us_nga_egm08_25).  PROJ fetches
    the chunks it needs from cdn.proj.org and caches them in its user data dir;
    for a fully offline machine run `projsync --file us_nga_egm96_15.tif
    --file us_nga_egm08_25.tif` once.  Returns None when no grid is reachable.
    """
    epsg = DATUM_EPSG.get(datum)
    if epsg is None:
        return np.zeros_like(np.asarray(lon, np.float64))
    try:
        import pyproj
        from pyproj import Transformer

        if os.environ.get("DW_PROJ_NETWORK", "1") != "0":
            pyproj.network.set_network_enabled(True)
        t = Transformer.from_crs(f"EPSG:4326+{epsg}", "EPSG:4979", always_xy=True)
        lon = np.asarray(lon, np.float64)
        lat = np.asarray(lat, np.float64)
        _, _, h = t.transform(lon, lat, np.zeros_like(lon))
        h = np.asarray(h, np.float64)
        if not np.isfinite(h).all() or np.allclose(h, 0.0):
            return None                      # PROJ silently fell back to "no grid"
        return h
    except Exception as e:  # noqa: BLE001
        print(f"[dem] geoid grid for {datum} unavailable: {e}")
        return None


def to_datum(array: np.ndarray, transform, crs, from_datum: str, to_datum_: str,
             samples: int = 17) -> tuple[np.ndarray, dict]:
    """Re-express heights on the (transform, crs) grid in another vertical datum.

    The geoid varies over tens of km, so N is evaluated on a coarse
    `samples x samples` lattice and bilinearly interpolated — exact to cm at
    scene scale and ~free.  Unknown datums, or a missing grid, leave the array
    unchanged and say so; the caller records it in the product metadata.
    """
    info = {"from": from_datum, "to": to_datum_, "applied": False}
    if not from_datum or not to_datum_ or from_datum == to_datum_:
        info["applied"] = from_datum == to_datum_ and bool(from_datum)
        info["note"] = "same datum" if info["applied"] else "datum unknown — not converted"
        return array, info
    from rasterio.warp import transform as warp_xy

    H, W = array.shape
    rows = np.linspace(0, H - 1, samples)
    cols = np.linspace(0, W - 1, samples)
    cc, rr = np.meshgrid(cols, rows)
    xs, ys = transform * (cc.ravel() + 0.5, rr.ravel() + 0.5)
    lon, lat = warp_xy(crs, "EPSG:4326", list(np.asarray(xs)), list(np.asarray(ys)))
    n_from = geoid_undulation(np.array(lon), np.array(lat), from_datum)
    n_to = geoid_undulation(np.array(lon), np.array(lat), to_datum_)
    if n_from is None or n_to is None:
        print(f"[dem] WARNING: cannot convert {from_datum} -> {to_datum_} "
              "(no PROJ geoid grid); heights left in the source datum")
        info["note"] = "geoid grid unavailable — not converted"
        return array, info
    # h_ellps = H_from + N_from = H_to + N_to  ->  H_to = H_from + N_from - N_to
    d = (n_from - n_to).reshape(samples, samples).astype(np.float32)
    from scipy.ndimage import map_coordinates

    ry = np.linspace(0, samples - 1, H, dtype=np.float32)
    rx = np.linspace(0, samples - 1, W, dtype=np.float32)
    gy, gx = np.meshgrid(ry, rx, indexing="ij")
    delta = map_coordinates(d, [gy, gx], order=1, mode="nearest")
    info.update(applied=True, shift_mean_m=float(delta.mean()),
                shift_range_m=[float(delta.min()), float(delta.max())])
    return (array + delta).astype(np.float32), info


# ---------------------------------------------------------------------
# public entry points
# ---------------------------------------------------------------------
def fetch_dem(transform, crs, width: int, height: int, *, source: str = "copernicus30",
              local_path: str = "", local_datum: str = "", cache_dir: str = "",
              timeout: float = 60.0) -> DemResult:
    """Coarse elevation reprojected onto the (transform, crs, width, height) grid,
    in the source's own vertical datum (`DemResult.datum`)."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    source = canonical_source(source)
    info = SOURCE_INFO[source]
    dst = np.full((height, width), np.nan, np.float32)
    used, note = [], info.get("note", "")
    datum = info["datum"] if source != "local" else local_datum

    if info["kind"] == "local" or local_path:
        if not local_path:
            return DemResult(dst, source, [], 0.0, "no --dem path given", datum)
        candidates = [local_path]
        if source == "cartodem":
            note = ("CartoDEM supplied as a local file (WGS84 ellipsoidal heights "
                    "per the NRSC spec — converted if a target datum is asked for)")
    else:
        cdir = Path(cache_dir) if cache_dir else default_cache_dir()
        bbox = _pad_bbox(_bounds_lonlat(transform, crs, width, height))
        path, how = _fetch_to_cache(source, bbox, cdir, timeout)
        print(f"[dem] {source}: {how}" + (f" -> {path}" if path else ""))
        candidates = [str(path)] if path else []
        if not path:
            note = how

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
    print(f"[dem] {source}: {len(used)} file(s), {cov * 100:.1f}% coverage, "
          f"datum {datum or 'unknown'}")
    return DemResult(dst, source, used, cov, note, datum)


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
                       local_path: str = "", local_datum: str = "",
                       target_datum: str = "", cache_dir: str = "") -> DemResult | None:
    """`fetch_dem` for a `preprocess.SceneMeta`, optionally re-expressed in
    `target_datum` (the anchor the output DSM should be in)."""
    if not meta.georeferenced and source != "local":
        print("[dem] scene is not georeferenced — a remote DEM cannot be located")
        return None
    h, w = height_shape
    res = fetch_dem(meta.transform, meta.crs, w, h, source=source,
                    local_path=local_path, local_datum=local_datum,
                    cache_dir=cache_dir)
    if target_datum and res.coverage > 0 and res.datum != target_datum:
        arr, dinfo = to_datum(res.array, meta.transform, meta.crs, res.datum, target_datum)
        res.array = arr
        if dinfo.get("applied"):
            res.note = (res.note + "; " if res.note else "") + \
                f"converted {res.datum} -> {target_datum} (mean {dinfo['shift_mean_m']:+.2f} m)"
            res.datum = target_datum
        else:
            res.note = (res.note + "; " if res.note else "") + dinfo.get("note", "")
    return res


def vertical_crs_wkt(horizontal_crs, datum: str) -> str | None:
    """Compound CRS (horizontal + vertical datum) for GeoTIFF tagging, or None.

    GDAL writes GeoTIFF 1.1 keys when the CRS has a vertical part, so a reader
    can see that `dsm_m.tif` is, e.g., UTM 45N + EGM2008 height.
    """
    epsg = DATUM_EPSG.get(datum)
    if epsg is None:
        return None
    try:
        from pyproj import CRS
        from pyproj.crs import CompoundCRS

        h = CRS.from_user_input(str(horizontal_crs))
        v = CRS.from_epsg(epsg)
        return CompoundCRS(name=f"{h.name} + {v.name}", components=[h, v]).to_wkt()
    except Exception as e:  # noqa: BLE001
        print(f"[dem] could not build a compound CRS: {e}")
        return None


def _prime(argv=None) -> None:
    """Fill the DEM cache for a scene so the geo path works offline later."""
    import argparse

    ap = argparse.ArgumentParser(description="prime the DEM cache for a scene")
    ap.add_argument("--prime", required=True, help="GeoTIFF / product folder of the scene")
    ap.add_argument("--sources", default="copernicus30,srtmgl1")
    ap.add_argument("--cache_dir", default="")
    a = ap.parse_args(argv)
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dwdata.preprocess import scene_geometry

    meta = scene_geometry(a.prime)
    for s in a.sources.split(","):
        r = fetch_dem(meta.transform, meta.crs, 64, 64, source=s.strip(),
                      cache_dir=a.cache_dir)
        print(f"  {s}: coverage {r.coverage:.2f}  {r.note}")


if __name__ == "__main__":
    _prime()
