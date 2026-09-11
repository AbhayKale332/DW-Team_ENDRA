"""Indian-imagery ingest — the domain the model is actually graded on.

**Why this module exists.**  Every open RGB+height pair we can train on is
foreign: GAMUS is five US cities, GeoNRW is North Rhine-Westphalia, SynRS3D is
synthetic, DFC2019 is Jacksonville/Omaha.  ISRO will evaluate on Cartosat-class
imagery over Indian cities, whose morphology (dense low-rise with narrow
irregular streets, flat roofs, very different roof albedo and street width) and
radiometry are both outside that training distribution.  v2 already demonstrated
what that costs: on Inria Austin — a *US* city at nearly the same GSD as GAMUS —
it put ~4 m of height on flat ground and the render was a mountain range.

**What is actually available.**  We looked for an open Indian dataset with
per-pixel height labels and there is not one:

| candidate | height labels over India? | usable here? |
|---|---|---|
| GAMUS / DFC2019 / GeoNRW | no | foreign only |
| DFC2023 Track 2 (has a **New Delhi** city) | yes, 2 m nDSM from Gaofen-7/WV stereo | IEEE DataPort login; no API — supported below via `--india_dir` once downloaded by hand |
| Open Buildings 2.5D, UT-GLOBUS | building heights only, 4 m+, no paired RGB | weak, needs its own basemap pairing |
| Bhoonidhi / Bhuvan (CartoDEM, Cartosat) | DEM at 10-30 m | too coarse for building height, fine for the *terrain* term (see `geo/`) |

So this module supports the two ingest paths that genuinely work:

1. **`pack_labeled`** — a local directory of paired rasters.  Use it for
   DFC2023's New Delhi tiles, for any Indian stereo/LiDAR product the team gets
   hold of, or for a hand-built set.  Pairing is by filename stem plus a suffix
   table, so it accepts most naming conventions without editing code.

2. **`pack_unlabeled`** — RGB-only Indian tiles, no heights.  These feed the
   mean-teacher consistency branch in `train.py`: the EMA teacher predicts a
   weakly-augmented view, the student is pulled towards it on a strongly
   augmented one, and the encoder's features migrate towards Indian radiometry
   and street morphology without ever needing a label.  This is the TSE-Net-style
   self-training the plan schedules for the finals, and unlabeled Indian imagery
   *is* obtainable (Bhuvan/Bhoonidhi browse exports, any tile source the team is
   licensed for).

`fetch_xyz_tiles` will build (2) from an XYZ template over the Indian city boxes
below, but **the URL is deliberately not defaulted**: which basemap a government
deliverable may train on is a licensing decision for the team, not for a script.
Point it at Bhuvan or at a source you hold rights to.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .packed import NO_LABEL, ShardWriter, store_exists

# Bounding boxes (min_lon, min_lat, max_lon, max_lat) picked for *morphological
# spread*, not size — the rubric grades stability across urban / sparse / hilly /
# forested, so the unlabeled set should contain all four.
INDIA_AOIS: dict[str, tuple[float, float, float, float, str]] = {
    # dense, irregular, low-rise-with-highrise urban
    "mumbai":      (72.79, 18.89, 72.99, 19.16, "urban"),
    "kolkata":     (88.30, 22.48, 88.44, 22.63, "urban"),
    "delhi":       (77.10, 28.52, 77.32, 28.70, "urban"),
    "bengaluru":   (77.53, 12.90, 77.68, 13.03, "urban"),
    "hyderabad":   (78.38, 17.36, 78.52, 17.47, "urban"),
    # planned grid / low density
    "chandigarh":  (76.72, 30.69, 76.82, 30.78, "sparse"),
    "gandhinagar": (72.60, 23.19, 72.70, 23.28, "sparse"),
    "jaipur":      (75.76, 26.86, 75.86, 26.95, "sparse"),
    # hills and steep terrain
    "shimla":      (77.13, 31.07, 77.22, 31.13, "hilly"),
    "dehradun":    (77.96, 30.29, 78.08, 30.38, "hilly"),
    "gangtok":     (88.58, 27.30, 88.65, 27.36, "hilly"),
    # forest / vegetation-dominated
    "coorg":       (75.72, 12.34, 75.85, 12.45, "forested"),
    "nilgiris":    (76.65, 11.38, 76.78, 11.48, "forested"),
    "sundarbans":  (88.70, 21.85, 88.85, 21.98, "forested"),
}

# Suffixes we accept when pairing files inside `--india_dir`.  First match wins.
RGB_SUFFIXES = ("_rgb", "_RGB", "_img", "_image", "_opt", "")
HGT_SUFFIXES = ("_ndsm", "_nDSM", "_NDSM", "_agl", "_AGL", "_dsm", "_height", "_h")
SEG_SUFFIXES = ("_seg", "_cls", "_CLS", "_label", "_mask")
RASTER_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


# ---------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------
def read_raster(path: Path) -> np.ndarray:
    """Any of GeoTIFF / PNG / JPEG -> ndarray, bands last."""
    if path.suffix.lower() in (".tif", ".tiff"):
        try:
            import rasterio

            with rasterio.open(path) as ds:
                a = ds.read()
            return np.transpose(a, (1, 2, 0)) if a.shape[0] > 1 else a[0]
        except Exception:  # noqa: BLE001 — fall through to PIL
            pass
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    return np.asarray(Image.open(path))


def raster_gsd_m(path: Path, default: float) -> float:
    """GSD from a GeoTIFF transform, else `default`."""
    if path.suffix.lower() not in (".tif", ".tiff"):
        return default
    try:
        import rasterio

        from .preprocess import _metres_per_unit

        with rasterio.open(path) as ds:
            if ds.crs is None:
                return default
            return float(abs(ds.transform.a)) * _metres_per_unit(
                ds.crs, ds.transform, ds.width, ds.height)
    except Exception:  # noqa: BLE001
        return default


def _stem_key(p: Path) -> str:
    s = p.stem
    for suf in sorted(RGB_SUFFIXES + HGT_SUFFIXES + SEG_SUFFIXES, key=len, reverse=True):
        if suf and s.endswith(suf):
            return s[: -len(suf)]
    return s


def pair_rasters(root: Path) -> list[dict]:
    """Group a directory tree into {stem, rgb, hgt, seg} records.

    A record needs an RGB; `hgt`/`seg` are None when nothing matched, which is
    how the same walker serves both the labeled and the unlabeled path.
    """
    by_key: dict[str, dict] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in RASTER_EXT:
            continue
        key = _stem_key(p)
        rec = by_key.setdefault(key, {"stem": key, "rgb": None, "hgt": None, "seg": None})
        s = p.stem
        if any(s.endswith(x) for x in HGT_SUFFIXES if x):
            rec["hgt"] = rec["hgt"] or p
        elif any(s.endswith(x) for x in SEG_SUFFIXES if x):
            rec["seg"] = rec["seg"] or p
        else:
            rec["rgb"] = rec["rgb"] or p
    return [r for r in by_key.values() if r["rgb"] is not None]


def _to_rgb_u8(a: np.ndarray) -> np.ndarray:
    from .preprocess import _to_uint8_rgb

    return _to_uint8_rgb(a)


def _grid(h: int, w: int, tile: int, stride: int):
    ys = list(range(0, max(1, h - tile + 1), stride)) or [0]
    xs = list(range(0, max(1, w - tile + 1), stride)) or [0]
    return [(y, x) for y in ys for x in xs if y + tile <= h and x + tile <= w]


# ---------------------------------------------------------------------
# packing
# ---------------------------------------------------------------------
def pack_labeled(src_dir: Path, out_dir: Path, tile_px: int, gsd_m: float,
                 max_tiles: int = 0, force: bool = False,
                 dsm_is_absolute: bool = False) -> dict | None:
    """Paired Indian rasters -> a packed store the trainer can sample directly.

    `dsm_is_absolute=True` says the height raster is elevation above sea level
    rather than above ground, in which case a morphological ground estimate is
    subtracted (the same coarse approximation GeoNRW gets, and just as
    auxiliary-grade — say so in the report if you use it).
    """
    if store_exists(out_dir) and not force:
        print(f"[india/labeled] already prepared -> {out_dir}")
        return None
    recs = [r for r in pair_rasters(src_dir) if r["hgt"] is not None]
    if not recs:
        print(f"[india/labeled] no rgb+height pairs under {src_dir} — nothing packed. "
              f"Expected e.g. delhi_001_rgb.tif + delhi_001_ndsm.tif")
        return None
    print(f"[india/labeled] {len(recs)} pairs under {src_dir} -> {out_dir}")

    w = ShardWriter(out_dir, tile_px=tile_px, gsd_m=gsd_m, shard_tiles=128)
    n = 0
    for rec in recs:
        try:
            rgb = _to_rgb_u8(read_raster(rec["rgb"]))
            hgt = np.asarray(read_raster(rec["hgt"]), np.float32)
            if hgt.ndim == 3:
                hgt = hgt[..., 0]
            if hgt.shape[:2] != rgb.shape[:2]:
                from .preprocess import resize

                hgt = resize(hgt, rgb.shape[:2], "bilinear")
            if dsm_is_absolute:
                hgt = ndsm_from_absolute(hgt)
            seg = None
            if rec["seg"] is not None:
                seg = np.asarray(read_raster(rec["seg"]), np.int32)
                if seg.ndim == 3:
                    seg = seg[..., 0]
                from .preprocess import resize

                seg = resize(seg, rgb.shape[:2], "nearest")
            valid = np.isfinite(hgt) & (hgt > -2.0) & (hgt < 500.0)
            hgt = np.where(valid, np.clip(hgt, 0.0, None), 0.0).astype(np.float32)
            for y, x in _grid(*rgb.shape[:2], tile_px, tile_px):
                sl = (slice(y, y + tile_px), slice(x, x + tile_px))
                v = valid[sl]
                if v.mean() < 0.5:                      # mostly no-data, skip
                    continue
                w.add(f"{rec['stem']}_{y}_{x}", rgb[sl], hgt[sl],
                      None if seg is None else np.clip(seg[sl], 0, 254).astype(np.uint8), v)
                n += 1
                if max_tiles and n >= max_tiles:
                    break
        except Exception as e:  # noqa: BLE001
            print(f"[india/labeled] skip {rec['stem']}: {e}")
        if max_tiles and n >= max_tiles:
            break
    idx = w.finalise()
    print(f"[india/labeled] packed {idx['n']} tiles @ {tile_px}px / {gsd_m} m")
    return idx


def pack_unlabeled(src_dir: Path, out_dir: Path, tile_px: int, gsd_m: float,
                   max_tiles: int = 0, force: bool = False) -> dict | None:
    """RGB-only Indian tiles -> a store with `valid` all False.

    Everything downstream reads `valid` before touching `height`, so an all-False
    mask is exactly "there is no supervision here" — it can never be mistaken for
    a 0 m label, which is the bug that put 41 % black-padding-at-0 m into v2.
    """
    if store_exists(out_dir) and not force:
        print(f"[india/unlabeled] already prepared -> {out_dir}")
        return None
    recs = pair_rasters(src_dir)
    if not recs:
        print(f"[india/unlabeled] no rasters under {src_dir}")
        return None
    print(f"[india/unlabeled] {len(recs)} scenes under {src_dir} -> {out_dir}")

    w = ShardWriter(out_dir, tile_px=tile_px, gsd_m=gsd_m, shard_tiles=256,
                    has_seg=False)
    zero = np.zeros((tile_px, tile_px), np.float32)
    novalid = np.zeros((tile_px, tile_px), bool)
    n = 0
    for rec in recs:
        try:
            rgb = _to_rgb_u8(read_raster(rec["rgb"]))
            for y, x in _grid(*rgb.shape[:2], tile_px, tile_px):
                tile = rgb[y:y + tile_px, x:x + tile_px]
                # a tile that is mostly one flat colour is a nodata block or an
                # ocean square; neither teaches the encoder anything
                if float(tile.std()) < 6.0:
                    continue
                w.add(f"{rec['stem']}_{y}_{x}", tile, zero, None, novalid)
                n += 1
                if max_tiles and n >= max_tiles:
                    break
        except Exception as e:  # noqa: BLE001
            print(f"[india/unlabeled] skip {rec['stem']}: {e}")
        if max_tiles and n >= max_tiles:
            break
    idx = w.finalise()
    print(f"[india/unlabeled] packed {idx['n']} tiles @ {tile_px}px / {gsd_m} m")
    return idx


def ndsm_from_absolute(dem: np.ndarray, win: int = 120) -> np.ndarray:
    """Absolute elevation -> height above ground, by a morphological opening.

    Coarse on purpose: a large-window minimum filter followed by a Gaussian is a
    standard cheap DTM proxy, and it is why a store built this way is auxiliary
    supervision rather than a target you would quote a number against.
    """
    dem = np.nan_to_num(dem, nan=float(np.nanmedian(dem)))
    try:
        from scipy.ndimage import gaussian_filter, minimum_filter

        g = gaussian_filter(minimum_filter(dem, size=win, mode="nearest"), sigma=win / 3.0)
    except Exception:  # noqa: BLE001
        g = np.full_like(dem, float(np.percentile(dem, 5)))
    return np.clip(dem - g, 0.0, None)


# ---------------------------------------------------------------------
# optional: build the unlabeled set straight from an XYZ tile source
# ---------------------------------------------------------------------
def _deg2tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    r = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(r)) / math.pi) / 2.0 * n)
    return x, y


def tile_gsd_m(z: int, lat: float) -> float:
    """Web-Mercator ground sample distance of a 256 px tile at zoom `z`."""
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** z)


def fetch_xyz_tiles(url_template: str, out_dir: Path, aois: dict | None = None,
                    zoom: int = 18, per_aoi: int = 60, mosaic: int = 3,
                    user_agent: str = "DepthWizard/1.0", timeout: float = 20.0) -> int:
    """Download XYZ tiles over the Indian AOIs into `out_dir` as PNGs.

    `url_template` uses `{z}/{x}/{y}`.  **There is no default** — pick a source
    you are licensed to train on (ISRO Bhuvan is the obvious one for an ISRO
    deliverable) and pass it explicitly.  `mosaic` stitches an NxN block of
    tiles into one scene so `pack_unlabeled` gets something bigger than 256 px
    to cut 512 px training tiles out of.
    """
    import urllib.request

    from PIL import Image

    if not url_template:
        raise ValueError(
            "fetch_xyz_tiles needs an explicit --india_tile_url; no basemap is "
            "defaulted because the licence choice belongs to the team.")
    aois = aois or INDIA_AOIS
    out_dir.mkdir(parents=True, exist_ok=True)
    n_out = 0
    for name, (lon0, lat0, lon1, lat1, _kind) in aois.items():
        x0, y0 = _deg2tile(lon0, lat1, zoom)          # lat1 = north edge
        x1, y1 = _deg2tile(lon1, lat0, zoom)
        got = 0
        for by in range(y0, y1 + 1, mosaic):
            for bx in range(x0, x1 + 1, mosaic):
                if got >= per_aoi:
                    break
                canvas = Image.new("RGB", (256 * mosaic, 256 * mosaic))
                ok = True
                for dy in range(mosaic):
                    for dx in range(mosaic):
                        u = (url_template.replace("{z}", str(zoom))
                             .replace("{x}", str(bx + dx)).replace("{y}", str(by + dy)))
                        try:
                            req = urllib.request.Request(u, headers={"User-Agent": user_agent})
                            with urllib.request.urlopen(req, timeout=timeout) as r:
                                import io

                                canvas.paste(Image.open(io.BytesIO(r.read())).convert("RGB"),
                                             (dx * 256, dy * 256))
                        except Exception:  # noqa: BLE001
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    canvas.save(out_dir / f"{name}_{zoom}_{bx}_{by}_rgb.png")
                    got += 1
                    n_out += 1
            if got >= per_aoi:
                break
        print(f"[india/xyz] {name}: {got} scenes @ z{zoom} "
              f"(~{tile_gsd_m(zoom, (lat0 + lat1) / 2):.2f} m/px)", flush=True)
    return n_out
