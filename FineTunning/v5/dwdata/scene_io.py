"""Reading real deliverable inputs: GeoTIFFs, NRSC Cartosat products, PAN+MX pairs.

v4's `read_scene` read bands `[1, 2, 3]` of one file.  The two Cartosat-2E
samples in `cartosat_2S_Sample/` show why that is not enough for the images the
judges will actually send (`CompetitionContext/V5_Research_Additions.md` §1):

* **The NRSC MERGED product is four single-band files** (`BAND1..4.tif`, 0.6 m,
  blue/green/red/NIR) plus `BAND_META.txt` — there is no RGB file to open.  And a
  4-band stack read as `[1, 2, 3]` is *BGR*.
* **11-bit UInt16 with NoData 0** and a rotated-scene collar (29 % of the PAN
  sample).  v4 stretched over the collar, predicted terrain on it and wrote it
  out as valid DSM.
* **20k x 20k pixels.**  3 bands x 829 MB of UInt16, before any float copy —
  the whole-array path cannot hold it, so `SceneSource` reads *row bands*
  (NRSC TIFFs are stored one row per block, so 2-D windows would re-read rows).
* The metadata carries the sun azimuth/elevation and the tilt, which the shadow
  layer and the shadow scale check (`viz/shadow.py`) use directly.

`SceneSource` is the one reader: it resolves any of

    scene.tif / scene.png / scene.jpg              a single image
    <product dir> | <product>.zip                  NRSC product (BAND*.tif + BAND_META*.txt)
    [<PAN product>, <MX product>]                  pan-sharpened on the fly (Brovey)

and hands out uint8 RGB rows plus a validity mask, with one radiometric mapping
per *scene* so every band of a windowed run sees identical colours.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

META_NAMES = ("BAND_META.txt", "BAND_METAP.txt")
_BAND_RE = re.compile(r"^BAND(\d)\.tif{1,2}$", re.I)


# ---------------------------------------------------------------------
# product metadata
# ---------------------------------------------------------------------
def parse_meta_txt(text: str) -> dict[str, str]:
    """`Key=Value` lines of an NRSC `BAND_META.txt` -> dict (values stripped)."""
    out = {}
    for ln in text.splitlines():
        if "=" in ln:
            k, v = ln.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _f(d: dict, k: str) -> float | None:
    try:
        return float(d[k])
    except (KeyError, ValueError):
        return None


@dataclass
class ProductInfo:
    """What an NRSC product folder says about itself."""

    root: str
    satellite: str = ""
    sensor: str = ""                 # "MX" | "PAN"
    level: str = ""                  # "MERGED" | "ORTHORECTIFIED" | "GEOREF" | ...
    date: str = ""
    sun_azimuth_deg: float | None = None
    sun_elevation_deg: float | None = None
    tilt_deg: float | None = None
    mean_elevation_m: float | None = None   # NRSC: WGS84 *ellipsoidal* — never use as terrain
    pixel_spacing_m: float | None = None
    bands: dict = field(default_factory=dict)   # band number -> path (GDAL-openable)
    raw: dict = field(default_factory=dict)

    @property
    def is_pan(self) -> bool:
        return self.sensor.upper() == "PAN" or (len(self.bands) == 1)

    def summary(self) -> dict:
        return {"satellite": self.satellite, "sensor": self.sensor, "level": self.level,
                "date": self.date, "sun_azimuth_deg": self.sun_azimuth_deg,
                "sun_elevation_deg": self.sun_elevation_deg, "tilt_deg": self.tilt_deg,
                "pixel_spacing_m": self.pixel_spacing_m,
                "mean_elevation_m_ellipsoidal": self.mean_elevation_m,
                "bands": {int(k): str(v) for k, v in self.bands.items()}}


def _listing(p: Path) -> list[tuple[str, str]]:
    """(name, GDAL path) for every file in a folder or a zip (recursively)."""
    if p.is_dir():
        return [(f.name, str(f)) for f in sorted(p.rglob("*")) if f.is_file()]
    if p.suffix.lower() == ".zip" and p.is_file():
        with zipfile.ZipFile(p) as z:
            return [(Path(n).name, f"/vsizip/{p}/{n}") for n in sorted(z.namelist())
                    if not n.endswith("/")]
    return []


def _read_text(gdal_path: str) -> str:
    if gdal_path.startswith("/vsizip/"):
        zpath, _, member = gdal_path[len("/vsizip/"):].partition(".zip/")
        with zipfile.ZipFile(zpath + ".zip") as z:
            return z.read(member).decode("latin-1")
    return Path(gdal_path).read_text(encoding="latin-1")


def find_product(path: str | Path) -> ProductInfo | None:
    """An NRSC product folder / zip -> ProductInfo, or None if it is not one.

    MX products use `BAND_META.txt`; a MERGED product carries both
    `BAND_META.txt` (MX) and `BAND_METAP.txt` (PAN) — the MX one wins, it is the
    one describing the delivered bands.
    """
    p = Path(path)
    files = _listing(p)
    if not files:
        return None
    bands = {}
    metas = {}
    for name, gp in files:
        m = _BAND_RE.match(name)
        if m:
            bands.setdefault(int(m.group(1)), gp)
        elif name.upper() == "BAND.TIF":
            bands.setdefault(1, gp)
        elif name in META_NAMES:
            metas[name] = gp
    if not bands:
        return None
    meta = {}
    for name in META_NAMES:            # BAND_META first: it describes the bands
        if name in metas:
            meta = parse_meta_txt(_read_text(metas[name]))
            break
    return ProductInfo(
        root=str(p), satellite=meta.get("SatID", ""), sensor=meta.get("Sensor", ""),
        level=meta.get("ProcessingLevel", ""), date=meta.get("DateOfPass", ""),
        sun_azimuth_deg=_f(meta, "SunAzimuthAtCenter"),
        sun_elevation_deg=_f(meta, "SunElevationAtCenter"),
        tilt_deg=_f(meta, "TiltAngle"), mean_elevation_m=_f(meta, "MeanElevation"),
        pixel_spacing_m=_f(meta, "PixelSpacingAlong"), bands=bands, raw=meta)


# ---------------------------------------------------------------------
# band selection
# ---------------------------------------------------------------------
def rgb_band_indexes(count: int, colorinterp=None, override=None) -> list[int]:
    """1-based band indexes to read as (R, G, B).

    Order of evidence: an explicit `override` (`--bands 3,2,1`), the file's own
    ColorInterp tags, then convention — 4+ bands are VNIR stacks in sensor order
    (Cartosat-2S/2E, WorldView-2 4-band, PlanetScope: B, G, R, NIR) so RGB is
    3,2,1; 3 bands are assumed RGB; 1 band is grey.
    """
    if override:
        idx = [int(b) for b in override]
        if len(idx) == 1:
            idx = idx * 3
        if len(idx) != 3 or not all(1 <= b <= count for b in idx):
            raise ValueError(f"--bands {override} does not fit a {count}-band input")
        return idx
    if colorinterp:
        names = [str(getattr(c, "name", c)).lower() for c in colorinterp]
        if all(n in names for n in ("red", "green", "blue")):
            return [names.index("red") + 1, names.index("green") + 1, names.index("blue") + 1]
    if count >= 4:
        return [3, 2, 1]
    if count == 3:
        return [1, 2, 3]
    return [1, 1, 1]


# ---------------------------------------------------------------------
# the reader
# ---------------------------------------------------------------------
class SceneSource:
    """Row-band access to any supported input, as uint8 RGB + validity.

    Radiometry: native values (UInt16 11-bit, float, uint8) are mapped to uint8
    per band by a *scene-wide* linear map between the 0.1 / 99.9 percentiles of
    **valid** pixels.  That keeps the full dynamic range (the percentile stretch
    of the preprocessing contract runs afterwards, on the uint8 image, exactly as
    in training) while making the mapping identical for every row band.
    """

    def __init__(self, inputs, bands=None, user_gsd_m: float = 0.0,
                 assumed_gsd_m: float = 0.5):
        if isinstance(inputs, (str, Path)):
            inputs = [inputs]
        self.inputs = [Path(x) for x in inputs]
        self.product: ProductInfo | None = None
        self._pan_path: str | None = None
        self._ms_paths: list[str] = []          # one path per band, or [one multi-band file]
        self._band_idx: list[int] = [1, 2, 3]
        self._array: np.ndarray | None = None   # PNG/JPG: whole image in memory
        self.pansharpened = False
        self._lo = self._hi = None

        prods = [find_product(p) for p in self.inputs]
        if len(self.inputs) == 2 and all(prods):
            pan = next((q for q in prods if q.is_pan), None)
            ms = next((q for q in prods if not q.is_pan), None)
            if pan is None or ms is None:
                raise ValueError("two products given but not one PAN + one MX")
            self.product = ms
            self._pan_path = pan.bands[1]
            self._ms_paths = [ms.bands[b] for b in sorted(ms.bands)]
            self._band_idx = [3, 2, 1] if len(ms.bands) >= 3 else [1, 1, 1]
            self.pansharpened = True
            self.product.level = self.product.level + "+PANSHARPENED(brovey)"
            grid_path = self._pan_path
        elif prods[0] is not None and len(self.inputs) == 1:
            self.product = prods[0]
            self._ms_paths = [self.product.bands[b] for b in sorted(self.product.bands)]
            n = len(self._ms_paths)
            self._band_idx = rgb_band_indexes(n, None, bands)
            grid_path = self._ms_paths[0]
        elif len(self.inputs) >= 3 and all(p.suffix.lower() in (".tif", ".tiff")
                                           for p in self.inputs):
            # loose single-band files: sort by the BANDn number when present
            def key(p):
                m = _BAND_RE.match(p.name)
                return int(m.group(1)) if m else 0
            self._ms_paths = [str(p) for p in sorted(self.inputs, key=key)]
            self._band_idx = rgb_band_indexes(len(self._ms_paths), None, bands)
            grid_path = self._ms_paths[0]
        else:
            p = self.inputs[0]
            if p.suffix.lower() in (".tif", ".tiff", ".gtif", ".gtiff", ".vrt"):
                self._ms_paths = [str(p)]
                grid_path = str(p)
            else:
                grid_path = None

        self._init_grid(grid_path, bands, user_gsd_m, assumed_gsd_m)

    # -- geometry ---------------------------------------------------------
    def _init_grid(self, grid_path, bands, user_gsd_m, assumed_gsd_m):
        from .preprocess import _metres_per_unit

        self.nodata = None
        if grid_path is None:                       # PNG / JPG
            from PIL import Image

            Image.MAX_IMAGE_PIXELS = None
            im = Image.open(self.inputs[0])
            a = np.asarray(im.convert("RGB")) if im.mode not in ("I;16", "I") else np.asarray(im)
            if a.ndim == 2:
                a = np.stack([a] * 3, -1)
            self._array = a
            self.height, self.width = a.shape[:2]
            self.transform = self.crs = None
            self.georeferenced = False
            self.gsd_m = float(user_gsd_m or assumed_gsd_m)
            self.gsd_source = "user" if user_gsd_m else "assumed"
            return
        import rasterio

        with rasterio.open(grid_path) as ds:
            self.height, self.width = ds.height, ds.width
            self.transform, self.crs = ds.transform, ds.crs
            self.nodata = ds.nodata
            count, ci = ds.count, ds.colorinterp
        if len(self._ms_paths) == 1 and not self.pansharpened:
            self._band_idx = rgb_band_indexes(count, ci, bands)
        self.georeferenced = self.crs is not None and abs(self.transform.a) > 0 \
            and not self.transform.is_identity
        native = abs(self.transform.a) * _metres_per_unit(
            self.crs, self.transform, self.width, self.height)
        self.gsd_m = float(user_gsd_m or (native if self.georeferenced else assumed_gsd_m))
        self.gsd_source = "user" if user_gsd_m else ("geotiff" if self.georeferenced else "assumed")

    # -- raw access -------------------------------------------------------
    def _read_native(self, r0: int, r1: int, step: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """(rows, W, 3) float32 native values + (rows, W) valid, for rows [r0, r1)."""
        if self._array is not None:
            a = self._array[r0:r1:step, ::step].astype(np.float32)
            return a[..., :3], np.ones(a.shape[:2], bool)
        import rasterio
        from rasterio.windows import Window

        h = r1 - r0
        out_shape = (max(1, -(-h // step)), max(1, -(-self.width // step)))
        win = Window(0, r0, self.width, h)

        def read_band(path, b):
            with rasterio.open(path) as ds:
                a = ds.read(b, window=win, out_shape=out_shape).astype(np.float32)
                nd = ds.nodata
            return a, nd

        if len(self._ms_paths) == 1:
            chans = [read_band(self._ms_paths[0], b) for b in self._band_idx]
        else:
            chans = [read_band(self._ms_paths[b - 1], 1) for b in self._band_idx]
        if self.pansharpened:
            return self._brovey(win, out_shape)
        nd = chans[0][1]
        rgb = np.stack([c[0] for c in chans], -1)
        valid = np.ones(out_shape, bool) if nd is None else ~np.all(rgb == nd, axis=-1)
        valid &= np.isfinite(rgb).all(-1)
        return rgb, valid

    def _brovey(self, win, out_shape):
        """MX resampled onto the PAN grid, then RGB * PAN / mean(B, G, R, NIR).

        GDAL's `gdal_pansharpen` default (weighted Brovey, equal weights over the
        spectral bands given).  The PAN band spans the visible + NIR, so the NIR
        band belongs in the intensity term.  For a registration-checked product
        use `tools/cartosat_to_rgb.py`, which measures and removes a PAN/MX shift
        first; this path trusts the two products' georeferencing.
        """
        import rasterio
        from rasterio.vrt import WarpedVRT
        from rasterio.warp import Resampling

        with rasterio.open(self._pan_path) as ps:
            pan = ps.read(1, window=win, out_shape=out_shape).astype(np.float32)
            nd = ps.nodata
            wt = ps.window_transform(win)
            sx = win.width / out_shape[1]
            sy = win.height / out_shape[0]
            from affine import Affine

            dst_tr = wt * Affine.scale(sx, sy)
            ms = []
            for p in self._ms_paths:
                with rasterio.open(p) as src, WarpedVRT(
                        src, crs=ps.crs, transform=dst_tr, width=out_shape[1],
                        height=out_shape[0], resampling=Resampling.bilinear) as v:
                    ms.append(v.read(1).astype(np.float32))
        ms = np.stack(ms, -1)
        inten = ms.mean(-1)
        ratio = np.where(inten > 0, pan / np.maximum(inten, 1e-3), 0.0)
        rgb = np.stack([ms[..., b - 1] * ratio for b in self._band_idx], -1)
        valid = (pan != (nd if nd is not None else -1)) & (inten > 0)
        return rgb, valid

    # -- radiometry -------------------------------------------------------
    def radiometry(self, max_samples: int = 4_000_000) -> tuple[np.ndarray, np.ndarray]:
        """Scene-wide per-band (lo, hi) in native units, from valid pixels only."""
        if self._lo is not None:
            return self._lo, self._hi
        n = self.height * self.width
        step = max(1, int(np.ceil(np.sqrt(n / max_samples))))
        rgb, valid = self._read_native(0, self.height, step)
        v = rgb[valid]
        if v.size == 0:
            lo, hi = np.zeros(3, np.float32), np.full(3, 255.0, np.float32)
        elif self._is_uint8_native():
            lo, hi = np.zeros(3, np.float32), np.full(3, 255.0, np.float32)
        else:
            lo = np.percentile(v, 0.1, axis=0).astype(np.float32)
            hi = np.percentile(v, 99.9, axis=0).astype(np.float32)
            hi = np.maximum(hi, lo + 1.0)
        self._lo, self._hi = lo, hi
        return lo, hi

    def _is_uint8_native(self) -> bool:
        if self._array is not None:
            return self._array.dtype == np.uint8
        import rasterio

        with rasterio.open(self._pan_path or self._ms_paths[0]) as ds:
            return ds.dtypes[0] == "uint8" and not self.pansharpened

    def read_rows(self, r0: int, r1: int, step: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """uint8 RGB (rows, W, 3) + valid (rows, W) for source rows [r0, r1)."""
        lo, hi = self.radiometry()
        rgb, valid = self._read_native(r0, r1, step)
        x = (rgb - lo) / (hi - lo)
        u8 = (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        u8[~valid] = 0
        return u8, valid

    def read_all(self, step: int = 1):
        return self.read_rows(0, self.height, step)

    def meta_dict(self) -> dict:
        d = {"inputs": [str(p) for p in self.inputs], "rgb_bands": self._band_idx,
             "pansharpened": self.pansharpened}
        if self.product:
            d["product"] = self.product.summary()
        return d


def is_product_or_multi(path_or_paths) -> bool:
    """True when the input needs `SceneSource` rather than a plain image read."""
    if isinstance(path_or_paths, (list, tuple)):
        return len(path_or_paths) > 1 or is_product_or_multi(path_or_paths[0])
    p = Path(path_or_paths)
    return p.is_dir() or p.suffix.lower() == ".zip"
