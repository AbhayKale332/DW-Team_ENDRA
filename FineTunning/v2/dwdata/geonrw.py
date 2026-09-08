"""GeoNRW loader.

Source: `torchgeo/geonrw` on HF — a single `nrw_dataset.tar.gz` (~32 GB).  It is
NOT per-tile streamable, so v2 downloads + extracts the tar once into the bounded
cache (`BoundedCacheHF.ensure_archive`) and reads tiles locally; the extracted
tree is LRU-managed like everything else.  On Kaggle, point `--data_source local
--local_root /kaggle/input/geonrw` at the mounted dataset instead.

Layout after extraction: city folders each with
    <utm1>_<utm2>_rgb.jp2   (1000x1000x3, ~0.1 m resampled to 1 m)
    <utm1>_<utm2>_dem.tif   (1000x1000, metres, LiDAR first-return -> ~DSM)
    <utm1>_<utm2>_seg.tif   (1000x1000, 11 classes)

GeoNRW ships an elevation DEM, not an nDSM.  We derive a cheap nDSM:
    nDSM ~= clip(DEM - DTM_proxy, 0),  DTM_proxy = smoothed large-window minimum.
Good enough for auxiliary supervision; see `.agents/Depth_Wizard_Plan.md` note.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import TileDatasetBase

# GeoNRW 11-class id -> GAMUS 7-class id (ground0 veg1 building2 water3 road4 bridge5 other7)
_GEONRW_TO_GAMUS = {
    0: 7, 1: 1, 2: 3, 3: 1, 4: 2, 5: 1, 6: 4, 7: 4, 8: 0, 9: 4, 10: 2,
}

TRAIN_CITIES = {
    "aachen", "bergisch", "bielefeld", "bochum", "bonn", "borken", "bottrop",
    "coesfeld", "dortmund", "dueren", "duisburg", "ennepetal", "erftstadt",
    "essen", "euskirchen", "gelsenkirchen", "guetersloh", "hagen", "hamm",
    "heinsberg", "herford", "hoexter", "kleve", "koeln", "krefeld", "leverkusen",
    "lippetal", "lippstadt", "lotte", "moenchengladbach", "moers", "muelheim",
    "muenster", "oberhausen", "paderborn", "recklinghausen", "remscheid",
    "siegen", "solingen", "wuppertal",
}
TEST_CITIES = {"duesseldorf", "herne", "neuss"}


def _read_raster(path: Path) -> np.ndarray:
    try:
        import rasterio

        with rasterio.open(path) as ds:
            arr = ds.read()
        return np.transpose(arr, (1, 2, 0)) if arr.shape[0] > 1 else arr[0]
    except Exception:  # noqa: BLE001
        from PIL import Image

        return np.asarray(Image.open(path))


def _ndsm_from_dem(dem: np.ndarray, win: int = 120) -> np.ndarray:
    dem = dem.astype(np.float32)
    dem = np.nan_to_num(dem, nan=float(np.nanmedian(dem)))
    try:
        from scipy.ndimage import gaussian_filter, minimum_filter

        ground = minimum_filter(dem, size=win, mode="nearest")
        ground = gaussian_filter(ground, sigma=win / 3.0)
    except Exception:  # noqa: BLE001 - numpy-only fallback (coarse)
        k = max(1, win // 8)
        h, w = dem.shape
        pad = np.pad(dem, k, mode="edge")
        ground = np.stack(
            [pad[i:i + h, j:j + w] for i in range(0, 2 * k + 1, k) for j in range(0, 2 * k + 1, k)]
        ).min(0)
    return np.clip(dem - ground, 0.0, None)


def discover_tiles(base: Path, split: str) -> list[Path]:
    cities = TRAIN_CITIES if split == "train" else TEST_CITIES
    out: list[Path] = []
    for p in base.rglob("*_rgb.jp2"):
        city = p.parent.name.lower()
        if any(city.startswith(c) or c in str(p).lower() for c in cities):
            out.append(p)
    return sorted(out)


class GeoNRWDataset(TileDatasetBase):
    def __init__(self, cfg, rgb_paths: list[Path], split: str, train: bool):
        stems = [p.name[: -len("_rgb.jp2")] for p in rgb_paths]
        super().__init__(cfg, stems, "geonrw", train)
        self.rgb_paths = rgb_paths

    def load_tile(self, idx: int):
        rp = self.rgb_paths[idx]
        dem_p = rp.with_name(rp.name.replace("_rgb.jp2", "_dem.tif"))
        seg_p = rp.with_name(rp.name.replace("_rgb.jp2", "_seg.tif"))

        rgb = np.asarray(_read_raster(rp))[..., :3].astype(np.uint8)
        dem = _read_raster(dem_p).astype(np.float32)
        ndsm = _ndsm_from_dem(dem)
        if seg_p.exists():
            seg_raw = _read_raster(seg_p).astype(np.int64)
            seg = np.vectorize(lambda v: _GEONRW_TO_GAMUS.get(int(v), 7))(seg_raw)
            has_seg = True
        else:
            seg = np.full(ndsm.shape, 7, np.int64)
            has_seg = False
        return rgb, ndsm, seg, self.cfg.geonrw_gsd_m, has_seg
