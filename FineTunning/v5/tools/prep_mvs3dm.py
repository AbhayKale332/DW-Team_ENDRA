"""IARPA MVS3DM (WorldView-3, San Fernando AR) -> Cartosat-like validation sites.

    python tools/prep_mvs3dm.py --root ~/dw/data/mvs3dm --out ~/dw/data/mvs3dm/sites

For each of the 8 lidar ground-truth tiles (`Challenge_Data_and_Software/Lidar_gt`,
0.3 m UTM 21S, absolute DSM) this writes `<out>/<site>/`:

* `image_bgrn.tif`  4-band UInt16 B, G, R, NIR at 0.3 m on the GT grid — the
  band order of a Cartosat-2S MERGED product, so `scene_io` reads RGB as 3,2,1.
  WV3 MSI bands 2, 3, 5, 7 are orthorectified with the scene RPCs over the GT
  DSM itself (true ortho: roofs land on their footprints), co-registered, and
  pansharpened with the PAN band.
* `gt_dsm.tif`, `gt_dtm.tif`, `gt_ndsm.tif`  float32, NoData -9999.  The raw
  LAZ is unclassified (every point is class 1), so the DTM is a progressive
  morphological filter (Zhang 2003) of the DSM — fine on this flat suburb, an
  approximation under large sheds.
* `preview.png`, `site.json`  quick look and provenance (scene, shifts, NCC).

**Registration.**  WV3 RPCs are good to a few metres, i.e. ~10 px at 0.3 m.
The image is warped onto the GT grid plus a `--pad` margin, then the shift that
maximises the normalised cross-correlation of gradient magnitudes (PAN vs DSM,
then MS intensity vs PAN) is cut out.  The peak NCC is recorded; a site whose
peak is flat should not be trusted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.warp import Resampling, reproject
from scipy import ndimage
from scipy.signal import fftconvolve

MS_BGRN = (2, 3, 5, 7)      # WV3 MSI: coastal, blue, green, yellow, red, red-edge, NIR1, NIR2
ND = -9999.0
# WV3 PAN scenes by mean off-nadir angle (IMD), most nadir first
SCENES = ["15JAN11135414", "15APR22140347", "15JAN05135727", "15APR03140238",
          "14NOV15135121", "15OCT03140452", "15SEP27140912", "15OCT22140432",
          "15SEP14140305"]


def fill_nan(a: np.ndarray) -> np.ndarray:
    bad = ~np.isfinite(a)
    if not bad.any():
        return a
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def dtm_from_dsm(dsm: np.ndarray, gsd: float, cell_m: float = 1.0,
                 windows_m=(3, 9, 21, 41, 61), slope: float = 0.05,
                 dh0: float = 0.3, dh_max: float = 2.5) -> np.ndarray:
    """Progressive morphological ground filter on a `cell_m` min-pooled grid."""
    k = max(1, int(round(cell_m / gsd)))
    H, W = dsm.shape
    f = fill_nan(dsm)
    ph, pw = -H % k, -W % k
    f = np.pad(f, ((0, ph), (0, pw)), mode="edge")
    z = f.reshape(f.shape[0] // k, k, f.shape[1] // k, k).min((1, 3))
    ground = np.ones_like(z, bool)
    for w_m in windows_m:
        w = max(3, int(w_m / cell_m) | 1)
        opened = ndimage.grey_opening(z, size=(w, w))
        ground &= (z - opened) <= min(dh_max, dh0 + slope * w * cell_m)
    g = np.where(ground, z, np.nan)
    g = ndimage.gaussian_filter(fill_nan(g), 3.0)
    dtm = ndimage.zoom(g, k, order=1)[:H, :W]
    return np.minimum(dtm, f[:H, :W]).astype(np.float32)


def grad_mag(a: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    a = ndimage.gaussian_filter(a.astype(np.float32), sigma)
    return np.hypot(ndimage.sobel(a, 0), ndimage.sobel(a, 1))


def best_shift(big: np.ndarray, ref: np.ndarray, valid=None) -> tuple[int, int, float]:
    """(dy, dx, ncc) such that big[dy:dy+H, dx:dx+W] best matches `ref`."""
    g1 = np.log1p(grad_mag(big))
    g2 = np.log1p(grad_mag(ref))
    if valid is not None:
        g2 = np.where(valid, g2, 0)
    m = valid if valid is not None else np.ones_like(g2, bool)
    t = np.where(m, g2 - g2[m].mean(), 0).astype(np.float32)
    num = fftconvolve(g1, t[::-1, ::-1], mode="valid")
    mk = m.astype(np.float32)[::-1, ::-1]
    s1 = fftconvolve(g1, mk, mode="valid")
    s2 = fftconvolve(g1 * g1, mk, mode="valid")
    n = mk.sum()
    var1 = np.maximum(s2 - s1 * s1 / n, 1e-6)
    ncc = num / np.sqrt(var1 * (t * t).sum())
    dy, dx = np.unravel_index(np.argmax(ncc), ncc.shape)
    return int(dy), int(dx), float(ncc[dy, dx])


def ortho(ntf: Path, bands, transform, shape, crs, dem: Path, dem_fill: float,
          resampling=Resampling.cubic) -> np.ndarray:
    out = np.zeros((len(bands), *shape), np.float32)
    with rasterio.open(ntf) as src:
        for i, b in enumerate(bands):
            reproject(rasterio.band(src, b), out[i], rpcs=src.rpcs, src_nodata=0,
                      dst_transform=transform, dst_crs=crs, dst_nodata=0,
                      resampling=resampling, RPC_DEM=str(dem),
                      RPC_DEMINTERPOLATION="bilinear",
                      RPC_DEM_MISSING_VALUE=f"{dem_fill:.3f}", NUM_THREADS="ALL_CPUS")
    return out


def pansharpen(ms: np.ndarray, pan: np.ndarray, ratio: float) -> np.ndarray:
    """Intensity-ratio (Brovey-style) sharpening with PAN fitted to MS intensity."""
    inten = ms.mean(0)
    ok = (pan > 0) & (inten > 0)
    lo = ndimage.gaussian_filter(pan, ratio / 2)
    A = np.stack([inten[ok], np.ones(ok.sum())], 1)
    a, b = np.linalg.lstsq(A, lo[ok], rcond=None)[0]
    fitted = np.maximum(a * inten + b, 1.0)
    out = ms * (pan / fitted)[None]
    out[:, ~ok] = 0
    return out


def write(path: Path, arr: np.ndarray, transform, crs, nodata, dtype, desc=None):
    arr = arr if arr.ndim == 3 else arr[None]
    prof = dict(driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0],
                dtype=dtype, crs=crs, transform=transform, nodata=nodata,
                compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(path, "w", **prof) as d:
        d.write(arr.astype(dtype))
        for i, s in enumerate(desc or []):
            d.set_band_description(i + 1, s)


def find(root: Path, kind: str, sid: str) -> Path | None:
    hits = sorted((root / "WV3" / kind).glob(f"*{sid}-*.NTF"))
    return hits[0] if hits else None


def preview(path: Path, rgb_bgrn: np.ndarray, ndsm: np.ndarray):
    from PIL import Image

    rgb = rgb_bgrn[[2, 1, 0]].transpose(1, 2, 0)
    v = rgb[rgb > 0]
    lo, hi = np.percentile(v, (2, 98)) if v.size else (0, 1)
    rgb = np.clip((rgb - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
    h = np.nan_to_num(np.clip(ndsm / 20.0, 0, 1))
    hm = (np.stack([h, h ** 0.5, 1 - h], -1) * 255).astype(np.uint8)
    im = np.concatenate([rgb, hm], 1)
    Image.fromarray(im).save(path)


def prep_site(gt_path: Path, root: Path, out: Path, scenes, pad: int) -> dict:
    name = gt_path.stem
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    with rasterio.open(gt_path) as g:
        dsm = g.read(1).astype(np.float32)
        tr, crs, gnd = g.transform, g.crs, g.nodata
    dsm[(dsm == gnd) | (dsm < -1000)] = np.nan
    # lidar spikes (birds, multipath: Explorer has a 366 m pixel) -> NoData
    med = ndimage.median_filter(fill_nan(dsm), size=7)
    spikes = np.abs(dsm - med) > 25.0
    dsm[spikes] = np.nan
    valid = np.isfinite(dsm)
    gsd = abs(tr.a)
    H, W = dsm.shape
    dtm = dtm_from_dsm(dsm, gsd)
    ndsm = np.where(valid, np.maximum(dsm - dtm, 0), np.nan)
    for fn, a in (("gt_dsm.tif", dsm), ("gt_dtm.tif", np.where(valid, dtm, np.nan)),
                  ("gt_ndsm.tif", ndsm)):
        write(d / fn, np.nan_to_num(a, nan=ND), tr, crs, ND, "float32")

    big_tr = tr * Affine.translation(-pad, -pad)
    big_shape = (H + 2 * pad, W + 2 * pad)
    filled = fill_nan(dsm)
    dem_fill = float(np.nanmedian(dsm))
    dem = d / "rpc_dem.tif"
    write(dem, np.pad(filled, pad, mode="edge"), big_tr, crs, None, "float32")

    info: dict = {"site": name, "shape": [H, W], "gsd_m": gsd, "crs": str(crs),
                  "gt_valid_frac": float(valid.mean()), "gt_spikes_removed": int(spikes.sum()),
                  "ndsm_p50_p95_max_m": [float(np.nanpercentile(ndsm, q)) for q in (50, 95, 100)]}
    for sid in scenes:
        pan_p, ms_p = find(root, "PAN", sid), find(root, "MSI", sid)
        if not (pan_p and ms_p):
            continue
        pan_big = ortho(pan_p, [1], big_tr, big_shape, crs, dem, dem_fill)[0]
        dy, dx, ncc = best_shift(pan_big, filled, valid)
        pan = pan_big[dy:dy + H, dx:dx + W]
        cover = float((pan > 0).mean())
        if cover < 0.98:
            print(f"[{name}] {sid}: covers {cover:.1%} — next scene")
            continue
        ms_big = ortho(ms_p, MS_BGRN, big_tr, big_shape, crs, dem, dem_fill)
        my, mx, mncc = best_shift(ms_big.mean(0), pan, pan > 0)
        ms = ms_big[:, my:my + H, mx:mx + W]
        img = pansharpen(ms, pan, ratio=4.0)
        img[:, ~((pan > 0) & (ms.min(0) > 0))] = 0
        write(d / "image_bgrn.tif", np.clip(np.rint(img), 0, 65535), tr, crs, 0, "uint16",
              desc=["blue", "green", "red", "nir"])
        preview(d / "preview.png", img, ndsm)
        info.update(scene=sid, pan=pan_p.name, msi=ms_p.name, image_cover=cover,
                    pan_shift_px=[dy - pad, dx - pad], pan_ncc=round(ncc, 3),
                    ms_shift_px=[my - pad, mx - pad], ms_ncc=round(mncc, 3))
        break
    else:
        info["error"] = "no scene covers this site"
    dem.unlink(missing_ok=True)
    (d / "site.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info))
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sites", nargs="*", default=None, help="GT tile stems; default all")
    ap.add_argument("--scenes", nargs="*", default=SCENES)
    ap.add_argument("--pad", type=int, default=64, help="registration search, px each way")
    a = ap.parse_args()
    gts = sorted((a.root / "Challenge_Data_and_Software" / "Lidar_gt").glob("*.tif"))
    if a.sites:
        gts = [g for g in gts if g.stem in a.sites]
    a.out.mkdir(parents=True, exist_ok=True)
    res = [prep_site(g, a.root, a.out, a.scenes, a.pad) for g in gts]
    (a.out / "sites.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
