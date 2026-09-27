"""IARPA MVS3DM -> packed store `mvs3dm/{train,val,test}` (0.5 m, 512 px tiles).

    python tools/pack_mvs3dm.py mosaic --root ~/dw/data/mvs3dm
    python tools/pack_mvs3dm.py pack   --root ~/dw/data/mvs3dm --data_root ~/dw/data/stores \
        --fetch --delete_raw

**mosaic**  Rasterises all 40 LAZ strips (616 M points, unclassified) onto one
0.5 m UTM 21S grid: DSM = highest return per cell, despiked (|z - median5| >
25 m -> NoData).  The ground is built from the *lowest* return per cell, which
reaches the soil under tree crowns through the multi-return pulses: min-pooled
to 1 m, low outliers dropped, then a progressive morphological filter
(Zhang 2003) up to 101 m windows for the warehouses.  nDSM = DSM - DTM >= 0.
Writes `mosaic/{dsm,dtm,ndsm,dem_filled}.tif` and checks the DSM against the 8
official `Lidar_gt` tiles.

**pack**  For each WV3 scene (PAN + MSI NTF; `TRAIN_SCENES`, 31 scenes up to
26 deg off-nadir, fetched on demand with `--fetch`, which `--delete_raw`
removes again once packed):
orthorectify onto the mosaic grid with the RPCs over the lidar DSM (true
ortho), estimate one RPC bias shift per scene from gradient NCC against the DSM
on a few lidar-dense blocks, pansharpen, map to uint8 by the scene-wide
0.1/99.9 % percentiles per band (what `dwdata/scene_io.SceneSource` does at
inference), then cut the fixed 512 px grid:

* `test`   tiles touching one of the 8 benchmark GT sites (the viewer's test
           set; `eval_test.py --source mvs3dm --split test`),
* dropped  tiles within `--buffer_m` of a site that do not touch it,
* `val`    10 % of the remaining 2x2-tile blocks (seed 0), split by location,
           so no view of a val location is ever trained on,
* `train`  the rest.

Scan-line gaps of 1-2 cells are filled where their neighbours are flat
(`fill_gaps`).  A tile is kept when >= 50 % of it has both an image and a label.  Labels are
metres above ground *with trees* (lidar), no classes (`has_seg` false).
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.warp import Resampling
from rasterio.windows import Window
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.prep_mvs3dm import SCENES, best_shift, fill_nan, find, ortho  # noqa: E402

S3 = "s3://spacenet-dataset/Hosted-Datasets/MVS_dataset/WV3"
# every WV3 scene up to 26 deg mean off-nadir (PAN IMD; Cartosat-2S tilts to
# about +-26), most nadir first, minus the austral-winter ones (Jun-Aug:
# 15JUN12/18/19/24/30, 15JUL14/19, 15AUG27): leaf-off crowns vs lidar trees
TRAIN_SCENES = SCENES + [
    "15DEC18140533", "15SEP08140733", "15DEC18140522", "15MAR21135704", "15DEC19142039",
    "16JAN13141501", "15FEB11135123", "15MAR15140133", "15DEC18140544", "15DEC18140510",
    "15MAY04135349", "15FEB12140652", "15DEC18140554", "15MAR08134953", "15JAN23134652",
    "15OCT23141928", "15MAY05140810", "15MAR22141208", "15DEC18140455", "15FEB06141035",
    "15SEP01135603", "15SEP15141840"]

GSD = 0.5
TILE = 512
ND = -9999.0
EPSG = "EPSG:32721"


# ---------------------------------------------------------------------
# mosaic
# ---------------------------------------------------------------------
def _write(path: Path, a: np.ndarray, tr, nodata=ND, dtype="float32"):
    a = a if a.ndim == 3 else a[None]
    with rasterio.open(path, "w", driver="GTiff", height=a.shape[1], width=a.shape[2],
                       count=a.shape[0], dtype=dtype, crs=EPSG, transform=tr,
                       nodata=nodata, compress="deflate", predictor=2 if dtype != "float32" else 3,
                       tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER") as d:
        d.write(a.astype(dtype))


def _nanmin_pool(a: np.ndarray, k: int) -> np.ndarray:
    H, W = a.shape
    a = np.pad(a, ((0, -H % k), (0, -W % k)), constant_values=np.nan)
    a = a.reshape(a.shape[0] // k, k, a.shape[1] // k, k)
    with np.errstate(all="ignore"):
        return np.nanmin(np.nanmin(a, 3), 1)


def pmf(z: np.ndarray, cell_m: float, windows_m=(3, 9, 21, 41, 61, 101),
        slope: float = 0.05, dh0: float = 0.3, dh_max: float = 3.0) -> np.ndarray:
    """Progressive morphological filter; returns a smooth, gap-free ground."""
    f = fill_nan(z)
    ground = np.isfinite(z)
    for w_m in windows_m:
        w = max(3, int(w_m / cell_m) | 1)
        opened = ndimage.grey_opening(f, size=(w, w))
        ground &= (f - opened) <= min(dh_max, dh0 + slope * w * cell_m)
    g = fill_nan(np.where(ground, f, np.nan))
    return ndimage.gaussian_filter(g, 2.0).astype(np.float32), ground


def mosaic(root: Path) -> None:
    import laspy

    out = root / "mosaic"
    out.mkdir(exist_ok=True)
    fs = sorted(glob.glob(str(root / "Lidar/LAZ/*.laz")))
    b = []
    for f in fs:
        with laspy.open(f) as r:
            b.append((*r.header.mins[:2], *r.header.maxs[:2]))
    b = np.array(b)
    x0, y1 = np.floor(b[:, 0].min() / GSD) * GSD, np.ceil(b[:, 3].max() / GSD) * GSD
    W = int(np.ceil((b[:, 2].max() - x0) / GSD))
    H = int(np.ceil((y1 - b[:, 1].min()) / GSD))
    tr = Affine(GSD, 0, x0, 0, -GSD, y1)
    print(f"[mosaic] {len(fs)} strips -> {W} x {H} @ {GSD} m  origin ({x0}, {y1})", flush=True)
    zmax = np.full(H * W, -np.inf, np.float32)
    zmin = np.full(H * W, np.inf, np.float32)
    npts = 0
    for i, f in enumerate(fs):
        with laspy.open(f) as r:
            for p in r.chunk_iterator(5_000_000):
                x, y, z = np.asarray(p.x), np.asarray(p.y), np.asarray(p.z, np.float32)
                c = ((x - x0) / GSD).astype(np.int64)
                rr = ((y1 - y) / GSD).astype(np.int64)
                ok = (c >= 0) & (c < W) & (rr >= 0) & (rr < H)
                idx = rr[ok] * W + c[ok]
                np.maximum.at(zmax, idx, z[ok])
                np.minimum.at(zmin, idx, z[ok])
                npts += int(ok.sum())
        print(f"  {i + 1}/{len(fs)} {Path(f).name}  {npts / 1e6:.0f} M pts", flush=True)
    dsm = zmax.reshape(H, W)
    del zmax
    dsm[~np.isfinite(dsm)] = np.nan
    zmin = zmin.reshape(H, W)
    zmin[~np.isfinite(zmin)] = np.nan
    cover = np.isfinite(dsm)
    print(f"[mosaic] lidar covers {cover.mean():.1%} of the grid "
          f"({cover.sum() * GSD * GSD / 1e6:.1f} km^2)", flush=True)

    fill = float(np.nanmedian(dsm[::16, ::16]))
    med = ndimage.median_filter(np.where(cover, dsm, fill), size=5)
    spikes = cover & (np.abs(dsm - med) > 25.0)
    del med
    dsm[spikes] = np.nan
    print(f"[mosaic] DSM spikes removed: {int(spikes.sum())}", flush=True)
    del spikes

    # ground from the lowest returns, 1 m cells
    g1 = _nanmin_pool(zmin, 2)
    del zmin
    med1 = ndimage.median_filter(np.nan_to_num(g1, nan=fill), size=7)
    g1[g1 < med1 - 3.0] = np.nan                 # below-ground noise (the -36 m points)
    del med1
    dtm1, ground = pmf(g1, 1.0)
    print(f"[mosaic] ground cells: {ground[np.isfinite(g1)].mean():.1%} of occupied 1 m cells",
          flush=True)
    del ground
    # Windows to 101 m miss warehouses wider than that (their roofs become
    # "ground"); windows to 301 m shave the ~15 m barranca the town sits on.
    # Neither is right where they disagree, so those cells get no label.
    wide, _ = pmf(g1, 1.0, windows_m=(3, 9, 21, 41, 61, 101, 201, 301))
    doubt1 = ndimage.binary_dilation(np.abs(dtm1 - wide) > 2.5, iterations=5)
    del g1, wide
    dtm = ndimage.zoom(dtm1, 2, order=1)[:H, :W]
    doubt = np.repeat(np.repeat(doubt1, 2, 0), 2, 1)[:H, :W]
    del dtm1, doubt1
    np.minimum(dtm, np.where(np.isfinite(dsm), dsm, np.inf), out=dtm)
    ndsm = np.where(np.isfinite(dsm) & ~doubt, np.maximum(dsm - dtm, 0), np.nan).astype(np.float32)
    print(f"[mosaic] ground ambiguous (no label): {doubt[cover].mean():.2%} of lidar cells",
          flush=True)
    del doubt

    _write(out / "dsm.tif", np.nan_to_num(dsm, nan=ND), tr)
    _write(out / "dtm.tif", np.where(cover, dtm, ND), tr)
    _write(out / "ndsm.tif", np.nan_to_num(ndsm, nan=ND), tr)
    del ndsm, dtm
    # the RPC DEM must be gap-free: fill holes from a 4 m nearest-neighbour grid
    k = 8
    coarse = _nanmin_pool(dsm, k)
    coarse = fill_nan(np.where(np.isfinite(coarse), coarse, np.nan)) if np.isfinite(coarse).any() \
        else np.full_like(coarse, fill)
    up = np.repeat(np.repeat(coarse, k, 0), k, 1)[:H, :W]
    _write(out / "dem_filled.tif", np.where(np.isfinite(dsm), dsm, up), tr, nodata=None)
    del up, coarse

    # sanity: our DSM vs the benchmark's own 0.3 m GT tiles
    rep = {"grid": {"width": W, "height": H, "gsd_m": GSD, "transform": list(tr)[:6],
                    "crs": EPSG}, "points": npts, "lidar_cover_frac": float(cover.mean()),
           "vs_official_gt": {}}
    from rasterio.warp import reproject

    for g in sorted((root / "Challenge_Data_and_Software/Lidar_gt").glob("*.tif")):
        with rasterio.open(g) as s:
            ref = np.full((H, W), np.nan, np.float32)
            reproject(rasterio.band(s, 1), ref, src_nodata=s.nodata, dst_transform=tr,
                      dst_crs=EPSG, dst_nodata=np.nan, resampling=Resampling.average)
        m = np.isfinite(ref) & np.isfinite(dsm)
        d = dsm[m] - ref[m]
        rep["vs_official_gt"][g.stem] = {"n": int(m.sum()), "median_m": float(np.median(d)),
                                         "mad_m": float(np.median(np.abs(d - np.median(d))))}
        print(f"[mosaic] {g.stem:22} ours - official: median {np.median(d):+.2f} m, "
              f"MAD {rep['vs_official_gt'][g.stem]['mad_m']:.2f} m", flush=True)
    (out / "mosaic.json").write_text(json.dumps(rep, indent=2))


# ---------------------------------------------------------------------
# pack
# ---------------------------------------------------------------------
def _site_boxes(root: Path) -> dict:
    out = {}
    for g in sorted((root / "Challenge_Data_and_Software/Lidar_gt").glob("*.tif")):
        with rasterio.open(g) as s:
            out[g.stem] = tuple(s.bounds)
    return out


def _tile_split(tr, H: int, W: int, sites: dict, buffer_m: float, val_frac: float):
    """{(r, c): 'train' | 'val' | 'test' | site-name} over the fixed tile grid."""
    nr, nc = H // TILE, W // TILE
    tm = TILE * GSD
    blocks = sorted({(r // 2, c // 2) for r in range(nr) for c in range(nc)})
    rng = np.random.default_rng(0)
    val_blocks = {blocks[i] for i in rng.permutation(len(blocks))[:int(round(len(blocks) * val_frac))]}
    split, site_of = {}, {}
    for r in range(nr):
        for c in range(nc):
            x0, y1 = tr.c + c * tm, tr.f - r * tm
            x1, y0 = x0 + tm, y1 - tm
            hit = [n for n, (a, b, cc, d) in sites.items() if x0 < cc and x1 > a and y0 < d and y1 > b]
            near = any(x0 < cc + buffer_m and x1 > a - buffer_m and y0 < d + buffer_m
                       and y1 > b - buffer_m for (a, b, cc, d) in sites.values())
            if hit:
                split[(r, c)], site_of[(r, c)] = "test", hit[0]
            elif not near:
                split[(r, c)] = "val" if (r // 2, c // 2) in val_blocks else "train"
    return split, site_of


def _scene_shift(pan_p: Path, ms_p: Path, dem: Path, dsm: np.ndarray, tr, dem_fill: float,
                 pad: int = 24, n_blocks: int = 4, blk: int = 1024):
    """Median RPC bias (PAN vs DSM, MS vs PAN) over the lidar-densest blocks."""
    H, W = dsm.shape
    dens = np.isfinite(dsm[:H // blk * blk, :W // blk * blk]).reshape(
        H // blk, blk, W // blk, blk).mean((1, 3))
    order = np.argsort(dens.ravel())[::-1][:n_blocks]
    ps, ms, nccs = [], [], []
    for o in order:
        r, c = np.unravel_index(o, dens.shape)
        r0, c0 = r * blk, c * blk
        ref = dsm[r0:r0 + blk, c0:c0 + blk]
        valid = np.isfinite(ref)
        btr = tr * Affine.translation(c0 - pad, r0 - pad)
        shape = (blk + 2 * pad, blk + 2 * pad)
        pan_big = ortho(pan_p, [1], btr, shape, EPSG, dem, dem_fill)[0]
        if (pan_big > 0).mean() < 0.9:
            continue
        dy, dx, ncc = best_shift(pan_big, fill_nan(ref), valid)
        pan = pan_big[dy:dy + blk, dx:dx + blk]
        ms_big = ortho(ms_p, ms_bands(ms_p), btr, shape, EPSG, dem, dem_fill)
        my, mx, _ = best_shift(ms_big.mean(0), pan, pan > 0)
        ps.append((dy - pad, dx - pad))
        ms.append((my - pad, mx - pad))
        nccs.append(ncc)
    if not ps:
        return None
    ps, ms = np.array(ps), np.array(ms)
    return {"pan_shift_px": np.median(ps, 0).round().astype(int).tolist(),
            "ms_shift_px": np.median(ms, 0).round().astype(int).tolist(),
            "pan_shift_spread_px": float(np.abs(ps - np.median(ps, 0)).max()),
            "ncc": [round(float(x), 3) for x in nccs]}


def ms_bands(path: Path) -> tuple[int, ...]:
    """1-based B, G, R, NIR of an MSI NTF, picked by centre wavelength (NITF
    ISUBCAT, nm).  WV3 ships 8-band products (2, 3, 5, 7) but the 2014 P005
    order is a 4-band B, G, R, N one (1, 2, 3, 4)."""
    want = (480.0, 545.0, 660.0, 830.0)
    with rasterio.open(path) as d:
        try:
            nm = [float(d.tags(i).get("NITF_ISUBCAT", "nan")) for i in range(1, d.count + 1)]
        except ValueError:
            nm = []
        count = d.count
    if nm and np.all(np.isfinite(nm)):
        return tuple(int(np.argmin([abs(x - w) for x in nm])) + 1 for w in want)
    return (2, 3, 5, 7) if count >= 8 else (1, 2, 3, 4)


def _pansharpen(ms: np.ndarray, pan: np.ndarray, ratio: float) -> np.ndarray:
    inten = ms.mean(0)
    ok = (pan > 0) & (ms.min(0) > 0)
    if ok.sum() < 1000:
        return np.zeros_like(ms)
    lo = ndimage.gaussian_filter(pan, ratio / 2)
    a, b = np.linalg.lstsq(np.stack([inten[ok], np.ones(ok.sum())], 1), lo[ok], rcond=None)[0]
    out = ms * (pan / np.maximum(a * inten + b, 1.0))[None]
    out[:, ~ok] = 0
    return out


def ortho_scene(sid: str, root: Path, mos: Path, tmp: Path, blk: int = 2048) -> dict | None:
    """Scene -> `tmp` 3-band UInt16 R, G, B on the mosaic grid + percentile bounds."""
    pan_p, ms_p = find(root, "PAN", sid), find(root, "MSI", sid)
    if not (pan_p and ms_p):
        return None
    dem = mos / "dem_filled.tif"
    with rasterio.open(mos / "dsm.tif") as d:
        tr, H, W = d.transform, d.height, d.width
        dsm = d.read(1)
    dsm[dsm == ND] = np.nan
    dem_fill = float(np.nanmedian(dsm[::16, ::16]))     # RPC DEM beyond the grid
    sh = _scene_shift(pan_p, ms_p, dem, dsm, tr, dem_fill)
    if sh is None:
        print(f"[{sid}] no lidar block inside the scene — skipped", flush=True)
        return None
    print(f"[{sid}] shifts {sh}", flush=True)
    (pdy, pdx), (mdy, mdx) = sh["pan_shift_px"], sh["ms_shift_px"]
    hist = np.zeros((3, 65536), np.int64)
    prof = dict(driver="GTiff", height=H, width=W, count=3, dtype="uint16", crs=EPSG,
                transform=tr, nodata=0, compress="deflate", predictor=2, tiled=True,
                blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
    n_done = 0
    with rasterio.open(tmp, "w", **prof) as out:
        for r0 in range(0, H, blk):
            for c0 in range(0, W, blk):
                h, w = min(blk, H - r0), min(blk, W - c0)
                if not np.isfinite(dsm[r0:r0 + h, c0:c0 + w]).any():
                    continue
                ptr = tr * Affine.translation(c0 + pdx, r0 + pdy)
                pan = ortho(pan_p, [1], ptr, (h, w), EPSG, dem, dem_fill)[0]
                if not (pan > 0).any():
                    continue
                mtr = tr * Affine.translation(c0 + pdx + mdx, r0 + pdy + mdy)
                ms = ortho(ms_p, ms_bands(ms_p), mtr, (h, w), EPSG, dem, dem_fill)
                sharp = _pansharpen(ms, pan, ratio=1.24 / GSD)
                rgb = np.clip(np.rint(sharp[[2, 1, 0]]), 0, 65535).astype(np.uint16)
                out.write(rgb, window=Window(c0, r0, w, h))
                lab = np.isfinite(dsm[r0:r0 + h, c0:c0 + w]) & (rgb.min(0) > 0)
                for i in range(3):
                    hist[i] += np.bincount(rgb[i][lab], minlength=65536)
                n_done += 1
    if not hist.sum():
        tmp.unlink(missing_ok=True)
        return None
    cdf = np.cumsum(hist, 1) / hist.sum(1, keepdims=True)
    lo = [int(np.searchsorted(cdf[i], 0.001)) for i in range(3)]
    hi = [int(np.searchsorted(cdf[i], 0.999)) for i in range(3)]
    print(f"[{sid}] {n_done} blocks, uint8 bounds lo {lo} hi {hi}", flush=True)
    return {**sh, "scene": sid, "pan": pan_p.name, "msi": ms_p.name, "lo": lo, "hi": hi,
            "blocks": n_done}


def fill_gaps(h: np.ndarray, valid: np.ndarray, iters: int = 2, flat_m: float = 1.0):
    """Fill 1-2 cell scan-line gaps, but only where the valid neighbours are flat
    (spread < `flat_m`), so a gap beside a wall never takes the roof's height."""
    for _ in range(iters):
        hz = np.where(valid, h, 0).astype(np.float32)
        n = ndimage.uniform_filter(valid.astype(np.float32), 3) * 9
        s = ndimage.uniform_filter(hz, 3) * 9
        mx = ndimage.maximum_filter(np.where(valid, h, -1e9), 3)
        mn = ndimage.minimum_filter(np.where(valid, h, 1e9), 3)
        ok = ~valid & (n >= 3.5) & (mx - mn < flat_m)
        if not ok.any():
            break
        h = np.where(ok, s / np.maximum(n, 1), h)
        valid = valid | ok
    return h, valid


def fetch_scene(root: Path, sid: str) -> list[Path]:
    """PAN + MSI NTF of one scene from the public bucket, if not already here.
    Returns the files it downloaded."""
    import subprocess

    got = []
    for b in ("PAN", "MSI"):
        if find(root, b, sid) is None:
            subprocess.run(["aws", "s3", "cp", "--no-sign-request", "--only-show-errors",
                            "--recursive", f"{S3}/{b}", str(root / "WV3" / b),
                            "--exclude", "*", "--include", f"*{sid}-*.NTF"], check=True)
            got += list((root / "WV3" / b).glob(f"*{sid}-*.NTF"))
    return got


def pack(root: Path, data_root: Path, scenes, buffer_m: float, val_frac: float,
         min_valid: float, max_h: float, fetch: bool = False,
         delete_raw: bool = False) -> None:
    from dwdata.packed import ShardWriter

    mos = root / "mosaic"
    with rasterio.open(mos / "ndsm.tif") as d:
        tr, H, W = d.transform, d.height, d.width
    split, site_of = _tile_split(tr, H, W, _site_boxes(root), buffer_m, val_frac)
    print(f"[pack] tile grid {H // TILE} x {W // TILE}: "
          + ", ".join(f"{s} {sum(v == s for v in split.values())}" for s in ("train", "val", "test")),
          flush=True)
    writers = {s: ShardWriter(data_root / "mvs3dm" / s, tile_px=TILE, gsd_m=GSD,
                              shard_tiles=128, has_seg=False) for s in ("train", "val", "test")}
    counts = {s: 0 for s in writers}
    info = {"scenes": [], "gsd_m": GSD, "tile_px": TILE, "buffer_m": buffer_m,
            "val_frac": val_frac, "min_valid": min_valid, "max_valid_height_m": max_h}
    for sid in scenes:
        tmp = root / f"_ortho_{sid}.tif"
        fetched = []
        try:
            fetched = fetch_scene(root, sid) if fetch else []
            si = ortho_scene(sid, root, mos, tmp)
        except Exception as e:  # noqa: BLE001  one odd product must not end a 5 h run
            import traceback

            traceback.print_exc()
            print(f"[{sid}] FAILED ({type(e).__name__}: {e}) — skipped", flush=True)
            info.setdefault("failed", []).append({"scene": sid, "error": f"{type(e).__name__}: {e}"})
            si = None
        if delete_raw:                          # only what this run downloaded
            for f in fetched:
                f.unlink(missing_ok=True)
        if si is None:
            tmp.unlink(missing_ok=True)
            continue
        lo, hi = np.array(si["lo"], np.float32), np.array(si["hi"], np.float32)
        kept = {s: 0 for s in writers}
        with rasterio.open(tmp) as im, rasterio.open(mos / "ndsm.tif") as nd:
            for (r, c), s in sorted(split.items()):
                win = Window(c * TILE, r * TILE, TILE, TILE)
                raw = im.read(window=win)
                if not raw.any():
                    continue
                h = nd.read(1, window=win)
                img_ok = raw.min(0) > 0
                h, lab = fill_gaps(h, h != ND)
                valid = img_ok & lab & (h <= max_h)
                if valid.mean() < min_valid:
                    continue
                u8 = np.clip((raw.astype(np.float32) - lo[:, None, None])
                             / np.maximum(hi - lo, 1)[:, None, None] * 255.0, 0, 255)
                rgb = np.where(img_ok[None], np.rint(u8), 0).astype(np.uint8).transpose(1, 2, 0)
                stem = f"{sid}_r{r:03d}_c{c:03d}" + (f"_{site_of[(r, c)]}" if s == "test" else "")
                writers[s].add(stem, rgb, np.where(valid, h, 0.0).astype(np.float32), None, valid)
                kept[s] += 1
        tmp.unlink(missing_ok=True)
        for s in kept:
            counts[s] += kept[s]
        print(f"[{sid}] kept {kept}  running {counts}", flush=True)
        info["scenes"].append({**si, "kept": kept})
    for s, w in writers.items():
        idx = w.finalise()
        print(f"[pack] mvs3dm/{s}: {idx['n']} tiles", flush=True)
    info["counts"] = counts
    info["test_sites"] = sorted(set(site_of.values()))
    (data_root / "mvs3dm" / "build_info.json").write_text(json.dumps(info, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["mosaic", "pack"])
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--data_root", type=Path, default=None)
    ap.add_argument("--scenes", nargs="*", default=TRAIN_SCENES)
    ap.add_argument("--fetch", action="store_true", help="download missing scenes from S3")
    ap.add_argument("--delete_raw", action="store_true",
                    help="delete the NTFs --fetch downloaded once their scene is packed")
    ap.add_argument("--buffer_m", type=float, default=100.0)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--min_valid", type=float, default=0.5)
    ap.add_argument("--max_valid_height_m", type=float, default=150.0)
    a = ap.parse_args()
    root = a.root.expanduser()
    if a.stage == "mosaic":
        mosaic(root)
    else:
        pack(root, (a.data_root or root / "stores").expanduser(), a.scenes, a.buffer_m,
             a.val_frac, a.min_valid, a.max_valid_height_m, a.fetch, a.delete_raw)


if __name__ == "__main__":
    main()
