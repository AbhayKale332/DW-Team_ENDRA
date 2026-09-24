"""nDSM + coarse DEM -> absolute DSM, without double-counting buildings.

**The mistake this file exists to avoid.**  Copernicus GLO-30 and SRTM are
*surface* models: their 30 m cells already contain the buildings and the canopy.
v3's `--dem` path added the DEM to the predicted nDSM straight, which counts every
structure twice — a 40 m tower over 12 m terrain comes out at ~80 m, and it does
so silently because the result still looks like a plausible elevation raster.

The fix is the one the plan specifies: **estimate the bare-earth DTM by fitting a
smooth surface through the DEM values at ground pixels only**, then add the nDSM
to that.

    ground mask  =  (Head C says ground/road/water)  AND  (predicted nDSM < tol)
    DTM(x, y)    =  smooth surface fitted to DEM[ground]
    DSM          =  DTM + nDSM

Two independent conditions on the mask, because either alone fails: the semantic
head is auxiliary and its class ids are unverified (README §1.6), while the height
threshold alone would call a flat rooftop "ground" whenever the terrain beneath it
happens to be low. Requiring both makes a wrong class id harmless.

The surface fit is a low-order 2-D polynomial by least squares on the ground
samples, with a residual correction blurred over a large window. A polynomial
alone cannot follow a valley; a blur alone leaks building height back in through
the gaps. Together they follow real terrain and stay bare-earth under buildings.

`refine_with_gcps` is the second calibration route the problem statement allows:
a handful of surveyed control points, fitted robustly (RANSAC) for scale and
offset. Use it when the panel supplies GCPs; it needs no DEM at all.

**v5: `dem_anchored`, the mode that matches how we are scored.**  The judges
compare the absolute DSM against SRTM / Copernicus 30 m (host FAQ).  Those are
surface models whose 30 m cells hold part of every building and canopy, so
`DTM + nDSM` adds structure height on top of a reference that already contains
some of it; and `fit_dtm` degrades to a flat median when a scene has < 32
ground pixels (closed canopy on a hillside — the exact "forested/hilly" case).
`dem_anchored` needs no ground pixels at all:

    A  = valid-pixel block mean onto ~30 m cells (k = 30 m / GSD pixels)
    U  = bilinear interpolation from cell centres back to pixels
    D  = U(X) + detail_gain * (nDSM - U(A(nDSM)))
    X  <- X + (DEM_c - A(D))            (Tobler 1979; X starts at DEM_c)

so every 30 m cell of the output averages to the DEM exactly, while the model
supplies everything finer than a cell.  If the judges aggregate to 30 m the
detail is free; if they compare per pixel it costs `detail_gain` x the nDSM's
high-pass energy — `--detail_gain` exposes that trade until the hosts say which
(`CompetitionContext/Host_Questions_Draft.md`).  The terrain layer is
`U(X - A(nDSM))`: the anchored surface minus the cell-mean structure height.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from config import GROUND_LIKE_IDS


@dataclass
class Calibration:
    dsm_m: np.ndarray                     # absolute DSM = DTM + nDSM
    dtm_m: np.ndarray                     # the fitted bare-earth surface
    ground_mask: np.ndarray               # what the fit was trained on
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------
def ground_mask(ndsm_m: np.ndarray, seg: np.ndarray | None = None,
                *, max_height_m: float = 1.5,
                ground_ids: tuple = GROUND_LIKE_IDS,
                min_frac: float = 0.02) -> np.ndarray:
    """Pixels we are willing to believe are bare earth.

    Falls back to the height criterion alone when the semantic agreement is too
    small to fit on — better a weaker mask than a fit on 40 pixels.
    """
    low = np.isfinite(ndsm_m) & (ndsm_m < max_height_m)
    if seg is None:
        return low
    sem = np.zeros_like(low)
    for i in ground_ids:
        sem |= (seg == i)
    both = low & sem
    if both.mean() >= min_frac:
        return both
    print(f"[calib] semantic ground ∩ low-nDSM is only {both.mean() * 100:.2f}% of the "
          f"scene — falling back to the height criterion alone "
          f"({low.mean() * 100:.1f}%)")
    return low


def _poly_design(y: np.ndarray, x: np.ndarray, order: int) -> np.ndarray:
    cols = [np.ones_like(x)]
    for o in range(1, order + 1):
        for i in range(o + 1):
            cols.append((x ** (o - i)) * (y ** i))
    return np.stack(cols, axis=-1)


def fit_dtm(dem_m: np.ndarray, mask: np.ndarray, *, order: int = 2,
            residual_window: int = 64, max_samples: int = 200_000,
            robust_iters: int = 2) -> tuple[np.ndarray, dict]:
    """Smooth bare-earth surface through `dem_m` at `mask` pixels.

    Stage 1 is a robust low-order polynomial (the regional trend); stage 2 adds a
    heavily smoothed residual so real valleys and ridges survive.  The residual is
    computed *only* from masked pixels and then filled outward, so a building's
    DEM value never enters it.
    """
    H, W = dem_m.shape
    yy, xx = np.mgrid[0:H, 0:W]
    ny = (yy / max(H - 1, 1)).astype(np.float32)
    nx = (xx / max(W - 1, 1)).astype(np.float32)

    sel = mask & np.isfinite(dem_m)
    n = int(sel.sum())
    if n < 32:
        med = float(np.nanmedian(dem_m)) if np.isfinite(dem_m).any() else 0.0
        return np.full((H, W), med, np.float32), {
            "method": "median", "n_ground_px": n,
            "note": "too few ground pixels to fit a surface"}

    sy, sx, sz = ny[sel], nx[sel], dem_m[sel].astype(np.float64)
    if n > max_samples:
        keep = np.random.default_rng(0).choice(n, max_samples, replace=False)
        sy, sx, sz = sy[keep], sx[keep], sz[keep]

    w = np.ones_like(sz)
    coef = None
    for _ in range(max(1, robust_iters)):
        A = _poly_design(sy, sx, order)
        Aw = A * w[:, None]
        coef, *_ = np.linalg.lstsq(Aw, sz * w, rcond=None)
        r = sz - A @ coef
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        # Tukey-style down-weighting: a chunk of canopy that slipped into the mask
        # should not tilt the whole regional trend
        w = np.clip(1.0 - (r / (4.0 * s)) ** 2, 0.0, 1.0) ** 2

    trend = (_poly_design(ny.ravel(), nx.ravel(), order) @ coef).reshape(H, W)

    resid = np.where(sel, dem_m - trend, np.nan).astype(np.float32)
    resid_s = _masked_smooth(resid, residual_window)
    dtm = (trend + np.nan_to_num(resid_s, nan=0.0)).astype(np.float32)
    return dtm, {
        "method": f"poly{order}+smoothed_residual",
        "n_ground_px": n, "ground_frac": float(sel.mean()),
        "residual_window_px": residual_window,
        "trend_range_m": [float(trend.min()), float(trend.max())],
    }


def _masked_smooth(a: np.ndarray, window: int) -> np.ndarray:
    """Gaussian smoothing that ignores NaNs (normalised convolution)."""
    m = np.isfinite(a).astype(np.float32)
    v = np.nan_to_num(a, nan=0.0).astype(np.float32)
    sigma = max(1.0, window / 3.0)
    try:
        from scipy.ndimage import gaussian_filter

        num = gaussian_filter(v * m, sigma=sigma, mode="nearest")
        den = gaussian_filter(m, sigma=sigma, mode="nearest")
    except Exception:  # noqa: BLE001
        from eval.landscape import _box_blur

        k = int(window) | 1
        num, den = _box_blur(v * m, k), _box_blur(m, k)
    out = num / np.maximum(den, 1e-4)
    out[den < 1e-3] = np.nan
    return out


# ---------------------------------------------------------------------
def calibrate(ndsm_m: np.ndarray, dem_m: np.ndarray, seg: np.ndarray | None = None,
              *, mode: str = "dtm_plus_ndsm", detail_gain: float = 1.0,
              gsd_m: float = 0.5, anchor_cell_m: float = 30.0, tobler_iters: int = 20,
              max_ground_height_m: float = 1.5, poly_order: int = 2,
              residual_window: int = 64) -> Calibration:
    """The whole decomposition, in one call.  `mode`: "dtm_plus_ndsm" (v4) or
    "dem_anchored" (v5 default for the judged product — see the module doc)."""
    from geo.dem import fill_nan

    if mode == "dem_anchored":
        return calibrate_anchored(ndsm_m, dem_m, detail_gain=detail_gain, gsd_m=gsd_m,
                                  anchor_cell_m=anchor_cell_m, iters=tobler_iters)
    if mode != "dtm_plus_ndsm":
        raise ValueError(f"unknown calibration mode {mode!r}")
    nd = np.nan_to_num(ndsm_m, nan=0.0)
    dem = fill_nan(dem_m)
    mask = ground_mask(nd, seg, max_height_m=max_ground_height_m)
    dtm, info = fit_dtm(dem, mask, order=poly_order, residual_window=residual_window)
    dsm = (dtm + np.clip(nd, 0.0, None)).astype(np.float32)
    dsm[~np.isfinite(ndsm_m)] = np.nan

    # How much building height was in the raw DEM — i.e. what we just avoided
    # counting twice.  Reported so the effect is visible, not asserted.
    struct = nd > 5.0
    over = float(np.nanmean(dem[struct] - dtm[struct])) if struct.any() else 0.0
    info.update({
        "mode": "dtm_plus_ndsm",
        "dem_minus_dtm_over_structures_m": over,
        "double_count_avoided_m": over,
        "dtm_range_m": [float(dtm.min()), float(dtm.max())],
        "dsm_range_m": [float(np.nanmin(dsm)), float(np.nanmax(dsm))],
        "ground_mask_frac": float(mask.mean()),
    })
    print(f"[calib] DTM {dtm.min():.1f}..{dtm.max():.1f} m from "
          f"{info['ground_mask_frac'] * 100:.1f}% ground pixels; the raw DEM sat "
          f"{over:+.2f} m above it over structures (that is the double-count avoided)")
    return Calibration(dsm, dtm, mask, info)


# ---------------------------------------------------------------------
# v5: DEM-anchored detail fusion
# ---------------------------------------------------------------------
def block_mean(a: np.ndarray, k: int, valid: np.ndarray | None = None) -> np.ndarray:
    """Mean over k x k blocks (edge blocks partial), NaN where a block is empty."""
    H, W = a.shape
    hb, wb = -(-H // k), -(-W // k)
    v = np.isfinite(a) if valid is None else (valid & np.isfinite(a))
    x = np.where(v, a, 0.0).astype(np.float64)
    pad = ((0, hb * k - H), (0, wb * k - W))
    xs = np.pad(x, pad).reshape(hb, k, wb, k).sum((1, 3))
    ns = np.pad(v.astype(np.float64), pad).reshape(hb, k, wb, k).sum((1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ns > 0, xs / np.maximum(ns, 1), np.nan)


def _lin_weights(n_out: int, k: float, n_in: int, o0: int = 0):
    """Indices + weights to interpolate cell-centred samples to pixel centres.

    The outer half-cell is *extrapolated* linearly from the two outermost
    centres rather than clamped: clamping makes a border cell's block mean of
    U(X) differ from X even for a plane, and the Tobler loop then pushes a
    ripple inwards from every scene edge (1.5 m on a 200 m slope in the test).
    """
    t = (np.arange(o0, o0 + n_out, dtype=np.float64) + 0.5) / k - 0.5
    if n_in == 1:
        z = np.zeros(n_out, np.int64)
        return z, z, np.zeros(n_out, np.float32)
    i0 = np.clip(np.floor(t), 0, n_in - 2).astype(np.int64)
    return i0, i0 + 1, (t - i0).astype(np.float32)


def upsample_rows(c: np.ndarray, k: float, r0: int, r1: int, W: int) -> np.ndarray:
    """Bilinear U of a cell grid `c` onto pixel rows [r0, r1) of a W-wide grid.

    Separable, so a row band of a 20k-wide scene costs two small gathers.
    """
    i0, i1, wy = _lin_weights(r1 - r0, k, c.shape[0], r0)
    rows = (1 - wy)[:, None] * c[i0] + wy[:, None] * c[i1]
    j0, j1, wx = _lin_weights(W, k, c.shape[1])
    return ((1 - wx)[None, :] * rows[:, j0] + wx[None, :] * rows[:, j1]).astype(np.float32)


def _fill_cells(c: np.ndarray) -> np.ndarray:
    from geo.dem import fill_nan

    return fill_nan(c) if np.isfinite(c).any() else np.zeros_like(c, np.float32)


class AnchoredFusion:
    """The cell-level half of `dem_anchored`; the pixel half is `rows()`.

    Everything that needs the whole scene lives on the ~30 m cell grid (a
    12 km scene is 400 x 400 cells), so the same object serves an in-memory
    array and a 20k x 20k windowed product: `calibrate_windowed` streams the
    nDSM through `rows()` band by band.

    `A(U(X))` is needed inside the Tobler loop.  Bilinear U in cell units is
    the same function whatever k is, so it is evaluated on a virtual grid of
    `s = min(k, 8)` pixels per cell — exact for whole cells, ~free for any k.
    """

    def __init__(self, dem_cells: np.ndarray, ndsm_cells: np.ndarray, k: float,
                 *, detail_gain: float = 1.0, iters: int = 20):
        self.k = float(k)
        self.lam = float(detail_gain)
        self.dem_c = _fill_cells(np.asarray(dem_cells, np.float64))
        self.m_c = _fill_cells(np.asarray(ndsm_cells, np.float64))
        s = int(max(1, min(round(self.k), 8)))
        hb, wb = self.dem_c.shape

        def AU(c):
            return block_mean(upsample_rows(c, s, 0, hb * s, wb * s), s)

        # A(D) = A(U(X)) + lam * (M - A(U(M)))   (A(nDSM) = M by definition)
        detail_mean = self.lam * (self.m_c - AU(self.m_c))
        x = self.dem_c.copy()
        res = []
        for _ in range(max(0, int(iters))):
            r = self.dem_c - (AU(x) + detail_mean)
            x += r
            res.append(float(np.sqrt(np.mean(r ** 2))))
        self.x_c = x
        self.info = {"mode": "dem_anchored", "detail_gain": self.lam,
                     "cell_px": self.k, "cells": list(self.dem_c.shape),
                     "tobler_residual_rms_m": res}

    def rows(self, ndsm_rows: np.ndarray, r0: int) -> tuple[np.ndarray, np.ndarray]:
        """(DSM, terrain) for nDSM rows starting at source row `r0`."""
        r1 = r0 + ndsm_rows.shape[0]
        W = ndsm_rows.shape[1]
        ux = upsample_rows(self.x_c, self.k, r0, r1, W)
        um = upsample_rows(self.m_c, self.k, r0, r1, W)
        dsm = ux + self.lam * (ndsm_rows - um)
        dtm = ux - um
        return dsm.astype(np.float32), dtm.astype(np.float32)


def anchor_cell_px(gsd_m: float, anchor_cell_m: float = 30.0) -> int:
    return max(1, int(round(anchor_cell_m / max(gsd_m, 1e-3))))


def calibrate_anchored(ndsm_m: np.ndarray, dem_m: np.ndarray, *, detail_gain: float = 1.0,
                       gsd_m: float = 0.5, anchor_cell_m: float = 30.0,
                       iters: int = 20) -> Calibration:
    """`dem_anchored` on in-memory arrays (the DEM already on the nDSM grid)."""
    k = anchor_cell_px(gsd_m, anchor_cell_m)
    valid = np.isfinite(ndsm_m)
    dem_c = block_mean(np.asarray(dem_m, np.float64), k, valid)
    m_c = block_mean(np.asarray(ndsm_m, np.float64), k, valid)
    fus = AnchoredFusion(dem_c, m_c, k, detail_gain=detail_gain, iters=iters)
    dsm, dtm = fus.rows(np.nan_to_num(ndsm_m, nan=0.0), 0)
    dsm[~valid] = np.nan
    a = block_mean(dsm, k, valid)
    fin = np.isfinite(a) & np.isfinite(dem_c)
    fus.info.update({
        "cell_mean_error_rms_m": float(np.sqrt(np.mean((a[fin] - dem_c[fin]) ** 2)))
        if fin.any() else None,
        "dsm_range_m": [float(np.nanmin(dsm)), float(np.nanmax(dsm))] if valid.any() else None,
    })
    print(f"[calib] dem_anchored: {k} px cells, detail_gain {detail_gain}, cell-mean "
          f"error vs DEM {fus.info['cell_mean_error_rms_m']:.3g} m")
    return Calibration(dsm, dtm, np.zeros_like(valid), fus.info)


def calibrate_windowed(ndsm_path: str, dem_cells: np.ndarray, ndsm_cells: np.ndarray,
                       k: int, out_dir, *, detail_gain: float = 1.0, iters: int = 20,
                       band_rows: int = 2048) -> dict:
    """Stream `ndsm_m.tif` -> `dsm_m.tif` + `dtm_m.tif` with `AnchoredFusion`.

    `dem_cells` is the DEM on the k-pixel cell grid (fetch it with the scene
    transform scaled by k — `geo.dem.fetch_dem` on the coarse grid), already in
    the target vertical datum; `ndsm_cells` comes from
    `infer.engine.predict_scene_windowed(block_px=k)`.
    """
    import rasterio
    from pathlib import Path
    from rasterio.windows import Window

    fus = AnchoredFusion(dem_cells, ndsm_cells, k, detail_gain=detail_gain, iters=iters)
    out_dir = Path(out_dir)
    paths = {"dsm": out_dir / "dsm_m.tif", "dtm": out_dir / "dtm_m.tif"}
    with rasterio.open(ndsm_path) as src:
        prof = src.profile.copy()
        H, W = src.height, src.width
        with rasterio.open(paths["dsm"], "w", **prof) as dd, \
                rasterio.open(paths["dtm"], "w", **prof) as dt:
            for r0 in range(0, H, band_rows):
                r1 = min(H, r0 + band_rows)
                w = Window(0, r0, W, r1 - r0)
                nd = src.read(1, window=w)
                fin = np.isfinite(nd)
                dsm, dtm = fus.rows(np.where(fin, nd, 0.0), r0)
                dsm[~fin] = np.nan
                dtm[~fin] = np.nan
                dd.write(dsm, 1, window=w)
                dt.write(dtm, 1, window=w)
    return {"dsm_path": str(paths["dsm"]), "dtm_path": str(paths["dtm"]), **fus.info}


# ---------------------------------------------------------------------
def refine_with_gcps(height_m: np.ndarray, gcps, *, iters: int = 200,
                     inlier_m: float = 1.0, seed: int = 0) -> tuple[np.ndarray, dict]:
    """RANSAC scale+offset from surveyed control points.

    `gcps` is an iterable of (row, col, elevation_m).  Solves
    `elev ≈ scale * height[row, col] + offset` and keeps the consensus with the
    most inliers, so one mis-keyed point cannot drag the whole raster.
    """
    pts = [(int(r), int(c), float(z)) for r, c, z in gcps]
    pts = [(r, c, z) for r, c, z in pts
           if 0 <= r < height_m.shape[0] and 0 <= c < height_m.shape[1]]
    if len(pts) < 2:
        return height_m, {"gcp": "need >= 2 in-bounds control points",
                          "n_gcps": len(pts)}
    h = np.array([height_m[r, c] for r, c, _ in pts], np.float64)
    z = np.array([p[2] for p in pts], np.float64)

    rng = np.random.default_rng(seed)
    best = (0, 1.0, float(np.median(z - h)))
    for _ in range(iters):
        i, j = rng.choice(len(pts), 2, replace=False)
        if abs(h[i] - h[j]) < 1e-6:
            continue
        s = (z[i] - z[j]) / (h[i] - h[j])
        o = z[i] - s * h[i]
        inl = int((np.abs(s * h + o - z) <= inlier_m).sum())
        if inl > best[0]:
            best = (inl, float(s), float(o))
    n_inl, s, o = best
    if n_inl >= 2:                                   # least-squares on the consensus
        keep = np.abs(s * h + o - z) <= inlier_m
        A = np.stack([h[keep], np.ones(keep.sum())], 1)
        sol, *_ = np.linalg.lstsq(A, z[keep], rcond=None)
        s, o = float(sol[0]), float(sol[1])
    out = (s * height_m + o).astype(np.float32)
    resid = s * h + o - z
    inl = np.abs(resid) <= inlier_m
    return out, {
        "gcp": "ransac_scale_offset", "n_gcps": len(pts), "inliers": int(inl.sum()),
        "scale": s, "offset_m": o,
        # both numbers, because they answer different questions: the inlier RMSE
        # is the fit quality, the all-points RMSE tells you a point was rejected
        "rmse_inliers_m": float(np.sqrt((resid[inl] ** 2).mean())) if inl.any() else None,
        "rmse_all_gcps_m": float(np.sqrt((resid ** 2).mean())),
        "rejected": [pts[i][:3] for i in np.nonzero(~inl)[0].tolist()],
    }


# ---------------------------------------------------------------------
# v5: GCPs against the terrain / structure split
# ---------------------------------------------------------------------
def refine_with_gcps_decomposed(dtm_m: np.ndarray, ndsm_m: np.ndarray, gcps, *,
                                iters: int = 300, inlier_m: float = 1.0,
                                seed: int = 0) -> tuple[np.ndarray, dict]:
    """GCPs fitted where each error actually lives.

    v4 fitted `s * DSM + o` to the whole absolute DSM, so a scale correction
    meant for building heights also stretched the terrain (a 1.05 scale on a
    400 m plateau is a 20 m error).  Here the model is

        z = DTM + a + b*x + c*y + s * nDSM

    an offset (plus a tilt when >= 4 points allow it) on the terrain, and a scale
    on the structures only.  Point counts decide how much is fitted: 1 -> offset;
    2-3 -> offset + scale; >= 4 -> offset + tilt + scale.  RANSAC over minimal
    sets, then least squares on the consensus.  `gcps`: (row, col, elevation_m).
    """
    H, W = dtm_m.shape
    pts = [(int(r), int(c), float(z)) for r, c, z in gcps
           if 0 <= int(r) < H and 0 <= int(c) < W]
    if not pts:
        return dtm_m + ndsm_m, {"gcp": "no in-bounds control points", "n_gcps": 0}
    r = np.array([p[0] for p in pts], np.float64)
    c = np.array([p[1] for p in pts], np.float64)
    z = np.array([p[2] for p in pts], np.float64)
    t = np.array([dtm_m[int(a), int(b)] for a, b in zip(r, c)], np.float64)
    h = np.array([ndsm_m[int(a), int(b)] for a, b in zip(r, c)], np.float64)
    y = z - t - h                                  # what is left to explain
    n = len(pts)
    xn, yn = c / max(W - 1, 1), r / max(H - 1, 1)
    if n >= 4:
        A = np.stack([np.ones(n), xn, yn, h], 1)
        kind = "offset+tilt+scale"
    elif n >= 2 and np.ptp(h) > 0.5:
        A = np.stack([np.ones(n), h], 1)
        kind = "offset+scale"
    else:
        A = np.ones((n, 1))
        kind = "offset"
    m = A.shape[1]
    rng = np.random.default_rng(seed)
    best = np.ones(n, bool)
    best_n = -1
    if n > m:
        for _ in range(iters):
            idx = rng.choice(n, m, replace=False)
            sol, *_ = np.linalg.lstsq(A[idx], y[idx], rcond=None)
            inl = np.abs(A @ sol - y) <= inlier_m
            if inl.sum() > best_n:
                best, best_n = inl, int(inl.sum())
    sol, *_ = np.linalg.lstsq(A[best], y[best], rcond=None)
    coef = dict(zip(["offset_m", "tilt_x_m", "tilt_y_m", "scale_minus_1"]
                    if m == 4 else (["offset_m", "scale_minus_1"] if m == 2 else ["offset_m"]),
                    [float(v) for v in sol]))
    off = coef.get("offset_m", 0.0)
    bx, by = coef.get("tilt_x_m", 0.0), coef.get("tilt_y_m", 0.0)
    ds = coef.get("scale_minus_1", 0.0)
    yy, xx = np.mgrid[0:H, 0:W]
    terrain = dtm_m + off + bx * (xx / max(W - 1, 1)) + by * (yy / max(H - 1, 1))
    dsm = (terrain + (1.0 + ds) * ndsm_m).astype(np.float32)
    resid = A @ sol - y
    return dsm, {"gcp": kind, "n_gcps": n, "inliers": int(best.sum()),
                 "structure_scale": 1.0 + ds, **coef,
                 "rmse_inliers_m": float(np.sqrt(np.mean(resid[best] ** 2))),
                 "rmse_all_gcps_m": float(np.sqrt(np.mean(resid ** 2))),
                 "rejected": [pts[i] for i in np.nonzero(~best)[0].tolist()]}
