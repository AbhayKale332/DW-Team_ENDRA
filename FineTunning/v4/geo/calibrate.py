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
              *, max_ground_height_m: float = 1.5, poly_order: int = 2,
              residual_window: int = 64) -> Calibration:
    """The whole decomposition, in one call."""
    from geo.dem import fill_nan

    dem = fill_nan(dem_m)
    mask = ground_mask(ndsm_m, seg, max_height_m=max_ground_height_m)
    dtm, info = fit_dtm(dem, mask, order=poly_order, residual_window=residual_window)
    dsm = (dtm + np.clip(ndsm_m, 0.0, None)).astype(np.float32)

    # How much building height was in the raw DEM — i.e. what we just avoided
    # counting twice.  Reported so the effect is visible, not asserted.
    struct = ndsm_m > 5.0
    over = float(np.nanmean(dem[struct] - dtm[struct])) if struct.any() else 0.0
    info.update({
        "dem_minus_dtm_over_structures_m": over,
        "double_count_avoided_m": over,
        "dtm_range_m": [float(dtm.min()), float(dtm.max())],
        "dsm_range_m": [float(dsm.min()), float(dsm.max())],
        "ground_mask_frac": float(mask.mean()),
    })
    print(f"[calib] DTM {dtm.min():.1f}..{dtm.max():.1f} m from "
          f"{info['ground_mask_frac'] * 100:.1f}% ground pixels; the raw DEM sat "
          f"{over:+.2f} m above it over structures (that is the double-count avoided)")
    return Calibration(dsm, dtm, mask, info)


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
