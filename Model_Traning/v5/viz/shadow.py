"""Shadows: seen in the image, cast by the predicted surface, and the agreement.

Two masks that should coincide when the heights are right:

* **in the image** — `detect_image_shadows`, the HSI ratio of Tsai (2006):
  shadow pixels are dark *and* shifted towards blue, so `(H + 1) / (I + 1)` is
  high there; Otsu picks the threshold per scene, a 3x3 opening and a minimum
  blob size remove speckle.
* **cast by the heights** — `cast_shadows(height, gsd, az, el)`: a pixel is in
  shadow when anything between it and the sun rises above the sun ray.

Their agreement (IoU) needs no labels, which is the point on Indian imagery:
there is no reference DSM for the judges' scenes, but there are shadows.  With
the sun known from `BAND_META.txt` (every NRSC product carries
`SunAzimuthAtCenter` / `SunElevationAtCenter`), `shadow_scale_check` searches one
height scale s maximising IoU(image, cast(s * nDSM)) — Kadhim & Mourshed (2018).
s ~ 1 is label-free evidence that the metric scale is right; s far from 1 says
the heights are systematically off.  For a PNG with an unknown sun, `fit_sun`
estimates az/el instead, and there the scale check is meaningless (scale and
elevation trade off exactly: shadow length = h / tan(el)), so it is not run.

Azimuth is clockwise from image-up (north for a north-up product), the same
convention as the viewer's `hillshade()`.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------
# shadows in the image
# ---------------------------------------------------------------------
def _otsu(v: np.ndarray, bins: int = 256) -> float:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    hist, edges = np.histogram(v, bins=bins)
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    w0 = np.cumsum(p)
    mu = np.cumsum(p * (edges[:-1] + edges[1:]) / 2)
    mt = mu[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        sb = (mt * w0 - mu) ** 2 / (w0 * (1 - w0))
    sb[~np.isfinite(sb)] = 0
    i = int(np.argmax(sb))
    return float((edges[i] + edges[i + 1]) / 2)


def detect_image_shadows(rgb_u8: np.ndarray, valid: np.ndarray | None = None,
                         min_blob_px: int = 8) -> np.ndarray:
    """Boolean shadow mask of an RGB image (HSI ratio + Otsu + cleanup)."""
    x = rgb_u8.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    inten = (r + g + b) / 3.0
    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b)) + 1e-6
    theta = np.arccos(np.clip(num / den, -1.0, 1.0))
    hue = np.where(b > g, 2 * np.pi - theta, theta) / (2 * np.pi)
    ratio = (hue + 1.0) / (inten + 1.0)
    v = np.ones(ratio.shape, bool) if valid is None else valid
    t_r = _otsu(ratio[v])
    t_i = _otsu(inten[v])
    return _despeckle((ratio > t_r) & (inten < t_i) & v, min_blob_px)


def _despeckle(m: np.ndarray, min_blob_px: int = 8) -> np.ndarray:
    """3x3 opening, then drop blobs under `min_blob_px`."""
    try:
        from scipy import ndimage as ndi

        m = ndi.binary_opening(m, structure=np.ones((3, 3), bool))
        lab, n = ndi.label(m)
        if n:
            sizes = np.bincount(lab.ravel())
            keep = sizes >= min_blob_px
            keep[0] = False
            m = keep[lab]
    except Exception:  # noqa: BLE001 — scipy is in requirements; stay usable without
        pass
    return m


# ---------------------------------------------------------------------
# shadows cast by a height map
# ---------------------------------------------------------------------
def sun_direction(az_deg: float) -> tuple[float, float]:
    """(drow, dcol) of one pixel step *towards* the sun (row grows downwards)."""
    a = np.radians(az_deg)
    return -np.cos(a), np.sin(a)


def cast_shadows(height: np.ndarray, gsd_m: float, az_deg: float, el_deg: float,
                 max_steps: int = 512) -> np.ndarray:
    """Pixels the surface itself shades from a sun at (az, el).

    Marches every pixel towards the sun in 1 px steps and compares against the
    sun ray `h + t * gsd * tan(el)`; stops as soon as no ray can still be
    blocked (the tallest remaining obstacle is below every ray).  Vectorised
    over the whole map per step, so cost is steps x pixels — use a working
    resolution of ~1k px (`shadow_products` does).
    """
    h = np.nan_to_num(np.asarray(height, np.float32), nan=np.nan)
    fin = np.isfinite(h)
    hf = np.where(fin, h, -np.inf).astype(np.float32)
    H, W = hf.shape
    dr, dc = sun_direction(az_deg)
    rise = float(gsd_m) * np.tan(np.radians(max(el_deg, 0.5)))
    hmax = float(np.max(hf[fin])) if fin.any() else 0.0
    hmin = float(np.min(hf[fin])) if fin.any() else 0.0
    steps = int(min(max_steps, np.ceil((hmax - hmin) / max(rise, 1e-6)) + 1))
    shade = np.zeros((H, W), bool)
    base = np.where(fin, hf, np.inf)
    for t in range(1, steps + 1):
        oy, ox = int(round(t * dr)), int(round(t * dc))
        if abs(oy) >= H or abs(ox) >= W:
            break
        # obstacle[y, x] = h[y + oy, x + ox]  (-inf outside the scene)
        ob = np.full((H, W), -np.inf, np.float32)
        ys = slice(max(0, -oy), min(H, H - oy))
        xs = slice(max(0, -ox), min(W, W - ox))
        yd = slice(max(0, oy), min(H, H + oy))
        xd = slice(max(0, ox), min(W, W + ox))
        ob[ys, xs] = hf[yd, xd]
        shade |= ob > base + t * rise
    return shade & fin


def iou(a: np.ndarray, b: np.ndarray, valid: np.ndarray | None = None) -> float:
    if valid is not None:
        a, b = a & valid, b & valid
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def _working(height, mask, valid, gsd_m, max_side: int = 1024):
    """Decimate to <= max_side px (shadows are a scene-scale comparison)."""
    s = max(1, int(np.ceil(max(height.shape) / max_side)))
    if s == 1:
        return height, mask, valid, gsd_m
    return (height[::s, ::s], mask[::s, ::s],
            None if valid is None else valid[::s, ::s], gsd_m * s)


def fit_sun(height: np.ndarray, gsd_m: float, img_mask: np.ndarray,
            valid: np.ndarray | None = None) -> dict:
    """Grid-search the sun (az 0-355, el 15-85) that best explains the image's
    shadows: a coarse 10 deg pass, then 5 deg around the winner.  Only for
    inputs without sun metadata."""
    h, m, v, g = _working(height, img_mask, valid, gsd_m, 256)

    def score(az, el):
        return iou(cast_shadows(h, g, az, el), m, v)

    best = max(((score(az, el), az, el) for az in range(0, 360, 10)
                for el in range(15, 86, 10)))
    _, az0, el0 = best
    best = max([best] + [(score(az % 360, el), az % 360, el)
                         for az in range(az0 - 10, az0 + 11, 5)
                         for el in range(max(15, el0 - 10), min(85, el0 + 10) + 1, 5)])
    return {"azimuth_deg": float(best[1]), "elevation_deg": float(best[2]),
            "shadow_iou": best[0], "source": "fit"}


def shadow_scale_check(height: np.ndarray, gsd_m: float, az: float, el: float,
                       img_mask: np.ndarray, valid: np.ndarray | None = None,
                       scales=None) -> dict:
    """Height scale s maximising IoU(image shadows, cast(s * height)).

    Evidence for the metric scale that uses the sun, not labels.  Reported,
    never applied: a scene with few shadows gives a flat, uninformative curve,
    which `curve` makes visible.
    """
    h, m, v, g = _working(height, img_mask, valid, gsd_m)
    if scales is None:
        scales = np.round(np.linspace(0.5, 2.0, 16), 3)
    base = np.nanmin(h) if np.isfinite(h).any() else 0.0
    curve = [(float(s), iou(cast_shadows(base + s * (h - base), g, az, el), m, v))
             for s in scales]
    s_best, i_best = max(curve, key=lambda t: t[1])
    i1 = iou(cast_shadows(h, g, az, el), m, v)
    return {"best_scale": s_best, "iou_best": i_best, "iou_at_1": i1,
            "curve": curve, "informative": bool(i_best - min(c[1] for c in curve) > 0.02)}


def reference_sun_check(rgb, gt, pred, valid, gsd_m: float) -> dict | None:
    """Image shadows vs the shadows the reference and the prediction would cast.

    GAMUS ships no sun metadata, so the sun is fitted on the *reference* heights
    against the image's own shadows (`fit_sun`), then the prediction is cast under
    that same sun.  Fitting on the prediction would let it grade itself.  Masks
    are at the working resolution `_working` picks (<= 1024 px), and both cast
    masks get the image detector's despeckle: sub-metre texture in a predicted
    ground plane otherwise casts pixel-sized "shadows" everywhere and the IoU
    measures that noise instead of the structures.
    """
    img = detect_image_shadows(rgb, valid)
    if not img[valid].any():
        return None
    sun = fit_sun(np.where(valid, gt, 0.0), gsd_m, img, valid)
    az, el = sun["azimuth_deg"], sun["elevation_deg"]
    g, m, v, gg = _working(np.where(valid, gt, 0.0), img, valid, gsd_m)
    p = _working(pred, img, valid, gsd_m)[0]
    cast_g = _despeckle(cast_shadows(g, gg, az, el))
    cast_p = _despeckle(cast_shadows(p, gg, az, el))
    return {"azimuth_deg": az, "elevation_deg": el,
            "image": m, "cast_ref": cast_g, "cast_pred": cast_p, "valid": v,
            "iou_ref": iou(cast_g, m, v), "iou_pred": iou(cast_p, m, v),
            "iou_pred_ref": iou(cast_p, cast_g, v),
            "image_shadow_frac": float(m[v].mean()) if v.any() else 0.0}


def shadow_products(out_dir, rgb_u8: np.ndarray, height_m: np.ndarray, meta) -> dict:
    """What `write_outputs` records: `shadow_img.png` + the `sun` block of meta.json."""
    from pathlib import Path

    from PIL import Image

    valid = getattr(meta, "valid", None)
    if valid is not None and valid.shape != height_m.shape:
        valid = None
    fin = np.isfinite(height_m)
    v = fin if valid is None else (valid & fin)
    img = detect_image_shadows(rgb_u8, v)
    Image.fromarray((img * 255).astype(np.uint8)).save(Path(out_dir) / "shadow_img.png")
    sun = getattr(meta, "sun", None)
    g = float(meta.gsd_m)
    if sun is not None:
        out = {"azimuth_deg": sun[0], "elevation_deg": sun[1], "source": "metadata"}
        h, m, vv, gg = _working(height_m, img, v, g)
        out["shadow_iou"] = iou(cast_shadows(h, gg, *sun), m, vv)
        if meta.gsd_source != "assumed":
            out["scale_check"] = shadow_scale_check(height_m, g, sun[0], sun[1], img, v)
    else:
        out = fit_sun(height_m, g, img, v)
    out["image_shadow_frac"] = float(img[v].mean()) if v.any() else 0.0
    out["files"] = ["shadow_img.png"]
    return out
