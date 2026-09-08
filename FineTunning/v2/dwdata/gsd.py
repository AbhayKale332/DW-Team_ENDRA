"""GSD (ground-sample-distance) canonicalisation and jitter.

Why (from `.agents/Depth_Wizard_Plan.md`): metres-per-pixel is what makes the
output metric.  GAMUS is 0.33 m, GeoNRW 1 m, SynRS3D 0.05-1 m, and ISRO will hand
us something else (~0.25-1.1 m).  So:

  * train: resample each tile to a *random* GSD in [min, max] (pixel spacing
    changes; height values in metres do NOT) so the model degrades gracefully
    across resolutions.
  * eval: resample every tile to a fixed canonical GSD (0.5 m), predict, resample
    the prediction back to native for scoring -- a geometry-preserving round trip.

Everything here is numpy; arrays are (H, W) or (H, W, 3).
"""

from __future__ import annotations

import numpy as np


def _resize(arr: np.ndarray, out_hw: tuple[int, int], order: str) -> np.ndarray:
    """order: 'bilinear' | 'nearest'.  Pure-numpy via PIL (already a Kaggle dep)."""
    from PIL import Image

    h, w = out_hw
    if arr.shape[:2] == (h, w):
        return arr
    resample = Image.BILINEAR if order == "bilinear" else Image.NEAREST
    if arr.ndim == 3:
        im = Image.fromarray(arr.astype(np.uint8))
        return np.asarray(im.resize((w, h), resample))
    im = Image.fromarray(arr.astype(np.float32), mode="F")
    return np.asarray(im.resize((w, h), resample), dtype=np.float32)


def rescale_to_gsd(
    rgb: np.ndarray,
    height: np.ndarray,
    seg: np.ndarray,
    src_gsd_m: float,
    dst_gsd_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Change pixel spacing from src to dst GSD. Heights are unchanged (metres)."""
    if abs(src_gsd_m - dst_gsd_m) < 1e-6:
        return rgb, height, seg
    scale = src_gsd_m / dst_gsd_m           # >1 -> upsample (finer grid)
    h, w = height.shape
    out_hw = (max(1, round(h * scale)), max(1, round(w * scale)))
    return (
        _resize(rgb, out_hw, "bilinear"),
        _resize(height, out_hw, "bilinear"),
        _resize(seg, out_hw, "nearest"),
    )


def center_crop_or_pad(arr: np.ndarray, size: int, pad_value: float = 0.0) -> np.ndarray:
    """Force (size, size[, C]) by centre crop and/or edge pad."""
    h, w = arr.shape[:2]
    # pad
    ph, pw = max(0, size - h), max(0, size - w)
    if ph or pw:
        pad = [(ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)]
        if arr.ndim == 3:
            pad.append((0, 0))
        arr = np.pad(arr, pad, mode="constant", constant_values=pad_value)
        h, w = arr.shape[:2]
    # crop
    top, left = (h - size) // 2, (w - size) // 2
    return arr[top: top + size, left: left + size]


def jitter_gsd(
    rgb: np.ndarray,
    height: np.ndarray,
    seg: np.ndarray,
    src_gsd_m: float,
    lo_m: float,
    hi_m: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Pick a random target GSD in [lo, hi], resample, return (…, chosen_gsd)."""
    dst = float(rng.uniform(lo_m, hi_m))
    r, h, s = rescale_to_gsd(rgb, height, seg, src_gsd_m, dst)
    return r, h, s, dst


def roundtrip_predict(
    predict_fn,
    rgb: np.ndarray,
    src_gsd_m: float,
    canonical_gsd_m: float,
) -> np.ndarray:
    """Eval helper: rgb(native) -> canonical -> predict -> back to native height map.

    `predict_fn(rgb_canonical: HxWx3 uint8) -> HxW float32` (metres).
    """
    h, w = rgb.shape[:2]
    scale = src_gsd_m / canonical_gsd_m
    canon_hw = (max(16, round(h * scale)), max(16, round(w * scale)))
    rgb_c = _resize(rgb, canon_hw, "bilinear").astype(np.uint8)
    pred_c = predict_fn(rgb_c).astype(np.float32)
    return _resize(pred_c, (h, w), "bilinear")
