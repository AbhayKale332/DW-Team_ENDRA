"""Train-time augmentation.

Two families, and they fix two different v2 failures:

**Geometric — `sample_window`.**  v2 resampled the *whole* source tile to a random
GSD and then centre-cropped/padded to 512.  Because GAMUS tiles are 1024 px at
0.33 m, any requested GSD above ~0.66 m produced an image smaller than the crop,
so it was zero-padded — and the padded pixels were marked *valid* with target
height 0.  Measured over the configured jitter range that is ~41 % of all training
pixels being black padding paired with a 0 m label.  v3 picks the crop **in source
pixels first** and resamples that window to the tile size, so a tile is full by
construction and the achievable GSD range is derived from the source extent.

**Photometric — `photometric_jitter`.**  v2 had none.  Trained on one sensor's
radiometry, the network learned "bright texture => tall", which is exactly what it
did on Inria Austin (median predicted height 4 m on a residential scene, ground
never flat).  Brightness / contrast / gamma / saturation / per-channel gain / blur
/ noise, on top of the scene-level percentile stretch in `preprocess.py`, is the
cheapest available defence before ISRO hands us an unseen sensor.
"""

from __future__ import annotations

import numpy as np

from .preprocess import resize


# ---------------------------------------------------------------------
# geometric
# ---------------------------------------------------------------------
def achievable_gsd_range(
    src_h: int, src_w: int, src_gsd_m: float, tile: int,
    lo_m: float, hi_m: float, min_window: int = 64,
) -> tuple[float, float]:
    """Clamp [lo, hi] to GSDs this source tile can actually fill.

    A window of `tile * dst/src` source pixels must fit inside the tile and stay
    above `min_window` px (below that we would be upsampling from mush).
    """
    max_win = min(src_h, src_w)
    hi_ok = max_win * src_gsd_m / tile        # coarsest GSD that still fills the crop
    lo_ok = min_window * src_gsd_m / tile     # finest GSD worth asking for
    lo = max(lo_m, lo_ok)
    hi = min(hi_m, hi_ok)
    if hi < lo:                                # source too small for the range
        lo = hi = max(lo_ok, min(hi_ok, hi_ok))
    return float(lo), float(hi)


def sample_window(
    src_h: int, src_w: int, src_gsd_m: float, tile: int,
    dst_gsd_m: float, rng: np.random.Generator,
) -> tuple[int, int, int, float]:
    """(top, left, window_px, effective_gsd) for a crop that fills `tile` exactly."""
    win = int(round(tile * dst_gsd_m / src_gsd_m))
    win = max(8, min(win, src_h, src_w))
    eff = win * src_gsd_m / tile              # the GSD actually delivered
    top = int(rng.integers(0, src_h - win + 1))
    left = int(rng.integers(0, src_w - win + 1))
    return top, left, win, float(eff)


def center_window(src_h: int, src_w: int, src_gsd_m: float, tile: int,
                  dst_gsd_m: float) -> tuple[int, int, int, float]:
    win = int(round(tile * dst_gsd_m / src_gsd_m))
    win = max(8, min(win, src_h, src_w))
    eff = win * src_gsd_m / tile
    return (src_h - win) // 2, (src_w - win) // 2, win, float(eff)


def crop_and_scale(
    rgb: np.ndarray, height: np.ndarray, seg: np.ndarray, valid: np.ndarray,
    top: int, left: int, win: int, tile: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Crop a `win` x `win` source window and resample it to `tile` x `tile`.

    Heights stay in metres.  `valid` rides along as a mask so nothing invalid is
    ever silently turned into a 0 m label (v2's other padding bug).
    """
    sl = (slice(top, top + win), slice(left, left + win))
    r, h, s, v = rgb[sl], height[sl], seg[sl], valid[sl]
    if win != tile:
        r = resize(r, (tile, tile), "bilinear")
        h = resize(h.astype(np.float32), (tile, tile), "bilinear")
        s = resize(s.astype(np.int32), (tile, tile), "nearest")
        # nearest on the mask: a resampled pixel is valid only if its source was
        v = resize(v.astype(np.uint8), (tile, tile), "nearest").astype(bool)
    # `np.array(..., copy=True)`, not `ascontiguousarray`: the source is a
    # read-only memmap slice and a view of it would make a non-writable tensor.
    return (np.array(r, dtype=np.uint8, copy=True),
            np.array(h, dtype=np.float32, copy=True),
            np.array(s, dtype=np.int64, copy=True),
            np.array(v, dtype=bool, copy=True))


def dihedral(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, k)
    if flip:
        out = np.fliplr(out)
    return np.ascontiguousarray(out)


# ---------------------------------------------------------------------
# photometric  (RGB only — heights and labels are never touched)
# ---------------------------------------------------------------------
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def photometric_jitter(rgb_u8: np.ndarray, cfg, rng: np.random.Generator) -> np.ndarray:
    x = rgb_u8.astype(np.float32) / 255.0

    if cfg.photo_gamma > 0:
        x = np.power(np.clip(x, 1e-4, 1.0),
                     float(np.exp(rng.uniform(-cfg.photo_gamma, cfg.photo_gamma))))
    if cfg.photo_channel_gain > 0:
        g = 1.0 + rng.uniform(-cfg.photo_channel_gain, cfg.photo_channel_gain, size=3)
        x = x * g.astype(np.float32)
    if cfg.photo_saturation > 0:
        grey = (x * _LUMA).sum(-1, keepdims=True)
        f = 1.0 + float(rng.uniform(-cfg.photo_saturation, cfg.photo_saturation))
        x = grey + (x - grey) * f
    if cfg.photo_contrast > 0:
        m = float(x.mean())
        f = 1.0 + float(rng.uniform(-cfg.photo_contrast, cfg.photo_contrast))
        x = m + (x - m) * f
    if cfg.photo_brightness > 0:
        x = x + float(rng.uniform(-cfg.photo_brightness, cfg.photo_brightness))

    x = np.clip(x, 0.0, 1.0)

    if cfg.photo_blur_p > 0 and rng.random() < cfg.photo_blur_p:
        x = _blur(x, float(rng.uniform(0.4, 1.3)))
    if cfg.photo_noise_std > 0:
        x = x + rng.normal(0.0, float(rng.uniform(0.0, cfg.photo_noise_std)), size=x.shape)

    return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _blur(x: np.ndarray, sigma: float) -> np.ndarray:
    try:
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(x, sigma=(sigma, sigma, 0), mode="nearest")
    except Exception:  # noqa: BLE001
        k = np.array([1.0, 2.0, 1.0], np.float32)
        k /= k.sum()
        for ax in (0, 1):
            x = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), ax, x)
        return x
