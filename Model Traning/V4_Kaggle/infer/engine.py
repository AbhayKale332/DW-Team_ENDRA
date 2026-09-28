"""The one inference path.

`predict_scene()` is the *only* place a height map is produced from an image, and
both the final evaluation (`eval/sliding.py`) and the CLI (`infer/predict.py`)
call it.  That is deliberate: v2 had one code path for validation (centre crop of
a normalised GAMUS tile) and a different one for real images (`predict_image.py`,
which resized any PNG to 512 regardless of its ground sample distance and used
ImageNet normalisation constants the training script also used but the encoder
does not want).  A validation number produced by a path the demo never runs is
not a validation number.

Contract in / out
-----------------
    rgb_u8   (H, W, 3) uint8 at `src_gsd_m` metres per pixel, *unstretched*
    ->  height_m (H, W) float32, nDSM in metres, on the SAME grid as the input

Internally: scene stretch -> resample to `spec.canonical_gsd_m` -> overlapping
`spec.tile_size` windows -> optional D4 TTA -> Hann-weighted blend -> resample the
height map back to the input grid.  Heights are metres, so the round trip does not
rescale them.
"""

from __future__ import annotations

import numpy as np
import torch

from dwdata.preprocess import (
    PreprocSpec, gsd_to_shape, hann2d, resize, stretch_scene, tile_origins,
)


@torch.no_grad()
def predict_canonical(
    model, rgb_canon_u8: np.ndarray, spec: PreprocSpec, device,
    *, tta: bool = False, tta_scales=(1.0,), amp_dtype=None,
    overlap: float = 0.25, batch_tiles: int = 4, want_seg: bool = False,
    progress=None,
):
    """Sliding-window prediction over an image already at the canonical GSD."""
    from models.tta import tta_predict

    S = spec.tile_size
    H, W = rgb_canon_u8.shape[:2]
    # Reflect-pad rather than zero-pad: a black border is a strong out-of-
    # distribution edge and v2 trained on ~41% of it, so we never introduce one.
    pad_h, pad_w = max(0, S - H), max(0, S - W)
    if pad_h or pad_w:
        rgb_canon_u8 = np.pad(rgb_canon_u8, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    Hp, Wp = rgb_canon_u8.shape[:2]

    ys, xs = tile_origins(Hp, S, overlap), tile_origins(Wp, S, overlap)
    win = hann2d(S)
    acc = np.zeros((Hp, Wp), np.float32)
    wsum = np.zeros((Hp, Wp), np.float32)
    seg_acc = None
    coords = [(y, x) for y in ys for x in xs]
    total = len(coords)

    for b0 in range(0, total, batch_tiles):
        chunk = coords[b0:b0 + batch_tiles]
        batch = np.stack([
            spec.normalise(rgb_canon_u8[y:y + S, x:x + S]) for y, x in chunk
        ])
        t = torch.from_numpy(batch).to(device, non_blocking=True)
        if tta:
            pred = tta_predict(model, t, tuple(tta_scales), "fused",
                               patch=spec.patch, amp_dtype=amp_dtype)
            seg = None
            if want_seg:
                with _autocast(device, amp_dtype):
                    seg = model(t)["seg"]
        else:
            with _autocast(device, amp_dtype):
                out = model(t)
            pred, seg = out["fused"], (out["seg"] if want_seg else None)
        pred = pred.float().cpu().numpy()[:, 0]
        if seg is not None:
            seg = seg.float().argmax(1).cpu().numpy()
            if seg_acc is None:
                seg_acc = np.zeros((Hp, Wp), np.int16)

        for i, (y, x) in enumerate(chunk):
            acc[y:y + S, x:x + S] += pred[i] * win
            wsum[y:y + S, x:x + S] += win
            if seg is not None:
                seg_acc[y:y + S, x:x + S] = seg[i]
        if progress:
            progress(min(b0 + batch_tiles, total), total)

    height = acc / np.maximum(wsum, 1e-6)
    height = height[:H, :W]
    if seg_acc is not None:
        seg_acc = seg_acc[:H, :W]
    return height.astype(np.float32), seg_acc


def _autocast(device, amp_dtype):
    if amp_dtype is None or device.type != "cuda":
        import contextlib

        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=amp_dtype)


@torch.no_grad()
def predict_scene(
    model, rgb_u8: np.ndarray, src_gsd_m: float, spec: PreprocSpec, device,
    *, tta: bool = False, tta_scales=(1.0,), amp_dtype=None, overlap: float = 0.25,
    batch_tiles: int = 4, want_seg: bool = False, progress=None,
):
    """RGB at any GSD -> nDSM in metres on the input grid (+ optional class map)."""
    H, W = rgb_u8.shape[:2]
    rgb_s = stretch_scene(rgb_u8, spec)
    canon_hw = gsd_to_shape(H, W, src_gsd_m, spec.canonical_gsd_m)
    rgb_c = resize(rgb_s, canon_hw, "bilinear")

    height_c, seg_c = predict_canonical(
        model, rgb_c, spec, device, tta=tta, tta_scales=tta_scales,
        amp_dtype=amp_dtype, overlap=overlap, batch_tiles=batch_tiles,
        want_seg=want_seg, progress=progress,
    )
    height = resize(height_c, (H, W), "bilinear").astype(np.float32)
    seg = resize(seg_c.astype(np.int32), (H, W), "nearest") if seg_c is not None else None
    return height, seg
