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

from pathlib import Path

import numpy as np
import torch

from dwdata.preprocess import (
    PreprocSpec, gsd_to_shape, hann2d, resize, scene_stretch_bounds, stretch_scene,
    tile_origins,
)


@torch.no_grad()
def predict_canonical(
    model, rgb_canon_u8: np.ndarray, spec: PreprocSpec, device,
    *, tta: bool = False, tta_scales=(1.0,), amp_dtype=None,
    overlap: float = 0.25, batch_tiles: int = 4, want_seg: bool = False,
    progress=None, want_std: bool = False,
):
    """Sliding-window prediction over an image already at the canonical GSD.

    `want_std` (v5): also blend Head B's per-pixel `b_std` (the spread of its bin
    distribution, metres) the same way as the height, and return it as a third
    value — `None` when the model does not emit it (the ONNX graph, v3).
    """
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
    std_acc = None
    coords = [(y, x) for y in ys for x in xs]
    total = len(coords)

    for b0 in range(0, total, batch_tiles):
        chunk = coords[b0:b0 + batch_tiles]
        batch = np.stack([
            spec.normalise(rgb_canon_u8[y:y + S, x:x + S]) for y, x in chunk
        ])
        t = torch.from_numpy(batch).to(device, non_blocking=True)
        std = None
        if tta:
            pred = tta_predict(model, t, tuple(tta_scales), "fused",
                               patch=spec.patch, amp_dtype=amp_dtype)
            seg = None
            if want_seg or want_std:
                with _autocast(device, amp_dtype):
                    out = model(t)
                seg = out["seg"] if want_seg else None
                std = out.get("b_std") if want_std else None
        else:
            with _autocast(device, amp_dtype):
                out = model(t)
            pred, seg = out["fused"], (out["seg"] if want_seg else None)
            std = out.get("b_std") if want_std else None
        pred = pred.float().cpu().numpy()[:, 0]
        if std is not None:
            std = std.float().cpu().numpy()[:, 0]
            if std_acc is None:
                std_acc = np.zeros((Hp, Wp), np.float32)
        if seg is not None:
            seg = seg.float().argmax(1).cpu().numpy()
            if seg_acc is None:
                seg_acc = np.zeros((Hp, Wp), np.int16)

        for i, (y, x) in enumerate(chunk):
            acc[y:y + S, x:x + S] += pred[i] * win
            wsum[y:y + S, x:x + S] += win
            if std is not None:
                std_acc[y:y + S, x:x + S] += std[i] * win
            if seg is not None:
                seg_acc[y:y + S, x:x + S] = seg[i]
        if progress:
            progress(min(b0 + batch_tiles, total), total)

    height = acc / np.maximum(wsum, 1e-6)
    height = height[:H, :W]
    if seg_acc is not None:
        seg_acc = seg_acc[:H, :W]
    if want_std:
        std_out = None if std_acc is None else \
            (std_acc / np.maximum(wsum, 1e-6))[:H, :W].astype(np.float32)
        return height.astype(np.float32), seg_acc, std_out
    return height.astype(np.float32), seg_acc


def _autocast(device, amp_dtype):
    if amp_dtype is None or device.type != "cuda":
        import contextlib

        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=amp_dtype)


def _fill_invalid(rgb_u8: np.ndarray, valid: np.ndarray | None,
                  fill: np.ndarray | None = None) -> np.ndarray:
    """Paint NoData with the scene's median colour.

    A Cartosat collar is value 0 — a hard black edge the network never saw in
    training (v2 learned a 4 m cliff from exactly that).  A flat median-colour
    field reads as featureless ground instead; its prediction is discarded
    anyway (NaN in the output).
    """
    if valid is None or valid.all():
        return rgb_u8
    if fill is None:
        fill = (np.median(rgb_u8[valid], axis=0) if valid.any()
                else np.full(3, 127)).astype(np.uint8)
    out = rgb_u8.copy()
    out[~valid] = fill
    return out


@torch.no_grad()
def predict_scene(
    model, rgb_u8: np.ndarray, src_gsd_m: float, spec: PreprocSpec, device,
    *, tta: bool = False, tta_scales=(1.0,), amp_dtype=None, overlap: float = 0.25,
    batch_tiles: int = 4, want_seg: bool = False, progress=None,
    valid: np.ndarray | None = None, return_std: bool = False,
):
    """RGB at any GSD -> nDSM in metres on the input grid (+ optional class map).

    `valid` (v5): NoData pixels are excluded from the stretch, painted with the
    median colour before the network sees them, and come back as NaN.
    `return_std` (v5): return `(height, seg, std)`, std = Head B's per-pixel
    spread in metres on the same grid, or None if the model has no Head B output.
    """
    H, W = rgb_u8.shape[:2]
    rgb_s = stretch_scene(rgb_u8, spec, valid=valid)
    rgb_s = _fill_invalid(rgb_s, valid)
    canon_hw = gsd_to_shape(H, W, src_gsd_m, spec.canonical_gsd_m)
    rgb_c = resize(rgb_s, canon_hw, "bilinear")

    res = predict_canonical(
        model, rgb_c, spec, device, tta=tta, tta_scales=tta_scales,
        amp_dtype=amp_dtype, overlap=overlap, batch_tiles=batch_tiles,
        want_seg=want_seg, progress=progress, want_std=return_std,
    )
    height_c, seg_c = res[0], res[1]
    height = resize(height_c, (H, W), "bilinear").astype(np.float32)
    seg = resize(seg_c.astype(np.int32), (H, W), "nearest") if seg_c is not None else None
    std = None
    if return_std and res[2] is not None:
        std = resize(res[2], (H, W), "bilinear").astype(np.float32)
    if valid is not None and not valid.all():
        height[~valid] = np.nan
        if std is not None:
            std[~valid] = np.nan
    if return_std:
        return height, seg, std
    return height, seg


# ---------------------------------------------------------------------
# v5: scenes that do not fit in memory
# ---------------------------------------------------------------------
def _geotiff_profile(src, dtype: str, nodata) -> dict:
    return dict(driver="GTiff", height=src.height, width=src.width, count=1,
                dtype=dtype, crs=src.crs, transform=src.transform, nodata=nodata,
                compress="deflate", tiled=True, blockxsize=512, blockysize=512,
                BIGTIFF="IF_SAFER", **({"predictor": 3} if dtype == "float32" else {}))


@torch.no_grad()
def predict_scene_windowed(
    model, source, spec: PreprocSpec, device, out_dir,
    *, tta: bool = False, tta_scales=(1.0,), amp_dtype=None, overlap: float = 0.25,
    batch_tiles: int = 4, band_rows: int = 2048, want_seg: bool = True,
    overview_max: int = 2048, block_px: int = 0, progress=None,
) -> dict:
    """Row-band inference over a `dwdata.scene_io.SceneSource`, streamed to disk.

    RAM is bounded by one band (~2048 source rows + a one-tile halo on each side)
    instead of the scene: a 20k x 20k Cartosat MERGED product is ~600 MP at the
    canonical 0.5 m, i.e. >10 GB of float buffers on the whole-array path.

    Everything scene-level is decided once, before the first band, so bands are
    radiometrically identical: the native->uint8 map (`SceneSource.radiometry`),
    the contract's percentile stretch and the NoData fill colour all come from
    one decimated read of the whole scene.

    Writes `ndsm_m.tif` (float32, NaN = NoData) and, with `want_seg`, `seg.tif`
    (uint8, 255 = NoData) on the *source* grid, and returns decimated overviews
    for the viewer plus the per-block nDSM sums that the DEM-anchored fusion
    needs (`geo/calibrate.py calibrate_windowed`).  `block_px` > 0 sets that
    block size (source pixels, normally 30 m / GSD).
    """
    import rasterio
    from rasterio.windows import Window

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    H, W, g = source.height, source.width, float(source.gsd_m)

    # -- scene-level radiometry, once ----------------------------------
    ov_step = max(1, int(np.ceil(max(H, W) / overview_max)))
    ov_rgb, ov_valid = source.read_rows(0, H, ov_step)
    bounds = scene_stretch_bounds(ov_rgb, spec.stretch_lo_pct, spec.stretch_hi_pct,
                                  valid=ov_valid) if spec.radiometric_stretch else None
    ov_rgb_s = stretch_scene(ov_rgb, spec, bounds=bounds)
    fill = (np.median(ov_rgb_s[ov_valid], axis=0) if ov_valid.any()
            else np.full(3, 127)).astype(np.uint8)

    # one tile of context above and below each band, in source pixels
    halo = int(np.ceil(spec.tile_size * spec.canonical_gsd_m / g))
    k = int(block_px) if block_px else 0
    if k:
        hb, wb = -(-H // k), -(-W // k)
        blk_sum = np.zeros((hb, wb), np.float64)
        blk_n = np.zeros((hb, wb), np.float64)

    ndsm_path = out_dir / "ndsm_m.tif"
    seg_path = out_dir / "seg.tif"
    std_path = out_dir / "ndsm_std_m.tif"       # opened on the first band that has one
    ds_std = None
    ov_h, ov_std = [], []
    stats_n, stats_sum = 0, 0.0
    lo_h, hi_h = np.inf, -np.inf
    below1 = 0

    with rasterio.open(ndsm_path, "w", **_geotiff_profile(source, "float32", np.nan)) as dn, \
            (rasterio.open(seg_path, "w", **_geotiff_profile(source, "uint8", 255))
             if want_seg else _NullCtx()) as ds_seg:
        bands = list(range(0, H, band_rows))
        for bi, r0 in enumerate(bands):
            r1 = min(H, r0 + band_rows)
            a0, a1 = max(0, r0 - halo), min(H, r1 + halo)
            rgb, valid = source.read_rows(a0, a1)
            rgb = stretch_scene(rgb, spec, bounds=bounds)
            rgb = _fill_invalid(rgb, valid, fill)
            hh, ww = rgb.shape[:2]
            canon_hw = gsd_to_shape(hh, ww, g, spec.canonical_gsd_m)
            rgb_c = resize(rgb, canon_hw, "bilinear")
            del rgb
            h_c, s_c, sd_c = predict_canonical(
                model, rgb_c, spec, device, tta=tta, tta_scales=tta_scales,
                amp_dtype=amp_dtype, overlap=overlap, batch_tiles=batch_tiles,
                want_seg=want_seg, want_std=True)
            del rgb_c
            h = resize(h_c, (hh, ww), "bilinear").astype(np.float32)[r0 - a0:r1 - a0]
            v = valid[r0 - a0:r1 - a0]
            h[~v] = np.nan
            dn.write(h, 1, window=Window(0, r0, W, r1 - r0))
            if sd_c is not None:
                sd = resize(sd_c, (hh, ww), "bilinear").astype(np.float32)[r0 - a0:r1 - a0]
                sd[~v] = np.nan
                if ds_std is None:
                    ds_std = rasterio.open(std_path, "w",
                                           **_geotiff_profile(source, "float32", np.nan))
                ds_std.write(sd, 1, window=Window(0, r0, W, r1 - r0))
                ov_std.append(sd[(-r0) % ov_step::ov_step, ::ov_step])
            if want_seg:
                s = resize(s_c.astype(np.int32), (hh, ww), "nearest")[r0 - a0:r1 - a0]
                s = np.where(v, s, 255).astype(np.uint8)
                ds_seg.write(s, 1, window=Window(0, r0, W, r1 - r0))
            # running statistics, overview rows, block sums
            fin = np.isfinite(h)
            if fin.any():
                hv = h[fin]
                stats_n += hv.size
                stats_sum += float(hv.sum())
                lo_h, hi_h = min(lo_h, float(hv.min())), max(hi_h, float(hv.max()))
                below1 += int((hv < 1.0).sum())
            first = (-r0) % ov_step
            ov_h.append(h[first::ov_step, ::ov_step])
            if k:
                _accumulate_blocks(h, r0, k, blk_sum, blk_n)
            if progress:
                progress(bi + 1, len(bands))

    if ds_std is not None:
        ds_std.close()
    height_ov = np.concatenate(ov_h, 0)[:ov_rgb.shape[0], :ov_rgb.shape[1]]
    std_ov = (np.concatenate(ov_std, 0)[:ov_rgb.shape[0], :ov_rgb.shape[1]]
              if ov_std and len(ov_std) == len(ov_h) else None)
    out = {
        "ndsm_path": str(ndsm_path), "seg_path": str(seg_path) if want_seg else None,
        "std_path": str(std_path) if ds_std is not None else None,
        "overview_std": std_ov,
        "overview_step": ov_step, "overview_height": height_ov,
        "overview_rgb": ov_rgb_s, "overview_valid": ov_valid,
        "stats": {"height_min_m": lo_h if stats_n else None,
                  "height_max_m": hi_h if stats_n else None,
                  "height_mean_m": stats_sum / stats_n if stats_n else None,
                  "frac_below_1m": below1 / stats_n if stats_n else None,
                  "valid_px": stats_n},
        "halo_px": halo, "band_rows": band_rows,
    }
    if k:
        with np.errstate(invalid="ignore"):
            out["block_mean"] = np.where(blk_n > 0, blk_sum / np.maximum(blk_n, 1), np.nan)
        out["block_px"] = k
    return out


def _accumulate_blocks(h: np.ndarray, r0: int, k: int, s: np.ndarray, n: np.ndarray):
    """Add the finite pixels of rows [r0, r0+len(h)) into k x k block sums."""
    rows = h.shape[0]
    W = h.shape[1]
    ridx = (np.arange(r0, r0 + rows) // k)
    cidx = (np.arange(W) // k)
    fin = np.isfinite(h)
    hv = np.where(fin, h, 0.0)
    for bi in np.unique(ridx):
        m = ridx == bi
        s[bi] += np.bincount(cidx, weights=hv[m].sum(0), minlength=s.shape[1])[:s.shape[1]]
        n[bi] += np.bincount(cidx, weights=fin[m].sum(0), minlength=n.shape[1])[:n.shape[1]]


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
