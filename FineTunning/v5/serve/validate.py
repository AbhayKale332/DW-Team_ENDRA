"""In-app validation and full-resolution AOIs for a finished job (plan C2, C6).

Both work on a job directory as `infer.predict.write_outputs` leaves it
(`meta.json`, `ndsm_m.npy`, optionally `dsm_m.npy` / `dtm_m.npy` /
`ndsm_std_m.npy`, and for a windowed run the full-resolution GeoTIFFs listed in
`meta["full_resolution"]`), so they serve the CLI's output folders as well as
the web service's jobs.

**`validate_reference`** — the problem statement's own deliverable: "validate
against a reference DSM".  Any GDAL-readable raster is
  1. reprojected onto the prediction grid (`rasterio.warp.reproject`, bilinear,
     its NoData respected) — or, for a job without georeferencing, resized onto
     it and flagged as pixel-aligned by assumption;
  2. classified as an absolute DSM or an nDSM (`kind="auto"` picks whichever of
     our two surfaces its median is closer to);
  3. converted to our DSM's vertical datum when both datums are known
     (`geo.dem.to_datum`; an ellipsoid/geoid mix-up is tens of metres);
  4. scored per pixel, on "confident" pixels only (Head B's spread under the
     threshold in `meta["uncertainty"]`), and per 30 m cell — the two readings of
     the judges' protocol (Host_Questions_Draft.md Q1).
It writes `gt_ndsm_m.npy` or `gt_dsm_m.npy` (the viewer's reference layers),
`err_ref_m.npy` and `validation.json`.

**`extract_aoi`** — a windowed (20k x 20k) job's viewer arrays are a <= 2048 px
overview.  This cuts a window, given in overview pixels, out of the
full-resolution products and the source image, and writes it as its own
product directory (`aoi_<row>_<col>_<h>_<w>/`) that the viewer loads like any
other.  Windows larger than `max_px` a side are decimated to fit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_meta(job: Path) -> dict:
    p = job / "meta.json"
    if not p.is_file():
        raise FileNotFoundError(f"{job} has no meta.json — not a finished product")
    return json.loads(p.read_text())


def _grid(meta: dict):
    """(Affine, CRS) of the viewer arrays, or (None, None)."""
    sc = meta.get("scene") or {}
    t = sc.get("transform")
    if not (sc.get("georeferenced") and t and (sc.get("crs_epsg") or sc.get("crs"))):
        return None, None
    from affine import Affine
    from rasterio.crs import CRS

    crs = CRS.from_epsg(sc["crs_epsg"]) if sc.get("crs_epsg") else CRS.from_user_input(sc["crs"])
    return Affine(*t[:6]), crs


def _opt(job: Path, name: str):
    p = job / name
    return np.load(p) if p.is_file() else None


def reference_on_grid(ref_path: str, shape, transform=None, crs=None) -> tuple[np.ndarray, str]:
    """Band 1 of `ref_path` on our grid; NaN where the reference has no data."""
    import rasterio

    with rasterio.open(ref_path) as src:
        nd = src.nodata
        if transform is not None and src.crs is not None:
            from rasterio.warp import Resampling, reproject

            out = np.full(shape, np.nan, np.float32)
            reproject(rasterio.band(src, 1), out, src_nodata=nd, dst_transform=transform,
                      dst_crs=crs, dst_nodata=np.nan, resampling=Resampling.bilinear)
            return out, "reprojected"
        a = src.read(1, out_shape=shape, resampling=rasterio.enums.Resampling.bilinear
                     ).astype(np.float32)
    if nd is not None:
        a[a == nd] = np.nan
    return a, "resized (no georeferencing on one side — pixel-aligned by assumption)"


def _cells(pred, ref, k: int):
    from geo.calibrate import block_mean

    both = np.isfinite(pred) & np.isfinite(ref)
    return block_mean(pred, k, both), block_mean(ref, k, both)


def validate_reference(job: str | Path, ref_path: str, *, kind: str = "auto",
                       ref_datum: str = "", anchor_cell_m: float = 30.0) -> dict:
    from eval.judge_proxy import _Acc
    from geo.calibrate import anchor_cell_px

    job = Path(job)
    meta = _load_meta(job)
    ndsm = np.load(job / "ndsm_m.npy").astype(np.float32)
    dsm, std = _opt(job, "dsm_m.npy"), _opt(job, "ndsm_std_m.npy")
    tr, crs = _grid(meta)
    ref, how = reference_on_grid(ref_path, ndsm.shape, tr, crs)
    fin = np.isfinite(ref)
    if not fin.any():
        raise ValueError("the reference does not overlap this scene")

    if kind == "auto":
        m = float(np.median(ref[fin]))
        kind = "ndsm"
        if dsm is not None and np.isfinite(dsm).any():
            if abs(m - np.nanmedian(dsm)) < abs(m - np.nanmedian(ndsm)):
                kind = "dsm"
    if kind == "dsm" and dsm is None:
        raise ValueError("reference is an absolute DSM but this job has none — "
                         "re-run with 'absolute DSM' on")

    out: dict = {"reference": Path(ref_path).name, "kind": kind, "placement": how,
                 "overlap_frac": float(fin.mean())}
    if kind == "dsm":
        ours = meta.get("vertical_datum") or ""
        if ref_datum and ours and ref_datum != ours and tr is not None:
            from geo.dem import to_datum

            ref, info = to_datum(ref, tr, crs, ref_datum, ours)
            out["datum_conversion"] = info
        else:
            out["datum_conversion"] = {"from": ref_datum or "unknown", "to": ours or "unknown",
                                       "applied": False,
                                       "note": "same datum" if ref_datum == ours and ours
                                       else "a datum is unknown — compared as delivered"}
    pred = dsm if kind == "dsm" else ndsm

    err = (pred - ref).astype(np.float32)
    allp = _Acc()
    allp.add(pred, ref)
    out["per_pixel"] = allp.result()
    unc = meta.get("uncertainty") or {}
    if std is not None and std.shape == pred.shape and unc.get("confident_threshold_m"):
        conf = std <= float(unc["confident_threshold_m"])
        c = _Acc()
        c.add(np.where(conf, pred, np.nan), ref)
        out["per_pixel_confident"] = {**c.result(), "threshold_m": unc["confident_threshold_m"]}
    gsd = float((meta.get("scene") or {}).get("gsd_m") or 0.5)
    k = anchor_cell_px(gsd, anchor_cell_m)
    if min(pred.shape) >= 2 * k:
        pc, rc = _cells(pred, ref, k)
        cc = _Acc()
        cc.add(pc, rc)
        out["per_cell"] = {**cc.result(), "cell_m": anchor_cell_m, "cell_px": k}
    ground = np.isfinite(ndsm) & (ndsm < 1.0) & np.isfinite(err)
    if ground.any():
        out["median_err_on_ground_m"] = float(np.median(err[ground]))

    np.save(job / f"gt_{kind}_m.npy", ref.astype(np.float32))
    np.save(job / "err_ref_m.npy", err)
    stale = job / f"gt_{'dsm' if kind == 'ndsm' else 'ndsm'}_m.npy"
    if stale.is_file():                        # one reference at a time
        stale.unlink()
    out["files"] = [f"gt_{kind}_m.npy", "err_ref_m.npy", "validation.json"]
    (job / "validation.json").write_text(json.dumps(out, indent=2))
    # keep meta.json's file list true: the viewer only requests listed files
    files = [f for f in meta.get("files", []) if not f.startswith("gt_")]
    meta["files"] = files + [f for f in out["files"] if f not in files]
    (job / "meta.json").write_text(json.dumps(meta, indent=2))
    return out


# ---------------------------------------------------------------------
# full-resolution AOIs
# ---------------------------------------------------------------------
def _read_tif(path: str, win, out_shape):
    import rasterio

    with rasterio.open(path) as d:
        return d.read(1, window=win, out_shape=out_shape,
                      resampling=rasterio.enums.Resampling.average).astype(np.float32)


def _input_path(job: Path, meta: dict):
    """The scene the job was run on: the service's upload, else meta's path."""
    for p in (job / "input", *sorted(job.glob("input.*"))):
        if p.exists():
            return p
    p = (meta.get("scene") or {}).get("path")
    return Path(p) if p and Path(p).exists() else None


def extract_aoi(job: str | Path, row: int, col: int, h: int, w: int, *,
                max_px: int = 2048, spec=None) -> dict:
    """Cut (row, col, h, w) — in the job's *viewer* pixels — at full resolution."""
    from affine import Affine
    from PIL import Image
    from rasterio.windows import Window

    from dwdata.preprocess import PreprocSpec, SceneMeta, scene_stretch_bounds, stretch_scene
    from infer.predict import write_outputs

    job = Path(job)
    meta = _load_meta(job)
    sc = meta.get("scene") or {}
    step = int((meta.get("windowed") or {}).get("overview_step") or 1)
    full = meta.get("full_resolution") or {}
    Hv, Wv = np.load(job / "ndsm_m.npy", mmap_mode="r").shape
    row, col = max(0, int(row)), max(0, int(col))
    h, w = min(int(h), Hv - row), min(int(w), Wv - col)
    if h < 8 or w < 8:
        raise ValueError("AOI too small (need at least 8 x 8 viewer pixels)")
    R0, C0, FH, FW = row * step, col * step, h * step, w * step
    if full.get("ndsm"):
        import rasterio

        with rasterio.open(full["ndsm"]) as d:
            FH, FW = min(FH, d.height - R0), min(FW, d.width - C0)
    d = max(1, int(np.ceil(max(FH, FW) / max_px)))
    # the decimated source read samples columns 0, d, 2d, ...: start on one
    FW, C0 = FW + C0 % d, C0 - C0 % d
    oh, ow = -(-FH // d), -(-FW // d)
    win = Window(C0, R0, FW, FH)

    def layer(key: str, npy: str):
        if full.get(key) and Path(full[key]).is_file():
            return _read_tif(full[key], win, (oh, ow))
        a = _opt(job, npy)
        return None if a is None else \
            a[R0:R0 + FH:d, C0:C0 + FW:d][:oh, :ow].astype(np.float32)

    ndsm = layer("ndsm", "ndsm_m.npy")
    dsm, dtm, std = layer("dsm", "dsm_m.npy"), layer("dtm", "dtm_m.npy"), layer("std",
                                                                                  "ndsm_std_m.npy")
    valid = np.isfinite(ndsm)

    rgb = None
    src_path = _input_path(job, meta) if step > 1 else None
    if src_path is not None:
        from dwdata.preprocess import open_scene

        src = open_scene(src_path, None, float(sc.get("gsd_m", 0.5)) / step)
        u8, v = src.read_rows(R0, R0 + FH, d)
        u8 = u8[:, C0 // d:C0 // d + ow][:oh, :ow]
        v = v[:, C0 // d:C0 // d + ow][:oh, :ow]
        spec = spec or PreprocSpec.from_dict(meta.get("preproc") or {})
        b = scene_stretch_bounds(u8, spec.stretch_lo_pct, spec.stretch_hi_pct, valid=v)
        rgb = stretch_scene(u8, spec, bounds=b) if spec.radiometric_stretch else u8
    if rgb is None or rgb.shape[:2] != ndsm.shape:
        im = np.asarray(Image.open(job / "rgb.png").convert("RGB"))
        rgb = im[row:row + h, col:col + w] if step > 1 else \
            im[R0:R0 + FH:d, C0:C0 + FW:d][:oh, :ow]
        if rgb.shape[:2] != ndsm.shape:
            from dwdata.preprocess import resize

            rgb = resize(rgb, ndsm.shape, "bilinear")

    tr, crs = _grid(meta)
    gsd_full = float(sc.get("gsd_m", 0.5)) / step
    ameta = SceneMeta(gsd_m=gsd_full * d, gsd_source=sc.get("gsd_source", "geotiff"),
                      georeferenced=tr is not None,
                      transform=(tr * Affine.scale(1 / step) * Affine.translation(C0, R0)
                                 * Affine.scale(d)) if tr is not None else None,
                      crs=crs, width=ndsm.shape[1], height=ndsm.shape[0],
                      path=str(sc.get("path", "")), valid=valid,
                      source=sc.get("source") or {})
    name = f"aoi_{R0}_{C0}_{FH}_{FW}"
    out = job / name
    spec = spec or PreprocSpec.from_dict(meta.get("preproc") or {})
    payload = write_outputs(out, name, rgb, ndsm, ameta, spec, dsm_abs=dsm,
                            extra={"aoi": {"parent": meta.get("stem"), "row": R0, "col": C0,
                                           "height": FH, "width": FW, "decimation": d,
                                           "overview_window": [row, col, h, w]}},
                            dtm=dtm, mesh=False, datum=meta.get("vertical_datum") or "",
                            geotiff=False, std=std)
    return {"dir": name, "shape": list(ndsm.shape), "decimation": d,
            "window_full_res": [R0, C0, FH, FW], "meta": payload}
