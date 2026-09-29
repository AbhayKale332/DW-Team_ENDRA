"""Standardised inference:  any image  ->  DSM in a standard geospatial format.

    # non-georeferenced (PNG/JPG)  -> relative DSM, you supply the scale
    python -m infer.predict scene.png --ckpt outputs/v3/best.pt --gsd 0.5

    # georeferenced GeoTIFF       -> absolute DSM, GSD read from the transform
    python -m infer.predict austin1.tif --ckpt outputs/v3/best.pt --tta

    # absolute DSM: terrain fetched automatically from Copernicus GLO-30
    python -m infer.predict scene.tif --ckpt outputs/v4/best.pt --absolute

    # ...or from a DEM you already have (CartoDEM, SRTM, anything GDAL reads)
    python -m infer.predict scene.tif --ckpt outputs/v4/best.pt --dem cartodem.tif

    # scale tied to surveyed control points instead of a DEM
    python -m infer.predict scene.tif --ckpt outputs/v4/best.pt --gcps gcps.csv

**Why this file exists in this shape.**  The model is only metric because its
input is at a known ground sample distance, normalised the way the encoder was
trained.  v2 had three different opinions about that: training normalised with
ImageNet constants, `predict_image.py`'s small-image path resized *any* PNG to
512x512 regardless of its GSD (destroying the metric scale outright), and the GSD
for a GeoTIFF had to be passed by hand.  Everything here instead goes through
`dwdata/preprocess.py` and `infer/engine.py`, and the normalisation constants,
canonical GSD and tile size are read from the **checkpoint**, not from a config
file that may have moved on.  If a checkpoint can be loaded, its inputs can be
reproduced exactly.

`--report` prints the resolved contract so it can be pasted into a bug report.

**v4 adds the georeferenced half.**  v3 produced an "absolute DSM" by adding the
coarse DEM to the predicted nDSM directly.  Copernicus GLO-30 and SRTM are
*surface* models, so that counts every building twice — a 40 m tower on 12 m
terrain came out near 80 m, quietly.  v4 routes it through `geo/calibrate.py`,
which fits the bare-earth DTM through ground pixels only (Head C's ground classes
AND a low predicted nDSM, both required) and adds the nDSM to that.  On a
synthetic scene with a 30 m tower the naive path gives 165 m, the calibrated path
gives 134.4 m against a true 135.0 m.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import parse_config  # noqa: E402
from dwdata.preprocess import PreprocSpec, read_scene  # noqa: E402
from geo.calibrate import refine_with_gcps  # noqa: E402
from infer.engine import predict_scene  # noqa: E402


# ---------------------------------------------------------------------
def load_model(ckpt: str, device, hf_token: str = ""):
    """Rebuild the network and its preprocessing spec from a checkpoint."""
    from models.heads import DepthWizardNet

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    spec = PreprocSpec.from_dict(ck["preproc"]) if "preproc" in ck else PreprocSpec()
    if "preproc" not in ck:
        print("[infer] checkpoint has no preproc block (pre-v3?) — using v3 defaults; "
              "metric scale may be wrong")

    cfg = parse_config([])
    for k, v in (ck.get("config") or {}).items():
        if hasattr(cfg, k) and not isinstance(getattr(cfg, k), tuple):
            setattr(cfg, k, v)
    cfg.hf_token = hf_token or cfg.hf_token
    cfg.encoder_model_id = spec.encoder_model_id
    cfg.tile_size = spec.tile_size
    cfg.canonical_gsd_m = spec.canonical_gsd_m
    cfg.grad_checkpoint_encoder = False

    model = DepthWizardNet(cfg).to(device).eval()
    miss, unexp = model.load_state_dict(ck["model"], strict=False)
    enc_missing = [m for m in miss if m.startswith("encoder.model.")]
    other_missing = [m for m in miss if not m.startswith("encoder.model.")]
    print(f"[infer] {ckpt}: epoch={ck.get('epoch')} "
          f"encoder_in_ckpt={ck.get('encoder_included', False)}")
    if other_missing:
        print(f"[infer] WARNING: {len(other_missing)} decoder/head tensors missing "
              f"from the checkpoint, e.g. {other_missing[:3]}")
    if enc_missing and not ck.get("encoder_included", False):
        print(f"[infer] encoder loaded from the hub ({len(enc_missing)} tensors) — "
              "expected when the encoder stayed frozen")
    if unexp:
        print(f"[infer] {len(unexp)} unexpected tensors ignored")
    return model, spec, cfg


# ---------------------------------------------------------------------
def to_absolute(height_m: np.ndarray, seg: np.ndarray | None, meta,
                *, dem_source: str, dem_path: str = "") -> tuple[np.ndarray, dict]:
    """nDSM (above ground) + coarse terrain -> absolute DSM, via a ground-only fit.

    The DEM is reprojected onto the prediction grid and then `geo.calibrate` fits
    the bare-earth surface through it at ground pixels only.  Adding the raw DEM
    instead — which is what v3 did — double-counts every structure, because both
    GLO-30 and SRTM already contain them.
    """
    from geo.calibrate import calibrate
    from geo.dem import load_dem_for_scene

    src = "local" if (dem_path and dem_source in ("", "local")) else dem_source
    res = load_dem_for_scene(meta, height_m.shape, source=src, local_path=dem_path)
    if res is None or res.coverage <= 0:
        return None, {"dem": (res.summary() if res else None),
                      "error": "no usable terrain coverage; nDSM returned unchanged"}
    cal = calibrate(height_m, res.array, seg)
    return cal.dsm_m, {"dem": res.summary(), "calibration": cal.info,
                       "dtm": cal.dtm_m, "ground_mask": cal.ground_mask}


def read_gcps(path: str) -> list[tuple[int, int, float]]:
    """CSV of `row,col,elevation_m` (a `#` or non-numeric first line is skipped)."""
    import csv

    out = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].strip().startswith("#"):
                continue
            try:
                out.append((int(float(row[0])), int(float(row[1])), float(row[2])))
            except (ValueError, IndexError):
                continue                                  # header line
    return out


def write_outputs(out_dir: Path, stem: str, rgb_u8, height_m, meta, spec,
                  dsm_abs=None, extra: dict | None = None, seg=None,
                  dtm=None, mesh: bool = True) -> dict:
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    lo, hi = float(np.nanmin(height_m)), float(np.nanmax(height_m))

    Image.fromarray(rgb_u8).save(out_dir / "rgb.png")
    # 16-bit PNG with an explicit affine encoding + the raw float32 array, so the
    # viewer can recover metres instead of guessing from a 0-255 ramp.
    span = max(hi - lo, 1e-6)
    Image.fromarray((((height_m - lo) / span) * 65535).astype(np.uint16)) \
        .save(out_dir / "ndsm16.png")
    np.save(out_dir / "ndsm_m.npy", height_m.astype(np.float32))

    wrote_tif = []
    if meta.georeferenced:
        try:
            import rasterio

            prof = dict(driver="GTiff", height=height_m.shape[0], width=height_m.shape[1],
                        count=1, dtype="float32", crs=meta.crs, transform=meta.transform,
                        compress="deflate", predictor=3, tiled=True, nodata=np.nan)
            with rasterio.open(out_dir / "ndsm_m.tif", "w", **prof) as d:
                d.write(height_m.astype(np.float32), 1)
                d.update_tags(1, UNITS="metres", PRODUCT="nDSM_above_ground")
            wrote_tif.append("ndsm_m.tif")
            if dsm_abs is not None:
                with rasterio.open(out_dir / "dsm_m.tif", "w", **prof) as d:
                    d.write(dsm_abs.astype(np.float32), 1)
                    d.update_tags(1, UNITS="metres_above_datum", PRODUCT="DSM_absolute")
                np.save(out_dir / "dsm_m.npy", dsm_abs.astype(np.float32))
                wrote_tif.append("dsm_m.tif")
            if dtm is not None:
                with rasterio.open(out_dir / "dtm_m.tif", "w", **prof) as d:
                    d.write(np.asarray(dtm, np.float32), 1)
                    d.update_tags(1, UNITS="metres_above_datum",
                                  PRODUCT="DTM_bare_earth_fitted")
                wrote_tif.append("dtm_m.tif")
        except Exception as e:  # noqa: BLE001
            print(f"[infer] GeoTIFF write skipped: {e}")
    else:
        print("[infer] input is not georeferenced -> relative DSM only "
              "(PNG/NPY); pass a GeoTIFF for a georeferenced product")

    if seg is not None:
        np.save(out_dir / "seg.npy", np.asarray(seg, np.int16))
        Image.fromarray(np.asarray(seg, np.uint8)).save(out_dir / "seg.png")

    mesh_files = []
    if mesh:
        try:
            from viz.mesh import write_mesh

            mesh_files = write_mesh(out_dir, height_m, rgb_u8,
                                    gsd_m=float(meta.gsd_m))
        except Exception as e:  # noqa: BLE001
            print(f"[infer] mesh export skipped: {e}")

    payload = {
        "stem": stem,
        "product": "rDSM" if not meta.georeferenced else "nDSM",
        "units": "metres_above_ground",
        "height_min_m": lo, "height_max_m": hi,
        "height_mean_m": float(np.nanmean(height_m)),
        "height_median_m": float(np.nanmedian(height_m)),
        "frac_below_1m": float((height_m < 1.0).mean()),
        "size_px": list(height_m.shape),
        "scene": meta.summary(),
        "preproc": spec.to_dict(),
        "files": ["rgb.png", "ndsm16.png", "ndsm_m.npy", *wrote_tif, *mesh_files],
        "ndsm16_encode": "height_m = height_min_m + (png16/65535)*(height_max_m-height_min_m)",
    }
    if extra:
        payload.update(extra)
    (out_dir / "meta.json").write_text(json.dumps(payload, indent=2))

    print(f"\nheight: {height_m.shape}  min={lo:.2f} m  max={hi:.2f} m  "
          f"mean={payload['height_mean_m']:.2f} m  median={payload['height_median_m']:.2f} m")
    print(f"        {payload['frac_below_1m'] * 100:.1f}% of pixels below 1 m "
          f"(a nadir urban/suburban scene is usually 40-70%)")
    if payload["frac_below_1m"] < 0.15:
        print("        WARNING: very little flat ground — the model may be reading "
              "texture as terrain on this imagery (check --gsd and the stretch)")
    print(f"saved -> {out_dir}")
    return payload


# ---------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="DepthWizard v3 inference")
    ap.add_argument("image")
    ap.add_argument("--ckpt", default="outputs/v3/best.pt")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--gsd", type=float, default=0.0,
                    help="metres/pixel of the input. Overrides a GeoTIFF's own "
                         "transform. Required for PNG/JPG to be metric.")
    ap.add_argument("--assumed-gsd", type=float, default=0.5,
                    help="fallback when the input carries no scale (default: the "
                         "model's canonical GSD, i.e. treat 1 px as 0.5 m)")
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--batch-tiles", type=int, default=4)
    ap.add_argument("--tta", action="store_true", help="8x dihedral TTA (~8x slower)")
    ap.add_argument("--tta-scales", default="1.0")
    ap.add_argument("--dem", default="",
                    help="local coarse DEM (CartoDEM/SRTM/anything GDAL reads)")
    ap.add_argument("--absolute", action="store_true",
                    help="fetch terrain automatically and output an absolute DSM")
    ap.add_argument("--dem-source", default="copernicus30",
                    choices=["copernicus30", "srtm30", "cartodem", "local"])
    ap.add_argument("--gcps", default="",
                    help="CSV of row,col,elevation_m -> RANSAC scale+offset refinement")
    ap.add_argument("--no-mesh", action="store_true",
                    help="skip the OBJ/glTF export for the 3D viewer")
    ap.add_argument("--max-side", type=int, default=0, help="downsample huge scenes first")
    ap.add_argument("--device", default="")
    ap.add_argument("--hf-token", default="")
    ap.add_argument("--report", action="store_true", help="print the contract and exit")
    a = ap.parse_args()

    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[infer] device: {device}")
    model, spec, cfg = load_model(a.ckpt, device, a.hf_token)

    rgb, meta = read_scene(a.image, user_gsd_m=a.gsd,
                           assumed_gsd_m=a.assumed_gsd or spec.canonical_gsd_m,
                           max_side=a.max_side)
    print(f"[infer] scene {meta.width}x{meta.height}  gsd={meta.gsd_m:.4f} m "
          f"({meta.gsd_source})  georeferenced={meta.georeferenced}")
    if meta.gsd_source == "assumed":
        print("[infer] NOTE: no scale metadata — heights are relative (rDSM). "
              "Pass --gsd for metric output.")
    print(f"[infer] contract: stretch={spec.radiometric_stretch} "
          f"({spec.stretch_lo_pct}/{spec.stretch_hi_pct} pct) -> "
          f"{spec.canonical_gsd_m} m/px -> {spec.tile_size}px tiles -> "
          f"mean={tuple(round(v, 3) for v in spec.mean)} std={tuple(round(v, 3) for v in spec.std)}")
    if a.report:
        return

    scales = tuple(float(s) for s in a.tta_scales.split(",") if s.strip())
    amp_dt = torch.bfloat16 if (device.type == "cuda" and cfg.amp_dtype == "bf16") \
        else (torch.float16 if device.type == "cuda" else None)

    def prog(done, total):
        if done == total or done % (max(1, total // 10)) < a.batch_tiles:
            print(f"  tiles {done}/{total}", flush=True)

    height, seg = predict_scene(
        model, rgb, meta.gsd_m, spec, device, tta=a.tta, tta_scales=scales,
        amp_dtype=amp_dt, overlap=a.overlap, batch_tiles=a.batch_tiles,
        want_seg=True, progress=prog,
    )

    dsm_abs, dtm, extra = None, None, {}
    if a.dem or a.absolute:
        if not meta.georeferenced and not a.dem:
            print("[infer] --absolute needs a georeferenced input (or pass --dem); "
                  "skipping the terrain step")
        else:
            dsm_abs, extra = to_absolute(height, seg, meta,
                                         dem_source=a.dem_source, dem_path=a.dem)
            dtm = extra.pop("dtm", None)
            extra.pop("ground_mask", None)
            if dsm_abs is not None:
                print(f"[infer] absolute DSM: {np.nanmin(dsm_abs):.1f}.."
                      f"{np.nanmax(dsm_abs):.1f} m")

    if a.gcps:
        target = dsm_abs if dsm_abs is not None else height
        refined, ginfo = refine_with_gcps(target, read_gcps(a.gcps))
        extra["gcp_refinement"] = ginfo
        print(f"[infer] GCPs: {ginfo}")
        if dsm_abs is not None:
            dsm_abs = refined
        else:
            height = refined

    out_dir = Path(a.out_dir) if a.out_dir else \
        Path(cfg.output_dir) / "predict" / Path(a.image).stem
    write_outputs(out_dir, Path(a.image).stem, rgb, height, meta, spec, dsm_abs,
                  extra, seg=seg, dtm=dtm, mesh=not a.no_mesh)


if __name__ == "__main__":
    main()
