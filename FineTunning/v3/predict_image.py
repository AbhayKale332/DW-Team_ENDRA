"""Single-image prediction, packaged for the Phase-0 3D viewer.

    # simplest form: everything defaults to the repo-root checkpoint + image
    python predict_image.py /teamspace/studios/this_studio/image.png \
        --ckpt /teamspace/studios/this_studio/best.pt

    # you know the ground sample distance -> the heights become metric
    python predict_image.py scene.png --ckpt best.pt --gsd 0.3

    # fastest look, no 8x dihedral averaging
    python predict_image.py scene.png --ckpt best.pt --no-tta

This is the v3 replacement for v2's `predict_image.py`, and it is a thin shell
around `infer/predict.py`'s machinery rather than a second implementation:

  * the preprocessing contract (encoder mean/std, canonical GSD, tile size,
    radiometric stretch) is read from the **checkpoint** via `PreprocSpec`, so
    this script cannot drift from training the way v2's did;
  * the image is *resampled to the canonical GSD and tiled*, never squashed to
    512x512 — v2's small-image path destroyed the metric scale outright;
  * inference goes through `infer.engine.predict_scene`, the same code path the
    final evaluation uses.

What it adds over `python -m infer.predict` is the **viewer contract**.
`DepthWizard/viewer/phase0_viewer.html` matches its file picker on substrings
("rgb" / "height") and decodes height with

    h_m = meta.height_min_m + (red/255) * (meta.height_max_m - meta.height_min_m)
    y_px = h_m / meta.gsd_m

so the export directory here is named to be picked up, and the 0-255 ramp is
stretched over a *robust* (percentile) height range instead of the raw min/max.
That matters: one hot pixel of LiDAR-noise-shaped prediction at 60 m would
otherwise compress a whole 15 m suburb into the bottom four grey levels.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from dwdata.preprocess import read_scene  # noqa: E402
from infer.engine import predict_scene  # noqa: E402
from infer.predict import load_model  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
_VIEWER_HTML = _HERE.parents[1] / "viewer" / "phase0_viewer.html"


def robust_range(h: np.ndarray, lo_pct: float, hi_pct: float) -> tuple[float, float]:
    """Percentile height range used for the 8/16-bit display ramp.

    Clamped so the low end never rises above 0 m: ground *is* the reference
    surface of an nDSM and the viewer should draw it flat, not floating.
    """
    lo = float(np.nanpercentile(h, lo_pct)) if lo_pct > 0 else float(np.nanmin(h))
    hi = float(np.nanpercentile(h, hi_pct)) if hi_pct < 100 else float(np.nanmax(h))
    lo = min(lo, 0.0)
    return lo, max(hi, lo + 1e-3)


def write_viewer_export(out_dir: Path, stem: str, rgb_u8: np.ndarray,
                        height_m: np.ndarray, gsd_m: float, spec, meta,
                        clip_pct: tuple[float, float], elapsed_s: float) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    H, W = height_m.shape
    lo, hi = robust_range(height_m, *clip_pct)
    span = hi - lo

    # Height is encoded as 8-bit RGB, not a 16-bit greyscale PNG: R is the high
    # byte and G the low byte of a 16-bit fixed-point ramp.  The viewer reads
    # exactly one 8-bit channel back out of a 2D canvas (R), so it sees the plain
    # 0-255 ramp it expects, while R*256+G still recovers the full 16 bits for
    # any other consumer.  A true 16-bit PNG would leave the 16->8 downconversion
    # up to the browser's image decoder, which is not worth depending on.
    norm = np.clip((height_m - lo) / span, 0.0, 1.0)
    q16 = (norm * 65535.0 + 0.5).astype(np.uint16)
    enc = np.stack([(q16 >> 8).astype(np.uint8),
                    (q16 & 0xFF).astype(np.uint8),
                    np.zeros(q16.shape, np.uint8)], axis=-1)
    Image.fromarray(rgb_u8).save(out_dir / "rgb.png")
    Image.fromarray(enc, mode="RGB").save(out_dir / "height16.png")
    np.save(out_dir / "height_m.npy", height_m.astype(np.float32))

    payload = {
        # --- consumed by phase0_viewer.html -------------------------
        "stem": stem,
        "height_min_m": lo,
        "height_max_m": hi,
        "gsd_m": float(gsd_m),
        "size_px": [W, H],
        # --- provenance / diagnostics -------------------------------
        "product": "nDSM" if meta.gsd_source != "assumed" else "rDSM",
        "units": "metres_above_ground",
        "display_clip_pct": list(clip_pct),
        "height_true_min_m": float(np.nanmin(height_m)),
        "height_true_max_m": float(np.nanmax(height_m)),
        "height_mean_m": float(np.nanmean(height_m)),
        "height_median_m": float(np.nanmedian(height_m)),
        "frac_below_1m": float((height_m < 1.0).mean()),
        "vertical_quantum_m": span / 255.0,
        "scene": meta.summary(),
        "preproc": spec.to_dict(),
        "elapsed_s": round(elapsed_s, 1),
        "files": ["rgb.png", "height16.png", "height_m.npy"],
        "height_encode": "q = R*256+G (or just R/255 for 8-bit); "
                          "h_m = height_min_m + (q/65535)*(height_max_m-height_min_m)",
    }
    (out_dir / "meta.json").write_text(json.dumps(payload, indent=2))
    return out_dir, payload


def report(payload: dict, out_dir: Path) -> None:
    p = payload
    print(f"\nheight  {p['size_px'][0]}x{p['size_px'][1]}px @ {p['gsd_m']:.3f} m/px "
          f"({p['scene']['gsd_source']})")
    print(f"        true range   {p['height_true_min_m']:+.2f} .. {p['height_true_max_m']:+.2f} m")
    print(f"        display ramp {p['height_min_m']:+.2f} .. {p['height_max_m']:+.2f} m "
          f"({p['display_clip_pct'][0]}/{p['display_clip_pct'][1]} pct, "
          f"{p['vertical_quantum_m']:.3f} m per grey level)")
    print(f"        mean {p['height_mean_m']:.2f} m · median {p['height_median_m']:.2f} m · "
          f"{p['frac_below_1m'] * 100:.1f}% below 1 m")
    if p["frac_below_1m"] < 0.15:
        print("        WARNING: little flat ground — check --gsd; the model may be "
              "reading texture as terrain at this scale")
    if p["product"] == "rDSM":
        print("        NOTE: no scale metadata and no --gsd, so the GSD was assumed. "
              "Shapes are right, absolute metres are not.")
    print(f"\nsaved -> {out_dir}")
    print(f"  {', '.join(p['files'])}, meta.json")
    print("\nopen the viewer and pick rgb.png + height16.png + meta.json:")
    print(f"  python -m http.server -d {_VIEWER_HTML.parent} 8000")
    print(f"  then http://localhost:8000/{_VIEWER_HTML.name}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="v3 single-image prediction -> Phase-0 viewer export")
    ap.add_argument("image", nargs="?", default="/teamspace/studios/this_studio/image.png")
    ap.add_argument("--ckpt", default="/teamspace/studios/this_studio/best.pt")
    ap.add_argument("--out-dir", default=None,
                    help="default: <output_dir>/viewer_sample/<stem>")
    ap.add_argument("--gsd", type=float, default=0.0,
                    help="metres per pixel of the input; makes the output metric")
    ap.add_argument("--assumed-gsd", type=float, default=0.5,
                    help="fallback GSD when the image carries no scale "
                         "(default 0.5 = the model's canonical GSD, i.e. 1 px = 0.5 m)")
    ap.add_argument("--tta", dest="tta", action="store_true", default=True,
                    help="8x dihedral TTA (default on — nadir imagery has no up)")
    ap.add_argument("--no-tta", dest="tta", action="store_false")
    ap.add_argument("--tta-scales", default="1.0")
    ap.add_argument("--overlap", type=float, default=0.5,
                    help="tile overlap; 0.5 gives a smoother Hann blend than eval's 0.25")
    ap.add_argument("--batch-tiles", type=int, default=4)
    ap.add_argument("--max-side", type=int, default=0, help="downsample huge scenes first")
    ap.add_argument("--clip-pct", default="0.5,99.5",
                    help="percentiles for the display ramp (use 0,100 for raw min/max)")
    ap.add_argument("--device", default="")
    ap.add_argument("--hf-token", default="")
    a = ap.parse_args()

    clip = tuple(float(v) for v in a.clip_pct.split(","))
    assert len(clip) == 2 and clip[0] < clip[1], "--clip-pct wants lo,hi"

    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[predict] device: {device}")
    model, spec, cfg = load_model(a.ckpt, device, a.hf_token)

    rgb, meta = read_scene(a.image, user_gsd_m=a.gsd,
                           assumed_gsd_m=a.assumed_gsd or spec.canonical_gsd_m,
                           max_side=a.max_side)
    print(f"[predict] {a.image}: {meta.width}x{meta.height} @ {meta.gsd_m:.4f} m/px "
          f"({meta.gsd_source}) -> canonical {spec.canonical_gsd_m} m/px, "
          f"{spec.tile_size}px tiles, tta={a.tta}")
    if device.type == "cpu" and a.tta:
        print("[predict] CPU + TTA means 8 forward passes per tile; --no-tta for a quick look")

    scales = tuple(float(s) for s in a.tta_scales.split(",") if s.strip())
    amp_dt = (torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16) \
        if device.type == "cuda" else None

    def prog(done, total):
        print(f"  tiles {done}/{total}", flush=True)

    t0 = time.time()
    height, _ = predict_scene(
        model, rgb, meta.gsd_m, spec, device, tta=a.tta, tta_scales=scales,
        amp_dtype=amp_dt, overlap=a.overlap, batch_tiles=a.batch_tiles, progress=prog,
    )
    elapsed = time.time() - t0

    stem = Path(a.image).stem
    out_dir = Path(a.out_dir) if a.out_dir else \
        Path(cfg.output_dir) / "viewer_sample" / stem
    out_dir, payload = write_viewer_export(
        out_dir, stem, rgb, height, meta.gsd_m, spec, meta, clip, elapsed)
    report(payload, out_dir)


if __name__ == "__main__":
    main()
