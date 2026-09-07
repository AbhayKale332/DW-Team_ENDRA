"""IMELE (IM2ELEVATION) baseline row on the GAMUS val split.

Phase 0 wants an apples-to-apples baseline: "Run existing IMELE notebook on the
same val split -> baseline row" (`.agents/Depth_Wizard_Plan.md`).

Workflow
--------
1. Run `Notebooks/depth-wiz.ipynb` (IMELE inference) with its `IMAGE_INPUT`
   pointed at the GAMUS val RGB tiles this project downloads, i.e.
       /kaggle/working/data/gamus/images/val/       (*_RGB.h5)
   The notebook currently reads image files; for GAMUS you either
     (a) dump the val RGBs to PNG first (see `dump_gamus_val_pngs` below), or
     (b) adapt its loader to read the *_RGB.h5 "image" key.
   It writes `<stem>_imele_prediction.npy` per tile into `imele_outputs/`.

2. Point this script at that folder + the GAMUS root:
       python eval_imele_on_gamus.py \
           --pred_dir /kaggle/working/imele_outputs \
           --data_root /kaggle/working/data/gamus \
           --out /kaggle/working/outputs/v1/imele_baseline.json

It reuses the exact metric code from `kaggle_phase0.py` so the numbers line up
with the DepthWizardNet row.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("DEPTHWIZ_SKIP_INSTALL", "1")
from kaggle_phase0 import Config, MetricAccum, _read_h5  # noqa: E402


def dump_gamus_val_pngs(data_root: str, out_dir: str) -> None:
    """Helper: write GAMUS val RGB tiles to PNG for the IMELE notebook."""
    from PIL import Image

    src = Path(data_root) / "images" / "val"
    dst = Path(out_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(src.glob("*_RGB.h5")):
        stem = p.name[: -len("_RGB.h5")]
        Image.fromarray(_read_h5(p).astype(np.uint8)).save(dst / f"{stem}.png")
    print(f"wrote PNGs -> {dst}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, help="dir of <stem>_imele_prediction.npy")
    ap.add_argument("--data_root", default=Config().data_root)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="imele_baseline.json")
    ap.add_argument("--align_scale", action="store_true",
                    help="least-squares fit a global scale+offset per tile before scoring "
                         "(IMELE was trained at a different GSD / height distribution)")
    args = ap.parse_args()

    cfg = Config()
    root = Path(args.data_root)
    gm = MetricAccum()
    cm = {i: MetricAccum() for i in range(len(cfg.class_names))}
    n_tiles = 0

    for npy in sorted(Path(args.pred_dir).glob("*_prediction.npy")):
        stem = npy.name.replace("_imele_prediction.npy", "").replace("_prediction.npy", "")
        agl_p = root / "heights" / args.split / f"{stem}_AGL.h5"
        if not agl_p.exists():
            continue
        pred = np.load(npy).astype(np.float32)
        gt = _read_h5(agl_p).astype(np.float32)
        cls = (
            _read_h5(root / "classes" / args.split / f"{stem}_CLS.h5").astype(np.int64)
            if (root / "classes" / args.split / f"{stem}_CLS.h5").exists()
            else np.zeros(gt.shape, np.int64)
        )
        if pred.shape != gt.shape:  # notebook may tile/upsample differently
            from PIL import Image

            pred = np.asarray(
                Image.fromarray(pred).resize(gt.shape[::-1], Image.BILINEAR), np.float32
            )

        valid = np.isfinite(gt) & (gt >= 0) & (gt <= cfg.max_valid_height_m) & np.isfinite(pred)
        p, t = pred[valid], gt[valid]
        if p.size == 0:
            continue
        if args.align_scale:
            A = np.stack([p, np.ones_like(p)], 1)
            s, b = np.linalg.lstsq(A, t, rcond=None)[0]
            p = s * p + b

        import torch

        gm.update(torch.from_numpy(p), torch.from_numpy(t))
        c = cls[valid]
        for ci, acc in cm.items():
            m = c == ci
            if m.any():
                acc.update(torch.from_numpy(p[m]), torch.from_numpy(t[m]))
        n_tiles += 1

    result = {
        "model": "IMELE (IM2ELEVATION, public Dublin checkpoint)",
        "split": args.split,
        "n_tiles": n_tiles,
        "align_scale": args.align_scale,
        "global": gm.result(),
        "per_class": {cfg.class_names[i]: a.result() for i, a in cm.items() if a.n > 0},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    g = result["global"]
    print(f"\nIMELE baseline on GAMUS {args.split} ({n_tiles} tiles):")
    print(f"  RMSE={g.get('rmse_m'):.3f} m  MAE={g.get('mae_m'):.3f} m  "
          f"r={g.get('pearson_r'):.3f}  delta1={g.get('delta1'):.3f}")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
