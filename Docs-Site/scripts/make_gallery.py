"""Render web-sized gallery images from model output folders.

Run from Docs-Site/:  python3 scripts/make_gallery.py
Colour-maps float height rasters (turbo, 2-98 % range unless given) and writes
JPEG/PNG files into src/assets/gallery/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from PIL import Image

DOCS = Path(__file__).resolve().parents[1]
ROOT = DOCS.parent
OUT = DOCS / "src/assets/gallery"
OUT.mkdir(parents=True, exist_ok=True)
SIZE = 640

SEG_COLORS = {  # Frontend/src/theme/classes.ts, by class id (config.CLASS_NAMES order)
    0: (0x9a, 0xa0, 0xa6), 1: (0xc8, 0xb5, 0x8a), 2: (0x9c, 0xcc, 0x65), 3: (0xe0, 0x5a, 0x47),
    4: (0x3a, 0x8f, 0xd9), 5: (0x5f, 0x63, 0x68), 6: (0x2e, 0x7d, 0x32), 7: (0, 0, 0),
}


def save_rgb(img: Image.Image, name: str):
    img = img.convert("RGB")
    img.thumbnail((SIZE, SIZE), Image.LANCZOS)
    img.save(OUT / name, quality=88)
    print("wrote", name)


def colorize(h: np.ndarray, name: str, cmap="turbo", vmin=None, vmax=None):
    h = np.asarray(h, dtype=np.float32)
    valid = np.isfinite(h)
    lo = np.nanpercentile(h[valid], 2) if vmin is None else vmin
    hi = np.nanpercentile(h[valid], 98) if vmax is None else vmax
    t = np.clip((np.nan_to_num(h, nan=lo) - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = (colormaps[cmap](t)[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = 0
    save_rgb(Image.fromarray(rgb), name)
    return float(lo), float(hi)


def seg_color(seg: np.ndarray, name: str):
    rgb = np.zeros((*seg.shape, 3), np.uint8)
    for k, c in SEG_COLORS.items():
        rgb[seg == k] = c
    save_rgb(Image.fromarray(rgb), name)


ranges = {}

# Cartosat-2E georeferenced job (absolute DSM)
job = ROOT / "Model_Traning/v5/outputs/jobs/88e63fd3329e"
save_rgb(Image.open(job / "rgb.png"), "carto_rgb.jpg")
ranges["carto_ndsm"] = colorize(np.load(job / "ndsm_m.npy"), "carto_ndsm.jpg", vmin=0)
ranges["carto_dsm"] = colorize(np.load(job / "dsm_m.npy"), "carto_dsm.jpg", cmap="terrain")
seg_color(np.load(job / "seg.npy"), "carto_seg.png")

# Cartosat-2S campus sample bundled with the frontend
camp = ROOT / "Frontend/public/samples/buildings_large_campus"
if camp.exists():
    save_rgb(Image.open(camp / "rgb.png"), "campus_rgb.jpg")
    ranges["campus_ndsm"] = colorize(np.load(camp / "ndsm_m.npy"), "campus_ndsm.jpg", vmin=0)

# MVS3DM LiDAR comparison sites
for site in ["Explorer", "MasterSequestered3"]:
    d = ROOT / "cartosat_2S_Sample/mvs3dm_results" / site
    save_rgb(Image.open(d / "output/rgb.png"), f"mvs_{site}_rgb.jpg")
    gt = np.array(Image.open(d / "lidar/gt_ndsm.tif"), dtype=np.float32)
    gt[gt < -1000] = np.nan
    pred = np.array(Image.open(d / "output/ndsm_m.tif"), dtype=np.float32)
    colorize(gt, f"mvs_{site}_gt.jpg", vmin=0, vmax=20)
    colorize(pred, f"mvs_{site}_pred.jpg", vmin=0, vmax=20)
    err = np.array(Image.open(d / "output/error_m.tif"), dtype=np.float32)
    err[np.abs(err) > 1000] = np.nan
    colorize(err, f"mvs_{site}_err.jpg", cmap="RdBu_r", vmin=-8, vmax=8)

(OUT / "ranges.json").write_text(json.dumps(ranges, indent=1))
print(ranges)
