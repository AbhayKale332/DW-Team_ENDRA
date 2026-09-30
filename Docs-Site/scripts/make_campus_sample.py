"""Pack the web app's bundled campus sample for the docs mini viewer.

Run from Docs-Site/:  python3 scripts/make_campus_sample.py
Reads  ../Frontend/public/samples/buildings_large_campus/
Writes public/samples/campus/{rgb.jpg, seg.png, heights.bin, buildings.json, meta.json}

heights.bin is little-endian uint16, row-major, GRID x GRID (2x2 block mean of
the 1024^2 prediction); height_m = v / 65535 * height_max_m.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

DOCS = Path(__file__).resolve().parents[1]
SRC = DOCS.parent / "Frontend/public/samples/buildings_large_campus"
OUT = DOCS / "public/samples/campus"
OUT.mkdir(parents=True, exist_ok=True)
GRID = 512

meta = json.loads((SRC / "meta.json").read_text())
h = np.load(SRC / "ndsm_m.npy").astype(np.float32)
H, W = h.shape
h = np.nan_to_num(h, nan=0.0).clip(min=0)
f = H // GRID
hs = h.reshape(GRID, f, GRID, f).mean(axis=(1, 3))
hmax = float(hs.max())
q = np.round(hs / hmax * 65535).astype("<u2")
(OUT / "heights.bin").write_bytes(q.tobytes())

Image.open(SRC / "rgb.png").convert("RGB").save(OUT / "rgb.jpg", quality=86, optimize=True)
Image.open(SRC / "seg.png").save(OUT / "seg.png", optimize=True)

obj = json.loads((SRC / "objects.json").read_text())
gsd = float(obj["grid"]["gsd_m"])
blds = []
for b in obj["buildings"]:
    blds.append({
        "poly": [[round(x, 1), round(y, 1)] for x, y in b["poly"]],
        "h": b.get("h"), "h_max": b.get("h_max"), "area_m2": b.get("area_m2"),
    })
(OUT / "buildings.json").write_text(json.dumps(blds, separators=(",", ":")))

t = meta["scene"]["transform"]
out_meta = {
    "grid": GRID, "src_px": [W, H], "gsd_m": gsd,
    "height_max_m": hmax, "height_max_src_m": float(h.max()),
    "crs": meta["scene"]["crs"], "transform": t,
    "classes": meta.get("classes"),
    "frac_below_1m": meta.get("frac_below_1m"),
    "n_buildings": len(blds),
}
(OUT / "meta.json").write_text(json.dumps(out_meta, indent=1))
for p in sorted(OUT.iterdir()):
    print(f"{p.name:16s} {p.stat().st_size/1024:8.1f} KB")
print(json.dumps({k: out_meta[k] for k in ("height_max_m", "height_max_src_m", "n_buildings")}))
print("building keys:", list(obj["buildings"][0].keys()))
