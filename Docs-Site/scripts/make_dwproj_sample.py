"""Pack a .dwproj project for the docs mini viewer (SceneViewer).

Run from Docs-Site/:
  python3 scripts/make_dwproj_sample.py ../Frontend/public/samples/Wankhede_Stadium_Mumbai/Wankhede_Stadium_Mumbai.dwproj wankhede
Writes public/samples/<name>/{rgb.jpg, seg.png, heights.bin, buildings.json, meta.json}

heights.bin is little-endian uint16, row-major, grid[1] rows x grid[0] cols
(FACTOR x FACTOR block mean of the prediction); height_m = v / 65535 * height_max_m.
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

DOCS = Path(__file__).resolve().parents[1]
FACTOR = 2

src, name = Path(sys.argv[1]), sys.argv[2]
OUT = DOCS / "public/samples" / name
OUT.mkdir(parents=True, exist_ok=True)
z = zipfile.ZipFile(src)
man = json.loads(z.read("manifest.json"))

h = np.load(io.BytesIO(z.read("ndsm_m.npy"))).astype(np.float32)
H, W = h.shape
h = np.nan_to_num(h, nan=0.0).clip(min=0)
gh, gw = H // FACTOR, W // FACTOR
hs = h[: gh * FACTOR, : gw * FACTOR].reshape(gh, FACTOR, gw, FACTOR).mean(axis=(1, 3))
hmax = float(hs.max())
q = np.round(hs / hmax * 65535).astype("<u2")
(OUT / "heights.bin").write_bytes(q.tobytes())

Image.open(io.BytesIO(z.read(man["imageName"]))).convert("RGB").save(OUT / "rgb.jpg", quality=86, optimize=True)
Image.open(io.BytesIO(z.read("seg.png"))).save(OUT / "seg.png", optimize=True)

obj = json.loads(z.read("objects.json"))
blds = [
    {"poly": [[round(x, 1), round(y, 1)] for x, y in b["poly"]], "h": b.get("h"), "h_max": b.get("h_max"), "area_m2": b.get("area_m2")}
    for b in obj.get("buildings", [])
]
(OUT / "buildings.json").write_text(json.dumps(blds, separators=(",", ":")))

scene = man["meta"]["scene"]
out_meta = {
    "grid": [gw, gh], "src_px": [W, H], "gsd_m": float(man["gsd"]),
    "height_max_m": hmax, "height_max_src_m": float(h.max()),
    "crs": scene.get("crs"), "transform": scene.get("transform"),
    "classes": man["meta"].get("classes"),
    "frac_below_1m": man["meta"].get("frac_below_1m"),
    "n_buildings": len(blds),
}
(OUT / "meta.json").write_text(json.dumps(out_meta, indent=1))
for p in sorted(OUT.iterdir()):
    print(f"{p.name:16s} {p.stat().st_size/1024:8.1f} KB")
print(json.dumps({k: out_meta[k] for k in ("grid", "src_px", "height_max_m", "n_buildings")}))
