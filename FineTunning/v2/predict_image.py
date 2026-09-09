"""Quick qualitative test of a trained checkpoint on a single image.

Small RGB image (jpg/png) -> whole image is resized to one tile:

    python predict_image.py path/to/img.jpg --ckpt outputs/v2/best.pt

Large GeoTIFF (aerial ortho) -> resampled to the model GSD and run with a
sliding 512 window, Hann-blended back into a full-resolution height map:

    python predict_image.py /teamspace/studios/this_studio/austin1.tif \
        --ckpt outputs/v2/best.pt --src-gsd 0.3 \
        --out-dir outputs/v2/predict/austin1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config import parse_config
from dwdata.base import IMAGENET_MEAN, IMAGENET_STD
from models.heads import DepthWizardNetV2

Image.MAX_IMAGE_PIXELS = None
TIF_SUFFIXES = {".tif", ".tiff"}


def load_model(ckpt: str, cfg, dev):
    model = DepthWizardNetV2(cfg).to(dev).eval()
    ck = torch.load(ckpt, map_location=dev)
    miss, unexp = model.load_state_dict(ck["model"], strict=False)
    print(f"loaded {ckpt}  epoch={ck.get('epoch')} stage={ck.get('stage')}  "
          f"(missing {len(miss)}, unexpected {len(unexp)})")
    return model


def _norm(rgb01: np.ndarray) -> np.ndarray:
    return (rgb01 - IMAGENET_MEAN) / IMAGENET_STD


def _hann2d(n: int) -> np.ndarray:
    w = np.hanning(n + 2)[1:-1]
    return np.clip(np.outer(w, w), 1e-3, None).astype(np.float32)


@torch.no_grad()
def _infer_tile(model, rgb01: np.ndarray, dev) -> np.ndarray:
    x = _norm(rgb01.astype(np.float32))
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(dev)
    out = model(x)
    return out["fused"][0, 0].float().cpu().numpy()


def _save_outputs(out_dir: Path, stem: str, rgb_u8: np.ndarray, height_m: np.ndarray,
                  gsd_m: float, transform=None, crs=None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lo, hi = float(np.nanmin(height_m)), float(np.nanmax(height_m))
    norm = (height_m - lo) / (hi - lo + 1e-6)

    Image.fromarray(rgb_u8).save(out_dir / "rgb.png")
    Image.fromarray((norm * 255).astype(np.uint8)).save(out_dir / "height.png")
    np.save(out_dir / "height_m.npy", height_m.astype(np.float32))

    if transform is not None:
        try:
            import rasterio
            with rasterio.open(
                out_dir / "height_m.tif", "w", driver="GTiff",
                height=height_m.shape[0], width=height_m.shape[1], count=1,
                dtype="float32", crs=crs, transform=transform, compress="deflate",
            ) as dst:
                dst.write(height_m.astype(np.float32), 1)
        except Exception as e:  # noqa: BLE001
            print("geotiff write skipped:", e)

    (out_dir / "meta.json").write_text(json.dumps({
        "stem": stem,
        "height_min_m": lo,
        "height_max_m": hi,
        "height_mean_m": float(np.nanmean(height_m)),
        "gsd_m": gsd_m,
        "size_px": list(height_m.shape),
    }, indent=2))
    print(f"height map: shape={height_m.shape}  min={lo:.2f}m  max={hi:.2f}m  "
          f"mean={float(np.nanmean(height_m)):.2f}m")
    print("saved ->", out_dir, "(rgb.png / height.png / height_m.npy / meta.json)")


def run_small(model, image: str, cfg, dev, out_dir: Path) -> None:
    s = cfg.tile_size
    rgb = Image.open(image).convert("RGB").resize((s, s), Image.BILINEAR)
    rgb01 = np.asarray(rgb, dtype=np.float32) / 255.0
    depth = _infer_tile(model, rgb01, dev)
    _save_outputs(out_dir, Path(image).stem, (rgb01 * 255).astype(np.uint8), depth,
                  cfg.canonical_gsd_m)


def run_tif(model, image: str, cfg, dev, out_dir: Path, src_gsd: float,
            overlap: float) -> None:
    import rasterio
    from rasterio.enums import Resampling

    s = cfg.tile_size
    dst_gsd = cfg.canonical_gsd_m
    with rasterio.open(image) as src:
        native = float(abs(src.res[0]))
        g = src_gsd if src_gsd > 0 else native
        scale = g / dst_gsd
        W = max(s, int(round(src.width * scale)))
        H = max(s, int(round(src.height * scale)))
        print(f"{image}: {src.width}x{src.height} @ {native:.3f}m "
              f"-> {W}x{H} @ {dst_gsd:.3f}m  (assumed src gsd {g:.3f}m)")
        rgb = src.read(
            indexes=[1, 2, 3], out_shape=(3, H, W),
            resampling=Resampling.bilinear,
        ).transpose(1, 2, 0)
        # rescale the geotransform to the resampled grid
        transform = src.transform * src.transform.scale(src.width / W, src.height / H)
        crs = src.crs

    rgb = rgb.astype(np.float32)
    if rgb.max() > 1.5:
        rgb /= 255.0
    rgb = np.clip(rgb, 0.0, 1.0)

    stride = max(1, int(round(s * (1.0 - overlap))))
    win = _hann2d(s)
    acc = np.zeros((H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)

    ys = list(range(0, max(1, H - s + 1), stride)) or [0]
    xs = list(range(0, max(1, W - s + 1), stride)) or [0]
    if ys[-1] != H - s and H > s:
        ys.append(H - s)
    if xs[-1] != W - s and W > s:
        xs.append(W - s)
    total = len(ys) * len(xs)
    print(f"sliding window: {s}px stride {stride}  -> {total} tiles")

    n = 0
    for y in ys:
        for x in xs:
            tile = rgb[y:y + s, x:x + s]
            ph, pw = tile.shape[:2]
            if (ph, pw) != (s, s):
                tile = np.pad(tile, ((0, s - ph), (0, s - pw), (0, 0)), mode="reflect")
            d = _infer_tile(model, tile, dev)[:ph, :pw]
            acc[y:y + ph, x:x + pw] += d * win[:ph, :pw]
            wsum[y:y + ph, x:x + pw] += win[:ph, :pw]
            n += 1
            if n % 5 == 0 or n == total:
                print(f"  {n}/{total}", flush=True)

    height = acc / np.maximum(wsum, 1e-6)
    _save_outputs(out_dir, Path(image).stem, (rgb * 255).astype(np.uint8), height,
                  dst_gsd, transform=transform, crs=crs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--ckpt", default="outputs/v2/best.pt")
    ap.add_argument("--out-dir", default=None,
                    help="output dir (default outputs/v2/predict/<stem>)")
    ap.add_argument("--src-gsd", type=float, default=0.0,
                    help="source ground sample distance in m for a GeoTIFF "
                         "(0 = use the file's own resolution)")
    ap.add_argument("--overlap", type=float, default=0.25,
                    help="tile overlap fraction for GeoTIFF sliding window")
    a = ap.parse_args()

    cfg = parse_config([])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev)
    model = load_model(a.ckpt, cfg, dev)

    out_dir = Path(a.out_dir) if a.out_dir else Path("outputs/v2/predict") / Path(a.image).stem

    if Path(a.image).suffix.lower() in TIF_SUFFIXES:
        run_tif(model, a.image, cfg, dev, out_dir, a.src_gsd, a.overlap)
    else:
        run_small(model, a.image, cfg, dev, out_dir)


if __name__ == "__main__":
    main()
