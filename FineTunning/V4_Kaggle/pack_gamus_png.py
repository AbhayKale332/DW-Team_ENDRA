"""Pack a *raw PNG* GAMUS tree into the memmap shard store training reads.

`prepare_data.py --datasets gamus` pulls one HDF5 per tile from the gated
`earthflow/GAMUS` repo.  The Kaggle mirror of the same data is already unpacked
into plain PNGs:

    <root>/{train,val,test}/images/gamus_XXX_1234.png    RGB
    <root>/{train,val,test}/depth/gamus_XXX_1234.png     AGL height

so there is nothing to download — this walks the pairs and writes exactly the
layout `dwdata/packed.PackedStore` expects (`shard_NNN_{rgb,hgt,cls,val}.npy`
plus `index.json`).  No class masks come with the PNG mirror, so every pixel is
written as `NO_LABEL`; `seg_ce_loss` short-circuits an all-unlabelled batch to
zero, so the 0.2-weight semantic head simply contributes nothing.

    python pack_gamus_png.py --src /kaggle/input/gamus --out /kaggle/temp/dwdata

Height decoding is the one thing that cannot be assumed: the PNG mirror may hold
metres (8-bit / float) or centimetres (16-bit).  `--depth_scale auto` guesses
from the dtype and the 99.9th percentile and *prints the resulting height
statistics* — check them against GAMUS's real range (0 to ~100 m AGL, median a
couple of metres) before letting a 12 h run start.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dwdata.packed import NO_LABEL, ShardWriter, store_exists  # noqa: E402

GAMUS_GSD_M = 0.33
_EXT = (".png", ".tif", ".tiff", ".jpg", ".jpeg")


def _imread(path: Path) -> np.ndarray:
    """Unchanged read — 16-bit depth PNGs must not be truncated to uint8."""
    import cv2

    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise OSError(f"unreadable: {path}")
    return a


def _rgb(path: Path) -> np.ndarray:
    a = _imread(path)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    a = a[..., :3][..., ::-1]                      # cv2 gives BGR
    if a.dtype != np.uint8:                        # 16-bit imagery -> 8-bit
        a = (a.astype(np.float32) / max(1.0, float(a.max())) * 255.0)
    return np.ascontiguousarray(a.astype(np.uint8))


def _square(a: np.ndarray, t: int) -> np.ndarray | None:
    """Centre-crop to t x t; None if the tile is smaller than t in either axis."""
    h, w = a.shape[:2]
    if h < t or w < t:
        return None
    top, left = (h - t) // 2, (w - t) // 2
    return a[top:top + t, left:left + t]


def _pairs(split_dir: Path, img_sub: str, dep_sub: str) -> list[tuple[str, Path, Path]]:
    imgs = {p.stem: p for p in sorted((split_dir / img_sub).iterdir())
            if p.suffix.lower() in _EXT}
    deps = {p.stem: p for p in sorted((split_dir / dep_sub).iterdir())
            if p.suffix.lower() in _EXT}
    # The mirror pairs by identical stem, but tolerate a _depth/_agl/_dsm suffix
    # on the label side rather than silently packing half the split.
    out = []
    for stem, ip in imgs.items():
        dp = deps.get(stem)
        if dp is None:
            for suf in ("_depth", "_agl", "_dsm", "_AGL", "_DEPTH"):
                if stem + suf in deps:
                    dp = deps[stem + suf]
                    break
        if dp is None and stem.startswith("gamus_"):
            dp = deps.get(stem[len("gamus_"):])
        if dp is not None:
            out.append((stem, ip, dp))
    return out


def _decode_scale(sample: np.ndarray, mode: str) -> float:
    if mode != "auto":
        return float(mode)
    if np.issubdtype(sample.dtype, np.floating):
        return 1.0
    if sample.dtype == np.uint8:
        return 1.0                                  # 0-255 already reads as metres
    p = float(np.percentile(sample.astype(np.float32), 99.9))
    return 0.01 if p > 1000.0 else 1.0              # uint16 centimetres vs metres


def pack_split(src: Path, out: Path, img_sub: str, dep_sub: str, n_tiles: int,
               depth_scale: str, tile_px: int, shard_tiles: int,
               force: bool) -> None:
    if store_exists(out) and not force:
        print(f"[{out.name}] already packed -> {out}")
        return

    pairs = _pairs(src, img_sub, dep_sub)
    if not pairs:
        raise RuntimeError(f"no image/depth pairs under {src}")
    if n_tiles and n_tiles < len(pairs):
        rng = np.random.default_rng(0)
        keep = rng.permutation(len(pairs))[:n_tiles]
        pairs = [pairs[i] for i in sorted(keep.tolist())]

    if tile_px <= 0:
        h, w = _imread(pairs[0][1]).shape[:2]
        tile_px = int(min(h, w))
        tile_px -= tile_px % 16                     # the model's patch grid
    scale = _decode_scale(_imread(pairs[0][2]), depth_scale)
    print(f"[{out.name}] {len(pairs)} pairs, tile_px={tile_px}, "
          f"depth x{scale:g} -> metres", flush=True)

    writer = ShardWriter(out, tile_px=tile_px, gsd_m=GAMUS_GSD_M,
                         shard_tiles=shard_tiles, has_seg=False)
    stats, done, skipped = [], 0, 0
    for stem, ip, dp in pairs:
        try:
            rgb_full = _rgb(ip)
            d = _imread(dp)
            if d.ndim == 3:
                d = d[..., 0]
            if d.shape[:2] != rgb_full.shape[:2]:
                # A label raster at a different resolution has to be put on the
                # image grid *before* either is cropped, or the two crops are of
                # different ground.  Nearest: heights must not be interpolated
                # across a roof edge.
                import cv2
                d = cv2.resize(d, (rgb_full.shape[1], rgb_full.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
            rgb, d = _square(rgb_full, tile_px), _square(d, tile_px)
            if rgb is None or d is None:
                skipped += 1
                continue
            agl = d.astype(np.float32) * scale
            valid = np.isfinite(agl) & (agl > -2.0) & (agl < 500.0)
            agl = np.where(valid, np.clip(agl, 0.0, None), 0.0)
            writer.add(stem, rgb, agl, None, valid)
            if len(stats) < 200:
                stats.append(agl[valid][::97] if valid.any() else np.zeros(1, np.float32))
            done += 1
        except Exception as e:  # noqa: BLE001
            print(f"[{out.name}] skip {stem}: {e}")
            skipped += 1
        if done and done % 250 == 0:
            print(f"  {done}/{len(pairs)}", flush=True)
    idx = writer.finalise()
    print(f"[{out.name}] packed {idx['n']} tiles into {len(idx['shards'])} shards "
          f"({skipped} skipped)")
    if stats:
        s = np.concatenate(stats)
        print(f"[{out.name}] height check  min {s.min():.2f}  median "
              f"{np.median(s):.2f}  p99 {np.percentile(s, 99):.2f}  "
              f"max {s.max():.2f}  (metres — GAMUS AGL is ~0-100 m)")


def find_src(base: Path) -> Path:
    """The directory that actually holds train/images, wherever it got mounted."""
    if (base / "train" / "images").is_dir():
        return base
    for p in sorted(base.rglob("train/images")):
        return p.parent.parent
    raise RuntimeError(f"no <root>/train/images under {base}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="GAMUS PNG root (or a parent of it)")
    ap.add_argument("--out", default="/kaggle/temp/dwdata",
                    help="data_root; the store lands in <out>/gamus/<split>")
    ap.add_argument("--images_sub", default="images")
    ap.add_argument("--depth_sub", default="depth")
    ap.add_argument("--train_tiles", type=int, default=0, help="0 = all")
    ap.add_argument("--val_tiles", type=int, default=0)
    ap.add_argument("--depth_scale", default="auto",
                    help="'auto', or a float multiplier into metres (0.01 = cm)")
    ap.add_argument("--tile_px", type=int, default=0, help="0 = detect")
    ap.add_argument("--shard_tiles", type=int, default=128)
    ap.add_argument("--prime_workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    src = find_src(Path(a.src))
    out = Path(a.out) / "gamus"
    print(f"source: {src}\ntarget: {out}")
    for split, n in (("train", a.train_tiles), ("val", a.val_tiles)):
        d = src / split
        if not d.is_dir():
            print(f"[{split}] absent — skipped")
            continue
        pack_split(d, out / split, a.images_sub, a.depth_sub, n, a.depth_scale,
                   a.tile_px, a.shard_tiles, a.force)

    # Same reason prepare_data.py does it: the 2/98 stretch bounds are a
    # full-tile histogram, and a training worker must never pay for one.
    from dwdata.packed import PackedStore
    for idx in sorted(out.glob("*/index.json")):
        st = PackedStore(idx.parent)
        st.prime_stretch_bounds(2.0, 98.0, workers=a.prime_workers)
        print(f"  bounds primed: {idx.parent} ({len(st)} tiles)")


if __name__ == "__main__":
    main()
