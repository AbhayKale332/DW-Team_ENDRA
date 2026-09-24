"""Step 0 of the v5 plan: audit the labels before spending GPU time on them.

    python tools/audit_labels.py --data_root /scratch/dwdata --out outputs/audit
    python tools/audit_labels.py --data_root ... --cartosat cartosat_2S_Sample/Cartosat-2E/247677521
    python tools/audit_labels.py --data_root ... --dfc23_raw /data/raw/dfc23/train

CPU only, reads the packed stores `prepare_data.py` wrote (no repack), and
writes `audit.json`, `audit.md` and `class_overlays/*.png`.  Every section is
independent: a store that is missing is skipped with a note.

**What each section decides** (the flag it feeds is printed next to it):

* `--resolution`  Coarse labels.  DFC23 / India nDSMs are ~2 m stereo resampled
  onto 0.5 m pixels, so they carry no detail below ~4 px.  For each store and
  k in (2, 4, 8) this measures the fraction of the label's variance that an
  k x k average-pool + bilinear upsample *throws away*.  A label that really is
  k x coarser loses almost nothing at k; GAMUS (true 0.33 m LiDAR) loses a lot.
  `coarse_pool` is the largest k at which a store keeps <= 25 % of GAMUS's
  residual.  `--dfc23_raw` additionally reads the raw rgb/ and dsm/ GeoTIFF
  pixel sizes, which is the direct answer when the raw scenes are to hand.
* `--vegetation`  Does a coarse label put trees at ~0 m?  An ExG threshold tau
  is calibrated on GAMUS as the F1-optimal separator of the tree class, then
  the median label height on ExG > tau pixels is compared across stores.  A
  coarse store whose green pixels sit under 1 m while GAMUS trees stand several
  metres tall -> `coarse_mask_veg true`, `coarse_veg_exg tau`.
* `--classes`  Pins the GAMUS class names: per-id pixel share, mean / median /
  p90 GT height, ExG share, and an overlay PNG per id.  Checks the result
  against `config.CLASS_NAMES` (the two tall ids must be building and tree, the
  flattest must be water) and says so if it disagrees.
* `--overlaps`  Held-out claims: stems *and* image content (a 32 x 32 thumbnail
  hash, so a re-cut tile with a new stem is still caught) of gamus/test vs
  gamus/train+val, and india_labeled/val vs every dfc23_*/train.
* `--cartosat`  Calibrates the Cartosat augmentations on the real product:
  the chroma-to-luma high-frequency energy ratio (-> `aug_pansharp_lo/hi`, by
  finding the chroma downsample factor that gives GAMUS tiles the same ratio)
  and stretched colour / haze statistics against GAMUS and DFC23 (-> whether a
  haze augmentation is needed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CLASS_NAMES, SEG_IGNORE_INDEX  # noqa: E402
from dwdata.dataset import seg_ids  # noqa: E402
from dwdata.packed import PackedStore, store_exists  # noqa: E402
from dwdata.preprocess import apply_stretch, scene_stretch_bounds  # noqa: E402

COARSE_KS = (2, 4, 8)
RESIDUAL_KEEP = 0.25          # a store is "k x coarse" when it keeps <= 25 % of GAMUS's residual


# ---------------------------------------------------------------------
# small numeric helpers (numpy only, so this runs on any CPU box)
# ---------------------------------------------------------------------
def pool_up(a: np.ndarray, k: int) -> np.ndarray:
    """k x k area mean, then bilinear back to the input shape (crop to a multiple of k)."""
    h, w = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    a = a[:h, :w]
    low = a.reshape(h // k, k, w // k, k).mean((1, 3))
    return bilinear(low, (h, w))


def bilinear(a: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """align_corners=False bilinear resize, the same convention torch uses."""
    H, W = out_hw
    h, w = a.shape

    def coords(n_out, n_in):
        x = (np.arange(n_out) + 0.5) * n_in / n_out - 0.5
        x = np.clip(x, 0, n_in - 1)
        i0 = np.floor(x).astype(int)
        i1 = np.minimum(i0 + 1, n_in - 1)
        return i0, i1, (x - i0).astype(np.float32)

    y0, y1, fy = coords(H, h)
    x0, x1, fx = coords(W, w)
    top = a[y0][:, x0] * (1 - fx) + a[y0][:, x1] * fx
    bot = a[y1][:, x0] * (1 - fx) + a[y1][:, x1] * fx
    return top * (1 - fy)[:, None] + bot * fy[:, None]


def box_blur(a: np.ndarray, r: int) -> np.ndarray:
    """(2r+1)^2 box mean with edge replication, via cumulative sums."""
    p = np.pad(a.astype(np.float64), r + 1, mode="edge")
    c = p.cumsum(0).cumsum(1)
    n = 2 * r + 1
    s = c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n]
    return (s / (n * n))[: a.shape[0], : a.shape[1]].astype(np.float32)


def exg(rgb_u8: np.ndarray) -> np.ndarray:
    """ExG = 2g - r - b on chromaticities (same formula as dataset.exg_mask)."""
    x = rgb_u8.astype(np.float32)
    s = x.sum(-1) + 1e-3
    return (2 * x[..., 1] - x[..., 0] - x[..., 2]) / s


def ycbcr(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = rgb.astype(np.float32) / 255.0
    y = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    return y, x[..., 2] - y, x[..., 0] - y          # Y, Cb~, Cr~ (the gpu_aug convention)


def hf_energy(a: np.ndarray, valid: np.ndarray | None = None, r: int = 1) -> float:
    d = a - box_blur(a, r)
    if valid is not None:
        d = d[valid]
    return float(np.mean(d.astype(np.float64) ** 2)) if d.size else 0.0


def chroma_luma_ratio(rgb: np.ndarray, valid: np.ndarray | None = None) -> float:
    y, cb, cr = ycbcr(rgb)
    ey = hf_energy(y, valid)
    return (hf_energy(cb, valid) + hf_energy(cr, valid)) / max(ey, 1e-12)


def simulate_pansharp(rgb: np.ndarray, f: float) -> np.ndarray:
    """numpy twin of gpu_aug.sensor_augment's pan-sharpen branch (one factor)."""
    x = rgb.astype(np.float32) / 255.0
    h, w = x.shape[:2]
    y = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    hs, ws = max(1, int(round(h / f))), max(1, int(round(w / f)))
    out = []
    for c in (x[..., 0] - y, x[..., 2] - y):
        # area-downsample by an arbitrary factor: box over the source footprint
        ys = np.linspace(0, h, hs + 1).astype(int)
        xs = np.linspace(0, w, ws + 1).astype(int)
        cs = c.cumsum(0).cumsum(1)
        cs = np.pad(cs, ((1, 0), (1, 0)))
        s = cs[ys[1:]][:, xs[1:]] - cs[ys[:-1]][:, xs[1:]] - cs[ys[1:]][:, xs[:-1]] \
            + cs[ys[:-1]][:, xs[:-1]]
        area = np.outer(np.diff(ys), np.diff(xs)).clip(1)
        out.append(bilinear((s / area).astype(np.float32), (h, w)))
    r = y + out[0]
    b = y + out[1]
    g = (y - 0.299 * r - 0.114 * b) / 0.587
    return (np.clip(np.stack([r, g, b], -1), 0, 1) * 255 + 0.5).astype(np.uint8)


def dark_channel(rgb: np.ndarray, valid: np.ndarray | None = None, r: int = 7) -> float:
    """Mean of the dark channel (He et al.): high = hazy.  Min over a (2r+1)^2 window."""
    m = rgb.min(-1).astype(np.float32)
    from scipy.ndimage import minimum_filter

    d = minimum_filter(m, size=2 * r + 1, mode="nearest")
    return float(d[valid].mean() if valid is not None else d.mean()) / 255.0


def _stretched(rgb: np.ndarray, lo_pct: float = 2.0, hi_pct: float = 98.0) -> np.ndarray:
    lo, hi = scene_stretch_bounds(rgb, lo_pct, hi_pct)
    return apply_stretch(rgb, lo, hi)


def _sample(n: int, k: int, seed: int = 0) -> np.ndarray:
    if k <= 0 or k >= n:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, k, replace=False))


def _open(root: Path, name: str, split: str) -> PackedStore | None:
    d = root / name / split
    return PackedStore(d) if store_exists(d) else None


def _label_stores(root: Path) -> dict[str, PackedStore]:
    """Every labelled train store under root (plus the ones only packed as val)."""
    out = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if d.name.startswith(("_", "india_unlabeled")):
            continue
        for split in ("train", "val"):
            st = _open(root, d.name, split)
            if st is not None:
                out[d.name] = st
                break
    return out


# ---------------------------------------------------------------------
# 1. label resolution
# ---------------------------------------------------------------------
def label_residuals(store: PackedStore, n_tiles: int, win: int = 256) -> dict:
    """Per k: fraction of label variance lost to a k x pool + upsample, plus grad energy."""
    lost = {k: [] for k in COARSE_KS}
    grad = []
    for i in _sample(len(store), n_tiles):
        _, h, _, v = store.get(int(i))
        t = min(win, h.shape[0], h.shape[1])
        y0, x0 = (h.shape[0] - t) // 2, (h.shape[1] - t) // 2
        h = h[y0:y0 + t, x0:x0 + t].astype(np.float32)
        v = v[y0:y0 + t, x0:x0 + t].astype(bool)
        if v.mean() < 0.99 or h.std() < 0.25:        # need a full, non-flat window
            continue
        var = float(h.var())
        for k in COARSE_KS:
            r = h[: (t // k) * k, : (t // k) * k] - pool_up(h, k)
            lost[k].append(float((r ** 2).mean()) / var)
        gy, gx = np.gradient(h)
        grad.append(float(np.mean(np.hypot(gx, gy))))
    return {"tiles": len(grad),
            "residual_frac": {k: (float(np.median(x)) if x else None) for k, x in lost.items()},
            "mean_abs_grad_m_per_px": float(np.median(grad)) if grad else None}


def raw_pixel_ratio(src: Path, n: int = 50) -> dict:
    """DFC23 raw: median DSM / RGB pixel-size ratio over `<src>/{rgb,dsm}/*.tif`."""
    import rasterio

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from prepare_data import _dfc23_pairs

    recs = _dfc23_pairs(src)[:n]
    ratios = []
    for r in recs:
        with rasterio.open(r["rgb"]) as a, rasterio.open(r["hgt"]) as b:
            ratios.append(abs(b.transform.a) / abs(a.transform.a))
    return {"pairs": len(ratios),
            "dsm_over_rgb_pixel": float(np.median(ratios)) if ratios else None}


def audit_resolution(root: Path, stores: dict, n_tiles: int, dfc23_raw: str) -> dict:
    res = {name: label_residuals(st, n_tiles) for name, st in stores.items()}
    ref = res.get("gamus", {}).get("residual_frac", {})
    rec = {}
    for name, r in res.items():
        if name == "gamus" or not r["tiles"]:
            continue
        pool = 1
        for k in COARSE_KS:
            a, b = r["residual_frac"].get(k), ref.get(k)
            if a is not None and b and a <= RESIDUAL_KEEP * b:
                pool = k
        r["effective_pool"] = pool
        if ref:
            g, gg = r["mean_abs_grad_m_per_px"], res["gamus"]["mean_abs_grad_m_per_px"]
            r["grad_energy_vs_gamus"] = (g / gg) if (g and gg) else None
        rec[name] = pool
    out = {"per_store": res, "recommended_pool": rec}
    coarse = [p for n, p in rec.items() if n.startswith(("dfc23", "india"))]
    if coarse:
        out["coarse_pool"] = int(max(set(coarse), key=coarse.count))
        # the same answer in metres — what the loss actually needs, since the
        # GSD jitter resamples crops away from the store's packed GSD
        metres = [p * float(stores[n].gsd_m) for n, p in rec.items()
                  if n.startswith(("dfc23", "india"))]
        out["coarse_label_m"] = round(float(np.median(metres)), 2)
    if dfc23_raw:
        out["dfc23_raw"] = raw_pixel_ratio(Path(dfc23_raw))
    return out


# ---------------------------------------------------------------------
# 2. vegetation in coarse labels
# ---------------------------------------------------------------------
def calibrate_exg(gamus: PackedStore, tree_id: int, n_tiles: int) -> dict:
    """F1-optimal ExG threshold for the GAMUS tree class, plus tree heights."""
    taus = np.round(np.arange(-0.05, 0.30, 0.01), 3)
    tp = np.zeros(len(taus))
    fp = np.zeros(len(taus))
    fn = np.zeros(len(taus))
    tree_h = []
    for i in _sample(len(gamus), n_tiles):
        rgb, h, cls, v = gamus.get(int(i))
        seg = seg_ids(cls, "gamus")
        ok = v.astype(bool) & (seg != SEG_IGNORE_INDEX)
        if not ok.any():
            continue
        e = exg(_stretched(rgb))[ok]
        t = seg[ok] == tree_id
        tree_h.append(h[ok][t])
        for j, tau in enumerate(taus):
            p = e > tau
            tp[j] += np.sum(p & t)
            fp[j] += np.sum(p & ~t)
            fn[j] += np.sum(~p & t)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    j = int(np.argmax(f1))
    th = np.concatenate(tree_h) if tree_h else np.zeros(0)
    return {"tau": float(taus[j]), "f1": float(f1[j]),
            "gamus_tree_median_m": float(np.median(th)) if th.size else None}


def green_heights(store: PackedStore, tau: float, n_tiles: int) -> dict:
    hs = []
    for i in _sample(len(store), n_tiles):
        rgb, h, _, v = store.get(int(i))
        m = v.astype(bool) & (exg(_stretched(rgb)) > tau)
        hs.append(h[m])
    a = np.concatenate(hs) if hs else np.zeros(0)
    if not a.size:
        return {"green_px": 0}
    return {"green_px": int(a.size), "median_m": float(np.median(a)),
            "frac_below_1m": float(np.mean(a < 1.0))}


def audit_vegetation(stores: dict, n_tiles: int) -> dict:
    if "gamus" not in stores:
        return {"skipped": "no gamus store to calibrate tau on"}
    tree_id = CLASS_NAMES.index("tree")
    cal = calibrate_exg(stores["gamus"], tree_id, n_tiles)
    per = {n: green_heights(st, cal["tau"], n_tiles) for n, st in stores.items()}
    tall = cal["gamus_tree_median_m"] or 0.0
    mask = {n: bool(p.get("median_m") is not None and p["median_m"] < 1.0 and tall >= 3.0)
            for n, p in per.items() if n.startswith(("dfc23", "india"))}
    return {"calibration": cal, "green_pixel_heights": per,
            "coarse_mask_veg": any(mask.values()) if mask else None,
            "coarse_veg_exg": cal["tau"], "per_store_mask": mask}


# ---------------------------------------------------------------------
# 3. GAMUS class pinning
# ---------------------------------------------------------------------
_TINT = np.array([[255, 0, 255], [255, 255, 0], [124, 252, 0], [255, 0, 0],
                  [0, 128, 255], [160, 160, 160], [0, 100, 0]], np.uint8)


def audit_classes(gamus: PackedStore, n_tiles: int, out_dir: Path) -> dict:
    n_ids = len(CLASS_NAMES)
    hs = [[] for _ in range(n_ids)]
    green = np.zeros(n_ids)
    count = np.zeros(n_ids)
    examples: dict[int, list[tuple[float, int]]] = {c: [] for c in range(n_ids)}
    for i in _sample(len(gamus), n_tiles):
        rgb, h, cls, v = gamus.get(int(i))
        seg = seg_ids(cls, "gamus")
        ok = v.astype(bool)
        e = exg(_stretched(rgb)) > 0.05
        for c in range(n_ids):
            m = ok & (seg == c)
            k = int(m.sum())
            if not k:
                continue
            count[c] += k
            green[c] += int((m & e).sum())
            hs[c].append(h[m][:: max(1, k // 20000)])
            # keep the three tiles where this id covers the most area
            examples[c] = sorted(examples[c] + [(k / m.size, int(i))], reverse=True)[:3]
    tot = max(count.sum(), 1)
    per = {}
    for c in range(n_ids):
        a = np.concatenate(hs[c]) if hs[c] else np.zeros(0)
        per[c] = {"name": CLASS_NAMES[c], "px_frac": float(count[c] / tot),
                  "mean_m": float(a.mean()) if a.size else None,
                  "median_m": float(np.median(a)) if a.size else None,
                  "p90_m": float(np.percentile(a, 90)) if a.size else None,
                  "green_frac": float(green[c] / max(count[c], 1))}
    # the evidence the pinning rests on
    means = {c: p["mean_m"] for c, p in per.items() if p["mean_m"] is not None and c != 0}
    checks = []
    if len(means) >= 3:
        tall = sorted(means, key=means.get)[-2:]
        flattest = sorted(means, key=means.get)[:2]
        want_tall = {CLASS_NAMES.index("building"), CLASS_NAMES.index("tree")}
        checks.append(("two tallest ids are building + tree", set(tall) == want_tall,
                       f"tallest ids {sorted(tall)}"))
        # GAMUS: water 0.19 m, low veg 0.38 m, ground 0.49 m — water must be
        # among the two lowest means (ties with low veg are within noise).
        checks.append(("water id is among the two flattest",
                       CLASS_NAMES.index("water") in flattest,
                       f"flattest ids {sorted(flattest)}"))
        tree, bld = CLASS_NAMES.index("tree"), CLASS_NAMES.index("building")
        checks.append(("tree id is greener than building id",
                       per[tree]["green_frac"] > per[bld]["green_frac"],
                       f"green {per[tree]['green_frac']:.2f} vs {per[bld]['green_frac']:.2f}"))
    # overlays, one PNG per id: the image with that class tinted
    ov = out_dir / "class_overlays"
    ov.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    for c, ex in examples.items():
        for j, (_, i) in enumerate(ex):
            rgb, _, cls, _ = gamus.get(i)
            img = _stretched(rgb).astype(np.float32)
            m = seg_ids(cls, "gamus") == c
            img[m] = 0.45 * img[m] + 0.55 * _TINT[c % len(_TINT)]
            Image.fromarray(img.astype(np.uint8)).save(
                ov / f"id{c}_{CLASS_NAMES[c]}_{j}_{gamus.stems[i]}.png")
    return {"per_id": per, "checks": [{"check": a, "ok": bool(b), "evidence": e}
                                      for a, b, e in checks],
            "pinning_ok": all(b for _, b, _ in checks) if checks else None,
            "overlays": str(ov)}


# ---------------------------------------------------------------------
# 4. held-out overlaps (stems + content)
# ---------------------------------------------------------------------
def thumb_hashes(store: PackedStore) -> dict[str, str]:
    out = {}
    for i in range(len(store)):
        rgb = np.asarray(store.rgb_view(i))
        s = max(1, rgb.shape[0] // 32)
        t = rgb[::s, ::s][:32, :32].mean(-1)
        q = (t > np.median(t)).astype(np.uint8)          # average-hash: robust to re-encode
        out[hashlib.sha1(q.tobytes()).hexdigest()] = store.stems[i]
    return out


def audit_overlaps(root: Path) -> dict:
    pairs = [(("gamus", "test"), ("gamus", "train")), (("gamus", "test"), ("gamus", "val"))]
    for d in sorted(root.glob("dfc23_*")):
        pairs.append((("india_labeled", "val"), (d.name, "train")))
    out = {}
    for a, b in pairs:
        sa, sb = _open(root, *a), _open(root, *b)
        if sa is None or sb is None:
            continue
        stems = set(sa.stems) & set(sb.stems)
        ha, hb = thumb_hashes(sa), thumb_hashes(sb)
        content = set(ha) & set(hb)
        out[f"{a[0]}/{a[1]} vs {b[0]}/{b[1]}"] = {
            "shared_stems": len(stems), "shared_content": len(content),
            "examples": sorted(stems)[:3] + [f"{ha[h]}~{hb[h]}" for h in sorted(content)[:3]]}
    return out


# ---------------------------------------------------------------------
# 5. Cartosat statistics
# ---------------------------------------------------------------------
def cartosat_window(path: str, rows: int = 4096, cols: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Central window of a product as uint8 RGB after the training stretch, + valid."""
    from dwdata.scene_io import SceneSource

    paths = path.split(",") if "," in path else path
    src = SceneSource(paths)
    r0 = max(0, (src.height - rows) // 2)
    rgb, valid = src.read_rows(r0, min(src.height, r0 + rows))
    c0 = max(0, (src.width - cols) // 2)
    rgb, valid = rgb[:, c0:c0 + cols], valid[:, c0:c0 + cols]
    lo, hi = scene_stretch_bounds(rgb, 2.0, 98.0, valid=valid)
    return apply_stretch(rgb, lo, hi), valid


def colour_stats(rgb: np.ndarray, valid: np.ndarray | None = None) -> dict:
    v = rgb[valid] if valid is not None else rgb.reshape(-1, 3)
    v = v.astype(np.float32) / 255.0
    return {"mean": [round(float(x), 4) for x in v.mean(0)],
            "std": [round(float(x), 4) for x in v.std(0)],
            "saturation": round(float((v.max(1) - v.min(1)).mean()), 4),
            "dark_channel": round(dark_channel(rgb, valid), 4)}


def store_colour_stats(store: PackedStore, n_tiles: int) -> dict:
    acc = {"mean": [], "std": [], "saturation": [], "dark_channel": [], "cl_ratio": []}
    for i in _sample(len(store), n_tiles):
        rgb = _stretched(np.asarray(store.rgb_view(int(i))))
        s = colour_stats(rgb)
        for k in ("mean", "std", "saturation", "dark_channel"):
            acc[k].append(s[k])
        acc["cl_ratio"].append(chroma_luma_ratio(rgb))
    if not acc["mean"]:
        return {}
    return {"mean": np.mean(acc["mean"], 0).round(4).tolist(),
            "std": np.mean(acc["std"], 0).round(4).tolist(),
            "saturation": round(float(np.mean(acc["saturation"])), 4),
            "dark_channel": round(float(np.mean(acc["dark_channel"])), 4),
            "chroma_luma_hf": round(float(np.median(acc["cl_ratio"])), 5)}


def match_pansharp_factor(target_ratio: float, tiles: list[np.ndarray],
                          factors=np.round(np.arange(1.0, 6.01, 0.25), 2)) -> dict:
    """The chroma downsample factor at which GAMUS's chroma/luma HF ratio equals Cartosat's."""
    curve = []
    for f in factors:
        r = [chroma_luma_ratio(simulate_pansharp(t, f) if f > 1 else t) for t in tiles]
        curve.append(float(np.median(r)))
    curve = np.asarray(curve)
    # the ratio falls monotonically with f; take the first crossing
    below = np.nonzero(curve <= target_ratio)[0]
    f = float(factors[below[0]]) if below.size else float(factors[-1])
    return {"factor": f, "curve": dict(zip([float(x) for x in factors],
                                           [round(float(c), 5) for c in curve]))}


def audit_cartosat(path: str, stores: dict, n_tiles: int) -> dict:
    rgb, valid = cartosat_window(path)
    c = colour_stats(rgb, valid)
    c["chroma_luma_hf"] = round(chroma_luma_ratio(rgb, valid), 5)
    refs = {n: store_colour_stats(st, n_tiles) for n, st in stores.items()
            if n == "gamus" or n.startswith("dfc23")}
    out = {"cartosat": c, "reference": refs}
    if "gamus" in stores:
        tiles = [_stretched(np.asarray(stores["gamus"].rgb_view(int(i)))[:512, :512])
                 for i in _sample(len(stores["gamus"]), min(n_tiles, 24))]
        m = match_pansharp_factor(c["chroma_luma_hf"], tiles)
        out["pansharp_match"] = m
        f = m["factor"]
        if f > 1.0:
            out["aug_pansharp_lo"] = round(max(1.25, 0.8 * f), 2)
            out["aug_pansharp_hi"] = round(1.25 * f, 2)
    dc = [r["dark_channel"] for r in refs.values() if r]
    if dc:
        out["haze_aug_needed"] = bool(c["dark_channel"] > max(dc) + 0.05)
    return out


# ---------------------------------------------------------------------
def suggested_flags(rep: dict) -> dict:
    f = {}
    r = rep.get("resolution", {})
    if "coarse_pool" in r:
        f["coarse_pool"] = str(r["coarse_pool"])
    if "coarse_label_m" in r:
        f["coarse_label_m"] = str(r["coarse_label_m"])
    v = rep.get("vegetation", {})
    if v.get("coarse_mask_veg") is not None:
        f["coarse_mask_veg"] = "true" if v["coarse_mask_veg"] else "false"
        f["coarse_veg_exg"] = str(v["coarse_veg_exg"])
    c = rep.get("cartosat", {})
    if "aug_pansharp_lo" in c:
        f["aug_pansharp_lo"] = str(c["aug_pansharp_lo"])
        f["aug_pansharp_hi"] = str(c["aug_pansharp_hi"])
    return f


def to_markdown(rep: dict) -> str:
    L = ["# Label audit (v5 Step 0)", ""]
    r = rep.get("resolution")
    if r:
        L += ["## Label resolution", "",
              "| store | tiles | lost @2x | lost @4x | lost @8x | grad vs GAMUS | pool |",
              "|---|---|---|---|---|---|---|"]
        for n, s in r["per_store"].items():
            rf = s["residual_frac"]
            fmt = lambda x: "-" if x is None else f"{x:.3f}"   # noqa: E731
            L.append(f"| {n} | {s['tiles']} | {fmt(rf.get(2))} | {fmt(rf.get(4))} | "
                     f"{fmt(rf.get(8))} | {fmt(s.get('grad_energy_vs_gamus'))} | "
                     f"{s.get('effective_pool', 1 if n == 'gamus' else '-')} |")
        if "dfc23_raw" in r:
            L += ["", f"Raw DFC23 DSM/RGB pixel ratio: {r['dfc23_raw']}"]
        L.append("")
    v = rep.get("vegetation")
    if v and "calibration" in v:
        c = v["calibration"]
        L += ["## Vegetation", "",
              f"ExG tau = {c['tau']} (F1 {c['f1']:.3f} on GAMUS trees; GAMUS tree "
              f"median {c['gamus_tree_median_m']} m)", "",
              "| store | green px | median label (m) | < 1 m |", "|---|---|---|---|"]
        for n, s in v["green_pixel_heights"].items():
            if s.get("green_px"):
                L.append(f"| {n} | {s['green_px']} | {s['median_m']:.2f} | "
                         f"{s['frac_below_1m']:.1%} |")
        L.append("")
    c = rep.get("classes")
    if c:
        L += ["## GAMUS classes", "", "| id | name | px | mean m | median m | p90 m | green |",
              "|---|---|---|---|---|---|---|"]
        for i, s in c["per_id"].items():
            f = lambda x: "-" if x is None else f"{x:.2f}"    # noqa: E731
            L.append(f"| {i} | {s['name']} | {s['px_frac']:.1%} | {f(s['mean_m'])} | "
                     f"{f(s['median_m'])} | {f(s['p90_m'])} | {s['green_frac']:.1%} |")
        L += [""] + [f"- {'OK ' if k['ok'] else 'FAIL'} {k['check']} ({k['evidence']})"
                     for k in c["checks"]] + [""]
    o = rep.get("overlaps")
    if o is not None:
        L += ["## Held-out overlaps", ""]
        L += [f"- {k}: {s['shared_stems']} stems, {s['shared_content']} by content"
              for k, s in o.items()] or ["- no pairs found"]
        L.append("")
    k = rep.get("cartosat")
    if k:
        L += ["## Cartosat", "", "```", json.dumps(k, indent=2), "```", ""]
    L += ["## Suggested flags", "", "```", json.dumps(rep.get("flags", {}), indent=2), "```"]
    return "\n".join(L) + "\n"


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data_root", required=True)
    p.add_argument("--out", default="outputs/audit")
    p.add_argument("--tiles", type=int, default=200, help="tiles sampled per store")
    p.add_argument("--dfc23_raw", default="", help="raw DFC23 dir with rgb/ and dsm/")
    p.add_argument("--cartosat", default="",
                   help="a product folder/zip/tif, or 'PAN,MX' for a pair")
    for s in ("resolution", "vegetation", "classes", "overlaps"):
        p.add_argument(f"--{s}", action="store_true")
    a = p.parse_args(argv)
    every = not any((a.resolution, a.vegetation, a.classes, a.overlaps))
    root, out = Path(a.data_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stores = _label_stores(root) if root.is_dir() else {}
    print(f"[audit] stores: {', '.join(stores) or 'none'}")

    rep: dict = {}
    if every or a.resolution:
        rep["resolution"] = audit_resolution(root, stores, a.tiles, a.dfc23_raw)
    if every or a.vegetation:
        rep["vegetation"] = audit_vegetation(stores, a.tiles)
    if (every or a.classes) and "gamus" in stores:
        rep["classes"] = audit_classes(stores["gamus"], a.tiles, out)
    if every or a.overlaps:
        rep["overlaps"] = audit_overlaps(root)
    if a.cartosat:
        rep["cartosat"] = audit_cartosat(a.cartosat, stores, a.tiles)
    rep["flags"] = suggested_flags(rep)

    (out / "audit.json").write_text(json.dumps(rep, indent=2, default=str))
    md = to_markdown(rep)
    (out / "audit.md").write_text(md)
    print(md)
    bad = [k for k, s in rep.get("overlaps", {}).items()
           if s["shared_stems"] or s["shared_content"]]
    if bad:
        print(f"[audit] HELD-OUT LEAK: {bad}")
    if rep.get("classes", {}).get("pinning_ok") is False:
        print("[audit] class pinning disagrees with config.CLASS_NAMES — fix before training")
    return rep


if __name__ == "__main__":
    main()
