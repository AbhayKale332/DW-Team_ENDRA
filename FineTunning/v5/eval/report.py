"""metrics.json, the viewer sample, and RGB | pred | GT | error strips."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def write_metrics_json(path, cfg, spec, history, best_rmse, elapsed_min, extra=None):
    from config import safe_config_dict

    payload = {
        "config": safe_config_dict(cfg),
        "preproc": spec.to_dict(),
        "history": history,
        "best_val_rmse_m": best_rmse,
        "elapsed_min": elapsed_min,
    }
    if extra:
        payload.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def colorize(a: np.ndarray, vmin: float = 0.0, vmax: float | None = None) -> np.ndarray:
    hi = float(vmax if vmax is not None else np.nanmax(a))
    x = np.clip((a - vmin) / max(hi - vmin, 1e-6), 0, 1)
    try:
        import matplotlib

        cmap = matplotlib.colormaps["turbo"]
        return (cmap(x)[..., :3] * 255).astype(np.uint8)
    except Exception:  # noqa: BLE001
        g = (x * 255).astype(np.uint8)
        return np.stack([g, g, g], -1)


def _predict(model, ds, i, cfg, spec, device):
    from infer.engine import predict_scene

    s = ds[i]
    rgb = s["rgb_u8"].numpy()
    amp_dt = (torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16) \
        if (cfg.amp and device.type == "cuda") else None
    height, _ = predict_scene(model, rgb, float(s["gsd_m"]), spec, device,
                              tta=False, amp_dtype=amp_dt,
                              batch_tiles=max(1, cfg.batch_size // 2))
    return s, rgb, height


@torch.no_grad()
def export_viewer_sample(model, ds, cfg, spec, device):
    """A metric-recoverable height map for `viewer/`.

    The 16-bit PNG carries an explicit affine encoding in meta.json AND a raw
    float32 .npy, so heights survive the trip to the renderer.  (v2 wrote a
    min-max normalised PNG whose scale lived only in a sibling file that the
    viewer never read, which is why its 3D range read 0-34 m regardless.)
    """
    from PIL import Image

    model.eval()
    s, rgb, pred = _predict(model, ds, 0, cfg, spec, device)
    gt = s["target"].numpy()
    d = Path(cfg.output_dir) / "viewer_sample"
    d.mkdir(parents=True, exist_ok=True)

    Image.fromarray(rgb).save(d / "rgb.png")
    lo, hi = 0.0, float(max(np.nanmax(pred), 1.0))
    Image.fromarray(((np.clip(pred, lo, hi) - lo) / (hi - lo) * 65535)
                    .astype(np.uint16)).save(d / "height16.png")
    np.save(d / "pred_ndsm_m.npy", pred.astype(np.float32))
    np.save(d / "gt_ndsm_m.npy", gt.astype(np.float32))
    (d / "meta.json").write_text(json.dumps({
        "stem": s["stem"], "src": s["src"],
        "height_min_m": lo, "height_max_m": hi,
        "gsd_m": float(s["gsd_m"]), "size_px": list(pred.shape),
        "encode": "height_m = height_min_m + (png16/65535)*(height_max_m-height_min_m)",
    }, indent=2))
    print(f"[viewer] wrote sample -> {d}")


@torch.no_grad()
def export_qualitative(model, ds, cfg, spec, device, n: int) -> list:
    """Write the RGB | pred | GT | |err| strips and return the raw triples.

    The arrays come back so `viz.figures` can build the scatter, the error
    histogram and the hillshade from the same predictions the strips show,
    instead of re-running inference on the same tiles a second time.
    """
    from PIL import Image

    model.eval()
    d = Path(cfg.output_dir) / "qualitative"
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(min(n, len(ds))):
        s, rgb, pred = _predict(model, ds, i, cfg, spec, device)
        gt = s["target"].numpy()
        vmax = float(max(gt.max(), pred.max(), 1.0))
        strip = np.concatenate([
            rgb, colorize(pred, 0, vmax), colorize(gt, 0, vmax),
            colorize(np.abs(pred - gt), 0, max(vmax * 0.4, 1.0)),
        ], axis=1)
        Image.fromarray(strip).save(d / f"{s['src']}_{s['stem']}.png")
        out.append((rgb, pred.astype(np.float32), gt.astype(np.float32)))
    print(f"[qual] wrote {len(out)} strips -> {d}  (RGB | pred | GT | |err|)")
    return out


def _gallery_stores(cfg) -> list[tuple[str, str, bool]]:
    """(store, split, seen_in_training) — one store per `gallery_sources` family.

    A family is prefix-matched (`dfc23` -> `dfc23_g050`), preferring a store the
    run trained on, and the split falls back val -> test -> train.  SynRS3D is
    packed train-only, so its tiles are training tiles and the report says so.
    """
    from dwdata.loaders import val_split_of
    from dwdata.packed import store_exists

    root = Path(cfg.data_root)
    trained = set(cfg.labeled_sources())
    on_disk = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    out = []
    for fam in (f.strip() for f in str(cfg.gallery_sources or "").split(",")):
        if not fam:
            continue
        names = [n for n in on_disk if n == fam or n.startswith(fam + "_")]
        names.sort(key=lambda n: (n not in trained, n))
        pick = None
        for name in names:
            for split in dict.fromkeys(s for s in (val_split_of(name), "test", "train") if s):
                if store_exists(root / name / split):
                    pick = (name, split, split == "train" and name in trained)
                    break
            if pick:
                break
        if pick:
            out.append(pick)
        else:
            print(f"[gallery] no prepared store for '{fam}' under {root}; skipped")
    return out


@torch.no_grad()
def export_gallery(model, cfg, spec, device) -> list[dict]:
    """A few predicted tiles per source family, for the report's gallery.

    Returns one entry per store: {store, split, seen_in_training, gsd_m, tiles},
    each tile {stem, rgb, pred, gt, valid, rmse_m, mae_m}.  Tiles are a seeded
    sample, not the sorted-stem prefix — the prefix is one city.  The per-tile
    numbers are captions, not metrics.
    """
    from dwdata.dataset import FullTileDataset
    from dwdata.loaders import _open, sample_indices

    k = int(getattr(cfg, "gallery_tiles", 0) or 0)
    if k <= 0:
        return []
    model.eval()
    out = []
    for name, split, seen in _gallery_stores(cfg):
        st = _open(Path(cfg.data_root), name, split)
        idx = sample_indices(len(st), min(k, len(st)), 0)
        ds = FullTileDataset(cfg, st, spec, name, length=min(k, len(st)), indices=idx)
        tiles = []
        for i in range(len(ds)):
            s, rgb, pred = _predict(model, ds, i, cfg, spec, device)
            gt, valid = s["target"].numpy(), s["valid"].numpy().astype(bool)
            e = (pred - gt)[valid]
            tiles.append({
                "stem": str(s["stem"]), "rgb": rgb, "pred": pred.astype(np.float32),
                "gt": gt.astype(np.float32), "valid": valid,
                "rmse_m": float(np.sqrt(np.mean(e ** 2))) if e.size else None,
                "mae_m": float(np.mean(np.abs(e))) if e.size else None,
            })
        out.append({"store": name, "split": split, "seen_in_training": seen,
                    "gsd_m": float(st.gsd_m), "tiles": tiles})
        print(f"[gallery] {name}/{split}: {len(tiles)} tiles"
              + ("  (training tiles — seen by the model)" if seen else ""))
    return out
