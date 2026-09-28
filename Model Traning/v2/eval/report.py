"""Write metrics.json, the viewer sample, and qualitative RGB|pred|GT|error strips."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def write_metrics_json(path, cfg, history, best_rmse, elapsed_min, extra=None):
    from config import safe_config_dict

    payload = {
        "config": safe_config_dict(cfg),
        "history": history,
        "best_val_rmse_m": best_rmse,
        "elapsed_min": elapsed_min,
    }
    if extra:
        payload.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def _colorize(a: np.ndarray, vmax: float | None = None) -> np.ndarray:
    lo = float(np.nanmin(a))
    hi = float(vmax if vmax is not None else np.nanmax(a))
    span = max(hi - lo, 1e-6)
    x = np.clip((a - lo) / span, 0, 1)
    try:
        import matplotlib

        try:
            cmap = matplotlib.colormaps["turbo"]           # mpl >= 3.6
        except Exception:  # noqa: BLE001
            from matplotlib import cm

            cmap = cm.get_cmap("turbo")
        return (cmap(x)[..., :3] * 255).astype(np.uint8)
    except Exception:  # noqa: BLE001
        g = (x * 255).astype(np.uint8)
        return np.stack([g, g, g], -1)


@torch.no_grad()
def export_viewer_sample(model, loader, cfg, device):
    from PIL import Image

    model.eval()
    batch = next(iter(loader))
    out = model(batch["image"][:1].to(device))
    pred = out["fused"].float()[0, 0].cpu().numpy()
    gt = batch["target"][0, 0].numpy()
    rgb = batch["rgb_u8"][0].numpy().astype(np.uint8)

    d = Path(cfg.output_dir) / "viewer_sample"
    d.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(d / "rgb.png")
    lo, hi = float(pred.min()), float(pred.max())
    span = max(hi - lo, 1e-6)
    Image.fromarray(((pred - lo) / span * 65535).astype(np.uint16)).save(d / "height16.png")
    np.save(d / "pred_ndsm_m.npy", pred.astype(np.float32))
    np.save(d / "gt_ndsm_m.npy", gt.astype(np.float32))
    with open(d / "meta.json", "w") as f:
        json.dump({
            "stem": batch["stem"][0], "src": batch["src"][0],
            "height_min_m": lo, "height_max_m": hi,
            "gsd_m": float(batch.get("gsd_m", [cfg.canonical_gsd_m])[0]),
            "size_px": cfg.tile_size,
            "encode": "height_m = height_min_m + (png16/65535)*(height_max_m-height_min_m)",
        }, f, indent=2)
    print(f"[viewer] wrote sample -> {d}")


@torch.no_grad()
def export_qualitative(model, loader, cfg, device, n: int):
    from PIL import Image

    model.eval()
    d = Path(cfg.output_dir) / "qualitative"
    d.mkdir(parents=True, exist_ok=True)
    done = 0
    for batch in loader:
        out = model(batch["image"].to(device))
        pred = out["fused"].float().cpu().numpy()[:, 0]
        gt = batch["target"].numpy()[:, 0]
        rgb = batch["rgb_u8"].numpy().astype(np.uint8)
        for i in range(len(pred)):
            vmax = float(max(gt[i].max(), pred[i].max(), 1.0))
            err = np.abs(pred[i] - gt[i])
            strip = np.concatenate([
                rgb[i],
                _colorize(pred[i], vmax),
                _colorize(gt[i], vmax),
                _colorize(err, vmax),
            ], axis=1)
            Image.fromarray(strip).save(d / f"{batch['src'][i]}_{batch['stem'][i]}.png")
            done += 1
            if done >= n:
                print(f"[qual] wrote {done} strips -> {d}")
                return
