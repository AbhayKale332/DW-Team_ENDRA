"""Full-tile evaluation through the *inference* path.

Training-time validation is one 512 centre crop per tile (fast, comparable
epoch to epoch).  The number that belongs in the report is this one: every pixel
of every val tile, scored at the tile's native GSD, produced by exactly the code
the demo runs.
"""

from __future__ import annotations

import numpy as np
import torch

from eval.metrics import Evaluator
from infer.engine import predict_scene


@torch.no_grad()
def sliding_eval(model, full_ds, cfg, spec, device, *, tta: bool = False,
                 max_tiles: int = 0, log_every: int = 25) -> dict:
    model.eval()
    ev = Evaluator(cfg)
    amp_dt = (torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16) \
        if (cfg.amp and device.type == "cuda") else None
    n = min(len(full_ds), max_tiles) if max_tiles else len(full_ds)
    for i in range(n):
        s = full_ds[i]
        rgb = s["rgb_u8"].numpy()
        gsd = float(s["gsd_m"])
        height, _ = predict_scene(model, rgb, gsd, spec, device, tta=tta,
                                  tta_scales=tuple(cfg.tta_scales), amp_dtype=amp_dt,
                                  overlap=0.25, batch_tiles=max(1, cfg.batch_size // 2))
        pred_t = torch.from_numpy(height).to(device)
        tgt_t, val_t = s["target"].to(device), s["valid"].to(device)
        ev.add(pred_t, tgt_t, val_t, s["cls"].to(device))
        ev.add_tiles(pred_t.unsqueeze(0), tgt_t.unsqueeze(0), val_t.unsqueeze(0), gsd)
        if log_every and (i + 1) % log_every == 0:
            print(f"  [sliding] {i + 1}/{n}", flush=True)
    return ev.result(tta)
