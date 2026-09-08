"""Streaming depth metrics.

`MetricAccum` is v1's exact online RMSE/MAE/Pearson/delta1 accumulator.
`evaluate()` adds:
  * per-land-cover breakdown (the rubric's "stability across urban/sparse/hilly/forested")
  * per-height-stratum breakdown + a height-balanced RMSE (mean of stratum RMSEs)
    — exposes tall-structure underestimation, which a global RMSE hides
  * optional 8x TTA path
"""

from __future__ import annotations

import torch

from config import HEIGHT_STRATA_M


class MetricAccum:
    def __init__(self):
        self.n = 0
        self.se = self.ae = 0.0
        self.sx = self.sy = self.sxx = self.syy = self.sxy = 0.0
        self.d1 = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.double()
        target = target.double()
        n = pred.numel()
        if n == 0:
            return
        self.n += n
        diff = pred - target
        self.se += float((diff ** 2).sum())
        self.ae += float(diff.abs().sum())
        self.sx += float(pred.sum())
        self.sy += float(target.sum())
        self.sxx += float((pred ** 2).sum())
        self.syy += float((target ** 2).sum())
        self.sxy += float((pred * target).sum())
        r = torch.maximum(pred + 1.0, target + 1.0) / torch.minimum(pred + 1.0, target + 1.0)
        self.d1 += int((r < 1.25).sum())

    def result(self) -> dict:
        if self.n == 0:
            return {"n": 0}
        rmse = (self.se / self.n) ** 0.5
        mae = self.ae / self.n
        cov = self.sxy / self.n - (self.sx / self.n) * (self.sy / self.n)
        vx = self.sxx / self.n - (self.sx / self.n) ** 2
        vy = self.syy / self.n - (self.sy / self.n) ** 2
        pearson = cov / ((vx * vy) ** 0.5 + 1e-12)
        return {"n": self.n, "rmse_m": rmse, "mae_m": mae,
                "pearson_r": pearson, "delta1": self.d1 / self.n}


@torch.no_grad()
def evaluate(model, loader, cfg, device, use_tta: bool = False) -> dict:
    from models.tta import tta_predict

    model.eval()
    gm = MetricAccum()
    cm = {i: MetricAccum() for i in range(len(cfg.class_names))} if cfg.per_class_metrics else {}
    strata = [MetricAccum() for _ in range(len(HEIGHT_STRATA_M) - 1)]

    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        tgt = batch["target"].to(device, non_blocking=True)
        val = batch["valid"].to(device, non_blocking=True).bool()
        if use_tta:
            pred = tta_predict(model, img, tuple(cfg.tta_scales), key="fused")
        else:
            with torch.autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
                pred = model(img)["fused"].float()
        p, t = pred[val], tgt[val]
        gm.update(p, t)

        if cm:
            cls = batch["cls"].to(device, non_blocking=True).unsqueeze(1)[val]
            for ci, acc in cm.items():
                sel = cls == ci
                if sel.any():
                    acc.update(p[sel], t[sel])

        for si in range(len(strata)):
            lo, hi = HEIGHT_STRATA_M[si], HEIGHT_STRATA_M[si + 1]
            sel = (t >= lo) & (t < hi)
            if sel.any():
                strata[si].update(p[sel], t[sel])

    out = {"global": gm.result()}
    if cm:
        out["per_class"] = {
            cfg.class_names[i]: acc.result() for i, acc in cm.items() if acc.n > 0
        }
    stratum_res = {}
    rmses = []
    for si in range(len(strata)):
        lo, hi = HEIGHT_STRATA_M[si], HEIGHT_STRATA_M[si + 1]
        label = f"{lo:g}-{hi:g}m" if hi < 1e8 else f"{lo:g}m+"
        r = strata[si].result()
        stratum_res[label] = r
        if r.get("n"):
            rmses.append(r["rmse_m"])
    out["per_stratum"] = stratum_res
    out["balanced_rmse_m"] = sum(rmses) / len(rmses) if rmses else None
    out["tta"] = use_tta
    return out
