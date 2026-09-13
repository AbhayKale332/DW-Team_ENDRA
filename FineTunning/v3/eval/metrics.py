"""Streaming depth metrics + the diagnostics v2 was missing.

Beyond global RMSE/MAE/r/delta1 this reports:
  * **per height stratum** and a **balanced RMSE** (mean of stratum RMSEs) — the
    global number is 65 % ground pixels and hides tall-structure error entirely.
    On the real v2 run global RMSE was 3.07 m while balanced RMSE was 4.19 m.
  * **tall-structure bias** — signed mean error above 15 m, which is the number
    that was -4.5 m in v2 and is the single biggest lever on the rubric.
  * **flat-ground bias** — signed mean error where GT < 1 m.  This is the metric
    that catches the Austin failure (predicting metres of height on flat ground)
    before it reaches a demo.
  * **per class id**, under neutral `classN` labels plus a measured histogram,
    because v1/v2's class *names* provably did not match GAMUS's id order.
"""

from __future__ import annotations

import torch

from config import HEIGHT_STRATA_M


class MetricAccum:
    def __init__(self):
        self.n = 0
        self.se = self.ae = self.sd = 0.0
        self.sx = self.sy = self.sxx = self.syy = self.sxy = 0.0
        self.d1 = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred, target = pred.double(), target.double()
        n = pred.numel()
        if n == 0:
            return
        self.n += n
        diff = pred - target
        self.se += float((diff ** 2).sum())
        self.ae += float(diff.abs().sum())
        self.sd += float(diff.sum())                     # signed -> bias
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
        n = self.n
        cov = self.sxy / n - (self.sx / n) * (self.sy / n)
        vx = self.sxx / n - (self.sx / n) ** 2
        vy = self.syy / n - (self.sy / n) ** 2
        return {
            "n": n,
            "rmse_m": (self.se / n) ** 0.5,
            "mae_m": self.ae / n,
            "bias_m": self.sd / n,
            "pearson_r": cov / ((vx * vy) ** 0.5 + 1e-12),
            "delta1": self.d1 / n,
        }


class Evaluator:
    """Accumulates every breakdown from (pred, target, valid, cls) batches."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.g = MetricAccum()
        self.cls = {i: MetricAccum() for i in range(len(cfg.class_names))}
        self.cls_px = {i: 0 for i in range(len(cfg.class_names))}
        self.cls_h = {i: 0.0 for i in range(len(cfg.class_names))}
        self.strata = [MetricAccum() for _ in range(len(HEIGHT_STRATA_M) - 1)]
        self.tall = MetricAccum()
        self.flat = MetricAccum()

    @torch.no_grad()
    def add(self, pred, target, valid, cls=None):
        valid = valid.bool()
        if not valid.any():
            return
        p, t = pred[valid], target[valid]
        self.g.update(p, t)
        for si in range(len(self.strata)):
            lo, hi = HEIGHT_STRATA_M[si], HEIGHT_STRATA_M[si + 1]
            sel = (t >= lo) & (t < hi)
            if sel.any():
                self.strata[si].update(p[sel], t[sel])
        sel = t >= 15.0
        if sel.any():
            self.tall.update(p[sel], t[sel])
        sel = t < 1.0
        if sel.any():
            self.flat.update(p[sel], t[sel])
        if cls is not None and self.cfg.per_class_metrics:
            c = cls[valid] if cls.shape == valid.shape else cls.unsqueeze(1)[valid]
            for i in self.cls:
                s = c == i
                if s.any():
                    self.cls[i].update(p[s], t[s])
                    self.cls_px[i] += int(s.sum())
                    self.cls_h[i] += float(t[s].sum())

    def result(self, tta: bool = False) -> dict:
        out = {"global": self.g.result(), "tta": tta}
        strat, rmses = {}, []
        for si in range(len(self.strata)):
            lo, hi = HEIGHT_STRATA_M[si], HEIGHT_STRATA_M[si + 1]
            label = f"{lo:g}-{hi:g}m" if hi < 1e8 else f"{lo:g}m+"
            r = self.strata[si].result()
            strat[label] = r
            if r.get("n"):
                rmses.append(r["rmse_m"])
        out["per_stratum"] = strat
        out["balanced_rmse_m"] = sum(rmses) / len(rmses) if rmses else None
        out["tall_gt15m"] = self.tall.result()
        out["flat_lt1m"] = self.flat.result()
        if self.cfg.per_class_metrics:
            out["per_class"] = {self.cfg.class_names[i]: a.result()
                                for i, a in self.cls.items() if a.n > 0}
            if self.cfg.dump_class_stats:
                tot = max(1, sum(self.cls_px.values()))
                out["class_stats"] = {
                    self.cfg.class_names[i]: {
                        "px_frac": self.cls_px[i] / tot,
                        "mean_gt_height_m": self.cls_h[i] / max(1, self.cls_px[i]),
                    } for i in self.cls if self.cls_px[i]
                }
        return out


@torch.no_grad()
@torch.no_grad()
def evaluate(model, loader, cfg, device, use_tta: bool = False,
             gpu_prep=None) -> dict:
    """`gpu_prep` is the trainer's `GpuPreproc`, or None for the all-CPU path.

    The decorator matters: without it the eval pass builds an autograd graph for
    a (B, 1, 512, 512) prediction it immediately throws away.  `Evaluator.add`
    was already `no_grad`, but by then the forward had allocated — which is why
    eval peaked higher than training and forced the batch down for both.
    """
    from models.tta import tta_predict

    model.eval()
    ev = Evaluator(cfg)
    amp_dt = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
    use_amp = cfg.amp and device.type == "cuda"
    for batch in loader:
        if gpu_prep is not None and "image_u8" in batch:
            batch = dict(batch)
            batch["image_u8"] = batch["image_u8"].to(device, non_blocking=True)
            batch = gpu_prep(batch, train=False)
        img = batch["image"].to(device, non_blocking=True)
        tgt = batch["target"].to(device, non_blocking=True)
        val = batch["valid"].to(device, non_blocking=True)
        if use_tta:
            pred = tta_predict(model, img, tuple(cfg.tta_scales), "fused",
                               amp_dtype=amp_dt if use_amp else None)
        else:
            with torch.autocast("cuda", dtype=amp_dt, enabled=use_amp):
                pred = model(img)["fused"]
            pred = pred.float()
        cls = batch["cls"].to(device, non_blocking=True).long().unsqueeze(1)
        ev.add(pred, tgt, val, cls)
    return ev.result(use_tta)


def format_line(m: dict) -> str:
    g = m["global"]
    tall, flat = m.get("tall_gt15m", {}), m.get("flat_lt1m", {})
    return (f"RMSE={g['rmse_m']:.3f} MAE={g['mae_m']:.3f} r={g['pearson_r']:.3f} "
            f"d1={g['delta1']:.3f} bal={m['balanced_rmse_m']:.3f} "
            f"tall_bias={tall.get('bias_m', float('nan')):+.2f} "
            f"flat_bias={flat.get('bias_m', float('nan')):+.2f}")
