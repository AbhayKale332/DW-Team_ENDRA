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
  * **per landscape** (urban / sparse / hilly / forested) — v4.  This is the
    rubric's own stability axis and nothing in v1-v3 answered it; the label is
    derived per tile from the GT height field (`eval/landscape.py`), so it needs
    no annotation and works on every source.
"""

from __future__ import annotations

import torch

from config import HEIGHT_STRATA_M, LANDSCAPE_NAMES


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

    def __init__(self, cfg, src: str = ""):
        self.cfg = cfg
        # NEON-like sources have no buildings: their "urban" tiles are canopy
        self.no_urban = bool(cfg.no_urban(src)) if hasattr(cfg, "no_urban") else False
        self.g = MetricAccum()
        self.cls = {i: MetricAccum() for i in range(len(cfg.class_names))}
        self.cls_px = {i: 0 for i in range(len(cfg.class_names))}
        self.cls_h = {i: 0.0 for i in range(len(cfg.class_names))}
        self.strata = [MetricAccum() for _ in range(len(HEIGHT_STRATA_M) - 1)]
        self.tall = MetricAccum()
        self.flat = MetricAccum()
        self.pred_strata = [MetricAccum() for _ in self.strata]
        self.pred_tall = MetricAccum()
        self.block30 = MetricAccum()
        self.block30_sizes = set()
        self.land = {k: MetricAccum() for k in LANDSCAPE_NAMES}
        self.land_tiles = {k: 0 for k in LANDSCAPE_NAMES}
        self.land_desc: list[dict] = []
        # v5: does the map have the GT's edges?
        self.edge = MetricAccum()
        self.grad_p = 0.0
        self.grad_t = 0.0

    @torch.no_grad()
    def add_tiles(self, pred, target, valid, gsd_m=1.0):
        """Per-tile landscape breakdown.  `pred/target/valid` are (B, [1,] H, W).

        `classify` is numpy, so the GT and its mask have to come back to the
        host — but they come back **once per batch**, not twice per tile plus an
        `.any()` sync per tile.  At 400 val tiles and an eval every other epoch
        that was ~24 000 device syncs a run for a breakdown that is pure
        bookkeeping.
        """
        if not getattr(self.cfg, "per_landscape_metrics", False):
            return
        from eval.landscape import classify

        p = pred if pred.dim() == 3 else pred[:, 0]
        t = target if target.dim() == 3 else target[:, 0]
        v = valid if valid.dim() == 3 else valid[:, 0]
        t_np = t.detach().to("cpu", torch.float32).numpy()
        v_np = v.detach().to("cpu", torch.bool).numpy()
        if torch.is_tensor(gsd_m):
            g_np = gsd_m.detach().to("cpu", torch.float32).reshape(-1).numpy()
        else:
            g_np = None
        for i in range(t_np.shape[0]):
            if not v_np[i].any():                       # host-side, free
                continue
            g = float(g_np[i % len(g_np)]) if g_np is not None else float(gsd_m)
            name, desc = classify(t_np[i], v_np[i], g, no_urban=self.no_urban)
            vi = v[i].bool()
            self.land[name].update(p[i][vi], t[i][vi])
            self.land_tiles[name] += 1
            if len(self.land_desc) < 64:
                self.land_desc.append({"landscape": name, **desc})

    @torch.no_grad()
    def add_spatial(self, pred, target, valid, step_m: float = 2.0, band_px: int = 2):
        """v5 sharpness metrics, on (B, 1, H, W) maps.

        * `edge_rmse_m` — RMSE on pixels within `band_px` of a GT height step
          (3x3 max - min > `step_m`): building walls and crown edges, where a
          blurred map pays most and a global RMSE hides it.
        * `grad_ratio` — sum |grad pred| / sum |grad GT| over valid pixel pairs.
          ~1 means as much relief as the reference; v4's blobs read well below 1.
        """
        import torch.nn.functional as F

        p = pred.float().reshape(-1, 1, *pred.shape[-2:])
        t = target.float().reshape_as(p)
        v = valid.bool().reshape_as(p)
        big = torch.where(v, t, torch.full_like(t, -1e6))
        small = torch.where(v, t, torch.full_like(t, 1e6))
        rng = F.max_pool2d(big, 3, 1, 1) + F.max_pool2d(-small, 3, 1, 1)
        edge = (rng > step_m) & (rng < 5e5)
        k = 2 * band_px + 1
        edge = F.max_pool2d(edge.float(), k, 1, band_px) > 0
        sel = edge & v
        if sel.any():
            self.edge.update(p[sel], t[sel])
        for d in (2, 3):
            dp = p.diff(dim=d).abs()
            dt = t.diff(dim=d).abs()
            vv = v.narrow(d, 0, v.shape[d] - 1) & v.narrow(d, 1, v.shape[d] - 1)
            self.grad_p += float(dp[vv].sum())
            self.grad_t += float(dt[vv].sum())

    @torch.no_grad()
    def add_30m(self, pred, target, valid, gsd_m):
        """Tile-aligned nDSM block diagnostic, not an absolute DEM score.

        Only complete, fully valid blocks count; incomplete tile margins drop.
        Pixel count is rounded to the closest achievable 30 m block size.
        """
        if not getattr(self.cfg, "eval_bias_diagnostics", False):
            return
        p = pred.reshape(-1, 1, *pred.shape[-2:])
        t = target.reshape_as(p)
        v = valid.reshape_as(p).bool() & torch.isfinite(p) & torch.isfinite(t)
        gsds = torch.as_tensor(gsd_m).reshape(-1).tolist()
        for i in range(p.shape[0]):
            g = float(gsds[i % len(gsds)])
            if not g > 0:
                raise ValueError("30 m diagnostics require a positive GSD")
            k = max(1, round(30.0 / g))
            self.block30_sizes.add(round(k * g, 6))
            if min(p.shape[-2:]) < k:
                continue
            # Invalid NaNs must not contaminate otherwise valid pooling cells.
            pp, tt, vv = pool_pair(torch.where(v[i:i+1], p[i:i+1], 0),
                                   torch.where(v[i:i+1], t[i:i+1], 0), v[i:i+1], k)
            vv = torch.nn.functional.max_pool2d((~v[i:i+1]).float(), k) == 0
            self.block30.update(pp[vv], tt[vv])

    @torch.no_grad()
    def add(self, pred, target, valid, cls=None):
        valid = valid.bool()
        if not valid.any():
            return
        p, t = pred[valid], target[valid]
        self.g.update(p, t)
        if getattr(self.cfg, "eval_bias_diagnostics", False):
            for si, accum in enumerate(self.pred_strata):
                lo, hi = HEIGHT_STRATA_M[si:si + 2]
                sel = (p >= lo) & (p < hi)
                accum.update(p[sel], t[sel])
            sel = p >= 15.0
            self.pred_tall.update(p[sel], t[sel])
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
        if getattr(self.cfg, "eval_bias_diagnostics", False):
            out["tall_pred15m"] = self.pred_tall.result()
            out["per_predicted_stratum"] = {
                label: accum.result() for label, accum in zip(strat, self.pred_strata)}
            out["ndsm_30m"] = {
                **self.block30.result(), "actual_block_sizes_m": sorted(self.block30_sizes),
                "protocol": "tile-aligned complete fully-valid block means; bias = pred - GT"}
        if self.edge.n:
            out["edge_rmse_m"] = self.edge.result()["rmse_m"]
            out["edge_band"] = self.edge.result()
        if self.grad_t > 0:
            out["grad_ratio"] = self.grad_p / self.grad_t
        if getattr(self.cfg, "per_landscape_metrics", False) and any(
                a.n for a in self.land.values()):
            per = {k: a.result() for k, a in self.land.items() if a.n > 0}
            for k in per:
                per[k]["tiles"] = self.land_tiles[k]
            out["per_landscape"] = per
            # the rubric grades *stability*, so report the spread explicitly
            r = [v["rmse_m"] for v in per.values()]
            out["landscape_rmse_spread_m"] = max(r) - min(r) if len(r) > 1 else 0.0
            out["landscape_worst"] = max(per, key=lambda k: per[k]["rmse_m"])
            out["landscape_descriptors"] = self.land_desc
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


def pool_pair(pred, target, valid, k: int):
    """k x k block means of pred / target, valid only where the whole block is.

    How a coarse-label source (DFC23 / India, ~2 m stereo on 0.5 m pixels) is
    scored in v5: at the resolution its labels actually have, so a sharp map is
    not charged for edges its reference cannot represent.
    """
    import torch.nn.functional as F

    vf = valid.float()
    n = F.avg_pool2d(vf, k)
    pp = F.avg_pool2d(pred.float() * vf, k) / n.clamp_min(1e-6)
    tt = F.avg_pool2d(target.float() * vf, k) / n.clamp_min(1e-6)
    return pp, tt, n > 0.999


def dataset_src(ds) -> str:
    """The `src` of a TileDataset / FullTileDataset, through any Subset wrappers."""
    while ds is not None and not hasattr(ds, "src") and hasattr(ds, "dataset"):
        ds = ds.dataset
    return str(getattr(ds, "src", "") or "")


@torch.no_grad()
def evaluate(model, loader, cfg, device, use_tta: bool = False,
             gpu_prep=None, pool: int = 1, mask_veg: bool = False,
             extra_pool: int = 1) -> dict:
    """`gpu_prep` is the trainer's `GpuPreproc`, or None for the all-CPU path.

    `extra_pool` adds a label-resolution companion from the same predictions,
    keeping the native metrics and avoiding another model forward pass.

    The decorator matters: without it the eval pass builds an autograd graph for
    a (B, 1, 512, 512) prediction it immediately throws away.  `Evaluator.add`
    was already `no_grad`, but by then the forward had allocated — which is why
    eval peaked higher than training and forced the batch down for both.

    `mask_veg` leaves out the pixels the training loss masks (ExG vegetation
    labelled < 1 m, from the batch's `veg`).  DFC23 / India put every tree at
    0 m, so on those sets a model that predicts canopy is charged for it; this
    is the score that separates "predicts trees" from "lifts the ground".
    """
    from models.tta import tta_predict

    model.eval()
    ev = Evaluator(cfg, dataset_src(getattr(loader, "dataset", None)))
    coarse = Evaluator(cfg, dataset_src(getattr(loader, "dataset", None))) if extra_pool > 1 else None
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
        if mask_veg and "veg" in batch:
            veg = batch["veg"].to(device, non_blocking=True).bool()
            val = val & ~(veg & (tgt < 1.0))
        if use_tta:
            pred = tta_predict(model, img, tuple(cfg.tta_scales), "fused",
                               amp_dtype=amp_dt if use_amp else None)
        else:
            with torch.autocast("cuda", dtype=amp_dt, enabled=use_amp):
                pred = model(img)["fused"]
            pred = pred.float()
        cls = batch["cls"].to(device, non_blocking=True).long().unsqueeze(1)
        if pool > 1:
            pred, tgt, val = pool_pair(pred, tgt, val, pool)
            cls = None
        ev.add(pred, tgt, val, cls)
        ev.add_spatial(pred, tgt, val)
        ev.add_30m(pred, tgt, val, torch.as_tensor(batch.get("gsd_m", 1.0)) * pool)
        ev.add_tiles(pred, tgt, val, torch.as_tensor(batch.get("gsd_m", 1.0)) * pool)
        if coarse is not None:
            pp, tt, vv = pool_pair(pred, tgt, val, extra_pool)
            gsd = torch.as_tensor(batch.get("gsd_m", 1.0)) * pool * extra_pool
            coarse.add(pp, tt, vv)
            coarse.add_spatial(pp, tt, vv)
            coarse.add_tiles(pp, tt, vv, gsd)
            coarse.add_30m(pp, tt, vv, gsd)
    result = ev.result(use_tta)
    if coarse is not None:
        result[f"pooled{extra_pool}"] = coarse.result(use_tta)
    return result


def format_line(m: dict) -> str:
    g = m["global"]
    tall, flat = m.get("tall_gt15m", {}), m.get("flat_lt1m", {})
    line = (f"RMSE={g['rmse_m']:.3f} MAE={g['mae_m']:.3f} r={g['pearson_r']:.3f} "
            f"d1={g['delta1']:.3f} bal={m['balanced_rmse_m']:.3f} "
            f"tall_bias={tall.get('bias_m', float('nan')):+.2f} "
            f"flat_bias={flat.get('bias_m', float('nan')):+.2f}")
    if "edge_rmse_m" in m:
        line += f" edge={m['edge_rmse_m']:.3f}"
    if "grad_ratio" in m:
        line += f" grad={m['grad_ratio']:.2f}"
    if m.get("per_landscape"):
        land = " ".join(f"{k[:4]}={v['rmse_m']:.2f}"
                        for k, v in sorted(m["per_landscape"].items()))
        line += f"  [{land}]"
    return line
