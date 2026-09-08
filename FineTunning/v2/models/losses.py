"""Losses for the 3-head net.

Regression terms (L1 + multi-scale gradient + SiLog) are lifted verbatim from
`FineTunning/v1/kaggle_phase0.py`.  New:
  * `bin_ce_loss`  — cross-entropy of Head B's per-pixel bin distribution against
    the bin that contains the GT height (fights the long tail).
  * `seg_ce_loss`  — Head C cross-entropy, only where `has_seg`.
  * `compute_losses` — weighted total over all heads + the fused prediction.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _mask(valid):
    return valid.bool()


def l1_loss(pred, target, valid):
    m = _mask(valid)
    return F.l1_loss(pred[m], target[m]) if m.any() else pred.sum() * 0.0


def gradient_loss(pred, target, valid, scales: int = 4):
    total = pred.sum() * 0.0
    p, t, v = pred, target, valid.float()
    for _ in range(scales):
        for d in (1, 2):
            dp = p.diff(dim=-d).abs()
            dt = t.diff(dim=-d).abs()
            m = (v.diff(dim=-d) == 0) & (v.narrow(-d, 0, v.shape[-d] - 1) > 0)
            if m.any():
                total = total + F.l1_loss(dp[m], dt[m])
        p = F.avg_pool2d(p, 2)
        t = F.avg_pool2d(t, 2)
        v = F.avg_pool2d(v, 2)
    return total / scales


def silog_loss(pred, target, valid, lam: float, shift: float):
    m = _mask(valid)
    if not m.any():
        return pred.sum() * 0.0
    g = torch.log(pred[m].clamp_min(0) + shift) - torch.log(target[m] + shift)
    return torch.sqrt((g ** 2).mean() - lam * (g.mean() ** 2) + 1e-7)


def regression_loss(pred, target, valid, cfg) -> tuple[torch.Tensor, dict]:
    l1 = l1_loss(pred, target, valid)
    grad = gradient_loss(pred, target, valid)
    sil = silog_loss(pred, target, valid, cfg.silog_lambda, cfg.silog_shift)
    total = cfg.w_l1 * l1 + cfg.w_grad * grad + cfg.w_silog * sil
    return total, {"l1": float(l1), "grad": float(grad), "silog": float(sil)}


def bin_ce_loss(probs, centres, target, valid) -> torch.Tensor:
    """probs (B,K,H,W), centres (B,K), target (B,1,H,W)."""
    m = _mask(valid)
    if not m.any():
        return probs.sum() * 0.0
    b, k = probs.shape[:2]
    # nearest bin index per pixel
    tgt = target.squeeze(1)                                   # (B,H,W)
    d = (tgt.unsqueeze(1) - centres.view(b, k, 1, 1)).abs()   # (B,K,H,W)
    bin_idx = d.argmin(dim=1)                                 # (B,H,W)
    logp = torch.log(probs.clamp_min(1e-8))
    ce = F.nll_loss(logp, bin_idx, reduction="none").unsqueeze(1)
    return ce[m].mean()


def seg_ce_loss(seg_logits, cls, has_seg) -> torch.Tensor:
    if has_seg is not None and not bool(has_seg.any()):
        return seg_logits.sum() * 0.0
    sel = has_seg.bool() if has_seg is not None else torch.ones(len(seg_logits), dtype=torch.bool)
    if not sel.any():
        return seg_logits.sum() * 0.0
    tgt = cls[sel].long()
    if (tgt != 7).sum() == 0:  # every pixel is ignore -> CE would be NaN
        return seg_logits.sum() * 0.0
    return F.cross_entropy(seg_logits[sel], tgt, ignore_index=7)


def compute_losses(out: dict, batch: dict, cfg) -> tuple[torch.Tensor, dict]:
    tgt, val = batch["target"], batch["valid"]
    la, sa = regression_loss(out["a"], tgt, val, cfg)
    lb, sb = regression_loss(out["b"], tgt, val, cfg)
    lf, sf = regression_loss(out["fused"], tgt, val, cfg)
    lbin = bin_ce_loss(out["b_probs"], out["b_centres"], tgt, val)
    lseg = seg_ce_loss(out["seg"], batch["cls"], batch.get("has_seg"))

    total = (
        cfg.w_head_a * la
        + cfg.w_head_b * lb
        + cfg.w_fused * lf
        + cfg.w_bin_ce * lbin
        + cfg.w_seg * lseg
    )
    stats = {
        "loss": float(total),
        "l1_a": sa["l1"], "l1_b": sb["l1"], "l1_fused": sf["l1"],
        "silog_fused": sf["silog"], "grad_fused": sf["grad"],
        "bin_ce": float(lbin), "seg_ce": float(lseg),
    }
    return total, stats
