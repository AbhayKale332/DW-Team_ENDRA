"""Losses.

Three things the measured v2 checkpoint told us to fix:

1. **Tall structures are underestimated.**  On GAMUS val tile DC_38_35, pixels
   with GT > 15 m had mean GT 20.0 m and mean prediction 15.5 m (MAE 5.3 m) while
   flat ground was near-perfect.  A pixel-uniform L1 is ~65 % ground pixels, so
   the tail contributes almost nothing to the gradient.  `stratum_weights()`
   re-weights each pixel by the inverse frequency of its height stratum (the same
   strata the balanced-RMSE metric uses), tracked with a running EMA so the
   weights are stable across batches.

2. **Head B never learned.**  v2's bin cross-entropy sat at 0.06-0.13 for the
   whole run — the nearest-bin target is trivially predictable when 65 % of the
   mass is one bin, so the head collapsed and the gate just copied Head A.  Here
   the bin loss is computed on *logits* (not on log(probs)), at half resolution,
   with the same stratum weighting, and the bin index comes from a batched
   `searchsorted` instead of materialising a (B, K, H, W) distance tensor.

3. **Texture is being read as height.**  On out-of-domain imagery v2 turned image
   texture into terrain (median predicted height 4 m on a residential Austin
   scene; the 3D render is a mountain range).  `normal_loss` matches surface
   orientation rather than raw gradients, and `flatness_loss` explicitly penalises
   curvature where the GT surface is locally flat — together they push the model
   towards piecewise-planar ground with discrete structures on top.

v4 changes one of these and adds two:

4. **The bin head still had a trivial target.**  v3 kept the hard nearest-bin
   index, which over 96 bins with ~65 % of the mass on the ground bin is almost
   free to predict — the same reason it collapsed in v2.  `bin_ce_loss` now
   builds a **Gaussian-smoothed** target over neighbouring bins
   (`bin_soft_sigma`, measured in bins) and `bin_entropy_reg` puts a floor on the
   *marginal* bin distribution's entropy, so a head that dumps every pixel into
   one bin is penalised directly instead of being rewarded for it.

5. **`consistency_loss`** is the unlabeled branch: an EMA teacher predicts a
   weakly-augmented view, the student is trained towards it on a strongly
   augmented one, and pixels the teacher is unsure about are dropped.  This is
   the only lever in the repo that can move the network towards Indian imagery,
   because no Indian RGB + height pairs are openly published (README §2).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from config import HEIGHT_STRATA_M, SEG_IGNORE_INDEX

_STRATA = torch.tensor(HEIGHT_STRATA_M[1:-1], dtype=torch.float32)  # inner edges


class StratumBalancer:
    """Running EMA of height-stratum frequencies -> per-pixel loss weights."""

    def __init__(self, beta: float = 0.5, clip: float = 5.0, momentum: float = 0.98):
        self.beta, self.clip, self.momentum = float(beta), float(clip), float(momentum)
        self.freq: torch.Tensor | None = None

    @torch.no_grad()
    def _update(self, idx: torch.Tensor, n_strata: int) -> torch.Tensor:
        counts = torch.bincount(idx.reshape(-1), minlength=n_strata).float()
        p = counts / counts.sum().clamp_min(1.0)
        if self.freq is None or self.freq.shape != p.shape:
            self.freq = p
        else:
            self.freq = self.momentum * self.freq.to(p.device) + (1 - self.momentum) * p
        return self.freq

    def weights(self, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """(B,1,H,W) float weights, mean ~1 over the valid pixels."""
        if self.beta <= 0:
            return torch.ones_like(target)
        edges = _STRATA.to(target.device)
        idx = torch.bucketize(target.detach(), edges)          # 0..len(edges)
        n = edges.numel() + 1
        freq = self._update(idx[valid] if valid.any() else idx, n)
        w = (1.0 / freq.clamp_min(1e-4)) ** self.beta
        # Normalise by the *frequency-weighted* mean, not the plain mean over
        # strata: a stratum with no pixels in this batch has an enormous raw
        # weight, and dividing by the plain mean let those empty strata drag
        # every real weight below the clip floor — which flattened the weights
        # back to uniform and quietly turned the tail re-weighting off.
        w = (w / (freq * w).sum().clamp_min(1e-6)).clamp(1.0 / self.clip, self.clip)
        out = w.to(target.dtype)[idx]
        m = out[valid].mean() if valid.any() else out.mean()
        return out / m.clamp_min(1e-6)


# ---------------------------------------------------------------------
def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    return x[m].mean() if m.any() else x.sum() * 0.0


def l1_loss(pred, target, valid, w=None):
    e = (pred - target).abs()
    if w is not None:
        e = e * w
    return _masked_mean(e, valid)


def silog_loss(pred, target, valid, lam: float, shift: float):
    if not valid.any():
        return pred.sum() * 0.0
    g = torch.log(pred[valid].clamp_min(0) + shift) - torch.log(target[valid] + shift)
    return torch.sqrt((g ** 2).mean() - lam * (g.mean() ** 2) + 1e-7)


def gradient_loss(pred, target, valid, scales: int = 4):
    """Multi-scale L1 on first differences — edge sharpness."""
    total = pred.sum() * 0.0
    p, t, v = pred, target, valid.float()
    for _ in range(scales):
        if p.shape[-1] < 4 or p.shape[-2] < 4:
            break
        for d in (1, 2):
            dp = p.diff(dim=-d)
            dt = t.diff(dim=-d)
            m = (v.diff(dim=-d) == 0) & (v.narrow(-d, 0, v.shape[-d] - 1) > 0.5)
            total = total + _masked_mean((dp - dt).abs(), m)
        p, t, v = F.avg_pool2d(p, 2), F.avg_pool2d(t, 2), F.avg_pool2d(v, 2)
    return total / scales


def normal_loss(pred, target, valid, gsd_m=None):
    """Cosine distance between predicted and GT surface normals.

    Gradients are converted to metres-per-metre using the tile's GSD, so the
    orientation the loss sees is the *physical* slope and is comparable across
    the scale-augmented range instead of drifting with pixel spacing.
    """
    if pred.shape[-1] < 2 or pred.shape[-2] < 2:
        return pred.sum() * 0.0
    if gsd_m is None:
        sx = sy = 1.0
    else:
        g = gsd_m.view(-1, 1, 1, 1).clamp_min(1e-3)
        sx = sy = 1.0 / g

    def nrm(z):
        dzdx = (z[..., :, 1:] - z[..., :, :-1])[..., :-1, :] * sx
        dzdy = (z[..., 1:, :] - z[..., :-1, :])[..., :, :-1] * sy
        n = torch.cat([-dzdx, -dzdy, torch.ones_like(dzdx)], dim=1)
        return n / n.norm(dim=1, keepdim=True).clamp_min(1e-6)

    m = (valid[..., :-1, :-1] & valid[..., 1:, :-1] & valid[..., :-1, 1:])
    cos = (nrm(pred) * nrm(target)).sum(dim=1, keepdim=True)
    return _masked_mean(1.0 - cos, m)


def flatness_loss(pred, target, valid, cls=None, flat_ids=(0, 3, 4), tol_m: float = 0.35):
    """Penalise predicted curvature where the ground truth is locally flat.

    This is the direct counter to "texture becomes terrain": on a road, a lawn or
    water the GT Laplacian is ~0, so any structure the network invents from image
    texture is paid for.  Restricted to GT-flat pixels, so real building edges are
    untouched.
    """
    lap = lambda z: (z[..., 1:-1, 1:-1] * 4.0 - z[..., :-2, 1:-1] - z[..., 2:, 1:-1]  # noqa: E731
                     - z[..., 1:-1, :-2] - z[..., 1:-1, 2:])
    if pred.shape[-1] < 3 or pred.shape[-2] < 3:
        return pred.sum() * 0.0
    lp, lt = lap(pred), lap(target)
    m = valid[..., 1:-1, 1:-1] & (lt.abs() < tol_m)
    if cls is not None:
        c = cls.unsqueeze(1) if cls.dim() == pred.dim() - 1 else cls
        c = c[..., 1:-1, 1:-1]
        sem = torch.zeros_like(m)
        for i in flat_ids:
            sem |= (c == i)
        # only tighten where we actually have labels
        if sem.any():
            m = m & sem
    return _masked_mean(lp.abs(), m)


def bin_target_index(centres, target, h, w_):
    """Nearest-bin index of every target pixel, at the logits' resolution.

    `centres` may carry a batch of 1 (one shared set of bin edges); it is
    broadcast to the target's batch rather than reshaped against it, because
    silently reinterpreting the flat buffer is how you get an index map that is
    the right shape and the wrong pixels.
    """
    k = centres.shape[1]
    b = target.shape[0]
    if centres.shape[0] != b:
        centres = centres.expand(b, k)
    tgt = F.interpolate(target, size=(h, w_), mode="nearest")
    mids = (0.5 * (centres[:, 1:] + centres[:, :-1])).contiguous()          # (B, K-1)
    idx = torch.searchsorted(mids, tgt.reshape(b, -1).contiguous().to(mids.dtype))
    return idx.clamp(0, k - 1).reshape(b, h, w_)


def bin_ce_loss(logits, centres, target, valid, w=None, soft_sigma: float = 0.0):
    """Cross-entropy of the per-pixel bin distribution, at the logits' own res.

    `soft_sigma > 0` replaces the one-hot target with a Gaussian over bin
    *indices*.  A hard target over 96 bins is nearly free to predict when most of
    the mass sits in one bin — which is how this head collapsed in v2 — whereas a
    smoothed target forces the distribution to have the right *shape* around the
    true height, not merely the right argmax.
    """
    b, k = logits.shape[:2]
    h, w_ = logits.shape[-2:]
    val = F.interpolate(valid.float(), size=(h, w_), mode="nearest") > 0.5
    if not val.any():
        return logits.sum() * 0.0
    idx = bin_target_index(centres, target, h, w_)
    if soft_sigma > 0:
        ar = torch.arange(k, device=logits.device, dtype=torch.float32)
        d = ar.view(1, k, 1, 1) - idx.unsqueeze(1).float()
        soft = torch.softmax(-(d ** 2) / (2.0 * float(soft_sigma) ** 2), dim=1)
        ce = -(soft * F.log_softmax(logits.float(), dim=1)).sum(1, keepdim=True)
    else:
        ce = F.cross_entropy(logits.float(), idx, reduction="none").unsqueeze(1)
    if w is not None:
        ce = ce * F.interpolate(w, size=(h, w_), mode="nearest")
    return _masked_mean(ce, val)


def bin_entropy_reg(logits, valid):
    """Negative entropy of the *marginal* bin distribution over valid pixels.

    Minimising it maximises how much of the bin range the head actually uses.
    Without it the cheapest solution is to put every pixel in the ground bin and
    let the gate copy Head A, which is the measured v2 failure.
    """
    h, w_ = logits.shape[-2:]
    val = F.interpolate(valid.float(), size=(h, w_), mode="nearest")
    p = F.softmax(logits.float(), dim=1)
    m = (p * val).sum(dim=(0, 2, 3)) / val.sum().clamp_min(1.0)
    m = m / m.sum().clamp_min(1e-8)
    return (m * torch.log(m.clamp_min(1e-8))).sum()          # == -entropy


def consistency_loss(student, teacher_pred, teacher_unc, conf_m: float = 1.5):
    """Student -> EMA-teacher agreement on unlabeled pixels (mean-teacher).

    Pixels whose teacher uncertainty exceeds `conf_m` metres are dropped, so the
    branch cannot pull the student towards the teacher's own hallucinations on
    out-of-domain imagery — which is the whole failure mode this is meant to fix.
    Returns (loss, kept_fraction).
    """
    keep = teacher_unc <= conf_m
    frac = float(keep.float().mean())
    if not keep.any():
        return student.sum() * 0.0, frac
    return _masked_mean((student - teacher_pred).abs(), keep), frac


def seg_ce_loss(seg_logits, cls):
    if cls is None:
        return seg_logits.sum() * 0.0
    tgt = cls.long()
    if (tgt != SEG_IGNORE_INDEX).sum() == 0:
        return seg_logits.sum() * 0.0
    return F.cross_entropy(seg_logits.float(), tgt, ignore_index=SEG_IGNORE_INDEX)


# ---------------------------------------------------------------------
def regression_terms(pred, target, valid, cfg, w, gsd):
    l1 = l1_loss(pred, target, valid, w)
    sil = silog_loss(pred, target, valid, cfg.silog_lambda, cfg.silog_shift)
    grad = gradient_loss(pred, target, valid)
    total = cfg.w_l1 * l1 + cfg.w_silog * sil + cfg.w_grad * grad
    return total, {"l1": float(l1.detach()), "silog": float(sil.detach()),
                   "grad": float(grad.detach())}


def compute_losses(out: dict, batch: dict, cfg, balancer: StratumBalancer):
    tgt, val = batch["target"], batch["valid"].bool()
    gsd = batch.get("gsd_m")
    w = balancer.weights(tgt, val)

    lf, sf = regression_terms(out["fused"], tgt, val, cfg, w, gsd)
    la, _ = regression_terms(out["a"], tgt, val, cfg, w, gsd)
    lb, _ = regression_terms(out["b"], tgt, val, cfg, w, gsd)

    l_norm = normal_loss(out["fused"], tgt, val, gsd)
    l_flat = flatness_loss(out["fused"], tgt, val, batch.get("cls"))
    l_bin = bin_ce_loss(out["b_logits"], out["b_centres"], tgt, val, w,
                        getattr(cfg, "bin_soft_sigma", 0.0))
    l_ent = bin_entropy_reg(out["b_logits"], val)
    l_seg = seg_ce_loss(out["seg"], batch.get("cls"))

    total = (cfg.w_fused * lf + cfg.w_head_a * la + cfg.w_head_b * lb
             + cfg.w_normal * l_norm + cfg.w_flat * l_flat
             + cfg.w_bin_ce * l_bin + getattr(cfg, "w_bin_entropy", 0.0) * l_ent
             + cfg.w_seg * l_seg)

    stats = {
        "loss": float(total.detach()),
        "l1": sf["l1"], "silog": sf["silog"], "grad": sf["grad"],
        "normal": float(l_norm.detach()), "flat": float(l_flat.detach()),
        "bin": float(l_bin.detach()), "bin_ent": float(-l_ent.detach()),
        "seg": float(l_seg.detach()),
        "alpha": float(out["alpha"].detach().mean()),
    }
    return total, stats
