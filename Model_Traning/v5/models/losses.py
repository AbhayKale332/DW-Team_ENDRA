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

from config import FLAT_CLASS_IDS, HEIGHT_STRATA_M, SEG_IGNORE_INDEX

_STRATA = torch.tensor(HEIGHT_STRATA_M[1:-1], dtype=torch.float32)  # inner edges


class StratumBalancer:
    """Running EMA of height-stratum frequencies -> per-pixel loss weights."""

    def __init__(self, beta: float = 0.5, clip: float = 5.0, momentum: float = 0.98):
        self.beta, self.clip, self.momentum = float(beta), float(clip), float(momentum)
        self.freq: torch.Tensor | None = None

    @torch.no_grad()
    def _update(self, idx: torch.Tensor, n_strata: int,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        # Weighted bincount rather than `idx[mask]`: the boolean gather has a
        # data-dependent output shape and syncs the device, and this runs on
        # every training step.
        w = None if mask is None else mask.reshape(-1).to(torch.float32)
        counts = torch.bincount(idx.reshape(-1), weights=w, minlength=n_strata).float()
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
        vf = valid.to(torch.float32)
        freq = self._update(idx, n, vf)
        w = (1.0 / freq.clamp_min(1e-4)) ** self.beta
        # Normalise by the *frequency-weighted* mean, not the plain mean over
        # strata: a stratum with no pixels in this batch has an enormous raw
        # weight, and dividing by the plain mean let those empty strata drag
        # every real weight below the clip floor — which flattened the weights
        # back to uniform and quietly turned the tail re-weighting off.
        w = (w / (freq * w).sum().clamp_min(1e-6)).clamp(1.0 / self.clip, self.clip)
        out = w.to(target.dtype)[idx]
        m = (out * vf.to(out.dtype)).sum() / vf.sum().clamp_min(1.0)
        return out / m.clamp_min(1e-6)


# ---------------------------------------------------------------------
def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Mean of `x` over the True entries of `m`, without a device sync.

    `x[m].mean()` is the obvious spelling and it is the wrong one here: boolean
    indexing has a data-dependent output shape, so it forces a device-to-host
    sync before the kernel can even be launched.  There are eight of these
    between the forward and the backward of one step, and every one of them
    stops the host running ahead of the card — which is most of what "the loader
    is starving the GPU" looked like in the v3 logs.

    `torch.where` is NaN-safe in the masked-out region (a plain `x * m` turns an
    inf there into a NaN that survives the sum) and its gradient is zero there,
    so this is numerically the same loss, one fused kernel, and no sync.
    """
    mf = m.to(x.dtype)
    if mf.shape != x.shape:
        mf = mf.expand_as(x)
    zero = torch.zeros((), dtype=x.dtype, device=x.device)
    return torch.where(mf > 0, x, zero).sum() / mf.sum().clamp_min(1.0)


def l1_loss(pred, target, valid, w=None):
    e = (pred - target).abs()
    if w is not None:
        e = e * w
    return _masked_mean(e, valid)


def silog_loss(pred, target, valid, lam: float, shift: float):
    """Scale-invariant log loss, masked without the three syncs `pred[valid]` costs.

    The variance is floored with `clamp_min` rather than nudged with `+ 1e-7`.
    `d/dx sqrt(x) = 1/(2 sqrt(x))`, so a batch where the prediction happens to
    track the target closely — a tile of flat ground, which is most of GAMUS —
    drives `x` towards zero and the gradient towards `1/(2 sqrt(1e-7)) ~ 1600`,
    *and keeps it there*, because the additive epsilon never stops the flow.
    `clamp_min` has exactly zero gradient below the floor, so the term stops
    contributing instead of contributing an enormous amount.  The floor is small
    enough (1e-8 -> a loss of 1e-4 at the optimum) that the reported value is
    strictly closer to zero than the old `+ 1e-7` spelling gave.
    """
    m = valid.to(pred.dtype)
    n = m.sum().clamp_min(1.0)
    g = torch.log(pred.clamp_min(0) + shift) - torch.log(target.clamp_min(0) + shift)
    g = torch.where(m > 0, g, torch.zeros((), dtype=g.dtype, device=g.device))
    mu = g.sum() / n
    var = (g ** 2).sum() / n - lam * (mu ** 2)
    return var.clamp_min(1e-8).sqrt()


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


def flatness_loss(pred, target, valid, cls=None, flat_ids=FLAT_CLASS_IDS,
                  tol_m: float = 0.35):
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
        # Unlabelled pixels count as eligible, and that is what makes this term
        # work at all on the GAMUS Kaggle mirror.  It ships no semantic raster,
        # so `cls` arrives entirely as SEG_IGNORE_INDEX — and intersecting with
        # the flat-class set then produced an EMPTY mask on every step of every
        # epoch.  The v4 Kaggle log prints `flat=0.000` from step 0 to the end
        # of the run: the planarity penalty, which README §3 calls the direct
        # counter to "texture becomes terrain" and which matters most on the
        # out-of-domain Indian imagery this model is for, was silently switched
        # off for the whole run while still carrying w_flat=0.2 in the config.
        #
        # Where there is no label the GT-Laplacian test above is the only
        # evidence available, and it is exactly the evidence this loss uses in
        # the `cls is None` branch — so an all-unlabelled dataset now behaves
        # identically to no dataset labels at all, which is plainly the intent.
        # A labelled pixel of a non-flat class is still excluded.
        sem = (c == SEG_IGNORE_INDEX)
        for i in flat_ids:
            sem |= (c == i)
        # Tighten to the flat classes unconditionally.  The old `if sem.any()`
        # guard was a per-step device sync bought to handle a batch with no
        # flat-class pixels at all — in which case `m & sem` is empty and
        # `_masked_mean` already returns a clean zero.
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
    # A detached 0-d tensor, not a float: `float(t)` here is a device sync in the
    # middle of the step, and the trainer only ever prints this every 25 steps.
    frac = keep.to(student.dtype).mean().detach()
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
    return total, {"l1": l1.detach(), "silog": sil.detach(), "grad": grad.detach()}


# ---------------------------------------------------------------------
# v5: coarse-label supervision
# ---------------------------------------------------------------------
def coarse_flags(batch: dict, cfg, device) -> torch.Tensor | None:
    """(B, 1, 1, 1) bool: which samples come from a coarse-label source.

    Built from `batch["src"]`, a host-side list of strings, so this costs no
    device sync.  None when the batch has no coarse sample (or the knob is off),
    which keeps every v4 code path byte-identical.
    """
    pref = tuple(x.strip() for x in str(getattr(cfg, "coarse_label_sources", "")).split(",")
                 if x.strip())
    src = batch.get("src")
    if not pref or src is None:
        return None
    if isinstance(src, str):
        src = [src]
    f = [str(x).startswith(pref) for x in src]
    if not any(f):
        return None
    return torch.tensor(f, dtype=torch.bool, device=device).view(-1, 1, 1, 1)


def coarse_pool_px(cfg, gsd_m: float) -> int:
    """Pool size, in pixels, that matches a coarse label at this GSD.

    `coarse_label_m > 0`: round(coarse_label_m / gsd_m), so the block is the
    label's own footprint whatever the crop was resampled to.  Otherwise the
    fixed `coarse_pool` (v5's first cut, and what `coarse_label_m 0` keeps).
    """
    m = float(getattr(cfg, "coarse_label_m", 0.0) or 0.0)
    if m > 0 and gsd_m and gsd_m > 0:
        return max(1, int(round(m / float(gsd_m))))
    return int(max(1, getattr(cfg, "coarse_pool", 4)))


# A block with a vegetation-masked part is still scored on the rest of it, as
# long as the rest is at least this share of the block.
COARSE_KEEP_MIN = 0.5


def coarse_terms(pred, target, valid, cfg, w=None, gsd=None, k: int | None = None,
                 keep=None):
    """`regression_terms` after k x k average pooling (default k = cfg.coarse_pool).

    A DFC23 / India label carries ~2 m of information per ~4 px; comparing the
    prediction's 4x4 block means to the label's leaves the network free to put
    real edges inside the block.  A pooled pixel counts only when every source
    pixel under it was valid, so NoData edges cannot drag a block mean down.

    `keep` (the vegetation mask's complement) is not NoData: a block with masked
    pixels is compared on the means of its kept pixels, prediction and label
    alike, and dropped only when under `COARSE_KEEP_MIN` of it is kept.  Folding
    it into `valid` made one flagged pixel void its whole block, so the ground
    and roof edges around every tree left the loss with the tree.
    """
    k = int(max(1, k if k is not None else getattr(cfg, "coarse_pool", 4)))
    vf = valid.to(pred.dtype)
    full = F.avg_pool2d(vf, k) > 0.999
    if keep is not None:
        vf = vf * keep.to(pred.dtype)
    n = F.avg_pool2d(vf, k)
    if keep is not None:
        full = full & (n >= COARSE_KEEP_MIN)
    pp = F.avg_pool2d(pred * vf, k) / n.clamp_min(1e-6)
    tt = F.avg_pool2d(target * vf, k) / n.clamp_min(1e-6)
    wp = F.avg_pool2d(w, k) if w is not None else None
    return regression_terms(pp, tt, full, cfg, wp, gsd)


# `alpha` and `b_std` join the original set for the same reason the rest are in
# it: `alpha` is reported every 25 steps and is the only published trace of the
# gate, and `b_std` is what `consistency_loss` thresholds the mean-teacher's
# pseudo-labels with, at metre precision.  Neither should be read in fp16.
_FP32_KEYS = ("fused", "a", "b", "seg", "b_logits", "b_centres",
              "alpha", "b_std")

# Probed whenever a forward goes non-finite; these names are what `train.py`
# prints.  Listed earliest-in-the-graph first, so the leftmost name in the report
# is the tensor that went bad on its own rather than by inheritance.
NONFINITE_PROBE_KEYS = ("b_centres", "b_logits", "a", "b", "fused")


def compute_losses(out: dict, batch: dict, cfg, balancer: StratumBalancer):
    # ---- every loss term is computed in fp32, unconditionally --------------
    # `compute_losses` is already called outside the autocast context, but that
    # only stops *new* ops from being autocast — the head outputs arrive as fp16
    # and fp16 inputs keep the arithmetic in fp16.  Two terms then overflow:
    #
    #   * `normal_loss` divides height differences by the GSD, so a 200 m
    #     building edge at 0.33 m/px becomes ~600, and the `n.norm()` that
    #     follows squares it — 3.6e5 against an fp16 ceiling of 65504.
    #   * `silog_loss` takes a sqrt whose gradient diverges near zero.
    #
    # The v4 Kaggle run's log shows exactly this: `nrm=nan` in the forward from
    # epoch 6 and inf gradients on 100 % of the optimiser steps in epochs 7-12.
    # Casting here costs ~10 MB of fp32 activations on a (2, 1, 512, 512) batch
    # and removes the whole failure mode; the ViT-L forward, which is where the
    # step time actually goes, is untouched and still runs in fp16.
    out = {k: (v.float() if (k in _FP32_KEYS and torch.is_tensor(v)
                             and v.dtype.is_floating_point) else v)
           for k, v in out.items()}

    # ---- which tensor went non-finite, named ------------------------------
    # The second v4 Kaggle run spent epochs 4 and 5 printing `l1=nan ... bin=3.77`
    # with no way to tell from the log which head produced the NaN; working it
    # out afterwards meant reasoning backwards from which loss terms survived.
    # These are `any()` reductions over tensors the step already has resident,
    # they stay on the device, and `train.py` only reads them on the steps it was
    # already going to print — so the answer is in the log for free.
    tgt_dev = batch["target"].device
    nf = torch.stack([(~torch.isfinite(out[k])).any() if k in out
                      else torch.zeros((), dtype=torch.bool, device=tgt_dev)
                      for k in NONFINITE_PROBE_KEYS])

    tgt, val = batch["target"].float(), batch["valid"].bool()
    # A single non-finite pixel in the packed target poisons the mean, the
    # backward and — through the optimiser state — everything after it.  One
    # cheap elementwise op buys immunity, and `_masked_mean`'s `torch.where`
    # already guarantees the excluded region cannot leak back in.
    val = val & torch.isfinite(tgt)
    tgt = torch.nan_to_num(tgt, 0.0, 0.0, 0.0)
    gsd = batch.get("gsd_m")
    if gsd is not None:
        gsd = gsd.float()
    w = balancer.weights(tgt, val)

    # v5: coarse-label samples (DFC23 / India) leave the pixel-level terms and
    # are scored on pooled blocks instead.  `val_f` is what every edge-sensitive
    # term sees.  The two regression parts are mixed by their share of valid
    # pixels, so a batch's total weight on regression is what v4 gave it.
    coarse = coarse_flags(batch, cfg, tgt.device)
    val_f = val if coarse is None else (val & ~coarse)
    lf, sf = regression_terms(out["fused"], tgt, val_f, cfg, w, gsd)
    la, _ = regression_terms(out["a"], tgt, val_f, cfg, w, gsd)
    lb, _ = regression_terms(out["b"], tgt, val_f, cfg, w, gsd)
    l_reg = cfg.w_fused * lf + cfg.w_head_a * la + cfg.w_head_b * lb
    l_coarse = tgt.new_zeros(())
    veg_drop = tgt.new_zeros(())
    if coarse is not None:
        val_c = val & coarse
        keep = None
        veg = batch.get("veg")
        if getattr(cfg, "coarse_mask_veg", False) and veg is not None:
            keep = ~(veg.bool() & (tgt < 1.0))
            # Share of the coarse samples' valid pixels the mask takes out,
            # logged as `veg=`.  resume-v2 ran tau = -0.05 with no way to see
            # how much of DFC23 / India that took.
            veg_drop = 1.0 - (val_c & keep).sum().float() / val_c.sum().float().clamp_min(1.0)
        # One pool size per sample (see `coarse_pool_px`): samples are grouped
        # by k and each group's loss is weighted by its valid pixels.  Reading
        # the per-sample GSDs back is one small device sync, paid only on
        # batches that carry a coarse sample with `coarse_label_m` set.
        groups = {None: coarse}
        if float(getattr(cfg, "coarse_label_m", 0.0) or 0.0) > 0 and gsd is not None:
            gl = gsd.reshape(-1).tolist()
            cl = coarse.reshape(-1).tolist()
            ks: dict = {}
            for i, (g, c) in enumerate(zip(gl, cl)):
                if c:
                    ks.setdefault(coarse_pool_px(cfg, g), []).append(i)
            groups = {}
            for kk, idx in ks.items():
                m = torch.zeros(len(gl), dtype=torch.bool)
                m[idx] = True
                groups[kk] = m.to(tgt.device).view(-1, 1, 1, 1)
        l_coarse = tgt.new_zeros(())
        n_tot = tgt.new_zeros(())
        kept = val_c if keep is None else val_c & keep
        for kk, sel in groups.items():
            vk = val_c & sel
            nk = (kept & sel).sum().float()
            cf, _ = coarse_terms(out["fused"], tgt, vk, cfg, w, gsd, k=kk, keep=keep)
            ca, _ = coarse_terms(out["a"], tgt, vk, cfg, w, gsd, k=kk, keep=keep)
            cb, _ = coarse_terms(out["b"], tgt, vk, cfg, w, gsd, k=kk, keep=keep)
            l_coarse = l_coarse + nk * (cfg.w_fused * cf + cfg.w_head_a * ca
                                        + cfg.w_head_b * cb)
            n_tot = n_tot + nk
        l_coarse = l_coarse / n_tot.clamp_min(1.0)
        n_f = val_f.sum().float()
        n_c = kept.sum().float()
        frac_c = n_c / (n_f + n_c).clamp_min(1.0)
        l_reg = (1.0 - frac_c) * l_reg + frac_c * getattr(cfg, "w_coarse", 1.0) * l_coarse

    l_norm = normal_loss(out["fused"], tgt, val_f, gsd)
    l_flat = flatness_loss(out["fused"], tgt, val_f, batch.get("cls"))
    l_bin = bin_ce_loss(out["b_logits"], out["b_centres"], tgt, val_f, w,
                        getattr(cfg, "bin_soft_sigma", 0.0))
    l_ent = bin_entropy_reg(out["b_logits"], val_f)
    l_seg = seg_ce_loss(out["seg"], batch.get("cls"))

    total = (l_reg
             + cfg.w_normal * l_norm + cfg.w_flat * l_flat
             + cfg.w_bin_ce * l_bin + getattr(cfg, "w_bin_entropy", 0.0) * l_ent
             + cfg.w_seg * l_seg)

    # Detached 0-d *tensors*, not floats: `float(t)` on a CUDA tensor is a full
    # device sync, and there were ten of them between the forward and the
    # backward of every step — enough to stop the CPU ever running ahead of the
    # GPU, which is most of what "low utilisation" looked like.  The trainer
    # materialises these only on the steps it actually prints.
    stats = {
        "loss": total.detach(),
        "l1": sf["l1"], "silog": sf["silog"], "grad": sf["grad"],
        "normal": l_norm.detach(), "flat": l_flat.detach(),
        "bin": l_bin.detach(), "bin_ent": -l_ent.detach(),
        "seg": l_seg.detach(),
        "coarse": l_coarse.detach(),
        "veg_drop": veg_drop.detach(),
        "alpha": out["alpha"].detach().mean(),
        "nonfinite": nf,
    }
    return total, stats
