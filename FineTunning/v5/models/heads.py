"""Heads + gated fusion + `DepthWizardNet`.

  Head A  metric nDSM regression   — softplus, not ReLU.  v2's `F.relu` on the raw
          conv output has exactly zero gradient wherever the head predicts a
          negative number, so a unit that drifts negative early stops learning.
  Head B  adaptive-bin classifier  — AdaBins-lite; owns the long tail.
  Head C  semantic segmentation    — auxiliary regulariser + the ground mask that
          `infer/calibrate.py` needs to tie an rDSM to a DEM.
  Fusion  per-pixel alpha over (A, B).

All heads run at H/2 and the final maps are upsampled once, at the end.

v4 adds one output: **`b_std`**, the standard deviation of Head B's bin
distribution, in metres.  It is a free per-pixel uncertainty — the bin head
already computes the distribution, so the second moment costs one extra
reduction — and the mean-teacher branch uses it to decide which of the teacher's
pseudo-labels on unlabeled Indian imagery are worth believing.  That also gives
Head B a second job, which is another reason for it not to collapse.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn

from .dpt import DPTTrunk
from .encoder import DINOv3Encoder


def _conv_block(dim: int, out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(dim, dim // 2, 3, padding=1, bias=False),
        nn.GroupNorm(8, dim // 2), nn.ReLU(True),
        nn.Conv2d(dim // 2, out, 1),
    )


class HeadA(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = _conv_block(dim, 1)
        # start near zero height: most pixels are ground
        nn.init.constant_(self.net[-1].bias, -2.0)

    def forward(self, x):
        return F.softplus(self.net(x))


class HeadB(nn.Module):
    """Global adaptive bin widths + per-pixel bin logits.

    `bin_centres` is the one path in this network with **no normalisation
    anywhere**: a global average pool straight into an unnormalised `nn.Linear`,
    whose output is then softmaxed.  That is what ended the second v4 Kaggle run,
    and the log says so precisely.  From e4 s1000 on, every line reads

        loss=nan (l1=nan sil=nan nrm=nan flat=nan bin=3.77 ent=2.97 seg=0.00 a=nan)

    and that combination has exactly one source.  `bin` and `ent` are computed
    from `b_logits`, which comes off `pixel_logits` — a `_conv_block`, so its
    first conv is followed by a GroupNorm and it stayed finite.  `bin` also reads
    `centres`, but only through `searchsorted` -> `clamp(0, K-1)`, which launders
    a NaN edge into a valid integer index.  Everything that went NaN — `l1`,
    `silog`, `normal`, `flat` and the reported `alpha` — reads `fused`, and
    `fused` reads `b["height"] = einsum(centres, probs)`.  So `centres` was NaN
    while `logits` was finite: the failure is in this method and nowhere else.

    Three things are fixed here, none of which change any parameter name, so
    existing checkpoints still load:

    1. **fp32.**  Under `torch.autocast` the `Linear` ran in fp16 on a pooled
       feature the trunk never bounds (`DPTTrunk` ends on a bare 1x1 conv).  One
       overflow to `inf` makes `softmax` return NaN for the whole row, and from
       there it is in `centres`, `fused`, and the gradient of every regression
       term.
    2. **RMS-normalised input.**  The global pool concentrates the gradient of
       every pixel in the tile onto this one small `Linear`, which is why it is
       the first thing to run away when the encoder unfreezes and the pooled
       feature scale shifts under it.  e3 is the log of that happening: `bin`
       leaves its 2.2-2.6 band at s1400 and thrashes between 2.6 and 6.74 —
       *above* `ln(96) = 4.56`, i.e. the head doing worse than a uniform
       prediction — for the rest of the epoch, because the bin edges themselves
       were moving faster than the pixel logits could track them.
    3. **A width floor.**  `torch.searchsorted` requires strictly increasing
       boundaries.  A saturated softmax gives bins whose width underflows to
       zero, so the edges are merely non-decreasing and the target index it
       returns is not defined.  Mixing in a little uniform mass keeps every bin
       open without meaningfully constraining the shape.

    The second moment is computed in fp32 for a separate reason: `std` is
    `sqrt(c2 - height**2)` and at the top of a 120 m range those two terms are
    both ~1.4e4, where fp16 has a resolution of 8.  The cancellation left `b_std`
    with ~3 m of pure quantisation noise, and `consistency_loss` gates the
    mean-teacher's pseudo-labels at `conf_m = 1.5` m — so the gate was being
    driven by rounding error rather than by the head's actual uncertainty.
    """

    # Softmax over 96 bins saturates long before this; the clamp only exists so
    # a diverging `Linear` cannot reach `inf` and take `softmax` with it.
    WIDTH_LOGIT_CLAMP = 15.0
    # Uniform mass mixed into the width distribution.  Each bin ends up at least
    # `WIDTH_FLOOR / K` of the range wide, so the edges are strictly increasing.
    WIDTH_FLOOR = 0.05

    def __init__(self, dim: int, n_bins: int, hmin: float, hmax: float):
        super().__init__()
        self.n_bins, self.hmin, self.hmax = n_bins, hmin, hmax
        self.bin_widths = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(dim, n_bins))
        self.pixel_logits = _conv_block(dim, n_bins)

    def bin_centres(self, x) -> torch.Tensor:
        # The pool is spelled out rather than run through `bin_widths[:2]` so it
        # can accumulate in fp32 *without* an `x.float()` first — that cast would
        # materialise a second copy of the whole (B, dim, H/2, W/2) trunk output,
        # ~134 MB per micro-batch at 512 px, and hold it live for the backward on
        # a profile whose own log reports 13-14 of 15 GiB in use.  `mean` over the
        # spatial dims with `dtype=` is exactly `AdaptiveAvgPool2d(1)` followed by
        # `Flatten`, at fp32 precision and fp16 footprint.  The two layers stay in
        # the Sequential anyway: they hold no parameters, so DDP does not see
        # them, and keeping them there is what keeps the trained `Linear` at
        # `bin_widths.2.*` so existing checkpoints load unchanged.
        with torch.autocast(x.device.type, enabled=False):
            g = x.mean(dim=(-2, -1), dtype=torch.float32)           # (B, C)
            g = g * torch.rsqrt(g.pow(2).mean(dim=1, keepdim=True) + 1e-6)
            logits = self.bin_widths[2](g).clamp(-self.WIDTH_LOGIT_CLAMP,
                                                 self.WIDTH_LOGIT_CLAMP)
            p = F.softmax(logits, dim=1)
            p = (p + self.WIDTH_FLOOR / self.n_bins) / (1.0 + self.WIDTH_FLOOR)
            w = p * (self.hmax - self.hmin)
            edges = F.pad(self.hmin + torch.cumsum(w, dim=1), (1, 0), value=self.hmin)
            return 0.5 * (edges[:, 1:] + edges[:, :-1])             # (B, K)

    def forward(self, x):
        centres = self.bin_centres(x)                               # fp32
        logits = self.pixel_logits(x)                               # (B, K, h, w)
        # Both moments in fp32: `einsum` is on autocast's fp16 list, and `c2`
        # reaches `hmax ** 2` = 1.4e4 where fp16 cannot resolve the difference
        # that `std` is made of.
        with torch.autocast(x.device.type, enabled=False):
            probs = F.softmax(logits.float(), dim=1)
            height = torch.einsum("bk,bkhw->bhw", centres, probs).unsqueeze(1)
            c2 = torch.einsum("bk,bkhw->bhw", centres ** 2, probs).unsqueeze(1)
            std = (c2 - height ** 2).clamp_min(0.0).sqrt()
        return {"height": height, "logits": logits, "centres": centres, "std": std}


class GatedFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(dim + 2, dim // 4, 3, padding=1, bias=False),
            nn.GroupNorm(8, dim // 4), nn.ReLU(True),
            nn.Conv2d(dim // 4, 1, 1),
        )

    def forward(self, feat, a, b, scale=None):
        # `b` arrives in fp32 (HeadB computes its moments there — see HeadB's
        # docstring) while `feat` and `a` are autocast fp16.  `torch.cat` type-
        # promotes, so concatenating them as-is would silently widen all `dim`
        # channels of `feat` to fp32 — ~70 MB per micro-batch at 512 px, on a
        # profile the log shows sitting at 13-14 of 15 GiB.  The gate is a conv
        # stack autocast would run in fp16 regardless, so the cast is free to
        # make explicit and narrow.
        #
        # `scale` is the per-sample factor `feat` was already multiplied by (see
        # `DepthWizardNet.forward`).  The gate's first conv has no bias and feeds
        # a GroupNorm, so scaling its whole input leaves alpha unchanged, but
        # only if `a` and `b` go in scaled along with `feat`.
        ga, gb = (a, b) if scale is None else (a * scale, b * scale)
        g = torch.cat([feat, ga.to(feat.dtype), gb.to(feat.dtype)], dim=1)
        alpha = torch.sigmoid(self.gate(g))
        # The blend itself stays in the wider of the two: this is the metric
        # height that every regression loss is computed against.
        alpha = alpha.to(b.dtype)
        return alpha * a.to(b.dtype) + (1.0 - alpha) * b, alpha


# ---------------------------------------------------------------------
# v5: the full-resolution detail path
# ---------------------------------------------------------------------
class DetailStem(nn.Module):
    """A shallow conv stem on the RGB itself, at full and half resolution.

    The ViT sees 16 px patches and the DPT trunk stops at H/2, so nothing in v4
    ever looked at the image above 1/4 resolution with more than one conv —
    roof edges, tree crowns and walls were reconstructed from patch tokens and
    then bilinearly upsampled.  This gives the heads (at H/2) and the upsampler
    (to H) direct access to image edges.
    """

    def __init__(self, c1: int = 32, c2: int = 64):
        super().__init__()
        # not `full` / `half`: `nn.Module.half` is the fp16 cast
        self.s1 = nn.Sequential(nn.Conv2d(3, c1, 3, padding=1, bias=False),
                                nn.GroupNorm(8, c1), nn.ReLU(True))
        self.s2 = nn.Sequential(nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False),
                                nn.GroupNorm(8, c2), nn.ReLU(True))

    def forward(self, image):
        f = self.s1(image)
        return f, self.s2(f)


def _bilinear_up2_logits() -> torch.Tensor:
    """(36,) logits that make `ConvexUp2x` reproduce bilinear 2x upsampling.

    Channel c = k*4 + di*2 + dj, neighbour k = dy*3 + dx at offset (dy-1, dx-1),
    output sub-pixel (di, dj).  With align_corners=False, sub-pixel 0 sits a
    quarter pixel before the source centre: weights 0.25 / 0.75 on rows i-1 / i;
    sub-pixel 1 uses rows i / i+1 at 0.75 / 0.25.
    """
    w1 = {0: {-1: 0.25, 0: 0.75}, 1: {0: 0.75, 1: 0.25}}
    out = torch.full((9, 2, 2), 1e-4)
    for dy in range(3):
        for dx in range(3):
            for di in range(2):
                for dj in range(2):
                    wy = w1[di].get(dy - 1, 0.0)
                    wx = w1[dj].get(dx - 1, 0.0)
                    if wy * wx > 0:
                        out[dy * 3 + dx, di, dj] = wy * wx
    return out.log().reshape(36)


class ConvexUp2x(nn.Module):
    """RAFT-style learned upsampling, H/2 -> H: every output pixel is a convex
    combination of its 3x3 low-res neighbourhood, weights predicted from the
    trunk features and the full-resolution stem.  Where the image has an edge
    the weights can snap to one side of it instead of averaging across — which
    is what a roof edge or a wall needs and what bilinear cannot do.

    The last conv is zero-weighted with the bilinear logits as its bias, so at
    initialisation this *is* bilinear upsampling (to ~1e-4) and a v4 checkpoint
    loaded into a v5 net predicts what it predicted before.  The neighbourhood
    is gathered with pad + 9 slices rather than `F.unfold`, which keeps the ONNX
    graph to Pad / Slice / Softmax / DepthToSpace-free reshapes.
    """

    def __init__(self, cin: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, hidden, 3, padding=1), nn.ReLU(True),
                                 nn.Conv2d(hidden, 36, 1))
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():
            self.net[-1].bias.copy_(_bilinear_up2_logits())

    def weights(self, feat):
        b, _, h, w = feat.shape
        return self.net(feat).float().view(b, 1, 9, 2, 2, h, w).softmax(2)

    @staticmethod
    def apply(t, m):
        """t (B, C, h, w), m from `weights` -> (B, C, 2h, 2w)."""
        b, c, h, w = t.shape
        tp = F.pad(t.float(), (1, 1, 1, 1), mode="replicate")
        nb = torch.stack([tp[..., dy:dy + h, dx:dx + w]
                          for dy in range(3) for dx in range(3)], 2)   # (B, C, 9, h, w)
        out = (m * nb.view(b, c, 9, 1, 1, h, w)).sum(2)                 # (B, C, 2, 2, h, w)
        return out.permute(0, 1, 4, 2, 5, 3).reshape(b, c, 2 * h, 2 * w)


class DepthWizardNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from config import N_SEG_CLASSES

        self.cfg = cfg
        self.encoder = DINOv3Encoder(cfg)
        d = cfg.decoder_dim
        self.trunk = DPTTrunk(self.encoder.hidden, d, cfg.grad_checkpoint_decoder)
        self.head_a = HeadA(d)
        self.head_b = HeadB(d, cfg.n_bins, cfg.bin_min_m, cfg.bin_max_m)
        self.head_c = _conv_block(d, N_SEG_CLASSES)
        self.fusion = GatedFusion(d)
        self.detail = bool(getattr(cfg, "detail_branch", False))
        if self.detail:
            c2 = int(getattr(cfg, "detail_dim", 64))
            self.stem = DetailStem(max(8, c2 // 2), c2)
            # zero-init: at t=0 the trunk features pass through unchanged
            self.inject = nn.Conv2d(d + c2, d, 1)
            nn.init.zeros_(self.inject.weight)
            nn.init.zeros_(self.inject.bias)
            self.upsampler = ConvexUp2x(d + c2 + 4 * max(8, c2 // 2))

    # Largest |feature| handed to the fp16 heads.  Their first conv sums
    # 9 x 256 products, so this keeps its output far inside fp16's 65504.
    FP16_FEAT_MAX = 64.0

    def forward(self, image: torch.Tensor) -> dict:
        """Under fp16 autocast, the parts with no normalisation run in fp32.
        bf16, fp32 and ONNX export take the same path as before.

        The DPT trunk is a residual stream with no normalisation, and neither
        the detail `inject` nor the upsampler's first conv (bias, then ReLU) has
        any either.  v4 was trained in bf16 on the H100, and its stream grows
        past fp16's 65504 on some tiles.  The first T4 warm start shows what
        follows: `inf` in the trunk output (NON-FINITE b_centres), NaN on
        dfc23 val, and 54 % then 97 % of optimiser steps skipped, because one
        bad micro-batch spoils all 16 in its step.

        The heads, the fusion gate and HeadB's bin-width pool each start with
        either a bias-free conv followed by GroupNorm or an RMS normalisation.
        That makes them invariant to a positive per-sample scale on their input.
        So they stay in fp16 and get a copy of the features scaled down to
        FP16_FEAT_MAX.  The result is exact apart from GroupNorm's eps.
        """
        hw = image.shape[-2:]
        dev = image.device.type
        fp16 = (torch.is_autocast_enabled(dev)
                and torch.get_autocast_dtype(dev) == torch.float16)
        fp32 = (lambda: torch.autocast(dev, enabled=False)) if fp16 else nullcontext
        feats = self.encoder(image)
        with fp32():
            x = self.trunk([f.float() for f in feats] if fp16 else feats)  # (B, d, H/2, W/2)
        up = lambda t: F.interpolate(t, size=hw, mode="bilinear", align_corners=False)  # noqa: E731
        up_h = up
        if self.detail:
            s_full, s_half = self.stem(image)
            if s_half.shape[-2:] != x.shape[-2:]:
                s_half = F.interpolate(s_half, size=x.shape[-2:], mode="bilinear",
                                       align_corners=False)
            with fp32():
                x = x + self.inject(torch.cat([x, s_half.to(x.dtype)], 1))
                if tuple(hw) == (2 * x.shape[-2], 2 * x.shape[-1]):
                    m = self.upsampler.weights(torch.cat(
                        [x, s_half.to(x.dtype), F.pixel_unshuffle(s_full, 2).to(x.dtype)], 1))
                    up_h = lambda t: ConvexUp2x.apply(t, m)  # noqa: E731
        scale = None
        if fp16:
            # Detached, because the heads are blind to it: no gradient flows through it.
            peak = x.detach().abs().amax(dim=(1, 2, 3), keepdim=True)
            scale = (self.FP16_FEAT_MAX / peak.clamp_min(1e-12)).clamp_max(1.0)
            x = (x * scale).to(torch.float16)
        a = self.head_a(x)
        b = self.head_b(x)
        fused, alpha = self.fusion(x, a, b["height"], scale)

        return {
            "a": up_h(a),
            "b": up_h(b["height"]),
            "fused": up_h(fused),
            "alpha": up(alpha),
            "b_std": up(b["std"]),
            "seg": up(self.head_c(x)),
            # kept at half res — the bin loss is computed against a downsampled
            # target rather than blowing a (B, K, H, W) tensor up to full size.
            "b_logits": b["logits"],
            "b_centres": b["centres"],
        }

    def decoder_parameters(self):
        for n, p in self.named_parameters():
            if p.requires_grad and not n.startswith("encoder."):
                yield p

    def head_state_dict(self) -> dict:
        """Everything except the public frozen encoder backbone.

        Encoder weights are only included once they have actually been trained,
        which `train.py` signals with `include_encoder=True`.
        """
        return {k: v for k, v in self.state_dict().items()
                if not k.startswith("encoder.model.")}

    def full_state_dict(self) -> dict:
        return self.state_dict()


# v3 checkpoints and any external code that imported the old name keep working.
DepthWizardNetV3 = DepthWizardNet
