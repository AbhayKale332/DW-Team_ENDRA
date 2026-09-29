"""Heads + gated fusion + `DepthWizardNetV3`.

  Head A  metric nDSM regression   — softplus, not ReLU.  v2's `F.relu` on the raw
          conv output has exactly zero gradient wherever the head predicts a
          negative number, so a unit that drifts negative early stops learning.
  Head B  adaptive-bin classifier  — AdaBins-lite; owns the long tail.
  Head C  semantic segmentation    — auxiliary regulariser + the ground mask that
          `infer/calibrate.py` needs to tie an rDSM to a DEM.
  Fusion  per-pixel alpha over (A, B).

All heads run at H/2 and the final maps are upsampled once, at the end.
"""

from __future__ import annotations

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
    """Global adaptive bin widths + per-pixel bin logits."""

    def __init__(self, dim: int, n_bins: int, hmin: float, hmax: float):
        super().__init__()
        self.n_bins, self.hmin, self.hmax = n_bins, hmin, hmax
        self.bin_widths = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(dim, n_bins))
        self.pixel_logits = _conv_block(dim, n_bins)

    def bin_centres(self, x) -> torch.Tensor:
        w = F.softmax(self.bin_widths(x), dim=1) * (self.hmax - self.hmin)
        edges = F.pad(self.hmin + torch.cumsum(w, dim=1), (1, 0), value=self.hmin)
        return 0.5 * (edges[:, 1:] + edges[:, :-1])            # (B, K)

    def forward(self, x):
        centres = self.bin_centres(x)
        logits = self.pixel_logits(x)                          # (B, K, h, w)
        probs = F.softmax(logits, dim=1)
        height = torch.einsum("bk,bkhw->bhw", centres, probs).unsqueeze(1)
        return {"height": height, "logits": logits, "centres": centres}


class GatedFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(dim + 2, dim // 4, 3, padding=1, bias=False),
            nn.GroupNorm(8, dim // 4), nn.ReLU(True),
            nn.Conv2d(dim // 4, 1, 1),
        )

    def forward(self, feat, a, b):
        alpha = torch.sigmoid(self.gate(torch.cat([feat, a, b], dim=1)))
        return alpha * a + (1.0 - alpha) * b, alpha


class DepthWizardNetV3(nn.Module):
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

    def forward(self, image: torch.Tensor) -> dict:
        hw = image.shape[-2:]
        x = self.trunk(self.encoder(image))          # (B, d, H/2, W/2)
        a = self.head_a(x)
        b = self.head_b(x)
        fused, alpha = self.fusion(x, a, b["height"])

        up = lambda t: F.interpolate(t, size=hw, mode="bilinear", align_corners=False)  # noqa: E731
        return {
            "a": up(a),
            "b": up(b["height"]),
            "fused": up(fused),
            "alpha": up(alpha),
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
