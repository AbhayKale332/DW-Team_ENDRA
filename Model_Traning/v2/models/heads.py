"""Prediction heads + gated fusion + the full DepthWizardNetV2.

  Head A  metric nDSM regression        (ReLU, metres)
  Head B  adaptive-bin classifier       (AdaBins-style: per-pixel softmax over
          learned bin widths -> height = sum p * bin_centre)  -> fixes the long tail
  Head C  6(+1)-class semantic seg       (free GAMUS supervision; ground mask later)
  GatedFusion  per-pixel alpha in [0,1] -> nDSM = alpha*A + (1-alpha)*B

`forward()` returns a dict so the trainer / TTA / eval can pick what they need.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .dpt import DPTTrunk
from .encoder import DINOv3Encoder


class HeadA(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(dim // 2, 32, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x):
        return F.relu(self.net(x))


class HeadB(nn.Module):
    """AdaBins-lite: global adaptive bin widths + per-pixel bin logits."""

    def __init__(self, dim: int, n_bins: int, hmin: float, hmax: float):
        super().__init__()
        self.n_bins = n_bins
        self.hmin, self.hmax = hmin, hmax
        self.bin_widths = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(dim, n_bins),
        )
        self.pixel_logits = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(dim // 2, n_bins, 1),
        )

    def bin_centres(self, x) -> torch.Tensor:
        w = F.softmax(self.bin_widths(x), dim=1) * (self.hmax - self.hmin)  # (B, n_bins)
        edges = self.hmin + torch.cumsum(w, dim=1)
        edges = F.pad(edges, (1, 0), value=self.hmin)
        return 0.5 * (edges[:, 1:] + edges[:, :-1])                        # (B, n_bins)

    def forward(self, x):
        centres = self.bin_centres(x)                                     # (B, K)
        probs = F.softmax(self.pixel_logits(x), dim=1)                     # (B, K, H, W)
        height = torch.einsum("bk,bkhw->bhw", centres, probs).unsqueeze(1)
        return {"height": F.relu(height), "probs": probs, "centres": centres}


class HeadC(nn.Module):
    def __init__(self, dim: int, n_classes: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(dim // 2, n_classes, 1),
        )

    def forward(self, x):
        return self.net(x)


class GatedFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(dim + 2, dim // 4, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(dim // 4, 1, 1),
        )

    def forward(self, feat, a, b):
        alpha = torch.sigmoid(self.gate(torch.cat([feat, a, b], dim=1)))
        return alpha * a + (1.0 - alpha) * b, alpha


class DepthWizardNetV2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = DINOv3Encoder(cfg)
        d = cfg.decoder_dim
        self.trunk = DPTTrunk(self.encoder.hidden, d, cfg.grad_checkpoint)
        self.head_a = HeadA(d)
        self.head_b = HeadB(d, cfg.n_bins, cfg.bin_min_m, cfg.bin_max_m)
        self.head_c = HeadC(d, len(cfg.class_names))
        self.fusion = GatedFusion(d)

    def forward(self, image: torch.Tensor) -> dict:
        feats = self.encoder(image)
        x = self.trunk(feats, image.shape[-2:])
        a = self.head_a(x)
        b = self.head_b(x)
        fused, alpha = self.fusion(x, a, b["height"])
        return {
            "a": a,
            "b": b["height"],
            "b_probs": b["probs"],
            "b_centres": b["centres"],
            "seg": self.head_c(x),
            "fused": fused,
            "alpha": alpha,
        }

    # decoder + heads only (encoder is frozen & public unless partially unfrozen)
    def trainable_state_dict(self) -> dict:
        sd = self.state_dict()
        keep = {}
        for k, v in sd.items():
            if not k.startswith("encoder.model.") or self.encoder.unfrozen_block_ids and any(
                f".{b}." in k for b in self.encoder.unfrozen_block_ids
            ):
                keep[k] = v
        return keep
