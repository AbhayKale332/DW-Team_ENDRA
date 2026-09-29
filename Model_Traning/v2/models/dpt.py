"""Shared DPT decoder trunk.

Same reassemble + RefineNet-fusion structure as v1's `DPTDecoder`, but the final
1x1 regression conv is removed — `forward()` returns the fused feature map
`(B, dim, H, W)` that all three heads consume.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


class ResidualConvUnit(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(F.relu(x))))


class FeatureFusionBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.rcu1 = ResidualConvUnit(c)
        self.rcu2 = ResidualConvUnit(c)
        self.out = nn.Conv2d(c, c, 1)

    def forward(self, x, skip=None):
        if skip is not None:
            x = x + self.rcu1(skip)
        x = self.rcu2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.out(x)


class DPTTrunk(nn.Module):
    def __init__(self, in_ch: int, dim: int, grad_checkpoint: bool = False):
        super().__init__()
        self.dim = dim
        self.grad_checkpoint = grad_checkpoint
        proj = [96, 192, 384, 768]
        self.proj = nn.ModuleList(nn.Conv2d(in_ch, p, 1) for p in proj)
        self.resample = nn.ModuleList([
            nn.ConvTranspose2d(proj[0], proj[0], 4, stride=4),
            nn.ConvTranspose2d(proj[1], proj[1], 2, stride=2),
            nn.Identity(),
            nn.Conv2d(proj[3], proj[3], 3, stride=2, padding=1),
        ])
        self.to_dim = nn.ModuleList(nn.Conv2d(p, dim, 3, padding=1, bias=False) for p in proj)
        self.fuse = nn.ModuleList(FeatureFusionBlock(dim) for _ in range(4))

    def forward(self, feats: list[torch.Tensor], out_hw) -> torch.Tensor:
        f = [self.resample[i](self.proj[i](feats[i])) for i in range(4)]
        f = [self.to_dim[i](f[i]) for i in range(4)]

        def run(fn, *a):
            if self.grad_checkpoint and self.training:
                return checkpoint(fn, *a, use_reentrant=False)
            return fn(*a)

        x = run(self.fuse[3], f[3])
        x = run(self.fuse[2], x, f[2])
        x = run(self.fuse[1], x, f[1])
        x = run(self.fuse[0], x, f[0])
        return F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
