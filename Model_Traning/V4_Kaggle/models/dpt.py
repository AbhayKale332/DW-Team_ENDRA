"""Shared DPT trunk.

Same reassemble + RefineNet fusion as v1/v2, with one deliberate change: it
returns features at **half** the input resolution and lets the heads run there.
v2 upsampled to full resolution inside the trunk, so Head B materialised a
(B, 128, 512, 512) bin-logit tensor — 3.4 GB at batch 12 in bf16, for no accuracy
benefit.  DPT's own design predicts at 1/2 and upsamples the *output*.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


class ResidualConvUnit(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(8, c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(8, c)

    def forward(self, x):
        y = self.conv1(F.relu(x, inplace=False))
        y = self.conv2(F.relu(self.bn1(y), inplace=True))
        return x + self.bn2(y)


class FeatureFusionBlock(nn.Module):
    """`use_skip=False` drops `rcu1` entirely.

    The deepest block is called with no skip connection, so its `rcu1` never
    takes part in the forward.  On one GPU that is only six wasted parameters;
    under DDP with `find_unused_parameters=False` the reducer waits forever for
    gradients that never arrive and the next forward dies with "Expected to have
    finished reduction in the prior iteration" (indices 60-65 = `fuse[3].rcu1`).
    """

    def __init__(self, c: int, use_skip: bool = True):
        super().__init__()
        self.rcu1 = ResidualConvUnit(c) if use_skip else None
        self.rcu2 = ResidualConvUnit(c)
        self.out = nn.Conv2d(c, c, 1)

    def forward(self, x, skip=None):
        if skip is not None:
            if self.rcu1 is None:
                raise RuntimeError("this FeatureFusionBlock was built without a "
                                   "skip path (use_skip=False)")
            s = self.rcu1(skip)
            # Align to the skip before adding. `resample[3]` is a stride-2 conv,
            # which ROUNDS UP (ceil), while the upsample below is a flat x2 — so
            # an odd token grid comes back one pixel too large and the add dies
            # with "The size of tensor a (6) must match the size of tensor b (5)".
            # A 512 px tile is a 32-grid and every TTA scale keeps it even, which
            # is why only small tile_size configs (a 64 px test tile at TTA 1.25
            # -> an 80 px input -> a 5-grid) ever hit it. On an even grid the
            # shapes already match and this is a no-op, so nothing about the real
            # run changes.
            if x.shape[-2:] != s.shape[-2:]:
                x = F.interpolate(x, size=s.shape[-2:], mode="bilinear",
                                  align_corners=False)
            x = x + s
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
            nn.ConvTranspose2d(proj[0], proj[0], 4, stride=4),   # /16 -> /4
            nn.ConvTranspose2d(proj[1], proj[1], 2, stride=2),   # /16 -> /8
            nn.Identity(),                                        # /16
            nn.Conv2d(proj[3], proj[3], 3, stride=2, padding=1),  # /16 -> /32
        ])
        self.to_dim = nn.ModuleList(nn.Conv2d(p, dim, 3, padding=1, bias=False) for p in proj)
        # fuse[3] is the deepest block and runs without a skip — no rcu1.
        self.fuse = nn.ModuleList(FeatureFusionBlock(dim, use_skip=i < 3)
                                  for i in range(4))

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        f = [self.to_dim[i](self.resample[i](self.proj[i](feats[i]))) for i in range(4)]

        def run(fn, *a):
            if self.grad_checkpoint and self.training:
                return checkpoint(fn, *a, use_reentrant=False)
            return fn(*a)

        x = run(self.fuse[3], f[3])            # /32 -> /16
        x = run(self.fuse[2], x, f[2])         # /16 -> /8
        x = run(self.fuse[1], x, f[1])         # /8  -> /4
        x = run(self.fuse[0], x, f[0])         # /4  -> /2
        return x                                # (B, dim, H/2, W/2)
