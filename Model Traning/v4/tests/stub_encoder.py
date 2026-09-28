"""A tiny stand-in for DINOv3 so the whole pipeline is testable offline.

Mimics what `models/encoder.py` relies on: a `.config` carrying hidden_size /
patch_size / num_hidden_layers, a `nn.ModuleList` of blocks (so the structural
block finder has something to find), and `hidden_states` from forward.

The block list is deliberately named `trunk_stack` — nothing like the
"layer" / "layers" / "blocks" names v2 searched for — because that name-based
lookup is precisely what raised `AttributeError: cannot locate transformer blocks
on the encoder` and killed the real run at epoch 21.
"""
from __future__ import annotations

import torch
from torch import nn


class _Cfg:
    def __init__(self, hidden, patch, layers):
        self.hidden_size, self.patch_size, self.num_hidden_layers = hidden, patch, layers


class StubViT(nn.Module):
    def __init__(self, hidden=32, patch=16, layers=8):
        super().__init__()
        self.config = _Cfg(hidden, patch, layers)
        self.embed = nn.Conv2d(3, hidden, patch, stride=patch)
        self.trunk_stack = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.cls = nn.Parameter(torch.zeros(1, 5, hidden))
        self.gc_enabled = False

    def gradient_checkpointing_enable(self, **kw):
        self.gc_enabled = True

    def forward(self, pixel_values, **kw):
        x = self.embed(pixel_values)
        b = x.shape[0]
        t = torch.cat([self.cls.expand(b, -1, -1), x.flatten(2).transpose(1, 2)], dim=1)
        hs = [t]
        for blk in self.trunk_stack:
            t = t + blk(t)
            hs.append(t)
        return type("Out", (), {"hidden_states": tuple(hs), "last_hidden_state": t})()


def use_stub(hidden=32, patch=16, layers=8):
    """Install the stub as the encoder builder; returns an undo callable."""
    from models.encoder import DINOv3Encoder

    prev = DINOv3Encoder.BUILDER
    DINOv3Encoder.BUILDER = lambda cfg: StubViT(hidden, patch, layers)
    return lambda: setattr(DINOv3Encoder, "BUILDER", prev)
