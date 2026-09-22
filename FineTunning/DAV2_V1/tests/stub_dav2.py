"""A tiny stand-in for Depth Anything V2 so the pipeline is testable offline.

Mimics exactly what `models/dav2.py` relies on:

  * `.config` carrying `patch_size`, `fusion_hidden_size` and a
    `backbone_config` with `hidden_size` / `num_hidden_layers`,
  * `.backbone` returning `feature_maps` as **token sequences** with one prefix
    token, which is the shape `DepthAnythingNeck` strips with `[:, 1:]`,
  * `.neck` — the **real** `DepthAnythingNeck`, built tiny.  The neck is the
    whole point of the swap and its reassemble/fusion geometry is exactly what
    the shape chain has to be checked against, so stubbing it would test
    nothing.  Falls back to a hand-rolled equivalent only when `transformers`
    is not installed at all.
  * `.head` — present, so the "discard the head" path is exercised.

The block list is deliberately named `trunk_stack` — nothing like
"layer"/"layers"/"blocks" — because that name-based lookup is what raised
`AttributeError: cannot locate transformer blocks on the encoder` and killed a
real run at epoch 21.  The structural finder in `dav2.py` has to find it anyway.
"""
from __future__ import annotations

import torch
from torch import nn


class _BackboneCfg:
    def __init__(self, hidden, layers):
        self.hidden_size = hidden
        self.num_hidden_layers = layers


class _Cfg:
    # Same four reassemble factors DAv2 uses, so the stub's token grid scales
    # 4 / 2 / 1 / 0.5 exactly like the real one.
    reassemble_factors = (4, 2, 1, 0.5)
    head_in_index = -1

    def __init__(self, hidden, patch, layers, fusion):
        self.patch_size = patch
        self.fusion_hidden_size = fusion
        self.reassemble_hidden_size = hidden
        self.neck_hidden_sizes = (hidden, hidden, hidden, hidden)
        self.backbone_config = _BackboneCfg(hidden, layers)
        self.hidden_size = hidden
        self.num_hidden_layers = layers


class _Out:
    def __init__(self, feature_maps):
        self.feature_maps = feature_maps


class StubBackbone(nn.Module):
    """Token-sequence feature maps, one prefix token, four taps."""

    def __init__(self, hidden, patch, layers):
        super().__init__()
        self.config = _BackboneCfg(hidden, layers)
        self.embeddings = nn.Module()
        # Masked-image-modelling only: never on the path to the taps, which is
        # what the unreachable-parameter probe has to notice.
        self.embeddings.mask_token = nn.Parameter(torch.zeros(1, hidden))
        self.embed = nn.Conv2d(3, hidden, patch, stride=patch)
        self.trunk_stack = nn.ModuleList(nn.Linear(hidden, hidden)
                                         for _ in range(layers))
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
        self.gc_enabled = False

    def gradient_checkpointing_enable(self, **kw):
        self.gc_enabled = True

    def gradient_checkpointing_disable(self):
        self.gc_enabled = False

    def forward(self, pixel_values, **kw):
        x = self.embed(pixel_values)
        b = x.shape[0]
        t = torch.cat([self.cls.expand(b, -1, -1), x.flatten(2).transpose(1, 2)],
                      dim=1)
        taps = []
        for i, blk in enumerate(self.trunk_stack):
            t = t + blk(t)
            if i % max(1, len(self.trunk_stack) // 4) == 0 and len(taps) < 4:
                taps.append(t)
        while len(taps) < 4:
            taps.append(t)
        return _Out(tuple(taps[:4]))


def _build_neck(cfg):
    from transformers.models.depth_anything.modeling_depth_anything import (
        DepthAnythingNeck,
    )

    return DepthAnythingNeck(cfg)


class StubDAV2(nn.Module):
    def __init__(self, hidden=32, patch=14, layers=8, fusion=16):
        super().__init__()
        self.config = _Cfg(hidden, patch, layers, fusion)
        self.backbone = StubBackbone(hidden, patch, layers)
        self.neck = _build_neck(self.config)
        # The relative-depth head DAV2Backbone is supposed to delete.
        self.head = nn.Conv2d(fusion, 1, 1)


def use_stub(hidden=32, patch=14, layers=8, fusion=16):
    """Install the stub as the DAv2 builder; returns an undo callable."""
    from models.dav2 import DAV2Backbone

    prev = DAV2Backbone.BUILDER
    DAV2Backbone.BUILDER = lambda cfg: StubDAV2(hidden, patch, layers, fusion)
    return lambda: setattr(DAV2Backbone, "BUILDER", prev)
