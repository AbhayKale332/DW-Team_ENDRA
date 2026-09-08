"""Shared fixtures.  Offline by default; `-m hf` enables Hub-touching tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

V2 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V2))


def pytest_configure(config):
    config.addinivalue_line("markers", "hf: touches the Hugging Face Hub (network)")


@pytest.fixture
def cfg():
    from config import Config

    c = Config()
    c.smoke = True
    c.apply_smoke()
    c.tile_size = 64
    c.decoder_dim = 32
    c.n_bins = 16
    c.output_dir = "/tmp/dw_v2_test_out"
    c.cache_dir = "/tmp/dw_v2_test_cache"
    return c.validate()


@pytest.fixture
def synthetic_tile():
    """(rgb u8 HxWx3, height f32 HxW, seg i64 HxW) with a raised 'building'."""
    rng = np.random.default_rng(0)
    h = w = 96
    rgb = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    height = np.zeros((h, w), np.float32)
    height[30:60, 30:60] = 18.0
    seg = np.zeros((h, w), np.int64)
    seg[30:60, 30:60] = 2
    return rgb, height, seg


class _StubEncoder:
    """Stand-in for DINOv3Encoder — no transformers, no gated download."""

    hidden = 96
    unfrozen_block_ids: list = []

    def __init__(self, cfg):
        import torch
        from torch import nn

        with torch.random.fork_rng():
            torch.manual_seed(0)  # deterministic -> save/reload tests are stable
            self._m = nn.Conv2d(3, self.hidden, 16, stride=16)
        self.frozen = True

    def __call__(self, image):
        f = self._m(image)
        return [f, f, f, f]

    # nn.Module-ish shims used by the trainer
    def to(self, *a, **k):
        self._m.to(*a, **k)
        return self

    def train(self, mode=True):
        self._m.train(mode)
        return self

    def eval(self):
        self._m.eval()
        return self

    def parameters(self):
        return self._m.parameters()

    def named_parameters(self, prefix=""):
        return self._m.named_parameters(prefix=prefix)

    def unfreeze_last_n_blocks(self, n):
        return []


@pytest.fixture
def fake_net(cfg, monkeypatch):
    import models.heads as mh

    monkeypatch.setattr(mh, "DINOv3Encoder", _StubEncoder)
    net = mh.DepthWizardNetV2(cfg)
    return net


@pytest.fixture
def trunk_and_heads(cfg):
    """DPT trunk + 3 heads + fusion, fed random encoder features (no transformers)."""
    import torch

    from models.dpt import DPTTrunk
    from models.heads import GatedFusion, HeadA, HeadB, HeadC

    d = cfg.decoder_dim
    in_ch = 96
    trunk = DPTTrunk(in_ch, d)
    feats = [torch.randn(2, in_ch, cfg.tile_size // 16, cfg.tile_size // 16) for _ in range(4)]
    x = trunk(feats, (cfg.tile_size, cfg.tile_size))
    return {
        "x": x,
        "head_a": HeadA(d),
        "head_b": HeadB(d, cfg.n_bins, cfg.bin_min_m, cfg.bin_max_m),
        "head_c": HeadC(d, len(cfg.class_names)),
        "fusion": GatedFusion(d),
        "cfg": cfg,
    }
