"""Validate against the *real* DINOv3 module (random weights, no download).

This is the regression test for the bug that killed the v2 run.  The actual HF
layout is `DINOv3ViTModel.model.layer` — a ModuleList nested under a child called
`model`.  v2 searched `self.model.{layer,layers,blocks}` and
`self.model.encoder.{...}`, found none of them, and raised
`AttributeError: cannot locate transformer blocks on the encoder` at epoch 21/30,
so the encoder was never unfrozen and the run died before its final evaluation.
"""
import pytest
import torch

transformers = pytest.importorskip("transformers")

from config import Config  # noqa: E402


@pytest.fixture
def small_dinov3():
    from transformers import DINOv3ViTConfig, DINOv3ViTModel
    from models.encoder import DINOv3Encoder

    hcfg = DINOv3ViTConfig(hidden_size=64, num_hidden_layers=6, num_attention_heads=4,
                           intermediate_size=128, patch_size=16, image_size=64)
    prev = DINOv3Encoder.BUILDER
    DINOv3Encoder.BUILDER = lambda cfg: DINOv3ViTModel(hcfg)

    c = Config()
    c.tile_size, c.decoder_dim, c.n_bins = 64, 32, 8
    c.encoder_feature_indices = (2, 3, 4, 6)
    c.grad_checkpoint_encoder = False
    yield c
    DINOv3Encoder.BUILDER = prev


def test_v2_name_lookup_really_does_fail():
    """Pin the root cause so nobody 'simplifies' the finder back."""
    from transformers import DINOv3ViTConfig, DINOv3ViTModel

    m = DINOv3ViTModel(DINOv3ViTConfig(
        hidden_size=64, num_hidden_layers=6, num_attention_heads=4,
        intermediate_size=128, patch_size=16, image_size=64))
    for attr in ("layer", "layers", "blocks"):
        enc = getattr(m, "encoder", m)
        assert getattr(enc, attr, None) is None and getattr(m, attr, None) is None


def test_v3_finds_the_blocks_structurally(small_dinov3):
    from models.encoder import DINOv3Encoder

    enc = DINOv3Encoder(small_dinov3)
    assert enc.blocks is not None
    assert len(enc.blocks) == 6
    assert enc.hidden == 64 and enc.patch == 16


def test_real_forward_and_unfreeze(small_dinov3):
    from models.heads import DepthWizardNetV3

    net = DepthWizardNetV3(small_dinov3)
    x = torch.randn(1, 3, 64, 64)

    out = net(x)
    assert out["fused"].shape == (1, 1, 64, 64) and torch.isfinite(out["fused"]).all()
    assert (out["fused"] >= 0).all()

    # this is the call that crashed v2
    net.encoder.set_frozen(False)
    net(x)["fused"].sum().backward()
    g = [p.grad for p in net.encoder.model.parameters() if p.grad is not None]
    assert g and any(t.abs().sum() > 0 for t in g), "encoder received no gradient"


def test_real_llrd_groups(small_dinov3):
    from models.heads import DepthWizardNetV3

    net = DepthWizardNetV3(small_dinov3)
    net.encoder.set_frozen(False)
    groups = net.encoder.llrd_param_groups(1e-4, 0.8, 0.05)
    lrs = sorted(g["lr"] for g in groups)
    assert len(groups) >= 6
    assert lrs[0] < lrs[-1] <= 1e-4 + 1e-12
    # every trainable encoder tensor lands in exactly one group
    n_in = sum(p.numel() for g in groups for p in g["params"])
    n_all = sum(p.numel() for p in net.encoder.model.parameters() if p.requires_grad)
    assert n_in == n_all


def test_prefix_tokens_are_dropped_correctly(small_dinov3):
    """Patch tokens are the trailing hp*wp entries whatever the prefix count."""
    from models.encoder import DINOv3Encoder

    enc = DINOv3Encoder(small_dinov3)
    feats = enc(torch.randn(2, 3, 64, 64))
    assert len(feats) == 4
    for f in feats:
        assert f.shape == (2, 64, 4, 4)      # 64/16 = 4 patches per side
