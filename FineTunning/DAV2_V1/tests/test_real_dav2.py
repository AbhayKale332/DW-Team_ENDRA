"""Validate against the *real* Depth Anything V2 module, not the stub.

Two tiers, because they fail for different reasons:

1. `small_dav2` builds a genuine `DepthAnythingForDepthEstimation` from a tiny
   config — real `Dinov2Backbone`, real `DepthAnythingNeck`, random weights, no
   download.  This is where the structural block finder and the
   unreachable-parameter probe are exercised against HF's actual module layout,
   which is the thing that killed the v2 run (`AttributeError: cannot locate
   transformer blocks on the encoder`, epoch 21/30) and the thing that would
   hang DDP if the probe missed a dead tensor.

2. `test_the_published_checkpoint_*` downloads the ungated
   `depth-anything/Depth-Anything-V2-Base-hf` and checks the 518 -> 37 -> 296
   -> 259 chain against the published weights rather than against arithmetic in
   a docstring.  Skipped without network.
"""
import os

import pytest
import torch

transformers = pytest.importorskip("transformers")

from config import Config

REAL = os.environ.get("DW_TEST_REAL_CKPT", "").lower() in ("1", "true", "yes")
needs_net = pytest.mark.skipif(
    not REAL, reason="set DW_TEST_REAL_CKPT=1 to download the real checkpoint")


def _tiny_config():
    from transformers import DepthAnythingConfig, Dinov2Config

    backbone = Dinov2Config(
        hidden_size=64, num_hidden_layers=6, num_attention_heads=4,
        intermediate_size=128, patch_size=14, image_size=56,
        out_indices=[3, 4, 5, 6], reshape_hidden_states=False)
    return DepthAnythingConfig(
        backbone_config=backbone, patch_size=14, fusion_hidden_size=16,
        reassemble_hidden_size=64,
        neck_hidden_sizes=[16, 16, 16, 16], head_hidden_size=8)


@pytest.fixture
def small_dav2():
    from models.dav2 import DAV2Backbone
    from transformers import DepthAnythingForDepthEstimation

    dacfg = _tiny_config()
    prev = DAV2Backbone.BUILDER
    DAV2Backbone.BUILDER = lambda cfg: DepthAnythingForDepthEstimation(dacfg)

    c = Config()
    c.tile_size, c.decoder_dim, c.n_bins = 56, 16, 8
    c.grad_checkpoint_encoder = False
    c.channels_last = False
    yield c
    DAV2Backbone.BUILDER = prev


def test_block_finder_locates_the_twelve_dinov2_blocks(small_dav2):
    """Structurally, not by name — the attribute name is a private detail of
    whichever `transformers` version happens to be installed."""
    from models.dav2 import DAV2Backbone

    enc = DAV2Backbone(small_dav2)
    assert enc.blocks is not None
    assert len(enc.blocks) == 6
    assert enc.patch == 14 and enc.hidden == 64


def test_shape_chain_through_the_real_neck(small_dav2):
    from models.dav2 import DAV2Backbone

    enc = DAV2Backbone(small_dav2).eval()
    with torch.no_grad():
        # 56 px / 14 = a 4-grid; the fusion stage ends at 8x that (32) and the
        # interpolate brings it to H/2.
        assert enc._neck_out(torch.zeros(2, 3, 56, 56)).shape == (2, 16, 32, 32)
        assert enc(torch.zeros(2, 3, 56, 56)).shape == (2, 16, 28, 28)


def test_the_probe_finds_the_structurally_dead_parameters(small_dav2):
    """Two of them are guaranteed by the architecture and both would hang DDP
    with `find_unused_parameters=False`:

      * `backbone.embeddings.mask_token` — masked-image-modelling only.
      * `neck.fusion_stage.layers.0.residual_layer1.*` — the deepest fusion
        layer is called with no residual, so its first residual unit never
        runs.  Exactly v4's `fuse[3].rcu1`, one level down.

    What is asserted is the *property* — nothing trainable is left without a
    gradient path — not a name list, because a name list is what the probe
    exists to replace.  The two known ones are checked as well, since if the
    probe stops finding them it has stopped working.
    """
    from models.dav2 import DAV2Backbone

    enc = DAV2Backbone(small_dav2)
    enc.set_frozen(False)

    enc(torch.zeros(1, 3, 56, 56)).sum().backward()
    stranded = [n for n, p in enc.model.named_parameters()
                if p.requires_grad and p.grad is None]
    assert stranded == [], stranded

    dead = [n for n, p in enc.model.named_parameters() if not p.requires_grad]
    assert any("mask_token" in n for n in dead), dead
    assert any("fusion_stage.layers.0.residual_layer1" in n for n in dead), dead


def test_the_head_is_gone_and_the_neck_is_not(small_dav2):
    from models.dav2 import DAV2Backbone

    enc = DAV2Backbone(small_dav2)
    assert getattr(enc.model, "head", None) is None
    assert sum(p.numel() for p in enc.model.neck.parameters()) > 0


def test_llrd_gives_the_neck_its_own_undecayed_group(small_dav2):
    from models.dav2 import DAV2Backbone

    enc = DAV2Backbone(small_dav2)
    enc.set_frozen(False)
    groups = enc.llrd_param_groups(base_lr=1e-4, decay=0.85, weight_decay=0.05)
    neck = [g for g in groups if g["name"].startswith("neck")]
    assert neck, [g["name"] for g in groups]
    assert all(abs(g["lr"] - 1e-4) < 1e-12 for g in neck)
    # …and the blocks still decay towards the input.
    blocks = [g for g in groups if g["name"].startswith("enc.d")]
    lrs = [g["lr"] for g in blocks]
    assert min(lrs) < max(lrs) <= 1e-4 + 1e-12


@needs_net
def test_the_published_checkpoint_has_the_geometry_this_variant_assumes():
    """518 -> 37x37 -> 296 -> 259, against the actual published weights."""
    from models.dav2 import DAV2Backbone

    c = Config()          # tile_size 518, encoder_patch 14, decoder_dim 128
    c.grad_checkpoint_encoder = False
    c.channels_last = False
    enc = DAV2Backbone(c).eval()

    assert enc.patch == 14
    assert enc.hidden == 768 and enc.n_layers == 12
    assert enc.fusion_hidden == c.decoder_dim, (
        f"decoder_dim {c.decoder_dim} should match this checkpoint's "
        f"fusion_hidden_size {enc.fusion_hidden}")

    with torch.no_grad():
        neck_out = enc._neck_out(torch.zeros(1, 3, 518, 518))
        out = enc(torch.zeros(1, 3, 518, 518))
    assert neck_out.shape == (1, 128, 296, 296), neck_out.shape
    assert out.shape == (1, c.decoder_dim, 259, 259), out.shape


@needs_net
def test_the_published_checkpoint_leaves_no_parameter_stranded():
    """The DDP deadlock check, on the real module.  A tensor that is trainable
    but never receives a gradient makes the reducer wait forever."""
    from models.dav2 import DAV2Backbone

    c = Config()
    c.grad_checkpoint_encoder = False
    c.channels_last = False
    enc = DAV2Backbone(c)
    enc.set_frozen(False)

    # 4 patches a side is the smallest grid the x0.5 reassemble can halve, and
    # it keeps this test to a few seconds on CPU.
    enc(torch.zeros(1, 3, 56, 56)).sum().backward()
    stranded = [n for n, p in enc.model.named_parameters()
                if p.requires_grad and p.grad is None]
    assert stranded == [], stranded
