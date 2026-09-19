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


def test_v2_name_lookup_is_not_something_you_can_rely_on():
    """Pin the root cause so nobody 'simplifies' the finder back.

    This used to assert that *no* name (`layer` / `layers` / `blocks`) resolves on
    a `DINOv3ViTModel`, which held for the transformers build it was written
    against and fails on others: on the Kaggle image
    `DINOv3ViTModel.layer` is a real `ModuleList` of `DINOv3ViTLayer`, and the
    test failed there while passing locally on transformers 5.17.

    Which is the point, and a sharper version of it than the original assertion
    made. The attribute name is a private detail of whichever transformers
    version happens to be installed, so a name lookup is not something the
    encoder can be built on — not because the name is always absent, but because
    whether it is present is not ours to decide. The finder therefore locates the
    block list structurally (`models/encoder.py:_find_blocks`, by ModuleList
    length against `num_hidden_layers`), and what is asserted here is the
    property that actually has to hold: it finds the right list whatever the
    attribute is called, including when it is called nothing recognisable.
    """
    import transformers
    from transformers import DINOv3ViTConfig, DINOv3ViTModel

    m = DINOv3ViTModel(DINOv3ViTConfig(
        hidden_size=64, num_hidden_layers=6, num_attention_heads=4,
        intermediate_size=128, patch_size=16, image_size=64))
    enc = getattr(m, "encoder", m)
    names = {a: getattr(enc, a, None) is not None or getattr(m, a, None) is not None
             for a in ("layer", "layers", "blocks")}
    # Informational, never asserted — it is exactly the thing that moves between
    # versions, and printing it is what makes a future surprise diagnosable.
    print(f"[test] transformers {transformers.__version__} exposes {names}")

    from models.encoder import DINOv3Encoder

    finder = DINOv3Encoder._find_blocks
    probe = type("P", (), {"model": m, "n_layers": 6})()
    found = finder(probe)
    assert found is not None
    assert len(found) == 6, f"structural finder got {len(found)} blocks, want 6"

    # …and still, with every name it could have keyed on removed.
    import torch.nn as nn

    class Renamed(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.trunk_stack = inner        # the one name nothing looks for

    inner = found
    probe2 = type("P", (), {"model": Renamed(inner), "n_layers": 6})()
    found2 = finder(probe2)
    assert found2 is not None and len(found2) == 6, \
        "the finder is keying on a name again — it must locate blocks by shape"


def test_v3_finds_the_blocks_structurally(small_dinov3):
    from models.encoder import DINOv3Encoder

    enc = DINOv3Encoder(small_dinov3)
    assert enc.blocks is not None
    assert len(enc.blocks) == 6
    assert enc.hidden == 64 and enc.patch == 16


def test_real_forward_and_unfreeze(small_dinov3):
    from models.heads import DepthWizardNet

    net = DepthWizardNet(small_dinov3)
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
    from models.heads import DepthWizardNet

    net = DepthWizardNet(small_dinov3)
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


def test_no_trainable_param_is_unreachable_after_unfreeze(small_dinov3):
    """The DDP precondition: everything that wants a gradient must get one.

    With `find_unused_parameters=False` the reducer buckets every
    `requires_grad` parameter at construction and then blocks until each one
    reports a gradient.  Unfreezing the encoder used to register three tensors
    that no gradient can reach — `embeddings.mask_token` (masked-image
    modelling only) and the backbone's trailing LayerNorm, which is applied to
    `last_hidden_state` while the taps read the pre-norm `hidden_states` — so
    the reduction never finished and the next forward died in
    `_rebuild_buckets`.  The 2xT4 run reported them as indices `1 413 414`, one
    step after the unfreeze at epoch 3.

    WHICH tensors those are is not part of the contract, and this test must not
    assert it.  `_freeze_unreachable` finds them by probing -- one forward and
    backward, and whatever comes back with `grad is None` is unreachable by
    definition -- precisely so it survives an HF refactor that a name list does
    not.  It earned that: on transformers as packaged in the Modal H100 image
    the backbone's trailing LayerNorm IS on the path to the taps, the probe
    correctly leaves `norm.weight`/`norm.bias` trainable, and asserting them
    frozen failed a suite that was describing a different version of the
    library rather than a broken model.

    So assert the invariant the probe exists to guarantee, which is the one DDP
    actually cares about: after the unfreeze, nothing that wants a gradient is
    starved of one.
    """
    from models.heads import DepthWizardNet

    net = DepthWizardNet(small_dinov3)
    net.encoder.set_frozen(False)

    enc = net.encoder.model
    # `mask_token` is structural, not version-dependent: the backbone reads it
    # only when `bool_masked_pos` is passed, and `_hidden_states` never passes
    # it, so no forward this project runs can reach it.
    assert enc.embeddings.mask_token.requires_grad is False
    # ...and the probe is not simply switching the whole backbone off
    assert sum(p.numel() for p in enc.parameters() if p.requires_grad) > 0

    out = net(torch.randn(1, 3, 64, 64))
    total = sum(v.float().sum() for v in out.values() if torch.is_tensor(v))
    total.backward()
    starved = [n for n, p in net.named_parameters()
               if p.requires_grad and p.grad is None]
    assert not starved, f"DDP would hang waiting on: {starved}"
