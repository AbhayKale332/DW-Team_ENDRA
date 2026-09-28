import pytest
import torch

from config import Config, N_SEG_CLASSES
from tests.stub_dav2 import use_stub


@pytest.fixture
def net():
    undo = use_stub(hidden=32, patch=14, layers=8)
    from models.heads import DepthWizardNet

    c = Config()
    c.tile_size = 56
    c.decoder_dim = 32
    c.n_bins = 8
    c.grad_checkpoint_encoder = False
    m = DepthWizardNet(c)
    yield m, c
    undo()


def test_forward_shapes(net):
    m, c = net
    out = m(torch.randn(2, 3, 56, 56))
    assert out["fused"].shape == (2, 1, 56, 56)
    assert out["a"].shape == out["b"].shape == out["fused"].shape
    assert out["seg"].shape == (2, N_SEG_CLASSES, 56, 56)
    assert out["alpha"].shape == (2, 1, 56, 56)
    # heads run at half res -> the bin logits must NOT be full resolution
    assert out["b_logits"].shape == (2, c.n_bins, 28, 28)
    assert out["b_centres"].shape == (2, c.n_bins)


def test_outputs_are_non_negative_metres(net):
    m, _ = net
    out = m(torch.randn(2, 3, 56, 56) * 5)
    for k in ("a", "b", "fused"):
        assert (out[k] >= 0).all(), k


def test_head_a_gradient_survives_negative_preactivation(net):
    """v2 used F.relu on the raw conv output, so a unit that drifted negative had
    exactly zero gradient forever.  softplus keeps it alive."""
    m, _ = net
    x = torch.full((1, 32, 8, 8), -50.0, requires_grad=True)
    y = m.head_a(x).sum()
    y.backward()
    assert x.grad.abs().sum() > 0


def test_bin_centres_are_sorted_and_in_range(net):
    m, c = net
    ctr = m.head_b.bin_centres(torch.randn(3, 32, 8, 8))
    assert ctr.shape == (3, c.n_bins)
    assert (ctr.diff(dim=1) > 0).all()
    assert ctr.min() >= c.bin_min_m and ctr.max() <= c.bin_max_m


def test_block_finder_ignores_naming(net):
    """The v2 crash: blocks were looked up by attribute name and not found."""
    m, _ = net
    assert m.encoder.blocks is not None
    assert len(m.encoder.blocks) == 8


def test_freeze_then_unfreeze_produces_encoder_gradients(net):
    m, _ = net
    assert m.encoder.frozen
    assert not any(p.requires_grad for p in m.encoder.model.parameters())
    m.encoder.set_frozen(False)
    out = m(torch.randn(1, 3, 56, 56))
    out["fused"].sum().backward()
    grads = [p.grad for p in m.encoder.model.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_llrd_groups_decay_towards_the_input(net):
    m, _ = net
    m.encoder.set_frozen(False)
    groups = m.encoder.llrd_param_groups(base_lr=1e-4, decay=0.8, weight_decay=0.05)
    lrs = [g["lr"] for g in groups]
    assert len(groups) > 2
    assert max(lrs) <= 1e-4 + 1e-12 and min(lrs) < max(lrs)
    # norms/biases must not get weight decay
    assert any(g["weight_decay"] == 0.0 for g in groups)


def test_head_state_dict_excludes_the_public_backbone_and_neck(net):
    """DAv2's neck is public pretrained weight too, so it has to be stripped
    alongside the backbone — and `encoder.proj` is random, so it must not be."""
    m, _ = net
    sd = m.head_state_dict()
    assert not any(k.startswith("encoder.model.") for k in sd)
    assert not any(".neck." in k for k in sd)
    assert any(k.startswith("encoder.proj.") for k in sd)
    assert any(k.startswith("head_a.") for k in sd)
    assert len(m.full_state_dict()) > len(sd)


def test_the_relative_depth_head_is_discarded(net):
    """DAv2's own head predicts relative inverse depth through a sigmoid.
    Leaving it attached would also hand DDP parameters no gradient reaches."""
    m, _ = net
    assert getattr(m.encoder.model, "head", None) is None


def test_proj_stays_trainable_while_the_encoder_is_frozen(net):
    """`encoder.proj` is randomly initialised, so it has to train during the
    `--freeze_epochs` warm start with the heads.  `train.py` puts it in the
    decoder group by matching `encoder.model.`, not `encoder.`."""
    m, _ = net
    assert m.encoder.frozen
    assert all(p.requires_grad for p in m.encoder.proj.parameters())
    names = [n for n, _ in m.named_parameters()
             if not n.startswith("encoder.model.")]
    assert "encoder.proj.weight" in names


def test_variable_input_size_is_accepted(net):
    """Any multiple of the patch size — TTA rescales to exactly that."""
    m, _ = net
    for s in (28, 84):
        assert m(torch.randn(1, 3, s, s))["fused"].shape == (1, 1, s, s)


def test_the_neck_output_is_projected_to_half_resolution(net):
    """The interpolate to H/2 is load-bearing: the neck's fusion stage ends at
    8x the token grid (56/14 = 4 -> 32), and without the step down HeadB's bin
    logits would be materialised there instead of at 28."""
    m, c = net
    x = m.encoder(torch.randn(2, 3, 56, 56))
    assert x.shape == (2, c.decoder_dim, 28, 28)
