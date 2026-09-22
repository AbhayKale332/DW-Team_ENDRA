"""The neck must survive an odd token grid — and 37 is odd.

v4's own DPT trunk had this bug: `resample[3]` is a stride-2 convolution, which
rounds the token grid UP, while the fusion block upsampled by a flat x2, so a
grid of 5 came back as 6 and the skip addition died with "The size of tensor a
(6) must match the size of tensor b (5)".  That never fired in a real v4 run
because a 512 px patch-16 tile is a 32-grid and every TTA scale kept it even.

**This variant's real runs are odd**: 518 / 14 = 37, and TTA 1.25 gives
648 / 14 = 46.  So the case v4 only ever hit in a 64 px test is the case that
now runs every step, which is why it is checked here against the actual
`DepthAnythingNeck` rather than against arithmetic.

DAv2's fusion stage passes an explicit `size=` taken from the next feature map,
so the alignment is exact by construction — but "by construction" is a claim
about somebody else's code across `transformers` versions, and that is exactly
the kind of claim worth a test.
"""
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from config import Config

from tests.stub_dav2 import use_stub


def _encoder():
    from models.dav2 import DAV2Backbone

    c = Config()
    c.decoder_dim = 8
    c.grad_checkpoint_encoder = False
    c.channels_last = False
    return DAV2Backbone(c).eval()


@pytest.mark.parametrize("grid", [2, 3, 4, 5, 7, 8, 13, 37])
def test_neck_handles_any_token_grid(grid):
    """A patch-14 input of `grid * 14` px must come out at half that."""
    undo = use_stub(hidden=16, patch=14, layers=4, fusion=8)
    try:
        enc = _encoder()
        px = grid * 14
        with torch.no_grad():
            out = enc(torch.zeros(1, 3, px, px))
        assert out.shape == (1, 8, px // 2, px // 2), out.shape
        assert torch.isfinite(out).all()
    finally:
        undo()


def test_the_fusion_stage_ends_at_eight_times_the_token_grid():
    """The 37 -> 296 -> 259 chain, spelled out.

    If a `transformers` release changes `head_in_index` or drops the last
    fusion layer's unconstrained x2, this is the assertion that catches it —
    and the symptom otherwise would be a silent halving of output detail.
    """
    undo = use_stub(hidden=16, patch=14, layers=4, fusion=8)
    try:
        enc = _encoder()
        with torch.no_grad():
            neck_out = enc._neck_out(torch.zeros(1, 3, 5 * 14, 5 * 14))
        assert neck_out.shape[-2:] == (5 * 8, 5 * 8), neck_out.shape
    finally:
        undo()
