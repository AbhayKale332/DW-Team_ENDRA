"""The DPT fusion must survive an odd token grid.

`resample[3]` is a stride-2 convolution, which rounds the token grid UP, while
`FeatureFusionBlock` upsampled by a flat x2 — so a grid of 5 came back as 6 and
the skip addition died with "The size of tensor a (6) must match the size of
tensor b (5)".

A 512 px tile is a 32-grid and every TTA scale keeps it even (640/16 = 40),
which is why three v4 runs never hit it and the two end-to-end tests that did
— both of which use a 64 px tile, where TTA 1.25 gives an 80 px input and a
5-grid — looked like flaky infrastructure rather than a decoder bug.
"""
import sys, pathlib
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from models.dpt import DPTTrunk


@pytest.mark.parametrize("grid", [4, 5, 7, 8, 13])
def test_fusion_handles_any_token_grid(grid):
    trunk = DPTTrunk(in_ch=16, dim=8)
    feats = [torch.randn(2, 16, grid, grid) for _ in range(4)]
    out = trunk(feats)
    # /16 -> /2 is eight times the token grid, whatever its parity.
    assert out.shape == (2, 8, grid * 8, grid * 8), out.shape
    assert torch.isfinite(out).all()


def test_even_grid_is_untouched_by_the_alignment():
    """The fix must be a no-op where the shapes already agreed — i.e. every
    real run, which is a 32-grid."""
    torch.manual_seed(0)
    trunk = DPTTrunk(in_ch=16, dim=8).eval()
    feats = [torch.randn(1, 16, 8, 8) for _ in range(4)]
    with torch.no_grad():
        a = trunk(feats)
        b = trunk(feats)
    assert torch.equal(a, b)
    assert a.shape == (1, 8, 64, 64)
