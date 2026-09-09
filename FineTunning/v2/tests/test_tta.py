import torch

from models.tta import _D4, _apply, _invert, tta_predict


def test_dihedral_inverse_identity():
    x = torch.randn(1, 1, 16, 16)
    for k, flip in _D4:
        y = _invert(_apply(x, k, flip), k, flip)
        assert torch.allclose(x, y, atol=1e-5)


def test_tta_of_equivariant_model_matches_plain():
    class Const(torch.nn.Module):
        def forward(self, img):
            # rotation/flip-equivariant: output depends only on per-pixel mean
            h = img.mean(dim=1, keepdim=True).abs()
            return {"fused": h}

    m = Const()
    img = torch.randn(2, 3, 16, 16)
    plain = m(img)["fused"]
    tta = tta_predict(m, img, scales=(1.0,), key="fused")
    assert torch.allclose(plain, tta, atol=1e-4)


def test_tta_shape_with_scales():
    class M(torch.nn.Module):
        def forward(self, img):
            return {"fused": img.mean(1, keepdim=True)}

    out = tta_predict(M(), torch.randn(1, 3, 32, 32), scales=(1.0, 1.3), key="fused")
    assert out.shape == (1, 1, 32, 32)
