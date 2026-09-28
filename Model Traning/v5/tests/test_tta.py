import torch

from models.tta import _D4, _apply, _invert, tta_predict


def test_d4_elements_are_distinct_and_invertible():
    x = torch.arange(16.0).view(1, 1, 4, 4)
    seen = set()
    for k, f in _D4:
        y = _apply(x, k, f)
        seen.add(tuple(y.flatten().tolist()))
        assert torch.equal(_invert(y, k, f), x)
    assert len(seen) == 8


def test_tta_of_an_equivariant_model_is_the_identity():
    class Eq(torch.nn.Module):
        def forward(self, x):
            return {"fused": x.mean(1, keepdim=True) * 2.0}

    x = torch.randn(2, 3, 32, 32)
    out = tta_predict(Eq(), x, (1.0,), "fused")
    assert torch.allclose(out, x.mean(1, keepdim=True) * 2.0, atol=1e-5)


def test_tta_averages_a_direction_sensitive_model():
    """A model that keys on orientation gets its bias averaged away."""
    class Biased(torch.nn.Module):
        def forward(self, x):
            r = torch.zeros_like(x[:, :1])
            r[..., : x.shape[-1] // 2] = 1.0      # always "left half is tall"
            return {"fused": r}

    out = tta_predict(Biased(), torch.randn(1, 3, 16, 16), (1.0,), "fused")
    assert abs(float(out.mean()) - 0.5) < 1e-5
    assert float(out.std()) < 0.5                 # smeared, not a hard edge


def test_multi_scale_returns_input_resolution():
    class M(torch.nn.Module):
        def forward(self, x):
            return {"fused": x[:, :1] * 0 + x.shape[-1]}

    out = tta_predict(M(), torch.randn(1, 3, 32, 32), (1.0, 1.5), "fused", patch=16)
    assert out.shape == (1, 1, 32, 32)
