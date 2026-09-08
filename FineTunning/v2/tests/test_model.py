import torch


def test_head_output_shapes(trunk_and_heads):
    t = trunk_and_heads
    x = t["x"]
    b, _, h, w = x.shape
    a = t["head_a"](x)
    hb = t["head_b"](x)
    seg = t["head_c"](x)
    fused, alpha = t["fusion"](x, a, hb["height"])
    assert a.shape == (b, 1, h, w)
    assert hb["height"].shape == (b, 1, h, w)
    assert seg.shape[1] == len(t["cfg"].class_names)
    assert fused.shape == (b, 1, h, w)
    assert alpha.shape == (b, 1, h, w)


def test_nonneg_and_bounds(trunk_and_heads):
    t = trunk_and_heads
    x = t["x"]
    a = t["head_a"](x)
    hb = t["head_b"](x)
    _, alpha = t["fusion"](x, a, hb["height"])
    assert (a >= 0).all()
    assert (hb["height"] >= 0).all()
    assert (alpha >= 0).all() and (alpha <= 1).all()


def test_bin_probs_sum_to_one(trunk_and_heads):
    hb = trunk_and_heads["head_b"](trunk_and_heads["x"])
    s = hb["probs"].sum(dim=1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-4)


def test_bin_centres_increasing(trunk_and_heads):
    c = trunk_and_heads["head_b"](trunk_and_heads["x"])["centres"]
    assert (c.diff(dim=1) > 0).all()


def test_fusion_between_experts(trunk_and_heads):
    t = trunk_and_heads
    x = t["x"]
    a = torch.zeros_like(t["head_a"](x))
    b = torch.ones_like(a) * 5.0
    fused, _ = t["fusion"](x, a, b)
    assert (fused >= -1e-4).all() and (fused <= 5.0 + 1e-4).all()
