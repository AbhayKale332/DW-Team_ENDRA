"""v5 detail branch: shapes, zero-init equivalence, old checkpoints, ONNX."""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from config import Config
from tests.stub_encoder import use_stub


def _cfg(detail: bool):
    c = Config()
    c.tile_size, c.decoder_dim, c.n_bins = 64, 32, 8
    c.grad_checkpoint_encoder = False
    c.detail_branch = detail
    c.detail_dim = 16
    return c


@pytest.fixture
def pair():
    undo = use_stub(hidden=32, patch=16, layers=8)
    from models.heads import DepthWizardNet

    torch.manual_seed(0)
    v4 = DepthWizardNet(_cfg(False)).eval()
    v5 = DepthWizardNet(_cfg(True)).eval()
    miss, unexp = v5.load_state_dict(v4.state_dict(), strict=False)
    yield v4, v5, miss, unexp
    undo()


def test_old_checkpoints_load_missing_only_the_new_modules(pair):
    _, _, miss, unexp = pair
    assert not unexp
    assert miss and all(k.startswith(("stem.", "inject.", "upsampler.")) for k in miss)


def test_zero_init_reproduces_the_v4_forward(pair):
    v4, v5, _, _ = pair
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        a, b = v4(x), v5(x)
    for k in ("fused", "a", "b"):
        assert b[k].shape == a[k].shape == (2, 1, 64, 64)
        # 1e-4 logit leak onto the non-bilinear neighbours: ~5 mm on a 30 m map
        assert torch.allclose(a[k], b[k], atol=1e-2, rtol=1e-3), (k, (a[k] - b[k]).abs().max())


def test_convex_upsampler_starts_as_bilinear():
    from models.heads import ConvexUp2x

    up = ConvexUp2x(8)
    t = torch.rand(1, 1, 7, 9)
    m = up.weights(torch.randn(1, 8, 7, 9))
    ref = F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)
    assert torch.allclose(ConvexUp2x.apply(t, m), ref, atol=2e-3)


def test_convex_upsampler_can_keep_an_edge():
    """With all weight on the centre neighbour the output is a 2x nearest
    upsample — the step stays one pixel wide instead of being smeared."""
    from models.heads import ConvexUp2x

    t = torch.zeros(1, 1, 4, 4)
    t[..., 2:] = 30.0
    m = torch.zeros(1, 1, 9, 2, 2, 4, 4)
    m[:, :, 4] = 1.0
    out = ConvexUp2x.apply(t, m)
    assert set(out.unique().tolist()) == {0.0, 30.0}


def test_detail_branch_trains(pair):
    _, v5, _, _ = pair
    v5.train()
    x = torch.randn(2, 3, 64, 64)
    out = v5(x)
    out["fused"].mean().backward()
    assert v5.upsampler.net[-1].weight.grad.abs().sum() > 0
    assert v5.inject.weight.grad.abs().sum() > 0


def test_onnx_export_with_the_detail_branch(tmp_path, monkeypatch):
    undo = use_stub(hidden=32, patch=16, layers=8)
    try:
        from config import safe_config_dict
        from dwdata.preprocess import PreprocSpec
        from models.heads import DepthWizardNet

        cfg = _cfg(True)
        net = DepthWizardNet(cfg).eval()
        spec = PreprocSpec(tile_size=64, canonical_gsd_m=0.5)
        ck = tmp_path / "best.pt"
        torch.save({"model": net.state_dict(), "preproc": spec.to_dict(),
                    "config": safe_config_dict(cfg), "epoch": 1,
                    "encoder_included": True}, ck)
        import dwdata.preprocess as pp

        monkeypatch.setattr(pp, "resolve_encoder_stats",
                            lambda *a, **k: (pp.DINOV3_SAT_MEAN, pp.DINOV3_SAT_STD))
        from infer.export_onnx import export

        out = tmp_path / "m.onnx"
        try:
            export(str(ck), str(out), opset=18, check=True)
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"torch.onnx export unavailable in this build: {e}")
        ort = pytest.importorskip("onnxruntime")
        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        x = np.random.RandomState(0).randn(1, 3, 64, 64).astype(np.float32)
        got = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        with torch.no_grad():
            ref = net(torch.from_numpy(x))["fused"].numpy()
        assert np.abs(got.reshape(ref.shape) - ref).max() < 1e-3
    finally:
        undo()


def test_fp16_forward_survives_a_trunk_past_fp16_range(pair):
    """A bf16-trained trunk (v4) can put features past 65504.  Under fp16
    autocast that used to be `inf` in the trunk output and NaN in every head.
    The unnormalised parts now run in fp32 and the heads get a scaled copy,
    which they cannot tell apart from the original."""
    _, v5, _, _ = pair
    with torch.no_grad():
        v5.trunk.fuse[0].out.weight.mul_(3e4)
        v5.trunk.fuse[0].out.bias.mul_(3e4)
        v5.upsampler.net[-1].weight.normal_(0, 0.01)   # a trained, non-bilinear upsampler
        v5.inject.weight.normal_(0, 0.01)
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        ref = v5(x)
        assert v5.trunk(v5.encoder(x)).abs().max() > 65504   # the premise
        with torch.autocast("cpu", dtype=torch.float16):
            got = v5(x)
    for k in ("fused", "a", "b", "alpha", "b_std", "seg", "b_logits", "b_centres"):
        assert torch.isfinite(got[k]).all(), k
        # fp16 rounding, not the rescale: the old path gave NaN here, not error
        err = (got[k].float() - ref[k]).abs().mean().item()
        assert err < 1e-3 * ref[k].abs().max().item() + 1e-3, (k, err)
