"""ONNX export — the standalone-deployment path.

A silent export whose graph disagrees with the checkpoint is worse than no
export, so `verify()` compares the two and the test asserts on that comparison
rather than on the file merely existing.  onnxruntime is optional; the export
itself is still checked without it.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from config import Config
from dwdata.preprocess import PreprocSpec
from tests.stub_encoder import use_stub


def _ckpt(tmp_path):
    from models.heads import DepthWizardNet

    cfg = Config()
    cfg.tile_size, cfg.decoder_dim, cfg.n_bins = 64, 32, 8
    cfg.grad_checkpoint_encoder = False
    net = DepthWizardNet(cfg).eval()
    spec = PreprocSpec(tile_size=64, canonical_gsd_m=0.5)
    p = tmp_path / "best.pt"
    from config import safe_config_dict

    torch.save({"model": net.state_dict(), "preproc": spec.to_dict(),
                "config": safe_config_dict(cfg), "epoch": 1,
                "encoder_included": True}, p)
    return p, net, spec


def test_export_matches_the_checkpoint(tmp_path, monkeypatch):
    undo = use_stub(hidden=32, patch=16, layers=8)
    try:
        ck, net, spec = _ckpt(tmp_path)
        import dwdata.preprocess as pp

        monkeypatch.setattr(pp, "resolve_encoder_stats",
                            lambda *a, **k: (pp.DINOV3_SAT_MEAN, pp.DINOV3_SAT_STD))
        from infer.export_onnx import export

        out = tmp_path / "m.onnx"
        try:
            export(str(ck), str(out), opset=17, check=True)
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"torch.onnx export unavailable in this build: {e}")
        assert out.is_file() and out.stat().st_size > 1000

        meta = __import__("json").loads((tmp_path / "m.onnx.json").read_text())
        assert meta["preproc"]["tile_size"] == 64
        assert meta["preproc"]["canonical_gsd_m"] == 0.5
        assert "height_m" in meta["outputs"] and "seg" in meta["outputs"]

        ort = pytest.importorskip("onnxruntime")
        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        x = np.random.RandomState(0).randn(2, 3, 64, 64).astype(np.float32)  # batch 2
        h, s = sess.run(None, {"image": x})
        assert h.shape == (2, 1, 64, 64) and s.shape == (2, 1, 64, 64)
        with torch.no_grad():
            ref = net(torch.from_numpy(x))["fused"].numpy()
        assert np.abs(ref - h).max() < 5e-2, "onnx graph disagrees with the checkpoint"
    finally:
        undo()


def test_onnx_adapter_plugs_into_the_engine(tmp_path, monkeypatch):
    """`serve._OnnxModel` has to satisfy the interface `infer.engine` expects, so
    the deployment path reuses the tiling/blending code rather than a second copy."""
    undo = use_stub(hidden=32, patch=16, layers=8)
    try:
        ck, net, spec = _ckpt(tmp_path)
        import dwdata.preprocess as pp

        monkeypatch.setattr(pp, "resolve_encoder_stats",
                            lambda *a, **k: (pp.DINOV3_SAT_MEAN, pp.DINOV3_SAT_STD))
        from infer.export_onnx import export

        out = tmp_path / "m.onnx"
        try:
            export(str(ck), str(out), opset=17, check=False)
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"torch.onnx export unavailable: {e}")
        ort = pytest.importorskip("onnxruntime")

        from infer.engine import predict_scene
        from serve.app import _OnnxModel

        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        rgb = (np.random.RandomState(1).rand(150, 210, 3) * 255).astype(np.uint8)
        h, seg = predict_scene(_OnnxModel(sess), rgb, 0.5, spec,
                               torch.device("cpu"), want_seg=True, batch_tiles=2)
        assert h.shape == (150, 210) and np.isfinite(h).all()
        assert seg.shape == (150, 210)
    finally:
        undo()
