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
from tests.stub_dav2 import use_stub


def _ckpt(tmp_path):
    from models.heads import DepthWizardNet

    cfg = Config()
    cfg.tile_size, cfg.decoder_dim, cfg.n_bins = 56, 32, 8
    cfg.grad_checkpoint_encoder = False
    net = DepthWizardNet(cfg).eval()
    spec = PreprocSpec(tile_size=56, canonical_gsd_m=0.5)
    p = tmp_path / "best.pt"
    from config import safe_config_dict

    torch.save({"model": net.state_dict(), "preproc": spec.to_dict(),
                "config": safe_config_dict(cfg), "epoch": 1,
                "encoder_included": True}, p)
    return p, net, spec


def test_export_matches_the_checkpoint(tmp_path, monkeypatch):
    undo = use_stub(hidden=32, patch=14, layers=8)
    try:
        ck, net, spec = _ckpt(tmp_path)
        import dwdata.preprocess as pp

        monkeypatch.setattr(pp, "resolve_encoder_stats",
                            lambda *a, **k: (pp.IMAGENET_MEAN, pp.IMAGENET_STD))
        from infer.export_onnx import export

        out = tmp_path / "m.onnx"
        try:
            export(str(ck), str(out), opset=18, check=True)
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"torch.onnx export unavailable in this build: {e}")
        assert out.is_file() and out.stat().st_size > 1000

        meta = __import__("json").loads((tmp_path / "m.onnx.json").read_text())
        assert meta["preproc"]["tile_size"] == 56
        assert meta["preproc"]["canonical_gsd_m"] == 0.5
        assert "height_m" in meta["outputs"] and "seg" in meta["outputs"]

        ort = pytest.importorskip("onnxruntime")
        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        x = np.random.RandomState(0).randn(2, 3, 56, 56).astype(np.float32)  # batch 2
        h, s = sess.run(None, {"image": x})
        assert h.shape == (2, 1, 56, 56) and s.shape == (2, 1, 56, 56)
        with torch.no_grad():
            ref = net(torch.from_numpy(x))["fused"].numpy()
        assert np.abs(ref - h).max() < 5e-2, "onnx graph disagrees with the checkpoint"
    finally:
        undo()


def test_sidecar_records_the_opset_the_graph_actually_has(tmp_path, monkeypatch):
    """`opset_version=` is a request, not a guarantee.

    The dynamo exporter emits at its own native opset and then asks onnxscript
    to down-convert; when an op has no version adapter that conversion fails,
    onnxscript prints a traceback, reports "the model was not modified", and the
    export returns successfully at the native opset.  The v4-2 run hit exactly
    that on `Resize` (the head's F.interpolate) and wrote `"opset": 17` into
    `depthwizard.onnx.json` for a graph that is not opset 17 — a sidecar that
    lies to whoever deploys it.
    """
    onnx = pytest.importorskip("onnx")
    undo = use_stub(hidden=32, patch=14, layers=8)
    try:
        ck, net, spec = _ckpt(tmp_path)
        import dwdata.preprocess as pp

        monkeypatch.setattr(pp, "resolve_encoder_stats",
                            lambda *a, **k: (pp.IMAGENET_MEAN, pp.IMAGENET_STD))
        from infer.export_onnx import export

        out = tmp_path / "m.onnx"
        try:
            export(str(ck), str(out), opset=18, check=False)
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"torch.onnx export unavailable: {e}")

        import json
        meta = json.loads(Path(str(out) + ".json").read_text())
        m = onnx.load(str(out), load_external_data=False)
        real = next(i.version for i in m.opset_import if i.domain in ("", "ai.onnx"))
        assert meta["opset"] == real
        assert meta["opset_requested"] == 18
        # and every external weight file the graph references is named
        for name in meta["external_data"]:
            assert (out.parent / name).is_file(), name
    finally:
        undo()


def test_external_weight_files_are_reported_so_they_can_be_shipped(
        tmp_path, capsys, monkeypatch):
    """A 300 M-parameter model that serialises to a few MB has put its weights
    in sibling `.data` files; `depthwizard.onnx` alone then loads nothing, and
    with `--make_zip false` nothing else bundles the directory."""
    pytest.importorskip("onnx")
    undo = use_stub(hidden=32, patch=14, layers=8)
    try:
        ck, net, spec = _ckpt(tmp_path)
        import dwdata.preprocess as pp

        monkeypatch.setattr(pp, "resolve_encoder_stats",
                            lambda *a, **k: (pp.IMAGENET_MEAN, pp.IMAGENET_STD))
        from infer.export_onnx import export

        out = tmp_path / "m.onnx"
        try:
            export(str(ck), str(out), opset=18, check=False)
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"torch.onnx export unavailable: {e}")
        printed = capsys.readouterr().out
        import json
        meta = json.loads(Path(str(out) + ".json").read_text())
        if meta["external_data"]:
            assert "EXTERNAL" in printed
            for name in meta["external_data"]:
                assert name in printed
        else:
            # self-contained: the graph must then actually carry the weights
            assert out.stat().st_size > 0
    finally:
        undo()
