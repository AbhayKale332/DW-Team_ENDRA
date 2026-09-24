"""`eval_test.py` scores a frozen checkpoint on a held-out store.

Runs the real entry point against a synthetic gamus/test store with the stub
encoder, so the wiring — store discovery, arch restore from config.json, the
strict-missing guard, plain/TTA/sliding — is exercised without a GPU or the
gated hub.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from config import Config
from dwdata.packed import ShardWriter
from tests.stub_encoder import use_stub


def _tiny_cfg() -> Config:
    c = Config()
    c.smoke = True
    c.apply_smoke()
    c.tile_size = 64
    c.decoder_dim = 32
    c.n_bins = 8
    c.num_workers = 0
    c.amp = False
    c.channels_last = False
    c.tta_scales = (1.0,)
    return c.validate()


@pytest.fixture
def test_store(tmp_path):
    d = tmp_path / "data" / "gamus" / "test"
    w = ShardWriter(d, tile_px=128, gsd_m=0.5, shard_tiles=2)
    rng = np.random.default_rng(1)
    for i in range(4):
        rgb = (rng.random((128, 128, 3)) * 60 + 90).astype(np.uint8)
        h = np.zeros((128, 128), np.float32)
        h[32:80, 32:80] = 18.0
        rgb[32:80, 32:80] = 220
        w.add(f"t{i}", rgb, h, np.where(h > 1, 2, 0).astype(np.uint8),
              np.ones((128, 128), bool))
    w.finalise()
    return tmp_path


def _write_ckpt_and_config(root: Path, cfg: Config) -> tuple[Path, Path]:
    from config import safe_config_dict
    from models.heads import DepthWizardNetV3

    undo = use_stub(hidden=32, patch=16, layers=8)
    try:
        model = DepthWizardNetV3(cfg)
        ck = root / "best.pt"
        torch.save({"model": model.state_dict(), "epoch": 7}, ck)
    finally:
        undo()
    conf = root / "config.json"
    conf.write_text(json.dumps(safe_config_dict(cfg), indent=2))
    return ck, conf


def _run(argv: list[str], monkeypatch):
    import eval_test

    undo = use_stub(hidden=32, patch=16, layers=8)
    monkeypatch.setattr(sys, "argv", ["eval_test.py", *argv])
    try:
        eval_test.main()
    finally:
        undo()


def test_scores_a_held_out_store(test_store, monkeypatch, capsys):
    cfg = _tiny_cfg()
    ck, conf = _write_ckpt_and_config(test_store, cfg)
    out = test_store / "out"
    _run(["--ckpt", str(ck), "--run_config", str(conf),
          "--data_root", str(test_store / "data"), "--source", "gamus",
          "--split", "test", "--sliding_tiles", "2", "--num_workers", "0",
          "--batch_size", "2", "--tta_scales", "1.0,1.5",
          "--out", str(out)], monkeypatch)

    m = json.loads((out / "test_metrics.json").read_text())
    # The override wins over the (1.0,) the run config carries.  (1.5, not
    # v4's 1.25: on this 64 px tile 1.25 is a 5-token grid, which v3's DPT
    # cannot fuse; the real 512 px tile goes to 640 px = 40 tokens.)
    assert m["config"]["tta_scales"] == [1.0, 1.5]
    for k in ("test_gamus_test_plain", "test_gamus_test_tta",
              "test_gamus_test_sliding_tta"):
        assert m[k]["global"]["n"] > 0, k
        assert np.isfinite(m[k]["global"]["rmse_m"])
        # Every breakdown the deck quotes must survive the trip to JSON.
        assert m[k]["balanced_rmse_m"] is not None
        assert "bias_m" in m[k]["tall_gt15m"]
        # ...and the landscape table v4/DAV2's metrics.json carry.
        assert sum(v["tiles"] for v in m[k]["per_landscape"].values()) > 0, k
    assert m["tiles_scored"] == 4
    # The store was written at 128 px; the centre-crop path uses cfg.tile_size.
    assert m["config"]["tile_size"] == 64
    assert "sliding window" in capsys.readouterr().out


def test_refuses_a_checkpoint_that_does_not_fit(test_store, monkeypatch):
    """A wrong --run_config silently half-loads under strict=False.

    That is the failure this guard exists for: the score would come out of a
    network with randomly initialised decoder blocks and look merely bad.
    """
    cfg = _tiny_cfg()
    ck, conf = _write_ckpt_and_config(test_store, cfg)
    wrong = json.loads(conf.read_text())
    wrong["decoder_dim"] = 64
    (test_store / "wrong.json").write_text(json.dumps(wrong))
    with pytest.raises(SystemExit, match="does not fit"):
        _run(["--ckpt", str(ck), "--run_config", str(test_store / "wrong.json"),
              "--data_root", str(test_store / "data"), "--sliding_tiles", "0",
              "--num_workers", "0", "--batch_size", "2",
              "--out", str(test_store / "out2")], monkeypatch)


def test_fit_keys_bridges_the_dinov3_layout_change():
    """The v3 best.pt (transformers 5.x) saved blocks as encoder.model.model.layer.*;
    Kaggle's 4.x builds encoder.model.layer.*, and all 408 came back missing."""
    from eval_test import fit_keys

    old = {"encoder.model.embeddings.cls_token": 0,
           "encoder.model.model.layer.0.norm1.weight": 1,
           "encoder.model.norm.weight": 2, "head.w": 3}
    new = {"encoder.model.embeddings.cls_token", "encoder.model.layer.0.norm1.weight",
           "encoder.model.norm.weight", "head.w"}
    sd, n = fit_keys(old, new)
    assert (n, set(sd), sd["encoder.model.layer.0.norm1.weight"]) == (1, new, 1)
    # ...and back, for a checkpoint saved under the newer layout.
    sd, n = fit_keys({"encoder.model.layer.3.mlp.w": 5},
                     {"encoder.model.model.layer.3.mlp.w"})
    assert (n, sd) == (1, {"encoder.model.model.layer.3.mlp.w": 5})
    # A name the model does not have is left alone for the missing-key guard.
    sd, n = fit_keys({"encoder.model.model.layer.0.x": 1}, {"decoder.y"})
    assert (n, sd) == (0, {"encoder.model.model.layer.0.x": 1})


def test_replicated_splits_the_batch_and_keeps_autocast():
    """Two-GPU eval must score exactly what one GPU does.  Autocast and grad
    mode are thread-local, so the worker threads have to re-enter both — a
    replica silently running fp32 (or building a graph) is the failure."""
    from eval_test import Replicated

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 2, 1)
            self.seen = []

        def forward(self, x):
            self.seen.append((torch.get_autocast_dtype("cpu"),
                              torch.is_autocast_enabled("cpu"), torch.is_grad_enabled()))
            y = self.conv(x)
            return {"fused": y[:, :1], "seg": y, "b_logits": y}

    net = Net().eval()
    rep = Replicated(net, [torch.device("cpu")] * 2)
    x = torch.randn(5, 3, 8, 8)
    with torch.no_grad():
        ref = net(x)
        out = rep(x)
        assert set(out) == {"fused", "seg"}      # b_logits never comes back
        for k in out:
            assert torch.allclose(out[k], ref[k]), k
        assert rep(x[:1])["fused"].shape[0] == 1  # batch smaller than devices
        with torch.autocast("cpu", dtype=torch.bfloat16):
            rep(x)
    seen = rep.replicas[0].seen + rep.replicas[1].seen
    assert (torch.bfloat16, True, False) in seen
    assert all(not grad for *_, grad in seen)
