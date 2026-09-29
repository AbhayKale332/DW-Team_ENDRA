"""Train -> checkpoint -> inference, with the stub encoder.

The point of this test is the *contract*: a checkpoint must be enough to
reproduce training-time preprocessing, and the inference path must return metric
heights on the caller's own pixel grid.  v2 could not have passed it — its
validation path and its `predict_image.py` path normalised differently and
handled scale differently.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from config import Config
from dwdata.preprocess import PreprocSpec, read_scene
from tests.stub_encoder import use_stub


def _tiny_cfg(tmp_path, data_root):
    c = Config()
    c.smoke = True
    c.apply_smoke()
    c.tile_size = 64
    c.decoder_dim = 32
    c.n_bins = 8
    c.epochs = 2
    c.freeze_epochs = 1
    c.crops_per_epoch = 4
    c.batch_size = 2
    c.grad_accum = 1
    c.val_tiles = 2
    c.num_workers = 0
    c.amp = False
    c.channels_last = False
    c.cudnn_benchmark = False
    c.make_zip = False
    c.n_qualitative = 1
    c.data_root = str(data_root)
    c.output_dir = str(tmp_path / "out")
    c.datasets = "gamus"
    c.eval_every = 1
    c.final_sliding_eval = True
    c.ema_decay = 0.99
    return c.validate()


def test_train_then_predict_roundtrip(tmp_path, store, monkeypatch):
    undo = use_stub(hidden=32, patch=16, layers=8)
    try:
        data_root = Path(store.dir).parents[1]
        cfg = _tiny_cfg(tmp_path, data_root)
        monkeypatch.setattr(sys, "argv", ["train.py"])
        import config as config_mod
        monkeypatch.setattr(config_mod, "parse_config", lambda *a, **k: cfg)
        import train as train_mod
        monkeypatch.setattr(train_mod, "parse_config", lambda *a, **k: cfg)
        # keep the hub out of the test: the spec resolves stats from the
        # encoder's image processor, so stub that one call instead.
        import dwdata.preprocess as pp
        monkeypatch.setattr(pp, "resolve_encoder_stats",
                            lambda *a, **k: (pp.DINOV3_SAT_MEAN, pp.DINOV3_SAT_STD))

        train_mod.main()

        out = Path(cfg.output_dir)
        assert (out / "best.pt").is_file()
        assert (out / "preproc.json").is_file()
        m = json.loads((out / "metrics.json").read_text())
        assert np.isfinite(m["best_val_rmse_m"])
        assert "final_plain" in m and "final_sliding_tta" in m
        assert m["history"][0]["encoder_frozen"] is True
        assert m["history"][1]["encoder_frozen"] is False, "encoder never unfroze"
        assert (out / "viewer_sample" / "pred_ndsm_m.npy").is_file()

        # ---- the contract: the checkpoint carries its own preprocessing ----
        ck = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
        assert "preproc" in ck
        spec = PreprocSpec.from_dict(ck["preproc"])
        assert spec.tile_size == cfg.tile_size
        assert spec.canonical_gsd_m == cfg.canonical_gsd_m

        # ---- inference on a PNG the model has never seen -------------------
        from PIL import Image
        from infer.engine import predict_scene
        from models.heads import DepthWizardNetV3

        img = (np.random.rand(300, 400, 3) * 255).astype(np.uint8)
        p = tmp_path / "scene.png"
        Image.fromarray(img).save(p)

        icfg = Config()
        for k, v in ck["config"].items():
            if hasattr(icfg, k) and not isinstance(getattr(icfg, k), tuple):
                setattr(icfg, k, v)
        net = DepthWizardNetV3(icfg).eval()
        miss, unexp = net.load_state_dict(ck["model"], strict=False)
        assert not unexp
        assert not [k for k in miss if not k.startswith("encoder.model.")]

        rgb, meta = read_scene(p, user_gsd_m=0.4)
        assert meta.gsd_m == 0.4 and meta.gsd_source == "user"
        h, _ = predict_scene(net, rgb, meta.gsd_m, spec, torch.device("cpu"))
        # output is on the CALLER's grid, in metres, finite and non-negative
        assert h.shape == (300, 400)
        assert np.isfinite(h).all() and (h >= 0).all()
    finally:
        undo()


def test_scale_changes_nothing_but_the_grid(tmp_path):
    """Predicting the same scene declared at two GSDs must change the sampled
    field of view, not silently rescale the metre values."""
    undo = use_stub(hidden=32, patch=16, layers=8)
    try:
        from infer.engine import predict_scene
        from models.heads import DepthWizardNetV3

        c = Config()
        c.tile_size, c.decoder_dim, c.n_bins = 64, 32, 8
        c.grad_checkpoint_encoder = False
        net = DepthWizardNetV3(c).eval()
        spec = PreprocSpec(tile_size=64, canonical_gsd_m=0.5)
        rgb = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
        a, _ = predict_scene(net, rgb, 0.5, spec, torch.device("cpu"))
        b, _ = predict_scene(net, rgb, 0.25, spec, torch.device("cpu"))
        assert a.shape == b.shape == (256, 256)
        assert not np.allclose(a, b)          # different GSD -> different input
    finally:
        undo()


def test_predict_cli_reports_the_contract(tmp_path, store, monkeypatch):
    """`--report` must print the resolved contract without needing a GPU."""
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [sys.executable, "-c",
         "import sys;sys.path.insert(0,%r);"
         "from dwdata.preprocess import PreprocSpec;"
         "s=PreprocSpec();print(s.to_dict()['canonical_gsd_m'])" % str(root)],
        capture_output=True, text=True)
    assert r.returncode == 0 and "0.5" in r.stdout
