"""`last_full.pt` must be a real resume, not a warm start.

v4's `--resume` loads `ck["model"]` with strict=False and nothing else, and its
checkpoints carry EMA-*merged* weights — so a session that died at hour 8
restarted the LR cosine from the top with fresh AdamW moments.  This test is the
gate on the replacement: the artefact has to carry the optimiser, scaler, EMA,
teacher, epoch, elapsed time, `best` and `history`, and a second run pointed at
it has to pick all of that up instead of starting over.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from config import Config
from tests.stub_encoder import use_stub


def _tiny_cfg(tmp_path, data_root, out_name="out"):
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
    c.make_figures = False
    c.make_report = False
    c.export_onnx = False
    c.final_sliding_eval = False
    c.n_qualitative = 1
    c.data_root = str(data_root)
    c.output_dir = str(tmp_path / out_name)
    c.datasets = "gamus"
    c.eval_every = 1
    c.ema_decay = 0.99
    c.save_full_state = True
    return c.validate()


def _run(cfg, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py"])
    import config as config_mod
    import dwdata.preprocess as pp
    import train as train_mod
    monkeypatch.setattr(config_mod, "parse_config", lambda *a, **k: cfg)
    monkeypatch.setattr(train_mod, "parse_config", lambda *a, **k: cfg)
    monkeypatch.setattr(pp, "resolve_encoder_stats",
                        lambda *a, **k: (pp.DINOV3_SAT_MEAN, pp.DINOV3_SAT_STD))
    train_mod.main()


def test_last_full_roundtrips_the_whole_training_state(tmp_path, store, monkeypatch):
    undo = use_stub(hidden=32, patch=16, layers=8)
    try:
        data_root = Path(store.dir).parents[1]
        cfg = _tiny_cfg(tmp_path, data_root)
        _run(cfg, monkeypatch)

        full = Path(cfg.output_dir) / "last_full.pt"
        assert full.is_file(), "--save_full_state wrote no last_full.pt"
        ck = torch.load(full, map_location="cpu", weights_only=False)

        # Everything a resume needs, and the things `last.pt` deliberately lacks.
        for key in ("model", "opt", "scaler", "ema", "ema_n", "epoch",
                    "elapsed_min", "best", "history", "encoder_frozen", "rng"):
            assert key in ck, f"last_full.pt is missing {key!r}"
        assert ck["epoch"] == cfg.epochs
        assert ck["encoder_frozen"] is False, "saved before the unfreeze?"
        assert np.isfinite(ck["best"])
        assert len(ck["history"]) == cfg.epochs
        assert ck["elapsed_min"] >= 0.0
        # LIVE weights, not the EMA-merged ones `last.pt` carries.
        last = torch.load(Path(cfg.output_dir) / "last.pt", map_location="cpu",
                          weights_only=False)
        shared = [k for k in ck["model"] if k in last["model"]
                  and ck["model"][k].dtype.is_floating_point]
        assert shared, "no comparable tensors between last.pt and last_full.pt"
        assert any(not torch.allclose(ck["model"][k], last["model"][k])
                   for k in shared), \
            "last_full.pt looks EMA-merged — it must carry the live weights"

        # ---- second session: resume, don't restart ----------------------
        cfg2 = _tiny_cfg(tmp_path, data_root, out_name="out2")
        cfg2.epochs = 3                    # one more epoch to actually run
        cfg2.resume = str(full)
        _run(cfg2, monkeypatch)

        full2 = Path(cfg2.output_dir) / "last_full.pt"
        ck2 = torch.load(full2, map_location="cpu", weights_only=False)
        assert ck2["epoch"] == 3, "resumed run did not continue the epoch count"
        # `history` came back and was appended to rather than started over.
        assert len(ck2["history"]) == 3
        assert [h["epoch"] for h in ck2["history"]] == [1, 2, 3]
        # The clock was back-dated, which is what keeps the cosine going.
        assert ck2["elapsed_min"] >= ck["elapsed_min"]
        # Optimiser state survived: the param-group structure has to match, and
        # AdamW's step counters must be non-zero rather than freshly built.
        assert len(ck2["opt"]["param_groups"]) == len(ck["opt"]["param_groups"])
        steps = [s.get("step") for s in ck2["opt"]["state"].values()]
        assert steps and all(float(s) > 0 for s in steps if s is not None)
    finally:
        undo()


def test_exports_only_run_does_not_erase_the_training_history(
        tmp_path, store, monkeypatch):
    """`run_kaggle.sh finalize` launches with --epochs 0.

    No epoch runs, so `history` starts empty and `best` starts at +inf — and the
    end-of-run `write_metrics_json` would then overwrite the committed
    metrics.json with an empty history, taking the training curves the figures
    are drawn from with it.  The recovery path reads both back off disk.
    """
    undo = use_stub(hidden=32, patch=16, layers=8)
    try:
        data_root = Path(store.dir).parents[1]
        cfg = _tiny_cfg(tmp_path, data_root, out_name="out_fin")
        _run(cfg, monkeypatch)

        mpath = Path(cfg.output_dir) / "metrics.json"
        before = json.loads(mpath.read_text())
        assert len(before["history"]) == cfg.epochs
        assert np.isfinite(before["best_val_rmse_m"])

        # Same output_dir, exports only.
        cfg2 = _tiny_cfg(tmp_path, data_root, out_name="out_fin")
        cfg2.epochs = 0
        cfg2.freeze_epochs = 0          # validate(): 0 <= freeze_epochs <= epochs
        cfg2.save_full_state = False
        _run(cfg2, monkeypatch)

        after = json.loads(mpath.read_text())
        assert [h["epoch"] for h in after["history"]] == \
               [h["epoch"] for h in before["history"]], \
            "the exports-only run wiped the training history"
        assert after["best_val_rmse_m"] == before["best_val_rmse_m"]
        # ...and it still did its actual job.
        assert "final_plain" in after
    finally:
        undo()


def test_save_full_state_is_off_by_default(tmp_path, store, monkeypatch):
    """The v4 behaviour is the default: no 6.4 GB artefact unless asked for."""
    undo = use_stub(hidden=32, patch=16, layers=8)
    try:
        data_root = Path(store.dir).parents[1]
        cfg = _tiny_cfg(tmp_path, data_root, out_name="out_off")
        cfg.save_full_state = False
        _run(cfg, monkeypatch)
        assert (Path(cfg.output_dir) / "last.pt").is_file()
        assert not (Path(cfg.output_dir) / "last_full.pt").exists()
    finally:
        undo()
