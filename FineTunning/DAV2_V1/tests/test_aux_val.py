"""Secondary val sets: scored and recorded, never selected on.

GAMUS is US *aerial* imagery, so the 400-tile prefix every run since v1 is scored
on cannot answer how the model does on satellite imagery it was not trained to
fit — and "v4 beat v4-2 on GAMUS val" is partly just v4 having trained on nothing
but the val domain.  DFC23 answers it.  But the primary has to stay the
selection criterion, or the only run-to-run comparison the project has is gone,
so this pins both halves: the secondary number is produced and recorded, and
`best.pt` still follows the primary even when the secondary is lower.
"""
import sys, json, pathlib
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from config import Config
from dwdata.packed import ShardWriter
from tests.stub_dav2 import use_stub


def _mk(root, name, split, n, tile, gsd):
    w = ShardWriter(root / name / split, tile_px=tile, gsd_m=gsd, shard_tiles=64,
                    has_seg=False)
    rng = np.random.default_rng(abs(hash(name + split)) % 2**31)
    for i in range(n):
        h = np.zeros((tile, tile), "float32"); h[8:40, 8:40] = 9.0
        w.add(f"{name}_{split}_{i}", (rng.random((tile, tile, 3)) * 255).astype("uint8"),
              h, None, np.ones((tile, tile), bool))
    w.finalise()


def test_secondary_val_is_scored_and_recorded_but_not_selected_on(tmp_path, monkeypatch):
    root = tmp_path / "data"
    _mk(root, "gamus", "train", 6, 64, 0.33); _mk(root, "gamus", "val", 4, 64, 0.33)
    _mk(root, "dfc23_g050", "train", 6, 64, 0.5); _mk(root, "dfc23_g050", "val", 4, 64, 0.5)

    cfg = Config(); cfg.smoke = True; cfg.apply_smoke()
    cfg.data_root = str(root); cfg.datasets = "gamus,dfc23_g050"
    cfg.tile_size = 56; cfg.decoder_dim = 32; cfg.n_bins = 8
    cfg.epochs = 1; cfg.freeze_epochs = 0; cfg.crops_per_epoch = 4
    cfg.batch_size = 2; cfg.num_workers = 0; cfg.val_tiles = 4
    cfg.output_dir = str(tmp_path / "out"); cfg.export_onnx = False
    cfg.make_figures = cfg.make_report = False; cfg.final_sliding_eval = False
    cfg.n_qualitative = 0; cfg.w_seg = 0.0

    undo = use_stub(hidden=32, patch=14, layers=4)
    try:
        monkeypatch.setattr(sys, "argv", ["train.py"])
        import config as config_mod, dwdata.preprocess as pp, train as train_mod
        monkeypatch.setattr(config_mod, "parse_config", lambda *a, **k: cfg)
        monkeypatch.setattr(train_mod, "parse_config", lambda *a, **k: cfg)
        monkeypatch.setattr(pp, "resolve_encoder_stats",
                            lambda *a, **k: (pp.IMAGENET_MEAN, pp.IMAGENET_STD))
        train_mod.main()
    finally:
        undo()

    m = json.loads((pathlib.Path(cfg.output_dir) / "metrics.json").read_text())
    rec = m["history"][-1]
    assert "val" in rec, "primary val missing"
    assert "val_dfc23_g050" in rec, f"secondary val missing: {sorted(rec)}"
    assert rec["val_dfc23_g050"]["global"]["rmse_m"] > 0
    # best.pt tracks the PRIMARY number, not the secondary one
    assert m["best_val_rmse_m"] == pytest.approx(rec["val"]["global"]["rmse_m"])
