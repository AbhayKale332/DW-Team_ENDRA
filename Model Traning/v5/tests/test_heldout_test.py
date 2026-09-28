"""The held-out test split: scored once at the end, never trained or selected on.

`best.pt` is picked on the GAMUS val prefix and `final_plain`/`final_tta` are
then reported on those same tiles, so the headline number is measured on the set
that chose the checkpoint.  `--test_sources gamus:test` is the fix.  This pins
the three things that make it a test set rather than a third val set: the tiles
never enter an epoch, the number lands in metrics.json under `test_*`, and
`best_val_rmse_m` still follows the primary val set.
"""
import sys, json, pathlib
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from config import Config
from dwdata.packed import ShardWriter
from tests.stub_encoder import use_stub

from tests.test_aux_val import _mk


def _cfg(tmp_path, root):
    cfg = Config(); cfg.smoke = True; cfg.apply_smoke()
    cfg.data_root = str(root); cfg.datasets = "gamus"
    cfg.tile_size = 64; cfg.decoder_dim = 32; cfg.n_bins = 8
    cfg.epochs = 1; cfg.freeze_epochs = 0; cfg.crops_per_epoch = 4
    cfg.batch_size = 2; cfg.num_workers = 0; cfg.val_tiles = 4
    cfg.output_dir = str(tmp_path / "out"); cfg.export_onnx = False
    cfg.make_figures = cfg.make_report = False; cfg.final_sliding_eval = False
    cfg.n_qualitative = 0; cfg.w_seg = 0.0
    return cfg


def _run(cfg, monkeypatch):
    undo = use_stub(hidden=32, patch=16, layers=4)
    try:
        monkeypatch.setattr(sys, "argv", ["train.py"])
        import config as config_mod, dwdata.preprocess as pp, train as train_mod
        monkeypatch.setattr(config_mod, "parse_config", lambda *a, **k: cfg)
        monkeypatch.setattr(train_mod, "parse_config", lambda *a, **k: cfg)
        monkeypatch.setattr(pp, "resolve_encoder_stats",
                            lambda *a, **k: (pp.DINOV3_SAT_MEAN, pp.DINOV3_SAT_STD))
        train_mod.main()
    finally:
        undo()


def test_test_split_is_scored_at_the_end_and_not_selected_on(tmp_path, monkeypatch):
    root = tmp_path / "data"
    _mk(root, "gamus", "train", 6, 64, 0.33)
    _mk(root, "gamus", "val", 4, 64, 0.33)
    _mk(root, "gamus", "test", 4, 64, 0.33)

    cfg = _cfg(tmp_path, root)
    cfg.test_sources = "gamus:test"; cfg.test_tiles = 4
    _run(cfg, monkeypatch)

    m = json.loads((pathlib.Path(cfg.output_dir) / "metrics.json").read_text())
    assert m.get("test_gamus_test_plain"), f"no test result: {sorted(m)}"
    assert m["test_gamus_test_plain"]["global"]["rmse_m"] > 0
    # Reported at the END, once — never per epoch, never in the history rows
    # the best.pt decision reads.
    for rec in m["history"]:
        assert not [k for k in rec if k.startswith("test")], \
            f"test metrics leaked into the per-epoch history: {sorted(rec)}"
    assert m["best_val_rmse_m"] == pytest.approx(
        m["history"][-1]["val"]["global"]["rmse_m"])


def test_test_source_that_the_run_also_uses_is_rejected(tmp_path):
    """A 'held-out' split the run trains or validates on must not parse."""
    root = tmp_path / "data"
    _mk(root, "gamus", "train", 2, 64, 0.33)
    _mk(root, "gamus", "val", 2, 64, 0.33)
    import config as config_mod
    for bad in ("gamus:train", "gamus:val", "gamus"):
        cfg = Config(); cfg.data_root = str(root); cfg.datasets = "gamus"
        cfg.test_sources = bad
        if bad == "gamus":            # bare name means gamus:test — legal
            cfg.validate()
            continue
        with pytest.raises(AssertionError):
            cfg.validate()


def test_missing_test_store_is_loud(tmp_path):
    """A val store that vanishes costs a number; a test store that vanishes
    silently restores the very headline it was meant to replace."""
    from dwdata.loaders import build_test_loaders
    from dwdata.preprocess import PreprocSpec
    root = tmp_path / "data"
    _mk(root, "gamus", "train", 2, 64, 0.33)
    cfg = Config(); cfg.data_root = str(root); cfg.test_sources = "gamus:test"
    cfg.datasets = "synrs3d_g1"       # keeps validate() out of the way
    spec = PreprocSpec()
    with pytest.raises(RuntimeError, match="no store exists"):
        build_test_loaders(cfg, spec)
