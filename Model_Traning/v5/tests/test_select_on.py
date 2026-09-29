"""`select_on`: best.pt on forested + sparse RMSE instead of global RMSE."""

import math
import sys

import pytest

from config import Config


def _m(forest=None, sparse=None, urban=None, g=1.0):
    pl = {}
    for nm, v in (("forested", forest), ("sparse", sparse), ("urban", urban)):
        if v is not None:
            pl[nm] = {"n": v[1], "rmse_m": v[0]}
    return {"global": {"rmse_m": g}, "per_landscape": pl}


def test_primary_pools_forest_and_sparse_by_pixels():
    from train import selection_score

    rec = {"val": _m(forest=(2.0, 300), sparse=(1.0, 100), urban=(9.0, 1000))}
    want = math.sqrt((4.0 * 300 + 1.0 * 100) / 400)
    assert selection_score(rec, "mvs3dm", ["mvs3dm"]) == pytest.approx(want)


def test_sources_are_averaged_and_pooled_line_preferred():
    from train import selection_score

    rec = {"val": _m(forest=(2.0, 100)),
           "val_neon": _m(forest=(9.0, 100)),            # full-res: charged for 1 m label blur
           "val_neon_pooled2": _m(forest=(4.0, 100))}
    assert selection_score(rec, "mvs3dm", ["neon", "mvs3dm"]) == pytest.approx(3.0)


def test_none_without_forest_or_sparse_tiles():
    from train import selection_score

    rec = {"val": _m(urban=(3.0, 10))}
    assert selection_score(rec, "mvs3dm", ["mvs3dm", "neon"]) is None


def test_config_helpers():
    c = Config()
    assert c.select_sources() == []
    c.select_on = " neon, mvs3dm ,"
    assert c.select_sources() == ["neon", "mvs3dm"]


def test_bench_mode_prints_one_report_and_saves_nothing(tmp_path, store, monkeypatch, capsys):
    """`bench_steps` runs the real train loop, prints `[bench] {json}`, exits."""
    import json
    from pathlib import Path

    from tests.stub_encoder import use_stub
    from tests.test_end_to_end import _tiny_cfg

    undo = use_stub(hidden=32, patch=16, layers=8)
    try:
        cfg = _tiny_cfg(tmp_path, Path(store.dir).parents[1])
        cfg.freeze_epochs = 0
        cfg.bench_steps = 3
        cfg.crops_per_epoch = 2 * 16
        cfg.landscape_sampler_boost = "forested:2,sparse:1.5"
        cfg.select_on = "gamus"
        cfg.opt_fused = True                 # CPU: must fall back, not crash
        monkeypatch.setattr(sys, "argv", ["train.py"])
        import train as train_mod
        monkeypatch.setattr(train_mod, "parse_config", lambda *a, **k: cfg)
        import dwdata.preprocess as pp
        monkeypatch.setattr(pp, "resolve_encoder_stats",
                            lambda *a, **k: (pp.DINOV3_SAT_MEAN, pp.DINOV3_SAT_STD))
        train_mod.main()
    finally:
        undo()
    # train.py tees its stdout into <output_dir>/run.log
    out = capsys.readouterr().out + (Path(cfg.output_dir) / "run.log").read_text()
    lines = [ln for ln in out.splitlines() if ln.startswith("[bench] {")]
    assert len(lines) == 1, out[-2000:]
    rep = json.loads(lines[0][len("[bench] "):])
    assert rep["img_s"] > 0 and 0.0 <= rep["wait_frac"] <= 1.0
    assert rep["batch_size"] == cfg.batch_size
    assert not (Path(cfg.output_dir) / "best.pt").exists()
