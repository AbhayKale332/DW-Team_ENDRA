"""`last_full.pt` must be a real resume, not a warm start.

v4's `--resume` loads `ck["model"]` with strict=False and nothing else, and its
checkpoints carry EMA-*merged* weights — so a session that died at hour 8
restarted the LR cosine from the top with fresh AdamW moments.  This test is the
gate on the replacement: the artefact has to carry the optimiser, scaler, EMA,
teacher, epoch, elapsed time, `best` and `history`, and a second run pointed at
it has to pick all of that up instead of starting over.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from config import Config
from tests.stub_dav2 import use_stub


def _tiny_cfg(tmp_path, data_root, out_name="out"):
    c = Config()
    c.smoke = True
    c.apply_smoke()
    c.tile_size = 56
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
                        lambda *a, **k: (pp.IMAGENET_MEAN, pp.IMAGENET_STD))
    train_mod.main()


def test_last_full_roundtrips_the_whole_training_state(tmp_path, store, monkeypatch):
    undo = use_stub(hidden=32, patch=14, layers=8)
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
    undo = use_stub(hidden=32, patch=14, layers=8)
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
    undo = use_stub(hidden=32, patch=14, layers=8)
    try:
        data_root = Path(store.dir).parents[1]
        cfg = _tiny_cfg(tmp_path, data_root, out_name="out_off")
        cfg.save_full_state = False
        _run(cfg, monkeypatch)
        assert (Path(cfg.output_dir) / "last.pt").is_file()
        assert not (Path(cfg.output_dir) / "last_full.pt").exists()
    finally:
        undo()


# ---------------------------------------------------------------------
# The launch-flag combination that made the artefact necessary and then
# withheld it.
# ---------------------------------------------------------------------
def test_budget_longer_than_the_session_warns_when_it_cannot_be_resumed(capsys):
    """`--max_minutes 960 --session_minutes 480 --save_full_state false`.

    `max_minutes` is the budget the LR cosine is *sized* for (train.py drives
    progress off max(epoch fraction, elapsed/max_minutes)), not a safety cap.
    Setting it past the session cap with nothing written to continue from does
    not shorten the run — it truncates the anneal and throws the rest away.  The
    v4-2 run went out exactly like this: it stopped at 81 % of the cosine with
    the LR still at 5.09e-05, left no `last_full.pt`, and scored 3.804 m against
    the previous run's 3.441 m on the same val prefix.
    """
    from config import parse_config

    parse_config(["--max_minutes", "960", "--session_minutes", "480",
                  "--save_full_state", "false"])
    out = capsys.readouterr().out
    assert "[config] !!" in out
    assert "50 %" in out          # 480 of 960
    assert "unrecoverable" in out


def test_no_warning_when_the_budget_fits_or_the_run_can_be_resumed(capsys):
    from config import parse_config

    # fits in one session
    parse_config(["--max_minutes", "480", "--session_minutes", "480"])
    assert "[config] !!" not in capsys.readouterr().out
    # two sessions, but resumable — the documented profile
    parse_config(["--max_minutes", "960", "--session_minutes", "480",
                  "--save_full_state", "true"])
    assert "[config] !!" not in capsys.readouterr().out
    # this IS the second session
    parse_config(["--max_minutes", "960", "--session_minutes", "480",
                  "--resume", "/tmp/last_full.pt"])
    assert "[config] !!" not in capsys.readouterr().out
    # no session cap at all
    parse_config(["--max_minutes", "960"])
    assert "[config] !!" not in capsys.readouterr().out


def test_sampler_weight_falls_back_from_family_store_to_source(tmp_path):
    """`dfc23:2` has to cover `dfc23_g050`.

    Sources packed one store per GSD family arrive suffixed, but the thing
    anyone writes a weight for is the source.  A missed fallback here is silent:
    `loaders.py` just defaults the store to 1.0 and the mix ratio is quietly not
    what `--sampler_weights` says.
    """
    from config import Config

    c = Config()
    c.sampler_weights = "gamus:2,dfc23:3,synrs3d:1,synrs3d_g1:5"
    assert c.sampler_weight("gamus") == 2.0
    assert c.sampler_weight("dfc23_g050") == 3.0      # prefix fallback
    assert c.sampler_weight("dfc23_g080") == 3.0
    assert c.sampler_weight("synrs3d_g05") == 1.0     # prefix fallback
    assert c.sampler_weight("synrs3d_g1") == 5.0      # exact beats prefix
    assert c.sampler_weight("geonrw") == 1.0          # default


def test_val_split_lookup_is_prefix_aware(tmp_path):
    """A source with no `_VAL_SPLIT` entry does not get "no val set" — it gets a
    slice of the first *train* store scored as if it were one (loaders.py)."""
    from dwdata.loaders import val_split_of

    assert val_split_of("gamus") == "val"
    assert val_split_of("dfc23_g050") == "val"
    assert val_split_of("synrs3d_g05") is None
    assert val_split_of("synrs3d") is None
    assert val_split_of("nonesuch") is None



# ---------------------------------------------------------------------
# Numbered training phases.  Run 1 finished all 24 epochs with the cosine at
# its floor, and every obvious way to continue it was broken: a plain --resume
# trained nothing, raising --epochs spiked the LR ~40x with no warmup, and the
# fresh session's output_dir had no best.pt.  These are the gates on the fix.
# ---------------------------------------------------------------------
def _log(cfg) -> str:
    """run.log, not capsys: train.py's Tee writes to sys.__stdout__."""
    return (Path(cfg.output_dir) / "run.log").read_text()


def _full(cfg):
    return torch.load(Path(cfg.output_dir) / "last_full.pt", map_location="cpu",
                      weights_only=False)


def _floor_lr(ck, mult):
    return float(ck["base_lrs"][0]) * mult / 1e2


def test_phase_progress_and_lr_maths():
    from train import lr_scale, phase_progress

    # Phase 1 is the old formula exactly.
    for ep, st, n, epochs, el, mm in [(1, 0, 10, 24, 0.0, 960.0),
                                      (5, 3, 10, 24, 100.0, 960.0),
                                      (24, 9, 10, 24, 470.0, 480.0),
                                      (3, 0, 7, 24, 900.0, 480.0)]:
        old = max((ep - 1 + st / n) / epochs, el / mm)
        assert phase_progress(ep, st, n, 1, epochs, el, mm) == pytest.approx(old)

    # Phase 2 of run 1: e25..e40, 800 steps an epoch, 0.3x the 3e-4 peak.
    base, mult, warm = 3e-4, 0.3, 0.05
    p0 = phase_progress(25, 0, 800, 25, 40, 0.0, 480.0)
    assert p0 == 0.0
    assert base * mult * lr_scale(p0, warm) == pytest.approx(9e-7)
    # 5 % of 16 epochs is 0.8 of an epoch: step 640 of e25 is the peak.
    pw = phase_progress(25, 640, 800, 25, 40, 5.0, 480.0)
    assert pw == pytest.approx(warm)
    assert base * mult * lr_scale(pw, warm) == pytest.approx(9e-5)
    # ...and it falls from there, to the floor at the end of e40.
    mid = base * mult * lr_scale(
        phase_progress(30, 0, 800, 25, 40, 90.0, 480.0), warm)
    assert 9e-7 < mid < 9e-5
    end = phase_progress(40, 800, 800, 25, 40, 270.0, 480.0)
    assert end == pytest.approx(1.0)
    assert base * mult * lr_scale(end, warm) == pytest.approx(9e-7)
    # What run 1 would have done with --epochs 40 and no phase: a jump.
    spike = base * lr_scale(phase_progress(25, 0, 800, 1, 40, 0.0, 960.0), 0.05)
    assert spike > 30 * 3e-6


def test_phase_two_starts_a_fresh_warmup_instead_of_spiking(
        tmp_path, store, monkeypatch):
    undo = use_stub(hidden=32, patch=14, layers=8)
    try:
        data_root = Path(store.dir).parents[1]
        cfg = _tiny_cfg(tmp_path, data_root, out_name="p1")
        _run(cfg, monkeypatch)
        ck = _full(cfg)
        assert ck["phase"]["id"] == 1 and ck["phase"]["start_epoch"] == 1
        assert ck["phase"]["progress"] >= 1.0 - 1e-6, "phase 1 did not finish"
        assert all("lr_start" in h and "lr_end" in h for h in ck["history"])

        cfg2 = _tiny_cfg(tmp_path, data_root, out_name="p2")
        cfg2.epochs = 4
        cfg2.phase = 2
        cfg2.resume = str(Path(cfg.output_dir) / "last_full.pt")
        _run(cfg2, monkeypatch)
        ck2 = _full(cfg2)

        assert ck2["phase"]["id"] == 2
        assert ck2["phase"]["start_epoch"] == 3
        assert ck2["phase"]["lr_mult"] == cfg2.phase_lr_mult
        h = ck2["history"]
        assert [r["epoch"] for r in h] == [1, 2, 3, 4]
        # e3's first step is at the phase-2 floor, not wherever the old
        # cosine's epoch fraction would have put it.
        assert h[2]["phase"] == 2
        # (rel, because the phase clock has ticked a few ms by step 0.)
        assert h[2]["lr_start"] == pytest.approx(
            _floor_lr(ck, cfg2.phase_lr_mult), rel=0.1)
        assert "[phase] 2 starts at epoch 3 of 4" in _log(cfg2)
    finally:
        undo()


def test_rerunning_the_same_phase_continues_it(tmp_path, store, monkeypatch):
    """A crashed phase 2, re-run with the same command, must not re-warm."""
    undo = use_stub(hidden=32, patch=14, layers=8)
    try:
        data_root = Path(store.dir).parents[1]
        cfg = _tiny_cfg(tmp_path, data_root, out_name="c1")
        _run(cfg, monkeypatch)

        # Phase 2 over e3..e5, killed by the session cap one step into e3.
        cfg2 = _tiny_cfg(tmp_path, data_root, out_name="c2")
        cfg2.epochs = 5
        cfg2.phase = 2
        cfg2.session_minutes = 1e-9
        cfg2.resume = str(Path(cfg.output_dir) / "last_full.pt")
        _run(cfg2, monkeypatch)
        ck2 = _full(cfg2)
        assert ck2["epoch"] == 3 and ck2["phase"]["start_epoch"] == 3

        # Same phase, same epochs: a continuation.
        cfg3 = _tiny_cfg(tmp_path, data_root, out_name="c3")
        cfg3.epochs = 5
        cfg3.phase = 2
        cfg3.resume = str(Path(cfg2.output_dir) / "last_full.pt")
        _run(cfg3, monkeypatch)
        ck3 = _full(cfg3)
        assert ck3["phase"]["id"] == 2
        assert ck3["phase"]["start_epoch"] == 3, "phase 2 restarted"
        assert [r["epoch"] for r in ck3["history"]] == [1, 2, 3, 4, 5]
        # e4 is a third of the way down phase 2's cosine, well off the floor.
        assert ck3["history"][3]["lr_start"] > \
            10 * _floor_lr(ck2, cfg3.phase_lr_mult)
        assert "continuing phase 2" in _log(cfg3)
        assert "starts at epoch" not in _log(cfg3)

        # And going backwards is refused outright.
        cfg4 = _tiny_cfg(tmp_path, data_root, out_name="c4")
        cfg4.epochs = 6
        cfg4.phase = 1
        cfg4.resume = str(Path(cfg3.output_dir) / "last_full.pt")
        with pytest.raises(SystemExit, match="already in phase 2"):
            _run(cfg4, monkeypatch)
    finally:
        undo()


def test_extending_an_annealed_run_without_phase_auto_bumps(
        tmp_path, store, monkeypatch):
    """Run 1's exact situation: --epochs raised, --phase not passed.

    Covered twice: a new checkpoint (records its progress) and a run-1 style
    one (no "phase" block, detected off the optimiser's last LR instead).
    """
    undo = use_stub(hidden=32, patch=14, layers=8)
    try:
        data_root = Path(store.dir).parents[1]
        cfg = _tiny_cfg(tmp_path, data_root, out_name="a1")
        _run(cfg, monkeypatch)
        src = Path(cfg.output_dir) / "last_full.pt"

        # Run-1 style: no phase block, optimiser LR at the floor.
        legacy = torch.load(src, map_location="cpu", weights_only=False)
        del legacy["phase"]
        for g, b in zip(legacy["opt"]["param_groups"], legacy["base_lrs"]):
            g["lr"] = b / 1e2
        leg_dir = tmp_path / "legacy"
        leg_dir.mkdir()
        torch.save(legacy, leg_dir / "last_full.pt")

        for tag, resume in (("new", src), ("old", leg_dir / "last_full.pt")):
            c = _tiny_cfg(tmp_path, data_root, out_name=f"a2_{tag}")
            c.epochs = 3
            c.resume = str(resume)
            assert c.phase == 1
            _run(c, monkeypatch)
            ck = _full(c)
            log = _log(c)
            assert "[phase] !!" in log and "already annealed" in log, tag
            assert ck["phase"]["id"] == 2, tag
            assert ck["phase"]["start_epoch"] == 3, tag
            assert ck["history"][2]["lr_start"] == \
                pytest.approx(_floor_lr(legacy, c.phase_lr_mult), rel=0.1), tag
    finally:
        undo()


def test_resume_into_a_fresh_dir_carries_best_pt(tmp_path, store, monkeypatch):
    """No later epoch beats `best` -> best.pt still exists, with the old epoch."""
    undo = use_stub(hidden=32, patch=14, layers=8)
    try:
        data_root = Path(store.dir).parents[1]
        cfg = _tiny_cfg(tmp_path, data_root, out_name="b1")
        _run(cfg, monkeypatch)
        old_best = torch.load(Path(cfg.output_dir) / "best.pt",
                              map_location="cpu", weights_only=False)

        # An unbeatable `best`, so the resumed epochs cannot write best.pt.
        ck = _full(cfg)
        ck["best"] = 0.0
        src = tmp_path / "prev"
        src.mkdir()
        torch.save(ck, src / "last_full.pt")
        shutil.copy2(Path(cfg.output_dir) / "best.pt", src / "best.pt")

        cfg2 = _tiny_cfg(tmp_path, data_root, out_name="b2")
        cfg2.epochs = 3
        cfg2.phase = 2
        cfg2.resume = str(src / "last_full.pt")
        _run(cfg2, monkeypatch)

        dst = Path(cfg2.output_dir) / "best.pt"
        assert dst.is_file(), "best.pt was not carried into the new output_dir"
        got = torch.load(dst, map_location="cpu", weights_only=False)
        assert got["epoch"] == old_best["epoch"]
        assert "[resume] carried best.pt" in _log(cfg2)
    finally:
        undo()


def test_resume_past_the_last_epoch_says_nothing_to_train(
        tmp_path, store, monkeypatch):
    undo = use_stub(hidden=32, patch=14, layers=8)
    try:
        data_root = Path(store.dir).parents[1]
        cfg = _tiny_cfg(tmp_path, data_root, out_name="n1")
        _run(cfg, monkeypatch)
        before = _full(cfg)

        cfg2 = _tiny_cfg(tmp_path, data_root, out_name="n2")
        cfg2.resume = str(Path(cfg.output_dir) / "last_full.pt")   # epochs == 2
        _run(cfg2, monkeypatch)

        log = _log(cfg2)
        assert "nothing to train" in log
        assert "--phase 2" in log
        assert "[phase]" not in log, "a finished run must not auto-bump"
        after = json.loads((Path(cfg2.output_dir) / "metrics.json").read_text())
        assert [h["epoch"] for h in after["history"]] == \
               [h["epoch"] for h in before["history"]]
        assert "final_plain" in after
    finally:
        undo()
