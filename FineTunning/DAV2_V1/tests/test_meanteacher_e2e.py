"""Train with an unlabeled store present, and prove the branch actually ran.

The failure this guards against is the v2 one: a feature that is configured,
logged about, and silently never executes.  So the assertion is not "the code
imports" — it is that the consistency statistic appears in the training log and
that the teacher's weights moved away from the student's initialisation.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

from config import Config
from dwdata.packed import PackedStore, ShardWriter
from tests.stub_dav2 import use_stub


@pytest.fixture
def unlabeled_store(tmp_path, store):
    """An `india_unlabeled` store beside the labeled one, with no supervision."""
    d = Path(store.dir).parents[1] / "india_unlabeled" / "train"
    w = ShardWriter(d, tile_px=256, gsd_m=0.56, shard_tiles=4, has_seg=False)
    rng = np.random.default_rng(3)
    for i in range(4):
        rgb = (rng.random((256, 256, 3)) * 90 + 60).astype(np.uint8)
        w.add(f"india{i}", rgb, np.zeros((256, 256), np.float32), None,
              np.zeros((256, 256), bool))
    w.finalise()
    return PackedStore(d)


def _cfg(tmp_path, data_root):
    c = Config()
    c.smoke = True
    c.apply_smoke()
    c.tile_size, c.decoder_dim, c.n_bins = 56, 32, 8
    c.epochs, c.freeze_epochs, c.crops_per_epoch = 2, 1, 6
    c.batch_size, c.grad_accum, c.val_tiles, c.num_workers = 2, 1, 2, 0
    c.amp = c.channels_last = c.cudnn_benchmark = False
    c.make_zip = c.make_figures = c.make_report = c.export_onnx = False
    c.n_qualitative = 1
    c.data_root = str(data_root)
    c.output_dir = str(tmp_path / "out")
    c.datasets = "gamus,india_unlabeled"
    c.eval_every = 1
    c.final_sliding_eval = False
    c.consistency_every = 1
    c.unlabeled_batch_frac = 1.0
    c.teacher_ema = 0.5                 # move fast so the test can see it move
    return c.validate()


def test_unlabeled_loader_yields_two_views(unlabeled_store, tmp_path, monkeypatch, store):
    from dwdata.loaders import build_unlabeled_loader
    from dwdata.preprocess import PreprocSpec

    cfg = _cfg(tmp_path, Path(store.dir).parents[1])
    dl = build_unlabeled_loader(cfg, PreprocSpec(tile_size=56, canonical_gsd_m=0.5))
    assert dl is not None
    b = next(iter(dl))
    assert b["image_weak"].shape == b["image_strong"].shape == (2, 3, 56, 56)
    assert "target" not in b and "valid" not in b, \
        "an unlabeled batch must not carry anything a loss could mistake for a label"
    assert not np.allclose(b["image_weak"].numpy(), b["image_strong"].numpy())


def test_branch_is_off_without_a_store(tmp_path, store):
    from dwdata.loaders import build_unlabeled_loader
    from dwdata.preprocess import PreprocSpec

    cfg = _cfg(tmp_path, Path(store.dir).parents[1])
    cfg.unlabeled_source = "nonexistent_source"
    assert build_unlabeled_loader(cfg, PreprocSpec(tile_size=56)) is None


def test_training_runs_the_branch(unlabeled_store, tmp_path, store, monkeypatch, capsys):
    import torch

    undo = use_stub(hidden=32, patch=14, layers=8)
    try:
        cfg = _cfg(tmp_path, Path(store.dir).parents[1])
        monkeypatch.setattr(sys, "argv", ["train.py"])
        import train as train_mod

        monkeypatch.setattr(train_mod, "parse_config", lambda *a, **k: cfg)
        import dwdata.preprocess as pp

        monkeypatch.setattr(pp, "resolve_encoder_stats",
                            lambda *a, **k: (pp.IMAGENET_MEAN, pp.IMAGENET_STD))
        train_mod.main()

        log = (Path(cfg.output_dir) / "run.log").read_text()
        assert "mean-teacher enabled" in log
        assert " con=" in log, "the consistency term never appeared in a train step"
        assert (Path(cfg.output_dir) / "best.pt").is_file()
    finally:
        undo()


def test_teacher_tracks_but_lags_the_student():
    import torch
    from torch import nn

    from train import MeanTeacher

    student = nn.Linear(4, 4)
    with torch.no_grad():
        student.weight.fill_(0.0)
    t = MeanTeacher(student, decay=0.5)
    before = t.model.weight.detach().clone()
    with torch.no_grad():
        student.weight.fill_(1.0)
    t.update(student)
    after = t.model.weight.detach().clone()
    assert not torch.allclose(before, after), "teacher never moved"
    assert (after < student.weight).all(), "teacher must lag, not copy"
    for p in t.model.parameters():
        assert not p.requires_grad
