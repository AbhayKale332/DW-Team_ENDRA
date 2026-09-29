"""v5 selection hygiene: seeded val sample and held-out overlap checks."""

import numpy as np
import pytest

from dwdata.loaders import _how, check_split_overlaps, sample_indices
from dwdata.packed import ShardWriter


def _store(root, name, split, stems):
    w = ShardWriter(root / name / split, tile_px=32, gsd_m=0.5, shard_tiles=8)
    for s in stems:
        w.add(s, np.zeros((32, 32, 3), np.uint8), np.zeros((32, 32), np.float32),
              None, np.ones((32, 32), bool))
    w.finalise()


def test_seeded_sample_is_random_sorted_and_reproducible():
    a = sample_indices(859, 400, 42)
    b = sample_indices(859, 400, 42)
    assert (a == b).all() and (np.diff(a) > 0).all() and len(a) == 400
    assert a.max() > 500                      # not the prefix
    assert sample_indices(859, 400, -1) is None
    assert sample_indices(10, 400, 42) is None
    assert _how(400, 859, 42) == "random 400 of 859, seed 42"
    assert "prefix" in _how(400, 859, -1)


def test_overlap_between_test_and_train_raises(tmp_path, cfg):
    _store(tmp_path, "gamus", "train", ["a", "b", "c"])
    _store(tmp_path, "gamus", "test", ["x", "y"])
    cfg.test_sources = "gamus:test"
    out = check_split_overlaps(cfg, tmp_path)
    assert out["gamus/test vs gamus/train"] == 0
    _store(tmp_path, "gamus", "val", ["y", "z"])
    with pytest.raises(AssertionError, match="not held out"):
        check_split_overlaps(cfg, tmp_path)


def test_india_val_is_checked_against_dfc23_train(tmp_path, cfg):
    _store(tmp_path, "india_labeled", "val", ["delhi_1", "delhi_2"])
    _store(tmp_path, "dfc23_g050", "train", ["delhi_2", "rio_9"])
    cfg.test_sources = ""
    with pytest.raises(AssertionError):
        check_split_overlaps(cfg, tmp_path)
