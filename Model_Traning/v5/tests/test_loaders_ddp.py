"""The sampler-sharding regression for the DDP port.

`build_loaders(..., rank, world_size, is_main)` has to do two things and the
rest of the suite checks neither: give every rank the *same number of steps*
(DDP deadlocks the moment two ranks disagree on how many all-reduces an epoch
has) and give them *different crops* (two ranks drawing the identical stream is
a wasted GPU).  Both defaults must also leave the single-process path untouched.
"""

import numpy as np
import pytest
from config import Config
from dwdata.packed import ShardWriter
from dwdata.preprocess import PreprocSpec


def _store(tmp_path, n=24, tile=64):
    root = tmp_path / "gamus" / "train"
    w = ShardWriter(root, tile_px=tile, gsd_m=0.5, shard_tiles=8)
    rng = np.random.default_rng(0)
    for i in range(n):
        rgb = (rng.random((tile, tile, 3)) * 60 + 90).astype(np.uint8)
        h = np.zeros((tile, tile), np.float32)
        h[tile // 4:tile // 2, tile // 4:tile // 2] = 12.0
        w.add(f"t{i}", rgb, h, np.where(h > 1, 2, 0).astype(np.uint8),
              np.ones((tile, tile), bool))
    w.finalise()
    return root


@pytest.fixture
def dcfg(tmp_path):
    c = Config()
    c.tile_size = 32
    c.smoke = True
    c = c.apply_smoke().validate()
    _store(tmp_path)
    c.data_root = str(tmp_path)
    c.datasets = "gamus"
    c.crops_per_epoch = 16
    c.batch_size = 2
    c.num_workers = 0
    c.gpu_augment = False
    c.radiometric_stretch = False
    return c


def _draws(sampler):
    return list(iter(sampler))


def test_world_size_halves_steps_and_matches_across_ranks(dcfg):
    from dwdata.loaders import build_loaders

    spec = PreprocSpec.from_config(dcfg)
    dl1, _, _ = build_loaders(dcfg, spec)                       # single process
    dl_r0, _, _ = build_loaders(dcfg, spec, rank=0, world_size=2, is_main=True)
    dl_r1, _, _ = build_loaders(dcfg, spec, rank=1, world_size=2, is_main=False)

    # Identical step counts on both ranks: this is what keeps the all-reduce
    # counts in lockstep.  `drop_last=True` is what makes it exact.
    assert len(dl_r0) == len(dl_r1)
    assert len(dl_r0) == len(dl1) // 2
    assert dl_r0.sampler.num_samples == dcfg.crops_per_epoch // 2


def test_ranks_draw_different_crops(dcfg):
    from dwdata.loaders import build_loaders

    spec = PreprocSpec.from_config(dcfg)
    dl_r0, _, _ = build_loaders(dcfg, spec, rank=0, world_size=2, is_main=True)
    dl_r1, _, _ = build_loaders(dcfg, spec, rank=1, world_size=2, is_main=False)
    a, b = _draws(dl_r0.sampler), _draws(dl_r1.sampler)
    assert len(a) == len(b)
    assert a != b, "both ranks drew the identical crop stream"


def test_defaults_are_the_single_process_path(dcfg):
    """No DDP arguments -> the explicit rank-0/world-size-1 call, exactly.

    Not "byte-for-byte v4": the port gives the sampler a seeded `torch.Generator`
    where v4 passed none, so the draw *stream* differs from v4's (it is now
    reproducible, which v4's was not).  What must hold is that omitting the new
    arguments is indistinguishable from passing the single-process values —
    same step count, same sampler size, and the same crops in the same order.
    """
    from dwdata.loaders import build_loaders

    spec = PreprocSpec.from_config(dcfg)
    dl_a, _, _ = build_loaders(dcfg, spec)
    dl_b, _, _ = build_loaders(dcfg, spec, rank=0, world_size=1, is_main=True)
    assert len(dl_a) == len(dl_b)
    assert dl_a.sampler.num_samples == dcfg.crops_per_epoch
    assert _draws(dl_a.sampler) == _draws(dl_b.sampler)
    # ...and rank 0 is the un-sharded stream, not a shard of it.
    dl_r0, _, _ = build_loaders(dcfg, spec, rank=0, world_size=2, is_main=True)
    assert dl_r0.sampler.num_samples < dl_a.sampler.num_samples


def test_unlabeled_loader_accepts_ddp_args(dcfg):
    from dwdata.loaders import build_unlabeled_loader

    spec = PreprocSpec.from_config(dcfg)
    # No unlabeled store prepared -> None, on every rank, with no exception.
    assert build_unlabeled_loader(dcfg, spec, rank=1, world_size=2,
                                  is_main=False) is None
