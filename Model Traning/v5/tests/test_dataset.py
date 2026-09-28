import numpy as np
import torch

from dwdata.dataset import FullTileDataset, TileDataset
from dwdata.preprocess import PreprocSpec


def _spec(cfg):
    return PreprocSpec.from_config(cfg, resolve_stats=False)


def test_train_sample_has_no_padding(cfg, store):
    """The v2 regression, end to end: ~41% of its training pixels were black
    zero-padding labelled 0 m and marked valid.  Here every pixel is real."""
    spec = _spec(cfg)
    ds = TileDataset(cfg, store, spec, "gamus", train=True, length=64)
    for i in range(40):
        s = ds[i]
        rgb = s["rgb_u8"].numpy()
        assert s["image"].shape == (3, cfg.tile_size, cfg.tile_size)
        assert s["valid"].float().mean() > 0.99, "padding leaked into the valid mask"
        # a fully black row/column is the fingerprint of constant-pad
        assert not (rgb.reshape(-1, 3).max(1) == 0).all()


def test_effective_gsd_stays_in_the_achievable_range(cfg, store):
    ds = TileDataset(cfg, store, _spec(cfg), "gamus", train=True, length=64)
    g = np.array([float(ds[i]["gsd_m"]) for i in range(50)])
    # 256 px @ 0.33 m filling a 64 px tile allows up to 256*0.33/64 = 1.32 m
    assert g.min() >= cfg.gsd_jitter_lo_m - 1e-6
    assert g.max() <= 1.32 + 1e-6
    assert g.std() > 0, "scale augmentation is not varying"


def test_augmentation_is_redrawn_every_epoch(cfg, store):
    """v2 seeded the RNG off (seed, index), so all 30 epochs replayed one frozen
    augmented copy of the dataset."""
    ds = TileDataset(cfg, store, _spec(cfg), "gamus", train=True, length=64)
    a, b = ds[0]["image"], ds[0]["image"]
    assert not torch.allclose(a, b)


def test_val_sample_is_deterministic_and_canonical(cfg, store):
    ds = TileDataset(cfg, store, _spec(cfg), "gamus", train=False, length=4)
    a, b = ds[1], ds[1]
    assert torch.allclose(a["image"], b["image"])
    assert abs(float(a["gsd_m"]) - cfg.canonical_gsd_m) < 0.02


def test_targets_are_metric_and_masked(cfg, store):
    ds = TileDataset(cfg, store, _spec(cfg), "gamus", train=False, length=4)
    s = ds[0]
    t, v = s["target"], s["valid"]
    assert t.shape == (1, cfg.tile_size, cfg.tile_size)
    assert (t[v] >= 0).all() and (t[v] <= cfg.max_valid_height_m).all()
    assert 10.0 < float(t.max()) < 14.0, "the 12 m block must survive resampling"


def test_full_tile_dataset_is_native_resolution(cfg, store):
    ds = FullTileDataset(cfg, store, _spec(cfg), "gamus")
    s = ds[0]
    assert s["rgb_u8"].shape == (256, 256, 3)
    assert abs(float(s["gsd_m"]) - 0.33) < 1e-6
