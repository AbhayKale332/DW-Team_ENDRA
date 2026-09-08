"""Build train / val DataLoaders for a training stage.

Stage F (finetune): a `ConcatDataset` of the requested datasets with a
`WeightedRandomSampler` so GAMUS:GeoNRW ~= the configured ratio regardless of
their raw sizes.
Stage P (pretrain): SynRS3D archive dataset (rotated per epoch by the trainer).
"""

from __future__ import annotations

import random

import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from .base import TileDatasetBase  # noqa: F401  (re-export for tests)


def _subset(stems: list[str], n: int, seed: int) -> list[str]:
    if not n or n >= len(stems):
        return stems
    rng = random.Random(seed)
    s = stems[:]
    rng.shuffle(s)
    return sorted(s[:n])


def build_finetune_loaders(cfg, token: str | None):
    from .gamus import GamusDataset, list_stems as gamus_stems, make_cache as gamus_cache

    names = cfg.dataset_list()
    weights_cfg = cfg.sampler_weight_map()
    datasets, per_ds_weight = [], []

    if "gamus" in names:
        cache = gamus_cache(cfg, token)
        tr = _subset(gamus_stems(cache, "train"), cfg.train_tiles, cfg.seed)
        va = _subset(gamus_stems(cache, "val"), cfg.val_tiles, cfg.seed + 1)
        datasets.append(GamusDataset(cfg, cache, "train", tr, train=True))
        per_ds_weight.append(weights_cfg.get("gamus", 1.0))
        val_ds = GamusDataset(cfg, cache, "val", va, train=False)
    else:
        val_ds = None

    if "geonrw" in names:
        try:
            from pathlib import Path

            from .geonrw import GeoNRWDataset, discover_tiles
            from .streaming import BoundedCacheHF

            if cfg.data_source == "local":
                base = Path(cfg.local_root)
            else:
                gc = BoundedCacheHF(cfg.geonrw_repo, cfg.cache_dir,
                                    int(cfg.cache_max_gib * 1024**3), token=token)
                base = gc.ensure_archive("nrw_dataset.tar.gz")
            tr_paths = discover_tiles(base, "train")
            tr_paths = tr_paths[:cfg.train_tiles] if cfg.train_tiles else tr_paths
            if not tr_paths:
                raise RuntimeError(f"no GeoNRW tiles found under {base}")
            datasets.append(GeoNRWDataset(cfg, tr_paths, "train", train=True))
            per_ds_weight.append(weights_cfg.get("geonrw", 1.0))
            if val_ds is None:
                va_paths = discover_tiles(base, "test")[: cfg.val_tiles or None]
                val_ds = GeoNRWDataset(cfg, va_paths, "test", train=False)
        except Exception as e:  # noqa: BLE001 — GeoNRW is auxiliary; GAMUS carries the run
            print(f"[geonrw] unavailable, continuing without it: {e}")

    if not datasets:
        raise RuntimeError(f"no finetune datasets built from {cfg.datasets!r}")

    concat = ConcatDataset(datasets)
    # per-sample weight = dataset weight / dataset length  -> equalises then re-ratios
    sample_w = np.concatenate([
        np.full(len(ds), w / max(1, len(ds)))
        for ds, w in zip(datasets, per_ds_weight)
    ])
    sampler = WeightedRandomSampler(sample_w.tolist(), num_samples=len(concat), replacement=True)

    dl_tr = DataLoader(
        concat, batch_size=cfg.batch_size, sampler=sampler,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers else None,
    )
    dl_va = DataLoader(
        val_ds, batch_size=max(1, cfg.batch_size // 2), shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers else None,
    ) if val_ds is not None else None
    return dl_tr, dl_va


def build_pretrain_dataset(cfg, token: str | None):
    from .streaming import BoundedCacheHF
    from .synrs3d import SynRS3DArchiveDataset

    cache = BoundedCacheHF(cfg.synrs3d_repo, cfg.cache_dir,
                           int(cfg.cache_max_gib * 1024**3), token=token)
    return SynRS3DArchiveDataset(cfg, cache, train=True)


def pretrain_loader(cfg, ds) -> DataLoader:
    return DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=False,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers else None,
    )
