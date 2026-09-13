"""DataLoader construction.

Stage-agnostic: one train loader over a weighted mix of whatever stores were
prepared, one cheap centre-crop val loader, and (optionally) a full-tile val set
for the final sliding-window evaluation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from .dataset import FullTileDataset, TileDataset
from .packed import PackedStore, store_exists

# Where each source's val split lives, if it has one.
_VAL_SPLIT = {"gamus": "val", "geonrw": "test", "synrs3d": None}


def _open(root: Path, name: str, split: str) -> PackedStore | None:
    d = root / name / split
    return PackedStore(d) if store_exists(d) else None


def _prime(store, spec, cfg, label: str) -> None:
    """Persist the per-tile stretch bounds before any worker forks.

    One pass over the store on the parent (threaded, I/O bound) replaces a
    full-tile histogram inside every crop in every worker for the rest of the
    run.  It writes `stretch_bounds_*.npy` next to the shards, so the second
    run of a Studio pays nothing at all.
    """
    if not spec.radiometric_stretch:
        return
    import time

    t0 = time.time()
    fresh = not store._bounds_path(spec.stretch_lo_pct, spec.stretch_hi_pct).is_file()
    ok = store.prime_stretch_bounds(
        spec.stretch_lo_pct, spec.stretch_hi_pct,
        workers=max(4, int(getattr(cfg, "num_workers", 8)) or 8))
    if ok and fresh:
        print(f"[data] {label}: stretch bounds computed for {len(store)} tiles "
              f"in {time.time() - t0:.0f}s (cached on disk from now on)")


def build_loaders(cfg, spec):
    root = Path(cfg.data_root)
    names = cfg.dataset_list()
    weights = cfg.sampler_weight_map()
    gpu_aug = bool(getattr(cfg, "gpu_augment", False))

    train_sets, train_w, missing = [], [], []
    for name in names:
        st = _open(root, name, "train")
        if st is None:
            missing.append(name)
            continue
        _prime(st, spec, cfg, f"{name}/train")
        train_sets.append(TileDataset(cfg, st, spec, name, train=True,
                                      length=len(st), gpu_augment=gpu_aug))
        train_w.append(float(weights.get(name, 1.0)))
        print(f"[data] {name}/train: {len(st)} tiles @ {st.tile_px}px / {st.gsd_m} m")
    if missing:
        print(f"[data] not prepared, skipped: {missing}  "
              f"(run `python prepare_data.py --datasets {','.join(missing)}`)")
    if not train_sets:
        raise RuntimeError(
            f"no prepared datasets under {root}. Run prepare_data.py first.")

    concat = ConcatDataset(train_sets)
    # Weight per sample so the *mix ratio* is what `sampler_weights` says,
    # independent of how many tiles each store happens to hold.
    sample_w = np.concatenate([
        np.full(len(ds), w / max(1, len(ds))) for ds, w in zip(train_sets, train_w)
    ])
    sampler = WeightedRandomSampler(
        sample_w.tolist(), num_samples=int(cfg.crops_per_epoch), replacement=True)

    dl_tr = DataLoader(
        concat, batch_size=cfg.batch_size, sampler=sampler,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers else None,
        # Workers are forked once and then live for the whole run, so paying a
        # few hundred ms each to warm their memmaps and import numpy's SIMD
        # paths is free; what is not free is the first batch of every epoch
        # stalling behind a fresh fork.
        pin_memory_device="cuda" if _cuda() else "",
    )

    # -- validation ------------------------------------------------------
    val_store, val_name = None, None
    for name in names:
        sp = _VAL_SPLIT.get(name)
        if sp and (st := _open(root, name, sp)) is not None:
            val_store, val_name = st, name
            break
    if val_store is None:
        # fall back to a held-out slice of the first train store
        val_store, val_name = train_sets[0].store, train_sets[0].src
        print(f"[data] no dedicated val split; using a slice of {val_name}/train")

    n_val = min(cfg.val_tiles, len(val_store))
    # Val runs every `eval_every` epochs, so it does not need — and must not
    # hold — a second full set of persistent workers: with num_workers sized for
    # the GPU (20+) that doubled the process count and had the two pools
    # fighting for the same cores through every training epoch.
    n_val_workers = min(4, cfg.num_workers)
    _prime(val_store, spec, cfg, f"{val_name}/val")
    dl_va = DataLoader(
        TileDataset(cfg, val_store, spec, val_name, train=False, length=n_val,
                    gpu_augment=gpu_aug),
        # Eval is inference-only: no activations are kept, so it fits a much
        # bigger batch than training and there is no reason to make the card
        # run it at the training batch.
        batch_size=max(1, int(cfg.batch_size * max(1, cfg.eval_batch_mult))),
        shuffle=False,
        num_workers=n_val_workers, pin_memory=True,
        persistent_workers=False,
        prefetch_factor=cfg.prefetch_factor if n_val_workers else None,
    )
    print(f"[data] val: {n_val} tiles from {val_name}  "
          f"({len(dl_tr)} train steps/epoch)")
    full = FullTileDataset(cfg, val_store, spec, val_name, length=n_val)
    return dl_tr, dl_va, full


def _cuda() -> bool:
    import torch

    return torch.cuda.is_available()
