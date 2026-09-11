"""DataLoader construction.

Stage-agnostic: one train loader over a weighted mix of whatever *labeled* stores
were prepared, one cheap centre-crop val loader, a full-tile val set for the
final sliding-window evaluation, and — when an unlabeled store exists — a
separate infinite loader feeding the mean-teacher branch.

The unlabeled loader is deliberately its own loader rather than another entry in
the weighted mix: it yields a different sample shape (two views, no target), it
runs at its own batch size, and it must be able to run out and restart
independently of the labeled epoch length.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from .dataset import FullTileDataset, TileDataset, UnlabeledTileDataset
from .packed import PackedStore, store_exists

# Where each source's val split lives, if it has one.
_VAL_SPLIT = {"gamus": "val", "geonrw": "test", "synrs3d": None,
              "india_labeled": "val", "india_unlabeled": None}


def _open(root: Path, name: str, split: str) -> PackedStore | None:
    d = root / name / split
    return PackedStore(d) if store_exists(d) else None


def build_unlabeled_loader(cfg, spec):
    """The mean-teacher branch's input, or None when no such store was prepared.

    Returns None rather than raising when the store is absent: the Indian
    unlabeled set is an optional add-on, and a run that has not got it should
    train exactly as it would have without the branch.
    """
    if cfg.w_consistency <= 0 or not cfg.unlabeled_source:
        return None
    root = Path(cfg.data_root)
    st = _open(root, cfg.unlabeled_source, "train")
    if st is None:
        if cfg.unlabeled_source in cfg.dataset_list():
            print(f"[data] --datasets asked for {cfg.unlabeled_source!r} but no store "
                  f"exists under {root / cfg.unlabeled_source}; consistency branch off. "
                  f"See `python prepare_data.py --datasets india --help`")
        return None
    bs = max(1, int(round(cfg.batch_size * cfg.unlabeled_batch_frac)))
    ds = UnlabeledTileDataset(cfg, st, spec, cfg.unlabeled_source, length=len(st))
    print(f"[data] {cfg.unlabeled_source}/unlabeled: {len(st)} tiles @ "
          f"{st.tile_px}px / {st.gsd_m} m -> mean-teacher batch {bs}")
    return DataLoader(
        ds, batch_size=bs, shuffle=True, num_workers=max(0, cfg.num_workers // 2),
        pin_memory=torch.cuda.is_available(), drop_last=True,
        persistent_workers=cfg.num_workers > 1,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 1 else None,
    )


class CyclicLoader:
    """An endless iterator over a DataLoader — the unlabeled set has no epoch."""

    def __init__(self, loader):
        self.loader = loader
        self._it = iter(loader)

    def next(self):
        try:
            return next(self._it)
        except StopIteration:
            self._it = iter(self.loader)
            return next(self._it)


def build_loaders(cfg, spec):
    root = Path(cfg.data_root)
    names = cfg.labeled_sources()
    weights = cfg.sampler_weight_map()

    train_sets, train_w, missing = [], [], []
    for name in names:
        st = _open(root, name, "train")
        if st is None:
            missing.append(name)
            continue
        train_sets.append(TileDataset(cfg, st, spec, name, train=True, length=len(st)))
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
        num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(), drop_last=True,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers else None,
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
    dl_va = DataLoader(
        TileDataset(cfg, val_store, spec, val_name, train=False, length=n_val),
        batch_size=max(1, cfg.batch_size), shuffle=False,
        num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers else None,
    )
    print(f"[data] val: {n_val} tiles from {val_name}  "
          f"({len(dl_tr)} train steps/epoch)")
    full = FullTileDataset(cfg, val_store, spec, val_name, length=n_val)
    return dl_tr, dl_va, full
