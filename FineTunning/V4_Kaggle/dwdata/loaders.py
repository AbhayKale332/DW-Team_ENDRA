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


def _prime(store, spec, cfg, label: str, is_main: bool = True) -> None:
    """Persist the per-tile stretch bounds before any worker forks.

    One pass over the store on the parent (threaded, I/O bound) replaces a
    full-tile histogram inside every crop in every worker for the rest of the
    run.  It writes `stretch_bounds_*.npy` next to the shards, so the second
    run of a Studio pays nothing at all.
    """
    if not spec.radiometric_stretch:
        return
    # Under DDP only rank 0 computes and writes `stretch_bounds_*.npy` (96 KB).
    # Two ranks racing on the same `.npy` write is the only place in the data
    # path where they would touch the same file, and the work is duplicated
    # anyway — ranks 1..N-1 pick the cached table up on first use.
    if not is_main:
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


def build_unlabeled_loader(cfg, spec, rank: int = 0, world_size: int = 1,
                           is_main: bool = True):
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
    _prime(st, spec, cfg, f"{cfg.unlabeled_source}/train", is_main)
    ds = UnlabeledTileDataset(cfg, st, spec, cfg.unlabeled_source, length=len(st),
                              gpu_augment=bool(getattr(cfg, "gpu_augment", False)))
    # This pool runs *alongside* the labeled one for the whole run, so it gets a
    # slice of the worker budget rather than a second full set: v3 measured two
    # full persistent pools fighting over the same cores (and tripping the
    # "24/28/35 worker processes" warning straight into a shared-memory Bus
    # error) as a bigger loss than the branch was worth.
    nw = max(0, min(4, cfg.num_workers // 3))
    # Rank-salted so the two processes do not consume identical unlabeled crops.
    gen = torch.Generator()
    gen.manual_seed(int(cfg.seed) + 2000 * int(rank))
    return DataLoader(
        ds, batch_size=bs, shuffle=True, num_workers=nw, generator=gen,
        pin_memory=_cuda(), pin_memory_device="cuda" if _cuda() else "",
        drop_last=True, persistent_workers=nw > 0,
        prefetch_factor=cfg.prefetch_factor if nw else None,
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


def build_loaders(cfg, spec, rank: int = 0, world_size: int = 1,
                  is_main: bool = True):
    """Build the train / val / full-tile loaders.

    Under DDP (`world_size > 1`) `cfg.batch_size` is a *per-process* micro-batch,
    so the global effective batch is `world_size x batch_size x grad_accum`.
    `cfg.crops_per_epoch` stays a *global* count: the train sampler draws
    `crops_per_epoch // world_size` per rank, so `len(dl_tr)` halves on two
    ranks and the epoch-fraction progress math in train.py needs no change.
    `drop_last=True` keeps `len(dl_tr)` identical on every rank, which is what
    keeps the forward/backward counts — and therefore the all-reduces — in
    lockstep.
    """
    root = Path(cfg.data_root)
    names = cfg.labeled_sources()
    weights = cfg.sampler_weight_map()
    gpu_aug = bool(getattr(cfg, "gpu_augment", False))

    train_sets, train_w, missing = [], [], []
    for name in names:
        st = _open(root, name, "train")
        if st is None:
            missing.append(name)
            continue
        _prime(st, spec, cfg, f"{name}/train", is_main)
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
    world_size = max(1, int(world_size))
    gen = torch.Generator()
    gen.manual_seed(int(cfg.seed) + 1000 * int(rank))
    sampler = WeightedRandomSampler(
        sample_w.tolist(), num_samples=int(cfg.crops_per_epoch) // world_size,
        replacement=True, generator=gen)

    dl_tr = DataLoader(
        concat, batch_size=cfg.batch_size, sampler=sampler,
        num_workers=cfg.num_workers, pin_memory=_cuda(), drop_last=True,
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
    # the GPU (16+) that doubled the process count and had the two pools
    # fighting for the same cores through every training epoch.
    # Under DDP only rank 0 ever iterates `dl_va` (evaluation is inference-only
    # and runs through the unwrapped module, so calling it on every rank would
    # be a NCCL mismatch).  Giving the other ranks workers for a loader they
    # never touch is pure oversubscription — on Kaggle's ~4 vCPU, 2 train
    # workers per rank plus two idle val pools is already more processes than
    # cores.  The loader itself is still built, so the return contract and the
    # single-process path are unchanged.
    n_val_workers = min(4, cfg.num_workers) if is_main else 0
    _prime(val_store, spec, cfg, f"{val_name}/val", is_main)
    dl_va = DataLoader(
        TileDataset(cfg, val_store, spec, val_name, train=False, length=n_val,
                    gpu_augment=gpu_aug),
        # Eval is inference-only: no activations are kept, so it fits a much
        # bigger batch than training and there is no reason to make the card
        # run it at the training batch.
        batch_size=max(1, int(cfg.batch_size * max(1, cfg.eval_batch_mult))),
        shuffle=False,
        num_workers=n_val_workers, pin_memory=_cuda(),
        persistent_workers=False,
        prefetch_factor=cfg.prefetch_factor if n_val_workers else None,
    )
    print(f"[data] val: {n_val} tiles from {val_name}  "
          f"({len(dl_tr)} train steps/epoch)")
    full = FullTileDataset(cfg, val_store, spec, val_name, length=n_val)
    return dl_tr, dl_va, full


def _cuda() -> bool:
    return torch.cuda.is_available()
