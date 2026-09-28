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

from .augment import achievable_gsd_range
from .dataset import FullTileDataset, TileDataset, UnlabeledTileDataset
from .packed import PackedStore, store_exists

# Where each source's val split lives, if it has one.  Sources that are packed
# one store per GSD family carry a suffix (`synrs3d_g05`, `dfc23_g050`), so the
# lookup is prefix-aware — a source that falls through here gets no val store,
# and `build_loaders` then silently scores a slice of the first *train* store
# instead, which is a val number that means nothing.
_VAL_SPLIT = {"gamus": "val", "geonrw": "test", "synrs3d": None,
              "india_labeled": "val", "india_unlabeled": None, "us3d": "val",
              "mvs3dm": "val"}
_VAL_SPLIT_PREFIX = (("synrs3d_", None), ("dfc23_", "val"))


def val_split_of(name: str) -> str | None:
    if name in _VAL_SPLIT:
        return _VAL_SPLIT[name]
    for pre, sp in _VAL_SPLIT_PREFIX:
        if name.startswith(pre):
            return sp
    return None


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
        train_w.append(float(cfg.sampler_weight(name)))
        # The GSD jitter range in the config is a *request*; what a store can
        # actually deliver is bounded by its own extent, because v3 moved to
        # picking the crop in source pixels first (see dwdata/augment.py).  A
        # 1024 px / 0.25 m GAMUS tile caps at 1024*0.25/512 = 0.5 m, so the
        # configured 1.20 m upper bound is silently unreachable and the real
        # scale span is 1.7x, not 4x.  That clamp is correct — it is what stops
        # v2's zero-padding bug — but it was invisible, so print it: mixing a
        # coarse source (DFC2019 at 1.3 m) is the only way to actually train the
        # coarse end, and you cannot tell that from the config alone.
        lo_g, hi_g = achievable_gsd_range(
            st.tile_px, st.tile_px, st.gsd_m, cfg.tile_size,
            cfg.gsd_jitter_lo_m, cfg.gsd_jitter_hi_m)
        note = ""
        if hi_g < cfg.gsd_jitter_hi_m - 1e-6:
            note = f"  [!] requested hi {cfg.gsd_jitter_hi_m:.2f} unreachable"
        print(f"[data] {name}/train: {len(st)} tiles @ {st.tile_px}px / {st.gsd_m} m"
              f"  gsd_jitter={lo_g:.2f}..{hi_g:.2f} m  seg={'yes' if st.has_seg else 'NO'}{note}")
    if missing:
        print(f"[data] not prepared, skipped: {missing}  "
              f"(run `python prepare_data.py --datasets {','.join(missing)}`)")
    # Every v4 run to date carried w_seg=0.2 against stores with no semantic
    # raster, so seg_ce_loss returned exactly 0.0 on every step of every epoch
    # and nobody noticed until the run was over.  It is one boolean, already in
    # index.json — say it at startup instead.
    if train_sets and float(getattr(cfg, "w_seg", 0.0)) > 0:
        seg_ok = [ds.src for ds in train_sets if ds.store.has_seg]
        if not seg_ok:
            print(f"[data] !! w_seg={cfg.w_seg} but NO train store has semantic labels — "
                  f"seg_ce_loss will be exactly 0.0 for the whole run and "
                  f"flatness_loss loses its ground/road/water restriction. "
                  f"Pass --w_seg 0, or prepare a source that ships classes.")
        elif len(seg_ok) < len(train_sets):
            print(f"[data] w_seg={cfg.w_seg}; semantic labels only from {seg_ok}")

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
        sp = val_split_of(name)
        if sp and (st := _open(root, name, sp)) is not None:
            val_store, val_name = st, name
            break
    if val_store is None:
        # fall back to a held-out slice of the first train store
        val_store, val_name = train_sets[0].store, train_sets[0].src
        print(f"[data] no dedicated val split; using a slice of {val_name}/train")

    dl_va, full, n_val = _val_loader(cfg, spec, val_store, val_name,
                                     gpu_aug, is_main)
    print(f"[data] val: {n_val} tiles from {val_name} "
          f"({_how(n_val, len(val_store), getattr(cfg, 'val_sample_seed', -1))})  "
          f"({len(dl_tr)} train steps/epoch)")
    check_split_overlaps(cfg, root, is_main)
    return dl_tr, dl_va, full


def _how(n_val: int, n_store: int, seed: int = -1) -> str:
    if n_val >= n_store:
        return "all"
    if seed is not None and int(seed) >= 0:
        return f"random {n_val} of {n_store}, seed {int(seed)}"
    return f"first {n_val} of {n_store}, prefix not sample"


def check_split_overlaps(cfg, root: Path, is_main: bool = True) -> dict:
    """Held-out claims, checked instead of asserted by comment.

    * every `--test_sources name:split` against `name/train` and `name/val`
    * `india_labeled/val` against every `dfc23_*/train` (both are cut from the
      DFC23 New Delhi scenes)

    Stems are compared as packed.  Any overlap raises — a test number measured
    on training tiles is worse than no number.  Returns the counts it printed.
    """
    out = {}

    def stems(name, split):
        st = _open(root, name, split)
        return set(st.stems) if st is not None and getattr(st, "stems", None) else None

    pairs = []
    for name, split in test_split_pairs(cfg):
        for other in ("train", val_split_of(name)):
            if other and other != split:
                pairs.append(((name, split), (name, other)))
    if (root / "india_labeled" / "val").exists():
        for d in sorted(root.glob("dfc23_*")):
            pairs.append((("india_labeled", "val"), (d.name, "train")))
    for a, b in pairs:
        sa, sb = stems(*a), stems(*b)
        if sa is None or sb is None:
            continue
        n = len(sa & sb)
        out[f"{a[0]}/{a[1]} vs {b[0]}/{b[1]}"] = n
        if is_main:
            print(f"[data] overlap {a[0]}/{a[1]} vs {b[0]}/{b[1]}: {n} shared stems "
                  f"of {len(sa)}")
        if n:
            raise AssertionError(
                f"{n} tiles of {a[0]}/{a[1]} are also in {b[0]}/{b[1]} — it is not "
                f"held out.  e.g. {sorted(sa & sb)[:3]}")
    return out


def _val_loader(cfg, spec, val_store, val_name: str, gpu_aug: bool,
                is_main: bool, n_tiles: int | None = None):
    """One val DataLoader + its full-tile twin, for any store.

    Factored out of `build_loaders` so the same recipe can serve the primary val
    set, the secondary out-of-domain ones (`build_aux_val_loaders`) *and* the
    held-out test stores (`build_test_loaders`), which score every tile rather
    than a 400-tile prefix and so pass their own `n_tiles`.
    """
    # `TileDataset`/`FullTileDataset` both index with `ti = i % len(store)`, so
    # a length of N selects the **first N tiles in sorted-stem order** — a
    # prefix, not a random sample.  That is deliberately left alone: v1-v4 were
    # all scored on the same prefix, and swapping in a random subset now would
    # silently break the only yardstick we have (v3's 2.723 m).  `--val_tiles 0`
    # scores the whole store instead, which is the honest number to report
    # alongside it.
    budget = cfg.val_tiles if n_tiles is None else n_tiles
    n_val = len(val_store) if budget <= 0 else min(budget, len(val_store))
    idx = sample_indices(len(val_store), n_val, getattr(cfg, "val_sample_seed", -1))
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
    dl = DataLoader(
        TileDataset(cfg, val_store, spec, val_name, train=False, length=n_val,
                    gpu_augment=gpu_aug, indices=idx),
        # Eval is inference-only: no activations are kept, so it fits a much
        # bigger batch than training and there is no reason to make the card
        # run it at the training batch.
        batch_size=max(1, int(cfg.batch_size * max(1, cfg.eval_batch_mult))),
        shuffle=False,
        num_workers=n_val_workers, pin_memory=_cuda(),
        persistent_workers=False,
        prefetch_factor=cfg.prefetch_factor if n_val_workers else None,
    )
    full = FullTileDataset(cfg, val_store, spec, val_name, length=n_val, indices=idx)
    return dl, full, n_val


def sample_indices(n_store: int, n: int, seed: int):
    """None (the v1-v4 prefix) or a sorted seeded random subset of tile indices.

    v4's val was `first 400 of 859` — 87.5 % urban against a 57.6 % urban test
    set, and best.pt was selected on it.  Sorted so the loader still reads the
    shards in order.
    """
    if seed is None or int(seed) < 0 or n >= n_store:
        return None
    return np.sort(np.random.default_rng(int(seed)).permutation(n_store)[:n])


def build_aux_val_loaders(cfg, spec, primary: str, is_main: bool = True,
                          gpu_aug: bool | None = None) -> dict:
    """Every *other* prepared val store named in `--datasets`, keyed by source.

    The primary val set is whichever source comes first in `--datasets` and has
    a val split, and with `gamus` first that stays GAMUS — which is the point:
    v1-v4 are all scored on the same 400-tile GAMUS prefix and that comparison
    is the only yardstick the project has.  But GAMUS is US *aerial* imagery,
    so it cannot answer the question the deliverable actually turns on, which is
    how the model does on satellite imagery it was not trained to fit.  DFC23
    Track 2 can, and reporting it next to GAMUS is what makes the dataset-mix
    question decidable: "v4 beat v4-2 on GAMUS val" is partly just v4 having
    trained on nothing but the val domain.

    These are reported, never selected on — `best.pt` stays tied to the primary.
    """
    root = Path(cfg.data_root)
    if gpu_aug is None:
        gpu_aug = bool(getattr(cfg, "gpu_augment", False)) and _cuda()
    out = {}
    for name in cfg.labeled_sources():
        if name == primary:
            continue
        sp = val_split_of(name)
        if not sp:
            continue
        st = _open(root, name, sp)
        if st is None:
            continue
        dl, full, n_val = _val_loader(cfg, spec, st, name, gpu_aug, is_main)
        out[name] = (dl, full)
        print(f"[data] val+ : {n_val} tiles from {name} "
              f"({_how(n_val, len(st), getattr(cfg, 'val_sample_seed', -1))})  — reported, not selected on")
    return out


# `--test_sources` entries may be written bare; this is the split they mean.
_TEST_DEFAULT_SPLIT = "test"


def test_split_pairs(cfg) -> list[tuple[str, str]]:
    """`--test_sources` parsed into (store, split) pairs.

    Deliberately NOT routed through `val_split_of`: a test store is named with
    its split spelled out precisely so nothing about it is inferred from a table
    that the training path also reads.  `gamus` and `gamus:test` mean the same
    thing; anything else must say its split.
    """
    out: list[tuple[str, str]] = []
    for part in str(getattr(cfg, "test_sources", "") or "").split(","):
        part = part.strip()
        if not part:
            continue
        name, _, split = part.partition(":")
        name, split = name.strip(), split.strip()
        if name:
            out.append((name, split or _TEST_DEFAULT_SPLIT))
    return out


def build_test_loaders(cfg, spec, is_main: bool = True,
                       gpu_aug: bool | None = None) -> dict:
    """Held-out test stores, keyed by "<store>/<split>".

    These are built for the post-training stage only and iterated exactly once,
    with `best.pt` already loaded.  They never reach `build_loaders`, so there
    is no path by which a tile in here is sampled into an epoch, and no path by
    which one moves the best.pt decision — which is the entire reason the split
    exists.  `gamus/test` is 2861 tiles the run has never seen; `gamus/val` is
    both the selection set and the set `final_*` is reported on.

    A missing store is a hard error, not a skipped line.  A val store that is
    absent costs a reported number; a *test* store that is absent silently
    turns the run's headline back into the number it was meant to replace.
    """
    root = Path(cfg.data_root)
    if gpu_aug is None:
        gpu_aug = bool(getattr(cfg, "gpu_augment", False)) and _cuda()
    n_tiles = int(getattr(cfg, "test_tiles", 0) or 0)
    out = {}
    for name, split in test_split_pairs(cfg):
        st = _open(root, name, split)
        if st is None:
            raise RuntimeError(
                f"--test_sources names {name}:{split} but no store exists at "
                f"{root / name / split}. Prepare it (01_data_prep.ipynb packs "
                f"gamus/test with --gamus_test 0) or drop it from the flag — "
                f"do not let a held-out number go quietly missing.")
        key = f"{name}/{split}"
        dl, full, n = _val_loader(cfg, spec, st, key, gpu_aug, is_main,
                                  n_tiles=n_tiles)
        out[key] = (dl, full)
        print(f"[data] test: {n} tiles from {key} ({_how(n, len(st), getattr(cfg, 'val_sample_seed', -1))}) "
              f"seg={'yes' if st.has_seg else 'NO'} — held out: scored once at "
              f"the end, never selected on")
    return out


def _cuda() -> bool:
    return torch.cuda.is_available()
