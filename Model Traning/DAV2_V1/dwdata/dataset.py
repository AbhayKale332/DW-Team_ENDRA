"""Torch datasets over the packed stores.

Sample dict (identical for every source, so the trainer is source-agnostic):

    image   (3, S, S) float32   encoder-normalised RGB
    rgb_u8  (S, S, 3) uint8     the exact bytes that produced `image` (for export)
    target  (1, S, S) float32   nDSM in metres, >= 0
    valid   (1, S, S) bool      real supervision only — never padding, never no-data
    cls     (S, S)    int64     class id, `SEG_IGNORE_INDEX` where unlabelled
    gsd_m   ()        float32   the GSD this crop was actually delivered at
    src/stem                    provenance

Under `gpu_augment` the worker hands over `image_u8` (S, S, 3) uint8 instead of
`image`/`rgb_u8`, and `cls` as uint8: the photometric jitter and the encoder
normalisation then run on the card over the whole batch (`dwdata/gpu_aug.py`).
That is 4x less host-to-device traffic per sample and ~22 ms less CPU per crop,
and it is what kept an H100 at 99-100 % utilisation through the v3 run.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from config import SEG_IGNORE_INDEX
from .augment import (
    achievable_gsd_range, center_window, crop_and_scale, dihedral,
    photometric_jitter, sample_window,
)
from .packed import NO_LABEL, PackedStore
from .preprocess import (
    PreprocSpec, apply_stretch, apply_stretch_lut, scene_stretch_bounds,
)


class TileDataset(Dataset):
    def __init__(self, cfg, store: PackedStore, spec: PreprocSpec, src: str,
                 train: bool, length: int | None = None,
                 gpu_augment: bool = False):
        self.cfg = cfg
        self.store = store
        self.spec = spec
        self.src = src
        self.train = train
        # When the trainer does the jitter + normalisation on the card, the
        # worker hands over raw uint8 HWC (`image_u8`) and skips both.  That
        # removes ~22 ms of the ~65 ms this method costs and cuts the
        # host-to-device payload from 3.1 MB to 0.79 MB per sample.  Default
        # off so a bare `TileDataset(...)` (tests, notebooks) is unchanged.
        self.gpu_augment = bool(gpu_augment)
        self.s = int(cfg.tile_size)
        # A train "epoch" is a fixed number of random crops; a val epoch is one
        # deterministic centre crop per tile.
        self.length = int(length) if length is not None else len(store)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int) -> dict:
        cfg, spec = self.cfg, self.spec
        if self.train:
            # Fresh entropy per __getitem__ so augmentation is re-drawn every
            # epoch (v2 seeded off the tile index, which froze one augmented
            # copy of the dataset for the whole run).
            rng = np.random.default_rng()
            # …but the *tile* comes from the index the sampler drew, not from a
            # second throw of the dice.  Same distribution (the weights are
            # uniform within a store), and it is the only version that lets the
            # page cache and the per-tile LUT cache do anything: re-rolling the
            # tile made every crop an independent random read into 27 GB of
            # shards, which on network-backed storage is the whole ballgame.
            ti = i % len(self.store)
        else:
            rng = np.random.default_rng(cfg.seed * 2_654_435_761 + i)
            ti = i % len(self.store)

        src_gsd = self.store.gsd_m
        H = W = self.store.tile_px

        # -- step 3/4: pick the window in SOURCE pixels, then resample --------
        # Window selection moved *ahead* of the read and the stretch: both used
        # to run over the whole tile and then have ~90 % of the result cropped
        # away.  The stretch is pointwise, so cropping first is identical.
        if self.train and rng.random() < cfg.gsd_jitter_p:
            lo_g, hi_g = achievable_gsd_range(
                H, W, src_gsd, self.s, cfg.gsd_jitter_lo_m, cfg.gsd_jitter_hi_m)
            dst_gsd = float(rng.uniform(lo_g, hi_g))
        else:
            dst_gsd = float(cfg.canonical_gsd_m)

        if self.train:
            top, left, win, eff = sample_window(H, W, src_gsd, self.s, dst_gsd, rng)
        else:
            top, left, win, eff = center_window(H, W, src_gsd, self.s, dst_gsd)

        rgb, hgt, cls, valid = self.store.get_window(ti, top, left, win)

        # -- step 2: scene-level radiometric stretch (same op as inference) --
        # Bounds still come from the whole scene (cached per tile); only the
        # window is mapped through the resulting LUT.
        if spec.radiometric_stretch:
            rgb = apply_stretch_lut(
                rgb, self.store.stretch_lut(ti, spec.stretch_lo_pct, spec.stretch_hi_pct))

        rgb, hgt, cls, valid = crop_and_scale(rgb, hgt, cls, valid,
                                              0, 0, win, self.s)

        # -- geometric + photometric augmentation ---------------------------
        if self.train:
            k, flip = int(rng.integers(0, 4)), bool(rng.random() < 0.5)
            rgb, hgt, cls, valid = (dihedral(a, k, flip) for a in (rgb, hgt, cls, valid))
            # The photometric half runs on the GPU in `gpu_augment` mode — see
            # dwdata/gpu_aug.py.  It is the single most expensive thing in this
            # method and it is pure pointwise arithmetic, so it has no business
            # on a CPU core that could be fetching the next crop instead.
            if not self.gpu_augment and rng.random() < cfg.photo_p:
                rgb = photometric_jitter(rgb, cfg, rng)

        # -- targets ---------------------------------------------------------
        valid = valid & np.isfinite(hgt) & (hgt >= 0.0) & (hgt <= cfg.max_valid_height_m)
        hgt = np.where(valid, hgt, 0.0).astype(np.float32)
        # `cls` travels as uint8 and is widened to int64 on the device.  At
        # 512 px an int64 label map is 2 MB — more wire than the image itself,
        # for eight distinct values.
        seg = np.where(cls == NO_LABEL, SEG_IGNORE_INDEX, cls).astype(np.uint8)
        seg = np.clip(seg, 0, SEG_IGNORE_INDEX)

        out = {
            "target": torch.from_numpy(hgt).unsqueeze(0),
            "valid": torch.from_numpy(np.ascontiguousarray(valid)).unsqueeze(0),
            "cls": torch.from_numpy(np.ascontiguousarray(seg)),
            "gsd_m": torch.tensor(eff, dtype=torch.float32),
            "src": self.src,
            "stem": self.store.stems[ti] if ti < len(self.store.stems) else str(ti),
        }
        if self.gpu_augment:
            # HWC uint8: permuting this to NCHW on the device lands directly in
            # `channels_last`, which is the layout the model wants anyway.
            out["image_u8"] = torch.from_numpy(np.ascontiguousarray(rgb))
        else:
            out["image"] = torch.from_numpy(spec.normalise(rgb))
            out["cls"] = out["cls"].long()
            # Only the CPU path carries the export copy; the trainer never sent
            # it to the device, and in gpu_augment mode `image_u8` *is* it.
            out["rgb_u8"] = torch.from_numpy(np.ascontiguousarray(rgb))
        return out


class FullTileDataset(Dataset):
    """Whole val tiles at their native GSD — the input to the sliding-window eval.

    Training-time eval uses `TileDataset(train=False)` (one centre crop, cheap).
    The final number the deck quotes should come from *this*, because it scores
    every pixel of the tile at native resolution, which is what a DSM deliverable
    actually is.
    """

    def __init__(self, cfg, store: PackedStore, spec: PreprocSpec, src: str,
                 length: int | None = None):
        self.cfg, self.store, self.spec, self.src = cfg, store, spec, src
        self.length = int(length) if length is not None else len(store)

    def __len__(self) -> int:
        return min(self.length, len(self.store))

    def __getitem__(self, i: int) -> dict:
        rgb, hgt, cls, valid = (np.array(a, copy=True) for a in self.store.get(i))
        if self.spec.radiometric_stretch:
            lo, hi = scene_stretch_bounds(rgb, self.spec.stretch_lo_pct,
                                          self.spec.stretch_hi_pct)
            rgb = apply_stretch(rgb, lo, hi)
        valid = (valid & np.isfinite(hgt) & (hgt >= 0.0)
                 & (hgt <= self.cfg.max_valid_height_m))
        seg = np.where(cls == NO_LABEL, SEG_IGNORE_INDEX, cls).astype(np.int64)
        return {
            "rgb_u8": torch.from_numpy(np.ascontiguousarray(rgb)),
            "target": torch.from_numpy(np.where(valid, hgt, 0.0).astype(np.float32)),
            "valid": torch.from_numpy(np.ascontiguousarray(valid)),
            "cls": torch.from_numpy(np.ascontiguousarray(np.clip(seg, 0, SEG_IGNORE_INDEX))),
            "gsd_m": torch.tensor(self.store.gsd_m, dtype=torch.float32),
            "src": self.src,
            "stem": self.store.stems[i] if i < len(self.store.stems) else str(i),
        }


class UnlabeledTileDataset(Dataset):
    """Two photometric views of the same crop, for the mean-teacher branch.

    The geometry is *shared* between the views on purpose: teacher and student
    then see pixel-aligned scenes, so the consistency loss needs no warp to
    compare them and cannot be quietly satisfied by learning the warp instead of
    the content.  What differs is radiometry — a weak view for the teacher, a
    strong one for the student — which is precisely the axis the Indian/Cartosat
    domain gap lives on.

    `valid` is not returned at all: there is no supervision here, and a store
    packed by `india.pack_unlabeled` carries an all-False mask so nothing
    downstream can mistake its zeros for 0 m labels.

    Same v3 loader economics as `TileDataset`: the window is chosen before the
    read, the stretch comes from the store's cached per-tile LUT, and under
    `gpu_augment` the worker hands over one uint8 crop and lets the card draw
    both views (which is also the only way the two views cost one crop's worth
    of I/O instead of two).
    """

    def __init__(self, cfg, store: PackedStore, spec: PreprocSpec, src: str,
                 length: int | None = None, gpu_augment: bool = False):
        self.cfg, self.store, self.spec, self.src = cfg, store, spec, src
        self.gpu_augment = bool(gpu_augment)
        self.s = int(cfg.tile_size)
        self.length = int(length) if length is not None else len(store)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int) -> dict:
        cfg, spec = self.cfg, self.spec
        rng = np.random.default_rng()
        # Same locality argument as TileDataset: the tile comes from the index
        # the sampler drew, so the page cache and the per-tile LUT cache can do
        # something.  Shuffle=True on the loader supplies the randomness.
        ti = i % len(self.store)

        src_gsd = self.store.gsd_m
        H = W = self.store.tile_px

        if rng.random() < cfg.gsd_jitter_p:
            lo_g, hi_g = achievable_gsd_range(
                H, W, src_gsd, self.s, cfg.gsd_jitter_lo_m, cfg.gsd_jitter_hi_m)
            dst_gsd = float(rng.uniform(lo_g, hi_g))
        else:
            dst_gsd = float(cfg.canonical_gsd_m)
        top, left, win, eff = sample_window(H, W, src_gsd, self.s, dst_gsd, rng)

        rgb, hgt, cls, _valid = self.store.get_window(ti, top, left, win)
        if spec.radiometric_stretch:
            rgb = apply_stretch_lut(
                rgb, self.store.stretch_lut(ti, spec.stretch_lo_pct, spec.stretch_hi_pct))
        rgb, hgt, cls, _ = crop_and_scale(rgb, hgt, cls,
                                          np.zeros((win, win), bool),
                                          0, 0, win, self.s)
        k, flip = int(rng.integers(0, 4)), bool(rng.random() < 0.5)
        rgb = dihedral(rgb, k, flip)

        out = {"gsd_m": torch.tensor(eff, dtype=torch.float32), "src": self.src}
        if self.gpu_augment:
            # One crop on the wire; `GpuPreproc.two_view` draws the weak and the
            # strong jitter on the card.
            out["image_u8"] = torch.from_numpy(np.ascontiguousarray(rgb))
        else:
            weak = photometric_jitter(rgb, _WeakPhoto(cfg), rng)
            strong = photometric_jitter(rgb, cfg, rng)
            out["image_weak"] = torch.from_numpy(spec.normalise(weak))
            out["image_strong"] = torch.from_numpy(spec.normalise(strong))
        return out


class _WeakPhoto:
    """The teacher's view: same knobs, a quarter of the amplitude, no noise.

    The teacher must stay close to the distribution it was trained on or its
    pseudo-labels are worthless; the *student* is the one that has to survive a
    different sensor.
    """

    _SCALE = 0.25

    def __init__(self, cfg):
        self._cfg = cfg

    def __getattr__(self, name: str):
        v = getattr(self._cfg, name)
        if name == "photo_blur_p":
            return 0.0
        if name == "photo_noise_std":
            return 0.0
        if name.startswith("photo_") and isinstance(v, (int, float)):
            return v * self._SCALE
        return v
