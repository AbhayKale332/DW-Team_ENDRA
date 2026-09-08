"""Shared dataset plumbing for v2 loaders (GAMUS / GeoNRW / SynRS3D).

Each concrete dataset yields the same dict so the trainer is source-agnostic:

    image   : (3, S, S) float32   ImageNet-normalised RGB
    rgb_u8  : (S, S, 3) uint8     raw crop (viewer export / qualitative)
    target  : (1, S, S) float32   nDSM in metres (>= 0)
    valid   : (1, S, S) bool      finite & in-range target
    cls     : (S, S)    int64     land-cover class id  (7 = "other"/ignore-ish)
    has_seg : ()        bool      whether cls is real supervision
    src     : str                 dataset name
    stem    : str                 tile id
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalise_rgb(rgb_u8: np.ndarray) -> np.ndarray:
    x = (rgb_u8.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(x, (2, 0, 1)).copy()


def random_crop_coords(h: int, w: int, s: int, rng: np.random.Generator) -> tuple[int, int]:
    return int(rng.integers(0, max(1, h - s + 1))), int(rng.integers(0, max(1, w - s + 1)))


def dihedral(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, k)
    if flip:
        out = np.fliplr(out)
    return np.ascontiguousarray(out)


def pack_sample(
    rgb_u8: np.ndarray,
    height: np.ndarray,
    seg: np.ndarray,
    *,
    max_h: float,
    has_seg: bool,
    src: str,
    stem: str,
) -> dict:
    valid = np.isfinite(height) & (height >= 0.0) & (height <= max_h)
    height = np.where(valid, height, 0.0).astype(np.float32)
    seg = np.clip(seg.astype(np.int64), 0, 7)
    return {
        "image": torch.from_numpy(normalise_rgb(rgb_u8)),
        "rgb_u8": torch.from_numpy(np.ascontiguousarray(rgb_u8.astype(np.uint8))),
        "target": torch.from_numpy(height).unsqueeze(0),
        "valid": torch.from_numpy(valid).unsqueeze(0),
        "cls": torch.from_numpy(np.ascontiguousarray(seg)),
        "has_seg": torch.tensor(bool(has_seg)),
        "src": src,
        "stem": stem,
    }


class TileDatasetBase(Dataset):
    """Common crop + GSD-jitter + dihedral-augment pipeline.

    Subclasses implement `load_tile(idx) -> (rgb_u8 HxWx3, height HxW, seg HxW,
    src_gsd_m, has_seg)`.
    """

    def __init__(self, cfg, stems: list[str], src_name: str, train: bool):
        self.cfg = cfg
        self.stems = stems
        self.src = src_name
        self.train = train
        self.s = cfg.tile_size

    def __len__(self) -> int:
        return len(self.stems)

    # subclass hook
    def load_tile(self, idx: int):  # pragma: no cover - abstract
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict:
        from .gsd import center_crop_or_pad, jitter_gsd, rescale_to_gsd

        # A single missing/corrupt tile (streaming cache eviction race, killed
        # download) must not abort a training stage — fall back to a nearby
        # index a bounded number of times.
        for _try in range(8):
            try:
                rgb, height, seg, src_gsd, has_seg = self.load_tile(idx)
                break
            except (FileNotFoundError, OSError) as e:
                nxt = (idx * 1_000_003 + 17) % max(1, len(self.stems))
                print(f"[{self.src}] tile {idx} unreadable ({e}); retrying with {nxt}")
                idx = nxt
        else:
            rgb, height, seg, src_gsd, has_seg = self.load_tile(idx)
        rng = np.random.default_rng(self.cfg.seed * 2_654_435_761 + idx * 2 + int(self.train))

        chosen_gsd = src_gsd
        if self.train and rng.random() < self.cfg.gsd_jitter_p:
            rgb, height, seg, chosen_gsd = jitter_gsd(
                rgb, height, seg, src_gsd,
                self.cfg.gsd_jitter_min_m, self.cfg.gsd_jitter_max_m, rng,
            )
        elif not self.train:
            rgb, height, seg = rescale_to_gsd(
                rgb, height, seg, src_gsd, self.cfg.canonical_gsd_m
            )
            chosen_gsd = self.cfg.canonical_gsd_m

        h, w = height.shape
        s = self.s
        if self.train and (h > s and w > s):
            top, left = random_crop_coords(h, w, s, rng)
            sl = (slice(top, top + s), slice(left, left + s))
            rgb, height, seg = rgb[sl], height[sl], seg[sl]
        else:
            rgb = center_crop_or_pad(rgb, s)
            height = center_crop_or_pad(height, s)
            seg = center_crop_or_pad(seg, s, pad_value=7)

        if self.train:
            k = int(rng.integers(0, 4))
            flip = bool(rng.random() < 0.5)
            rgb, height, seg = (dihedral(a, k, flip) for a in (rgb, height, seg))

        sample = pack_sample(
            rgb, height, seg,
            max_h=self.cfg.max_valid_height_m,
            has_seg=has_seg, src=self.src, stem=self.stems[idx],
        )
        sample["gsd_m"] = float(chosen_gsd)
        return sample
