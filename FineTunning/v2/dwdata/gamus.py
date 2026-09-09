"""GAMUS loader (streamed).

earthflow/GAMUS layout (verified in v1):
    images/<split>/<STEM>_RGB.h5   key "image" uint8   (1024,1024,3)
    heights/<split>/<STEM>_AGL.h5  key "image" float32 (1024,1024)  metres AGL == nDSM
    classes/<split>/<STEM>_CLS.h5  key "image" float32 (1024,1024)  class id {0..6}
"""

from __future__ import annotations

import numpy as np

from .base import TileDatasetBase
from .streaming import BoundedCacheHF, wait_for

_SUB = {"rgb": ("images", "RGB"), "agl": ("heights", "AGL"), "cls": ("classes", "CLS")}


def _read_h5(path) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as f:
        key = "image" if "image" in f else list(f.keys())[0]
        return np.asarray(f[key][()])


def make_cache(cfg, token: str | None) -> BoundedCacheHF:
    return BoundedCacheHF(
        cfg.gamus_repo, cfg.cache_dir, int(cfg.cache_max_gib * 1024**3), token=token
    )


def list_stems(cache: BoundedCacheHF, split: str) -> list[str]:
    return cache.list_stems(f"images/{split}/", "_RGB.h5")


class GamusDataset(TileDatasetBase):
    def __init__(self, cfg, cache: BoundedCacheHF, split: str, stems: list[str], train: bool):
        super().__init__(cfg, stems, "gamus", train)
        self.cache = cache
        self.split = split

    def load_tile(self, idx: int):
        stem = self.stems[idx]
        p_rgb = wait_for(lambda: self.cache.get(f"images/{self.split}/{stem}_RGB.h5"))
        p_agl = wait_for(lambda: self.cache.get(f"heights/{self.split}/{stem}_AGL.h5"))
        try:
            p_cls = wait_for(lambda: self.cache.get(f"classes/{self.split}/{stem}_CLS.h5"))
            cls = _read_h5(p_cls).astype(np.int64)
            has_seg = True
        except Exception:  # noqa: BLE001
            cls = None
            has_seg = False

        rgb = _read_h5(p_rgb).astype(np.uint8)
        agl = _read_h5(p_agl).astype(np.float32)
        if cls is None:
            cls = np.zeros(agl.shape, np.int64)
        return rgb, agl, cls, self.cfg.gamus_gsd_m, has_seg
