"""SynRS3D loader (synthetic pretrain).

Source: `JTRNEO/SynRS3D` on HF — 17 zip archives (4-16 GB each, 102 GB total),
NOT per-tile streamable.  v2 downloads a rotating subset of archives via
`BoundedCacheHF.ensure_archive`, extracts, reads `.tif` tiles, and lets the LRU
cache evict whole archives as it goes.  69,667 tiles, 512x512, GSD 0.05-1 m.

Archive member layout:
    opt/<name>.tif        RGB
    gt_nDSM/<name>.tif     nDSM, metres
    gt_ss_mask/<name>.tif  8-class land cover

SynRS3D 8-class -> GAMUS 7-class:
    1 Bareland->ground  2 Rangeland->veg  3 Developed->ground  4 Road->road
    5 Trees->veg  6 Water->water  7 Agriculture->veg  8 Buildings->building  0->other
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import TileDatasetBase
from .streaming import BoundedCacheHF

_SYN_TO_GAMUS = {0: 7, 1: 0, 2: 1, 3: 0, 4: 4, 5: 1, 6: 3, 7: 1, 8: 2}

# rough GSD prior per archive family; GSD jitter dominates during pretrain anyway
_ARCHIVE_GSD = {"g005": 0.3, "g05": 0.7}

ARCHIVES = [
    "SynRS3D/grid_g005_low_v1.zip", "SynRS3D/grid_g005_mid_v1.zip",
    "SynRS3D/grid_g005_mid_v2.zip", "SynRS3D/grid_g005_high_v1.zip",
    "SynRS3D/grid_g05_low_v1.zip", "SynRS3D/grid_g05_mid_v1.zip",
    "SynRS3D/grid_g05_mid_v2.zip", "SynRS3D/grid_g05_high_v1.zip",
    "SynRS3D/terrain_g005_low_v1.zip", "SynRS3D/terrain_g005_mid_v1.zip",
    "SynRS3D/terrain_g05_low_v1.zip", "SynRS3D/terrain_g05_mid_v1.zip",
]


def _read_tif(path: Path) -> np.ndarray:
    try:
        import tifffile

        return tifffile.imread(path)
    except Exception:  # noqa: BLE001
        from PIL import Image

        return np.asarray(Image.open(path))


def _archive_gsd(rel: str) -> float:
    for k, v in _ARCHIVE_GSD.items():
        if f"_{k}_" in rel:
            return v
    return 0.5


class SynRS3DArchiveDataset(TileDatasetBase):
    """Reads whatever archives are currently extracted in the cache.

    The trainer calls `ensure_ready(k)` before each epoch to rotate `k` archives
    into the cache; the dataset then indexes every tile under those dirs.
    """

    def __init__(self, cfg, cache: BoundedCacheHF, train: bool = True):
        super().__init__(cfg, [], "synrs3d", train)
        self.cache = cache
        self._tiles: list[tuple[Path, float]] = []  # (opt_path, gsd)

    def ensure_ready(self, n_archives: int, epoch: int) -> None:
        pick = [ARCHIVES[(epoch * n_archives + i) % len(ARCHIVES)] for i in range(n_archives)]
        tiles: list[tuple[Path, float]] = []
        for rel in pick:
            try:
                d = self.cache.ensure_archive(rel)
            except Exception as e:  # noqa: BLE001
                print(f"[synrs3d] skip {rel}: {e}")
                continue
            gsd = _archive_gsd(rel)
            for opt in sorted(d.rglob("opt/*.tif")):
                tiles.append((opt, gsd))
        if self.cfg.pretrain_tiles and len(tiles) > self.cfg.pretrain_tiles:
            rng = np.random.default_rng(self.cfg.seed + epoch)
            idx = rng.choice(len(tiles), self.cfg.pretrain_tiles, replace=False)
            tiles = [tiles[i] for i in idx]
        self._tiles = tiles
        self.stems = [p.stem for p, _ in tiles]

    def load_tile(self, idx: int):
        opt, gsd = self._tiles[idx]
        ndsm_p = opt.parent.parent / "gt_nDSM" / opt.name
        seg_p = opt.parent.parent / "gt_ss_mask" / opt.name

        rgb = np.asarray(_read_tif(opt))[..., :3].astype(np.uint8)
        ndsm = _read_tif(ndsm_p).astype(np.float32)
        if seg_p.exists():
            seg_raw = _read_tif(seg_p).astype(np.int64)
            seg = np.vectorize(lambda v: _SYN_TO_GAMUS.get(int(v), 7))(seg_raw)
            has_seg = True
        else:
            seg = np.full(ndsm.shape, 7, np.int64)
            has_seg = False
        return rgb, ndsm, seg, gsd, has_seg
