"""Materialised tile store: `.npy` memmap shards.

v2 streamed every tile from the Hugging Face hub inside the DataLoader workers,
behind a bounded-LRU cache that re-`rglob`'d the whole cache directory on each
download.  With a 50 GiB cap over an ~80 GiB dataset that means constant
eviction + re-download, and the training loop spends its life waiting on the
network: the real v2 run managed 278 steps/epoch and died at 75 minutes having
seen ~1.4 M crops' worth of wall-clock but only ~20 epochs of a 30-epoch plan.

v3 splits that in two.  `prepare_data.py` materialises the tiles **once** into
contiguous `.npy` shards; training then memory-maps them, so a crop is a page
fault instead of an HTTPS round trip and the GPU is the bottleneck again.

Layout (one directory per dataset/split):

    index.json                {"tile_px", "gsd_m", "n", "has_seg", "shards": [...]}
    shard_000_rgb.npy         (n, T, T, 3) uint8
    shard_000_hgt.npy         (n, T, T)    float16   metres, nDSM/AGL
    shard_000_cls.npy         (n, T, T)    uint8     class id, 255 == unlabelled
    shard_000_val.npy         (n, T, T)    bool      source validity (no-data mask)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

NO_LABEL = 255


class ShardWriter:
    """Append tiles; rolls to a new shard every `shard_tiles`."""

    def __init__(self, out_dir: str | Path, tile_px: int, gsd_m: float,
                 shard_tiles: int = 200, has_seg: bool = True):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.tile_px = int(tile_px)
        self.gsd_m = float(gsd_m)
        self.shard_tiles = int(shard_tiles)
        self.has_seg = bool(has_seg)
        self.shards: list[dict] = []
        self._i = 0                 # index within the current shard
        self._arrays: dict | None = None
        self._stems: list[str] = []

    def _open_shard(self) -> None:
        name = f"shard_{len(self.shards):03d}"
        n, t = self.shard_tiles, self.tile_px
        mk = lambda suf, shape, dt: np.lib.format.open_memmap(  # noqa: E731
            self.dir / f"{name}_{suf}.npy", mode="w+", dtype=dt, shape=shape)
        self._arrays = {
            "name": name,
            "rgb": mk("rgb", (n, t, t, 3), np.uint8),
            "hgt": mk("hgt", (n, t, t), np.float16),
            "cls": mk("cls", (n, t, t), np.uint8),
            "val": mk("val", (n, t, t), np.bool_),
        }
        self._i = 0
        self._stems = []

    def add(self, stem: str, rgb: np.ndarray, hgt: np.ndarray,
            cls: np.ndarray | None, valid: np.ndarray | None) -> None:
        t = self.tile_px
        assert rgb.shape[:2] == (t, t), f"{stem}: rgb {rgb.shape} != {t}"
        if self._arrays is None or self._i >= self.shard_tiles:
            self.close()
            self._open_shard()
        a = self._arrays
        a["rgb"][self._i] = rgb[..., :3].astype(np.uint8)
        a["hgt"][self._i] = np.asarray(hgt, np.float32).astype(np.float16)
        a["cls"][self._i] = (np.full((t, t), NO_LABEL, np.uint8) if cls is None
                             else np.clip(cls, 0, 254).astype(np.uint8))
        v = np.isfinite(np.asarray(hgt, np.float32)) if valid is None else valid.astype(bool)
        a["val"][self._i] = v
        self._stems.append(stem)
        self._i += 1

    def close(self) -> None:
        if self._arrays is None:
            return
        a, n = self._arrays, self._i
        # trim the shard to the tiles actually written
        for suf in ("rgb", "hgt", "cls", "val"):
            p = self.dir / f"{a['name']}_{suf}.npy"
            arr = np.load(p, mmap_mode="r")[:n]
            tmp = p.with_suffix(".tmp.npy")
            np.save(tmp, np.ascontiguousarray(arr))
            del arr
            tmp.replace(p)
        self.shards.append({"file": a["name"], "n": n, "stems": self._stems})
        self._arrays = None

    def finalise(self) -> dict:
        self.close()
        idx = {
            "tile_px": self.tile_px, "gsd_m": self.gsd_m,
            "has_seg": self.has_seg,
            "n": sum(s["n"] for s in self.shards),
            "shards": self.shards,
        }
        (self.dir / "index.json").write_text(json.dumps(idx, indent=2))
        return idx


class PackedStore:
    """Read-only memmap view over one dataset/split directory."""

    def __init__(self, root: str | Path):
        self.dir = Path(root)
        self.index = json.loads((self.dir / "index.json").read_text())
        self.tile_px = int(self.index["tile_px"])
        self.gsd_m = float(self.index["gsd_m"])
        self.has_seg = bool(self.index.get("has_seg", True))
        self.stems: list[str] = []
        self._map: list[tuple[int, int]] = []       # (shard_i, row)
        for si, sh in enumerate(self.index["shards"]):
            for r in range(sh["n"]):
                self._map.append((si, r))
            self.stems.extend(sh["stems"])
        # memmaps are opened lazily and per-process, so DataLoader workers each
        # get their own handles instead of inheriting a half-consumed one.
        self._mm: dict[tuple[int, str], np.ndarray] = {}

    def __len__(self) -> int:
        return len(self._map)

    def _arr(self, si: int, suf: str) -> np.ndarray:
        key = (si, suf)
        a = self._mm.get(key)
        if a is None:
            name = self.index["shards"][si]["file"]
            a = self._mm[key] = np.load(self.dir / f"{name}_{suf}.npy", mmap_mode="r")
        return a

    def get(self, i: int):
        si, r = self._map[i]
        rgb = np.asarray(self._arr(si, "rgb")[r])
        hgt = np.asarray(self._arr(si, "hgt")[r], dtype=np.float32)
        cls = np.asarray(self._arr(si, "cls")[r])
        val = np.asarray(self._arr(si, "val")[r])
        return rgb, hgt, cls, val

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_mm"] = {}                # never pickle memmaps into a worker
        return d


def store_exists(root: str | Path) -> bool:
    return (Path(root) / "index.json").is_file()
