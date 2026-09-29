import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config  # noqa: E402
from dwdata.packed import ShardWriter  # noqa: E402


@pytest.fixture
def cfg():
    c = Config()
    c.tile_size = 64
    c.smoke = True
    return c.apply_smoke().validate()


@pytest.fixture
def store(tmp_path):
    """A tiny packed store: flat ground with one 12 m block."""
    from dwdata.packed import PackedStore

    d = tmp_path / "gamus" / "train"
    w = ShardWriter(d, tile_px=256, gsd_m=0.33, shard_tiles=4)
    rng = np.random.default_rng(0)
    for i in range(6):
        rgb = (rng.random((256, 256, 3)) * 60 + 90).astype(np.uint8)
        h = np.zeros((256, 256), np.float32)
        h[64:160, 64:160] = 12.0
        rgb[64:160, 64:160] = 210
        w.add(f"t{i}", rgb, h, np.where(h > 1, 2, 0).astype(np.uint8),
              np.ones((256, 256), bool))
    w.finalise()
    return PackedStore(d)
