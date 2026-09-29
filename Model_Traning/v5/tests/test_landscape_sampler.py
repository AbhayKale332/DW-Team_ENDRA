"""`landscape_sampler_boost`: forest / sparse tiles drawn more, store mix unchanged."""

import numpy as np
import pytest

from config import LANDSCAPE_NAMES, Config
from dwdata.packed import PackedStore, ShardWriter
from dwdata.preprocess import PreprocSpec


def _heights(kind: str, t: int, rng) -> np.ndarray:
    h = np.zeros((t, t), np.float32)
    if kind == "forested":      # rough canopy over most of the tile
        h[:, : int(0.8 * t)] = rng.uniform(5, 20, (t, int(0.8 * t)))
    elif kind == "urban":       # flat roofs: tall, but smooth
        for y in range(0, t, t // 4):
            h[y + 4: y + t // 8 + 4, 8: t - 8] = 12.0
    return h                    # "sparse": bare ground


def _store(root, name, kinds, t=128, gsd=0.5):
    d = root / name / "train"
    w = ShardWriter(d, tile_px=t, gsd_m=gsd, shard_tiles=4, has_seg=False)
    rng = np.random.default_rng(0)
    for i, k in enumerate(kinds):
        w.add(f"{k}{i}", np.full((t, t, 3), 120, np.uint8), _heights(k, t, rng), None,
              np.ones((t, t), bool))
    w.finalise()
    return PackedStore(d)


@pytest.fixture
def lcfg(tmp_path):
    c = Config()
    c.tile_size = 64
    c.smoke = True
    c = c.apply_smoke().validate()
    c.data_root = str(tmp_path)
    c.output_dir = str(tmp_path / "out")
    c.num_workers = 0
    c.gpu_augment = False
    c.radiometric_stretch = False
    return c


def test_tile_classes_and_cache(tmp_path, lcfg):
    from dwdata.loaders import LANDSCAPE_CACHE, tile_landscapes

    st = _store(tmp_path, "gamus", ["forested", "sparse", "urban", "forested"])
    cls = tile_landscapes(st, lcfg, "gamus/train")
    names = [LANDSCAPE_NAMES[i] for i in cls]
    assert names == ["forested", "sparse", "urban", "forested"]
    assert (st.dir / LANDSCAPE_CACHE).is_file()
    # the cache is what the second call returns
    np.save(st.dir / LANDSCAPE_CACHE, np.zeros(4, np.int8))
    assert (tile_landscapes(st, lcfg, "gamus/train") == 0).all()


def test_classes_measured_at_canonical_gsd(tmp_path, lcfg):
    """A 0.25 m store is block-averaged to 0.5 m before classifying."""
    from dwdata.loaders import tile_landscapes

    st = _store(tmp_path, "gamus", ["sparse", "urban"], t=256, gsd=0.25)
    names = [LANDSCAPE_NAMES[i] for i in tile_landscapes(st, lcfg, "gamus/train")]
    assert names == ["sparse", "urban"]


def test_weights_have_mean_one_and_follow_the_boost():
    from dwdata.loaders import landscape_tile_weights

    f, s, u = (LANDSCAPE_NAMES.index(n) for n in ("forested", "sparse", "urban"))
    w = landscape_tile_weights(np.array([f, s, u, u]), {"forested": 2.0, "sparse": 1.5})
    assert w.mean() == pytest.approx(1.0)
    assert w[0] / w[2] == pytest.approx(2.0)
    assert w[1] / w[2] == pytest.approx(1.5)


def test_sampler_keeps_store_mix(tmp_path, lcfg):
    from dwdata.loaders import build_loaders

    _store(tmp_path, "gamus", ["forested", "urban", "urban", "urban"])
    _store(tmp_path, "us3d", ["sparse", "urban"])
    lcfg.datasets = "gamus,us3d"
    lcfg.sampler_weights = "gamus:3,us3d:1"
    lcfg.landscape_sampler_boost = "forested:2,sparse:1.5"
    spec = PreprocSpec.from_config(lcfg)
    dl, _, _ = build_loaders(lcfg, spec)
    w = np.asarray(dl.sampler.weights, np.float64)
    ga, us = w[:4], w[4:]
    assert ga.sum() / us.sum() == pytest.approx(3.0)          # mix unchanged
    assert ga[0] / ga[1] == pytest.approx(2.0)                # forest tile boosted
    assert us[0] / us[1] == pytest.approx(1.5)                # sparse tile boosted


def test_boost_off_is_uniform_within_store(tmp_path, lcfg):
    from dwdata.loaders import LANDSCAPE_CACHE, build_loaders

    st = _store(tmp_path, "gamus", ["forested", "urban"])
    lcfg.datasets = "gamus"
    dl, _, _ = build_loaders(lcfg, PreprocSpec.from_config(lcfg))
    w = np.asarray(dl.sampler.weights)
    assert w[0] == pytest.approx(w[1])
    assert not (st.dir / LANDSCAPE_CACHE).exists()            # nothing computed


def test_bad_boost_class_is_rejected():
    c = Config()
    c.landscape_sampler_boost = "forest:2"
    with pytest.raises(ValueError):
        c.validate()
