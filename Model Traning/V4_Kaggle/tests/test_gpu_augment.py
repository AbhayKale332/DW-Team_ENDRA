"""The uint8 hand-over path — the one the H100 run actually uses.

`Config.apply_smoke()` turns `gpu_augment` off, so every other end-to-end test
here exercises the all-CPU fallback.  That is exactly how the v4 draft came to
ship without `dwdata/gpu_aug.py` at all while its tests stayed green, so this
file drives the real path: loader hands over uint8, `GpuPreproc` normalises and
jitters on the device, training and evaluation both consume it.

`GpuPreproc` is device-agnostic, so all of this runs on CPU in CI and is the
same code the card runs.
"""
import sys
from pathlib import Path

import numpy as np
import torch

from config import Config
from dwdata.dataset import TileDataset
from dwdata.gpu_aug import GpuPreproc
from dwdata.preprocess import PreprocSpec
from tests.stub_encoder import use_stub

DEV = torch.device("cpu")


def _spec(cfg):
    return PreprocSpec.from_config(cfg, resolve_stats=False)


def test_worker_hands_over_uint8_not_float(cfg, store):
    """The whole point of the mode: 0.79 MB a sample on the wire, not 3.1 MB."""
    ds = TileDataset(cfg, store, _spec(cfg), "gamus", train=True, length=8,
                     gpu_augment=True)
    s = ds[0]
    assert "image_u8" in s and "image" not in s
    assert s["image_u8"].dtype == torch.uint8
    assert s["image_u8"].shape == (cfg.tile_size, cfg.tile_size, 3)
    assert s["cls"].dtype == torch.uint8, "an int64 label map is more wire than the image"
    assert "rgb_u8" not in s, "image_u8 already is the export copy"


def test_gpu_preproc_reproduces_the_cpu_normalisation(cfg, store):
    """`normalise` is part of the inference contract, so the device path has to
    agree with the numpy one to float rounding — jitter off, eval semantics."""
    spec = _spec(cfg)
    cpu_ds = TileDataset(cfg, store, spec, "gamus", train=False, length=4)
    gpu_ds = TileDataset(cfg, store, spec, "gamus", train=False, length=4,
                         gpu_augment=True)
    prep = GpuPreproc(spec, cfg, DEV)
    for i in range(4):
        a = cpu_ds[i]["image"]
        b = prep({"image_u8": gpu_ds[i]["image_u8"].unsqueeze(0)},
                 train=False)["image"][0]
        assert torch.allclose(a, b, atol=2e-6), float((a - b).abs().max())


def test_jitter_is_gated_and_bounded(cfg, store):
    spec = _spec(cfg)
    prep = GpuPreproc(spec, cfg, DEV)
    x = torch.rand(16, 3, 32, 32)
    y = prep.jitter(x.clone())
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert (y >= 0).all() and (y <= 1).all(), "jitter must stay in [0, 1]"
    assert not torch.allclose(y, x), "photo_p=0.9 — something should have moved"


def test_two_view_is_weak_and_strong_off_one_crop(cfg, store):
    """Teacher and student must see the same geometry and different radiometry."""
    spec = _spec(cfg)
    prep = GpuPreproc(spec, cfg, DEV)
    u8 = torch.randint(40, 200, (4, 32, 32, 3), dtype=torch.uint8)
    out = prep.two_view({"image_u8": u8.clone()})
    assert set(out) == {"image_weak", "image_strong"}
    w, s = out["image_weak"], out["image_strong"]
    assert w.shape == s.shape == (4, 3, 32, 32)
    assert torch.isfinite(w).all() and torch.isfinite(s).all()
    assert not torch.allclose(w, s)

    # the weak view is the gentler one: measured against the un-jittered image
    base = prep({"image_u8": u8.clone()}, train=False)["image"]
    dw = float((w - base).abs().mean())
    dsg = float((s - base).abs().mean())
    assert dw < dsg, f"weak ({dw:.4f}) should perturb less than strong ({dsg:.4f})"


def test_train_and_eval_end_to_end_on_the_uint8_path(tmp_path, store, monkeypatch):
    """Two epochs with gpu_augment on, through the real trainer.

    This is the regression test for the v4 draft: it dropped `gpu_aug.py`, the
    `_HOST_ONLY` skip, the windowed reads and the cached stretch bounds, and no
    test noticed because every test ran the fallback.
    """
    undo = use_stub(hidden=32, patch=16, layers=8)
    try:
        c = Config()
        c.smoke = True
        c.apply_smoke()
        c.gpu_augment = True                 # the thing under test
        c.tile_size, c.decoder_dim, c.n_bins = 64, 32, 8
        c.epochs, c.freeze_epochs, c.crops_per_epoch = 2, 1, 4
        c.batch_size, c.grad_accum, c.val_tiles, c.num_workers = 2, 1, 2, 0
        c.amp = c.channels_last = c.cudnn_benchmark = False
        c.make_zip = c.make_figures = c.make_report = c.export_onnx = False
        c.n_qualitative = 1
        c.eval_every, c.final_sliding_eval, c.ema_decay = 1, True, 0.99
        c.datasets = "gamus"
        c.data_root = str(Path(store.dir).parents[1])
        c.output_dir = str(tmp_path / "out")
        c.validate()

        monkeypatch.setattr(sys, "argv", ["train.py"])
        import train as train_mod

        monkeypatch.setattr(train_mod, "parse_config", lambda *a, **k: c)
        import dwdata.preprocess as pp

        monkeypatch.setattr(pp, "resolve_encoder_stats",
                            lambda *a, **k: (pp.DINOV3_SAT_MEAN, pp.DINOV3_SAT_STD))
        train_mod.main()

        out = Path(c.output_dir)
        log = (out / "run.log").read_text()
        assert "GPU augmentation on" in log, "the uint8 path never engaged"
        assert "img/s" in log, "the throughput counter is how loader stalls get seen"
        assert (out / "best.pt").is_file()

        import json
        m = json.loads((out / "metrics.json").read_text())
        assert np.isfinite(m["best_val_rmse_m"]), "eval did not survive the uint8 path"
        assert np.isfinite(m["final_plain"]["global"]["rmse_m"])
        assert np.isfinite(m["final_tta"]["global"]["rmse_m"])
        assert np.isfinite(m["final_sliding_tta"]["global"]["rmse_m"])
        # the qualitative/viewer exports read FullTileDataset, which is never on
        # the uint8 path — they must still work
        assert (out / "viewer_sample" / "pred_ndsm_m.npy").is_file()
    finally:
        undo()


def test_stretch_bounds_are_computed_once_and_cached(store, cfg):
    """`_prime` is what moved a full-tile histogram out of every crop."""
    from dwdata.loaders import _prime

    spec = _spec(cfg)
    p = store._bounds_path(spec.stretch_lo_pct, spec.stretch_hi_pct)
    assert not p.is_file()
    _prime(store, spec, cfg, "gamus/train")
    assert p.is_file(), "bounds were not persisted next to the shards"
    tbl = np.load(p)
    assert tbl.shape == (len(store), 2, 3)
    assert (tbl[:, 1] > tbl[:, 0]).all(), "hi must exceed lo on every channel"
    # and the crop path now reads them instead of recomputing
    lut = store.stretch_lut(0, spec.stretch_lo_pct, spec.stretch_hi_pct)
    assert lut.shape == (256, 3) and lut.dtype == np.uint8


def test_unlabeled_loader_on_the_uint8_path(tmp_path, store):
    """The mean-teacher branch under gpu_augment: one crop on the wire, two
    views drawn on the device.  Without this the branch would be the one caller
    of `GpuPreproc.two_view` and nothing would ever call it."""
    from dwdata.loaders import build_unlabeled_loader
    from dwdata.packed import PackedStore, ShardWriter

    d = Path(store.dir).parents[1] / "india_unlabeled" / "train"
    w = ShardWriter(d, tile_px=256, gsd_m=0.56, shard_tiles=4, has_seg=False)
    rng = np.random.default_rng(3)
    for i in range(4):
        rgb = (rng.random((256, 256, 3)) * 90 + 60).astype(np.uint8)
        w.add(f"india{i}", rgb, np.zeros((256, 256), np.float32), None,
              np.zeros((256, 256), bool))
    w.finalise()
    PackedStore(d)

    c = Config()
    c.smoke = True
    c.apply_smoke()
    c.gpu_augment = True
    c.tile_size, c.num_workers, c.batch_size = 64, 0, 2
    c.unlabeled_batch_frac = 1.0
    c.data_root = str(Path(store.dir).parents[1])
    spec = PreprocSpec(tile_size=64, canonical_gsd_m=0.5)

    dl = build_unlabeled_loader(c, spec)
    assert dl is not None
    b = next(iter(dl))
    assert "image_u8" in b and "image_weak" not in b, \
        "under gpu_augment the worker must ship one uint8 crop, not two floats"
    assert "target" not in b and "valid" not in b, \
        "an unlabeled batch must not carry anything a loss could mistake for a label"

    out = GpuPreproc(spec, c, DEV).two_view(dict(b))
    assert out["image_weak"].shape == out["image_strong"].shape == (2, 3, 64, 64)
    assert not torch.allclose(out["image_weak"], out["image_strong"])
