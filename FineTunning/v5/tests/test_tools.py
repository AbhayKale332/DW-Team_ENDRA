"""tools/audit_labels.py, tools/cartosat_to_rgb.py and eval_test.py.

All on tiny synthetic stores / GeoTIFFs written by the test, never real data.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CLASS_NAMES, Config, safe_config_dict  # noqa: E402
from dwdata.packed import ShardWriter  # noqa: E402
from tests.stub_encoder import use_stub  # noqa: E402


# ---------------------------------------------------------------------
# audit_labels
# ---------------------------------------------------------------------
def _scene(rng, tile, coarse: int = 1, veg_flat: bool = False):
    """rgb, height, GAMUS-id cls: blocks (buildings, grey), trees (green), ground."""
    h = np.full((tile, tile), 0.4, np.float32)          # ground ~0.4 m, as in GAMUS
    cls = np.full((tile, tile), CLASS_NAMES.index("ground"), np.uint8)
    rgb = np.zeros((tile, tile, 3), np.uint8)
    rgb[:] = (150, 140, 120)
    for _ in range(6):
        y, x = rng.integers(0, tile - 40, 2)
        s = int(rng.integers(12, 36))
        h[y:y + s, x:x + s] = rng.uniform(5, 20)
        cls[y:y + s, x:x + s] = CLASS_NAMES.index("building")
        rgb[y:y + s, x:x + s] = (200, 200, 205)
    for _ in range(6):
        y, x = rng.integers(0, tile - 20, 2)
        s = int(rng.integers(6, 16))
        h[y:y + s, x:x + s] = 0.2 if veg_flat else rng.uniform(4, 12)
        cls[y:y + s, x:x + s] = CLASS_NAMES.index("tree")
        rgb[y:y + s, x:x + s] = (40, 150, 40)
    wy, wx = rng.integers(0, tile - 20, 2)
    cls[wy:wy + 12, wx:wx + 12] = CLASS_NAMES.index("water")
    rgb[wy:wy + 12, wx:wx + 12] = (30, 50, 90)
    h[wy:wy + 12, wx:wx + 12] = 0.0
    h += rng.normal(0, 0.05, h.shape).astype(np.float32)
    h = np.maximum(h, 0)
    if coarse > 1:
        from tools.audit_labels import pool_up
        h = pool_up(h, coarse).astype(np.float32)
    rgb = np.clip(rgb.astype(int) + rng.integers(-8, 9, rgb.shape), 0, 255).astype(np.uint8)
    return rgb, h, cls


def _store(root, name, split, n, tile=128, gsd=0.33, seed=0, **kw):
    w = ShardWriter(root / name / split, tile_px=tile, gsd_m=gsd, shard_tiles=64)
    rng = np.random.default_rng(seed)
    for i in range(n):
        rgb, h, cls = _scene(rng, tile, **kw)
        w.add(f"{name}_{split}_{i}", rgb, h, cls, np.ones((tile, tile), bool))
    w.finalise()


def test_audit_finds_coarse_labels_flat_trees_and_pins_classes(tmp_path):
    from tools.audit_labels import main

    root = tmp_path / "data"
    _store(root, "gamus", "train", 12, seed=1)
    _store(root, "gamus", "test", 4, seed=2)
    _store(root, "dfc23_g050", "train", 12, seed=3, coarse=4, veg_flat=True)
    rep = main(["--data_root", str(root), "--out", str(tmp_path / "audit"),
                "--tiles", "12"])

    res = rep["resolution"]
    assert res["recommended_pool"]["dfc23_g050"] == 4, res
    assert rep["flags"]["coarse_pool"] == "4"
    assert rep["flags"]["coarse_label_m"] == str(round(4 * 0.33, 2))   # in metres
    veg = rep["vegetation"]
    assert veg["coarse_mask_veg"] is True
    assert veg["green_pixel_heights"]["dfc23_g050"]["median_m"] < 1.0
    assert veg["calibration"]["f1"] > 0.8
    assert rep["classes"]["pinning_ok"] is True, rep["classes"]["checks"]
    assert list((tmp_path / "audit" / "class_overlays").glob("id6_tree_*.png"))
    assert all(v["shared_stems"] == 0 and v["shared_content"] == 0
               for v in rep["overlaps"].values())
    assert (tmp_path / "audit" / "audit.md").read_text().startswith("# Label audit")


def test_audit_catches_a_retitled_duplicate_tile(tmp_path):
    from tools.audit_labels import audit_overlaps

    root = tmp_path / "data"
    _store(root, "gamus", "train", 4, seed=5)
    _store(root, "gamus", "test", 4, seed=5)       # same pixels, new stems
    out = audit_overlaps(root)
    k = "gamus/test vs gamus/train"
    assert out[k]["shared_stems"] == 0 and out[k]["shared_content"] == 4


def test_pansharp_match_recovers_a_known_chroma_downsample():
    from tools.audit_labels import chroma_luma_ratio, match_pansharp_factor, simulate_pansharp

    rng = np.random.default_rng(0)
    tiles = [_scene(rng, 128)[0] for _ in range(6)]
    target = np.median([chroma_luma_ratio(simulate_pansharp(t, 2.75)) for t in tiles])
    m = match_pansharp_factor(float(target), tiles)
    assert abs(m["factor"] - 2.75) <= 0.25, m


# ---------------------------------------------------------------------
# cartosat_to_rgb
# ---------------------------------------------------------------------
rasterio = pytest.importorskip("rasterio")


def _texture(h, w, seed=0):
    from tools.audit_labels import box_blur

    rng = np.random.default_rng(seed)
    a = box_blur(rng.random((h, w)).astype(np.float32), 2)
    return (a - a.min()) / (np.ptp(a) + 1e-6)


def _write_pair(d: Path, shift_px=(0.0, 0.0)):
    """PAN 0.6 m (512^2) and 4-band MX 1.6 m, same ground; MX georeferencing
    deliberately off by `shift_px` PAN pixels (dy, dx)."""
    from rasterio.transform import from_origin

    x0, y0 = 500000.0, 2250000.0
    base = _texture(1536, 1536)                     # 0.2 m "truth"
    pan = base.reshape(512, 3, 512, 3).mean((1, 3))
    (d / "pan").mkdir(parents=True)
    with rasterio.open(d / "pan" / "BAND.tif", "w", driver="GTiff", height=512, width=512,
                       count=1, dtype="uint16", crs="EPSG:32645",
                       transform=from_origin(x0, y0, 0.6, 0.6), nodata=0) as ds:
        ds.write((200 + 1500 * pan).astype(np.uint16), 1)
    (d / "pan" / "BAND_META.txt").write_text("Sensor=PAN\nProcessingLevel=GEOREF\n")
    # MX grid: 1.6 m = 8 truth px; 1536/8 = 192 px
    mx = base.reshape(192, 8, 192, 8).mean((1, 3))
    dy, dx = shift_px
    tr = from_origin(x0 - dx * 0.6, y0 + dy * 0.6, 1.6, 1.6)
    (d / "mx").mkdir(parents=True)
    for b, gain in ((1, 0.8), (2, 0.9), (3, 1.0), (4, 1.2)):
        with rasterio.open(d / "mx" / f"BAND{b}.tif", "w", driver="GTiff", height=192,
                           width=192, count=1, dtype="uint16", crs="EPSG:32645",
                           transform=tr, nodata=0) as ds:
            ds.write((100 + 900 * gain * mx).astype(np.uint16), 1)
    (d / "mx" / "BAND_META.txt").write_text("Sensor=MX\nProcessingLevel=ORTHO\n")


def test_phase_correlation_sign_and_subpixel():
    from tools.cartosat_to_rgb import phase_correlate

    a = _texture(256, 256, 3)
    b = np.roll(a, (-5, 3), axis=(0, 1))            # b at (y, x) shows a at (y+5, x-3)
    dy, dx, ratio = phase_correlate(a, b)
    assert abs(dy - 5) < 0.3 and abs(dx + 3) < 0.3 and ratio > 8


def test_measured_shift_is_removed_and_output_is_rgb(tmp_path):
    from tools.cartosat_to_rgb import main, measure_shift

    _write_pair(tmp_path, shift_px=(4.0, -6.0))
    m = measure_shift(str(tmp_path / "pan" / "BAND.tif"),
                      [str(tmp_path / "mx" / f"BAND{b}.tif") for b in range(1, 5)],
                      size=384, n_windows=1)
    assert m["ok"] and m["min_peak_ratio"] > 8
    # the MX origin sits 2.4 m north / 3.6 m east of the truth, so on the PAN
    # grid its content shows up 4 rows high and 6 cols east: it must move
    # (+4 rows, -6 cols), i.e. 2.4 m south and 3.6 m west.
    assert abs(m["dy_px"] - 4.0) < 0.3 and abs(m["dx_px"] + 6.0) < 0.3, m
    assert m["shift_north_m"] == pytest.approx(-2.4, abs=0.2)
    assert m["shift_east_m"] == pytest.approx(-3.6, abs=0.2)

    out = tmp_path / "rgb.tif"
    rep = main(["--pan", str(tmp_path / "pan"), "--mx", str(tmp_path / "mx"),
                "--out", str(out), "--engine", "python", "--window", "384",
                "--windows", "1"])
    assert rep["applied_shift_m"] != [0.0, 0.0]
    with rasterio.open(out) as ds:
        assert ds.count == 3 and ds.dtypes[0] == "uint16" and ds.nodata == 0
        assert [c.name for c in ds.colorinterp] == ["red", "green", "blue"]
        assert abs(ds.transform.a - 0.6) < 1e-9
        rgb = ds.read()
    # after registration the sharpened red band tracks PAN closely
    with rasterio.open(tmp_path / "pan" / "BAND.tif") as ps:
        pan = ps.read(1).astype(np.float32)
    v = rgb[0] > 0
    r = np.corrcoef(rgb[0][v].astype(np.float32), pan[v])[0, 1]
    assert r > 0.9
    # and it reads back through the normal scene reader as R, G, B
    from dwdata.scene_io import SceneSource

    src = SceneSource(out)
    assert src._band_idx == [1, 2, 3]


# ---------------------------------------------------------------------
# eval_test
# ---------------------------------------------------------------------
def _eval_store(root, split, n, tile=64):
    w = ShardWriter(root / "gamus" / split, tile_px=tile, gsd_m=0.5, shard_tiles=64)
    rng = np.random.default_rng(7)
    for i in range(n):
        h = np.zeros((tile, tile), np.float32)
        h[16:40, 16:40] = 9.0
        rgb = (rng.random((tile, tile, 3)) * 80 + 100).astype(np.uint8)
        rgb[40:52, 16:40] = 20                      # a shadow south of the block
        w.add(f"t{i}", rgb, h, np.where(h > 1, 3, 1).astype(np.uint8),
              np.ones((tile, tile), bool))
    w.finalise()


def _ckpt(path, detail: bool):
    import dwdata.preprocess as pp
    from models.heads import DepthWizardNet

    c = Config()
    c.tile_size, c.decoder_dim, c.n_bins = 64, 32, 8
    c.detail_branch, c.detail_dim = detail, 16
    c.canonical_gsd_m = 0.5
    torch.manual_seed(0)
    m = DepthWizardNet(c)
    spec = pp.PreprocSpec(tile_size=64, canonical_gsd_m=0.5)
    torch.save({"model": m.state_dict(), "config": safe_config_dict(c),
                "preproc": spec.to_dict(), "epoch": 3, "encoder_included": True}, path)


@pytest.mark.parametrize("detail", [False, True])
def test_eval_test_scores_v4_and_v5_shaped_checkpoints(tmp_path, monkeypatch, detail):
    undo = use_stub(hidden=32, patch=16, layers=8)
    try:
        import eval_test

        root = tmp_path / "data"
        _eval_store(root, "test", 4)
        ck = tmp_path / "best.pt"
        _ckpt(ck, detail)
        out = tmp_path / ("v5" if detail else "v4")
        eval_test.main(["--ckpt", str(ck), "--data_root", str(root), "--source", "gamus",
                        "--split", "test", "--tiles", "0", "--sliding_tiles", "2",
                        "--tta_scales", "1.0", "--out", str(out), "--num_workers", "0",
                        "--batch_size", "2", "--qualitative", "2", "--amp", "false"])
    finally:
        undo()
    r = json.loads((out / "test_metrics.json").read_text())
    assert r["config"]["detail_branch"] is detail          # restored from the checkpoint
    p = r["test_gamus_test_plain"]
    assert p["global"]["n"] == 4 * 64 * 64
    assert "edge_rmse_m" in p and "grad_ratio" in p
    assert "test_gamus_test_tta" in r and "test_gamus_test_sliding_tta" in r
    assert len(list((out / "qual").glob("*.npz"))) == 2
    assert len(r["qualitative"]["tiles"]) == 2


def test_eval_test_refuses_a_checkpoint_missing_head_weights(tmp_path):
    undo = use_stub(hidden=32, patch=16, layers=8)
    try:
        import eval_test

        ck = tmp_path / "best.pt"
        _ckpt(ck, False)
        d = torch.load(ck, weights_only=False)
        d["model"] = {k: v for k, v in d["model"].items() if not k.startswith("head_b.")}
        torch.save(d, ck)
        c = Config()
        with pytest.raises(SystemExit, match="missing"):
            eval_test.load_checkpoint(c, str(ck), "", [], torch.device("cpu"))
    finally:
        undo()


def test_fit_keys_renames_across_transformers_layouts():
    from eval_test import fit_keys

    sd = {"encoder.model.model.layer.0.w": 1, "head.x": 2}
    out, n = fit_keys(sd, {"encoder.model.layer.0.w", "head.x"})
    assert n == 1 and "encoder.model.layer.0.w" in out and out["head.x"] == 2


def test_compare_writes_the_cross_version_table_and_strips(tmp_path):
    import eval_test

    def res(rmse, edge):
        return {"test_gamus_test_plain": {
                    "global": {"rmse_m": rmse, "mae_m": rmse / 2, "pearson_r": 0.9},
                    "per_stratum": {"0-2m": {"n": 5, "rmse_m": rmse, "bias_m": 0.1}},
                    "edge_rmse_m": edge, "grad_ratio": 0.8},
                "test_gamus_test_tta": {"global": {"rmse_m": rmse - 0.05}}}

    rng = np.random.default_rng(0)
    rgb = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
    for name, r in (("a", res(3.6, 5.0)), ("b", res(3.3, 4.0))):
        d = tmp_path / name
        (d / "qual").mkdir(parents=True)
        (d / "test_metrics.json").write_text(json.dumps(r))
        np.savez_compressed(d / "qual" / "t0.npz", rgb=rgb, gt=np.ones((32, 32), np.float16),
                            valid=np.ones((32, 32), bool), pred=np.ones((32, 32), np.float16),
                            gsd_m=0.5)
    md = tmp_path / "cmp" / "v3_v4_v5.md"
    eval_test.main(["--compare", f"{tmp_path / 'a'},{tmp_path / 'b'}", "--labels", "v4,v5",
                    "--md", str(md)])
    text = md.read_text()
    assert "| RMSE plain | 3.600 | 3.300 |" in text
    assert "| edge RMSE | 5.000 | 4.000 |" in text
    assert (tmp_path / "cmp" / "strips" / "strip_t0.png").is_file()


# ---------------------------------------------------------------------
# V4_modal/v5_flags.py
# ---------------------------------------------------------------------
def _v5_flags():
    import importlib.util

    p = Path(__file__).resolve().parents[2] / "V4_modal" / "v5_flags.py"
    spec = importlib.util.spec_from_file_location("_v5_flags", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_v5_flags_are_real_fields_and_parse():
    from dataclasses import fields

    from config import parse_config

    m = _v5_flags()
    assert not set(m.FLAGS) - {f.name for f in fields(Config)}
    cfg = parse_config([x for k, v in m.FLAGS.items() for x in (f"--{k}", v)])
    assert cfg.detail_branch and cfg.val_sample_seed == 42
    assert cfg.coarse_label_sources == "dfc23,india_labeled" and cfg.coarse_pool == 4
    assert cfg.aug_pansharp_p == 0.5 and cfg.aug_gray_p == 0.1
    assert cfg.output_dir == "/results/v5" and cfg.onnx_opset == 18
    assert cfg.bin_max_m >= cfg.max_valid_height_m         # Head B covers every label
    assert cfg.coarse_label_m == 2.0 and cfg.unfreeze_warmup_epochs == 1.0
    assert cfg.datasets.startswith("gamus,")


def test_v5_flags_take_only_step0_keys_from_an_audit(tmp_path):
    m = _v5_flags()
    a = tmp_path / "audit.json"
    a.write_text(json.dumps({"flags": {"coarse_pool": "8", "coarse_mask_veg": "true",
                                       "epochs": "99"}}))
    f = m.with_audit(str(a))
    assert f["coarse_pool"] == "8" and f["coarse_mask_veg"] == "true"
    assert f["epochs"] == m.FLAGS["epochs"]
