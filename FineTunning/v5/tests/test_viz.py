"""Figures and the validation report.

These are deliverables, not decoration: "validate estimated structural heights
against reference datasets" is in the problem statement, and a report that
crashes on a missing key at the end of a five-hour run is a lost run.  So the
tests hammer the degenerate inputs — no history, no landscape block, no samples.
"""
import json

import numpy as np
import pytest

from viz.figures import colorize, hillshade, make_all, strip
from viz.report_html import build


def _metrics(with_landscape=True, with_history=True):
    val = {
        "global": {"rmse_m": 2.6, "mae_m": 1.2, "bias_m": -0.2, "pearson_r": 0.91,
                   "delta1": 0.67, "n": 1_000_000},
        "balanced_rmse_m": 3.5,
        "tall_gt15m": {"bias_m": -2.4, "rmse_m": 5.1, "n": 9000},
        "flat_lt1m": {"bias_m": 0.3, "rmse_m": 0.7, "n": 24000},
        "per_stratum": {"0-2m": {"n": 500, "rmse_m": 0.4, "mae_m": 0.3, "bias_m": 0.1},
                        "20m+": {"n": 200, "rmse_m": 5.0, "mae_m": 4.0, "bias_m": -4.1}},
        "per_class": {"class0": {"rmse_m": 1.1, "mae_m": 0.5}},
        "class_stats": {"class0": {"px_frac": 0.3, "mean_gt_height_m": 0.2}},
    }
    if with_landscape:
        val["per_landscape"] = {
            "urban": {"rmse_m": 2.4, "mae_m": 1.2, "bias_m": -0.2, "pearson_r": 0.9,
                      "tiles": 80, "n": 900},
            "hilly": {"rmse_m": 5.8, "mae_m": 3.9, "bias_m": -2.2, "pearson_r": 0.6,
                      "tiles": 4, "n": 40}}
        val["landscape_rmse_spread_m"] = 3.4
        val["landscape_worst"] = "hilly"
    return {
        "config": {"encoder_model_id": "x", "datasets": "gamus", "epochs": 4},
        "preproc": {"canonical_gsd_m": 0.5, "tile_size": 512, "version": "v4"},
        "elapsed_min": 12.0,
        "history": ([{"epoch": i, "train_loss": 5.0 / i, "encoder_frozen": i < 2,
                      "val": val} for i in range(1, 5)] if with_history else []),
        "final_sliding_tta": val,
    }


def _sample(n=128):
    rng = np.random.default_rng(0)
    gt = np.zeros((n, n), np.float32)
    gt[20:60, 20:60] = 18.0
    pred = np.clip(gt * 0.85 + rng.normal(0, 0.4, gt.shape).astype(np.float32), 0, None)
    rgb = (rng.random((n, n, 3)) * 255).astype(np.uint8)
    return rgb, pred, gt


def test_every_figure_renders(tmp_path):
    made = make_all(tmp_path, _metrics(), [_sample()], 0.5)
    for key in ("curves", "per_stratum", "per_landscape", "scatter",
                "error_hist", "hillshade", "qualitative"):
        assert key in made, f"missing figure: {key}"
        assert (tmp_path / made[key]).stat().st_size > 1000


def test_metric_only_run_still_produces_figures(tmp_path):
    made = make_all(tmp_path, _metrics(), None, 0.5)
    assert "curves" in made and "per_stratum" in made
    assert "scatter" not in made          # nothing to scatter without pixels


def test_degenerate_metrics_do_not_crash(tmp_path):
    made = make_all(tmp_path, {"history": []}, None, 0.5)
    assert made == {}
    made = make_all(tmp_path, _metrics(with_landscape=False, with_history=False),
                    [_sample()], 0.5)
    assert "per_landscape" not in made and "scatter" in made


def test_report_is_self_contained(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps(_metrics()))
    make_all(tmp_path / "figures", _metrics(), [_sample()], 0.5)
    out = build(tmp_path)
    html = out.read_text()
    assert out.stat().st_size > 20_000
    # no external references at all: it has to open on an air-gapped machine
    assert "src=\"data:image/png;base64," in html
    assert "http://" not in html and "https://" not in html
    for must in ("RMSE", "urban", "hilly", "balanced", "Inference contract",
                 "does not establish", "unverified"):
        assert must in html, f"report is missing {must!r}"


def test_report_survives_a_run_with_no_figures(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps(_metrics(with_landscape=False)))
    html = build(tmp_path).read_text()
    assert "RMSE" in html and "<img" not in html


def test_hillshade_is_flat_on_flat_ground():
    flat = np.zeros((64, 64), np.float32)
    sh = hillshade(flat, 0.5, 315, 45)
    assert sh.std() < 1.0
    bumpy = flat.copy()
    bumpy[20:40, 20:40] = 10.0
    assert hillshade(bumpy, 0.5, 315, 45).std() > 5.0


def test_colorize_and_strip_shapes():
    a = np.linspace(0, 10, 64 * 64).reshape(64, 64).astype(np.float32)
    c = colorize(a, 0, 10)
    assert c.shape == (64, 64, 3) and c.dtype == np.uint8
    rgb, pred, gt = _sample(64)
    s = strip(rgb, pred, gt)
    assert s.shape == (64, 64 * 4, 3)


def _gallery_entry(store, split, seen=False, n=2):
    tiles = []
    for i in range(n):
        rgb, pred, gt = _sample(64)
        valid = np.ones_like(gt, bool)
        valid[:4] = False                         # a NoData band, drawn grey
        tiles.append({"stem": f"{store}_{i}", "rgb": rgb, "pred": pred, "gt": gt,
                      "valid": valid, "rmse_m": 1.5, "mae_m": 0.8})
    return {"store": store, "split": split, "seen_in_training": seen,
            "gsd_m": 0.5, "tiles": tiles}


def test_gallery_renders_into_the_report(tmp_path):
    gallery = [_gallery_entry("synrs3d_g05", "train", seen=True),
               _gallery_entry("dfc23_g050", "val"),
               _gallery_entry("india_labeled", "val"),
               _gallery_entry("us3d", "val"),
               {"store": "empty", "split": "val", "tiles": []}]
    (tmp_path / "metrics.json").write_text(json.dumps(_metrics()))
    made = make_all(tmp_path / "figures", _metrics(), [_sample()], 0.5, gallery=gallery)
    for s in ("synrs3d_g05", "dfc23_g050", "india_labeled", "us3d"):
        assert (tmp_path / "figures" / made[f"gallery_{s}"]).stat().st_size > 5000
    assert "gallery_empty" not in made
    index = json.loads((tmp_path / "figures" / "gallery.json").read_text())
    assert [e["store"] for e in index] == ["synrs3d_g05", "dfc23_g050", "india_labeled",
                                    "us3d"]
    assert "rgb" not in index[0]["tiles"][0]      # metadata only, no arrays
    html = build(tmp_path).read_text()
    assert "Sample tiles by dataset" in html
    assert html.count("src=\"data:image/jpeg;base64,") == 4
    assert "SynRS3D" in html and "DFC23" in html and "New Delhi" in html
    assert "US3D" in html and "WorldView-3" in html
    assert "training tiles" in html               # synrs3d is train-only; say so


def test_gallery_store_resolution(tmp_path):
    from config import Config
    from eval.report import _gallery_stores

    for name, split in (("synrs3d_g05", "train"), ("synrs3d_g1", "train"),
                        ("dfc23_g050", "train"), ("dfc23_g050", "val"),
                        ("india_labeled", "val"), ("us3d", "train"), ("us3d", "val")):
        d = tmp_path / name / split
        d.mkdir(parents=True)
        (d / "index.json").write_text("{}")
    cfg = Config(data_root=str(tmp_path), datasets="gamus,synrs3d_g1",
                 gallery_sources="synrs3d,dfc23,india_labeled,us3d,geonrw")
    got = _gallery_stores(cfg)
    # the store this run trained on wins; val beats train; untrained is fine
    assert got == [("synrs3d_g1", "train", True), ("dfc23_g050", "val", False),
                   ("india_labeled", "val", False), ("us3d", "val", False)]


def _shadowed_scene(n=96):
    """A 12 m block lit from the south-east, its shadow painted dark-blue in the RGB."""
    from viz.shadow import cast_shadows

    gt = np.zeros((n, n), np.float32)
    gt[30:50, 30:50] = 12.0
    shade = cast_shadows(gt, 0.5, 135.0, 45.0)
    rgb = np.full((n, n, 3), (150, 140, 120), np.uint8)
    rgb[gt > 0] = (200, 190, 180)
    rgb[shade] = (20, 25, 60)
    pred = gt * 0.8
    return rgb, pred, gt, shade


def test_tile_shadows_fit_the_sun_on_the_reference():
    from viz.shadow import reference_sun_check

    rgb, pred, gt, _ = _shadowed_scene()
    sh = reference_sun_check(rgb, gt, pred, np.ones_like(gt, bool), 0.5)
    assert sh is not None and sh["image"].any()
    assert sh["iou_ref"] > 0.5
    # a 20 % short prediction casts a 20 % short shadow: close, not better
    assert 0 < sh["iou_pred"] <= sh["iou_ref"] + 1e-6


def test_landscape_gallery_and_shadows_render_into_the_report(tmp_path):
    from viz.shadow import reference_sun_check

    classes = {}
    for c in ("urban", "sparse", "forested"):
        rgb, pred, gt, _ = _shadowed_scene()
        valid = np.ones_like(gt, bool)
        classes[c] = [{"stem": f"{c}_0", "rgb": rgb, "pred": pred, "gt": gt,
                       "valid": valid, "landscape": c, "rmse_m": 1.0, "mae_m": 0.5,
                       "bias_m": -0.4, "descriptors": {"frac_tall": 0.04},
                       "shadow": reference_sun_check(rgb, gt, pred, valid, 0.5)}]
    lg = {"store": "gamus", "split": "val", "gsd_m": 0.5, "scanned": 40,
          "counts": {"urban": 30, "sparse": 7, "hilly": 0, "forested": 3},
          "classes": classes}
    (tmp_path / "metrics.json").write_text(json.dumps(_metrics()))
    made = make_all(tmp_path / "figures", _metrics(), [_sample()], 0.5,
                    landscape_gallery=lg)
    for c in classes:
        assert (tmp_path / "figures" / made[f"landscape_{c}"]).stat().st_size > 5000
    assert "shadows" in made
    index = json.loads((tmp_path / "figures" / "landscape_gallery.json").read_text())
    assert "image" not in index["classes"]["urban"]["tiles"][0]["shadow"]  # no arrays
    html = build(tmp_path).read_text()
    assert "What each landscape looks like" in html
    assert "<h3>forested</h3>" in html and "<h3>sparse</h3>" in html
    assert "classified as hilly" in html          # absent class is said, not hidden
    assert "Shadow consistency" in html
    assert html.count("src=\"data:image/jpeg;base64,") == 4


def test_landscape_gallery_absent_is_quiet(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps(_metrics()))
    make_all(tmp_path / "figures", _metrics(), [_sample()], 0.5, landscape_gallery=None)
    html = build(tmp_path).read_text()
    assert "What each landscape looks like" not in html
    assert "Shadow consistency" not in html
