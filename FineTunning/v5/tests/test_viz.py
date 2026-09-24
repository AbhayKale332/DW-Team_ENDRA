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
