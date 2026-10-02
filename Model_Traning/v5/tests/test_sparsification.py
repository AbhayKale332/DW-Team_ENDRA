"""Sparsification / AUSE / AURG (eval/sparsification.py) on small synthetic arrays."""
import numpy as np
import pytest
import torch

from eval.sparsification import SparsificationMeter, format_line, sparsification


def _scene(n=4096, seed=0):
    rng = np.random.default_rng(seed)
    gt = rng.random(n) * 20
    err = rng.normal(0, 1, n) * (0.2 + rng.random(n) * 3)   # heteroscedastic
    return gt + err, gt, err


def test_hand_worked_case():
    """intervals=2, errors 0, 0, 3, 4: RMSE(all) 2.5, and 0 once the two big ones go."""
    gt = np.zeros(4)
    pred = np.array([0.0, 0.0, 3.0, 4.0])
    perfect = sparsification(pred, gt, np.abs(pred), intervals=2, min_px=1)
    assert perfect["curve_rmse_m"] == pytest.approx([2.5, 0.0, 0.0])
    assert perfect["ause_rmse_m"] == pytest.approx(0.0)
    assert perfect["aurg_rmse_m"] == pytest.approx(2.5 - 0.625)
    # a constant sigma cannot rank anything: the curve stays at RMSE(all) until the end
    flat = sparsification(pred, gt, np.ones(4), intervals=2, min_px=1)
    assert flat["curve_rmse_m"] == pytest.approx([2.5, 2.5, 0.0])
    assert flat["ause_rmse_m"] == pytest.approx(1.875 - 0.625)
    assert flat["aurg_rmse_m"] == pytest.approx(2.5 - 1.875)


def test_perfect_sigma_has_zero_ause():
    pred, gt, err = _scene()
    r = sparsification(pred, gt, np.abs(err))
    assert abs(r["ause_rmse_m"]) < 1e-9
    assert r["aurg_rmse_m"] > 0.3 * r["rmse_m"]
    assert r["curve_rmse_m"][0] == pytest.approx(r["rmse_m"])
    assert r["curve_rmse_m"] == pytest.approx(r["oracle_rmse_m"])
    # a monotone function of |err| ranks the same pixels: still the oracle
    assert abs(sparsification(pred, gt, 0.1 + np.abs(err) ** 2)["ause_rmse_m"]) < 1e-9


def test_random_sigma_is_worse_than_informative_sigma():
    pred, gt, err = _scene()
    rng = np.random.default_rng(1)
    rand = sparsification(pred, gt, rng.random(err.size))
    noisy = sparsification(pred, gt, np.abs(err) + rng.normal(0, 0.3, err.size))
    assert rand["ause_rmse_m"] > 0.1
    assert 0 < noisy["ause_rmse_m"] < rand["ause_rmse_m"]
    assert noisy["aurg_rmse_m"] > rand["aurg_rmse_m"]
    # random removal leaves the RMSE flat: no gain beyond the final-point artefact
    assert abs(rand["aurg_rmse_m"]) < 0.1 * rand["rmse_m"]


def test_inverted_sigma_has_negative_gain():
    pred, gt, err = _scene()
    r = sparsification(pred, gt, -np.abs(err))
    assert r["aurg_rmse_m"] < 0 and r["ause_rmse_m"] > 0


def test_masks_non_finite_and_invalid_pixels():
    pred, gt, err = _scene(n=1000)
    sigma = np.abs(err)
    base = sparsification(pred, gt, sigma)
    pred2, sigma2 = pred.copy(), sigma.copy()
    pred2[:10] = np.nan
    sigma2[10:20] = np.inf
    valid = np.ones(1000, bool)
    valid[20:30] = False
    r = sparsification(pred2, gt, sigma2, valid)
    assert r["n"] == 970 and base["n"] == 1000
    assert sparsification(pred[:10], gt[:10], sigma[:10]) is None       # under min_px


def test_confident_cut_matches_predict_rule():
    """max(1 m, median sigma), as infer/predict.py uncertainty_summary."""
    gt = np.zeros(100)
    pred = np.r_[np.zeros(50), np.full(50, 4.0)]
    sigma = np.r_[np.full(50, 0.2), np.full(50, 0.5)]   # median 0.35 -> floored to 1 m
    r = sparsification(pred, gt, sigma, min_px=1)
    assert r["confident_threshold_m"] == 1.0 and r["confident_frac"] == 1.0
    sigma = np.r_[np.full(50, 1.5), np.full(50, 6.0)]   # median 3.75
    r = sparsification(pred, gt, sigma, min_px=1)
    assert r["confident_threshold_m"] == pytest.approx(3.75)
    assert r["confident_frac"] == 0.5 and r["confident_rmse_m"] == 0.0


def test_meter_averages_tiles_and_takes_torch_batches():
    pred, gt, err = _scene(n=2 * 32 * 32)
    sig = np.abs(err)
    shape = (2, 1, 32, 32)
    t = lambda a: torch.from_numpy(a.reshape(shape).astype(np.float32))  # noqa: E731
    m = SparsificationMeter(intervals=10)
    assert m.result() == {} and format_line({}) == "no sigma"
    valid = torch.ones(shape, dtype=torch.bool)
    m.add_batch(t(pred), t(gt), t(sig), valid)
    r = m.result()
    assert r["n_tiles"] == 2 and r["n_px"] == 2 * 32 * 32
    assert len(r["curve_rmse_m"]) == 11 and len(r["removed"]) == 11
    assert abs(r["ause_rmse_m"]) < 1e-5          # float32 round trip
    one = sparsification(pred[:1024], gt[:1024], sig[:1024], intervals=10)
    two = sparsification(pred[1024:], gt[1024:], sig[1024:], intervals=10)
    assert r["rmse_m"] == pytest.approx((one["rmse_m"] + two["rmse_m"]) / 2, rel=1e-5)
    assert "AUSE" in format_line(r)
