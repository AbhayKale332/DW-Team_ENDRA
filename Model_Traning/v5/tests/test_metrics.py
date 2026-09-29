import torch

from config import Config
from eval.metrics import Evaluator, MetricAccum, format_line


def test_perfect_prediction():
    a = MetricAccum()
    t = torch.rand(1000) * 30
    a.update(t, t)
    r = a.result()
    assert r["rmse_m"] < 1e-9 and r["mae_m"] < 1e-9
    assert abs(r["bias_m"]) < 1e-9 and r["delta1"] == 1.0
    assert r["pearson_r"] > 0.999


def test_bias_is_signed():
    a = MetricAccum()
    t = torch.full((100,), 10.0)
    a.update(t - 3.0, t)
    assert abs(a.result()["bias_m"] + 3.0) < 1e-9      # under-prediction -> negative


def test_balanced_rmse_exposes_tail_error_the_global_hides():
    """The v2 story in one test: 3.07 m global vs 4.19 m balanced."""
    cfg = Config()
    ev = Evaluator(cfg)
    tgt = torch.cat([torch.zeros(9000), torch.full((100,), 25.0)])   # 99% ground
    pred = torch.cat([torch.zeros(9000), torch.full((100,), 10.0)])  # tail is 15 m low
    valid = torch.ones_like(tgt, dtype=torch.bool)
    ev.add(pred, tgt, valid)
    r = ev.result()
    assert r["global"]["rmse_m"] < 2.0                 # global barely notices
    assert r["balanced_rmse_m"] > 7.0                  # balanced does
    assert r["tall_gt15m"]["bias_m"] < -14.0           # and names the direction


def test_flat_bias_catches_hallucinated_ground_height():
    """The Austin failure mode: metres of height invented on flat ground."""
    cfg = Config()
    ev = Evaluator(cfg)
    tgt = torch.zeros(1000)
    ev.add(torch.full((1000,), 4.0), tgt, torch.ones(1000, dtype=torch.bool))
    r = ev.result()
    assert r["flat_lt1m"]["bias_m"] > 3.9
    assert "flat_bias=+4.00" in format_line(r)


def test_class_stats_report_measured_frequency():
    cfg = Config()
    ev = Evaluator(cfg)
    tgt = torch.cat([torch.zeros(800), torch.full((200,), 12.0)])
    cls = torch.cat([torch.zeros(800, dtype=torch.long),
                     torch.full((200,), 2, dtype=torch.long)])
    ev.add(tgt, tgt, torch.ones(1000, dtype=torch.bool), cls)
    st = ev.result()["class_stats"]
    assert abs(st["other"]["px_frac"] - 0.8) < 1e-6
    assert abs(st["low_veg"]["mean_gt_height_m"] - 12.0) < 1e-6


def test_empty_accumulator_is_safe():
    assert MetricAccum().result() == {"n": 0}


def test_edge_rmse_and_grad_ratio_see_a_blurred_map():
    """v5: a map with the right means but smeared walls scores worse on
    `edge_rmse_m` and reads < 1 on `grad_ratio`."""
    import torch.nn.functional as F

    cfg = Config()
    t = torch.zeros(1, 1, 64, 64)
    t[..., 20:44, 20:44] = 20.0
    v = torch.ones_like(t, dtype=torch.bool)
    blur = F.avg_pool2d(F.pad(t, (3, 3, 3, 3), mode="replicate"), 7, 1)
    sharp_ev, blur_ev = Evaluator(cfg), Evaluator(cfg)
    for ev, p in ((sharp_ev, t), (blur_ev, blur)):
        ev.add(p, t, v)
        ev.add_spatial(p, t, v)
    rs, rb = sharp_ev.result(), blur_ev.result()
    assert rs["edge_rmse_m"] < 1e-6 < rb["edge_rmse_m"]
    assert abs(rs["grad_ratio"] - 1.0) < 1e-6
    assert rb["grad_ratio"] < 1.0 or rb["edge_rmse_m"] > 3.0


def test_pooled_scoring_forgives_sub_block_detail():
    from eval.metrics import pool_pair

    t = torch.zeros(1, 1, 16, 16)
    t[..., ::2, :] = 4.0                      # stripes finer than the 4x4 block
    p = torch.full_like(t, 2.0)               # the block mean
    v = torch.ones_like(t, dtype=torch.bool)
    pp, tt, vv = pool_pair(p, t, v, 4)
    assert vv.all() and torch.allclose(pp, tt)
