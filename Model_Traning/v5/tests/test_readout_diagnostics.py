import pytest
import torch
from torch import nn

from config import Config
from eval.metrics import Evaluator, evaluate
from models.heads import HeadB


@pytest.mark.parametrize("prob,expected", [
    ([.5, .2, .05, .1, .15], (.5 + .4 + .15) / .75),
    ([.15, .1, .05, .2, .5], 3.45 / .75),
    ([.1, .2, .4, .2, .1], 3.0),
    ([.2, .2, .2, .2, .2], 3.0),
])
def test_single_mode_stops_at_other_peak(prob, expected):
    class FixedLogits(nn.Module):
        def forward(self, x):
            return torch.tensor(prob).log().reshape(1, 5, 1, 1)

    head = HeadB(32, 5, 0, 6)
    head.pixel_logits = FixedLogits()
    head.bin_centres = lambda x: torch.arange(1., 6.).reshape(1, 5)
    x = torch.zeros(1, 32, 1, 1)
    baseline = head(x)
    head.readout = "single_mode"
    actual = head(x)
    assert actual["height"].item() == pytest.approx(expected)
    assert torch.equal(actual["std"], baseline["std"])


def test_prediction_conditioning_uses_prediction_and_signed_error():
    cfg = Config()
    cfg.eval_bias_diagnostics = True
    ev = Evaluator(cfg)
    ev.add(torch.tensor([1., 16., 25.]), torch.tensor([20., 10., 30.]),
           torch.ones(3, dtype=torch.bool))
    result = ev.result()
    assert result["tall_pred15m"]["n"] == 2
    assert result["tall_pred15m"]["bias_m"] == pytest.approx(.5)
    assert result["tall_gt15m"]["bias_m"] == pytest.approx(-12.)


def test_30m_uses_each_samples_gsd_and_excludes_invalid_cells():
    cfg = Config()
    cfg.eval_bias_diagnostics = True
    ev = Evaluator(cfg)
    p = torch.full((2, 1, 120, 120), 3.)
    t = torch.ones_like(p)
    v = torch.ones_like(p, dtype=torch.bool)
    t[0, 0, 0, 0] = float("nan")
    v[0, 0, 0, 0] = False
    ev.add_30m(p, t, v, torch.tensor([.5, 1.]))
    result = ev.result()["ndsm_30m"]
    assert result["n"] == 3 + 16
    assert result["bias_m"] == pytest.approx(2.)
    assert result["rmse_m"] == pytest.approx(2.)
    assert result["actual_block_sizes_m"] == [30.]


def test_diagnostics_are_opt_in():
    ev = Evaluator(Config())
    assert "ndsm_30m" not in ev.result()


def test_spatial_metrics_accept_full_tile_2d_maps():
    ev = Evaluator(Config())
    t = torch.zeros(8, 8)
    t[:, 4:] = 10.
    ev.add_spatial(t, t.unsqueeze(0), torch.ones_like(t, dtype=torch.bool).unsqueeze(0))
    result = ev.result()
    assert result["edge_rmse_m"] == 0
    assert result["grad_ratio"] == 1


def test_pooled_companion_scores_same_prediction_without_another_forward():
    class FixedModel(nn.Module):
        calls = 0

        def forward(self, image):
            self.calls += 1
            return {"fused": torch.full((1, 1, 8, 8), 3.)}

    cfg = Config()
    cfg.amp = False
    batch = {"image": torch.zeros(1, 3, 8, 8), "target": torch.ones(1, 1, 8, 8),
             "valid": torch.ones(1, 1, 8, 8, dtype=torch.bool),
             "cls": torch.zeros(1, 8, 8, dtype=torch.long), "gsd_m": torch.tensor([.5])}
    model = FixedModel()
    result = evaluate(model, [batch], cfg, torch.device("cpu"), extra_pool=2)
    assert model.calls == 1
    assert result["global"]["n"] == 64
    assert result["pooled2"]["global"]["n"] == 16
    assert result["pooled2"]["global"]["rmse_m"] == 2
