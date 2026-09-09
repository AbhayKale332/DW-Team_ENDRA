import numpy as np
import torch

from eval.metrics import MetricAccum


def test_matches_numpy_reference():
    rng = np.random.default_rng(0)
    p = rng.random(5000) * 30
    t = p + rng.normal(0, 2, 5000)
    acc = MetricAccum()
    acc.update(torch.from_numpy(p), torch.from_numpy(t))
    r = acc.result()
    assert abs(r["rmse_m"] - np.sqrt(np.mean((p - t) ** 2))) < 1e-6
    assert abs(r["mae_m"] - np.mean(np.abs(p - t))) < 1e-6
    assert abs(r["pearson_r"] - np.corrcoef(p, t)[0, 1]) < 1e-3


def test_streaming_equivalence():
    rng = np.random.default_rng(1)
    p = rng.random(4000) * 10
    t = rng.random(4000) * 10
    one = MetricAccum()
    one.update(torch.from_numpy(p), torch.from_numpy(t))
    many = MetricAccum()
    for i in range(0, 4000, 400):
        many.update(torch.from_numpy(p[i:i + 400]), torch.from_numpy(t[i:i + 400]))
    assert abs(one.result()["rmse_m"] - many.result()["rmse_m"]) < 1e-6


def test_empty():
    assert MetricAccum().result() == {"n": 0}
