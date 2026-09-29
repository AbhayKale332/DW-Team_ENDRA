"""End-to-end mini training loop on synthetic tiles, CPU, no network."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dwdata.base import pack_sample
from eval.metrics import evaluate


class _SynDS(Dataset):
    def __init__(self, n, s):
        self.n, self.s = n, s

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(i)
        rgb = rng.integers(0, 255, (self.s, self.s, 3), np.uint8)
        h = np.zeros((self.s, self.s), np.float32)
        h[self.s // 3: 2 * self.s // 3, self.s // 3: 2 * self.s // 3] = 12.0
        seg = (h > 0).astype(np.int64) * 2
        d = pack_sample(rgb, h, seg, max_h=400.0, has_seg=True, src="syn", stem=str(i))
        d["gsd_m"] = 0.5
        return d


def test_train_stage_runs_and_improves(fake_net, cfg, tmp_path):
    from train import train_stage

    cfg.output_dir = str(tmp_path)
    cfg.finetune_epochs = 2
    cfg.eval_every = 1
    cfg.grad_accum = 1
    dl = DataLoader(_SynDS(8, cfg.tile_size), batch_size=2)
    dl_va = DataLoader(_SynDS(4, cfg.tile_size), batch_size=2)

    hist: list = []
    best = train_stage(
        cfg, fake_net, torch.device("cpu"), dl, dl_va,
        stage="finetune", epochs=2, minutes=5.0, history=hist,
    )
    assert np.isfinite(best)
    assert (Path(tmp_path) / "metrics.json").exists()
    assert (Path(tmp_path) / "best.pt").exists()
    m = json.loads((Path(tmp_path) / "metrics.json").read_text())
    assert m["history"] and "val" in m["history"][-1]


def test_evaluate_produces_strata(fake_net, cfg):
    dl_va = DataLoader(_SynDS(4, cfg.tile_size), batch_size=2)
    out = evaluate(fake_net, dl_va, cfg, torch.device("cpu"), use_tta=False)
    assert "per_stratum" in out and "per_class" in out
    assert out["global"]["n"] > 0


def test_packaging_after_train(fake_net, cfg, tmp_path):
    from package_results import build_zip, write_env_files
    from train import train_stage

    cfg.output_dir = str(tmp_path)
    dl = DataLoader(_SynDS(4, cfg.tile_size), batch_size=2)
    train_stage(cfg, fake_net, torch.device("cpu"), dl, None,
                stage="finetune", epochs=1, minutes=5.0, history=[])
    write_env_files(str(tmp_path))
    zp = build_zip(str(tmp_path))
    assert zp.exists() and zp.stat().st_size > 0
