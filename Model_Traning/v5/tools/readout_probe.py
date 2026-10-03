"""Frozen-checkpoint validation sweep; no optimisation or checkpoint writes.

Run from v5: python tools/readout_probe.py --ckpt best.pt --data_root /path \
    --out outputs/readout_probe --tiles 128
Uses the same seeded tile sample for every candidate.  This is a pilot; confirm
the chosen candidate on full validation before using held-out test for reporting.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config, safe_config_dict
from dwdata.dataset import TileDataset
from dwdata.packed import PackedStore
from dwdata.preprocess import PreprocSpec
from eval.metrics import evaluate
from eval_test import Replicated, load_checkpoint
from models.losses import coarse_pool_px
from train import resolve_hf_token


class ScaledInput(torch.nn.Module):
    """One scaled forward pass, returned to the original target grid in metres."""

    def __init__(self, model, scale):
        super().__init__()
        self.model = model
        self.scale = scale

    def forward(self, image):
        h, w = image.shape[-2:]
        size = tuple(max(16, int(round(s * self.scale / 16)) * 16) for s in (h, w))
        scaled = F.interpolate(image, size=size, mode="bilinear", align_corners=False)
        pred = self.model(scaled)["fused"].float()
        return {"fused": F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiles", type=int, default=128, help="0 = full validation")
    ap.add_argument("--sources", default="mvs3dm,us3d,neon,gamus")
    ap.add_argument("--candidates", default="", help="comma list; empty runs all")
    ap.add_argument("--batch_size", type=int, default=2)
    a = ap.parse_args()
    if a.tiles < 0:
        ap.error("--tiles must be nonnegative")
    cfg = Config()
    cfg.hf_token = resolve_hf_token(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_checkpoint(cfg, a.ckpt, "", [], device)
    cfg.data_root = a.data_root
    cfg.gpu_augment = False
    cfg.eval_bias_diagnostics = True
    cfg.per_landscape_metrics = True
    cfg.amp_dtype = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp16"
    cfg.batch_size = max(1, a.batch_size)
    cfg.eval_batch_mult = 1
    cfg.num_workers = 2
    spec = PreprocSpec.from_dict(ck["preproc"]) if "preproc" in ck else PreprocSpec.from_config(cfg)
    del ck
    selected = {"head": "fused"}
    model.register_forward_hook(lambda module, inputs, out: {**out, "fused": out[selected["head"]]})
    if torch.cuda.device_count() > 1:
        model = Replicated(model, [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())])
    replicas = list(model.replicas) if isinstance(model, Replicated) else [model]
    candidates = [
        ("baseline", "mean", None, "fused", False, (1.,)),
        ("head_a", "mean", None, "a", False, (1.,)),
        ("head_b", "mean", None, "b", False, (1.,)),
        ("head_b_single_mode", "single_mode", None, "b", False, (1.,)),
        ("fused_single_mode", "single_mode", None, "fused", False, (1.,)),
        ("fused_b50", "mean", .5, "fused", False, (1.,)),
        ("fused_single_mode_b50", "single_mode", .5, "fused", False, (1.,)),
        ("baseline_d4", "mean", None, "fused", True, (1.,)),
        ("baseline_d4_zoom150", "mean", None, "fused", True, (1.5,)),
        ("baseline_zoom125", "mean", None, "fused", False, (1.25,)),
        ("baseline_zoom150", "mean", None, "fused", False, (1.5,)),
        ("baseline_d4_zoom125", "mean", None, "fused", True, (1.25,)),
    ]
    if a.candidates:
        wanted = set(a.candidates.split(","))
        unknown = wanted - {c[0] for c in candidates}
        if unknown:
            ap.error(f"unknown candidates: {sorted(unknown)}")
        candidates = [c for c in candidates if c[0] in wanted]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"checkpoint": a.ckpt, "seed": 42, "split": "val", "tile_limit": a.tiles,
              "protocol": "paired seeded centre crops; bias = prediction - GT; 30m nDSM proxy",
              "config": safe_config_dict(cfg), "sources": {}}
    rows = ["| source | candidate | RMSE | GT-tall bias | pred-tall bias | flat bias | 30m RMSE | 30m bias | edge RMSE | grad |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for src in a.sources.split(","):
        store = PackedStore(Path(a.data_root) / src / "val")
        n = min(a.tiles, len(store)) if a.tiles else len(store)
        idx = np.sort(np.random.default_rng(42).permutation(len(store))[:n])
        ds = TileDataset(cfg, store, spec, src, train=False, length=n, indices=idx)
        loader = DataLoader(ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                            pin_memory=device.type == "cuda", persistent_workers=True)
        report["sources"][src] = {"indices": idx.tolist(), "store_tiles": len(store),
                                   "stems": [store.stems[int(i)] for i in idx], "candidates": {}}
        for name, readout, weight, head, tta, scales in candidates:
            for replica in replicas:
                replica.head_b.readout = readout
                replica.eval_head_b_weight = weight
            selected["head"] = head
            cfg.tta_scales = scales
            start = time.time()
            coarse_sources = tuple(s.strip() for s in cfg.coarse_label_sources.split(",") if s.strip())
            extra_pool = coarse_pool_px(cfg, cfg.canonical_gsd_m) if coarse_sources and src.startswith(coarse_sources) else 1
            evaluated_model = ScaledInput(model, scales[0]) if not tta and scales != (1.,) else model
            metrics = evaluate(evaluated_model, loader, cfg, device, use_tta=tta, extra_pool=extra_pool)
            metrics["seconds"] = round(time.time() - start, 2)
            metrics["readout"] = {"bin": readout, "head_b_weight": weight,
                                   "head": head, "tta": tta, "scales": scales}
            report["sources"][src]["candidates"][name] = metrics
            def value(key, field):
                return metrics.get(key, {}).get(field, float("nan"))
            vals = [value("global", "rmse_m"), value("tall_gt15m", "bias_m"),
                    value("tall_pred15m", "bias_m"), value("flat_lt1m", "bias_m"),
                    value("ndsm_30m", "rmse_m"), value("ndsm_30m", "bias_m"),
                    metrics.get("edge_rmse_m", float("nan")), metrics.get("grad_ratio", float("nan"))]
            row = f"| {src} | {name} | " + " | ".join(f"{v:.3f}" for v in vals) + " |"
            rows.append(row)
            print(row + f" ({metrics['seconds']:.0f}s)", flush=True)
            (out / "probe_metrics.json").write_text(json.dumps(report, indent=2))
            title = "# Frozen readout full validation" if a.tiles == 0 else "# Frozen readout validation pilot"
            (out / "PROBE.md").write_text(title + "\n\n" + report["protocol"] +
                                          "\n\n" + "\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
