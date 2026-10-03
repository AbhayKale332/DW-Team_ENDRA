"""Paired full-native-grid validation, using the exact HF inference source."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np


class Accum:
    def __init__(self):
        self.n = 0
        self.sse = self.sae = self.bias_sum = 0.0

    def add(self, error):
        d = np.asarray(error, dtype=np.float64).reshape(-1)
        self.n += d.size
        self.sse += float(np.dot(d, d))
        self.sae += float(np.abs(d).sum())
        self.bias_sum += float(d.sum())

    def result(self):
        return {"n": self.n, "sse": self.sse, "sae": self.sae, "bias_sum": self.bias_sum,
                "rmse_m": math.sqrt(self.sse / self.n) if self.n else None,
                "mae_m": self.sae / self.n if self.n else None,
                "bias_m": self.bias_sum / self.n if self.n else None}


def valid_mask(target, source_valid, max_height=150.0):
    return source_valid.astype(bool) & np.isfinite(target) & (target >= 0) & (target <= max_height)


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(8 * 2**20):
            h.update(chunk)
    return h.hexdigest()


def load_support(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("hf-code", "support-code", "store", "ckpt", "out"):
        ap.add_argument(f"--{name}", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.hf_code)
    import torch
    import config
    from infer.predict import load_model
    from infer.engine import count_tiles, predict_scene

    config.LANDSCAPE_NAMES = ("urban", "sparse", "hilly", "forested")
    packed = load_support("_dw_packed", Path(a.support_code, "dwdata/packed.py"))
    landscape = load_support("_dw_landscape", Path(a.support_code, "eval/landscape.py"))
    store = packed.PackedStore(a.store)
    assert len(store) == 859
    device = torch.device("cuda")
    model, spec, cfg = load_model(a.ckpt, device, os.environ.get("HF_TOKEN", ""))
    assert cfg.checkpoint_epoch == 7 and cfg.checkpoint_tensor_count == 925 and cfg.detail_branch
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
    modes = (("baseline", False, 1.0), ("d4_1.5", True, 1.5))
    names = [m[0] for m in modes]
    keys = ["global", "tall_gt15m", "flat_lt1m"] + [f"class:{c}" for c in cfg.class_names]
    keys += [f"landscape:{c}" for c in config.LANDSCAPE_NAMES]
    accum = {name: {key: Accum() for key in keys} for name in names}
    seconds = {name: 0.0 for name in names}
    out = Path(a.out)
    report = {
        "status": "running", "split": "val", "tiles_total": len(store), "tiles_completed": 0,
        "protocol": "full native-grid GAMUS labels; deployed HF predict_scene; paired baseline / D4 + 1.5x",
        "label_grid_px": store.tile_px, "source_gsd_m": store.gsd_m,
        "max_valid_height_m": 150.0, "overlap": 0.5, "batch_tiles": 0,
        "preproc": spec.to_dict(), "checkpoint_epoch": cfg.checkpoint_epoch,
        "checkpoint_tensors": cfg.checkpoint_tensor_count, "detail_branch": cfg.detail_branch,
        "checkpoint_sha256": file_sha256(a.ckpt), "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(), "hf_space_commit": "36afc327ba4112390ba08ef6836dad140a98a3c7",
        "hf_source_sha256": {str(p.relative_to(a.hf_code)): file_sha256(p)
                             for p in Path(a.hf_code).rglob("*.py")},
        "store_index_sha256": file_sha256(Path(a.store, "index.json")),
        "indices": list(range(len(store))), "stems": store.stems, "candidates": {},
        "tiles_per_scene": count_tiles(store.tile_px, store.tile_px, store.gsd_m, spec, 0.5),
    }
    started = time.perf_counter()

    def save():
        for name in names:
            report["candidates"][name] = {"metrics": {k: v.result() for k, v in accum[name].items()},
                                         "inference_seconds": seconds[name]}
        report["wall_seconds"] = time.perf_counter() - started
        tmp = out / "accuracy.json.tmp"
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(out / "accuracy.json")

    save()
    with (out / "per_tile.jsonl").open("w", buffering=1) as rows:
        for i in range(len(store)):
            rgb, target, cls, source_valid = store.get(i)
            valid = valid_mask(target, source_valid)
            assert valid.any()
            kind, desc = landscape.classify(np.where(valid, target, 0.0), valid, store.gsd_m)
            p_target, p_cls = target[valid], cls[valid]
            tall, flat = p_target >= 15, p_target < 1
            row = {"index": i, "stem": store.stems[i], "landscape": kind,
                   "landscape_descriptors": desc, "candidates": {}}
            predictions = {}
            for name, tta, scale in modes:
                begin = time.perf_counter()
                # Raw store RGB: predict_scene applies the scene stretch once.
                height, seg, std = predict_scene(
                    model, rgb, store.gsd_m, spec, device, tta=tta, tta_scales=(scale,),
                    amp_dtype=amp_dtype, overlap=0.5, batch_tiles=0, want_seg=True, want_std=True,
                )
                elapsed = time.perf_counter() - begin
                seconds[name] += elapsed
                assert height.shape == seg.shape == std.shape == target.shape
                assert np.isfinite(height).all() and np.isfinite(std).all() and (std >= 0).all()
                assert seg.min() >= 0 and seg.max() < config.SEG_IGNORE_INDEX
                error = height[valid].astype(np.float64) - p_target.astype(np.float64)
                meters = accum[name]
                meters["global"].add(error)
                meters["tall_gt15m"].add(error[tall])
                meters["flat_lt1m"].add(error[flat])
                meters[f"landscape:{kind}"].add(error)
                for class_id, class_name in enumerate(cfg.class_names):
                    meters[f"class:{class_name}"].add(error[p_cls == class_id])
                per_tile = Accum()
                per_tile.add(error)
                row["candidates"][name] = {**per_tile.result(), "inference_seconds": elapsed}
                if i < 8:
                    predictions[name] = height
                del height, seg, std
            assert row["candidates"]["baseline"]["n"] == row["candidates"]["d4_1.5"]["n"]
            rows.write(json.dumps(row) + "\n")
            if predictions:
                np.savez_compressed(out / f"qual_{i:04d}.npz", rgb=rgb, target=target, valid=valid,
                                    cls=cls, **predictions)
            report["tiles_completed"] = i + 1
            if i == 0 or (i + 1) % 10 == 0:
                save()
                print(f"[accuracy] {i + 1}/{len(store)} native tiles, "
                      f"baseline={accum['baseline']['global'].result()['rmse_m']:.6f}, "
                      f"D4+1.5={accum['d4_1.5']['global'].result()['rmse_m']:.6f}, "
                      f"elapsed={report['wall_seconds']:.1f}s", flush=True)
    report["status"] = "completed"
    for key in keys:
        assert accum['baseline'][key].n == accum['d4_1.5'][key].n
    save()
    print("[accuracy] completed all 859 paired native-grid validation tiles", flush=True)


if __name__ == "__main__":
    main()
