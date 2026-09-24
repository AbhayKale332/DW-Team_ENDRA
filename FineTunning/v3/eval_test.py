"""Score a frozen v3 checkpoint on a held-out packed store.

    python eval_test.py --ckpt best.pt --data_root /kaggle/temp/dwdata \
        --run_config config.json --source gamus --split test \
        --tiles 0 --sliding_tiles 400 --out outputs/v3_test

Why this file exists and not `train.py --epochs 0`: v3 has no test path at all —
`build_loaders` only ever opens `train` and the one val split in `_VAL_SPLIT`.
v4 grew one (`--test_sources`, V4_Kaggle/train.py:988) but v4's HeadB is not
numerically the same head v3 trained: it RMS-normalises the pooled feature,
runs the width softmax in fp32 and floors the bin widths (V4_Kaggle/models/
heads.py).  Every parameter name survived, so a v3 checkpoint *loads* there
without complaint and then predicts through different bin centres — wrong
numbers, no error.  So the v3 weights are scored by v3 code, and only the packed
store is shared (`dwdata/packed.py` is byte-identical across the two trees).

The protocol mirrors v4's `test_gamus_test_*` exactly — all tiles centre-crop
plain + TTA, sliding window + TTA over the first `--sliding_tiles` — so the two
runs' numbers sit in the same table.  Two things are needed on top of the v3
config for that to hold:

  * `--tta_scales 1.0,1.25`.  v3 trained with `tta_scales=(1.0,)` and
    `_restore_arch` would carry that over; v4 and DAV2 score TTA at (1.0, 1.25).
    `parse_config` skips tuples, so this is the only way to set it.
  * `per_landscape_metrics` is forced on, so the output carries the same
    `per_landscape` breakdown v4/DAV2's metrics.json do.

The same script also scores v4's secondary val sets (`dfc23_g050/val`,
`india_labeled/val`: `--tiles 400 --sliding_tiles 0 --skip_tta`, i.e. the
400-tile prefix, centre-crop, plain — how `val_<source>` is computed in v4).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config, parse_config, safe_config_dict
from dwdata.dataset import FullTileDataset, TileDataset
from dwdata.packed import PackedStore, store_exists
from dwdata.preprocess import PreprocSpec
from eval.metrics import evaluate, format_line
from eval.sliding import sliding_eval
from models.heads import DepthWizardNetV3
from train import resolve_hf_token

# Config keys that describe the *run*, not the network.  Restoring these from a
# training config.json would point the eval back at the Lightning studio.
_SKIP = {"output_dir", "resume", "data_root", "datasets", "make_zip", "smoke",
         "num_workers", "prefetch_factor", "batch_size", "epochs", "max_minutes"}


# DINOv3ViTModel's blocks sit at `layer.N` in transformers 4.56/4.57 and at
# `model.layer.N` in 5.x.  v3 trained under 5.x, so best.pt holds
# `encoder.model.model.layer.N.*`; Kaggle's image ships 4.x (which v3's
# `transformers>=4.56` accepts) and builds `encoder.model.layer.N.*`.  Same
# tensors, one prefix level apart — without this all 24 blocks come back
# "missing" (408 tensors) and the guard below refuses.
_KEY_ALIASES = (("encoder.model.model.layer.", "encoder.model.layer."),
                ("encoder.model.layer.", "encoder.model.model.layer."))


def fit_keys(sd: dict, want) -> tuple[dict, int]:
    """Rename checkpoint keys across `_KEY_ALIASES`, in either direction.

    Only onto a name the model actually has and the checkpoint does not, so a
    genuinely different network still reaches the missing-key guard below.
    """
    want = set(want)
    out, n = {}, 0
    for k, v in sd.items():
        if k not in want:
            for old, new in _KEY_ALIASES:
                nk = new + k[len(old):]
                if k.startswith(old) and nk in want and nk not in sd:
                    k, n = nk, n + 1
                    break
        out[k] = v
    return out, n


def _own_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--ckpt", required=True, help="the frozen checkpoint to score")
    p.add_argument("--run_config", default="",
                   help="config.json from the run that produced --ckpt; the "
                        "architecture fields are restored from it so the "
                        "state_dict cannot silently half-load")
    p.add_argument("--source", default="gamus")
    p.add_argument("--split", default="test")
    p.add_argument("--tiles", type=int, default=0, help="0 = the whole store")
    p.add_argument("--sliding_tiles", type=int, default=400, help="0 = skip")
    p.add_argument("--out", default="outputs/v3_test")
    p.add_argument("--skip_plain", action="store_true")
    p.add_argument("--skip_tta", action="store_true")
    p.add_argument("--tta_scales", default="",
                   help="comma list, e.g. 1.0,1.25 (v4/DAV2's); empty keeps the "
                        "run config's")
    return p.parse_known_args(argv)


def _restore_arch(cfg: Config, run_config: str, argv: list[str]) -> None:
    """Overlay the training run's config onto `cfg`, CLI flags winning.

    `decoder_dim`, `n_bins`, `encoder_feature_indices` and friends decide the
    shapes.  Get one wrong and `load_state_dict(strict=False)` reports it in a
    count nobody reads, then scores a partly-random network.  Keys in `_SKIP`
    describe the *run* (where its data and output lived), not the network, and
    restoring those would point this eval back at the Lightning studio.
    """
    if not run_config:
        return
    d = json.loads(Path(run_config).read_text())
    explicit = {x[2:].split("=")[0] for x in argv if x.startswith("--")}
    names = set(vars(cfg)) - _SKIP - explicit
    for k, v in d.items():
        if k in names:
            cur = getattr(cfg, k)
            setattr(cfg, k, tuple(v) if isinstance(cur, tuple) else v)
    print(f"[cfg] architecture restored from {run_config}")


def main() -> None:
    a, rest = _own_args(sys.argv[1:])
    cfg = parse_config(rest)
    _restore_arch(cfg, a.run_config, rest)
    cfg.hf_token = resolve_hf_token(cfg)
    cfg.gpu_augment = False        # the eval path normalises on the host
    cfg.per_landscape_metrics = True
    if a.tta_scales:
        cfg.tta_scales = tuple(float(x) for x in a.tta_scales.split(",") if x.strip())
    print(f"[cfg] tta_scales={tuple(cfg.tta_scales)}")

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = cfg.tf32
    torch.backends.cudnn.allow_tf32 = cfg.tf32
    with contextlib.suppress(Exception):
        torch.set_float32_matmul_precision(cfg.matmul_precision)
    if cfg.sdp_flash:
        with contextlib.suppress(Exception):
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)

    n_gpu = torch.cuda.device_count()
    device = torch.device("cuda" if n_gpu else "cpu")
    print(f"[env] torch={torch.__version__} gpus={n_gpu}"
          + (f" {torch.cuda.get_device_properties(0).name}" if n_gpu else ""))

    spec = PreprocSpec.from_config(cfg)
    print(f"[preproc] {spec.canonical_gsd_m} m/px, tile {spec.tile_size}, "
          f"stretch={spec.radiometric_stretch}, "
          f"mean={tuple(round(v, 3) for v in spec.mean)}")

    d = Path(cfg.data_root) / a.source / a.split
    if not store_exists(d):
        raise SystemExit(f"no packed store at {d} — run pack_gamus_png.py first")
    store = PackedStore(d)
    n = min(a.tiles, len(store)) if a.tiles else len(store)
    print(f"[data] {a.source}/{a.split}: {len(store)} tiles @ {store.tile_px}px "
          f"/ {store.gsd_m} m — scoring {n}")
    if spec.radiometric_stretch:
        t0 = time.time()
        # One threaded pass on the parent; every worker would otherwise redo a
        # full-tile histogram inside every crop.  Cached next to the shards.
        if store.prime_stretch_bounds(spec.stretch_lo_pct, spec.stretch_hi_pct,
                                      workers=max(4, cfg.num_workers or 8)):
            print(f"[data] stretch bounds ready in {time.time() - t0:.0f}s")

    dl = DataLoader(
        TileDataset(cfg, store, spec, a.source, train=False, length=n),
        batch_size=max(1, int(cfg.batch_size * max(1, cfg.eval_batch_mult))),
        shuffle=False, num_workers=cfg.num_workers, pin_memory=n_gpu > 0,
        persistent_workers=False,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers else None)
    full = FullTileDataset(cfg, store, spec, a.source, length=n)

    model = DepthWizardNetV3(cfg).to(device)
    if cfg.channels_last and n_gpu:
        model = model.to(memory_format=torch.channels_last)
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    sd, renamed = fit_keys(ck.get("model", ck), model.state_dict().keys())
    if renamed:
        print(f"[ckpt] {renamed} encoder keys renamed across the transformers "
              f"DINOv3 layout change (4.x layer.N <-> 5.x model.layer.N)")
    # Both failure modes below mean the same thing — this config describes a
    # different network from the one that was trained — and both would otherwise
    # produce a plausible-looking bad score instead of an error.  A shape clash
    # raises; a name that is simply absent comes back in `missing` and leaves
    # randomly initialised weights in the graph.
    try:
        miss, unexp = model.load_state_dict(sd, strict=False)
    except RuntimeError as e:
        raise SystemExit(f"checkpoint does not fit this config — wrong "
                         f"--run_config?\n{e}") from e
    print(f"[ckpt] {a.ckpt}: {len(sd)} tensors, missing {len(miss)}, "
          f"unexpected {len(unexp)}"
          + (f"  epoch={ck['epoch']}" if isinstance(ck, dict) and "epoch" in ck else ""))
    if miss:
        raise SystemExit(f"checkpoint does not fit this config — missing "
                         f"{len(miss)} parameters, first few: {miss[:8]}")
    model.eval()

    res: dict = {"checkpoint": str(a.ckpt), "store": str(d), "tiles_scored": n,
                 "store_tiles": len(store), "config": safe_config_dict(cfg),
                 "preproc": spec.to_dict()}
    tag = f"test_{a.source}_{a.split}"

    for suffix, on, tta in (("plain", not a.skip_plain, False),
                            ("tta", cfg.tta and not a.skip_tta, True)):
        if not on:
            continue
        t0 = time.time()
        res[f"{tag}_{suffix}"] = evaluate(model, dl, cfg, device, use_tta=tta)
        print(f"[test] {tag}_{suffix:<6} {format_line(res[f'{tag}_{suffix}'])}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
        (out_dir / "test_metrics.json").write_text(json.dumps(res, indent=2))

    if a.sliding_tiles:
        t0 = time.time()
        res[f"{tag}_sliding_tta"] = sliding_eval(
            model, full, cfg, spec, device, tta=cfg.tta,
            max_tiles=min(a.sliding_tiles, n))
        print(f"[test] {tag}_sliding_tta {format_line(res[f'{tag}_sliding_tta'])}"
              f"  ({time.time() - t0:.0f}s)")
        print("       ^ full-tile sliding window at native GSD, held-out tiles — "
              "THIS is the number to quote")

    (out_dir / "test_metrics.json").write_text(json.dumps(res, indent=2))
    print(f"[out] {out_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
