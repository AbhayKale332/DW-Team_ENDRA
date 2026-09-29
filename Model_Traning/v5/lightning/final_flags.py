"""The final v5 fine-tune: one Lightning H100, no DFC23, trees and sparse land first.

Warm start from `resume-v4-1.6`'s best.pt (Kaggle run `v5_probe_v4init_mvs3dm`).
Read with `V4_modal/v5_flags.py` beside it: this is `FLAGS` from there plus the
overrides below, each with its reason.  `final_h100.sh` calls `build()` through
the CLI at the bottom; nothing here is hand-copied into the shell script.

What that run's metrics.json says, and what each group below answers:

* trees are the worst class — GAMUS `tree` RMSE 4.19 m / delta1 0.43, flat
  from e1 to e9; MVS3DM `forested` 2.07 m vs `sparse` 0.96 m; > 15 m is
  6.9 m under on MVS3DM                           -> DATA, SAMPLING, SELECTION
* the map is over-smoothed — grad_ratio 0.23 (MVS3DM) / 0.39 (GAMUS)
                                                  -> SHARPNESS
* DFC23 / india_labeled put 97 % of green pixels at < 1 m: they teach
  "a tree is ground" and the ExG mask only patched it       -> both dropped
* 2x T4 fp16: 1.7 img/s, ~108 k crops in 10 h     -> H100 sizing from a sweep
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V5 = HERE.parent
sys.path.insert(0, str(V5.parent / "V4_modal"))
sys.path.insert(0, str(V5))

from v5_flags import FLAGS as BASE  # noqa: E402

# Effective batch every LR below was set for (the Kaggle run: 2 x 1 x 16).
REF_EFFECTIVE_BATCH = 32

OVERRIDES = {
    # ---- DATA: every labeled store whose heights keep trees; DFC23 and
    # india_labeled (trees = 0 m) are gone.  mvs3dm stays first, so the primary
    # val set — and the per-epoch number — is the one resume-v4-1.6 reported.
    # neon (forest / savanna / scrub LiDAR) is inserted by build() when present.
    "datasets": "mvs3dm,gamus,us3d,synrs3d_g05,synrs3d_g1",
    "sampler_weights": "gamus:2.5,mvs3dm:1.5,us3d:1,neon:2,synrs3d_g05:1,synrs3d_g1:0.5",
    # Within a store, draw forest tiles 2x and sparse tiles 1.5x as often as
    # urban ones.  The store mix above is unchanged by this.
    "landscape_sampler_boost": "forested:2.0,sparse:1.5",
    # ---- no coarse-label source left except NEON (1 m LiDAR rasters on 0.5 m
    # pixels: scored on 2x2 blocks so the upsampling blur is never taught).
    "coarse_label_sources": "",
    "coarse_label_m": "1.0",
    "coarse_mask_veg": "false",         # NEON's trees are real heights
    # ---- SELECTION: best.pt on forested + sparse RMSE, not global RMSE, which
    # urban ground dominates.
    "select_on": "mvs3dm",
    # ---- SHARPNESS: gradient and normal terms up; the balancer stays at the
    # values v4 showed not to cost the 0-2 m stratum.
    "w_grad": "1.0",
    "w_normal": "0.5",
    "stratum_balance_beta": "0.5",
    "stratum_weight_clip": "5.0",
    # ---- the warm start's own architecture: 96 bins over 0-120 m, detail branch
    "bin_max_m": "120",
    "detail_branch": "true",
    # ---- SCHEDULE: warm start, all 24 blocks trainable from step 0 with the
    # encoder LR ramped in over epoch 1.  LRs are 0.5x the defaults (the plateaued
    # run used 0.3x; the data mix changed, so it gets a little more room).
    "freeze_epochs": "0",
    "encoder_unfreeze_blocks": "0",
    "llrd": "0.90",
    "unfreeze_warmup_epochs": "1.0",
    "learning_rate": "1.5e-4",
    "encoder_lr": "3e-5",
    "warmup_frac": "0.05",
    "epochs": "30",
    "max_minutes": "330",
    # Big epochs: 4-5 val sets per eval are then < 10 % of wall-clock.
    "crops_per_epoch": "48000",
    "eval_every": "1",
    # ---- AUGMENTATION: NEON's source is 0.1 m aerial, sharper than Cartosat
    "photo_blur_p": "0.3",
    # ---- H100 (batch / accum / compile / workers / eval mult come from sizing)
    "amp_dtype": "bf16",
    "grad_checkpoint_encoder": "false",
    "opt_fused": "true",
    "gpu_augment": "true",
    # ---- OUTPUT
    "save_full_state": "true",
    "full_state_every": "3",
    "session_minutes": "0",
    "test_sources": "mvs3dm:test,gamus:test",
    "final_sliding_eval": "true",
    "tta": "true",
    "export_onnx": "true",
    "make_zip": "false",
    "gallery_sources": "synrs3d,us3d,mvs3dm",
    "landscape_gallery_source": "gamus",
}

# Keys the sweep measures; build() refuses to run without them.
SIZED = ("batch_size", "grad_accum", "compile_model", "num_workers",
         "prefetch_factor", "eval_batch_mult")


def build(data_root: str | Path, sizing: dict | None = None,
          output_dir: str | None = None, resume: str | None = None) -> dict:
    """The full flag dict for this machine and this data root."""
    root = Path(data_root)
    f = {**BASE, **OVERRIDES, "data_root": str(root)}
    for k in ("dfc23", "india_labeled", "india_unlabeled"):
        assert k not in f["datasets"], f"{k} must not be in the final run"
    if (root / "neon" / "train" / "index.json").is_file():
        ds = f["datasets"].split(",")
        f["datasets"] = ",".join([ds[0], "neon"] + ds[1:])
        f["coarse_label_sources"] = "neon"
        # NEON's CHM has no buildings; the roughness rule calls 1024 of its 1846
        # train tiles (every closed-canopy site) "urban" — so without this the
        # forest boost and select_on would skip exactly the forests (eda_neon.py)
        f["landscape_no_urban_sources"] = "neon"
        f["select_on"] = "neon,mvs3dm"
        f["gallery_sources"] += ",neon"
        f["landscape_gallery_source"] = "neon"
        if (root / "neon" / "test" / "index.json").is_file():
            f["test_sources"] = "neon:test," + f["test_sources"]
    for src in list(f["datasets"].split(",")):
        if not (root / src / "train" / "index.json").is_file():
            raise SystemExit(f"[flags] {root / src / 'train'} missing — run `final_h100.sh fetch`")
    if sizing:
        for k in SIZED:
            if k in sizing:
                f[k] = str(sizing[k]).lower() if isinstance(sizing[k], bool) else str(sizing[k])
        # sqrt LR scaling when the sweep's batch moved the effective batch
        eff = int(f["batch_size"]) * int(f["grad_accum"])
        if eff != REF_EFFECTIVE_BATCH:
            s = math.sqrt(eff / REF_EFFECTIVE_BATCH)
            for k in ("learning_rate", "encoder_lr"):
                f[k] = f"{float(OVERRIDES[k]) * s:.3g}"
    if output_dir:
        f["output_dir"] = str(output_dir)
    if resume:
        f["resume"] = str(resume)
    return f


def to_argv(flags: dict) -> list[str]:
    return [x for k, v in flags.items() for x in (f"--{k}", str(v))]


def check(flags: dict) -> None:
    """Every key must be a real Config field: parse_config ends in
    parse_known_args, so a typo is silently dropped."""
    from dataclasses import fields

    from config import Config, parse_config

    names = {x.name for x in fields(Config)}
    bad = sorted(set(flags) - names)
    if bad:
        raise SystemExit(f"[flags] not v5 config fields: {bad}")
    cfg = parse_config(to_argv(flags))
    for k in ("batch_size", "grad_accum", "select_on", "landscape_sampler_boost",
              "coarse_label_sources", "landscape_no_urban_sources", "learning_rate",
              "encoder_lr"):
        got, want = getattr(cfg, k), flags.get(k)
        if want is not None and str(got) != str(type(got)(want)):
            raise SystemExit(f"[flags] {k}: parsed {got!r}, wanted {want!r}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--sizing", default="")
    ap.add_argument("--output_dir", default="")
    ap.add_argument("--resume", default="")
    ap.add_argument("--format", choices=("argv", "json"), default="argv")
    ap.add_argument("--no_check", action="store_true")
    a = ap.parse_args()
    sz = json.loads(Path(a.sizing).read_text()) if a.sizing and Path(a.sizing).is_file() else None
    fl = build(a.data_root, sz, a.output_dir or None, a.resume or None)
    if not a.no_check:
        check(fl)
    if a.format == "json":
        print(json.dumps(fl, indent=2))
    else:
        print("\n".join(to_argv(fl)))
