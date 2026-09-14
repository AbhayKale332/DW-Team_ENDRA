"""DepthWizard v3 — configuration.

One dataclass, every knob CLI-overridable (same pattern as v1/v2).

What changed vs v2 (see README.md §1 for the full post-mortem):
  * scale augmentation is *crop-then-resample*, so a tile is never zero-padded
  * data is materialised once into memmap shards instead of streamed per step
  * encoder is fully unfrozen after a short warmup, with layer-wise LR decay
  * heads run at 1/2 resolution (DPT-standard) instead of full res
  * height-stratum balanced loss weighting + weight EMA
  * the train-time preprocessing contract is serialised into the checkpoint so
    `infer/predict.py` cannot drift from it
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, fields
from pathlib import Path

_V3_DIR = Path(__file__).resolve().parent

_DEFAULT_OUTPUT_DIR = os.environ.get("DW_OUTPUT_DIR", str(_V3_DIR / "outputs" / "v3"))
_DEFAULT_DATA_ROOT = os.environ.get("DW_DATA_ROOT", str(_V3_DIR / "data"))

# GAMUS ships integer class ids in classes/<split>/<STEM>_CLS.h5.  v1/v2 asserted
# the name order below, but the measured val histogram contradicts it (id 5 is
# 16% of all pixels, id 0 is 0.1% — see README §1.6).  Until the ids are pinned
# down empirically, v3 reports per-class metrics under *neutral* labels and dumps
# a class histogram + mean-height table (`--dump_class_stats`) so the mapping can
# be fixed from evidence rather than assumption.
CLASS_IDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
CLASS_NAMES: tuple[str, ...] = tuple(f"class{i}" for i in CLASS_IDS)
N_SEG_CLASSES = 8          # 7 real ids + 7 == ignore
SEG_IGNORE_INDEX = 7

# Per-height-stratum edges (m) for the stratified / balanced RMSE table.
HEIGHT_STRATA_M: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0, 20.0, 1e9)


@dataclass
class Config:
    # ----- data ------------------------------------------------------
    datasets: str = "gamus,synrs3d"         # comma list of {gamus,geonrw,synrs3d}
                                            # (a source that was not prepared is skipped)
    data_root: str = _DEFAULT_DATA_ROOT     # where prepare_data.py wrote the shards
    sampler_weights: str = "gamus:1,geonrw:1,synrs3d:1"
    gamus_repo: str = "earthflow/GAMUS"
    geonrw_repo: str = "torchgeo/geonrw"
    synrs3d_repo: str = "JTRNEO/SynRS3D"

    # An "epoch" is a fixed number of random crops, decoupled from tile count so
    # the LR schedule and the wall-clock budget stay predictable.
    crops_per_epoch: int = 12000
    val_tiles: int = 400
    max_valid_height_m: float = 200.0       # AGL above this is LiDAR noise, not a building

    # ----- geometry / GSD -------------------------------------------
    tile_size: int = 512                    # model input, multiple of patch(16)
    canonical_gsd_m: float = 0.5            # the GSD the model works at
    gsd_jitter_lo_m: float = 0.30           # train-time scale augmentation range
    gsd_jitter_hi_m: float = 1.20
    gsd_jitter_p: float = 0.9

    # ----- radiometry / domain gap ----------------------------------
    # v2 trained on raw GAMUS bytes with zero photometric augmentation and then
    # hallucinated ~4 m of height on flat ground in a different sensor's imagery
    # (Inria Austin).  Both knobs below exist to close that gap, and the stretch
    # is part of the *inference contract* — it runs identically in predict.py.
    radiometric_stretch: bool = True        # per-scene 2/98-percentile stretch
    stretch_lo_pct: float = 2.0
    stretch_hi_pct: float = 98.0
    photo_p: float = 0.9                    # prob. a train crop gets photometric jitter
    photo_brightness: float = 0.25
    photo_contrast: float = 0.25
    photo_saturation: float = 0.35
    photo_gamma: float = 0.35               # gamma ~ exp(U(-g, g))
    photo_channel_gain: float = 0.12        # per-channel multiplicative gain
    photo_blur_p: float = 0.2
    photo_noise_std: float = 0.015

    # ----- model -----------------------------------------------------
    encoder_model_id: str = "facebook/dinov3-vitl16-pretrain-sat493m"
    encoder_feature_indices: tuple = (6, 12, 18, 24)
    decoder_dim: int = 256
    n_bins: int = 96
    bin_min_m: float = 0.0
    bin_max_m: float = 120.0

    # ----- schedule --------------------------------------------------
    epochs: int = 26
    freeze_epochs: int = 2                  # encoder frozen for this many epochs
    max_minutes: float = 330.0              # hard wall-clock cap for the whole run
    learning_rate: float = 3e-4             # decoder + heads
    encoder_lr: float = 6e-5                # top encoder block; decayed downward
    llrd: float = 0.80                      # layer-wise LR decay per block
    weight_decay: float = 0.05
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    # Sized for a 48 GB L40S running ViT-L/16 at 512 px with the encoder
    # UNFROZEN and no gradient checkpointing: ~1.05 GB of activations per
    # sample plus ~5 GB of weights/grads/AdamW state.  `train_L40S.sh` probes
    # the real number on the real card at startup and overrides this; the
    # default is the conservative value that is known to fit.
    batch_size: int = 24
    grad_accum: int = 2                     # effective batch 48
    eval_batch_mult: int = 2                # eval keeps no activations -> 2x batch
    num_workers: int = 12
    prefetch_factor: int = 6
    ema_decay: float = 0.9995               # 0 -> disable weight EMA

    # ----- precision / perf -----------------------------------------
    amp: bool = True
    amp_dtype: str = "bf16"                 # bf16 on Hopper
    channels_last: bool = True
    grad_checkpoint_encoder: bool = False    # trades ~35% throughput for VRAM
    grad_checkpoint_decoder: bool = False
    compile_model: bool = False             # recompiles on unfreeze; opt-in
    # Photometric jitter + encoder normalisation run on the GPU over the whole
    # batch instead of per crop in a DataLoader worker, and the worker hands
    # over uint8 HWC.  ~3x the loader throughput and 4x less H2D traffic; see
    # dwdata/gpu_aug.py for the measurements.  Turn it off to fall back to the
    # all-CPU path (identical maths, much slower).
    gpu_augment: bool = True
    sdp_flash: bool = True                  # prefer the fused SDPA kernels
    cudnn_benchmark: bool = True
    tf32: bool = True
    matmul_precision: str = "high"
    seed: int = 42

    # ----- loss ------------------------------------------------------
    w_l1: float = 1.0
    w_grad: float = 0.5
    w_silog: float = 1.0
    silog_lambda: float = 0.85
    silog_shift: float = 1.0
    w_head_a: float = 0.3                   # auxiliary; `fused` is the product
    w_head_b: float = 0.3
    w_fused: float = 1.0
    w_bin_ce: float = 0.5
    w_seg: float = 0.2
    w_normal: float = 0.3                   # surface-normal loss -> keeps flat things flat
    w_flat: float = 0.2                     # planarity penalty on ground/road/water
    stratum_balance_beta: float = 0.5       # 0 -> plain pixel-uniform loss
    stratum_weight_clip: float = 5.0

    # ----- eval / io -------------------------------------------------
    eval_every: int = 2
    tta: bool = True
    tta_scales: tuple = (1.0,)              # dihedral only by default; add 1.25 for +scale
    final_sliding_eval: bool = True         # full-tile sliding-window eval at the end
    per_class_metrics: bool = True
    dump_class_stats: bool = True
    class_names: tuple = CLASS_NAMES
    output_dir: str = _DEFAULT_OUTPUT_DIR
    n_qualitative: int = 12

    # ----- misc ------------------------------------------------------
    hf_token: str = ""
    smoke: bool = False
    resume: str = ""                        # checkpoint to resume/warm-start from
    make_zip: bool = True

    # -----------------------------------------------------------------
    def apply_smoke(self) -> "Config":
        if not self.smoke:
            return self
        self.crops_per_epoch = 24
        self.val_tiles = 4
        self.epochs = 2
        self.freeze_epochs = 1
        self.max_minutes = 20.0
        self.batch_size = 2
        self.grad_accum = 1
        self.eval_batch_mult = 1
        self.num_workers = 0
        self.compile_model = False
        self.cudnn_benchmark = False
        self.eval_every = 1
        self.n_qualitative = 2
        self.final_sliding_eval = False
        self.ema_decay = 0.0
        return self

    def validate(self) -> "Config":
        assert self.tile_size % 16 == 0, "tile_size must be a multiple of 16"
        assert self.amp_dtype in ("bf16", "fp16")
        assert self.gsd_jitter_lo_m <= self.gsd_jitter_hi_m
        assert 0 <= self.freeze_epochs <= self.epochs
        try:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        return self

    def dataset_list(self) -> list[str]:
        return [d.strip() for d in self.datasets.split(",") if d.strip()]

    def sampler_weight_map(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for part in self.sampler_weights.split(","):
            if ":" in part:
                k, v = part.split(":")
                out[k.strip()] = float(v)
        return out


def parse_config(argv: list[str] | None = None) -> Config:
    cfg = Config()
    p = argparse.ArgumentParser(description="DepthWizard v3")
    for f in fields(cfg):
        cur = getattr(cfg, f.name)
        if isinstance(cur, tuple):
            continue
        t = type(cur)
        if t is bool:
            p.add_argument(
                f"--{f.name}", nargs="?", const=True, default=None,
                type=lambda x: str(x).lower() in ("1", "true", "yes", "y"),
            )
        else:
            p.add_argument(f"--{f.name}", type=t)
    args, _ = p.parse_known_args(argv)
    for k, v in vars(args).items():
        if v is not None:
            setattr(cfg, k, v)
    return cfg.apply_smoke().validate()


def safe_config_dict(cfg: Config) -> dict:
    from dataclasses import asdict

    d = asdict(cfg)
    d.pop("hf_token", None)
    return d
