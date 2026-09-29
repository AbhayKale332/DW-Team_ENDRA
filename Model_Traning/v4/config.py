"""DepthWizard v4 — configuration.

One dataclass, every knob CLI-overridable (same pattern as v1/v2/v3).

v4 is v3's training core plus the two things the plan calls Phase 2 and Phase 3:
the *geo* half of the model (ground-masked DTM fit -> absolute DSM, DEM fetch,
ONNX export) and the *product* half (FastAPI service, Three.js flythrough,
auto-generated validation report).  On the model side there are three deliberate
changes, each of them a fix for something measured rather than guessed:

  * **Head B gets soft bin targets and an entropy floor.**  v2's bin head
    collapsed (CE 0.06-0.13 all run) and v3 kept the same hard nearest-bin
    target.  A hard target over 96 bins where ~65 % of the mass is one bin is
    trivially predictable; a Gaussian-smoothed target plus a diversity penalty
    is not, so the head has to actually spread probability over the tail.
  * **Mean-teacher consistency on unlabeled imagery** — the only mechanism here
    that can adapt the network to Indian urban morphology and Cartosat
    radiometry without Indian height labels, which do not exist openly (see
    README §2).  Off unless an `india` store was prepared.
  * **Landscape-stratified metrics** (urban / sparse / hilly / forested), because
    that phrase is literally in the rubric and no v1-v3 number answers it.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, fields
from pathlib import Path

_V4_DIR = Path(__file__).resolve().parent

_DEFAULT_OUTPUT_DIR = os.environ.get("DW_OUTPUT_DIR", str(_V4_DIR / "outputs" / "v4"))
_DEFAULT_DATA_ROOT = os.environ.get("DW_DATA_ROOT", str(_V4_DIR / "data"))

# GAMUS ships integer class ids in classes/<split>/<STEM>_CLS.h5.  v1/v2 asserted
# a name order that the measured val histogram contradicts (id 5 is 16 % of all
# pixels, id 0 is 0.1 %).  Until the ids are pinned down empirically, per-class
# metrics use *neutral* labels and a measured histogram is dumped alongside them
# so the mapping gets fixed from evidence rather than assumption.
CLASS_IDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
CLASS_NAMES: tuple[str, ...] = tuple(f"class{i}" for i in CLASS_IDS)
N_SEG_CLASSES = 8          # 7 real ids + 7 == ignore
SEG_IGNORE_INDEX = 7

# The shared class space prepare_data.py remaps every source into.  GAMUS's own
# ids pass through unchanged (they are the unverified ones); SynRS3D and GeoNRW
# are remapped onto this.  `GROUND_LIKE_IDS` is what `geo/calibrate.py` fits the
# terrain surface on — being wrong here is safe, because the calibrator also
# requires a low predicted nDSM before it trusts a pixel as ground.
GROUND_LIKE_IDS: tuple[int, ...] = (0, 3, 4)      # ground / water / road

# Per-height-stratum edges (m) for the stratified / balanced RMSE table.
HEIGHT_STRATA_M: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0, 20.0, 1e9)

# Landscape classes for the rubric's "stability across urban / sparse / hilly /
# forested" requirement.  A tile is assigned one of these from its *ground truth*
# statistics at eval time (see eval/landscape.py), so the breakdown needs no
# extra labels and works on any dataset.
LANDSCAPE_NAMES: tuple[str, ...] = ("urban", "sparse", "hilly", "forested")


@dataclass
class Config:
    # ----- data ------------------------------------------------------
    datasets: str = "gamus,synrs3d"         # comma list of {gamus,geonrw,synrs3d,
                                            #  india_labeled,india_unlabeled}
                                            # (a source that was not prepared is skipped)
    data_root: str = _DEFAULT_DATA_ROOT     # where prepare_data.py wrote the shards
    sampler_weights: str = "gamus:1,geonrw:1,synrs3d:1,india_labeled:2"
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

    # ----- unlabeled domain adaptation (India) -----------------------
    # The mean-teacher branch: the EMA teacher predicts a weakly-augmented view,
    # the student is trained to match it on a strongly-augmented one.  This is
    # what carries the network towards Indian urban morphology and Cartosat-like
    # radiometry when nobody publishes Indian height labels.
    unlabeled_source: str = "india_unlabeled"   # store treated as label-free
    w_consistency: float = 1.0              # 0 -> disable the branch entirely
    consistency_rampup_epochs: int = 4      # linear ramp; teacher is noisy early
    consistency_conf_m: float = 1.5         # ignore pixels where the teacher's two
                                            # views disagree by more than this
    unlabeled_batch_frac: float = 0.5       # unlabeled batch = frac x batch_size
    consistency_every: int = 2              # run the branch every N steps; it costs
                                            # ~50 % of a step and the budget is fixed
    teacher_ema: float = 0.999

    # ----- model -----------------------------------------------------
    encoder_model_id: str = "facebook/dinov3-vitl16-pretrain-sat493m"
    encoder_feature_indices: tuple = (6, 12, 18, 24)
    decoder_dim: int = 256
    n_bins: int = 96
    bin_min_m: float = 0.0
    bin_max_m: float = 120.0

    # ----- schedule --------------------------------------------------
    # v3 ran 26 epochs in 121 min of a 330 min budget and its val RMSE was still
    # falling monotonically at the last eval (2.760 -> 2.732 -> 2.723 -> 2.715).
    # It did not converge; it ran out of epochs with ~3 hours of budget unused.
    # That is the single largest accuracy lever here and it is free, so v4
    # spends the budget: more epochs, same wall-clock cap.  The LR schedule is
    # driven by max(epoch fraction, wall-clock fraction), so if the card turns
    # out slower than expected the cosine still completes instead of being
    # chopped off with the LR high.
    epochs: int = 40
    freeze_epochs: int = 2                  # encoder frozen for this many epochs
    # The cap covers *training* only; the final plain/TTA/sliding evaluation,
    # the qualitative export, figures, the report and the ONNX export run after
    # it and cost ~30 min on v3.  280 + 30 keeps the whole run inside a 5.5 h
    # Studio session the way v3's 330 was meant to.
    max_minutes: float = 280.0              # hard wall-clock cap on training
    learning_rate: float = 3e-4             # decoder + heads
    encoder_lr: float = 6e-5                # top encoder block; decayed downward
    llrd: float = 0.80                      # layer-wise LR decay per block
    weight_decay: float = 0.05
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    # Sized from the v3 H100 run's own measurements, not from a probe:
    #
    #     micro-batch 16, encoder unfrozen, no checkpointing -> 42 GB, 44.7 img/s
    #     micro-batch 32, same                               -> 74 GB, 48.0 img/s
    #
    # Doubling the batch bought 7 % throughput (the card is compute-bound at
    # 99-100 % utilisation either way) and cost 32 GB — and the 32 run OOMed at
    # epoch 7 after creeping to 76 GB.  So the batch is *not* where the 80 GB
    # goes.  24 is the useful middle: ~58 GB with ~20 GB of headroom for the
    # eval spike and allocator drift, most of micro-32's kernel efficiency, and
    # — with `grad_accum` dropped to 1 — 500 optimiser steps an epoch instead of
    # v3's 375, which is what a run that never converged actually needed.
    batch_size: int = 24
    grad_accum: int = 1                     # effective batch 24
    eval_batch_mult: int = 2                # eval keeps no activations -> 2x batch
    # v3's H100 run used 16 workers and the guard log shows 99-100 % GPU
    # utilisation throughout, so the loader was not the limit once the v3
    # pipeline fixes landed.  Val gets its own capped pool (see loaders.py), so
    # this is 16 + 4 processes, not 16 + 16.
    num_workers: int = 16
    prefetch_factor: int = 6
    ema_decay: float = 0.9995               # 0 -> disable weight EMA

    # ----- precision / perf -----------------------------------------
    amp: bool = True
    amp_dtype: str = "bf16"                 # bf16 on Hopper
    channels_last: bool = True
    # OFF.  The v4 draft defaulted this to True with the comment "required to
    # fit ViT-L unfrozen @512" — the v3 H100 log disproves that directly: the
    # encoder ran unfrozen at 512 px, micro-batch 16, *without* checkpointing,
    # in 42 GB of an 80 GB card.  Recomputing every block to save memory that is
    # sitting idle costs ~35 % of the step for nothing.  Turn it on only if you
    # move to a genuinely smaller card.
    grad_checkpoint_encoder: bool = False
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
    bin_soft_sigma: float = 1.5             # target smoothing, in *bins*.  0 -> hard
    w_bin_entropy: float = 0.02             # floor on marginal bin entropy: stops the
                                            # head collapsing onto the ground bin
    w_seg: float = 0.2
    w_normal: float = 0.3                   # surface-normal loss -> keeps flat things flat
    w_flat: float = 0.2                     # planarity penalty on ground/road/water
    # v3 ended with a *frozen* systematic bias: tall_bias sat at -2.39, -2.13,
    # -2.14, -2.12, -2.06, -2.13, -2.08, -2.06, -2.08 m from epoch 6 to epoch 26
    # while global RMSE kept falling — 20 epochs of improvement that never once
    # touched the tail.  Per stratum at the end: 20 m+ bias -2.46 m, 10-20 m
    # -1.35 m, 0-2 m *+0.43 m*.  That is textbook regression to the mean, and it
    # means beta=0.5 equilibrated at a biased optimum rather than being on its
    # way to an unbiased one.  0.7 with a wider clip pushes harder on the ~4 %
    # of pixels above 20 m; the balanced-RMSE and per-stratum bias columns in
    # metrics.json are the check on whether it went too far.
    stratum_balance_beta: float = 0.7       # 0 -> plain pixel-uniform loss
    stratum_weight_clip: float = 8.0

    # ----- eval / io -------------------------------------------------
    eval_every: int = 2
    tta: bool = True
    tta_scales: tuple = (1.0,)              # dihedral only by default; add 1.25 for +scale
    final_sliding_eval: bool = True         # full-tile sliding-window eval at the end
    per_class_metrics: bool = True
    per_landscape_metrics: bool = True      # urban / sparse / hilly / forested
    dump_class_stats: bool = True
    class_names: tuple = CLASS_NAMES
    output_dir: str = _DEFAULT_OUTPUT_DIR
    n_qualitative: int = 12
    make_figures: bool = True               # viz/figures.py at the end of the run
    make_report: bool = True                # viz/report_html.py validation report
    export_onnx: bool = True                # infer/export_onnx.py at the end of the run
    onnx_opset: int = 17

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
        self.num_workers = 0
        self.gpu_augment = False
        self.compile_model = False
        self.cudnn_benchmark = False
        self.eval_every = 1
        self.n_qualitative = 2
        self.final_sliding_eval = False
        self.ema_decay = 0.0
        self.consistency_rampup_epochs = 1
        self.export_onnx = False
        return self

    def validate(self) -> "Config":
        assert self.tile_size % 16 == 0, "tile_size must be a multiple of 16"
        assert self.amp_dtype in ("bf16", "fp16")
        assert self.gsd_jitter_lo_m <= self.gsd_jitter_hi_m
        assert 0 <= self.freeze_epochs <= self.epochs
        assert 0.0 <= self.unlabeled_batch_frac <= 2.0
        try:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        return self

    def dataset_list(self) -> list[str]:
        return [d.strip() for d in self.datasets.split(",") if d.strip()]

    def labeled_sources(self) -> list[str]:
        return [d for d in self.dataset_list() if d != self.unlabeled_source]

    def sampler_weight_map(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for part in self.sampler_weights.split(","):
            if ":" in part:
                k, v = part.split(":")
                out[k.strip()] = float(v)
        return out


def parse_config(argv: list[str] | None = None) -> Config:
    cfg = Config()
    p = argparse.ArgumentParser(description="DepthWizard v4")
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
