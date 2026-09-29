"""DepthWizard v5 — configuration.

One dataclass, every knob CLI-overridable (same pattern as v1-v4).

v5 is V4_Kaggle plus the fixes the v3/v4 comparison, the host FAQ and the
Cartosat-2E samples asked for (plan: `CompetitionContext/V5_Research_Additions.md`
and `FineTunning/v5/README.md` §v5):

  * **The class space is pinned from measurement** (below), which also fixes the
    flatness loss and the DTM ground mask — both were aimed at id 3, which is
    the ~9.4 m building class, and missed ground and road entirely.
  * **Coarse-label supervision.**  DFC23 / India nDSMs are ~2 m stereo
    upsampled onto 0.5 m pixels; they are scored after 4x pooling and kept out
    of every edge-sensitive term, so their blur stops being taught as texture.
  * **A full-resolution detail branch** (`detail_branch`) and edge-aware convex
    upsampling of the height maps.
  * **The geo half targets the judges' metric**: absolute DSM against SRTM /
    Copernicus 30 m on Cartosat-2S MERGED products (`geo/`, `infer/`).

What follows is the v4 account, kept because every v4 knob is still here.

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

_DEFAULT_OUTPUT_DIR = os.environ.get("DW_OUTPUT_DIR", str(_V4_DIR / "outputs" / "v5"))
_DEFAULT_DATA_ROOT = os.environ.get("DW_DATA_ROOT", str(_V4_DIR / "data"))

# GAMUS ships integer class ids in classes/<split>/<STEM>_CLS.h5.  v1-v4 used
# neutral names ("class0".."class6") and dumped a measured histogram so the
# mapping could be fixed from evidence.  v5 pins it from that evidence — v4
# Modal `test_gamus_test_tta.class_stats`, 2861 tiles:
#
#   id   px frac   mean GT (m)   reads as
#    0     5.3 %       1.24      unlabelled / other (0.1 % of v3 val)
#    1    17.9 %       0.49      ground
#    2    19.4 %       0.38      low vegetation
#    3    20.6 %       9.44      building   (r 0.86-0.88, negative tall bias)
#    4     1.7 %       0.19      water      (MAE 0.06 m — the flattest class)
#    5    14.7 %       1.06      road
#    6    20.4 %      10.23      tree       (lowest delta1 of any class)
#
# This matches the published GAMUS order (background, ground, low vegetation,
# building, water, road, tree).  It is evidence-based, not verified against
# rendered tiles yet: confirm with `tools/audit_labels.py --classes` before
# quoting per-class numbers by name.
#
# What the pinning fixes: v4 had `GROUND_LIKE_IDS = flat_ids = (0, 3, 4)` —
# i.e. the *building* class was treated as flat ground and ids 1 / 5 (ground,
# road) were not.  The flatness term therefore never touched ground, and the
# DTM ground mask fit terrain through rooftops.
CLASS_IDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
CLASS_NAMES: tuple[str, ...] = ("other", "ground", "low_veg", "building",
                                "water", "road", "tree")
N_SEG_CLASSES = 8          # 7 real ids + 7 == ignore
SEG_IGNORE_INDEX = 7

# Ids the DTM fit (`geo/calibrate.py`) may treat as bare earth.  Low vegetation
# is included: grass and crops sit at ~0.4 m mean, and the calibrator also
# demands a low predicted nDSM before it trusts any pixel.
GROUND_LIKE_IDS: tuple[int, ...] = (1, 2, 4, 5)   # ground / low veg / water / road
# Ids on which `flatness_loss` may penalise predicted curvature (still gated by
# a GT-flat Laplacian, so kerbs and embankments are untouched).
FLAT_CLASS_IDS: tuple[int, ...] = (1, 2, 4, 5)

# prepare_data.py packs SynRS3D and GeoNRW into an older "shared" space
# (0 ground, 1 vegetation, 2 building, 3 water, 4 road — `SYN_TO_SHARED`),
# which disagrees with GAMUS's own ids above: shared 2 (building) is GAMUS
# low-veg, shared 3 (water) is GAMUS building.  The seg head was trained on
# both at once.  Rather than repack 84 GiB, `dataset.py` remaps at load time.
# Shared 1 folds tree + rangeland + agriculture together and cannot be split
# back, so it becomes ignore.
SHARED_TO_GAMUS: dict[int, int] = {0: 1, 1: SEG_IGNORE_INDEX, 2: 3, 3: 4, 4: 5}
SHARED_SPACE_SOURCES: tuple[str, ...] = ("synrs3d", "geonrw")

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
    # Comma list of prepared store names:
    #   gamus  geonrw  india_labeled  india_unlabeled  us3d  mvs3dm
    #   synrs3d_g005 / synrs3d_g05 / synrs3d_g1   (one store per GSD family)
    #   dfc23_g050 / dfc23_g080 / …               (one store per measured GSD)
    # A source that was not prepared is skipped with a printed note.  The
    # suffixed names exist because `index.json` carries one `gsd_m` per store
    # and `dataset.py` turns it straight into the crop's effective GSD, so
    # pooling families under one nominal is a scale lie about the training set.
    # The FIRST name here that has a val split becomes the primary val set and
    # the one best.pt is selected on — keep `gamus` first to stay comparable
    # with v1-v4.
    datasets: str = "gamus,synrs3d_g05,synrs3d_g1"
    data_root: str = _DEFAULT_DATA_ROOT     # where prepare_data.py wrote the shards
    # Mix ratio, not tile counts — loaders.py normalises by store size.  Real
    # imagery is weighted over synthetic, and DFC23 is the only source that is
    # actually the deployment domain.
    sampler_weights: str = ("gamus:2,dfc23:2,geonrw:1,"
                            "synrs3d_g005:1,synrs3d_g05:1,synrs3d_g1:1,"
                            "india_labeled:2")
    gamus_repo: str = "earthflow/GAMUS"
    geonrw_repo: str = "torchgeo/geonrw"
    synrs3d_repo: str = "JTRNEO/SynRS3D"

    # An "epoch" is a fixed number of random crops, decoupled from tile count so
    # the LR schedule and the wall-clock budget stay predictable.
    crops_per_epoch: int = 12000
    # Final run: per-tile sampling boost by GT landscape class
    # (`eval/landscape.classify`: urban / sparse / hilly / forested), e.g.
    # "forested:2,sparse:1.5".  Applied *within* each store and renormalised,
    # so the store mix stays what `sampler_weights` says; only which tiles of a
    # store get drawn changes.  Classes are cached next to the shards as
    # `landscape_v1.npy`.  Empty -> uniform within a store (v1-v5 behaviour).
    landscape_sampler_boost: str = ""
    # Sources whose labels hold no buildings (NEON's CHM: roofs masked out), so a
    # tall tile the rule calls "urban" is canopy.  `classify` divides roughness by
    # mean height, and a closed 30 m canopy is smooth for its height: on NEON
    # 1024 of 1846 train tiles (every HARV / BART / GRSM / MLBS / TALL tile) came
    # out "urban" (tools/eda_neon.py).  For these sources urban -> forested, in
    # the sampler boost, the per-landscape metrics and `select_on`.
    landscape_no_urban_sources: str = ""
    # v5: >= 0 draws the val tiles as a seeded random sample instead of the
    # sorted-stem prefix.  v4's prefix was 87.5 % urban against a 57.6 % urban
    # test set and best.pt was selected on it.  -1 keeps the v1-v4 prefix.
    val_sample_seed: int = -1
    val_tiles: int = 400                    # 0 -> score the whole val store.
                                            # Non-zero selects the FIRST n tiles in
                                            # sorted-stem order (a prefix, not a
                                            # random sample) — kept at 400 so the
                                            # number stays comparable to v1-v4.
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
    # v5 — the Cartosat-2S look (`CompetitionContext/V5_Research_Additions.md` §1).
    # The judges' 0.6 m RGB is a PAN+MX *merge*: luminance at 0.6 m, colour at
    # 1.6 m.  `aug_pansharp` reproduces that on training crops — Cb/Cr are
    # area-downsampled by U(lo, hi) and upsampled back, luminance untouched — so
    # blurred chroma over sharp edges is not out-of-distribution at test time.
    # 1.6 / 0.6 = 2.67; the range brackets it.  Measure the real ratio with
    # `tools/audit_labels.py --cartosat` before trusting these bounds.
    aug_pansharp_p: float = 0.0             # v5 run: 0.5 (v5_flags.py)
    aug_pansharp_lo: float = 2.0
    aug_pansharp_hi: float = 3.3
    # Grayscale crops, so a PAN-only upload (one 0.6 m band) is still usable.
    aug_gray_p: float = 0.0                 # v5 run: 0.1

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
    # v5: a full-resolution RGB stem fused before the heads + RAFT-style convex
    # 2x upsampling of the height maps (models/heads.py).  Both zero-initialised,
    # so a freshly built v5 net starts out computing exactly what v4 did (up to
    # the upsampler, which starts as a 3x3 mean instead of bilinear).
    detail_branch: bool = False             # v5 run: true (v5_flags.py)
    detail_dim: int = 64
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
    # How many transformer blocks, counted from the output, become trainable at
    # the unfreeze.  0 = all of them, which is what the v4 Kaggle run did: 303 M
    # encoder parameters against 3 453 GAMUS tiles, at 2.1 img/s instead of the
    # 4.5 img/s the frozen encoder managed.  On a small labelled set the lower
    # blocks of a SAT-493M ViT-L are already better general-purpose overhead
    # imagery features than 42 passes over five US cities can make them, and
    # every block left frozen is ~12.6 M fewer parameters of gradient *and* of
    # AdamW state (~150 MB a block on a 15 GiB T4) and a faster step.  16 keeps
    # the top two thirds adaptable and roughly halves the unfrozen cost.
    encoder_unfreeze_blocks: int = 0
    # v5: after `freeze_epochs`, ramp the encoder groups' LR linearly from 0 over
    # this many epochs, and keep the decoder's AdamW state.  v1-v4 rebuilt the
    # optimiser at the unfreeze: the decoder lost its moments and 300 M encoder
    # parameters started at ~peak LR with no warm-up — the point v4's notes say
    # Head B's bin widths began to run away.  0 = the old step change.
    unfreeze_warmup_epochs: float = 1.0
    # The cap covers *training* only; the final plain/TTA/sliding evaluation,
    # the qualitative export, figures, the report and the ONNX export run after
    # it and cost ~30 min on v3.  280 + 30 keeps the whole run inside a 5.5 h
    # Studio session the way v3's 330 was meant to.
    max_minutes: float = 280.0              # hard wall-clock cap on training
    learning_rate: float = 3e-4             # decoder + heads
    encoder_lr: float = 6e-5                # top encoder block; decayed downward
    # Layer-wise LR decay, applied per block as `encoder_lr * llrd ** (L - depth)`.
    # 0.80 over a 24-block ViT-L puts the bottom block at 0.8**24 = 0.5 % of
    # `encoder_lr`, i.e. 2.8e-7 — the v4 Kaggle log prints exactly that range
    # ("lr 2.83e-07..3.00e-04").  The lower half of the encoder was therefore
    # paying full price in gradients, DDP all-reduce traffic and AdamW state to
    # move essentially not at all.  0.90 keeps the same shape over a span where
    # every unfrozen block has a usable LR; pair it with
    # `encoder_unfreeze_blocks` rather than widening the span further.
    llrd: float = 0.90                      # layer-wise LR decay per block
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
    # ----- fp16 loss scaling (Turing/T4 profile) --------------------
    # torch's GradScaler defaults are init_scale=2**16, growth_interval=2000 and
    # NO lower bound on the scale.  All three hurt here, and the third one ended
    # the v4 Kaggle run: see `_floor_scale` in train.py for the full account.
    #
    #   * 2**16 is high enough that the first four accumulation boundaries of
    #     epoch 1 were skipped just backing it down.  2**13 starts inside the
    #     range the run actually settled at and the scaler grows from there.
    #   * growth_interval 2000 is longer than an entire epoch on this profile
    #     (500 optimiser steps), so a scale that backed off once could not
    #     recover within the run.  500 lets it climb back each epoch.
    #   * amp_min_scale is the floor that makes a long run of skips survivable
    #     instead of terminal.  Below ~1.0 there is no headroom left to reclaim
    #     anyway: a skip at scale 1.0 means the true gradient is non-finite.
    amp_init_scale: float = 2.0 ** 13
    amp_growth_interval: int = 500
    amp_min_scale: float = 1.0
    amp_skip_patience: int = 50             # consecutive floored skips -> warn
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
    # torch.optim.AdamW(fused=True): one kernel per step instead of a foreach
    # chain over ~300 tensors.  CUDA only; ignored on CPU.
    opt_fused: bool = False
    # Throughput probe (tools/h100_sweep.py).  > 0: run this many *timed*
    # micro-steps after a warm-up, print one `[bench] {json}` line (img/s,
    # data-wait share, peak reserved VRAM, the largest eval batch multiple that
    # fits under `bench_vram_ceiling_gb`) and exit — no eval, no checkpoint.
    bench_steps: int = 0
    bench_vram_ceiling_gb: float = 72.0
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
    # v5 — coarse-label supervision.  DFC23 / India nDSMs are ~2 m stereo,
    # bilinearly upsampled 4x onto 0.5 m pixels (prepare_data.py `_dfc23_pairs`):
    # no detail below ~4 px.  v4 fitted them pixel for pixel, gradient / normal /
    # flatness terms included, and 40 % of its crops came from them — that is
    # the smooth-blob look on satellite imagery.  Samples whose `src` starts with
    # one of these are scored only after `coarse_pool` x `coarse_pool` average
    # pooling (L1 + SILog + gradient) and are excluded from the normal, flatness
    # and bin terms.  Empty string -> v4 behaviour.
    coarse_label_sources: str = ""          # v5 run: "dfc23,india_labeled"
    coarse_pool: int = 4
    # The label's real resolution in METRES.  > 0 sizes the pool per sample as
    # round(coarse_label_m / gsd_m): the GSD jitter resamples 90 % of crops, and
    # a DFC23 tile (512 px @ 0.5 m) lands at 0.30-0.50 m, where a fixed 4 px
    # block is 1.2-2.0 m — finer than a ~2 m label, so part of its upsampling
    # blur was still being taught.  0 keeps the fixed `coarse_pool` pixels.
    coarse_label_m: float = 0.0             # v5 run: 2.0 (tools/audit_labels.py)
    w_coarse: float = 1.0
    # When the audit shows the coarse labels put trees at ~0 m, mask vegetation
    # (ExG > coarse_veg_exg) with label < 1 m out of those samples, so "a tree
    # is flat ground" is not taught.  ExG = 2g - r - b on chromaticities.
    coarse_mask_veg: bool = False
    coarse_veg_exg: float = 0.10

    # ----- eval / io -------------------------------------------------
    eval_every: int = 2
    # Final run: what best.pt is selected on.  Empty -> the primary val set's
    # global RMSE (v1-v5).  Otherwise a comma list of val sources (the primary
    # and/or aux val sets) whose forested + sparse tiles are pooled into one
    # pixel-weighted RMSE — the canopy and rural maps are what this run is for,
    # and a global RMSE is dominated by urban ground.
    select_on: str = ""
    tta: bool = True
    # 0.5 m / 1.25 = 0.4 m effective, inside the trained 0.33-0.66 m jitter band.
    # Final-eval only; best.pt is still selected on plain no-TTA centre-crop.
    tta_scales: tuple = (1.0, 1.25)         # parse_config skips tuples -> edit here, not CLI
    final_sliding_eval: bool = True         # full-tile sliding-window eval at the end

    # ----- held-out test sets ----------------------------------------
    # Stores that are scored ONCE, at the very end, with best.pt already loaded
    # — never during training, never part of the best.pt decision.  That is the
    # whole point: `val_tiles` selects the first 400 GAMUS val tiles, best.pt is
    # selected on them (train.py), and `final_plain`/`final_tta`/
    # `final_sliding_tta` are then reported on those same tiles, so the headline
    # number is measured on the set the checkpoint was picked on.  A store named
    # here is measured on tiles the run never saw.
    #
    # "<store>:<split>", comma separated; a bare "<store>" means "<store>:test".
    # A split that the run also trains or validates on is rejected in validate()
    # rather than quietly reported as held out.
    test_sources: str = ""                  # e.g. "gamus:test"
    test_tiles: int = 0                     # 0 -> the whole store (no prefix)
    # Sliding+TTA is the honest protocol but costs ~6 s/tile; 400 keeps the test
    # sliding number the same size as the val one it sits next to.  0 -> all.
    # Skipped entirely when final_sliding_eval is false.
    test_sliding_tiles: int = 400
    per_class_metrics: bool = True
    per_landscape_metrics: bool = True      # urban / sparse / hilly / forested
    dump_class_stats: bool = True
    class_names: tuple = CLASS_NAMES
    output_dir: str = _DEFAULT_OUTPUT_DIR
    n_qualitative: int = 12
    # Per-dataset sample gallery in the report: source families (prefix-matched
    # against the stores under data_root, trained on or not), and tiles each.
    # Each family shows one store, from its val split if it has one, else test,
    # else train, and the report says which.  "" or 0 turns it off.
    gallery_sources: str = "synrs3d,dfc23,india_labeled,us3d,mvs3dm"
    gallery_tiles: int = 3
    # Per-landscape examples in the report: tiles per class (urban / sparse /
    # hilly / forested, the same GT rule as `per_landscape`) from one store,
    # found by scanning a seeded `landscape_gallery_scan` tiles of it.  With
    # `landscape_gallery_shadows` each tile also gets image shadows vs the
    # shadows cast by the reference and by the prediction.  0 turns it off.
    landscape_gallery_source: str = "gamus"
    landscape_gallery_tiles: int = 2
    landscape_gallery_scan: int = 300
    landscape_gallery_shadows: bool = True
    make_figures: bool = True              # viz/figures.py at the end of the run
    make_report: bool = True                # viz/report_html.py validation report
    export_onnx: bool = True                # infer/export_onnx.py at the end of the run
    onnx_opset: int = 17

    # ----- misc ------------------------------------------------------
    hf_token: str = ""
    smoke: bool = False
    resume: str = ""                        # checkpoint to resume/warm-start from
    make_zip: bool = True

    # ----- multi-session resume (Kaggle) -----------------------------
    # `best.pt`/`last.pt` are *deployment* checkpoints: EMA-merged weights plus
    # the preprocessing contract, and nothing else.  Restarting a killed session
    # from one of those restarts the LR cosine from the top with fresh AdamW
    # moments.  `last_full.pt` is the separate artefact that carries optimiser,
    # scaler, EMA, teacher, RNG and elapsed time, so a 12 h Kaggle session that
    # dies at hour 8 is actually recoverable.  ~6.4 GB, ~30 s to write.
    save_full_state: bool = False
    full_state_every: int = 1               # write last_full.pt every N epochs
    # `max_minutes` is the *total* training budget across all sessions and is
    # what drives the cosine; `session_minutes` caps *this* session only.
    # A two-session run is --max_minutes 960 --session_minutes 480 in both.
    session_minutes: float = 0.0            # 0 -> unlimited (single session)

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
        self.gallery_tiles = 1
        self.landscape_gallery_tiles = 1
        self.landscape_gallery_scan = 8
        self.final_sliding_eval = False
        self.test_tiles = 4
        self.ema_decay = 0.0
        self.consistency_rampup_epochs = 1
        self.export_onnx = False
        self.full_state_every = 1
        return self

    def validate(self) -> "Config":
        assert self.tile_size % 16 == 0, "tile_size must be a multiple of 16"
        assert self.amp_dtype in ("bf16", "fp16")
        assert self.gsd_jitter_lo_m <= self.gsd_jitter_hi_m
        assert 0 <= self.freeze_epochs <= self.epochs
        assert 0.0 <= self.unlabeled_batch_frac <= 2.0
        assert self.encoder_unfreeze_blocks >= 0
        # A floor above the initial scale would pin the scale there and defeat
        # the backoff the scaler exists to perform.
        assert 0.0 < self.amp_min_scale <= self.amp_init_scale
        assert all(v > 0 for v in self.landscape_boost_map().values()), \
            "landscape_sampler_boost values must be > 0"
        # A "held-out" split that the run trains or validates on is not held
        # out, and reporting it as a test number would be worse than reporting
        # nothing.  Caught here, at parse time, rather than three hours in.
        from dwdata.loaders import test_split_pairs, val_split_of
        for _name, _split in test_split_pairs(self):
            if _name not in self.labeled_sources():
                continue
            if _split == "train" or _split == val_split_of(_name):
                raise AssertionError(
                    f"--test_sources {_name}:{_split} is the same split this run "
                    f"{'trains' if _split == 'train' else 'validates'} on — it is "
                    f"not held out. Use a split no other cell touches "
                    f"(e.g. gamus:test).")
        # The cosine is driven by max(epoch fraction, elapsed/max_minutes)
        # (train.py), so `max_minutes` is the budget the schedule is SIZED for,
        # not a safety cap.  If this session stops before that budget is spent
        # and nothing was written to continue from, the anneal is simply lost:
        # the v4-2 run was launched --max_minutes 600 --session_minutes 480
        # --save_full_state false, died at the session cap with progress 0.81,
        # and ended at lr 5.09e-05 instead of the 8e-06 floor its predecessor
        # reached.  It scored 3.804 m against that predecessor's 3.441 m.
        # A warning, not an assert: a deliberate short probe of a long schedule
        # is a legitimate thing to run.
        if self.session_minutes > 0 and self.max_minutes > self.session_minutes \
                and not self.save_full_state and not self.resume:
            print(f"[config] !! max_minutes={self.max_minutes:.0f} exceeds "
                  f"session_minutes={self.session_minutes:.0f}, so this session "
                  f"stops at {100.0 * self.session_minutes / self.max_minutes:.0f} % "
                  f"of the LR cosine — with save_full_state=false and no "
                  f"--resume, the rest of the anneal is unrecoverable.  Pass "
                  f"--save_full_state true and resume in a second session, or "
                  f"set --max_minutes {self.session_minutes:.0f} to fit one.")
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

    def landscape_boost_map(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for part in str(self.landscape_sampler_boost or "").split(","):
            if ":" in part:
                k, v = part.split(":")
                k = k.strip()
                if k not in LANDSCAPE_NAMES:
                    raise ValueError(f"landscape_sampler_boost: unknown class {k!r} "
                                     f"(expected one of {LANDSCAPE_NAMES})")
                out[k] = float(v)
        return out

    def no_urban(self, src: str) -> bool:
        """True when `src` (or its base, e.g. `neon` for `neon/test`) has no buildings."""
        names = [s.strip() for s in str(self.landscape_no_urban_sources or "").split(",") if s.strip()]
        base = str(src or "").split("/", 1)[0]
        return any(base == n or base.startswith(n + "_") for n in names)

    def select_sources(self) -> list[str]:
        return [s.strip() for s in str(self.select_on or "").split(",") if s.strip()]

    def sampler_weight(self, name: str, default: float = 1.0) -> float:
        """Weight for one store, falling back to its base source name.

        Sources packed one store per GSD family arrive as `dfc23_g050` /
        `synrs3d_g05`, but the thing anyone actually wants to weight is the
        *source*.  So `dfc23:2` covers every `dfc23_*` family, while a fully
        qualified `synrs3d_g1:3` still wins over a bare `synrs3d:1` — exact
        match first, then the longest prefix.
        """
        w = self.sampler_weight_map()
        if name in w:
            return w[name]
        parts = name.split("_")
        for i in range(len(parts) - 1, 0, -1):
            base = "_".join(parts[:i])
            if base in w:
                return w[base]
        return default


def parse_config(argv: list[str] | None = None) -> Config:
    cfg = Config()
    p = argparse.ArgumentParser(description="DepthWizard v5")
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
