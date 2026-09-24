"""DepthWizard DAV2/V1 — configuration.

One dataclass, every knob CLI-overridable (same pattern as v1-v4).

This is v4's training core with **one variable changed**: the encoder is Depth
Anything V2 Base (a DINOv2 ViT-B/14 plus its own pretrained DPT neck) instead
of DINOv3 ViT-L/16 SAT-493M plus a randomly-initialised trunk.  Everything the
change forces — patch 14, tile 518, decoder_dim 128, no explicit feature
indices — is marked below.  Two settings are reverted to v3's values because
v4's measurements say v4 was wrong about them (`stratum_balance_beta`,
`stratum_weight_clip`), and `llrd` is re-derived for a 12-block encoder.

The data mix, the losses, the mean-teacher branch and the metrics are v4's,
unchanged, so the comparison is a comparison.

Defaults are H100-friendly.  The Kaggle 2xT4 profile lives entirely in
`run_kaggle.sh`'s launch flags — no Turing constants baked in here.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, fields
from pathlib import Path

_DAV2_DIR = Path(__file__).resolve().parent

_DEFAULT_OUTPUT_DIR = os.environ.get("DW_OUTPUT_DIR", str(_DAV2_DIR / "outputs" / "dav2-v1"))
_DEFAULT_DATA_ROOT = os.environ.get("DW_DATA_ROOT", str(_DAV2_DIR / "data"))

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
    # Comma list of prepared store names:
    #   gamus  geonrw  india_labeled  india_unlabeled
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
    val_tiles: int = 400                    # 0 -> score the whole val store.
                                            # Non-zero selects the FIRST n tiles in
                                            # sorted-stem order (a prefix, not a
                                            # random sample) — kept at 400 so the
                                            # number stays comparable to v1-v4.
    max_valid_height_m: float = 200.0       # AGL above this is LiDAR noise, not a building

    # ----- geometry / GSD -------------------------------------------
    tile_size: int = 518                    # model input, multiple of patch(14).
                                            # 518 = 37 x 14 is DAv2's own training
                                            # resolution, so no position-embedding
                                            # interpolation is needed.
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
    # Depth Anything V2 Base: a DINOv2 ViT-B/14 encoder plus the DPT decoder it
    # was trained with.  Ungated — unlike DINOv3-SAT, this one needs no HF_TOKEN
    # (GAMUS still does).  LICENCE: DAv2-Base is CC-BY-NC-4.0, i.e. NOT
    # commercially usable; only DAv2-Small is Apache-2.0.  If the submission
    # needs a commercial licence, the only change is this string.
    encoder_model_id: str = "depth-anything/Depth-Anything-V2-Base-hf"
    encoder_patch: int = 14                 # DINOv2 is patch-14
    # The taps are no longer ours to choose: DAv2's own `backbone_config.out_indices`
    # picks them, and its neck was trained against exactly those four.
    decoder_dim: int = 128                  # == DAv2-Base's fusion_hidden_size,
                                            # which makes encoder.proj a free 1x1
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
    # 0 = all 12 blocks, which at 86 M parameters is affordable where ViT-L's
    # 303 M was not — and a full unfreeze is the recipe v3 actually won with.
    encoder_unfreeze_blocks: int = 0
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
    # 0.85 over 12 blocks puts the bottom block at 0.85**12 = 14 % of
    # `encoder_lr` — a usable span end to end.  (0.80**12 = 6.9 % is the
    # dead-lower-half problem that cost v4 throughput for no movement.)
    llrd: float = 0.85                      # layer-wise LR decay per block
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
    # Reverted to v3's values.  v4 raised these to 0.7/8.0 to attack the tall
    # bias and the measured result was worse in both directions: balanced RMSE
    # stalled at 4.165 against v3's 3.584, and `flat_bias` never moved off
    # +1.08 m (v3: +0.396 m).  The backbone is the variable under test here, so
    # the loss weighting goes back to the configuration that has the better
    # number attached to it.
    stratum_balance_beta: float = 0.5       # 0 -> plain pixel-uniform loss
    stratum_weight_clip: float = 5.0

    # ----- eval / io -------------------------------------------------
    eval_every: int = 2
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
    make_figures: bool = True               # viz/figures.py at the end of the run
    make_report: bool = True                # viz/report_html.py validation report
    export_onnx: bool = True                # infer/export_onnx.py at the end of the run
    # 18, not 17: the dynamo exporter emits Resize at 18 and has no adapter
    # down to 17, so a request for 17 printed a traceback and fell back to 18
    # anyway (run-1 log).  Asking for what it produces keeps the log clean.
    onnx_opset: int = 18

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
    # In phase > 1 (below) `max_minutes` is THAT PHASE's budget, measured from
    # the phase's own start, not the all-sessions total.
    session_minutes: float = 0.0            # 0 -> unlimited (single session)

    # ----- numbered training phases ----------------------------------
    # Each phase is its own warmup + cosine over epochs [phase start, epochs].
    # Resuming with the SAME --phase continues that phase's cosine (a crashed
    # session); resuming with a HIGHER --phase starts a fresh warmup from the
    # floor at the next epoch.  Without this, extending a finished run by
    # raising --epochs jumps the LR from the annealed floor to wherever
    # max(epoch fraction, clock fraction) lands — ~40x in one step for run 1.
    phase: int = 1
    phase_lr_mult: float = 0.3              # peak LR of a phase > 1, relative to
                                            # the run's saved base_lrs (run 1:
                                            # 3e-4 decoder, 6e-5 encoder top)
    phase_warmup_frac: float = 0.05         # of the phase's own progress

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
        self.test_tiles = 4
        self.ema_decay = 0.0
        self.consistency_rampup_epochs = 1
        self.export_onnx = False
        self.full_state_every = 1
        return self

    def validate(self) -> "Config":
        assert self.encoder_patch > 0
        assert self.tile_size % self.encoder_patch == 0, (
            f"tile_size {self.tile_size} must be a multiple of encoder_patch "
            f"{self.encoder_patch}")
        assert self.amp_dtype in ("bf16", "fp16")
        assert self.gsd_jitter_lo_m <= self.gsd_jitter_hi_m
        assert 0 <= self.freeze_epochs <= self.epochs
        assert 0.0 <= self.unlabeled_batch_frac <= 2.0
        assert self.encoder_unfreeze_blocks >= 0
        assert self.phase >= 1, f"--phase must be >= 1, got {self.phase}"
        assert 0.0 < self.phase_lr_mult <= 1.0, (
            f"--phase_lr_mult must be in (0, 1], got {self.phase_lr_mult}")
        # A floor above the initial scale would pin the scale there and defeat
        # the backoff the scaler exists to perform.
        assert 0.0 < self.amp_min_scale <= self.amp_init_scale
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
