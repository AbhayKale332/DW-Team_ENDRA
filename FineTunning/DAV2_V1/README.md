# DepthWizard DAV2/V1 — Depth Anything V2 backbone (Kaggle 2× T4 / H100)

A **one-variable experiment**, not a new generation.  Everything here is
`FineTunning/V4_Kaggle`'s training and evaluation path, copied unchanged, with
the encoder swapped:

| | V4_Kaggle | DAV2/V1 |
|---|---|---|
| encoder | DINOv3 ViT-L/16 SAT-493M, 303 M | DINOv2 ViT-B/14 (DAv2 Base), 86 M |
| decoder | `models/dpt.py`, **randomly initialised** | DAv2's own **pretrained** DPT neck |
| pretraining | satellite imagery, self-supervised | monocular depth, task-supervised |
| patch / tile | 16 / 512 | 14 / **518** (37 × 14) |
| gated | yes (`HF_TOKEN`) | **no** (GAMUS still is) |
| licence | Apache-2.0 | **CC-BY-NC-4.0** — see §6 |

For everything this variant did *not* change — the data mix, the losses, Head
B's bin head, the mean-teacher branch, the landscape-stratified metrics, the
DDP traps, the 19 GB output quota, the two-session resume — read
`../V4_Kaggle/README.md`.  It is not duplicated here, on purpose: a second copy
would be a second thing to keep true.

---

## 1. The question this answers

Every DepthWizard run so far (v1→v4) used DINOv3 with a random DPT trunk.  The
v4 H100 run plateaued at GAMUS val RMSE ~3.85 and its `flat_bias` term never
moved (+1.08 m against v3's +0.396 m), while v3's own 2.715 came from a
different data mix, a different val subset and a full-backbone unfreeze.  So
there is **no clean read on whether the backbone is the limiting factor**.

Two things change at once here and both are deliberate:

1. **The decoder is no longer random.**  DAv2's reassemble + fusion stages
   arrive with depth-specific priors.  That is the whole reason to pick DAv2
   over plain DINOv2 — if the decoder were going to be random anyway, the
   encoder swap alone would be the cheaper test.
2. **The encoder is 3.5× smaller.**  On the Kaggle 2×T4 target that is the
   difference between `--grad_checkpoint_encoder true` at `--batch_size 2`
   (what `../V4_Kaggle/README.md` §5.2 was forced into) and running
   uncheckpointed at `--batch_size 6`, i.e. roughly 2–3× the optimiser steps
   per 12 h session.  Step count is what the v4 Kaggle run was actually
   starved of.

**The bet, stated plainly:** DINOv3-SAT-493M is pretrained on *satellite*
imagery and DAv2 is not.  This trades domain-matched pretraining for
task-matched pretraining.  It may lose.  It is cheap to find out, and the
answer is worth having either way.

**Not the goal:** chasing v3's 2.715.  DAV2/V1 is judged on
`test_gamus_test_*`, `dfc23_g050` and `india_labeled` — the deployment-relevant
domains.  `final_*` is never quoted; it is scored on the very tiles `best.pt`
was selected on.

**Rejected alternative — Marigold V2.**  A QLoRA fine-tune of
Qwen-Image-Edit-2509 (~20 B DiT + VAE) that its own authors trained for >5 days
on a single 32 GB GPU at batch size 1.  It does not fit a 16 GB Turing card, and
it outputs affine-invariant depth, which `.agents/Depth_Wizard_Plan.md:17` rules
out as this model's output.

---

## 2. The model change

`models/dav2.py` holds one class, `DAV2Backbone`, that stands in for v4's
`DINOv3Encoder` **and** `DPTTrunk` together — so `DepthWizardNet.forward` needed
no structural change and the four heads are untouched.  `models/dpt.py` and
`models/encoder.py` are not shipped.

It is built from HF `DepthAnythingForDepthEstimation`, keeping `.backbone` and
`.neck` and **deleting `.head`** (which predicts relative *inverse* depth
through a sigmoid; this model predicts metric nDSM in metres).

The shape chain, for a 518 px tile:

```
backbone(pixel_values).feature_maps      4 taps, (B, 1 + 37·37, 768)
  → neck(taps, patch_h=37, patch_w=37)
      reassemble ×(4, 2, 1, 0.5)         148 / 74 / 37 / 19
      fusion upsamples ×2, four times    →  296 × 296, C=128
  → fusion_stage[-1]                     (B, 128, 296, 296)
  → interpolate to (H/2, W/2)            (B, 128, 259, 259)
  → 1×1 Conv2d(128 → decoder_dim)        (B, decoder_dim, 259, 259)
```

Three things in that chain are load-bearing and each has a test:

* **518 = 37 × 14** is DAv2's own training resolution, so no position-embedding
  interpolation is needed.
* **The interpolate to H/2** restores the geometry v4's `DPTTrunk` had.  Without
  it Head B's bin logits would be `(B, 96, 296, 296)` — 1.3 GB at batch 8 in
  fp16, for no accuracy benefit.  `tests/test_model.py` asserts the half-res
  shape.
* **37 is odd.**  v4's own trunk had a bug where a stride-2 resample rounded the
  token grid up against a flat ×2 upsample; it never fired in a real v4 run
  because a 512 px patch-16 tile is a 32-grid.  Here the real runs are odd every
  step, so `tests/test_neck_odd_grid.py` checks grids 2…37 against the actual
  `DepthAnythingNeck`.

### What was ported from `models/encoder.py`, and why

* **`_find_blocks`** — locates the transformer blocks *structurally* (the
  `nn.ModuleList` whose length equals `num_hidden_layers`) rather than by
  attribute name.  v2 died at epoch 21/30 with `AttributeError: cannot locate
  transformer blocks on the encoder`, so the encoder was never unfrozen and the
  run ended before its final evaluation.  The search is scoped to the backbone
  so the neck's own ModuleLists cannot win the fallback.
* **The unreachable-parameter probe** — one tiny forward/backward, and anything
  that comes back with `grad is None` is frozen.  Under DDP with
  `find_unused_parameters=False` a trainable parameter that never receives a
  gradient is a deadlock, not a waste.  This architecture has two such places
  and both are structural: `backbone.embeddings.mask_token`, and
  `neck.fusion_stage.layers.0.residual_layer1.*` — the deepest fusion layer is
  called with no residual, so its first residual unit never runs.  That is
  exactly v4's `fuse[3].rcu1`, one level down and in somebody else's code.
  `tests/test_real_dav2.py` asserts the *property* (nothing trainable is left
  stranded), not a name list, because a name list is what the probe replaces.
* **Grad checkpointing is toggled once**, in `set_frozen`, never per step — that
  was a measured v4 defect.

### Three deliberate differences from the ported code

* **The neck freezes and unfreezes with the encoder.**  It is pretrained
  depth-specific weight, not a random trunk, so training it from step 0 against
  random heads is the one thing `--freeze_epochs` exists to prevent.
* **The neck gets its own LLRD-exempt group at `encoder_lr`.**  It should
  neither be trained at the decoder's 3e-4 nor be buried at the bottom of the
  layer-wise ladder with the patch embedding.
* **`encoder.proj` sits outside `encoder.model`.**  `head_state_dict()` strips
  the `encoder.model.` prefix, which is now the backbone *and* the neck — both
  public weights.  The 1×1 projection is randomly initialised, so it must be
  saved, and it must train during the frozen warm start.  `train.py`'s optimiser
  therefore matches `encoder.model.`, not `encoder.`; matching the bare prefix
  would leave `proj` with no optimiser group at all for the first two epochs.

---

## 3. Config deltas from V4_Kaggle

| Field | v4 | DAV2/V1 | Why |
|---|---|---|---|
| `encoder_model_id` | `facebook/dinov3-vitl16-pretrain-sat493m` | `depth-anything/Depth-Anything-V2-Base-hf` | the swap |
| `encoder_patch` | *(new)* | `14` | DINOv2 is patch-14 |
| `tile_size` | 512 | **518** | 37 × 14, DAv2's native resolution |
| `decoder_dim` | 256 | **128** | `fusion_hidden_size` for DAv2-Base; the 1×1 proj becomes free |
| `encoder_feature_indices` | (6,12,18,24) | *removed* | DAv2's `backbone_config.out_indices` already picks the taps, and its neck was trained against exactly those four |
| `stratum_balance_beta` | 0.7 | **0.5** | reverted to v3's: 0.7 stalled balanced RMSE at 4.165 vs v3's 3.584 |
| `stratum_weight_clip` | 8.0 | **5.0** | same |
| `encoder_unfreeze_blocks` | 16 | **0** (all 12) | 86 M fits; v3's full unfreeze is the recipe that worked |
| `llrd` | 0.90 | **0.85** | over 12 blocks, 0.85¹² ≈ 14 % of `encoder_lr` — usable end to end (0.80¹² = 6.9 % is the dead-lower-half problem again) |
| `grad_checkpoint_encoder` | False (H100) | False | the VRAM it buys is no longer scarce even on a T4 |
| `validate()` | `tile_size % 16` | `tile_size % encoder_patch` | |

Defaults stay H100-friendly.  **The T4 profile lives entirely in launch flags**,
exactly as `../V4_Kaggle/README.md` §5.2 established — no Turing constants baked
into the dataclass.

`dwdata/preprocess.py` carries the input contract and gained four fixes:
`patch` defaults to 14; `from_config()` now actually **forwards** it (v4 never
did, which was harmless while both ends were 16 and would silently tile a
patch-14 model on a patch-16 grid here — `infer/engine.py:68` reads
`spec.patch`); the pre-resolve fallback is ImageNet rather than the DINOv3-SAT
constants, which would shift every input by ~0.3 σ against this encoder; and
`version` is `"dav2-v1"`, so a v4 checkpoint cannot be loaded under this
contract without somebody noticing.

**No change was needed to the crop path.**  `augment.crop_and_scale` resamples a
`win × win` source window to `tile × tile` and `sample_window` clamps
`win = min(win, src_h, src_w)`, so a 512 px GAMUS tile fills a 518 px crop at an
effective GSD of 0.494 m — recorded in the sample's `gsd_m` field, not silently
lost.

---

## 4. Layout

Copied unchanged from `V4_Kaggle`: `dwdata/` · `eval/` · `infer/engine.py` ·
`infer/export_onnx.py` · `viz/figures.py` · `viz/report_html.py` ·
`models/losses.py` · `models/ema.py` · `models/tta.py` · `train.py` ·
`prepare_data.py` · `package_results.py` · `pytest.ini`.

**Not shipped:** `geo/` · `serve/` · `viewer/` · `viz/mesh.py` ·
`infer/predict.py` · `pack_gamus_png.py` · `models/dpt.py` ·
`models/encoder.py`.  Verified by import trace: nothing reachable from
`train.py` touches any of them.  `infer/predict.py`'s only part the training
path needed was `load_model`, which now lives in `infer/load.py` (the rest of
that module is the georeferenced CLI and imports `geo.calibrate`).

**New or modified:** `config.py` · `models/dav2.py` · `models/heads.py` ·
`dwdata/preprocess.py` · `infer/load.py` · `main.py` (no `--serve`) ·
`run_kaggle.sh` · `dav2_train.ipynb` · `tests/`.

---

## 5. Running it

```bash
bash run_kaggle.sh check      # GPUs, NCCL, /kaggle/{working,temp}, /dev/shm, tests
bash run_kaggle.sh prepare    # the ~14 GB pack, in a CPU notebook
bash run_kaggle.sh link       # symlink farm over /kaggle/input/*
bash run_kaggle.sh smoke      # the first real 2-process DDP run (~10 min)
bash run_kaggle.sh train      # the real run
```

Do not skip `smoke`; it is the first thing that opens a NCCL process group.

The 2×T4 profile, and what it is spending the smaller encoder on:

```bash
--amp_dtype fp16                  # Turing has no bf16; the GradScaler path exists
--grad_checkpoint_encoder false   # was forced true for ViT-L
--batch_size 6 --grad_accum 2     # 2 ranks × 6 × 2 = global 24, parity with v3/v4
--eval_batch_mult 4
--num_workers 2 --prefetch_factor 2
--compile_model false
--consistency_every 2 --w_consistency 1.0
--save_full_state true --make_zip false
```

Static VRAM per card falls from V4_Kaggle's measured ~8.3 GB to roughly
**~2.6 GB** (86 M × 4 copies for live / grads / AdamW ×2, plus the EMA shadow
and the mean-teacher deepcopy).

Read the `vram=NG` field the train log prints every 25 steps.  With headroom, go
to `--batch_size 8 --grad_accum 2` (global 32) rather than cutting `grad_accum`
to 1 — more optimiser steps is the point, but global-batch parity with v3/v4 is
what keeps the LR schedule honest.  If it is tight, back off in V4_Kaggle's
documented order: `--consistency_every 4`, then `--ema_decay 0`, then
`--w_consistency 0`.

### Verification, in order

1. `pytest -q` — no GPU, no network.  146 tests.
2. `DW_TEST_REAL_CKPT=1 pytest -q tests/test_real_dav2.py` — downloads the
   published checkpoint and confirms the 518 → 37 → 296 → 259 chain and the
   no-stranded-parameter property against the real weights, not against the
   arithmetic in a docstring.
3. `bash run_kaggle.sh check`, then `smoke`.
4. **The first 200 steps of the real run**, four things:
   * `train_loss` is finite.  fp16 + GradScaler on a freshly-attached neck is
     the most likely place for a NaN, and a trainer can report PASS with
     `train_loss = NaN`.
   * `vram=NG` leaves ≥ 3 GB headroom for the eval spike.
   * img/s is meaningfully above V4_Kaggle's 2.1 img/s unfrozen.  If it is not,
     the 86 M encoder is not buying what §5 claims and the batch size should be
     re-derived from a measurement.
   * the `[optim]` line's LR span covers all 12 blocks at usable values —
     roughly 8e-06..3e-04, not something ending in e-07.
5. **Verify from the filesystem.**  In a notebook, a green cell and a frozen log
   pane prove nothing; use `ls`, `du -sh`, `metrics.json`, not the output pane.

### Scoring

Compare `test_gamus_test_*`, `dfc23_g050` and `india_labeled` against the v4 run
in `../V4_modal/Output/`.  Never quote `final_*`.  Report per-stratum
`flat_bias` / `tall_bias` explicitly — that pair is the diagnostic that told us
v4's problem, and it is the first place a backbone change should show up.

---

## 6. Licence

**Depth Anything V2 Base is CC-BY-NC-4.0 — non-commercial.**  Only DAv2-Small is
Apache-2.0.  This matters for the SIH writeup: if the submission needs a
commercially usable model, the fallback is

```bash
--encoder_model_id depth-anything/Depth-Anything-V2-Small-hf
```

and nothing else has to change: the patch size is still 14, and
`DAV2Backbone`'s 1×1 `proj` reads the checkpoint's own `fusion_hidden_size`
(64 for Small) and maps it to `decoder_dim` whatever that is.  Setting
`--decoder_dim 64` to match makes the projection free again, as it is for Base;
`tests/test_real_dav2.py` asserts that equality for the Base default.

The DepthWizard code in this directory is under the repository's own licence
(see `../../LICENSE`).
