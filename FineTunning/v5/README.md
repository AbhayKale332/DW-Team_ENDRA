# DepthWizard v5 — the ISRO metric, v3-or-better detail, and a shadow mask

SIH 2026, PS 26175 (ISRO): single-view optical RGB → metric DSM → navigable 3D
flythrough.  **§0 is v5.**  Everything from §1 on is the v4 README this tree was
copied from (`FineTunning/V4_Kaggle/`), kept because the v4 machinery (DDP,
resume, the preprocessing contract, the FastAPI deliverable) is unchanged
underneath.  The plan is `CompetitionContext/V5_Research_Additions.md`, and the
open questions for the hosts are in `CompetitionContext/Host_Questions_Draft.md`.

## 0. v5

### 0.1 Why v5 exists

How the hosts score it (`CompetitionContext/FAQs.md`): **50 % accuracy** is
RMSE, MAE and r of the **absolute DSM against SRTM or Copernicus 30 m**, on
Cartosat-2S 0.6 m GeoTIFFs across urban, sparse, hilly and forested scenes.
**50 % visualisation**.  So the calibration of the absolute DSM and the
Cartosat input path decide the accuracy half.  The model mostly feeds the
visual half, and there v4 had a problem: it was better than v3 on GAMUS test
(TTA RMSE 3.321 vs 3.565) but rendered **blobs**.  The four reasons:

1. 40 % of its crops were DFC23 / India labels: ~2 m stereo heights,
   bilinearly upsampled onto 0.5 m pixels, fitted pixel by pixel.
2. `flatness_loss` treated their unlabelled pixels as eligible.
3. Wrong class ids.  `GROUND_LIKE_IDS = (0, 3, 4)` included the 9.4 m
   *building* class and missed ground and road.
4. No full-resolution path: the finest features were 16 px patches.

### 0.2 What changed (by plan step)

| Step | Change | Where |
|---|---|---|
| 0 | **Label audit.** Measures the labels' true resolution (→ `coarse_pool`), whether coarse labels put trees at 0 m (→ `coarse_mask_veg`, ExG τ calibrated on GAMUS trees), pins the GAMUS class names from per-id heights plus overlay PNGs, checks held-out overlaps by stem *and* image content, and calibrates the Cartosat augmentations on the real product | `tools/audit_labels.py` |
| 1 | **Class ids pinned** (1 ground, 2 low veg, 3 building, 4 water, 5 road, 6 tree); ground-like = (1, 2, 4, 5); SynRS3D's packed ids remapped at load | `config.py`, `dwdata/dataset.py` |
| 1 | **Coarse supervision**: DFC23 / India scored after 4×4 pooling (L1 + SILog + gradient) and excluded from the normal, flatness and bin terms | `models/losses.py` |
| 1 | **Cartosat augmentations**: pan-sharpen simulation (chroma at 1/2–1/3.3 resolution), grayscale | `dwdata/gpu_aug.py` |
| 2 | **Detail branch**: full-resolution RGB stem + RAFT-style convex 2× upsampling, zero-initialised, so v4 checkpoints still load and the branch starts out as v4 | `models/heads.py` |
| 3 | **Seeded val sample** (v4's val was a prefix, 87.5 % urban vs a 57.6 % urban test set); `edge_rmse_m`, `grad_ratio` | `dwdata/loaders.py`, `eval/metrics.py` |
| 3 | **One protocol for v3 / v4 / v5**: `eval_test.py` scores any checkpoint, v3's through v3's own code (`--model_code ../v3`), adds sharpness metrics to every version, qualitative strips and shadow IoU, and writes `outputs/v3_v4_v5.md` | `eval_test.py` |
| 5 | **The run profile**: v3's five reverts + `detail_branch`, coarse flags, Cartosat augs, `val_sample_seed 42`, `/results/v5`; `with_audit()` applies the Step 0 numbers | `../V4_modal/v5_flags.py` |
| A1 | **DEM-anchored DSM**: `DSM = U(DEM) + λ·[nDSM − U(A(nDSM))]` with Tobler iterations, so every 30 m cell averages to the DEM exactly and the model supplies only what is finer (v4's `DTM + nDSM` double-counted everything SRTM / COP30 already contain) | `geo/calibrate.py` |
| A2 | **Datum-correct DEMs**: every source carries its vertical datum (SRTM EGM96, COP30 EGM2008, CartoDEM ellipsoidal); `srtm30` renamed to what it is (a mixed AWS mosaic); native SRTM via OpenTopography (`OPENTOPO_KEY`); offline DEM cache; compound vertical CRS on output | `geo/dem.py` |
| A4 | **GCPs** in lon/lat + datum; offset / tilt on the terrain, scale on the nDSM only | `geo/calibrate.py` |
| A5 | **Shadow scale check**: the height scale that best explains the image's shadows, under the product's sun (label-free evidence of metric scale) | `viz/shadow.py` |
| A7 | **Judge proxy**: scene → windowed DSM → scored against SRTM and COP30, per pixel and per 30 m cell, with the datum sanity number | `eval/judge_proxy.py` |
| B | **Cartosat input**: NRSC product folders / zips (BAND3,2,1), PAN + MX pan-sharpened in memory, per-band stretch on valid pixels, NoData → NaN, row-band windowed inference (a 20k × 20k scene peaked at 5.9 GB RAM), `max_side` transform fix | `dwdata/scene_io.py`, `infer/engine.py` |
| B | **PAN + MX → registered RGB GeoTIFF**: phase-correlation shift measurement, then pan-sharpening (`gdal_pansharpen` if on PATH, else built-in Brovey) | `tools/cartosat_to_rgb.py` |
| C1 | Viewer shows what is scored: Surface / Structures / Terrain toggle, lon/lat probe | `viewer/` |
| C2 | **In-app validation**: upload any reference GeoTIFF; it is reprojected onto the prediction grid, datum-converted, and scored per pixel, on confident pixels, and per 30 m cell | `serve/validate.py`, `POST /api/reference/{job}` |
| C3 | **Shadow mask**: shadows in the image, cast by the heights (live with the sun sliders), and their agreement | `viz/shadow.py`, `viewer/` |
| C4 | **Vertical walls**: the mesh splits at height steps above a threshold (default 2.5 m) and fills them with shaded vertical quads, so roofs no longer smear down a slope | `viewer/app.js buildGeometry` |
| C5 | **Uncertainty**: Head B's per-pixel spread (`b_std`, computed and dropped in v4) is written as `ndsm_std_m.npy` / `.tif`, with a confidence drape and metrics on confident pixels | `infer/engine.py`, `infer/predict.py` |
| C6 | **Whole scene first, detail on demand**: shift-drag a box on the 2-D map to load that area at full resolution out of a windowed product | `serve/validate.py extract_aoi`, `POST /api/aoi/{job}` |

### 0.2.1 Architecture review fixes (after the first v5 cut)

| Issue | Fix |
|---|---|
| `coarse_pool` was fixed in pixels, but the GSD jitter puts DFC23 crops at 0.30–0.50 m, where 4 px is 1.2–2.0 m — finer than a ~2 m label, so part of its blur was still taught | `coarse_label_m` (metres): each sample is pooled over `round(coarse_label_m / gsd)` px, grouped by pool size; the audit reports it; `v5_flags` sets 2.0 |
| The ONNX graph (the standalone deployment) had no `b_std`, so the confidence layer and confident-pixel metrics vanished under `--onnx` | third output `height_std_m`; `_OnnxModel` passes it through; older 2-output graphs still load |
| Head B's bins stopped at 120 m while v5 trains on labels up to 150 m | `bin_max_m 150` in `v5_flags` |
| The unfreeze rebuilt the optimiser: decoder AdamW state lost, encoder at ~peak LR with no warm-up | encoder groups appended to the live optimiser (same layout a resume rebuilds), encoder LR ramped 0 → 1 over `unfreeze_warmup_epochs` (1.0) |

### 0.3 Runbook

```bash
cd FineTunning/v5

# Step 0 — audit (CPU).  Paste nothing by hand: the flags come from audit.json.
python tools/audit_labels.py --data_root /scratch/dwdata --out outputs/audit \
    --cartosat ../../cartosat_2S_Sample/Cartosat-2E/247677521

# the Bhubaneswar PAN + MX pair -> one registered 0.6 m RGB GeoTIFF
python tools/cartosat_to_rgb.py --pan ../../cartosat_2S_Sample/5132211 \
    --mx ../../cartosat_2S_Sample/5132611 --out bhubaneswar_rgb.tif

# train on Modal: the v5 tree, the v5 profile, the audit's Step 0 numbers
cd ../V4_modal
python v5_flags.py ../v5/outputs/audit/audit.json          # print + parse-check
DW_CODE=v5 DW_AUDIT=../v5/outputs/audit/audit.json modal run modal_app.py::smoke
DW_CODE=v5 DW_AUDIT=../v5/outputs/audit/audit.json modal run --detach modal_app.py::train

# one protocol for all three versions, then the table
cd ../v5
python eval_test.py --ckpt best_3.7.pt  --data_root D --tta_scales 1.0,1.25 \
    --max_valid_height_m 150 --qualitative 12 --out outputs/v4_test
python eval_test.py --model_code ../v3 --ckpt best_2.715.pt --data_root D \
    --tta_scales 1.0,1.25 --max_valid_height_m 150 --qualitative 12 --out outputs/v3_test
python eval_test.py --ckpt /results/v5/best.pt --data_root D --tta_scales 1.0,1.25 \
    --max_valid_height_m 150 --qualitative 12 --out outputs/v5_test
python eval_test.py --compare outputs/v3_test,outputs/v4_test,outputs/v5_test \
    --labels v3,v4,v5 --md outputs/v3_v4_v5.md

# the accuracy half, the way the judges score it
python -m eval.judge_proxy ../../cartosat_2S_Sample/Cartosat-2E/247677521 \
    --landscape sparse --ckpt outputs/v5/best.pt --out outputs/judge_proxy

# the deliverable: upload -> DSM -> viewer (reference upload, AOIs, shadows)
python -m serve.app --ckpt outputs/v5/best.pt --port 8000
python -m geo.dem --prime scene.tif      # fill the DEM cache for offline demos
```

`--qualitative 12` samples the same 12 tiles in every run (`--qual_seed 42`),
so `--compare` can stitch strips RGB | GT | v3 | v4 | v5 | shadow agreement.
For named tiles use `--qual_stems a,b,c` instead.

### 0.4 What is verified, and what is not

Verified in this tree (`pytest -q`: stub encoder, synthetic fixtures, no GPU,
no network):
* every item in §0.2 has tests, including: a MERGED folder / zip stacks
  BAND3,2,1; windowed = whole-array within 1e-3 m across band seams; the
  anchored DSM's 30 m cell means equal the DEM; GCPs leave the nDSM scale
  unchanged; a box casts a shadow h / tan(el) long; the phase correlation
  recovers a known PAN/MX shift and its sign; the audit recovers a 4× coarse
  label, flat "trees" and a retitled duplicate tile; the reference validation
  reprojects a lon/lat DSM and recovers a 2 m offset; an AOI cut from a
  windowed product matches the full-resolution GeoTIFF pixel for pixel;
  `eval_test.py` loads v4- and v5-shaped checkpoints and refuses one with
  missing head weights; the Modal image carries the v5 profile into the
  container.
* The viewer was driven in headless Chromium (Playwright, SwiftShader) against
  the real service with a stand-in model: surface toggle, walls on/off,
  confidence and shadow layers, reference upload (lon/lat GeoTIFF → reprojected
  → scored), AOI select → load → back, probe with lon/lat, and no console
  errors.

**Not** done yet, and it matters:
* **No real checkpoint has been scored through the v5 code**, and no training
  run has happened.  The cloud session that wrote this had no access to
  huggingface.co, so the published v3 / v4 / DAv2 weights and the DINOv3
  encoder could not be downloaded.  The first thing to run on a GPU box is the
  three `eval_test.py` lines above with the existing weights (`best_2.715` =
  v3, `best_3.7` = v4).
* The audit and `cartosat_to_rgb.py` have run on synthetic data only.  The
  `coarse_pool` / `coarse_mask_veg` / pan-sharpen numbers in `v5_flags.py` are
  provisional until `audit.json` exists.
* The judge proxy has run on a synthetic scene against live Copernicus data,
  not on the Cartosat samples.
* The mesh is a decimated regular grid with walls.  The plan's RTIN adaptive
  mesh (Martini) is not implemented.  Thin towers narrower than the
  decimation step can still drop out at the "Fast" mesh setting.
* The geoid test skips without network (PROJ fetches the EGM grids from
  cdn.proj.org).  Run `projsync` once on an offline demo machine.
* `λ` (`--detail-gain`) stays 1.0 until the hosts answer Q1 of the draft.

---

# DepthWizard v4 (Kaggle 2× T4) — Phase 2 + Phase 3

SIH 2026, PS 26175 (ISRO): single-view optical RGB → metric DSM → navigable 3D
flythrough.

> **This directory is the Kaggle variant of `FineTunning/v4/`.** It is a full
> copy with three additions — DistributedDataParallel over two GPUs, full-state
> multi-session resume (`last_full.pt`), and a 16 GB-Turing flag profile — and
> `run_lightning.sh` replaced by `run_kaggle.sh` + `kaggle_train.ipynb`. The
> model, losses, preprocessing contract, tests and FastAPI deliverable are
> unchanged, and `FineTunning/v4/` itself is untouched and remains the H100
> runbook. **Jump to §5 for the Kaggle runbook.**

v3 was a training pipeline. **v4 is the deliverable**: the same training core
plus the geo half (ground-masked DTM fit, DEM fetch, GCP refinement, ONNX
export), the product half (FastAPI service, standalone Three.js viewer,
auto-generated validation report), and three model changes each driven by a
measurement rather than a hunch.

```
FineTunning/V4_Kaggle/
  config.py prepare_data.py train.py main.py
  run_kaggle.sh  kaggle_train.ipynb        <- the Kaggle launch surface (§5)
  dwdata/    preprocess (the contract) · augment · packed · dataset · loaders · india
  models/    encoder · dpt · heads · losses · ema · tta
  geo/       dem (COP30 / SRTM / CartoDEM) · calibrate (DTM fit, GCP RANSAC)
  eval/      metrics · landscape · sliding · report
  infer/     engine (the one inference path) · predict (CLI) · export_onnx
  viz/       figures · report_html · mesh (glTF / OBJ)
  serve/     FastAPI: upload → DSM → viewer
  viewer/    standalone dual-pane Three.js app (vendored, no build step)
  tests/     113 offline tests, no GPU and no network
```

---

## 1. What changed from v3, and why

Everything below is a fix for something that was **measured**, not a redesign for
its own sake. v3's own post-mortem of v2 is still worth reading — it is
`../v3/README.md` §1 and none of it is superseded.

### 1.1 `--dem` was double-counting every building

v3's absolute-DSM path added the coarse DEM to the predicted nDSM directly.
Copernicus GLO-30 and SRTM are **surface** models: their cells already contain
buildings and canopy. So a 40 m tower on 12 m terrain came out near 80 m — and
it came out looking like a perfectly plausible elevation raster, which is the
dangerous part.

v4 routes it through `geo/calibrate.py`, which does what the plan actually
specified: fit the bare-earth DTM through **ground pixels only**, then add.

```
ground   = (Head C says ground/road/water)  AND  (predicted nDSM < 1.5 m)
DTM      = robust poly-2 trend + NaN-aware smoothed residual, fitted on ground
DSM      = DTM + nDSM
```

Both mask conditions are required, so a wrong semantic class id — and the ids
*are* unverified, see §6 — cannot poison the fit. Measured on a synthetic scene
with a 30 m tower over known terrain:

| | DSM at the tower |
|---|---|
| truth | 135.0 m |
| v4 calibrated | **134.4 m** |
| v3 naive `dem + ndsm` | 165.0 m |

`tests/test_calibrate.py` and `tests/test_geotiff_e2e.py` pin this, the second on
a real GeoTIFF with a real 30 m DEM reprojected onto the prediction grid.

### 1.2 Head B still had a trivial target

v2's bin head collapsed (CE 0.06–0.13 for the whole run) and the gate just copied
Head A. v3 diagnosed it and added stratum weighting, but kept the **hard**
nearest-bin target — which over 96 bins with ~65 % of the mass on the ground bin
is still nearly free to predict. Two changes:

* `bin_soft_sigma` (default 1.5 bins) replaces the one-hot target with a Gaussian
  over neighbouring bins, so the head has to get the *shape* of the distribution
  right, not just the argmax.
* `w_bin_entropy` puts a floor on the **marginal** bin distribution's entropy, so
  dumping every pixel into one bin is penalised directly.

`tests/test_consistency.py` asserts the property that matters: a collapsed head
now scores worse than a spread one, and the soft optimum is a distribution rather
than a spike.

### 1.3 Nothing answered the rubric's stability axis

"Performance stability across urban, sparse, hilly, and forested landscapes" is
50 % of the accuracy score, and no v1–v3 number addressed it. The nearest thing
was a per-*class* table whose ids are provably wrong.

`eval/landscape.py` classifies every validation tile from its **own ground-truth
height field** — relief, tall-pixel fraction, canopy roughness — so it needs no
annotation and works on any dataset:

```
relief    >= 6.0 m   -> hilly       (terrain-scale undulation dominates)
frac_tall <  0.12    -> sparse      (barely any structure)
roughness >= 0.22    -> forested    (rough relative to its own height)
otherwise            -> urban
```

Every result now carries `per_landscape`, `landscape_rmse_spread_m`,
`landscape_worst`, and the raw descriptors so the assignment can be audited
rather than trusted. These are heuristics and the report says so out loud.

### 1.4 Head B now also produces an uncertainty

`b_std` — the standard deviation of the bin distribution, in metres. It costs one
extra reduction (the head already computes the distribution) and it gates the
pseudo-labels in §2. It also gives the bin head a second job, which is one more
reason for it not to collapse.

### 1.5 Smaller things

* `run_lightning.sh` began with `set -euo pipefail` and the docs said to run it
  with `sh`. `/bin/sh` is dash on Lightning's images and dash has no `pipefail`,
  so the script aborted on line 1 under its own documented invocation. Fixed, and
  it now picks `python` vs `python3` rather than assuming.
* `serve/app.py` deliberately has **no** `from __future__ import annotations`:
  FastAPI resolves endpoint signatures at definition time, and postponed
  annotations turn `file: UploadFile` into an unresolvable string. The symptom is
  a 500 at request time, not an import error — it would have shipped.
* `pin_memory=True` on a CPU box warns on every DataLoader; now conditional.
* `DepthWizardNetV3` → `DepthWizardNet`, with the old name kept as an alias so
  v3 checkpoints still load.

---

### 1.6 The v3 run's own logs (`Logs/v3/`) — and why most of v4's perf work is undo

This is the largest single change in v4 and it is almost entirely *restoration*.
The v4 draft was branched from v3 **before** v3's performance work landed, so it
silently reverted every fix v3 had made to the data pipeline and the training
step. Side by side against `Logs/v3/run.log`, the v4 draft would have been
roughly 2.5x slower per image than the run that produced the good checkpoint.
What came back:

| what the v4 draft did | what v3 does | cost of the draft |
|---|---|---|
| `PackedStore.get()` — materialise all four full-tile planes per crop | `get_window()` — slice the memmap first | ~9 MB read per crop, ~90 % discarded |
| full-tile histogram per crop for the 2/98 stretch | bounds computed once per tile and persisted next to the shards (`prime_stretch_bounds`) | 13.4 ms of an 82 ms `__getitem__`, in every worker, forever |
| `np.percentile` on uint8 | 256-bin histogram | ~15x on a 1 Mpx tile |
| float stretch image | 256-entry LUT gather | no float image at all |
| PIL resize (scalar C) | OpenCV `INTER_AREA`/`INTER_LINEAR` | ~40 % of what was left of the crop budget |
| CPU photometric jitter, float32 CHW on the wire | `dwdata/gpu_aug.py`, uint8 HWC on the wire | 3.1 MB → 0.79 MB per sample, ~22 ms per crop |
| `rgb_u8` shipped to the GPU every step | kept host-side (`_HOST_ONLY`) | ~25 MB/step of pure H2D |
| `gradient_checkpointing_enable()` **inside `forward()`** | toggled once in `set_frozen` | walks every ViT-L submodule, every step |
| `.float()` on every encoder tap | taps stay bf16 under autocast | ~540 MB allocated and re-read per step |
| `transpose(1,2).reshape(...)` on the taps | a real view + permute into channels-last | a full copy of every tap |
| ten `float(tensor)` calls between forward and backward | detached 0-d tensors, read only on logged steps | ten device syncs per step — the host can never run ahead |
| `ModelEMA` per-key Python loop | `torch._foreach_*` | ~800 kernel launches per step |
| rebuilding the trainable-parameter list each optimiser step | hoisted per epoch | walks 324 M params per step |
| a second full persistent worker pool for val | val capped at 4 workers | v3's log shows the 24/28/35-worker warning and then a shared-memory `Bus error` |

v4 goes a little further than v3 in three places, for the same reason:

* **`_masked_mean` no longer uses boolean indexing.** `x[m].mean()` has a
  data-dependent output shape, so it syncs the device before the kernel can
  launch; `torch.where(m, x, 0).sum() / m.sum()` is the same number, one fused
  kernel, no sync, and NaN-safe in the masked-out region (`x * m` is not — `inf *
  0` is `NaN`). Same treatment for `silog_loss`, `StratumBalancer` (weighted
  `bincount` instead of `idx[valid]`) and `flatness_loss`.
* **`MeanTeacher` got `ModelEMA`'s fused update.** It is a second 324 M-param
  EMA running inside the training step; the per-key loop it shipped with would
  have cost more than the branch it serves.
* **The landscape breakdown transfers once per batch, not twice per tile.** At
  400 val tiles and an eval every other epoch that was ~24 000 syncs a run.

#### The three failures in `run.log`

1. **Two OOMs, both at micro-batch 32.** The L40S attempt died at epoch 4
   (44 GB); the H100 attempt reached epoch 7 and died on a 768 MB allocation
   with 78.4 GB in use and 2.36 GB reserved-but-unallocated — VRAM had drifted
   from 70 GB at epoch 3 to 76 GB at epoch 7. Both tracebacks asked for
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` by name; v4 sets it in
   `train.py` and exports it from the launch script.
2. **A shared-memory `Bus error`** when an autotuner walked the worker count up
   to 35 on a 16-vCPU box. v4 sets the worker budget statically (16 train + 4
   val) and, because the workers now hand over uint8, each sample on the wire is
   a quarter of the size.
3. **`STARVED` in `gpu_disk_guard.log`** on the runs that read shards straight
   off network-backed storage. The run that finished read from `/tmp`;
   v4's `run_lightning.sh stage` made that copy a first-class step; on Kaggle
   the equivalent is keeping the pack small enough to live in page cache (§5.4),
   because there is nowhere local to stage 32 GB to.

Once those were out of the way the guard log is unambiguous: **median GPU
utilisation 99 %, median VRAM 47 GB of 80 GB.** The loader was not the limit;
the card was compute-bound with a third of its memory idle.

#### Why the batch is 24 and not 32

From v3's own log, encoder unfrozen, no checkpointing:

| micro-batch | VRAM | throughput |
|---|---|---|
| 16 | 42 GB | 44.7 img/s |
| 32 | 74 GB (→76 GB, then OOM) | 48.0 img/s |

Doubling the batch bought **7 %** and cost 32 GB, because the card was already
at 99-100 % utilisation either way. So the spare 33 GB does not belong in the
batch. v4 uses 24 (~58 GB, ~20 GB of headroom for the eval spike and allocator
drift) with `grad_accum` dropped to **1**, which buys the kernel efficiency
without the OOM and gives 500 optimiser steps an epoch instead of v3's 375.
What the VRAM headroom actually buys is `grad_checkpoint_encoder false` — the v4
draft defaulted it to `true` with the comment *"required to fit ViT-L unfrozen
@512"*, which v3's log disproves directly: 42 GB at batch 16, no checkpointing,
on an 80 GB card. Recomputing every block to save memory that is sitting idle
cost ~35 % of the step for nothing.

#### Two accuracy findings the metrics hand you

* **v3 never converged.** It finished 26 epochs in 121 minutes of a 330-minute
  budget, and val RMSE was still falling monotonically at the last eval
  (2.760 → 2.732 → 2.723 → 2.715). ~3 hours of paid-for compute went unused.
  v4 runs 40 epochs. The LR schedule is driven by `max(epoch fraction, wall-clock
  fraction)`, so a slower card still completes the cosine instead of being
  chopped off with the LR high.
* **The tail bias was frozen, not converging.** `tall_bias` sat at −2.39, −2.13,
  −2.14, −2.12, −2.06, −2.13, −2.08, −2.06, −2.08 m from epoch 6 to epoch 26
  while global RMSE kept improving — twenty epochs that never once touched the
  tail. Final per stratum: 20 m+ bias **−2.46 m**, 10–20 m −1.35 m, 0–2 m
  **+0.43 m**. That is textbook regression to the mean, and it says
  `stratum_balance_beta=0.5` equilibrated at a biased optimum rather than
  travelling towards an unbiased one. v4 raises it to **0.7** with the clip at
  **8.0**. Replayed over v3's own measured val height distribution (61 % 0-2 m,
  10 % 2-5 m, 16 % 5-10 m, 9 % 10-20 m, 4 % 20 m+) that moves the per-pixel loss
  weights like this:

  | | 0-2 m | 2-5 m | 5-10 m | 10-20 m | 20 m+ | tall : ground |
  |---|---|---|---|---|---|---|
  | v3 `0.5 / 5.0` | 0.641 | 1.574 | 1.249 | 1.669 | 2.504 | 3.90x |
  | v4 `0.7 / 8.0` | 0.504 | 1.771 | 1.282 | 1.923 | 3.393 | 6.73x |

  `balanced_rmse_m` and the per-stratum `bias_m` column are the check on whether
  that went too far — if the tall bias crosses into positive territory while
  `balanced_rmse_m` rises, back it off with `--stratum_balance_beta 0.5`.

Kept unchanged, because v3's numbers say they worked: the DINOv3-SAT ViT-L
encoder and its (6, 12, 18, 24) taps, the DPT decoder at half resolution, the
three-head + gated-fusion design, the adaptive bin widths, every loss weight
except the stratum knobs, the 2-epoch freeze warmup, LLRD 0.8, lr 3e-4 / 6e-5,
the cosine-with-warmup schedule, EMA 0.9995, the GSD-jitter and photometric
augmentation ranges, the 2/98 stretch contract, and dihedral TTA (which took the
final RMSE from 2.715 to **2.605** and is the reason `--tta` stays on).

---

## 2. Indian data — what exists, what does not, and what v4 does about it

**Every labelled source available is foreign.** GAMUS is five US cities, GeoNRW
is North Rhine-Westphalia, SynRS3D is synthetic, DFC2019 is Jacksonville/Omaha.
ISRO will evaluate on Cartosat-class imagery over Indian cities, whose morphology
(dense low-rise, narrow irregular streets, flat roofs, different roof albedo) and
radiometry are both outside that distribution. v2 already showed what this costs:
on Inria *Austin* — a US city at essentially GAMUS's own GSD — it put ~4 m of
height on flat ground and the render was a crumpled mountain range.

We went looking for an open Indian RGB + per-pixel-height dataset. There is not
one:

| candidate | Indian height labels? | usable automatically? |
|---|---|---|
| GAMUS / DFC2019 / GeoNRW / SynRS3D | no | — |
| **DFC2023 Track 2** (includes a **New Delhi** city) | yes — 2 m nDSM from Gaofen-7 / WorldView stereo | **no API**: IEEE DataPort login. Supported here via `--india_dir` once downloaded by hand. |
| Google Open Buildings 2.5D, UT-GLOBUS | building heights only, ≥4 m, no paired RGB | weak; needs its own basemap pairing and licence call |
| Bhoonidhi / Bhuvan (CartoDEM, Cartosat) | DEM at 10–30 m | too coarse for building height — but exactly right for the *terrain* term, see `geo/dem.py` |
| SAC's own reference repo (`IMG-PROCESS-SAC/SIH2026`) | README only, no data | — |

So v4 supports the two ingest paths that actually work, and is honest that
neither is a labelled Indian benchmark.

**(a) Labelled, from a local directory** — `dwdata/india.py:pack_labeled`.
Pairs files by stem (`delhi_001_rgb.tif` + `delhi_001_ndsm.tif` [+ `_seg`]), cuts
tiles, and writes a normal packed store. Point it at DFC2023's New Delhi tiles, at
any stereo/LiDAR product the team obtains, or at CartoDEM-derived pairs
(`--india_absolute_dsm` subtracts a morphological ground estimate). The
train/val split is by **scene**, not by tile — tiles from one raster are
near-duplicates and splitting them at random would leak.

```bash
python prepare_data.py --datasets india_labeled --india_dir ~/indian_pairs
python train.py --datasets gamus,synrs3d_g05,india_labeled \
                --sampler_weights gamus:1,synrs3d:1,india_labeled:3
```

For DFC23 Track 2 specifically use `--datasets dfc23 --dfc23_dir …` instead
(§5.11): its rgb/ and dsm/ rasters share filenames and differ only by parent
directory, which the stem-suffix pairing above cannot see.

Adding a source touches five places and there is no plugin registry:
`config.py`'s `datasets` docstring and `sampler_weights` default;
`prepare_data.py`'s dispatch chain and argparse; `dwdata/loaders.py`'s
`_VAL_SPLIT` / `val_split_of` (a missing entry does not mean "no val set" — it
means a slice of the first *train* store gets scored as one); and
`run_kaggle.sh`'s `link` whitelist (a name absent there is silently skipped).
Sources packed one store per GSD family arrive suffixed (`synrs3d_g05`,
`dfc23_g050`); both the val-split and the sampler-weight lookups fall back from
the store name to the base source, so `dfc23:2` covers every `dfc23_*` family.

**(b) Unlabelled, via mean-teacher domain adaptation** — the lever that works
without labels, and the one the plan schedules for the finals. An EMA teacher
predicts a weakly-augmented view of an Indian tile; the student is trained
towards it on a strongly-augmented one; pixels where the teacher's own `b_std`
exceeds `consistency_conf_m` are dropped, so the branch cannot chase the
teacher's hallucinations. The encoder's features migrate towards Indian
radiometry and street morphology with no height labels anywhere.

```bash
python prepare_data.py --datasets india_unlabeled --india_dir ~/bhuvan_exports
python train.py --datasets gamus,synrs3d,india_unlabeled
```

`dwdata/india.py:INDIA_AOIS` carries 14 city boxes chosen for **morphological
spread** — dense urban (Mumbai, Kolkata, Delhi, Bengaluru, Hyderabad), planned
low-density (Chandigarh, Gandhinagar, Jaipur), hills (Shimla, Dehradun, Gangtok)
and forest (Coorg, Nilgiris, Sundarbans) — so the unlabeled set spans the four
landscapes the rubric grades. `fetch_xyz_tiles` will download over them from an
XYZ template, but **there is no default URL**: which basemap a government
deliverable may train on is the team's licence decision, not a script's. Bhuvan
is the obvious one for an ISRO deliverable.

**State this plainly in the deck.** Unlabeled adaptation reduces the domain gap;
it is not evidence of Indian accuracy. Numbers on ISRO imagery have to be
measured on ISRO imagery. The auto-generated report says exactly this in its
"what this does not establish" section, because a panel that catches an overclaim
discounts everything else you said.

The branch is **off** unless the store exists, and it costs ~50 % of a step, so
it runs every `consistency_every` (default 2) steps.

---

## 3. The elevation module

Architecture is v3's, with the additions above:

```
RGB scene (any size, any GSD)
  └─ scene 2/98-percentile stretch          ← identical in train and inference
  └─ resample to 0.5 m/px canonical GSD
  └─ 512² windows  [train: one crop, chosen in SOURCE px then resampled]
       └─ DINOv3-SAT ViT-L/16   [frozen 2 epochs, then FULLY trainable w/ LLRD 0.8]
            └─ DPT trunk → features at H/2
                 ├─ Head A  metric nDSM       softplus, metres
                 ├─ Head B  96 adaptive bins  soft targets + entropy floor → also b_std
                 ├─ Head C  semantics         aux; the ground mask geo/calibrate needs
                 └─ gated fusion α·A + (1−α)·B  → nDSM
  └─ [unlabeled] EMA teacher on a weak view → student on a strong one, gated by b_std
  └─ [inference] 8× dihedral TTA → Hann-blended stitch → back to the input grid
  └─ [georeferenced] + DEM → ground-masked DTM fit → absolute DSM GeoTIFF
```

**The inference contract** is unchanged and still the point: training and
inference build their tensors with the same code (`dwdata/preprocess.py`), and
the resolved recipe is serialised into every checkpoint, so `infer/predict.py`
reads it from the weights rather than from a config that may have moved on.

```bash
# georeferenced GeoTIFF → nDSM on the source grid; GSD from the transform
python -m infer.predict scene.tif --ckpt outputs/v4/best.pt --tta

# absolute DSM, terrain fetched from the public Copernicus GLO-30 bucket
python -m infer.predict scene.tif --ckpt outputs/v4/best.pt --absolute

# ...or from CartoDEM / SRTM / any GDAL-readable DEM you hold
python -m infer.predict scene.tif --ckpt outputs/v4/best.pt --dem cartodem.tif

# non-georeferenced PNG/JPG → rDSM.  --gsd makes it metric.
python -m infer.predict scene.png --ckpt outputs/v4/best.pt --gsd 0.5

# scale tied to surveyed control points (row,col,elevation_m CSV), RANSAC-fitted
python -m infer.predict scene.tif --ckpt outputs/v4/best.pt --gcps gcps.csv

# print the resolved contract and exit
python -m infer.predict scene.tif --ckpt outputs/v4/best.pt --report
```

Outputs: `ndsm_m.tif` (float32, source CRS + transform, pixel-identical grid),
`dsm_m.tif` and `dtm_m.tif` with `--absolute`/`--dem`, `ndsm_m.npy`,
`ndsm16.png` (16-bit, decode affine in `meta.json`), `seg.png`, `rgb.png`,
`terrain.glb` + `terrain.obj`/`.mtl`/`texture.png` for the viewer, and
`meta.json` carrying the contract, the scene metadata and distribution
diagnostics.

---

## 4. The visualization module

### 4.1 The viewer (`viewer/index.html`)

Open the file. That is the whole install — three.js r128 is vendored beside it,
there is no build step, no bundler and no network call, so it behaves identically
from a USB stick and from the server. That is the "standalone deployment" half of
the rubric.

Two synchronised panes:

* **Map pane** — height colormap / hillshade / optical / reference / signed-error
  layers, optional contours, and the 3D camera's footprint drawn on it so the two
  panes are visibly the same place.
* **Flythrough pane** — the DSM as a textured mesh with the **original optical
  image draped**. Orbit, **first-person** (pointer-lock WASD, Shift sprint,
  Space/C for altitude), and a cinematic drone orbit for the demo video.
  Sun azimuth/elevation with real shadows, vertical exaggeration, wireframe, and
  a mesh-detail budget (150k / 500k / 1.5M vertices).

Analysis, all in metres:

* **click-to-probe** — height at a pixel, plus the reference value and the error
  when a reference is loaded
* **two-point slope** — ground distance, Δh, slope in % and degrees
* **elevation profile** — predicted vs reference along the measured line
* **live validation** — RMSE / MAE / bias / Pearson r computed in the browser
  against `gt_ndsm_m.npy`
* **an automatic warning** when implausibly little of the scene is below 1 m,
  which is the signature of the model reading texture as terrain

**Projection accuracy is graded, so the mapping is stated rather than implied**:
vertex (row, col) sits at world `(x = col·gsd, y = height_m, z = row·gsd)` with UV
`(col/(W−1), 1 − row/(H−1))` — pixel-centre to vertex, no half-pixel drift. Same
convention in `viz/mesh.py` and in the viewer, and `tests/test_mesh.py` asserts it.

Heights are read from `ndsm_m.npy` as float32, not from the PNG: a 16-bit PNG
drawn into a 2-D canvas is down-converted to 8 bits by the browser, which would
quantise a 0–60 m range into 0.23 m steps before anything renders. The PNG path
still exists as a fallback and warns when it is used.

```bash
# try it with no model at all — a synthetic scene with exact known block heights
python viewer/make_sample.py
python -m http.server -d viewer 8000    # then open ?result=./sample
```

### 4.2 The service (`serve/app.py`)

```bash
bash run_kaggle.sh serve                         # torch
python -m serve.app --onnx outputs/v4/depthwizard.onnx    # CPU, no HF token
```

Drag an image onto `/`, watch the progress bar, and land in the flythrough. It
calls the same `infer.engine.predict_scene` the evaluation calls — a number on
the report is a number the demo reproduces. Jobs are plain directories on disk in
exactly the layout the viewer already reads: no database, no queue, nothing to
lose when a process dies mid-demo.

`GET /api/health` · `POST /api/predict` · `GET /api/job/{id}` ·
`GET /api/result/{id}/{file}` · `GET /api/report/{id}` · `GET /view/{id}`

### 4.3 Figures and the validation report

Produced by the training run itself, so a finished run is a deliverable rather
than a checkpoint someone still has to write a notebook around. Re-render any
time with `bash run_kaggle.sh report`.

`viz/figures.py`: training curves (with the unfreeze marked), per-stratum RMSE
and bias, per-landscape RMSE, predicted-vs-reference hexbin, signed-error
histogram split flat/tall, hillshade pred vs reference, and the qualitative
contact sheet.

`viz/report_html.py`: one self-contained HTML file — every figure inlined as a
data URI, no external references at all, so it emails, prints to PDF and opens on
an air-gapped machine. It leads with balanced RMSE and the two bias numbers
rather than the headline, and it ends with a section on what it does **not**
establish.

---

## 5. Running it on Kaggle (2× T4)

This directory is the Kaggle variant of `FineTunning/v4/`. Everything else in it
is identical — same model, same losses, same preprocessing contract, same tests,
same FastAPI deliverable — and `FineTunning/v4/` is unchanged and still the
H100/Lightning runbook. The differences are all here: DDP over two GPUs,
full-state multi-session resume, and a data/VRAM profile that fits a 16 GB
Turing card. The dataset layout is byte-identical to v4's; only the paths move.

Set the accelerator to **GPU T4 x2**, turn internet on, and add `HF_TOKEN` under
Add-ons → Secrets (DINOv3-SAT and GAMUS are both gated). Then, in order:

```bash
bash run_kaggle.sh check     # GPUs, NCCL, /kaggle/working + /kaggle/temp + /dev/shm, tests
bash run_kaggle.sh prepare   # in a CPU notebook, once  -> ~14 GB, then Save Version
bash run_kaggle.sh link      # symlink farm over the attached dataset(s)
bash run_kaggle.sh smoke     # the first real 2-process DDP run, ~10 min
bash run_kaggle.sh train     # the real run
```

`kaggle_train.ipynb` ships these as four cells. **Start the real run with Save
Version → Save & Run All (Commit)**: the headless commit is what gets the full
~12 h and persists `/kaggle/working` as the notebook's output. An interactive
session idles out long before the run finishes.

Do not skip `check`, and do not skip `smoke` — `smoke` is the first thing that
actually opens a NCCL process group, and it costs ten minutes against a run
that costs sixteen hours.

### 5.1 Why the H100 profile does not transfer

Kaggle breaks four of the assumptions `config.py`'s defaults are justified
against, and they are four different problems:

1. **16 GB per card, Turing.** No bf16 tensor cores, and the run holds *three*
   fp32 copies of a 321 M-param network (live, the `ModelEMA` shadow, the
   `MeanTeacher` deepcopy).
2. **Two GPUs**, which is only useful with real DDP — and this codebase has
   four specific traps for it (§5.3).
3. **~12 h per session**, against a full 40 × 12 000-crop schedule that is ~17 h
   even on both cards.
4. **~19 GB of persistent `/kaggle/working`**, and `/kaggle/input` is a
   read-only network mount that random 512 px crops will starve on.

### 5.2 The flag profile, and where each number came from

Static VRAM on one card, before a single activation:

| | GB |
|---|---|
| params (321 M, fp32) | 1.28 |
| grads (= the DDP buckets, with `gradient_as_bucket_view`) | 1.28 |
| AdamW `exp_avg` + `exp_avg_sq` | 2.57 |
| `ModelEMA` fp32 shadow | 1.28 |
| `MeanTeacher` deepcopy (only when `india_unlabeled` is prepared) | 1.28 |
| autocast fp16 weight casts | ~0.6 |
| **total** | **~8.3** of ~15 usable |

~6.5 GB left for activations is why `grad_checkpoint_encoder` has to be **on**
here — the exact opposite of the H100 default, and exactly the case its comment
in `config.py` reserves it for ("Turn it on only if you move to a genuinely
smaller card").

None of these are new dataclass fields; `config.py`'s auto-generated argparse
already exposes every scalar, so the whole profile is launch flags:

```bash
--amp_dtype fp16                # Turing has no bf16; the GradScaler path is already there
--grad_checkpoint_encoder true
--batch_size 2 --grad_accum 6   # 2 ranks x 2 x 6 = global effective batch 24,
                                #   exact parity with the H100 run
--eval_batch_mult 8             # eval is no_grad; batch 2 would be absurd
--num_workers 2                 # T4 x2 gives ~4 vCPU; 2/rank + the capped val pool
--prefetch_factor 2
--compile_model false
--max_minutes 480 --session_minutes 480
--save_full_state true --make_zip false
```

Under DDP **`batch_size` is a per-process micro-batch**, so the global effective
batch is `world_size × batch_size × grad_accum`. `crops_per_epoch` stays a
*global* count — each rank draws `crops_per_epoch // world_size`, so `len(dl_tr)`
halves and the epoch-fraction progress math needs no change.

Start at `--batch_size 2 --grad_accum 6`, read the `vram=NG` field the train log
already prints every 25 steps, and move to `--batch_size 4 --grad_accum 3` if
there is headroom. If it is still tight, in order of preference:
`--consistency_every 4`, then `--ema_decay 0` (costs the 2–5 % RMSE the EMA is
worth), then `--w_consistency 0` (frees the teacher's whole 1.28 GB).

Turing specifics: `tf32` and `sdp_flash` are harmless no-ops (flash SDPA needs
sm_80+; the mem-efficient kernel is picked instead, and the enables are already
wrapped in `contextlib.suppress`). Keep `channels_last` and `cudnn_benchmark`
on — they still help the DPT trunk's convolutions.

### 5.3 The four DDP traps, and what was done about them

A generic "wrap it in `DistributedDataParallel`" port breaks here four ways,
and only one of them is loud:

**(a) The encoder unfreeze invalidates the wrapper.** `train.py` flips
`core.encoder.set_frozen(False)` at `epoch == freeze_epochs + 1`. DDP registers
gradient buckets for the parameters that have `requires_grad=True` *at
construction time*, so a wrapper built at epoch 1 with the encoder frozen has no
buckets for the 300 M encoder parameters — from epoch 3 the two ranks silently
train different encoders. `find_unused_parameters=True` does **not** fix this: it
handles parameters that go unused, not parameters that appear later. The wrapper
is therefore *rebuilt* inside the same unfreeze block, and the run logs the
bucket parameter count either side of it (`~21M -> ~321M`) precisely because this
is the failure that otherwise looks like a slightly worse run rather than an
error.

That count is read off **DDP's reducer**, not off `requires_grad`. The flags are
already flipped by the time the rebuild block asks, so counting trainable
parameters on the module would report the post-unfreeze number on both sides and
confirm nothing; the reducer's bucket buffers are the registration, fixed at
construction. The run warns if the number fails to grow.

Zero-code fallback if the rebuild ever misbehaves: `--freeze_epochs 0` — which
works only because the encoder is unfrozen **before** `_wrap` builds the wrapper.
Unfreezing it afterwards would reintroduce exactly this bug inside the escape
hatch, and with `freeze_epochs == 0` the rebuild block never fires to correct it.
The construction-time reduction set is logged unconditionally
(`[ddp] reduction set at construction: 321.xM params`) so that path has a trace
of its own.

**(b) There are two backwards per step.** The mean-teacher consistency branch is
a second forward/backward — through a DDP wrapper that is a second full 1.28 GB
all-reduce per micro-step, over PCIe, on cards with no NVLink. The consistency
forward therefore runs through the **unwrapped** `core`, and it runs **before**
the labeled forward/backward rather than after it. The ordering is not cosmetic:
DDP's reducer is armed by the wrapper's forward and disarmed by the autograd
callback at the end of that backward, so a consistency backward placed *after*
the labeled one would drop a rank-local gradient into a bucket that had already
been all-reduced — and `opt.step()` fires immediately afterwards, so every rank
would apply a different update and the two models would drift apart. Going
first, the gradient is already sitting in `.grad` (which, with
`gradient_as_bucket_view=True`, *is* the bucket) when the labeled backward
triggers the reduction, and is averaged along with everything else. It is still
a separate backward, and the unlabeled tensors are dropped before the labeled
forward, so the step still peaks at `max(labeled, unlabeled)` rather than their
sum.

**(c) `grad_accum` multiplies communication.** DDP all-reduces on every backward
but the optimiser steps every `grad_accum` micro-steps, and the T4 profile needs
a high `grad_accum`. Non-boundary micro-steps run under `model.no_sync()`: one
all-reduce per optimiser step instead of `grad_accum` of them.

**(d) A parameter that never receives a gradient hangs the reducer.** With
`find_unused_parameters=False` the reducer buckets every `requires_grad`
parameter at construction and then blocks until each one reports a gradient. One
tensor that the forward cannot reach is enough: the reduction never completes,
and the failure surfaces one step later, in the *next* forward, as
`RuntimeError: Expected to have finished reduction in the prior iteration`. It
has bitten twice, at two different levels:

* `DPTTrunk.fuse[3]` is the deepest fusion block and runs with no skip
  connection, so its `rcu1` never participated in the forward — six tensors,
  reported as indices 60–65. Fixed structurally: `FeatureFusionBlock(use_skip=False)`
  does not build an `rcu1` at all (`models/dpt.py`).
* The **encoder unfreeze** registers ~415 more tensors, three of which are off
  the path to the taps: `embeddings.mask_token` (masked-image modelling; it is
  read only when `bool_masked_pos` is passed, which it never is) and the
  backbone's trailing `norm.weight` / `norm.bias`, which `DINOv3ViTModel`
  applies to `last_hidden_state` while the taps read the *pre-norm*
  `hidden_states`. Reported as indices `1 413 414`, one step after the unfreeze.

The second one is handled by `DINOv3Encoder._freeze_unreachable`, called from
`set_frozen(False)` — so it covers the unfreeze block, `--freeze_epochs 0` and
the resume path alike, and it runs before `build_optimizer` and `_wrap`, which
both filter on `requires_grad`. It finds the dead set by **probing**: one tiny
forward/backward through the encoder on a deterministic ramp (no `torch.randn`,
so a resumed run's RNG state is untouched), and anything that comes back with
`grad is None` is unreachable by definition. That survives a backbone swap or an
HF refactor in a way a hard-coded name list would not; the name list exists only
as the fallback for when the probe cannot run. Both ranks probe the same
structure and freeze the same set, so the reduction sets stay identical. The
result is logged either way — `[model] N unreachable encoder parameter(s) left
frozen` or `[model] encoder probe: every trainable parameter reaches the taps`.

Switching `find_unused_parameters=True` on instead would also clear the error,
at the cost of an autograd-graph traversal on every iteration — for parameters
that are dead in every iteration and can simply be turned off once.

Two things that are deliberately *not* problems: there is no BatchNorm anywhere
(GroupNorm in the DPT trunk and heads, LayerNorm in the encoder), so there are no
running buffers to sync and `broadcast_buffers=False` is safe; and a read-only
data root already degrades gracefully, because the `stretch_bounds_*.npy` write
is caught and the table kept in memory.

There is deliberately **no gradient-compression hook**. `fp16_compress_hook`
looks attractive on PCIe cards, but this profile runs `amp_dtype=fp16`, so the
`GradScaler` is live and gradients reach the reducer scaled by up to 65536 —
right at fp16's max of 65504. Compressing them would overflow.
`bf16_compress_hook` needs sm_80+.

Everything inference-only — per-epoch eval, the final eval, sliding eval, the
qualitative and viewer exports — goes through `core`, gated to rank 0, because
calling the wrapper a different number of times per rank is a NCCL mismatch.
Ranks 1..N-1 wait at an end-of-epoch barrier, then DDP is **torn down before**
the 20–60 min post-training stage rather than after it, and they exit cleanly.

### 5.4 Data: a ~14 GB pack, prepared *on* Kaggle

Packed size is exactly `7 · T²` bytes per tile, and GAMUS dominates because it is
written at `tile_px=1024` → 7.34 MB/tile. The v4 default 4000+400 pack is ~32 GB.

```bash
--gamus_train 1200 --gamus_val 160 --synrs3d_archives 1
   ≈ 8.8 GB  +  1.2 GB  +  ~4 GB   ≈  14 GB
```

Two reasons for ~14 GB and not 32: it fits Kaggle's ~29 GB RAM as page cache
after the first epoch, so the `/kaggle/input` network mount stops being the
bottleneck; and it fits the 19 GB output quota *during* preparation. Fewer unique
tiles costs less than it looks — the sampler oversamples with `replacement=True`
anyway.

Prepare it on Kaggle rather than uploading 14 GB:

1. A **CPU notebook**, internet on, `HF_TOKEN` as a Secret:
   `bash run_kaggle.sh prepare`. GAMUS deletes each raw file right after packing,
   so the transient peak is only the shard-trim copy (~1.9 GB). The same command
   pre-computes the 2/98 stretch bounds, so no training worker ever pays a
   full-tile histogram.
2. **Save Version** → the committed output becomes a dataset.
3. Attach it to the training notebook and run `bash run_kaggle.sh link`.

`link` builds a symlink farm at `/kaggle/temp/dwdata/{gamus,synrs3d}` over
whatever is under `/kaggle/input`, so `cfg.data_root` stays a single root even
when the two sources were packed in separate notebook runs. The same mechanism
attaches a previous run's `last_full.pt`.

Check rather than assume: `check` prints free space on `/kaggle/working`,
`/kaggle/temp` and `/dev/shm`. `/dev/shm` in particular — v3 already hit a
shared-memory `Bus error` from too many workers, and a small `/dev/shm` also
pushes NCCL off its SHM transport.

### 5.5 The 19 GB output quota

With `--make_zip false` the run's own footprint is `best.pt` 1.28 + `last.pt`
1.28 + `last_full.pt` 6.4 + `depthwizard.onnx` 1.3 + logs/figures/report <0.1 ≈
**10.3 GB**.

`--make_zip false` is not optional: `build_zip` is `shutil.make_archive` with no
exclusions and no source cleanup, so it duplicates all ~4 GB of `.pt`/`.onnx`
(which barely deflate) and takes the total to ~18 GB. Kaggle already versions the
whole working directory as the notebook's output, so the zip buys nothing here.

### 5.6 Two sessions, and a real resume

`--resume` in `FineTunning/v4/` is a **warm start, not a resume**: it loads
`ck["model"]` with `strict=False` and nothing else, and the checkpoints persist
no optimiser, scaler, EMA counter or epoch — so a session that dies at hour 8
restarts the LR cosine from the top with fresh AdamW moments. They also carry
*EMA-merged* weights, which is right for deployment and wrong for resuming.

This variant adds a separate artefact rather than changing what `best.pt` and
`last.pt` mean. With `--save_full_state true`, rank 0 writes `last_full.pt` next
to `last.pt`, atomically (`.tmp` then `Path.replace`, so a killed session cannot
leave a truncated 6.4 GB file), carrying the **live** weights plus optimiser,
scaler, EMA shadow and counter, teacher, epoch, elapsed minutes, `best`,
`history`, `encoder_frozen` and the RNG states. On restore the encoder is
unfrozen *before* the optimiser is rebuilt, so the param-group structure matches
`load_state_dict`, and `t0` is back-dated by `elapsed_min` — which is what makes
the wall-clock half of the progress fraction continue down the cosine instead of
re-warming from the floor.

Cost: ~6.4 GB and ~30 s against a ~20-min epoch, under 3 %. `--full_state_every 2`
halves it.

Two time knobs, because one cannot do both jobs:

- **`--max_minutes`** — the *total* training budget across all sessions. It
  drives the cosine and the global stop. Unchanged meaning for a single session.
- **`--session_minutes`** (new, default `0` = unlimited) — *this* session's cap.
  When it trips, `last_full.pt` is written and training stops.

So a two-session run is `--max_minutes 960 --session_minutes 480` in **both**
sessions, with session 2 adding
`--resume /kaggle/input/<run1>/outputs/v4/last_full.pt`. Under DDP the stop
verdict is all-reduced with `MAX`, so the ranks cannot break on different
iterations and desync the collective count.

If a session dies *after* training but *before* the exports,
`bash run_kaggle.sh finalize` runs the final eval, sliding eval, figures, report
and ONNX against the committed `best.pt` on a single GPU in a short notebook. It
launches with `--epochs 0`, so no epoch runs and `history` starts empty — the
trainer reads the committed `metrics.json` back in that case rather than
overwriting it with an empty history, which is what the figures' training curves
are drawn from. Point `--output_dir` at a *copy* of the previous output if you
want the original version left untouched regardless.

### 5.7 What to watch

| | |
|---|---|
| prepare (1200 GAMUS + 1 SynRS3D archive) | 30–60 min, network-bound, one time, CPU notebook |
| epochs 1–2, encoder frozen | ~22 min/epoch (measured: 4.5 img/s) |
| epochs 3–16, top 16 blocks trainable, checkpointed | ~32 min/epoch (measured: 2.1 img/s with all 24 unfrozen) |
| eval every epoch (rank 0 only) | ~1.5 min each |
| final eval + TTA + sliding eval | 30–60 min |
| figures + report + ONNX | 5–10 min |
| **total** | **~16–17 h**, i.e. two sessions at `--session_minutes 480` |

In the first ten minutes of the real run, read the line the trainer already
prints every 25 steps:

- `img/s` should be roughly **double** the single-GPU figure. If it is low *and*
  `vram=` is low too, the loader is starving — that is the §5.4 page-cache
  problem, not a GPU problem.
- `vram=` should sit near **12–13 G**. Climbing epoch over epoch is allocator
  drift, the thing that killed v3's batch-32 attempt.
- At startup, `[ddp] reduction set at construction: 21.x M params (encoder
  FROZEN)` — or `321.x M params (encoder trainable)` under `--freeze_epochs 0`.
- At epoch `freeze_epochs + 1`, `[ddp] wrapper rebuilt at unfreeze: bucket params
  21.x M -> 321.x M` must appear. This is trap (a), and it is the one failure
  that is otherwise completely silent. The number is read off DDP's reducer, and
  a `[ddp] WARNING` is printed if it fails to grow.
- `run.log` must contain **one** set of loss lines, not two interleaved ones —
  that is the proof rank 1's stdout redirect worked.
- **The skipped-step fraction at the end of each epoch.** See §5.9 — this is the
  line that would have caught the 2025-09-15 run four hours earlier.

Then deliberately kill a session around epoch 4 and resume from `last_full.pt`:
`run.log` should show the LR continuing down the cosine rather than re-warming.

### 5.9 The 2025-09-15 run: half the budget trained nothing

The first full Kaggle run finished cleanly, reported `sliding+TTA RMSE 3.959 m`,
and produced every artefact except the ONNX graph. It also wasted about four of
its eight training hours, and its own log says so if you read the right line:

```
e1:   7/500 optimiser step(s) skipped on non-finite gradients
e2:   2/500
e6: 281/500        <- fp16 gradients start overflowing
e7: 500/500        <- 100 % of the epoch
e8: 500/500   e9: 500/500   e10: 500/500   e11: 500/500   e12: 330/330
```

`torch.amp.GradScaler` halves the loss scale on every skipped step, has a growth
ceiling and **no floor**. Starting from `2**16`, backoff number 166 takes the
scale below the smallest fp32 subnormal, so it becomes *exactly* `0.0` — and
`unscale_` then divides every gradient by zero, finds `inf`, and skips. Forever.
There is no path back, because growth needs `growth_interval` consecutive
*successful* steps.

Epochs 7–12 ran ~240 minutes of forward and backward passes and applied **zero**
optimiser updates. The tell was in plain sight and nobody was looking for it:

```
eval e6   RMSE=3.947 MAE=2.319 r=0.899 d1=0.482 bal=4.299
eval e8   RMSE=3.947 MAE=2.319 r=0.899 d1=0.482 bal=4.299
eval e10  RMSE=3.947 MAE=2.319 r=0.899 d1=0.482 bal=4.299
eval e12  RMSE=3.947 MAE=2.319 r=0.899 d1=0.482 bal=4.299
```

Byte-identical, because the weights genuinely never moved. `best.pt` is the
epoch-6 checkpoint. `3.947 m` is a 6-epoch result wearing a 12-epoch run's
wall-clock.

What overflowed, and why it took until epoch 6: `compute_losses` is called
outside the autocast block, but the head outputs arrive as fp16 and fp16 inputs
keep the arithmetic in fp16. `normal_loss` divides height differences by the
0.33 m GSD, so a tall building edge becomes ~600 and the squared norm after it
~3.6e5 — against an fp16 ceiling of 65504. The log prints `nrm=nan` from epoch 6
onward. `silog_loss` was the second path: its `sqrt(var + 1e-7)` has a gradient
of ~1600 as the variance approaches zero, which is what a tile of flat ground
produces, and the additive epsilon never stops that flow.

Five changes, in four files:

| Where | Change |
|---|---|
| `models/losses.py` | every loss term computes in **fp32**, unconditionally — removes the overflow at source |
| `models/losses.py` | `silog_loss` floors the variance with `clamp_min` (zero gradient below the floor) instead of nudging it with `+ 1e-7` |
| `train.py` | the loss scale has a floor (`--amp_min_scale`, default 1.0), so a run of skips is recoverable rather than terminal |
| `train.py` | skips are reported as a **fraction**, a dead epoch is called out explicitly, and two consecutive dead epochs abort to the export stage instead of burning the budget |
| `config.py` | `--amp_init_scale 2**13` (not `2**16`, which skipped the first four boundaries of epoch 1 just backing down) and `--amp_growth_interval 500` (not 2000, which was longer than an entire epoch on this profile) |

Two more things the same log exposed, neither of them the crash:

- **`flat=0.000` on every step of every epoch.** The GAMUS Kaggle mirror ships
  no semantic raster, so `cls` arrives entirely as `SEG_IGNORE_INDEX` — and
  `flatness_loss` intersected its GT-flatness mask with the flat-*class* set,
  which is empty. The planarity penalty §3 calls the direct counter to "texture
  becomes terrain", and which matters most on the out-of-domain Indian imagery
  this model is for, was switched off for the whole run while still carrying
  `w_flat = 0.2`. Unlabelled pixels now count as eligible, so an all-unlabelled
  dataset behaves exactly like no `cls` at all.
- **No `depthwizard.onnx`.** The exporter traced a dynamic batch axis on a
  **batch-1** example; `torch.export` treats 1 as a special size and specialises
  rather than keeping it symbolic, so both the `dynamic_shapes` attempt and the
  `dynamic_axes` fallback died with `ConstraintViolationError: ... resulted in a
  specialized value of 1`. It now traces at batch 2 and verifies at batch 1,
  which also proves the axis came out dynamic.

And two accuracy changes that follow from what the run measured rather than from
what it broke:

- `--encoder_unfreeze_blocks 16` with `--llrd 0.90`. The old profile unfroze all
  24 blocks at `--llrd 0.80`, which puts the bottom block at `0.8**24` = 0.5 % of
  `--encoder_lr` — the log prints the range as `lr 2.83e-07..3.00e-04`. The lower
  half of a 303 M-parameter ViT-L was paying full price in gradients, all-reduce
  traffic and AdamW state to move essentially not at all, and throughput fell
  from 4.5 to 2.1 img/s for it. `--encoder_unfreeze_blocks 0` restores the old
  behaviour.
- `--epochs 16 --eval_every 1`. `progress = max(epoch fraction, wall-clock
  fraction)` and 40 epochs was never reachable, so the epoch half of that max was
  dead all run and the cosine was driven purely by the clock. And at
  `eval_every 2`, `best.pt` had six selection points across 12 epochs — too
  coarse when `d1` went 0.493 → 0.533 → 0.482 across e2/e4/e6 while RMSE fell
  monotonically.

### 5.8 Running it on an H100 instead

Use `FineTunning/v4/` and its `run_lightning.sh`; that tree is untouched and its
defaults (batch 24, bf16, no checkpointing, 16 workers, `--max_minutes 280`) are
justified against measurements from the v3 H100 run. Nothing here is a
replacement for it.

---

### 5.10 The `v4-2` run, and what came out of it

`outputs/v4-2/` scored **3.804 m** sliding+TTA against `outputs/v4/`'s **3.441 m**
on the *same* 400-tile GAMUS prefix. The `hilly` count going 60 → 0 between them
is the intended `eval/landscape.py` relief fix, not a different val set:
`sparse`/`forested` are 21/20 in both, and 299 urban + 60 hilly = v4-2's 359
urban. It is a real regression, and the post-mortem produced most of what
follows.

**The LR cosine was cut off at 81 %.** `train.py` drives the schedule off
`max(epoch fraction, elapsed / max_minutes)`, so `--max_minutes` is the budget
the anneal is *sized for*, not a safety cap. v4-2 went out as `--max_minutes 600
--epochs 16` with `--session_minutes 480` and `--save_full_state false`: the
session cap fired (`run.log`: `session cap 480 min hit`) with the LR still at
5.09e-05, where v4 had annealed to 8.32e-06 — and with no `last_full.pt` the
remaining three epochs could not be recovered. `config.validate()` now warns on
that exact combination and `train.py` prints a `[sched]` line at startup saying
where the cosine will actually stop. The committed profile is a two-session
`--epochs 24 --max_minutes 960 --session_minutes 480 --save_full_state true`,
identical in both sessions, with only `--resume` added to the second.

**`--w_seg 0` had gone stale.** It was reasoned from the GAMUS PNG mirror
shipping no class rasters, which was true while GAMUS was the only source.
SynRS3D ships them (`[data]` prints `seg=yes`), so v4-2 computed a real
segmentation CE on every mixed batch and multiplied it by zero; the `seg=2.13`
in the step lines is the raw unweighted value, never in the gradient. The weight
is back at `config.py`'s 0.2, which also restores `flatness_loss`'s
ground/road/water restriction.

**SynRS3D's GSD table was wrong, though not in a way that bit this run.**
`_SYN_GSD` read `{"g005": 0.3, "g05": 0.7}`. The dataset card
(<https://huggingface.co/datasets/JTRNEO/SynRS3D>) publishes *ranges*:
`g005` 0.05–0.30 m, `g05` 0.30–0.60 m, `g1` 0.60–1.00 m. So 0.3 was the ceiling
of g005 rather than anything representative and 0.7 was outside g05 altogether.
The hardcoded 0.5 m the packer actually used is inside g05's real range, so the
one archive v4-2 trained on was labelled about right by accident — but pooling
archives under one nominal is a 3–10× scale lie the moment a second family is
added, and `dataset.py` turns `store.gsd_m` straight into the crop's effective
GSD. Families are now packed into **separate stores** (`synrs3d_g005`,
`synrs3d_g05`, `synrs3d_g1`) at the midpoint of their published range.

That also fixed a claim this file used to make. `SYNRS3D_ARCHIVES` was ordered
"coarse-GSD family first (they give the model the >0.66 m scale range that
GAMUS's 1024 px tiles physically cannot reach)" — but it started with g05, which
tops out at 0.60 m, and contained no `g1` archive at all. The g1 family is 5,537
images and is the only thing in the mix that reaches past GAMUS's ceiling; it is
now the first three entries, and `run_kaggle.sh prepare` takes
`--synrs3d_archives 4` (all of g1, plus the g05 archive v4-2 used).

### 5.11 DFC23 Track 2

The first source in the mix that is actually the deployment domain: real 0.5 m
(SuperView-1) and 0.8 m (Gaofen-2) satellite optical with a real nDSM reference.
GAMUS is US *aerial*, SynRS3D is synthetic, GeoNRW's nDSM is a DEM proxy.

```bash
python prepare_data.py --data_root $DATA --datasets dfc23 \
       --dfc23_dir /kaggle/input/<ds>/track2/train
python train.py --datasets gamus,synrs3d_g05,synrs3d_g1,dfc23_g050
```

Three things about it are deliberate.

**SAR is ignored.** Track 2 ships a co-registered SAR band, but the packed store
is 3-channel (`dwdata/packed.py`) and DINOv3's patch embedding is `Conv2d(3, …)`,
so fusing it means a store-format change *and* encoder surgery — for a signal a
Cartosat-class single-view optical deployment does not have at inference.

**The val split is carved by scene, not by tile**, for the same reason
`prepare_india_labeled`'s is: tiles cut from one scene are near-duplicates, so a
tile-level split leaks the val set into training. `track2/val` and
`track2_test_data` ship no reference nDSM (the contest withheld it), so the only
usable labels are under `track2/train` and the holdout has to come from there.

**It is reported, never selected on.** `build_loaders` takes the first source in
`--datasets` with a val split as the primary, so with `gamus` first the primary
stays the 400-tile GAMUS prefix every run since v1 has been scored on, and that
is what `best.pt` tracks. Every other prepared val store is scored alongside it
(`build_aux_val_loaders`) and written into `metrics.json` as `val_<source>`.
That second number is the point: "v4 beat v4-2 on GAMUS val" is partly just v4
having trained on nothing but the val domain, and with one in-domain number
there is no way to tell that apart from real generalisation.

Measured on the 605-scene `track2_test_data` download (the test split ships
rgb + sar and no reference, so this is the imagery, not the labels):

| property | value |
|---|---|
| scene size | **512 × 512, all 605** — not large scenes |
| RGB | `uint8`, 3-band |
| SAR | `float32`, 1-band, ~0.03–12 (unused) |
| pixel size in the transform | 0.5 m, uniform |
| CRS | **`None`, all 605** |
| black/nodata | none worth the name — max 2.56 % zero-luminance, no scene above 5 % |
| per-scene dynamic range | p2 spans 0–90 DN, p98 spans 107–249 DN |

Three consequences worth knowing before you set the flags.

**The GSD comes from `--dfc23_gsd`, not from the file.** `raster_gsd_m` returns
its default when `ds.crs is None` (it cannot know the transform's units without
one), so every scene lands in one `dfc23_g050` store. The 0.5 m default is right
— the transform agrees and it matches SuperView-1 — but the per-GSD grouping is
inert on this data. It stays in because the contest also sourced 0.8 m Gaofen-2,
and a future drop that carries a CRS would otherwise pool them silently.

**There is no scale-augmentation headroom.** A 512 px store at a 512 px model
input caps the achievable GSD at `512 × 0.5 / 512 = 0.5 m`, so DFC23's jitter
range collapses to 0.30–0.50 m and the `[!] requested hi 1.20 unreachable`
warning fires for it too. Do not "fix" this by packing a smaller tile: a store
whose tiles are smaller than `tile_size` gets upsampled crops, which is worse.
DFC23 is in the mix for domain realism; the scale range comes from GAMUS
(0.30–0.66) and `synrs3d_g1` (0.30–0.80).

**One tile per scene**, so the scene-level split is here indistinguishable from a
tile-level one. It is still the correct thing to do and still tested — it starts
to matter the moment a source ships scenes bigger than one tile.

Budget: 1.84 MB per packed 512 px tile, so the ~1,773-scene Track 2 training
split is ≈3.3 GB.

The reference nDSMs have a failure mode worth knowing about before you weight
this source. On the New Delhi training scene `GF2_NewDelhi_28.5557_77.1194`,
**every** pixel above 100 m — 2,615 of them, up to 183.2 m — lies in rows 0-31,
a ribbon glued to the tile's top border. The RGB underneath is ordinary city
(luminance 119.5 and texture std 33.6, against 133.5 / 38.8 for the rest of the
tile); there is no structure there. They are stereo blunders.

Nothing downstream catches that on its own: `pack_labeled` filters at 500 m and
`dataset.py` clamps at `--max_valid_height_m` 200, so a phantom 183 m target on
a normal rooftop trains as fact — and `StratumBalancer` (beta 0.7, clip 8) gives
the tallest stratum the *largest* loss weight in the batch, on a model whose
measured problem is already tall-structure bias (`tall_bias −1.70`, 20 m+ bias
−2.20). So `prepare_data.py` now reports the tall mass split by border vs
interior after packing:

```
[dfc23] HEIGHT CHECK dfc23_g050/train: median 0.00 m  p99 99.7 m  max 183.1 m  exactly-0 85.6 %
[dfc23]   >100 m: 4.256 % of border-32px pixels vs 0.000 % of interior   [!] concentrated
          at the tile border — stereo blunders, not buildings.
```

`--dfc23_max_height_m 100` marks those pixels **invalid** rather than clipping
them — a blunder is an unknown, not a building of the ceiling height, and
clipping would train 100 m as fact. On that scene it takes p99 from 99.7 m to
28.0 m, which is a plausible New Delhi figure. It is off by default and the
check is a report, not a silent edit: real buildings do sit at tile edges,
because DFC23's tiles are cut from larger scenes, and three scenes are not
enough to know how common this is. Pack the real split, read the line, decide.

Also note the label distribution is not GAMUS-shaped. The three New Delhi nDSMs
run 69-87 % at exactly 0.0 m with a second mass at 20-50 m (9-24 %) and very
little between 0.5 and 5 m — consistent with a ~2 m stereo product resampled to
0.5 m. GAMUS's val prefix, by contrast, is 49 % under 2 m and 9 % above 20 m.
Mixing the two shifts what the stratum balancer sees; watch `per_stratum` rather
than assuming the weights transfer.

The pairing needed one new piece. DFC23 discriminates optical from height by
**parent directory** (`rgb/P_0199.tif`, `dsm/P_0199.tif`, identical filenames),
while `dwdata.india.pair_rasters` keys on filename *suffix* — handed that tree
directly it claims whichever file it walks first as the RGB and every scene comes
out unlabelled. `prepare_dfc23` pairs across the directories and symlinks into
the `<stem>_rgb` / `<stem>_ndsm` convention; everything after that is
`pack_labeled`, unchanged. `prepare_data.py` prints a `HEIGHT CHECK` per store —
DFC23 ships an nDSM, so the median belongs near 0, and a median metres above zero
is an absolute DSM that would poison the target silently.

---

## 6. Reading the results

`metrics.json` carries `final_plain`, `final_tta` **and** `final_sliding_tta`.
Of the three, quote the last: it scores every pixel of every val tile at native
GSD through the same code the demo runs.

All three, though, are scored on the **val** tiles — and `best.pt` is selected
on those same tiles (`val_tiles` takes the first 400 in sorted-stem order), so a
`final_*` number is measured on the set that chose the checkpoint. Run with
`--test_sources gamus:test` and `metrics.json` also carries
`test_gamus_test_plain` / `_tta` / `_sliding_tta`, scored once at the end on a
split no epoch and no eval ever touched. When those exist, they are the numbers
to quote, and the HTML report headlines them automatically; `final_*` stays for
the v1–v4 comparison. `--test_tiles` sizes the centre-crop sweep (0 = all) and
`--test_sliding_tiles` caps the sliding one (~6 s/tile).

One thing to know before you read those three against each other — on v3 they
came out **2.715 / 2.605 / 2.723 m**, so the sliding number was the *worst* of
the three. That is not a regression in the sliding path: it scores 415 M pixels
of whole tiles against the centre crop's 104 M, and whole tiles include the tile
edges and the low-texture margins a centre crop never sees. It is the harder and
more honest number, which is why it is the one to quote — but do not present the
gap as "TTA stopped working". Dihedral TTA is worth ~4 % (2.715 → 2.605) on a
like-for-like comparison and that is why `--tta` stays on.

Watch six numbers, not one:

| field | what it catches | v2's value |
|---|---|---|
| `global.rmse_m` | headline | 3.068 m |
| `balanced_rmse_m` | tail error the global hides | 4.19 m |
| `tall_gt15m.bias_m` | tall-structure underestimation | ≈ −4.5 m |
| `flat_lt1m.bias_m` | hallucinated ground height (the Austin failure) | ≈ 0 in-domain |
| `landscape_rmse_spread_m` | the rubric's stability axis | never measured |
| `landscape_worst` | which landscape to fix next | never measured |

---

## 7. Known limits, stated plainly

* **No Indian accuracy number exists yet, and this repo cannot produce one.** §2.
  The unlabeled branch narrows the gap; only ISRO imagery measures it.
* **The class-id mapping is unverified.** GAMUS's measured val histogram
  contradicts the name order v1/v2 asserted — id 5 ("bridge") is 16 % of all
  pixels, id 0 ("ground") is 0.1 %. Per-class metrics use neutral `classN` labels
  plus a measured histogram. Fix it against the GAMUS paper before any per-class
  claim reaches a judge. `geo/calibrate.py` is designed so a wrong id here is
  harmless (§1.1).
* **Landscape labels are heuristics** derived from GT geometry, not annotations,
  and because our target is nDSM the terrain has largely been differenced out —
  so a near-zero `hilly` count on GAMUS is expected, and is itself the finding.
* **GAMUS 1024² tiles cannot supply a GSD coarser than 0.66 m** at a 512 tile.
  That is geometry, not a setting; coarse-GSD robustness comes from the SynRS3D
  `g1` family (0.6–1.0 m) and from DFC23's 0.8 m Gaofen-2 scenes. Note the
  `g05` family the mix used to rest on tops out at 0.60 m and never covered it
  (§5.10).
* **A GSD family is a range, and `index.json` holds one scalar.** SynRS3D
  publishes GSD per *folder* — there is no per-image GSD anywhere in the
  archives — so a `g05` tile carries its family's 0.45 m midpoint and is still
  up to ±33 % off its own true GSD. Splitting the families into separate stores
  bounds that; it does not remove it.
* **DFC23 is one contest's imagery, not a neutral satellite benchmark.** It is
  the honest out-of-domain number available today and it is reported next to
  GAMUS rather than in place of it — but seventeen contest cities are not
  Cartosat, and the second val column should be read as "generalises off GAMUS",
  not as an ISRO number.
* **GeoNRW's nDSM is a proxy** (`DEM − smoothed large-window minimum`). Auxiliary
  supervision, not a target you quote against. Off by default.
* **torch's oneDNN CPU convolution returns NaN on gVisor/AMD containers, and it
  is not our arithmetic.** On a Modal gVisor box (43-core `AuthenticAMD`,
  `Model name: unknown`, family 191) `torch 2.10.0+cpu` dispatches oneDNN
  v3.7.1's AVX-512 kernels and `F.conv2d` intermittently returns NaN *from
  finite inputs and finite weights*. Caught in `GatedFusion.gate[0]`
  (`Conv2d(34, 8, 3, padding=1, bias=False)`): gate input finite, `absmax 57.7`;
  weights finite, `absmax 0.057`; output 7688/8192 NaN. Replaying the captured
  tensors in a fresh process is clean, so the trigger is process state, not
  data. It is not confined to inference — a bare `net(t)` on one tile fails, so
  training sees it too as skipped non-finite gradient steps.

  The rate tracks vector width, which is the tell:

  | oneDNN setting | non-finite forwards |
  |---|---|
  | `mkldnn.enabled = False` | 0 / 3000 |
  | `ONEDNN_MAX_CPU_ISA=SSE41` | 0 / 3000 |
  | `ONEDNN_MAX_CPU_ISA=AVX2` | 8 / 3000 |
  | `ONEDNN_MAX_CPU_ISA=AVX512_CORE` | 37 / 3000 |
  | default (AVX-512 + VNNI) | 32 / 2000 |

  The CPU advertises `avx512_vp2intersect` while reporting `AuthenticAMD` — a
  combination that does not exist on real AMD silicon — so this is gVisor's
  synthetic CPUID leading oneDNN to pick kernels the host does not actually
  honour. Export `ONEDNN_MAX_CPU_ISA=SSE41` when running on such a container;
  with it the suite is 169 passed, and the only remaining failures are the
  torchvision pin note below. No source guard is carried for this: the defect is in
  the container, and a host-sniffing workaround in the model would be wrong on
  real Zen 4/5 hardware, where AVX-512 oneDNN is correct and fast.

* **`torchvision`/`torchaudio` can be ABI-mismatched against `torch`, which
  silently unplugs the real encoder tests.** `requirements.txt` deliberately
  does not pin torch (§5), so an image can end up with `torchvision 0.23.0+cu129`
  (which pairs with torch 2.8) beside `torch 2.10.0+cpu`. Then
  `transformers.modeling_layers` raises `operator torchvision::nms does not
  exist`, `tests/test_real_dinov3.py` reports 5 errors + 1 failure, and the real
  DINOv3 path — the one a GPU run actually uses — goes unexercised while the
  stub-encoder tests stay green. Check `torchvision.__version__` matches the
  torch build before trusting a green suite. The mismatch has a second face: on
  a Modal container with `torch 2.10.0+cu128` the import itself dies as
  `partially initialized module 'torchvision' has no attribute 'extension'`
  before any op is looked up. Install the pair in one resolve off the pytorch
  index and neither face appears — `uv pip install torch==2.10.0 torchvision
  --index-url .../whl/cu128`, with torchvision left unpinned so the torch pin
  selects its matching build. `--extra-index-url` is the wrong flag here: it
  leaves PyPI in play for the other half of the pair.

* **v4 has not been trained yet.** Everything here is verified by 113 tests, by a
  real end-to-end CLI run, by a real GeoTIFF round trip, and by driving the
  viewer in a headless browser against a real prediction directory — but the
  accuracy numbers are still to be earned.

## 8. Licensing

| component | licence | note |
|---|---|---|
| DINOv3 ViT-L/16 SAT-493M | Meta DINOv3 licence | commercial use permitted with conditions |
| GAMUS | see the HF dataset card | gated |
| SynRS3D | see the HF dataset card | NeurIPS'24 |
| DFC23 Track 2 | IEEE GRSS contest terms | **attribution is mandatory** — see below |
| three.js r128 | MIT | vendored in `viewer/vendor/`, notice retained |
| Copernicus DEM GLO-30 | free, attribution required | ESA / Copernicus |
| DA-V2 **Small** | Apache-2.0 | Base/Large/Giant are CC-BY-NC-4.0 — do **not** use them |

Ship this table with the deliverable. A licence challenge on a government
deliverable is cheap to prevent and expensive to lose.

The DFC23 terms are not a formality — they require, verbatim, in any publication
using the data:

> "[REF. NO.] 2023 IEEE GRSS Data Fusion Contest. Online:
> www.grss-ieee.org/technical-committees/image-analysis-and-data-fusion/"

plus an *Acknowledgement* section thanking the IEEE GRSS Image Analysis and Data
Fusion Technical Committee, the Aerospace Information Research Institute of the
Chinese Academy of Sciences, Universität der Bundeswehr München and GEOVIS Earth
Technology Co., Ltd. for organizing the Data Fusion Contest, and a citation of
Huang et al., 2022, *Urban Building Classification (UBC) — A Dataset for
Individual Building Detection and Classification from Satellite Imagery*,
CVPR 2022, pp. 1413–1421. See `CompetitionContext/DFC23_Building_Classification.md`.

```bash
python -m pytest -q        # 113 passed
```
