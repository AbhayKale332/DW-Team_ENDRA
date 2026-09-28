# DepthWizard v4 — Phase 2 + Phase 3

SIH 2026, PS 26175 (ISRO): single-view optical RGB → metric DSM → navigable 3D
flythrough.

v3 was a training pipeline. **v4 is the deliverable**: the same training core
plus the geo half (ground-masked DTM fit, DEM fetch, GCP refinement, ONNX
export), the product half (FastAPI service, standalone Three.js viewer,
auto-generated validation report), and three model changes each driven by a
measurement rather than a hunch.

```
FineTunning/v4/
  config.py prepare_data.py train.py main.py run_lightning.sh
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
   `train.py` and exports it from `run_lightning.sh`.
2. **A shared-memory `Bus error`** when an autotuner walked the worker count up
   to 35 on a 16-vCPU box. v4 sets the worker budget statically (16 train + 4
   val) and, because the workers now hand over uint8, each sample on the wire is
   a quarter of the size.
3. **`STARVED` in `gpu_disk_guard.log`** on the runs that read shards straight
   off network-backed storage. The run that finished read from `/tmp`;
   `run_lightning.sh stage` now makes that copy a first-class step.

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
python prepare_data.py --datasets india_labeled --india_dir ~/dfc23_newdelhi
python train.py --datasets gamus,synrs3d,india_labeled \
                --sampler_weights gamus:1,synrs3d:1,india_labeled:3
```

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
bash run_lightning.sh serve                      # torch
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
time with `bash run_lightning.sh report`.

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

## 5. Running it on Lightning AI (1× H100 80 GB)

```bash
export HF_TOKEN=hf_...          # DINOv3-SAT and GAMUS are both gated

bash run_lightning.sh check     # deps + offline tests + GPU report
bash run_lightning.sh prepare   # ONE TIME, ~40 min, ~35 GB on the persistent disk
bash run_lightning.sh stage     # copy the shards to local NVMe  <-- do not skip
bash run_lightning.sh india --datasets india_unlabeled --india_dir ~/tiles   # optional
bash run_lightning.sh smoke     # ~10 min end-to-end
bash run_lightning.sh train     # the real run
```

`prepare` is resumable and idempotent — a split with an `index.json` is skipped.
Because the studio disk persists, later runs start training immediately.

**`stage` matters.** Training does ~12 000 random 512 px crops an epoch out of
~27 GB of shards: small random reads. On the box's own NVMe the page cache
absorbs them; off a network mount they are the entire bottleneck, and no worker
count fixes it — `Logs/v3/gpu_disk_guard.log` prints `STARVED` for exactly the
attempts that skipped this, and the run that finished read from `/tmp`. `train`
picks the local copy up automatically when it holds a packed store
(`DW_NO_AUTOSTAGE=1` to opt out).

### The defaults, and where each number came from

Everything here is read off `Logs/v3/` — see §1.6 for the measurements.

| knob | v4 | why |
|---|---|---|
| `batch_size 24`, `grad_accum 1` | ~58 GB | micro-32 was 74 GB for +7 % and OOMed twice; micro-16 left 38 GB idle |
| `grad_checkpoint_encoder false` | +~35 % step | v3 ran ViT-L unfrozen @512 in 42 GB; the memory it saves is memory nothing wants |
| `eval_batch_mult 2` | eval at 48 | eval keeps no activations |
| `num_workers 16`, `prefetch_factor 6` | 99-100 % GPU util in v3 | plus 4 capped val workers, not a second full pool |
| `gpu_augment true` | 0.79 MB/sample on the wire | jitter + normalisation are pointwise; they belong on the card |
| `amp_dtype bf16`, `tf32`, `channels_last`, `sdp_flash` | Hopper defaults | flash SDPA matters most: 1024 tokens of attention at 512 px |
| `epochs 40`, `max_minutes 280` | v3 used 121 min of 330 and was still improving | the cap covers training only; the final eval + exports add ~30 min |
| `stratum_balance_beta 0.7`, clip `8.0` | tall bias frozen at −2.1 m for 20 epochs | the one deliberate accuracy change |

`torch.compile` is **off** by default — the freeze→unfreeze transition forces a
recompile mid-run, and a five-hour run should not discover a Dynamo bug at epoch
3. Enable with `--compile_model true` if you have warmed it in a smoke run first.

| stage | ~time |
|---|---|
| prepare (4000 GAMUS + 2 SynRS3D archives) | 30–60 min, network-bound, one time |
| stage to local NVMe | 3–8 min, one time per box |
| epochs 1–2, encoder frozen | ~3 min/epoch (~70 img/s) |
| epochs 3–40, encoder trainable | ~4–5 min/epoch (~48 img/s; +~25 % with the unlabeled branch at `consistency_every 2`) |
| eval every 2 epochs | ~0.5 min each |
| final eval + TTA + sliding-window eval | 20–35 min |
| figures + report + ONNX | 3–6 min |
| **total** | **≈ 4–5 h**, training hard-capped by `--max_minutes 280` |

`last.pt` and `best.pt` are written every epoch; `--resume outputs/v4/last.pt`
picks up a killed box. The final evaluation, the figures, the report and the ONNX
export are each guarded separately — a five-hour run must not lose its report
because matplotlib choked, or its ONNX because the report did.

Watch the `img/s` and `vram=` fields in the train log: they are there so the two
failure modes are visible while the run is happening rather than afterwards.
`img/s` far below ~48 with the encoder trainable means the loader is behind;
`vram=` climbing epoch over epoch is the drift that killed v3's batch-32 attempt.

Knobs if memory or time is tight:

```bash
--batch_size 16 --grad_accum 2         # back to exactly v3's effective batch, ~42 GB
--grad_checkpoint_encoder true         # only on a genuinely smaller card; ~35 % slower
--freeze_epochs 0                      # skip the warmup (riskier early gradients)
--datasets gamus                       # drop SynRS3D
--epochs 26                            # exactly v3's schedule
--tta_scales 1.0,1.25                  # multi-scale TTA at the end (slower)
--stratum_balance_beta 0.5             # back to v3's tail weighting
--w_consistency 0                      # disable the unlabeled branch outright
--consistency_every 4                  # ...or just run it less often
```

---

## 6. Reading the results

`metrics.json` carries `final_plain`, `final_tta` **and** `final_sliding_tta`.
Quote the last one: it scores every pixel of every val tile at native GSD through
the same code the demo runs.

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
  mix.
* **GeoNRW's nDSM is a proxy** (`DEM − smoothed large-window minimum`). Auxiliary
  supervision, not a target you quote against. Off by default.
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
| three.js r128 | MIT | vendored in `viewer/vendor/`, notice retained |
| Copernicus DEM GLO-30 | free, attribution required | ESA / Copernicus |
| DA-V2 **Small** | Apache-2.0 | Base/Large/Giant are CC-BY-NC-4.0 — do **not** use them |

Ship this table with the deliverable. A licence challenge on a government
deliverable is cheap to prevent and expensive to lose.

```bash
python -m pytest -q        # 113 passed
```
