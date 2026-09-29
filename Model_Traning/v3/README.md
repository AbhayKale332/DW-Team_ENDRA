# DepthWizard v3 — single-view metric height estimation

v3 is a rewrite of `../v2/` driven by evidence from the actual v2 run
(`metrics.json`, `run.log` and the `austin1` prediction in the shared `v2.zip`),
not by guesswork. Section 1 is the post-mortem; sections 2-6 are how to run it.

---

## 1. What actually went wrong in v2

The v2 run reached **best val RMSE 3.068 m at epoch 18** — a 7.6 % improvement on
v1's 3.321 m, from a much larger machine and a much more complicated pipeline.
Six separate defects explain the gap between that and what the architecture
should have delivered.

### 1.1 The run crashed at epoch 21/30 and never produced a final result

```
File "models/encoder.py", line 58, in _blocks
    raise AttributeError("cannot locate transformer blocks on the encoder")
```

`_blocks()` looked for `model.layer`, `model.layers`, `model.blocks` and the same
three on `model.encoder`. The real HF layout is **`DINOv3ViTModel.model.layer`** —
a `ModuleList` nested under a child named `model`, which that search never
visits. Verified directly:

```
top-level children: ['embeddings', 'rope_embeddings', 'model', 'norm']
ModuleLists found:  'model.layer': len=6
v2 lookup:          layer -> None   layers -> None   blocks -> None
```

Consequences, all of them in the shipped run: **the encoder was never unfrozen**,
so the entire run trained ~11 M decoder parameters on a frozen backbone (v1's
architecture with extra steps); the process died at 75 min of a 300 min budget;
and `final_plain`, `final_tta`, the viewer sample and the qualitative strips were
never written — `metrics.json` has no `final_*` key at all. The 8x TTA that the
plan calls "free accuracy" has never once been measured.

*v3:* the block list is found **structurally** — the `nn.ModuleList` whose length
equals `config.num_hidden_layers` — so a naming change cannot break it.
`tests/test_real_dinov3.py` pins both the old failure and the new behaviour
against the real `DINOv3ViTModel`.

### 1.2 ~41 % of every training pixel was black padding labelled 0 m

`TileDatasetBase.__getitem__` resampled the **whole** source tile to a random GSD
and then centre-cropped/padded to 512. GAMUS tiles are 1024 px at 0.33 m, so any
requested GSD above `1024 x 0.33 / 512 = 0.66 m` produces an image *smaller* than
the crop, and `center_crop_or_pad` fills the rest with zeros. With
`gsd_jitter ~ U(0.25, 2.0)` at `p = 0.8`:

| | measured |
|---|---|
| P(a jittered tile needs padding) | **0.765** |
| P(a training sample is >50 % padding) | **0.491** |
| mean real-pixel fraction per sample | **0.588** |
| SynRS3D mean real-pixel fraction | **0.338** |

Worse, `pack_sample` marks those pixels **valid** (`height >= 0` is true for the
zeros), so the loss actively taught the network "black region ⇒ 0 m". GeoNRW at
the fine end went the other way: 0.25 m against a 1 m source upsamples a
1000² tile to 4000², ~112 MB per tile in the DataLoader.

*v3:* the crop window is chosen **in source pixels first** and that window is
resampled to the tile size (`dwdata/augment.py:sample_window`). A tile is full by
construction, the achievable GSD range is derived from the source extent
(`achievable_gsd_range`), and the validity mask is carried through resampling so
no-data can never become a 0 m label. `tests/test_augment.py` and
`tests/test_dataset.py` assert both.

### 1.3 The model does not survive a change of sensor

This is the failure in the recorded flythrough. Predicting `austin1.tif` (Inria
Aerial, 0.3 m — essentially the same GSD as GAMUS):

| | GAMUS val tile `DC_38_35` | Inria `austin1` |
|---|---|---|
| median predicted height | — | **4.04 m** |
| pixels below 1 m | 56.4 % (GT 58.2 %) | **28.3 %** |
| mean predicted height | 3.67 m (GT 4.10) | **5.24 m** |

In-domain the model is fine; out-of-domain it puts ~4 m of height on flat ground,
and the render is a crumpled mountain range rather than flat ground with discrete
buildings. v2 had **no photometric augmentation of any kind** and no radiometric
normalisation, so it keyed on GAMUS's absolute brightness and colour balance.
ISRO will supply Cartosat imagery with different radiometry at final evaluation.

*v3:* three defences.
1. **Scene-level 2/98-percentile stretch**, applied identically in training and
   inference (`dwdata/preprocess.py:stretch_scene`) — removes the sensor's
   exposure and white balance from the input.
2. **Photometric jitter** — brightness, contrast, gamma, saturation, per-channel
   gain, blur, noise (`dwdata/augment.py:photometric_jitter`).
3. **`flatness_loss`** — penalises predicted curvature wherever the GT surface is
   locally flat, which directly prices in "invented terrain from texture", plus
   `normal_loss` on GSD-normalised physical slope.

`infer/predict.py` prints the fraction of pixels below 1 m and warns when it is
implausibly low, so this failure is caught before it reaches a demo.

### 1.4 Tall structures are systematically underestimated, and Head B never helped

On `DC_38_35`, pixels above 15 m: **GT mean 20.0 m, predicted 15.5 m, MAE 5.3 m**,
while flat ground was accurate to 0.48 m. Global RMSE (3.07 m) hides this
completely; the balanced RMSE in the same run was **4.19 m**.

Head B (adaptive bins) was supposed to fix exactly this and did not: its
cross-entropy sat at 0.06-0.13 for the entire run. With ~65 % of pixels in one
bin the nearest-bin target is trivially predictable, so the head collapsed and the
learned gate just copied Head A.

*v3:* `StratumBalancer` re-weights every pixel by the inverse frequency of its
height stratum (the same strata the balanced-RMSE metric uses), tracked with a
running EMA. The bin loss is computed on logits at half resolution with the same
weighting and a batched `searchsorted` instead of a `(B, K, H, W)` distance
tensor. `Evaluator` now reports `tall_gt15m.bias_m` and `flat_lt1m.bias_m`
directly, so both failure directions are visible every eval.

### 1.5 The training loop was I/O-bound, and stage P was actively harmful

Every tile was fetched from the Hub inside the DataLoader workers, behind a
50 GiB LRU cache over an ~80 GiB dataset that re-`rglob`'d the entire cache
directory on each download. And the SynRS3D pretrain rotated a *different*
archive in each epoch, so its train loss went **up**:

```
pretrain e1 9.62  e2 7.02  e3 8.83  e4 10.70  e5 8.84  e6 13.07
```

Six epochs and ~25 % of the compute budget for a stage whose loss ended 36 %
higher than it started, because each epoch changed the height distribution under
a schedule that had no idea.

*v3:* `prepare_data.py` materialises tiles **once** into `.npy` memmap shards; a
crop becomes a page fault instead of an HTTPS round trip. Sources are mixed in
**one** weighted sampler rather than trained in sequence, so every batch sees the
distribution the LR schedule was built for.

### 1.6 The per-class table in the deck is wrong

v1/v2 both assert `("ground", "vegetation", "building", "water", "road",
"bridge", "other")` for GAMUS's class ids. The measured val histogram says
otherwise: id 5 ("bridge") is **7.27 M pixels — 16 % of the split**, and id 0
("ground") is **41 975 pixels — 0.1 %**. Bridges are not 16 % of an aerial scene
and ground is not 0.1 %. The id order is not what the code claims.

*v3:* per-class metrics are reported under neutral `class0..class6` labels
alongside a measured `class_stats` block (pixel fraction and mean GT height per
id), so the mapping gets fixed from evidence. Fix it before quoting per-class
numbers to an ISRO judge. The semantic head is weighted 0.2 and is auxiliary.

### 1.7 Smaller things, fixed

* **Wrong normalisation constants.** DINOv3 SAT-493M was not pretrained with the
  ImageNet mean/std that v1/v2 used. v3 asks the encoder's own
  `AutoImageProcessor` and falls back to the published SAT constants.
* **`hidden_states` is `None` on transformers 5.x** when `output_hidden_states` is
  set only via `from_pretrained`. v2's code path raises `TypeError` at the first
  forward on a current Lightning image. v3 passes the flag per call.
* **`F.relu` output head** has exactly zero gradient once a unit drifts negative.
  v3 uses `softplus` with a -2.0 bias init (most pixels are ground).
* **Head B at full resolution** materialised a `(B, 128, 512, 512)` logit tensor.
  DPT predicts at 1/2 and upsamples the output; v3 does the same (~4x cheaper).
* **Augmentation was frozen across epochs** — v2 seeded the RNG off
  `(seed, tile_index)`, so all 30 epochs replayed one augmented copy of the data.
* **Validation used a different code path from inference**, so no validation
  number ever exercised what the demo runs. v3 has one path (§3).

---

## 2. Architecture

```
RGB scene (any size, any GSD)
  └─ scene 2/98-percentile stretch          ← identical in train and inference
  └─ resample to 0.5 m/px canonical GSD
  └─ 512² windows  [train: one crop, chosen in SOURCE px then resampled]
       └─ DINOv3-SAT ViT-L/16   [frozen 2 epochs, then FULLY trainable w/ LLRD 0.8]
            └─ DPT trunk → features at H/2
                 ├─ Head A  metric nDSM      softplus, metres
                 ├─ Head B  96 adaptive bins  (owns the long tail)
                 ├─ Head C  semantics         (aux; ground mask for DEM calibration)
                 └─ gated fusion α·A + (1-α)·B  → nDSM
  └─ [inference] 8× dihedral TTA → Hann-blended stitch → back to the input grid
```

Losses: `L1 + SiLog + multi-scale gradient` (all stratum-weighted) `+ normal
+ flatness + bin-CE + seg-CE`, plus weight EMA.

---

## 3. The inference contract — "same format as training"

This was the second half of the brief, and it is the reason `dwdata/preprocess.py`
exists. **Training and inference share one implementation**, and the resolved
recipe is serialised into every checkpoint:

```python
ck = torch.load("best.pt")
ck["preproc"]
# {'encoder_model_id': 'facebook/dinov3-vitl16-pretrain-sat493m',
#  'mean': (0.43, 0.411, 0.296), 'std': (0.213, 0.156, 0.143),
#  'canonical_gsd_m': 0.5, 'tile_size': 512, 'patch': 16,
#  'radiometric_stretch': True, 'stretch_lo_pct': 2.0, 'stretch_hi_pct': 98.0,
#  'target': 'nDSM_agl_metres', 'version': 'v3'}
```

`infer/predict.py` reads the contract **from the checkpoint**, never from a config
file that may have moved on. The five steps, in order:

| # | step | where |
|---|---|---|
| 1 | read RGB + resolve GSD (GeoTIFF transform → metres, handles geographic CRS and feet; else `--gsd`; else assumed) | `preprocess.read_scene` |
| 2 | scene 2/98-percentile stretch | `preprocess.stretch_scene` |
| 3 | resample to `canonical_gsd_m` (heights are metres — never rescaled) | `preprocess.to_canonical` |
| 4 | overlapping `tile_size` windows, reflect-padded, Hann-blended | `infer.engine` |
| 5 | normalise with the encoder's own mean/std | `PreprocSpec.normalise` |

Then the height map is resampled back to **the caller's own pixel grid**.

```bash
# georeferenced GeoTIFF -> absolute-grid nDSM GeoTIFF; GSD from the transform
python -m infer.predict scene.tif --ckpt outputs/v3/best.pt --tta

# non-georeferenced PNG/JPG -> rDSM. Pass --gsd to make it metric.
python -m infer.predict scene.png --ckpt outputs/v3/best.pt --gsd 0.5

# nDSM + coarse DEM -> absolute DSM (the plan's decomposition)
python -m infer.predict scene.tif --ckpt outputs/v3/best.pt --dem srtm_30m.tif

# print the resolved contract and exit
python -m infer.predict scene.tif --ckpt outputs/v3/best.pt --report
```

Outputs in `--out-dir`: `ndsm_m.tif` (float32, source CRS + transform,
pixel-identical grid), `dsm_m.tif` when `--dem` is given, `ndsm_m.npy`,
`ndsm16.png` (16-bit, with the affine encoding written into `meta.json` so metres
are recoverable), `rgb.png`, and `meta.json` carrying the full contract, the
scene metadata and distribution diagnostics.

**Verified end to end** — `tests/test_engine.py` asserts the blend reproduces a
constant exactly, has no seams, and that metre values are invariant to the GSD
round trip; a real GeoTIFF round trip was checked to be pixel-identical to its
source, and `--dem` correctly reprojects a 30 m DEM onto the prediction grid.

---

## 4. Running it on Lightning AI (1× L40S 48 GB)

Two scripts, two machine types, because downloading 45 GB of datasets on a GPU
is the most expensive thing this repo can do:

```bash
export HF_TOKEN=hf_...          # DINOv3-SAT and GAMUS are both gated

# --- on a CHEAP CPU Studio -------------------------------------------------
sh prepare_data.sh --background       # fetch + pack + stretch bounds + encoder
                                      # ~40-60 min, ~45 GB on the persistent disk

# --- switch the Studio to an L40S ------------------------------------------
sh train_L40S.sh --show-tuning        # resolve flags and batch, launch nothing
sh train_L40S.sh --smoke              # ~10 min end-to-end
sh train_L40S.sh --background         # the real run
```

`prepare_data.sh` is resumable and idempotent — a split with an `index.json` is
skipped — and it writes a `.dw_data_ready` stamp that `train_L40S.sh` refuses to
start without. Because the Studio disk persists, the GPU box starts training
inside a couple of minutes.

### 4.1 How the L40S gets filled

The first real run showed steps moving with VRAM pinned near 20 GiB of 48 and
GPU util in the fifties. That was two independent problems:

**The batch was never measured.** The tuner picked it from a VRAM bracket whose
top tier was `>= 40 GiB -> 16`, and 48 falls in it. Sixteen 512 px crops through
an unfrozen ViT-L is ~20 GiB — the flat line exactly. `tools/probe_batch.py` now
runs a real forward/backward/AdamW step on the actual card **with the encoder
unfrozen** (the peak that matters is the one after `--freeze_epochs`, not the
cheap warmup) and returns the largest batch that fits under `DW_VRAM_HEADROOM`.
The answer is cached per card in `~/.dw_probe_<gpu>_t<tile>`; an OOM deletes it.

Two flags were also being dropped silently, because the tuner matched option
names with a whole-line exact match: `--compile` never matched `--compile_model`
and `--grad_checkpoint` never matched `--grad_checkpoint_encoder`. Both now fall
back to a prefix match, and `--show-tuning` prints every flag it resolved.

**The loader could not fill it.** One GAMUS crop cost ~82 ms of CPU:

| | before | after |
|---|---|---|
| per-tile stretch bounds (full-tile histogram, per crop, per worker) | 13.4 ms | 0 — precomputed by `prepare_data.sh` |
| photometric jitter (single-threaded numpy) | 17.3 ms | 0 — batched on the GPU, ~2 ms/batch |
| `normalise` → fp32 CHW | 4.5 ms | 0 — on the GPU |
| PIL resize + copies | ~22 ms | ~22 ms |
| **per sample** | **75.6 ms** | **22.2 ms** |
| **per worker** | **13.2/s** | **44.9/s** |
| **host→device per sample** | 7.34 MB | 2.36 MB |

(synrs3d, whose tiles are 512 px rather than 1024: 34.1 ms → 11.3 ms.)

The bounds are a property of the *scene*, so they are computed once per store
and persisted as `stretch_bounds_2_98.npy` next to the shards. The jitter and
the normalisation are pointwise arithmetic, so they belong on the card
(`--gpu_augment`, `dwdata/gpu_aug.py`); the worker hands over uint8 HWC, which
permutes into `channels_last` on the device for free. The maths is unchanged —
`photo_p=0` reproduces `PreprocSpec.normalise` to 7e-4, and the jittered
distribution matches the numpy one in mean, std and both tails.

Three smaller things in the same direction: `cls` travels as uint8 and is
widened to int64 on the device (an int64 label map is 2 MB at 512 px, more wire
than the image); the train loader no longer carries the `rgb_u8` export copy;
and `evaluate()` is now `@torch.no_grad()`, which is why eval can run at
`eval_batch_mult` times the training batch instead of forcing both down.

`--stage-local` is conditional now. On a Lightning Studio `/tmp` and
`/teamspace/studios/this_studio` are the same overlay mount, so the old
unconditional 45 GB copy moved data from a disk to itself and cost ~10 minutes of
L40S time for nothing. It only copies across a genuinely different filesystem;
otherwise a niced background pass warms the page cache while the model loads.

### 4.2 Budget

| stage | ~time |
|---|---|
| prepare (4000 GAMUS + 2 SynRS3D archives), on a CPU box | 30-60 min, network-bound, one time |
| GPU startup: deps check, encoder load, batch probe | ~3 min (probe cached after the first run) |
| epochs 1-2, encoder frozen | ~3 min/epoch |
| epochs 3-26, encoder trainable | ~9-12 min/epoch |
| final eval + TTA + sliding-window eval | 20-35 min |
| **total** | **≈ 5-6 h**, hard-capped by `--max_minutes 330` |

If the box dies, `last.pt` and `best.pt` are written every epoch and
`train_L40S.sh` warm-starts from `last.pt` automatically.

### 4.3 Knobs

```bash
DW_BATCH=32 sh train_L40S.sh           # skip the probe, force a batch
DW_VRAM_HEADROOM=0.93 sh train_L40S.sh --force   # re-probe, fill more of the card
DW_EFFECTIVE_BATCH=48 sh train_L40S.sh # batch x grad_accum target
DW_LOADER_WORKERS=14 sh train_L40S.sh  # more/fewer DataLoader workers
DW_COMPILE=1 sh train_L40S.sh          # torch.compile (~1.2x, multi-minute warmup)
DW_WARM_CACHE=0 sh train_L40S.sh       # skip the background page-cache warm
```

```bash
--gpu_augment false                    # fall back to the all-CPU pipeline
--grad_checkpoint_encoder true         # ~35% slower, ~2.5x smaller activations
--freeze_epochs 0                      # skip the warmup (riskier early gradients)
--datasets gamus                       # drop SynRS3D
--tta_scales 1.0,1.25                  # multi-scale TTA at the end (slower)
--stratum_balance_beta 0.75            # push harder on tall structures
```

The watchdog in `train_L40S.sh` samples `nvidia-smi` every 2 minutes and says
which of the two failure modes it is seeing — low util *and* low VRAM means the
loader, high util with low VRAM means the batch — so the next run is better
sized than this one. The verdict is repeated at the end and lands in
`RUN_MANIFEST.txt`.

---

## 5. Reading the results

`metrics.json` now carries `final_plain`, `final_tta` **and**
`final_sliding_tta`. Quote the last one: it scores every pixel of every val tile
at native GSD through the same code the demo runs.

Watch four numbers, not one:

| field | what it catches | v2's value |
|---|---|---|
| `global.rmse_m` | headline | 3.068 m |
| `balanced_rmse_m` | tail error the global hides | 4.19 m |
| `tall_gt15m.bias_m` | tall-structure underestimation | ≈ −4.5 m |
| `flat_lt1m.bias_m` | hallucinated ground height (the Austin failure) | ≈ 0 in-domain |

`class_stats` prints the measured pixel fraction and mean GT height per class id —
use it to fix §1.6 before any per-class claim goes in the deck.

---

## 6. Layout

| path | role |
|---|---|
| `config.py` | one dataclass, every knob `--flag`-overridable |
| `prepare_data.py` | HF → memmap shards, one time, resumable |
| `dwdata/preprocess.py` | **the shared contract** — normalisation, GSD, tiling |
| `dwdata/augment.py` | crop-then-resample scale aug + photometric jitter |
| `dwdata/packed.py` | memmap shard writer/reader |
| `dwdata/dataset.py`, `loaders.py` | samples and DataLoaders |
| `models/` | encoder (robust blocks + LLRD), DPT trunk, heads, losses, EMA, TTA |
| `eval/metrics.py` | global / per-class / per-stratum / tall / flat |
| `eval/sliding.py` | full-tile eval **through the inference path** |
| `infer/engine.py` | the one sliding-window prediction implementation |
| `infer/predict.py` | the CLI: PNG/JPG/GeoTIFF → DSM + GeoTIFF/PNG/NPY |
| `run_lightning.sh` | check / prepare / smoke / train / predict |
| `tests/` | 65 offline tests, no GPU and no network required |

```bash
python -m pytest -q        # 65 passed
```

---

## 7. Known limits, stated plainly

* **GAMUS 1024² tiles cannot supply a GSD coarser than 0.66 m** at a 512 tile —
  that is geometry, not a setting. Coarse-GSD robustness comes from the SynRS3D
  mix; if you train `--datasets gamus` alone, expect degradation above ~0.7 m.
* **GeoNRW's nDSM is a proxy** (`DEM − smoothed large-window minimum`) because
  GeoNRW ships elevation, not height above ground. It is auxiliary supervision,
  not a precise target, and it is off by default.
* **The class-id mapping is unverified** (§1.6). Neutral labels until it is.
* **v3 has not been trained yet.** Everything here is verified by tests, by a real
  end-to-end CLI run on synthetic data, and by measurements taken from the v2
  checkpoint — but the accuracy numbers are still to be earned.
