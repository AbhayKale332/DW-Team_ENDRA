# Outcomes — versions & experiments

Fill in after each Kaggle run. Numbers here feed the idea deck.

## v1 — Phase 0 spike (`FineTunning/v1/kaggle_phase0.py`)

DINOv3-SAT (frozen) → DPT decoder → metric-nDSM head (Head A only).
GAMUS subset (`train_subset` / `val_subset`), 512² crops @ native 0.33 m GSD.
Source log: `FineTunning/v1/Phase0_Cell_Ouptut.md` (full run, 12 epochs, 28 min).

| Run | Encoder | Train / Val batches | Epochs | Val RMSE (m) | MAE (m) | Pearson r | δ1 | Notes |
|---|---|---|---|---|---|---|---|---|
| e11 (best) | dinov3-vitl16-pretrain-sat493m | 1200 / 300 | 12 | **3.321** | 1.632 | 0.856 | 0.620 | new best → `best.pt` |
| e12 (final) | dinov3-vitl16-pretrain-sat493m | 1200 / 300 | 12 | 3.340 | 1.622 | 0.855 | 0.626 | last epoch |

Per-epoch val RMSE (m): e1 5.29 · e2 4.14 · e3 4.18 · e4 4.34 · e5 3.61 · e6 3.51 ·
e7 3.39 · e8 3.37 · e11 3.32 · e12 3.34 — steep drop to e5, then a shallow plateau
(3.6 → 3.3 over 7 more epochs).

Per-land-cover val RMSE / MAE (m), epoch 11 (from the eval log):

| | ground | vegetation | building | water | road | bridge | other |
|---|---|---|---|---|---|---|---|
| RMSE | 4.520 | 1.475 | 1.506 | 4.862 | 1.684 | 2.567 | 4.882 |
| MAE  | 3.265 | 0.454 | 0.479 | 3.046 | 0.390 | 1.018 | 3.573 |
| n (px) | 41 975 | 8.68 M | 10.90 M | 7.58 M | 0.59 M | 7.27 M | 8.85 M |

**Read:** structure classes (vegetation / building / road) are already good (RMSE ≈ 1.5 m);
the error is concentrated in `water`, `other`, and `ground` (RMSE 4.5–4.9 m) — flat/low
classes where absolute metres and the long tail hurt most. Early epochs had `ground`
RMSE ≈ 2 m and it *drifted up* as the model over-fit tall structures. This is the case for
v2's Head B (adaptive bins for the tail) + GSD jitter + more varied terrain data.

### Configuration (v1 Phase 0)

| Field | Value |
|---|---|
| Encoder | `facebook/dinov3-vitl16-pretrain-sat493m`, **frozen**, fp16 weights (`encoder_half=True`) |
| Encoder taps | hidden_states (6, 12, 18, 24); patch 16; 5 prefix tokens; hidden 1024 |
| Decoder | DPT (reassemble → RefineNet fusion), `decoder_dim=256`, grad-checkpointed |
| Heads | Head A only — metric nDSM regression, `ReLU` (non-negative) |
| Trainable params | 11.3 M / 314.4 M total |
| Input | 512×512 crop (random for train, centre for val) @ native 0.33 m GSD, ImageNet norm |
| Targets | AGL/nDSM in metres, valid mask = finite ∧ 0 ≤ h ≤ 400 m |
| Loss | `1.0·L1 + 0.5·grad(4-scale) + 1.0·SiLog(λ=0.85, shift=1.0)` |
| Optim | AdamW lr 3e-4, wd 1e-2, OneCycleLR (pct_start 0.1), grad-clip 1.0 |
| Batch | `batch_size=4`, `grad_accum=2` defaults; run log shows `nn.DataParallel` over 2×T4, per-GPU batch ~1 |
| Precision | AMP (fp16 autocast), `channels_last`, `cudnn.benchmark=False` |
| Hardware / time | 2× Tesla T4 (Kaggle), 28 min wall-clock, hard cap 150 min |
| Seed | 42 |
| Artifacts | `outputs/v1/{metrics.json, best.pt (decoder+head, no encoder), viewer_sample/}` |

### Baseline — IMELE on the same GAMUS val split (`eval_imele_on_gamus.py`)

| Variant | RMSE (m) | MAE (m) | Pearson r |
|---|---|---|---|
| raw metres | _pending_ | — | — |
| per-tile scale+offset aligned | _pending_ | — | — |

_Needs `Notebooks/depth-wiz.ipynb` run with `IMAGE_INPUT` = the GAMUS val RGB tiles, then
`python eval_imele_on_gamus.py --pred_dir … [--align_scale]`._

---

## v2 — Phase 2 (`FineTunning/v2/`)  _(in progress)_

Streamed **SynRS3D pretrain → GAMUS + GeoNRW fine-tune**. DINOv3-SAT encoder (frozen, last
4 blocks unfrozen for the final ~30% of epochs) → shared DPT trunk →
**Head A (metric) + Head B (adaptive bins) + Head C (6-class semantics) + gated fusion**.
GSD canonicalization (0.5 m) + jitter (0.25–2.0 m). 8× dihedral + 2-scale TTA at eval.
Target hardware: 1× RTX PRO 6000 (96 GB), ~6–8 GPU-h.

| Run | Stage | Datasets | Epochs | Val RMSE (m) | RMSE+TTA (m) | MAE | r | δ1 | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-09-09 | P + F | SynRS3D → GAMUS+GeoNRW | 6 + **20 of 30** | **3.068** (e18) | _never measured_ | 1.538 | 0.890 | 0.615 | **crashed at e21**; 75 min of a 300 min budget |

**The run did not finish.** `train.py` died in `encoder.unfreeze_last_n_blocks`:

```
AttributeError: cannot locate transformer blocks on the encoder
```

`models/encoder.py::_blocks()` searched `model.{layer,layers,blocks}` and
`model.encoder.{...}`; the real HF layout is `DINOv3ViTModel.model.layer`. So the
encoder was **never unfrozen** (the run trained ~11 M decoder params on a frozen
backbone — v1's architecture), and `final_plain` / `final_tta` / `viewer_sample` /
`qualitative` were never written. **The 8× TTA number in the deck does not exist yet.**

Per-epoch val RMSE (m): e2 3.68 · e4 3.68 · e6 3.58 · e8 3.34 · e10 3.39 · e12 3.55 ·
e14 3.21 · e16 3.21 · **e18 3.07** · e20 3.18 — still improving when it died.

Stage P (SynRS3D pretrain) train loss went **up**: 9.62 → 7.02 → 8.83 → 10.70 →
8.84 → 13.07, because each epoch rotated in a different archive with a different
height distribution under a schedule that assumed one. ~25 % of the compute budget.

### Where the error actually is (measured, GAMUS val tile `DC_38_35`)

| region | GT mean | pred mean | MAE |
|---|---|---|---|
| GT < 1 m (flat ground, roads) | ~0.2 m | 0.48 m | **0.48 m** |
| GT > 15 m (tall structures) | 20.00 m | 15.45 m | **5.33 m** |

Global RMSE 3.07 m; **balanced RMSE 4.19 m**. In-domain the model is good on flat
ground and underestimates tall structures by ~4.5 m. Head B (adaptive bins) was
supposed to fix this and did not — its CE sat at 0.06–0.13 all run, i.e. it
collapsed and the gate just copied Head A.

### The cross-sensor failure (this is what the flythrough video shows)

`austin1.tif` (Inria Aerial, 0.3 m — nearly the same GSD as GAMUS):

| | GAMUS val | Inria austin1 |
|---|---|---|
| pixels below 1 m | 56 % (GT 58 %) | **28 %** |
| median predicted height | — | **4.04 m** |
| mean predicted height | 3.67 m (GT 4.10) | **5.24 m** |

Out of domain the model puts ~4 m of height on flat ground and the render is a
crumpled mountain range instead of flat ground with discrete buildings. v2 had
**zero photometric augmentation** and no radiometric normalisation. ISRO will
supply Cartosat imagery with different radiometry at final evaluation, so this is
the single biggest risk to the 50 % accuracy score.

### Two silent data bugs

1. **~41 % of every training pixel was black padding labelled 0 m and marked
   valid.** `TileDatasetBase` resampled the whole tile to a random GSD *then*
   centre-cropped/padded to 512; GAMUS 1024 px @ 0.33 m cannot fill a 512 crop
   beyond 0.66 m GSD, and the jitter range went to 2.0 m. Measured:
   P(padding | jittered) = 0.765, P(>50 % padded) = 0.491.
2. **Augmentation was frozen across epochs** — the RNG was seeded off
   `(seed, tile_index)`, so all 30 epochs replayed one augmented copy of the data.

### ⚠ The per-class table above (and in v1) is wrong

GAMUS class ids do not match the assumed name order. Measured on the val split:
id 5 ("bridge") is **7.27 M px = 16 % of the split**; id 0 ("ground") is
**41 975 px = 0.1 %**. Bridges are not 16 % of an aerial scene. **Do not quote
per-class numbers to an ISRO judge until the id order is verified against the
GAMUS paper.** v3 reports neutral `class0..class6` labels plus a measured
histogram so the mapping can be fixed from evidence.

---

## v3 — `FineTunning/v3/`  _(built, tested, not yet trained)_

Rewrite driven by the measurements above. See `FineTunning/v3/README.md` §1 for
the full post-mortem. Headline changes:

* crop-then-resample scale augmentation → **no padding, ever** (fixes 41 % of pixels)
* structural transformer-block discovery → **the encoder actually unfreezes**, fully,
  with layer-wise LR decay (0.8) after a 2-epoch frozen warmup
* scene percentile stretch + photometric jitter + `flatness`/`normal` losses →
  cross-sensor robustness
* height-stratum inverse-frequency loss weighting → tall-structure tail
* data materialised once into memmap shards → GPU-bound instead of network-bound
* one sources mix in one sampler instead of a sequential pretrain stage
* **one inference path**, its contract serialised into every checkpoint
* new diagnostics: `balanced_rmse_m`, `tall_gt15m.bias_m`, `flat_lt1m.bias_m`,
  `class_stats`, and `final_sliding_tta` (full tiles, native GSD, inference path)

| Run | Datasets | Epochs | RMSE (m) | balanced | tall bias | flat bias | Notes |
|---|---|---|---|---|---|---|---|
| _pending_ | GAMUS + SynRS3D | 26 | — | — | — | — | 1× H100, ~5–6 h |

---

## v4 — `FineTunning/v4/`  _(built, tested, not yet trained)_

Phase 2 + Phase 3 in one tree: v3's training core plus the geo half
(ground-masked DTM fit, DEM fetch, GCP RANSAC, ONNX export) and the product half
(FastAPI service, standalone Three.js viewer, auto-generated validation report).
See `FineTunning/v4/README.md` §1 for the full rationale.

**Model changes, each fixing something measured:**

* `--dem` in v3 **double-counted every building** — GLO-30 and SRTM are surface
  models, so `dem + ndsm` put a 30 m tower on 12 m terrain at ~165 m instead of
  135 m. `geo/calibrate.py` now fits the bare-earth DTM through ground pixels
  only (Head C's ground classes **and** a low predicted nDSM, both required) and
  adds the nDSM to that. Measured on a synthetic scene: 134.4 m against a true
  135.0 m, DTM recovered to 0.4 m mean error.
* Head B kept a **hard** nearest-bin target in v3 — nearly free to predict when
  65 % of the mass is one bin, which is why it collapsed in v2. Now a
  Gaussian-smoothed target (`bin_soft_sigma 1.5`) plus an entropy floor on the
  marginal bin distribution (`w_bin_entropy`).
* Head B also emits `b_std` (bin-distribution sigma, in metres) — a free
  per-pixel uncertainty that gates the unlabeled branch.
* **Per-landscape metrics** (urban / sparse / hilly / forested), derived per tile
  from the GT height field, so the rubric's own stability axis is finally
  answerable. Nothing in v1–v3 measured it.

**Indian data — what we found.** No open dataset pairs Indian RGB with per-pixel
heights. DFC2023 Track 2 *does* include a New Delhi city (2 m nDSM from Gaofen-7
/ WorldView stereo) but is behind an IEEE DataPort login with no API; Open
Buildings 2.5D and UT-GLOBUS give building heights only, with no paired RGB;
Bhoonidhi/CartoDEM is 10–30 m, too coarse for structure height but right for the
terrain term. SAC's own reference repo has a README and no data.

So v4 ships two ingest paths and is explicit that neither is an Indian benchmark:
`india_labeled` (a local directory of paired rasters — DFC2023 New Delhi once
downloaded by hand, or any product the team obtains) and `india_unlabeled`
(RGB-only tiles feeding a **mean-teacher** branch: EMA teacher on a weak view,
student on a strong one, pixels gated by the teacher's `b_std`). 14 Indian AOIs
spanning all four landscape types are defined; no basemap URL is defaulted,
because that licence call belongs to the team.

**Deliverables the run now produces itself:** figures, a self-contained HTML
validation report, `terrain.glb`/`.obj`, and an ONNX graph verified against the
checkpoint (max |torch − onnx| < 0.05 m).

**Verified locally without a GPU:** 113 tests pass; a real GeoTIFF round trip
keeps its CRS and transform pixel-identically; the FastAPI service survives
path-traversal probes; the viewer was driven in headless Chrome against a real
`infer.predict` output directory and read back the correct GSD, shape, product
and live metrics.

| Run | Datasets | Epochs | RMSE (m) | balanced | tall bias | flat bias | landscape spread | Notes |
|---|---|---|---|---|---|---|---|---|
| _pending_ | GAMUS + SynRS3D (+ india_unlabeled) | 26 | — | — | — | — | — | 1× H100, ~5–6 h |

