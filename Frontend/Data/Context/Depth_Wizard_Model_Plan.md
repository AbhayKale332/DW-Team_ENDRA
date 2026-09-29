## This is how i am building the my ML model + Some processing things

# DepthWizard — Plan (SIH 2026, PS 26175, ISRO)

## Context

**The ask.** Single-view RGB → DSM, two modes (non-georeferenced → rDSM, GeoTIFF → absolute metric DSM), plus an interactive 3D flythrough shipped as a standalone app. Scored **50% DSM accuracy** (RMSE/MAE/correlation vs LiDAR, *stability across urban / sparse / hilly / forested*) and **50% rendering + UX** (projection accuracy, navigability, stability, standalone deployment).

**Where you are.** `Notebooks/depth-wiz.ipynb` has working IM2ELEVATION (IMELE) inference on Kaggle. `FineTunning/` is a clean versioned Kaggle-training scaffold but `v1/config.py` is still the stock Llama/alpaca LoRA template — nothing depth-related. No local GPU (Intel Iris Xe, 15 GB RAM). Draft idea deck exists.

**Constraints (from you).** Idea PPT in <2 weeks (~20 Sep 2026), finals ~Dec 2026. Team of 6 but only 1–2 can do ML. Compute: Kaggle T4×2 (30 h/wk), Colab, Lightning AI free credits. Viewer: React + Three.js, reusing GeoLibre code and design language.

---

## Brutally honest verdict on the original idea

Viable, but three parts of it are actively working against you:

1. **The framing is wrong, and it's the PS's own framing.** "Depth Anything gives relative depth → rescale to metric with SRTM" will lose on RMSE. In a nadir remote-sensing image the GSD is *known*, so height in metres is directly learnable — there is no scale ambiguity to resolve. Any team that regresses **metric nDSM** end-to-end beats any team that regresses relative depth and fits a global affine to it. GAMUS labels are already metric nDSM. Use DA-V2 as a *pretrained encoder*, never as the relative-depth *output*.

2. **The ensemble is redundant.** A fine-tuned DA-V2 and IM2ELEVATION are the same function class on the same input; their errors correlate hard. Realistic gain from fusing them: low single-digit % RMSE, at 2–3× inference cost and a third training loop. Worse, IMELE's public checkpoint is Dublin-only at 0.5 m/px and will be strictly weaker than a fine-tuned modern backbone — naively fusing a much weaker expert *hurts*. With 1–2 ML people and 30 GPU-h/week you cannot afford it.

3. **Two factual errors in the deck that a SAC/ISRO judge will catch.** IM2ELEVATION does *not* "detect building footprints, measure shadow length, and use sun elevation angle" — it is a plain SENet-154 encoder–decoder with skip fusion; there is no explicit shadow geometry anywhere in it. And "CNN = fine detail / Transformer = global context" is a slide story, not a mechanism: a DPT decoder already fuses multi-scale features. Fix both before submission.

Also: **DDPM/diffusion is out.** 10–100× slower inference and far more training compute than you have. And your current effort split is ~90% model / 10% viewer, against a 50/50 rubric.

**What I'd do instead** is below. It keeps a genuine, defensible ensemble story — but built from experts that actually disagree.

---

## The core technical insight (this is the pitch)

The PS complains that *"foundational monocular depth models are trained largely on natural, egocentric imagery"* and therefore suffer a domain gap. **Do not paper over that gap with fine-tuning — remove it at the root by changing the foundation model.**

Meta released **DINOv3 ViT-L/16 pretrained on SAT-493M**: 493M Maxar ortho satellite images at 0.6 m GSD (`facebook/dinov3-vitl16-pretrain-sat493m`, 300M params — verified to exist). It is a satellite-native foundation encoder designed for frozen-backbone dense prediction. Nobody else in this hackathon will be using it.

Second insight: **decompose the output correctly.**

```
DSM_absolute(x,y) = DTM_coarse(x,y)   +   nDSM_predicted(x,y)
                    ↑ from Copernicus GLO-30 /   ↑ metric, learned from
                      SRTM / CartoDEM via          RGB — this is where
                      the GeoTIFF transform        the ML lives
```

Terrain comes from geometry and public DEMs (that is what handles *hilly*); structure height comes from the network (that is what handles *urban/forested*). Nobody needs to "calibrate relative depth to metres" — that step disappears. This decomposition is the single highest-leverage decision in the plan.

Third: **GSD conditioning.** Metres-per-pixel is what makes the output metric. GAMUS is 0.33 m; ISRO will hand you something else at final evaluation (Cartosat-3 ≈ 0.25–1.1 m). Canonicalise: resample every input to a fixed working GSD (0.5 m) using the GeoTIFF transform, predict, resample back. Train with random GSD jitter (0.25–2.0 m) so it degrades gracefully. Most teams will skip this and their metres will simply be wrong on ISRO's imagery.

---

## Model architecture

One network, `DepthWizardNet`, trained once:

```
RGB tile (512×512 @ 0.5 m canonical GSD)
   │
   ├─► DINOv3 ViT-L/16 SAT-493M encoder   [frozen; optionally last 4 blocks unfrozen late]
   │     multi-scale tokens from layers {5, 11, 17, 23}
   │
   └─► DPT decoder (reassemble + fusion blocks, trainable)
         │
         ├─ Head A: metric nDSM regression        → SiLog + L1 + gradient/normal loss
         ├─ Head B: adaptive-bin classifier       → AdaBins/HTC-DC style softmax·bin-centres
         └─ Head C: 6-class semantic segmentation → CE  (GAMUS labels; free supervision)
                     │
                     └─► used at inference as the ground/road mask for DEM alignment
         │
         └─ learned per-pixel gate α combines A and B  →  final nDSM (metres)
```

**Why each piece earns its place:**

- **Frozen DINOv3-SAT encoder** — satellite-domain features, and freezing drops the trainable set to ~30M params so ViT-L fits comfortably on a T4 at batch 8 with AMP. Meta's own results show frozen DINOv3 + light heads are SOTA on dense prediction incl. canopy height. This is what makes ViT-L feasible on free compute.
- **Head A + Head B is the real ensemble.** Height distributions are severely long-tailed (the GAMUS paper documents this; TSE-Net's whole contribution is fighting it). Pure regression systematically *underestimates tall buildings*, which is exactly what inflates RMSE. A bin-classification head fixes the tail; a gated combination of the two is a principled, honest "learned fusion of complementary experts" — one training run, one forward pass.
- **Head C is free.** GAMUS ships pixel-wise semantic labels; the paper's premise is that height and land cover are strongly correlated, so it regularises. And it produces the ground mask the calibration module needs. Also gives you **per-landcover RMSE** — which is literally the rubric's "stability across urban, sparse, hilly, forested."
- **TTA at inference:** 8× dihedral (flip + rot90) + 2-scale. Nadir imagery is rotation-invariant, so this is legitimately free accuracy — typically 3–8% RMSE, zero training cost. *This* is your cheap ensemble.

**The second expert, if and only if time remains (gated):** DA-V2-Small DPT (Apache-2.0) fine-tuned on the same data. It has a genuinely different prior — egocentric depth pretraining vs satellite SSL — so its errors decorrelate, unlike DA-V2 + IMELE. Fuse only if it improves held-out val RMSE. **IM2ELEVATION stays in the results table as a published baseline you beat**, not in the pipeline. That table is worth more to your score than the fusion is.

> **Licensing (check before finals — an ISRO deliverable matters).** DA-V2 **Small is Apache-2.0; Base/Large/Giant are CC-BY-NC-4.0** (non-commercial). DINOv3 ships under Meta's DINOv3 license, which permits commercial use with conditions. GeoLibre is MIT. Use DINOv3-SAT + DA-V2-**Small** only, and put a licence table in the docs.

---

## Data strategy

| Dataset | Role | Specs | Why |
|---|---|---|---|
| **GAMUS** (`earthflow/GAMUS`) | Primary train/val | 11,507 tiles, 1024², **0.33 m**, 5 US cities, nDSM + 6-class semantics; 6,304/1,059/4,144 split | Mentor-recommended; real, metric, large, has the semantic labels Head C needs |
| **SynRS3D** (`JTRNEO/SynRS3D`) | Pretraining | 69,667 synthetic tiles, **GSD 0.05–1 m**, exact nDSM, 8 land-cover classes, 6 city styles | Covers ISRO's likely GSD range and gives *free multi-GSD supervision*; NeurIPS'24 spotlight |
| **GeoNRW** | Landscape generalisation | ~7,783 tiles, 1 m, Germany, RGB + DEM + seg | **GAMUS is 100% flat US urban.** This is your only real source of *hilly and forested* — the exact axis the rubric grades stability on |
| DFC2019 / US3D | Optional extra | 2,783 tiles, 1024² | JAX/OMA; partially overlaps GAMUS |
| ISPRS Vaihingen/Potsdam | Optional eval-only | 0.05–0.09 m | Cross-resolution generalisation evidence |

**Recipe:** pretrain on SynRS3D (cheap, synthetic, GSD-diverse) → fine-tune on GAMUS + GeoNRW jointly at canonical 0.5 m → evaluate held-out, reporting global *and per-landcover* metrics.

**Domain-gap insurance — no longer optional, and now built (v4).** We went looking for an open Indian RGB + per-pixel-height dataset. **There is not one.** DFC2023 Track 2 *does* include a New Delhi city (2 m nDSM from Gaofen-7 / WorldView stereo) but sits behind an IEEE DataPort login with no API; Google Open Buildings 2.5D and UT-GLOBUS give building heights only, with no paired RGB and at ≥4 m; Bhoonidhi/CartoDEM is 10–30 m — too coarse for structure height, but exactly right for the *terrain* term. SAC's own reference repo (`IMG-PROCESS-SAC/SIH2026`) is a README with no data.

So `FineTunning/v4/dwdata/india.py` ships the two paths that actually work:

* **`india_labeled`** — a local directory of paired rasters (`<stem>_rgb.*` + `<stem>_ndsm.*`), which ingests DFC2023's New Delhi tiles once someone downloads them by hand, or any stereo/LiDAR product the team obtains. Split by *scene*, not by tile.
* **`india_unlabeled`** — RGB-only Indian tiles feeding a **mean-teacher** branch in `train.py`: an EMA teacher predicts a weakly-augmented view, the student is pulled towards it on a strongly-augmented one, and pixels are gated by the teacher's own bin-distribution sigma so it cannot chase its own hallucinations. This is the TSE-Net-style self-training below, implemented, and it is the only lever available without labels.

14 Indian AOIs spanning urban / sparse / hilly / forested are defined. **No basemap URL is defaulted** — which imagery a government deliverable may train on is the team's licence decision, not a script's; Bhuvan is the obvious choice for an ISRO deliverable.

**Say this out loud in the deck.** Unlabeled adaptation narrows the domain gap; it is *not* evidence of Indian accuracy. Only ISRO's own imagery measures that. The auto-generated validation report states it in its own "what this does not establish" section, because a panel that catches an overclaim discounts everything else you said.

---

## Scale calibration module (a graded milestone — do not treat it as glue)

**Georeferenced GeoTIFF path** (`rasterio`, already installed):
1. Read CRS + affine transform → compute true GSD → resample to 0.5 m working grid.
2. Tile → predict nDSM (metric) + semantic mask → stitch with cosine-weighted overlap blending (your IMELE notebook already has the tiling/stitching logic to reuse).
3. Fetch terrain: **Copernicus DEM GLO-30** from the public AWS bucket `s3://copernicus-dem-30m/` (COGs, 1°×1° tiles, no auth) — with SRTM 30 m and ISRO **CartoDEM** (Bhuvan) as selectable sources. *Mentioning CartoDEM to ISRO judges is worth doing.*
4. Reproject the DEM to the image grid → **estimate the DTM by fitting a smooth surface only over ground pixels.** This matters, and v3 got it wrong: GLO-30 and SRTM are *surface* models, so v3's `--dem` path (`dem + ndsm`) counted every building twice — a 30 m tower over 12 m terrain came out near 165 m instead of 135 m, and it did so silently, because the result still looks like a plausible elevation raster.

   `v4/geo/calibrate.py` fixes it: the ground mask requires **both** Head C's ground/road/water classes **and** a low predicted nDSM, so a wrong semantic class id (they are unverified — see below) cannot poison the fit; the surface is a robust order-2 polynomial trend plus a NaN-aware smoothed residual, fitted only on masked pixels. Measured on a synthetic scene with known terrain: DTM recovered to 0.4 m mean error, DSM at the tower 134.4 m against a true 135.0 m.
5. Optional GCP refinement: RANSAC-fit scale + offset on supplied control points.
6. `DSM = DTM_fitted + nDSM_pred` → write GeoTIFF with CRS, transform, nodata, and a sidecar JSON of provenance + metrics.

**Non-georeferenced PNG/JPG path:** output rDSM normalised to [0,1] plus the raw metric prediction under an assumed GSD, with a GSD slider in the UI that live-rescales the mesh. Be explicit in the docs that this is assumption-conditioned — judges reward honesty about it.

**Validation report** (auto-generated per run, built in `v4/viz/report_html.py`): RMSE, MAE, Pearson r, δ-thresholds, a building-balanced RMSE (RMSE averaged over height strata — the metric that exposes tall-building underestimation), and **a per-landscape breakdown across urban / sparse / hilly / forested**. That last one is the rubric's own stability axis and *nothing in v1–v3 measured it*; the nearest thing was a per-landcover table whose class ids are provably wrong. `v4/eval/landscape.py` derives the label per tile from the ground-truth height field (relief, tall-pixel fraction, canopy roughness), so it needs no annotation and works on any dataset — heuristics with published thresholds, and the descriptors ship next to every result so the assignment can be audited.

The report is one self-contained HTML file with every figure inlined as a data URI: it emails, prints to PDF, and opens on an air-gapped machine. It directly answers the "validate against reference datasets" deliverable.

---

## Visualization (50% of the score — staff it accordingly)

**Base: fork GeoLibre** (`github.com/opengeos/GeoLibre`, **MIT**). Tauri v2 + React + TypeScript + MapLibre GL JS + deck.gl, already shipping slope / aspect / hillshade / curvature / viewshed as browser WASM, already builds native Windows/macOS/Linux binaries, already runs fully offline.

That fork hands you, for near-zero cost: the standalone-deployment requirement, offline/air-gapped operation (ISRO cares), a mature design language, and several of the PS's own analysis tools. Keep its attribution and MIT notice.

**What you add — two synchronized panes:**

1. **GIS pane** (MapLibre + deck.gl, GeoLibre-native): georeferenced DSM as colormap/hillshade overlay, reference-DSM difference layer, error heatmap, contours, GeoTIFF export.
2. **Flythrough pane** (React Three Fiber / Three.js — new): the graded centrepiece.
   - DSM heightmap → tiled `BufferGeometry` with distance-based LOD; original RGB draped as texture (this *is* "projection accuracy" — get the UV↔geo mapping exactly right and say so).
   - Cameras: **first-person** (pointer-lock WASD + look), orbit, and a cinematic drone path for the demo video.
   - Analysis in-scene: click-to-read height, two-point slope, elevation profile line chart, vertical exaggeration slider, colormap toggle, sun/shadow control.
   - Side-by-side predicted vs reference with a live metrics panel.

**Backend:** FastAPI + ONNX Runtime (export the model to ONNX; it makes the desktop bundle self-contained and CPU-capable for judges without a GPU). Ship both a `docker compose` for the web demo and a Tauri installer for standalone.

**Team split (matches your 1–2 ML / 6 total):** 2 on ML+calibration, 3 on the GeoLibre fork + Three.js viewer + backend, 1 on docs/deck/demo video.

---

## Phasing

**Phase 0 — Spike, now → ~13 Sep (goal: real numbers in the deck).**
- Rewrite `FineTunning/v1/{config,train}.py` from the Llama template to the depth task; keep `main.py`/`install_dependency.py` as-is (they work).
- GAMUS loader; DINOv3-SAT frozen encoder + DPT head + Head A only; ~2–3 h on Kaggle → **first GAMUS val RMSE**.
- Run existing IMELE notebook on the same val split → baseline row.
- Load one predicted DSM into a bare Three.js scene → one screenshot.

**Phase 1 — Idea deck, → 20 Sep.** Rewrite around the DINOv3-SAT + DTM-decomposition + GSD-canonicalisation story. Delete the shadow-geometry and CNN-vs-Transformer claims. Include: the real Phase-0 RMSE vs the IMELE baseline, the architecture diagram, the GeoLibre-fork screenshot, and a licence/compute feasibility slide.

**Phase 2 — Model, Oct. — built in `FineTunning/v4/`, not yet trained.** One mixed-sampler stage (the sequential SynRS3D pretrain was actively harmful in v2 and is gone); Heads A + B + C with gated fusion; TTA; per-height-stratum, per-class **and per-landscape** eval; ONNX export verified against the checkpoint; GeoTIFF I/O and the Copernicus GLO-30 / SRTM / CartoDEM fetcher with the ground-masked DTM fit; FastAPI backend. Remaining: run it on the H100 and fill in `CompetitionContext/Outcome.md`.

**Phase 3 — Integration, Nov. — viewer and service built in `FineTunning/v4/`.** Standalone dual-pane Three.js app (vendored three.js r128, no build step, opens from a USB stick): map pane with height / hillshade / optical / reference / error layers and contours, flythrough pane with the optical image draped, orbit + pointer-lock first-person + cinematic drone path, sun and shadow control, vertical exaggeration, click-to-probe height, two-point slope, elevation profile, and live RMSE/MAE/bias/r against a reference surface. FastAPI upload→DSM→viewer service on the same origin, running the *same* `predict_scene` the evaluation runs. Auto-generated figures and validation report. Remaining: Tauri + Docker packaging, and the gated extras (DA-V2-Small second expert).

**A note on effort split.** The plan warned that the split was ~90 % model / 10 % viewer against a 50/50 rubric. v4 closes that: roughly a third of the new code is the viewer, the service and the report generator.

**Phase 4 — Dec.** Robustness (huge images, odd CRSs, cloud/nodata), docs, demo video, dry runs on unseen imagery.

## Repo layout

As built, in `FineTunning/v4/`:

```
FineTunning/v4/
  config.py prepare_data.py train.py main.py run_lightning.sh
  dwdata/    preprocess.py (THE contract: normalisation, GSD, tiling — shared by
             train and inference), augment.py, packed.py, dataset.py, loaders.py,
             india.py (labeled + unlabeled Indian ingest, AOI table)
  models/    encoder.py (DINOv3-SAT, structural block discovery, LLRD),
             dpt.py, heads.py (A/B/C + gated fusion + b_std), losses.py, ema.py, tta.py
  geo/       dem.py (COP30 / SRTM / CartoDEM / local), calibrate.py (ground-masked
             DTM fit + GCP RANSAC)
  eval/      metrics.py (global + per-class + per-stratum + per-landscape),
             landscape.py, sliding.py, report.py
  infer/     engine.py (the ONE inference path), predict.py (CLI), export_onnx.py
  viz/       figures.py, report_html.py, mesh.py (glTF + OBJ)
  serve/     app.py (FastAPI: upload -> DSM -> viewer) + static/
  viewer/    index.html + app.js + vendor/three.min.js  (standalone, no build)
  tests/     113 offline tests, no GPU and no network
```

Earlier versions are kept as-is: `v1/` (Phase-0 Kaggle spike), `v2/` (the run that
crashed at epoch 21), `v3/` (the rewrite whose post-mortem drove v4).

## Verification

- **Unit:** GSD round-trip (resample→predict→resample back preserves geometry); GeoTIFF CRS/transform preserved on write; tile-stitch seam continuity < 0.05 m; DEM reprojection alignment on a known tile.
- **Model:** held-out GAMUS test — RMSE, MAE, Pearson r, δ₁; **per-landcover and per-height-stratum RMSE**; cross-dataset zero-shot on GeoNRW and Vaihingen (proves the stability claim).
- **Baselines to beat and publish:** IMELE checkpoint, a plain DA-V2-Small fine-tune, and your own regression-head-only ablation. The ablation table is deck material.
- **End-to-end:** upload a held-out GeoTIFF with a known LiDAR DSM → auto validation report → mesh loads and is navigable at ≥30 fps → exported GeoTIFF opens correctly in QGIS.
- **Deployment:** Tauri build runs on a clean offline machine; `docker compose up` serves the web demo; ONNX CPU inference completes on a laptop.

## Risks

| Risk | Mitigation |
|---|---|
| GAMUS is 80 GB and all-flat-US-urban | Stream/subset via HF; GeoNRW + SynRS3D supply hills, forest, GSD diversity |
| ISRO's evaluation imagery is a different GSD/sensor/geography | GSD canonicalisation + jitter training; optional self-training on Indian imagery |
| ViT-L too slow on free T4s | Encoder frozen by default (~30M trainable); ViT-B/16 LVD fallback; Lightning credits for the final run |
| GLO-30/SRTM are surface models → double-counted buildings | Fit the DTM only on Head C's ground/road pixels |
| Viewer under-delivered (it's half the score) | 3 of 6 people on it from Phase 2; GeoLibre fork removes the packaging risk entirely |
| Licence challenge on a government deliverable | DINOv3-SAT + DA-V2-**Small** + MIT three.js/GeoLibre only; ship a licence table (v4 README §8) |
| **No open Indian height labels exist** — so no Indian accuracy number can be produced before the finals | Unlabeled mean-teacher adaptation (built); `india_labeled` ingest ready for DFC2023 New Delhi or any product the team obtains; and *say so in the deck* rather than implying coverage we do not have |
| A surface DEM silently double-counts buildings and the output still looks plausible | Fixed in `v4/geo/calibrate.py` and pinned by tests; the reported `double_count_avoided_m` makes the effect visible per scene |
| Class ids are unverified, so any per-landcover claim is unsafe | Neutral `classN` labels + a measured histogram everywhere; the DEM calibrator requires a height test as well as a class test, so a wrong id is harmless there |
