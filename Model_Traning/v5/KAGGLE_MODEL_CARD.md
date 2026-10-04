# DepthWizard v5 — satellite RGB to height above ground

DepthWizard v5 estimates a **normalised digital surface model (nDSM), in metres above ground**, from a single optical RGB image at a known ground sample distance (GSD). It predicts surface heights such as buildings and vegetation; it does not directly predict terrain elevation or an absolute elevation above sea level.

This release is the **`forest-final` PyTorch checkpoint**, selected at epoch 7 of the `v5_final_forest` fine-tune on a Modal H100 on 29 September 2026. It was warm-started from `resume-v4-1.6`, with emphasis on forested and sparse landscapes. The project supports geospatial reconstruction and 3D visualisation for SIH 2026, problem statement 26175 (ISRO).

- **Model code:** [DepthWizard repository](https://github.com/AbhayKale332/DepthWizard), directory `Model_Traning/v5`.
- **Download handle:** `abhaydkale232/depthwizard-v5/pyTorch/forest-final/1`.
- **Release evidence:** the bundled `RESULTS.md`, `config.json` and `preproc.json` describe this checkpoint. They take precedence over historical runbooks and default settings in the repository.

## Release contents

| File | Purpose |
|---|---|
| `best.pt` | Selected PyTorch inference checkpoint, approximately 2.50 GB (decimal). Load with the v5 architecture and checkpoint loader. |
| `config.json` | Saved architecture, data mixture, loss weights and training settings. Paths refer to the training machine. |
| `preproc.json` | Input preprocessing contract, including normalisation, canonical GSD and tile size. |
| `RESULTS.md` | Run summary, per-epoch validation metrics and evaluation caveats. |

This version contains these four files. It does **not** include an ONNX graph, full-state training resume checkpoint, evaluation datasets or a held-out test report. `best.pt` is the selected model; `last_full.pt`, mentioned in the results, is a separate training artifact outside this release.

## Reported validation results

The table below reproduces the **epoch 7 validation results** from the bundled `RESULTS.md`. RMSE is in metres; lower is better.

| Dataset / measure | RMSE (m) |
|---|---:|
| Checkpoint selection score: forested + sparse, NEON and MVS3DM | **2.481** |
| MVS3DM, overall | 1.545 |
| MVS3DM, forested / sparse | 1.95 / 0.86 |
| NEON, overall | 3.444 |
| NEON, forested / sparse | 4.11 / 0.68 |
| GAMUS, overall | 2.893 |
| GAMUS, urban | 3.26 |
| US3D, overall | 3.872 |

**These are validation scores from the sets used during model development, not held-out test accuracy.** The selection score pools forested and sparse errors within each source by pixel count, computes that source's RMSE, then averages the source scores. NEON is a coarse-label source, with pooled-resolution evaluation used for selection when available. The selection score is not an overall RMSE across all datasets.

GAMUS was evaluated on **200 validation tiles at corrected 0.25 m source GSD**. Older runs used different sampling or GSD, so their headline numbers are not directly comparable. Held-out comparisons between the starting and final checkpoints, a visual validation report, and ONNX export were skipped in this run.

The reported MVS3DM / NEON gradient ratios were **0.23 / 0.66**. Height RMSE alone does not establish that fine structure or canopy detail is faithfully reconstructed.

## Architecture and training

The saved configuration specifies:

- **Encoder:** DINOv3 ViT-L/16, `facebook/dinov3-vitl16-pretrain-sat493m`, with features from blocks 6, 12, 18 and 24.
- **Decoder:** DPT-style trunk, width 256, with a full-resolution RGB detail branch of width 64.
- **Prediction heads:** metric height regression, adaptive height bins and auxiliary semantic segmentation; a learned gate fuses the height predictions.
- **Height bins:** 96 bins over 0–120 m. The training validity ceiling is 150 m; this does not establish reliable accuracy throughout that range.
- **Training stores:** MVS3DM, NEON, GAMUS, US3D and SynRS3D (`g05` and `g1`). DFC23 and `india_labeled` are excluded from this final mixture. NEON uses 1 m coarse-label supervision.
- **Fine-tune:** H100, bfloat16, batch 16 with gradient accumulation 2, all encoder blocks trainable, layer-wise learning-rate decay 0.9 and seed 42. The training budget was 106 minutes across two sessions; the run summary reports approximately 108 minutes to completion.

This is a fine-tuned height model, not a generic relative-depth estimator. Known GSD and the saved preprocessing recipe are essential to its metric interpretation.

## Input contract

Use the v5 inference pipeline to reproduce training preprocessing:

1. Read optical imagery in **RGB channel order** and determine GSD in metres per pixel. GeoTIFFs supply georeferencing; PNG/JPEG inputs require a reliable supplied GSD.
2. Apply **per-scene, per-channel 2nd–98th percentile stretch**, using valid pixels.
3. Resample to **0.5 m per pixel**. Heights remain in metres.
4. Process **512 × 512** tiles; scene inference uses overlapping tiles and Hann blending.
5. Convert to float32 CHW, divide byte values by 255, and normalise with **mean `[0.430, 0.411, 0.296]`** and **std `[0.213, 0.156, 0.143]`**.

Do not replace these with ImageNet statistics or resize an arbitrary image to 512 × 512 while ignoring its physical scale. `preproc.json` has `version: "v3"`; that identifies the inherited preprocessing contract, not the model release version.

## Download and run

Download version 1 with KaggleHub:

```python
from pathlib import Path
import kagglehub

model_dir = Path(kagglehub.model_download(
    "abhaydkale232/depthwizard-v5/pyTorch/forest-final/1"
))
print(model_dir / "best.pt")
```

Install `kagglehub` first (`pip install kagglehub`). Access to this model requires permission and Kaggle authentication where applicable.

The checkpoint requires the project's custom model code; it is not a standalone TorchScript module or a Hugging Face `AutoModel` checkpoint. See the variation's **Usage** section for environment setup, scene inference, direct Python loading and optional ONNX export.

## Outputs and absolute elevation

The model exposes `fused` (nDSM), `seg` (semantic logits) and `b_std` (spread of the adaptive-bin head), alongside training-oriented outputs. Semantic IDs are: **0 other, 1 ground, 2 low vegetation, 3 building, 4 water, 5 road, 6 tree**. Segmentation is an auxiliary prediction; its accuracy is not established by the height metrics above.

The scene pipeline can write height arrays, visual previews and georeferenced products. The bin-head spread is an uncertainty indicator in metres, **not a calibrated confidence interval or a guaranteed error bound**.

Absolute DSM generation is a separate geospatial operation that requires a georeferenced image and an external elevation reference. The v5 default `dem_anchored` mode constrains coarse-cell means to the reference DEM and adds model detail at finer scales. SRTM and Copernicus products contain surface contributions; simply adding nDSM to them can double-count buildings and vegetation. Keep the reference and output vertical datums consistent.

**Agreement with an anchor DEM is partly imposed by this calibration.** Scoring against the same DEM is not independent evidence of fine-resolution height accuracy.

## Limitations and intended use

Use this release for research, prototyping, geospatial visualisation and evaluation against independent height references. Validate it on the intended sensor, landscape and GSD before relying on its measurements.

- This release reports no held-out test results or Cartosat/India-specific accuracy.
- Shadows, occlusion, sensor differences, spectral/radiometric shifts and incorrect GSD can affect predictions. Trees and tall structures remain challenging.
- Landscape categories used for sampling and evaluation are heuristics, not independently annotated scene classes.
- A single RGB view cannot resolve all terrain and object geometry. Fine structures may be smoothed or missed even when aggregate RMSE is acceptable.
- Absolute elevations depend on the external DEM, its coverage, resolution and vertical datum.

## Licensing and attribution

The Kaggle variation is currently labelled **MIT**, and the repository code has an MIT license. The **DINOv3 backbone has separate Meta DINOv3 license terms**, which must also be reviewed when using or redistributing a model that incorporates it. The Kaggle label does not replace those terms or the source datasets' conditions.

Training data sources retain their own licenses and attribution requirements. Consult the source cards and original publications for [GAMUS](https://huggingface.co/datasets/earthflow/GAMUS), [SynRS3D](https://huggingface.co/datasets/JTRNEO/SynRS3D), NEON, MVS3DM and US3D before reproducing training or redistributing data. External DEM products also have their own attribution requirements.

For bug reports, include the versioned Kaggle handle, repository revision, saved preprocessing contract, input sensor/GSD and inference options through the [project issue tracker](https://github.com/AbhayKale332/DepthWizard/issues).
