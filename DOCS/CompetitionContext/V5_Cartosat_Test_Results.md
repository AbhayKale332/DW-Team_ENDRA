# Cartosat frozen-checkpoint evaluation

## Scope and interpretation

All 67 prepared, georeferenced Cartosat crops completed baseline inference on Modal L4 using `v5_final_forest/best.pt` (epoch 7). No training or checkpoint changes were made.

There is no scene DSM ground truth. These results measure pipeline behavior and coarse reference agreement; they do not measure building or tree height accuracy.

Full-resolution nDSM, absolute DSM, inferred DTM, segmentation and height-bin spread are retained on Modal. DSMs use Copernicus GLO-30 as their elevation anchor, so agreement with that same reference is self-consistency. Copernicus is a surface model that includes buildings and vegetation, not a bare-earth terrain survey. [Official Copernicus data description](https://registry.opendata.aws/copernicus-dem/).

SRTM and NASADEM were obtained through ten regional OpenTopography API requests covering five geographic areas. Both references covered all crops. NASADEM reprocesses SRTM observations from February 2000, so SRTM and NASADEM are not independent of each other, and historical land-cover differences can affect comparisons. [OpenTopography API](https://opentopography.org/developers), [NASADEM description and survey dates](https://portal.opentopography.org/raster?jobId=rt1729712823394).

## Coarse DEM agreement

Values below are cell-count-weighted RMSE in metres over input-aligned blocks of approximately 30 m, including partial boundary blocks. They are a judge proxy, not scores on a specified official reference grid. Ground-height and tall-object bias cannot be validated without independent scene truth.

| Inputs | Crops | SRTM RMSE | NASADEM RMSE | Copernicus self-consistency RMSE |
|---|---:|---:|---:|---:|
| Cartosat-3 RGB | 22 | 2.942 | 2.197 | 0.024 |
| Cartosat-3 PAN | 12 | 3.824 | 2.791 | 0.348 |
| Cartosat-2E | 11 | 3.511 | 2.650 | 0.252 |
| Cartosat-2S | 13 | 2.783 | 2.711 | 0.195 |
| Cartosat-2E hills | 9 | 3.849 | 3.100 | 0.119 |

## Protocol

- Default fused mean readout; no TTA, sharpening or changes to loss weights.
- Input GSD comes from each GeoTIFF: RGB Cartosat-3 approximately 0.45 m, PAN 0.28 m, Cartosat-2 approximately 0.60 m.
- Four-band stacks use the existing RGB band selection; PAN is repeated across the three input channels.
- SRTM/NASADEM heights are converted from EGM96 to EGM2008 using locally cached PROJ geoid grids. All comparisons required successful datum conversion.
- Missing reference support is excluded from scores. Unknown or unconvertible vertical datums are reported as unavailable.
- Pixel and coarse-cell RMSE, MAE, signed bias, correlation, and median residual over model-predicted ground are recorded per crop. Predicted-ground residuals are not ground-truth ground bias.
- Height percentiles and the fraction predicted above 20 m use the overview grid, at most 1024 pixels on the long side. Full-resolution min/max/mean and valid pixel counts are recorded separately.
- Inference processing across the 67 crops took about 12 minutes; this excludes model loading, CPU preparation, uploads, and packaging.

## Artifacts

- Portable gallery: `Model_Traning/v5/outputs/v5/cartosat_benchmark/modal/index.html`.
- Per-crop measurements, provenance and output grid audits: `Model_Traning/v5/outputs/v5/cartosat_benchmark/modal/metrics.json`.
- Persistent full-resolution products: `depthwizard-results:/cartosat_benchmark/scenes/<crop>/` in Modal environment `gpu`.
- Input dataset: [Abhay-Kale/C2-C3-Test](https://huggingface.co/datasets/Abhay-Kale/C2-C3-Test). The existing repository is public; its visibility was preserved.

## What to do next

Review the gallery for obvious domain failures, especially PAN, clouds, and bare hill terrain. Do not interpret small Copernicus errors as evidence of accurate building heights: anchoring enforces that agreement. Independent building/forest height labels or a scene DSM are needed before claiming a Cartosat tall-object bias or choosing a training fix from these scenes.

The earlier paired validation probe remains the evidence for head-readout choices: it did not justify changing the default fusion or adding sharpening.

## Checks

All 67 output sets passed audits for native dimensions, original input transform and horizontal CRS, GSD, and EGM2008 DSM/DTM tags. Local DEM, calibration, and scoring checks passed (22 tests); one geoid-network test was skipped locally. The actual Modal comparisons successfully converted both regional references for every crop using cached geoid grids. `git diff --check` passed. Nothing was committed.

## Reproduction

Deploy `Model_Traning/V4_modal/cartosat_modal.py` in Modal environment `gpu`. Run `prep`, then `benchmark`, then `package` from app `dw-cartosat`. The named secrets are `dw-cartosat-hf`, `dw-hf`, and `dw-opentopo`; credentials are not in code or reports. Full source products are included in the Hugging Face upload; inference covers the 67 prepared crops, not the multi-gigabyte original mosaics.

CPU prep call: `fc-01M3YN1RJBW32V7XQ94NK1Y16G`. GPU inference call: `fc-01M3YN6FRTY68P1RAB92XD3QQE`. Audit/package call: `fc-01M3YNYW66QW34J387QHCE4CP9`.

## Input upload verification

The final Hugging Face revision is `2d5a346ca8d468a0de3f7bd868d196a2f6b5b69e`. All 201 Cartosat source/supporting files (17,539,483,238 bytes, approximately 17.54 GB) were verified by repository path and byte size. This includes 65 files from the RGB Cartosat-3 root, 27 PAN crop/supporting files, and 109 Cartosat-2 files. The unrelated old `mvs3dm_results/` and `mvs3dm_results.zip` are excluded from the current dataset revision. `upload_verification.json` records the verification.

A representative eight-crop contact sheet is saved as `Model_Traning/v5/outputs/v5/cartosat_benchmark/modal/CONTACT_SHEET.png`; the complete gallery covers all 67 crops.
