# Validation examples

Prepared on 2026-10-03 using Modal shell `ta-01M40Y06K6143RHDGADD4Y3VMS`, environment `gpu`, and the NEON data API. Seven examples: three GAMUS nDSM ground-truth pairs, two NEON LiDAR DSM/nDSM ground-truth pairs, and two Cartosat DSM loading demos. Optical inputs are 1024 × 1024; NEON DSM/DTM references retain their native 1 m grid (512 × 512 over the same extent). Source paths and label provenance are in `manifest.json`.

## nDSM: independent dataset references

Three GAMUS validation tiles: `DC_02_26`, `DC_04_23`, and `DC_04_27`, under `ndsm_ground_truth/`.

1. Open the tile's `input_rgb.png` in DepthWizard.
2. Set **Resolution (m / pixel)** to **0.25**, as recorded in the source dataset, and run estimation.
3. If the result is in absolute elevation mode, switch **Info → Product → Above ground (nDSM)** before comparing.
4. Open **Validation**, choose **Height above ground (nDSM)**, and load the same tile's `reference_ndsm_m.npy`.
5. Inspect RMSE, MAE, bias, the error layer, and compare swipe. Use **Export report** to save the metrics.

The reference is a 2D float32 array in metres. Source-invalid pixels were converted to NaN so Validation excludes them. Packed float16 source precision is retained; exporting as float32 does not restore lost precision. PNGs have no map coordinates, so these references use matching extent and orientation.

## DSM and nDSM: independent NEON LiDAR references

Two forest validation tiles under `dsm_ground_truth/`:

- `neon_HARV_2024_724000_4697000_r0_c0`
- `neon_BART_2024_319000_4873000_r0_c0`

The RGB tiles were exported from the existing Modal NEON validation store. Matching 2024-08 LiDAR DSM and DTM tiles were downloaded from NEON and cropped to exactly the RGB extent. Reference elevations use **NAVD88 (Geoid12A realization)**, as documented in the [NEON Elevation–LiDAR product](https://data.neonscience.org/data-products/DP3.30024.001).

| File | Purpose |
|---|---|
| `input_rgb.tif` | Georeferenced RGB input at 0.5 m/pixel; use this for inference |
| `input_rgb.png` | Same pixels as a preview, without map coordinates |
| `reference_dsm_m.tif` | Independent LiDAR surface elevation, 1 m/pixel |
| `ground_dtm_m.tif` | Matching LiDAR bare-ground elevation for anchoring |
| `reference_ndsm_m.tif` | Independent height above ground: LiDAR DSM − LiDAR DTM |
| `reference_packed_chm_m.npy` | Original packed canopy-height label, including its validity mask; not the same label as DSM − DTM |
| `provenance.json` | Source product, flight month, filenames, checksums, crop and datum |

### Test DSM accuracy

1. Open **`input_rgb.tif`**, keep **Resolution → Auto** (0.5 m/pixel), and run estimation.
2. In **Info → Product**, set **Datum of a loaded DEM file → Not stated**, then click **Load DEM file…** and select the same folder's `ground_dtm_m.tif`. The interface currently has no NAVD88 option; this setting uses the published values without conversion. The actual datum is recorded in the provenance files.
3. Set **Structure already in the DEM → 0%**, because the DTM is bare ground. Choose **Elevation (DSM)** for the product.
4. Open **Validation**, select **Elevation (DSM)**, and load the same folder's **`reference_dsm_m.tif`**.
5. Inspect metrics, error layer and compare swipe, then export the report. The reference is independent LiDAR, so a perfect zero error is not expected.

DepthWizard anchors terrain on approximately 30 m cells even when the loaded DTM is finer. DSM error therefore includes terrain interpolation and model height error. Using the supplied DTM also makes this a height-model evaluation conditioned on known ground elevations; it is not an evaluation of independently predicted terrain.

### Test nDSM accuracy on the same tile

Switch **Info → Product → Above ground (nDSM)**, remove any previously loaded DSM reference from Validation, select **Height above ground (nDSM)**, and load **`reference_ndsm_m.tif`**. Coordinates align the native 1 m LiDAR reference to the prediction grid. Interpolation does not create additional LiDAR detail.

The alternate `reference_packed_chm_m.npy` contains the source dataset's canopy-height labels at the 0.5 m image grid, with invalid pixels set to NaN. That label excludes some buildings and wires; DSM − DTM measures all surface structures. Both references are useful, but their metrics answer different questions.

## DSM: precomputed model outputs for testing the feature

Two Cartosat-2S scenes under `dsm_model_output/`: `2e_buildings_village` and `2e_hill_forest_edge`.

**These are generated model outputs, not independent ground truth.** Use them to test DSM loading, alignment, comparison, and reporting. Do not report their comparison metrics as independently measured model accuracy.

Each folder includes:

| File | Purpose |
|---|---|
| `input_optical.tif` | Original georeferenced optical input, 0.6 m/pixel |
| `rgb.png` | Optical preview supplied with the model result |
| `dsm_m.tif` | Generated absolute surface elevation, metres, EGM2008 |
| `ndsm_m.tif` | Generated height above ground, metres |
| `dtm_m.tif` | Generated terrain used by that result; not an independent terrain survey |
| `meta.json` | Original model metadata; describes its nDSM output |
| `dsm_result.zip` | Prepared absolute DSM result bundle that opens without running inference |

### Quick DSM panel check without inference

1. Use **File → Open…** (Ctrl+O) and select `dsm_result.zip`.
2. Open **Validation** and choose **Elevation (DSM)**.
3. Load `dsm_m.tif` from the same folder.
4. Expect RMSE and MAE approximately **0 m**, bias approximately **0 m**, and **100%** within ±1 m/±2 m. The bundle and reference contain the same generated DSM: this is a deliberate check of the comparison workflow.

### Compare a new estimate

Open `input_optical.tif` and run inference. Switch to **Info → Product → Elevation (DSM)** after DEM anchoring, then load `dsm_m.tif` in Validation with **Elevation (DSM)** selected. Check the prediction's vertical datum against **EGM2008**. Different model versions or terrain sources can change the results.

Loading the DSM against an above-ground result only offers a single **Remove ground offset** adjustment. That adjustment cannot remove spatially varying terrain, especially in the hill scene; compare the same height type on both sides.

NEON source data are licensed under CC BY 4.0. Cite [Elevation–LiDAR (DP3.30024.001)](https://doi.org/10.48443/sxrt-ne87), [camera imagery (DP3.30010.001)](https://doi.org/10.48443/8vq2-s021), and [ecosystem structure (DP3.30015.001)](https://doi.org/10.48443/8qst-0w84) when using these examples. The API key and temporary download URLs are not stored in these files.

## Verify the files

With NumPy, Pillow, rasterio, and pyproj installed:

```sh
python3 validation-examples/verify.py
```

This checks image/reference sizes, valid heights, horizontal coordinates, matching extents at different resolutions, DSM = nDSM + DTM, and the ready-to-open Cartosat DSM bundles.
