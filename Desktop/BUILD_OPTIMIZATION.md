# Build and memory optimization measurements

Measured on Modal container `ta-01M436VK4JPABGHCHD442QE3FS`, 2026-10-04, starting from GitHub commit `28ce8b8` on `main`. Linux x64, Node 22.18.0, Python 3.12.6, CPU torch 2.6.0, ONNX Runtime 1.22.1. Windows/macOS native builds were not run; use the existing release matrix to verify those platforms.

| Measurement | Before (MiB) | After (MiB) | Reduction |
| --- | ---: | ---: | ---: |
| Linux AppImage (without model) | 541.0 | 422.2 | 22.0% |
| Installed Linux app (without model) | 1432.8 | 1184.9 | 17.3% |
| Frozen inference runtime | 1031.3 | 861.9 | 16.4% |
| Frontend build output (includes maps in baseline) | 134.3 | 118.9 | 11.5% |
| Frontend build peak RSS | 2059.9 | 1874.0 | 9.0% |
| 32 MP windowed buffer benchmark peak RSS | 734.0 | 469.4 | 36.0% |

## Changes

- Collect the ONNX code and native providers through normal PyInstaller analysis/hooks instead of collecting the entire packages and their test fixtures/tooling.
- Remove upstream PyTorch native test programs after freezing. Preserve tensor libraries, provider libraries and the shared-memory helper. CPU torch remains because the existing tiling/preprocessing engine uses it.
- Package complete sample `.dwproj` archives and their advertised input/height previews. Omit loose duplicate rasters and intermediate results from installers. The two selected offline projects and advertised previews were compared byte-for-byte inside the actual ASAR.
- Generate source maps only with `DW_BUILD_SOURCEMAPS=1`. They were already excluded from installers; this reduces build output and peak build RAM.
- Transfer the isolated terrain height copy to the worker instead of structured-cloning it again; skip cancelled builds before dispatch.
- Close CPU ImageBitmaps when their GPU textures are disposed, and release the probe bitmap if resizing fails. Keep live bitmaps for context restoration.
- Divide blend accumulators in place. Copy only decimated overview pixels so windowed inference releases old full-resolution row bands. Construct ONNX class logits directly in float32 instead of allocating an int64 one-hot cube and conversion intermediates.

No weights, model precision, input preprocessing, tiling parameters, output formats or rendering-quality settings were changed.

## Verification

- Production frontend build, frontend lint and all three desktop integration tests pass.
- Image bitmap lifecycle tests, bundled project tests and existing terrain geometry tests pass.
- Engine, input/windowing and ONNX-adapter focused tests pass, including uncertainty and class-clamping checks.
- A new weak-reference regression test checks that older row-band buffers are freed. It fails on the original engine and passes on the optimized engine.
- Frozen inference smoke tests pass with both the deterministic graph and the actual exported V5 model at `v5_final_forest/desktop_onnx_v0.1.2/external-v0.1.4/depthwizard.onnx`.
- Packaged Electron starts under Xvfb without a model. With the actual V5 graph, it passes image upload, local inference, result loading and terrain rendering. Headless tests run as root using a test-only `--no-sandbox` launcher; production sandbox preferences are unchanged.
- The 32 MP benchmark uses an 8192 x 4096 synthetic raster, 512-row bands, 128-pixel overview and deterministic predictor. It isolates row-buffer retention, not real-model compute or total model inference memory. Height, segmentation and uncertainty pixel hashes and aggregate statistics match exactly. The measured RAM savings are workload-specific.
- Broader frontend tests: 199 pass, with two existing failures and one failed suite caused by missing `public/samples/synthetic-city` fixtures. Those failures reproduce on the original checkout.
- Broader selected Python tests have an existing `test_export_matches_the_checkpoint` mismatch; it also reproduces on the original checkout. These training/export issues were not changed.

## Artifacts and reproduction

All implementation and builds ran on Modal. Persistent working tree: `/mnt/depthwizard-results/build-optimization/repo`. Measurements, benchmark script, logs and patch: `/mnt/depthwizard-results/build-optimization`.

Optimized installer: `repo/Desktop/linux/release/without-model/DepthWizard-0.1.4-linux-x86_64-without-model.AppImage`. No model weights are embedded; it supports installing a local model or the existing hosted inference path.

The standalone `benchmark-windowed.py` in the results directory compares the saved original engine (`engine-before.py`) with this working tree. Run it with the same CPU Python environment and `before` / `after` arguments. `sizes.json`, the two `windowed-*.json` files and `/usr/bin/time -v` outputs contain raw measurements.

Build with the documented dependencies: `npm ci` in Desktop and Frontend, install the Python requirements and CPU torch, run `python shared/build-backend.py`, then `DW_BUNDLE_MODEL=0 npm run desktop:linux`. For the model-included variant, stage the same verified model and keep `DW_BUNDLE_MODEL=1`.

## Offline sample selection

Offline installers now bundle only `PNG-Wankhede_Stadium_Mumbai` and `GeoReferenced .tif From Cartosat2S`, including their complete project archives and input/height previews. Other sample folders are excluded from desktop packaging, including preview-only folders. Website source samples are unchanged. The desktop server generates its sample index from the packaged folders, avoiding a stale index containing removed projects.

The actual packaged ASAR contains exactly two projects and no other sample directories. A live Electron check confirms the two sample IDs and HTTP 200 responses for both project downloads and all four previews. `two-samples-package.json` and `two-samples-live-index.json` capture those checks.
