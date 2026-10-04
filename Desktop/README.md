# DepthWizard desktop

Electron wraps the existing React app. The launcher serves the built UI on a loopback address, starts a bundled V5 Python/ONNX inference process when a model is available, and keeps all project data, model imports and generated results in the OS application data directory. The renderer has no Node access. End users need neither Python nor Node. Model-included installers select local CPU inference automatically and work offline. Model-free installers select hosted Hugging Face inference automatically and require internet; users can import a compatible ONNX model later.

Targets: Windows 10/11 x64 (`.exe` installer, primary), macOS Intel (`x64`) and Apple Silicon (`arm64`, separate `.dmg`/`.zip`), Linux x64 (`.AppImage`). Builds must use the target OS and architecture because the inference runtime contains native Python libraries. The workflow builds Linux on Ubuntu 22.04; test on other distributions before promising support. CPU inference is the baseline; GPU acceleration is not configured.

## Source layout

- `shared/`: Electron launcher, model installation, local API, Python inference runtime, tests, shared icon and staged resources.
- `mac/`: macOS packaging configuration; Intel and Apple Silicon installers go into `mac/release/<variant>/`.
- `windows/`: Windows packaging configuration, NSIS installer options and Windows icon; installers go into `windows/release/<variant>/`.
- `linux/`: Linux packaging configuration; AppImages go into `linux/release/<variant>/`.
- `package.json` and `electron-builder.config.cjs`: desktop dependencies, commands and shared packaging settings.

The React UI and web server remain in `Frontend`. Desktop builds compile that UI and bundle its `dist` and server modules. Shared runtime code is maintained once for all platforms.

## Develop on this machine with the temporary model

Use Node 22.12+ and the existing project Python environment:

```bash
cd Desktop
npm ci
npm ci --prefix ../Frontend
npm run desktop:install
DW_PYTHON="$(realpath ../.venv/bin/python)" \
DW_ONNX_MODEL="$(realpath ../best.onyx/v5_probe_v4init_onnx/depthwizard.onnx)" \
npm run desktop:dev
```

The Python environment needs the inference dependencies listed below. The supplied temporary model is used by path; it is not committed or silently included in release builds. `DW_USER_DATA=/tmp/depthwizard-dev` can isolate development project/settings data from the installed application.

## Build locally

Install Node 22.12+ and Python 3.11 on each target OS. From `Desktop`:

```bash
npm ci
npm ci --prefix ../Frontend
npm run desktop:install
python -m venv .desktop-venv
```

Activate that environment (`.desktop-venv\Scripts\Activate.ps1` on Windows PowerShell; `source .desktop-venv/bin/activate` on macOS/Linux), then:

```bash
python -m pip install -r shared/requirements.txt
```

Install CPU torch, retained by the existing tiling/TTA code:

```bash
# Windows / Linux
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
# Apple Silicon
python -m pip install torch==2.6.0
# Intel macOS (last supported official Intel wheel)
python -m pip install torch==2.2.2
```

NumPy is pinned to 1.26.4 to keep the Intel macOS torch bridge compatible. ONNX Runtime, SciPy, Pillow and pyproj are also pinned to versions with official Python 3.11 Intel macOS wheels. Then:

```bash
npm run desktop:test
python shared/inference-smoke.py
npm run desktop:backend
DW_BUNDLE_MODEL=0 npm run desktop:win    # run on Windows, using Bash
DW_BUNDLE_MODEL=0 npm run desktop:mac    # run on the matching Mac architecture
DW_BUNDLE_MODEL=0 npm run desktop:linux  # run on Linux
```

Output goes to `Desktop/windows/release/<variant>/`, `Desktop/mac/release/<variant>/`, or `Desktop/linux/release/<variant>/` on the corresponding build machine. `npm run desktop:pack` creates an unpacked app for inspection. Run `npm run desktop:smoke` to test source startup after building the UI; on a headless Linux machine use `xvfb-run -a npm run desktop:smoke`. Set `DW_DESKTOP_EXECUTABLE` to test an unpacked executable instead. The smoke test uses isolated temporary application data.

The backend build deliberately excludes model training dependencies. It still ships CPU torch for the existing inference engine, so application downloads are substantial even without weights. `beforePack` fails if the native inference executable is missing instead of producing an incomplete installer.

## Supply / replace the model

The format is **ONNX**, with extension `.onnx` (not `.onyx`). Use a compatible DepthWizard V5 export with input `image` (normalised float32 NCHW tiles) and ordered outputs `height_m`, `seg`, and optionally `height_std_m`.

Keep the complete export together:

```text
depthwizard.onnx
depthwizard.onnx.json
depthwizard.onnx.data   # required if the graph stores external weights
```

The `.json` must contain the original `preproc` contract and `external_data` list. Do not rename files independently; the graph references its external weight filenames. A different network architecture/output contract requires an adapter, not just renaming its file.

- **Separate model release asset:** ZIP the files at the archive root and upload the ZIP to GitHub Releases. Users extract it and choose **Model → Install ONNX model…**. The app copies the files into its application data folder. This lets you release the final model later without rebuilding every installer. Replacing a model stops active inference. Imported model folders remain available until removed manually.
- **Model included in installer:** put the files in `shared/resources/model/` before building, or set `DW_MODEL_DIR` to the export directory when running `desktop:win` / `desktop:mac` / `desktop:linux`. The builder copies this folder into application resources and selects local inference on first launch. This is the default build variant and requires a staged model. Set `DW_BUNDLE_MODEL=0` to create the separate `without-model` variant. An imported model takes precedence over bundled weights and selects local inference automatically. Explicit inference choices remain saved across restarts.
- **Development only:** `DW_ONNX_MODEL` selects a graph by absolute path and `DW_PYTHON` selects the Python executable; neither is needed by installed users.

For your temporary model on Linux:

```bash
DW_MODEL_DIR="$(realpath ../best.onyx/v5_probe_v4init_onnx)" npm run desktop:linux
```

Do not add large weights to Git. Their `.data` sidecars are just as necessary as the small `.onnx` graph. The app remains usable for existing projects and bundled samples without a model. Without local weights, new predictions use the hosted Hugging Face service. It never substitutes a mock prediction automatically.

## GitHub Releases

Commit these changes and push them to GitHub. Open **Actions → Build desktop releases → Run workflow** to produce downloadable artifacts for all four OS/architecture combinations. Manual runs do not publish a release.

To build and upload a **draft** GitHub release:

```bash
git tag desktop-v0.1.3
git push origin desktop-v0.1.3
```

Use a new semantic version for each release. The workflow sets the application version from the tag, builds and smoke-tests each native inference runtime and both packaged variants (`with-model` and `without-model`), uploads installers and `SHA256SUMS.txt` to a draft release. Review the installers before publishing the draft. Existing releases can have their assets replaced when rerunning the same tag workflow.

The workflow finds the newest published `DepthWizard-V5-*-model*.zip` asset and verifies its GitHub SHA-256 digest before staging. It authenticates GitHub downloads while this repository is private, without forwarding the token to the asset CDN. Optional repository **Actions variables** `DESKTOP_MODEL_URL` and `DESKTOP_MODEL_SHA256`, or manual workflow inputs, override that source. The ZIP must have one graph and all its sidecars at its root. A missing or invalid model fails the build rather than producing an incomplete model-included installer. Large sample projects excluded from Git are not downloaded automatically; committed samples are included. To distribute additional sample data offline, add it to `Frontend/public/samples` before building.

Signing is optional for test builds. Configure per-platform **Actions secrets** for public distribution:

| Secret | Purpose |
|---|---|
| `DESKTOP_WIN_CSC_LINK`, `DESKTOP_WIN_CSC_KEY_PASSWORD` | Windows signing certificate and password |
| `DESKTOP_MAC_CSC_LINK`, `DESKTOP_MAC_CSC_KEY_PASSWORD` | Apple Developer ID certificate and password |
| `DESKTOP_APPLE_ID`, `DESKTOP_APPLE_APP_SPECIFIC_PASSWORD`, `DESKTOP_APPLE_TEAM_ID` | macOS notarization |

No credentials or `.env` files are packaged. Auto-update is not enabled; installers and model bundles can be downloaded separately from GitHub Releases.

## Offline scope and troubleshooting

The desktop runtime uses one tile per batch and at most four CPU threads to limit memory use and thread contention on laptops. Large images and TTA can still take several minutes with this model; CPU timing is not equivalent to the hosted GPU service. `DW_BATCH_TILES` can override the batch size for profiling on larger machines.

Model-included installations use local ONNX Runtime on CPU. Model-free installations use hosted inference until a model is imported. Rendering, measurements, validation, local project files and exports run locally. Basemaps and automatic remote DEM anchoring start disabled in desktop mode; live map/OSM services still need internet when enabled. Load a local DEM or use project-bundled geographic data for offline work. The desktop relay allows the existing restricted Hugging Face inference routes and blocks remote Overpass proxy requests; direct external map/OSM sources remain online features.

**Model → Open application data** contains `inference.log`, generated `jobs/`, imported `models/`, model selection and the saved UI port. The port stays stable so browser project storage persists across app restarts. If another program occupies that port, startup reports an error; free the port instead of deleting it and losing access to the old browser storage origin. **Model → Restart inference service** reloads the selected model. Close the app before manually removing job/model folders.

## Download page

`Docs-Site` exposes `/download/`, which reads published desktop releases through `/api/releases`. It selects installers by OS, architecture and model variant; no release URLs need manual updates. `/api/download` redirects allowed published assets to GitHub’s signed CDN URLs, without proxying installer bytes. While the repository is private, set `GITHUB_RELEASES_TOKEN` in the hosting provider’s server environment to a GitHub token with repository read access. Keep it server-side. After the repository becomes public, this optional variable can be removed and the release list uses public GitHub download URLs. Draft releases are never listed.
