# DepthWizard — frontend

Desktop installers for Windows, macOS and Linux: see [desktop build and release instructions](desktop/README.md).

Web application for **single-view height estimation and 3D flythrough** (SIH 2026, problem statement 26175, ISRO).

It does five things:

1. Takes an optical RGB image (PNG, JPG or GeoTIFF).
2. Sends it to the DepthWizard model, which runs as a private Hugging Face Space.
3. Drapes the image on the estimated surface model in an interactive 3D view.
4. Provides height, slope and profile analysis, and validation against a reference raster.
5. Exports the result as GeoTIFF, GLB, OBJ, PLY, STL, PNG or JPG.

## Quick start

```bash
npm install
cp .env.example .env        # then put your Hugging Face token in HF_TOKEN
npm run dev                 # http://localhost:5173
```

Two ways to try it without the model:

- **Samples:** open *Try a sample scene* in the viewport, or **File → Sample scenes**. These are precomputed and work offline.
- **Offline mock:** open `http://localhost:5173/?mock`. Every run uses an offline mock provider.

## Access to the private model Space

The Space is private, so every request must carry a Hugging Face token. The token never reaches the browser.

- **Where it is stored:** `HF_TOKEN` in `.env` (or several: `HF_TOKENS=a,b,c` / `HF_TOKEN_2`, `HF_TOKEN_3` …; when one fails — ZeroGPU quota used up, token rejected or rate-limited, or a run error — the server switches to the next and the app retries). It deliberately has no `VITE_` prefix; Vite would embed a `VITE_` variable in the JavaScript bundle.
- **How it is used:** the browser only talks to its own origin at `/hf-space/*`. The server forwards those requests to the Space and adds `Authorization: Bearer $HF_TOKEN`.
  - In development and `npm run preview`, the server is Vite's proxy (`vite.config.ts`).
  - In production, it is `server/serve.mjs` (Node ≥ 20, no dependencies). It serves `dist/` and proxies the same path.
- **Vercel:** `vercel.json` routes `/hf-space/*` to a server function that uses the same server-only `HF_TOKEN`. Configure `HF_TOKEN` and, if needed, `VITE_SPACE_ID` or `HF_SPACE_URL` in the Vercel project settings, then redeploy. Do not use a `VITE_` prefix for the token. Vercel Functions cap request bodies at 4.5 MB and limit execution duration by plan; for larger images or longer inference, deploy with `npm run serve` on a Node host instead.
- **After editing `.env`:** restart `npm run dev` (or `npm run serve`). The header status dot shows whether the model is reachable. **File → Settings → Test connection** re-checks it.

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | Hugging Face token with read access to the Space. **Required**, server-side only. |
| `HF_TOKENS`, `HF_TOKEN_2` … | Extra tokens from other accounts, used in turn when a token fails (quota, rejected token, run error). |
| `HF_TOKEN_COOLDOWN_MIN` | How long an exhausted token is skipped when the Space gives no reset time (default 60). |
| `VITE_SPACE_ID` | `owner/space` of the model; the proxy target is derived from it. |
| `HF_SPACE_URL` | Optional explicit target, for example `https://owner-space.hf.space`. |
| `PORT` | Port for `npm run serve` (default 8080). |

## Scripts

| Command | What it does |
|---|---|
| `npm run dev` | Development server with the authenticating proxy |
| `npm run build` | Type-check and production build into `dist/` |
| `npm run serve` | Production server: `dist/` plus the authenticating proxy |
| `npm test` | Unit tests (Vitest): parsers, metrics, mesh/UV contract, picking, georeferencing |
| `npm run e2e` | End-to-end tests (Playwright, offline mock provider, axe accessibility scan) |
| `npm run make-sample` | Regenerate the procedural *River valley* sample |

Docker: `docker build -t depthwizard . && docker run -e HF_TOKEN=hf_… -p 8080:8080 depthwizard`

## Architecture

| Layer | Choice |
|---|---|
| App | React 19, TypeScript (strict), Vite |
| UI kit | Mantine 9. Sharp theme with 2 px radii; light by default, with a dark theme |
| 3D | three.js through React Three Fiber and drei |
| Terrain shading | A GPU material (`three-custom-shader-material`) that draws every layer: optical, height tint, colormap, hillshade, slope, contours, reference, error, compare swipe |
| Heavy work | Web Workers via comlink. The mesh is built off the main thread, including vertical walls for buildings |
| State | zustand |
| Charts | Apache ECharts (modular imports) |
| Geo | geotiff.js (read and write), proj4 |
| Persistence | fflate zip files for `.dwproj` projects; IndexedDB for Recent |

```
src/
  api/        provider interface · Gradio Space provider (REST via proxy) · offline mock · typed errors
  domain/     normalised Scene / Reference / Metrics types
  lib/        npy, GeoTIFF, georeferencing, heights, metrics, picking, exporters, projects
  workers/    terrain mesh builder
  features/
    shell/       header, File/View/Tools/Help menus, status bar, shortcuts
    input/       project panel (image, resolution, TTA, run)
    processing/  run orchestration, progress stepper, error recovery
    viewport/    canvas, cameras (orbit / map / walk / flight / tour), overlays
    analysis/    probe, measure, profile, histogram
    validation/  reference alignment, metrics, charts, report
    inspector/   Layers · Analysis · Validation · Info
    files/       open / save / recent / export
    help/        settings, shortcuts, model info, docs
server/     production server and proxy
scripts/    sample generator
```

### Coordinate and projection contract

Grid pixel `(col, row)` maps to scene coordinates as follows:

- `x = (col − (W−1)/2)·gsd` (east)
- `z = (row − (H−1)/2)·gsd` (south)
- `y = h − base` (up, metres)

Texture coordinates are pixel-centre exact: `uv = ((col+0.5)/W, (row+0.5)/H)`, with textures uploaded as `flipY = false` (the glTF convention). This is the same contract as the backend's `viz/mesh.py`, and it is covered by unit tests.

### What the deployed model returns

The Space returns **height above ground (nDSM)** as float32 metres, plus previews and a textured mesh.

- **Plain images:** shown as an **rDSM**.
- **GeoTIFFs:** the app reads the georeferencing in the browser and restores it on the (possibly downsampled) height grid. The result is a georeferenced **nDSM**, and GeoTIFF exports open in the right place in QGIS.
- **Absolute DSM** (DEM calibration) needs a backend that provides it. The provider interface already has a capability flag for it.
