# Stand-out features — what DepthWizard lacks, ranked

Checked 2026-10-02. Primary sources only; every claim about the codebase cites a
repo path, every external claim links its owner. Companions: `V5_Research_Additions.md`
(datum facts, DEM-anchoring maths, judge format), `FAQs.md` (host answers),
`ProblemStatment.json`. Where this file repeats a V5 fact it points back rather
than re-deriving it.

---

## 0. What the judges reward (recap)

| | |
|---|---|
| Accuracy, 50 % | RMSE / MAE / r of the **absolute DSM vs SRTM or Copernicus 30 m** on Cartosat-2S 0.6 m GeoTIFFs; "stability across urban, sparse, hilly, and forested" (`ProblemStatment.json`; `FAQs.md` Evaluation 1–2). |
| Visualization, 50 % | Projection accuracy, visual fidelity, navigability, UI, **software stability, standalone deployment** (`ProblemStatment.json`). |
| Explicit asks | Scale calibration from "scene-level statistics, low-resolution DEMs, **semantic priors**, or minimal GCPs"; an "**immersive**" navigable environment; "**validate** estimated height values against reference datasets"; Theme: **Disaster Management** (`ProblemStatment.json`). |

A feature stands out if it moves one of these and no other team is likely to have it.

## 1. Already implemented — checked, not proposed

Read directly in the code. Nothing below is recommended again.

| Area | Where |
|---|---|
| nDSM / rDSM / absolute DSM products, honest product labels | `Frontend/src/lib/product.ts`, `Frontend/src/domain/types.ts` |
| DEM anchoring without double-counting (Tobler block-mean fusion), DTM split, DSM/nDSM/DTM GeoTIFF export | `Frontend/src/lib/dem/anchor.ts`, `Frontend/src/features/files/exports.ts` (`geotiff`, `geotiff-ndsm`, `geotiff-dtm`) |
| GCPs: elevation (offset/plane) and lat/lon (affine georeference) | `Frontend/src/lib/gcp.ts`; Python `Model_Traning/v5/infer/predict.py` (lon/lat/elev + datum) |
| Datum conversion, COP30 default, compound vertical CRS — **Python only** | `Model_Traning/v5/geo/dem.py` (`copernicus30`, `geoid_undulation`, `to_datum`), `infer/predict.py` (`_geo_crs`, `VERTICAL_DATUM` tag) |
| Per-pixel uncertainty σ — **written, not shown** | `models/heads.py` (`std`), `infer/export_onnx.py` (`height_std_m`), `infer/predict.py` (`ndsm_std_m.npy/.tif`, `meta.json["uncertainty"]`) |
| NRSC product reader (BAND1..4 + `BAND_META`, 3,2,1 order, NoData collar, 20k² row-band reads), PAN+MX Brovey pan-sharpening — **Python only** | `Model_Traning/v5/dwdata/scene_io.py`, `tools/cartosat_to_rgb.py` |
| Shadow-based, label-free scale check from sun angles — **Python only** | `Model_Traning/v5/viz/shadow.py` |
| Sliding-window tiling with seam blending, TTA | `Model_Traning/v5/infer/engine.py`, `eval/sliding.py`, `models/tta.py` |
| ONNX export with parity check (3.96 MB graph + 1.3 GB external weights) | `infer/export_onnx.py`, `best.onyx/v5_probe_v4init_onnx/`, `Docs-Site/src/content/docs/system/onnx.mdx` |
| Self-hosted FastAPI service + static viewer, judge-proxy and landscape eval | `serve/app.py`, `eval/judge_proxy.py`, `eval/landscape.py` |
| 3D viewer: orbit / map / walk / flight / tour cameras; layers optical, height tint, colormaps, hillshade, slope, contours, reference, error, compare swipe; vertical building walls | `Frontend/src/features/viewport/cameras/*`, `scene/terrainMaterial.ts`, `workers/terrainMesh.ts` |
| Probe, point-to-point measure, profile (CSV), histogram | `Frontend/src/lib/analysis.ts` |
| Validation vs any uploaded reference (CRS-aware alignment), strata metrics, HTML report | `Frontend/src/lib/reference.ts`, `features/validation/ValidationTab.tsx`, `features/validation/report.ts` |
| Buildings (footprint + median/p95 height + area), trees, water as 3D objects | `Frontend/src/lib/objects.ts`, `domain/types.ts:58`, `scene/Objects.tsx` |
| Flood: connected bathtub + LISFLOOD-FP-style inertial shallow-water simulation | `Frontend/src/lib/usecases/flood.ts`, `floodSim.ts` |
| Telecom coverage with ITU-R P.526 knife-edge diffraction (line of sight over the DSM) | `Frontend/src/lib/usecases/telecom.ts` |
| Cloud detection and fill; OSM overlay; POIs; basemap; `.dwproj` projects; mesh export GLB/OBJ/PLY/STL; 16-bit PNG / NPY | `lib/cloud.ts`, `lib/osm.ts`, `lib/poi.ts`, `lib/tiles.ts`, `lib/dwproj.ts`, `lib/exporters/*` |

**So the gaps are mostly "the Python pipeline can, the web app the judges click
cannot" — plus a few genuinely new capabilities.**

## 2. Ranked recommendations

Ranked by stand-out value per unit of effort. S ≈ ≤ 1 day, M ≈ 2–4 days, L ≈ a week+.

| # | Feature | Criterion it moves | Value | Effort |
|---|---|---|---|---|
| 1 | Datum-correct COP30 / SRTM anchoring in the web app + "score as the judges" panel | Accuracy | Very high | M |
| 2 | Drop-in NRSC Cartosat product (folder / zip) in the web app | Accuracy, projection accuracy | High | S–M |
| 3 | Uncertainty layer in the UI, with calibration evidence | Validation, honesty | High | S |
| 4 | Quota-free inference: self-hosted provider (+ in-browser WebGPU stretch) | Stability, standalone deployment | High | S–M (L) |
| 5 | 3D change detection between two dates (DSM of difference) | Disaster theme, uniqueness | High | M |
| 6 | Polygon area / volume / cut-fill tool | Structural analysis, disaster use | Medium–high | S |
| 7 | One-click flythrough video export | Visualization, demo | Medium | S |
| 8 | CityJSON LoD1 city model + building inventory export | Standard formats, urban planning | Medium | S |
| 9 | Semantic-prior height checks (OSM, Open Buildings 2.5D, Meta CHM) | Scale-calibration milestone, forest/urban stability | Medium | M |
| 10 | Rooftop solar potential from the DSM | Uniqueness, Indian relevance | Medium | M |
| 11 | WebXR immersive mode | "Immersive" visualization | Medium | S–M |
| 12 | Full-resolution terrain via RTIN / quadtree LOD | Visual fidelity, navigability | Medium | M–L |

### 2.1 Datum-correct COP30 / SRTM anchoring in the web app (#1)

**Gap.** The web app anchors only to AWS Terrain Tiles or a user file
(`Frontend/src/lib/dem/sources.ts`, `sourceId: 'terrain-tiles' | 'local-file'`), and
states "No geoid conversion is applied anywhere" (`sources.ts:6`,
`features/anchoring/ElevationReference.tsx:27`). Terrain Tiles are an SRTM/GMTED
mosaic, described by their owner as "30 meters (90 meters nominal quality)" SRTM on
land, with no stated vertical datum
([joerd data-sources](https://github.com/tilezen/joerd/blob/master/docs/data-sources.md)).
The Python pipeline already defaults to COP30 and converts datums
(`Model_Traning/v5/geo/dem.py`), so **the judged UI and the measured pipeline
disagree**. The Space returns only an nDSM; the absolute DSM is made in the browser
(`Frontend/README.md`, "What the deployed model returns").

**Why it stands out.** The judges score against SRTM *or* COP30 (`FAQs.md`).
COP30 is EGM2008 and a surface model "including buildings, infrastructure and
vegetation", LE90 < 4 m
([Copernicus DEM collection](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM));
SRTM is EGM96. The EGM2008 − EGM96 separation reaches −3.7 m (Shimla) and +4.7 m
(Gangtok) — a constant bias that adds straight into RMSE (`V5_Research_Additions.md` §2,
GeographicLib). A "Score as: SRTM (EGM96) / Copernicus (EGM2008)" switch is a visible,
judge-specific answer no generic team will have.

**Feasibility (measured 2026-10-02 with `curl -r 0-15 -H Origin:`).**
- COP30 tiles on AWS are COGs, no account needed
  ([registry](https://registry.opendata.aws/copernicus-dem/)); the bucket answered
  `206 Partial Content` but **sent no `Access-Control-Allow-Origin`**, so the browser
  cannot read it directly → relay it through the existing server, exactly as
  `Frontend/server/overpass.mjs` / `api/overpass.mjs` already relay Overpass.
- Geoid grids: PROJ's CDN serves `us_nga_egm96_15.tif` (2.6 MB) and
  `us_nga_egm08_25.tif` (76.9 MB) as COGs ([cdn.proj.org](https://cdn.proj.org/)); both
  returned `206` with `access-control-allow-origin: *` — the browser can range-read
  only the few tiles over a scene. geotiff.js fetches "only the necessary portions" of
  a COG and supports `window` reads
  ([geotiff.js README](https://github.com/geotiffjs/geotiff.js)).
- Native SRTM GL1 needs an OpenTopography key (`geo/dem.py` `OPENTOPO_URL`), which
  belongs server-side like `HF_TOKEN`.

**Build.** Add `copernicus30` and `srtmgl1` samplers + a geoid sampler to
`lib/dem/sources.ts`; an output-datum selector in `ElevationReference.tsx`; a `/dem`
relay in `server/serve.mjs` and `api/`; write the vertical datum into
`lib/exporters/raster.ts`. In `features/validation/ValidationTab.tsx`, a "Fetch SRTM /
COP30 as reference" button that also aggregates the prediction to 30 m cells and
reports per-pixel and per-cell RMSE/MAE/r per landscape (`lib/metrics.ts`).

**Caveat.** Scoring against the *same* DEM used as the anchor is ~0 per cell by
construction (found with `eval/judge_proxy.py`). The panel must anchor on one DEM
and score on the other, or label the result "consistency check".

### 2.2 Drop-in NRSC Cartosat product in the web app (#2)

**Gap.** The web app accepts one file (`features/input/ProjectPanel.tsx:52`
`multiple={false}`), reads it whole with `file.arrayBuffer()` (`lib/input.ts:47`)
under a 512 MB cap (`lib/input.ts:8`), and drapes bands `[0,1,2]`
(`lib/geotiff.ts:77`). The NRSC MERGED product is **four single-band 0.6 m files of
829 MB each** plus `BAND_META.txt`, band order B,G,R,NIR (`V5_Research_Additions.md` §1).
So a 4-band stack drapes as BGR, and the MERGED product cannot be opened at all. The
Python reader already handles all of this (`Model_Traning/v5/dwdata/scene_io.py`
docstring).

**Why it stands out.** It is the closest thing we have to the judges' input format
(inferred from the free Bhoonidhi sample; the FAQ only says "georeferenced RGB Images
(like tiff files)"). "Drop the raw NRSC folder, nothing else" is a strong live-demo
moment and removes a projection-accuracy failure mode.

**Feasibility.** Folder pick via `<input webkitdirectory>` is
[Baseline 2025](https://developer.mozilla.org/en-US/docs/Web/API/HTMLInputElement/webkitdirectory);
geotiff.js reads windows and overviews from a `Blob` without loading the file
([README](https://github.com/geotiffjs/geotiff.js)), which lifts the 512 MB cap.
`BAND_META` carries `SunAzimuthAtCenter` / `SunElevationAtCenter`
(`viz/shadow.py` docstring) → set the viewer sun (`store/view.ts` already has
`sunAzimuth`/`sunElevation`) so rendered shadows match the image, and show the
shadow-agreement scale check that `viz/shadow.py` computes
([Kadhim & Mourshed 2018](https://ieeexplore.ieee.org/abstract/document/8126256/)).

**Build.** `lib/input.ts`, `lib/geotiff.ts` (band map 3,2,1 for 4-band VNIR, NoData
collar → alpha), `features/files/openFile.ts`, `ProjectPanel.tsx`; send the product
zip to the Space. **Unverified:** whether the deployed Space's `/predict` passes a
zip to `SceneSource` — its code is not in this repo.

### 2.3 Uncertainty layer in the UI, with calibration evidence (#3)

**Gap.** σ is computed and written (§1), but the web app fetches only `ndsm_m.npy`,
`meta.json`, `seg.png`, `objects.json`
(`Frontend/src/api/gradio/GradioSpaceProvider.ts:199-213`). A `confidence` colormap
is defined (`theme/colormaps.ts:32`, `scene/terrainMaterial.ts:8`) but no layer uses
it. Nothing in `Model_Traning/v5/eval/` measures whether σ tracks error.

**Why it stands out.** It answers "validate estimated height values" on scenes with
no reference — the judges' own situation — and makes the known weak spots
(tall structures, canopy; `Docs-Site/src/content/docs/reference/limitations.mdx`)
visible instead of hidden. Per-pixel uncertainty for depth regression is established
([Kendall & Gal, NeurIPS 2017](https://arxiv.org/abs/1703.04977)); the standard
check is a sparsification curve with AUSE / AURG
([Poggi et al., CVPR 2020](https://arxiv.org/abs/2005.06209);
[code](https://github.com/mattpoggi/mono-uncertainty)).

**Build (S).** Fetch `ndsm_std_m.npy`; add a Confidence layer and "h ± σ" to the
probe; in the validation tab, "RMSE on confident pixels" and a sparsification plot.
Compute AUSE on the held-out stores in `Model_Traning/v5/eval_test.py` so the claim
on the slide is measured. σ also feeds #5's detection threshold.

### 2.4 Quota-free inference: self-hosted provider, in-browser stretch (#4)

**Gap.** Providers are `'gradio-space' | 'mock'` only (`Frontend/src/api/registry.ts`).
The live path depends on ZeroGPU quota, handled by rotating tokens
(`Frontend/README.md`, `server/tokens.mjs`). Wiring the self-hosted service is
roadmap item 5 (`Docs-Site/.../reference/limitations.mdx`); `serve/app.py` already
loads an ONNX backend (`load_backend(onnx=…)`).

**Why it stands out.** "Software stability" and "successful standalone deployment" are
scored; a quota error during judging costs points directly. The FAQ accepts a locally
hosted web app (`FAQs.md`, Execution 1).

**Feasibility.** (a) A `depthwizard-serve` provider calling `serve/app.py` — S–M.
(b) Stretch, in-browser: ONNX Runtime Web caps a single ArrayBuffer at ~2 GB and
wasm memory at 4 GB, and loads external weights via `externalData`
([ORT large models](https://onnxruntime.ai/docs/tutorials/web/large-models.html));
WebGPU ships by default in Chrome and Edge
([ORT WebGPU EP](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)). Our graph is
1.3 GB of external fp32 weights (`best.onyx/…/depthwizard.onnx.data`) — under the
limits on paper, but **unverified** in a browser; Depth Anything V2 runs in the
browser at ViT-S size ([onnx-community/depth-anything-v2-small](https://huggingface.co/onnx-community/depth-anything-v2-small)),
so a distilled or fp16 student is the realistic route — L.

### 2.5 3D change detection between two dates (#5)

**Gap.** No bi-temporal feature (no "change" use case in
`Frontend/src/features/usecases/`).

**Why it stands out.** The problem's theme is Disaster Management
(`ProblemStatment.json`). Elevation change from optical pairs alone is a recognised
research task — MTBIT / SUNet on the 3DCD dataset produce a 2D change map *and* a
3D elevation-change map "without the need to rely directly on elevation data during the
inference step" ([Marsocci et al., ISPRS JPRS 196, 2023](https://arxiv.org/abs/2205.15903)).
Collapsed buildings, new construction, landslide scars and debris volume all fall out
of one DSM of difference.

**Build (M).** Load a second scene or `.dwproj`, align it with the existing CRS-aware
`lib/reference.ts`, difference the **nDSMs** (terrain anchoring cancels), threshold with
a minimum level of detection `t·√(σ₁² + σ₂²)` — the uncertainty-thresholded DoD of
[Wheaton et al., ESPL 2010](https://research.aber.ac.uk/en/publications/accounting-for-uncertainty-in-dems-from-repeat-topographic-survey/) — using σ from #3.
Report volume gained / lost and diff `objects.json` building polygons (new / removed /
lowered). The `diverging` colormap already exists (`theme/colormaps.ts`). New file
`lib/usecases/change.ts`, panel in `features/usecases/UseCasesTab.tsx`.

**Caveat.** Model RMSE is 3–5 m (`limitations.mdx`), so only multi-storey changes are
reliable; say so on screen.

### 2.6 Polygon area / volume / cut-fill tool (#6)

**Gap.** `measure()` is point-to-point only (`Frontend/src/lib/analysis.ts:52`);
"volume" appears only in the flood simulator.

**Why.** The problem asks for "analysis of structural heights"; volume is the number a
disaster or mining officer actually needs (debris, stockpile, reservoir, building
volume). QGIS's `native:rastersurfacevolume` defines the method set to mirror — count
only above / below a base level, or add / subtract volume below it — and reports
volume, area and pixel count
([QGIS source](https://api.qgis.org/api/qgsalgorithmrastersurfacevolume_8cpp_source.html)).

**Build (S).** Lasso polygon in `features/analysis/ToolPalette.tsx` / `store/tool.ts`;
base = user level or a plane through the polygon rim; compute in
`workers/analysis.worker.ts` on `scene.heights` at the true GSD.

### 2.7 One-click flythrough video export (#7)

**Gap.** No recorder (no `MediaRecorder` / `captureStream` in `Frontend/src`); the
team records demos with an external tool (`DOCS/demo-recording-prompt.md`). The Tour
camera exists (`features/viewport/cameras/TourRig.tsx`).

**Why.** Visualization is half the score; a "Record tour → WebM" button turns every
judge's run into a shareable flythrough. `canvas.captureStream()` is Baseline since
January 2020
([MDN](https://developer.mozilla.org/en-US/docs/Web/API/HTMLCanvasElement/captureStream));
`MediaRecorder` writes it to a Blob, Baseline since April 2021
([MDN](https://developer.mozilla.org/en-US/docs/Web/API/MediaRecorder)).

**Build (S).** `features/files/exports.ts` (new kind), `TourRig.tsx` start/stop hook.

### 2.8 CityJSON LoD1 city model + building inventory (#8)

**Gap.** Buildings already carry footprint, median and p95 height and area
(`domain/types.ts:58-66`), but exports are mesh or raster only
(`features/files/exports.ts` `ExportKind`); no city-model format.

**Why.** "A standard geospatial format" is required; a semantic 3D city model is what
urban-planning and disaster-exposure users load. CityJSON 2.0 implements a subset of
the OGC CityGML 3.0 data model, encodes LoD1 buildings as `"Solid"` with `"lod": "1"`,
stores integer vertices with a `transform`, and asks for a 3D CRS in `referenceSystem`
([CityJSON 2.0.1](https://www.cityjson.org/specs/2.0.1/)) — a JSON writer, no
dependencies. Add an inventory CSV / GeoJSON: storeys ≈ h / 3 m (OSM's default floor
height, [Key:building:levels](https://wiki.openstreetmap.org/wiki/Key:building:levels)),
footprint and gross floor area.

**Build (S).** New `lib/exporters/cityjson.ts` (base = DTM under the footprint,
top = `h`), wire into `exports.ts` and `features/shell/MenuBar.tsx`.

### 2.9 Semantic-prior height checks (#9)

**Gap.** The OSM overlay already reads `building:levels` (`lib/hoverInfo.ts:150`) but
only displays it. "Semantic priors" is one of the four calibration routes the problem
names (`ProblemStatment.json`).

**Why.** Gives a label-free, per-scene metric-scale check on Indian imagery, and
evidence for the forested and urban stability axes.
- OSM `height` / `building:levels × 3 m` vs predicted building `h` → robust scale ratio
  and bias, optionally applied as a scale on the nDSM only (as GCPs do).
- [Open Buildings 2.5D Temporal](https://sites.research.google/gr/open-buildings/temporal/):
  building heights, effective 4 m, 2016–2023, covers India, CC BY 4.0 / ODbL; its 1.5 m
  height MAE was measured only in North America, Europe and Japan.
- [Meta / WRI 1 m canopy height](https://registry.opendata.aws/dataforgood-fb-forests/):
  global CHM COGs, CC BY 4.0 — a canopy reference that matches the user's
  tree-inclusive goal (DFC23 labels trees 0 m).

**Build (M).** `lib/osm.ts` + `lib/objects.ts` matching, a "Semantic check" card in
`ValidationTab.tsx`; Open Buildings / CHM fetched server-side.
**Unverified:** OSM height-tag coverage in Indian cities; CORS on the CHM bucket; Open
Buildings needs Earth Engine or GCS, not a browser fetch.

### 2.10 Rooftop solar potential (#10)

**Gap.** Nothing solar in the code. The pieces exist: building footprints, a sun
direction in the viewer (`store/view.ts`), and DSM shadow casting in Python
(`viz/shadow.py` `cast_shadows`).

**Why.** Turns the DSM into a national-priority answer: PM-Surya Ghar funds rooftop
solar for one crore households with ₹75,021 crore
([PIB, Cabinet approval](https://www.pib.gov.in/PressReleaseIframePage.aspx?PRID=2010130&reg=48&lang=2)).
Height is what decides roof shading; a 2D roof map cannot do this.

**Build (M).** Sun positions over a day / year from the NREL SPA (±0.0003°,
[Reda & Andreas 2004](https://doi.org/10.1016/j.solener.2003.12.003)); per-roof-pixel
sunlit hours with DSM shadowing in a worker; optional clear-sky irradiance following
GRASS `r.sun` (beam, diffuse, reflected, "shadowing effect of the local topography")
([r.sun manual](https://grass.osgeo.org/grass-stable/manuals/r.sun.html)). Lead with
sun-hours and a roof ranking; kWh needs an irradiance model and is secondary. New
`lib/usecases/solar.ts`, panel in `UseCasesTab.tsx`.

### 2.11 WebXR immersive mode (#11)

**Gap.** No XR (no `xr` / `VRButton` usage in `Frontend/src`). The problem asks for an
"immersive" environment with "first-person navigation".

**Feasibility (S–M).** three.js needs `renderer.xr.enabled = true`, a `VRButton` and
`setAnimationLoop`
([three.js manual](https://threejs.org/docs/#manual/en/introduction/How-to-create-VR-content));
`@react-three/xr` 6.6.31 declares peers `@react-three/fiber >=8`, `react >=18`
(npm registry, checked 2026-10-02), so it fits the app's fiber 9 / React 19
(`Frontend/package.json`); it ships `createXRStore`, `<XR>` and a teleport tutorial
([pmndrs/xr](https://github.com/pmndrs/xr)). WebXR is a W3C Candidate Recommendation
Draft ([WebXR Device API](https://www.w3.org/TR/webxr/)). Map the Walk rig to teleport.
Value depends on a headset at judging; the WebXR emulator covers the demo.

### 2.12 Full-resolution terrain via RTIN / quadtree LOD (#12)

**Gap.** The mesh is a regular grid decimated to a vertex budget
(`workers/terrainMesh.ts:16`), and the display image is capped at 8192 px
(`lib/geotiff.ts:67`) — a 20k² Cartosat scene loses ~2.5× before decimation.
Vertical walls are done; RTIN, proposed in `V5_Research_Additions.md` §3.9, is not.

**Feasibility (M–L).** Martini builds error-bounded RTIN meshes in milliseconds from
(2ᵏ+1)² grids, ISC licence
([mapbox/martini](https://github.com/mapbox/martini); npm `@mapbox/martini` 0.2.0).
Chunk the scene into 257² tiles with screen-space error per tile: sharp roof edges
where they matter, flat ground nearly free.

## 3. Cheap hygiene (not ranked as stand-out)

- **COG output.** The backend writes tiled DEFLATE GeoTIFF without overviews
  (`infer/predict.py:244-246`); GDAL's COG driver adds overviews and puts IFDs first so
  one 16 KB range request reads them ([GDAL COG](https://gdal.org/en/stable/drivers/raster/cog.html)).
  Browser exports use geotiff.js `writeArrayBuffer`, which "writes the values
  uncompressed" ([README](https://github.com/geotiffjs/geotiff.js)).
- **STAC Item sidecar** with model version, DEM source, datum and GSD, so every DSM
  carries its provenance ([STAC spec](https://stacspec.org/en/about/stac-spec/)).

## 4. Checked and rejected

- **Viewshed / line-of-sight** — the telecom tool already ray-marches the DSM (`lib/usecases/telecom.ts`).
- **Flood inundation** — done twice over (`flood.ts`, `floodSim.ts`).
- **DSM → DTM / nDSM separation, building and tree heights** — done (`lib/dem/anchor.ts`, `lib/objects.ts`).
- **Fusing / super-resolving the coarse DEM with the model** — that is the DEM anchoring.
- **PAN-sharpening, large-scene tiling, seam blending** — done in Python (`scene_io.py`, `engine.py`); #2 brings the product path to the web app.
- **Explainability (attention maps)** — not in the rubric; σ (#3) is the trust signal judges can read.
- **QGIS plugin** — GeoTIFF / glTF exports already open in QGIS (`Frontend/README.md`); a plugin duplicates the app.
- **3D Tiles / Cesium** — rejected in `V5_Research_Additions.md` §5 (heavy; glTF + COG cover "standard format").
- **Off-nadir roof-offset height cues** — Cartosat MERGED tilt was 3.2° on the sample (`V5_Research_Additions.md` §1); too small to matter.

## 5. Verified vs inferred

Verified 2026-10-02: every codebase claim (read or grepped at commit `f367300`); CORS
and range behaviour of the COP30 bucket, PROJ CDN and Terrain Tiles (`curl` with an
`Origin` header); npm peer dependencies of `@react-three/xr`; each linked page above.
Not fetched directly: NREL's SPA page did not resolve, so the ±0.0003° figure rests on
the Reda & Andreas paper as indexed; QGIS volume methods come from the algorithm's
source help text, not the user manual.
Inferred: that judges send the MERGED product; that the Space handles a product zip;
the value and effort ranks; that ViT-L runs in-browser only after distillation or fp16.
