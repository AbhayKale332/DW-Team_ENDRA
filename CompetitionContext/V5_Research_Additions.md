# V5 — research: additions that raise the score

Checked 2026-09-24. Primary sources where reachable; fallbacks are marked.
Companions: `FAQs.md` (host answers), `Cartosat_2S_Spec.md`, the V5 plan
(`FineTunning/v5/README.md` once written). Local sensor facts come from the two
Cartosat-2E samples in `cartosat_2S_Sample/` (git-ignored).

The FAQ lives at [SIH-DepthWizard-2026 issue #1](https://github.com/IMG-PROCESS-SAC/SIH-DepthWizard-2026/issues/1).
[SIH2026](https://github.com/IMG-PROCESS-SAC/SIH2026) still holds only
`README.md` (latest commit `6601349`, "Add FAQs section") — no data.

---

## 0. What decides the score

| | |
|---|---|
| Accuracy, 50 % | RMSE, MAE, r of the **absolute DSM vs SRTM or Copernicus 30 m** (FAQ, Evaluation 2), on **Cartosat-2S 0.6 m** GeoTIFFs (FAQ, Evaluation 1). "For GeoTIFF images, the values match DEM heights" (FAQ, Model 1). No reference data is provided; ranking is relative. |
| Visualization, 50 % | Projection accuracy, visual fidelity, navigability, UI, stability, standalone deployment. A locally hosted web app is accepted (FAQ, Execution 1). |

So the calibration of the absolute DSM and the Cartosat input path decide the
accuracy half; GAMUS RMSE moves it only weakly.

## 1. Local Cartosat-2E samples (measured with `gdalinfo`, `BAND_META.txt`)

| | `Cartosat-2E/247677521` — NRSC **MERGED** | `5132211` PAN + `5132611` MX |
|---|---|---|
| Place / date | Kurnool Ultra Mega Solar Park, 15.65 N 78.30 E, 6 Jan 2022 | Bhubaneswar, 20.32 N 85.83 E, 16 May 2020 |
| Files | `BAND1..4.tif`, **four single-band 0.6 m GeoTIFFs**, 829 MB each | PAN `BAND.tif` 0.6 m (833 MB); MX `BAND1..4.tif` 1.6 m |
| Level | `ProcessingLevel=MERGED`, ORTHO, `DEMSource=SRTM90` | PAN GEOREF (CDEM10); MX ORTHO (ASTER30) |
| Grid | UTM 44N (EPSG:32644), 20362 × 20357 | UTM 45N (EPSG:32645), 20491 × 20332 / 7687 × 7640 |
| Radiometry | UInt16, 11-bit, very dark: B1 mean 131 σ 13, B3 mean 150 σ 40 | UInt16, 11-bit; PAN mean 475 σ 86 |
| NoData | 0, rotated-scene collar | 0; 29 % of PAN is collar |
| TIFF layout | strips, one row per block | same |
| Sun / tilt | az 136.0°, el 37.7° / 3.2° | az 86.0°, el 57.8° / −8.9° |
| `MeanElevation` | 328 | −23 (ellipsoidal; geoid ≈ −62 m there) |

Band order B1 blue 0.43–0.52 µm, B2 green 0.52–0.61, B3 red 0.61–0.69, B4 NIR
0.76–0.90 ([eoPortal Cartosat-2E](https://www.eoportal.org/satellite-missions/cartosat-2e), citing ISRO) → **RGB = bands 3,2,1**.
The judges' "0.6 m RGB TIFF" most likely arrives as the MERGED product: four
separate band files.

## 2. Verified answers

| Item | Answer | Source |
|---|---|---|
| C2S GSD | PAN 0.6 m, MX 1.6 m, 9 × 9 km | [Bhoonidhi C2S spec](https://bhoonidhi.nrsc.gov.in/bhoonidhi_resources/help/sampleprods/Cartosat-2S/C2S-Specs.pdf) |
| NRSC "RGB" | Merged / PAN-sharpened PAN+MX, method unstated. Free sample `…/sampleprods/Cartosat-2S/C2E/247677521_C2E_Merge_solarpanels.zip` (= the local Kurnool product) | [NRSC workshop](https://www.nrsc.gov.in/nrscnew/assets/pdf/announcements/Bhoonidhi_2022/4-BWS.pdf), [sample products](https://bhoonidhi.nrsc.gov.in/bhoonidhi/help/sampleProducts.html) |
| Ortho accuracy | < 1 px plains, ≤ 2 px hills (CartoDEM ortho) | NRSC workshop |
| Vertical datums | **SRTM GL1: EGM96**. **Copernicus GLO-30: EGM2008 (EPSG:3855)**, LE90 < 4 m. **CartoDEM: WGS84 ellipsoidal**, "a DSM and not a bare earth surface" | [USGS SRTM](https://data.usgs.gov/datacatalog/metadata/USGS.EROS5e83a3ee1af480c5.xml), [COP-DEM handbook v5](https://dataspace.copernicus.eu/sites/default/files/media/files/2024-06/geo1988-copernicusdem-spe-002_producthandbook_i5.0.pdf), [CartoDEM spec](https://bhoonidhi.nrsc.gov.in/bhoonidhi_resources/help/sampleprods/Cartosat-1/DEM/CartoDEM-Specs.pdf) |
| Geoid N (EGM2008 / EGM96) | Bhubaneswar −62.02 / −62.41 m; Bengaluru −86.16 / −86.42; Shimla −37.55 / −41.25; Gangtok −39.24 / −34.56. COP30 − SRTM datum offset for the same surface: −0.4 m (Bhubaneswar), −3.7 m (Shimla), +4.7 m (Gangtok). Ellipsoidal vs orthometric: 37–86 m | [GeographicLib GeoidEval](https://geographiclib.sourceforge.io/cgi-bin/GeoidEval) |
| What the 30 m references measure | Both are **surface** models. COP30 "includes buildings, infrastructure and vegetation" (handbook §2.2); SRTM C-band phase centre sits inside canopy ([Kellndorfer 2004](https://www.sciencedirect.com/science/article/abs/pii/S0034425704002330)). Removing trees/buildings from COP30 cut MAE vs bare earth 5.15 → 2.88 m (forest), 1.61 → 1.12 m (built-up) ([FABDEM, Hawker 2022](https://doi.org/10.1088/1748-9326/ac4d4f)). Ranking: COP30 ≈ FABDEM > ALOS > NASADEM > SRTM ([Bielski 2024](https://arxiv.org/abs/2302.08425)) | as cited |
| Open C2S over hills/forest | < 5 m imagery is priced for non-government users; only the Bhoonidhi samples are free | [Bhoonidhi brochure 2025](https://www.nrsc.gov.in/nrscnew/assets/pdf/brochures/Bhoonidhi_Brochure_2025.pdf) |

## 3. Ranked additions

| # | Addition | Criterion | Impact | Effort |
|---|---|---|---|---|
| 1 | DEM-anchored detail fusion as the GeoTIFF default | Accuracy | Very high | S |
| 2 | Datum-correct DEM stack (native SRTMGL1 + COP30, CartoDEM ellipsoid→geoid, compound vertical CRS, offline fallback) | Accuracy, stability | Very high | S–M |
| 3 | Ask the hosts three protocol questions | Accuracy (sets λ) | High | S |
| 4 | Judge-mirror imagery + `judge_proxy.py` | Accuracy, 4-landscape story | High | M |
| 5 | Viewer renders the absolute DSM; georef bug fixes | Projection accuracy, fidelity | High | S |
| 6 | In-app validation vs COP30/SRTM and any uploaded reference | Visualization + PS deliverable | High | M |
| 7 | Shadow-based scale check from `BAND_META` sun elevation | Scale-calibration milestone | Medium | S |
| 8 | GCPs in lon/lat/elev; offset/tilt on DTM, scale on nDSM | Scale-calibration milestone | Medium | S |
| 9 | Edge-aware meshing: vertical walls + RTIN | Fidelity, navigability | Medium–high | M |
| 10 | Uncertainty layer (`b_std`) | Visualization, honesty | Medium | S |
| 11 | Weak Indian references (Meta 1 m CHM, Open Buildings 2.5D, ICESat-2 ATL08) | Stability evidence | Medium | M |

### 3.1 DEM-anchored detail fusion
`A` = valid-pixel area mean onto the DEM grid, `U` = interpolation back.
`DSM = U(DEM) + λ·[nDSM − U(A(nDSM))]`, then a few Tobler block-residual
iterations `D ← D + U(DEM − A(D))` so each cell mean equals the DEM
([Tobler 1979](https://doi.org/10.1080/01621459.1979.10481647)).
`DTM + nDSM` double-counts what the 30 m surface models already hold, and
`fit_dtm` (`geo/calibrate.py`) falls back to a scene median below 32 ground
pixels — the closed-canopy-hillside case. Anchoring needs no ground pixels.
*Inferred:* against a 30 m reference compared per pixel, the detail term only
adds error (RMSE grows with λ × high-pass energy); if the judges aggregate to
30 m, λ = 1 is free. #3 decides.

### 3.2 Datum-correct DEM stack
`geo/dem.py`'s `srtm30` reads AWS Terrain Tiles — Web-Mercator, mixed sources
([joerd data-sources](https://github.com/tilezen/joerd/blob/master/docs/data-sources.md)),
not native SRTM GL1. Native SRTMGL1 / COP30 / NASADEM via the
[OpenTopography API](https://opentopography.org/developers) (free key). CartoDEM
raw is ~62 m too low at Bhubaneswar. Write compound vertical CRS
(e.g. EPSG:32645+3855 / +5773; GDAL writes GeoTIFF 1.1 for a vertical CRS,
[GDAL GTiff](https://gdal.org/en/stable/drivers/raster/gtiff.html)). Offline:
today no internet ⇒ no absolute DSM at all; ship a DEM cache and accept a
user DEM. RPC heights are ellipsoidal ([rasterio](https://rasterio.readthedocs.io/en/stable/topics/georeferencing.html)).

### 3.3 Questions for the hosts (post on issue #1)
1. Is our DSM aggregated to the DEM grid, or the DEM resampled to ours — with which resampler?
2. SRTM GL1, Copernicus GLO-30, or both — and which vertical datum?
3. Are NoData pixels excluded?

### 3.4 Judge-mirror imagery
- Local C2E MERGED Kurnool (judge format; sparse + hilly ridges; low sun).
- Local C2E Bhubaneswar PAN+MX, pan-sharpened (urban, flat).
- Maxar Open Data "North India Floods" (Sikkim), WV02 0.56 m, 26° off-nadir, sun az 145.2° el 51.2° in STAC, CC BY-NC 4.0 ([collection](https://maxar-opendata.s3.amazonaws.com/events/India-Floods-Oct-2023/collection.json)) — hilly/forested.
- NAIP 0.3/0.6/1.0 m + USGS 3DEP LiDAR DSM / HAG (Planetary Computer STAC `naip`, `3dep-lidar-dsm`, `3dep-lidar-hag`) — LiDAR truth, US only.
Score against both SRTM and COP30, per pixel and at 30 m. Evaluation use only; state licences.

### 3.5 Viewer shows what is scored (code gaps, verified)
- `viewer/app.js` loads only `ndsm_m.npy`; `dsm_m.npy` is never rendered.
- `SceneMeta.summary()` omits `transform` ⇒ `lonLatAt()` always null.
- `read_scene(max_side=…)` rescales `gsd_m` but not `transform` ⇒ GeoTIFF output and DEM fetch misregistered (reachable from `serve/app.py`).

### 3.6 In-app validation
Reproject any uploaded reference (any CRS/datum) onto the prediction grid;
auto-fetch COP30 + SRTM and show "what the judges will see": RMSE/MAE/r per
pixel and at 30 m, per landscape, with error layer and profile.

### 3.7 Shadow-based scale check
With sun elevation known, search one scale s maximising IoU between image
shadows and shadows cast by s·nDSM ([Kadhim & Mourshed 2018](https://ieeexplore.ieee.org/abstract/document/8126256/)).
For PNG with unknown sun, scale and elevation are confounded — not usable.

### 3.8 GCPs
`read_gcps` takes row/col; `refine_with_gcps` applies `s·DSM + o` to the whole
DSM, scaling terrain too. Accept lon/lat/elev + datum; offset (+ tilt with ≥ 3)
on the DTM, scale on the nDSM only.

### 3.9 Edge-aware meshing
Strided decimation drops thin towers and drapes roof texture on sloped walls.
Split vertices at height discontinuities and add vertical wall quads; RTIN via
[Mapbox Martini](https://github.com/mapbox/martini) (ISC). Textures ≤ 4096²
([MDN](https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/WebGL_best_practices)).

### 3.10 Uncertainty
`models/heads.py` emits `b_std`; `predict_scene` and the viewer drop it.

### 3.11 Weak Indian references (deck, not training truth)
[Meta/WRI 1 m canopy height](https://registry.opendata.aws/dataforgood-fb-forests/) (CC BY 4.0);
[Open Buildings 2.5D Temporal](https://sites.research.google/gr/open-buildings/temporal/) (~4 m, India);
[ICESat-2 ATL08](https://nsidc.org/data/atl08/versions/7) along-track terrain/canopy.

## 4. Positioning
No clean public GAMUS leaderboard: HTC-DC Net ([2309.16486](https://arxiv.org/abs/2309.16486)) and
SynRS3D/RS3DAda ([2406.18151](https://arxiv.org/abs/2406.18151)) do not report GAMUS test.
[TerraHeight-S](https://huggingface.co/benfox6515/TerraHeight-S) reports GAMUS *val* RMSE 2.616 / MAE 1.312 / r 0.924
on its own protocol. Publish ours.

## 5. Checked and rejected
- Desktop packaging — the FAQ accepts a local web app.
- GSD estimation for PNG — relative heights "at any acceptable scale".
- FABDEM as anchor — bare earth vs a DSM reference, and CC BY-NC-SA. For a DTM layer, [GEDTM30](https://pmc.ncbi.nlm.nih.gov/articles/PMC12296579/) (CC BY 4.0, EGM2008).
- Co-registering the DEM to the image — the judges presumably sample on the image geotransform (inferred).
- 3D Tiles / Cesium — heavy; glTF + COG satisfy "standard format".

## 6. Verified vs inferred
Verified 2026-09-24: every table row and URL above; the code gaps (read
directly); local `BAND_META` fields. Secondary: band wavelengths (eoPortal).
Inferred: the judges' input is pan-sharpened/MERGED; `MeanElevation` is
ellipsoidal; the detail-vs-RMSE trade-off; impact ranks; that the judges
sample on the image geotransform.
