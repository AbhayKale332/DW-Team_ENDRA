# US3D store and measured GSDs (2026-09-26)

What was done on the Lightning Studio on 2026-09-26: US3D (DFC2019 Track 1) was
added as a packed store, its data was audited, and the pixel sizes of US3D and
GAMUS were measured and published. The packer defaults that came out of it are
commit `0ac4e0f` (`GAMUS_GSD_M = 0.25`, `US3D_GSD_M = 0.32`).

## Outcome

| Kaggle dataset | Version | `gsd_m` | Layout on Kaggle |
|---|---|---|---|
| `abhaydkale232/depthwizard-us3d` | **3** | 0.30 → **0.32** | flat: `train/`, `val/` |
| `abhaydkale232/depthwizard-gamus` | **2** | 0.33 → **0.25** | nested, unchanged: `train/train/`, `val/val/`, `test/test/` |

Both are private. Each new version was downloaded back and checked:
- every `index.json` reads the new value and is byte-identical to the edited file;
- the file list has the same names and byte sizes as before (US3D 96 files, GAMUS 286).

Shards and `stretch_bounds_2_98.npy` were not touched. The stretch bounds are
per tile and don't depend on GSD.

**GAMUS RMSE is a new baseline.** With `gsd_m` 0.25 the val/test centre crop is
the whole 1024 px tile. Before, it was 776 px at a true ~0.38 m/px, not 0.5 m/px.
Re-score the old `best.pt` on the corrected `gamus/test` before comparing runs.
The old checkpoint will probably score worse there: it learned GAMUS at the
wrong scale, and the new inputs sit at the coarse end of what it saw. The Modal
volume copies of GAMUS still say 0.33 until they are edited.

## 1. US3D raw data

The data comes from IEEE DataPort (dataset 6870): `Train-Track1-RGB.zip`,
`Train-Track1-Truth.zip` and `Validate-Track1.zip`. All three passed `unzip -t`.

It was extracted to `~/data/DFC2019-Track1/` with a README:

```
train/rgb  2,783  *_RGB.tif  uint8 ×3
train/agl  2,783  *_AGL.tif  float32, m above ground (trees included)
train/cls  2,783  *_CLS.tif  uint8 ASPRS codes {2, 5, 6, 9, 17, 65}
val/rgb       50  *_RGB.tif
val/msi       50  *_MSI.tif  uint16 ×8 (read with tifffile; rasterio misreads it)
```

Facts about the raw data:
- Names are `CITY_TILE_VIEW`. There are 108 locations with 1–43 views each.
- Each view's labels are in that view's own geometry, so views of one location
  don't align pixel for pixel.
- The GeoTIFFs have no georeferencing.
- `Validate-Track1` has no ground truth, and there is no test split.

## 2. Packed store `us3d`

`prepare_data.py --datasets us3d --us3d_dir ~/data/DFC2019-Track1/train` produced:

| Split | Tiles | Locations | Skipped | Size |
|---|---|---|---|---|
| train | 2,492 | 97 | 0 | 17.0 GiB |
| val | 291 | 11 | 0 | 2.0 GiB |

- The split is by location. Val holds JAX_028, 072, 113, 122, 175, 224 and
  OMA_059, 287, 288, 292, 364.
- A pixel is invalid if its height is non-finite, ≤ −2 m or > 150 m, or its RGB is
  pure black. Nearly all invalid pixels come from black RGB (0.19% of train pixels).
- Classes map to GAMUS ids: 2→ground, 5→tree, 6→building, 9→water, 17→other.
  Code 65 and any other id is stored as **254**, not 255: the packer clips to
  0–254. Training still ignores it, via a later clip to the ignore index 7.

## 3. EDA and loader check (outputs in `~/dwdata/us3d_eda/`)

- **Loader:** the v5 profile runs on CPU with 0 non-finite batches and valid
  fraction 0.998. Augmentation keeps the image and height aligned
  (`loader_crops.png`).
- **Height strata (0–2 / 2–5 / 5–10 / 10–20 / 20+ m):** train 66 / 10 / 13 / 9 / 2 %,
  val 72 / 8 / 9 / 8 / 3 %. GAMUS val is about 61% in 0–2 m.
- **Median heights:** ground 0.0 m, building 6.5 m (train) / 10.5 m (val),
  tree 7.8 m / 6.5 m.
- **Suspect labels:**
  - about 5% of tree pixels are under 1 m, mostly crown edges and leaf-off snow views;
  - about 4% of building pixels are under 2 m;
  - one JAX_164 building is 0 m in all its views.
- **Things to watch:**
  - OMA_059 (an airport, nearly all 0 m) is 41 of the 291 val tiles, which flatters val.
  - Winter OMA views are paired with leaf-on heights.
  - v5 asks for GSD jitter up to 1.2 m, but US3D can reach only 0.64 m.
  - The DINOv3 image processor is gated, so it needs an HF token on the Studio.

## 4. GSD measurement (`~/dwdata/gsd_measurements/`, also copied to `SIH/gsd_measurements/`)

Pixel size was measured on objects of regulation size, directly on the raw tiles.

**Method** (`scripts/gsd_measure2.py`):
1. Sample a whiteness profile along each axis of the object.
2. Find the line centres to sub-pixel precision.
3. Check each measurement against the object's own line spacing:
   - tennis: baseline→service 5.485 m and doubles→singles 1.37 m;
   - football: yard lines 4.572 m;
   - soccer: goal area 18.29 × 5.49 m.
4. Keep a row only if the check **and** the two axes agree within 5%.

| Dataset | City | Kept | Median m/px | IQR |
|---|---|---|---|---|
| US3D | JAX | 14 | 0.313 | 0.310–0.319 |
| US3D | OMA | 12 | 0.337 | 0.328–0.346 |
| **US3D** | all (13 views, 5 locations) | 26 | **0.321** | 0.313–0.334 |
| GAMUS | PHL | 9 | 0.253 | 0.253–0.254 |
| GAMUS | NYC | 7 | 0.251 | 0.250–0.251 |
| GAMUS | DC | 3 | 0.256 | 0.253–0.260 |

Notes on the measurements:
- **Spread across views:** the same US3D object varies by 2–10% between views.
  Off-nadir views can be anisotropic: JAX_144_015 measures 0.344 along one axis
  and 0.378 across it.
- **Resolution limit:** at about 0.31 m/px the 1.37 m tennis check is only about
  4.4 px, so most court short-axis rows fail it and only the long axis is kept.
- **GAMUS cities:** GAMUS contains only DC, NYC and PHL. EarthNets README:
  "Remove the cities (OMA and JAX)". The paper's "0.33m" doesn't match any city.
- **Rejected method:** comparing building-footprint areas against GAMUS was not
  used. Rowhouse blocks make the component sizes non-comparable.

## 5. Kaggle lessons

- **Zip layout:** Kaggle extracts `<split>.zip` into `<split>/`. Members named
  `<split>/<file>` therefore give `<split>/<split>/`. Us3d v1 was nested this way
  and was replaced; `run_kaggle.sh link` handles both layouts.
- **Descriptions:** `kaggle datasets metadata --update` changes the description
  without a new version. Kaggle truncates the description at a `<...>`
  HTML-like token.
- **Listings:** `kaggle datasets files` lists at most 200 files per page. Follow
  `--page-token` to see all of them.

## Where things are (Lightning Studio)

| Path | Contents |
|---|---|
| `~/data/DFC2019-Track1/` | raw US3D, plus the three original zips in `~/data/` |
| `~/dwdata/us3d/` | packed store, now `gsd_m` 0.32, with `index.json.bak` = 0.30 |
| `~/dwdata/us3d_eda/` | EDA scripts, logs and figures |
| `~/dwdata/gamus/val/` | GAMUS val as downloaded, still at 0.33 |
| `~/dwdata/gsd_measurements/` | measurements: CSVs, 88 annotated PNGs, scripts, README |
