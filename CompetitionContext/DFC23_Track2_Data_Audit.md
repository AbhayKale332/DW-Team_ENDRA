# DFC23 Track 2 — what we measured, and what we still need to know

Written for an agent running on a machine that holds the **full** DFC23 Track 2
dataset. Everything below was measured on two small local downloads; the whole
point of this file is that those samples are too small to settle the questions
in §3, and every one of those questions changes a flag or a line of code.

Companion: `CompetitionContext/DFC23_Building_Classification.md` (the contest
terms and the S3 URIs). Code: `FineTunning/V4_Kaggle/prepare_data.py`
(`prepare_dfc23`, `_dfc23_pairs`, `_dfc23_height_check`),
`FineTunning/V4_Kaggle/dwdata/india.py` (`pack_labeled`),
`FineTunning/V4_Kaggle/tests/test_dfc23.py`.

---

## 1. What was measured, and on what

Two downloads, both small. **Mark everything here as provisional.**

| sample | what it held | n |
|---|---|---|
| `track2_test_data/` | `rgb/` + `sar/`, no reference | 605 scenes |
| `track2_train_data/` | a flat folder, files from mixed source dirs | 2 RGB + 3 nDSM, **1 complete pair** |

### 1.1 Imagery geometry — from 605 test scenes

| property | value |
|---|---|
| scene size | **512 × 512, all 605** — these are tiles, not large scenes |
| RGB | `uint8`, 3-band |
| SAR | `float32`, 1-band, ~0.03–11.9, mean 0.182, no NaN |
| transform pixel size | 0.5 m, uniform |
| CRS | **`None` on all 605** |
| black / nodata in RGB | negligible: max 2.56 % zero-luminance, no scene above 5 % |
| per-scene dynamic range | p2 spans 0–90 DN, p98 spans 107–249 DN |

### 1.2 The train sample — 3 nDSMs, 1 paired

All `GF2_NewDelhi_*` (Gaofen-2, New Delhi), 512 × 512, 0.5 m, and **these carry
`crs=EPSG:32643`** where the test split carried none. That difference matters:
`raster_gsd_m` (`dwdata/india.py`) returns its `default` when `ds.crs is None`,
because it cannot know the transform's units without one. So on test-shaped
files the GSD comes from `--dfc23_gsd`; on train-shaped files it is read from the
GeoTIFF. Both give 0.5 m here, so the outcome is the same — but do not assume
the mechanism.

The float32 files are **nDSMs, not SAR**: median exactly 0.0, floored at 0,
versus SAR's min 0.026 / mean 0.182. Verified co-registered with their RGB
(identical geotransform) on the one complete pair.

### 1.3 Height distribution — 3 scenes, and not GAMUS-shaped

| band | scene A | scene B | scene C | GAMUS val prefix |
|---|---|---|---|---|
| exactly 0.0 m | 85.6 % | 68.9 % | 77.1 % | — |
| 0–2 m | 87.0 % | 69.3 % | 77.4 % | 49 % |
| 2–5 m | 0.17 % | 0.42 % | 0.29 % | 9 % |
| 5–10 m | 1.76 % | 5.19 % | 2.59 % | 17 % |
| 10–20 m | 0.39 % | 0.79 % | 1.91 % | 15 % |
| 20–50 m | 9.04 % | 24.2 % | 17.8 % | 9 % (20 m+) |
| 100 m+ | **1.00 %** | 0 | 0 | — |

Bimodal with a hollow middle — consistent with a ~2 m stereo product resampled
onto the 0.5 m grid. Ground is **exactly 0.0**, so real ground and any
zero-filled nodata are indistinguishable; there is no NaN and no `nodata` tag.

### 1.4 A confirmed artefact

On `GF2_NewDelhi_28.5557_77.1194`, **every** pixel above 100 m — 2,615 of them,
up to 183.2 m — lies in rows 0–31, a ribbon on the tile's top border. The RGB
underneath is ordinary city: luminance 119.5 and texture std 33.6, against
133.5 / 38.8 for the rest of the tile. No structure. Stereo blunders.

Nothing downstream catches it: `pack_labeled` filters at 500 m and
`dwdata/dataset.py` clamps at `--max_valid_height_m` 200, so a phantom 183 m
target on a normal rooftop trains as fact — and `StratumBalancer`
(`stratum_balance_beta` 0.7, clip 8) gives the tallest stratum the *largest*
loss weight in the batch, on a model whose measured failure is already
tall-structure bias (`tall_bias −1.70`, 20 m+ bias −2.20 in the `v4-2` run).

`prepare_data.py` now reports the tall mass split border vs interior:

```
[dfc23] HEIGHT CHECK dfc23_g050/train: median 0.00 m  p99 99.7 m  max 183.1 m  exactly-0 85.6 %
[dfc23]   >100 m: 4.256 % of border-32px pixels vs 0.000 % of interior   [!] concentrated
          at the tile border — stereo blunders, not buildings.
```

`--dfc23_max_height_m 100` marks such pixels **invalid** rather than clipping
them (a blunder is an unknown, not a building of the ceiling height). It is
**off by default** and the check reports rather than edits, because real
buildings do sit at tile edges and one scene cannot establish prevalence.

---

## 2. What the code expects

`prepare_dfc23` wants a directory holding **sibling `rgb/` and `dsm/`
subdirectories with matching filenames**:

```
<--dfc23_dir>/
  rgb/P_0199.tif
  dsm/P_0199.tif        # same stem, different parent dir
  sar/P_0199.tif        # ignored
```

It accepts `rgb|opt|optical|image|images` and `dsm|ndsm|nDSM|height|agl|gt_nDSM`.
Given anything else it prints `need an rgb-like and a dsm-like subdirectory …
found [...]` and skips cleanly. **A flat folder does not work** — the local
train download was flat, with files from three source dirs collapsed together
and a browser-renamed `… (1).tif` collision. Preserve the tree.

It then: groups scenes by measured GSD into one store per family
(`dfc23_g050`, `dfc23_g080`, …); splits **by scene, not by tile**; symlinks into
the `<stem>_rgb` / `<stem>_ndsm` convention `pair_rasters` understands; and hands
off to `pack_labeled`. SAR is ignored by design — the packed store is 3-channel
and DINOv3's patch embed is `Conv2d(3, …)`.

Relevant flags: `--dfc23_dir --dfc23_tile --dfc23_max --dfc23_val_frac
--dfc23_gsd --dfc23_max_height_m --dfc23_absolute_dsm`.

Two known constraints, both geometry, neither a bug:

* **No scale-augmentation headroom.** A 512 px store at a 512 px model input caps
  achievable GSD at `512 × 0.5 / 512 = 0.5 m`, so DFC23's jitter collapses to
  0.30–0.50 and `[!] requested hi 1.20 unreachable` fires. Do **not** pack a
  smaller tile to "fix" it — a store with tiles smaller than `tile_size` yields
  upsampled crops, which is worse. Scale range comes from GAMUS (0.30–0.66) and
  `synrs3d_g1` (0.30–0.80).
* **One tile per 512 px scene**, so the scene-level split is currently
  indistinguishable from a tile-level one. It is still correct and still tested;
  it starts to matter if any scene is bigger than one tile (see Q3).

Packed cost: **1.84 MB per 512 px tile** (rgb 0.79 + hgt 0.52 + cls 0.26 +
val 0.26).

---

## 3. Open questions — the reason this file exists

Each has a decision attached. Answer with numbers, not impressions, and say the
sample size.

### Q1 — Inventory and layout
Full tree of the real dataset. How many scenes in `track2/train/{rgb,dsm,sar}`?
Do all three have identical stem sets, or are there orphans (an RGB with no
nDSM is silently dropped by `_dfc23_pairs`)? How do the top-level `rgb/ sar/
dsm/` folders relate to `track2/train/*` — duplicates, a superset, or different
scenes? **Decision:** whether `--dfc23_dir` points at `track2/train` or
somewhere else, and whether any scenes are being silently lost.

### Q2 — City coverage
Filenames mix `P_XXXX` (no city), `SV_<City>_<lat>_<lon>` and
`GF2_<City>_<lat>_<lon>`. How many scenes per city and per sensor prefix? Is
there a mapping from `P_XXXX` to a city (a metadata file, or the CRS/UTM zone)?
**Decision:** whether New Delhi is a large enough slice to weight up, and whether
to report a Delhi-only val number — that is the closest thing this project has
to an Indian accuracy figure, and §7 of the V4 README currently says none exists.

### Q3 — Scene size and GSD across the whole split
Are all train scenes 512 × 512, or are some larger? Histogram of
`transform.a`, and how many scenes have a CRS vs `None`. Which UTM zones appear?
**Decision:** `--dfc23_tile`; whether the multi-GSD store split actually
activates (0.5 m SuperView-1 vs 0.8 m Gaofen-2 must not share a nominal); and
whether the scene-level split is doing real work.

### Q4 — Artefact prevalence *(highest value)*
Across **all** train nDSMs: what fraction of scenes have >100 m mass, and in
those, what fraction of that mass is within 32 px of a border vs the interior?
Is it only the top edge or all four? Is there a cleaner cut than 100 m — e.g. a
visible knee in the global height histogram? **Decision:** whether
`--dfc23_max_height_m` should be on by default and at what value. If the
artefact is in a handful of scenes, dropping those scenes beats a global
ceiling; if it is endemic, the ceiling is right; if border-tall mass tracks
interior-tall mass, there is no artefact and the ceiling would destroy real data.

### Q5 — Is any of the tall mass real?
Per city, the max and p99.9 nDSM. 183 m in New Delhi is implausible; 183 m in a
DFC23 city with genuine towers may be real. Cross-check the tallest interior
(non-border) blobs against their RGB — does a shadow or a footprint support the
height? **Decision:** whether a single global ceiling is even the right shape of
fix, or whether it must be per-city.

### Q6 — Nodata
Ground reads exactly 0.0 and there is no NaN or `nodata` tag in the sample. Is
that true across the split — any NaN, any `-9999`, any `nodata` set? Are there
scenes that are mostly zero (an empty tile is indistinguishable from flat
ground, and `pack_labeled` drops tiles under 50 % valid, which will never fire
if every pixel is "valid")? **Decision:** whether the `valid` mask needs a real
nodata rule instead of `isfinite & > −2 & < 500`. This is the failure that put
41 % black padding-at-zero into v2's training set — see `dwdata/india.py`'s
module docstring.

### Q7 — Global height histogram
The §1.3 table over the whole split, not 3 scenes. **Decision:**
`--sampler_weights` (currently `gamus:2,dfc23:2,…`) and whether
`stratum_balance_beta` still makes sense with a bimodal source in the mix.

### Q8 — `buildings_only_train.json`
Nothing in the repo produced it and nothing reads it; `pycocotools` is not a
dependency. What are its `categories` (single `building` class, or the 12 roof
types collapsed)? Do its `image` entries match `track2/train` stems? **Decision:**
whether a building mask is cheap enough to rasterise into the `cls` plane — DFC23
currently packs `cls=None` / `has_seg=False`, and `w_seg` 0.2 is carried by
SynRS3D alone.

### Q9 — Co-registration at scale
Verified on one pair. Do `rgb` and `dsm` share a geotransform on **every** scene?
Any size mismatches? (`pack_labeled` resizes a mismatched height raster onto the
RGB grid, so a mismatch is survivable but silent.) **Decision:** whether to add
an assertion rather than a silent resize.

### Q10 — End-to-end
Run the real packer on the real split and paste the full output: scene counts,
per-family stores, every `HEIGHT CHECK` line, total GB. **Decision:** confirms
the flags before a 960-minute training run depends on them.

---

## 4. How to answer them

Needs `rasterio` and `numpy`. Read-only until Q10.

```bash
git -C <repo> pull
cd <repo>/FineTunning/V4_Kaggle
python -m pytest -q                      # 152 expected; a failure here first
```

Q10, once Q1 has established the path:

```bash
python prepare_data.py --data_root /tmp/dfc23_probe --datasets dfc23 \
       --dfc23_dir <.../track2/train> --dfc23_tile 512
```

For Q4–Q7, sample **every** scene, not a subset — prevalence is the question.
Report counts and percentages with the denominator stated.

## 5. Deliverable

Update **this file** in place: fold the answers into §1, replacing the
provisional numbers and saying what n each is now based on. Then, in §3, replace
each question with the decision it resolved to and the flag value that follows.
Open a question that the data raises and this file did not anticipate rather
than forcing it into an existing one. Anything that turns out to be a code bug —
Q6 and Q9 are the likely candidates — gets a failing test first.
