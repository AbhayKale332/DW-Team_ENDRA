# DFC23 Track 2 — what we measured, and what we decided

Measured on the **full** dataset: `track2/train` = **1773 scenes** (rgb + dsm +
sar), `track2/val` = 579 (rgb + sar), `track2_test_data` = 605 (rgb + sar).
Every number below states the n it rests on. The provisional numbers this file
used to carry — 605 test scenes and 3 New Delhi nDSMs — have been replaced.

Companion: `CompetitionContext/DFC23_Building_Classification.md` (the contest
terms and the S3 URIs). Code: `FineTunning/V4_Kaggle/prepare_data.py`
(`prepare_dfc23`, `_dfc23_pairs`, `_dfc23_height_check`),
`FineTunning/V4_Kaggle/dwdata/india.py` (`pack_labeled`),
`FineTunning/V4_Kaggle/tests/test_dfc23.py`.

Dataset on this machine: `/teamspace/studios/this_studio/kaggle_upload/DFC23`
(12 GB, of which ~5.5 GB is a duplicate — see Q11).

---

## 1. What was measured, and on what

### 1.0 Inventory — n = the whole dataset

| tree | rgb | dsm | sar | stems |
|---|---|---|---|---|
| `track2/train/` | 1773 | 1773 | 1773 | **identical sets, 0 orphans** |
| `track2/val/` | 579 | — | 579 | `P_0000…`, disjoint from test |
| `track2_test_data/` | 605 | — | 605 | `P_….`, disjoint from val |
| `rgb/ dsm/ sar/` (top level) | 1773 | 1773 | 1773 | **byte-identical copy of `track2/train`** |

Only `track2/train` carries reference nDSMs, so it is the only split that can be
packed. Nothing is silently lost: `_dfc23_pairs` drops an RGB with no nDSM, and
there are none.

### 1.1 Imagery geometry — n = 1773 train scenes (all of them)

| property | value | n |
|---|---|---|
| scene size | **512 × 512, all 1773** — tiles, not large scenes | 1773/1773 |
| RGB | `uint8`, 3-band | 1773/1773 |
| SAR | `float32`, 1-band | 1773/1773 |
| nDSM | `float32`, 1-band | 1773/1773 |
| transform pixel size | **0.5 m, uniform — including every Gaofen-2 scene** | 1773/1773 |
| CRS | **present on all 1773**, 8 UTM zones | 1773/1773 |
| `nodata` tag | `None` on rgb and dsm | 1773/1773 |

UTM zones: 32633 (782), 32723 (369), 32630 (151), 32643 (131), 32618 (108),
32611 (108), 32631 (90), 32756 (34).

**The train split is not shaped like the test split.** Train scenes all carry a
CRS; val and test carry **`crs=None`** (checked on the first 400 of each). So on
train, `raster_gsd_m` (`dwdata/india.py`) reads 0.5 m from the GeoTIFF and
`--dfc23_gsd` is never consulted. It would be consulted on val/test — which have
no nDSM and are never packed. Both paths give 0.5 m; the mechanism differs.

### 1.2 City and sensor coverage — n = 1773

| city | SV | GF2 | total | share |
|---|---|---|---|---|
| Berlin | 566 | — | 566 | 31.9 % |
| Brasilia | 199 | 47 | 246 | 13.9 % |
| Copenhagen | 216 | — | 216 | 12.2 % |
| Portsmouth | 151 | — | 151 | 8.5 % |
| **New Delhi** | 116 | 15 | **131** | **7.4 %** |
| New York | 108 | — | 108 | 6.1 % |
| San Diego | 108 | — | 108 | 6.1 % |
| Rio | 93 | — | 93 | 5.2 % |
| Barcelona | 90 | — | 90 | 5.1 % |
| Sydney | 34 | — | 34 | 1.9 % |
| São Luís | 30 | — | 30 | 1.7 % |

**`P_XXXX` never appears in train.** It is the val/test naming only, so the
"is there a `P_XXXX` → city mapping" problem does not arise for anything we
train on. Every train stem is `SV_<City>_<lat>_<lon>` or `GF2_<City>_<lat>_<lon>`.

### 1.3 Height distribution — n = 1773 scenes, 464,781,312 pixels

| band | DFC23 train (all 1773) | GAMUS val prefix |
|---|---|---|
| exactly 0.0 m | **66.628 %** | — |
| 0–2 m (incl. exact 0) | **69.974 %** | 49 % |
| 2–5 m | 3.758 % | 9 % |
| 5–10 m | 7.177 % | 17 % |
| 10–20 m | 9.259 % | 15 % |
| 20–50 m | 9.140 % | 9 % (20 m+) |
| 50–100 m | 0.646 % | — |
| 100–150 m | 0.046 % | — |
| 150–200 m | 0.00026 % | — |
| 200 m+ | **0 pixels** | — |
| below 0 m | **0 pixels** | — |

Bimodal with a hollow middle, as the 3-scene sample suggested — but the middle
is less hollow than that sample implied (5–20 m is 16.4 % here against 2.2–6.0 %
in the three New Delhi tiles). The per-scene exactly-0 fraction ranges
18.6 %–100.0 %, mean 66.5 %, median 67.5 %. Ground is **exactly 0.0**, there is
no NaN, no negative pixel and no `nodata` tag anywhere in the split.

### 1.4 The artefact — measured, and rarer than one scene suggested

Across all 1773 scenes, **the border-blunder is not endemic. It is two scenes.**

| threshold | scenes with any | pixels | border share | border density | interior density | ratio |
|---|---|---|---|---|---|---|
| >60 m | 241 (13.59 %) | 1,651,521 | 22.93 % | 0.3477 % | 0.3577 % | **0.97×** |
| >80 m | 92 (5.19 %) | 490,864 | 20.77 % | 0.0936 % | 0.1093 % | **0.86×** |
| >100 m | 37 (2.09 %) | 214,540 | 16.30 % | 0.0321 % | 0.0505 % | **0.64×** |
| >120 m | 24 (1.35 %) | 86,419 | 17.00 % | 0.0135 % | 0.0202 % | **0.67×** |
| >150 m | **1 (0.06 %)** | 1,203 | 100 % | 0.0011 % | 0 % | — |

The 32 px border band is 23.44 % of a 512 px tile. Tall mass is **under**-
represented there at every threshold up to 120 m. There is no global border
artefact; the §1.4 that used to stand here generalised from one tile.

**The 36 scenes above 100 m that are not artefacts** are Rio (20 scenes, max
148.5 m), New York (10, max 144.9 m) and Berlin (6, max 140.5 m) — real
high-rise, tall in the interior, and confirmed against the RGB: the New York
blob at 144.9 m is a compact 130 × 142 px footprint at luminance 146 against 99
for the rest of the tile (a bright roof with its shadow), the Berlin one is 62 %
flat within 10 % of its maximum (a roof plateau).

**The two that are artefacts** are both Gaofen-2 New Delhi tiles on the top
latitude row of the Delhi mosaic:

| scene | max | interior max | tall px | where |
|---|---|---|---|---|
| `GF2_NewDelhi_28.5557_77.1194` | 183.2 m | 89.9 m | 2,615 >100 m | rows 0–23, **289 columns wide**, 39 % filled |
| `GF2_NewDelhi_28.5557_77.1210` | 90.1 m | 28.5 m | 150 >54 m | rows 0–19, 26 columns |

The shape is the tell. A blunder is a **ribbon** spanning most of a tile edge; a
real building clipped by a tile boundary is a **compact blob** — `SV_Berlin_
52.4900_13.4933` puts 471 px above 100 m in a 26 × 23 box at 79 % fill, and its
125.2 m maximum reappears in the interior of three adjacent Berlin tiles, which
is what a real tower straddling a tile cut looks like. The other 129 New Delhi
scenes never exceed 89.9 m in their interiors, and the twelve tiles adjacent to
the blunder scene all cap at 29–31 m. 183 m in that part of Delhi is not real.

The RGB under the ribbon is ordinary city: luminance 102.4 / std 23.2 against
133.2 / 38.7 for the rest of the tile. No structure.

Total artefact mass: **2,765 px out of 464,781,312 — 0.0006 % of the split.**

### 1.5 Nodata — the real one is in the RGB, not the height

The height raster has no nodata of any kind (§1.3), so
`valid = isfinite & > −2 & < 500` is **`True` for every pixel of all 1773
scenes**, and `pack_labeled`'s "drop a tile under 50 % valid" gate could never
fire. That gate was dead code on this dataset.

What it should have been firing on is in the optical raster. **12 of 1773 scenes
carry all-black RGB** where the tile falls outside the optical footprint, and the
nDSM under it reads exactly 0.0 — a legitimate-looking flat-ground label that
nothing could reject:

| scene | black RGB | of those, labelled exactly 0 m |
|---|---|---|
| `SV_Berlin_52.4902_13.5090` | **53.09 %** | 73.4 % |
| `SV_Berlin_52.4906_13.5090` | 43.03 % | 79.9 % |
| `SV_Berlin_52.4902_13.5081` | 26.96 % | 55.1 % |
| `SV_Berlin_52.4920_13.5090` | 18.08 % | 71.8 % |
| `SV_Berlin_52.4905_13.5080` | 17.50 % | 67.5 % |
| 7 more (Berlin ×6, Sydney ×1) | 0.78 – 7.55 % | 76 – 94 % |

This is the 41 %-black-padding-at-0 m failure that `dwdata/india.py`'s module
docstring opens with, present again and undetected. Fixed: `pack_labeled` now
takes a pixel that is zero in all three RGB bands as invalid
(`black_rgb_is_nodata=True`). On the real split that drops
`SV_Berlin_52.4902_13.5090` entirely — 1506 tiles packed from 1507 train scenes
— and gives the 50 %-valid gate something to fire on.

### 1.6 Co-registration — n = 1773, all of them

`rgb` and `dsm` share a geotransform **exactly on 1773 of 1773** scenes, and all
are 512 × 512, so `pack_labeled`'s resize never runs on this dataset. (`sar`
shares it too, on all 1773.) No bug — but the resize no longer happens silently:
it now prints both shapes and says to check the pairing, because on a paired
product a mismatch means the pairing is wrong far more often than it means the
height product is coarser.

### 1.7 Building annotations

`track2/buildings_only_train.json` is COCO: **1773 images, stems matching
`track2/train` exactly** (0 either way), **125,153 polygon annotations**, a
**single `building` category** (not the 12 UBC roof types), mean 70.6 buildings
per scene, median 49, max 677, and 3 scenes with none.

### 1.8 Packed cost — measured

1506 train tiles = 2.6 GiB, 266 val tiles = 0.5 GiB → **1.77 MB per 512 px
tile**, 3.1 GiB for the whole DFC23 contribution.

---

## 2. What the code expects

`prepare_dfc23` wants a directory holding **sibling `rgb/` and `dsm/`
subdirectories with matching filenames**:

```
<--dfc23_dir>/
  rgb/GF2_Brasilia_-15.8652_-47.9337.tif
  dsm/GF2_Brasilia_-15.8652_-47.9337.tif    # same stem, different parent dir
  sar/GF2_Brasilia_-15.8652_-47.9337.tif    # ignored
```

It accepts `rgb|opt|optical|image|images` and `dsm|ndsm|nDSM|height|agl|gt_nDSM`.
Given anything else it prints `need an rgb-like and a dsm-like subdirectory …
found [...]` and skips cleanly. **A flat folder does not work.** Preserve the tree.

It then: groups scenes by measured GSD into one store per family; splits **by
scene, not by tile**; symlinks into the `<stem>_rgb` / `<stem>_ndsm` convention
`pair_rasters` understands; and hands off to `pack_labeled`. SAR is ignored by
design — the packed store is 3-channel and DINOv3's patch embed is `Conv2d(3,…)`.

Relevant flags: `--dfc23_dir --dfc23_tile --dfc23_max --dfc23_val_frac
--dfc23_gsd --dfc23_max_height_m --dfc23_absolute_dsm`.

Three constraints, all geometry, none a bug:

* **One GSD family.** Every train scene is 0.5 m, Gaofen-2 included. The
  multi-GSD store split is correct code that never activates here — see Q3.
* **No scale-augmentation headroom.** A 512 px store at a 512 px model input caps
  achievable GSD at `512 × 0.5 / 512 = 0.5 m`, so DFC23's jitter collapses to
  0.30–0.50 and `[!] requested hi 1.20 unreachable` fires. Do **not** pack a
  smaller tile to "fix" it — a store with tiles smaller than `tile_size` yields
  upsampled crops, which is worse. Scale range comes from GAMUS (0.30–0.66) and
  `synrs3d_g1` (0.30–0.80).
* **One tile per 512 px scene**, confirmed over all 1773, so the scene-level
  split is indistinguishable from a tile-level one in effect. It is still correct
  and still tested.

---

## 3. The decisions these questions resolved to

### Q1 — Inventory and layout → `--dfc23_dir <root>/track2/train`
1773 scenes in each of `track2/train/{rgb,dsm,sar}`, **identical stem sets, zero
orphans**, so nothing is silently dropped by `_dfc23_pairs`. The top-level
`rgb/ dsm/ sar/` are a byte-identical duplicate of `track2/train`, not the
"curated sample triplets" the dataset README claims (see Q11). `track2/val` and
`track2_test_data` have no `dsm/`, so they cannot be packed at all.
**Flag:** `--dfc23_dir /teamspace/studios/this_studio/kaggle_upload/DFC23/track2/train`.

### Q2 — City coverage → no weighting change; report a Delhi val number
11 cities, 2 sensors, table in §1.2. New Delhi is **131 of 1773 = 7.4 %**, split
by the default `--dfc23_val_frac 0.15` into 111 train / 20 val scenes. `P_XXXX`
is val/test naming only and never appears in train, so no `P_XXXX` → city
mapping is needed. **Decision:** do not weight New Delhi up — 7.4 % of one of
five sources, oversampled, would make the run a Delhi run, and DFC23's value in
the mix is real-satellite radiometry across 11 cities rather than Delhi alone.
Do report the Delhi-only val figure: **20 val scenes / 20 tiles**, which is a
small number to quote and must be quoted with its n. This is still the first
Indian accuracy figure the project has; V4 README §7 should be updated to say
"n = 20 scenes, New Delhi, DFC23 held-out" rather than "none exists".

### Q3 — Scene size and GSD → `--dfc23_tile 512`, one store, split still correct
All 1773 train scenes are 512 × 512 at 0.5 m with a CRS (8 UTM zones, §1.1).
**Gaofen-2 is 0.5 m here too, not 0.8 m** — the 62 GF2 scenes are resampled onto
the same grid as SuperView-1, so the multi-GSD split produces exactly one family,
`dfc23_g050`, and the 0.5-vs-0.8 hazard the code guards against does not exist in
this data. `--dfc23_gsd` is never consulted on train (CRS present everywhere);
it would be on val/test, which are never packed. One tile per scene, so the
scene-level split is currently doing no work *that a tile split would not also
do* — keep it, it costs nothing and is what makes the val number safe if a
larger scene ever arrives. **Flags:** `--dfc23_tile 512`, `--dfc23_gsd 0.5`
(unused but correct).

### Q4 — Artefact prevalence → `--dfc23_max_height_m` stays **off by default**
**The artefact is 2 scenes out of 1773 (0.11 %), not a property of the dataset.**
Over all 1773 scenes, >100 m mass is **0.64× as dense in the 32 px border band as
in the interior**, and ≤1.0× at every threshold from 60 m to 120 m (§1.4). The
border-vs-interior signature that justified a ceiling does not survive the full
split; it was one tile generalised.

A `--dfc23_max_height_m 100` default would **delete 211,925 px of genuine
high-rise** across 36 Rio / New York / Berlin scenes — and still miss
`GF2_NewDelhi_28.5557_77.1210`, whose blunder tops out at 90.1 m. That is the
"the ceiling would destroy real data" branch of this question, and it is the one
the data picked.

**Flags:** `--dfc23_max_height_m 0.0` is the right default and does not change.
**For the 960-minute run, pass `--dfc23_max_height_m 150`**: nothing real
anywhere in the split exceeds 148.5 m, so it costs exactly **1,203 px in one
scene (0.00026 % of the split)** and caps the single implausible extreme that
`StratumBalancer` would otherwise weight most heavily. It is a cheap cap, not a
fix — the residual 2,765 px of 100–150 m ribbon in the two Delhi tiles is
recorded as Q15 rather than paid for with a lower global ceiling.

### Q5 — Is any of the tall mass real? → yes, and a global ceiling is the wrong shape
Per city, n = 1773 (max, median p99.9, max p99.9, scenes >100 m, interior max):

| city | n | max | p99.9 median | scenes >100 m | interior max |
|---|---|---|---|---|---|
| Rio | 93 | 148.5 | 47.6 | 20 | 148.5 |
| New York | 108 | 144.9 | 71.2 | 10 | 144.9 |
| Berlin | 566 | 140.5 | 27.0 | 6 | 140.5 |
| **New Delhi** | 131 | **183.2** | 29.4 | 1 | **89.9** |
| Barcelona | 90 | 84.7 | 37.8 | 0 | 84.7 |
| Copenhagen | 216 | 76.1 | 26.2 | 0 | 76.1 |
| Portsmouth | 151 | 72.5 | 16.5 | 0 | 72.5 |
| Sydney | 34 | 71.9 | 40.8 | 0 | 71.9 |
| São Luís | 30 | 44.3 | 22.6 | 0 | 44.3 |
| Brasilia | 246 | 29.0 | 10.5 | 0 | 29.0 |
| San Diego | 108 | 17.4 | 10.3 | 0 | 17.4 |

New Delhi is the only city whose maximum is not also its interior maximum, and
the gap is 93 m. Rio, New York and Berlin reach 140–148 m **in the interior**,
supported by the RGB (compact bright footprints, roof plateaus — §1.4).
**Decision:** a single global ceiling is indeed the wrong shape of fix, but so is
a per-city table — the correct discriminator turned out to be **ribbon vs blob**,
not city and not height. Two named scenes carry all of it. No per-city flag is
added; §1.4 records the shape test so the next person can apply it.

### Q6 — Nodata → the height raster is clean; the **RGB** was the bug
No NaN, no `-9999`, no `nodata` tag, no negative pixel; global min 0.0, global
max 183.2, over all 1773 scenes. `valid` was therefore `True` everywhere and the
50 %-valid drop was dead code. **But 12 scenes carry all-black RGB over nDSM 0.0**
— up to 53 % of a tile, 73 % of it labelled exactly 0 m (§1.5). That is the v2
failure verbatim. **Code change, failing test first:** `pack_labeled` gained
`black_rgb_is_nodata=True`; a pixel zero in all three RGB bands is no longer
supervised. Tests `test_black_rgb_is_not_supervised_as_flat_ground` and
`test_a_mostly_black_scene_is_dropped_entirely`. No new flag — this is on
unconditionally, and on a fully imaged tile it costs nothing.

### Q7 — Global height histogram → keep `dfc23:2`, keep `stratum_balance_beta 0.7`
§1.3, n = 464,781,312 px. The 3-scene sample overstated how hollow the middle is:
5–20 m is **16.4 %** of the split, not the 2–6 % those tiles showed, and
20–50 m (9.1 %) matches GAMUS's 20 m+ band almost exactly. What is genuinely
unlike GAMUS is the ground mass — **66.6 % exactly 0.0** against GAMUS's 49 %
under 2 m. **Decision:** leave `--sampler_weights gamus:2,dfc23:2,…` alone; DFC23
is not so distributionally odd that it needs a different weight, and it is the
only real-satellite source. Leave `stratum_balance_beta 0.7` alone too — the
worry was that a bimodal source would hand a runaway weight to a thin tall
stratum, but DFC23's tall bands are *denser* than GAMUS's, not thinner, so beta
0.7 with clip 8 is doing what it was tuned to do. With the measured
`tall_bias −1.70` (under-prediction of tall) still open, up-weighting tall is the
right direction; this stays worth watching, not worth changing blind.

### Q8 — `buildings_only_train.json` → worth rasterising, but not in this run
COCO, 1773 images matching `track2/train` stems exactly, 125,153 polygons, a
**single `building` class** (the 12 UBC roof types are collapsed), polygon
`segmentation`, mean 70.6 buildings per scene (§1.7). So a binary building mask
for every DFC23 tile is genuinely cheap — one `pycocotools` dependency, or a
hand-rolled polygon fill, and the `cls` plane is already allocated (0.26 MB of
the 1.77 MB per tile is being written as `NO_LABEL` today). **Decision:** the
data supports it and it would give `w_seg 0.2` a second source instead of
SynRS3D alone — but it is a 1-class mask against a 0..6 shared class space, so it
needs a class-id decision and a `has_seg` story, and it is not a flag flip. Not
in this run; logged for the next one. DFC23 continues to pack `cls=None` /
`has_seg=False`, which §1.7's numbers confirm is honest.

### Q9 — Co-registration at scale → no assertion; the resize now says so
`rgb` and `dsm` share a geotransform on **1773 of 1773** scenes and all are
512 × 512, so the resize path never executes on DFC23 (§1.6). No bug, and an
assertion would be wrong — `pack_labeled` also serves `india_labeled`, where a
coarser height product is a legitimate input. **Code change, failing test first:**
the resize now prints both shapes and says to check the pairing, so it can never
be silent again. Test `test_a_resized_height_raster_says_so`.

### Q10 — End-to-end → confirmed, 3.1 GiB, 1506 + 266 tiles
```
$ python prepare_data.py --data_root <probe> --datasets dfc23 \
      --dfc23_dir .../DFC23/track2/train --dfc23_tile 512

[dfc23] 1773 rgb+nDSM pairs under .../track2/train (rgb/ + dsm/)
[dfc23] dfc23_g050: 1773 scenes @ 0.500 m -> 1507 train / 266 val (split by scene)
[dfc23/dfc23_g050/val] 266 pairs ... -> .../dfc23_g050/val
[dfc23/dfc23_g050/val] packed 266 tiles @ 512px / 0.5 m  seg=NO
[dfc23] HEIGHT CHECK dfc23_g050/val: median 0.00 m  p99 43.8 m  max 123.4 m  exactly-0 65.5 %  (over 266 tiles)
[dfc23]   >100 m: 0.029 % of border-32px pixels vs 0.009 % of interior
[dfc23/dfc23_g050/train] 1507 pairs ... -> .../dfc23_g050/train
[dfc23/dfc23_g050/train] packed 1506 tiles @ 512px / 0.5 m  seg=NO
[dfc23] HEIGHT CHECK dfc23_g050/train: median 0.00 m  p99 43.5 m  max 183.1 m  exactly-0 66.7 %  (over 1506 tiles)
[dfc23]   >100 m: 0.033 % of border-32px pixels vs 0.058 % of interior

prepared stores:
  dfc23_g050/train: 1506 tiles @ 512px / 0.5 m  (2.6 GiB)
  dfc23_g050/val: 266 tiles @ 512px / 0.5 m  (0.5 GiB)
```
41 s wall. **One GSD family**, as Q3 predicted. **1506 of 1507** train scenes
packed — the missing one is `SV_Berlin_52.4902_13.5090`, dropped by the Q6 fix
for being 53 % black. No scene leak between the splits (verified on the packed
stems). The train `HEIGHT CHECK` border/interior line now reads **0.033 % vs
0.058 %** and does not raise `[!]`, which is the whole-split truth from §1.4;
before the Q13 fix the same store printed `[!] concentrated at the tile border`.
The val store's 0.029 vs 0.009 is a 266-tile subset artefact — it is a different,
smaller n, and §1.4 is the number to trust. **Settled flags for the run:**
`--dfc23_dir .../track2/train --dfc23_tile 512 --dfc23_val_frac 0.15
--dfc23_max_height_m 150`, no `--dfc23_absolute_dsm` (median is 0.00 m, so these
are nDSMs).

---

## 4. Questions the data raised that this file did not anticipate

### Q11 — The top-level `rgb/ dsm/ sar/` are a full duplicate of `track2/train`
Not "curated sample triplets (Portsmouth, New York x2, New Delhi)" as the dataset
`README.md` says: 1773 files each, identical stem sets to `track2/train`, and
md5-identical on the files spot-checked. That is **~5.5 GB of the 12 GB**.
`_dfc23_pairs` would accept `<root>/` itself and pack exactly the same 1773
scenes, so there is no correctness hazard, only disk. **Decision:** point
`--dfc23_dir` at `track2/train` (Q1) and fix the dataset README's Contents
section; deleting the duplicate is a call for whoever owns the Kaggle upload,
not for this audit.

### Q12 — DFC23's own val and test splits cannot validate anything
`track2/val` (579) and `track2_test_data` (605) ship rgb + sar and **no nDSM**,
and their stems (`P_XXXX`) carry no city. So the DFC23 val number this project
reports is a **scene-level holdout of train** (266 scenes), not DFC23's val
split, and the Delhi-only figure rests on **20 held-out scenes** (Q2). The 1184
unlabelled val+test scenes are, however, exactly what `pack_unlabeled` and the
mean-teacher branch want — real satellite radiometry over 11 cities with no
labels needed. **Open:** whether to pack them as an unlabelled source. Not
decided here; it changes the consistency branch, not a DFC23 flag.

### Q13 — `_dfc23_height_check` was drawing its verdict from the first 64 tiles
The border/interior line is a prevalence claim, and it was computed over
`min(len(st), 64)` tiles taken from the **front** of the store. On the real
1506-tile train store that slice caught `GF2_NewDelhi_28.5557_77.1194` and almost
none of the Rio / New York high-rise scenes, and printed
`[!] concentrated at the tile border — stereo blunders, not buildings. Consider
--dfc23_max_height_m` — **the opposite of the whole-split answer**, and the exact
input that would have put a 100 m ceiling on by default and deleted 211,925 px of
real target. **Code change, failing test first:** the check now reads every tile
in the store (strided only above 4096, which no DFC23 store reaches) and prints
the n it used. Test
`test_the_height_check_samples_the_whole_store_not_just_the_front`. A strided
sample is not enough either — the five tallest Rio tiles hold 137k of the 214k
pixels above 100 m, so any subsample that misses them moves the verdict by 5×.

### Q14 — 0.5 m Gaofen-2 contradicts the code's own premise
`prepare_dfc23`'s docstring and `test_differing_gsds_are_packed_into_separate_
stores` both rest on "DFC23 optical is SuperView-1 at 0.5 m *and* Gaofen-2 at
0.8 m". In this data all 62 Gaofen-2 scenes are 0.5 m. The code is right to
group by *measured* GSD and the test is right to test that it can, but the
premise as stated is wrong for this release. **Open:** whether another DFC23
release does ship 0.8 m Gaofen-2, in which case `--dfc23_gsd` and the family
split matter and the store names change. Left as documentation, no code change —
the measured grouping handles either case.

### Q15 — 2,765 px of blunder survive, in two named scenes
With `--dfc23_max_height_m 150` (Q4), `GF2_NewDelhi_28.5557_77.1194` keeps 1,412
px of 100–150 m ribbon and `GF2_NewDelhi_28.5557_77.1210` keeps all 150 px of its
54–90 m ribbon. That is 0.0006 % of the split, but `StratumBalancer`
(`stratum_balance_beta 0.7`, clip 8) gives the tallest stratum the largest loss
weight in the batch, so the effective weight is higher than the pixel count.
**Open:** the clean fix is a scene denylist (`--dfc23_exclude <stem>,<stem>`)
rather than any global height rule, since the discriminator is ribbon-vs-blob
(Q5) and only two scenes are affected. Not implemented — it is new surface before
a 960-minute run, and 0.0006 % is a defensible thing to carry. Revisit if the
run's tall-bias metrics move the wrong way.

---

## 5. Reproducing this

Needs `rasterio` and `numpy`; this machine also needed `pytest`,
`python-multipart`, `onnx`, `onnxruntime` and `transformers` before the suite ran
whole (`pip install pytest rasterio python-multipart onnx onnxruntime
transformers "numpy<2"`).

```bash
cd <repo>/FineTunning/V4_Kaggle
python -m pytest -q                      # 156 now (152 + Q6 ×2, Q9 ×1, Q13 ×1)
python prepare_data.py --data_root /tmp/dfc23_probe --datasets dfc23 \
       --dfc23_dir <.../DFC23/track2/train> --dfc23_tile 512
```

Q4–Q7 were measured by reading all 1773 `dsm/` rasters and their `rgb/`
counterparts directly (~37 s, 16 processes), not through the packer, so the
numbers in §1 are independent of the packer's own sampling.
