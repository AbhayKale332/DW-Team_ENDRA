# Extra training data for v5 without DFC

Context: `Notebook/kaggle_train_gamus_synrs3d.ipynb` trains on GAMUS + SynRS3D only.
It drops DFC23 and also `india_labeled`, which is DFC23's 131 New Delhi scenes.
These are the candidates for adding supervision back. Each gets an entry in that
notebook's `SOURCES` dict once it exists as a packed store.

**Provenance.** The GeoNRW section was checked against this repo on 2026-09-25.
The other sections are from general knowledge and were not verified. Confirm
licence, GSD and label format before building a loader for any of them.

## Summary

| Dataset | GSD | Height label | Code in repo | Effort | Verdict |
|---|---|---|---|---|---|
| GeoNRW | 1.0 m aerial | Absolute DEM → nDSM by a crude ground filter | Yes | Low | Try first, at low weight |
| NAIP + USGS 3DEP | 0.6 m aerial | LiDAR DSM − DTM (you build it) | No | High | Best GSD match to Cartosat |
| FLAIR #1 (IGN) | 0.2 m aerial | Height band in the patch | No | Medium | Check the band first |
| ISPRS Potsdam / Vaihingen | 5–9 cm aerial | Photogrammetric nDSM | No | Low–medium | Too small to matter much |
| US3D (DFC19) | 0.3 m WorldView-3 | LiDAR AGL | No | Medium | Closest real satellite domain, but it's DFC |

## GeoNRW (already supported)

- **What's in the repo:**
  - `prepare_data.py --datasets geonrw` downloads `nrw_dataset.tar.gz` (~32 GB)
    from the HF repo `torchgeo/geonrw` (`--geonrw_repo`) and packs 1000 px tiles
    at 1.0 m (`GEONRW_GSD_M`).
  - It holds out the cities `duesseldorf`, `herne` and `neuss`, and packs train only.
  - `--geonrw_max` defaults to 2500 tiles, about 15 GB at ~6 MB/tile.
  - `run_kaggle.sh link` recognises a `depthwizard-geonrw` store.
  - `config.SHARED_SPACE_SOURCES` remaps its class ids to GAMUS's ids at load time.
- **The catch is label quality.** GeoNRW ships an absolute DEM, not heights above
  ground. `_ndsm_from_dem` (`prepare_data.py:666`) estimates the ground as a
  120 px minimum filter plus a Gaussian blur, subtracts it and clips at 0.
  - On slopes this puts flat ground above 0 m, and `flat_bias` is already the
    stuck term (see the v4 findings).
  - The code's own docstring calls it "auxiliary-only".
- **How to add it:**
  1. Pack it on Modal, not Kaggle: the 32 GB tar plus its extraction won't fit on
     Kaggle's disk.
     `python prepare_data.py --datasets geonrw --geonrw_max 2500`
  2. Publish it as `depthwizard-geonrw` with `V4_modal/03_publish_kaggle.ipynb`.
  3. Add `"geonrw": 0.5` to `SOURCES`.
- **What to watch:** compare `flat_bias` on `gamus/val` against the run without
  GeoNRW. If it moves the wrong way, drop GeoNRW or build a proper nDSM from a real
  DTM (NRW publishes open DGM1 terrain models).

## NAIP + USGS 3DEP LiDAR (build your own)

- NAIP US aerial RGB(+NIR) is 0.6 m in recent years, the same GSD as Cartosat-2S
  MERGED. 3DEP LiDAR provides both a DSM (first return) and a DTM (ground), so
  DSM − DTM is a real height map, not an estimate.
- Both are public domain and on AWS / Planetary Computer.
- Cost: a new `prepare_*` function, co-registration and a choice of US cities.
  Mismatched capture dates between image and LiDAR (new buildings, cut trees)
  need a change mask or careful scene selection.

## FLAIR #1 (IGN France)

- 0.2 m RGB + NIR patches (512 px) with a fifth band of height above ground, plus
  land-cover labels.
- **Check before use:** how the height band was made (LiDAR vs photogrammetry),
  and its units and scaling (it may be quantised to uint8). A quantised band caps
  what the model can learn about tall buildings.

## ISPRS Potsdam / Vaihingen

- Very high resolution (5 cm / 9 cm) with photogrammetric nDSMs, about 70 large
  tiles in total.
- Good for sharp building edges once downsampled to ~0.5 m, but after
  downsampling it's a small dataset. Use a low sampler weight so it isn't
  oversampled into memorisation.

## US3D (IEEE DFC 2019)

- WorldView-3 0.3 m satellite imagery, Jacksonville and Omaha, with LiDAR
  above-ground height. It's the closest real satellite domain to Cartosat of
  anything here.
- It's DFC data (2019, not DFC23), so it's excluded under the current "no DFC"
  rule. Add it only if that rule meant DFC23 specifically.

## Not candidates

- **india_labeled:** it's DFC23 New Delhi. The only Indian height data available,
  but excluded for the same reason as DFC23.
- **india_unlabeled:** no labels. Only useful for the mean-teacher consistency
  branch, and that store doesn't exist.
