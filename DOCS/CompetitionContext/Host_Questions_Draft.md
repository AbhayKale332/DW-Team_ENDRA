# Draft: questions for the hosts (to post on issue #1)

**Status: draft, not posted.** A team member posts it on
[IMG-PROCESS-SAC/SIH-DepthWizard-2026#1](https://github.com/IMG-PROCESS-SAC/SIH-DepthWizard-2026/issues/1).
Code that depends on the answers:

| Question | Code that changes with the answer | Current default |
|---|---|---|
| Q1 | `--detail-gain` (`infer/predict.py`, `geo/calibrate.py`); which of `per_pixel` / `per_30m_cell` in `eval/judge_proxy.py` we optimise | `--detail-gain 1.0`: best if they aggregate our DSM to 30 m |
| Q2 | `--anchor-dem`, `--out-datum` (`infer/predict.py`, `geo/dem.py`) | COP30, EGM2008 |
| Q3 | NoData handling in `write_outputs` | NaN written as NoData |
| Q4 | `dwdata/scene_io.py` band selection | NRSC product → bands 3,2,1; 3-band file → its ColorInterp |
| Q5 | windowed inference threshold (`serve/app.py windowed_mp`) | row-band streaming above 40 MP |

Q1–Q3 are the ones the plan needs (V5 plan A6). Q4 and Q5 are optional: post
them only if the thread is still active.

---

**Text to post:**

> Thank you for the answers so far. A few follow-ups on the absolute-DSM
> accuracy metric, because they change how we should calibrate the output:
>
> 1. **Grid and resampling.** Our DSM is at the image's 0.6 m grid and the
>    reference DEM is 30 m. Will you (a) aggregate our DSM onto the 30 m DEM
>    grid, for example by block mean, or (b) resample the DEM onto our 0.6 m grid
>    and compare pixel by pixel? If (b), which resampler (nearest, bilinear,
>    cubic)?
> 2. **Reference DEM and vertical datum.** Will the reference be SRTM GL1,
>    Copernicus GLO-30, or both? SRTM heights are relative to EGM96 and
>    Copernicus heights to EGM2008. Over India the two geoids differ by up to
>    about 5 m (for example −3.7 m at Shimla and +4.7 m at Gangtok). Which datum
>    should our GeoTIFF heights use? We tag the output with a compound vertical
>    CRS (for example `EPSG:32644+3855`). Will you read that tag, or assume a
>    datum?
> 3. **NoData.** A Cartosat scene has a black NoData collar around the imaged
>    area. We write those pixels as NoData (NaN in a float32 GeoTIFF). Are
>    NoData pixels excluded from RMSE, MAE and correlation?
> 4. *(optional)* **Input format.** Will the evaluation GeoTIFFs be NRSC
>    products (separate `BAND1..4.tif` files plus `BAND_META.txt`, 11-bit UInt16,
>    band order B, G, R, NIR), like the Bhoonidhi MERGED sample, or pre-made
>    3-band RGB GeoTIFFs?
> 5. *(optional)* **Scene size.** Full scenes (about 20,000 × 20,000 px at
>    0.6 m) or subsets?

---

Background for whoever posts it: `CompetitionContext/V5_Research_Additions.md`
§2 (datums and geoid numbers) and §3.1 (why the answer to Q1 moves
`detail_gain`).
