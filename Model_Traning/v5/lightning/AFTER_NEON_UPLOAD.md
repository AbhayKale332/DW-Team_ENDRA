# After the NEON upload: running the final H100 fine-tune

This is the runbook for the step after the NEON agent reports that
`abhaydkale232/depthwizard-neon` is on Kaggle. Budget about 6 hours of H100 time:

| Step | Time |
|---|---|
| fetch | 30–40 min |
| sweep | 15–20 min |
| smoke | ~8 min |
| train | 330 min |
| test and package | ~20 min |

Everything large lives on the Lightning box. Nothing gets downloaded to the laptop
except the final results, and only the parts you choose (see step 8).

---

## 1. Check the NEON dataset before spending H100 time

Do this from any machine with the kaggle CLI (the NEON CPU Studio is fine):

```bash
kaggle datasets files abhaydkale232/depthwizard-neon --page-size 200
```

Check each of these:

- [ ] Files sit under `train/`, `val/` and `test/`. There is **no** `train/train/` nesting.
- [ ] Each split has `index.json`, `stretch_bounds_2_98.npy` and `shard_NNN_{rgb,hgt,cls,val}.npy`.
- [ ] The agent's report says `tile_px 1024`, `gsd_m 0.5` and `has_seg false`.
- [ ] Forest sites show a real share of pixels above 3 m, and the sparse sites less. If the > 3 m share is ~0 everywhere, the DSM − DTM label is broken, so don't train on it.
- [ ] Registration dropped fewer than about 1/3 of the tiles at every site.
- [ ] The Kaggle file names and sizes match the local zips; the agent's step (d) checks this.

**Done 2026-09-29** (version 1: 1846 / 118 / 232 tiles). `tools/eda_neon.py`
(`--store <root>/neon --out <dir>`) found two things the run has to handle.
Both are in the code now:

- **The landscape rule calls NEON's closed forests "urban".** It divides
  roughness by mean height, and a 30 m canopy is smooth for its height. That put
  1024 of 1846 train tiles in "urban", including every tile at HARV, BART, GRSM,
  MLBS and TALL. NEON's CHM has no buildings, so `final_flags.py` sets
  `landscape_no_urban_sources neon`, which maps urban to forested for NEON in the
  sampler boost, the per-landscape metrics and `select_on`.
- **About 0.01 % of pixels are spikes, not trees.** Examples: a 74 m canyon wall
  at MOAB, 113 m wire remnants at LAJA, the SERC flux tower, and single-cell
  spikes. `fetch` runs `pack_neon.py clean`, a per-site cap of 1.5 × the median
  tile p99.9 + 10 m. That keeps WREF's 80 m firs and TEAK / SOAP's 60–70 m pines.

It also confirmed the rest:
- every 2 × 2 label block is one lidar cell on the even grid, and `cls` is all
  unlabelled;
- there is no split leakage (stems, km tiles or image hashes);
- the image sits a consistent ~1 px (0.5 m) from the lidar. That is inside one
  label cell and fits sunlit crown sides, not misregistration, so it is not
  corrected.

If any box fails, fix NEON first. The run also works **without** NEON, because
`fetch` skips a missing `depthwizard-neon`, but then you lose the forest data.

## 2. Get the code onto Lightning

The final-run code is uncommitted on the laptop.

1. Commit and push it, using the commit message from the session.
2. On the Studio, clone or pull the repo. The folder you work in is the v5 directory
   (`Model_Traning/v5`), where `train.py` and `lightning/` live:

```bash
cd /teamspace/studios/this_studio
git clone https://<GITHUB_PAT>@github.com/<you>/DepthWizard.git   # or: cd DepthWizard && git pull
cd DepthWizard/Model_Traning/v5
ls lightning/final_h100.sh tools/h100_sweep.py tools/kaggle_fetch.py   # all three must exist
```

## 3. Switch the Studio to an H100

- Pick the H100 option with the **most vCPUs** on offer. The sweep will tell you
  if the data loader, not the GPU, is the bottleneck.
- Switching the machine type **wipes `/tmp`**. That's fine: `fetch` re-downloads
  everything from Kaggle.
- Don't copy datasets into `/teamspace`. It must stay under 50 GB.

## 4. Credentials and the checkpoint to start from

```bash
# Kaggle API key (kaggle.com -> Settings -> Create New Token)
mkdir -p ~/.kaggle && nano ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json

# DINOv3-SAT is gated on Hugging Face
export HF_TOKEN=hf_...

# The Kaggle notebook whose Output has outputs/v5_probe_v4init_mvs3dm/best.pt
# (the resume-v4-1.6 run). Take the slug from its URL: kaggle.com/code/abhaydkale232/<slug>
export PREV_KERNEL=abhaydkale232/<slug>
```

`kaggle kernels output` reads that notebook's **latest** version. If you ran the
notebook again after resume-v4-1.6, its output no longer holds that `best.pt`. In
that case:

1. Download `best.pt` from the right version's Output tab.
2. Upload it to the Studio.
3. Point the script at it instead:

```bash
export PREV_PATH=/teamspace/studios/this_studio/prev/best.pt   # PREV_KERNEL is then not needed
```

The `export` lines only last for the shell you type them in. If you open a new
terminal, export them again.

## 5. First pass: check, fetch, sweep, smoke

Run these first and read the output before starting the 5.5 h run:

```bash
bash lightning/final_h100.sh check fetch sweep smoke --background
tail -f /teamspace/studios/this_studio/dw_final/final.log
```

**`check`:**
- The GPU is listed as H100.
- `bf16=True`.
- `cores=` shows how many CPUs the loader gets.
- The dry-run prints how many GB it will fetch against the free space on `/tmp`. It
  stops here if the space isn't there; see Troubleshooting.
- pytest passes.

**`fetch`:**
- A line per dataset, then `store GSDs match the measured values`.
- `/tmp/dwdata` shows `mvs3dm gamus us3d synrs3d_g05 synrs3d_g1 neon`, and **no**
  `dfc23` or `india_labeled` (the script refuses to go on if either appears).
- `neon: per-site height cap`, then `[clean] neon/train: N px above the site cap
  dropped` and the same for val and test. Measured on v1: train 81 583, val
  16 423 and test 724 px (0.0045 % of the labels). About 80 k of those are MOAB
  canyon walls (cap 19 m); the rest are mostly LAJA, GUAN, SRER, JORN and CLBJ.
- `[ckpt] ... {'epoch': ..., 'encoder_included': True, 'bin_max_m': 120.0, ...}`.
  If `encoder_included` is `False`, it's the wrong file.

**`sweep`:**
- One line per configuration, then the choice and a verdict, for example:

  ```
  [sweep] b= 32 acc=1 compile=1 ...  52.3 img/s  wait=  1%  peak= 61.0 GB  util= 97.1 %  ...  [ok]
  [sweep] CHOSEN  b= 32 ...
  ```
- **What good looks like:** `wait` under 5 %, `util` 95 % or more, `peak` at or
  under 72 GB.
- A `CPU-BOUND` verdict means the loader is starving the GPU. Stop and move to an
  H100 with more vCPUs, then run `sweep` again with `FORCE_SWEEP=1`.
- The result is saved to `/teamspace/studios/this_studio/dw_final/sizing.json`,
  and later steps reuse it.

**`smoke`:** it must end with `DONE`. Look at the `[data]` lines in its output:
- `neon/train: ... seg=NO`.
- `[data] <store>/train landscape: urban=… sparse=… forested=… (x% of draws)`.
  Forested and sparse should hold a bigger share of the draws than of the tiles.
  For `neon/train`, expect `urban=0` and roughly `sparse≈430 forested≈1420`. A
  large `urban` count means `landscape_no_urban_sources` did not reach the run.
- Train lines carry `crs=` (the NEON coarse loss) and `wait=` values in the low
  single digits.

## 6. The real run

```bash
bash lightning/final_h100.sh train test package --background
tail -f /teamspace/studios/this_studio/dw_final/final.log
```

**During training, watch:**
- `img/s` within ~5 % of the sweep's figure, and `wait=` staying low.
- Every eval prints `select=… m` (the forest + sparse error). A new `best.pt` is
  saved when it drops.
- `nvidia-smi` in a second terminal. Utilisation is also logged every 15 s to
  `outputs/v5_final_forest/gpu_util.csv`.

**If the Studio stops or restarts mid-run:**
1. Export the variables from step 4 again.
2. Run `bash lightning/final_h100.sh fetch train test package --background`.
   `fetch` only re-downloads what `/tmp` lost, and `train` resumes from
   `last_full.pt`, which is saved every 3 epochs.

## 7. Judge the result, by eye first

Everything is in `/teamspace/studios/this_studio/dw_final/outputs/v5_final_forest/`.

1. **`validation_report.html`:** the landscape gallery, NEON forest and sparse
   tiles, and the MVS3DM and US3D strips. Compare against resume-v4-1.6's report
   in `outputs/v5/warm_start/`. You're looking for:
   - individual crowns instead of blobs
   - canopy height kept, not flattened to ground
   - isolated trees in open land that survive
   - roofs that are still clean
2. **`test_final_<src>/` vs `test_start_<src>/` and `start_vs_final_<src>.md`:** the
   same held-out tiles scored by the new model and by the checkpoint it started
   from, 16 image strips each.
3. **Numbers, as a sanity check only:**
   - `grad_ratio` should rise toward 0.5 (it was 0.23 on MVS3DM).
   - GAMUS `tree` delta1 (was 0.43) and the MVS3DM `forested` RMSE (was 2.07 m)
     should improve.
   - GAMUS `urban` and US3D should not get worse by more than ~0.1 m.
4. **`gpu_util_summary.txt`:** mean GPU utilisation should be in the 90s. If not,
   the log says whether the loader or eval ate the time.

## 8. Bring results home, carefully (the laptop disk is nearly full)

The zip is `/teamspace/studios/this_studio/dw_final/results_v5_final_forest.zip`.
It holds `best.pt` (~1–2 GB), the ONNX model, the report, metrics and strips.

- Check `df -h /` on the laptop first.
- If space is tight, download only `validation_report.html`, `metrics.json`,
  `run.log` and `start_vs_final_*.md` from the Studio's file browser.
- Keep `best.pt` and the ONNX file on Lightning or publish them to Kaggle, rather
  than on the laptop.

Then **stop the Studio**, because the H100 bills while it is idle. Losing `/tmp`
costs nothing, since Kaggle holds the data.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `check`: not enough disk on `/tmp` | Use an H100 option with a bigger local disk. Or drop `gamus` `test` from `DATASETS` in `final_h100.sh` (it is only used by the `test` step), and remove `gamus:test` from `test_sources` in `final_flags.py`. |
| `fetch`: 401 / 403 | Check `~/.kaggle/kaggle.json` and that it has `chmod 600`. The datasets are private to `abhaydkale232`. |
| `[ckpt] no outputs/v5_probe_v4init_mvs3dm/best.pt` | The notebook's latest version isn't the resume-v4-1.6 run. Use `PREV_PATH` (step 4). |
| `stale GSD in: gamus/...` | An old Kaggle version of gamus or us3d was fetched. Delete `/tmp/kin/datasets/abhaydkale232/depthwizard-gamus` and fetch again. |
| `sweep`: every config `failed` | Open the `log` path the sweep printed. It is usually `HF_TOKEN`, the checkpoint path, or a missing store. |
| `CPU-BOUND` verdict | Use more vCPUs. Rerun with `FORCE_SWEEP=1 bash lightning/final_h100.sh sweep`. |
| OOM during `train` | Handled: the script restarts from `last_full.pt` with the next-smaller batch it measured. |
| No `select=` line in the evals | Neither `neon` nor `mvs3dm` val has forested or sparse tiles. Selection then falls back to overall RMSE, which is still safe. |
| `train` crashed (not an OOM) | Read `outputs/v5_final_forest/train_console.log`, fix it, then run `train` again. It resumes from the last full-state checkpoint if one exists. |
