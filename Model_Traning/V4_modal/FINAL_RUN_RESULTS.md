# DepthWizard v5 final run — results

_Written 2026-09-29 20:46 UTC by `dw-report` on Modal, from `run.log` on the `depthwizard-results` Volume (env `gpu`)._

## Verdict

> Finished: `DONE — best centre-crop val RMSE 2.481 m (108 min)`

- **best.pt = epoch 7, selection score 2.481 m** (epoch 1: 2.717 m, so 0.236 m lower).
- mvs3dm val RMSE at best: **1.545 m** (start checkpoint resume-v4-1.6: 1.615 m).
- neon forest RMSE at best: **4.11 m** (epoch 1: 4.61 m).
- GAMUS urban at best: 3.26 m (epoch 1: 3.30 m) — the runbook's guard is 'no worse than ~0.1 m'.

_Selection score = forested + sparse RMSE averaged over the neon and mvs3dm **val** sets (the number best.pt is chosen on). Lower is better._

## Every eval (validation sets, metres)

| epoch | select | mvs3dm RMSE | mvs3dm fore / spar | neon RMSE | neon fore / spar | gamus RMSE | gamus urban | us3d RMSE | grad mvs3dm / neon |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2.717 | 1.602 | 2.02 / 0.91 | 3.861 | 4.61 / 0.75 | 2.937 | 3.30 | 3.953 | 0.23 / 0.66 |
| 2 | 2.598 | 1.580 | 1.99 / 0.88 | 3.647 | 4.35 / 0.71 | 2.944 | 3.30 | 3.899 | 0.23 / 0.65 |
| 3 | 2.541 | 1.562 | 1.97 / 0.87 | 3.549 | 4.24 / 0.69 | 2.913 | 3.28 | 3.841 | 0.23 / 0.65 |
| 4 | 2.512 | 1.554 | 1.96 / 0.87 | 3.500 | 4.18 / 0.69 | 2.911 | 3.27 | 3.857 | 0.23 / 0.66 |
| 5 | 2.487 | 1.549 | 1.96 / 0.86 | 3.454 | 4.12 / 0.68 | 2.903 | 3.27 | 3.871 | 0.23 / 0.66 |
| 6 | 2.482 | 1.546 | 1.95 / 0.86 | 3.447 | 4.11 / 0.68 | 2.899 | 3.27 | 3.866 | 0.23 / 0.66 |
| 7 ★ | 2.481 | 1.545 | 1.95 / 0.86 | 3.444 | 4.11 / 0.68 | 2.893 | 3.26 | 3.872 | 0.23 / 0.66 |

Caveats: GAMUS is scored on 200 val tiles here (older runs: 400) and at the corrected 0.25 m GSD, so it is not comparable with older GAMUS numbers. `grad` is the sharpness ratio (goal: rise toward 0.5).

## Run facts

- H100 on Modal, 2 CPU cores / 8 GiB, batch 16 × accum 2, bf16, data read straight off the Volume.
- Mean train speed ~40.7 img/s; training clock reached 106 of 106 min.
- Two sessions: 18:48–19:59 UTC (cancelled after epoch 4's `last_full.pt` to extend the budget 90 → 106 min), then a full-state resume from 20:01 with the LR cosine held at 71 %.
- `[resume] continuing at epoch 5, best 2.512 m, 64 min of --max_minutes 106 used; budget changed, cosine held at 71 % (saved) instead of restarting from 61 %`
- GPU: 459 samples (every 15 s): mean 88.5 %, p10 80 %, max memory 46.7 GB

## Files

In `depthwizard-results:/v5_final_forest/`: `best.pt` (the model to use), `last_full.pt` (resume state), `run.log`, `metrics.json`, `gpu_util.csv`, `config.json`, `sizing.json`.

```bash
modal volume ls  -e gpu depthwizard-results /v5_final_forest
modal volume get -e gpu depthwizard-results /v5_final_forest/RESULTS.md .
modal volume get -e gpu depthwizard-results /v5_final_forest/metrics.json .
# best.pt is ~2.5 GB — the laptop disk is nearly full, fetch only if needed
```

## Not done yet (skipped on the paid H100 to save credit)

1. **Held-out test, start vs final** — `eval_test.py` on neon / mvs3dm / gamus **test** for both `prev/best.pt` (resume-v4-1.6) and this `best.pt`, same tiles (what `final_h100.sh test` does). This is the fair answer to 'did GAMUS regress?'. ~20–30 min on an L4 (~$0.80/h) or free on Kaggle.
2. `validation_report.html` (landscape gallery, strips) — judge crowns / canopy height by eye.
3. ONNX export for the serving stack.
