# GAMUS RMSE investigation — 2026-10-03

**Confirmed improvement:** D4 test-time augmentation with 1.5× input scaling reduces GAMUS validation RMSE from **2.774581 to 2.590420 m** across all 859 tiles: **0.184160 m (6.64%) lower**, with weights fixed. It increases runtime from 67.78 s to 960.38 s (14.17×) and worsens error on true tall pixels. Results are from deterministic 512 px centre crops; held-out test and native sliding evaluation were not run.

## Prior results and comparison protocol

The previous 2026-10-02 chat and its saved [readout report](V5_Readout_Probe_Results.md) were recovered. The model is epoch 7 of `v5_final_forest/best.pt`. The headline **2.480854 m** in the training report is the NEON/MVS3DM forested+sparse selection score, not GAMUS RMSE.

| Existing evaluation | GAMUS RMSE |
|---|---:|
| Epoch 7 training evaluation, 200 seeded val tiles | 2.892936 m |
| Readout pilot baseline, 128 seeded val tiles | 2.882592 m |
| Readout pilot D4 + 1.5× input, same 128 tiles | 2.524681 m |
| Full readout confirmation baseline, all 859 val tiles | 2.774581 m |
| Full confirmation, 50% Head B blend | 2.771011 m |
| Full confirmation, D4 + 1.5× input, all 859 val tiles | **2.590420 m** |

Changing the cohort explains the different baseline values; these are not successive training improvements. Sources: [training metrics](../../Model_Traning/v5/outputs/v5/modal/metrics.json), [pilot](../../Model_Traning/v5/outputs/v5/readout_probe/modal/probe_metrics.json), [full confirmation](../../Model_Traning/v5/outputs/v5/readout_confirmation/modal/probe_metrics.json).

The new run compares baseline and the already-selected pilot zoom candidate on all 859 GAMUS validation tiles. Both use the same 512 px deterministic centre-crop path, corrected source GSD 0.25 m, valid pixels, preprocessing, checkpoint, BF16 and batch size 2. The 1.5× candidate resizes the normalized input to 768 px, averages eight D4 transforms, and returns predictions to the same 512 px target grid. It changes neither target heights nor evaluated region. This measures the existing centre-crop protocol, not native-resolution sliding evaluation or held-out test performance. [Evaluator](../../Model_Traning/v5/tools/readout_probe.py), [TTA](../../Model_Traning/v5/models/tta.py).

## Full GAMUS validation result

| Metric | Baseline | D4 + 1.5× | Change |
|---|---:|---:|---:|
| Global RMSE | 2.774581 m | **2.590420 m** | **−0.184160 m (−6.64%)** |
| Global MAE | 1.345533 m | 1.263796 m | −0.081737 m |
| Urban RMSE (465 tiles) | 3.211026 m | 3.061205 m | −0.149821 m |
| Sparse RMSE (99 tiles) | 2.210237 m | 1.253103 m | −0.957134 m |
| Forest RMSE (295 tiles) | 2.126347 m | 2.051557 m | −0.074789 m |
| Building RMSE | 3.322058 m | 2.919535 m | −0.402523 m |
| Tree RMSE | 4.163721 m | 4.058267 m | −0.105454 m |
| Edge-band RMSE | 3.752002 m | 3.596419 m | −0.155583 m |
| 30 m nDSM block RMSE | 1.285452 m | 1.115329 m | −0.170123 m |
| True-height ≥15 m RMSE | 5.792292 m | **5.979674 m** | **+0.187382 m** |
| True-height ≥15 m bias (prediction − GT) | −2.091797 m | −3.138403 m | 1.046606 m more negative |
| Runtime | 67.78 s | 960.38 s | 14.17× |

Both candidates scored the same **222,847,123 valid pixels**. The zoom gain is largest on sparse tiles and buildings. Tree RMSE improves by only 0.105 m; trees remain the main error contributor. The ≥15 m group gets worse, so this setting suits the global GAMUS RMSE metric but is a real compromise if tall-building heights matter most. The 30 m pooled metric is an nDSM check, not an absolute DSM score.

The original 128-tile pilot is a subset of the full set (indices and stems match). Subtracting its SSE from full-set SSE gives the remaining 731 tiles an aggregate-derived RMSE of **2.755093 m baseline vs 2.601842 m zoom**. This supports the gain beyond the pilot, though it is not an independent test and no per-tile differences were saved for confidence intervals.

Raw completed artifacts: [full JSON](../../Model_Traning/v5/outputs/v5/gamus_zoom_confirmation/modal/probe_metrics.json), [Markdown table](../../Model_Traning/v5/outputs/v5/gamus_zoom_confirmation/modal/PROBE.md), [run provenance](../../Model_Traning/v5/outputs/v5/gamus_zoom_confirmation/modal/run.json). The held-out test split was preserved.

## Further scaling comparison — completed

The authorized follow-up evaluated plain 1.25×, plain 1.5×, and D4 + 1.25× on all 859 GAMUS validation tiles. Checkpoint path, indices, stems, valid pixel counts, true-height/class/landscape cohort counts, and preprocessing match the prior comparison. The new baseline global metrics reproduce the old ones exactly.

| Inference | RMSE (m) | MAE (m) | Runtime (s) | True ≥15 m RMSE (m) |
|---|---:|---:|---:|---:|
| Single pass | 2.774581 | 1.345533 | 83.85 | 5.792292 |
| Plain 1.25× | 2.706537 | 1.322277 | 105.43 | 5.973975 |
| Plain 1.5× | 2.621131 | 1.272752 | 146.82 | 5.999211 |
| D4 + 1.25× | 2.652125 | 1.307279 | 637.56 | 5.932389 |
| D4 + 1.5× (previous) | 2.590420 | 1.263796 | 960.38 | 5.979674 |

**D4 + 1.5× remains the best confirmed global GAMUS setting (2.590420 m). Plain 1.5× is the faster alternative (2.621131 m):** it retains 83.32% of the prior RMSE gain and is only 0.030711 m higher. Its recorded runtime is 146.82 s versus the earlier 960.38 s D4 result. Timing is observed L4 evaluation duration, not a repeated throughput benchmark; the old baseline took 67.78 s, while the new baseline took 83.85 s.

Plain 1.5× gives urban/sparse/forested RMSE of 3.092189 / 1.399414 / 2.060940 m, building RMSE 3.005126 m, and tree RMSE 4.089428 m. Plain 1.25× worsens forested and tree error versus baseline. Every scaled setting worsens true-height ≥15 m RMSE; D4 + 1.25× has the least tall RMSE increase among these scaled candidates, but takes 637.56 s and has higher global RMSE than plain 1.5×.

The new run is `dw-gamus-confirmation` call `fc-01M40SBFSEMCJHSTWB082V7XTR`, one L4, 2 CPU cores, 8192 MiB RAM, 40-minute timeout, output volume `depthwizard-results:/v5_gamus_scale_comparison_20261003/`. No training or held-out test evaluation was performed. Per-tile error vectors were not saved; no confidence interval or significance claim is made.

Completed evidence: [comparison report](../../Model_Traning/v5/outputs/v5/gamus_scale_comparison/modal/COMPARISON.md), [new raw metrics](../../Model_Traning/v5/outputs/v5/gamus_scale_comparison/modal/probe_metrics.json), [combined metrics and cohort checks](../../Model_Traning/v5/outputs/v5/gamus_scale_comparison/modal/comparison.json), [run provenance](../../Model_Traning/v5/outputs/v5/gamus_scale_comparison/modal/run.json). The documentation landscape page now leads with a separate GAMUS-only score/table and includes the scaling comparison; four-dataset pooled accuracy appears as supporting evidence.

## Hugging Face Blackwell runtime — completed

The private Space was tested on its actual RTX PRO 6000 Blackwell **48 GB MIG partition** with the confirmed epoch-7 checkpoint. Its old v3 implementation silently missed six decoder tensors and ignored twelve checkpoint tensors. The model implementation and loader now match v5; boot confirms all 925 tensors loaded, zero missing/unexpected keys, and the detail branch enabled.

For a 1024×1024 scene at a declared 0.5 m/pixel (9 canonical tiles), two complete requests per mode took: **single pass 26.91–33.08 s**, **plain 1.5× 26.81–28.34 s**, **native D4 28.50–32.50 s**, and **D4 + 1.5× 29.98–30.67 s**. These include queue/allocation, model transfer, CPU objects/mesh/viewer/artifacts, and response; automatic downloads were disabled. Warm `predict_scene` timings inside one lease were 0.211 / 0.475 / 1.568 / 3.689 s. The overlapping complete-request ranges do not prove equal latency; compute cost is higher, and larger scenes or other GSD may change the balance.

All 28 local tests, 16 live inference cases, and 8 complete requests passed. Scaled modes are available through `/predict_scaled`; the visible checkbox selects native D4. The timing imagery has no ground-truth heights, so a full GAMUS sliding-window accuracy evaluation remains necessary to establish deployed-pipeline accuracy. Existing centre-crop scores above retain their original protocol. [Full report and evidence](../../Model_Traning/v5/outputs/v5/hf_blackwell_benchmark/REPORT.md).

## What should change next

1. **Choose inference according to the judging objective.** D4 + 1.5× remains the best confirmed global GAMUS setting, 6.64% below baseline. Plain 1.5× is the cheaper alternative, 5.53% below baseline with 83.32% of the D4 gain retained. The 1.25× variants did not lower the incumbent RMSE. All scaled candidates worsen true-tall RMSE. Do not make scaling a global default: US3D worsened from 3.698 to 4.050 m in the pilot.
2. **Use the intended validation objective before another training run.** GAMUS should be the first dataset and `select_on` should be empty to select its global RMSE. `--select_on gamus` still selects only forested+sparse GAMUS tiles and excludes urban tiles. Start a separate run from weights-only `best.pt`; a full-state resume restores the old forest selection score and optimizer schedule. Keep NEON/MVS3DM/US3D validation as regression checks. See the exact override fragment in the [audit](../../Model_Traning/v5/outputs/v5/readout_confirmation/modal/RESULTS_AUDIT.md).
3. **Investigate GAMUS trees before adding a more complex model.** On full baseline validation, trees are 22.0% of valid pixels but 49.6% of squared error; buildings contribute 24.7%. During the previous seven-epoch run, GAMUS tree RMSE barely changed (4.229 → 4.215 m on the fixed 200-tile cohort). Inspect the largest tree errors and RGB/height alignment, then try one bounded GAMUS-focused fine-tune with replay from other sources. A small fused-head squared-error term is a testable hypothesis because the present regression objective uses L1/SILog plus spatial penalties rather than unweighted MSE. No benefit is established yet; change one factor at a time.

A lower-rate restart is also untested. The current checkpoint already uses EMA (`ema_decay=0.9995`); adding another averaging method alone is not an established fix. Keep the starting model as an incumbent and compare it using exactly the same validation protocol before promoting any new weights. [Training config](../../Model_Traning/v5/outputs/v5/modal/metrics.json), [losses](../../Model_Traning/v5/models/losses.py), [checkpoint selection](../../Model_Traning/v5/train.py).

## Changes the evidence does not support

- **A global height offset:** full-validation bias is only +0.0282 m. Even the same-set optimum constant correction would lower RMSE by just 0.000143 m. This is an analytic optimistic bound, not a deployable calibration result.
- **A blanket upward correction for tall predictions:** GT-conditioned and prediction-conditioned biases have opposite signs. The GT-tall subset cannot be identified from ground truth at inference.
- **Single-mode Head B or stronger fusion:** single-mode worsens the pilot; 50:50 fusion buys only 0.00357 m on full GAMUS and raises flat-ground bias.
- **More edge weight on faith:** the prior run already used `w_grad=1.0`, and higher gradient energy from single-mode readout came with worse edge RMSE.

Evidence and calculations are detailed in the [existing-results audit](../../Model_Traning/v5/outputs/v5/readout_confirmation/modal/RESULTS_AUDIT.md). The official [GAMUS class mapping](https://github.com/EarthNets/RSI-MMSegmentation) also matches the local ground/low-vegetation/building/water/road/tree diagnostic labels.

## Reproduction and scope

- New runner: [gamus_confirmation_modal.py](../../Model_Traning/V4_modal/gamus_confirmation_modal.py).
- Modal app: `dw-gamus-confirmation`, environment `gpu`; call `fc-01M4066HPF224NG0JCY2MJFGE4`.
- Output volume: `depthwizard-results:/v5_gamus_zoom_confirmation_20261003/`.
- Local output: [gamus_zoom_confirmation/modal](../../Model_Traning/v5/outputs/v5/gamus_zoom_confirmation/modal/).
- Compute requested: one L4, 2 CPU cores, 8 GiB RAM, 40-minute function timeout.
- Existing readout CPU preflight passed; all 925 checkpoint tensors loaded with zero missing/unexpected keys. The new baseline exactly reproduced the earlier full-validation score.
- Original training/evaluation implementation and checkpoint remain unchanged; only a separate Modal runner and investigation artifacts were added. Held-out test was not evaluated.
- Skills used: `research`, `ponytail`, and the repository's [Modal skill](../../.claude/skills/modal-ai/SKILL.md). Existing authenticated Modal credentials sufficed.
