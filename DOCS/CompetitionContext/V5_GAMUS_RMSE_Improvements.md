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

## What should change next

1. **Treat inference scale as the first lever.** The full GAMUS validation confirms D4 + 1.5× input improves global RMSE by 6.64%. Test plain 1.5× inference without D4, and a 1.25× alternative, on validation to find a cheaper or better tail-height tradeoff. These variants are untested. Do not make this a global default: US3D worsened from 3.698 to 4.050 m in the pilot.
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
