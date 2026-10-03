# V5 frozen-checkpoint evaluation on Modal

Completed 2026-10-02. Follow-up to `V5_Next_Run_Research.md`.

**Decision: retain epoch 7 `best.pt` and the learned fusion / mean bin readout.** The 50% Head B blend has effectively identical selection error on full validation, while raising flat-ground predictions. Single-mode readout worsens accuracy. A 1.5× zoom improves three datasets but substantially worsens US3D, so it should not become a global inference setting.

## Protocol and provenance

- Checkpoint: `depthwizard-results:/v5_final_forest/best.pt`, epoch 7, 2,504,595,420 bytes; all 925 tensors loaded with zero missing or unexpected keys.
- Compute: Modal L4, BF16, batch size 2, 2 CPU cores and 8 GiB requested RAM. Existing image, data volumes and `dw-hf` secret were reused. No Kaggle execution, Git commit, or training run was launched.
- Pilot: nine candidates on the same seed-42 sample of up to 128 **validation** tiles per source; all 118 NEON tiles were included.
- Confirmation: baseline and fixed 50% Head B fusion on **all 2,088 validation tiles** (820 MVS3DM, 291 US3D, 118 NEON, 859 GAMUS). Each tile contributes the existing deterministic 512 px centre crop, not a full native-tile/sliding prediction. The two candidates use identical valid pixels.
- `bias_m` always means **prediction minus GT**. Tall subsets use ≥15 m. Height strata conditioned on prediction are also saved, including the >20 m tail.
- The 30 m diagnostic averages prediction and GT into non-overlapping crop-aligned blocks, using each sample’s delivered GSD. Only complete blocks with every pixel valid count; tile margins and partly invalid blocks are excluded. All measured block sizes were 30 m.
- This is a **30 m nDSM diagnostic**, not the judges’ absolute DSM score. Terrain, vertical datum, DEM alignment and comparison against SRTM/COP30 are absent.
- NEON confirmation additionally scores 2× pooled predictions at its 1 m label resolution, from the same model forward. Pooled landscape metrics now use the multiplied GSD.
- `single_mode` keeps the descending basin around the highest-probability bin and renormalises its probabilities; the learned gate is recomputed from the resulting Head B height. The reported `b_std` retains the original distribution’s spread.
- D4-only and D4+1.5× zoom were compared separately to isolate the zoom effect from rotation/flip averaging. BF16/device rounding and the changed validation cohort mean these scores should not be treated as a before/after training gain against the old 2.481 m score.

## Full-validation bias diagnosis

| Source | Pixels | GT ≥15 m bias | Pred ≥15 m bias | GT ≥20 m bias | Pred ≥20 m bias | 30 m RMSE | 30 m bias | Valid 30 m blocks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mvs3dm | 199,566,244 | -6.189 | +0.836 | -8.833 | +2.204 | +0.112 | -0.030 | 14,673 |
| us3d | 76,114,083 | -5.543 | -0.570 | -9.683 | -0.747 | +2.696 | -0.344 | 16,577 |
| neon | 30,174,897 | -1.180 | +0.133 | -1.833 | +0.154 | +1.537 | +0.057 | 4,658 |
| gamus | 222,847,123 | -2.092 | +1.261 | -2.218 | +1.487 | +1.285 | +0.029 | 54,032 |

MVS3DM reproduces the low correlation inside the GT-tall subset (r = 0.135). Its −6.19 m GT-tall bias coexists with +0.84 m prediction-tall bias and only −0.03 m 30 m bias. This supports the uncertainty explanation and provides no evidence for a global upward height correction.

US3D retains a modest source-specific calibration error. Its GT ≥20 m error is much larger than its error conditioned on predicted height. Prediction-conditioned and pooled biases are useful diagnostics, but cannot alone prove that every building is calibrated: conditioning averages can hide subgroups, and pixels/30 m blocks are spatially dependent.

NEON and GAMUS have small positive 30 m mean bias. Their prediction-tall bias is also positive. A global loss change that pushes every tall prediction upward would therefore have opposing effects across sources.

## Full-validation 50% Head B confirmation

| Source | Baseline RMSE | B50 RMSE | RMSE change | Baseline flat bias | B50 flat bias | Baseline 30 m bias | B50 30 m bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| mvs3dm | 1.533125 | 1.531457 | -0.001667 | 0.355855 | 0.363971 | -0.029653 | -0.041229 |
| us3d | 3.899122 | 3.894451 | -0.004672 | 0.266779 | 0.327076 | -0.343596 | -0.288003 |
| neon | 3.443492 | 3.445071 | 0.001579 | 0.433582 | 0.450091 | 0.056720 | 0.063642 |
| gamus | 2.774581 | 2.771011 | -0.003569 | 0.446571 | 0.500795 | 0.029317 | 0.076924 |

Selection score: **baseline 2.474438228 m; B50 2.474437306 m**. This uses training’s selection formula: pixel-pool forested+sparse RMSE within each source, then average NEON’s 1 m pooled result with MVS3DM. The numerical difference is below one micrometre and has no practical meaning; the MVS3DM gain and NEON regression cancel. This is a full-validation cohort, not the historical capped cohort.

## Nine-candidate pilot: overall RMSE

| Candidate | MVS3DM | US3D | NEON | GAMUS |
|---|---:|---:|---:|---:|
| baseline | 1.529 | 3.698 | 3.443 | 2.883 |
| head_a | 1.536 | 3.686 | 3.443 | 2.892 |
| head_b | 1.527 | 3.709 | 3.458 | 2.875 |
| head_b_single_mode | 1.957 | 3.807 | 3.672 | 3.154 |
| fused_single_mode | 1.582 | 3.699 | 3.508 | 2.919 |
| fused_b50 | 1.527 | 3.692 | 3.445 | 2.878 |
| fused_single_mode_b50 | 1.636 | 3.716 | 3.507 | 2.969 |
| baseline_d4 | 1.497 | 3.746 | 3.401 | 2.819 |
| baseline_d4_zoom150 | 1.461 | 4.050 | 3.363 | 2.525 |

Single-mode fusion worsens edge RMSE on all four pilot datasets. On MVS3DM, its gradient ratio rises from 0.233 to 0.284 while edge RMSE worsens from 2.282 to 2.359 m. More gradient energy therefore does not establish better boundaries. Pure single-mode Head B is worse still.

Compared with D4 at native scale, 1.5× zoom lowers MVS3DM RMSE from 1.497 to 1.461 m, NEON from 3.401 to 3.363 m, and GAMUS from 2.819 to 2.525 m. US3D moves the opposite way, from 3.746 to 4.050 m, with predicted-tall bias worsening to −5.263 m. A global zoom switch is unsuitable for the satellite-building task. These zoom and single-mode comparisons remain pilot measurements; their large regressions were sufficient to exclude them from the two-candidate full confirmation.

## Implementation and validation

- `v5/eval/metrics.py`: opt-in prediction-conditioned height metrics, strictly valid 30 m blocks, and a pooled label-resolution companion. Spatial metrics now accept full-tile 2D maps; pooled landscape metrics use their actual GSD.
- `v5/eval_test.py`: `--bin_readout mean|single_mode`, `--head_b_weight 0..1`, and `--prediction_head fused|a|b`; replicated evaluation carries both heads. Defaults retain the current inference behaviour.
- `v5/models/heads.py`: vectorised single-mode readout and transient evaluation fusion override; no checkpoint parameter names or tensors changed.
- `v5/tools/readout_probe.py` and `V4_modal/readout_modal.py`: paired sweeps using the frozen checkpoint, with CPU preflight and a 90-minute GPU timeout.
- **18 focused checks passed on Modal CPU**: four single-mode distributions, prediction conditioning/sign, per-sample GSD and invalid-block exclusion, opt-in metrics, 2D spatial metrics, single-forward pooling, and the nine existing model checks. Both GPU calls completed successfully.
- Recorded evaluation passes total **24.55 minutes** on L4 across both calls, excluding startup/preflight/idle time. This used Modal compute; no zero-cost claim is made and exact billing was not queried.

## Next action

Keep the existing checkpoint and defaults. These measurements do not justify a global tall-height reweighting, SILog change, or stronger edge loss. A reduced-LR/SWA cycle remains an untested training hypothesis rather than a measured improvement. Before spending the remaining training budget, score the absolute DSM path with `eval/judge_proxy.py` on representative target scenes against the actual reference DEM; the small nDSM mean biases do not validate terrain anchoring or datum handling.

## Reproduction and artifacts

From `Model_Traning/V4_modal`, deploy `modal deploy -e gpu readout_modal.py`, then call the deployed functions through the Modal SDK:

```python
import modal
preflight = modal.Function.from_name("dw-readout", "preflight", environment_name="gpu")
probe = modal.Function.from_name("dw-readout", "probe", environment_name="gpu")
print(preflight.remote())
print(probe.spawn(tiles=128, run="v5_readout_probe_repeat").object_id)
print(probe.spawn(tiles=0, candidates="baseline,fused_b50",
                  run="v5_readout_confirmation_repeat").object_id)
```

- Pilot call: `fc-01M3YH97FDZJ3TM594RCKGC48P`.
- Confirmation call: `fc-01M3YJ95X37FEEPB3BX97EWB7M`.
- Persistent outputs: `depthwizard-results:/v5_readout_probe/` and `/v5_readout_confirmation/`; each contains `probe_metrics.json`, `PROBE.md`, and `probe.log`.
- Local copies: [pilot metrics](../../Model_Traning/v5/outputs/v5/readout_probe/modal/probe_metrics.json) and [full-validation metrics](../../Model_Traning/v5/outputs/v5/readout_confirmation/modal/probe_metrics.json).
- Followed the project’s `.claude/skills/modal-ai/SKILL.md` for Modal operation. Its linked reference files were absent; existing authenticated Modal SDK/CLI and project infrastructure supplied the working implementation.
