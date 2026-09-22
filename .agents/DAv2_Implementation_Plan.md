# DAV2/V1 — Depth Anything V2 backbone for DepthWizard

## Context

Every DepthWizard run so far (v1→v4) has used **DINOv3 ViT-L/16 SAT-493M** as the
encoder with a randomly-initialised DPT trunk bolted on top. The v4 H100 run
plateaued at GAMUS val RMSE ~3.85 and its `flat_bias` term never moved
(+1.08 m against v3's +0.396 m), while v3's own 2.715 came from a different data
mix, a different val subset and a full-backbone unfreeze — so there is no clean
read on whether the *backbone* is the limiting factor.

This variant answers that by swapping in **Depth Anything V2 Base**: a DINOv2
ViT-B/14 encoder **plus its pretrained DPT decoder**, trained specifically for
monocular depth. Two things change at once and both are deliberate:

1. The decoder is no longer random — DAv2's reassemble + fusion stages arrive
   with depth-specific priors, which is the whole reason to pick DAv2 over plain
   DINOv2.
2. The encoder drops from 303 M to 86 M parameters. On the Kaggle 2×T4 target
   that is the difference between `grad_checkpoint_encoder` being forced ON at
   `batch_size 2` (what `V4_Kaggle/README.md` §5.2 had to do) and running
   uncheckpointed at `batch_size 6–8` — roughly 2–3× the optimiser steps per
   12 h session, which is what the v4 Kaggle run was actually starved of.

**The bet, stated plainly:** DINOv3-SAT-493M is pretrained on *satellite*
imagery and DAv2 is not. We are trading domain-matched pretraining for
task-matched pretraining. It may lose. It is cheap to find out, and the
answer is worth having either way.

**Not the goal:** chasing v3's 2.715. Per the user's decision, DAV2/V1 is judged
on `test_gamus_test_*`, `dfc23_g050` and `india_labeled` — the deployment-relevant
domains. `final_*` is never quoted (it is scored on the tiles `best.pt` was
selected on).

**Rejected:** Marigold V2. It is a QLoRA fine-tune of Qwen-Image-Edit-2509
(~20 B DiT + VAE) that its own authors trained for >5 days on a single 32 GB GPU
at batch size 1. It does not fit a 16 GB Turing card, and it outputs
affine-invariant depth, which `.agents/Depth_Wizard_Plan.md:17` rules out as the
model's output.

---

## 1. Layout — `FineTunning/DAV2_V1/`

Self-contained, following the `V4_Kaggle` precedent, but copying **only the
training + evaluation path**. Verified by import trace: nothing reachable from
`train.py` touches `geo/`, `serve/` or `viewer/`.

**Copy from `FineTunning/V4_Kaggle/`, unchanged:**
`dwdata/` (all 8 files) · `eval/` (metrics, landscape, sliding, report) ·
`infer/engine.py`, `infer/export_onnx.py` · `viz/figures.py`, `viz/report_html.py` ·
`models/losses.py`, `models/ema.py`, `models/tta.py` · `train.py` ·
`prepare_data.py` · `package_results.py` · `pytest.ini` · `requirements.txt`

**Do not copy:** `geo/` · `serve/` · `viewer/` · `viz/mesh.py` ·
`infer/predict.py` (imports `geo.calibrate`) · `pack_gamus_png.py` ·
`Kaggle_File_Structure.md`

**New or modified:** `config.py` · `models/dav2.py` (new) · `models/heads.py` ·
`dwdata/preprocess.py` · `main.py` (drop the `--serve` branch) · `run_kaggle.sh` ·
`dav2_train.ipynb` · `README.md` · `tests/`

`models/dpt.py` and `models/encoder.py` are **not** copied — DAv2's own neck
replaces both.

---

## 2. The model change

### 2.1 `models/dav2.py` (new) — `DAV2Backbone`

One module that stands in for `DINOv3Encoder` **and** `DPTTrunk` together, so
`DepthWizardNet.forward` needs no structural change.

Built from HF `DepthAnythingForDepthEstimation` (`depth-anything/Depth-Anything-V2-Base-hf`,
ungated — no `HF_TOKEN` needed for it), keeping `.backbone` (Dinov2Backbone) and
`.neck`, and **discarding `.head`** (it predicts relative inverse depth; we
predict metric nDSM).

Forward, for a 518 px input:

```
backbone(pixel_values).feature_maps     # 4 taps, out_indices from DAv2's own config
  -> neck(features, patch_h=37, patch_w=37)
     reassemble x(4, 2, 1, 0.5) on the 37x37 grid -> 148 / 74 / 37 / 18
     fusion upsamples x2 from coarsest        -> final at 296 x 296, C=128
  -> take fusion_stage[-1]                     # (B, 128, 296, 296)
  -> F.interpolate to (H//2, W//2) = 259x259   # restores the existing head geometry
  -> 1x1 Conv2d(fusion_hidden_size -> decoder_dim)
  -> (B, decoder_dim, H/2, W/2)                # exactly DPTTrunk's old contract
```

The interpolate-to-H/2 step is what keeps `tests/test_model.py::test_forward_shapes`
(`b_logits` at half res) and HeadB's memory cost unchanged. Without it the bin
logits would be `(B, 96, 296, 296)` — 1.3 GB at batch 8 in fp16.

Interface `train.py` depends on, ported from `V4_Kaggle/models/encoder.py`:

| Member | Notes |
|---|---|
| `BUILDER` class attr | test-stub injection, same pattern as `encoder.py:29` |
| `.hidden`, `.patch` | read from `backbone.config` / `config.patch_size` (14) |
| `.frozen` | flag |
| `set_frozen(frozen, top_blocks=0)` | reuse `encoder.py`'s structural `_find_blocks` verbatim — it locates the `nn.ModuleList` of length `num_hidden_layers`, which works on Dinov2Backbone's 12 blocks without change. **The neck freezes and unfreezes with the encoder.** |
| `llrd_param_groups(base_lr, decay, wd)` | reuse `encoder.py:261` logic for the 12 blocks; the **neck gets its own undecayed group at `encoder_lr`** — it is pretrained, not random, so it should not be trained at the decoder's 3e-4 |
| `train(mode)` | keeps the backbone in `eval()` while frozen |
| `_set_grad_checkpointing(on)` | `backbone.gradient_checkpointing_enable(...)`, toggled once in `set_frozen` (never per-step — that was a v4 defect) |

### 2.2 `models/heads.py`

Two lines. `HeadA/HeadB/HeadC/GatedFusion` and the whole output dict are untouched.

```python
self.encoder = DAV2Backbone(cfg)     # was DINOv3Encoder(cfg) + DPTTrunk(...)
# delete self.trunk
...
x = self.encoder(image)              # was self.trunk(self.encoder(image))
```

`head_state_dict()` (`heads.py:214`) filters on the prefix `encoder.model.` —
update it to match whatever `DAV2Backbone` names its HF submodule, or the
frozen-encoder checkpoints will silently carry 86 M of public weights.

---

## 3. `config.py` deltas

| Field | v4 Kaggle | DAV2/V1 | Why |
|---|---|---|---|
| `encoder_model_id` | `facebook/dinov3-vitl16-pretrain-sat493m` | `depth-anything/Depth-Anything-V2-Base-hf` | the swap |
| `encoder_patch` | *(new field)* | `14` | DINOv2 is patch-14 |
| `tile_size` | 512 | **518** (37×14) | DAv2's native training resolution; no pos-embed interpolation |
| `decoder_dim` | 256 | **128** | `fusion_hidden_size` for DAv2-Base; the 1×1 proj becomes free |
| `encoder_feature_indices` | (6,12,18,24) | *removed* | DAv2's own `backbone_config.out_indices` already picks the taps |
| `stratum_balance_beta` | 0.7 | **0.5** | v4 lesson: 0.7 stalled balanced RMSE at 4.165 vs v3's 3.584 |
| `stratum_weight_clip` | 8.0 | **5.0** | same |
| `encoder_unfreeze_blocks` | 16 | **0** (all 12) | 86 M fits; v3's full unfreeze is the recipe that worked |
| `llrd` | 0.90 | **0.85** | over 12 blocks, 0.85¹² ≈ 14 % of `encoder_lr` — a usable span end to end (0.80¹² = 6.9 % is the dead-lower-half problem again) |
| `grad_checkpoint_encoder` | True | **False** | the VRAM it bought is no longer scarce |
| `validate()` | `tile_size % 16 == 0` | `tile_size % encoder_patch == 0` | |

Defaults stay the H100-friendly ones (you said an H100 is coming). **The T4
profile lives entirely in launch flags**, exactly as `V4_Kaggle/README.md` §5.2
established — no Turing constants baked into the dataclass.

Data mix is unchanged from V4_Kaggle: `gamus, synrs3d_g05, synrs3d_g1` +
`dfc23`, `india_labeled`, with `india_unlabeled` driving the mean-teacher.

---

## 4. `dwdata/preprocess.py` deltas

`PreprocSpec` is already parameterised for this — it has a `patch` field and a
`resolve_encoder_stats()` that falls back to ImageNet for any non-`sat` model id.
Three small fixes:

1. `PreprocSpec.patch` default `16` → `14`.
2. `from_config()` (line 74) does **not** forward `patch` — add
   `patch=int(cfg.encoder_patch)`. `infer/engine.py:68` reads `spec.patch`, so
   without this, inference tiles on a patch-16 grid against a patch-14 model.
3. `from_config()`'s pre-resolve fallback (line 75) hardcodes `DINOV3_SAT_MEAN/STD`
   → make it ImageNet. DAv2 uses ImageNet statistics; the SAT constants shift
   every input by ~0.3σ.
4. Bump `PreprocSpec.version` to `"dav2-v1"` so a v4 checkpoint cannot be loaded
   under this contract without noticing.

**No change needed to the crop path.** `augment.crop_and_scale` resamples a
`win×win` source window to `tile×tile`, and `sample_window` clamps
`win = min(win, src_h, src_w)`, so a 512 px GAMUS tile fills a 518 px crop at an
effective GSD of 0.494 m — recorded in the sample's `gsd_m` field, not silently
lost.

---

## 5. Kaggle 2×T4 launch profile

Static VRAM per card falls from ~8.3 GB (V4_Kaggle's measured table) to roughly
**~2.6 GB** — 86 M params × 4 copies (live / grads / AdamW ×2) + the EMA shadow
+ the mean-teacher deepcopy. That is the budget being spent:

```bash
--amp_dtype fp16                  # Turing has no bf16; the GradScaler path exists
--grad_checkpoint_encoder false   # was forced true for ViT-L
--batch_size 6 --grad_accum 2     # 2 ranks x 6 x 2 = global 24, parity with v3/v4
--eval_batch_mult 4
--num_workers 2 --prefetch_factor 2
--compile_model false
--max_minutes 480 --session_minutes 480
--save_full_state true --make_zip false
--consistency_every 2 --w_consistency 1.0   # mean-teacher at full strength
```

Read the `vram=NG` field the train log already prints every 25 steps. If there
is headroom, move to `--batch_size 8 --grad_accum 2` (global 32) rather than
cutting `grad_accum` to 1 — more optimiser steps is the point, but global-batch
parity with v3/v4 is what keeps the LR schedule honest. If it is tight, back off
in V4_Kaggle's documented order: `--consistency_every 4`, then `--ema_decay 0`,
then `--w_consistency 0`.

Kaggle notebook: internet **on** (GAMUS still needs `HF_TOKEN`; the DAv2
checkpoint does not). Licence note for the SIH writeup: **DAv2 Base is
CC-BY-NC-4.0** (non-commercial). Only DAv2-Small is Apache-2.0. Flag this in
`README.md` — if the submission needs a commercially usable model, the fallback
is ViT-S, and the only config change is the model id.

---

## 6. Tests

**Delete** (their dependencies are not copied): `test_calibrate.py`,
`test_geotiff_e2e.py`, `test_serve.py`, `test_mesh.py`, `test_real_dinov3.py`.

**New:**
- `tests/stub_dav2.py` — offline stand-in in the spirit of `tests/stub_encoder.py`:
  a fake Dinov2Backbone returning `feature_maps` plus a real (tiny)
  `DepthAnythingNeck`, installed via `DAV2Backbone.BUILDER`. Keep the deliberately
  odd block-list name so the structural `_find_blocks` is genuinely exercised.
- `tests/test_real_dav2.py` — the CPU proof-of-checkpoint test, mirroring what
  `test_real_dinov3.py` did: load the real ungated checkpoint, one forward, assert
  the neck output shape and that the dead-parameter probe finds nothing unexpected
  after a backward. Marked to skip without network.

**Update for patch-14 / 518:** `test_model.py` (tile 518→ b_logits at 259),
`test_dpt_odd_grid.py` (now an odd-*patch*-grid test against the neck),
`test_end_to_end.py`, `test_onnx.py`, `test_preprocess.py`,
`test_full_state_resume.py`.

---

## 7. Verification

1. **Offline** — `pytest -q` in `FineTunning/DAV2_V1/`, no GPU, no network.
   Every test that was green in V4_Kaggle minus the five deleted ones.
2. **Shape proof** — `test_real_dav2.py` with network on: confirms the 518 →
   37×37 → 296 → 259 chain against the actual published checkpoint, not my
   arithmetic.
3. **`bash run_kaggle.sh check`** — GPUs, NCCL, `/kaggle/working` + `/kaggle/temp`
   + `/dev/shm`, test suite.
4. **`bash run_kaggle.sh smoke`** — the first real 2-process DDP run (~10 min).
   Do not skip it; it is the first thing that opens a NCCL process group.
5. **First 200 steps of the real run, check four things:**
   - `train_loss` is finite. fp16 + GradScaler on a freshly-attached neck is the
     most likely place for a NaN, and per the depth-anything skill's own warning
     a trainer can report PASS with `train_loss = NaN`.
   - `vram=NG` leaves ≥3 GB headroom for the eval spike.
   - img/s is meaningfully above V4_Kaggle's 2.1 img/s unfrozen — if it is not,
     the 86 M encoder is not buying what this plan claims and the batch size
     should be re-derived from a measurement, not from §5.
   - the optimiser log's LR span covers all 12 blocks at usable values.
6. **Against Modal's lesson** — in a Modal/Kaggle notebook, a green cell and a
   frozen log prove nothing. Verify from the filesystem (`ls`, `du -sh`,
   `metrics.json`), not the output pane.
7. **Scoring** — compare `test_gamus_test_*`, `dfc23_g050` and `india_labeled`
   against the v4 run's numbers in `FineTunning/V4_modal/Output/`. Never quote
   `final_*`. Report per-stratum `flat_bias` / `tall_bias` explicitly — that pair
   is the diagnostic that told us v4's problem, and it is the first place a
   backbone change should show up.

---

## 8. Commit

Commit when the offline tests pass and again after the run lands. **Do not push** —
that is yours.
