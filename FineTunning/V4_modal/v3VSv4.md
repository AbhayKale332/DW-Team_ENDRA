# v3 vs v4-modal — why v3 won, and what v5 reverts

Sources: `Logs/v3 (2)/{config,metrics}.json` + `run.log`, `FineTunning/v3/outputs/v3/run.log`,
`FineTunning/V4_modal/Output/{config,metrics}.json` + `run.log` + `02-train_Executted.ipynb`,
`FineTunning/V4_modal/Output/Visualization/{v3,v4_Modal}.jpeg`.
Config produced: `FineTunning/V4_modal/v5_flags.py`.

---

## 0. Verdict

**v4-modal is a regression.** On a like-for-like stratum distribution it scores **3.369 m**
against v3's **2.605 m** — 0.764 m worse. **85.5 % of that gap is the 0–2 m height band**,
caused primarily by `stratum_balance_beta 0.5 → 0.7`, which is a knob whose entire purpose is
to move loss mass off the abundant low stratum and onto the rare tall one. It did exactly that,
and the trade was bad: it bought −0.17 MSE on 10–20 m and −0.04 on 20 m+ for **+3.90 on 0–2 m**.

Ship v3 until v5 beats it.

---

## 1. Headline numbers

Both runs: `facebook/dinov3-vitl16-pretrain-sat493m`, identical preproc
(`0.5 m/px, tile 512, stretch=True`, mean `(0.43, 0.411, 0.296)`), `decoder_dim 256`,
`n_bins 96`, `crops_per_epoch 12000`, one H100, ~2 h.

| | v3 (ep 26) | v4-modal (ep 27) | Δ |
|---|---|---|---|
| `final_plain` RMSE | **2.715** | 3.755 | +1.04 |
| `final_tta` RMSE | **2.605** | 3.631 | +1.03 |
| `final_sliding_tta` RMSE | **2.723** | 3.646 | +0.92 |
| MAE (tta) | **1.288** | 1.953 | +0.67 |
| pearson r | **0.923** | 0.910 | worse |
| δ1 | **0.669** | 0.599 | worse |
| balanced RMSE | **3.497** | 3.948 | worse |
| flat_bias | **+0.38** | +1.00 | 2.6× worse |
| tall_bias | −2.12 | −2.05 | ≈ tie |
| elapsed | 153 min | 143 min | — |
| optimizer steps | 9 750 | 15 000 | v4 did **more** |
| samples seen | ~312 k | ~360 k | v4 saw **more** |

v4 was not step-starved or data-starved. It did more work for a worse result.

---

## 2. The val sets are not the same — a quarter of the gap is measurement

Both logs print `val: 400 tiles from gamus`, which is what made this look like a clean A/B.
It isn't. The ground-truth distributions differ materially:

| stratum | v3 val frac | v4 val frac | v3 RMSE | v4 RMSE | v3 bias | v4 bias |
|---|---|---|---|---|---|---|
| 0–2 m | 0.614 | 0.493 | **1.565** | 2.968 | +0.415 | +0.985 |
| 2–5 m | 0.102 | 0.093 | **2.651** | 3.294 | +0.572 | +0.842 |
| 5–10 m | 0.157 | 0.176 | **2.837** | 3.337 | −0.359 | −0.028 |
| 10–20 m | 0.087 | **0.152** | 4.795 | **4.586** | −1.375 | −1.056 |
| 20 m+ | 0.042 | **0.087** | 5.638 | **5.553** | −2.499 | −2.348 |

v4's val carries **2.1× the tall-pixel fraction**, and tall is the hardest band for both
models. `class_stats` independently confirms different tile populations (class6 mean GT
10.04 m in v3 vs 14.54 m in v4; class2 0.265 m vs 0.601 m).

**Provenance.** v3 took GAMUS from the gated HF repo — `RUN_MANIFEST.txt`:
`fetch: bulk snapshot (dw_fetch.py)`, `sizing: gamus_train=4000 gamus_val=400`, packed to
3837 train tiles. v4 ran through the Kaggle PNG mirror path (`modal_app.py:245`,
`akashch1512/gamusdataset`) with `--train_tiles 0 --val_tiles 0`, giving 5004 train / 859 val,
of which v4 took `first 400 of 859, prefix not sample`.

**Reweighting v4's per-stratum errors onto v3's stratum distribution:**

```
v3 global RMSE            2.605   (recomposes exactly from per-stratum — decomposition verified)
v4 global RMSE            3.631   (on its own, harder val)
v4 reweighted to v3 val   3.369
like-for-like gap        +0.764 m   (raw headline gap was +1.026 m)
```

A quarter of the apparent gap was the val set. **Three quarters is a real regression.**

`balanced_rmse_m`, which weights strata equally and is therefore already distribution-free,
agrees independently: **v3 3.497 vs v4 3.948**.

---

## 3. Where the 0.764 m lives

Gap decomposed in MSE units under v3's stratum weights:

| stratum | ΔMSE (v4 − v3) | share of gap |
|---|---|---|
| **0–2 m** | **+3.903** | **+85.5 %** |
| 2–5 m | +0.389 | +8.5 % |
| 5–10 m | +0.483 | +10.6 % |
| 10–20 m | −0.169 | −3.7 % |
| 20 m+ | −0.040 | −0.9 % |
| **total** | **+4.566** | 100 % |

**85.5 % of the regression is the 0–2 m band.** v4 is marginally *better* than v3 on both
tall strata and 1.9× worse on ground (1.565 → 2.968 m RMSE, bias +0.415 → +0.985 m).

v4 traded the ground plane for a rounding error on tall structures.

---

## 4. Root cause

### 4.1 Primary: `stratum_balance_beta 0.5 → 0.7`

`StratumBalancer` (`V4_Kaggle/models/losses.py:54-94`) is **algorithmically identical** to v3's
(`v3/models/losses.py:40-72`); v4 only made the masking sync-free. `HEIGHT_STRATA_M` is
identical in both: `(0.0, 2.0, 5.0, 10.0, 20.0, 1e9)`. **Only `beta` and `clip` changed.**

Effective per-pixel loss weights at v4's training stratum mix:

```
beta=0.5 clip=5:  0-2m=0.675  2-5m=1.555  5-10m=1.131  10-20m=1.217  20m+=1.602   tall/flat=2.37x
beta=0.7 clip=8:  0-2m=0.558  2-5m=1.794  5-10m=1.149  10-20m=1.274  20m+=1.871   tall/flat=3.35x
```

`beta 0.5 → 0.7` raised the tall:flat ratio **41 %** and cut the absolute 0–2 m weight **17 %** —
on the stratum holding ~50 % of all pixels and 85 % of the regression. The sign and relative
magnitude of every stratum's error change matches this prediction.

> **`stratum_weight_clip 5 → 8` was inert.** At these stratum frequencies the clip never binds
> in *either* config (verified numerically). Reverting the clip alone changes nothing — which is
> likely why a previous attempt at this fix did not move the number.

### 4.2 Secondary: two new bin-head terms, both fighting the ground's zero-mode

Neither existed in v3.

- **`bin_soft_sigma: 1.5`.** A Gaussian soft label at σ=1.5 bins carries ~1.8 nats of
  irreducible cross-entropy. v4's raw bin CE ended at **2.53**; v3's, on hard targets, at
  **0.19**. Not directly comparable, but ~0.7 nats of genuine excess remains — head B is
  under-fit, not merely measured differently.
- **`w_bin_entropy: 0.02`.** `bin_entropy_reg` (`losses.py:276-288`) puts a floor on the
  *marginal* bin distribution's entropy — it pushes toward uniform. A ground plane wants the
  opposite: one sharp mode at zero. Across 30 epochs `ent` moved only 4.51 → 3.03.

Both terms blur precisely the distribution feature that the 0–2 m stratum depends on.

### 4.3 Plausible but unmeasured

Reverted on the grounds of *"v3 did it differently and v3 won"*, **not** on isolating evidence:

- `encoder_unfreeze_blocks 16` — v3 trained 303.1 M encoder params (`encoder TRAINABLE`,
  52 param groups); v4 trained 221.6 M (34 groups), freezing the patch embedding and blocks
  1–8, where a satellite-pretrained ViT keeps its texture→structure mapping.
- `llrd 0.80 → 0.90`.

These need an ablation to attribute. Do not claim them as causes.

---

## 5. The demo images corroborate it

`Output/Visualization/{v3,v4_Modal}.jpeg` — same scene, same declared GSD, same 484×363 px:

| | v3 | v4-modal |
|---|---|---|
| Height range | 0.00 – 30.78 m | 0.00 – 59.30 m |
| mean / median | 9.54 / 8.84 m | 5.24 / **0.02 m** |
| % below 1 m | 9.3 % | 66.9 % |
| probe A | px(201,127) = **22.97 m** | px(204,123) = **7.21 m** |

Those probes are 4 px apart and disagree by 15.8 m. There is no GT for this scene, so neither
can be scored — but two things are checkable:

- **v4 zeroes the tree canopy.** The forested half of the scene renders flat dark blue; median
  0.02 m means half the pixels are pinned to hard zero. For a GAMUS-style AGL target that
  includes vegetation, that is a failure — and it is the same pathology as `flat_lt1m` r = 0.16
  in the metrics. The model is emitting a constant, not a surface. **This is section 3 showing
  up on screen.**
- **v4 has a 59.3 m hotspot** on a small structure in a scene whose main building v3 reads at
  ~23–30 m. Almost certainly a spike, and it drags the colour scale so everything real
  compresses into blue-green.

Against v3: its richness is *partly* the artifact its own UI flags — `9.3 % of pixels below 1 m
(a nadir urban scene is usually 40–70 %)` plus *"Very little flat ground — the model may be
reading texture as terrain."* v3 is not recovering a clean ground plane here either.

**Honest read: v3 over-predicts ground, v4 destroys everything that isn't a building.** v4's
failure is the more damaging one for a demo — a flat blue forest reads as "the model doesn't
work." And where there *is* GT to check, v3's flat_bias of +0.38 vs v4's +1.00 says v3 handles
ground better. Visual impression and metrics agree.

---

## 6. Full config diff

### Changed

| flag | v3 | v4 | v5 | note |
|---|---|---|---|---|
| `stratum_balance_beta` | 0.5 | **0.7** | 0.5 | the 85.5 % |
| `stratum_weight_clip` | 5.0 | 8.0 | 5.0 | inert; reverted for provenance |
| `encoder_unfreeze_blocks` | *absent (=all)* | **16** | 0 | 303.1 M vs 221.6 M trainable |
| `llrd` | 0.80 | 0.90 | 0.80 | block 1 needs deeper decay once it trains |
| `batch_size` / `grad_accum` | 16 / 2 | 24 / 1 | 16 / 2 | **forced** — see §7 |
| `bin_soft_sigma` | *absent* | **1.5** | 0.0 | bin CE 2.53 vs 0.19 |
| `w_bin_entropy` | *absent* | **0.02** | 0.0 | fights the zero-mode |
| `epochs` | 26 | 30 | 26 | v4 flat from ep 22 (3.780→3.755 over 8) |
| `onnx_opset` | — | 17 | 18 | 17 conversion failed, silently kept 18 |
| `datasets` | `gamus,synrs3d` | +dfc23, +india_labeled, split synrs3d | **keep v4** | dfc23 is the only held-out domain |
| `sampler_weights` | uniform `1:1:1` | `gamus:3,dfc23:2,…` | **keep v4** | not implicated |
| `max_valid_height_m` | 200 | 150 | **keep v4** | not implicated |
| `tta_scales` | `[1.0]` | `[1.0, 1.25]` | keep v4 | not implicated |
| `eval_every` | 2 | 1 | keep v4 | |
| `max_minutes` | 330 | 240 | 240 | neither run hit the cap |

### New in v4, kept
`test_sources` / `test_tiles` / `test_sliding_tiles`, `per_landscape_metrics`, `export_onnx`,
`make_figures`, `make_report`, `save_full_state`, `full_state_every`, `amp_*`, `session_minutes`.

### New in v4, **deliberately dropped from the shipped config**
`w_consistency`, `unlabeled_source`, `teacher_ema`, `consistency_rampup_epochs`,
`consistency_conf_m`, `consistency_every`, `unlabeled_batch_frac` — see §8.1.

---

## 7. v5

`FineTunning/V4_modal/v5_flags.py`. All 30 flags validated against the `config.py` dataclass
(`config.py:443-454` auto-exposes every field as `--<field>`).

```python
from v5_flags import FLAGS as V5
modal_app.argv(**V5)
```

Rendered argv:

```
python train.py --data_root /scratch/dwdata --output_dir /mnt/depthwizard-results/v5 \
  --amp_dtype bf16 --grad_checkpoint_encoder false --grad_checkpoint_decoder false \
  --eval_batch_mult 2 --num_workers 12 --prefetch_factor 6 --compile_model false \
  --epochs 26 --eval_every 1 --max_minutes 240 --session_minutes 0 \
  --save_full_state true --full_state_every 5 --make_zip false \
  --datasets gamus,synrs3d_g1,synrs3d_g05,dfc23_g050,india_labeled \
  --sampler_weights gamus:3,dfc23:2,synrs3d_g05:1,synrs3d_g1:0.5,india_labeled:1 \
  --max_valid_height_m 150 --w_seg 0.2 \
  --test_sources gamus:test --test_tiles 0 --test_sliding_tiles 400 \
  --stratum_balance_beta 0.5 --stratum_weight_clip 5.0 \
  --bin_soft_sigma 0.0 --w_bin_entropy 0.0 \
  --encoder_unfreeze_blocks 0 --batch_size 16 --grad_accum 2 --llrd 0.80 \
  --onnx_opset 18
```

**`batch_size 16 / grad_accum 2` is not a free choice.** It is forced by the encoder revert:
v3 peaked at **80 345 MiB of 81 559** on an H100 at that shape with a fully-unfrozen encoder.
`24 × 1` with all 24 blocks training will OOM. `encoder_unfreeze_blocks: 0` means *all of it*
(`models/encoder.py:83`).

**Fallback if head B collapses again.** `losses.py:13` records v2's bin CE sitting at 0.06–0.13
while the head learned nothing — a *low* CE is the collapse signature, and v3's 0.19 is not far
off it. If v5's bin CE drops below ~0.15, reintroduce `bin_soft_sigma` at **0.5**, not 1.5, and
leave `w_bin_entropy` at 0.

---

## 8. Blockers — read before launching v5

### 8.1 v4's config claims a pipeline that never ran

`Output/config.json` ships `w_consistency: 1.0`, `teacher_ema: 0.999`,
`consistency_rampup_epochs: 4`, `unlabeled_source: "india_unlabeled"`. **None of it executed.**

`india_unlabeled` was not in `--datasets`, so `dwdata/loaders.py:89` found no store and returned
`None`. The warning at `loaders.py:92` is gated on the source being in `dataset_list()`, so it
**printed nothing**. Confirmed: no `con=` term appears in any of v4's 1665 log lines — every
per-step line, epoch 1 to 30, reads `(l1= sil= nrm= flat= bin= ent= seg= a=)`.

Anyone reading `config.json` beside `metrics.json` would conclude semi-supervised domain
adaptation produced the 2.77 m India number. It did not.

- **Fix:** make the `loaders.py:92` warning unconditional.
- **Until then:** keep these keys out of any shipped config. A config claiming a branch that
  never ran is worse than no config.

### 8.2 The val split is a prefix, and model selection ran on it

```
[data] val: 400 tiles from gamus (first 400 of 859, prefix not sample)
```

`loaders.py:244` prints `prefix not sample` and then takes a prefix anyway. The consequence:

| | urban | sparse | forested |
|---|---|---|---|
| v4 val prefix (400) | 87.5 % | 7.5 % | 5.0 % |
| GAMUS test (2861) | 57.6 % | 11.8 % | 30.5 % |

**87 % urban validation against a 58 % urban test set, forest under-represented 6×** — and
`best.pt` (epoch 27) was selected on it.

`v5_flags.py` stubs `val_sample_seed` in `FIXES`, commented out, **because no such flag exists
yet.** It needs a seeded permutation in `loaders.py` (~5 lines). Until that lands, v5's val is
the same skewed prefix and v5-vs-v3 will need the same reweighting done in §2.

### 8.3 The shipped ONNX cannot be loaded

`depthwizard.onnx` is 3.97 MB — graph only. It references `depthwizard.onnx.data` for external
weights (~0.9 GB for 221.6 M fp32 params) and **that file is not in `Output/`**.

Separately, the opset downgrade failed (`run.log:1612-1643`):
`Failed to convert the model to the target version 17 using the ONNX C API. The model was not
modified` → `RuntimeError: axes_input_to_attribute.h:55`. The file is opset **18** while
`depthwizard.onnx.json` records `"opset_requested": 17`. Hence `onnx_opset: 18` in v5.

Also missing from `Output/`: `best.pt`, the 12 qualitative strips, the 7 figures,
`viewer_sample`.

### 8.4 The India number is near-domain, not out-of-domain

`india_labeled` has **81 training tiles** at sampler weight 1.0 of total weight 7.5 →
~1600 of 12 000 crops/epoch, i.e. each tile seen ~20× per epoch, ~600× over the run. The 35-tile
India val is a disjoint tile split but almost certainly the same acquisition and geography.

The trajectory shows it: GAMUS val flattens after epoch 22 (3.780 → 3.755 over 8 epochs) while
India improves monotonically to the last epoch (2.824 → 2.771). That divergence is the signature
of memorising a small pool.

**2.77 m is not evidence of generalisation to Indian imagery.** DFC23 at **4.97 m** — genuinely
held out, never in the sampler at eval time, `tall_gt15m` RMSE 7.97 m — is the honest transfer
number.

### 8.5 Unverified: GAMUS test-split disjointness

`gamus/test` (2861 tiles) is asserted held out by comment (`modal_app.py:71-76`) but nothing in
the run log prints an ID-overlap check, and `pack_gamus_png.py:265` only packs `train` and `val`.
Add an assertion that `test_ids ∩ train_ids = ∅` and print it. The held-out claim is currently
taken on faith.

---

## 9. Other findings from the v4 audit

- **`gsd_jitter_hi_m: 1.2` was unreachable for every dataset** (`[!] requested hi 1.20
  unreachable` ×5) — the coarsest source tops out at 0.66 m. Scale augmentation ran over a
  narrower range than configured. Matters if deployment GSD is coarser than 0.66 m.
- **Two v4 test numbers differ by 0.55 m for a non-modelling reason.** The log annotates
  `test_gamus_test_sliding_tta` (3.873) with *"THIS is the number to quote"* while
  `test_gamus_test_tta` reads 3.321 — but the first covers 400 tiles and the second all 2861,
  with 87.8 % vs 57.6 % urban mixes. Since sparse tiles score RMSE 1.28 on the full set,
  dropping them inflates the "quotable" number. Run sliding over all 2861, or quote both with
  tile counts attached.
- **δ1 is close to meaningless here.** It is a ratio threshold and ~46 % of val pixels are under
  1 m. Quote per-stratum RMSE + bias instead — it is the honest framing and it happens to be the
  one that shows the model's real strength (r = 0.91 on structure).
- **v4 left ~100 min of a 240 min budget unused.** Not waste — the curve was flat from epoch 22 —
  but a different configuration could spend it.

---

## 10. Reproducing the analysis

Everything in §2–4 comes from the two `metrics.json` files and is re-derivable:

- stratum fractions and per-stratum RMSE: `final_tta.per_stratum[*].{n,rmse_m,bias_m}`
- reweighting: `sqrt(Σ_s f3[s] · rmse4[s]²)` where `f3` is v3's stratum fraction
- gap attribution: `f3[s] · (rmse4[s]² − rmse3[s]²)`, sums to `mse4 − mse3`
- balancer weights: `w = (1/p)^beta`, normalised by `Σ p·w`, clipped to `[1/clip, clip]`,
  renormalised to mean 1 — `losses.py:85-94`

The v3 decomposition recomposes to its reported global RMSE exactly (2.605), which validates
the arithmetic.
