# V5: research for the next ~2 h fine-tune

Checked 2026-10-02. Primary sources where reachable; fallbacks are marked.
Companions: `V5_Research_Additions.md` (what decides the score), the last run's
`Model_Traning/v5/outputs/v5/modal/{RESULTS.md,run.log,metrics.json}`, the flags
it ran with (`Model_Traning/v5/lightning/final_flags.py` over
`Model_Traning/V4_modal/v5_flags.py`, launched by `V4_modal/final_modal.py`).

The question has three parts. What should one ~2 h H100 run (~$8 of the ~$10 left)
change? What fixes (a) tall objects coming out ~6 m low and (b) soft edges? Does a
second cosine cycle from the converged checkpoint pay?

Each section keeps two things apart. **Source** is what a paper or repo shows.
**Here** is my own inference for this project.

---

## 0. What the last run actually measured

These numbers come from `metrics.json`, val sets, at epoch 1 and epoch 7 (best).
"Tall" is GT ≥ 15 m and "flat" is GT < 1 m (`eval/metrics.py:168`).

| Source | 0–2 m share / bias | 5–10 m bias | 10–20 m bias | 20 m+ share / bias | tall px, bias, r inside the tall set | grad ratio | edge RMSE |
|---|---|---|---|---|---|---|---|
| MVS3DM (select) | 71 % / +0.33 | −2.01 | −3.62 | **0.03 %** / −8.86 | 99 k px (0.2 %), **−6.11**, r = **0.13** | 0.23 → 0.23 | 2.358 → 2.281 |
| NEON | 50 % / +0.42 | +1.05 | +0.38 | 20.7 % / −1.83 | 9.1 M px, −1.18, r = 0.76 | 0.66 → 0.66 | 5.156 → 4.646 |
| GAMUS | 60 % / +0.56 | −0.56 | −1.73 | 4.6 % / −1.90 | 4.0 M px, −1.89, r = 0.77 | 0.38 → 0.39 | 3.805 → 3.771 |
| US3D | 72 % / +0.26 | −1.03 | −1.39 | 3.1 % / **−9.66** (RMSE 16.4) | 3.1 M px, −5.58, r = 0.76 | 0.41 → 0.43 | 6.002 → 5.780 |

How to read it:

- **The headline "−6 m tall bias" means two different things.** On MVS3DM it
  rests on 0.2 % of pixels, and inside that set prediction and GT barely
  correlate (r = 0.13). The model cannot tell 15 m from 25 m there, and the
  number is close to noise. MVS3DM's material under-prediction is in 5–20 m
  (−2.0 / −3.6 m on 12 % of pixels: trees and sheds). The real tall-object
  problem is **US3D 20 m+ (−9.7 m, 3.1 % of pixels)**, i.e. tall buildings.
- **Ground is over-predicted on every source (+0.26 to +0.56 m) while tall is
  under-predicted.** That is the signature of a regressor that shrinks
  towards the frequent value. It is not an offset.
- **The bias depends on how rare tall pixels are in the source.** NEON, where
  21 % of pixels are > 20 m, is only −1.8 m there. US3D and MVS3DM, where they
  are rare, are −9 m.
- **`grad` did not move in 7 epochs, or across runs.** `final_flags.py`
  records 0.23 (MVS3DM) for `resume-v4-1.6`, trained at `w_grad 0.5` /
  `w_normal 0.3`. This run doubled `w_grad` to 1.0 and raised `w_normal` to 0.5,
  and it still reads 0.23. Edge RMSE did fall 3 % on MVS3DM and 10 % on NEON.
- **Most of the run's gain is new data, not schedule.** NEON was new
  (forest 4.61 → 4.11). MVS3DM, already in the mix, moved 1.602 → 1.545, and
  the start checkpoint scored 1.615. Train loss fell 8.15 → 7.21 while LR
  annealed to its 1 % floor (`train.py:134`, `final_div=1e2`). A falling train
  loss at LR → 0 is mostly annealing noise reduction. It does not show the model
  is under-trained.

The loss as it ran (`config.json` in `metrics.json`):

| Term | Knob(s) = value | Code |
|---|---|---|
| L1, stratum-weighted | `w_l1 1.0`, `stratum_balance_beta 0.5`, `stratum_weight_clip 5.0`, strata 0/2/5/10/20 m | `models/losses.py:54-116`, `config.py:101` |
| SILog on log(h + 1), **not** stratum-weighted | `w_silog 1.0`, `silog_lambda 0.85`, `silog_shift 1.0` | `losses.py:126` |
| Multi-scale gradient, 4 scales | `w_grad 1.0` | `losses.py:148` |
| Normals / flatness | `w_normal 0.5` / `w_flat 0.2` | |
| Bin CE (hard target) / entropy | `w_bin_ce 0.5`, `bin_soft_sigma 0`, `w_bin_entropy 0`, `n_bins 96`, `bin_max_m 120` | `losses.py:252` |
| Heads | `w_fused 1.0`, `w_head_a 0.3`, `w_head_b 0.3`; logged gate `a` ≈ 0.73–0.80, so `fused` is ~¾ Head A (regression) | `models/heads.py:151-179` |
| LR | decoder 1.5e-4, encoder 3e-5, `llrd 0.9` (bottom group 2.39e-6, as logged), cosine to 1 %, EMA 0.9995 | `train.py:134`, `models/encoder.py:261` |

## 1. Tall-object underestimation

### 1.1 Why it happens

**Imbalanced regression (source).** Ren et al., *Balanced MSE*, CVPR 2022
([arXiv 2203.16427](https://arxiv.org/abs/2203.16427), §3, Eq. 3.3) show that
MSE under a skewed label distribution fits *p_train(y|x)*. The prediction is
pulled towards frequent labels by the ratio *p_train(y)/p_bal(y)*. Yang et al.,
*Delving into Deep Imbalanced Regression*, ICML 2021
([arXiv 2102.09554](https://arxiv.org/abs/2102.09554), §4, Table 4) measure the
same effect on dense depth (NYUD2-DIR). Few-shot depth ranges have 3.6× the RMSE
of many-shot ones (2.123 vs 0.591 m).

**HTC-DC Net names it for heights (source).** Chen, Shi, Xiong, Zhu
([arXiv 2309.16486](https://arxiv.org/abs/2309.16486), abstract and §III-B2) say
the long-tailed height distribution makes networks "biased and tend to
underestimate". Tolan et al. 2024 is the closest prior work to this model:
SSL ViT-L on 0.5 m Maxar imagery, NEON CHM labels, DPT decoder
([arXiv 2304.07213](https://arxiv.org/abs/2304.07213), §3.2). They added a
classification output "to avoid a bias toward small predicted values".

**The metric itself (here; plain probability, no paper needed).** Take a
perfectly calibrated predictor, ŷ = E[y | x]. Its error conditioned on the
*GT* being high, E[ŷ − y | y ≥ 15], is negative whenever the image does not
fully determine y. The flip side is a positive error on y < 1 m. That is exactly
the +0.3 / −6 pattern above. So `tall_bias` cannot separate "biased model" from
"uncertain model". Two other numbers can:

- the bias conditioned on the **prediction**, E[y − ŷ | ŷ ≥ 15];
- the bias after block-averaging to 30 m, which is what the judges see
  against SRTM / COP30.

**The loss geometry in this repo (here, from the code).**

- `silog_loss` works on log(h + 1). Its per-metre gradient is 1/(h + 1): 1.0 at
  ground and 0.03 at 30 m. So the SILog term barely pushes a tall pixel up, and
  it is not multiplied by the stratum weights (`losses.py:317-322`).
- λ = 0.85 makes SILog 85 % scale-invariant. A tile whose heights are all
  10 % low pays only the (1 − λ) share of the mean-log term.
- Eigen et al. 2014 ([arXiv 1406.2283](https://arxiv.org/abs/1406.2283), §3.3,
  Eq. 4) use **λ = 0.5**: "λ = 1 is the scale-invariant error exactly … λ = 0.5 …
  produces good absolute-scale predictions". Depth Anything V2's metric
  fine-tune also uses 0.5 ([`metric_depth/util/loss.py`](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/metric_depth/util/loss.py), `SiLogLoss(lambd=0.5)`).
  AdaBins ([arXiv 2011.14141](https://arxiv.org/abs/2011.14141), §3.4) and
  Tolan et al. (§3.2, Eq. 1) use 0.85.
- The stratum balancer's top stratum is "20 m+". A 60 m tower pixel is weighted
  the same as a 21 m tree.

**Averaging at the output (source, edge-related).** `fused` is a per-pixel blend
of two heads, and Head B returns the *expectation* over its bin distribution
(`heads.py:145`). Where the distribution is bimodal (roof vs ground at an
edge), the mean lands between the modes. Chen et al., ICCV 2019
([paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_On_the_Over-Smoothing_Problem_of_CNN_Based_Disparity_Estimation_ICCV_2019_paper.pdf), §3.3, Table 1)
found most edge pixels in stereo networks have multimodal output
distributions. See §2.3.

### 1.2 Documented fixes and how big they were

| Fix | Source | Measured effect | Cost here |
|---|---|---|---|
| Label-distribution smoothing (LDS) + feature smoothing (FDS) | DIR, Table 4 (NYUD2-DIR) | Few RMSE 2.123 → 1.880; **Many 0.591 → 0.670 (worse)** | LDS = a smoothed stratum weight, small code change |
| Balanced MSE (GAI / BNI / BMC) | Balanced MSE, Table 2 (NYUD2-DIR) | Few 2.123 → 1.703; **Many 0.591 → 0.692 (worse)**; on depth they fix σ_noise = 1 because inter-pixel dependency breaks BMC (§4.2.3) | New loss term |
| Inverse-frequency reweighting | Balanced MSE §1 and Fig. 3: "reweighting has limited effectiveness" and is sensitive to seed (supplementary Fig. 7) | qualitative | Already here (`StratumBalancer`) |
| Log-inverse sampling (milder than 1/freq) | Tolan et al., App. B.3: samples weighted by 1/ln(count per 1 m bin) "so as not to overly bias the model towards the relatively few high canopy height samples" | not ablated | Sampler change |
| Classification (256 uniform 1 m bins) instead of scalar regression | Tolan et al., Table 1: ViT-L Sat18M NEON MAE 2.9 → 2.7, **mean error −1.4 → −0.9 m**; São Paulo ME −1.9 → −2.1 | ~0.5 m less negative bias on NEON | Head B already exists, but the gate gives it ~¼ weight |
| Separate foreground / background bin distributions (head-tail cut at 1 m) | HTC-DC Net, Table VI | building RMSE 3.83 → 3.70 (LA), 12.83 → 12.43 (Guangzhou) | New head |
| AdaBins bin-centre Chamfer loss | AdaBins §3.4, Eq. 5–6 (β = 0.1), Table 4 | improves AbsRel; not tail-specific | Small; this repo has no Chamfer term (`bin_ce_loss` uses centres only to index) |
| Log / spacing-increasing bins | DORN ([arXiv 1806.02446](https://arxiv.org/abs/1806.02446), SID) | for depth, where far errors are tolerated | **Wrong direction for heights**: it coarsens exactly the tall tail |
| Post-hoc rescale with an independent sensor | Tolan et al. §3.3–3.4, Table 2: a GEDI model rescales the CHM, factor clipped to [0.5, 2] | reduces bias vs GEDI | Not available for buildings; the DEM-anchoring step in `infer/` plays a similar role for the absolute DSM |

The pattern is the same in every paper that measures it. Tail-favouring losses
buy tail accuracy with head accuracy. This repo already measured that trade: v4's
β 0.5 → 0.7 bought −0.17 MSE at 10–20 m for **+3.90 at 0–2 m**
(`v5_flags.py`, `REVERTS`). The judges score an absolute DSM against a 30 m DEM.
Over most of a scene the 30 m cell is dominated by ground, so a flat-ground
regression costs more than a tall-tail gain pays (here).

### 1.3 What that means for this repo (here)

1. **Measure before changing the loss.** Score on val (no training):
   - tall bias conditioned on prediction;
   - 30 m block-mean bias (`pool_pair(..., k=60)` at 0.5 m already exists in
     `eval/metrics.py`);
   - Head A, Head B and `fused` separately for tall bias.

   If prediction-conditioned and 30 m biases are ≈ 0, the model is calibrated.
   Then the −6 m is uncertainty. A loss change will only trade it against
   ground, and the right fix (if any) is a visual de-shrink at render time.
2. If you do push on the tail, push only **inside** the tail. Split the
   balancer's top stratum (20–40, 40 m+) and leave β / clip at the values v4
   proved safe. That redistributes weight among tall pixels without taking any
   from 0–2 m.
3. `silog_lambda 0.85 → 0.5` is the one config-only knob with a primary source
   for absolute scale (Eigen §3.3). It costs nothing to set. Its size here is
   unknown.

## 2. Soft edges

### 2.1 Is `grad = 0.23` a target? (here)

`grad_ratio` = Σ|∇pred| / Σ|∇GT| (`eval/metrics.py:126-147`). Three reasons it is
a poor target:

- A predictor that minimises L1 / L2 under any uncertainty is smoother than the
  GT, so its ratio is below 1 by construction.
- Noise in the GT inflates the denominator. MVS3DM's GT is unclassified LAZ
  gridded at 0.3 m (`tools/prep_mvs3dm.py` docstring) and is the noisiest
  reference here. NEON's 1 m CHM is the smoothest, and it reads 0.66.
- The ratio rewards any high-frequency output, including invented texture: the
  "texture becomes terrain" failure `flatness_loss` exists to stop.

Better edge measures, in the sources:

- Tolan et al.'s Edge Error compares Sobel maps of prediction and GT,
  normalised by both (App. C.3, Alg. 1). "Lower is better", and it does not
  reward noise.
- Depth Pro's boundary F1 ([arXiv 2410.02073](https://arxiv.org/abs/2410.02073), §3.2, Eq. 3)
  scores neighbouring-pixel ratio contours with precision and recall, weighted
  over thresholds from 5 to 25 %.

The repo's own `edge_rmse_m` (RMSE within 2 px of a GT step > 2 m) is already
this kind of measure, and it did improve.

### 2.2 Gradient-matching losses: already here, and more weight did nothing

- **MiDaS** (Ranftl et al., TPAMI, [arXiv 1907.01341](https://arxiv.org/abs/1907.01341), Eq. 11–12):
  multi-scale gradient matching of the residual, K = 4 scales, α = 0.5. "This
  term biases discontinuities to be sharp and to coincide with discontinuities
  in the ground truth." `gradient_loss` (`losses.py:148`) is this term:
  4 scales, |∇pred − ∇gt| equals |∇R|. MiDaS also trims the 20 % largest
  residuals (Eq. 7).
- **Depth Anything V2** ([arXiv 2406.09414](https://arxiv.org/abs/2406.09414), §5.2):
  L_ssi : L_gm = 1 : 2, and "L_gm is super beneficial to the depth sharpness
  when using synthetic images". §2 argues that real labels "overlook certain
  details … resulting in over-smoothed depth predictions", while synthetic
  labels get "all fine details … correctly labeled".
- **Depth Pro** (§3.2; App. B.5):
  - Table 11: adding gradient losses in stage 1 roughly doubles Hypersim
    boundary F1 (0.221 → 0.391 on all data, 0.442 with SSI-gradients on
    synthetic only).
  - Table 12: once gradient losses exist, adding more derivative terms
    (MSGE, Laplacian) moves F1 only 0.461 → 0.465.
  - Table 13: their synthetic-only "sharpening" stage 2 has *lower* boundary F1
    than single-stage training (0.465 vs 0.478). It buys metric accuracy, not
    edges.

**Here:** the repo is past the big step that Depth Pro's Table 11 measures, and
this run's doubling of `w_grad` left `grad` flat. More gradient weight, or a
synthetic-only sharpening phase, is not supported by the evidence for one run.

### 2.3 Readout: the cheapest edge lever, and it needs no training

Chen et al., ICCV 2019 (§3.3, Eq. 2–3, Table 2): replace the full-distribution
expectation with a **single-modal weighted average**. Find the argmax bin, walk
left and right while probability keeps descending, and renormalise inside that
window. The paper applies it at inference only. It "is consistently better than
full-band" on soft-edge error. Example: PDSNet trained with regression, Sceneflow
average SEE 1.97 → 1.32. Training with a Gaussian-smoothed CE target (σ = 2 bins)
lowered it further, to 1.04.

SMD-Nets (Tosi et al., CVPR 2021, [arXiv 2104.03866](https://arxiv.org/abs/2104.03866), §3, Fig. 2)
reach the same conclusion with a bimodal mixture whose most-likely mode is read
out: sharp transitions instead of "bleeding".

AdaBins (§3.3) gives the opposite caution. Reading the single argmax bin
centre, as DORN does, gives discretisation artefacts. The single-modal window
is the middle ground.

**Here:**

- Head B already outputs a 96-bin distribution, and the fused map is ~¾ Head A.
  So two readouts can be scored on the existing `best.pt` without training:
  1. Head B single-modal;
  2. `fused` with Head B replaced by its single-modal readout.
- This needs new code in `models/heads.py` (`HeadB.forward`, as an alternative
  to the einsum at line 145) behind an eval flag. Reach `eval_test.py` through
  the model.
- If it helps, ship it in `infer/` with no retraining.
- Caveat: `bin_soft_sigma` was reverted to 0 for ground-plane reasons
  (`v5_flags.py`). Chen's Gaussian-CE result does not override that measured
  v4 regression.

### 2.4 Resolution and upsampling

- **Source.** Depth Pro (§1): "a high resolution is necessary but not sufficient
  to improve boundary accuracy". FeatUp (ICLR 2024, [arXiv 2403.10516](https://arxiv.org/abs/2403.10516))
  and LoftUp ([arXiv 2504.14032](https://arxiv.org/abs/2504.14032)) are learned
  feature upsamplers for frozen foundation-model features. Each is a new module
  with its own training, and LoftUp's abstract reports no depth numbers.
- **Here.** The repo already has the cheap version:
  - a full-resolution RGB stem;
  - RAFT-style convex 2× upsampling of the heads (`heads.py`, `DetailStem`,
    `ConvexUp2x`).

  A new upsampler is not a one-run change. Inference at a finer effective GSD
  *is* free to test. A 16 px patch at 0.5 m is 8 m of ground, wider than many
  buildings. Training already saw 0.30 m crops (`gsd_jitter_lo_m 0.30`), so
  running at 1.5× zoom (0.33 m) stays inside the trained band:
  `eval_test.py --tta_scales 1.5` or `1.0,1.5`. That checks whether more tokens
  per building sharpens edges and lifts tall objects, before any GPU training.
- **Augmentation.** `photo_blur_p 0.3` and `aug_pansharp_p 0.5` deliberately
  degrade the input towards Cartosat. Depth Pro trains with the same 30 % random
  blur (Table 16), so this is not obviously harmful (here). Keep them.

## 3. Warm restarts and the encoder LR

### 3.1 Evidence

| Source | Setting | Finding |
|---|---|---|
| SGDR, Loshchilov & Hutter, ICLR 2017 ([arXiv 1608.03983](https://arxiv.org/abs/1608.03983), §3, Eq. 5) | from-scratch CIFAR / WRN | Restarts improve *anytime* performance. They "often temporarily worsen performance", so the incumbent is the end-of-cycle model. "It might be of great interest to decrease η_max … at every new restart." |
| Gupta et al. 2023 ([arXiv 2308.04014](https://arxiv.org/abs/2308.04014), §1 findings, §4.4) | re-warming a converged LM | Re-warming "first increases the loss on upstream and downstream data", then improves. A higher max LR means more adaptation and more forgetting. **Same-data re-warm** (§4.4) shows the same rise-then-fall: optimisation dynamics, not just data shift. LM evidence; transfer to ViT dense regression is unverified. |
| SWA, Izmailov et al., UAI 2018 ([arXiv 1803.05407](https://arxiv.org/abs/1803.05407), §4.2 Table 2, §4.3) | 10 epochs of cyclic LR **from converged** torchvision ResNet / DenseNet | Averaging the weights gives **+0.6 to +0.9 % ImageNet top-1**. The best constant LR is "some intermediate value between the largest and the smallest learning rate" of the original schedule. An EMA under a *decaying* schedule "perform[s] comparably to conventional SGD at convergence" (§4 intro). |
| Model soups, Wortsman et al., ICML 2022 ([arXiv 2203.05482](https://arxiv.org/abs/2203.05482)); WiSE-FT, CVPR 2022 ([arXiv 2109.01903](https://arxiv.org/abs/2109.01903)) | fine-tunes from one init | Fine-tuned models "often appear to lie in a single low error basin". Averaging their weights improves accuracy and robustness. WiSE-FT: +4 to 6 pp under distribution shift, no extra inference cost. |

**Here.**

- The last run *was* a warm restart from `resume-v4-1.6`, with LR re-warmed to
  1.5e-4 over 5 %. It showed no visible damage at the first eval (MVS3DM 1.615
  → 1.602 after epoch 1), but its gain is confounded with adding NEON.
- A second cycle on **unchanged** data and loss is the SGDR/SWA case. Expect a
  small gain, of the order of the last three epochs' drift (2.512 → 2.481). That
  is a guess: none of these sources measured dense regression.
- The averaging part has the better evidence. The repo's EMA (0.9995 ≈ a
  2000-step horizon, under a cosine to 1 %) is the "EMA under a decaying
  schedule" that SWA found adds little.
- A flat LR segment with uniform averaging is what SWA measured.
- Averaging `prev/best.pt` with this run's `best.pt` (WiSE-FT style) is a
  zero-training experiment. Both share an init lineage. Check it on the NEON
  forest numbers, because prev never saw NEON.

### 3.2 Encoder LR and layer decay in the reference repos

| Model / repo | Encoder LR | Decoder LR | Ratio | Layer decay | Schedule |
|---|---|---|---|---|---|
| Depth Anything V2 metric ([`metric_depth/train.py`](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/metric_depth/train.py) L36, L100–102, L142–145) | 5e-6 | 5e-5 | 1 : 10 | none | poly 0.9, AdamW wd 0.01, bs 2/GPU, 518 px |
| Depth Pro (Table 16) | 1.28e-5 | 1.28e-4 | 1 : 10 | none; pretrained LayerNorm frozen | 1 % warmup, 80 % constant, 19 % ×0.1 |
| BEiT fine-tune ([arXiv 2106.08254](https://arxiv.org/abs/2106.08254), App. H–I) | layer-decayed | n/a | — | **0.65** (B, ImageNet and ADE20K), **0.75** (L) | cosine |
| Tolan et al. (§3.2) | frozen ("best results were obtained by freezing all layers") | — | — | — | one-cycle |
| DINOv3 sat canopy height ([arXiv 2508.10104](https://arxiv.org/abs/2508.10104), §8.2, Table 17) | frozen | DPT only | — | — | — |
| **This repo, last run** | 3e-5 (top block) | 1.5e-4 | 1 : 5 | 0.9 over 24 blocks → 2.4e-6 at the embeddings | cosine to 1 % |

**Here.**

- The encoder top-block LR is 6× Depth Anything V2's metric fine-tune, and the
  ratio is half the 1 : 10 that both DAV2 and Depth Pro use.
- The two satellite-height references freeze the encoder entirely.
- None of this shows the current LR is wrong: it has been fine-tuned over
  several runs and is past its pretrained state. It does argue that a *second*
  cycle should not re-warm the encoder to 3e-5.
- 0.3–0.5× the last peak follows SGDR's "decrease η_max" and Gupta's
  forgetting trade, and keeps the encoder near DAV2's regime.

## 4. Other levers with primary evidence

- **Trimmed losses for noisy LiDAR.** These come from MiDaS (Eq. 7, top 20 %
  dropped), Depth Pro (top 20 % per image on real data, Table 17) and DAV2
  (top 10 % on pseudo-labels, §5.2). **Here:** do not add one in this run.
  Trimming the largest residuals per tile removes exactly the under-predicted
  tall pixels and would worsen (a).
- **Off-nadir parallax** (Christie et al., CVPR 2020,
  [arXiv 2007.00729](https://arxiv.org/abs/2007.00729), §3.3, §4.1). US3D labels
  are projected into each oblique view. Jacksonville and Omaha, the cities in
  US3D val here, have "limited off-nadir angles", so oblique view geometry
  explains little of the US3D tall error.
- **Not checked in depth:** IM2HEIGHT
  ([arXiv 1802.10249](https://arxiv.org/abs/1802.10249)) and the GAMUS paper.
  I found no tail-specific treatment cited from them, and I did not read them in
  full. Metric3D / UniDepth's canonical-camera idea is already this repo's
  `canonical_gsd_m` + GSD jitter.
- **SynRS3D** ([arXiv 2406.18151](https://arxiv.org/abs/2406.18151), §4, Eq. 1)
  trains height with plain Smooth-L1 on DINOv2 + DPT. It gives no tail-specific
  recipe to copy.

## Recommendation for the next ~2 h run

Budget: H100 at $3.95/h plus 2 cores / 8 GiB is ≈ $8.2 for 2 h
(`final_modal.py` docstring), out of ~$9.80. So step 1 must run free on Kaggle,
not on Modal. Ranked by value per dollar:

**1. Eval gate before spending: free on Kaggle, no training.**
- **What.** Score `best.pt` on MVS3DM, US3D and GAMUS val/test with:
  - `eval_test.py --tta_scales 1.5` (CLI only);
  - three additions that need new code in `eval/metrics.py`:
    1. tall bias conditioned on the prediction;
    2. 30 m block-mean bias via `pool_pair(k=60)`;
    3. Head A / Head B / `fused` tall bias plus a Head B single-modal readout
       (§2.3, new code in `models/heads.py`).
- **Expected effect.** It decides whether (a) is a fixable bias or calibrated
  uncertainty. It decides whether (b) can be fixed at inference: single-modal
  readout, 1.5× zoom. If either readout helps, ship it in `infer/` and skip any
  loss change aimed at it.
- **Risk.** None to the model; ~1 h of Kaggle time.
- **Source.** Probability argument (§1.1); Chen et al. ICCV 2019 Table 2;
  SMD-Nets; Depth Pro §1.

**2. Tail-only loss rebalancing. Do it only if step 1 shows prediction-conditioned or 30 m bias on tall areas.**
- **What.**
  - Give the balancer its own strata `(0, 2, 5, 10, 20, 40, ∞)`. This needs new
    code: a `LOSS_STRATA_M` constant in `config.py` used by `_STRATA` in
    `models/losses.py:51`. Leave `HEIGHT_STRATA_M` alone so `bal=` stays
    comparable across runs.
  - Keep `stratum_balance_beta 0.5` and `stratum_weight_clip 5.0`.
  - Set `silog_lambda 0.5`; this is config only.
- **Expected effect.** Some of US3D's −9.7 m at 20 m+ and MVS3DM's −3.6 m at
  10–20 m recovered; the size is unknown. My guess is 1–3 m on US3D 20 m+, at
  +0.05–0.15 m flat bias.
- **Risk.**
  - Medium: every source shows a head cost (DIR Table 4, Balanced MSE Table 2),
    and v4 paid +3.90 MSE at 0–2 m for a β change.
  - `select_on` (NEON + MVS3DM forest/sparse) cannot see US3D/GAMUS urban
    tall gains, so judge this run on the logged per-source `tall_bias`, not
    `select`.
- **Source.** DIR; Balanced MSE; Tolan App. B.3 (mild tail weighting); Eigen
  §3.3 (λ = 0.5 for absolute scale).

**3. A second, lower cosine cycle with real weight averaging.**
- **What.**
  - Set `learning_rate 6e-5` and `encoder_lr 1.2e-5` (0.4× last peak), keeping
    `llrd 0.9` and `warmup_frac 0.05`.
  - Warm start from `best.pt` via `prev/best.pt`, as the launcher already does.
  - Add uniform averaging of the weights over the last ~40 % of steps. This
    needs new code in `train.py`, next to `ModelEMA`:
    `torch.optim.swa_utils.AveragedModel`, or an EMA whose decay follows
    1 − 1/n. The net has only GroupNorm / LayerNorm, so there are no
    BatchNorm statistics to recompute.
- **Expected effect.** Small: `select` ≈ 2.43–2.47 (my estimate). Lower
  variance run-to-run.
- **Risk.**
  - Low at 0.4×.
  - At the full 1.5e-4 / 3e-5, expect an early rise first (SGDR §3, Gupta
    §4.4), which a 2 h budget may not fully recover.
- **Source.** SGDR §3; Gupta et al. §1, §4.4; SWA Table 2 and §4 intro.

**4. Do not raise `w_grad` or add a synthetic-only sharpening phase.** This is a
negative recommendation. It is listed because it is the obvious move and the
evidence says it will not pay:
- The last run doubled `w_grad` and `grad` stayed at 0.23.
- Depth Pro's extra derivative losses moved F1 by 0.004 (Table 12).
- Its synthetic-only stage 2 lowered F1 (Table 13).

Put the edge effort into step 1's readout and zoom tests, and judge edges on
`edge_rmse_m` (or a Tolan-style edge error), not `grad`.

Where evidence is thin:
- No primary source measures any of these changes on a satellite nDSM with a
  DINOv3 encoder.
- The tail-loss numbers come from NYU indoor depth and age regression.
- The warm-restart numbers come from classification and language models.
- The single-modal readout numbers come from stereo cost volumes.

The effect sizes above are therefore direction-of-effect claims, not forecasts.
