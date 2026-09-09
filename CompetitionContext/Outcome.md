# Outcomes — versions & experiments

Fill in after each Kaggle run. Numbers here feed the idea deck.

## v1 — Phase 0 spike (`FineTunning/v1/kaggle_phase0.py`)

DINOv3-SAT (frozen) → DPT decoder → metric-nDSM head (Head A only).
GAMUS subset (`train_subset` / `val_subset`), 512² crops @ native 0.33 m GSD.
Source log: `FineTunning/v1/Phase0_Cell_Ouptut.md` (full run, 12 epochs, 28 min).

| Run | Encoder | Train / Val batches | Epochs | Val RMSE (m) | MAE (m) | Pearson r | δ1 | Notes |
|---|---|---|---|---|---|---|---|---|
| e11 (best) | dinov3-vitl16-pretrain-sat493m | 1200 / 300 | 12 | **3.321** | 1.632 | 0.856 | 0.620 | new best → `best.pt` |
| e12 (final) | dinov3-vitl16-pretrain-sat493m | 1200 / 300 | 12 | 3.340 | 1.622 | 0.855 | 0.626 | last epoch |

Per-epoch val RMSE (m): e1 5.29 · e2 4.14 · e3 4.18 · e4 4.34 · e5 3.61 · e6 3.51 ·
e7 3.39 · e8 3.37 · e11 3.32 · e12 3.34 — steep drop to e5, then a shallow plateau
(3.6 → 3.3 over 7 more epochs).

Per-land-cover val RMSE / MAE (m), epoch 11 (from the eval log):

| | ground | vegetation | building | water | road | bridge | other |
|---|---|---|---|---|---|---|---|
| RMSE | 4.520 | 1.475 | 1.506 | 4.862 | 1.684 | 2.567 | 4.882 |
| MAE  | 3.265 | 0.454 | 0.479 | 3.046 | 0.390 | 1.018 | 3.573 |
| n (px) | 41 975 | 8.68 M | 10.90 M | 7.58 M | 0.59 M | 7.27 M | 8.85 M |

**Read:** structure classes (vegetation / building / road) are already good (RMSE ≈ 1.5 m);
the error is concentrated in `water`, `other`, and `ground` (RMSE 4.5–4.9 m) — flat/low
classes where absolute metres and the long tail hurt most. Early epochs had `ground`
RMSE ≈ 2 m and it *drifted up* as the model over-fit tall structures. This is the case for
v2's Head B (adaptive bins for the tail) + GSD jitter + more varied terrain data.

### Configuration (v1 Phase 0)

| Field | Value |
|---|---|
| Encoder | `facebook/dinov3-vitl16-pretrain-sat493m`, **frozen**, fp16 weights (`encoder_half=True`) |
| Encoder taps | hidden_states (6, 12, 18, 24); patch 16; 5 prefix tokens; hidden 1024 |
| Decoder | DPT (reassemble → RefineNet fusion), `decoder_dim=256`, grad-checkpointed |
| Heads | Head A only — metric nDSM regression, `ReLU` (non-negative) |
| Trainable params | 11.3 M / 314.4 M total |
| Input | 512×512 crop (random for train, centre for val) @ native 0.33 m GSD, ImageNet norm |
| Targets | AGL/nDSM in metres, valid mask = finite ∧ 0 ≤ h ≤ 400 m |
| Loss | `1.0·L1 + 0.5·grad(4-scale) + 1.0·SiLog(λ=0.85, shift=1.0)` |
| Optim | AdamW lr 3e-4, wd 1e-2, OneCycleLR (pct_start 0.1), grad-clip 1.0 |
| Batch | `batch_size=4`, `grad_accum=2` defaults; run log shows `nn.DataParallel` over 2×T4, per-GPU batch ~1 |
| Precision | AMP (fp16 autocast), `channels_last`, `cudnn.benchmark=False` |
| Hardware / time | 2× Tesla T4 (Kaggle), 28 min wall-clock, hard cap 150 min |
| Seed | 42 |
| Artifacts | `outputs/v1/{metrics.json, best.pt (decoder+head, no encoder), viewer_sample/}` |

### Baseline — IMELE on the same GAMUS val split (`eval_imele_on_gamus.py`)

| Variant | RMSE (m) | MAE (m) | Pearson r |
|---|---|---|---|
| raw metres | _pending_ | — | — |
| per-tile scale+offset aligned | _pending_ | — | — |

_Needs `Notebooks/depth-wiz.ipynb` run with `IMAGE_INPUT` = the GAMUS val RGB tiles, then
`python eval_imele_on_gamus.py --pred_dir … [--align_scale]`._

---

## v2 — Phase 2 (`FineTunning/v2/`)  _(in progress)_

Streamed **SynRS3D pretrain → GAMUS + GeoNRW fine-tune**. DINOv3-SAT encoder (frozen, last
4 blocks unfrozen for the final ~30% of epochs) → shared DPT trunk →
**Head A (metric) + Head B (adaptive bins) + Head C (6-class semantics) + gated fusion**.
GSD canonicalization (0.5 m) + jitter (0.25–2.0 m). 8× dihedral + 2-scale TTA at eval.
Target hardware: 1× RTX PRO 6000 (96 GB), ~6–8 GPU-h.

| Run | Stage | Datasets | Epochs | Val RMSE (m) | RMSE+TTA (m) | MAE | r | δ1 | Notes |
|---|---|---|---|---|---|---|---|---|---|
| _pending_ | P + F | SynRS3D → GAMUS+GeoNRW | 6 + 30 | — | — | — | — | — | first v2 run |
