# Outcomes — versions & experiments

Fill in after each Kaggle run. Numbers here feed the idea deck.

## v1 — Phase 0 spike (`FineTunning/v1/kaggle_phase0.py`)

DINOv3-SAT (frozen) → DPT decoder → metric-nDSM head (Head A only).
GAMUS subset (`train_subset` / `val_subset`), 512² crops @ native 0.33 m GSD.

| Run | Encoder | Train / Val tiles | Epochs | Val RMSE (m) | MAE (m) | Pearson r | δ1 | Notes |
|---|---|---|---|---|---|---|---|---|
| _pending_ | dinov3-vitl16-sat493m | 1200 / 300 | 12 | — | — | — | — | first run |

Per-land-cover val RMSE (from `outputs/v1/metrics.json`):

| building | vegetation | ground | road | water | bridge |
|---|---|---|---|---|---|
| — | — | — | — | — | — |

### Baseline — IMELE on the same GAMUS val split (`eval_imele_on_gamus.py`)

| Variant | RMSE (m) | MAE (m) | Pearson r |
|---|---|---|---|
| raw metres | — | — | — |
| per-tile scale+offset aligned | — | — | — |
