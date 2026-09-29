# DepthWizard — Phase 0 (v1 spike)

**Goal:** a first *real* GAMUS validation RMSE for the idea deck, from the
architecture the plan actually pitches — not the IMELE notebook.

> DINOv3-SAT (frozen) → DPT decoder → single metric-nDSM regression head (Head A only).
> No bin head, no semantics head, no DEM calibration — those are Phase 2.
> See `../../.agents/Depth_Wizard_Plan.md`.

## Files

| File | Role |
|---|---|
| `kaggle_phase0.py` | **the whole spike in one file** — GAMUS download, model, train, eval, viewer export |
| `eval_imele_on_gamus.py` | IMELE baseline row on the *same* GAMUS val split, *same* metric code |
| `../../viewer/phase0_viewer.html` | bare Three.js scene — drop in the exported sample, get the screenshot |
| `config.py` / `train.py` / `main.py` | thin shims so `python main.py` still works |

## Run on Kaggle (GPU T4 ×2)

1. New notebook → **Settings: Accelerator = "GPU T4 x2", Internet = ON**.
2. DINOv3-SAT is **gated**. Accept the license at
   <https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m>, create an
   HF token, then **Add-ons → Secrets → add `HF_TOKEN`**.
3. In a cell:

   ```python
   !git clone https://github.com/<you>/SIH.git
   %cd SIH/FineTunning/v1
   !python kaggle_phase0.py
   ```

   or `!python main.py` (same thing; Step 1 installs deps first).

Defaults: 1200 train / 300 val tiles, 512² crops at native 0.33 m GSD, batch 8
across both T4s, 12 epochs, hard stop at 150 min. First eval prints after epoch 1.

### Useful overrides

```bash
# faster smoke run
!python kaggle_phase0.py --train_subset 300 --val_subset 120 --epochs 4 --max_train_minutes 40

# lighter encoder if the SAT weights are a problem
!python kaggle_phase0.py --encoder_model_id facebook/dinov3-vitb16-pretrain-lvd1689m --batch_size 12

# use BOTH T4s (nn.DataParallel) — only on an image where DP is known good.
# torch>=2.10 / cu12.8 images crash with it; the default is single-device.
!python kaggle_phase0.py --data_parallel true

# out of memory? levers, most effective first:
!python kaggle_phase0.py --train_size 448          # token count is quadratic
!python kaggle_phase0.py --batch_size 2 --grad_accum 8
!python kaggle_phase0.py --decoder_dim 192
!python kaggle_phase0.py --encoder_half false      # only if fp16 encoder hurts val metrics

# use a pre-mounted Kaggle Dataset instead of downloading
!python kaggle_phase0.py --data_source local --data_root /kaggle/input/gamus
```

## Outputs (`/kaggle/working/outputs/v1/`)

- `metrics.json` — per-epoch train loss + val **RMSE / MAE / Pearson r / δ1**, global
  **and per land-cover class** (ground / vegetation / building / water / road / bridge / other).
  *(class→name mapping is a best guess — confirm against the GAMUS paper before the deck.)*
- `best.pt` — decoder + head weights (encoder is frozen & public, not saved).
- `viewer_sample/` — `rgb.png`, `height16.png`, `pred_ndsm_m.npy`, `meta.json`
  → open `viewer/phase0_viewer.html`, load those 3 files, screenshot.

## IMELE baseline row

```python
# 1. dump GAMUS val RGBs to PNG for the existing notebook
%cd SIH/FineTunning/v1
python -c "from eval_imele_on_gamus import dump_gamus_val_pngs as d; d('/kaggle/working/data/gamus','/kaggle/working/gamus_val_png')"
# 2. run Notebooks/depth-wiz.ipynb with IMAGE_INPUT = that folder  -> imele_outputs/*.npy
# 3. score it with the same metrics
python eval_imele_on_gamus.py --pred_dir /kaggle/working/imele_outputs --align_scale
```

`--align_scale` least-squares fits a per-tile scale+offset first (IMELE's public
checkpoint is Dublin-only at 0.5 m/px, so raw metres won't match GAMUS) — report
both the aligned and raw numbers.

## Known Phase-0 shortcuts (deliberate; fixed in Phase 2)

- Val = deterministic centre `train_size` crop, not the full tile.
- No GSD canonicalisation/jitter — trains and evals at GAMUS's native 0.33 m.
- Single-device by default (`--data_parallel true` for DP where it's stable; no DDP).
- Frozen encoder loaded in fp16 by default (`--encoder_half false` to disable).
- Semantic mapping for per-class metrics is assumed, not verified.
