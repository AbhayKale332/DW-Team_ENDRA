# DepthWizard v2 — streamed, multi-dataset, 3-head training

v1 (`../v1/`) is left untouched. v2 implements the `.agents/Depth_Wizard_Plan.md`
Phase-2 architecture:

```
RGB tile (canonical 0.5 m GSD, GSD-jittered in training)
  └─ DINOv3-SAT ViT-L/16  [frozen; last 4 blocks unfrozen for the final ~30% of stage F]
       └─ shared DPT trunk
            ├─ Head A  metric nDSM regression        (SiLog + L1 + gradient)
            ├─ Head B  adaptive-bin classifier        (AdaBins-lite; fixes the long tail)
            ├─ Head C  7-class semantic segmentation   (free GAMUS/GeoNRW/SynRS3D labels)
            └─ gated fusion α·A + (1-α)·B  →  final nDSM
  └─ eval: 8× dihedral + 2-scale TTA
```

Data: **SynRS3D pretrain → GAMUS + GeoNRW fine-tune**, all *streamed* through a
bounded-LRU cache (`data/streaming.py`) so a 96 GB-VRAM / ~100 GB-disk vast.ai box
never has to hold the full ~260 GB of source data.

## Layout

| Path | Role |
|---|---|
| `config.py` | one dataclass, every knob, `--flag` overridable |
| `data/streaming.py` | `BoundedCacheHF` — on-demand HF fetch + size-capped LRU eviction (files *and* archives) |
| `data/{gamus,geonrw,synrs3d}.py` | per-source tile readers → a common sample dict |
| `data/gsd.py` | GSD canonicalisation (eval) + jitter (train) |
| `data/loaders.py` | stage-F weighted `ConcatDataset`; stage-P archive-rotating dataset |
| `models/{encoder,dpt,heads,losses,tta}.py` | the network + losses + TTA |
| `eval/{metrics,report}.py` | global + per-class + per-height-stratum metrics; metrics.json / qualitative PNGs |
| `train.py` | two-stage budget-capped trainer → checkpoints → zip → cloudflared |
| `package_results.py` | `results_v2.zip` + `cloudflared tunnel --url` share |
| `tests/` | `pytest` — offline unit tests + `-m hf` online tests |

## 1. Test first (Kaggle, or anywhere with torch)

```bash
cd FineTunning/v2
pip install -r requirements-kaggle.txt
pytest -q -m "not hf"          # offline: config, streaming LRU, GSD, model, losses, TTA, metrics, checkpoint, packaging, mini-train
pytest -q -m hf                # online: pulls 2 real tiles per HF repo (needs HF_TOKEN for GAMUS)
```

Then a real (tiny) end-to-end run on a Kaggle **T4**:

```bash
python main.py --smoke --datasets gamus
# ~15 min. Expect: outputs/v2/metrics.json with a finite RMSE, best.pt, results_v2.zip
```

Only move to vast.ai once both of the above are green.

## 2. Full run (vast.ai — 1× RTX PRO 6000, 96 GB, 48 vCPU)

```bash
git clone <repo> && cd SIH/FineTunning/v2
pip install -r requirements-kaggle.txt
export HF_TOKEN=...            # DINOv3-SAT + GAMUS are gated
python main.py                # pretrain → finetune → TTA → zip → cloudflared
```

At the end it prints:

```
DOWNLOAD: https://<random>.trycloudflare.com/results_v2.zip
```

Pull it into Kaggle:

```python
!wget -q https://<random>.trycloudflare.com/results_v2.zip && unzip -o results_v2.zip
```

Useful overrides:

```bash
python main.py --skip_pretrain true          # GAMUS+GeoNRW only  (~3.5-4.5 h)
python main.py --datasets gamus --pretrain_dataset ""   # v1-style single dataset  (~2-3 h)
python main.py --cache_max_gib 70            # bigger LRU cache if disk allows
python main.py --batch_size 48 --num_workers 24
python main.py --init_from outputs/v2/stageP_last.pt --skip_pretrain true   # resume at stage F
python main.py --data_source local --local_root /kaggle/input/geonrw        # pre-mounted data
```

## 3. Expected wall-clock (RTX PRO 6000, single card)

| Stage | Data | ~Epochs | Est. time | Notes |
|---|---|---|---|---|
| P — SynRS3D pretrain | rotating 2 archives/epoch, ≤20 k tiles | 6 | **1.5–2.5 h** | includes zip download+extract I/O; capped at `--pretrain_minutes 120` |
| F — GAMUS+GeoNRW finetune | ~6.3 k GAMUS + ~1.7 k GeoNRW, 3:1 sampled | 30, unfreeze @ e21 | **3.0–4.5 h** | GAMUS warms the cache after epoch 1; capped at `--finetune_minutes 300` |
| Final eval + 8× TTA | ~600 val tiles | — | **15–25 min** | plain + TTA both reported |
| Package + tunnel | — | — | **~5 min** | |
| **Total** | | | **≈ 5–7.5 h** | fits a 6–8 GPU-h session; caps guarantee it finishes |

If the budget is tight, `--skip_pretrain true` drops ~2 h with a modest accuracy
cost; SynRS3D pretrain is a "stretch" in the plan, not a dependency.

## 4. What the run saves (inside `results_v2.zip`)

| File | Contents |
|---|---|
| `best.pt` | DPT trunk + Head A/B/C + fusion (+ any unfrozen encoder blocks). Encoder backbone **not** saved (frozen + public). ~150–250 MB. |
| `stageP_last.pt` | pretrain checkpoint → stage F is re-runnable without redoing P |
| `last.pt` | last finetune epoch |
| `metrics.json` | per-epoch train loss; val **RMSE / MAE / Pearson r / δ1** — global + per-land-cover + per-height-stratum (0–2 / 2–5 / 5–10 / 10–20 / 20 m+) + **balanced RMSE**; `final_plain` vs `final_tta` |
| `config.json` | full resolved config (no token) |
| `run.log` | complete console output (stdout+stderr teed) |
| `pip_freeze.txt`, `env.txt` | reproducibility (torch / CUDA / GPU) |
| `viewer_sample/` | `rgb.png`, `height16.png`, `pred_ndsm_m.npy`, `gt_ndsm_m.npy`, `meta.json` → drop into `../../viewer/phase0_viewer.html` for the deck screenshot |
| `qualitative/` | RGB \| pred \| GT \| error strips across land-cover types |

## 5. Known v2 shortcuts (deliberate; revisit for the finals)

- **GeoNRW & SynRS3D are archive-based on HF** (a 32 GB tar / 17 zips), not
  per-tile — v2 downloads + extracts them into the LRU cache and reads locally.
  Budget the extra I/O (≈15–40 min of the first stage).
- **GeoNRW nDSM is a proxy**: `clip(DEM − smoothed-large-window-minimum, 0)`, since
  GeoNRW ships an elevation DEM, not an nDSM. Fine as auxiliary supervision; not a
  precise target.
- **SynRS3D per-archive GSD** is a rough prior; GSD jitter dominates during pretrain.
- Val = deterministic centre crop at canonical 0.5 m GSD (no full-tile stitching yet).
- No DEM/DTM calibration, no ONNX export, no viewer — those are plan Phase 3.
- Semantic class maps (GeoNRW-11→7, SynRS3D-8→7) are best-effort; verify against
  each dataset paper before quoting per-class numbers in the deck.
