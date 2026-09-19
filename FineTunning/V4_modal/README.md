# DepthWizard v4 on Modal — 1× H100, ~$14 a run

Glue only: `modal_app.py` mounts [`../V4_Kaggle`](../V4_Kaggle) verbatim at
`/root/dw` and shells out to its `train.py` / `prepare_data.py` /
`pack_gamus_png.py`. No library code is duplicated, and editing that tree needs
no rebuild (`copy=False`). Its DDP machinery short-circuits at `world_size == 1`,
so we run a plain `python train.py` — no `torchrun`, no NCCL env.

Why `V4_Kaggle` and not `v4`: same architecture, but it carries the fixes the two
real runs paid for — fp32 losses, the GradScaler floor, `last_full.pt` resume,
DFC23 ingest, aux val sets. **All flag rationale lives in
[`V4_Kaggle/README.md` §5.2](../V4_Kaggle/README.md)** and is not repeated here.

## Setup

```bash
cd FineTunning && uv sync && .venv/bin/modal setup
modal secret create dw-hf      HF_TOKEN=hf_...          # DINOv3-SAT is gated
modal secret create dw-kaggle  KAGGLE_USERNAME=... KAGGLE_KEY=...   # prepare only
```

Accept the licence for `facebook/dinov3-vitl16-pretrain-sat493m` first, or the
image build fails — it bakes the encoder in so cold starts make no hub calls.

## Run

Cheapest first. Nothing runs an H100 until the two above it pass.

```bash
modal run modal_app.py::show_tuning            # $0     flags + cost, no container
modal run modal_app.py::check                  # ~$0.15 image, flags, 175 tests
modal run --detach modal_app.py::prepare       # ~$1.30 build the pack, ONCE
modal run modal_app.py::smoke                  # ~$1    ~10 min
modal run --detach modal_app.py::train --epochs 3 --extra '--max_minutes 20'
modal run --detach modal_app.py::train         # ~$14   the real run
modal app logs depthwizard-v4
```

`--extra` is appended raw and always wins. `finalize` re-runs only the exports
from a committed `metrics.json` — ~$2 instead of ~$14 when a container dies
after training but before them.

Get the weights out (`best.pt` alone is useless — `preproc.json` is the
preprocessing contract `infer/engine.py` reads back):

```bash
for f in best.pt metrics.json config.json preproc.json; do
  modal volume get depthwizard-results v4m/$f .; done
```

## Cost

$3.95/h H100 + ~$0.95/h for 12 cores and 48 GiB, at ~45 img/s.

| epochs | hours | $/run |
|---:|---:|---:|
| 26 | 2.6 | ~12.9 |
| **30** (default) | **2.9** | **~14.3** |
| 40 | 3.7 | ~18.0 |

Two full runs plus a one-off `prepare` (~$1.30) is about the whole $30 of
monthly credit. **`IMG_PER_S = 45.0` is an estimate from the v3 H100 run, not a
Modal measurement** — the 3-epoch calibration above replaces it, and the whole
table moves with it.

The RTX PRO 6000 (96 GB, $3.03/h) matches an H100 on BF16 tensor throughput but
has 1.79 TB/s against 3.35 TB/s; the DPT convs, GroupNorms and fp32 loss stack
are bandwidth-bound, so it lands ~0.75× — and Modal bills CPU/RAM *per hour on
top*, so the slower card pays that overhead longer. ~$16/run, and slower.

## The three things that are Modal-specific

**Staging is not optional.** The trainer opens every shard `mmap_mode="r"` and
reads random ~0.79 MB windows ~45/s over ~36 GB. That is what
`Logs/v3/gpu_disk_guard.log` calls `STARVED` off a network mount, and a Volume
is FUSE-backed — so `prepare` packs to local `/scratch` and copies over
sequentially, and `train` rsyncs back before starting. `DW_NO_AUTOSTAGE=1` to
measure the difference once.

**`--session_minutes 0`.** That second cap firing before the other two is what
truncated v4-2's cosine at 81% and cost 0.36 m. One container has no 12 h
session to survive. `--max_minutes` still *sizes* the cosine, so `::train
--epochs N` re-derives it — never move one without the other.

**GAMUS comes from the gated HF repo**, not the Kaggle PNG mirror. The mirror
existed because the fetch used to spend two requests per file (~30 000 for
train, HEAD then GET) and re-spend them on every resume, so it never got past a
429; `prepare_data.py` now lists the tree once and fetches each file with one
ranged GET, skipping what is already complete. The mirror's cost was the class
rasters — `has_seg=no`, so `seg_ce_loss` was 0.0 every step.

## Pre-flight before the H100

`pre_flight.ipynb` is a CPU notebook that checks every assumption
`02_train.ipynb` makes, for well under a dollar: the `index.json` counts, the
shards themselves, the held-out contract (`gamus/test` disjoint from train and
val, `--test_sources gamus:val` *rejected*), the 175 offline tests, a val-vs-test
distribution comparison, and a full `train.main()` end to end on the real
Volume with a stub encoder. It ends in one verdict line. Run it first.

## Read `prepare`'s tile counts

v3 wrote a `gamus/val` holding 129 of 400 tiles and still reported success.
Expect `gamus/train ~5004`, `gamus/val ~859`, `gamus/test ~2861`, `synrs3d_g1`,
`synrs3d_g05`, `dfc23_g050/train ~1506`, `dfc23_g050/val ~266` — every line
saying `bounds`, not `NO-BOUNDS`. `gamus/test` is held out: it is packed, it is
never in `--datasets`, and no epoch or in-training eval touches it.

## The held-out test split

`02_train.ipynb` passes `--test_sources gamus:test`. Nothing else changes: the
sampler, the val loader and the `best.pt` decision are exactly what v4 ran, so
the run-to-run comparison survives. What is added is one pass at the very end,
after the checkpoint is frozen — `test_gamus_test_plain` and `_tta` over all
2861 tiles, `_sliding_tta` over the first 400 (`--test_sliding_tiles`, because
sliding+TTA is ~6 s/tile and the whole store would be ~4.8 unattended H100
hours).

**Quote a `test_*` number, not a `final_*` one.** `best.pt` is selected on the
first 400 `gamus/val` tiles and `final_plain` / `final_tta` /
`final_sliding_tta` are reported on those same 400 — the v4 headline was
measured on the set that chose the checkpoint. `final_*` stays in `metrics.json`
because it is the only thing comparable with v1–v4.

`--test_sources` is checked at parse time: a split the run also trains or
validates on (`gamus:train`, `gamus:val`) is rejected rather than reported as
held out, and a store that is named but absent raises instead of quietly
leaving the val number as the headline.

## Watch in the log

Skipped-gradient fraction ~0 (bf16 disables the scaler; anything else means
`--amp_dtype` didn't take) · `nrm=` never `nan` · **eval RMSE changing between
epochs** — byte-identical lines are a dead epoch · `vram=` flat near 58 G, not
creeping · `[sched]` at startup saying the cosine reaches 1.0 · `img/s` near 45,
or the loader is starving (check `/dev/shm` in `check`).

## Limits

Modal's rates are hardcoded in `modal_app.py` and drive the projection only.
`torch==2.10.0+cu128` is pinned because `requirements.txt` deliberately omits
torch; bump `TORCH` if that wheel is gone. Scope is one good `best.pt` — no
serving endpoint, no publish step. `python test_modal_app.py` checks the argv
plumbing with no account and no GPU.
