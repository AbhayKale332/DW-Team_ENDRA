# DepthWizard v4 — the two-Studio runbook

Same shape as v3's: the data stage is network-bound and the training stage is
GPU-bound, so they run on different machines and the expensive one never
downloads anything.

```bash
export HF_TOKEN=hf_...          # DINOv3-SAT and GAMUS are both gated

# ---- Studio 1: any cheap CPU box ------------------------------------------
sh shell_scripts/v4/prepare_data.sh --background
tail -f ~/DepthWizard-results/logs/prepare_v4_*.log

# ---- switch the Studio's machine to 1x H100 80 GB, then -------------------
sh shell_scripts/v4/train_h100.sh --background
tail -f ~/DepthWizard-results/logs/train_v4_*.log
```

| file | what it does |
|---|---|
| `prepare_data.sh` | fetch → pack → prefetch the encoder → precompute stretch bounds → stamp READY |
| `train_h100.sh` | preflight → stage to local NVMe → train → evaluate → figures/report/ONNX → manifest |
| `dw_fetch.py` | one bulk `snapshot_download` per repo, so the pack can run offline |

## What is different from `shell_scripts/*.sh` (v3)

**The training script no longer autotunes the batch.** v3's did, and it produced
both OOMs in `Logs/v3/run.log`: it read `VRAM >= 70 GiB` and set batch 32, which
reached epoch 7, drifted from 70 GB to 76 GB and died on a 768 MB allocation.
v4's `config.py` already holds the sizing, derived from v3's own measurements
(micro-batch 16 → 42 GB / 44.7 img/s; micro-batch 32 → 74 GB / 48.0 img/s, then
OOM). Section 9c now sets only what genuinely varies per machine: the loader
worker count, and a smaller batch if the card is *not* an 80 GB Hopper. Run
`--show-tuning` to see the exact command it would launch.

**The OOM retry keeps the effective batch.** v3's halved `--batch_size` and left
`--grad_accum` alone, so every retry silently changed the effective batch the LR
schedule was tuned for. v4's halves the micro-batch and doubles the
accumulation: 24×1 → 12×2 → 6×4 → 3×8, all effective 24.

**`--stage-local` is on by default.** `/teamspace` is network-backed and
training does ~12 000 random 512 px crops an epoch out of ~27 GB of shards.
`Logs/v3/gpu_disk_guard.log` prints `STARVED` for exactly the attempts that read
across it; the run that finished read from `/tmp`. `--no-stage-local` opts out,
`DW_STAGE_SLACK_MB` tunes the free-space requirement.

**The fetch stage is a bulk snapshot.** `prepare_data.py` downloads GAMUS one
file per tile — three files a tile, ~13 200 requests for a 4000+400 pack. That
earns a CAS 429 partway through, and a failed tile is swallowed as `skip: ...`:
v3 wrote a `gamus/val` store holding **129 of 400 tiles** and still stamped it
READY. `dw_fetch.py` asks for each repo subtree once, and the pack then runs with
`HF_HUB_OFFLINE=1` where a missing file is a hard error. Section 13 gates the
READY stamp on the tile yield regardless, so a partial pack cannot reach the GPU.

**The watchdog's advice is inverted.** v3's printed *"HEADROOM: raise the batch"*
whenever VRAM was under 40 % — which is the advice that produced the batch-32
OOM. v4 expects ~58 GB at 99 % utilisation, so it warns about the two things
that actually go wrong: a starved loader, and VRAM drifting upward across the run.

## Reusing v3's shards

v4's `ShardWriter`/`PackedStore` and its GAMUS, SynRS3D and GeoNRW packers are
unchanged from v3, so a `DepthWizard-data` written by v3's `prepare_data.sh` is
directly usable. `prepare_data.sh` detects it, says so, and only adds the stretch
bounds and the READY stamp rather than repacking 35 GB. `--reprepare` overrides.

## Flags worth knowing

```bash
sh prepare_data.sh --plan                 # repos + GiB, download nothing
sh prepare_data.sh --datasets all         # + geonrw (32 GB tar, slowest source)
sh prepare_data.sh --india-dir ~/tiles    # + Indian imagery -> mean-teacher branch
sh prepare_data.sh --reprepare            # re-fetch and repack from scratch

sh train_h100.sh --check                  # deps + the offline test suite + GPU report
sh train_h100.sh --show-tuning            # resolve every flag, launch nothing
sh train_h100.sh --smoke                  # ~10 min end-to-end
sh train_h100.sh --force                  # retrain over a completed run
sh train_h100.sh --epochs 56              # anything unrecognised goes to train.py
DW_BATCH=32 sh train_h100.sh              # override the batch (v3 OOMed at 32)
DW_COMPILE=true sh train_h100.sh          # torch.compile; smoke it first
```

Both scripts are resumable. `prepare_data.sh` keeps `_dl/` on failure so a retry
is a repack, not a re-download; `train_h100.sh` writes `last.pt` every epoch and
warm-starts from it.
