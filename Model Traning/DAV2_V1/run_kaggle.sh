#!/bin/bash
# DepthWizard DAV2/V1 — Kaggle (GPU T4 x2, 2 x 16 GB Turing) runbook.
#
#   bash run_kaggle.sh check              # GPUs, NCCL, disk, /dev/shm, tests
#   bash run_kaggle.sh prepare            # the ~14 GB pack, in a CPU notebook
#   bash run_kaggle.sh link               # symlink farm over /kaggle/input/*
#   bash run_kaggle.sh smoke              # the first real 2-process DDP run
#   bash run_kaggle.sh train              # the real run (torchrun, 2 processes)
#   bash run_kaggle.sh train --resume /kaggle/input/<prev>/last_full.pt
#   bash run_kaggle.sh phase2             # e25-e40 + india_labeled, from run 1
#   bash run_kaggle.sh finalize           # post-training eval/figures/report/ONNX
#   bash run_kaggle.sh onnx | report
#
# Prerequisites:
#   HF_TOKEN as a Kaggle Secret.  **GAMUS still needs it; the encoder no longer
#   does** — `depth-anything/Depth-Anything-V2-Base-hf` is ungated, unlike
#   DINOv3-SAT.  train.py reads the secret via UserSecretsClient automatically;
#   export it here too if you are running prepare_data.py directly.
#
# LICENCE: DAv2-Base is CC-BY-NC-4.0 (non-commercial).  Only DAv2-Small is
# Apache-2.0.  If the submission needs a commercially usable model, the fallback
# is `--encoder_model_id depth-anything/Depth-Anything-V2-Small-hf` and nothing
# else changes.
#
# Start the real run through **Save Version -> Save & Run All (Commit)**.  The
# headless commit is what gets the full ~12 h and persists /kaggle/working as the
# notebook's output; an interactive session idles out long before the run ends.
#
# Do NOT put credentials in this file.

# `sh` is dash on Debian/Ubuntu images and dash has no `pipefail`; set it only if
# the shell actually has it (same preamble as v4's run_lightning.sh, and for the
# same reason — `set -o pipefail` on line 1 aborted that script under dash).
set -eu
# shellcheck disable=SC3040
(set -o pipefail) 2>/dev/null && set -o pipefail || true
cd "$(dirname "$0")"

CMD="${1:-train}"
if [ $# -gt 0 ]; then shift; fi   # `shift || true` is fatal in dash

OUT="${DW_OUTPUT_DIR:-/kaggle/working/outputs/dav2-v1}"
# `/kaggle/input` is a read-only network mount and random 518 px crops out of it
# are small random reads — exactly the access pattern v3's guard log called
# "STARVED".  `link` builds a symlink farm under $DATA so cfg.data_root stays a
# single root even when the pack arrived as two separate attached datasets.
DATA="${DW_DATA_ROOT:-/kaggle/temp/dwdata}"
KIN="${DW_KAGGLE_INPUT:-/kaggle/input}"
NPROC="${DW_NPROC:-2}"

if command -v python >/dev/null 2>&1; then PY=python; else PY=python3; fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "note: HF_TOKEN not exported — train.py will try the Kaggle Secret."
fi

# Common to every launch that touches CUDA.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Kaggle's two T4s have no NVLink and P2P over the PCIe topology is frequently
# advertised but broken; SHM/socket transport is both correct and no slower here.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
# ~4 vCPU total: 2 train workers per rank + the capped val pool is the budget.
export TOKENIZERS_PARALLELISM=false

case "$CMD" in
  check)
    $PY -m pip install -q -r requirements.txt || true
    $PY - <<'PY'
import shutil, torch, torch.distributed as dist
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU{i}: {p.name} {p.total_memory/1024**3:.1f} GiB sm_{p.major}{p.minor}")
    if p.major < 8:
        print("       -> Turing: no bf16 tensor cores. Use --amp_dtype fp16.")
print("nccl available:", dist.is_nccl_available())
for d in ("/kaggle/working", "/kaggle/temp", "/dev/shm"):
    try:
        t, u, f = shutil.disk_usage(d)
        print(f"  {d:16s} {f/1024**3:6.1f} GiB free of {t/1024**3:.1f}")
    except OSError:
        print(f"  {d:16s} (absent)")
PY
    echo "nproc: $(nproc)"
    $PY -m pytest -q
    ;;

  prepare)
    # ~14 GB, which is the number that matters twice: it fits inside the 19 GB
    # /kaggle/working quota *during* packing, and it fits in Kaggle's ~29 GB RAM
    # as page cache afterwards, so training off the read-only input mount is not
    # I/O bound.  GAMUS at tile_px=1024 is 7.34 MB/tile and dominates; SynRS3D is
    # already 512 px at 1.84 MB/tile.
    #   1200 x 7.34 MB + 160 x 7.34 MB + ~4 GB  ~=  14 GB
    # Fewer unique tiles costs less than it looks: the sampler oversamples with
    # replacement=True anyway.
    PREP_ROOT="${DW_PREPARE_ROOT:-/kaggle/working/dwdata}"
    $PY prepare_data.py --data_root "$PREP_ROOT" --datasets gamus,synrs3d \
      --gamus_train 1200 --gamus_val 160 --synrs3d_archives 4 "$@"
    # Pre-compute the 2/98 stretch bounds here, once, so no training worker ever
    # pays a full-tile histogram — and so rank 1 never has to (it is gated off
    # that write; see dwdata/loaders.py `_prime`).
    DW_PREP_ROOT="$PREP_ROOT" $PY - <<'PY'
import os
from pathlib import Path
from dwdata.packed import PackedStore
root = Path(os.environ["DW_PREP_ROOT"])
for idx in sorted(root.glob("*/*/index.json")):
    st = PackedStore(idx.parent)
    st.prime_stretch_bounds(2.0, 98.0, workers=4)
    print(f"  bounds primed: {idx.parent} ({len(st)} tiles)")
PY
    du -sh "$PREP_ROOT"
    echo "now: Save Version -> the committed output becomes a dataset to attach."
    ;;

  link)
    # cfg.data_root is a single root laid out <root>/<source>/<split>/index.json,
    # but the packed stores arrive as separate Kaggle datasets mounted as
    #   $KIN/datasets/abhaydkale232/depthwizard-gamus/train/train/index.json
    #   $KIN/datasets/abhaydkale232/depthwizard-synrs3d-g05/train/index.json
    # (Kaggle zips a folder and nests it once more, hence train/train).  So we
    # find every index.json, take <split> from the directory holding it and
    # <source> from the path component that names the dataset
    # (depthwizard-synrs3d-g05 -> synrs3d_g05), and symlink per split.  Also
    # still accepts the old <slug>/dwdata/<source>/<split>/ output layout.
    # Symlinks cost nothing; the stores already ship stretch_bounds_2_98.npy, so
    # nothing ever needs to write into the read-only mount.
    mkdir -p "$DATA"
    n=0
    LIST="$DATA/.stores"
    find -L "$KIN" -maxdepth 8 -name index.json 2>/dev/null | sort > "$LIST"
    while IFS= read -r idx; do
      store="$(dirname "$idx")"
      split="$(basename "$store")"
      case "$split" in train|val|test) ;; *) continue ;; esac
      name=""
      for comp in $(echo "${store#"$KIN"/}" | tr '/' ' '); do
        c="$(echo "${comp#depthwizard-}" | tr '-' '_')"
        # Sources packed one store per GSD family arrive suffixed (synrs3d_g05,
        # dfc23_g050), so those patterns are globs.  A store whose name is not
        # matched here is silently skipped and train.py then reports it as
        # "not prepared" — check this list first when a source goes missing.
        case "$c" in
          gamus|geonrw|synrs3d|synrs3d_g*|dfc23|dfc23_g*|india_labeled|india_unlabeled) name="$c" ;;
        esac
      done
      [ -n "$name" ] || continue
      mkdir -p "$DATA/$name"
      rm -rf "${DATA:?}/$name/$split"
      ln -s "$store" "$DATA/$name/$split"
      echo "  $DATA/$name/$split -> $store"
      n=$((n + 1))
    done < "$LIST"
    rm -f "$LIST"
    if [ "$n" -eq 0 ]; then
      echo "no packed store found under $KIN — attach the prepared dataset first"; exit 1
    fi
    find -L "$DATA"/ -name index.json | sed 's/^/  /'
    ;;

  smoke)
    # The first real DDP test, and it is cheap: apply_smoke() gives 24 crops,
    # 2 epochs, num_workers=0.  It does NOT turn make_zip off, hence the flag.
    # Success = a clean exit, ONE set of loss lines in run.log (proving the
    # rank-1 stdout redirect worked), best.pt/last.pt/metrics.json written once.
    torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" train.py \
      --smoke --datasets gamus --output_dir "$OUT" --make_zip false "$@"
    ;;

  train)
    # ---- what this profile is buying, and what it is spending ---------------
    # The encoder drops from ViT-L/16's 303 M parameters to ViT-B/14's 86 M, so
    # static VRAM per card falls from V4_Kaggle's measured ~8.3 GB to roughly
    # ~2.6 GB (86 M x 4 copies for live/grads/AdamW-x2, plus the EMA shadow and
    # the mean-teacher deepcopy).  That is the entire point of the swap on this
    # card, and it is spent in exactly two places:
    #
    #   --grad_checkpoint_encoder false   V4_Kaggle had to force this ON to fit
    #                                     ViT-L at all, at ~35 % of the step.
    #   --batch_size 6                    against V4_Kaggle's 2.
    #
    # 2 ranks x 6 x grad_accum 2 = global 24, exact parity with v3 and v4 — so
    # the LR schedule stays comparable and the only variable is the backbone.
    # What changes is the number of *optimiser steps* per session, which is
    # roughly 2-3x, and step count is what the v4 Kaggle run was starved of.
    #
    # Read the `vram=NG` field train.py prints every 25 steps:
    #   * headroom (>= 3 GB spare for the eval spike) -> --batch_size 8
    #     --grad_accum 2 (global 32).  Raise the batch, NOT by cutting
    #     grad_accum to 1: more steps is the goal, but global-batch parity with
    #     v3/v4 is what keeps the LR schedule honest.
    #   * tight -> back off in V4_Kaggle's documented order:
    #     --consistency_every 4, then --ema_decay 0, then --w_consistency 0.
    #
    # Turing has no bf16 tensor cores, so fp16 + GradScaler, and config.py's
    # amp_init_scale / amp_growth_interval / amp_min_scale are the three fixes
    # for the run that skipped 500/500 optimiser steps for five straight epochs
    # after the scale underflowed to zero.  See the V4_Kaggle README for that
    # autopsy in full; nothing about it changed here.
    #
    # --epochs 24 / --max_minutes 960 / --session_minutes 480 is a TWO-session
    # profile and all three numbers must be passed identically in both sessions.
    # `progress = max(epoch fraction, elapsed/max_minutes)`, so `max_minutes` is
    # the budget the cosine is SIZED for, not a safety cap.  A session that died
    # MID-schedule is continued by adding, and changing nothing else:
    #   --resume /kaggle/input/<session-1>/outputs/dav2-v1/last_full.pt
    # Once the schedule has FINISHED (run 1 did: all 24 epochs, lr at the 3e-6
    # floor), a plain --resume trains nothing, and raising --epochs alone would
    # spike the LR — continuing needs a new --phase.  That is `phase2` below.
    #
    # --make_zip false is not optional: package_results.build_zip is
    # shutil.make_archive with no exclusions, so it duplicates GB of .pt/.onnx
    # that barely deflate and pushes the run past the 19 GB output quota.
    torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" train.py \
      --data_root "$DATA" --output_dir "$OUT" \
      --datasets gamus,synrs3d_g05,synrs3d_g1 \
      --amp_dtype fp16 --grad_checkpoint_encoder false \
      --batch_size 6 --grad_accum 2 --eval_batch_mult 4 \
      --num_workers 2 --prefetch_factor 2 --compile_model false \
      --consistency_every 2 --w_consistency 1.0 \
      --epochs 24 --eval_every 1 --max_minutes 960 --session_minutes 480 \
      --save_full_state true --make_zip false "$@"
    ;;

  phase2)
    # Phase 2: 16 more epochs (e25-e40) from run 1's last_full.pt, with its own
    # warmup + cosine at 0.3x run 1's peak LR, and india_labeled added to the
    # mix (sampler weight 2 via config.py's default --sampler_weights).  gamus
    # stays FIRST, so best.pt is still selected on the GAMUS val prefix and the
    # number stays comparable with v1-v4; india_labeled's val split is scored
    # and reported every epoch by build_aux_val_loaders, never selected on.
    #
    # Attach run 1's notebook output as an input.  Its last_full.pt is found
    # under $KIN; DW_RESUME=<path> overrides the search.  train.py copies run
    # 1's best.pt into $OUT if phase 2 never beats it.
    #
    # Re-running this exact command after a crash, with DW_RESUME pointing at
    # the PHASE-2 last_full.pt, continues phase 2 instead of restarting it: the
    # checkpoint records the phase id, start epoch and phase clock.
    #
    # Budget: 16 x (~16.6 min train + ~1 min India eval) ~= 280 min of
    # training; + ~40 min val finals + ~85 min gamus/test (2861 tiles plain +
    # TTA, 400 sliding) + setup ~= 7 h, well inside Kaggle's 12 h.
    RESUME="${DW_RESUME:-}"
    if [ -z "$RESUME" ]; then
      CANDS="$(find -L "$KIN" -maxdepth 8 -name last_full.pt 2>/dev/null | sort)"
      NC="$(printf '%s' "$CANDS" | grep -c . || true)"
      if [ "$NC" -ne 1 ]; then
        echo "phase2: expected exactly one last_full.pt under $KIN, found $NC:"
        if [ -n "$CANDS" ]; then printf '%s\n' "$CANDS" | sed 's/^/  /'; fi
        echo "attach run 1's notebook output, or set DW_RESUME=<path>/last_full.pt"
        exit 1
      fi
      RESUME="$CANDS"
    fi
    echo "phase2: resuming from $RESUME"
    torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" train.py \
      --data_root "$DATA" --output_dir "$OUT" \
      --datasets gamus,synrs3d_g05,synrs3d_g1,india_labeled \
      --amp_dtype fp16 --grad_checkpoint_encoder false \
      --batch_size 6 --grad_accum 2 --eval_batch_mult 4 \
      --num_workers 2 --prefetch_factor 2 --compile_model false \
      --consistency_every 2 --w_consistency 1.0 \
      --resume "$RESUME" --phase 2 --phase_lr_mult 0.3 --phase_warmup_frac 0.05 \
      --epochs 40 --eval_every 1 --max_minutes 480 --session_minutes 480 \
      --test_sources gamus:test \
      --save_full_state true --make_zip false "$@"
    ;;

  finalize)
    # The post-training stage (final plain/TTA eval, sliding eval, qualitative
    # export, figures, report, ONNX) is 20-60 min on a T4 and needs one GPU and
    # no training.  Running it in its own short notebook decouples it from the
    # 12 h cap, and it is the recovery path when a session dies after training
    # but before the exports.  --epochs 0 skips the loop entirely; train.py then
    # reads `history` and `best` back out of the committed metrics.json instead
    # of rewriting it with an empty history and losing the training curves.
    $PY train.py --data_root "$DATA" --output_dir "$OUT" \
      --datasets gamus,synrs3d_g05,synrs3d_g1 --amp_dtype fp16 \
      --batch_size 6 --eval_batch_mult 4 --num_workers 2 \
      --epochs 0 --freeze_epochs 0 --make_zip false "$@"
    ;;

  onnx)
    $PY -m infer.export_onnx --ckpt "${OUT}/best.pt" \
      --out "${OUT}/depthwizard.onnx" "$@"
    ;;

  report)
    $PY -m viz.figures "${OUT}"
    $PY -m viz.report_html "${OUT}"
    ;;

  *)
    echo "usage: bash run_kaggle.sh {check|prepare|link|smoke|train|phase2|finalize|onnx|report} [flags]"
    exit 1;;
esac
