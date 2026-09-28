#!/bin/bash
# DepthWizard v4 — Kaggle (GPU T4 x2, 2 x 16 GB Turing) runbook.
#
#   bash run_kaggle.sh check              # GPUs, NCCL, disk, /dev/shm, tests
#   bash run_kaggle.sh prepare            # the ~14 GB pack, in a CPU notebook
#   bash run_kaggle.sh link               # symlink farm over /kaggle/input/*
#   bash run_kaggle.sh smoke              # the first real 2-process DDP run
#   bash run_kaggle.sh train              # the real run (torchrun, 2 processes)
#   bash run_kaggle.sh train --resume /kaggle/input/<prev>/last_full.pt
#   bash run_kaggle.sh finalize           # post-training eval/figures/report/ONNX
#   bash run_kaggle.sh predict scene.tif --ckpt outputs/v4/best.pt --absolute
#   bash run_kaggle.sh onnx | serve | report
#
# Prerequisites:
#   HF_TOKEN as a Kaggle Secret (DINOv3-SAT and GAMUS are both gated).
#   train.py reads it via UserSecretsClient automatically; export it here too if
#   you are running prepare_data.py directly.
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

OUT="${DW_OUTPUT_DIR:-/kaggle/working/outputs/v4}"
# `/kaggle/input` is a read-only network mount and random 512 px crops out of it
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
    # cfg.data_root is a single root, but a pack prepared in two notebook runs
    # arrives as two /kaggle/input paths.  Symlinks cost nothing and keep the
    # layout byte-identical to a local v4 tree: <root>/<source>/<split>/.
    mkdir -p "$DATA"
    n=0
    for d in "$KIN"/*/dwdata/* "$KIN"/*/*; do
      [ -d "$d" ] || continue
      name="$(basename "$d")"
      # Sources packed one store per GSD family arrive suffixed
      # (synrs3d_g05, dfc23_g050), so those patterns are globs.  A store whose
      # name is not matched here is silently skipped and train.py then reports
      # it as "not prepared" — check this list first when a source goes missing.
      case "$name" in
        gamus|geonrw|synrs3d|synrs3d_*|dfc23|dfc23_*|india_labeled|india_unlabeled) ;;
        *) continue ;;
      esac
      [ -n "$(find "$d" -name index.json -print -quit 2>/dev/null)" ] || continue
      rm -rf "${DATA:?}/$name"
      ln -s "$d" "$DATA/$name"
      echo "  $DATA/$name -> $d"
      n=$((n + 1))
    done
    if [ "$n" -eq 0 ]; then
      echo "no packed store found under $KIN — attach the prepared dataset first"; exit 1
    fi
    find "$DATA"/ -name index.json | sed 's/^/  /'
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
    # ---- what the 2025-09-15 run measured, and what changed because of it ----
    # That run trained for 452 min and got 3.947 m.  Its log shows why that is
    # not the number this profile is capable of:
    #
    #   e6:  281/500 optimiser steps skipped on non-finite gradients
    #   e7 .. e11:  500/500 skipped.   e12: 330/330 skipped.
    #
    # fp16 gradients overflowed in epoch 6, GradScaler backed the loss scale off
    # 281 times, and at backoff #166 the scale underflowed fp32 to *exactly*
    # zero — after which `unscale_` divides every gradient by zero and no step
    # can ever be finite again.  Epochs 7-12 ran ~240 min of forward and
    # backward passes and applied ZERO updates; the eval lines for e6, e8, e10
    # and e12 are byte-identical because the weights genuinely never moved, and
    # best.pt is the epoch-6 checkpoint.  Half the wall-clock budget bought
    # nothing.  Fixed in three places:
    #
    #   models/losses.py   every loss term now computes in fp32, which removes
    #                      the overflow at source (normal_loss divides height
    #                      differences by a 0.33 m GSD, so a tall edge reaches
    #                      ~600 and the squared norm after it 3.6e5, against an
    #                      fp16 ceiling of 65504 — hence `nrm=nan` in the log).
    #   train.py           the loss scale has a floor (`--amp_min_scale`), so a
    #                      run of skips is recoverable rather than terminal, and
    #                      a dead epoch is reported as "500/500 (100 %)" and
    #                      aborts the run after two rather than being logged as
    #                      a bare count six times.
    #   config.py          init_scale 2**13 not 2**16 (the old default skipped
    #                      the first four boundaries of epoch 1 just backing
    #                      down) and growth_interval 500 not 2000, which was
    #                      longer than an entire epoch on this profile.
    #
    # --encoder_unfreeze_blocks 16 and --llrd 0.90 are the accuracy half.  The
    # old run unfroze all 24 blocks at --llrd 0.80, which puts the bottom block
    # at 0.8**24 = 0.5 % of --encoder_lr (the log prints the range as
    # "2.83e-07..3.00e-04"): the lower half of a 303 M-parameter ViT-L paid full
    # price in gradients, all-reduce traffic and AdamW state to move
    # essentially not at all, and dropped throughput from 4.5 to 2.1 img/s.
    # Freezing the bottom 8 and decaying at 0.90 gives every trainable block a
    # usable LR, frees ~1.2 GB, and buys back enough step time for ~15 epochs in
    # the same 480 min.  Use `--encoder_unfreeze_blocks 0` for the old
    # whole-encoder behaviour.
    #
    # --epochs 24 / --max_minutes 960 / --session_minutes 480 is a TWO-session
    # profile, and all three numbers have to be passed identically in both
    # sessions.  `progress = max(epoch fraction, elapsed/max_minutes)`, so
    # `max_minutes` is the budget the cosine is SIZED for, not a safety cap:
    # 24 epochs at the measured rate (~22 min frozen, ~40 min after the 16-block
    # unfreeze) is ~960 min, so the epoch term and the clock term land together
    # instead of one of them being dead the whole run.
    #
    # The v4-2 run is the cautionary tale.  It went out as --max_minutes 600
    # --epochs 16 --session_minutes 480 --save_full_state false: the session cap
    # fired at 81 % of the cosine with the LR still at 5.09e-05, and with no
    # last_full.pt the remaining anneal could not be resumed.  It scored 3.804 m
    # against v4's 3.441 m on the same 400-tile val prefix.  train.py now prints
    # a [sched] line at startup saying exactly where the cosine will stop, and
    # config.validate() warns on this combination.
    #
    # Session 2 adds, and changes nothing else:
    #   --resume /kaggle/input/<session-1>/outputs/v4/last_full.pt
    #
    # --eval_every 1 because 12 epochs at eval_every 2 gave best.pt only six
    # selection points, and the run's own numbers show why that is too coarse:
    # d1 went 0.493 -> 0.533 -> 0.482 across e2/e4/e6 while RMSE fell, so the
    # chosen epoch is decided by which epochs happened to be sampled.  An eval
    # is ~1.5 min here against ~32 min of training.
    #
    # 2 ranks x micro-batch 2 x grad_accum 6 = global effective batch 24, exact
    # parity with the H100 profile.  Read `vram=NG` off the log line train.py
    # prints every 25 steps and move to --batch_size 4 --grad_accum 3 if there
    # is headroom (the partial unfreeze above should leave some; the old run sat
    # at 14 of 15 GiB with all 24 blocks trainable).  Static VRAM before
    # activations is ~8.3 GB of ~15 usable (params 1.28 + grads 1.28 + AdamW
    # 2.57 + EMA 1.28 + teacher 1.28 + casts), which is why
    # grad_checkpoint_encoder — off by default on an H100 — has to be on here.
    # Turing has no bf16 tensor cores, so fp16 + GradScaler.
    #
    # --make_zip false is not optional: package_results.build_zip is
    # shutil.make_archive with no exclusions and no source cleanup, so it
    # duplicates ~4 GB of .pt/.onnx that barely deflate and pushes the run past
    # the 19 GB output quota.  Kaggle versions /kaggle/working anyway.
    torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" train.py \
      --data_root "$DATA" --output_dir "$OUT" \
      --datasets gamus,synrs3d_g05,synrs3d_g1 \
      --amp_dtype fp16 --grad_checkpoint_encoder true \
      --batch_size 2 --grad_accum 6 --eval_batch_mult 8 \
      --num_workers 2 --prefetch_factor 2 --compile_model false \
      --encoder_unfreeze_blocks 16 --llrd 0.90 \
      --epochs 24 --eval_every 1 --max_minutes 960 --session_minutes 480 \
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
      --batch_size 2 --eval_batch_mult 8 --num_workers 2 \
      --epochs 0 --freeze_epochs 0 --make_zip false "$@"
    ;;

  predict)
    $PY -m infer.predict "$@"
    ;;

  onnx)
    $PY -m infer.export_onnx --ckpt "${OUT}/best.pt" \
      --out "${OUT}/depthwizard.onnx" "$@"
    ;;

  serve)
    $PY -m serve.app --ckpt "${OUT}/best.pt" "$@"
    ;;

  report)
    $PY -m viz.figures "${OUT}"
    $PY -m viz.report_html "${OUT}"
    ;;

  *)
    echo "usage: bash run_kaggle.sh {check|prepare|link|smoke|train|finalize|predict|onnx|serve|report} [flags]"
    exit 1;;
esac
