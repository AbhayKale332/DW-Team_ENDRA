#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# DepthWizard v5 — the FINAL fine-tune, on one Lightning.ai H100.
#
# Warm start from resume-v4-1.6's best.pt (Kaggle run v5_probe_v4init_mvs3dm).
# No DFC23 / india_labeled (they label trees 0 m); NEON forest + scrub LiDAR
# when `depthwizard-neon` exists on Kaggle; forest/sparse-weighted sampling and
# checkpoint selection; batch / compile / workers measured on this card.
# Flags and their reasons: lightning/final_flags.py.
#
#   export HF_TOKEN=hf_...                  # DINOv3-SAT is gated
#   export PREV_KERNEL=abhaydkale232/<the notebook that ran v5_probe_v4init_mvs3dm>
#   bash lightning/final_h100.sh all --background      # check fetch sweep smoke train test package
#   tail -f ~/dw_final/final.log
#
# Steps (each also runs alone, in this order):
#   check     GPU, credentials, disk, deps, pytest
#   fetch     Kaggle stores -> /tmp/kin (per file, resumable) -> link -> /tmp/dwdata;
#             best.pt -> /tmp/prev (only best.pt + metrics.json, not last_full.pt)
#   sweep     measure batch x torch.compile on this card -> $WORK/sizing.json
#   smoke     60 micro-steps with the chosen sizing, end to end
#   train     the run (resumes from last_full.pt if one is there)
#   test      held-out neon/mvs3dm/gamus test: this run vs the checkpoint it started from
#   package   one zip of what matters + a GPU-utilisation summary
#
# Env knobs: WORK (default /teamspace/studios/this_studio/dw_final or ~/dw_final),
# DATA_ROOT (/tmp/dwdata), KIN (/tmp/kin), PREV_DIR (/tmp/prev), PREV_PATH (use
# this best.pt instead of fetching), RUN (v5_final_forest), PY (python),
# SWEEP_BATCHES (16,24,32,40,48), CEILING_GB (72), SKIP_TESTS=1.
# /tmp is local NVMe and ephemeral: after a Studio restart re-run `fetch`
# (it skips what is still there) and then `train` resumes from last_full.pt.
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V5="$(cd "$HERE/.." && pwd)"
if [ -d /teamspace/studios/this_studio ]; then
  WORK="${WORK:-/teamspace/studios/this_studio/dw_final}"
else
  WORK="${WORK:-$HOME/dw_final}"
fi
DATA_ROOT="${DATA_ROOT:-/tmp/dwdata}"
KIN="${KIN:-/tmp/kin}"
PREV_DIR="${PREV_DIR:-/tmp/prev}"
PREV_RUN="${PREV_RUN:-v5_probe_v4init_mvs3dm}"
RUN="${RUN:-v5_final_forest}"
OUT="$WORK/outputs/$RUN"
SIZING="$WORK/sizing.json"
PY="${PY:-python}"
CEILING_GB="${CEILING_GB:-72}"
SWEEP_BATCHES="${SWEEP_BATCHES:-16,24,32,40,48}"
OWNER="abhaydkale232"
# <slug>:<splits>.  neon is optional: the run trains without it until it exists.
DATASETS=(
  "$OWNER/depthwizard-mvs3dm:train,val,test"
  "$OWNER/depthwizard-gamus:train,val,test"
  "$OWNER/depthwizard-us3d:train,val"
  "$OWNER/depthwizard-synrs3d-g05:train"
  "$OWNER/depthwizard-synrs3d-g1:train"
  "$OWNER/depthwizard-neon:train,val,test"
)
OPTIONAL="depthwizard-neon"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=1
export DW_DATA_ROOT="$DATA_ROOT" DW_KAGGLE_INPUT="$KIN" DW_OUTPUT_DIR="$OUT"
mkdir -p "$WORK"

log()  { echo "[final $(date +%H:%M:%S)] $*"; }
die()  { echo "[final] ERROR: $*" >&2; exit 1; }

prev_ckpt() { if [ -n "${PREV_PATH:-}" ]; then echo "$PREV_PATH"; else echo "$PREV_DIR/best.pt"; fi; }

load_flags() {   # $1 = output dir, $2 = resume path -> global array A
  # Through a file, not `< <(...)`: a failing process substitution is invisible
  # to `set -e`, and train.py would then run on its config.py defaults.
  local f; f="$(mktemp)"
  "$PY" "$HERE/final_flags.py" --data_root "$DATA_ROOT" --sizing "$SIZING" \
        --output_dir "$1" --resume "$2" > "$f" || die "final_flags.py failed"
  mapfile -t A < "$f"; rm -f "$f"
  [ "${#A[@]}" -gt 10 ] || die "final_flags.py produced no flags"
}

# ---------------------------------------------------------------------------
do_check() {
  log "GPU:"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
    || die "no nvidia-smi — is this the H100 Studio?"
  nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "H100" \
    || log "!! not an H100 — the sweep will still size the run to this card"
  [ -n "${HF_TOKEN:-}" ] || die "export HF_TOKEN (DINOv3-SAT is gated)"
  [ -f "$HOME/.kaggle/kaggle.json" ] || [ -n "${KAGGLE_KEY:-}" ] \
    || die "no ~/.kaggle/kaggle.json (chmod 600) or KAGGLE_USERNAME/KAGGLE_KEY"
  log "installing deps"
  "$PY" -m pip install -q -r "$V5/requirements.txt" kaggle
  "$PY" - <<'EOF'
import os, torch
print(f"[final] torch {torch.__version__} cuda {torch.version.cuda} "
      f"bf16={torch.cuda.is_available() and torch.cuda.is_bf16_supported()} "
      f"cores={len(os.sched_getaffinity(0))}")
EOF
  log "disk:"; df -h /tmp "$WORK" | sed 's/^/[final]   /'
  "$PY" "$V5/tools/kaggle_fetch.py" datasets --dest "$KIN" --dry_run \
      --optional "$OPTIONAL" $(printf -- '--dataset %s ' "${DATASETS[@]}")
  if [ "${SKIP_TESTS:-0}" != "1" ]; then
    log "pytest"; (cd "$V5" && "$PY" -m pytest -q -x)
  fi
}

do_fetch() {
  log "datasets -> $KIN"
  "$PY" "$V5/tools/kaggle_fetch.py" datasets --dest "$KIN" \
      --optional "$OPTIONAL" $(printf -- '--dataset %s ' "${DATASETS[@]}")
  (cd "$V5" && bash run_kaggle.sh link)
  ls -la "$DATA_ROOT" | sed 's/^/[final]   /'
  if [ -f "$DATA_ROOT/neon/train/index.json" ]; then
    # per-site height cap (cliff / tower / wire spikes, tools/eda_neon.py).
    # Caps land in $DATA_ROOT/neon/build_info.json, so a re-fetch reuses them;
    # on a Kaggle version already capped it drops ~nothing.
    log "neon: per-site height cap"
    (cd "$V5" && "$PY" tools/pack_neon.py clean --data_root "$DATA_ROOT" --work "$DATA_ROOT/neon")
  fi
  "$PY" - "$DATA_ROOT" "$V5" <<'EOF'
import sys
sys.path.insert(0, sys.argv[2])
from prepare_data import stale_gsd_stores
bad = stale_gsd_stores(sys.argv[1])
if bad:
    raise SystemExit("[final] stale GSD in: " + "; ".join(bad) + " — a newer Kaggle version is needed")
print("[final] store GSDs match the measured values")
EOF
  for s in dfc23 dfc23_g050 india_labeled; do
    [ ! -e "$DATA_ROOT/$s" ] || die "$DATA_ROOT/$s exists — the final run must not see it; remove the link"
  done
  if [ -n "${PREV_PATH:-}" ]; then
    log "using PREV_PATH=$PREV_PATH"
  else
    [ -n "${PREV_KERNEL:-}" ] || die "export PREV_KERNEL=$OWNER/<notebook slug> (or PREV_PATH=/path/best.pt)"
    "$PY" "$V5/tools/kaggle_fetch.py" checkpoint --kernel "$PREV_KERNEL" \
        --run "$PREV_RUN" --dest "$PREV_DIR"
  fi
  [ -f "$(prev_ckpt)" ] || die "no warm-start checkpoint at $(prev_ckpt)"
}

do_sweep() {
  [ -f "$(prev_ckpt)" ] || die "run fetch first"
  (cd "$V5" && "$PY" tools/h100_sweep.py --data_root "$DATA_ROOT" --resume "$(prev_ckpt)" \
      --out "$SIZING" --work /tmp/dw_sweep --batches "$SWEEP_BATCHES" \
      --ceiling_gb "$CEILING_GB" ${FORCE_SWEEP:+--force})
}

do_smoke() {
  [ -f "$SIZING" ] || die "run sweep first"
  local sm="/tmp/dw_smoke"
  rm -rf "$sm"
  load_flags "$sm" "$(prev_ckpt)"
  local bs; bs=$("$PY" -c "import json;print(json.load(open('$SIZING'))['batch_size'])")
  (cd "$V5" && "$PY" train.py "${A[@]}" --epochs 1 --crops_per_epoch $((bs * 60)) \
      --max_minutes 15 --test_sources "" --make_figures false --make_report false \
      --export_onnx false --save_full_state false --val_tiles 32 --final_sliding_eval false \
      --tta false) 2>&1 | tee "$WORK/smoke.log"
  grep -q "DONE" "$WORK/smoke.log" || die "smoke run did not finish — see $WORK/smoke.log"
  grep -E "\[data\]|img/s" "$WORK/smoke.log" | tail -n 25 | sed 's/^/[final]   /'
}

train_once() {   # $1 = resume path
  load_flags "$OUT" "$1"
  mkdir -p "$OUT"
  (cd "$V5" && "$PY" train.py "${A[@]}") 2>&1 | tee -a "$OUT/train_console.log"
  return "${PIPESTATUS[0]}"
}

do_train() {
  [ -f "$SIZING" ] || die "run sweep first"
  mkdir -p "$OUT"
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw \
      --format=csv -l 15 >> "$OUT/gpu_util.csv" 2>/dev/null &
  local smi=$!
  trap 'kill $smi 2>/dev/null || true' RETURN
  local resume
  if [ -f "$OUT/last_full.pt" ]; then resume="$OUT/last_full.pt"; log "resuming $resume"
  else resume="$(prev_ckpt)"; log "warm start from $resume"; fi
  if train_once "$resume"; then return 0; fi
  # OOM fallback: the next-smaller batch the sweep measured, from last_full.pt.
  if grep -q "OutOfMemory\|out of memory" "$OUT/train_console.log"; then
    log "OOM — falling back to the next-smaller measured config"
    "$PY" - "$SIZING" <<'EOF' || die "no smaller measured config left"
import json, sys
p = sys.argv[1]; s = json.load(open(p))
fb = s.get("fallbacks") or []
if not fb:
    raise SystemExit(1)
nxt = fb.pop(0)
s.update(nxt); s["fallbacks"] = fb; s["eval_batch_mult"] = max(1, int(s.get("eval_batch_mult", 2)) - 1)
json.dump(s, open(p, "w"), indent=2)
print(f"[final] now batch {nxt['batch_size']} x accum {nxt['grad_accum']} compile={nxt['compile_model']}")
EOF
    local r2="$(prev_ckpt)"; [ -f "$OUT/last_full.pt" ] && r2="$OUT/last_full.pt"
    train_once "$r2"
  else
    die "train.py failed (not an OOM) — see $OUT/train_console.log"
  fi
}

do_test() {
  local ck="$OUT/best.pt"; [ -f "$ck" ] || ck="$OUT/last.pt"
  [ -f "$ck" ] || die "no checkpoint in $OUT"
  local start; start="$(prev_ckpt)"
  local bs; bs=$("$PY" -c "import json;print(json.load(open('$SIZING'))['batch_size'])" 2>/dev/null || echo 16)
  for src in neon mvs3dm gamus; do
    [ -f "$DATA_ROOT/$src/test/index.json" ] || { log "no $src/test, skipped"; continue; }
    local common=(--data_root "$DATA_ROOT" --source "$src" --split test --tiles 0
                  --sliding_tiles 200 --qualitative 16 --max_valid_height_m 150
                  --batch_size "$bs" --eval_batch_mult 2 --num_workers 8 --amp_dtype bf16)
    (cd "$V5" && "$PY" eval_test.py --ckpt "$ck" "${common[@]}" --out "$OUT/test_final_$src")
    if [ -f "$start" ]; then
      (cd "$V5" && "$PY" eval_test.py --ckpt "$start" "${common[@]}" --out "$OUT/test_start_$src")
      (cd "$V5" && "$PY" eval_test.py --compare "$OUT/test_start_$src,$OUT/test_final_$src" \
          --labels resume-v4-1.6,final --md "$OUT/start_vs_final_$src.md")
    fi
  done
}

do_package() {
  local z="$WORK/results_$RUN.zip"
  [ -f "$SIZING" ] && cp "$SIZING" "$OUT/sizing.json"
  "$PY" - "$OUT" <<'EOF' | tee "$OUT/gpu_util_summary.txt"
import csv, sys, statistics as st
from pathlib import Path
p = Path(sys.argv[1]) / "gpu_util.csv"
if not p.is_file():
    raise SystemExit("[final] no gpu_util.csv")
rows = [r for r in csv.reader(open(p)) if r and not r[0].startswith("timestamp")]
u = [float(r[1].split()[0]) for r in rows]; m = [float(r[2].split()[0]) / 1024 for r in rows]
u_s = sorted(u)
print(f"[final] GPU util over {len(u)} samples: mean {st.mean(u):.1f} %, "
      f"p10 {u_s[len(u_s) // 10]:.0f} %, memory used max {max(m):.1f} GB")
EOF
  (cd "$OUT" && zip -q -r "$z" best.pt depthwizard.onnx* metrics.json run.log validation_report.html \
      gpu_util.csv gpu_util_summary.txt sizing.json start_vs_final_*.md test_final_* test_start_* \
      qualitative figures 2>/dev/null || true)
  log "-> $z ($(du -h "$z" | cut -f1))"
}

# ---------------------------------------------------------------------------
STEPS=()
BG=0
for a in "$@"; do
  case "$a" in
    --background) BG=1 ;;
    all) STEPS+=(check fetch sweep smoke train test package) ;;
    check|fetch|sweep|smoke|train|test|package) STEPS+=("$a") ;;
    *) die "unknown argument $a (steps: check fetch sweep smoke train test package all; --background)" ;;
  esac
done
[ "${#STEPS[@]}" -gt 0 ] || { sed -n '2,32p' "$0"; exit 1; }
if [ "$BG" = 1 ]; then
  nohup bash "$0" "${STEPS[@]}" >> "$WORK/final.log" 2>&1 &
  echo "[final] running in background (pid $!): tail -f $WORK/final.log"
  exit 0
fi
for s in "${STEPS[@]}"; do
  log "===== $s ====="
  "do_$s"
done
log "done: ${STEPS[*]}"
