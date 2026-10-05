#!/bin/bash
# ---------------------------------------------------------------------------
# DepthWizard v2 — Lightning AI Studio boot script.
#
# This script runs every time the Studio starts, from the home directory.
# Logs from previous runs: ~/.lightning_studio/logs/
#
# ! fast_load
# DepthWizard-results/v2/best.pt
# DepthWizard-results/v2/last.pt
# DepthWizard-results/v2/stageP_last.pt
#
# What it does, end to end:
#   1. clone (or pull) AbhayKale332/DW-Team_ENDRA into  <studio home>/DepthWizard
#   2. install the v2 dependency set (requirements-kaggle.txt + extras)
#   3. run the two-stage fine-tune (SynRS3D pretrain -> GAMUS+GeoNRW finetune)
#      with every cache redirected off the repo and onto a disposable path
#   4. package + verify the results into  <studio home>/DepthWizard-results/v2
#   5. delete every byte that is re-downloadable (datasets, HF blobs, pip /
#      compile caches) so the Studio lands back under the 100 GB quota
#
# Storage model (Lightning: 100 GB persistent, ~200 GB tolerated at runtime):
#   DepthWizard-results/   ~1-2 GB   KEEP   checkpoints, metrics, logs, zip
#   DepthWizard/           ~50 MB    KEEP   the git checkout
#   dw_cache/              up to     WIPE   streamed GAMUS/GeoNRW/SynRS3D tiles
#   hf_home/               ~2 GB     WIPE   DINOv3 encoder weights (re-pullable)
#   pip / triton / inductor caches     WIPE
#
# Usage (from the Studio home dir, or from inside a checkout):
#   sh on_start.sh                              # full run + cleanup
#   sh on_start.sh --smoke --datasets gamus     # ~15 min sanity run first
#   sh on_start.sh --install-only               # just deps
#   sh on_start.sh --skip_pretrain true         # stage F only (~3.5-4.5 h)
#   sh on_start.sh --force                      # retrain even if a run finished
#   sh on_start.sh --keep-cache                 # skip the dataset wipe
#   sh on_start.sh --background                 # detach, return the shell
#   HF_TOKEN=hf_xxx sh on_start.sh              # gated DINOv3-SAT + GAMUS
# Any unrecognised flag is forwarded verbatim to main.py -> train.py.
# ---------------------------------------------------------------------------

# The Studio hook and the documented `sh on_start.sh` both may hand us dash;
# this script uses bash arrays, so re-exec under bash before anything else.
if [ -z "${BASH_VERSION:-}" ]; then exec /usr/bin/env bash "$0" "$@"; fi

set -uo pipefail

# ===========================================================================
# 0. Wrapper flags  (everything else is forwarded to main.py)
# ===========================================================================
DO_INSTALL=1
DO_TRAIN=1
DO_CLEAN=1
WIPE_DATASETS=1
FORCE=0
BACKGROUND=0
AUTO_RESUME=1
REINSTALL=0
TRAIN_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --install-only)  DO_TRAIN=0 ;;
        --skip-install)  DO_INSTALL=0 ;;
        --reinstall)     REINSTALL=1 ;;
        --no-clean)      DO_CLEAN=0 ;;
        --keep-cache)    WIPE_DATASETS=0 ;;
        --force|--fresh) FORCE=1 ;;
        --background)    BACKGROUND=1 ;;
        --no-resume)     AUTO_RESUME=0 ;;
        *)               TRAIN_ARGS+=("$arg") ;;
    esac
done

# ===========================================================================
# 1. Paths — Lightning Studio home, or wherever this script lives
# ===========================================================================
SELF_DIR="$(cd "$(dirname "$0")" 2>/dev/null && pwd || echo "$PWD")"

if [ -d "/teamspace/studios/this_studio" ]; then
    WORK_ROOT="/teamspace/studios/this_studio"      # Lightning AI persistent disk
else
    WORK_ROOT="${DW_WORK_ROOT:-$HOME}"
fi

# If this script sits inside an existing checkout, use that instead of cloning.
if [ -d "$SELF_DIR/FineTunning/v2" ]; then
    REPO_DIR="$SELF_DIR"
    IN_PLACE=1
else
    REPO_DIR="$WORK_ROOT/DepthWizard"
    IN_PLACE=0
fi

V2_DIR="$REPO_DIR/FineTunning/v2"

# A smoke run is a throwaway sanity check: it gets its own output dir so its
# 1-epoch checkpoints can never be mistaken for the real run's (auto-resume
# would otherwise warm-start a full run from them) and it never marks the
# Studio as "trained".
SMOKE=0
case " ${TRAIN_ARGS[*]-} " in *" --smoke"*) SMOKE=1 ;; esac

if [ "$SMOKE" = "1" ]; then
    RESULTS_DIR="${DW_OUTPUT_DIR:-$WORK_ROOT/DepthWizard-results/smoke}"
else
    RESULTS_DIR="${DW_OUTPUT_DIR:-$WORK_ROOT/DepthWizard-results/v2}"
fi
CACHE_DIR="${DW_CACHE_DIR:-$WORK_ROOT/dw_cache}"
HF_HOME_DIR="${HF_HOME:-$WORK_ROOT/hf_home}"
LOG_DIR="$WORK_ROOT/DepthWizard-results/logs"
SENTINEL="$RESULTS_DIR/.dw_complete"
DEPS_STAMP="$WORK_ROOT/.dw_deps_ok"
RUN_ID="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$RESULTS_DIR" "$CACHE_DIR" "$HF_HOME_DIR" "$LOG_DIR"
BOOT_LOG="$LOG_DIR/on_start_${RUN_ID}.log"

# Detach so the Studio's start hook (and an interactive shell) returns at once.
if [ "$BACKGROUND" = "1" ]; then
    export DW_ALREADY_DETACHED=1
    filtered=()
    for a in "$@"; do [ "$a" = "--background" ] || filtered+=("$a"); done
    setsid nohup bash "$0" ${filtered[@]+"${filtered[@]}"} >"$BOOT_LOG" 2>&1 &
    echo "[dw] detached (pid $!) — tail -f $BOOT_LOG"
    exit 0
fi

exec > >(tee -a "$BOOT_LOG") 2>&1

log()  { echo "[dw $(date +%H:%M:%S)] $*"; }
warn() { echo "[dw $(date +%H:%M:%S)] WARN: $*" >&2; }
die()  { echo "[dw $(date +%H:%M:%S)] FATAL: $*" >&2; exit 1; }

log "=============================================================="
log "DepthWizard v2 — Lightning AI run ${RUN_ID}"
log "  work root : $WORK_ROOT"
log "  repo      : $REPO_DIR$([ "$IN_PLACE" = 1 ] && echo '  (in-place checkout)')"
log "  results   : $RESULTS_DIR   (kept)"
log "  cache     : $CACHE_DIR     (deleted after training)"
log "=============================================================="

# ===========================================================================
# 2. Disk helpers
# ===========================================================================
avail_gib() { df -P -BG "${1:-$WORK_ROOT}" 2>/dev/null | awk 'NR==2 {gsub(/G/,"",$4); print $4+0}'; }

disk_report() {
    log "--- disk ---"
    df -h "$WORK_ROOT" | sed 's/^/[dw]   /'
    for d in "$RESULTS_DIR" "$CACHE_DIR" "$HF_HOME_DIR" "$REPO_DIR"; do
        [ -d "$d" ] && printf '[dw]   %6s  %s\n' "$(du -sh "$d" 2>/dev/null | cut -f1)" "$d"
    done
}

# rm -rf with a guard: only ever inside the work root / home, never empty, never "/".
safe_rm() {
    local p="$1"
    [ -n "$p" ] || return 0
    [ -e "$p" ] || return 0
    case "$p" in
        /|/home|/root|/teamspace|/teamspace/studios|"$WORK_ROOT"|"$HOME") warn "refusing to delete $p"; return 0 ;;
        "$WORK_ROOT"/*|"$HOME"/*|/tmp/*) ;;
        *) warn "refusing to delete outside the work root: $p"; return 0 ;;
    esac
    rm -rf -- "$p" 2>/dev/null || true
}

# ===========================================================================
# 3. Environment — every cache lands somewhere we control (and can delete)
# ===========================================================================
export HF_HOME="$HF_HOME_DIR"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DOWNLOAD_TIMEOUT=60
export DW_OUTPUT_DIR="$RESULTS_DIR"          # read by FineTunning/v2/config.py
export DW_CACHE_DIR="$CACHE_DIR"             # read by FineTunning/v2/config.py
export PIP_CACHE_DIR="$WORK_ROOT/.cache/pip"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export TORCHINDUCTOR_CACHE_DIR="$WORK_ROOT/.cache/inductor"
export TRITON_CACHE_DIR="$WORK_ROOT/.cache/triton"
export TORCH_HOME="$WORK_ROOT/.cache/torch"
export XDG_CACHE_HOME="$WORK_ROOT/.cache"
export MPLCONFIGDIR="$WORK_ROOT/.cache/mpl"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1                     # 22 dataloader workers on 24 vCPU
export PYTHONUNBUFFERED=1
mkdir -p "$PIP_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR"

# --- Hugging Face token (DINOv3-SAT encoder + GAMUS are both gated) --------
if [ -z "${HF_TOKEN:-}" ]; then
    for f in "$WORK_ROOT/.hf_token" "$HOME/.hf_token" "$REPO_DIR/.hf_token"; do
        if [ -r "$f" ]; then HF_TOKEN="$(tr -d ' \t\r\n' < "$f")"; break; fi
    done
fi
HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-${HUGGINGFACE_TOKEN:-}}}"
export HF_TOKEN
[ -n "$HF_TOKEN" ] || warn "no HF_TOKEN — gated DINOv3-SAT / GAMUS downloads will fail.
       Set it in the Studio env, or:  echo hf_xxx > $WORK_ROOT/.hf_token"

PY="${DW_PYTHON:-}"
[ -n "$PY" ] || PY="$(command -v python || command -v python3)"
[ -n "$PY" ] || die "no python interpreter found"

# ===========================================================================
# 4. Clone / pull the repository
# ===========================================================================
GH_TOKEN="${DW_GITHUB_TOKEN:-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}"
GH_REPO="github.com/AbhayKale332/DW-Team_ENDRA.git"
GH_URL="https://x-access-token:${GH_TOKEN}@${GH_REPO}"

if [ "$IN_PLACE" = "1" ]; then
    log "using the checkout this script lives in — skipping clone"
elif [ -d "$REPO_DIR/.git" ]; then
    log "pulling latest main…"
    # Pass the token on the command line only; never persist it in .git/config.
    if git -C "$REPO_DIR" fetch --quiet "$GH_URL" main; then
        if [ "${DW_GIT_RESET:-0}" = "1" ]; then
            git -C "$REPO_DIR" reset --hard --quiet FETCH_HEAD
        elif ! git -C "$REPO_DIR" merge --ff-only --quiet FETCH_HEAD 2>/dev/null; then
            # Local edits in the Studio are worth more than being on tip-of-main:
            # keep them and say so rather than silently resetting them away.
            warn "cannot fast-forward (local commits/edits in $REPO_DIR) — training
       the checkout as-is. Force the update with:  DW_GIT_RESET=1 sh on_start.sh"
        fi
    else
        warn "fetch failed — continuing with the checkout on disk"
    fi
else
    log "cloning DepthWizard…"
    git clone --depth 1 --quiet "$GH_URL" "$REPO_DIR" || die "clone failed"
    git -C "$REPO_DIR" remote set-url origin "https://${GH_REPO}"   # strip the token
fi

[ -d "$V2_DIR" ] || die "expected $V2_DIR — wrong repo layout?"
log "HEAD: $(git -C "$REPO_DIR" log --oneline -1 2>/dev/null || echo n/a)"

# ===========================================================================
# 5. Dependencies
# ===========================================================================
REQ="$V2_DIR/requirements-kaggle.txt"
[ -f "$REQ" ] || die "missing $REQ"
REQ_HASH="$(md5sum "$REQ" | cut -d' ' -f1)"

if [ "$DO_INSTALL" = "1" ]; then
    if [ "$REINSTALL" = "0" ] && [ -f "$DEPS_STAMP" ] && grep -qx "$REQ_HASH" "$DEPS_STAMP"; then
        log "dependencies already installed for this requirements file — skipping"
    else
        log "installing dependencies (this image ships torch/CUDA; we add the rest)…"
        "$PY" -m pip install --no-input -q --upgrade pip >/dev/null 2>&1 || true
        "$PY" -m pip install --no-input -q -r "$REQ" || die "pip install -r $REQ failed"
        # Optional: parallel Rust downloader — GAMUS/SynRS3D are ~100 GB of pulls.
        if "$PY" -m pip install --no-input -q hf_transfer >/dev/null 2>&1; then
            export HF_HUB_ENABLE_HF_TRANSFER=1
            log "hf_transfer enabled (faster Hub downloads)"
        fi
        echo "$REQ_HASH" > "$DEPS_STAMP"
        "$PY" -m pip cache purge >/dev/null 2>&1 || true
    fi
fi
"$PY" -c "import hf_transfer" >/dev/null 2>&1 && export HF_HUB_ENABLE_HF_TRANSFER=1

log "verifying the runtime…"
"$PY" - <<'PYCHECK' || die "dependency check failed — see the traceback above"
import importlib, sys
missing = [m for m in ("torch", "transformers", "huggingface_hub", "h5py",
                       "tifffile", "safetensors", "numpy", "PIL")
           if not importlib.util.find_spec(m)]
if missing:
    sys.exit(f"missing modules: {missing}")
import torch
print(f"[dw]   torch {torch.__version__}  cuda={torch.cuda.is_available()} "
      f"({torch.version.cuda})  gpus={torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"[dw]   GPU{i}: {p.name}  {p.total_memory/1024**3:.0f} GB")
if not torch.cuda.is_available():
    print("[dw]   WARN: no GPU visible — this will be unusably slow for a full run")
PYCHECK

if [ "$DO_TRAIN" = "0" ]; then
    log "--install-only: done."
    exit 0
fi

# ===========================================================================
# 6. Already finished?  A Studio restart must not silently retrain.
# ===========================================================================
if [ -f "$SENTINEL" ] && [ "$FORCE" = "0" ]; then
    log "a completed run is already in $RESULTS_DIR:"
    sed 's/^/[dw]   /' "$SENTINEL"
    log "re-run with --force to train again. Nothing to do."
    disk_report
    exit 0
fi

# ===========================================================================
# 7. Size the streaming cache to the disk we actually have
# ===========================================================================
# Each dataset repo gets its own independently-capped subdir, and stage F holds
# GAMUS + GeoNRW at once — so the live footprint is ~2x the cap. GeoNRW also
# needs its ~32 GB tar.gz plus the extracted tree on disk simultaneously, so the
# cap must stay comfortably above that or eviction thrashes.
AVAIL="$(avail_gib "$WORK_ROOT")"; AVAIL="${AVAIL:-0}"
if [ -n "${DW_CACHE_MAX_GIB:-}" ]; then
    CACHE_GIB="$DW_CACHE_MAX_GIB"
else
    CACHE_GIB=$(( (AVAIL - 55) / 2 ))
    [ "$SMOKE" = "1" ] && CACHE_GIB=12
    [ "$CACHE_GIB" -gt 50 ] && CACHE_GIB=50
    [ "$CACHE_GIB" -lt 12 ] && CACHE_GIB=12
fi
log "disk available: ${AVAIL} GiB -> per-dataset cache cap ${CACHE_GIB} GiB (~$((CACHE_GIB*2)) GiB peak)"
[ "$AVAIL" -lt 45 ] && warn "under 45 GiB free — expect eviction thrash; free space or pass --smoke"

# ===========================================================================
# 8. Auto-resume from whatever a previous (killed) run left behind
# ===========================================================================
RESUME_ARGS=()
if [ "$AUTO_RESUME" = "1" ] && [ "$FORCE" = "0" ] && [ "$SMOKE" = "0" ]; then
    case " ${TRAIN_ARGS[*]-} " in
        *--init_from*|*--skip_pretrain*) ;;                     # caller decides
        *)
            if [ -f "$RESULTS_DIR/last.pt" ]; then
                RESUME_ARGS=(--init_from "$RESULTS_DIR/last.pt" --skip_pretrain true)
                log "resuming stage F from last.pt (pretrain skipped)"
            elif [ -f "$RESULTS_DIR/stageP_last.pt" ]; then
                RESUME_ARGS=(--init_from "$RESULTS_DIR/stageP_last.pt" --skip_pretrain true)
                log "resuming from the stage-P checkpoint (pretrain skipped)"
            fi
            ;;
    esac
fi

# ===========================================================================
# 9. Disk watchdog — logs usage, and sheds re-downloadable bytes if we get tight
# ===========================================================================
GUARD_PID=""
GUARD_LOG="$LOG_DIR/disk_guard_${RUN_ID}.log"
start_disk_guard() {
    # Own session + its own log file: the loop must never hold the training
    # stdout pipe open, and `kill` has to take the sleeping child with it.
    setsid bash -c '
        while true; do
            sleep 300
            free=$(df -P -BG "'"$WORK_ROOT"'" 2>/dev/null | awk "NR==2 {gsub(/G/,\"\",\$4); print \$4+0}")
            free=${free:-999}
            echo "[dw-guard $(date +%H:%M:%S)] free=${free}GiB cache=$(du -sh "'"$CACHE_DIR"'" 2>/dev/null | cut -f1)"
            if [ "$free" -lt 12 ]; then
                echo "[dw-guard] LOW DISK (${free}GiB) — dropping partial downloads"
                find "'"$CACHE_DIR"'" "'"$HF_HOME_DIR"'" -name "*.incomplete" -delete 2>/dev/null
                find "'"$CACHE_DIR"'" -type d -name ".locks" -exec rm -rf {} + 2>/dev/null
                rm -rf "'"$PIP_CACHE_DIR"'" "'"$TORCHINDUCTOR_CACHE_DIR"'"/* 2>/dev/null
            fi
        done
    ' >"$GUARD_LOG" 2>&1 </dev/null &
    GUARD_PID=$!
    log "disk watchdog running (pid $GUARD_PID) -> $GUARD_LOG"
}
stop_disk_guard() {
    [ -n "$GUARD_PID" ] || return 0
    kill -- "-$GUARD_PID" 2>/dev/null || kill "$GUARD_PID" 2>/dev/null
    pkill -P "$GUARD_PID" 2>/dev/null
    GUARD_PID=""
}

# ===========================================================================
# 10. Cleanup
# ===========================================================================
# light: only ever-disposable build/compile junk (safe after a failure — the
#        dataset cache is left intact so the next start resumes cheaply)
# full : light + every streamed dataset byte + the HF blob cache
cleanup_storage() {
    local mode="$1"
    local before after
    before="$(avail_gib "$WORK_ROOT")"; before="${before:-0}"
    log "cleanup ($mode)…"

    "$PY" -m pip cache purge >/dev/null 2>&1 || true
    safe_rm "$PIP_CACHE_DIR"
    safe_rm "$TORCHINDUCTOR_CACHE_DIR"
    safe_rm "$TRITON_CACHE_DIR"
    safe_rm "$WORK_ROOT/.cache/matplotlib"; safe_rm "$MPLCONFIGDIR"
    safe_rm "$V2_DIR/bin"                      # cloudflared binary, if it self-installed
    safe_rm "$V2_DIR/.pytest_cache"; safe_rm "$V2_DIR/.ruff_cache"
    find "$REPO_DIR" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null
    find "$CACHE_DIR" "$HF_HOME_DIR" -name '*.incomplete' -delete 2>/dev/null
    safe_rm "/tmp/dw_cache"

    if [ "$mode" = "full" ]; then
        # The streamed tiles: GAMUS h5, the GeoNRW tar + extracted tree, SynRS3D
        # zips. All re-downloadable, none referenced by the saved checkpoints.
        safe_rm "$CACHE_DIR"
        safe_rm "$V2_DIR/outputs/dw_cache"     # in case a default-path run happened
        safe_rm "$V2_DIR/data"                 # local_root, only used with --data_source local
        # DINOv3-SAT encoder weights (~1.2 GB) — frozen, public, re-pulled on demand.
        if [ "${DW_KEEP_MODEL_CACHE:-0}" != "1" ]; then
            safe_rm "$HF_HOME_DIR/hub"
            safe_rm "$HF_HOME_DIR/xet"
            safe_rm "$HOME/.cache/huggingface"
        else
            safe_rm "$HF_HOME_DIR/hub/datasets--"*
        fi
        safe_rm "$WORK_ROOT/.cache/torch"
        mkdir -p "$CACHE_DIR"
    fi
    sync 2>/dev/null || true
    after="$(avail_gib "$WORK_ROOT")"; after="${after:-0}"
    log "cleanup done — reclaimed ~$(( after - before )) GiB (now ${after} GiB free)"
}

on_interrupt() {
    warn "interrupted — saving what exists and releasing scratch space"
    stop_disk_guard
    cleanup_storage light
    disk_report
    exit 130
}
trap on_interrupt INT TERM

# ===========================================================================
# 11. Train
# ===========================================================================
cd "$V2_DIR" || die "cannot cd $V2_DIR"

# Ours first, the caller's last: argparse keeps the final occurrence, so any of
# these can be overridden straight from the command line.
BASE_ARGS=(
    --skip-install                       # section 5 already did it
    --cache_max_gib "$CACHE_GIB"
    --make_zip true
    --share_cloudflared false            # Lightning has persistent disk; no tunnel needed
    --keep_alive_minutes 0
)

log "launching: $PY -u main.py ${BASE_ARGS[*]} ${RESUME_ARGS[*]-} ${TRAIN_ARGS[*]-}"
disk_report
start_disk_guard
T0=$SECONDS

"$PY" -u main.py "${BASE_ARGS[@]}" ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"}
RC=$?

stop_disk_guard
trap - INT TERM
ELAPSED=$(( (SECONDS - T0) / 60 ))
log "training exited rc=$RC after ${ELAPSED} min"

# ===========================================================================
# 12. Save / verify the results
# ===========================================================================
# train.py writes best.pt / last.pt / metrics.json / config.json / run.log /
# viewer_sample/ / qualitative/ straight into $RESULTS_DIR and zips them. If the
# packaging step failed mid-run, redo it here so the deliverable always exists.
if [ ! -f "$RESULTS_DIR/results_v2.zip" ] && [ -f "$RESULTS_DIR/metrics.json" ]; then
    log "rebuilding results_v2.zip…"
    "$PY" package_results.py --output_dir "$RESULTS_DIR" || warn "packaging failed"
fi

{
    echo "run_id      : $RUN_ID"
    echo "finished_at : $(date -Is)"
    echo "exit_code   : $RC"
    echo "elapsed_min : $ELAPSED"
    echo "commit      : $(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo n/a)"
    echo "host        : $(hostname)  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    echo "args        : ${BASE_ARGS[*]} ${RESUME_ARGS[*]-} ${TRAIN_ARGS[*]-}"
    echo
    echo "--- artifacts ---"
    ( cd "$RESULTS_DIR" && find . -maxdepth 2 -type f -printf '%10s  %p\n' 2>/dev/null | sort -k2 )
    echo
    echo "--- checkpoint sha256 ---"
    ( cd "$RESULTS_DIR" && sha256sum ./*.pt 2>/dev/null )
} > "$RESULTS_DIR/RUN_MANIFEST.txt"

cp -f "$BOOT_LOG" "$RESULTS_DIR/on_start.log" 2>/dev/null || true

MISSING=""
for f in best.pt metrics.json config.json run.log; do
    [ -s "$RESULTS_DIR/$f" ] || MISSING="$MISSING $f"
done

if [ -f "$RESULTS_DIR/metrics.json" ]; then
    "$PY" - "$RESULTS_DIR/metrics.json" <<'PYSUM' || true
import json, sys
m = json.load(open(sys.argv[1]))
best = m.get("best_val_rmse_m")
print(f"[dw]   best val RMSE : {best if best not in (None, float('inf')) else 'n/a'} m")
for k in ("final_plain", "final_tta"):
    g = (m.get(k) or {}).get("global")
    if g:
        print(f"[dw]   {k:12}: RMSE={g['rmse_m']:.3f} MAE={g['mae_m']:.3f} "
              f"r={g['pearson_r']:.3f} d1={g['delta1']:.3f}")
print(f"[dw]   epochs logged : {len(m.get('history', []))}")
PYSUM
fi

# ===========================================================================
# 13. Reclaim the disk
# ===========================================================================
if [ "$RC" -eq 0 ] && [ -z "$MISSING" ]; then
    if [ "$SMOKE" = "1" ]; then
        log "smoke run OK -> $RESULTS_DIR  (no completion marker; run without --smoke for the real thing)"
    else
        printf 'run_id=%s\nfinished_at=%s\nelapsed_min=%s\n' "$RUN_ID" "$(date -Is)" "$ELAPSED" > "$SENTINEL"
    fi
    if [ "$SMOKE" = "1" ]; then
        cleanup_storage light            # keep the warm cache for the real run
    elif [ "$DO_CLEAN" = "1" ] && [ "$WIPE_DATASETS" = "1" ]; then
        cleanup_storage full
    elif [ "$DO_CLEAN" = "1" ]; then
        cleanup_storage light
        log "--keep-cache: dataset cache left in $CACHE_DIR ($(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1))"
    fi
else
    [ -n "$MISSING" ] && warn "missing expected artifacts:$MISSING"
    warn "run did not complete cleanly — keeping the dataset cache so the next
       start resumes instead of re-downloading. Wipe it by hand with:
         rm -rf $CACHE_DIR"
    [ "$DO_CLEAN" = "1" ] && cleanup_storage light
fi

# ===========================================================================
# 14. Summary
# ===========================================================================
log "=============================================================="
log "RESULTS (persisted): $RESULTS_DIR"
ls -lh "$RESULTS_DIR" 2>/dev/null | sed 's/^/[dw]   /'
disk_report
log "download the bundle with:  lightning download ... or the Studio file browser"
log "  $RESULTS_DIR/results_v2.zip"
if [ -s "$GUARD_LOG" ]; then
    log "disk watchdog (last 3 samples):"
    tail -3 "$GUARD_LOG" | sed 's/^/[dw]   /'
    cp -f "$GUARD_LOG" "$RESULTS_DIR/disk_guard.log" 2>/dev/null || true
fi
log "boot log: $BOOT_LOG"
log "=============================================================="
exit "$RC"
