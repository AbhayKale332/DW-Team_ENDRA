#!/bin/bash
# ---------------------------------------------------------------------------
# DepthWizard v3 — Lightning AI Studio boot script.
#
# This script runs every time the Studio starts, from the home directory.
# Logs from previous runs: ~/.lightning_studio/logs/
#
# ! fast_load
# DepthWizard-results/v3/best.pt
# DepthWizard-results/v3/last.pt
# DepthWizard-results/results_v3.zip
#
# What it does, end to end:
#   1. clone (or pull) akashch1512/DepthWizard into  <studio home>/DepthWizard
#   2. install FineTunning/v3/requirements.txt
#   3. PREPARE — materialise HF datasets into memmap shards on the persistent
#      disk (one time, resumable, idempotent). Training never touches the
#      network afterwards.
#   4. TRAIN  — a single run (no two-stage pretrain; v2's stage P was harmful,
#      see README §1.5), auto-resuming from last.pt if a previous boot died
#   5. package + verify the results into  <studio home>/DepthWizard-results/v3
#   6. delete the scratch that is cheap to recreate (download staging, HF blobs,
#      pip / compile caches) — but NEVER the packed shards, which are the whole
#      point of the prepare step
#
# Storage model (Lightning: 100 GB persistent, ~200 GB tolerated at runtime):
#   DepthWizard-results/   ~1-2 GB   KEEP   checkpoints, metrics, logs, zip
#   DepthWizard-data/      ~35 GB    KEEP   packed shards (re-prepare = ~40 min)
#   DepthWizard/           ~50 MB    KEEP   the git checkout
#   DepthWizard-data/_dl/  up to     WIPE   download staging (32 GB for GeoNRW)
#   hf_home/               ~2 GB     WIPE   DINOv3 encoder weights (re-pullable)
#   pip / triton / inductor caches     WIPE
#
# The v2 script wiped its dataset cache on every successful run because v2
# streamed tiles per step. v3 inverts that: the shards ARE the artifact that
# makes the second run start instantly, so they survive cleanup unless you
# explicitly pass --wipe-data.
#
# Usage (from the Studio home dir, or from inside a checkout):
#   sh on_start.sh                              # prepare (if needed) + train + cleanup
#   sh on_start.sh --check                      # deps + offline tests + GPU report
#   sh on_start.sh --smoke                      # ~10 min sanity run, own data root
#   sh on_start.sh --install-only               # just deps
#   sh on_start.sh --prepare-only               # just the data pack
#   sh on_start.sh --skip-prepare               # train against what is on disk
#   sh on_start.sh --datasets gamus             # prepare AND train on GAMUS only
#   sh on_start.sh --force                      # retrain even if a run finished
#   sh on_start.sh --wipe-data                  # also drop the shards at the end
#   sh on_start.sh --background                 # detach, return the shell
#   HF_TOKEN=hf_xxx sh on_start.sh              # gated DINOv3-SAT + GAMUS
# Any unrecognised flag is forwarded verbatim to train.py (see config.py).
#
# Prepare-side sizing is env-driven, since those flags belong to prepare_data.py
# rather than to train.py:
#   DW_GAMUS_TRAIN=4000  DW_GAMUS_VAL=400  DW_SYNRS3D_ARCHIVES=2  DW_GEONRW_MAX=2500
# ---------------------------------------------------------------------------

# The Studio hook and the documented `sh on_start.sh` both may hand us dash;
# this script uses bash arrays, so re-exec under bash before anything else.
if [ -z "${BASH_VERSION:-}" ]; then exec /usr/bin/env bash "$0" "$@"; fi

set -uo pipefail

# ===========================================================================
# 0. Wrapper flags  (everything else is forwarded to train.py)
# ===========================================================================
DO_INSTALL=1
DO_PREPARE=1
DO_TRAIN=1
DO_CHECK=0
DO_CLEAN=1
WIPE_DATA=0
REPREPARE=0
FORCE=0
BACKGROUND=0
AUTO_RESUME=1
REINSTALL=0
TRAIN_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --install-only)  DO_PREPARE=0; DO_TRAIN=0 ;;
        --skip-install)  DO_INSTALL=0 ;;
        --reinstall)     REINSTALL=1 ;;
        --prepare-only)  DO_TRAIN=0 ;;
        --skip-prepare)  DO_PREPARE=0 ;;
        --reprepare)     REPREPARE=1 ;;
        --check)         DO_CHECK=1; DO_PREPARE=0; DO_TRAIN=0 ;;
        --no-clean)      DO_CLEAN=0 ;;
        --wipe-data)     WIPE_DATA=1 ;;
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
if [ -d "$SELF_DIR/FineTunning/v3" ]; then
    REPO_DIR="$SELF_DIR"; IN_PLACE=1
elif [ -d "$SELF_DIR/../../FineTunning/v3" ]; then
    REPO_DIR="$(cd "$SELF_DIR/../.." && pwd)"; IN_PLACE=1      # script lives in v3/
else
    REPO_DIR="$WORK_ROOT/DepthWizard"; IN_PLACE=0
fi

V3_DIR="$REPO_DIR/FineTunning/v3"

# A smoke run is a throwaway sanity check. It gets its own output dir so its
# 2-epoch checkpoints can never be mistaken for the real run's (auto-resume
# would otherwise warm-start a full run from them), its own small data root so
# a 64-tile pack can never masquerade as the real 4000-tile one (prepare skips
# any split that already has an index.json), and it never marks the Studio as
# "trained".
SMOKE=0
case " ${TRAIN_ARGS[*]-} " in *" --smoke"*) SMOKE=1 ;; esac

if [ "$SMOKE" = "1" ]; then
    RESULTS_DIR="${DW_OUTPUT_DIR:-$WORK_ROOT/DepthWizard-results/smoke}"
    DATA_ROOT="${DW_DATA_ROOT:-$WORK_ROOT/DepthWizard-data-smoke}"
else
    RESULTS_DIR="${DW_OUTPUT_DIR:-$WORK_ROOT/DepthWizard-results/v3}"
    DATA_ROOT="${DW_DATA_ROOT:-$WORK_ROOT/DepthWizard-data}"
fi
HF_HOME_DIR="${HF_HOME:-$WORK_ROOT/hf_home}"
LOG_DIR="$WORK_ROOT/DepthWizard-results/logs"
SENTINEL="$RESULTS_DIR/.dw_complete"
DEPS_STAMP="$WORK_ROOT/.dw_v3_deps_ok"
RUN_ID="$(date +%Y%m%d_%H%M%S)"

# build_zip() writes results_<basename>.zip into the PARENT of the output dir.
ZIP_PATH="$(dirname "$RESULTS_DIR")/results_$(basename "$RESULTS_DIR").zip"

mkdir -p "$RESULTS_DIR" "$DATA_ROOT" "$HF_HOME_DIR" "$LOG_DIR"
BOOT_LOG="$LOG_DIR/on_start_v3_${RUN_ID}.log"

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
log "DepthWizard v3 — Lightning AI run ${RUN_ID}"
log "  work root : $WORK_ROOT"
log "  repo      : $REPO_DIR$([ "$IN_PLACE" = 1 ] && echo '  (in-place checkout)')"
log "  results   : $RESULTS_DIR   (kept)"
log "  data      : $DATA_ROOT   (kept — packed shards)"
log "=============================================================="

# ===========================================================================
# 2. Disk helpers
# ===========================================================================
avail_gib() { df -P -BG "${1:-$WORK_ROOT}" 2>/dev/null | awk 'NR==2 {gsub(/G/,"",$4); print $4+0}'; }

disk_report() {
    log "--- disk ---"
    df -h "$WORK_ROOT" | sed 's/^/[dw]   /'
    for d in "$RESULTS_DIR" "$DATA_ROOT" "$DATA_ROOT/_dl" "$HF_HOME_DIR" "$REPO_DIR"; do
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
export DW_OUTPUT_DIR="$RESULTS_DIR"          # read by FineTunning/v3/config.py
export DW_DATA_ROOT="$DATA_ROOT"             # read by FineTunning/v3/config.py
export PIP_CACHE_DIR="$WORK_ROOT/.cache/pip"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export TORCHINDUCTOR_CACHE_DIR="$WORK_ROOT/.cache/inductor"
export TRITON_CACHE_DIR="$WORK_ROOT/.cache/triton"
export TORCH_HOME="$WORK_ROOT/.cache/torch"
export XDG_CACHE_HOME="$WORK_ROOT/.cache"
export MPLCONFIGDIR="$WORK_ROOT/.cache/mpl"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1                     # 12 dataloader workers on 24 vCPU
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
# No credentials live in this file (see run_lightning.sh: a token committed to a
# script ends up in git history). For a private checkout, put a PAT in the
# Studio env as DW_GITHUB_TOKEN, or in $WORK_ROOT/.gh_token, or configure
#   git config --global credential.helper store
GH_REPO="github.com/akashch1512/DepthWizard.git"
GH_TOKEN="${DW_GITHUB_TOKEN:-}"
if [ -z "$GH_TOKEN" ]; then
    for f in "$WORK_ROOT/.gh_token" "$HOME/.gh_token"; do
        if [ -r "$f" ]; then GH_TOKEN="$(tr -d ' \t\r\n' < "$f")"; break; fi
    done
fi
if [ -n "$GH_TOKEN" ]; then
    GH_URL="https://x-access-token:${GH_TOKEN}@${GH_REPO}"
else
    GH_URL="https://${GH_REPO}"
fi

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

[ -d "$V3_DIR" ] || die "expected $V3_DIR — wrong repo layout?"
log "HEAD: $(git -C "$REPO_DIR" log --oneline -1 2>/dev/null || echo n/a)"

# ===========================================================================
# 5. Dependencies
# ===========================================================================
REQ="$V3_DIR/requirements.txt"
[ -f "$REQ" ] || die "missing $REQ"
REQ_HASH="$(md5sum "$REQ" | cut -d' ' -f1)"

if [ "$DO_INSTALL" = "1" ]; then
    if [ "$REINSTALL" = "0" ] && [ -f "$DEPS_STAMP" ] && grep -qx "$REQ_HASH" "$DEPS_STAMP"; then
        log "dependencies already installed for this requirements file — skipping"
    else
        log "installing dependencies (this image ships torch/CUDA; we add the rest)…"
        "$PY" -m pip install --no-input -q --upgrade pip >/dev/null 2>&1 || true
        "$PY" -m pip install --no-input -q -r "$REQ" || die "pip install -r $REQ failed"
        # Optional: parallel Rust downloader — the prepare step pulls ~40 GB.
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
                       "tifffile", "rasterio", "safetensors", "scipy",
                       "numpy", "PIL")
           if not importlib.util.find_spec(m)]
if missing:
    sys.exit(f"missing modules: {missing}")
import torch, transformers
print(f"[dw]   torch {torch.__version__}  cuda={torch.cuda.is_available()} "
      f"({torch.version.cuda})  gpus={torch.cuda.device_count()}")
print(f"[dw]   transformers {transformers.__version__}  (DINOv3 needs >= 4.56)")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"[dw]   GPU{i}: {p.name}  {p.total_memory/1024**3:.0f} GB")
if not torch.cuda.is_available():
    print("[dw]   WARN: no GPU visible — this will be unusably slow for a full run")
PYCHECK

# --- --check: the offline test suite, no GPU and no network required -------
if [ "$DO_CHECK" = "1" ]; then
    log "running the offline test suite…"
    ( cd "$V3_DIR" && "$PY" -m pytest -q )
    RC=$?
    log "--check: pytest exited rc=$RC"
    exit "$RC"
fi

if [ "$DO_PREPARE" = "0" ] && [ "$DO_TRAIN" = "0" ]; then
    log "--install-only: done."
    exit 0
fi

# ===========================================================================
# 6. Which datasets, and are they already packed?
# ===========================================================================
# `--datasets a,b` is a train.py flag, but prepare_data.py takes the same name —
# so honour one spelling for both and never download a source the run won't use.
DATASETS="${DW_PREPARE_DATASETS:-}"
if [ -z "$DATASETS" ]; then
    prev=""
    for a in ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"}; do
        case "$a" in
            --datasets=*) DATASETS="${a#--datasets=}" ;;
            *) [ "$prev" = "--datasets" ] && DATASETS="$a" ;;
        esac
        prev="$a"
    done
fi
DATASETS="${DATASETS:-gamus,synrs3d}"          # config.py's default

# A source is "prepared" when every split prepare_data.py writes for it exists.
store_ok() { [ -f "$DATA_ROOT/$1/index.json" ]; }
source_prepared() {
    case "$1" in
        gamus)   store_ok gamus/train && store_ok gamus/val ;;
        synrs3d) store_ok synrs3d/train ;;
        geonrw)  store_ok geonrw/train ;;
        *)       return 1 ;;
    esac
}

NEED_PREP=""
for d in ${DATASETS//,/ }; do
    source_prepared "$d" || NEED_PREP="$NEED_PREP,$d"
done
NEED_PREP="${NEED_PREP#,}"

case ",$DATASETS," in
    *,gamus,*) ;;
    *) warn "GAMUS is not in --datasets — it is the only source prepare_data.py
       writes a dedicated val split for, so validation will fall back to a slice
       of the training set (see dwdata/loaders.py)." ;;
esac

# ===========================================================================
# 7. Already finished?  A Studio restart must not silently retrain.
# ===========================================================================
if [ "$DO_TRAIN" = "1" ] && [ -f "$SENTINEL" ] && [ "$FORCE" = "0" ]; then
    log "a completed run is already in $RESULTS_DIR:"
    sed 's/^/[dw]   /' "$SENTINEL"
    log "re-run with --force to train again. Nothing to do."
    disk_report
    exit 0
fi

# ===========================================================================
# 8. Cleanup  (defined early so the interrupt trap can use it)
# ===========================================================================
# light: only ever-disposable build/compile junk plus the download staging dir
#        (safe after a failure — the packed shards are left intact so the next
#        start goes straight to training)
# full : light + the HF blob cache
# data : full + the packed shards themselves (--wipe-data; ~40 min to rebuild)
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
    safe_rm "$V3_DIR/.pytest_cache"; safe_rm "$V3_DIR/.ruff_cache"
    find "$REPO_DIR" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null
    # prepare_data.py unlinks these itself on the happy path; a killed prepare
    # leaves the half-downloaded GeoNRW tar (32 GB) or a SynRS3D archive behind.
    safe_rm "$DATA_ROOT/_dl"
    find "$DATA_ROOT" "$HF_HOME_DIR" -name '*.incomplete' -delete 2>/dev/null

    if [ "$mode" = "full" ] || [ "$mode" = "data" ]; then
        # DINOv3-SAT encoder weights (~1.2 GB) — public, re-pulled on demand.
        if [ "${DW_KEEP_MODEL_CACHE:-0}" != "1" ]; then
            safe_rm "$HF_HOME_DIR/hub"
            safe_rm "$HF_HOME_DIR/xet"
            safe_rm "$HOME/.cache/huggingface"
        else
            safe_rm "$HF_HOME_DIR/hub/datasets--"*
        fi
        safe_rm "$WORK_ROOT/.cache/torch"
    fi

    if [ "$mode" = "data" ]; then
        warn "--wipe-data: deleting the packed shards in $DATA_ROOT
       ($(du -sh "$DATA_ROOT" 2>/dev/null | cut -f1)) — the next run re-prepares (~40 min)"
        for d in gamus synrs3d geonrw; do safe_rm "$DATA_ROOT/$d"; done
    fi

    sync 2>/dev/null || true
    after="$(avail_gib "$WORK_ROOT")"; after="${after:-0}"
    log "cleanup done — reclaimed ~$(( after - before )) GiB (now ${after} GiB free)"
}

on_interrupt() {
    warn "interrupted — saving what exists and releasing scratch space"
    stop_disk_guard 2>/dev/null || true
    cleanup_storage light
    disk_report
    exit 130
}

# ===========================================================================
# 9. Disk watchdog — logs usage, and sheds re-downloadable bytes if we get tight
# ===========================================================================
GUARD_PID=""
GUARD_LOG="$LOG_DIR/disk_guard_v3_${RUN_ID}.log"
start_disk_guard() {
    # Own session + its own log file: the loop must never hold the training
    # stdout pipe open, and `kill` has to take the sleeping child with it.
    setsid bash -c '
        while true; do
            sleep 300
            free=$(df -P -BG "'"$WORK_ROOT"'" 2>/dev/null | awk "NR==2 {gsub(/G/,\"\",\$4); print \$4+0}")
            free=${free:-999}
            echo "[dw-guard $(date +%H:%M:%S)] free=${free}GiB data=$(du -sh "'"$DATA_ROOT"'" 2>/dev/null | cut -f1) staging=$(du -sh "'"$DATA_ROOT"'/_dl" 2>/dev/null | cut -f1)"
            if [ "$free" -lt 12 ]; then
                echo "[dw-guard] LOW DISK (${free}GiB) — dropping partial downloads"
                find "'"$DATA_ROOT"'" "'"$HF_HOME_DIR"'" -name "*.incomplete" -delete 2>/dev/null
                find "'"$DATA_ROOT"'" "'"$HF_HOME_DIR"'" -type d -name ".locks" -exec rm -rf {} + 2>/dev/null
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

trap on_interrupt INT TERM
cd "$V3_DIR" || die "cannot cd $V3_DIR"

# ===========================================================================
# 10. Prepare — HF -> memmap shards, once, on the persistent disk
# ===========================================================================
# Sizes default to README §4's pack (~35 GB); a smoke run packs a token amount
# into its own root so it can finish in minutes.
if [ "$SMOKE" = "1" ]; then
    GAMUS_TRAIN="${DW_GAMUS_TRAIN:-64}";   GAMUS_VAL="${DW_GAMUS_VAL:-16}"
    SYN_ARCHIVES="${DW_SYNRS3D_ARCHIVES:-1}"; GEONRW_MAX="${DW_GEONRW_MAX:-64}"
else
    GAMUS_TRAIN="${DW_GAMUS_TRAIN:-4000}"; GAMUS_VAL="${DW_GAMUS_VAL:-400}"
    SYN_ARCHIVES="${DW_SYNRS3D_ARCHIVES:-2}"; GEONRW_MAX="${DW_GEONRW_MAX:-2500}"
fi

if [ "$DO_PREPARE" = "1" ] && { [ -n "$NEED_PREP" ] || [ "$REPREPARE" = "1" ]; }; then
    TO_PREP="$DATASETS"
    [ "$REPREPARE" = "1" ] || TO_PREP="$NEED_PREP"

    # Rough packed cost, so a doomed pack fails in the first second and not at
    # 90 %: GAMUS 6.3 MB/1024px tile, SynRS3D ~7 GiB/archive, GeoNRW 6 MB/tile.
    NEED_GIB=0
    case ",$TO_PREP," in *,gamus,*)   NEED_GIB=$(( NEED_GIB + (GAMUS_TRAIN + GAMUS_VAL) * 63 / 10000 )) ;; esac
    case ",$TO_PREP," in *,synrs3d,*) NEED_GIB=$(( NEED_GIB + SYN_ARCHIVES * 7 )) ;; esac
    case ",$TO_PREP," in *,geonrw,*)  NEED_GIB=$(( NEED_GIB + GEONRW_MAX * 6 / 1000 + 35 )) ;; esac   # +35 = the tar
    AVAIL="$(avail_gib "$WORK_ROOT")"; AVAIL="${AVAIL:-0}"
    log "preparing [$TO_PREP] -> $DATA_ROOT  (~${NEED_GIB} GiB needed, ${AVAIL} GiB free)"
    if [ "$AVAIL" -lt "$NEED_GIB" ]; then
        warn "not enough free disk for this pack. Shrink it, e.g.
       DW_GAMUS_TRAIN=2000 sh on_start.sh   /   sh on_start.sh --datasets gamus"
    fi

    PREP_ARGS=(--data_root "$DATA_ROOT" --datasets "$TO_PREP"
               --gamus_train "$GAMUS_TRAIN" --gamus_val "$GAMUS_VAL"
               --synrs3d_archives "$SYN_ARCHIVES" --geonrw_max "$GEONRW_MAX"
               --workers "${DW_PREP_WORKERS:-12}")
    [ "$REPREPARE" = "1" ] && PREP_ARGS+=(--force)

    start_disk_guard
    P0=$SECONDS
    "$PY" -u main.py --skip-install --prepare "${PREP_ARGS[@]}"
    PRC=$?
    stop_disk_guard
    log "prepare exited rc=$PRC after $(( (SECONDS - P0) / 60 )) min"

    if [ "$PRC" -ne 0 ]; then
        # Partial shards are still usable — a split only gets its index.json once
        # it is complete — so report what landed rather than deleting anything.
        warn "prepare failed. Packed stores that DID complete:"
        find "$DATA_ROOT" -name index.json -printf '       %h\n' 2>/dev/null
        cleanup_storage light
        disk_report
        exit "$PRC"
    fi

    # Re-check: a source can "succeed" and still have written nothing (a 401 on a
    # gated repo is caught per-file), and training on a missing store is worse
    # than stopping here.
    STILL_MISSING=""
    for d in ${DATASETS//,/ }; do
        source_prepared "$d" || STILL_MISSING="$STILL_MISSING $d"
    done
    [ -n "$STILL_MISSING" ] && warn "prepare left these sources unpacked:$STILL_MISSING
       (gated repo without HF_TOKEN? train.py will skip them)"
    safe_rm "$DATA_ROOT/_dl"
elif [ "$DO_PREPARE" = "1" ]; then
    log "all requested sources already packed in $DATA_ROOT — skipping prepare"
fi

log "--- packed stores ---"
find "$DATA_ROOT" -name index.json 2>/dev/null | sort | while IFS= read -r p; do
    "$PY" - "$p" <<'PYIDX' 2>/dev/null || true
import json, sys
from pathlib import Path
p = Path(sys.argv[1]); i = json.loads(p.read_text())
gib = sum(f.stat().st_size for f in p.parent.glob("*.npy")) / 1024 ** 3
print(f"[dw]   {p.parent.parent.name + '/' + p.parent.name:<16} {i['n']:>6} tiles "
      f"@ {i['tile_px']}px / {i['gsd_m']} m  ({gib:.1f} GiB)")
PYIDX
done

if [ "$DO_TRAIN" = "0" ]; then
    log "--prepare-only: done."
    [ "$DO_CLEAN" = "1" ] && cleanup_storage light
    disk_report
    exit 0
fi

# ===========================================================================
# 11. Auto-resume from whatever a previous (killed) run left behind
# ===========================================================================
# v3 has one stage, so resuming is just --resume: train.py loads the weights
# with strict=False and restarts the schedule. best.pt survives either way.
RESUME_ARGS=()
if [ "$AUTO_RESUME" = "1" ] && [ "$FORCE" = "0" ] && [ "$SMOKE" = "0" ]; then
    case " ${TRAIN_ARGS[*]-} " in
        *--resume*) ;;                                   # caller decides
        *)
            if [ -f "$RESULTS_DIR/last.pt" ]; then
                RESUME_ARGS=(--resume "$RESULTS_DIR/last.pt")
                log "warm-starting from last.pt (a previous boot did not finish)"
            elif [ -f "$RESULTS_DIR/best.pt" ]; then
                RESUME_ARGS=(--resume "$RESULTS_DIR/best.pt")
                log "warm-starting from best.pt"
            fi
            ;;
    esac
fi

# ===========================================================================
# 12. Train
# ===========================================================================
# Ours first, the caller's last: argparse keeps the final occurrence, so any of
# these can be overridden straight from the command line.
BASE_ARGS=(
    --skip-install                       # section 5 already did it
    --data_root "$DATA_ROOT"
    --output_dir "$RESULTS_DIR"
    --datasets "$DATASETS"
    --make_zip true
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
# 13. Save / verify the results
# ===========================================================================
# train.py writes best.pt / last.pt / metrics.json / config.json / preproc.json /
# run.log / viewer_sample/ / qualitative/ into $RESULTS_DIR and zips them. If the
# packaging step failed mid-run, redo it here so the deliverable always exists.
if [ ! -f "$ZIP_PATH" ] && [ -f "$RESULTS_DIR/metrics.json" ]; then
    log "rebuilding $(basename "$ZIP_PATH")…"
    "$PY" - "$RESULTS_DIR" <<'PYZIP' || warn "packaging failed"
import sys
sys.path.insert(0, ".")
from package_results import build_zip, write_env_files
write_env_files(sys.argv[1])
print(f"[dw]   {build_zip(sys.argv[1])}")
PYZIP
fi

{
    echo "run_id      : $RUN_ID"
    echo "finished_at : $(date -Is)"
    echo "exit_code   : $RC"
    echo "elapsed_min : $ELAPSED"
    echo "commit      : $(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo n/a)"
    echo "host        : $(hostname)  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    echo "datasets    : $DATASETS"
    echo "data_root   : $DATA_ROOT"
    echo "args        : ${BASE_ARGS[*]} ${RESUME_ARGS[*]-} ${TRAIN_ARGS[*]-}"
    echo
    echo "--- packed stores ---"
    find "$DATA_ROOT" -name index.json -printf '%h\n' 2>/dev/null | sort
    echo
    echo "--- artifacts ---"
    ( cd "$RESULTS_DIR" && find . -maxdepth 2 -type f -printf '%10s  %p\n' 2>/dev/null | sort -k2 )
    echo
    echo "--- checkpoint sha256 ---"
    ( cd "$RESULTS_DIR" && sha256sum ./*.pt 2>/dev/null )
} > "$RESULTS_DIR/RUN_MANIFEST.txt"

cp -f "$BOOT_LOG" "$RESULTS_DIR/on_start.log" 2>/dev/null || true

MISSING=""
for f in best.pt metrics.json config.json preproc.json run.log; do
    [ -s "$RESULTS_DIR/$f" ] || MISSING="$MISSING $f"
done

# README §5: quote final_sliding_tta, and watch four numbers, not one.
if [ -f "$RESULTS_DIR/metrics.json" ]; then
    "$PY" - "$RESULTS_DIR/metrics.json" <<'PYSUM' || true
import json, sys

m = json.load(open(sys.argv[1]))
best = m.get("best_val_rmse_m")
print(f"[dw]   best centre-crop val RMSE : "
      f"{best if best not in (None, float('inf')) else 'n/a'} m")
print(f"[dw]   epochs logged             : {len(m.get('history', []))}")


def num(v, sign=False):
    if not isinstance(v, (int, float)):
        return "n/a"
    return f"{v:+.3f}" if sign else f"{v:.3f}"


for k in ("final_plain", "final_tta", "final_sliding_tta"):
    d = m.get(k)
    if not isinstance(d, dict):
        continue
    g = d.get("global") or {}
    tall, flat = d.get("tall_gt15m") or {}, d.get("flat_lt1m") or {}
    print(f"[dw]   {k:18}: RMSE={num(g.get('rmse_m'))} MAE={num(g.get('mae_m'))} "
          f"r={num(g.get('pearson_r'))} d1={num(g.get('delta1'))} "
          f"bal={num(d.get('balanced_rmse_m'))} "
          f"tall_bias={num(tall.get('bias_m'), sign=True)} "
          f"flat_bias={num(flat.get('bias_m'), sign=True)}")
if "final_sliding_tta" in m:
    print("[dw]   ^ final_sliding_tta is the number to quote (README §5)")
else:
    print("[dw]   WARN: no final_sliding_tta — the final eval did not complete")
PYSUM
fi

# ===========================================================================
# 14. Reclaim the disk
# ===========================================================================
if [ "$RC" -eq 0 ] && [ -z "$MISSING" ]; then
    if [ "$SMOKE" = "1" ]; then
        log "smoke run OK -> $RESULTS_DIR  (no completion marker; run without --smoke for the real thing)"
    else
        printf 'run_id=%s\nfinished_at=%s\nelapsed_min=%s\n' "$RUN_ID" "$(date -Is)" "$ELAPSED" > "$SENTINEL"
    fi
    if [ "$DO_CLEAN" = "0" ]; then
        :
    elif [ "$WIPE_DATA" = "1" ]; then
        cleanup_storage data
    else
        cleanup_storage full
        log "packed shards kept in $DATA_ROOT ($(du -sh "$DATA_ROOT" 2>/dev/null | cut -f1)) — the next run starts training immediately"
    fi
else
    [ -n "$MISSING" ] && warn "missing expected artifacts:$MISSING"
    warn "run did not complete cleanly — keeping everything re-usable so the next
       start resumes instead of re-downloading:
         checkpoints  $RESULTS_DIR
         shards       $DATA_ROOT"
    [ "$DO_CLEAN" = "1" ] && cleanup_storage light
fi

# ===========================================================================
# 15. Summary
# ===========================================================================
log "=============================================================="
log "RESULTS (persisted): $RESULTS_DIR"
ls -lh "$RESULTS_DIR" 2>/dev/null | sed 's/^/[dw]   /'
disk_report
log "download the bundle with:  lightning download ... or the Studio file browser"
log "  $ZIP_PATH"
log "predict with the trained checkpoint:"
log "  cd $V3_DIR && $PY -m infer.predict --ckpt $RESULTS_DIR/best.pt --image <path>"
if [ -s "$GUARD_LOG" ]; then
    log "disk watchdog (last 3 samples):"
    tail -3 "$GUARD_LOG" | sed 's/^/[dw]   /'
    cp -f "$GUARD_LOG" "$RESULTS_DIR/disk_guard.log" 2>/dev/null || true
fi
log "boot log: $BOOT_LOG"
log "=============================================================="
exit "$RC"
