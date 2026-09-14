#!/bin/bash
# ---------------------------------------------------------------------------
# DepthWizard v4 — PART 1 of 2:  DATA PREPARATION  (run this on a CHEAP CPU box)
#
# Nothing here needs a GPU: it is network- and disk-bound start to finish. Run it
# on the cheapest Studio you have, then switch the machine to an H100 and run
# train_h100.sh. Re-downloading 40 GB on an H100 is the most expensive thing
# this repo can do.
#
# Four stages, in this order, because each one is what makes the next cheap:
#
#   FETCH   one bulk snapshot_download per repo into the _dl/<subdir> layout
#           prepare_data.py already expects                    (dw_fetch.py)
#   PACK    main.py --prepare with HF_HUB_OFFLINE=1 — reads the staged files off
#           local disk and never opens a socket
#   ENCODER prefetch DINOv3-SAT so the H100 run needs no network at all
#   BOUNDS  precompute each tile's 2/98 stretch bounds, so the GPU run's loader
#           never recomputes them per crop
#
# Why offline packing: prepare_data.py fetches GAMUS one file per tile, three
# files a tile — ~13 200 requests for a 4000+400 pack. That earns a CAS 429
# partway through, and a tile whose download failed is swallowed as "skip: ...".
# The v3 run wrote gamus/val with 129 of 400 tiles and still stamped it READY.
# Offline packing cannot do that (a missing file is a hard error) and section 12
# gates the stamp on the tile yield regardless.
#
# Everything lands on the Lightning persistent disk:
#   <root>/DepthWizard/            git checkout                     ~50 MB
#   <root>/DepthWizard-data/       packed memmap shards             ~35 GB
#   <root>/DepthWizard-data/_dl/   staged raw files    (wiped after a clean pack)
#   <root>/hf_home/hub/            DINOv3 encoder weights           ~1-2 GB
#   <root>/DepthWizard-data/.dw_data_ready_v4   the "GPU may start" stamp
#
# REUSING v3's SHARDS: v4's ShardWriter/PackedStore and its GAMUS, SynRS3D and
# GeoNRW packers are byte-identical to v3's, so a DepthWizard-data written by
# v3's prepare_data.sh is directly usable. This script detects that and only
# stamps it, rather than repacking 35 GB for nothing. `--reprepare` overrides.
#
# Usage (from the persistent-storage root, i.e. ~ on Lightning):
#   sh prepare_data.sh                     # fetch + pack + encoder + bounds
#   sh prepare_data.sh --background        # detach, return the shell
#   sh prepare_data.sh --plan              # show repos + GiB, download nothing
#   sh prepare_data.sh --datasets gamus    # one source
#   sh prepare_data.sh --datasets all      # gamus,synrs3d,geonrw
#   sh prepare_data.sh --india-dir ~/tiles # + Indian imagery (mean-teacher)
#   sh prepare_data.sh --fetch-only        # stage raw files, don't pack
#   sh prepare_data.sh --skip-fetch        # pack from what is already in _dl
#   sh prepare_data.sh --fetch-workers 16  # more parallelism (8 is the default)
#   sh prepare_data.sh --keep-dl           # keep _dl after a clean pack
#   sh prepare_data.sh --reprepare         # repack from scratch
#   sh prepare_data.sh --verify-only       # re-check yields, rewrite the stamp
#   HF_TOKEN=hf_xxx sh prepare_data.sh     # gated DINOv3-SAT + GAMUS
#
# Pack sizing (forwarded to prepare_data.py):
#   DW_GAMUS_TRAIN=4000  DW_GAMUS_VAL=400  DW_SYNRS3D_ARCHIVES=2  DW_GEONRW_MAX=2500
# Fetch sizing:
#   DW_FETCH_MAX_GIB=250  DW_GAMUS_REPO=org/name  DW_FETCH_WORKERS=8
# Yield gate:
#   DW_MIN_YIELD=0.75     # refuse the READY stamp below this fraction of tiles
# ---------------------------------------------------------------------------

if [ -z "${BASH_VERSION:-}" ]; then exec /usr/bin/env bash "$0" "$@"; fi

set -uo pipefail

# ===========================================================================
# 0. Flags
# ===========================================================================
DO_INSTALL=1
DO_FETCH=1
DO_PREPARE=1
DO_PREFETCH=1
DO_BOUNDS=1
REINSTALL=0
REPREPARE=0
BACKGROUND=0
DO_CLEAN=1
KEEP_DL=0
VERIFY_ONLY=0
PLAN_ONLY=0
FETCH_WORKERS="${DW_FETCH_WORKERS:-8}"
DATASETS_CLI=""
INDIA_DIR="${DW_INDIA_DIR:-}"
EXTRA_ARGS=()

prev=""
for arg in "$@"; do
    case "$arg" in
        --install-only)  DO_FETCH=0; DO_PREPARE=0; DO_PREFETCH=0; DO_BOUNDS=0 ;;
        --skip-install)  DO_INSTALL=0 ;;
        --reinstall)     REINSTALL=1 ;;
        --fetch-only)    DO_PREPARE=0; DO_PREFETCH=0; DO_BOUNDS=0 ;;
        --skip-fetch)    DO_FETCH=0 ;;
        --prepare-only)  DO_FETCH=0; DO_PREFETCH=0 ;;
        --prefetch-only) DO_FETCH=0; DO_PREPARE=0; DO_BOUNDS=0 ;;
        --skip-prefetch) DO_PREFETCH=0 ;;
        --skip-bounds)   DO_BOUNDS=0 ;;
        --plan)          PLAN_ONLY=1; DO_PREPARE=0; DO_PREFETCH=0; DO_BOUNDS=0 ;;
        --keep-dl)       KEEP_DL=1 ;;
        --reprepare|--force) REPREPARE=1 ;;
        --verify-only)   VERIFY_ONLY=1; DO_INSTALL=0; DO_FETCH=0; DO_PREPARE=0; DO_PREFETCH=0 ;;
        --no-clean)      DO_CLEAN=0 ;;
        --background)    BACKGROUND=1 ;;
        --datasets=*)    DATASETS_CLI="${arg#--datasets=}" ;;
        --india-dir=*|--india_dir=*) INDIA_DIR="${arg#*=}" ;;
        --fetch-workers=*) FETCH_WORKERS="${arg#--fetch-workers=}" ;;
        --datasets|--fetch-workers|--india-dir|--india_dir) ;;   # values below
        *)               case "$prev" in
                             --datasets)          DATASETS_CLI="$arg" ;;
                             --fetch-workers)     FETCH_WORKERS="$arg" ;;
                             --india-dir|--india_dir) INDIA_DIR="$arg" ;;
                             *)                   EXTRA_ARGS+=("$arg") ;;
                         esac ;;
    esac
    prev="$arg"
done

# ===========================================================================
# 1. Paths
# ===========================================================================
SELF_DIR="$(cd "$(dirname "$0")" 2>/dev/null && pwd || echo "$PWD")"

if [ -d "/teamspace/studios/this_studio" ]; then
    WORK_ROOT="/teamspace/studios/this_studio"
else
    WORK_ROOT="${DW_WORK_ROOT:-$HOME}"
fi

# This script lives in shell_scripts/v4/, so the checkout is two levels up.
if   [ -d "$SELF_DIR/FineTunning/v4" ];          then REPO_DIR="$SELF_DIR"; IN_PLACE=1
elif [ -d "$SELF_DIR/../../FineTunning/v4" ];    then REPO_DIR="$(cd "$SELF_DIR/../.." && pwd)"; IN_PLACE=1
elif [ -d "$SELF_DIR/../FineTunning/v4" ];       then REPO_DIR="$(cd "$SELF_DIR/.." && pwd)"; IN_PLACE=1
else REPO_DIR="$WORK_ROOT/DepthWizard"; IN_PLACE=0; fi
V4_DIR="$REPO_DIR/FineTunning/v4"

DATA_ROOT="${DW_DATA_ROOT:-$WORK_ROOT/DepthWizard-data}"
HF_HOME_DIR="${HF_HOME:-$WORK_ROOT/hf_home}"
LOG_DIR="$WORK_ROOT/DepthWizard-results/logs"
READY_STAMP="$DATA_ROOT/.dw_data_ready_v4"
DEPS_STAMP="$WORK_ROOT/.dw_v4_deps_ok"
RUN_ID="$(date +%Y%m%d_%H%M%S)"

FETCHER=""
for c in "$SELF_DIR/dw_fetch.py" "$REPO_DIR/shell_scripts/v4/dw_fetch.py" \
         "$WORK_ROOT/dw_fetch.py"; do
    [ -f "$c" ] && { FETCHER="$c"; break; }
done

mkdir -p "$DATA_ROOT" "$HF_HOME_DIR" "$LOG_DIR"
BOOT_LOG="$LOG_DIR/prepare_v4_${RUN_ID}.log"

if [ "$BACKGROUND" = "1" ]; then
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
log "DepthWizard v4 — PREPARE (data only, no GPU needed)  ${RUN_ID}"
log "  work root : $WORK_ROOT"
log "  repo      : $REPO_DIR$([ "$IN_PLACE" = 1 ] && echo '  (in-place checkout)')"
log "  data      : $DATA_ROOT   (persisted — this is the whole point)"
log "  hf cache  : $HF_HOME_DIR (persisted — encoder weights)"
log "  fetcher   : ${FETCHER:-<missing dw_fetch.py — falling back to per-file streaming>}"
log "=============================================================="

# ===========================================================================
# 2. Disk helpers
# ===========================================================================
avail_gib() { df -P -BG "${1:-$WORK_ROOT}" 2>/dev/null | awk 'NR==2 {gsub(/G/,"",$4); print $4+0}'; }

disk_report() {
    log "--- disk ---"
    df -h "$WORK_ROOT" | sed 's/^/[dw]   /'
    for d in "$DATA_ROOT" "$DATA_ROOT/_dl" "$HF_HOME_DIR" "$REPO_DIR"; do
        [ -d "$d" ] && printf '[dw]   %6s  %s\n' "$(du -sh "$d" 2>/dev/null | cut -f1)" "$d"
    done
}

safe_rm() {
    local p="$1"
    [ -n "$p" ] || return 0
    [ -e "$p" ] || return 0
    case "$p" in
        /|/home|/root|/teamspace|/teamspace/studios|"$WORK_ROOT"|"$HOME"|"$DATA_ROOT")
            warn "refusing to delete $p"; return 0 ;;
        "$WORK_ROOT"/*|"$HOME"/*|/tmp/*) ;;
        *) warn "refusing to delete outside the work root: $p"; return 0 ;;
    esac
    rm -rf -- "$p" 2>/dev/null || true
}

# ===========================================================================
# 3. Environment — Xet, not the retired hf_transfer
# ===========================================================================
export HF_HOME="$HF_HOME_DIR"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
export DW_DATA_ROOT="$DATA_ROOT"
export PIP_CACHE_DIR="$WORK_ROOT/.cache/pip"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export TORCH_HOME="$WORK_ROOT/.cache/torch"
export XDG_CACHE_HOME="$WORK_ROOT/.cache"
export MPLCONFIGDIR="$WORK_ROOT/.cache/mpl"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "$PIP_CACHE_DIR" "$MPLCONFIGDIR"

if [ -z "${HF_TOKEN:-}" ]; then
    for f in "$WORK_ROOT/.hf_token" "$HOME/.hf_token" "$REPO_DIR/.hf_token"; do
        if [ -r "$f" ]; then HF_TOKEN="$(tr -d ' \t\r\n' < "$f")"; break; fi
    done
fi
HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-${HUGGINGFACE_TOKEN:-}}}"
export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
[ -n "$HF_TOKEN" ] || warn "no HF_TOKEN — gated DINOv3-SAT / GAMUS downloads will fail.
       Set it in the Studio env, or:  echo hf_xxx > $WORK_ROOT/.hf_token"

PY="${DW_PYTHON:-}"
[ -n "$PY" ] || PY="$(command -v python || command -v python3)"
[ -n "$PY" ] || die "no python interpreter found"

# ===========================================================================
# 4. Clone / pull
# ===========================================================================
GH_REPO="github.com/akashch1512/DepthWizard.git"
GH_TOKEN="${DW_GITHUB_TOKEN:-}"
if [ -z "$GH_TOKEN" ]; then
    for f in "$WORK_ROOT/.gh_token" "$HOME/.gh_token"; do
        if [ -r "$f" ]; then GH_TOKEN="$(tr -d ' \t\r\n' < "$f")"; break; fi
    done
fi
if [ -n "$GH_TOKEN" ]; then GH_URL="https://x-access-token:${GH_TOKEN}@${GH_REPO}"
else                        GH_URL="https://${GH_REPO}"; fi

if [ "$IN_PLACE" = "1" ]; then
    log "using the checkout this script lives in — skipping clone"
elif [ -d "$REPO_DIR/.git" ]; then
    log "pulling latest main…"
    if git -C "$REPO_DIR" fetch --quiet "$GH_URL" main; then
        if [ "${DW_GIT_RESET:-0}" = "1" ]; then
            git -C "$REPO_DIR" reset --hard --quiet FETCH_HEAD
        elif ! git -C "$REPO_DIR" merge --ff-only --quiet FETCH_HEAD 2>/dev/null; then
            warn "cannot fast-forward (local commits/edits) — using the checkout as-is.
       Force with:  DW_GIT_RESET=1 sh prepare_data.sh"
        fi
    else
        warn "fetch failed — continuing with the checkout on disk"
    fi
else
    log "cloning DepthWizard…"
    git clone --depth 1 --quiet "$GH_URL" "$REPO_DIR" || die "clone failed"
    git -C "$REPO_DIR" remote set-url origin "https://${GH_REPO}"
fi

[ -d "$V4_DIR" ] || die "expected $V4_DIR — wrong repo layout?"
log "HEAD: $(git -C "$REPO_DIR" log --oneline -1 2>/dev/null || echo n/a)"

# ===========================================================================
# 5. Dependencies
# ===========================================================================
# Only what the pack needs. torch comes with the image; v4's requirements also
# carry fastapi/onnx for the serving half, which this stage never touches — but
# installing the file as written keeps the two Studios' environments identical,
# and pip is not the slow part of this script.
REQ="$V4_DIR/requirements.txt"
[ -f "$REQ" ] || die "missing $REQ"
REQ_HASH="$(md5sum "$REQ" | cut -d' ' -f1)"

deps_present() {
    "$PY" - >/dev/null 2>&1 <<'PYDEPS'
import importlib.util, sys
mods = ("torch", "transformers", "huggingface_hub", "h5py", "tifffile",
        "rasterio", "safetensors", "scipy", "numpy", "PIL", "cv2")
sys.exit(1 if [m for m in mods if not importlib.util.find_spec(m)] else 0)
PYDEPS
}

install_deps() {
    log "installing dependencies…"
    "$PY" -m pip install --no-input -q --upgrade pip >/dev/null 2>&1 || true
    "$PY" -m pip install --no-input -q -r "$REQ" || die "pip install -r $REQ failed"
    # Xet is the transport now; hf_xet is what actually moves the bytes fast.
    "$PY" -m pip install --no-input -q --upgrade "huggingface_hub[hf_xet]" >/dev/null 2>&1 \
        || warn "could not install hf_xet — downloads will use plain HTTPS"
    echo "$REQ_HASH" > "$DEPS_STAMP"
    "$PY" -m pip cache purge >/dev/null 2>&1 || true
}

if [ "$DO_INSTALL" = "1" ]; then
    if [ "$REINSTALL" = "0" ] && [ -f "$DEPS_STAMP" ] \
       && grep -qx "$REQ_HASH" "$DEPS_STAMP" && deps_present; then
        log "dependencies already satisfied for this requirements file — skipping"
    else
        install_deps
    fi
fi

# hf_transfer was retired in favour of Xet.
unset HF_HUB_ENABLE_HF_TRANSFER
if "$PY" -c "import hf_xet" >/dev/null 2>&1; then
    export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
    export HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-16}"
    log "Xet high-performance transfer on (range gets: $HF_XET_NUM_CONCURRENT_RANGE_GETS)"
else
    warn "hf_xet not installed — falling back to plain HTTPS.  pip install hf_xet"
fi

if [ "$VERIFY_ONLY" = "0" ]; then
    deps_present || die "runtime is incomplete — re-run with --reinstall"
    "$PY" - <<'PYCHECK'
import huggingface_hub, torch, transformers
print(f"[dw]   torch {torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"[dw]   transformers {transformers.__version__}  (DINOv3 needs >= 4.56)")
print(f"[dw]   huggingface_hub {huggingface_hub.__version__}")
try:
    import cv2
    print(f"[dw]   opencv {cv2.__version__}  (SIMD resize in the training loader)")
except Exception:
    print("[dw]   WARN: no opencv — the loader falls back to PIL's scalar resampler")
if not torch.cuda.is_available():
    print("[dw]   (no GPU here — correct: this stage is network bound)")
PYCHECK
fi

# ===========================================================================
# 6. Which datasets
# ===========================================================================
DATASETS="${DATASETS_CLI:-${DW_PREPARE_DATASETS:-gamus,synrs3d}}"
[ "$DATASETS" = "all" ] && DATASETS="gamus,synrs3d,geonrw"

store_ok() { [ -f "$DATA_ROOT/$1/index.json" ]; }
source_prepared() {
    case "$1" in
        gamus)            store_ok gamus/train && store_ok gamus/val ;;
        synrs3d)          store_ok synrs3d/train ;;
        geonrw)           store_ok geonrw/train ;;
        india_unlabeled)  store_ok india_unlabeled/train ;;
        india_labeled)    store_ok india_labeled/train ;;
        *)                return 1 ;;
    esac
}

NEED_PREP=""
for d in ${DATASETS//,/ }; do
    source_prepared "$d" || NEED_PREP="$NEED_PREP,$d"
done
NEED_PREP="${NEED_PREP#,}"

# v3's stores are v4's stores — same ShardWriter, same index.json, same packers.
if [ -z "$NEED_PREP" ] && [ ! -f "$READY_STAMP" ] && [ -f "$DATA_ROOT/.dw_data_ready" ]; then
    log "every source in [$DATASETS] is already packed, by v3's prepare_data.sh.
       v4's ShardWriter/PackedStore and its packers are unchanged from v3, so
       these shards are used as-is rather than repacked. (--reprepare overrides.)"
fi

case ",$DATASETS," in
    *,gamus,*) ;;
    *) warn "GAMUS is not in --datasets — it is the only source with a dedicated
       val split, so training will validate on a slice of train instead." ;;
esac

GAMUS_TRAIN="${DW_GAMUS_TRAIN:-4000}"; GAMUS_VAL="${DW_GAMUS_VAL:-400}"
SYN_ARCHIVES="${DW_SYNRS3D_ARCHIVES:-2}"; GEONRW_MAX="${DW_GEONRW_MAX:-2500}"
export DW_SYNRS3D_ARCHIVES="$SYN_ARCHIVES"

# ===========================================================================
# 7. Cleanup + traps
# ===========================================================================
cleanup_scratch() {
    local before after
    before="$(avail_gib "$WORK_ROOT")"; before="${before:-0}"
    log "cleaning scratch…"
    "$PY" -m pip cache purge >/dev/null 2>&1 || true
    safe_rm "$PIP_CACHE_DIR"
    find "$DATA_ROOT" "$HF_HOME_DIR" -name '*.incomplete' -delete 2>/dev/null
    find "$HF_HOME_DIR" -type d -name '.locks' -exec rm -rf {} + 2>/dev/null
    safe_rm "$HF_HOME_DIR/hub/datasets--"*
    find "$REPO_DIR" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null
    sync 2>/dev/null || true
    after="$(avail_gib "$WORK_ROOT")"; after="${after:-0}"
    log "scratch cleared — reclaimed ~$(( after - before )) GiB (now ${after} GiB free)"
}

on_interrupt() {
    warn "interrupted — keeping BOTH the completed shards and the staged _dl files,
       so a re-run resumes instead of re-downloading"
    stop_disk_guard 2>/dev/null || true
    disk_report
    exit 130
}
trap on_interrupt INT TERM

# ===========================================================================
# 8. Disk watchdog
# ===========================================================================
GUARD_PID=""
GUARD_LOG="$LOG_DIR/disk_guard_prepare_v4_${RUN_ID}.log"
start_disk_guard() {
    setsid bash -c '
        while true; do
            sleep 300
            free=$(df -P -BG "'"$WORK_ROOT"'" 2>/dev/null | awk "NR==2 {gsub(/G/,\"\",\$4); print \$4+0}")
            free=${free:-999}
            echo "[dw-guard $(date +%H:%M:%S)] free=${free}GiB data=$(du -sh "'"$DATA_ROOT"'" 2>/dev/null | cut -f1) staging=$(du -sh "'"$DATA_ROOT"'/_dl" 2>/dev/null | cut -f1)"
            if [ "$free" -lt 12 ]; then
                echo "[dw-guard] LOW DISK (${free}GiB) — dropping partial downloads"
                find "'"$DATA_ROOT"'" "'"$HF_HOME_DIR"'" -name "*.incomplete" -delete 2>/dev/null
                find "'"$HF_HOME_DIR"'" -type d -name ".locks" -exec rm -rf {} + 2>/dev/null
                rm -rf "'"$PIP_CACHE_DIR"'" 2>/dev/null
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

cd "$V4_DIR" || die "cannot cd $V4_DIR"

# ===========================================================================
# 9. FETCH — one bulk snapshot per repo, into the _dl layout pack expects
# ===========================================================================
FETCH_RC=0
FETCH_LIST=""
for d in ${NEED_PREP//,/ }; do
    case "$d" in gamus|synrs3d|geonrw) FETCH_LIST="$FETCH_LIST,$d" ;; esac
done
[ "$REPREPARE" = "1" ] && {
    FETCH_LIST=""
    for d in ${DATASETS//,/ }; do
        case "$d" in gamus|synrs3d|geonrw) FETCH_LIST="$FETCH_LIST,$d" ;; esac
    done
}
FETCH_LIST="${FETCH_LIST#,}"

if [ "$PLAN_ONLY" = "1" ]; then
    [ -n "$FETCHER" ] || die "--plan needs dw_fetch.py next to this script"
    PLAN_LIST=""
    for d in ${DATASETS//,/ }; do
        case "$d" in gamus|synrs3d|geonrw) PLAN_LIST="$PLAN_LIST,$d" ;; esac
    done
    "$PY" -u "$FETCHER" --repo-root "$V4_DIR" --data-root "$DATA_ROOT" \
          --datasets "${PLAN_LIST#,}" --workers "$FETCH_WORKERS" --list
    exit $?
fi

if [ "$DO_FETCH" = "1" ] && [ -n "$FETCH_LIST" ] && [ -n "$FETCHER" ]; then
    log "bulk-fetching [$FETCH_LIST] with $FETCH_WORKERS workers…"
    start_disk_guard
    F0=$SECONDS
    "$PY" -u "$FETCHER" --repo-root "$V4_DIR" --data-root "$DATA_ROOT" \
          --datasets "$FETCH_LIST" --workers "$FETCH_WORKERS"
    FETCH_RC=$?
    stop_disk_guard
    log "fetch exited rc=$FETCH_RC after $(( (SECONDS - F0) / 60 )) min"
    if [ "$FETCH_RC" -ne 0 ]; then
        warn "bulk fetch incomplete — the pack step will go online for whatever is
       missing (slower, and 429-prone). Re-run this script to resume the fetch."
    fi
elif [ "$DO_FETCH" = "1" ] && [ -z "$FETCHER" ]; then
    warn "dw_fetch.py not found — skipping the bulk stage. Put it next to this
       script to avoid ~13 000 per-file requests and the 429s they earn."
elif [ "$DO_FETCH" = "1" ]; then
    log "every requested source is already packed — nothing to fetch"
fi

# ===========================================================================
# 10. PACK — offline when the staging looks complete
# ===========================================================================
PRC=0
if [ "$DO_PREPARE" = "1" ] && { [ -n "$NEED_PREP" ] || [ "$REPREPARE" = "1" ]; }; then
    TO_PREP="$DATASETS"
    [ "$REPREPARE" = "1" ] || TO_PREP="$NEED_PREP"

    NEED_GIB=0
    case ",$TO_PREP," in *,gamus,*)   NEED_GIB=$(( NEED_GIB + (GAMUS_TRAIN + GAMUS_VAL) * 63 / 10000 )) ;; esac
    case ",$TO_PREP," in *,synrs3d,*) NEED_GIB=$(( NEED_GIB + SYN_ARCHIVES * 7 )) ;; esac
    case ",$TO_PREP," in *,geonrw,*)  NEED_GIB=$(( NEED_GIB + GEONRW_MAX * 6 / 1000 )) ;; esac
    AVAIL="$(avail_gib "$WORK_ROOT")"; AVAIL="${AVAIL:-0}"
    log "packing [$TO_PREP] -> $DATA_ROOT  (~${NEED_GIB} GiB of shards, ${AVAIL} GiB free)"
    [ "$AVAIL" -lt "$(( NEED_GIB + 10 ))" ] && warn "only ${AVAIL} GiB free for
       ~${NEED_GIB} GiB of shards plus staging — the pack may run out of disk"

    PREP_ARGS=(--data_root "$DATA_ROOT" --datasets "$TO_PREP"
               --gamus_train "$GAMUS_TRAIN" --gamus_val "$GAMUS_VAL"
               --synrs3d_archives "$SYN_ARCHIVES" --geonrw_max "$GEONRW_MAX"
               --workers "${DW_PREP_WORKERS:-12}")
    [ -n "$INDIA_DIR" ] && PREP_ARGS+=(--india_dir "$INDIA_DIR")
    [ "$REPREPARE" = "1" ] && PREP_ARGS+=(--force)

    start_disk_guard
    P0=$SECONDS

    # Offline first: the staged files are right there, and a missing one should
    # be a loud failure rather than 271 quiet "skip" lines.  v4's prepare_data.py
    # falls back to listing the staging directory when the Hub API is
    # unreachable, which is what makes HF_HUB_OFFLINE survivable at all.
    OFFLINE_OK=0
    if [ "$FETCH_RC" -eq 0 ] && [ "$DO_FETCH" = "1" ] && [ -n "$FETCHER" ] \
       && [ "${DW_PREPARE_OFFLINE:-1}" = "1" ]; then
        log "packing offline (HF_HUB_OFFLINE=1) from $DATA_ROOT/_dl"
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
            "$PY" -u main.py --skip-install --prepare "${PREP_ARGS[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
        PRC=$?
        if [ "$PRC" -eq 0 ]; then
            OFFLINE_OK=1
        else
            warn "offline pack failed rc=$PRC — retrying with the network on"
        fi
    fi

    if [ "$OFFLINE_OK" = "0" ]; then
        for attempt in 1 2 3; do
            "$PY" -u main.py --skip-install --prepare "${PREP_ARGS[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
            PRC=$?
            [ "$PRC" -eq 0 ] && break
            [ "$attempt" = "3" ] && break
            wait_s=$(( 60 * attempt ))
            warn "pack attempt $attempt failed rc=$PRC — sleeping ${wait_s}s (429s clear
       on their own; completed splits are skipped on the retry)"
            sleep "$wait_s"
        done
    fi

    stop_disk_guard
    log "pack exited rc=$PRC after $(( (SECONDS - P0) / 60 )) min"

    if [ "$PRC" -ne 0 ]; then
        warn "pack failed. Stores that DID complete (a split only gets an index.json
       when it is whole, so these are safe to keep and resume from):"
        find "$DATA_ROOT" -name index.json -printf '       %h\n' 2>/dev/null
        warn "staged files kept in $DATA_ROOT/_dl — the retry will not re-download"
        disk_report
        exit "$PRC"
    fi
elif [ "$DO_PREPARE" = "1" ]; then
    log "all requested sources already packed in $DATA_ROOT — nothing to do"
fi

# ===========================================================================
# 11. PREFETCH — the DINOv3 encoder, so the H100 run needs no network at all
# ===========================================================================
if [ "$DO_PREFETCH" = "1" ]; then
    log "prefetching encoder weights into $HF_HOME_DIR …"
    DW_ENCODER_ID="${DW_ENCODER_ID:-}" "$PY" - "$V4_DIR" <<'PYPULL' || warn "encoder prefetch
       failed — the GPU run will pull it at startup (needs network + HF_TOKEN)"
import os, sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
ids = {i.strip() for i in os.environ.get("DW_ENCODER_ID", "").split(",") if i.strip()}

# Ask config.py, rather than regexing repo ids out of the source: the encoder is
# a config field, and a scrape that silently matches the wrong string caches the
# wrong weights and the GPU box discovers it at minute one.
if not ids:
    try:
        from config import Config
        ids.add(Config().encoder_model_id)
    except Exception as e:
        print(f"[dw]   cannot read encoder id from config.py: {e}")

if not ids:
    print("[dw]   no encoder repo id — skipping prefetch")
    raise SystemExit(0)

from huggingface_hub import snapshot_download
tok = os.environ.get("HF_TOKEN") or None
rc = 0
for rid in sorted(ids):
    try:
        p = snapshot_download(
            repo_id=rid, repo_type="model", token=tok, max_workers=8,
            allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "*.bin"],
        )
        n = sum(1 for _ in Path(p).rglob("*.safetensors")) + \
            sum(1 for _ in Path(p).rglob("*.bin"))
        print(f"[dw]   cached {rid}  ({n} weight file(s))")
        if n == 0:
            print(f"[dw]   WARN: {rid} cached no weights — the GPU run will go online")
            rc = 1
    except Exception as e:
        print(f"[dw]   FAILED {rid}: {type(e).__name__}: {e}")
        rc = 1
raise SystemExit(rc)
PYPULL
fi

# ===========================================================================
# 12. BOUNDS — precompute the per-tile radiometric stretch bounds
# ===========================================================================
# The 2/98 percentile bounds are a property of the scene, not of the crop, but a
# naive loader recomputes them inside every __getitem__: a 3 MB full-tile read
# plus a 3-channel histogram, ~13 ms of an ~82 ms sample, in every worker
# independently. Doing it here writes one small `stretch_bounds_2_98.npy` next
# to each store's shards, which the GPU run just memory-maps. ~90 s of CPU here
# buys that back for every crop of every epoch.
#
# v4's build_loaders() also calls prime_stretch_bounds() and it is idempotent, so
# this is belt and braces — but it is the difference between the H100 paying the
# one-off cost at startup and not paying it at all.
if [ "$DO_BOUNDS" = "1" ] && { [ "$DO_PREPARE" = "1" ] || [ "$VERIFY_ONLY" = "1" ]; }; then
    log "precomputing per-tile stretch bounds (so the GPU run never does)…"
    "$PY" - "$DATA_ROOT" "${DW_PREP_WORKERS:-12}" "$V4_DIR" <<'PYBOUNDS' || warn "stretch-bound
       precompute failed — training still works, it just primes them at startup"
import sys, time
from pathlib import Path

sys.path.insert(0, sys.argv[3])
from dwdata.packed import PackedStore, store_exists

root, workers = Path(sys.argv[1]), int(sys.argv[2])
found = 0
for idx in sorted(root.glob("*/*/index.json")):
    d = idx.parent
    if not store_exists(d):
        continue
    found += 1
    st = PackedStore(d)
    t0 = time.time()
    ok = st.prime_stretch_bounds(2.0, 98.0, workers=workers)
    key = f"{d.parent.name}/{d.name}"
    print(f"[dw]   {key:<18} {len(st):>6} tiles  "
          f"{'ok' if ok else 'FAILED'}  {time.time() - t0:.0f}s")
if not found:
    print("[dw]   no packed stores found — nothing to prime")
PYBOUNDS
fi

# ===========================================================================
# 13. VERIFY — tile yield, not just "index.json exists"
# ===========================================================================
# v3 packed gamus/val with 129 of 400 tiles and still wrote a READY stamp. An
# index.json is necessary, not sufficient.
log "--- packed stores ---"
"$PY" - "$DATA_ROOT" "$GAMUS_TRAIN" "$GAMUS_VAL" "$GEONRW_MAX" "${DW_MIN_YIELD:-0.75}" <<'PYVERIFY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
want = {
    "gamus/train": int(sys.argv[2]),
    "gamus/val": int(sys.argv[3]),
    "geonrw/train": int(sys.argv[4]),
}
min_yield = float(sys.argv[5])
low = []

for idx in sorted(root.glob("*/*/index.json")):
    i = json.loads(idx.read_text())
    key = f"{idx.parent.parent.name}/{idx.parent.name}"
    gib = sum(f.stat().st_size for f in idx.parent.glob("*.npy")) / 1024**3
    bounds = "bounds" if list(idx.parent.glob("stretch_bounds_*.npy")) else "  -   "
    n, req = i["n"], want.get(key)
    frac = f"{n/req:5.0%}" if req else "    -"
    print(f"[dw]   {key:<18} {n:>6} tiles  {frac} of requested  "
          f"@ {i['tile_px']}px / {i['gsd_m']} m  ({gib:5.1f} GiB)  {bounds}")
    if req and n < req * min_yield:
        low.append((key, n, req))

if low:
    print("[dw]   ---")
    for key, n, req in low:
        print(f"[dw]   LOW YIELD: {key} packed {n} of {req} requested tiles")
    print(f"[dw]   Under DW_MIN_YIELD={min_yield:g} — almost always a partly-staged source:")
    print("[dw]     sh prepare_data.sh --reprepare   (re-fetches, then repacks)")
    sys.exit(3)
PYVERIFY
YIELD_RC=$?

MISSING=""
for d in ${DATASETS//,/ }; do
    source_prepared "$d" || MISSING="$MISSING $d"
done

if [ -n "$MISSING" ] || [ "$YIELD_RC" -ne 0 ]; then
    safe_rm "$READY_STAMP"
    [ -n "$MISSING" ] && warn "these sources are NOT packed:$MISSING
       (a gated repo without HF_TOKEN is the usual cause)"
    [ "$YIELD_RC" -ne 0 ] && warn "refusing the READY stamp — see LOW YIELD above.
       train_h100.sh will not start against half a val split."
    warn "staged files kept in $DATA_ROOT/_dl so the retry is a repack, not a redownload"
    disk_report
    exit 1
fi

{
    echo "ready_at    : $(date -Is)"
    echo "run_id      : $RUN_ID"
    echo "version     : v4"
    echo "datasets    : $DATASETS"
    echo "data_root   : $DATA_ROOT"
    echo "hf_home     : $HF_HOME_DIR"
    echo "commit      : $(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo n/a)"
    echo "sizing      : gamus_train=$GAMUS_TRAIN gamus_val=$GAMUS_VAL synrs3d_archives=$SYN_ARCHIVES geonrw_max=$GEONRW_MAX"
    echo "fetch       : bulk snapshot (dw_fetch.py) rc=$FETCH_RC"
    echo "host        : $(hostname)"
    echo
    echo "--- stores ---"
    find "$DATA_ROOT" -name index.json -printf '%h\n' 2>/dev/null | sort
} > "$READY_STAMP"

if [ "$KEEP_DL" = "1" ]; then
    log "--keep-dl: staging kept at $DATA_ROOT/_dl ($(du -sh "$DATA_ROOT/_dl" 2>/dev/null | cut -f1))"
else
    log "dropping staging $DATA_ROOT/_dl ($(du -sh "$DATA_ROOT/_dl" 2>/dev/null | cut -f1))"
    safe_rm "$DATA_ROOT/_dl"
fi
[ "$DO_CLEAN" = "1" ] && cleanup_scratch

log "=============================================================="
log "DATA READY (v4)"
log "  shards : $DATA_ROOT  ($(du -sh "$DATA_ROOT" 2>/dev/null | cut -f1))"
log "  encoder: $HF_HOME_DIR ($(du -sh "$HF_HOME_DIR" 2>/dev/null | cut -f1))"
log "  stamp  : $READY_STAMP"
log ""
log "Next: switch the Studio to an H100 and run"
log "  sh shell_scripts/v4/train_h100.sh --background"
disk_report
log "prepare log: $BOOT_LOG"
log "=============================================================="
exit 0
