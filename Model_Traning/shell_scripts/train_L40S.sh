#!/bin/bash
# ---------------------------------------------------------------------------
# DepthWizard v3 — PART 2 of 2:  TRAIN  (run this on the L40S Studio)
#
# ! fast_load
# DepthWizard-results/v3/best.pt
# DepthWizard-results/v3/last.pt
# DepthWizard-results/results_v3.zip
#
# Assumes prepare_data.sh already staged everything on the persistent disk. It
# does NOT download datasets — it refuses to start if the shards are missing,
# because re-downloading 45 GB on a GPU box is the most expensive thing this
# repo can do.
#
#   reads   <root>/DepthWizard-data/       packed memmap shards + stretch bounds
#   reads   <root>/hf_home/hub/            DINOv3 encoder weights
#   writes  <root>/DepthWizard-results/v3/ checkpoints, metrics, zip
#
# ---- Why the L40S was only half busy, and what is done about it -----------
# The symptom was steps moving with VRAM pinned near 20 GiB of 48 and GPU util
# in the fifties. That is two separate problems wearing one coat:
#
#   1. THE BATCH WAS NEVER THE MEASURED ONE.  The tuner picked it from a VRAM
#      bracket whose top tier was ">= 40 GiB -> 16", and 48 falls in it. Sixteen
#      512 px crops through an unfrozen ViT-L is ~20 GiB — the flat line exactly.
#      It now runs `tools/probe_batch.py`, which does a real forward / backward /
#      AdamW step on this card with the encoder unfrozen and returns the largest
#      batch that fits under DW_VRAM_HEADROOM (default 88%). Cached per card.
#      Two further flags were being dropped outright because the tuner matched
#      option names exactly: `--compile` never matched `--compile_model`, and
#      `--grad_checkpoint` never matched `--grad_checkpoint_encoder`.
#
#   2. THE LOADER COULD NOT FILL IT.  A GAMUS crop cost ~82 ms of CPU, of which
#      ~13 ms recomputed a whole-tile percentile histogram (a property of the
#      scene, recomputed per crop, in every worker) and ~22 ms did photometric
#      jitter and fp32 normalisation that a GPU does for the whole batch in ~2 ms.
#      Now: prepare_data.sh persists the stretch bounds, the jitter and the
#      normalisation moved to the card (--gpu_augment), and workers hand over
#      uint8 HWC — 22 ms/crop and 2.4 MB/sample instead of 76 ms and 7.3 MB.
#      Measured: 13 -> 45 samples/s per worker on gamus, 29 -> 88 on synrs3d.
#
# `--stage-local` is now conditional: on a Lightning Studio /tmp and
# /teamspace/studios/this_studio are the same overlay mount, so the old
# unconditional 45 GB copy moved data from a disk to itself and cost ~10 min of
# L40S time for nothing. It only copies across a genuinely different filesystem,
# and otherwise warms the page cache in the background instead.
#
# If the probed batch still OOMs, the run halves it and retries (9d); nothing is
# lost, since it resumes from last.pt.
#
# Usage (from the persistent-storage root, i.e. ~ on Lightning):
#   sh train_L40S.sh --background                 # the recommended first run
#   sh train_L40S.sh --show-tuning                # resolve flags, launch nothing
#   sh train_L40S.sh --smoke                      # ~10 min sanity run
#   sh train_L40S.sh --force                      # retrain even if one finished
#   sh train_L40S.sh --no-resume                  # ignore last.pt
#   sh train_L40S.sh --datasets gamus             # subset of what is packed
#   sh train_L40S.sh --check                      # deps + tests + GPU report
#   sh train_L40S.sh --online                     # allow Hub access
#   sh train_L40S.sh --allow-prepare              # last resort: pack data here
#   sh train_L40S.sh --no-autotune                # leave sizing to config.py
#
# Environment knobs:
#   DW_BATCH=32              skip the probe, use this batch
#   DW_VRAM_HEADROOM=0.92    let the probe fill more of the card
#   DW_EFFECTIVE_BATCH=48    batch x grad_accum target
#   DW_LOADER_WORKERS=14     override the worker count
#   DW_WARM_CACHE=0          skip the background page-cache warm
#   DW_COMPILE=1             torch.compile the model (slow first epoch, ~1.2x after)
# Any unrecognised flag is forwarded verbatim to train.py, and the caller's
# value always beats the tuner's.
# ---------------------------------------------------------------------------

if [ -z "${BASH_VERSION:-}" ]; then exec /usr/bin/env bash "$0" "$@"; fi

set -uo pipefail

# ===========================================================================
# 0. Wrapper flags  (everything else is forwarded to train.py)
# ===========================================================================
DO_INSTALL=1
DO_CHECK=0
DO_CLEAN=1
FORCE=0
BACKGROUND=0
AUTO_RESUME=1
REINSTALL=0
ALLOW_PREPARE=0
OFFLINE=1
WIPE_CACHE=0
AUTOTUNE="${DW_AUTOTUNE:-1}"
STAGE_LOCAL="${DW_STAGE_LOCAL:-0}"
SHOW_TUNING=0
TRAIN_ARGS=()
AUTO_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --skip-install)   DO_INSTALL=0 ;;
        --reinstall)      REINSTALL=1 ;;
        --check)          DO_CHECK=1 ;;
        --no-clean)       DO_CLEAN=0 ;;
        --force|--fresh)  FORCE=1 ;;
        --background)     BACKGROUND=1 ;;
        --no-resume)      AUTO_RESUME=0 ;;
        --allow-prepare)  ALLOW_PREPARE=1 ;;
        --online)         OFFLINE=0 ;;
        --wipe-cache)     WIPE_CACHE=1 ;;
        --no-autotune)    AUTOTUNE=0 ;;
        --stage-local)    STAGE_LOCAL=1 ;;
        --no-stage-local) STAGE_LOCAL=0 ;;
        --show-tuning)    SHOW_TUNING=1 ;;
        *)                TRAIN_ARGS+=("$arg") ;;
    esac
done

# ===========================================================================
# 1. Paths — the layout prepare_data.sh wrote
# ===========================================================================
SELF_DIR="$(cd "$(dirname "$0")" 2>/dev/null && pwd || echo "$PWD")"

if [ -d "/teamspace/studios/this_studio" ]; then
    WORK_ROOT="/teamspace/studios/this_studio"
else
    WORK_ROOT="${DW_WORK_ROOT:-$HOME}"
fi

if [ -d "$SELF_DIR/FineTunning/v3" ]; then
    REPO_DIR="$SELF_DIR"; IN_PLACE=1
elif [ -d "$SELF_DIR/../../FineTunning/v3" ]; then
    REPO_DIR="$(cd "$SELF_DIR/../.." && pwd)"; IN_PLACE=1
else
    REPO_DIR="$WORK_ROOT/DepthWizard"; IN_PLACE=0
fi
V3_DIR="$REPO_DIR/FineTunning/v3"

SMOKE=0
case " ${TRAIN_ARGS[*]-} " in *" --smoke"*) SMOKE=1 ;; esac

if [ "$SMOKE" = "1" ]; then
    RESULTS_DIR="${DW_OUTPUT_DIR:-$WORK_ROOT/DepthWizard-results/smoke}"
else
    RESULTS_DIR="${DW_OUTPUT_DIR:-$WORK_ROOT/DepthWizard-results/v3}"
fi
DATA_ROOT="${DW_DATA_ROOT:-$WORK_ROOT/DepthWizard-data}"
PERSIST_DATA_ROOT="$DATA_ROOT"                 # never reassigned; 9b may move DATA_ROOT
HF_HOME_DIR="${HF_HOME:-$WORK_ROOT/hf_home}"
LOG_DIR="$WORK_ROOT/DepthWizard-results/logs"
SENTINEL="$RESULTS_DIR/.dw_complete"
READY_STAMP="$PERSIST_DATA_ROOT/.dw_data_ready"
DEPS_STAMP="$WORK_ROOT/.dw_v3_deps_ok"
RUN_ID="$(date +%Y%m%d_%H%M%S)"

ZIP_PATH="$(dirname "$RESULTS_DIR")/results_$(basename "$RESULTS_DIR").zip"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
BOOT_LOG="$LOG_DIR/train_${RUN_ID}.log"

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
log "DepthWizard v3 — TRAIN (L40S)  ${RUN_ID}"
log "  work root : $WORK_ROOT"
log "  repo      : $REPO_DIR$([ "$IN_PLACE" = 1 ] && echo '  (in-place checkout)')"
log "  data      : $DATA_ROOT"
log "  results   : $RESULTS_DIR"
log "=============================================================="

# ===========================================================================
# 2. Disk helpers
# ===========================================================================
avail_gib() { df -P -BG "${1:-$WORK_ROOT}" 2>/dev/null | awk 'NR==2 {gsub(/G/,"",$4); print $4+0}'; }

disk_report() {
    log "--- disk ---"
    df -h "$WORK_ROOT" | sed 's/^/[dw]   /'
    for d in "$RESULTS_DIR" "$PERSIST_DATA_ROOT" "$HF_HOME_DIR" "$REPO_DIR"; do
        [ -d "$d" ] && printf '[dw]   %6s  %s\n' "$(du -sh "$d" 2>/dev/null | cut -f1)" "$d"
    done
}

safe_rm() {
    local p="$1"
    [ -n "$p" ] || return 0
    [ -e "$p" ] || return 0
    case "$p" in
        /|/home|/root|/teamspace|/teamspace/studios|"$WORK_ROOT"|"$HOME") warn "refusing to delete $p"; return 0 ;;
        "$PERSIST_DATA_ROOT"|"$PERSIST_DATA_ROOT"/gamus|"$PERSIST_DATA_ROOT"/synrs3d|"$PERSIST_DATA_ROOT"/geonrw)
            warn "refusing to delete packed shards: $p"; return 0 ;;
        "$WORK_ROOT"/*|"$HOME"/*|/tmp/*) ;;
        *) warn "refusing to delete outside the work root: $p"; return 0 ;;
    esac
    rm -rf -- "$p" 2>/dev/null || true
}

# ===========================================================================
# 3. Environment
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
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Hopper: TF32 everywhere, and an allocator that tolerates a big, varying batch.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NVIDIA_TF32_OVERRIDE="${NVIDIA_TF32_OVERRIDE:-1}"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export TORCH_CUDNN_V8_API_ENABLED=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-}"
unset CUBLAS_WORKSPACE_CONFIG 2>/dev/null || true
mkdir -p "$PIP_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR"

if [ -z "${HF_TOKEN:-}" ]; then
    for f in "$WORK_ROOT/.hf_token" "$HOME/.hf_token" "$REPO_DIR/.hf_token"; do
        if [ -r "$f" ]; then HF_TOKEN="$(tr -d ' \t\r\n' < "$f")"; break; fi
    done
fi
HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-${HUGGINGFACE_TOKEN:-}}}"
export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

PY="${DW_PYTHON:-}"
[ -n "$PY" ] || PY="$(command -v python || command -v python3)"
[ -n "$PY" ] || die "no python interpreter found"

# ===========================================================================
# 4. PREFLIGHT — is the data actually here?
# ===========================================================================
store_ok() { [ -f "$DATA_ROOT/$1/index.json" ]; }
source_prepared() {
    case "$1" in
        gamus)   store_ok gamus/train && store_ok gamus/val ;;
        synrs3d) store_ok synrs3d/train ;;
        geonrw)  store_ok geonrw/train ;;
        *)       return 1 ;;
    esac
}

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
if [ -z "$DATASETS" ] && [ -f "$READY_STAMP" ]; then
    DATASETS="$(awk -F': *' '/^datasets/ {print $2; exit}' "$READY_STAMP")"
fi
DATASETS="${DATASETS:-gamus,synrs3d}"
[ "$DATASETS" = "all" ] && DATASETS="gamus,synrs3d,geonrw"

MISSING_SRC=""
for d in ${DATASETS//,/ }; do
    source_prepared "$d" || MISSING_SRC="$MISSING_SRC $d"
done

if [ "$DO_CHECK" = "0" ] && [ "$SHOW_TUNING" = "0" ] && [ -n "$MISSING_SRC" ]; then
    if [ "$ALLOW_PREPARE" = "1" ]; then
        warn "--allow-prepare: packing$MISSING_SRC on the GPU box. Slow and expensive;
       prefer prepare_data.sh on a CPU Studio."
    else
        echo
        die "no packed shards for:$MISSING_SRC   (looked under $DATA_ROOT)

       This script will not download datasets on a GPU box. Run the data stage on
       a cheap CPU Studio first:

         sh prepare_data.sh --datasets $DATASETS --background

       …then come back here. To pack on this machine anyway:

         sh train_L40S.sh --allow-prepare"
    fi
elif [ "$DO_CHECK" = "0" ]; then
    log "packed shards found for [$DATASETS] — no dataset download needed"
    [ -f "$READY_STAMP" ] && sed 's/^/[dw]   /' "$READY_STAMP" | head -6
fi

if [ "$OFFLINE" = "1" ] && [ -d "$HF_HOME_DIR/hub" ] \
   && find "$HF_HOME_DIR/hub" \( -name '*.safetensors' -o -name '*.bin' \) 2>/dev/null | head -1 | grep -q .; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    log "running offline (encoder weights cached in $HF_HOME_DIR)"
elif [ "$OFFLINE" = "1" ]; then
    warn "no cached encoder weights in $HF_HOME_DIR — staying online for the first
       model pull. Fix with:  sh prepare_data.sh --prefetch-only"
fi

# ===========================================================================
# 5. Repo + dependencies
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
    if [ "${DW_GIT_PULL:-1}" = "1" ]; then
        log "pulling latest main…"
        if git -C "$REPO_DIR" fetch --quiet "$GH_URL" main; then
            if [ "${DW_GIT_RESET:-0}" = "1" ]; then
                git -C "$REPO_DIR" reset --hard --quiet FETCH_HEAD
            elif ! git -C "$REPO_DIR" merge --ff-only --quiet FETCH_HEAD 2>/dev/null; then
                warn "cannot fast-forward (local commits/edits) — training the checkout
       as-is. Force with:  DW_GIT_RESET=1 sh train_L40S.sh"
            fi
        else
            warn "fetch failed — continuing with the checkout on disk"
        fi
    fi
else
    log "cloning DepthWizard…"
    git clone --depth 1 --quiet "$GH_URL" "$REPO_DIR" || die "clone failed"
    git -C "$REPO_DIR" remote set-url origin "https://${GH_REPO}"
fi

[ -d "$V3_DIR" ] || die "expected $V3_DIR — wrong repo layout?"
log "HEAD: $(git -C "$REPO_DIR" log --oneline -1 2>/dev/null || echo n/a)"

REQ="$V3_DIR/requirements.txt"
[ -f "$REQ" ] || die "missing $REQ"
REQ_HASH="$(md5sum "$REQ" | cut -d' ' -f1)"

deps_present() {
    "$PY" - >/dev/null 2>&1 <<'PYDEPS'
import importlib.util, sys
mods = ("torch", "transformers", "huggingface_hub", "h5py", "tifffile",
        "rasterio", "safetensors", "scipy", "numpy", "PIL")
sys.exit(1 if [m for m in mods if not importlib.util.find_spec(m)] else 0)
PYDEPS
}

if [ "$DO_INSTALL" = "1" ]; then
    if [ "$REINSTALL" = "0" ] && [ -f "$DEPS_STAMP" ] \
       && grep -qx "$REQ_HASH" "$DEPS_STAMP" && deps_present; then
        log "dependencies already satisfied for this requirements file — skipping"
    else
        log "installing dependencies (this image ships torch/CUDA; we add the rest)…"
        ( unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
          "$PY" -m pip install --no-input -q --upgrade pip >/dev/null 2>&1 || true
          "$PY" -m pip install --no-input -q -r "$REQ" ) || die "pip install -r $REQ failed"
        echo "$REQ_HASH" > "$DEPS_STAMP"
        "$PY" -m pip cache purge >/dev/null 2>&1 || true
    fi
fi

log "verifying the runtime…"
"$PY" - <<'PYCHECK' || die "dependency check failed — see the traceback above"
import importlib.util, sys
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
names = []
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    names.append(p.name)
    print(f"[dw]   GPU{i}: {p.name}  {p.total_memory/1024**3:.0f} GB  sm_{p.major}{p.minor}")
if not torch.cuda.is_available():
    print("[dw]   WARN: no GPU visible — this is the GPU stage; fix the machine type")
elif not any(k in n for n in names for k in ("L40S", "L40", "H100", "H200", "A100")):
    print(f"[dw]   NOTE: expected an L40S, got {names}")
PYCHECK

cd "$V3_DIR" || die "cannot cd $V3_DIR"

if [ "$DO_CHECK" = "1" ]; then
    log "running the offline test suite…"
    "$PY" -m pytest -q
    RC=$?
    log "--check: pytest exited rc=$RC"
    exit "$RC"
fi

# ===========================================================================
# 6. Already finished?  A Studio restart must not silently retrain.
# ===========================================================================
if [ -f "$SENTINEL" ] && [ "$FORCE" = "0" ] && [ "$SHOW_TUNING" = "0" ]; then
    log "a completed run is already in $RESULTS_DIR:"
    sed 's/^/[dw]   /' "$SENTINEL"
    log "re-run with --force to train again. Nothing to do."
    disk_report
    exit 0
fi

# ===========================================================================
# 7. Cleanup + interrupt handling  (shards are never in scope here)
# ===========================================================================
cleanup_storage() {
    local mode="$1" before after
    before="$(avail_gib "$WORK_ROOT")"; before="${before:-0}"
    log "cleanup ($mode)…"
    "$PY" -m pip cache purge >/dev/null 2>&1 || true
    safe_rm "$PIP_CACHE_DIR"
    safe_rm "$TORCHINDUCTOR_CACHE_DIR"
    safe_rm "$TRITON_CACHE_DIR"
    safe_rm "$WORK_ROOT/.cache/matplotlib"; safe_rm "$MPLCONFIGDIR"
    safe_rm "$V3_DIR/.pytest_cache"; safe_rm "$V3_DIR/.ruff_cache"
    find "$REPO_DIR" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null
    find "$HF_HOME_DIR" -type d -name '.locks' -exec rm -rf {} + 2>/dev/null
    if [ "$mode" = "cache" ]; then
        warn "--wipe-cache: dropping the encoder weights in $HF_HOME_DIR"
        safe_rm "$HF_HOME_DIR/hub"; safe_rm "$HF_HOME_DIR/xet"
    fi
    sync 2>/dev/null || true
    after="$(avail_gib "$WORK_ROOT")"; after="${after:-0}"
    log "cleanup done — reclaimed ~$(( after - before )) GiB (now ${after} GiB free)"
}

on_interrupt() {
    warn "interrupted — last.pt stays put, so the next start resumes"
    stop_disk_guard 2>/dev/null || true
    cleanup_storage light
    disk_report
    exit 130
}

# --- watchdog: disk, and the GPU numbers that explain a stalled run ---------
GUARD_PID=""
GUARD_LOG="$LOG_DIR/gpu_disk_guard_${RUN_ID}.log"
start_disk_guard() {
    setsid bash -c '
        while true; do
            sleep 120
            free=$(df -P -BG "'"$WORK_ROOT"'" 2>/dev/null | awk "NR==2 {gsub(/G/,\"\",\$4); print \$4+0}")
            free=${free:-999}
            read -r util used total < <(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
                --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d "," )
            util=${util:-0}; used=${used:-0}; total=${total:-1}
            pct=$(( used * 100 / total ))
            echo "[guard $(date +%H:%M:%S)] gpu_util=${util}%  vram=${used}/${total}MiB (${pct}%)  disk_free=${free}GiB"
            if [ "$util" -lt 50 ] && [ "$pct" -lt 45 ]; then
                echo "[guard] STARVED: low util AND low VRAM -> the LOADER is the limit,"
                echo "[guard]          not the batch. Check --gpu_augment landed, then raise"
                echo "[guard]          DW_LOADER_WORKERS (currently one per vCPU minus 2)."
            elif [ "$util" -ge 80 ] && [ "$pct" -lt 55 ]; then
                echo "[guard] HEADROOM: card is busy but only ${pct}% of VRAM is in use ->"
                echo "[guard]          re-probe with more headroom:"
                echo "[guard]          DW_VRAM_HEADROOM=0.93 sh train_L40S.sh --force"
            fi
            if [ "$free" -lt 12 ]; then
                echo "[guard] LOW DISK (${free}GiB) — dropping compile caches"
                rm -rf "'"$PIP_CACHE_DIR"'" "'"$TORCHINDUCTOR_CACHE_DIR"'"/* "'"$TRITON_CACHE_DIR"'"/* 2>/dev/null
            fi
        done
    ' >"$GUARD_LOG" 2>&1 </dev/null &
    GUARD_PID=$!
    log "gpu/disk watchdog running (pid $GUARD_PID) -> $GUARD_LOG"
}
stop_disk_guard() {
    [ -n "$GUARD_PID" ] || return 0
    kill -- "-$GUARD_PID" 2>/dev/null || kill "$GUARD_PID" 2>/dev/null
    pkill -P "$GUARD_PID" 2>/dev/null
    GUARD_PID=""
}

trap on_interrupt INT TERM

# ===========================================================================
# 8. Escape hatch: pack here only if explicitly asked
# ===========================================================================
if [ -n "$MISSING_SRC" ] && [ "$ALLOW_PREPARE" = "1" ]; then
    PREP_LIST="$(echo "$MISSING_SRC" | tr ' ' ',' | sed 's/^ *//;s/ *$//;s/ /,/g')"
    start_disk_guard
    ( unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
      "$PY" -u main.py --skip-install --prepare \
          --data_root "$DATA_ROOT" --datasets "$PREP_LIST" \
          --gamus_train "${DW_GAMUS_TRAIN:-4000}" --gamus_val "${DW_GAMUS_VAL:-400}" \
          --synrs3d_archives "${DW_SYNRS3D_ARCHIVES:-2}" --geonrw_max "${DW_GEONRW_MAX:-2500}" \
          --workers "${DW_PREP_WORKERS:-12}" )
    PRC=$?
    stop_disk_guard
    safe_rm "$DATA_ROOT/_dl"
    [ "$PRC" -eq 0 ] || die "prepare failed rc=$PRC"
fi

# ===========================================================================
# 9. Auto-resume from whatever a previous (killed) run left behind
# ===========================================================================
RESUME_ARGS=()
if [ "$AUTO_RESUME" = "1" ] && [ "$FORCE" = "0" ] && [ "$SMOKE" = "0" ]; then
    case " ${TRAIN_ARGS[*]-} " in
        *--resume*) ;;                                   # caller decides
        *)
            if [ -f "$RESULTS_DIR/last.pt" ]; then
                RESUME_ARGS=(--resume "$RESULTS_DIR/last.pt")
                log "warm-starting from last.pt (a previous run did not finish)"
            elif [ -f "$RESULTS_DIR/best.pt" ]; then
                RESUME_ARGS=(--resume "$RESULTS_DIR/best.pt")
                log "warm-starting from best.pt"
            fi
            ;;
    esac
fi

# ===========================================================================
# 9b. Stage the shards on a genuinely faster disk (only if there is one)
# ===========================================================================
# The old version copied 45 GB to /tmp unconditionally when asked. On a
# Lightning Studio /tmp and /teamspace/studios/this_studio are the SAME overlay
# mount — `df` reports one device for both — so that copy moved 45 GB from a
# disk to itself, cost ~10 minutes of L40S time, and made the loader no faster.
# Now the copy only happens when the target is a different filesystem.
#
# What does help on one device is the page cache: the shards are read as random
# windows, and a sequential pre-read pulls them in at streaming speed. That runs
# in the background while the model loads and the batch is probed, so it is free.
if [ "$SHOW_TUNING" = "0" ]; then
    LOCAL_ROOT="${DW_LOCAL_DATA_ROOT:-/tmp/DepthWizard-data}"
    SRC_DEV="$(stat -f -c %i "$DATA_ROOT" 2>/dev/null || echo a)"
    DST_DEV="$(stat -f -c %i "$(dirname "$LOCAL_ROOT")" 2>/dev/null || echo b)"

    if [ "$STAGE_LOCAL" = "1" ] && [ "$SRC_DEV" = "$DST_DEV" ]; then
        warn "--stage-local ignored: $(dirname "$LOCAL_ROOT") and $DATA_ROOT are the
       same filesystem, so copying the shards there would buy nothing and cost
       ~10 min of GPU time. Warming the page cache instead."
        STAGE_LOCAL=0
    fi

    if [ "$STAGE_LOCAL" = "1" ]; then
        NEED_MB="$(du -sm --exclude=_dl "$DATA_ROOT" 2>/dev/null | cut -f1)"; NEED_MB="${NEED_MB:-0}"
        FREE_MB="$(df -Pm "$(dirname "$LOCAL_ROOT")" 2>/dev/null | awk 'NR==2 {print $4+0}')"
        FREE_MB="${FREE_MB:-0}"
        if [ "$FREE_MB" -lt "$(( NEED_MB + 10240 ))" ]; then
            warn "--stage-local: need $(( NEED_MB / 1024 )) GiB on $(dirname "$LOCAL_ROOT"), only
       $(( FREE_MB / 1024 )) GiB free — training from $DATA_ROOT instead"
        else
            log "staging $(( NEED_MB / 1024 )) GiB of shards -> $LOCAL_ROOT …"
            S0=$SECONDS
            mkdir -p "$LOCAL_ROOT"
            if rsync -a --exclude '_dl' "$DATA_ROOT/" "$LOCAL_ROOT/" 2>/dev/null \
               || cp -r "$DATA_ROOT/." "$LOCAL_ROOT/"; then
                DATA_ROOT="$LOCAL_ROOT"
                export DW_DATA_ROOT="$DATA_ROOT"
                log "staged in $(( (SECONDS - S0) / 60 )) min — training reads from local disk"
            else
                warn "staging failed — falling back to $DATA_ROOT"
            fi
        fi
    fi

    # Page-cache warm: sequential, niced, capped at what RAM can actually hold,
    # and detached so it overlaps the model build and the batch probe.
    if [ "${DW_WARM_CACHE:-1}" = "1" ]; then
        RAM_MB="$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo 2>/dev/null || echo 0)"
        WARM_MB=$(( RAM_MB * 60 / 100 ))
        if [ "$WARM_MB" -gt 1024 ]; then
            log "warming the page cache with up to $(( WARM_MB / 1024 )) GiB of shards (background)"
            setsid nohup bash -c '
                budget=$(( '"$WARM_MB"' * 1024 * 1024 ))
                used=0
                for f in $(find "'"$DATA_ROOT"'" -name "*_rgb.npy" -o -name "*_hgt.npy" | sort); do
                    sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
                    [ $(( used + sz )) -gt "$budget" ] && break
                    nice -n 19 dd if="$f" of=/dev/null bs=8M status=none 2>/dev/null
                    used=$(( used + sz ))
                done
                echo "warmed $(( used / 1024 / 1024 / 1024 )) GiB"
            ' >"$LOG_DIR/cache_warm_${RUN_ID}.log" 2>&1 </dev/null &
            WARM_PID=$!
        fi
    fi
fi

# ===========================================================================
# 9c. Autotune — resolve the REAL flag names, then size them by measurement
# ===========================================================================
# Two fixes over the previous version, both of which were leaving the card idle:
#
#   1. `pick` matched option names with `grep -ixE` — a whole-line exact match.
#      So `--compile` never matched v3's `--compile_model` and `--grad_checkpoint`
#      never matched `--grad_checkpoint_encoder`; both concepts were reported as
#      "no flag" and silently skipped. It now falls back to a prefix match.
#   2. The batch came from a VRAM bracket, and 48 GiB landed in the ">= 40 -> 16"
#      bucket. Sixteen 512 px crops through an unfrozen ViT-L is ~20 GiB — the
#      exact "flat 20 GB on a 48 GB card" this run was showing. The bracket is
#      now only a fallback; the real number comes from `tools/probe_batch.py`,
#      which runs a real forward/backward/AdamW step on this card with the
#      encoder unfrozen and reports the largest batch that fits.
if [ "$AUTOTUNE" = "1" ]; then
    GPU_GIB="$("$PY" -c "import torch;print(int(torch.cuda.get_device_properties(0).total_memory/1024**3) if torch.cuda.is_available() else 0)" 2>/dev/null || echo 0)"
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' ' '_')"
    NPROC="$(nproc 2>/dev/null || echo 8)"
    HELP_TXT="$("$PY" main.py --help 2>&1)"
    OPTS="$(printf '%s\n' "$HELP_TXT" | grep -oE '\-\-[a-zA-Z0-9][a-zA-Z0-9_-]*' | sort -u)"

    log "autotune: ${GPU_NAME:-?} ${GPU_GIB} GiB VRAM, ${NPROC} vCPU, $(printf '%s' "$OPTS" | wc -l) options in main.py"

    # Exact name first, then a prefix match — `--compile` -> `--compile_model`.
    pick() {
        local pat m
        for pat in "$@"; do
            m="$(printf '%s\n' "$OPTS" | grep -ixE -- "--$pat" | head -1)"
            [ -n "$m" ] && { printf '%s' "$m"; return 0; }
        done
        for pat in "$@"; do
            m="$(printf '%s\n' "$OPTS" | grep -ixE -- "--$pat[_-].*" | head -1)"
            [ -n "$m" ] && { printf '%s' "$m"; return 0; }
        done
        return 1
    }
    # argparse store_true has no metavar after it; anything else takes a value.
    takes_value() { printf '%s' "$HELP_TXT" | grep -qE -- "\\$1[ =](\{|<|[A-Z]|[0-9])"; }
    have_arg() { case " ${TRAIN_ARGS[*]-} " in *" $1 "*|*" $1="*) return 0 ;; esac; return 1; }

    tune() {   # tune <value> <name-regex>...
        local val="$1"; shift
        local flag
        flag="$(pick "$@")" || { log "  - no flag for [$1] — skipped"; return 0; }
        if have_arg "$flag"; then log "  = $flag left at your value"; return 0; fi
        if takes_value "$flag"; then
            AUTO_ARGS+=("$flag" "$val"); log "  + $flag $val"
        elif [ "$val" = "true" ] || [ "$val" = "1" ]; then
            AUTO_ARGS+=("$flag"); log "  + $flag (store_true)"
        else
            log "  - $flag is a switch; cannot set it to $val — skipped"
        fi
    }

    # --- feeding the card ---------------------------------------------------
    # Do this before the batch, because the probe's own worker count does not
    # matter but the loader's does, and a starved loader is what an operator
    # mistakes for "the batch is too small".
    W=$(( NPROC - 2 )); [ "$W" -lt 2 ] && W=2; [ "$W" -gt 16 ] && W=16
    W="${DW_LOADER_WORKERS:-$W}"
    tune "$W" 'num_workers' 'workers' 'dataloader_workers' 'n_workers'
    tune 6    'prefetch_factor' 'prefetch'
    # The whole point of gpu_augment: the jitter and the encoder normalisation
    # stop being 22 ms of single-threaded numpy per crop and become ~2 ms of
    # pointwise kernels per batch, and the worker ships uint8 instead of fp32.
    tune true 'gpu_augment'
    tune true 'channels_last' 'memory_format'
    tune true 'sdp_flash'
    export OMP_NUM_THREADS=1

    # --- memory/speed trades that are backwards on a card with room ---------
    tune false 'grad_checkpoint_encoder' 'grad_checkpoint' 'gradient_checkpointing'
    tune false 'grad_checkpoint_decoder'
    tune bf16  'amp_dtype' 'precision' 'dtype' 'mixed_precision'
    tune true  'amp' 'use_amp'
    tune true  'tf32' 'allow_tf32'

    # --- batch: measured on this card, not guessed from a bracket -----------
    if [ -n "${DW_BATCH:-}" ]; then
        BS="$DW_BATCH"
        log "  batch: DW_BATCH=$BS (probe skipped)"
    else
        # The probe costs ~90 s and the answer only changes when the card, the
        # tile size or the checkpointing flag changes — so cache it on those.
        PROBE_KEY="${GPU_NAME}_t$("$PY" -c "from config import Config;print(Config().tile_size)" 2>/dev/null || echo 512)"
        PROBE_STAMP="$WORK_ROOT/.dw_probe_${PROBE_KEY}"
        if [ -s "$PROBE_STAMP" ] && [ "$FORCE" = "0" ]; then
            BS="$(cat "$PROBE_STAMP")"
            log "  batch: $BS (cached probe for $PROBE_KEY — delete $PROBE_STAMP to re-probe)"
        else
            log "  batch: probing this card with a real unfrozen-encoder step…"
            PROBE_OUT="$("$PY" -u -m tools.probe_batch --headroom "${DW_VRAM_HEADROOM:-0.88}" 2>&1)"
            printf '%s\n' "$PROBE_OUT" | sed 's/^/[dw]   /'
            BS="$(printf '%s\n' "$PROBE_OUT" | grep -oE 'DW_PROBE_BATCH=[0-9]+' | tail -1 | cut -d= -f2)"
            if [ -n "$BS" ] && [ "$BS" -gt 0 ] 2>/dev/null; then
                echo "$BS" > "$PROBE_STAMP"
            else
                # Bracket fallback, with a tier that knows an L40S exists.
                if   [ "$GPU_GIB" -ge 70 ]; then BS=48
                elif [ "$GPU_GIB" -ge 44 ]; then BS=28
                elif [ "$GPU_GIB" -ge 20 ]; then BS=10
                else                             BS=4; fi
                warn "probe produced no answer — falling back to the ${GPU_GIB} GiB bracket ($BS)"
            fi
        fi
    fi
    tune "$BS" 'batch_size' 'batch' 'bs' 'train_batch_size' 'per_device_batch_size'

    # Keep the *effective* batch near 48 regardless of what fits: accumulation
    # is free throughput-wise once the card is full, and the LR schedule was
    # written for a batch of that order.
    TARGET_EFF="${DW_EFFECTIVE_BATCH:-48}"
    ACC=$(( (TARGET_EFF + BS - 1) / BS )); [ "$ACC" -lt 1 ] && ACC=1
    tune "$ACC" 'grad_accum' 'grad_accum_steps' 'accum_steps' 'gradient_accumulation_steps'
    log "  effective batch: $BS x $ACC = $(( BS * ACC ))"

    # Eval allocates no activations, so it holds a bigger batch than training
    # and there is no reason to make the card run it at the training batch.
    tune "${DW_EVAL_BATCH_MULT:-2}" 'eval_batch_mult'

    # torch.compile is a real ~1.2x on an L40S but it pays a multi-minute
    # warmup AND recompiles when the encoder unfreezes, so it is opt-in: worth
    # it for a 26-epoch run, not for a smoke test.
    if [ "${DW_COMPILE:-0}" = "1" ] && [ "$SMOKE" = "0" ]; then
        tune true 'compile_model' 'compile' 'torch_compile'
    fi

    export DW_BATCH_SIZE="${DW_BATCH_SIZE:-$BS}"
    export DW_NUM_WORKERS="${DW_NUM_WORKERS:-$W}"

    if [ "${#AUTO_ARGS[@]}" -eq 0 ]; then
        warn "main.py --help matched none of the expected concepts. Look at the real
       names and pass them yourself:
         cd $V3_DIR && $PY main.py --help | grep -iE 'batch|worker|precision|amp|compile'"
    fi
fi

if [ "$SHOW_TUNING" = "1" ]; then
    log "--show-tuning: would launch"
    log "  $PY -u main.py --skip-install --data_root $DATA_ROOT --output_dir $RESULTS_DIR \\"
    log "     --datasets $DATASETS --make_zip true ${AUTO_ARGS[*]-} ${RESUME_ARGS[*]-} ${TRAIN_ARGS[*]-}"
    exit 0
fi

# ===========================================================================
# 9d. Train, halving the batch if the tuner overshot
# ===========================================================================
BASE_ARGS=(
    --skip-install                       # section 5 already did it
    --data_root "$DATA_ROOT"
    --output_dir "$RESULTS_DIR"
    --datasets "$DATASETS"
    --make_zip true
)

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/[dw]   /'
disk_report
start_disk_guard
T0=$SECONDS
RC=1

for attempt in 1 2 3; do
    ATTEMPT_LOG="$LOG_DIR/train_${RUN_ID}_try${attempt}.log"
    log "launching (attempt $attempt): $PY -u main.py ${BASE_ARGS[*]} ${AUTO_ARGS[*]-} ${RESUME_ARGS[*]-} ${TRAIN_ARGS[*]-}"

    "$PY" -u main.py "${BASE_ARGS[@]}" ${AUTO_ARGS[@]+"${AUTO_ARGS[@]}"} \
          ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"} 2>&1 \
        | tee "$ATTEMPT_LOG"
    RC="${PIPESTATUS[0]}"

    [ "$RC" -eq 0 ] && break
    grep -qiE "out of memory|CUDA error: out of memory" "$ATTEMPT_LOG" || break

    # OOM: the tuner overshot. Halve and resume — nothing done so far is lost.
    NEW_BS=$(( ${BS:-32} / 2 ))
    [ "$NEW_BS" -lt 1 ] && break
    warn "CUDA OOM at batch ${BS:-?} — retrying at $NEW_BS"
    # A cached probe that OOMs is a wrong cached probe; drop it so the next
    # start measures again instead of walking into the same wall.
    [ -n "${PROBE_STAMP:-}" ] && rm -f "$PROBE_STAMP" 2>/dev/null
    for i in "${!AUTO_ARGS[@]}"; do
        [ "${AUTO_ARGS[$i]}" = "${BS:-}" ] && AUTO_ARGS[$i]="$NEW_BS" && break
    done
    BS="$NEW_BS"
    if [ -f "$RESULTS_DIR/last.pt" ] && [ "${#RESUME_ARGS[@]}" -eq 0 ]; then
        RESUME_ARGS=(--resume "$RESULTS_DIR/last.pt")
    fi
    "$PY" -c "import torch; torch.cuda.empty_cache()" >/dev/null 2>&1 || true
    sleep 20
done

stop_disk_guard
trap - INT TERM
ELAPSED=$(( (SECONDS - T0) / 60 ))
log "training exited rc=$RC after ${ELAPSED} min"

# ===========================================================================
# 10. Save / verify the results
# ===========================================================================
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
    echo "data_root   : $DATA_ROOT   (shards prepared beforehand)"
    echo "staged_local: $STAGE_LOCAL"
    echo "offline     : ${HF_HUB_OFFLINE:-0}"
    echo "autotune    : ${AUTO_ARGS[*]-<off>}"
    echo "args        : ${BASE_ARGS[*]} ${AUTO_ARGS[*]-} ${RESUME_ARGS[*]-} ${TRAIN_ARGS[*]-}"
    echo "peak_vram   : $(grep -ho 'vram=[0-9]*' "$GUARD_LOG" 2>/dev/null | cut -d= -f2 | sort -n | tail -1) MiB"
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

cp -f "$BOOT_LOG" "$RESULTS_DIR/train_boot.log" 2>/dev/null || true
[ -f "$READY_STAMP" ] && cp -f "$READY_STAMP" "$RESULTS_DIR/DATA_READY.txt" 2>/dev/null || true

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
# 11. Utilisation verdict — so the next run is better sized than this one
# ===========================================================================
if [ -s "$GUARD_LOG" ]; then
    PEAK="$(grep -ho 'vram=[0-9]*' "$GUARD_LOG" | cut -d= -f2 | sort -n | tail -1)"
    TOTAL="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)"
    AVGUTIL="$(grep -ho 'gpu_util=[0-9]*' "$GUARD_LOG" | cut -d= -f2 | awk '{s+=$1; n++} END {print (n ? int(s/n) : 0)}')"
    if [ -n "$PEAK" ] && [ -n "$TOTAL" ] && [ "$TOTAL" -gt 0 ]; then
        PCT=$(( PEAK * 100 / TOTAL ))
        log "utilisation: peak VRAM ${PEAK}/${TOTAL} MiB (${PCT}%), mean GPU util ${AVGUTIL}%"
        if [ "$PCT" -lt 55 ] && [ "$AVGUTIL" -ge 80 ]; then
            log "  compute-bound with VRAM to spare -> next run:"
            log "    DW_VRAM_HEADROOM=0.93 sh train_L40S.sh --force   # re-probe the batch"
        elif [ "$AVGUTIL" -lt 50 ]; then
            log "  input-bound (util ${AVGUTIL}%) -> the loader, not the batch, is the limit:"
            log "    grep -c gpu_augment $RESULTS_DIR/run.log         # did GPU aug land?"
            log "    DW_LOADER_WORKERS=$(nproc) sh train_L40S.sh --force"
        fi
    fi
fi

# ===========================================================================
# 12. Reclaim the disk  (shards always survive — that is the contract)
# ===========================================================================
if [ "$RC" -eq 0 ] && [ -z "$MISSING" ]; then
    if [ "$SMOKE" = "1" ]; then
        log "smoke run OK -> $RESULTS_DIR  (no completion marker)"
    else
        printf 'run_id=%s\nfinished_at=%s\nelapsed_min=%s\n' "$RUN_ID" "$(date -Is)" "$ELAPSED" > "$SENTINEL"
    fi
    if [ "$DO_CLEAN" = "1" ]; then
        if [ "$WIPE_CACHE" = "1" ]; then cleanup_storage cache; else cleanup_storage light; fi
    fi
    log "shards kept in $PERSIST_DATA_ROOT ($(du -sh "$PERSIST_DATA_ROOT" 2>/dev/null | cut -f1))"
else
    [ -n "$MISSING" ] && warn "missing expected artifacts:$MISSING"
    warn "run did not complete cleanly — keeping everything re-usable so the next
       start resumes instead of re-downloading:
         checkpoints  $RESULTS_DIR
         shards       $PERSIST_DATA_ROOT"
    [ "$DO_CLEAN" = "1" ] && cleanup_storage light
fi

# ===========================================================================
# 13. Summary
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
    log "gpu/disk watchdog (last 4 samples):"
    tail -4 "$GUARD_LOG" | sed 's/^/[dw]   /'
    cp -f "$GUARD_LOG" "$RESULTS_DIR/gpu_disk_guard.log" 2>/dev/null || true
fi
log "boot log: $BOOT_LOG"
log "=============================================================="
exit "$RC"
