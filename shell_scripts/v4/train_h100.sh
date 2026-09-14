#!/bin/bash
# ---------------------------------------------------------------------------
# DepthWizard v4 — PART 2 of 2:  TRAIN  (run this on the H100 Studio)
#
# ! fast_load
# DepthWizard-results/v4/best.pt
# DepthWizard-results/v4/last.pt
# DepthWizard-results/v4/validation_report.html
# DepthWizard-results/results_v4.zip
#
# Assumes prepare_data.sh already staged everything on the persistent disk. It
# does NOT download datasets — it refuses to start if the shards are missing,
# because re-downloading 40 GB on an H100 is the most expensive thing this repo
# can do.
#
#   reads   <root>/DepthWizard-data/       packed memmap shards
#   reads   <root>/hf_home/hub/            DINOv3 encoder weights
#   writes  <root>/DepthWizard-results/v4/ checkpoints, metrics, figures,
#                                          validation_report.html, ONNX, zip
#
# ---- What this does differently from v3's train_h100.sh -------------------
# v3's script guessed the sizing on the card and got it wrong twice. Its tuner
# read `GPU_GIB >= 70` and set batch 32 with grad_accum 1; that run reached
# epoch 7, drifted from 70 GB to 76 GB, and died on a 768 MB allocation. The
# retry halved to 16. Logs/v3/run.log has both tracebacks.
#
# So this script does NOT autotune the batch. v4's config.py already carries the
# numbers, and they come from v3's own measurements rather than from a rule of
# thumb — encoder unfrozen, no checkpointing:
#
#     micro-batch 16 -> 42 GB, 44.7 img/s
#     micro-batch 32 -> 74 GB, 48.0 img/s, then OOM at epoch 7
#
# Doubling the batch bought 7 % because the card was already at 99-100 %
# utilisation either way; it is not where the spare 33 GB should go. v4 uses 24
# (~58 GB, ~20 GB of headroom) with grad_accum 1. What the headroom actually
# buys is gradient checkpointing OFF, which is worth ~35 % of the step.
#
# Section 9c only sets what genuinely depends on *this* machine: the loader
# worker count, and a smaller batch if the card turns out not to be an H100.
# Everything else is config.py's, and `--show-tuning` prints the exact command.
#
# The OOM-halving retry is kept (section 9d) — it is cheap insurance, and it now
# halves from a number that should never need it.
#
# Usage (from the persistent-storage root, i.e. ~ on Lightning):
#   sh train_h100.sh --background                 # the recommended first run
#   sh train_h100.sh --show-tuning                # resolve flags, launch nothing
#   sh train_h100.sh --smoke                      # ~10 min sanity run
#   sh train_h100.sh --force                      # retrain even if one finished
#   sh train_h100.sh --no-resume                  # ignore last.pt
#   sh train_h100.sh --check                      # deps + tests + GPU report
#   sh train_h100.sh --no-stage-local             # read shards off /teamspace
#   sh train_h100.sh --online                     # allow Hub access
#   sh train_h100.sh --datasets gamus,synrs3d,india_unlabeled   # + mean teacher
#   DW_BATCH=32 sh train_h100.sh                  # override the batch yourself
#   DW_COMPILE=true sh train_h100.sh              # opt into torch.compile
# Any unrecognised flag is forwarded verbatim to train.py, and the caller's
# value always beats this script's.
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
# ON by default. /teamspace is network-backed, and random 512 px crops out of
# 27 GB of shards across it is what "STARVED" means in Logs/v3/gpu_disk_guard.log.
# The v3 run that finished read from /tmp.
STAGE_LOCAL="${DW_STAGE_LOCAL:-1}"
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

if   [ -d "$SELF_DIR/FineTunning/v4" ];       then REPO_DIR="$SELF_DIR"; IN_PLACE=1
elif [ -d "$SELF_DIR/../../FineTunning/v4" ]; then REPO_DIR="$(cd "$SELF_DIR/../.." && pwd)"; IN_PLACE=1
elif [ -d "$SELF_DIR/../FineTunning/v4" ];    then REPO_DIR="$(cd "$SELF_DIR/.." && pwd)"; IN_PLACE=1
else REPO_DIR="$WORK_ROOT/DepthWizard"; IN_PLACE=0; fi
V4_DIR="$REPO_DIR/FineTunning/v4"

SMOKE=0
case " ${TRAIN_ARGS[*]-} " in *" --smoke"*) SMOKE=1 ;; esac

if [ "$SMOKE" = "1" ]; then
    RESULTS_DIR="${DW_OUTPUT_DIR:-$WORK_ROOT/DepthWizard-results/smoke_v4}"
else
    RESULTS_DIR="${DW_OUTPUT_DIR:-$WORK_ROOT/DepthWizard-results/v4}"
fi
DATA_ROOT="${DW_DATA_ROOT:-$WORK_ROOT/DepthWizard-data}"
PERSIST_DATA_ROOT="$DATA_ROOT"                 # never reassigned; 9b may move DATA_ROOT
HF_HOME_DIR="${HF_HOME:-$WORK_ROOT/hf_home}"
LOG_DIR="$WORK_ROOT/DepthWizard-results/logs"
SENTINEL="$RESULTS_DIR/.dw_complete"
READY_STAMP="$PERSIST_DATA_ROOT/.dw_data_ready_v4"
[ -f "$READY_STAMP" ] || READY_STAMP="$PERSIST_DATA_ROOT/.dw_data_ready"   # v3's stamp
DEPS_STAMP="$WORK_ROOT/.dw_v4_deps_ok"
RUN_ID="$(date +%Y%m%d_%H%M%S)"

ZIP_PATH="$(dirname "$RESULTS_DIR")/results_$(basename "$RESULTS_DIR").zip"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
BOOT_LOG="$LOG_DIR/train_v4_${RUN_ID}.log"

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
log "DepthWizard v4 — TRAIN (H100)  ${RUN_ID}"
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
        "$PERSIST_DATA_ROOT"|"$PERSIST_DATA_ROOT"/gamus|"$PERSIST_DATA_ROOT"/synrs3d|"$PERSIST_DATA_ROOT"/geonrw|"$PERSIST_DATA_ROOT"/india_labeled|"$PERSIST_DATA_ROOT"/india_unlabeled)
            warn "refusing to delete packed shards: $p"; return 0 ;;
        "$WORK_ROOT"/*|"$HOME"/*|/tmp/*) ;;
        *) warn "refusing to delete outside the work root: $p"; return 0 ;;
    esac
    rm -rf -- "$p" 2>/dev/null || true
}

# ===========================================================================
# 3. Environment — Hopper
# ===========================================================================
export HF_HOME="$HF_HOME_DIR"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DOWNLOAD_TIMEOUT=60
export DW_OUTPUT_DIR="$RESULTS_DIR"          # read by FineTunning/v4/config.py
export DW_DATA_ROOT="$DATA_ROOT"             # read by FineTunning/v4/config.py
export PIP_CACHE_DIR="$WORK_ROOT/.cache/pip"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export TORCHINDUCTOR_CACHE_DIR="$WORK_ROOT/.cache/inductor"
export TRITON_CACHE_DIR="$WORK_ROOT/.cache/triton"
export TORCH_HOME="$WORK_ROOT/.cache/torch"
export XDG_CACHE_HOME="$WORK_ROOT/.cache"
export MPLCONFIGDIR="$WORK_ROOT/.cache/mpl"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# expandable_segments is what BOTH v3 OOM tracebacks asked for by name. The H100
# one died on a 768 MB allocation with 78.4 GB in use and 2.36 GB
# reserved-but-unallocated — that gap is allocator fragmentation, and this is
# the fix for it. train.py also sets it via os.environ.setdefault; exporting it
# here covers anything that imports torch before train.py runs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NVIDIA_TF32_OVERRIDE="${NVIDIA_TF32_OVERRIDE:-1}"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export TORCH_CUDNN_V8_API_ENABLED=1
# Deterministic cuBLAS workspaces serialise the GEMM stream; nothing here needs
# bitwise determinism, and the seed is set in train.py.
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
        gamus)            store_ok gamus/train && store_ok gamus/val ;;
        synrs3d)          store_ok synrs3d/train ;;
        geonrw)           store_ok geonrw/train ;;
        india_labeled)    store_ok india_labeled/train ;;
        india_unlabeled)  store_ok india_unlabeled/train ;;
        *)                return 1 ;;
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

# india_unlabeled is optional by design: train.py turns the mean-teacher branch
# off when the store is absent, so a missing one is a note, not a failure.
MISSING_SRC=""
for d in ${DATASETS//,/ }; do
    case "$d" in
        india_unlabeled|india_labeled)
            source_prepared "$d" || warn "no $d store — the mean-teacher branch stays off.
       Prepare it with:  sh prepare_data.sh --datasets $d --india-dir <tiles>" ;;
        *) source_prepared "$d" || MISSING_SRC="$MISSING_SRC $d" ;;
    esac
done

if [ "$DO_CHECK" = "0" ] && [ "$SHOW_TUNING" = "0" ] && [ -n "$MISSING_SRC" ]; then
    if [ "$ALLOW_PREPARE" = "1" ]; then
        warn "--allow-prepare: packing$MISSING_SRC on the GPU box. Slow and expensive;
       prefer prepare_data.sh on a CPU Studio."
    else
        echo
        die "no packed shards for:$MISSING_SRC   (looked under $DATA_ROOT)

       This script will not download datasets on an H100. Run the data stage on a
       cheap CPU Studio first:

         sh shell_scripts/v4/prepare_data.sh --datasets $DATASETS --background

       …then come back here. To pack on this machine anyway:

         sh train_h100.sh --allow-prepare"
    fi
elif [ "$DO_CHECK" = "0" ] && [ -z "$MISSING_SRC" ]; then
    log "packed shards found for [$DATASETS] — no dataset download needed"
    [ -f "$READY_STAMP" ] && sed 's/^/[dw]   /' "$READY_STAMP" | head -7
elif [ -n "$MISSING_SRC" ]; then
    # --show-tuning and --check both reach here; say what is missing rather than
    # printing the reassuring line the real launch path would have refused on.
    warn "no packed shards for:$MISSING_SRC  (looked under $DATA_ROOT).
       A real launch would refuse here; continuing because this is $([ "$SHOW_TUNING" = 1 ] && echo --show-tuning || echo --check)."
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
       as-is. Force with:  DW_GIT_RESET=1 sh train_h100.sh"
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

[ -d "$V4_DIR" ] || die "expected $V4_DIR — wrong repo layout?"
log "HEAD: $(git -C "$REPO_DIR" log --oneline -1 2>/dev/null || echo n/a)"

REQ="$V4_DIR/requirements.txt"
[ -f "$REQ" ] || die "missing $REQ"
REQ_HASH="$(md5sum "$REQ" | cut -d' ' -f1)"

deps_present() {
    "$PY" - >/dev/null 2>&1 <<'PYDEPS'
import importlib.util, sys
mods = ("torch", "transformers", "huggingface_hub", "h5py", "tifffile",
        "rasterio", "safetensors", "scipy", "numpy", "PIL", "cv2", "matplotlib")
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
# opencv is the loader's SIMD resize; without it the crop path falls back to
# PIL's scalar resampler, which v3 measured at ~40 % of the per-crop budget.
try:
    import cv2
    print(f"[dw]   opencv {cv2.__version__}  (SIMD resize in the loader)")
except Exception:
    print("[dw]   WARN: no opencv — the loader will use PIL and feed the card ~40 % slower")
names = []
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    names.append(p.name)
    print(f"[dw]   GPU{i}: {p.name}  {p.total_memory/1024**3:.0f} GB  sm_{p.major}{p.minor}")
if not torch.cuda.is_available():
    print("[dw]   WARN: no GPU visible — this is the GPU stage; fix the machine type")
elif not any("H100" in n or "H200" in n for n in names):
    print(f"[dw]   NOTE: expected an H100, got {names} — section 9c will resize the batch")
PYCHECK

cd "$V4_DIR" || die "cannot cd $V4_DIR"

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
    safe_rm "$V4_DIR/.pytest_cache"; safe_rm "$V4_DIR/.ruff_cache"
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
# The thresholds are v4's, not v3's. v3's guard said "HEADROOM: raise the batch"
# whenever VRAM was under 40 %, which is the advice that produced the batch-32
# OOM. On v4 a healthy run sits near 58 GB / 99 % util, so the interesting
# signals are the opposite two: a starved loader, and VRAM *drifting upward*
# across the run — which is exactly what preceded the v3 crash (70 GB at epoch
# 3, 76 GB at epoch 7, dead).
GUARD_PID=""
GUARD_LOG="$LOG_DIR/gpu_disk_guard_v4_${RUN_ID}.log"
start_disk_guard() {
    setsid bash -c '
        first_used=0
        while true; do
            sleep 120
            free=$(df -P -BG "'"$WORK_ROOT"'" 2>/dev/null | awk "NR==2 {gsub(/G/,\"\",\$4); print \$4+0}")
            free=${free:-999}
            read -r util used total < <(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
                --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d "," )
            util=${util:-0}; used=${used:-0}; total=${total:-1}
            pct=$(( used * 100 / total ))
            echo "[guard $(date +%H:%M:%S)] gpu_util=${util}%  vram=${used}/${total}MiB (${pct}%)  disk_free=${free}GiB"
            if [ "$first_used" -eq 0 ] && [ "$used" -gt 4096 ]; then first_used=$used; fi
            if [ "$util" -lt 50 ] && [ "$pct" -lt 40 ]; then
                echo "[guard] STARVED: low util AND low VRAM -> the loader is the limit,"
                echo "[guard]          not the batch. Check the img/s line in the train log,"
                echo "[guard]          then: --stage-local (default on) or DW_LOADER_WORKERS=N."
            elif [ "$pct" -ge 90 ]; then
                echo "[guard] TIGHT: ${pct}% of VRAM in use. v3 OOMed from here."
                echo "[guard]        If it crashes, 9d halves the batch and resumes from last.pt."
            elif [ "$first_used" -gt 0 ] && [ "$used" -gt $(( first_used + 8192 )) ]; then
                echo "[guard] DRIFT: VRAM up $(( (used - first_used) / 1024 )) GiB since the first"
                echo "[guard]        steady sample. This is the allocator creep that killed the"
                echo "[guard]        v3 batch-32 run. expandable_segments is on; watch it."
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
# 9b. Stage the shards on local NVMe  (ON by default — see the flag comment)
# ===========================================================================
# Training does ~12 000 random 512 px crops an epoch out of ~27 GB of shards:
# small random reads. Local NVMe lets the page cache absorb them; a network
# mount cannot, and no worker count fixes it. This is a copy — the originals
# stay on the persistent disk, which is why local scratch being ephemeral is
# fine.
STAGED=0                       # what the manifest reports — the outcome, not the flag
if [ "$STAGE_LOCAL" = "1" ] && [ "$SHOW_TUNING" = "0" ]; then
    LOCAL_ROOT="${DW_LOCAL_DATA_ROOT:-/tmp/DepthWizard-data}"
    NEED_MB="$(du -sm --exclude=_dl "$DATA_ROOT" 2>/dev/null | cut -f1)"; NEED_MB="${NEED_MB:-0}"
    FREE_MB="$(df -Pm "$(dirname "$LOCAL_ROOT")" 2>/dev/null | awk 'NR==2 {print $4+0}')"
    FREE_MB="${FREE_MB:-0}"
    # Slack on top of the shards: the run also writes checkpoints and, on some
    # images, /tmp is the same filesystem as everything else. Lower it on a box
    # where local disk is tight but still faster than the network mount.
    SLACK_MB="${DW_STAGE_SLACK_MB:-10240}"
    if [ "$NEED_MB" -eq 0 ]; then
        warn "--stage-local: nothing to copy from $DATA_ROOT"
    elif [ "$FREE_MB" -lt "$(( NEED_MB + SLACK_MB ))" ]; then
        warn "--stage-local: need ${NEED_MB} MiB + ${SLACK_MB} MiB of slack on
       $(dirname "$LOCAL_ROOT"), only ${FREE_MB} MiB free — training from $DATA_ROOT
       instead. Expect the guard to print STARVED if that is a network mount.
       Override the slack with DW_STAGE_SLACK_MB."
    else
        log "staging ${NEED_MB} MiB of shards -> $LOCAL_ROOT …"
        S0=$SECONDS
        mkdir -p "$LOCAL_ROOT"
        if rsync -a --exclude '_dl' "$DATA_ROOT/" "$LOCAL_ROOT/" 2>/dev/null \
           || cp -r "$DATA_ROOT/." "$LOCAL_ROOT/"; then
            DATA_ROOT="$LOCAL_ROOT"
            export DW_DATA_ROOT="$DATA_ROOT"
            STAGED=1
            log "staged in $(( (SECONDS - S0) / 60 )) min — training reads from local disk"
        else
            warn "staging failed — falling back to $DATA_ROOT"
        fi
    fi
fi

# ===========================================================================
# 9c. Machine-dependent flags ONLY.  The rest is config.py's job.
# ===========================================================================
# v3's tuner guessed the batch from VRAM and OOMed the run twice. v4's config.py
# already holds numbers measured on this exact card, so the only things resolved
# here are the two that genuinely vary per machine: how many loader workers this
# box can feed, and a smaller batch if the card is not an 80 GB Hopper.
BS=""
ACCUM=""
if [ "$AUTOTUNE" = "1" ] && [ "$SMOKE" = "0" ]; then
    GPU_GIB="$("$PY" -c "import torch;print(int(torch.cuda.get_device_properties(0).total_memory/1024**3) if torch.cuda.is_available() else 0)" 2>/dev/null || echo 0)"
    NPROC="$(nproc 2>/dev/null || echo 8)"
    CFG_BS="$("$PY" -c "import sys;sys.path.insert(0,'.');from config import Config;print(Config().batch_size)" 2>/dev/null || echo 24)"
    CFG_W="$("$PY" -c "import sys;sys.path.insert(0,'.');from config import Config;print(Config().num_workers)" 2>/dev/null || echo 16)"

    log "sizing: ${GPU_GIB} GiB VRAM, ${NPROC} vCPU  (config.py: batch $CFG_BS, workers $CFG_W)"

    have_arg() { case " ${TRAIN_ARGS[*]-} " in *" $1 "*|*" $1="*) return 0 ;; esac; return 1; }

    # --- batch ---------------------------------------------------------
    # Effective batch is held near 24 across cards so the LR schedule, which was
    # tuned at that scale, keeps meaning the same thing.
    if [ -n "${DW_BATCH:-}" ]; then
        BS="$DW_BATCH"; ACCUM="${DW_GRAD_ACCUM:-1}"
        log "  DW_BATCH=$BS (yours) accum=$ACCUM"
    elif [ "$GPU_GIB" -ge 70 ]; then
        BS="$CFG_BS"; ACCUM=""                     # trust config.py exactly
        log "  batch $BS, grad_accum 1 — config.py's H100 numbers, left alone"
    elif [ "$GPU_GIB" -ge 40 ]; then
        BS=12; ACCUM=2
        log "  batch 12 x accum 2 — ${GPU_GIB} GiB card, effective batch 24"
    elif [ "$GPU_GIB" -ge 20 ]; then
        BS=6; ACCUM=4
        AUTO_ARGS+=(--grad_checkpoint_encoder true)
        log "  batch 6 x accum 4 + gradient checkpointing — ${GPU_GIB} GiB card"
    elif [ "$GPU_GIB" -gt 0 ]; then
        BS=2; ACCUM=12
        AUTO_ARGS+=(--grad_checkpoint_encoder true)
        warn "  ${GPU_GIB} GiB card — batch 2 x accum 12 + gradient checkpointing.
       This will be slow; the defaults assume an 80 GB H100."
    fi
    if [ -n "$BS" ] && ! have_arg --batch_size; then AUTO_ARGS+=(--batch_size "$BS"); fi
    if [ -n "$ACCUM" ] && ! have_arg --grad_accum; then AUTO_ARGS+=(--grad_accum "$ACCUM"); fi

    # --- feeding the card ----------------------------------------------
    # v3's H100 run used 16 workers and its guard log shows 99-100 % GPU
    # utilisation throughout, so 16 is a measurement, not a guess. The val
    # loader gets its own pool of 4 (capped inside dwdata/loaders.py), so the
    # real process count is W + 4 — which is why this does not simply take
    # every core.
    W="${DW_LOADER_WORKERS:-}"
    if [ -z "$W" ]; then
        if   [ "$NPROC" -ge 16 ]; then W="$CFG_W"
        elif [ "$NPROC" -ge 8 ];  then W=$(( NPROC - 4 ))
        else                           W=4; fi
    fi
    have_arg --num_workers || AUTO_ARGS+=(--num_workers "$W")
    log "  num_workers $W (+4 for the val pool) on ${NPROC} vCPU"
    [ "$W" -le 4 ] && export OMP_NUM_THREADS=2 || export OMP_NUM_THREADS=1

    # --- torch.compile: opt-in, and say why ----------------------------
    # Off by default. The encoder unfreeze at epoch 3 changes the forward's
    # `frozen` branch, so Dynamo re-traces mid-run; a four-hour run should not
    # discover an Inductor bug at epoch 3. Set DW_COMPILE=true after a smoke run
    # has exercised it.
    if [ "${DW_COMPILE:-false}" = "true" ] && ! have_arg --compile_model; then
        AUTO_ARGS+=(--compile_model true)
        log "  DW_COMPILE=true — torch.compile on (expect a slow step 0, twice)"
    fi
fi

# ===========================================================================
# 9d. Train, halving the batch if the card disagrees with the estimate
# ===========================================================================
BASE_ARGS=(
    --skip-install                       # section 5 already did it
    --data_root "$DATA_ROOT"
    --output_dir "$RESULTS_DIR"
    --make_zip true
)
# $DATASETS was resolved from the caller's own --datasets when they passed one;
# adding it again would put the flag in the command line twice and make the
# manifest read as if two different values had been requested.
case " ${TRAIN_ARGS[*]-} " in
    *" --datasets "*|*" --datasets="*) ;;
    *) BASE_ARGS+=(--datasets "$DATASETS") ;;
esac

if [ "$SHOW_TUNING" = "1" ]; then
    log "--show-tuning: would launch"
    log "  cd $V4_DIR && $PY -u main.py \\"
    log "     ${BASE_ARGS[*]} \\"
    log "     ${AUTO_ARGS[*]-} ${RESUME_ARGS[*]-} ${TRAIN_ARGS[*]-}"
    log ""
    log "  data_root resolves to : $DATA_ROOT"
    log "  everything not listed above comes from FineTunning/v4/config.py"
    exit 0
fi

nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>/dev/null | sed 's/^/[dw]   /'
disk_report
start_disk_guard
T0=$SECONDS
RC=1

for attempt in 1 2 3; do
    ATTEMPT_LOG="$LOG_DIR/train_v4_${RUN_ID}_try${attempt}.log"
    log "launching (attempt $attempt): $PY -u main.py ${BASE_ARGS[*]} ${AUTO_ARGS[*]-} ${RESUME_ARGS[*]-} ${TRAIN_ARGS[*]-}"

    "$PY" -u main.py "${BASE_ARGS[@]}" ${AUTO_ARGS[@]+"${AUTO_ARGS[@]}"} \
          ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"} 2>&1 \
        | tee "$ATTEMPT_LOG"
    RC="${PIPESTATUS[0]}"

    [ "$RC" -eq 0 ] && break
    grep -qiE "out of memory|CUDA error: out of memory" "$ATTEMPT_LOG" || break

    # OOM. Halve the micro-batch and double the accumulation, so the effective
    # batch — and therefore the LR schedule — is unchanged. Nothing done so far
    # is lost: every epoch wrote last.pt.
    CUR_BS="${BS:-24}"
    NEW_BS=$(( CUR_BS / 2 ))
    [ "$NEW_BS" -lt 1 ] && break
    NEW_ACCUM=$(( ${ACCUM:-1} * 2 ))
    warn "CUDA OOM at batch ${CUR_BS} — retrying at ${NEW_BS} x accum ${NEW_ACCUM}
       (same effective batch, so the schedule is unaffected)"
    NEXT=()
    skip=0
    for a in ${AUTO_ARGS[@]+"${AUTO_ARGS[@]}"}; do
        if [ "$skip" = "1" ]; then skip=0; continue; fi
        case "$a" in
            --batch_size|--grad_accum) skip=1 ;;
            *) NEXT+=("$a") ;;
        esac
    done
    AUTO_ARGS=(${NEXT[@]+"${NEXT[@]}"} --batch_size "$NEW_BS" --grad_accum "$NEW_ACCUM")
    BS="$NEW_BS"; ACCUM="$NEW_ACCUM"
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
    echo "version     : v4"
    echo "finished_at : $(date -Is)"
    echo "exit_code   : $RC"
    echo "elapsed_min : $ELAPSED"
    echo "commit      : $(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo n/a)"
    echo "host        : $(hostname)  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    echo "datasets    : $DATASETS"
    echo "data_root   : $DATA_ROOT   (shards prepared beforehand)"
    echo "staged_local: $STAGED  (requested: $STAGE_LOCAL)"
    echo "offline     : ${HF_HUB_OFFLINE:-0}"
    echo "machine_args: ${AUTO_ARGS[*]-<none — config.py used as-is>}"
    echo "args        : ${BASE_ARGS[*]} ${AUTO_ARGS[*]-} ${RESUME_ARGS[*]-} ${TRAIN_ARGS[*]-}"
    echo "peak_vram   : $(grep -ho 'vram=[0-9]*' "$GUARD_LOG" 2>/dev/null | cut -d= -f2 | sort -n | tail -1) MiB"
    echo "img_per_s   : $(grep -ho '[0-9.]* img/s' "$RESULTS_DIR/run.log" 2>/dev/null | cut -d' ' -f1 | sort -n | tail -1) peak"
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
# v4's extra deliverables are each produced under their own try/except in
# train.py, so a missing one is a note rather than a failed run.
SOFT_MISSING=""
if [ "$SMOKE" = "0" ]; then
    # config.apply_smoke() turns export_onnx and final_sliding_eval off, so on a
    # smoke run their absence is the design, not a failure.
    for f in validation_report.html depthwizard.onnx figures; do
        [ -e "$RESULTS_DIR/$f" ] || SOFT_MISSING="$SOFT_MISSING $f"
    done
fi
[ -n "$SOFT_MISSING" ] && warn "these v4 deliverables were not produced:$SOFT_MISSING
       (each is guarded separately in train.py — check run.log for the reason;
        re-make them without retraining:  cd $V4_DIR && sh run_lightning.sh report)"

if [ -f "$RESULTS_DIR/metrics.json" ]; then
    "$PY" - "$RESULTS_DIR/metrics.json" <<'PYSUM' || true
import json, sys

m = json.load(open(sys.argv[1]))
best = m.get("best_val_rmse_m")
print(f"[dw]   best centre-crop val RMSE : "
      f"{best if best not in (None, float('inf')) else 'n/a'} m")
hist = m.get("history", [])
print(f"[dw]   epochs logged             : {len(hist)}")


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
    land = d.get("per_landscape")
    if land:
        cells = "  ".join(f"{n}={num((v or {}).get('rmse_m'))}" for n, v in sorted(land.items()))
        print(f"[dw]   {'':18}  landscapes: {cells}  "
              f"spread={num(d.get('landscape_rmse_spread_m'))} "
              f"worst={d.get('landscape_worst', '-')}")

if "final_sliding_tta" in m:
    print("[dw]   ^ final_sliding_tta is the number to quote (README v4 §6)")
elif m.get("config", {}).get("smoke"):
    print("[dw]   (no final_sliding_tta — --smoke turns the sliding eval off)")
else:
    print("[dw]   WARN: no final_sliding_tta — the final eval did not complete")

# Did it converge, or just run out of epochs? This is the v3 finding: 26 epochs
# in 121 min of a 330 min budget with val RMSE still falling every eval.
evals = [h["val"]["global"]["rmse_m"] for h in hist
         if h.get("val", {}).get("global", {}).get("rmse_m") is not None]
if len(evals) >= 3:
    tail = evals[-3:]
    drop = tail[0] - tail[-1]
    print(f"[dw]   last 3 evals              : "
          + " -> ".join(f"{v:.3f}" for v in tail) + f"  ({drop:+.3f} m)")
    if drop > 0.005:
        print("[dw]   still improving at the last eval — this run is epoch-limited,")
        print("[dw]   not converged. Next:  sh train_h100.sh --force --epochs "
              f"{int(len(hist) * 1.5)}")

# The tail bias is the one v4 deliberately moved; say whether it moved.
f = m.get("final_tta") or m.get("final_plain") or {}
tb = (f.get("tall_gt15m") or {}).get("bias_m")
if isinstance(tb, (int, float)):
    v3 = -2.12
    print(f"[dw]   tall (>15 m) bias         : {tb:+.2f} m   (v3 finished at {v3:+.2f} m)")
    if tb > 0.5:
        print("[dw]   OVERSHOT: tall structures now over-predicted. Back the tail")
        print("[dw]   re-weighting off:  --stratum_balance_beta 0.5 --stratum_weight_clip 5")
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
        IPS="$(grep -ho '[0-9.]* img/s' "$RESULTS_DIR/run.log" 2>/dev/null | cut -d' ' -f1 | sort -n | tail -1)"
        [ -n "$IPS" ] && log "             peak throughput ${IPS} img/s  (v3 on this card: 48.0)"
        if [ "$AVGUTIL" -lt 60 ]; then
            log "  input-bound (util ${AVGUTIL}%) -> the loader, not the batch, is the limit:"
            log "    sh train_h100.sh --stage-local --force      # shards onto local NVMe"
            log "    DW_LOADER_WORKERS=$(nproc) sh train_h100.sh --force"
        elif [ "$PCT" -lt 55 ] && [ "$AVGUTIL" -ge 90 ]; then
            log "  compute-bound with VRAM to spare. Note that v3 measured only +7 %"
            log "  throughput for 2x the batch on this card, so a bigger batch is the"
            log "  weakest of the options — prefer more epochs (--epochs), or"
            log "  DW_COMPILE=true, before DW_BATCH=$(( ${BS:-24} + 8 ))."
        else
            log "  sizing looks right: high utilisation, VRAM used but not cornered."
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
log "download the bundle:  $ZIP_PATH"
log "the validation report (open it in the Studio file browser):"
log "  $RESULTS_DIR/validation_report.html"
log "predict with the trained checkpoint:"
log "  cd $V4_DIR && $PY -m infer.predict --ckpt $RESULTS_DIR/best.pt --image <path> --absolute"
log "serve it (FastAPI + the 3D flythrough on :8000):"
log "  cd $V4_DIR && $PY -m serve.app --ckpt $RESULTS_DIR/best.pt"
if [ -s "$GUARD_LOG" ]; then
    log "gpu/disk watchdog (last 4 samples):"
    tail -4 "$GUARD_LOG" | sed 's/^/[dw]   /'
    cp -f "$GUARD_LOG" "$RESULTS_DIR/gpu_disk_guard.log" 2>/dev/null || true
fi
log "boot log: $BOOT_LOG"
log "=============================================================="
exit "$RC"
