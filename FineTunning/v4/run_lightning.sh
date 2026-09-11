#!/bin/bash
# DepthWizard v4 — Lightning AI (1x H100 80GB) runbook.
#
#   bash run_lightning.sh check            # deps + offline tests + GPU report
#   bash run_lightning.sh prepare          # one-time data pack (persistent disk)
#   bash run_lightning.sh india            # optional: Indian imagery for the
#                                        #   mean-teacher branch (see README §2)
#   bash run_lightning.sh smoke            # ~10 min end-to-end sanity run
#   bash run_lightning.sh train            # the real run
#   bash run_lightning.sh train --epochs 32 --batch_size 16   # extra flags pass through
#   bash run_lightning.sh predict scene.tif --ckpt outputs/v4/best.pt --absolute
#   bash run_lightning.sh onnx             # export for CPU/standalone deployment
#   bash run_lightning.sh serve            # FastAPI + the 3D viewer on :8000
#   bash run_lightning.sh report           # re-render figures + validation report
#
# Prerequisites in the Studio environment:
#   export HF_TOKEN=hf_...     # DINOv3-SAT and GAMUS are both gated
#
# Do NOT put credentials in this file. If you clone a private repo, use
#   git config --global credential.helper store
# or a Studio secret; a token committed to a script ends up in git history.

# `sh` is dash on Debian/Ubuntu images (Lightning included) and dash has no
# `pipefail`, so `set -euo pipefail` aborted this script on line 1 under the
# exact invocation v3's README told you to use.  Set it only if the shell has it.
set -eu
# shellcheck disable=SC3040
(set -o pipefail) 2>/dev/null && set -o pipefail || true
cd "$(dirname "$0")"

CMD="${1:-train}"
if [ $# -gt 0 ]; then shift; fi   # `shift || true` is fatal in dash
OUT="${DW_OUTPUT_DIR:-outputs/v4}"

# Some images ship only `python3`; a venv ships only `python`.  Pick whichever
# exists rather than assuming, because getting this wrong fails at the last line
# of a long script instead of the first.
if command -v python >/dev/null 2>&1; then PY=python; else PY=python3; fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "WARNING: HF_TOKEN is not set — the gated DINOv3-SAT encoder and GAMUS will 401."
fi

case "$CMD" in
  check)
    $PY -m pip install -q -r requirements.txt
    $PY -m pytest -q
    $PY - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU{i}: {p.name} {p.total_memory/1024**3:.0f} GiB")
PY
    ;;

  prepare)
    # ~25 GB for 4000 GAMUS tiles + ~8 GB per SynRS3D archive. Resumable.
    $PY prepare_data.py --datasets gamus,synrs3d \
      --gamus_train 4000 --gamus_val 400 --synrs3d_archives 2 "$@"
    ;;

  india)
    # Two ways in, and neither downloads a basemap on your behalf — see
    # dwdata/india.py for why that licence call is the team's, not a script's.
    #
    #   labeled  : paired rasters you already hold (DFC2023 New Delhi, a stereo
    #              product, anything with <stem>_rgb.* + <stem>_ndsm.*)
    #   unlabeled: RGB-only Indian tiles -> the mean-teacher branch
    #
    #   babash run_lightning.sh india --datasets india_labeled   --india_dir ~/dfc23
    #   babash run_lightning.sh india --datasets india_unlabeled --india_dir ~/bhuvan
    $PY prepare_data.py --datasets india_unlabeled "$@"
    ;;

  smoke)
    $PY train.py --smoke --datasets gamus "$@"
    ;;

  train)
    # Defaults are tuned for 1x H100 80 GB / ~24 vCPU. See README §5.
    $PY train.py "$@"
    ;;

  predict)
    $PY -m infer.predict "$@"
    ;;

  onnx)
    $PY -m infer.export_onnx --ckpt "${OUT}/best.pt" \
      --out "${OUT}/depthwizard.onnx" "$@"
    ;;

  serve)
    # --onnx runs with no CUDA, no transformers and no HF token: that is the
    # build that goes on a judge's laptop.
    $PY -m serve.app --ckpt "${OUT}/best.pt" "$@"
    ;;

  report)
    $PY -m viz.figures "${OUT}"
    $PY -m viz.report_html "${OUT}"
    ;;

  *)
    echo "usage: bash run_lightning.sh {check|prepare|india|smoke|train|predict|onnx|serve|report} [flags]"
    exit 1;;
esac
