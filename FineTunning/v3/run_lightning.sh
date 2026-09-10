#!/bin/bash
# DepthWizard v3 — Lightning AI (1x H100 80GB) runbook.
#
#   sh run_lightning.sh check            # deps + offline tests, no GPU needed
#   sh run_lightning.sh prepare          # one-time data pack (persistent disk)
#   sh run_lightning.sh smoke            # ~10 min end-to-end sanity run
#   sh run_lightning.sh train            # the real run
#   sh run_lightning.sh train --epochs 32 --batch_size 16   # extra flags pass through
#
# Prerequisites in the Studio environment:
#   export HF_TOKEN=hf_...     # DINOv3-SAT and GAMUS are both gated
#
# Do NOT put credentials in this file. If you clone a private repo, use
#   git config --global credential.helper store
# or a Studio secret; a token committed to a script ends up in git history.

set -euo pipefail
cd "$(dirname "$0")"

CMD="${1:-train}"; shift || true

if [ -z "${HF_TOKEN:-}" ]; then
  echo "WARNING: HF_TOKEN is not set — the gated DINOv3-SAT encoder and GAMUS will 401."
fi

case "$CMD" in
  check)
    python -m pip install -q -r requirements.txt
    python -m pytest -q
    python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU{i}: {p.name} {p.total_memory/1024**3:.0f} GiB")
PY
    ;;

  prepare)
    # ~25 GB for 4000 GAMUS tiles + ~8 GB per SynRS3D archive. Resumable.
    python prepare_data.py --datasets gamus,synrs3d \
      --gamus_train 4000 --gamus_val 400 --synrs3d_archives 2 "$@"
    ;;

  smoke)
    python train.py --smoke --datasets gamus --share_cloudflared false "$@"
    ;;

  train)
    # Defaults are tuned for 1x H100 80 GB / ~24 vCPU. See README §4.
    python train.py "$@"
    ;;

  predict)
    python -m infer.predict "$@"
    ;;

  *)
    echo "usage: sh run_lightning.sh {check|prepare|smoke|train|predict} [flags]"; exit 1;;
esac
