"""Run only the evaluation / testing pipeline on a trained checkpoint.

Skips Stage P (pretrain) and Stage F (finetune) entirely — just loads a
checkpoint and runs the same final-eval block that train.py does:
plain eval, TTA eval, viewer sample, qualitative strips, metrics.json.

    python test_pipeline.py --ckpt outputs/v2/best.pt

Extra --flags forward to the config parser (e.g. --val_tiles 64, --tta false,
--output_dir outputs/v2/test).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import parse_config
from dwdata.loaders import build_finetune_loaders
from eval.metrics import evaluate
from eval.report import export_qualitative, export_viewer_sample, write_metrics_json
from models.heads import DepthWizardNetV2
from train import resolve_hf_token, set_seed


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--ckpt", default="outputs/v2/best.pt")
    args, rest = ap.parse_known_args()

    cfg = parse_config(rest)
    ckpt = Path(args.ckpt).expanduser()
    if not ckpt.is_file():
        sys.exit(f"checkpoint not found: {ckpt}")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.hf_token = resolve_hf_token(cfg)
    set_seed(cfg.seed)

    torch.backends.cuda.matmul.allow_tf32 = cfg.tf32
    torch.backends.cudnn.allow_tf32 = cfg.tf32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()} device={device}")

    model = DepthWizardNetV2(cfg).to(device)
    ck = torch.load(ckpt, map_location=device)
    sd = ck.get("model", ck)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[ckpt] loaded {ckpt}  epoch={ck.get('epoch')} stage={ck.get('stage')}  "
          f"(missing {len(missing)}, unexpected {len(unexpected)})")
    model.eval()

    # ---- validation / test loader only ----------------------------
    _, dl_va = build_finetune_loaders(cfg, cfg.hf_token or None)
    if dl_va is None:
        sys.exit("no validation/test loader was built — check dataset config")
    print(f"[data] test batches={len(dl_va)}")

    t0 = time.time()
    final = {}

    plain = evaluate(model, dl_va, cfg, device, use_tta=False)
    final["final_plain"] = plain
    print(f"[final] plain RMSE={plain['global']['rmse_m']:.3f}  "
          f"MAE={plain['global']['mae_m']:.3f}  d1={plain['global']['delta1']:.3f}  "
          f"balanced={plain['balanced_rmse_m']}")

    if cfg.tta:
        tta = evaluate(model, dl_va, cfg, device, use_tta=True)
        final["final_tta"] = tta
        print(f"[final] TTA   RMSE={tta['global']['rmse_m']:.3f}")

    try:
        export_viewer_sample(model, dl_va, cfg, device)
        export_qualitative(model, dl_va, cfg, device, cfg.n_qualitative)
    except Exception as e:  # noqa: BLE001
        print(f"[report] export skipped: {e}")

    metrics_path = out_dir / "metrics_test.json"
    write_metrics_json(metrics_path, cfg, [], plain["global"]["rmse_m"],
                       (time.time() - t0) / 60, extra=final)
    print(f"\n[done] metrics -> {metrics_path}")
    print(json.dumps(final.get("final_tta", final["final_plain"])["global"], indent=2, default=str))


if __name__ == "__main__":
    main()
