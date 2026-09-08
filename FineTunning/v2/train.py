"""DepthWizard v2 trainer — SynRS3D pretrain -> GAMUS+GeoNRW fine-tune.

Single GPU (target: 1x RTX PRO 6000, 96 GB).  bf16 autocast, channels_last, no
grad-checkpoint.  Each stage is wall-clock capped and checkpointed so a killed
vast.ai box is resumable.  At the end everything is zipped and shared over a
cloudflared quick tunnel.

    python train.py                       # full run (reads config defaults)
    python train.py --smoke               # tiny CPU/T4 sanity run
    python train.py --skip_pretrain       # GAMUS+GeoNRW only
    python train.py --datasets gamus      # v1-style single-dataset run
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config, parse_config, safe_config_dict
from eval.metrics import evaluate
from eval.report import (
    export_qualitative,
    export_viewer_sample,
    write_metrics_json,
)
from models.heads import DepthWizardNetV2
from models.losses import compute_losses


class Tee:
    """Mirror stdout/stderr to output_dir/run.log.  Proxies any other stream
    attribute (isatty / fileno / encoding / …) to the real stdout so libraries
    that introspect the stream (transformers, tqdm) keep working."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", buffering=1)
        self.stream = sys.stdout

    def write(self, s):
        self.stream.write(s)
        try:
            self.f.write(s)
        except Exception:  # noqa: BLE001
            pass
        return len(s)

    def flush(self):
        self.stream.flush()
        self.f.flush()

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self.__dict__["stream"], name)


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def resolve_hf_token(cfg: Config) -> str:
    if cfg.hf_token:
        return cfg.hf_token
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        if os.environ.get(var):
            return os.environ[var]
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:  # noqa: BLE001
        return ""


def _amp_dtype(cfg: Config):
    return torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16


def build_optimizer(cfg: Config, model: nn.Module, extra_params=None, extra_lr_mult=1.0):
    base = [p for n, p in model.named_parameters()
            if p.requires_grad and not n.startswith("encoder.model.")]
    groups = [{"params": base, "lr": cfg.learning_rate}]
    if extra_params:
        groups.append({"params": extra_params, "lr": cfg.learning_rate * extra_lr_mult})
    return torch.optim.AdamW(groups, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)


def train_stage(
    cfg: Config,
    model: DepthWizardNetV2,
    device,
    dl_tr,
    dl_va,
    *,
    stage: str,
    epochs: int,
    minutes: float,
    history: list,
    unfreeze: bool = False,
    rotate_fn=None,
) -> float:
    """Run one stage; return best val RMSE seen (inf if no val)."""
    use_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and use_cuda and cfg.amp_dtype == "fp16")
    opt = build_optimizer(cfg, model)
    sched = None
    best = float("inf")
    t0 = time.time()
    unfreeze_epoch = round(cfg.unfreeze_at_frac * epochs) if unfreeze and cfg.unfreeze_last_n else None
    out_dir = Path(cfg.output_dir)

    for epoch in range(1, epochs + 1):
        if rotate_fn is not None:
            rotate_fn(epoch)
            dl_tr = rotate_fn.loader  # type: ignore[attr-defined]

        if unfreeze_epoch and epoch == unfreeze_epoch:
            newly = model.encoder.unfreeze_last_n_blocks(cfg.unfreeze_last_n)
            if newly:
                opt = build_optimizer(cfg, model, newly, cfg.unfreeze_lr_mult)

        if sched is None:
            steps = max(1, len(dl_tr) // cfg.grad_accum) * max(1, epochs - epoch + 1)
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=[g["lr"] for g in opt.param_groups],
                total_steps=steps, pct_start=0.1,
            )

        model.train()
        model.encoder.train()
        opt.zero_grad(set_to_none=True)
        run = {"loss": 0.0}
        nb = 0
        stop = False

        for step, batch in enumerate(dl_tr):
            batch = _to_device(batch, device)
            ctx = (
                torch.autocast("cuda", dtype=_amp_dtype(cfg))
                if (cfg.amp and use_cuda)
                else contextlib.nullcontext()
            )
            with ctx:
                out = model(batch["image"])
            loss, stats = compute_losses(out, batch, cfg)
            loss = loss / cfg.grad_accum
            (scaler.scale(loss) if scaler.is_enabled() else loss).backward()

            if (step + 1) % cfg.grad_accum == 0:
                if scaler.is_enabled():
                    scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg.grad_clip
                )
                if scaler.is_enabled():
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
                if sched.last_epoch < sched.total_steps - 1:
                    sched.step()

            run["loss"] += stats["loss"]
            nb += 1
            if step % 20 == 0:
                el = (time.time() - t0) / 60
                print(f"  [{stage}] e{epoch} s{step}/{len(dl_tr)} loss={stats['loss']:.3f} "
                      f"(l1_f={stats['l1_fused']:.2f} bin={stats['bin_ce']:.2f} "
                      f"seg={stats['seg_ce']:.2f}) lr={opt.param_groups[0]['lr']:.2e} {el:.0f}min")
            if (time.time() - t0) / 60 > minutes:
                print(f"  [{stage}] wall-clock cap {minutes:.0f} min hit — stopping stage")
                stop = True
                break

        rec = {"stage": stage, "epoch": epoch, "train_loss": run["loss"] / max(1, nb)}
        if dl_va is not None and (epoch % cfg.eval_every == 0 or epoch == epochs or stop):
            if use_cuda:
                torch.cuda.empty_cache()
            m = evaluate(model, dl_va, cfg, device, use_tta=False)
            rec["val"] = m
            g = m["global"]
            print(f"  [{stage}] eval e{epoch}  RMSE={g['rmse_m']:.3f}  MAE={g['mae_m']:.3f}  "
                  f"r={g['pearson_r']:.3f}  d1={g['delta1']:.3f}  bal={m['balanced_rmse_m']}")
            if g["rmse_m"] < best:
                best = g["rmse_m"]
                torch.save(
                    {"stage": stage, "epoch": epoch, "config": safe_config_dict(cfg),
                     "metrics": m, "model": model.trainable_state_dict()},
                    out_dir / "best.pt",
                )
                print(f"  [{stage}] new best {best:.3f} -> best.pt")

        history.append(rec)
        torch.save({"stage": stage, "epoch": epoch, "model": model.trainable_state_dict()},
                   out_dir / ("stageP_last.pt" if stage == "pretrain" else "last.pt"))
        write_metrics_json(out_dir / "metrics.json", cfg, history, best,
                           (time.time() - t0) / 60)
        if stop:
            break
    return best


def _to_device(batch: dict, device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


class _Rotator:
    """Per-epoch SynRS3D archive rotation, exposes `.loader`."""

    def __init__(self, cfg, ds):
        from dwdata.loaders import pretrain_loader

        self.cfg = cfg
        self.ds = ds
        self._mk = pretrain_loader
        self.loader = None

    def __call__(self, epoch: int):
        n = 2 if not self.cfg.smoke else 1
        self.ds.ensure_ready(n, epoch - 1)
        print(f"[pretrain] epoch {epoch}: {len(self.ds)} tiles ready")
        self.loader = self._mk(self.cfg, self.ds)


def main() -> None:
    cfg = parse_config()
    out_dir = Path(cfg.output_dir)
    sys.stdout = sys.stderr = Tee(out_dir / "run.log")
    cfg.hf_token = resolve_hf_token(cfg)
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = False

    n_gpu = torch.cuda.device_count()
    device = torch.device("cuda" if n_gpu else "cpu")
    print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()} gpus={n_gpu}")
    for i in range(n_gpu):
        print(f"[env]   GPU{i}: {torch.cuda.get_device_name(i)}")
    if not cfg.hf_token:
        print("[warn] no HF token — gated DINOv3 / GAMUS downloads will fail")
    (out_dir / "config.json").write_text(json.dumps(safe_config_dict(cfg), indent=2))

    model = DepthWizardNetV2(cfg).to(device)
    if cfg.channels_last and n_gpu:
        model = model.to(memory_format=torch.channels_last)
    if cfg.init_from and Path(cfg.init_from).is_file():
        sd = torch.load(cfg.init_from, map_location=device).get("model", {})
        missing, unexp = model.load_state_dict(sd, strict=False)
        print(f"[init] loaded {cfg.init_from}  (missing {len(missing)}, unexpected {len(unexp)})")
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable {n_tr/1e6:.1f}M params")

    history: list = []
    t_start = time.time()

    # ---- Stage P: pretrain -----------------------------------------
    if cfg.pretrain_dataset == "synrs3d" and not cfg.skip_pretrain:
        try:
            from dwdata.loaders import build_pretrain_dataset

            ds = build_pretrain_dataset(cfg, cfg.hf_token or None)
            rot = _Rotator(cfg, ds)
            train_stage(cfg, model, device, None, None, stage="pretrain",
                        epochs=cfg.pretrain_epochs, minutes=cfg.pretrain_minutes,
                        history=history, rotate_fn=rot)
        except Exception as e:  # noqa: BLE001 — pretrain is best-effort
            print(f"[pretrain] skipped due to error: {e}")

    # ---- Stage F: finetune ----------------------------------------
    from dwdata.loaders import build_finetune_loaders

    dl_tr, dl_va = build_finetune_loaders(cfg, cfg.hf_token or None)
    print(f"[data] finetune train batches={len(dl_tr)}  "
          f"val batches={len(dl_va) if dl_va else 0}")
    best = train_stage(cfg, model, device, dl_tr, dl_va, stage="finetune",
                       epochs=cfg.finetune_epochs, minutes=cfg.finetune_minutes,
                       history=history, unfreeze=True)

    # ---- Final eval (+ TTA) --------------------------------------
    final = {}
    if dl_va is not None:
        plain = evaluate(model, dl_va, cfg, device, use_tta=False)
        final["final_plain"] = plain
        print(f"[final] plain RMSE={plain['global']['rmse_m']:.3f}")
        if cfg.tta:
            tta = evaluate(model, dl_va, cfg, device, use_tta=True)
            final["final_tta"] = tta
            print(f"[final] TTA   RMSE={tta['global']['rmse_m']:.3f}")
        try:
            export_viewer_sample(model, dl_va, cfg, device)
            export_qualitative(model, dl_va, cfg, device, cfg.n_qualitative)
        except Exception as e:  # noqa: BLE001
            print(f"[report] export skipped: {e}")

    write_metrics_json(out_dir / "metrics.json", cfg, history, best,
                       (time.time() - t_start) / 60, extra=final)

    print("\n" + "=" * 64)
    print(f"DONE — best val RMSE {best:.3f} m   ({(time.time()-t_start)/60:.0f} min)")
    print("=" * 64)

    # ---- Package + share --------------------------------------
    if cfg.make_zip:
        try:
            from package_results import (
                build_zip,
                keep_alive,
                share_via_cloudflared,
                write_env_files,
            )

            write_env_files(cfg.output_dir)
            zp = build_zip(cfg.output_dir)
            if cfg.share_cloudflared:
                url, procs = share_via_cloudflared(str(zp), cfg.keep_alive_minutes)
                keep_alive(procs, cfg.keep_alive_minutes if url else 0.0)
        except Exception as e:  # noqa: BLE001
            print(f"[package] failed: {e}  (artifacts still in {cfg.output_dir})")


if __name__ == "__main__":
    main()
