"""DepthWizard v4 trainer — single stage, encoder unfrozen after a warmup,
plus an optional mean-teacher branch on unlabeled imagery.

    python train.py --smoke                      # tiny CPU/T4 sanity run
    python train.py                              # full run (config defaults)
    python train.py --datasets gamus --epochs 20
    python train.py --resume outputs/v4/last.pt
    python train.py --datasets gamus,synrs3d,india_unlabeled   # + domain adaptation

Design notes that differ from v2 and why:

* **One stage, not two.**  v2 spent 6 epochs and ~25 % of its budget pretraining
  on SynRS3D through a rotating archive cache whose train loss went *up*
  (9.6 -> 7.0 -> 8.8 -> 10.7 -> 8.8 -> 13.1) because each epoch swapped in a
  different archive with a different height distribution and the LR schedule had
  no idea.  v3 mixes the sources in one weighted sampler instead, so every batch
  sees the same distribution the schedule was built for.

* **Warmup-freeze, then train the encoder.**  `--freeze_epochs` (default 2) lets
  the randomly-initialised decoder stop producing garbage gradients, then the
  whole ViT-L is unfrozen with layer-wise LR decay.  v2 intended to unfreeze 4
  blocks at 70 % and crashed doing it, so its encoder never moved at all.

* **Everything is checkpointed every epoch and the run is wall-clock capped**, so
  a killed box or an exhausted budget still leaves a usable best.pt — and the
  final evaluation is guarded so a failure there cannot destroy the run's output
  the way v2's unfreeze crash did.

* **Mean-teacher on unlabeled imagery (v4).**  Every labelled source available is
  foreign — GAMUS is five US cities, GeoNRW is German, SynRS3D is synthetic — and
  ISRO evaluates on Indian scenes.  When an `india_unlabeled` store exists, an
  EMA teacher predicts a weakly-augmented view of an Indian tile and the student
  is pulled towards it on a strongly-augmented one, weighted by the teacher's own
  bin-distribution sigma so its uncertain pixels are ignored.  No Indian labels
  exist to do this properly; this is the honest second-best, and it is off unless
  the store is there.

* **The run ends with its own artefacts.**  Figures, a self-contained HTML
  validation report and an ONNX graph are produced by the training process, so a
  finished run is a deliverable rather than a checkpoint someone still has to
  write a notebook around.
"""

from __future__ import annotations

import contextlib
import copy
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config, parse_config, safe_config_dict
from dwdata.preprocess import PreprocSpec
from eval.metrics import evaluate, format_line
from eval.report import export_qualitative, export_viewer_sample, write_metrics_json
from models.ema import ModelEMA
from models.heads import DepthWizardNet
from models.losses import StratumBalancer, compute_losses, consistency_loss


class Tee:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", buffering=1)
        self.stream = sys.__stdout__

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


def lr_scale(progress: float, warmup: float = 0.05, final_div: float = 1e2) -> float:
    """OneCycle-shaped multiplier driven by *stage progress*, not a step count.

    Progress is the max of (epoch fraction, wall-clock fraction), so a run that
    hits its time cap still finishes its anneal instead of being chopped off with
    the LR high — and a schedule can never desynchronise from the real step count
    the way v2's precomputed `OneCycleLR(total_steps=...)` did.
    """
    p = min(max(progress, 0.0), 1.0)
    floor = 1.0 / final_div
    if p < warmup:
        return floor + (1.0 - floor) * (p / max(1e-8, warmup))
    q = (p - warmup) / max(1e-8, 1.0 - warmup)
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * q))


def build_optimizer(cfg: Config, model: DepthWizardNet, with_encoder: bool):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad or n.startswith("encoder."):
            continue
        (no_decay if p.ndim <= 1 or n.endswith(".bias") else decay).append(p)
    groups = [
        {"params": decay, "lr": cfg.learning_rate,
         "weight_decay": cfg.weight_decay, "name": "dec"},
        {"params": no_decay, "lr": cfg.learning_rate,
         "weight_decay": 0.0, "name": "dec.nd"},
    ]
    if with_encoder:
        groups += model.encoder.llrd_param_groups(
            cfg.encoder_lr, cfg.llrd, cfg.weight_decay)
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=(0.9, 0.999))
    base = [g["lr"] for g in opt.param_groups]
    n_tr = sum(p.numel() for g in groups for p in g["params"])
    print(f"[optim] {len(groups)} groups, {n_tr / 1e6:.1f}M trainable params, "
          f"lr {min(base):.2e}..{max(base):.2e}")
    return opt, base


class MeanTeacher:
    """An EMA copy of the network, used to pseudo-label unlabeled imagery.

    Distinct from `ModelEMA`, which averages weights for *evaluation*: this one
    has to run a forward pass inside the training step, and swapping weights in
    and out of the live model twice per step to do that would cost more than the
    duplicate parameters.

    Confidence comes from Head B's bin-distribution sigma rather than from a
    second augmented forward pass.  It is a real predictive-uncertainty estimate,
    it is already in metres so it can be thresholded directly, and it costs one
    reduction instead of a whole extra ViT-L forward.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = copy.deepcopy(model).eval()
        for prm in self.model.parameters():
            prm.requires_grad_(False)
        self.decay = float(decay)
        self.n = 0
        self._pairs: list | None = None      # (teacher, live) lists, built lazily

    def _build_pairs(self, model: nn.Module):
        """Pair teacher and live tensors once, for the same reason `ModelEMA`
        does: the per-key Python loop launched ~800 kernels per optimiser step
        on a ViT-L.  `_foreach_*` is two calls and no temporaries.

        Integer buffers are copied outright rather than averaged, so they are
        kept in a separate list and handled on their own.
        """
        sd = model.state_dict()
        tsd = self.model.state_dict()
        tf, lf, ti, li = [], [], [], []
        for k, t in tsd.items():
            v = sd.get(k)
            if v is None or v.shape != t.shape:
                continue
            if t.dtype.is_floating_point and v.dtype == t.dtype:
                tf.append(t)
                lf.append(v)
            else:
                ti.append(t)
                li.append(v)
        self._pairs = [tf, lf, ti, li]
        return self._pairs

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        tf, lf, ti, li = self._pairs or self._build_pairs(model)
        if tf:
            torch._foreach_mul_(tf, d)
            torch._foreach_add_(tf, lf, alpha=1.0 - d)
        for t, v in zip(ti, li):
            t.copy_(v)

    @torch.no_grad()
    def pseudo_label(self, image):
        out = self.model(image)
        return out["fused"].float(), out["b_std"].float()

    def sync_architecture(self, model: nn.Module) -> None:
        """Re-clone after the encoder unfreezes, so buffers/flags stay aligned."""
        self.model = copy.deepcopy(model).eval()
        for prm in self.model.parameters():
            prm.requires_grad_(False)
        self._pairs = None          # the old pairs point at freed tensors


# `rgb_u8` is carried for the qualitative export, which runs off the val
# datasets, never off a training batch — shipping it to the GPU every step was
# ~25 MB/step of pure H2D traffic.  In `gpu_augment` mode the train loader does
# not produce it at all.
_HOST_ONLY = ("rgb_u8",)


def _to_device(batch: dict, device, skip: tuple = _HOST_ONLY) -> dict:
    return {k: (v.to(device, non_blocking=True)
                if (torch.is_tensor(v) and k not in skip) else v)
            for k, v in batch.items()}


def main() -> None:
    cfg = parse_config()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = Tee(out_dir / "run.log")
    cfg.hf_token = resolve_hf_token(cfg)
    set_seed(cfg.seed)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.backends.cuda.matmul.allow_tf32 = cfg.tf32
    torch.backends.cudnn.allow_tf32 = cfg.tf32
    with contextlib.suppress(Exception):
        torch.set_float32_matmul_precision(cfg.matmul_precision)
    if cfg.sdp_flash:
        # ViT-L at 512 px is 1024 tokens of pure attention; without this torch
        # can silently pick the math kernel and materialise a
        # (B, heads, 1024, 1024) score matrix — 2.7 GB a layer at batch 24, and
        # roughly half the step time.
        with contextlib.suppress(Exception):
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)

    n_gpu = torch.cuda.device_count()
    device = torch.device("cuda" if n_gpu else "cpu")
    print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()} gpus={n_gpu}")
    for i in range(n_gpu):
        p = torch.cuda.get_device_properties(i)
        print(f"[env]   GPU{i}: {p.name}  {p.total_memory / 1024 ** 3:.0f} GiB")
    if not cfg.hf_token:
        print("[warn] no HF token — gated DINOv3 / GAMUS access will fail")

    # ---- the preprocessing contract, resolved once and pinned to the run ----
    spec = PreprocSpec.from_config(cfg)
    print(f"[preproc] {spec.canonical_gsd_m} m/px, tile {spec.tile_size}, "
          f"stretch={spec.radiometric_stretch}, mean={tuple(round(v, 3) for v in spec.mean)}")
    (out_dir / "config.json").write_text(json.dumps(safe_config_dict(cfg), indent=2))
    (out_dir / "preproc.json").write_text(json.dumps(spec.to_dict(), indent=2))

    from dwdata.loaders import CyclicLoader, build_loaders, build_unlabeled_loader

    dl_tr, dl_va, full_ds = build_loaders(cfg, spec)
    dl_un = build_unlabeled_loader(cfg, spec)
    un_iter = CyclicLoader(dl_un) if dl_un is not None else None

    # Jitter + normalisation for whole batches, on the card.  `gpu_prep` is a
    # no-op on a batch that already carries a normalised `image`, so the eval
    # and export paths work either way.
    gpu_prep = None
    if cfg.gpu_augment:
        from dwdata.gpu_aug import GpuPreproc

        gpu_prep = GpuPreproc(spec, cfg, device)
        print(f"[perf] GPU augmentation on — workers hand over uint8 "
              f"({cfg.tile_size}x{cfg.tile_size}x3 = "
              f"{cfg.tile_size ** 2 * 3 / 1e6:.1f} MB/sample instead of "
              f"{cfg.tile_size ** 2 * 3 * 4 / 1e6:.1f} MB)")

    model = DepthWizardNet(cfg).to(device)
    if cfg.channels_last and n_gpu:
        model = model.to(memory_format=torch.channels_last)
    if cfg.resume and Path(cfg.resume).is_file():
        ck = torch.load(cfg.resume, map_location=device, weights_only=False)
        miss, unexp = model.load_state_dict(ck.get("model", ck), strict=False)
        print(f"[resume] {cfg.resume}: missing {len(miss)}, unexpected {len(unexp)}")
    if cfg.compile_model and n_gpu:
        try:
            model = torch.compile(model, dynamic=False)
            print("[model] torch.compile enabled")
        except Exception as e:  # noqa: BLE001
            print(f"[model] torch.compile unavailable: {e}")

    core = model._orig_mod if hasattr(model, "_orig_mod") else model
    teacher = MeanTeacher(core, cfg.teacher_ema) if un_iter is not None else None
    if teacher is not None:
        teacher.model.to(device)
        print(f"[mt] mean-teacher enabled (decay {cfg.teacher_ema}, "
              f"conf <= {cfg.consistency_conf_m} m, ramp {cfg.consistency_rampup_epochs} ep)")
    balancer = StratumBalancer(cfg.stratum_balance_beta, cfg.stratum_weight_clip)
    ema = ModelEMA(core, cfg.ema_decay) if cfg.ema_decay > 0 else None
    scaler = torch.amp.GradScaler(
        "cuda", enabled=cfg.amp and n_gpu > 0 and cfg.amp_dtype == "fp16")
    amp_dt = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16

    opt, base_lrs = build_optimizer(cfg, core, with_encoder=cfg.freeze_epochs <= 0)
    if cfg.freeze_epochs <= 0:
        core.encoder.set_frozen(False)

    history: list = []
    best = float("inf")
    t0 = time.time()
    stop = False

    for epoch in range(1, cfg.epochs + 1):
        # ---- unfreeze the encoder once the decoder has warmed up ----------
        if cfg.freeze_epochs and epoch == cfg.freeze_epochs + 1:
            core.encoder.set_frozen(False)
            opt, base_lrs = build_optimizer(cfg, core, with_encoder=True)
            if ema is not None:
                ema = ModelEMA(core, cfg.ema_decay)   # shadow now covers the encoder
            if teacher is not None:
                teacher.sync_architecture(core)
                teacher.model.to(device)
            if n_gpu:
                torch.cuda.empty_cache()

        model.train()
        core.encoder.train()
        opt.zero_grad(set_to_none=True)
        # Built once per epoch: the old inline comprehension walked every
        # parameter of a 300 M-param ViT-L on every optimiser step.
        trainable = [p for p in model.parameters() if p.requires_grad]
        nb = 0
        n_steps = max(1, len(dl_tr))
        # Throughput, measured between log lines rather than from the epoch
        # start, so the number reacts to a loader that falls behind instead of
        # being averaged flat by a fast first few steps.  This is the one line
        # that says whether the card is actually being fed.
        t_win = time.time()
        s_win = 0
        # Losses accumulate on the device and are read once, at the end of the
        # epoch, for the same reason.
        run_loss_t = torch.zeros((), device=device, dtype=torch.float64)
        n_bad = 0

        for step, batch in enumerate(dl_tr):
            progress = max(
                (epoch - 1 + step / n_steps) / max(1, cfg.epochs),
                ((time.time() - t0) / 60.0) / max(1e-8, cfg.max_minutes),
            )
            sc = lr_scale(progress, cfg.warmup_frac)
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * sc

            batch = _to_device(batch, device)
            if gpu_prep is not None:
                batch = gpu_prep(batch, train=True)
            ctx = (torch.autocast("cuda", dtype=amp_dt)
                   if (cfg.amp and n_gpu) else contextlib.nullcontext())
            with ctx:
                out = model(batch["image"])
            loss, stats = compute_losses(out, batch, cfg, balancer)

            # The non-finite guard used to read `float(stats["loss"])` here,
            # which is a device sync on EVERY micro-step.  The comment said the
            # stall was free because the backward was already queued — it is
            # not: the sync stops the host from running ahead, so the next
            # batch's H2D copy, GPU augmentation and kernel launches cannot
            # overlap the current step's compute.  On a card that is otherwise
            # kept fed that is several percent of the run.
            #
            # The check now happens once per *optimiser* step, on the gradient
            # norm `clip_grad_norm_` already computes.  That is also the more
            # correct place: a non-finite micro-batch poisons the whole
            # accumulated gradient anyway, and the old code responded by
            # zeroing every accumulated micro-batch, good ones included.
            # `stats["loss"]` is already a detached 0-d tensor.  nan_to_num
            # keeps one poisoned batch from turning the whole epoch's reported
            # train_loss into NaN — the count of skipped steps is reported
            # separately, so nothing is hidden.
            run_loss_t += torch.nan_to_num(stats["loss"].double(), 0.0, 0.0, 0.0)
            nb += 1

            (scaler.scale(loss / cfg.grad_accum) if scaler.is_enabled()
             else loss / cfg.grad_accum).backward()

            # ---- unlabeled branch: pull the student towards the EMA teacher --
            # Backwarded *separately*, after the labeled graph has been freed.
            # Summing the two losses and backwarding once is the same gradient,
            # but it holds both sets of activations live at the same time —
            # (batch + unlabeled_batch) images of a ViT-L at 512 px — and that
            # is what turns a comfortable 58 GB step into an OOM.  Two backwards
            # into the same accumulated .grad cost one extra graph traversal and
            # peak at max(labeled, unlabeled) instead of their sum.
            if teacher is not None and step % max(1, cfg.consistency_every) == 0:
                ramp = min(1.0, epoch / max(1, cfg.consistency_rampup_epochs))
                ub = _to_device(un_iter.next(), device)
                if gpu_prep is not None:
                    ub = gpu_prep.two_view(ub)
                with torch.no_grad(), ctx:
                    t_pred, t_std = teacher.pseudo_label(ub["image_weak"])
                with ctx:
                    s_pred = model(ub["image_strong"])["fused"]
                l_con, kept = consistency_loss(
                    s_pred.float(), t_pred, t_std, cfg.consistency_conf_m)
                l_con = cfg.w_consistency * ramp * l_con
                (scaler.scale(l_con / cfg.grad_accum) if scaler.is_enabled()
                 else l_con / cfg.grad_accum).backward()
                run_loss_t += torch.nan_to_num(l_con.detach().double(), 0.0, 0.0, 0.0)
                # Device-side adds, so the logged `loss=` is the total the
                # optimiser actually saw without costing a sync to say so.
                stats["loss"] = stats["loss"] + l_con.detach()
                stats["con"] = l_con.detach()
                stats["con_keep"] = kept

            if (step + 1) % cfg.grad_accum == 0:
                if scaler.is_enabled():
                    scaler.unscale_(opt)
                gn = nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
                if torch.isfinite(gn):                 # the one sync per step
                    if scaler.is_enabled():
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    if ema is not None:
                        ema.update(core)
                    if teacher is not None:
                        teacher.update(core)
                else:
                    n_bad += 1
                    if n_bad <= 5 or n_bad % 50 == 0:
                        print(f"  [e{epoch} s{step}] non-finite gradient "
                              f"(#{n_bad}) — step skipped", flush=True)
                opt.zero_grad(set_to_none=True)
            s_win += batch["target"].shape[0]
            if step % 25 == 0:
                # Only here do the remaining stats get pulled off the device.
                st = {k: float(v) for k, v in stats.items()}
                el = (time.time() - t0) / 60
                dt = max(1e-6, time.time() - t_win)
                ips, t_win, s_win = s_win / dt, time.time(), 0
                vram = (f" vram={torch.cuda.max_memory_allocated() / 1024 ** 3:.0f}G"
                        if n_gpu else "")
                con = (f" con={st['con']:.3f}/{st['con_keep']:.0%}"
                       if "con" in st else "")
                print(f"  e{epoch} s{step}/{n_steps} loss={st['loss']:.3f} "
                      f"(l1={st['l1']:.2f} sil={st['silog']:.2f} "
                      f"nrm={st['normal']:.3f} flat={st['flat']:.3f} "
                      f"bin={st['bin']:.2f} ent={st['bin_ent']:.2f} "
                      f"seg={st['seg']:.2f} a={st['alpha']:.2f}{con}) "
                      f"lr={opt.param_groups[0]['lr']:.2e} "
                      f"{ips:.1f} img/s{vram} {el:.0f}min", flush=True)
            if (time.time() - t0) / 60 > cfg.max_minutes:
                print(f"  wall-clock cap {cfg.max_minutes:.0f} min hit")
                stop = True
                break

        run_loss = float(run_loss_t)
        if n_bad:
            print(f"  e{epoch}: {n_bad} optimiser step(s) skipped on non-finite "
                  f"gradients")
        rec = {"epoch": epoch, "train_loss": run_loss / max(1, nb),
               "encoder_frozen": core.encoder.frozen,
               "minutes": (time.time() - t0) / 60}

        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs or stop:
            if n_gpu:
                torch.cuda.empty_cache()
            swap = ema.swapped(ema, core) if ema is not None else contextlib.nullcontext()
            with swap:
                m = evaluate(model, dl_va, cfg, device, use_tta=False,
                             gpu_prep=gpu_prep)
                rec["val"] = m
                print(f"  eval e{epoch}  {format_line(m)}")
                if m["global"]["rmse_m"] < best:
                    best = m["global"]["rmse_m"]
                    _save(out_dir / "best.pt", core, spec, cfg, epoch, m, ema)
                    print(f"  new best {best:.3f} m -> best.pt")

        history.append(rec)
        _save(out_dir / "last.pt", core, spec, cfg, epoch, rec.get("val"), ema)
        write_metrics_json(out_dir / "metrics.json", cfg, spec, history, best,
                           (time.time() - t0) / 60)
        if stop:
            break

    # ---- final evaluation --------------------------------------------
    final = {}
    if (out_dir / "best.pt").is_file():
        ck = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
        core.load_state_dict(ck["model"], strict=False)
        print(f"[final] evaluating best.pt (epoch {ck.get('epoch')})")

    for name, fn in (
        ("final_plain", lambda: evaluate(model, dl_va, cfg, device, use_tta=False,
                                         gpu_prep=gpu_prep)),
        ("final_tta", (lambda: evaluate(model, dl_va, cfg, device, use_tta=True,
                                        gpu_prep=gpu_prep))
         if cfg.tta else None),
    ):
        if fn is None:
            continue
        try:
            final[name] = fn()
            print(f"[final] {name:12s} {format_line(final[name])}")
        except Exception:  # noqa: BLE001
            print(f"[final] {name} failed:\n{traceback.format_exc()}")

    if cfg.final_sliding_eval:
        try:
            from eval.sliding import sliding_eval

            final["final_sliding_tta"] = sliding_eval(
                model, full_ds, cfg, spec, device, tta=cfg.tta)
            print(f"[final] sliding+TTA {format_line(final['final_sliding_tta'])}")
            print("        ^ full val tiles at native GSD through the inference "
                  "path — this is the number to quote")
        except Exception:  # noqa: BLE001
            print(f"[final] sliding eval failed:\n{traceback.format_exc()}")

    samples = []
    try:
        export_viewer_sample(model, full_ds, cfg, spec, device)
        samples = export_qualitative(model, full_ds, cfg, spec, device,
                                     cfg.n_qualitative) or []
    except Exception:  # noqa: BLE001
        print(f"[report] export skipped:\n{traceback.format_exc()}")

    write_metrics_json(out_dir / "metrics.json", cfg, spec, history, best,
                       (time.time() - t0) / 60, extra=final)

    # ---- deliverables: figures, the validation report, the ONNX graph ----
    # Each is guarded separately.  A run that trains for five hours must not lose
    # its report because matplotlib choked, and must not lose its ONNX export
    # because the report did.
    if cfg.make_figures:
        try:
            import json as _json

            from viz.figures import make_all

            make_all(out_dir / "figures",
                     _json.loads((out_dir / "metrics.json").read_text()),
                     samples[:6], cfg.canonical_gsd_m)
        except Exception:  # noqa: BLE001
            print(f"[viz] figures skipped:\n{traceback.format_exc()}")

    if cfg.make_report:
        try:
            from viz.report_html import build as build_report

            build_report(out_dir)
        except Exception:  # noqa: BLE001
            print(f"[report] html skipped:\n{traceback.format_exc()}")

    if cfg.export_onnx and (out_dir / "best.pt").is_file():
        try:
            from infer.export_onnx import export as export_onnx

            export_onnx(str(out_dir / "best.pt"), str(out_dir / "depthwizard.onnx"),
                        opset=cfg.onnx_opset, hf_token=cfg.hf_token)
        except Exception:  # noqa: BLE001
            print(f"[onnx] export skipped:\n{traceback.format_exc()}")

    print("\n" + "=" * 64)
    print(f"DONE — best centre-crop val RMSE {best:.3f} m "
          f"({(time.time() - t0) / 60:.0f} min)")
    fs = final.get("final_sliding_tta")
    if fs:
        print(f"       quote this: sliding+TTA {format_line(fs)}")
    print(f"       artefacts -> {out_dir}")
    print("=" * 64)

    if cfg.make_zip:
        try:
            from package_results import build_zip, write_env_files

            write_env_files(cfg.output_dir)
            print(f"[package] {build_zip(cfg.output_dir)}")
        except Exception as e:  # noqa: BLE001
            print(f"[package] failed: {e}  (artifacts still in {cfg.output_dir})")


def _save(path: Path, core, spec: PreprocSpec, cfg, epoch: int, metrics, ema):
    """Checkpoints carry their own preprocessing contract.

    This is what makes `infer/predict.py --ckpt X` reproduce training-time inputs
    without being handed a config: the recipe travels with the weights.
    """
    sd = ema.state_dict() if ema is not None else None
    model_sd = core.head_state_dict() if core.encoder.frozen else core.full_state_dict()
    if sd is not None:
        model_sd = {k: sd.get(k, v) for k, v in model_sd.items()}
    torch.save({
        "model": model_sd,
        "preproc": spec.to_dict(),
        "config": safe_config_dict(cfg),
        "epoch": epoch,
        "metrics": metrics,
        "encoder_included": not core.encoder.frozen,
    }, path)


if __name__ == "__main__":
    main()
