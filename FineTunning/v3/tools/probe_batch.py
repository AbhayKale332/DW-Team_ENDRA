"""Find the largest training batch this card actually holds, by trying it.

Why not a table
---------------
`train_L40S.sh` used to pick the batch from a VRAM bracket — 48 GiB fell in the
">= 40" bucket and got 16, which on an L40S is ~20 GiB used out of 48 and is
exactly the symptom that started this: steps moving, half the card idle.

A bracket cannot know the things that actually set the number: whether the
encoder is frozen (the warmup epochs use a third of the memory of the epochs
after it), whether gradient checkpointing is on, which SDPA kernel gets picked,
what the decoder width is.  So this measures instead.  It builds the real
`DepthWizardNetV3`, **unfreezes the encoder** — the peak that matters is the one
after `--freeze_epochs`, not the cheap warmup — and runs a real
forward/backward/AdamW step at each candidate until one fits inside
`--headroom` of the card.

Two steps per candidate, not one: the first allocates the AdamW exp_avg /
exp_avg_sq state, and the second is the one whose peak is representative.

    python -m tools.probe_batch --headroom 0.88
    -> DW_PROBE_BATCH=32

Prints the chosen value on the last line as `DW_PROBE_BATCH=<n>` so a shell can
read it with a single grep, and exits non-zero if nothing fits.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from config import parse_config  # noqa: E402


def _free(*objs) -> None:
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def try_batch(model, cfg, bs: int, device, amp_dt) -> float:
    """Peak bytes for one real train step at `bs`, or raise torch.OutOfMemoryError."""
    from models.losses import StratumBalancer, compute_losses

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-6)
    bal = StratumBalancer(cfg.stratum_balance_beta, cfg.stratum_weight_clip)
    s = cfg.tile_size
    torch.cuda.reset_peak_memory_stats()
    try:
        for _ in range(2):
            batch = {
                "image": torch.randn(bs, 3, s, s, device=device).to(
                    memory_format=torch.channels_last),
                "target": torch.rand(bs, 1, s, s, device=device) * 20.0,
                "valid": torch.ones(bs, 1, s, s, dtype=torch.bool, device=device),
                "cls": torch.zeros(bs, s, s, dtype=torch.long, device=device),
                "gsd_m": torch.full((bs,), 0.5, device=device),
            }
            with torch.autocast("cuda", dtype=amp_dt, enabled=cfg.amp):
                out = model(batch["image"])
            loss, _ = compute_losses(out, batch, cfg, bal)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            del out, loss, batch
        peak = torch.cuda.max_memory_allocated()
    finally:
        _free(opt, bal)
    return float(peak)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--headroom", type=float, default=0.88,
                    help="fraction of total VRAM the step may peak at")
    ap.add_argument("--max-batch", type=int, default=64)
    ap.add_argument("--min-batch", type=int, default=2)
    args, rest = ap.parse_known_args()

    cfg = parse_config(rest)
    if not torch.cuda.is_available():
        print("no CUDA device — nothing to probe", file=sys.stderr)
        print(f"DW_PROBE_BATCH={cfg.batch_size}")
        return 0

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.backends.cuda.matmul.allow_tf32 = cfg.tf32
    total = torch.cuda.get_device_properties(0).total_memory
    budget = total * args.headroom
    amp_dt = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16

    from models.heads import DepthWizardNetV3

    model = DepthWizardNetV3(cfg).to(device)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    # The number we need is the post-unfreeze peak. Probing the frozen model
    # would hand back a batch that OOMs at epoch freeze_epochs + 1 — which is
    # what the halve-and-resume retry in the shell script was there to catch.
    model.encoder.set_frozen(False)
    model.train()

    name = torch.cuda.get_device_properties(0).name
    print(f"[probe] {name}  {total / 1024 ** 3:.0f} GiB, budget "
          f"{budget / 1024 ** 3:.1f} GiB ({args.headroom:.0%}), "
          f"tile {cfg.tile_size}, encoder unfrozen, "
          f"ckpt={cfg.grad_checkpoint_encoder}", flush=True)

    ladder = [b for b in (64, 56, 48, 40, 36, 32, 28, 24, 20, 16, 12, 8, 6, 4, 2)
              if args.min_batch <= b <= args.max_batch]
    chosen = 0
    for bs in ladder:
        try:
            peak = try_batch(model, cfg, bs, device, amp_dt)
        except torch.OutOfMemoryError:
            print(f"[probe]   batch {bs:>3}: OOM", flush=True)
            _free()
            continue
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            print(f"[probe]   batch {bs:>3}: OOM", flush=True)
            _free()
            continue
        pct = peak / total
        print(f"[probe]   batch {bs:>3}: peak {peak / 1024 ** 3:5.1f} GiB "
              f"({pct:.0%}){'  <- fits' if peak <= budget else '  over budget'}",
              flush=True)
        _free()
        if peak <= budget:
            chosen = bs
            break

    if not chosen:
        print("[probe] nothing on the ladder fits — check tile_size / the model",
              file=sys.stderr)
        return 1
    print(f"[probe] chosen batch {chosen}", flush=True)
    print(f"DW_PROBE_BATCH={chosen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
