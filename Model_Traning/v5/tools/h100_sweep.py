"""Measure, don't guess: size the final run to the card it is on.

    python tools/h100_sweep.py --data_root /tmp/dwdata --resume /tmp/prev/best.pt \
        --out ~/sizing.json [--batches 16,24,32,40,48] [--ceiling_gb 72]

Each candidate is a real `train.py` process with the final run's flags
(`lightning/final_flags.py`) plus `--bench_steps`: same data, same checkpoint,
same model, same loss.  After a warm-up (which is where `torch.compile`
compiles), it times N micro-steps and prints one `[bench] {json}` line with
img/s, the share of time blocked on the loader, peak *reserved* VRAM and the
largest eval batch that fits.  While it runs, this script polls `nvidia-smi`
and averages GPU utilisation and power over exactly the timed window.

Pick rule: the highest img/s whose peak reserved memory is <= `--ceiling_gb`
(v3 ran micro-batch 32 at 74 GB of 80 and died at epoch 7 on fragmentation
drift, so the last ~8 GB are not usable headroom).  Within 2 %, the larger
batch wins: fewer optimiser / EMA steps per image.

What this measured before (v3 / v4 on an H100): micro-batch 16 -> 42 GB,
44.7 img/s; 32 -> 74 GB, 48.0 img/s; GPU util 96 %.  So the card is compute
bound early and extra VRAM buys little speed by itself; the sweep settles
how much, on this model (v5 adds the detail branch), instead of assuming it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

V5 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V5 / "lightning"))
sys.path.insert(0, str(V5))

REF_EFFECTIVE_BATCH = 32


def usable_cores() -> int:
    """Cores this process may run on.  `os.cpu_count()` reports the host's
    (93 on a Modal H100 box against a 4-core reservation)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 4


def default_workers() -> int:
    return max(4, min(32, usable_cores() - 2))


def accum_for(batch: int) -> int:
    return max(1, round(REF_EFFECTIVE_BATCH / batch))


class GpuPoller(threading.Thread):
    """(t, util %, power W, mem MiB) samples from nvidia-smi, every `dt` s."""

    def __init__(self, dt: float = 0.5):
        super().__init__(daemon=True)
        self.dt, self.samples, self._stop_ev = dt, [], threading.Event()

    def run(self):
        q = ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw,memory.used",
             "--format=csv,noheader,nounits", "-i", "0"]
        while not self._stop_ev.is_set():
            try:
                out = subprocess.run(q, capture_output=True, text=True, timeout=5).stdout
                u, p, m = (float(x) for x in out.strip().split(",")[:3])
                self.samples.append((time.time(), u, p, m))
            except Exception:  # noqa: BLE001 — no GPU / no nvidia-smi: just no util column
                pass
            self._stop_ev.wait(self.dt)

    def stop(self):
        self._stop_ev.set()

    def window(self, t0: float, t1: float) -> dict:
        s = [x for x in self.samples if t0 <= x[0] <= t1]
        if not s:
            return {}
        n = len(s)
        return {"gpu_util": sum(x[1] for x in s) / n, "power_w": sum(x[2] for x in s) / n,
                "mem_used_gb": max(x[3] for x in s) / 1024, "n_samples": n}


def parse_bench(log: str) -> tuple[dict | None, float | None]:
    rep, start = None, None
    for ln in log.splitlines():
        if ln.startswith("[bench] window start"):
            start = float(ln.split()[-1])
        elif ln.startswith("[bench] {"):
            rep = json.loads(ln[len("[bench] "):])
    return rep, start


def run_one(args, batch: int, compile_: bool, workers: int, prefetch: int) -> dict:
    from final_flags import build, to_argv

    accum = accum_for(batch)
    steps = max(3 * accum, (48 if batch <= 16 else 36))
    sizing = {"batch_size": batch, "grad_accum": accum, "compile_model": compile_,
              "num_workers": workers, "prefetch_factor": prefetch, "eval_batch_mult": 2}
    out = Path(tempfile.mkdtemp(prefix=f"sweep_b{batch}_c{int(compile_)}_", dir=args.work))
    fl = build(args.data_root, sizing, str(out), args.resume)
    fl.update({
        "bench_steps": str(steps), "bench_vram_ceiling_gb": str(args.ceiling_gb),
        "epochs": "1", "max_minutes": "60", "eval_every": "1000",
        # enough crops that the sampler never runs out before the window ends
        "crops_per_epoch": str(batch * (steps + max(10, 2 * accum) + 8)),
        "test_sources": "", "make_figures": "false", "make_report": "false",
        "export_onnx": "false", "save_full_state": "false",
    })
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               PYTORCH_ALLOC_CONF="expandable_segments:True", OMP_NUM_THREADS="1")
    poll = GpuPoller()
    poll.start()
    t_start = time.time()
    p = subprocess.run([sys.executable, "train.py", *to_argv(fl)], cwd=str(V5), env=env,
                       capture_output=True, text=True, timeout=args.timeout_s)
    poll.stop()
    log = p.stdout + "\n" + p.stderr
    (out / "sweep.log").write_text(log)
    rep, start = parse_bench(log)
    row = {"batch_size": batch, "grad_accum": accum, "compile_model": compile_,
           "num_workers": workers, "prefetch_factor": prefetch,
           "rc": p.returncode, "wall_s": time.time() - t_start, "log": str(out / "sweep.log")}
    if rep is None:
        row["status"] = "oom" if "OutOfMemory" in log or "out of memory" in log else "failed"
        row["tail"] = log.strip().splitlines()[-5:]
        return row
    row.update(rep)
    row.update(poll.window(start or rep["window_end"] - rep["seconds"], rep["window_end"]))
    row["status"] = "over" if rep.get("peak_reserved_gb", 0) > args.ceiling_gb else "ok"
    return row


def pick(rows: list[dict]) -> dict | None:
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        return None
    top = max(r["img_s"] for r in ok)
    near = [r for r in ok if r["img_s"] >= 0.98 * top]
    return max(near, key=lambda r: (r["batch_size"], r["img_s"]))


def fmt(r: dict) -> str:
    return (f"b={r['batch_size']:>3} acc={r['grad_accum']} compile={int(r['compile_model'])} "
            f"w={r['num_workers']} pf={r['prefetch_factor']}  "
            + (f"{r['img_s']:6.1f} img/s  wait={r.get('wait_frac', 0):4.0%}  "
               f"peak={r.get('peak_reserved_gb', 0):5.1f} GB  util={r.get('gpu_util', float('nan')):5.1f} %  "
               f"{r.get('power_w', float('nan')):5.0f} W  eval_mult={r.get('eval_batch_mult', '-')}"
               if "img_s" in r else "") + f"  [{r['status']}]")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--resume", required=True, help="the warm-start best.pt")
    ap.add_argument("--out", required=True, help="sizing.json to write")
    ap.add_argument("--work", default=tempfile.gettempdir())
    ap.add_argument("--batches", default="16,24,32,40,48")
    ap.add_argument("--ceiling_gb", type=float, default=72.0)
    ap.add_argument("--compile_top", type=int, default=2,
                    help="also try torch.compile on the N fastest eager batches")
    ap.add_argument("--workers", type=int, default=0, help="0 -> usable cores - 2")
    ap.add_argument("--timeout_s", type=int, default=1200)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out).expanduser()
    if out.is_file() and not a.force:
        print(f"[sweep] {out} exists (use --force to re-measure):\n{out.read_text()}")
        return
    Path(a.work).mkdir(parents=True, exist_ok=True)
    workers = a.workers or default_workers()
    print(f"[sweep] usable cores {usable_cores()} -> num_workers {workers}; "
          f"ceiling {a.ceiling_gb:.0f} GB reserved", flush=True)

    rows = []
    # 1. eager, ascending batch; stop once a batch no longer fits
    for b in (int(x) for x in a.batches.split(",") if x.strip()):
        r = run_one(a, b, False, workers, 4)
        rows.append(r)
        print(f"[sweep] {fmt(r)}", flush=True)
        if r["status"] in ("oom", "over"):
            break
        if r["status"] == "failed":
            print("\n".join(r.get("tail", [])))
            raise SystemExit(f"[sweep] train.py failed at batch {b} — see {r['log']}")
    # 2. torch.compile on the fastest eager batches that fit
    eager_ok = sorted((r for r in rows if r["status"] == "ok"), key=lambda r: -r["img_s"])
    for r0 in eager_ok[:max(0, a.compile_top)]:
        r = run_one(a, r0["batch_size"], True, workers, 4)
        rows.append(r)
        print(f"[sweep] {fmt(r)}", flush=True)
    best = pick(rows)
    if best is None:
        raise SystemExit("[sweep] no configuration fit under the ceiling — see the logs above")
    # 3. is the loader the limit?  More prefetch before blaming the CPU count.
    if best.get("wait_frac", 0) > 0.05:
        for pf in (8,):
            r = run_one(a, best["batch_size"], best["compile_model"], workers, pf)
            rows.append(r)
            print(f"[sweep] {fmt(r)}", flush=True)
        best = pick(rows)
    verdict = []
    if best.get("wait_frac", 0) > 0.05:
        verdict.append(f"CPU-BOUND: {best['wait_frac']:.0%} of each step waits on the loader with "
                       f"{workers} workers on {usable_cores()} cores — pick a Lightning H100 "
                       f"with more vCPUs; the GPU is idle that share of the time")
    if best.get("gpu_util", 100) < 90:
        verdict.append(f"GPU util {best.get('gpu_util', 0):.0f} % in the timed window")
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if ok_rows:
        lo = min(ok_rows, key=lambda r: r["batch_size"])
        if lo is not best and lo["img_s"] > 0:
            verdict.append(f"batch {best['batch_size']} vs {lo['batch_size']}: "
                           f"{100 * (best['img_s'] / lo['img_s'] - 1):+.0f} % img/s, "
                           f"{best.get('peak_reserved_gb', 0) - lo.get('peak_reserved_gb', 0):+.0f} GB")
    chosen = {k: best[k] for k in ("batch_size", "grad_accum", "compile_model",
                                   "num_workers", "prefetch_factor")}
    chosen["eval_batch_mult"] = int(best.get("eval_batch_mult", 2))
    rep = {**chosen, "img_s": best["img_s"], "peak_reserved_gb": best.get("peak_reserved_gb"),
           "gpu_util": best.get("gpu_util"), "wait_frac": best.get("wait_frac"),
           "ceiling_gb": a.ceiling_gb, "verdict": verdict, "rows": rows,
           "fallbacks": [{k: r[k] for k in ("batch_size", "grad_accum", "compile_model")}
                         for r in sorted(ok_rows, key=lambda r: -r["batch_size"])
                         if r["batch_size"] < best["batch_size"]]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(f"[sweep] CHOSEN  {fmt(best)}")
    for v in verdict:
        print(f"[sweep] {v}")
    print(f"[sweep] -> {out}")


if __name__ == "__main__":
    main()
