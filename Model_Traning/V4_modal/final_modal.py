"""DepthWizard v5 FINAL fine-tune on Modal, sized for a ~$9 budget.

The Modal port of `v5/lightning/final_h100.sh`: the same flags
(`v5/lightning/final_flags.py`), the same warm start (resume-v4-1.6's best.pt), but
every step that does not need a GPU runs on a 2-core CPU container first, and the
GPU container only trains.  Test / report / ONNX run afterwards on a cheap card.

    cd Model_Traning/V4_modal
    modal deploy -e gpu final_modal.py                       # DW_GPU=H100 (default)
    PY=~/.local/share/uv/tools/modal/bin/python              # the python that has `modal`
    $PY -c "import modal; print(modal.Function.from_name('dw-final', 'prep', environment_name='gpu').spawn().object_id)"
    $PY -c "import modal; print(modal.Function.from_name('dw-final', 'train', environment_name='gpu').spawn(minutes=75).object_id)"
    modal app logs -e gpu dw-final

`deploy` + `spawn`, not `modal run --detach`: killing a local `modal run` client
cancelled a detached run on 2026-09-29.  A spawned call on a deployed app runs to
the end whatever happens to the laptop.

Data: the Kaggle stores, pulled onto the `depthwizard-data` Volume (env `gpu`)
under `kin/datasets/abhaydkale232/<slug>/` (the /kaggle/input layout);
warm start at `prev/best.pt`.  The Volume root also holds v4's stores
(stale-GSD gamus, dfc23_g050, india_labeled); nothing here links them.

Cost ($/h, Modal 2026-09): H100 3.95, A100-80 2.50, A100-40 2.10, L40S 1.95,
L4 0.80; + 0.047 per physical core and 0.008 per GiB.  `train` carries a hard
`timeout`, so a hang cannot spend past it.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
CODE = "/root/dw"
V5 = f"{CODE}/v5"
DATA, RESULTS = "/mnt/depthwizard-data", "/mnt/depthwizard-results"
KIN = f"{DATA}/kin"
PREV = f"{DATA}/prev/best.pt"
RUN = "v5_final_forest"
OUT = f"{RESULTS}/{RUN}"
ENCODER = "facebook/dinov3-vitl16-pretrain-sat493m"
TORCH, TORCH_INDEX = "2.10.0", "https://download.pytorch.org/whl/cu128"

# Chosen at deploy time: `DW_GPU=A100-80GB modal deploy ...` to change the card.
GPU = os.environ.get("DW_GPU", "H100")
TRAIN_TIMEOUT_MIN = int(os.environ.get("DW_TIMEOUT_MIN", "110"))
CORES, MEM_MIB = 2.0, 8192       # the user's budget rule: 2 cores / 8 GiB (v4 used ~4 GiB)

# Read locally at deploy.  Inside a container (including the _bake build step,
# which runs before the code is added) the list is not needed: the image exists.
_REQ_FILE = HERE.parent / "v5" / "requirements.txt"
_REQS = [ln.split("#")[0].strip() for ln in _REQ_FILE.read_text().splitlines()
         if ln.split("#")[0].strip()] if _REQ_FILE.is_file() else []

hf = modal.Secret.from_name("dw-hf", required_keys=["HF_TOKEN"])


def _bake() -> None:
    """The gated encoder in the image: no hub calls on a billed GPU cold start."""
    from huggingface_hub import snapshot_download
    from transformers import AutoImageProcessor
    tok = os.environ["HF_TOKEN"]
    snapshot_download(ENCODER, token=tok)
    try:
        AutoImageProcessor.from_pretrained(ENCODER, token=tok)
    except Exception as e:  # noqa: BLE001 — preprocess.py has a constant fallback
        print(f"[dw] processor not cached ({e}); constant fallback applies")


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install(f"torch=={TORCH}", extra_index_url=TORCH_INDEX)
    .uv_pip_install(*_REQS)
    .env({"HF_HOME": "/root/hf", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
          "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .run_function(_bake, secrets=[hf])
    .add_local_dir(HERE.parent / "v5", V5,
                   ignore=["**/__pycache__", "outputs", "data", "*.pt", "*.onnx*", "**/.pytest_cache"])
    .add_local_file(HERE / "v5_flags.py", f"{CODE}/V4_modal/v5_flags.py")
)

app = modal.App("dw-final", image=image)
data_vol = modal.Volume.from_name("depthwizard-data")
res_vol = modal.Volume.from_name("depthwizard-results")


# --- helpers ----------------------------------------------------------------
def sh(cmd: list[str], cwd: str = V5, env: dict | None = None) -> None:
    print(f"\n[dw] $ {shlex.join(cmd)}", flush=True)
    t = time.time()
    rc = subprocess.run(cmd, cwd=cwd, env={**os.environ, **(env or {})}, check=False).returncode
    if rc:
        raise RuntimeError(f"[dw] exit {rc} after {(time.time() - t) / 60:.1f} min: {cmd[:3]}")
    print(f"[dw] ok ({(time.time() - t) / 60:.1f} min)", flush=True)


def link(kin: str, root: str) -> None:
    """`run_kaggle.sh link`: <root>/<source>/<split> -> the pulled stores."""
    shutil.rmtree(root, ignore_errors=True)
    sh(["bash", "run_kaggle.sh", "link"], env={"DW_KAGGLE_INPUT": kin, "DW_DATA_ROOT": root})
    for s in ("dfc23", "dfc23_g050", "india_labeled"):
        if Path(root, s).exists():
            raise RuntimeError(f"{root}/{s} is linked — the final run must not see it")
    sys.path.insert(0, V5)
    from prepare_data import stale_gsd_stores
    bad = stale_gsd_stores(root)
    if bad:
        raise RuntimeError("stale GSD: " + "; ".join(bad))
    for idx in sorted(Path(root).glob("*/*/index.json")):
        d = json.loads(idx.read_text())
        print(f"[dw]   {idx.parents[1].name}/{idx.parent.name:<6} n={d['n']:>5} "
              f"tile={d['tile_px']} gsd={d['gsd_m']} seg={d.get('has_seg')}", flush=True)


def flags_for(root: str, sizing: dict, out: str, resume: str, **ov) -> list[str]:
    sys.path.insert(0, f"{V5}/lightning")
    sys.path.insert(0, V5)
    import final_flags
    f = final_flags.build(root, sizing, out, resume)
    f.update({k: str(v).lower() if isinstance(v, bool) else str(v) for k, v in ov.items()})
    final_flags.check(f)
    return final_flags.to_argv(f)


def _every(sec: float, fn, stop: threading.Event) -> None:
    def loop():
        while not stop.wait(sec):
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                print(f"[dw] background: {e}", flush=True)
    threading.Thread(target=loop, daemon=True).start()


# --- 1. CPU prep: everything that must not burn GPU minutes -----------------
@app.function(cpu=CORES, memory=MEM_MIB, timeout=90 * 60, volumes={DATA: data_vol})
def prep(bench_steps: int = 40, workers: str = "3", batch: int = 16) -> dict:
    """Link, NEON spike cap, landscape + stretch caches (written next to the
    shards on the Volume, so `train` reuses them), then time the train loader on
    these 2 cores straight off the Volume, and time a Volume -> local copy.
    Those two numbers pick the card and whether `train` stages."""
    root = "/tmp/dwdata"
    link(KIN, root)
    if Path(root, "neon/train/index.json").is_file():
        # before the landscape caches: clean() deletes neon's (they were cut
        # from the uncapped labels)
        sh([sys.executable, "tools/pack_neon.py", "clean", "--data_root", root])
    data_vol.commit()

    import torch
    sys.path.insert(0, V5)
    from config import parse_config
    from dwdata.loaders import build_loaders
    from dwdata.preprocess import PreprocSpec

    res: dict = {"cores": len(os.sched_getaffinity(0))}
    for w in [int(x) for x in workers.split(",")]:
        sizing = {"batch_size": batch, "grad_accum": 1, "compile_model": False,
                  "num_workers": w, "prefetch_factor": 4, "eval_batch_mult": 2}
        cfg = parse_config(flags_for(root, sizing, "/tmp/prep_out", PREV))
        spec = PreprocSpec.from_config(cfg)
        dl_tr, _, _ = build_loaders(cfg, spec)
        data_vol.commit()                       # landscape caches, first pass only
        it = iter(dl_tr)
        for _ in range(5):
            next(it)
        t, n = time.time(), 0
        for _ in range(bench_steps):
            b = next(it)
            n += int(next(v for v in b.values() if torch.is_tensor(v)).shape[0])
        ips = n / (time.time() - t)
        res[f"loader_img_s_w{w}"] = round(ips, 1)
        print(f"[bench] workers={w} batch={batch}: {ips:.1f} img/s off the Volume", flush=True)
        del it, dl_tr

    # Volume -> local disk, 8 threads, on the biggest train split
    src = max(Path(KIN).glob("datasets/*/*/**/train"), key=lambda p: sum(
        f.stat().st_size for f in p.glob("*.npy")))
    files = sorted(src.glob("*_rgb.npy"))[:8]
    os.makedirs("/tmp/stage_probe", exist_ok=True)
    t = time.time()
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda f: shutil.copyfile(f, f"/tmp/stage_probe/{f.name}"), files))
    gb = sum(f.stat().st_size for f in files) / 1e9
    res["stage_MB_s"] = round(gb * 1e3 / (time.time() - t))
    print(f"[bench] Volume -> local: {gb:.1f} GB at {res['stage_MB_s']} MB/s", flush=True)
    Path(DATA, "prep_bench.json").write_text(json.dumps(res, indent=1))
    data_vol.commit()
    print(f"[dw] prep done: {res}", flush=True)
    return res


# --- 2. GPU: train only -----------------------------------------------------
def _stage(dst: str) -> str:
    """train + val splits only (test is scored later, off the GPU box)."""
    t = time.time()
    jobs = []
    for f in Path(KIN).rglob("*"):
        rel = f.relative_to(KIN)
        if f.is_file() and "test" not in rel.parts and ".part_" not in str(rel) \
                and f.name != ".done":
            jobs.append((f, Path(dst) / rel))
    for _, d in jobs:
        d.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(lambda p: shutil.copyfile(*p), jobs))
    gb = sum(d.stat().st_size for _, d in jobs) / 1e9
    print(f"[dw] staged {len(jobs)} files, {gb:.0f} GB in {(time.time() - t) / 60:.1f} min", flush=True)
    return dst


@app.function(gpu=GPU, cpu=CORES, memory=MEM_MIB, ephemeral_disk=512 * 1024,
              timeout=TRAIN_TIMEOUT_MIN * 60, secrets=[hf],
              volumes={DATA: data_vol.with_mount_options(read_only=True), RESULTS: res_vol})
def train(minutes: float = 75, stage: bool = True, batch: int = 16, accum: int = 2,
          workers: int = 3, crops_per_epoch: int = 36000, val_tiles: int = 200,
          extra: str = "") -> None:
    """`minutes` is train.py's max_minutes: the LR cosine is sized to it and the
    loop stops there.  Evals, best.pt and last_full.pt land on the results
    Volume as they happen, so the function timeout loses at most an epoch."""
    t0 = time.time()
    kin = _stage("/scratch/kin") if stage else KIN
    root = "/tmp/dwdata"
    link(kin, root)
    Path(OUT).mkdir(parents=True, exist_ok=True)
    resume = f"{OUT}/last_full.pt" if Path(OUT, "last_full.pt").is_file() else PREV
    print(f"[dw] {'resuming' if resume != PREV else 'warm start from'} {resume}", flush=True)
    sizing = {"batch_size": batch, "grad_accum": accum, "compile_model": False,
              "num_workers": workers, "prefetch_factor": 4, "eval_batch_mult": 2}
    Path(OUT, "sizing.json").write_text(json.dumps(sizing, indent=1))
    argv = flags_for(
        root, sizing, OUT, resume,
        max_minutes=minutes, epochs=100, crops_per_epoch=crops_per_epoch,
        val_tiles=val_tiles, eval_every=1, full_state_every=2,
        final_sliding_eval=False, tta=False, test_sources="",
        export_onnx=False, make_report=False, make_figures=False,
    ) + shlex.split(extra)

    stop = threading.Event()
    smi = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw",
         "--format=csv", "-l", "15"], stdout=open(f"{OUT}/gpu_util.csv", "a"))
    _every(600, res_vol.commit, stop)
    print(f"[dw] setup {(time.time() - t0) / 60:.1f} min; training {minutes} min on {GPU}", flush=True)
    try:
        sh([sys.executable, "train.py", *argv])
    finally:
        stop.set()
        smi.terminate()
        res_vol.commit()
        print(f"[dw] total {(time.time() - t0) / 60:.1f} min", flush=True)
