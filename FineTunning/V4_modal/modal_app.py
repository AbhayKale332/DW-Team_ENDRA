"""DepthWizard v4 on Modal — 1x H100.

Glue only. The image mounts ../V4_Kaggle verbatim at /root/dw and shells out to
its train.py / prepare_data.py / pack_gamus_png.py. Nothing here is duplicated
from that tree, and its DDP code short-circuits at world_size==1, so we run a
plain `python train.py` — no torchrun.

    modal run modal_app.py::show_tuning        # $0
    modal run modal_app.py::check              # image + 175 tests
    modal run --detach modal_app.py::prepare   # build the pack, once
    modal run modal_app.py::smoke              # ~10 min
    modal run --detach modal_app.py::train
    modal run --detach modal_app.py::finalize  # exports only, from metrics.json

Secrets (you create these):
    modal secret create dw-hf      HF_TOKEN=hf_...
    modal secret create dw-kaggle  KAGGLE_USERNAME=... KAGGLE_KEY=...
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import modal

CODE = "/root/dw"
DATA, RESULTS, SCRATCH = "/data", "/results", "/scratch/dwdata"
ENCODER = "facebook/dinov3-vitl16-pretrain-sat493m"

# V4_Kaggle/requirements.txt omits torch on purpose ("the host's CUDA build is
# kept"). Modal has no host build, so we pin it. This is what outputs/v4/run.log
# records. If the wheel is gone, bump it and re-run `check`.
TORCH = "2.10.0"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"

_LOCAL = Path(__file__).parent.parent / "V4_Kaggle"
_REQS = [
    ln.split("#")[0].strip()
    for ln in (_LOCAL / "requirements.txt").read_text().splitlines()
    if ln.split("#")[0].strip()
]

# --- the profile ----------------------------------------------------------
# Every key is a config.py dataclass field; parse_config auto-generates one
# --<field> per field, so this dict IS the command line. Rationale for each
# number lives in V4_Kaggle/README.md 5.2 — not repeated here.
#
# Two that are specific to this port:
#   batch_size 24  — config.py:184-190 measured 32 -> 74 GB then OOM at e7.
#   session_minutes 0 — that second cap truncating the cosine is what cost
#                       v4-2 0.36 m. One Modal container has no session to survive.
FLAGS = {
    "data_root": SCRATCH, "output_dir": f"{RESULTS}/v4m",
    # gamus MUST stay first: loaders.py:31-43 selects best.pt on the first
    # source that has a val split.
    "datasets": "gamus,synrs3d_g1,synrs3d_g05,dfc23_g050",
    "amp_dtype": "bf16",                 # Hopper: disables the GradScaler path entirely
    "grad_checkpoint_encoder": "false", "grad_checkpoint_decoder": "false",
    "batch_size": "24", "grad_accum": "1", "eval_batch_mult": "2",
    "num_workers": "12", "prefetch_factor": "6", "compile_model": "false",
    "encoder_unfreeze_blocks": "0", "llrd": "0.90",
    "epochs": "30", "eval_every": "1",
    "max_minutes": "150", "session_minutes": "0",
    "save_full_state": "true", "full_state_every": "5",
    "make_zip": "false",                 # build_zip duplicates ~4 GB onto a billed volume
    # The held-out split: 2861 gamus/test tiles, deliberately absent from
    # "datasets" above so no epoch and no in-training eval touches them. Scored
    # once at the end with best.pt frozen -> metrics.json test_gamus_test_*.
    # Every final_* number is scored on the gamus/val prefix best.pt was
    # SELECTED on; this is the one that is not. See V4_modal/README.md.
    "test_sources": "gamus:test",
    "test_tiles": "0",                   # 0 = all 2861, centre crop plain + TTA
    "test_sliding_tiles": "400",         # sliding+TTA is ~6 s/tile
}

CROPS_PER_EPOCH = 12000                  # config.py:91
IMG_PER_S = 45.0                         # v3 on an H100; replace from `calibrate`
RATE = 3.95 + 12 * 0.0472 + 48 * 0.0080  # $/h: H100 + 12 cores + 48 GiB


# Two sliding+TTA passes now, not one: 400 val tiles and 400 held-out
# gamus/test tiles, plus a batched centre-crop sweep over all 2861 test tiles.
FINAL_H = 1.5


def cost(epochs: int) -> tuple[float, float]:
    """(hours, dollars) for one run, including the final eval stage."""
    h = epochs * CROPS_PER_EPOCH / IMG_PER_S / 3600 + FINAL_H
    return h, h * RATE


def argv(**ov: str) -> list[str]:
    """The exact argv train.py sees. `extra` is appended raw and wins (argparse
    takes the last occurrence)."""
    extra = ov.pop("extra", "")
    f = FLAGS | {k: str(v) for k, v in ov.items() if v not in ("", 0, None)}
    return [x for k, v in f.items() for x in (f"--{k}", v)] + shlex.split(extra)


# --- image ----------------------------------------------------------------
hf = modal.Secret.from_name("dw-hf", required_keys=["HF_TOKEN"])
kg = modal.Secret.from_name("dw-kaggle", required_keys=["KAGGLE_USERNAME", "KAGGLE_KEY"])


def _bake() -> None:
    """Pull the gated encoder at build time.

    Otherwise two hub calls happen on every $3.95/h cold start — the weights and
    a separate AutoImageProcessor lookup (preprocess.py:105-124) — and
    train.py:314-315 only *warns* about a missing token, so it fails late and
    illegibly. This makes it a free build-time error.
    """
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
    .apt_install("libgl1", "libglib2.0-0", "rsync", "unzip")   # cv2 deps, stage, kaggle
    .uv_pip_install(f"torch=={TORCH}", extra_index_url=TORCH_INDEX)
    .uv_pip_install(*_REQS, "kaggle>=1.6.0")
    .env({"HF_HOME": "/root/hf", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
          "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .run_function(_bake, secrets=[hf])
    .add_local_dir(_LOCAL, CODE, ignore=["**/__pycache__", "outputs", "data", "*.pt"])
)

app = modal.App("depthwizard-v4", image=image)
data_vol = modal.Volume.from_name("depthwizard-data", create_if_missing=True)
res_vol = modal.Volume.from_name("depthwizard-results", create_if_missing=True)

GPU_FN = dict(  # noqa: C408 — one config, three functions
    gpu="H100", cpu=12.0, memory=49152, ephemeral_disk=102400, timeout=8 * 3600,
    volumes={DATA: data_vol.with_mount_options(read_only=True), RESULTS: res_vol},
    secrets=[hf],
)


# --- helpers --------------------------------------------------------------
def sh(cmd: list[str], cwd: str = CODE) -> None:
    print(f"\n[dw] $ {shlex.join(cmd)}", flush=True)
    t = time.time()
    if subprocess.run(cmd, cwd=cwd, check=False).returncode:
        raise RuntimeError(f"[dw] failed after {(time.time() - t) / 60:.1f} min: {cmd[0]}")
    print(f"[dw] ok ({(time.time() - t) / 60:.1f} min)", flush=True)


def stores(root: str) -> int:
    """Print every store's tile count. Read these.

    v3 wrote a gamus/val holding 129 of 400 tiles and still reported success,
    because a tile whose download failed was swallowed as `skip: ...`.
    """
    n = 0
    print(f"\n[dw] stores under {root}:")
    for idx in sorted(Path(root).glob("*/*/index.json")):
        n += 1
        d = json.loads(idx.read_text())
        # .parents[1], not .parent — .parent is the split. That is the ba78dd8 bug.
        name = f"{idx.parents[1].name}/{idx.parent.name}"
        gib = sum(f.stat().st_size for f in idx.parent.glob("*.npy")) / 2**30
        ok = "bounds" if (idx.parent / "stretch_bounds_2_98.npy").exists() else "NO-BOUNDS"
        print(f"    {name:<20} n={d['n']:>6}  tile={d['tile_px']:>4}  "
              f"gsd={d['gsd_m']:<5}  seg={'yes' if d.get('has_seg') else 'no':<3}  "
              f"{gib:6.2f} GiB  {ok}")
    return n


# --- functions ------------------------------------------------------------
@app.function(gpu="T4", cpu=4.0, memory=16384, timeout=1800)
def check() -> None:
    """Cheapest proof the image is sane: imports, flags, the 175 offline tests."""
    import shutil
    sh([sys.executable, "-c", _HW])
    print(f"[dw] /dev/shm {shutil.disk_usage('/dev/shm').total / 2**30:.1f} GiB, "
          f"nproc {os.cpu_count()}")

    # config.py's parser ends in parse_known_args, so a flag that is not a real
    # dataclass field is SILENTLY DROPPED and the run proceeds on the default you
    # thought you overrode. A whole GPU-hour for a typo.
    from dataclasses import fields
    sys.path.insert(0, CODE)
    from config import Config
    unknown = sorted((set(FLAGS) | {"smoke", "freeze_epochs", "resume"})
                     - {f.name for f in fields(Config)})
    if unknown:
        raise RuntimeError(f"[dw] not Config fields, would be silently dropped: {unknown}")
    print(f"[dw] {len(FLAGS) + 3} flags all exist in Config")

    sh([sys.executable, "-m", "pytest", "-q"])


_HW = """
import torch, os
print("torch", torch.__version__, "cuda", torch.version.cuda)
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu{i}: {p.name} {p.total_memory/2**30:.1f}GiB sm_{p.major}{p.minor}"
          f"{'' if p.major >= 8 else '  NO bf16 -> --amp_dtype fp16'}")
import cv2, rasterio, h5py, tifffile, transformers  # noqa
print("imports ok | encoder cached:", os.path.isdir("/root/hf/hub"))
"""


@app.function(cpu=8.0, memory=32768, ephemeral_disk=204800, timeout=6 * 3600,
              volumes={DATA: data_vol, RESULTS: res_vol}, secrets=[hf, kg])
def prepare(datasets: str = "gamus,synrs3d,dfc23", force: bool = False) -> None:
    """Download -> pack -> prime bounds -> commit. CPU only; run once.

    Packs to local /scratch, never straight to the Volume: ShardWriter writes
    through open_memmap and trims by rewriting (packed.py:52,88), the exact
    random-write pattern FUSE is worst at. Finished .npy files are copied over
    sequentially at the end.

    Idempotent per store (every packer checks store_exists), so a failure
    resumes. force=True repacks.
    """
    want = datasets.split(",")
    dl = "/scratch/_dl"
    Path(SCRATCH).mkdir(parents=True, exist_ok=True)
    if any(Path(DATA).glob("*/*/index.json")):
        print("[dw] resuming from what is already on the volume")
        # /_dl/ is raw staging, never a store -- see push() in 01_data_prep.
        sh(["rsync", "-a", "--exclude", "/_dl/", f"{DATA}/", f"{SCRATCH}/"],
           cwd="/")
    fl = ["--force"] if force else []

    if "gamus" in want:
        # The Kaggle PNG mirror, not the gated HF repo: prepare_gamus issues one
        # hf_hub_download per file (~13 200 requests) and earns a CAS 429 that it
        # swallows per tile. Cost: no class rasters, so gamus has_seg=no.
        src = f"{dl}/gamus"
        _kaggle("akashch1512/gamusdataset", src, ["train", "val"])
        sh([sys.executable, "pack_gamus_png.py", "--src", src,
            "--out", SCRATCH,          # never omit: default is /kaggle/temp/dwdata
            "--depth_scale", "auto", "--train_tiles", "0", "--val_tiles", "0",
            "--prime_workers", "8", *fl])

    if "synrs3d" in want:
        # 4 archives = all three of g1 (the only source that reaches past GAMUS's
        # 0.66 m ceiling) + one g05. _syn_family splits them into per-GSD stores.
        sh([sys.executable, "prepare_data.py", "--data_root", SCRATCH,
            "--datasets", "synrs3d", "--synrs3d_archives", "4", *fl])

    if "dfc23" in want:
        src = f"{dl}/dfc23"
        _kaggle("abhaydkale232/dfc23-track2-height-estimation", src,
                ["track2/train/rgb", "track2/train/dsm"])
        # Top-level rgb/dsm/sar are a byte-identical copy of track2/train
        # (~5.5 of the 12 GB); SAR is unused, the store is 3-channel.
        for junk in ("rgb", "dsm", "sar", "track2/train/sar", "track2/val",
                     "track2_test_data"):
            sh(["rm", "-rf", f"{src}/{junk}"], cwd="/")
        sh([sys.executable, "prepare_data.py", "--data_root", SCRATCH,
            "--datasets", "dfc23", "--dfc23_dir", f"{src}/track2/train",
            "--dfc23_tile", "512", "--dfc23_val_frac", "0.15",
            "--dfc23_max_height_m", "150",   # DFC23_Track2_Data_Audit.md's value
            *fl])

    # Prime percentile bounds while still on /scratch — this writes into each
    # store dir and DATA is read-only during training. packed.py:240-245 fails
    # soft and recomputes in-process otherwise, which is a silent per-epoch tax.
    sh([sys.executable, "-c", _PRIME])
    stores(SCRATCH)
    # --exclude /_dl/: prepare_data.py stages raw downloads into
    # {data_root}/_dl, which is INSIDE SCRATCH. Without this the throwaway
    # .h5 files land on a billed Volume and nothing ever reads them back.
    sh(["rsync", "-a", "--info=progress2", "--exclude", "/_dl/",
        f"{SCRATCH}/", f"{DATA}/"], cwd="/")
    data_vol.commit()
    stores(DATA)
    print("\n[dw] expect: gamus/train ~3453  gamus/val ~859  synrs3d_g1  synrs3d_g05"
          "  dfc23_g050/train ~1506  dfc23_g050/val ~266")


def _kaggle(slug: str, dest: str, expect: list[str]) -> None:
    if all((Path(dest) / e).exists() for e in expect):
        return print(f"[dw] {slug} already at {dest}")
    sh(["kaggle", "datasets", "download", "-d", slug, "-p", dest, "--unzip"], cwd="/")
    if missing := [e for e in expect if not (Path(dest) / e).exists()]:
        raise RuntimeError(f"[dw] {slug} unpacked without {missing}; layout changed. "
                           f"Top level: {sorted(p.name for p in Path(dest).iterdir())[:20]}")


_PRIME = f"""
import sys; sys.path.insert(0, "{CODE}")
from pathlib import Path
from dwdata.packed import PackedStore
for i in sorted(Path("{SCRATCH}").glob("*/*/index.json")):
    if (i.parent / "stretch_bounds_2_98.npy").exists(): continue
    print("priming:", i.parent, flush=True)
    PackedStore(i.parent).prime_stretch_bounds(2.0, 98.0, workers=8)
"""


def _go(**ov: str) -> None:
    a = argv(**ov)
    out = a[a.index("--output_dir") + 1]
    Path(out).mkdir(parents=True, exist_ok=True)

    # Stage off the Volume. Not an optimisation: the trainer opens every shard
    # with mmap_mode="r" (packed.py:141,198) and reads random ~0.79 MB windows
    # ~45/s over ~36 GB. That is what Logs/v3/gpu_disk_guard.log calls STARVED
    # when it is served off a network mount, and a Volume is FUSE-backed.
    if os.environ.get("DW_NO_AUTOSTAGE") != "1":
        sh(["rsync", "-a", "--info=progress2", "--exclude", "/_dl/",
            f"{DATA}/", f"{SCRATCH}/"], cwd="/")
        if not stores(SCRATCH):
            raise RuntimeError(f"[dw] nothing staged from {DATA} — run `prepare` first")

    try:
        sh([sys.executable, "train.py", *a])
    finally:
        # Commit even on failure: a crash that wrote last_full.pt is resumable,
        # and one that died in the exports is a ~$2 `finalize`, not a ~$14 retrain.
        res_vol.commit()


@app.function(**GPU_FN)
def smoke(extra: str = "") -> None:
    """24 crops, 2 epochs, batch 2 (config.py:313-334). ~10 min. Do not skip it."""
    _go(output_dir=f"{RESULTS}/smoke_v4m", extra=f"--smoke {extra}")


@app.function(**GPU_FN)
def train(epochs: int = 0, batch_size: int = 0, datasets: str = "",
          resume: str = "", extra: str = "") -> None:
    """The real run. Use --detach.

        modal run --detach modal_app.py::train
        modal run --detach modal_app.py::train --epochs 3 --extra '--max_minutes 20'
    """
    ov: dict = {"batch_size": batch_size, "datasets": datasets,
                "resume": resume, "extra": extra}
    if epochs:
        # max_minutes SIZES the LR cosine (train.py:626-632). Moving epochs
        # without it is how v4-2 annealed to 5.09e-05 instead of 8.32e-06.
        ov |= {"epochs": epochs, "max_minutes": round(cost(epochs)[0] * 60 * 0.9)}
    h, d = cost(int(ov.get("epochs") or FLAGS["epochs"]))
    print(f"[dw] projected {h:.2f} h, ${d:.2f} at {IMG_PER_S} img/s")
    _go(**ov)


@app.function(**GPU_FN)
def finalize(extra: str = "") -> None:
    """Exports only. train.py:938-948 reads history/best back out of the
    committed metrics.json and runs final TTA, sliding eval, figures, ONNX."""
    _go(epochs="0", freeze_epochs="0", extra=extra)


@app.local_entrypoint()
def show_tuning(epochs: int = 0, img_per_s: float = 0.0) -> None:
    """Resolve every flag and print the cost. Starts no container."""
    global IMG_PER_S
    if img_per_s:
        IMG_PER_S = img_per_s
    n = epochs or int(FLAGS["epochs"])
    print("python train.py " + shlex.join(argv(**({"epochs": epochs} if epochs else {}))))
    print(f"\n{'epochs':>7} {'hours':>7} {'$':>8}   (at {IMG_PER_S} img/s)")
    for e in sorted({16, 26, 30, 40, n}):
        h, d = cost(e)
        print(f"{e:>7} {h:>7.2f} {d:>8.2f}{'  <-' if e == n else ''}")
    print("\nIMG_PER_S is an estimate until you run:"
          "\n  modal run --detach modal_app.py::train --epochs 3 --extra '--max_minutes 20'")
