"""Frozen validation on Modal's existing model/data volumes.

modal deploy -e gpu readout_modal.py
Then spawn preflight (CPU) and probe (L4) through the deployed dw-readout app.
Results persist under depthwizard-results:/v5_readout_probe.
"""
from pathlib import Path
import json
import subprocess
import sys
import threading

import modal

from final_modal import DATA, RESULTS, KIN, V5, image, hf, data_vol, res_vol, link, _every

app = modal.App("dw-readout", image=image)
CHECKPOINT = f"{RESULTS}/v5_final_forest/best.pt"
SOURCES = "mvs3dm,us3d,neon,gamus"


@app.function(cpu=2, memory=8192, timeout=600,
              volumes={DATA: data_vol.with_mount_options(read_only=True),
                       RESULTS: res_vol.with_mount_options(read_only=True)})
def preflight():
    root = "/tmp/dwdata"
    link(KIN, root)
    if not Path(CHECKPOINT).is_file():
        raise FileNotFoundError(CHECKPOINT)
    stores = {}
    for src in SOURCES.split(","):
        path = Path(root, src, "val", "index.json")
        stores[src] = json.loads(path.read_text())
        for shard in stores[src]["shards"]:
            for suffix in ("rgb", "hgt", "cls", "val"):
                if not (path.parent / f"{shard['file']}_{suffix}.npy").is_file():
                    raise FileNotFoundError(f"{src}/{shard['file']}_{suffix}.npy")
    subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_readout_diagnostics.py",
                    "tests/test_model.py"], cwd=V5, check=True)
    return {"checkpoint_bytes": Path(CHECKPOINT).stat().st_size,
            "stores": {src: {k: d[k] for k in ("n", "gsd_m", "tile_px")}
                       for src, d in stores.items()}}


@app.function(gpu="L4", cpu=2, memory=8192, timeout=90 * 60, secrets=[hf],
              volumes={DATA: data_vol.with_mount_options(read_only=True), RESULTS: res_vol})
def probe(tiles: int = 128, candidates: str = "", run: str = "v5_readout_probe"):
    if "/" in run or run in ("", ".", "..", "v5_final_forest"):
        raise ValueError("run must name a separate results folder")
    root = "/tmp/dwdata"
    link(KIN, root)
    out = Path(RESULTS, run)
    out.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    _every(60, res_vol.commit, stop)
    try:
        with (out / "probe.log").open("w", buffering=1) as log:
            proc = subprocess.Popen([sys.executable, "-u", "tools/readout_probe.py",
                                     "--ckpt", CHECKPOINT, "--data_root", root,
                                     "--out", str(out), "--tiles", str(tiles),
                                     "--sources", SOURCES, "--candidates", candidates],
                                    cwd=V5, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
            if proc.wait():
                raise RuntimeError(f"probe exited {proc.returncode}; see {out}/probe.log")
    finally:
        stop.set()
        res_vol.commit()
    return str(out / "probe_metrics.json")
