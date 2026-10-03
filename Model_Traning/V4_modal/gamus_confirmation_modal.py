"""Confirm the existing 1.5x D4 pilot on all GAMUS validation centre crops.

Deploy in environment gpu, then spawn confirm. Reuses the readout evaluator;
never writes a checkpoint or reads the held-out test split.
"""
from pathlib import Path
import subprocess
import sys
import threading

import modal

from final_modal import DATA, RESULTS, KIN, V5, image, hf, data_vol, res_vol, link, _every

app = modal.App("dw-gamus-confirmation", image=image)


@app.function(gpu="L4", cpu=2, memory=8192, timeout=40 * 60, secrets=[hf],
              volumes={DATA: data_vol.with_mount_options(read_only=True), RESULTS: res_vol})
def confirm():
    root = "/tmp/dwdata"
    link(KIN, root)
    out = Path(RESULTS, "v5_gamus_zoom_confirmation_20261003")
    out.mkdir(exist_ok=False)
    stop = threading.Event()
    _every(60, res_vol.commit, stop)
    try:
        with (out / "probe.log").open("w", buffering=1) as log:
            proc = subprocess.Popen(
                [sys.executable, "-u", "tools/readout_probe.py",
                 "--ckpt", f"{RESULTS}/v5_final_forest/best.pt",
                 "--data_root", root, "--out", str(out), "--tiles", "0",
                 "--sources", "gamus", "--candidates", "baseline,baseline_d4_zoom150"],
                cwd=V5, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
            if proc.wait():
                raise RuntimeError(f"Evaluation exited {proc.returncode}; see {out}/probe.log")
    finally:
        stop.set()
        res_vol.commit()
    return str(out / "probe_metrics.json")
