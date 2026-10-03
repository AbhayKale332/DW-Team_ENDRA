"""Compare inference settings on all GAMUS validation centre crops.

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
def confirm(candidates: str = "baseline,baseline_d4_zoom150",
            output_name: str = "v5_gamus_zoom_confirmation_20261003"):
    if not output_name or Path(output_name).name != output_name or output_name in (".", ".."):
        raise ValueError("output_name must be one directory name")
    root = "/tmp/dwdata"
    link(KIN, root)
    # Check the new single-pass scale adapter before loading the checkpoint.
    import torch
    sys.path.insert(0, V5)
    from tools.readout_probe import ScaledInput

    class ConstantHeight(torch.nn.Module):
        def forward(self, image):
            self.shape = image.shape[-2:]
            return {"fused": image[:, :1] * 0 + 7.0}

    probe = ConstantHeight()
    for scale, size in ((1.25, 40), (1.5, 48)):
        # Patch rounding turns 32 * 1.25 into 32 (ties to even).
        expected = max(16, int(round(size / 16)) * 16)
        prediction = ScaledInput(probe, scale)(torch.zeros(1, 3, 32, 32))["fused"]
        assert probe.shape == (expected, expected)
        assert prediction.shape == (1, 1, 32, 32)
        assert torch.allclose(prediction, torch.full_like(prediction, 7.0))
    print("[dw] single-pass scale shape and metre-height checks passed", flush=True)
    out = Path(RESULTS, output_name)
    out.mkdir(exist_ok=False)
    stop = threading.Event()
    _every(60, res_vol.commit, stop)
    try:
        with (out / "probe.log").open("w", buffering=1) as log:
            proc = subprocess.Popen(
                [sys.executable, "-u", "tools/readout_probe.py",
                 "--ckpt", f"{RESULTS}/v5_final_forest/best.pt",
                 "--data_root", root, "--out", str(out), "--tiles", "0",
                 "--sources", "gamus", "--candidates", candidates],
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
