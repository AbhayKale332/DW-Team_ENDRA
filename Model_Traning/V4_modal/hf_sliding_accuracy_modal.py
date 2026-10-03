"""Frozen native-grid GAMUS validation through the deployed HF source."""
from pathlib import Path
import modal

from final_modal import DATA, RESULTS, V5, image, hf, data_vol, res_vol

HF_LOCAL = Path("/home/abhay/F/7th SEM/SIH/SingleViewHeigthEstimation")
HF_REMOTE = "/root/hf_space"
evaluation_image = image.add_local_file(HF_LOCAL / "config.py", f"{HF_REMOTE}/config.py")
for directory in ("models", "infer", "dwdata"):
    evaluation_image = evaluation_image.add_local_dir(
        HF_LOCAL / directory, f"{HF_REMOTE}/{directory}",
        ignore=["**/__pycache__", "**/*.pyc"],
    )

app = modal.App("dw-hf-sliding-accuracy", image=evaluation_image)


@app.function(cpu=2, memory=8192, timeout=600,
              volumes={DATA: data_vol.with_mount_options(read_only=True),
                       RESULTS: res_vol.with_mount_options(read_only=True)})
def preflight():
    import json
    import subprocess
    import sys

    path = Path(DATA, "kin/datasets/abhaydkale232/depthwizard-gamus/val/val")
    index = json.loads((path / "index.json").read_text())
    assert index["n"] == 859 and index["gsd_m"] == 0.25 and index["tile_px"] == 1024
    for shard in index["shards"]:
        for suffix in ("rgb", "hgt", "cls", "val"):
            assert (path / f"{shard['file']}_{suffix}.npy").is_file()
    assert Path(RESULTS, "v5_final_forest/best.pt").is_file()
    subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_hf_sliding_accuracy.py"],
                   cwd=V5, check=True)
    return {"n": index["n"], "tile_px": index["tile_px"], "gsd_m": index["gsd_m"],
            "checkpoint_bytes": Path(RESULTS, "v5_final_forest/best.pt").stat().st_size}


@app.function(gpu="L4", cpu=2, memory=8192, timeout=40 * 60,
              secrets=[hf], max_containers=1,
              volumes={DATA: data_vol.with_mount_options(read_only=True), RESULTS: res_vol})
def evaluate(output_name: str = "v5_gamus_hf_sliding_accuracy_20261003"):
    import subprocess
    import sys
    import threading
    from final_modal import _every

    if not output_name or Path(output_name).name != output_name:
        raise ValueError("output_name must be one directory name")
    out = Path(RESULTS, output_name)
    out.mkdir(exist_ok=False)
    stop = threading.Event()
    _every(60, res_vol.commit, stop)
    try:
        with (out / "accuracy.log").open("w", buffering=1) as log:
            proc = subprocess.Popen(
                [sys.executable, "-u", "tools/hf_sliding_accuracy.py",
                 "--hf-code", HF_REMOTE, "--support-code", V5,
                 "--store", f"{DATA}/kin/datasets/abhaydkale232/depthwizard-gamus/val/val",
                 "--ckpt", f"{RESULTS}/v5_final_forest/best.pt", "--out", str(out)],
                cwd=V5, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
            if proc.wait():
                raise RuntimeError(f"Accuracy check exited {proc.returncode}")
    finally:
        stop.set()
        res_vol.commit()
    return str(out / "accuracy.json")
