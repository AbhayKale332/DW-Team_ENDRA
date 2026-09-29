"""Zip the run's artefacts + record the environment."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def write_env_files(out_dir: str) -> None:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                capture_output=True, text=True, timeout=120).stdout
        (d / "pip_freeze.txt").write_text(freeze)
    except Exception:  # noqa: BLE001
        pass
    lines = [f"python: {sys.version}"]
    try:
        import torch

        lines += [f"torch: {torch.__version__}", f"cuda: {torch.version.cuda}"]
        for i in range(torch.cuda.device_count()):
            lines.append(f"gpu{i}: {torch.cuda.get_device_name(i)}")
    except Exception:  # noqa: BLE001
        pass
    (d / "env.txt").write_text("\n".join(lines))


def build_zip(out_dir: str) -> Path:
    d = Path(out_dir)
    base = d.parent / f"results_{d.name}"
    return Path(shutil.make_archive(str(base), "zip", root_dir=str(d)))
