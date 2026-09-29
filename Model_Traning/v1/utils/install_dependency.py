"""Step 1 helper: install project dependencies into the *current* Python env.

On Kaggle there is no `uv` / virtualenv workflow — the notebook runs against a
single system interpreter. So we just shell out to `pip` for the interpreter
that is running this process (`sys.executable`).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REQUIREMENTS_FILE = Path(__file__).resolve().parent.parent / "requirements-kaggle.txt"


def running_on_kaggle() -> bool:
    return (
        "KAGGLE_KERNEL_RUN_TYPE" in os.environ
        or "KAGGLE_URL_BASE" in os.environ
        or Path("/kaggle").exists()
    )


def install_dependencies(requirements_file: Path = REQUIREMENTS_FILE) -> None:
    if not requirements_file.exists():
        raise FileNotFoundError(
            f"Requirements file not found: {requirements_file}"
        )

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-input",
        "--disable-pip-version-check",
        "-q",
        "-r",
        str(requirements_file),
    ]

    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
    )

    print("[setup] dependencies installed")
    

def setup_project() -> None:
    """Local dev convenience: keep using uv when it's available, else pip."""
    if running_on_kaggle():
        install_dependencies()
        return

    from shutil import which

    if which("uv"):
        subprocess.run(["uv", "sync"], check=True)
    else:
        install_dependencies()
