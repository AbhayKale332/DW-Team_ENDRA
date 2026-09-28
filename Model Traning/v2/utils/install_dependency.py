"""Step 1: install v2 deps into the current interpreter (Kaggle / vast.ai)."""

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
        raise FileNotFoundError(requirements_file)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-input",
         "--disable-pip-version-check", "-q", "-r", str(requirements_file)],
        check=True,
    )
    print("[setup] dependencies installed")


def setup_project() -> None:
    install_dependencies()
