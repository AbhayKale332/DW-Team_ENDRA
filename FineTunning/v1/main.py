"""Entrypoint for fine-tuning v1.

Kaggle usage (in a notebook cell):

    !git clone https://github.com/<you>/SIH.git
    %cd SIH/FineTunning/v1
    !python main.py

Step 1 installs dependencies into the Kaggle runtime.
Step 2 runs the fine-tuning.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# make `import config`, `import train`, `import utils...` work regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.install_dependency import setup_project


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="skip Step 1 (deps already installed this session)",
    )
    parser.add_argument(
        "--install-only",
        action="store_true",
        help="run Step 1 and exit",
    )
    args = parser.parse_args()

    # Step 1: dependencies -> current (Kaggle) env
    if not args.skip_install:
        print("=== Step 1: install dependencies ===")
        setup_project()

    if args.install_only:
        return

    # Step 2: train  (imported AFTER install so heavy deps exist)
    print("=== Step 2: fine-tune ===")
    from train import run

    run()


if __name__ == "__main__":
    main()
