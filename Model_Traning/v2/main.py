"""Entrypoint for DepthWizard v2.

    !git clone https://github.com/<you>/SIH.git
    %cd SIH/FineTunning/v2
    !python main.py                 # install deps, then train
    !python main.py --smoke         # tiny sanity run first
    !python main.py --install-only  # just deps

Unknown flags (--datasets, --batch_size, …) forward to `train.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.install_dependency import setup_project


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-install", action="store_true")
    p.add_argument("--install-only", action="store_true")
    args, _ = p.parse_known_args()

    if not args.skip_install:
        print("=== Step 1: install dependencies ===")
        setup_project()
    if args.install_only:
        return

    print("=== Step 2: train ===")
    from train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
