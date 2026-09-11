"""Entrypoint for DepthWizard v4.

    python main.py --prepare --datasets gamus,synrs3d   # one-time data pack
    python main.py --prepare --datasets india_unlabeled --india_dir ~/tiles
    python main.py --smoke                              # tiny sanity run
    python main.py                                      # full training run
    python main.py --serve                              # FastAPI + 3D viewer
    python main.py --install-only

Unknown flags (--epochs, --batch_size, ...) forward to train.py / prepare_data.py.

`run_lightning.sh` is the preferred entry point on a Studio — it also runs the
test suite and prints the GPU report before touching anything.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def install() -> None:
    req = HERE / "requirements.txt"
    print(f"=== installing {req} ===")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)],
                   check=False)


def main() -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--skip-install", action="store_true")
    p.add_argument("--install-only", action="store_true")
    p.add_argument("--prepare", action="store_true",
                   help="run prepare_data.py instead of training")
    p.add_argument("--serve", action="store_true",
                   help="run the FastAPI service instead of training")
    p.add_argument("-h", "--help", action="store_true")
    args, rest = p.parse_known_args()

    if args.help:
        print(__doc__)
        return
    if not args.skip_install:
        install()
    if args.install_only:
        return

    if args.prepare:
        from prepare_data import main as prep_main

        sys.argv = [sys.argv[0]] + rest
        prep_main()
        return

    if args.serve:
        from serve.app import main as serve_main

        sys.argv = [sys.argv[0]] + rest
        serve_main()
        return

    from train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
