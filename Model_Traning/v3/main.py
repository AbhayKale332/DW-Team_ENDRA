"""Entrypoint for DepthWizard v3.

    python main.py --prepare --datasets gamus,synrs3d   # one-time data pack
    python main.py --smoke                              # tiny sanity run
    python main.py                                      # full training run
    python main.py --install-only

Unknown flags (--epochs, --batch_size, ...) forward to train.py / prepare_data.py.
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
    p.add_argument("-h", "--help", action="store_true")
    args, rest = p.parse_known_args()

    if args.help:
        print(__doc__)
        # Then the *real* option list.  `--help` used to print only the
        # docstring, so anything parsing it (shell_scripts/train_h100.sh's
        # autotuner) saw six option names in prose and could not tell which of
        # them take a value — it silently dropped every flag it tried to set and
        # the run kept config.py's 16 GB-card defaults on an 80 GB card.
        try:
            from config import Config
            from dataclasses import fields
            print("training options (config.py):\n")
            for f in fields(Config()):
                cur = getattr(Config(), f.name)
                if isinstance(cur, tuple):
                    continue
                meta = "BOOL" if isinstance(cur, bool) else type(cur).__name__.upper()
                print(f"  --{f.name} {meta}    (default: {cur!r})")
        except Exception as e:  # noqa: BLE001
            print(f"  (could not introspect config.py: {e})")
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

    from train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
