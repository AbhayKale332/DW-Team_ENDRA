"""Config for fine-tuning version v1 — DepthWizard Phase 0.

Everything now lives in the single self-contained runner `kaggle_phase0.py`
(so it can be pasted into a Kaggle cell with no other project files). This
module just re-exports the dataclass for anything that still does
`from config import CONFIG`.

Tweak knobs in `kaggle_phase0.py::Config`, or pass them on the CLI:

    python kaggle_phase0.py --epochs 6 --train_subset 400 --batch_size 12
"""

from __future__ import annotations

from kaggle_phase0 import Config

CONFIG = Config()
