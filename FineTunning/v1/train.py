"""Step 2: the fine-tuning run.

DepthWizard Phase 0 is implemented as one self-contained file, `kaggle_phase0.py`
(GAMUS download + DINOv3-SAT encoder + DPT decoder + metric-nDSM head + train +
eval). This module keeps the `main.py` -> `train.run()` entrypoint working by
delegating to it.

Run either way:
    python main.py              # Step 1 installs deps, then calls run()
    python kaggle_phase0.py     # same thing, standalone
"""

from __future__ import annotations


def run() -> None:
    from kaggle_phase0 import main

    main()
