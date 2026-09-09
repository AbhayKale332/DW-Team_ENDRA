"""Ensure `FineTunning/v2` is importable as the package root for the test suite,
regardless of the directory pytest is invoked from (repo root, v2/, or tests/).
`pytest.ini`'s `pythonpath = .` covers pytest >= 7; this covers everything else.
"""

import sys
from pathlib import Path

_V2 = str(Path(__file__).resolve().parent)
if _V2 not in sys.path:
    sys.path.insert(0, _V2)
