"""Test set-up: the package under test is THIS checkout's `atkdf`, never
another copy that happens to be installed, and the tests' own helpers
(`_signals`) import by name."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
for p in (str(TESTS), str(ROOT)):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
for name in [n for n in sys.modules if n == "atkdf" or n.startswith("atkdf.")]:
    del sys.modules[name]

FIXTURES = TESTS / "fixtures"
