"""The optional door to atkdsp's kernels — today, its DMR / P25 sync correlator.

atkdf runs on numpy alone; atkdsp is optional (ATKDF_PLAN §0, §1). When ATK
hosts the engine it INJECTS its own kernels (`dsp_native.kernels().DmrSync`,
native DLL or numpy twin — ATK knows which and says so on Setup), so nothing
here is on ATK's path. This module is for atkdf on its own: tests, a script,
reprocessing a clip. It looks, in order, for

  1. an `atkdsp` package already importable;
  2. `$ATKDSP_HOME` (the repo root, or its `python` folder);
  3. a sibling checkout beside this repo (`..\\atkdsp\\python` — Bill's layout,
     D:\\Analyst_Toolkit\\atkdsp),

and prefers the native library when it loads (the twin when it does not —
same constructor, same hit dicts, slower). Nothing found means pilots are off
and the reason says where it looked; nothing else in atkdf changes.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger("atkdf.atkdsp")

_state: dict = {"mod": None, "why": "", "checked": False}


def _candidates() -> list:
    out = []
    env = os.environ.get("ATKDSP_HOME")
    if env:
        p = Path(env)
        out += [p / "python", p]
    out.append(Path(__file__).resolve().parents[2] / "atkdsp" / "python")
    return out


def module():
    """The `atkdsp` package, or None (with `why()` saying where it looked)."""
    if _state["checked"]:
        return _state["mod"]
    _state["checked"] = True
    try:
        _state["mod"] = importlib.import_module("atkdsp")
        log.info("atkdsp found on the import path: %s", getattr(_state["mod"], "__file__", "?"))
        return _state["mod"]
    except ImportError:
        pass
    tried = []
    for c in _candidates():
        tried.append(str(c))
        if not (c / "atkdsp" / "__init__.py").is_file():
            continue
        sys.path.insert(0, str(c))
        try:
            _state["mod"] = importlib.import_module("atkdsp")
            log.info("atkdsp found at %s", c)
            return _state["mod"]
        except Exception as exc:                          # noqa: BLE001
            tried[-1] += f" ({type(exc).__name__}: {exc})"
            try:
                sys.path.remove(str(c))
            except ValueError:
                pass
    _state["why"] = "atkdsp not found (looked on the import path, " + "; ".join(tried) + ")"
    log.info(_state["why"])
    return None


def why() -> str:
    module()
    return _state["why"]


def sync_factory(prefer_native: bool = True):
    """(DmrSync class, description) — native when it loads, else the numpy
    twin — or (None, the reason)."""
    mod = module()
    if mod is None:
        return None, why()
    if prefer_native:
        try:
            ok = bool(getattr(mod, "available", lambda: False)())
        except Exception as exc:                          # noqa: BLE001
            ok = False
            log.info("atkdsp native library unavailable: %s", exc)
        if ok and hasattr(mod, "DmrSync"):
            return mod.DmrSync, f"atkdsp native {getattr(mod, 'version', lambda: '?')()}"
    try:
        ref = importlib.import_module("atkdsp.reference")
        return ref.DmrSync, "atkdsp numpy twin (slow — build atkdsp for live use)"
    except Exception as exc:                              # noqa: BLE001
        return None, f"atkdsp found but its twin would not import: {exc}"


def reset_for_tests() -> None:
    _state.update({"mod": None, "why": "", "checked": False})
