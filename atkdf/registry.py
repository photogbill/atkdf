"""The registry: emitters keyed by identity, their bearing history, and the
JSONL log beside a clip.

Every bearing the engine produces is filed here under its emitter key
(`protocol:system:id`) or, with no identity, under a tracker's `track:<id>`.
The log is JSONL — one `BearingRecord` per line, each carrying `site_id` and
the passport it was measured under — so a second vehicle's log merges by
concatenation and `fusion`'s per-source bias groups already know what to do
with two sites (umbrella §8.10). "How good is this fix?" is answered by the
records on disk, not by memory.

A fix is computed on demand from an emitter's FUSABLE bearings (absolute,
positioned, not heard via a repeater — `BearingRecord.to_lob` enforces it)
through the moved `fusion.ml_fix`. The grid posterior (Phase 2) is the
multimodal view; this is the summary ellipse.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import fusion
from .records import BearingRecord, Emitter, EmitterKey


def key_of(rec: BearingRecord) -> str:
    """The registry key for a record: its emitter identity, or its track id."""
    if rec.emitter_key:
        return rec.emitter_key
    if rec.track_id:
        return f"track:{rec.track_id}"
    return ""


class Registry:
    """Emitters and their bearings. Optionally append-logs to a JSONL file."""

    def __init__(self, site_id: str = "", log_path=None, fleet_map=None):
        self.site_id = site_id
        self.emitters: dict = {}
        self.by_key_bearings: dict = {}
        self._log = None
        if log_path is not None:
            self._log = open(log_path, "a", encoding="utf-8")
        self.fleet = dict(fleet_map or {})           # key -> {name, role, team}

    # -- ingest --------------------------------------------------------------
    def add(self, rec: BearingRecord) -> Emitter | None:
        key = key_of(rec)
        if not key:
            return None
        em = self.emitters.get(key)
        if em is None:
            k = rec.key
            em = Emitter(key=key, protocol=(k.protocol if k else ""),
                         system=(k.system if k else ""))
            fm = self.fleet.get(key) or self.fleet.get(rec.emitter_key)
            if fm:
                em.name = fm.get("name", "")
                em.role = fm.get("role", "")
                em.team = fm.get("team", "")
            self.emitters[key] = em
            self.by_key_bearings[key] = []
        em.observe(rec)
        self.by_key_bearings[key].append(rec)
        if self._log is not None:
            self._log.write(rec.to_json() + "\n")
            self._log.flush()
        return em

    def add_all(self, recs) -> None:
        for r in recs:
            self.add(r)

    # -- query ---------------------------------------------------------------
    def get(self, key: str) -> Emitter | None:
        return self.emitters.get(key)

    def bearings(self, key: str) -> list:
        return list(self.by_key_bearings.get(key, ()))

    def fusable_bearings(self, key: str) -> list:
        return [r for r in self.by_key_bearings.get(key, ()) if r.fusable]

    def fix(self, key: str, max_bearings: int = 200, **kw):
        """A position fix for one emitter from its fusable bearings, or None
        when there are too few. Straight through the moved `fusion.ml_fix`,
        which carries the bias groups and the honest ellipse."""
        lobs = [r.to_lob(position_sigma_m=r.quality.get("position_sigma_m", 0.0))
                for r in self.fusable_bearings(key)[-max_bearings:]]
        lobs = [l for l in lobs if l is not None]
        if len(lobs) < 2:
            return None
        return fusion.ml_fix(lobs, **kw)

    # -- merge ---------------------------------------------------------------
    @classmethod
    def from_logs(cls, paths, site_id: str = "", fleet_map=None) -> "Registry":
        """Build a registry by reading one or more JSONL logs — how a second
        vehicle's records join (merge by file, no migration)."""
        reg = cls(site_id=site_id, fleet_map=fleet_map)
        for p in ([paths] if isinstance(paths, (str, Path)) else paths):
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        reg.add(BearingRecord.from_json(line))
        return reg

    def close(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def load_fleet_map(path) -> dict:
    """A fleet map CSV `id,name,role,team,protocol,system` -> {key: {...}}.
    The key is `protocol:system:id`, so an alert can read a name."""
    import csv
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            proto = (row.get("protocol") or "").strip()
            system = (row.get("system") or "").strip()
            rid = (row.get("id") or "").strip()
            if not rid:
                continue
            key = str(EmitterKey(proto, system, rid))
            out[key] = {"name": (row.get("name") or "").strip(),
                        "role": (row.get("role") or "").strip(),
                        "team": (row.get("team") or "").strip()}
    return out
