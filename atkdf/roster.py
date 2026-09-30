"""The roster: who is expected, who is silent, who just keyed up — for relief use.

The point Bill made: "If Radio ID 473 isn't checking in and it's a dangerous
area, and we pick up his radio as it checks in automatically with the repeater,
it could be a life or death situation." So the roster watches expected IDs and
raises an alert the operator can read before they finish reading the ID —
`heard via repeater` distinct from `located`, a name instead of a number.

Three things it does (umbrella §7.3):
  * silence thresholds — silent past its threshold is flagged, keys-up-after-
    silence is an alert (with the bearing the engine just measured);
  * a keep-alive learner — many radios check in periodically (ARS, GPS
    reports); the roster learns each ID's inter-arrival distribution online and
    flags a MISSED expected check-in at 3σ late, before a human notices the gap;
  * fleet-map names — an alert reads "Medic 2 - Rodriguez", not "473".

It holds no bearings itself; it is fed observation times and asks the registry
for the last bearing when it raises an alert.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class KeepAlive:
    """An online estimate of one ID's check-in interval (Welford mean/var of
    inter-arrival gaps). `late(now)` is how many σ past the mean it is."""

    last_t: float = -1.0
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def observe(self, t: float) -> None:
        if self.last_t >= 0 and t > self.last_t:
            gap = t - self.last_t
            self.n += 1
            d = gap - self.mean
            self.mean += d / self.n
            self.m2 += d * (gap - self.mean)
        self.last_t = t

    @property
    def std(self) -> float:
        return math.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else 0.0

    def sigma_late(self, now: float) -> float:
        if self.n < 3 or self.mean <= 0 or self.last_t < 0:
            return 0.0
        s = self.std or (0.25 * self.mean)      # until spread is known, assume 25%
        return (now - self.last_t - self.mean) / s

    def periodic(self) -> bool:
        """Does this ID check in regularly enough to trust a missed-check-in
        alert? (low coefficient of variation)."""
        return self.n >= 3 and self.mean > 0 and (self.std / self.mean) < 0.5


@dataclass
class RosterEntry:
    key: str
    name: str = ""
    role: str = ""
    team: str = ""
    expected: bool = False
    silence_s: float = 0.0          # 0 = use the roster default
    last_heard: float = -1.0
    keepalive: KeepAlive = field(default_factory=KeepAlive)
    silent_flagged: bool = False


@dataclass
class Alert:
    kind: str                       # keyup_after_silence | missed_checkin | silent
    key: str
    t_utc: float
    label: str = ""
    detail: str = ""


class Roster:
    """Expected IDs, silence, keep-alive. Feed it `observe(key, t)` for every
    burst and call `check(now)` on a timer; both return any alerts raised."""

    def __init__(self, fleet_map=None, default_silence_s: float = 600.0,
                 expected_keys=None, red_zone: bool = False):
        self.entries: dict = {}
        self.default_silence = float(default_silence_s)
        self.red_zone = bool(red_zone)
        for key, fm in (fleet_map or {}).items():
            self.entries[key] = RosterEntry(key=key, name=fm.get("name", ""),
                                            role=fm.get("role", ""), team=fm.get("team", ""),
                                            expected=True)
        for key in (expected_keys or []):
            self.entries.setdefault(key, RosterEntry(key=key, expected=True))

    def _entry(self, key: str) -> RosterEntry:
        e = self.entries.get(key)
        if e is None:
            e = RosterEntry(key=key)
            self.entries[key] = e
        return e

    def observe(self, key: str, t_utc: float) -> list:
        """A burst from `key` at `t`. Returns a keyup-after-silence alert when
        it broke a silence past its threshold."""
        e = self._entry(key)
        alerts = []
        thr = e.silence_s or self.default_silence
        if e.last_heard >= 0 and (t_utc - e.last_heard) >= thr:
            alerts.append(Alert(kind="keyup_after_silence", key=key, t_utc=t_utc,
                                label=self._label(e),
                                detail=f"keyed up after {(t_utc - e.last_heard)/60:.0f} min silent"))
        e.silent_flagged = False
        e.last_heard = t_utc
        e.keepalive.observe(t_utc)
        return alerts

    def check(self, now: float) -> list:
        """Timer sweep: missed periodic check-ins (3σ late) and newly silent
        expected IDs. Each is raised once until the ID is heard again."""
        alerts = []
        for e in self.entries.values():
            if e.last_heard < 0:
                continue
            thr = e.silence_s or self.default_silence
            if e.keepalive.periodic() and e.keepalive.sigma_late(now) >= 3.0 and not e.silent_flagged:
                e.silent_flagged = True
                alerts.append(Alert(kind="missed_checkin", key=e.key, t_utc=now,
                                    label=self._label(e),
                                    detail=(f"missed its check-in: mean interval "
                                            f"{e.keepalive.mean/60:.1f} min, now "
                                            f"{(now - e.last_heard)/60:.1f} min silent")))
            elif e.expected and (now - e.last_heard) >= thr and not e.silent_flagged:
                e.silent_flagged = True
                zone = " in a red zone" if self.red_zone else ""
                alerts.append(Alert(kind="silent", key=e.key, t_utc=now, label=self._label(e),
                                    detail=f"silent {(now - e.last_heard)/60:.0f} min{zone}"))
        return alerts

    @staticmethod
    def _label(e: RosterEntry) -> str:
        return (f"{e.name} ({e.role})" if e.name and e.role else e.name) or e.key
