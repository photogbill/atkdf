"""A bearing tracker for emitters WITHOUT identity.

Analog FM has no ID; encrypted TETRA hides its ISSI. Intermittent bursts from
several radios on one channel must stay several tracks, not collapse into one
average, so this is the mechanism that carries the networks the decoder cannot
name (umbrella §7.4) — not a fallback.

Global nearest neighbour first (JPDA if it ever earns it): each track predicts
its next bearing from a constant-turn-rate model, a new bearing is gated by
angular distance and assigned to the nearest track, unassigned bearings spawn
tracks, and a track with no support for `coast_s` dies. Slot and sync type are
kept apart — a bearing in slot 1 never associates with a slot-2 track — so two
handsets alternating on one channel stay two tracks.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from .records import wrap180


@dataclass
class Track:
    id: str
    bearing_deg: float
    rate_dps: float = 0.0
    t_utc: float = 0.0
    n: int = 0
    slot: object = None
    sync_type: str = ""
    sigma_deg: float = 5.0
    alive: bool = True
    history: list = field(default_factory=list)

    def predict(self, t: float) -> float:
        return (self.bearing_deg + self.rate_dps * (t - self.t_utc)) % 360.0


class BearingTracker:
    """Feed it bearings (each: t_utc, bearing_deg, sigma_deg, slot, sync_type);
    it returns the track id each was assigned to. `gate_deg` is the largest
    innovation accepted; `coast_s` is how long a track survives unsupported."""

    def __init__(self, gate_deg: float = 12.0, coast_s: float = 8.0,
                 rate_gain: float = 0.3, pos_gain: float = 0.5):
        self.gate = float(gate_deg)
        self.coast = float(coast_s)
        self.rate_gain = float(rate_gain)
        self.pos_gain = float(pos_gain)
        self.tracks: dict = {}
        self._ids = itertools.count(1)

    def _new(self, t, brg, sig, slot, sync) -> Track:
        tid = f"T{next(self._ids)}"
        tk = Track(id=tid, bearing_deg=brg % 360.0, t_utc=t, n=1, slot=slot,
                   sync_type=sync, sigma_deg=sig)
        tk.history.append((t, brg))
        self.tracks[tid] = tk
        return tk

    def update(self, t_utc: float, bearing_deg: float, sigma_deg: float = 5.0,
               slot=None, sync_type: str = "") -> str:
        """Associate one bearing; returns the track id it went to (new or
        existing)."""
        self._reap(t_utc)
        best, best_d = None, self.gate
        for tk in self.tracks.values():
            if not tk.alive or tk.slot != slot:
                continue
            d = abs(wrap180(bearing_deg - tk.predict(t_utc)))
            # gate scaled by the track's and the bearing's uncertainty
            gate = self.gate + 0.5 * (tk.sigma_deg + sigma_deg)
            if d < best_d and d < gate:
                best, best_d = tk, d
        if best is None:
            return self._new(t_utc, bearing_deg, sigma_deg, slot, sync_type).id
        # update the assigned track (α-β on bearing and rate)
        pred = best.predict(t_utc)
        innov = wrap180(bearing_deg - pred)
        dt = max(1e-3, t_utc - best.t_utc)
        best.bearing_deg = (pred + self.pos_gain * innov) % 360.0
        best.rate_dps += self.rate_gain * innov / dt
        best.rate_dps = max(-180.0, min(180.0, best.rate_dps))
        best.t_utc = t_utc
        best.n += 1
        best.sigma_deg = 0.7 * best.sigma_deg + 0.3 * sigma_deg
        best.history.append((t_utc, bearing_deg))
        return best.id

    def _reap(self, now: float) -> None:
        for tk in self.tracks.values():
            if tk.alive and (now - tk.t_utc) > self.coast:
                tk.alive = False

    def live_tracks(self, now: float | None = None) -> list:
        if now is not None:
            self._reap(now)
        return [t for t in self.tracks.values() if t.alive]
