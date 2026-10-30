"""Reference emitters: the radio compass, the mounting offset, and an error
model learned from truth.

A catalogued fixed emitter (an LTE eNodeB, a DMR/P25 repeater — public,
surveyed positions) that the array can hear is a free, continuous check on the
whole DF chain (umbrella §5.4, §8.8):

  * Radio compass. The array's measured angle to a reference, minus the true
    bearing to it from the map, is the vehicle heading plus the mounting
    offset. Valid at a stop and in a turn — it is what rescues heading when GPS
    course is useless (`heading.update_compass`).
  * Mounting offset. Solved once from the first drive (the part of the radio
    compass that does not change with heading), stored, re-checked each session.
  * Error model learned from truth. Every bearing to a reference has a KNOWN
    answer, so the residual as a function of SNR (and, later, speed/turn/band)
    is measured, not assumed — and applied to the target's bearings and handed
    to `fusion` as the σ. The 3°-bias experiment showed an assumed model gives
    0% ellipse coverage; this replaces the assumption with a measurement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import geodesy as geo
from .records import wrap180


@dataclass
class CompassObs:
    t_utc: float
    heading_deg: float          # measured array-forward heading = true bearing - measured rel
    sigma_deg: float
    ref_key: str


def radio_compass(rx_lat: float, rx_lon: float, ref_lat: float, ref_lon: float,
                  measured_rel_deg: float, sigma_deg: float, t_utc: float,
                  ref_key: str = "", position_sigma_m: float = 50.0) -> CompassObs:
    """One heading observation from a reference. `measured_rel_deg` is the
    array angle to the reference in the ARRAY frame (clockwise from forward);
    the true bearing comes from the map. heading = true - relative. The
    reference's own position uncertainty adds to σ at the measured range."""
    true_brg = geo.bearing_between(rx_lat, rx_lon, ref_lat, ref_lon)
    rng = max(geo.distance_between(rx_lat, rx_lon, ref_lat, ref_lon), 1.0)
    pos_sig = math.degrees(position_sigma_m / rng)
    heading = (true_brg - measured_rel_deg) % 360.0
    return CompassObs(t_utc=t_utc, heading_deg=heading,
                      sigma_deg=math.hypot(sigma_deg, pos_sig), ref_key=ref_key)


class MountingOffset:
    """The fixed angle between the array's forward mark and the platform axis.

    Estimated as the circular mean of (compass heading − GPS course) while
    moving straight at speed, where GPS course IS the platform axis. Once
    known, it is subtracted so the radio compass reports true heading."""

    def __init__(self):
        self._s = 0.0
        self._c = 0.0
        self.n = 0

    def observe(self, compass_heading_deg: float, gps_course_deg: float) -> None:
        d = math.radians(wrap180(compass_heading_deg - gps_course_deg))
        self._s += math.sin(d)
        self._c += math.cos(d)
        self.n += 1

    @property
    def offset_deg(self) -> float:
        return math.degrees(math.atan2(self._s, self._c)) if self.n else 0.0

    @property
    def sigma_deg(self) -> float:
        if self.n < 2:
            return float("nan")
        r = math.hypot(self._s, self._c) / self.n
        return math.degrees(math.sqrt(max(0.0, -2.0 * math.log(max(r, 1e-9))))) / math.sqrt(self.n)


class ErrorModel:
    """The bearing error as a function of SNR, learned from references.

    `observe(snr_db, residual_deg)` with each reference bearing's residual
    against its known truth; `sigma_for(snr_db)` returns the RMS residual in
    the nearest SNR bin (with a global fallback), which is the σ to attach to a
    TARGET bearing at that SNR and to hand to `fusion`. Bins are 3 dB."""

    def __init__(self, bin_db: float = 3.0, floor_deg: float = 0.5):
        self.bin = float(bin_db)
        self.floor = float(floor_deg)
        self._sum2: dict = {}
        self._n: dict = {}

    def observe(self, snr_db: float, residual_deg: float) -> None:
        b = int(round(snr_db / self.bin))
        self._sum2[b] = self._sum2.get(b, 0.0) + float(residual_deg) ** 2
        self._n[b] = self._n.get(b, 0) + 1

    def _rms(self, keys) -> float:
        s = sum(self._sum2[k] for k in keys)
        n = sum(self._n[k] for k in keys)
        return math.sqrt(s / n) if n else float("nan")

    def sigma_for(self, snr_db: float) -> float:
        b = int(round(snr_db / self.bin))
        for spread in (0, 1, 2):
            keys = [k for k in (b - spread, b, b + spread) if k in self._n]
            if sum(self._n[k] for k in keys) >= 5:
                return max(self.floor, self._rms(keys))
        allk = list(self._n)
        return max(self.floor, self._rms(allk)) if allk else float("nan")

    @property
    def n(self) -> int:
        return sum(self._n.values())


class Catalogue:
    """An in-memory reference catalogue satisfying ports.Catalogue, and a
    learner that promotes a repeatedly-seen unknown into an entry once enough
    bearings fix it (umbrella §5.3: 'grows from your own drives')."""

    def __init__(self, refs=()):
        from .ports import Reference
        self._by_key = {r.key: r for r in refs}
        self._Reference = Reference

    def lookup(self, key: str):
        return self._by_key.get(key)

    def near(self, lat: float, lon: float, radius_m: float) -> list:
        return [r for r in self._by_key.values()
                if geo.distance_between(lat, lon, r.lat, r.lon) <= radius_m]

    def add(self, ref) -> None:
        self._by_key[ref.key] = ref

    def at_frequency(self, fc_hz: float, tol_hz: float = 3_125.0) -> list:
        """References whose catalogued frequency is `fc_hz` (within half the
        narrowest LMR raster) — how `identity.path_evidence` recognises a known
        repeater output, or a repeater input / simplex channel, on a band with
        no duplex rule."""
        f = float(fc_hz)
        return [r for r in self._by_key.values()
                if r.fc_hz and abs(float(r.fc_hz) - f) <= tol_hz]

    def learn(self, key: str, lat: float, lon: float, position_sigma_m: float,
              kind: str = "", fc_hz: float = 0.0) -> None:
        """Record a learned position for a key seen from enough angles."""
        self._by_key[key] = self._Reference(key=key, lat=lat, lon=lon, kind=kind,
                                            fc_hz=fc_hz, position_sigma_m=position_sigma_m,
                                            verified=False, source="learned")
