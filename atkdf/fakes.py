"""Fake ports: the smallest thing that satisfies each protocol, for tests and
for the parts of `sim` that do not need a real model. None of these pretends
to be more than it is -- a flat earth is flat, free space is free space, an
empty catalogue is empty -- so a test that needs realism reaches for `sim`.
"""

from __future__ import annotations

import math
from typing import Iterator

import numpy as np

from .geodesy import bearing_between, distance_between
from .ports import (BurstEvent, Catalogue, CueSource, Detection, Edge, Elevation,
                    Fix2D, PathLoss, Positioning, Profile, Reference, RoadGraph)


class StaticPositioning:
    """The receiver never moves. `course_deg` is the array's forward heading
    (a tripod points somewhere), σ small and constant."""

    def __init__(self, lat: float, lon: float, heading_deg: float = 0.0,
                 sigma_m: float = 5.0):
        self.lat, self.lon = float(lat), float(lon)
        self.heading_deg = float(heading_deg)
        self.sigma_m = float(sigma_m)

    def at(self, t_utc: float) -> Fix2D:
        return Fix2D(lat=self.lat, lon=self.lon, t_utc=float(t_utc),
                     course_deg=self.heading_deg, speed_mps=0.0,
                     fix_age_s=0.0, sigma_m=self.sigma_m)


class TrackPositioning:
    """A GPS track, interpolated. `rows` are (t_utc, lat, lon[, course, speed]).
    Course and speed are derived from the two nearest fixes when not given;
    σ grows with the gap to the nearest fix (a fix far in time is a wide one)."""

    def __init__(self, rows, sigma_m: float = 8.0, sigma_grow_mps: float = 3.0):
        r = sorted(rows, key=lambda x: x[0])
        self.t = np.array([x[0] for x in r], dtype=float)
        self.lat = np.array([x[1] for x in r], dtype=float)
        self.lon = np.array([x[2] for x in r], dtype=float)
        self.course = np.array([x[3] if len(x) > 3 and x[3] is not None else np.nan
                                for x in r], dtype=float)
        self.speed = np.array([x[4] if len(x) > 4 and x[4] is not None else np.nan
                               for x in r], dtype=float)
        self.sigma_m = float(sigma_m)
        self.sigma_grow = float(sigma_grow_mps)

    def at(self, t_utc: float) -> Fix2D:
        t = float(t_utc)
        i = int(np.clip(np.searchsorted(self.t, t), 1, len(self.t) - 1))
        t0, t1 = self.t[i - 1], self.t[i]
        f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        lat = self.lat[i - 1] + f * (self.lat[i] - self.lat[i - 1])
        lon = self.lon[i - 1] + f * (self.lon[i] - self.lon[i - 1])
        d = distance_between(self.lat[i - 1], self.lon[i - 1], self.lat[i], self.lon[i])
        dt = max(t1 - t0, 1e-6)
        speed = d / dt
        course = bearing_between(self.lat[i - 1], self.lon[i - 1], self.lat[i], self.lon[i]) \
            if d > 0.5 else (self.course[i] if not np.isnan(self.course[i]) else None)
        age = min(abs(t - t0), abs(t - t1))
        return Fix2D(lat=float(lat), lon=float(lon), t_utc=t, course_deg=course,
                     speed_mps=float(speed), fix_age_s=float(age),
                     sigma_m=self.sigma_m + self.sigma_grow * age)


class FlatEarth:
    """No terrain: a profile of zeros. Turns the terrain likelihood into a
    no-op so a test can isolate everything else."""

    def profile(self, a, b, step_m: float = 30.0) -> Profile:
        d = distance_between(a[0], a[1], b[0], b[1])
        n = max(2, int(d / max(step_m, 1.0)) + 1)
        return Profile(dist_m=np.linspace(0, d, n), height_m=np.zeros(n),
                       resolution_m=float(step_m))


class RidgeEarth:
    """One triangular ridge of `height_m` at `ridge_frac` along every profile:
    enough terrain to prove the shadowing likelihood does something, without a
    DEM."""

    def __init__(self, height_m: float = 150.0, ridge_frac: float = 0.5):
        self.height_m = float(height_m)
        self.ridge_frac = float(ridge_frac)

    def profile(self, a, b, step_m: float = 30.0) -> Profile:
        d = distance_between(a[0], a[1], b[0], b[1])
        n = max(3, int(d / max(step_m, 1.0)) + 1)
        x = np.linspace(0, 1, n)
        h = self.height_m * (1.0 - np.abs(x - self.ridge_frac) / max(self.ridge_frac,
                                                                    1 - self.ridge_frac))
        return Profile(dist_m=np.linspace(0, d, n), height_m=np.maximum(h, 0.0),
                       resolution_m=float(step_m))


class FreeSpaceLoss:
    """Free-space path loss plus a knife-edge diffraction penalty for the
    worst obstruction in the profile. A model, honest about being one."""

    def loss_db(self, profile: Profile, f_hz: float, h_tx: float, h_rx: float) -> float:
        d = float(profile.dist_m[-1]) or 1.0
        lam = 299_792_458.0 / f_hz
        fspl = 20 * math.log10(4 * math.pi * d / lam)
        h = np.asarray(profile.height_m, dtype=float)
        if h.size >= 3 and float(np.max(h)) > 0:
            dm = profile.dist_m
            los = h_tx + (h_rx - h_tx) * dm / d
            clear = h - los
            i = int(np.argmax(clear))
            if clear[i] > 0:
                d1 = max(dm[i], 1.0)
                d2 = max(d - dm[i], 1.0)
                v = clear[i] * math.sqrt(2 * (d1 + d2) / (lam * d1 * d2))
                if v > -0.78:
                    fspl += 6.9 + 20 * math.log10(math.hypot(v - 0.1, 1) + v - 0.1)
        return float(fspl)


class EmptyCatalogue:
    def lookup(self, key: str):
        return None

    def near(self, lat, lon, radius_m):
        return []


class ListCatalogue:
    """A catalogue backed by a list of References."""

    def __init__(self, refs=()):
        self.refs = list(refs)

    def lookup(self, key: str):
        return next((r for r in self.refs if r.key == key), None)

    def near(self, lat, lon, radius_m):
        return [r for r in self.refs
                if distance_between(lat, lon, r.lat, r.lon) <= radius_m]

    def at_frequency(self, fc_hz: float, tol_hz: float = 3_125.0):
        return [r for r in self.refs if r.fc_hz and abs(r.fc_hz - float(fc_hz)) <= tol_hz]


class NoRoads:
    def nearest_edge(self, lat, lon):
        return None

    def neighbours(self, edge):
        return []

    def route_time_s(self, a, b):
        return distance_between(a[0], a[1], b[0], b[1]) / 13.4    # ~30 mph


class ListEvents:
    """A fixed list of BurstEvents, replayed in time order."""

    def __init__(self, events=()):
        self._events = sorted(events, key=lambda e: e.t_utc)

    def events(self) -> Iterator[BurstEvent]:
        yield from self._events


class ListCues:
    """A `CueSource` over a list of Detections, in time order — the scripted
    cue stream the scheduler is tested against. `replay(scheduler)` drives a
    scheduler through them, calling `on_cue` at each detection's `t0`,
    `on_burst_end` at its `t1`, and `tick` in between at `tick_s`, which is
    how a scripted drive runs without a clock."""

    def __init__(self, detections, tick_s: float = 0.5):
        self._dets = sorted(detections, key=lambda d: d.t0)
        self.tick_s = float(tick_s)

    def cues(self) -> Iterator[Detection]:
        return iter(self._dets)

    def replay(self, scheduler, until: float | None = None) -> list:
        """Returns the scheduler's decisions, in order."""
        events = []
        for d in self._dets:
            events.append((d.t0, 0, "cue", d))
            if d.t1 is not None:
                events.append((d.t1, 1, "end", d))
        t_end = until if until is not None else (max(t for t, *_ in events) + 10.0
                                                 if events else 0.0)
        t = 0.0
        while t <= t_end + 1e-9:
            events.append((t, -1, "tick", None))
            t += self.tick_s
        events.sort(key=lambda e: (e[0], e[1]))
        out = []
        for t, _order, kind, d in events:
            if kind == "tick":
                out.extend(scheduler.tick(t))
            elif kind == "cue":
                r = scheduler.on_cue(d, t)
                if r is not None:
                    out.append(r)
            else:
                scheduler.on_burst_end(d, t)
        return out

