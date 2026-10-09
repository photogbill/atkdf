"""Where the array is and which way it faces, when it is not moving — and
setting that heading from a reference instead of a compass.

THE STATIC CASE (Bill's first DF test, 2026-10-07: a table or a tripod, five
whips, a known transmitter). The engine needs the ARRAY's forward heading to
turn a relative bearing into a true one. A GPS course cannot give it — the
array is not moving, and the engine's GPS-course model says so with a 45°
sigma at a stop. `FixedSite` supplies the heading as what it is: entered (a
compass on the forward mark) or aligned (below), with a small sigma of its
own, and the position if one is known. Without a position the bearings are
still true bearings; they just cannot be drawn on a map.

ALIGNING ON A REFERENCE (the radio compass, umbrella §5.3, in its simplest
form). A transmitter at a KNOWN position gives a known true bearing from the
array. The array measures a relative bearing to it. The heading that makes
the two agree is

    heading = expected_true − measured_relative

taken as a circular mean over many bearings, so one bad one does not set it.
The spread of those bearings is reported with it: a wide spread means the
reference is not clean (multipath, a second emitter, too weak) and the
heading should not be trusted — which is said, not silently applied.
"""

from __future__ import annotations

import math

import numpy as np

from .ports import Fix2D


class FixedSite:
    """A receiver that stays put. `lat`/`lon` may be None (no position:
    bearings stay true bearings but carry no location). `heading_deg` is the
    array's forward heading, degrees true (element 0's direction for a UCA);
    None means unknown, and bearings are then relative only."""

    def __init__(self, lat: float | None = None, lon: float | None = None,
                 heading_deg: float | None = None, heading_sigma_deg: float = 2.0,
                 sigma_m: float = 5.0):
        self.lat = None if lat is None else float(lat)
        self.lon = None if lon is None else float(lon)
        self.heading_deg = None if heading_deg is None else float(heading_deg) % 360.0
        self.heading_sigma_deg = float(heading_sigma_deg)
        self.sigma_m = float(sigma_m)

    def at(self, t_utc: float) -> Fix2D:
        return Fix2D(lat=self.lat, lon=self.lon, t_utc=float(t_utc),
                     course_deg=None, speed_mps=0.0, fix_age_s=0.0,
                     sigma_m=self.sigma_m, heading_deg=self.heading_deg,
                     heading_sigma_deg=(self.heading_sigma_deg
                                        if self.heading_deg is not None else None))


def wrap180(a):
    """Degrees into (−180, 180]."""
    return (np.asarray(a, dtype=float) + 180.0) % 360.0 - 180.0


def circular_mean_deg(angles_deg) -> tuple:
    """(mean in [0, 360), circular standard deviation in degrees)."""
    a = np.radians(np.asarray(angles_deg, dtype=float))
    if a.size == 0:
        return float("nan"), float("nan")
    c, s = float(np.mean(np.cos(a))), float(np.mean(np.sin(a)))
    r = math.hypot(c, s)
    mean = math.degrees(math.atan2(s, c)) % 360.0
    std = math.degrees(math.sqrt(-2.0 * math.log(max(r, 1e-12))))
    return mean, std


def align_heading(measured_rel_deg, expected_true_deg: float) -> dict:
    """The array heading that makes measured RELATIVE bearings to a reference
    agree with its known TRUE bearing. Returns {heading_deg, spread_deg, n,
    trustworthy, sentence}. `trustworthy` is False when the bearings scatter
    by more than 10° or there are fewer than 5 of them."""
    rel = np.asarray(list(measured_rel_deg), dtype=float)
    n = int(rel.size)
    if n == 0:
        return {"heading_deg": None, "spread_deg": None, "n": 0,
                "trustworthy": False,
                "sentence": "no bearings on the reference yet"}
    mean_rel, spread = circular_mean_deg(rel)
    heading = (float(expected_true_deg) - mean_rel) % 360.0
    ok = n >= 5 and spread <= 10.0
    why = ("" if ok else
           (" — too few bearings" if n < 5 else
            f" — they scatter {spread:.1f}°, so the reference is not clean "
            f"(multipath, another emitter, or too weak)"))
    return {"heading_deg": heading, "spread_deg": spread, "n": n,
            "trustworthy": ok,
            "sentence": (f"array heading {heading:.1f}° true from {n} bearing(s), "
                         f"spread {spread:.1f}°{why}")}
