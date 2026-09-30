"""Vehicle heading from GPS, with an honest sigma — and a radio compass to pull
it back when a reference is in view.

GPS gives position, not heading. Course over ground is trustworthy at speed on
a straight run and undefined at a stop; a bearing taken mid-turn with a stale
heading is 45 deg wrong at 90 deg/s (umbrella §8.2). So heading is a filtered
estimate with a sigma that GROWS at a stop and in a turn and SHRINKS on a
radio-compass observation, and every bearing is converted to absolute with
that sigma — a mid-turn bearing comes out WIDE, not dropped.

The filter is a constant-turn-rate model on the GPS course (a small analytic
Kalman update, not a library dependency): state is (heading, turn_rate). At a
stop, course is unobservable, so the update is skipped and the heading simply
ages (sigma grows). A radio-compass observation — the measured array angle to a
catalogued emitter minus its true bearing from the map (`references`, Phase 2)
— is a direct heading measurement and is fused in with its own sigma.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .records import wrap180, wrap360


def _wrap_rad(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class Heading:
    heading_deg: float
    sigma_deg: float
    turn_rate_dps: float = 0.0
    t_utc: float = 0.0
    source: str = ""


class HeadingFilter:
    """Constant-turn-rate heading from GPS course, plus compass fixes.

    Tuning, all in degrees / seconds:
      process_sigma_dps   how fast the true turn rate can change (random walk)
      stop_speed_mps      below this, course is unobservable (a stop)
      slow_speed_mps      below this, course is noisy; its sigma is inflated
      max_sigma_deg       the cap while unobserved — a heading this uncertain
                          is 'unknown', and a bearing built on it is very wide
    """

    def __init__(self, process_sigma_dps: float = 8.0, stop_speed_mps: float = 0.7,
                 slow_speed_mps: float = 2.5, base_course_sigma_deg: float = 4.0,
                 max_sigma_deg: float = 90.0):
        self.q = float(process_sigma_dps)
        self.stop = float(stop_speed_mps)
        self.slow = float(slow_speed_mps)
        self.base = float(base_course_sigma_deg)
        self.max_sigma = float(max_sigma_deg)
        self.h = None                # heading rad
        self.rate = 0.0              # deg/s
        self.var = (max_sigma_deg) ** 2   # heading variance, deg^2
        self.t = None

    def _predict(self, t: float) -> None:
        if self.t is None:
            self.t = t
            return
        dt = max(0.0, t - self.t)
        self.t = t
        if self.h is not None:
            self.h = _wrap_rad(self.h + math.radians(self.rate) * dt)
        # the heading drifts by the (uncertain) turn rate; variance grows with dt
        self.var = min(self.max_sigma ** 2, self.var + (self.q * dt) ** 2)

    def update_gps(self, t_utc: float, course_deg, speed_mps: float,
                   fix_sigma_deg: float | None = None) -> Heading:
        """Fold in a GPS fix. At a stop, course is unobservable and only the
        prediction runs (sigma grows). At speed, course is a heading
        measurement whose sigma is inflated when slow."""
        self._predict(t_utc)
        if course_deg is not None and speed_mps >= self.stop:
            # course noise: base at speed, inflated toward a stop
            if speed_mps < self.slow:
                meas_sig = self.base * (self.slow / max(speed_mps, 1e-3))
            else:
                meas_sig = self.base
            if fix_sigma_deg:
                meas_sig = math.hypot(meas_sig, fix_sigma_deg)
            self._fuse(math.radians(course_deg), meas_sig, t_utc)
        return self.current(t_utc)

    def update_compass(self, t_utc: float, heading_deg: float, sigma_deg: float) -> Heading:
        """A radio-compass heading observation (reference angle vs map bearing).
        Valid at a stop and in a turn, so it is what rescues heading when GPS
        course is useless."""
        self._predict(t_utc)
        self._fuse(math.radians(heading_deg), sigma_deg, t_utc, is_turn_obs=False)
        return self.current(t_utc)

    def _fuse(self, meas_rad: float, meas_sigma_deg: float, t: float,
              is_turn_obs: bool = True) -> None:
        if self.h is None:
            self.h = meas_rad
            self.var = meas_sigma_deg ** 2
            return
        r = meas_sigma_deg ** 2
        innov = _wrap_rad(meas_rad - self.h)
        gain = self.var / (self.var + r)
        self.h = _wrap_rad(self.h + gain * innov)
        # a GPS course update also informs the turn rate (change of heading /
        # dt); a compass fix does not move the platform, so it does not
        new_var = (1 - gain) * self.var
        if is_turn_obs:
            self.rate = math.degrees(gain * innov) / max(1e-3, self._dt_since_rate(t))
        self.var = new_var
        self._last_rate_t = t

    def _dt_since_rate(self, t: float) -> float:
        prev = getattr(self, "_last_rate_t", None)
        return (t - prev) if prev is not None else 1.0

    def current(self, t_utc: float | None = None) -> Heading:
        if t_utc is not None and self.t is not None and t_utc > self.t:
            self._predict(t_utc)
        h = 0.0 if self.h is None else math.degrees(self.h) % 360.0
        return Heading(heading_deg=h, sigma_deg=min(self.max_sigma, math.sqrt(self.var)),
                       turn_rate_dps=self.rate, t_utc=(t_utc if t_utc is not None else self.t or 0.0))
