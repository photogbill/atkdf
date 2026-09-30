"""Applying atkdaq's receiver calibration, and keeping track of it.

WHAT THE HEADER SAYS, AND WHAT IT MEANS. atkdaq aligns the channels to the
nearest sample and REPORTS the rest: per channel, a complex weight w_i and a
fractional delay frac_i (samples, channel i behind channel 0). Its model,
fitted on the noise source, is

    X_i(f) = X_0(f) · exp(−j·2π·f·frac_i/fs) / w_i

with f the baseband offset from the tuner in Hz. So the correction is

    Y_i(f) = w_i · X_i(f) · exp(+j·2π·f·frac_i/fs)

and for a narrowband signal at baseband offset f0 it is the single factor
c_i(f0) = w_i · exp(+j·2π·f0·frac_i/fs). **Applying w alone is right only at
band centre**: at 2.4 MSPS a 0.3-sample residual is 22° on a signal 500 kHz
off centre, which is a bearing error that looks like a manifold error and
is not one. Every path in this module applies both.

THE PASSPORT. Every frame names the calibration it was aligned and weighted
by (`passport`); every bearing records it (`records.Passport`), so "which of
these bearings were measured under the cal that went bad at 14:02?" is a
filter, not an investigation.

THE ON-AIR MONITOR. The noise source calibrates the receivers, periodically,
blind. A catalogued reference in view does it continuously and for free:
while parked, its steering vector should not move. When it does, the
receivers' calibration has moved, and the monitor says so — recalibrate
when the reference says so, not on a timer (umbrella §5.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .atkq import Frame, Header
from .records import Passport


def factors(header: Header, f0_hz: float = 0.0) -> np.ndarray:
    """c_i(f0) = w_i · exp(+j·2π·f0·frac_i/fs), one per receiver channel."""
    w = header.weights
    frac = header.frac_delays
    return w * np.exp(2j * np.pi * float(f0_hz) * frac / float(header.fs_hz))


def band_factors(header: Header, freqs_hz) -> np.ndarray:
    """(n_channels, K) correction for spectra at baseband offsets `freqs_hz`."""
    f = np.asarray(freqs_hz, dtype=float).reshape(1, -1)
    return header.weights[:, None] * np.exp(
        2j * np.pi * f * header.frac_delays[:, None] / float(header.fs_hz))


def apply(x: np.ndarray, header: Header, f0_hz: float | None = None) -> np.ndarray:
    """Calibrate receiver-channel samples `x` (n, N) by the header's numbers.

    With `f0_hz` given, the narrowband factor for a signal at that baseband
    offset (cheap, exact for a signal much narrower than fs/frac). Without,
    the full correction across the band by FFT — each bin gets its own
    phase slope. The FFT treats the block as periodic, so the first and last
    few samples carry the wrap; a fractional delay under a sample moves
    them by under a sample's worth, which no estimator here can see.
    """
    x = np.asarray(x)
    if f0_hz is not None:
        return (x * factors(header, f0_hz)[:, None]).astype(np.complex64)
    n = x.shape[1]
    X = np.fft.fft(x, axis=1)
    f = np.fft.fftfreq(n, 1.0 / header.fs_hz)
    return np.fft.ifft(X * band_factors(header, f), axis=1).astype(np.complex64)


def calibrate(frame: Frame, f0_hz: float | None = None) -> np.ndarray:
    """A frame's payload, calibrated: (n_channels, N) complex64."""
    return apply(frame.iq(), frame.header, f0_hz)


def passport_of(header: Header, geometry_id: str = "", manifold_id: str = "",
                site_id: str = "", device: str | None = None) -> Passport:
    """The Passport a frame's header describes."""
    dev = device or ("synthetic" if header.device_kind == 2 else "kraken")
    return Passport(session=header.session_ns, passport=int(header.passport),
                    t_utc=header.t_utc, fc_hz=float(header.fc_hz), fs_hz=float(header.fs_hz),
                    gain_tenths_db=int(header.gain_tenths_db),
                    delays=[int(d) for d in header.delays],
                    weights=[[float(w.real), float(w.imag)] for w in header.weights],
                    frac_delays=[float(f) for f in header.frac_delays],
                    spread_deg=header.spread_deg,
                    drops=[int(c.drops) for c in header.channels],
                    segment=int(header.segment), device=dev,
                    geometry_id=geometry_id, manifold_id=manifold_id, site_id=site_id)


#: How old a calibration may be before its bearings are flagged, by platform
#: mode. A tuner's phase drifts with temperature; a vehicle's box heats and
#: cools faster than a tripod's. These are the thresholds for a CAVEAT, not
#: a refusal — atkdaq's own policy recalibrates, and the on-air monitor is
#: the better judge whenever a reference is in view.
CAL_AGE_S = {"static": (900.0, 3600.0), "mobile": (120.0, 600.0)}


def cal_age_verdict(header: Header, mode: str = "static") -> tuple:
    """('fresh' | 'aging' | 'stale', sentence)."""
    aging, stale = CAL_AGE_S.get(mode, CAL_AGE_S["static"])
    age = header.cal_age_ms / 1000.0
    if age < aging:
        return "fresh", ""
    if age < stale:
        return "aging", (f"the receiver calibration is {age / 60:.0f} min old; "
                         f"tuner phases drift with temperature")
    return "stale", (f"the receiver calibration is {age / 60:.0f} min old, past the "
                     f"{stale / 60:.0f} min this platform mode trusts")


@dataclass
class CoherenceMonitor:
    """Watches a reference emitter's steering vector while parked.

    `observe(vector, passport_uid)` with the measured array response to the
    reference (any scale; normalised here to element 0). The first vector
    after a new passport is the baseline. `drift_deg` is the largest
    per-element phase change since then; after `persist` consecutive
    observations beyond `threshold_deg` the monitor asks for a
    recalibration (`needs_cal`) and says which element moved.

    Only meaningful when neither the array nor the reference has moved: the
    caller feeds it only while the platform is stationary.
    """

    threshold_deg: float = 5.0
    persist: int = 3
    baseline: np.ndarray | None = None
    passport_uid: str = ""
    drift_deg: float = 0.0
    worst_element: int = -1
    over: int = 0
    needs_cal: bool = False
    history: list = field(default_factory=list)

    @staticmethod
    def _norm(v) -> np.ndarray:
        v = np.asarray(v, dtype=complex).reshape(-1)
        return v / (v[0] / abs(v[0])) / np.linalg.norm(v) if abs(v[0]) > 0 else v

    def observe(self, vector, passport_uid: str = "") -> dict:
        v = self._norm(vector)
        if self.baseline is None or passport_uid != self.passport_uid:
            self.baseline = v
            self.passport_uid = passport_uid
            self.over = 0
            self.needs_cal = False
            self.drift_deg = 0.0
            self.worst_element = -1
            return {"state": "baseline", "drift_deg": 0.0}
        d = np.degrees(np.angle(v * np.conj(self.baseline)))
        self.worst_element = int(np.argmax(np.abs(d)))
        self.drift_deg = float(np.max(np.abs(d)))
        self.history.append(self.drift_deg)
        self.over = self.over + 1 if self.drift_deg > self.threshold_deg else 0
        if self.over >= self.persist:
            self.needs_cal = True
        return {"state": "recalibrate" if self.needs_cal else "steady",
                "drift_deg": self.drift_deg, "element": self.worst_element}

    def sentence(self) -> str:
        if self.baseline is None:
            return "no reference observed yet"
        if self.needs_cal:
            return (f"the reference's steering vector has moved {self.drift_deg:.1f}° on "
                    f"element {self.worst_element} since the calibration — the receivers' "
                    f"calibration has drifted; recalibrate")
        return f"reference steady ({self.drift_deg:.1f}° since the calibration)"
