"""Burst-level integration: one bearing per transmission, not one per frame.

WHY. The KrakenSDR's frames are 50 ms. Estimating a bearing per frame turns a
3 s push-to-talk into sixty bearings — sixty draws of the same multipath, the
same manifold error and the same heading, which fusion then treats as sixty
independent looks and shrinks the ellipse by √60 for nothing (the same failure
`fusion`'s docstring measures for a shared bias, inside one transmission). The
sensitivity budget (`CUED_DF_PLAN.md` §7A) counts on the opposite: ONE
covariance over the WHOLE burst, ~45 dB of snapshot gain, one bearing.

WHAT IT DOES. The engine hands each frame's band-limited, calibrated snapshots
of a burst (`doa.band_snapshots`, element order, every frame calibrated by its
OWN header) to `BurstIntegrator.contribute`, keyed by who is transmitting. The
integrator sums Y·Yᴴ across frames, so the covariance at the end is exactly the
one a single long window would have given — calibrated snapshots from frames
under different passports are the same element-domain quantity and may be
summed; the passports that contributed are recorded.

A burst ENDS when (whichever first):
  * nothing has been added for `gap_s` of stream time (the transmitter keyed
    off — or a decoder stopped reporting it; a short noise-source cal inside a
    burst is shorter than the default gap and does not split it);
  * it has run `max_s` (a carrier left on, or a long static dwell: a bearing
    every `max_s` keeps the display and the log moving);
  * on a moving platform, the heading has changed more than
    `max_heading_change_deg` since it began — integration happens in the ARRAY
    frame, and a vehicle turning mid-burst would smear the relative bearing;
  * the Kraken retunes (a new segment), or the caller flushes (end of a file,
    DF stopped).

WHAT IT DOES NOT DO. Choose an estimator, or make a record: the engine does
both from the finished `Burst` (`Burst.covariance`), exactly as it does for a
single frame, so integrated and per-frame bearings are the same arithmetic on
different amounts of data.

KEYS. A decoder-identified burst is keyed by its emitter key and slot (two
TDMA slots are two radios and never share a covariance). An energy-clocked or
operator-chosen target with no identity is keyed by its RF centre (100 Hz
grid) and bandwidth. The segment is part of every key.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from . import doa as _doa
from .records import EmitterKey

log = logging.getLogger("atkdf.integrate")

#: defaults (the engine passes PlatformMode's values)
GAP_S = 0.25
MAX_S = 5.0


def burst_key(event, segment: int, f_hz: float, bw_hz: float) -> tuple:
    """Who is transmitting, for integration: identity + slot, else frequency."""
    if event is not None and event.id:
        ident = str(EmitterKey(event.protocol, event.system, event.id))
    else:
        ident = f"band:{int(round(float(f_hz) / 100.0)) * 100}:{int(round(float(bw_hz)))}"
    slot = getattr(event, "slot", None) if event is not None else None
    return (ident, slot, int(segment))


def _wrap180(a: float) -> float:
    return (float(a) + 180.0) % 360.0 - 180.0


@dataclass
class Burst:
    """One transmission being integrated (or finished)."""

    bid: int
    bkey: tuple
    event: object                 # the first BurstEvent (identity, sync type)
    f_hz: float                   # the signal's RF centre
    bw_hz: float
    segment: int
    n_el: int
    S: np.ndarray = None          # Σ Y·Yᴴ, element order, calibrated
    cols: int = 0                 # snapshot columns summed
    neff: int = 0                 # effective independent snapshots
    n0: int = 0                   # first stream index (channel 0, inclusive)
    n1: int = 0                   # last stream index (exclusive)
    t0: float = 0.0               # UTC of the first sample integrated
    t1: float = 0.0               # UTC of the last
    tw: float = 0.0               # Σ t_mid · neff (for the snapshot-weighted mid time)
    frames: int = 0
    contributions: int = 0
    passports: list = field(default_factory=list)
    heading0: float | None = None
    snr_db: list = field(default_factory=list)
    channel_db: list | None = None
    pilot_hits: list = field(default_factory=list)
    end_reason: str = ""
    #: UTC of the last frame whose sync word opened the squelch (`engine`
    #: holds the squelch open for SYNC_HOLD_S after it, like a receiver)
    last_sync_t: float | None = None
    _last_frame: int = -1

    def __post_init__(self):
        if self.S is None:
            self.S = np.zeros((self.n_el, self.n_el), dtype=np.complex128)

    def add(self, Y: np.ndarray, neff: int, n_lo: int, n_hi: int, t_lo: float, t_hi: float,
            frame_seq: int, passport: str = "", snr_db: float | None = None,
            channel_db=None) -> None:
        Y = np.asarray(Y, dtype=np.complex128)
        if Y.ndim != 2 or Y.shape[0] != self.n_el:
            raise ValueError(f"snapshots must be ({self.n_el}, K), not {Y.shape}")
        first = self.contributions == 0
        self.S += Y @ Y.conj().T
        self.cols += int(Y.shape[1])
        self.neff += int(max(1, neff))
        if first:
            self.n0, self.t0 = int(n_lo), float(t_lo)
        self.n1, self.t1 = max(self.n1, int(n_hi)), max(self.t1, float(t_hi))
        self.tw += 0.5 * (float(t_lo) + float(t_hi)) * max(1, neff)
        self.contributions += 1
        if frame_seq != self._last_frame:
            self.frames += 1
            self._last_frame = frame_seq
        if passport and passport not in self.passports:
            self.passports.append(passport)
        if snr_db is not None:
            self.snr_db.append(float(snr_db))
        if channel_db:
            self.channel_db = list(channel_db)

    @property
    def t_mid(self) -> float:
        return self.tw / self.neff if self.neff else self.t0

    @property
    def duration_s(self) -> float:
        return max(0.0, self.t1 - self.t0)

    @property
    def key(self) -> str:
        ident = self.bkey[0]
        return "" if ident.startswith("band:") else ident

    def covariance(self, geometry=None, fb=None) -> "_doa.Covariance":
        """The burst's covariance — what `doa.estimate` takes. Forward-backward
        averaging only where the geometry allows it (as `doa.covariance`)."""
        if self.cols <= 0:
            raise ValueError("an empty burst has no covariance")
        R = self.S / self.cols
        R_raw = R
        perm = geometry.reflection_permutation() if geometry is not None else None
        use_fb = bool(fb) if fb is not None else perm is not None
        if use_fb:
            R = _doa.forward_backward(R, perm)
        return _doa.Covariance(R=R, snapshots=int(self.neff), f_hz=float(self.f_hz),
                               bw_hz=float(self.bw_hz), fb=use_fb,
                               R_raw=(R_raw if use_fb else None))

    def quality(self) -> dict:
        """What the record carries about the integration itself."""
        q = {"integration": "burst",
             "integrated_ms": round(1000.0 * self.duration_s, 1),
             "integrated_frames": int(self.frames),
             "n_contributions": int(self.contributions),
             "burst_end": self.end_reason or "open"}
        if len(self.passports) > 1:
            q["passports"] = list(self.passports)
        if self.snr_db:
            q["band_snr_db"] = round(float(np.median(self.snr_db)), 1)
            q["band_snr_db_max"] = round(float(np.max(self.snr_db)), 1)
        if self.channel_db:
            q["channel_db"] = list(self.channel_db)
        return q


class BurstIntegrator:
    """Holds the open bursts; `contribute` adds to (or starts) one and returns
    any it had to finish first; `expire` finishes those whose transmitter has
    gone quiet; `flush` finishes everything."""

    def __init__(self, gap_s: float = GAP_S, max_s: float = MAX_S,
                 max_heading_change_deg: float = 0.0):
        self.gap_s = float(gap_s)
        self.max_s = float(max_s)
        self.max_heading_change_deg = float(max_heading_change_deg)
        self.open: dict = {}
        self._next = 1
        self.finished_count = 0

    def configure(self, gap_s=None, max_s=None, max_heading_change_deg=None) -> None:
        if gap_s is not None:
            self.gap_s = float(gap_s)
        if max_s is not None:
            self.max_s = float(max_s)
        if max_heading_change_deg is not None:
            self.max_heading_change_deg = float(max_heading_change_deg)

    def _finish(self, bkey, reason: str) -> Burst:
        b = self.open.pop(bkey)
        b.end_reason = reason
        self.finished_count += 1
        log.debug("burst %d %s finished (%s): %.0f ms, %d frames, %d snapshots",
                  b.bid, b.bkey[0], reason, 1000 * b.duration_s, b.frames, b.neff)
        return b

    def contribute(self, bkey: tuple, *, event, f_hz: float, bw_hz: float, segment: int,
                   Y, neff: int, n_lo: int, n_hi: int, t_lo: float, t_hi: float, fs: float,
                   frame_seq: int = 0, passport: str = "", snr_db=None, channel_db=None,
                   heading_deg=None) -> tuple:
        """Add one frame's snapshots of a burst. Returns (burst, finished):
        the burst they went into, and any burst finished to make room (the
        same key's previous transmission, ended by a gap, the length cap or a
        turn)."""
        finished = []
        b = self.open.get(bkey)
        if b is not None:
            gap = (int(n_lo) - b.n1) / float(fs)
            reason = ""
            if int(n_lo) < b.n0:
                reason = "stream went backwards"
            elif gap > self.gap_s:
                reason = "gap"
            elif (float(t_hi) - b.t0) > self.max_s:
                reason = "max_integration"
            elif (self.max_heading_change_deg > 0 and heading_deg is not None
                  and b.heading0 is not None
                  and abs(_wrap180(float(heading_deg) - b.heading0)) > self.max_heading_change_deg):
                reason = "turned"
            if reason:
                finished.append(self._finish(bkey, reason))
                b = None
        if b is None:
            n_el = int(np.asarray(Y).shape[0])
            b = Burst(bid=self._next, bkey=bkey, event=event, f_hz=float(f_hz),
                      bw_hz=float(bw_hz), segment=int(segment), n_el=n_el,
                      heading0=(float(heading_deg) if heading_deg is not None else None))
            self._next += 1
            self.open[bkey] = b
            log.debug("burst %d %s opened at %.4f MHz", b.bid, bkey[0], f_hz / 1e6)
        b.add(Y, neff, n_lo, n_hi, t_lo, t_hi, frame_seq, passport, snr_db, channel_db)
        return b, finished

    def expire(self, now_n: int, fs: float) -> list:
        """Finish every burst nothing has been added to for `gap_s` before
        stream index `now_n` (the end of the frame just processed)."""
        out = []
        for k in [k for k, b in self.open.items() if (int(now_n) - b.n1) / float(fs) > self.gap_s]:
            out.append(self._finish(k, "gap"))
        return out

    def flush(self, reason: str = "flush") -> list:
        return [self._finish(k, reason) for k in list(self.open)]
