"""The engine: frames in, bearing records out. What `atkdf.Engine` does per
frame is the core loop the umbrella plan sketches.

    frame -> skip if NOISE_ON / MISALIGNED / not calibrated (why_not_usable)
          -> calibrate (weights AND the fractional-delay slope)
          -> for each decoder burst that lands in this frame:
                 band-limited snapshots in the burst's channel
                 doa.estimate -> a Measurement + a 360-pt likelihood
                 absolute bearing = heading(t) + relative, with heading sigma
                 a BearingRecord, tagged direct/repeated, keyed by identity
          -> (no burst, mobile survey window: one bearing on the strongest peak)

Static vs mobile is a PlatformMode: window length, which estimator, the
quality gate, and how heading sigma is treated. The estimators are the same.

The engine holds only what it needs frame to frame: the current passport, the
platform mode, and a short ring of recent frames so a burst that spans a frame
boundary can be assembled. Everything durable (the registry, the posterior)
is a port or a later module; the engine emits records and lets them consume.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import calibration
from . import doa as _doa
from .atkq import Flag, Frame
from .manifold import Geometry, IdealManifold, Manifold
from .ports import BurstEvent, Fix2D
from .records import BearingRecord, EmitterKey, Passport, wrap360


@dataclass
class PlatformMode:
    """The switch that makes one estimator set behave as static or mobile."""

    name: str = "static"
    window_ms: float = 200.0           # survey window when there is no burst
    estimator: str = "music"
    min_confidence: float = 0.15
    survey: bool = True                # emit a bearing per window without a burst?
    heading_required: bool = False     # drop a bearing that has no heading?

    @classmethod
    def static(cls) -> "PlatformMode":
        return cls(name="static", window_ms=1000.0, estimator="music",
                   survey=True, heading_required=False)

    @classmethod
    def mobile(cls) -> "PlatformMode":
        return cls(name="mobile", window_ms=50.0, estimator="music",
                   survey=False, heading_required=True)


def channel_band(header, event: BurstEvent) -> tuple:
    """The emitter's baseband band [lo, hi] relative to the tuner, from its RF
    centre and bandwidth. Clipped to the digitised band."""
    f0 = (event.fc_hz - header.fc_hz) if event.fc_hz else 0.0
    bw = event.bw_hz or 12_500.0
    half = header.fs_hz * 0.45
    return max(-half, f0 - bw / 2), min(half, f0 + bw / 2)


@dataclass
class Engine:
    """Build one with the geometry, a manifold and the ports it needs. Call
    `process(frame, events=...)` per frame; it returns the BearingRecords that
    frame produced (possibly none). `positioning` gives the receiver's fix and
    heading; without it, bearings are relative only."""

    geometry: Geometry
    manifold: Manifold | None = None
    positioning: object | None = None
    catalogue: object | None = None
    mode: PlatformMode = field(default_factory=PlatformMode.static)
    site_id: str = ""
    manifold_id: str = ""
    passport: Passport | None = None
    _pending: list = field(default_factory=list)
    n_frames: int = 0
    n_records: int = 0

    def __post_init__(self):
        if self.manifold is None:
            self.manifold = IdealManifold(self.geometry)
        self.manifold_id = self.manifold_id or getattr(self.manifold, "id", "") or "ideal"

    # -- the per-frame loop --------------------------------------------------
    def process(self, frame: Frame, events=()) -> list:
        self.n_frames += 1
        h = frame.header
        if h.has(Flag.SEGMENT_START):
            self._pending.clear()
        if not h.usable_for_df:
            return []
        self.passport = calibration.passport_of(
            h, geometry_id=self.geometry.id, manifold_id=self.manifold_id,
            site_id=self.site_id)
        y = self.geometry.element_order(calibration.calibrate(frame))
        s0 = h.stream_sample0
        s1 = s0 + h.samples_per_ch
        out = []
        used_burst = False
        for e in events:
            if e.segment is not None and e.segment != h.segment:
                continue
            n0 = e.n0 if e.n0 is not None else int(round((e.t_utc - h.t_utc) * h.fs_hz) + s0)
            n1 = e.n1 if e.n1 is not None else int(n0 + max(1, e.span_s * h.fs_hz))
            lo, hi = max(n0, s0), min(n1, s1)
            if hi - lo < 64:
                continue
            used_burst = True
            seg = y[:, lo - s0:hi - s0]
            rec = self._bearing(h, seg, frame, band=channel_band(h, e), event=e,
                                span=(n0, n1))
            if rec is not None:
                out.append(rec)
        if not used_burst and self.mode.survey:
            rec = self._bearing(h, y, frame, band=None, event=None, span=(s0, s1))
            if rec is not None:
                out.append(rec)
        self.n_records += len(out)
        return out

    # -- one bearing ---------------------------------------------------------
    def _bearing(self, h, seg, frame, band, event, span) -> BearingRecord | None:
        if seg.shape[1] < 64:
            return None
        f_hz = float(event.fc_hz) if (event and event.fc_hz) else float(h.fc_hz)
        if band is not None:
            # an FFT fine enough to put ~16 bins in the channel, so MDL sees a
            # clean rank-1-plus-white-noise spectrum, but never longer than the
            # burst
            bw = max(band[1] - band[0], 1.0)
            nfft = int(2 ** round(math.log2(max(64, min(seg.shape[1] // 4,
                                                        16 * h.fs_hz / bw)))))
            Y, _freqs, neff = _doa.band_snapshots(seg, h.fs_hz, band[0], band[1], nfft=nfft)
            cov = _doa.covariance(Y, f_hz, geometry=self.geometry, snapshots=neff,
                                  bw_hz=band[1] - band[0])
            src = cov
        else:
            src = seg
        try:
            res = _doa.estimate(src, self.manifold, f_hz, method=self.mode.estimator,
                                geometry=self.geometry)
        except Exception:                                   # noqa: BLE001
            return None
        if res.quality.get("confidence", 0.0) < self.mode.min_confidence:
            return None

        fix = self.positioning.at(h.t_utc) if self.positioning is not None else None
        heading = fix.course_deg if fix is not None else None
        heading_sig = self._heading_sigma(fix) if fix is not None else None
        if heading is None and self.mode.heading_required:
            return None
        theta_abs = wrap360(heading + res.theta_deg) if heading is not None else None
        sigma = res.sigma_deg
        if heading_sig is not None:
            sigma = math.hypot(res.sigma_deg, heading_sig)

        key = ""
        path_tag = "unknown"
        slot = sync = ""
        if event is not None and event.protocol:
            key = str(EmitterKey(event.protocol, event.system, event.id))
            path_tag = self._path_tag(event)
            slot, sync = event.slot, event.sync_type

        rec = BearingRecord(
            t_utc=h.t_utc, theta_rel_deg=float(res.theta_deg), sigma_deg=float(sigma),
            estimator=res.estimator, segment=int(h.segment), sample_span=tuple(span),
            fc_hz=f_hz, bw_hz=(band[1] - band[0]) if band else 0.0,
            emitter_key=key, slot=slot, sync_type=sync or "", path_tag=path_tag,
            heading_deg=heading, heading_sigma_deg=heading_sig, theta_abs_deg=theta_abs,
            quality=dict(res.quality), likelihood=res.likelihood,
            lat=(fix.lat if fix else None), lon=(fix.lon if fix else None),
            fix_age_s=(fix.fix_age_s if fix else None),
            platform_mode=self.mode.name, site_id=self.site_id,
            passport_uid=(self.passport.uid if self.passport else ""),
            manifold=getattr(self.manifold, "kind", "ideal"),
            notes=list(res.caveats))
        return rec

    def _heading_sigma(self, fix: Fix2D) -> float:
        """How well the GPS course is known: undefined at a stop, noisy slow,
        good at speed. A crude but honest model until `heading.py` and the
        radio compass replace it."""
        if fix.speed_mps < 0.5:
            return 45.0
        if fix.speed_mps < 3.0:
            return 20.0
        return max(3.0, 8.0 * (1.0 + fix.fix_age_s / 5.0) / max(1.0, fix.speed_mps / 5.0))

    def _path_tag(self, event: BurstEvent) -> str:
        """direct / repeated / unknown. DMR's sync pattern says BS- or
        MS-sourced; a BS-sourced (outbound) burst carries the handset's ID
        with the REPEATER's bearing and must never be fused as the handset's
        position (umbrella 7.1). The catalogue's repeater plan is the other
        source when the sync type is silent."""
        st = (event.sync_type or "").lower()
        if st in ("bs", "base", "bs_voice", "bs_data", "outbound"):
            return "repeated"
        if st in ("ms", "mobile", "ms_voice", "ms_data", "direct", "dmo"):
            return "direct"
        if self.catalogue is not None and event.fc_hz:
            ref = self.catalogue.lookup(str(EmitterKey(event.protocol, event.system, event.id)))
            if ref is not None and ref.kind in ("repeater_output", "bs"):
                return "repeated"
        return "unknown"
