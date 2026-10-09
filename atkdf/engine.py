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
    #: An energy gate for band work (a decoder burst, or an operator's DF
    #: target): when the band's power over the floor (`doa.band_snapshots`
    #: stats) is below this, no bearing is produced — a handheld that is not
    #: keyed is not a bearing of the noise. None = no gate (the decoder
    #: already said something transmitted).
    gate_snr_db: float | None = None
    #: What the last frame came to, for a live display: state is one of
    #: unusable | no_target | squelched | low_confidence | bearing | error,
    #: with the band SNR, per-channel band power and why. Replaced, never
    #: mutated, so a reader on another thread sees a whole dict.
    last_status: dict = field(default_factory=dict)
    #: The DoaResult behind the last bearing (its pseudo-spectrum is what a
    #: polar display draws).
    last_result: object | None = None
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
            self.last_status = {"t_utc": h.t_utc, "state": "unusable",
                                "why": "; ".join(h.why_not_usable()),
                                "passport": int(h.passport)}
            return []
        self.passport = calibration.passport_of(
            h, geometry_id=self.geometry.id, manifold_id=self.manifold_id,
            site_id=self.site_id)
        # Two views of the frame, each made only if something needs it. Band
        # work (a burst, a DF target) calibrates PER FFT BIN inside
        # band_snapshots — exact, and it skips the whole-band FFT
        # calibration that costs two thirds of a frame's time at 2.4 MSPS.
        # The survey path (no band) still uses the whole-band calibration.
        cache: dict = {}

        def raw():                        # receiver order, uncalibrated
            if "raw" not in cache:
                cache["raw"] = frame.iq()
            return cache["raw"]

        def calibrated():                 # element order, whole-band calibrated
            if "cal" not in cache:
                cache["cal"] = self.geometry.element_order(calibration.calibrate(frame))
            return cache["cal"]

        s0 = h.stream_sample0
        s1 = s0 + h.samples_per_ch
        out = []
        used_burst = False
        self.last_status = {"t_utc": h.t_utc, "state": "no_target",
                            "passport": int(h.passport)}
        for e in events:
            if e.segment is not None and e.segment != h.segment:
                continue
            n0 = e.n0 if e.n0 is not None else int(round((e.t_utc - h.t_utc) * h.fs_hz) + s0)
            n1 = e.n1 if e.n1 is not None else int(n0 + max(1, e.span_s * h.fs_hz))
            lo, hi = max(n0, s0), min(n1, s1)
            if hi - lo < 64:
                continue
            used_burst = True
            rec = self._bearing(h, raw()[:, lo - s0:hi - s0], frame,
                                band=channel_band(h, e), event=e, span=(n0, n1))
            if rec is not None:
                out.append(rec)
        if not used_burst and self.mode.survey:
            rec = self._bearing(h, calibrated(), frame, band=None, event=None,
                                span=(s0, s1))
            if rec is not None:
                out.append(rec)
        self.n_records += len(out)
        return out

    # -- one bearing ---------------------------------------------------------
    def _bearing(self, h, seg, frame, band, event, span) -> BearingRecord | None:
        """`seg` is RAW receiver-order samples when `band` is given (calibrated
        per bin here), and calibrated element-order samples when it is not."""
        if seg.shape[1] < 64:
            return None
        f_hz = float(event.fc_hz) if (event and event.fc_hz) else float(h.fc_hz)
        status = {"t_utc": h.t_utc, "state": "bearing", "passport": int(h.passport),
                  "fc_hz": f_hz, "bw_hz": (band[1] - band[0]) if band else 0.0}
        if band is not None:
            # an FFT fine enough to put ~16 bins in the channel, so MDL sees a
            # clean rank-1-plus-white-noise spectrum, but never longer than the
            # burst
            bw = max(band[1] - band[0], 1.0)
            nfft = int(2 ** round(math.log2(max(64, min(seg.shape[1] // 4,
                                                        16 * h.fs_hz / bw)))))
            st: dict = {}
            Y, _freqs, neff = _doa.band_snapshots(seg, h.fs_hz, band[0], band[1],
                                                  nfft=nfft, header=h, stats=st)
            Y = self.geometry.element_order(Y)
            status["band_snr_db"] = round(st.get("snr_db", 0.0), 1)
            status["channel_db"] = st.get("channel_db", [])
            if self.gate_snr_db is not None and st.get("snr_db", 0.0) < self.gate_snr_db:
                status["state"] = "squelched"
                status["why"] = (f"band {st.get('snr_db', 0.0):.1f} dB over the floor, "
                                 f"gate {self.gate_snr_db:.1f} dB")
                self.last_status = status
                return None
            cov = _doa.covariance(Y, f_hz, geometry=self.geometry, snapshots=neff,
                                  bw_hz=band[1] - band[0])
            src = cov
        else:
            src = seg
        try:
            res = _doa.estimate(src, self.manifold, f_hz, method=self.mode.estimator,
                                geometry=self.geometry)
        except Exception as exc:                            # noqa: BLE001
            status["state"] = "error"
            status["why"] = f"{type(exc).__name__}: {exc}"
            self.last_status = status
            return None
        status["theta_rel_deg"] = float(res.theta_deg)
        status["confidence"] = float(res.quality.get("confidence", 0.0))
        self.last_result = res
        if res.quality.get("confidence", 0.0) < self.mode.min_confidence:
            status["state"] = "low_confidence"
            status["why"] = (f"confidence {res.quality.get('confidence', 0.0):.2f} "
                             f"< {self.mode.min_confidence:.2f}")
            self.last_status = status
            return None

        fix = self.positioning.at(h.t_utc) if self.positioning is not None else None
        if fix is not None and getattr(fix, "heading_deg", None) is not None:
            # the array's own heading (tripod, aligned, or mounting-corrected)
            heading = float(fix.heading_deg)
            hs = getattr(fix, "heading_sigma_deg", None)
            heading_sig = float(hs) if hs is not None else 2.0
        else:
            heading = fix.course_deg if fix is not None else None
            heading_sig = self._heading_sigma(fix) if fix is not None else None
        if heading is None and self.mode.heading_required:
            status["state"] = "low_confidence"
            status["why"] = "no heading (mobile mode needs one)"
            self.last_status = status
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

        quality = dict(res.quality)
        if "band_snr_db" in status:
            quality["band_snr_db"] = status["band_snr_db"]
            quality["channel_db"] = status["channel_db"]
        rec = BearingRecord(
            t_utc=h.t_utc, theta_rel_deg=float(res.theta_deg), sigma_deg=float(sigma),
            estimator=res.estimator, segment=int(h.segment), sample_span=tuple(span),
            fc_hz=f_hz, bw_hz=(band[1] - band[0]) if band else 0.0,
            emitter_key=key, slot=slot, sync_type=sync or "", path_tag=path_tag,
            heading_deg=heading, heading_sigma_deg=heading_sig, theta_abs_deg=theta_abs,
            quality=quality, likelihood=res.likelihood,
            lat=(fix.lat if fix else None), lon=(fix.lon if fix else None),
            fix_age_s=(fix.fix_age_s if fix else None),
            platform_mode=self.mode.name, site_id=self.site_id,
            passport_uid=(self.passport.uid if self.passport else ""),
            manifold=getattr(self.manifold, "kind", "ideal"),
            notes=list(res.caveats))
        status["theta_abs_deg"] = theta_abs
        status["sigma_deg"] = float(sigma)
        self.last_status = status
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
