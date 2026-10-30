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

BURST INTEGRATION (2026-10-29, DF_P25_IMPROVEMENT_PLAN §6.1). With
`PlatformMode.integrate`, band work is not estimated per frame: each frame's
calibrated snapshots go into a per-transmitter accumulator
(`integrate.BurstIntegrator`) and ONE bearing comes out per transmission, over
all of it, when it ends — so a 3 s push-to-talk is one bearing with the whole
burst's snapshots, not sixty correlated ones. While a burst is open,
`last_status` / `last_result` carry an interim estimate on everything
integrated so far (the live needle) without adding records. `flush()` ends
every open burst (the end of a file, DF stopped). Per-frame behaviour is
unchanged when `integrate` is off.

PILOTS (§4.3). With a `pilots.PilotStage`, every band target is also streamed
through atkdsp's DMR / P25 sync correlator; sync hits give the steering vector
from the known waveform. The pilot decides WHICH signal is the target; the
whole-burst covariance, being more precise, gives the bearing whenever it
agrees (`_choose` has the rule and the measurement behind it). Two clean DMR
reads that agree also supply the path tag when the decoder gave none.

PATH TAG (§4.1). `identity.path_evidence` with the engine's `bandplan`: the
sync type, then P25's modulation / message direction, then the frequency (a
site IDEN table, the region's duplex rule), then the catalogue. The reason is
written on the bearing.

The engine holds only what it needs frame to frame: the current passport, the
platform mode, the open bursts and the pilot streams. Everything durable (the
registry, the posterior) is a port or a later module; the engine emits
records and lets them consume.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace

import numpy as np

from . import calibration
from . import doa as _doa
from . import identity as _identity
from . import integrate as _integrate
from .atkq import Flag, Frame
from .manifold import Geometry, IdealManifold, Manifold
from .ports import BurstEvent, Fix2D
from .records import BearingRecord, EmitterKey, Passport, wrap360

log = logging.getLogger("atkdf.engine")


@dataclass
class PlatformMode:
    """The switch that makes one estimator set behave as static or mobile."""

    name: str = "static"
    window_ms: float = 200.0           # survey window when there is no burst
    estimator: str = "music"
    min_confidence: float = 0.15
    survey: bool = True                # emit a bearing per window without a burst?
    heading_required: bool = False     # drop a bearing that has no heading?
    #: Burst integration (off by default: the per-frame contract is unchanged).
    integrate: bool = False
    #: A burst is over when nothing has been added for this long.
    burst_gap_s: float = 0.25
    #: ...or when it has run this long (a bearing at least this often).
    max_integration_s: float = 5.0
    #: ...or (moving) when the heading has turned this much since it began; 0 = no cap.
    max_heading_change_deg: float = 0.0

    @classmethod
    def static(cls) -> "PlatformMode":
        return cls(name="static", window_ms=1000.0, estimator="music",
                   survey=True, heading_required=False,
                   max_integration_s=5.0, max_heading_change_deg=0.0)

    @classmethod
    def mobile(cls) -> "PlatformMode":
        return cls(name="mobile", window_ms=50.0, estimator="music",
                   survey=False, heading_required=True,
                   max_integration_s=1.0, max_heading_change_deg=5.0)


def channel_band(header, event: BurstEvent) -> tuple:
    """The emitter's baseband band [lo, hi] relative to the tuner, from its RF
    centre and bandwidth. Clipped to the digitised band."""
    f0 = (event.fc_hz - header.fc_hz) if event.fc_hz else 0.0
    bw = event.bw_hz or 12_500.0
    half = header.fs_hz * 0.45
    return max(-half, f0 - bw / 2), min(half, f0 + bw / 2)


def _wrap180(a: float) -> float:
    return (float(a) + 180.0) % 360.0 - 180.0


#: how far apart the pilot and covariance bearings may be, in combined sigmas,
#: before the record says so
DISAGREE_SIGMAS = 3.0


def _sync_opens_squelch(hits) -> bool:
    """A SYNC squelch, the way a P25 / DMR receiver squelches: a sync word that
    read cleanly (or that the correlator confirmed) says the target is on the
    air even when the energy gate says the band is too weak — which is exactly
    where the pilot is worth having. Payload partial matches do not open it,
    and pure noise crosses the correlator's threshold about once in five
    minutes, never with a confirmed read (atkdsp `sync_default_threshold`)."""
    return any(getattr(h, "confirmed", False) or getattr(h, "clean", False)
               for h in (hits or ()))


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
    #: A `bandplan.BandPlan` for the path tag by frequency (P25 has no
    #: BS/MS sync word). None = sync type and catalogue only, as before.
    bandplan: object | None = None
    #: A `pilots.PilotStage` (sync-aided steering vectors). None = off. The
    #: pilot bearing becomes the record only on real sync evidence
    #: (`pilots.evidence_enough`); otherwise it rides along in `quality`.
    pilots: object | None = None
    #: What the last frame came to, for a live display: state is one of
    #: unusable | no_target | squelched | low_confidence | bearing | error,
    #: with the band SNR, per-channel band power and why. Replaced, never
    #: mutated, so a reader on another thread sees a whole dict. While a
    #: burst is integrating, state is "bearing" with `interim: True`.
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
        self._integ = _integrate.BurstIntegrator()
        self._fs = 2_400_000.0
        self._t_map = (0.0, 0, 2_400_000.0)        # (t_utc, stream_sample0, fs) of the last frame

    # -- the per-frame loop --------------------------------------------------
    def process(self, frame: Frame, events=()) -> list:
        self.n_frames += 1
        h = frame.header
        out = []
        self._fs = float(h.fs_hz)
        if h.has(Flag.SEGMENT_START):
            self._pending.clear()
            if self._integ.open:
                out += self._finish_bursts(self._integ.flush("retune"))
            if self.pilots is not None:
                self.pilots.reset()
        if not h.usable_for_df:
            self.last_status = {"t_utc": h.t_utc, "state": "unusable",
                                "why": "; ".join(h.why_not_usable()),
                                "passport": int(h.passport)}
            self.n_records += len(out)
            return out
        self._t_map = (float(h.t_utc), int(h.stream_sample0), float(h.fs_hz))
        self.passport = calibration.passport_of(
            h, geometry_id=self.geometry.id, manifold_id=self.manifold_id,
            site_id=self.site_id)
        self._integ.configure(self.mode.burst_gap_s, self.mode.max_integration_s,
                              self.mode.max_heading_change_deg)
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
            band = channel_band(h, e)
            seg = raw()[:, lo - s0:hi - s0]
            if self.mode.integrate:
                out += self._contribute(h, seg, band, e, lo, hi)
                continue
            hits = self._pilot_feed(h, seg, band, e, lo, final=True)
            rec = self._bearing(h, seg, frame, band=band, event=e, span=(n0, n1),
                                pilot_hits=hits)
            if rec is not None:
                out.append(rec)
        if not used_burst and self.mode.survey:
            rec = self._bearing(h, calibrated(), frame, band=None, event=None,
                                span=(s0, s1))
            if rec is not None:
                out.append(rec)
        if self.mode.integrate and self._integ.open:
            out += self._finish_bursts(self._integ.expire(s1, h.fs_hz))
        self.n_records += len(out)
        return out

    def flush(self, reason: str = "flush") -> list:
        """End every open burst now (end of a file, DF stopped) and return
        their records."""
        recs = self._finish_bursts(self._integ.flush(reason))
        self.n_records += len(recs)
        return recs

    @property
    def open_bursts(self) -> int:
        return len(self._integ.open)

    # -- shared pieces -------------------------------------------------------
    def _t_of(self, n: int) -> float:
        t0, s0, fs = self._t_map
        return t0 + (int(n) - s0) / fs

    def _band_snapshots(self, h, seg, band) -> tuple:
        """(Y element order, n_eff, stats) for raw receiver-order `seg`."""
        # an FFT fine enough to put ~16 bins in the channel, so MDL sees a
        # clean rank-1-plus-white-noise spectrum, but never longer than the
        # burst
        bw = max(band[1] - band[0], 1.0)
        nfft = int(2 ** round(math.log2(max(64, min(seg.shape[1] // 4,
                                                    16 * h.fs_hz / bw)))))
        st: dict = {}
        Y, _freqs, neff = _doa.band_snapshots(seg, h.fs_hz, band[0], band[1],
                                              nfft=nfft, header=h, stats=st)
        return self.geometry.element_order(Y), neff, st

    def _pilot_feed(self, h, seg, band, event, n_lo, final=False, skey=None) -> list:
        """Stream a target's samples through the sync correlator. `seg` is raw
        receiver-order; it is calibrated at the target's offset first."""
        if self.pilots is None or not getattr(self.pilots, "available", False):
            return []
        f0 = 0.5 * (band[0] + band[1])
        try:
            x_el = self.geometry.element_order(calibration.apply(seg, h, f0_hz=f0))
            key = skey if skey is not None else ("frame", int(n_lo))
            return self.pilots.feed(key, x_el, int(n_lo), float(h.fs_hz), f0,
                                    t_of=self._t_of,
                                    protocol=(event.protocol if event else ""), final=final)
        except Exception as exc:                            # noqa: BLE001
            log.warning("pilot stage failed: %s: %s", type(exc).__name__, exc)
            return []

    def _heading_at(self, t_utc: float) -> tuple:
        """(fix, heading_deg, heading_sigma_deg) at a UTC instant."""
        fix = self.positioning.at(t_utc) if self.positioning is not None else None
        if fix is not None and getattr(fix, "heading_deg", None) is not None:
            # the array's own heading (tripod, aligned, dual-antenna GNSS, or
            # mounting-corrected)
            hs = getattr(fix, "heading_sigma_deg", None)
            return fix, float(fix.heading_deg), (float(hs) if hs is not None else 2.0)
        if fix is not None and fix.course_deg is not None:
            return fix, float(fix.course_deg), self._heading_sigma(fix)
        return fix, None, None

    def _estimate(self, src, f_hz):
        return _doa.estimate(src, self.manifold, f_hz, method=self.mode.estimator,
                             geometry=self.geometry)

    def _choose(self, cov_res, hits, f_hz) -> tuple:
        """(primary DoaResult, extra quality, extra notes).

        MEASURED, 2026-10-29 (tests/test_pilots.py): with no interference the
        burst's covariance beats the pilot — it integrates the whole burst,
        the pilot only the 5 ms sync windows (at −9 dB per element over 1 s:
        covariance 0.7–1.9°, pilot up to 6.9°). The pilot's value is OWNERSHIP
        — which signal is the target — plus timing and the path tag. So:

          * pilot evidence too thin (`pilots.evidence_enough`): covariance,
            the pilot bearing alongside;
          * pilot and covariance AGREE (within 3 combined sigmas): the
            covariance bearing, `method: pilot-confirmed`;
          * they DISAGREE but one of the covariance's other solutions sits on
            the pilot bearing: that solution, `method: pilot-selected` — the
            covariance's precision, the stronger co-channel signal set aside;
          * they disagree and no covariance solution fits: the pilot bearing,
            `method: pilot`.
        """
        extra, notes = {}, []
        pres = None
        if hits:
            from .pilots import pilot_doa
            try:
                pres = pilot_doa(hits, self.manifold, f_hz, self.geometry)
            except Exception as exc:                        # noqa: BLE001
                log.warning("pilot bearing failed: %s: %s", type(exc).__name__, exc)
        if pres is None:
            if cov_res is not None:
                cov_res.quality.setdefault("method", "covariance")
            return cov_res, extra, notes
        from .pilots import evidence_enough
        pq = pres.quality
        for k in ("n_hits", "n_clean", "n_periodic", "n_confirmed", "protocol", "sync"):
            if k in pq:
                extra[f"pilot_{k}" if not k.startswith("pilot") else k] = pq[k]
        extra["pilot_theta_deg"] = round(float(pres.theta_deg), 2)
        if cov_res is None:
            return pres, extra, notes
        if not evidence_enough(pq):
            cov_res.quality.setdefault("method", "covariance")
            return cov_res, extra, notes
        d = _wrap180(cov_res.theta_deg - pres.theta_deg)
        lim = DISAGREE_SIGMAS * math.hypot(pres.sigma_deg, cov_res.sigma_deg)
        extra["pilot_vs_covariance_deg"] = round(float(d), 2)
        if abs(d) <= lim:
            cov_res.quality["method"] = "pilot-confirmed"
            return cov_res, extra, notes
        extra["covariance_theta_deg"] = round(float(cov_res.theta_deg), 2)
        extra["covariance_estimator"] = cov_res.estimator
        for th in list(cov_res.all_theta_deg or [])[1:]:
            if abs(_wrap180(th - pres.theta_deg)) <= lim:
                lik = _doa.build_likelihood([(th, 1.0)], cov_res.sigma_deg)
                q = dict(cov_res.quality)
                q["method"] = "pilot-selected"
                res = replace(cov_res, theta_deg=float(th), likelihood=lik, quality=q)
                notes.append(f"a stronger signal on this channel at "
                             f"{cov_res.theta_deg % 360:.1f}° (relative) was set aside: "
                             f"the sync word picked the covariance solution at "
                             f"{th % 360:.1f}°")
                return res, extra, notes
        notes.append(f"the covariance bearing ({cov_res.estimator}) is {abs(d):.1f}° from "
                     f"the sync-aided one — a co-channel emitter or multipath is pulling "
                     f"it; the sync-aided bearing is this emitter's")
        return pres, extra, notes

    # -- one bearing (per frame) ------------------------------------------------
    def _bearing(self, h, seg, frame, band, event, span, pilot_hits=None) -> BearingRecord | None:
        """`seg` is RAW receiver-order samples when `band` is given (calibrated
        per bin here), and calibrated element-order samples when it is not."""
        if seg.shape[1] < 64:
            return None
        f_hz = float(event.fc_hz) if (event and event.fc_hz) else float(h.fc_hz)
        status = {"t_utc": h.t_utc, "state": "bearing", "passport": int(h.passport),
                  "fc_hz": f_hz, "bw_hz": (band[1] - band[0]) if band else 0.0}
        extra_q = {}
        if band is not None:
            Y, neff, st = self._band_snapshots(h, seg, band)
            status["band_snr_db"] = round(st.get("snr_db", 0.0), 1)
            status["channel_db"] = st.get("channel_db", [])
            if self.gate_snr_db is not None and st.get("snr_db", 0.0) < self.gate_snr_db \
                    and not _sync_opens_squelch(pilot_hits):
                status["state"] = "squelched"
                status["why"] = (f"band {st.get('snr_db', 0.0):.1f} dB over the floor, "
                                 f"gate {self.gate_snr_db:.1f} dB")
                self.last_status = status
                return None
            src = _doa.covariance(Y, f_hz, geometry=self.geometry, snapshots=neff,
                                  bw_hz=band[1] - band[0])
            extra_q["band_snr_db"] = status["band_snr_db"]
            extra_q["channel_db"] = status["channel_db"]
        else:
            src = seg
        try:
            cov_res = self._estimate(src, f_hz)
        except Exception as exc:                            # noqa: BLE001
            cov_res = None
            if not pilot_hits:
                status["state"] = "error"
                status["why"] = f"{type(exc).__name__}: {exc}"
                self.last_status = status
                return None
        res, q2, notes = self._choose(cov_res, pilot_hits or [], f_hz)
        if res is None:
            status["state"] = "error"
            status["why"] = "no estimate"
            self.last_status = status
            return None
        extra_q.update(q2)
        return self._record(h.t_utc, h, res, f_hz, band, event, span, status,
                            extra_q, notes, pilot_hits or [])

    # -- burst integration -----------------------------------------------------
    def _contribute(self, h, seg, band, event, lo, hi) -> list:
        """Add one frame's slice of a burst to its accumulator; return the
        records of any burst that finished to make room."""
        f_hz = float(event.fc_hz) if event.fc_hz else float(h.fc_hz)
        bw = band[1] - band[0]
        status = {"t_utc": h.t_utc, "state": "bearing", "passport": int(h.passport),
                  "fc_hz": f_hz, "bw_hz": bw}
        Y, neff, st = self._band_snapshots(h, seg, band)
        status["band_snr_db"] = round(st.get("snr_db", 0.0), 1)
        status["channel_db"] = st.get("channel_db", [])
        bkey = _integrate.burst_key(event, h.segment, f_hz, bw)
        hits = self._pilot_feed(h, seg, band, event, lo, skey=bkey)
        out = []
        sync_open = _sync_opens_squelch(hits)
        if sync_open and self.gate_snr_db is not None \
                and st.get("snr_db", 0.0) < self.gate_snr_db:
            status["squelch"] = "opened by the sync word"
        if self.gate_snr_db is not None and st.get("snr_db", 0.0) < self.gate_snr_db \
                and not sync_open:
            b = self._integ.open.get(bkey)
            if b is not None and hits:
                b.pilot_hits.extend(hits)
            status["state"] = "squelched"
            status["why"] = (f"band {st.get('snr_db', 0.0):.1f} dB over the floor, "
                             f"gate {self.gate_snr_db:.1f} dB")
            self.last_status = status
            return out
        t_lo, t_hi = self._t_of(lo), self._t_of(hi)
        _fix, heading, _hs = self._heading_at(0.5 * (t_lo + t_hi))
        burst, finished = self._integ.contribute(
            bkey, event=event, f_hz=f_hz, bw_hz=bw, segment=h.segment, Y=Y, neff=neff,
            n_lo=lo, n_hi=hi, t_lo=t_lo, t_hi=t_hi, fs=float(h.fs_hz),
            frame_seq=int(h.seq), passport=(self.passport.uid if self.passport else ""),
            snr_db=st.get("snr_db"), channel_db=st.get("channel_db"), heading_deg=heading)
        if finished:
            out += self._finish_bursts(finished)
        burst.pilot_hits.extend(hits)
        # the interim needle: everything integrated so far, no record
        try:
            cov = burst.covariance(self.geometry)
            res = self._estimate(cov, f_hz)
            res, _q, _n = self._choose(res, burst.pilot_hits, f_hz)
            self.last_result = res
            status.update({"interim": True, "theta_rel_deg": float(res.theta_deg),
                           "confidence": float(res.quality.get("confidence", 0.0)),
                           "integrated_ms": round(1000.0 * burst.duration_s, 1),
                           "n_snapshots": int(burst.neff), "estimator": res.estimator,
                           "method": res.quality.get("method", "covariance")})
            if heading is not None:
                status["theta_abs_deg"] = wrap360(heading + res.theta_deg)
        except Exception as exc:                            # noqa: BLE001
            status["state"] = "error"
            status["why"] = f"{type(exc).__name__}: {exc}"
        self.last_status = status
        return out

    def _finish_bursts(self, bursts) -> list:
        out = []
        for b in bursts:
            try:
                rec = self._burst_record(b)
            except Exception as exc:                        # noqa: BLE001
                log.warning("burst %s could not be estimated: %s: %s", b.bkey[0],
                            type(exc).__name__, exc)
                rec = None
            if rec is not None:
                out.append(rec)
        return out

    def _burst_record(self, b) -> BearingRecord | None:
        if self.pilots is not None:
            b.pilot_hits.extend(self.pilots.finish(b.bkey, self._fs, self._t_of,
                                                   protocol=getattr(b.event, "protocol", "")))
        status = {"t_utc": b.t_mid, "state": "bearing", "fc_hz": b.f_hz, "bw_hz": b.bw_hz,
                  "integrated_ms": round(1000.0 * b.duration_s, 1),
                  "n_snapshots": int(b.neff), "burst_end": b.end_reason}
        try:
            cov_res = self._estimate(b.covariance(self.geometry), b.f_hz)
        except Exception as exc:                            # noqa: BLE001
            cov_res = None
            if not b.pilot_hits:
                status.update({"state": "error", "why": f"{type(exc).__name__}: {exc}"})
                self.last_status = status
                return None
        res, q2, notes = self._choose(cov_res, b.pilot_hits, b.f_hz)
        if res is None:
            return None
        q = b.quality()
        q.update(q2)
        band = (-0.5 * b.bw_hz, 0.5 * b.bw_hz)
        return self._record(b.t_mid, None, res, b.f_hz, band, b.event, (b.n0, b.n1),
                            status, q, notes, b.pilot_hits, segment=b.segment,
                            passport_uid=(b.passports[0] if b.passports else ""))

    # -- the record ------------------------------------------------------------
    def _record(self, t_utc, h, res, f_hz, band, event, span, status, extra_q, notes,
                pilot_hits, segment=None, passport_uid=None) -> BearingRecord | None:
        status = dict(status)
        status["theta_rel_deg"] = float(res.theta_deg)
        status["confidence"] = float(res.quality.get("confidence", 0.0))
        status["estimator"] = res.estimator
        status["method"] = res.quality.get("method", extra_q.get("method", "covariance"))
        status.pop("interim", None)
        self.last_result = res
        if res.quality.get("confidence", 0.0) < self.mode.min_confidence:
            status["state"] = "low_confidence"
            status["why"] = (f"confidence {res.quality.get('confidence', 0.0):.2f} "
                             f"< {self.mode.min_confidence:.2f}")
            self.last_status = status
            return None
        fix, heading, heading_sig = self._heading_at(t_utc)
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
        why = ""
        slot = sync = ""
        if event is not None:
            ev = event
            from .pilots import source_tag
            pst = source_tag(pilot_hits) if pilot_hits else ""
            if pst and not (event.sync_type or "").strip():
                ev = replace(event, sync_type=pst)
            if event.protocol and event.id:
                key = str(EmitterKey(event.protocol, event.system, event.id))
            path_tag, why = _identity.path_evidence(ev, self.catalogue, self.bandplan)
            slot, sync = event.slot, ev.sync_type

        quality = dict(res.quality)
        quality.update(extra_q)
        if why:
            quality["path_evidence"] = why
        rec = BearingRecord(
            t_utc=float(t_utc), theta_rel_deg=float(res.theta_deg), sigma_deg=float(sigma),
            estimator=res.estimator,
            segment=int(segment if segment is not None else (h.segment if h is not None else 0)),
            sample_span=tuple(int(v) for v in span),
            fc_hz=f_hz, bw_hz=(band[1] - band[0]) if band else 0.0,
            emitter_key=key, slot=slot, sync_type=sync or "", path_tag=path_tag,
            heading_deg=heading, heading_sigma_deg=heading_sig, theta_abs_deg=theta_abs,
            quality=quality, likelihood=res.likelihood,
            lat=(fix.lat if fix else None), lon=(fix.lon if fix else None),
            fix_age_s=(fix.fix_age_s if fix else None),
            platform_mode=self.mode.name, site_id=self.site_id,
            passport_uid=(passport_uid if passport_uid is not None
                          else (self.passport.uid if self.passport else "")),
            manifold=getattr(self.manifold, "kind", "ideal"),
            notes=list(res.caveats) + list(notes))
        status["theta_abs_deg"] = theta_abs
        status["sigma_deg"] = float(sigma)
        status["path_tag"] = path_tag
        self.last_status = status
        log.debug("bearing %s %.1f° rel (%s, σ %.1f°) %s", key or f"{f_hz / 1e6:.4f} MHz",
                  res.theta_deg, res.estimator, sigma, path_tag)
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
        """direct / repeated / unknown (`identity.path_evidence`, kept for
        callers of the old name)."""
        return _identity.path_tag_of(event, self.catalogue, self.bandplan)
