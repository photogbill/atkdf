"""Pilot-aided bearings: the steering vector from a KNOWN piece of the waveform.

THE IDEA (umbrella §6.5; DF_P25_IMPROVEMENT_PLAN §4.3; DEML — Li, Halder,
Stoica & Viberg, IEEE TSP 1995). Every DMR burst and every P25 frame starts
with a 24-symbol sync word the transmitter is obliged to send. Correlate each
element's samples against that word and the per-element complex amplitude IS
the array's response to that transmitter — its steering vector — measured with
~14 dB of processing gain, owned by THAT emitter alone: a co-channel signal, a
noise burst or an analogue carrier on the same channel does not correlate with
the sync and does not pull the bearing. The covariance methods cannot say that;
with a stronger interferer in the channel MUSIC returns the interferer.

With the waveform known, the maximum-likelihood bearing of one source is the
peak of |a(θ)ᴴ v|² / |a(θ)|² — Bartlett on the outer product of the measured
steering vector. Several sync hits in a burst (P25 repeats its frame sync every
180 ms LDU; DMR every 30 ms slot) are combined as Σ wₖ vₖ vₖᴴ, each normalised
(the transmitter's phase differs hit to hit and must not), weighted by its
post-correlation SNR. No source count, no noise subspace, no multipath model.

THE CORRELATOR is atkdsp's DmrSync (`src/dmr_sync.c`, its twin in
`atkdsp.reference`) — DMR's nine sync words (ETSI TS 102 361-1) and P25's
frame sync (TIA-102.BAAA), a CFO bank, sub-sample timing, and per-channel
`amps` whose ratios are the steering vector. ATK injects it; on its own atkdf
finds it through `_atkdsp`. Nothing is duplicated here: this module is the
CHANNELISER in front of it (the correlator wants one 12.5 kHz channel at
24–48 kHz) and the ESTIMATOR behind it.

THE CHANNELISER, per target, streaming across frames:
  * calibrated (each frame's own narrowband factor at the target's offset —
    exact for a 12.5 kHz channel) and in element order before anything else;
  * mixed to DC with the phase taken from the ABSOLUTE stream index, so frame
    boundaries do not jump the phase;
  * low-passed by FFT with a cosine taper, overlap-save style — `history`
    samples of the previous frame in front, `lookahead` held back until the
    next frame — so a sync word straddling a frame boundary is whole;
  * decimated by picking samples on a FIXED grid (absolute index ≡ 0 mod D),
    so the output does not depend on how the stream was chunked.
A gap in the stream (a different burst, a retune) starts the stream afresh.

WHICH HITS COUNT (found by the tests, 2026-10-29). Inside a strong 4FSK burst
the correlator also reports PARTIAL matches in the payload — random symbols
that half-resemble a sync word — and, with no protocol given, the DMR
correlator finds them in P25 and vice versa. For the bearing that is harmless:
a partial match is still THAT signal correlated against a slightly wrong
template, so its amplitudes are still its steering vector (weighted down by
their score). For anything that reads the LABEL it is not, so:
  * `select` keeps the protocol whose syncs read CLEANLY (≤ 1 symbol error)
    and drops the other family's hits;
  * `mark_periodic` confirms by timing: true syncs recur at the frame period
    (P25 180 ms LDUs, DMR 30 ms) and payload partials do not — this is what
    still recognises real syncs at low SNR, where noise costs them symbol
    errors and the correlator's own `confirmed` flag goes false;
  * the engine makes the pilot bearing the record only on that evidence (one
    clean read, or two periodic / confirmed hits), and the DMR path tag needs
    two clean reads that agree.

PATH TAG. A cleanly-read DMR sync says who sent it (BS / MS / direct); the
engine uses that when the decoder gave no sync type. P25's frame sync carries
no such word (inbound and outbound share it) — `bandplan` supplies P25's tag.

The outputs are `PilotHit`s and, from a set of them, a `doa.DoaResult` with
estimator ``pilot`` — the same record shape as every other bearing, so the
display can colour it by how it was produced.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from . import doa as _doa
from .manifold import GRID_STEP_DEG

log = logging.getLogger("atkdf.pilots")

#: the correlator's channel rate (5 samples/symbol at 4800 Bd); atkdsp
#: accepts 9.6–384 kHz and recommends 24 or 48 kHz
DEFAULT_RATE = 24_000.0
#: the taper of the channel low-pass, as fractions of the output rate
PASS_FRAC, STOP_FRAC = 0.30, 0.46
#: a hit below this correlation score is not used for a bearing
MIN_SCORE = 0.05
SETS = ("p25", "dmr")
#: a sync read with at most this many symbol errors is CLEAN (a true sync reads
#: 0 down to a few dB SNR; payload partial matches read several)
CLEAN_ERRORS = 1
#: frame periods a true sync recurs at, seconds, and the timing tolerance
PERIOD_S = {"p25": 0.180, "dmr": 0.030}
PERIOD_TOL_S = 1.5 / 4800.0


@dataclass
class PilotHit:
    """One sync word found, normalised from the correlator's dict."""

    n_center: int                 # absolute stream index of the sync's centre
    t_utc: float
    amps: np.ndarray              # element order, calibrated; amps/amps[0] = steering vector
    score: float
    protocol: str                 # p25 | dmr
    name: str = ""
    source: str = ""              # BS | MS | direct | "" (P25: none)
    kind: str = ""
    slot: int = 0
    errors: int = 24
    confirmed: bool = False
    cfo_hz: float = 0.0
    window: int = 0               # correlator window, samples at the channel rate
    periodic: bool = False        # recurs at the protocol's frame period (`mark_periodic`)

    @property
    def clean(self) -> bool:
        return int(self.errors) <= CLEAN_ERRORS

    @property
    def snr_post(self) -> float:
        """Post-correlation SNR of the amplitude estimate (linear): the score
        is the fraction of the window's energy the sync explains, so the
        in-band SNR is s/(1−s) and the correlation integrates `window`
        samples of it."""
        s = min(max(float(self.score), 1e-6), 0.999)
        return max(1, int(self.window)) * s / (1.0 - s)


class _Stream:
    """One target's channelised stream and its correlators."""

    def __init__(self, D: int, n_lo: int):
        self.D = int(D)
        self.tail = None            # (n_el, k) mixed, calibrated, full-rate samples
        self.tail0 = int(n_lo)      # absolute index of tail[:, 0]
        self.emitted_to = int(math.ceil(n_lo / D) * D)
        self.last_end = int(n_lo)
        self.corr: dict = {}        # set -> (DmrSync, dec0)


class PilotStage:
    """Streams each target's samples through atkdsp's sync correlator.

    `sync_factory(sample_rate, n_channels=…, sync_set=…)` builds a correlator
    (atkdsp.DmrSync or its twin); None asks `_atkdsp` to find one. When none is
    found the stage is `available == False` and says why; the engine then runs
    without pilots."""

    def __init__(self, sync_factory=None, sample_rate: float = DEFAULT_RATE,
                 sets=SETS, cfo_span_hz: float = 1500.0, history: int = 2048,
                 lookahead: int = 1024, min_score: float = MIN_SCORE, backend: str = ""):
        if sync_factory is None:
            from . import _atkdsp
            sync_factory, backend = _atkdsp.sync_factory()
        self.factory = sync_factory
        self.backend = backend or ("injected" if sync_factory is not None else "")
        self.rate = float(sample_rate)
        self.sets = tuple(s.lower() for s in sets)
        self.cfo_span_hz = float(cfo_span_hz)
        self.history = int(history)
        self.lookahead = int(lookahead)
        self.min_score = float(min_score)
        self._streams: dict = {}
        self.hits_total = 0
        self.why = "" if sync_factory is not None else (backend or "no sync correlator")
        if sync_factory is None:
            log.warning("pilots off: %s", self.why)
        else:
            log.info("pilots on: %s, channel rate %.0f Hz, sets %s", self.backend,
                     self.rate, ",".join(self.sets))

    @property
    def available(self) -> bool:
        return self.factory is not None

    def describe(self) -> str:
        return (f"pilots: {self.backend}, {self.rate / 1e3:.0f} kHz, {'/'.join(self.sets)}"
                if self.available else f"pilots off: {self.why}")

    def reset(self) -> None:
        """Forget every stream (a retune)."""
        self._streams.clear()

    def _decimation(self, fs: float) -> int:
        D = max(1, int(round(float(fs) / self.rate)))
        r = float(fs) / D
        if not 9_600.0 <= r <= 384_000.0:
            raise ValueError(f"cannot make a 9.6–384 kHz channel from {fs:.0f} Hz (D={D})")
        return D

    def _sets_for(self, protocol: str) -> tuple:
        p = (protocol or "").lower()
        if p.startswith("p25"):
            return ("p25",) if "p25" in self.sets else ()
        if p.startswith("dmr"):
            return ("dmr",) if "dmr" in self.sets else ()
        return self.sets

    # -- the channeliser ----------------------------------------------------
    def feed(self, skey, x_el, n_lo: int, fs: float, f0_hz: float, t_of=None,
             protocol: str = "", final: bool = False) -> list:
        """Feed calibrated, element-ordered samples `x_el` (n_el, L) of one
        target, starting at absolute stream index `n_lo`, the target at
        baseband offset `f0_hz`. Returns the hits whose peaks are now final.
        `t_of(n)` maps a stream index to UTC (else t_utc is 0)."""
        if not self.available:
            return []
        x_el = np.asarray(x_el)
        n_el, L = x_el.shape
        D = self._decimation(fs)
        st = self._streams.get(skey)
        if st is not None and (int(n_lo) != st.last_end or st.D != D):
            log.debug("pilot stream %s restarts (expected %d, got %d)", skey, st.last_end, n_lo)
            st = None
        if st is None:
            st = _Stream(D, int(n_lo))
            self._streams[skey] = st
        n_abs = int(n_lo) + np.arange(L, dtype=np.float64)
        ph = 2.0 * np.pi * ((float(f0_hz) / float(fs)) * n_abs % 1.0)
        y = x_el.astype(np.complex128) * np.exp(-1j * ph)[None, :]
        buf = y if st.tail is None else np.concatenate([st.tail, y], axis=1)
        buf0 = st.tail0 if st.tail is not None else int(n_lo)
        buf1 = buf0 + buf.shape[1]
        X = np.fft.fft(buf, axis=1)
        f = np.abs(np.fft.fftfreq(buf.shape[1], 1.0 / float(fs)))
        rate = float(fs) / D
        p, s = PASS_FRAC * rate, STOP_FRAC * rate
        H = np.where(f <= p, 1.0, np.where(f >= s, 0.0,
                                            0.5 * (1.0 + np.cos(np.pi * (f - p) / (s - p)))))
        lp = np.fft.ifft(X * H[None, :], axis=1)
        end = buf1 if final else buf1 - self.lookahead
        hits = []
        if end > st.emitted_to:
            grid = np.arange(st.emitted_to, end, D)
            dec = lp[:, grid - buf0].astype(np.complex64)
            dec_start = st.emitted_to // D
            for sset in self._sets_for(protocol):
                pair = st.corr.get(sset)
                if pair is None:
                    c = self.factory(rate, n_channels=n_el, sync_set=sset,
                                     cfo_span_hz=self.cfo_span_hz)
                    pair = (c, dec_start)
                    st.corr[sset] = pair
                c, dec0 = pair
                raw = list(c.process(dec))
                if final:
                    raw += list(c.flush())
                for hdict in raw:
                    hits.append(self._hit(hdict, sset, dec0, D, c, t_of))
            st.emitted_to = int(grid[-1]) + D
        keep0 = max(buf0, st.emitted_to - self.history)
        st.tail = buf[:, keep0 - buf0:]
        st.tail0 = keep0
        st.last_end = buf1
        if final:
            self._streams.pop(skey, None)
        self.hits_total += len(hits)
        if hits:
            log.debug("pilot %s: %d hit(s) %s", skey, len(hits),
                      ", ".join(f"{h.name} s={h.score:.2f} e={h.errors}" for h in hits))
        return hits

    def finish(self, skey, fs: float, t_of=None, protocol: str = "") -> list:
        """End a target's stream: flush the held-back samples and any pending
        sync (the end of a burst, a file or a DF run)."""
        st = self._streams.get(skey)
        if st is None or st.tail is None:
            self._streams.pop(skey, None)
            return []
        n_el = st.tail.shape[0]
        return self.feed(skey, np.zeros((n_el, 0), np.complex64), st.last_end, fs, 0.0,
                         t_of, protocol, final=True)

    @staticmethod
    def _hit(h: dict, sset: str, dec0: int, D: int, c, t_of) -> PilotHit:
        L = int(getattr(c, "L", 0) or round(24 * (c.sample_rate / 4800.0)))
        centre_dec = dec0 + int(h["start"]) + float(h.get("frac", 0.0)) + (L - 1) / 2.0
        n_center = int(round(centre_dec * D))
        return PilotHit(n_center=n_center, t_utc=float(t_of(n_center)) if t_of else 0.0,
                        amps=np.asarray(h["amps"], dtype=np.complex128).copy(),
                        score=float(h["score"]), protocol=sset, name=str(h.get("name", "")),
                        source=str(h.get("source", "")), kind=str(h.get("kind", "")),
                        slot=int(h.get("slot", 0) or 0), errors=int(h.get("errors", 24)),
                        confirmed=bool(h.get("confirmed", False)),
                        cfo_hz=float(h.get("cfo_hz", 0.0)), window=L)


# ---------------------------------------------------------------------------
# which hits count
# ---------------------------------------------------------------------------

def dominant_protocol(hits) -> str:
    """The protocol whose syncs read cleanly, by majority; "" when none did."""
    votes: dict = {}
    for h in hits:
        if h.clean:
            votes[h.protocol] = votes.get(h.protocol, 0) + 1
    if not votes:
        return ""
    ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return ""
    return ranked[0][0]


def mark_periodic(hits, fs_hz: float = 2_400_000.0) -> list:
    """Flag hits that have a partner at a whole number of the protocol's frame
    periods (within 1.5 symbols) — true syncs, even when noise has cost them
    symbol errors. Returns the hits (flags set in place)."""
    by: dict = {}
    for h in hits:
        by.setdefault(h.protocol, []).append(h)
    for proto, hs in by.items():
        period = PERIOD_S.get(proto)
        if not period or len(hs) < 2:
            continue
        t = np.array([h.n_center for h in hs], dtype=float) / float(fs_hz)
        for i, h in enumerate(hs):
            d = np.abs(t - t[i])
            k = np.round(d / period)
            ok = (k >= 1) & (np.abs(d - k * period) <= PERIOD_TOL_S)
            h.periodic = bool(np.any(ok))
    return hits


def select(hits, fs_hz: float = 2_400_000.0) -> list:
    """The hits that belong to the signal's own protocol, timing-marked."""
    hits = [h for h in hits if np.all(np.isfinite(h.amps))]
    proto = dominant_protocol(hits)
    if proto:
        hits = [h for h in hits if h.protocol == proto]
    return mark_periodic(hits, fs_hz)


def evidence_enough(quality: dict) -> bool:
    """Is a pilot bearing's sync evidence enough to make it the record? One
    clean read, or two hits confirmed by timing or by the correlator."""
    return (quality.get("n_clean", 0) >= 1 or quality.get("n_periodic", 0) >= 2
            or quality.get("n_confirmed", 0) >= 2)


# ---------------------------------------------------------------------------
# the estimator
# ---------------------------------------------------------------------------

def pilot_doa(hits, manifold, f_hz: float, geometry=None, step_deg: float = GRID_STEP_DEG,
              min_score: float = MIN_SCORE):
    """A bearing from sync hits — the known-waveform ML estimate — as a
    `doa.DoaResult` with estimator ``pilot``; None when no hit is usable."""
    use = [h for h in select(hits) if h.score >= min_score and np.linalg.norm(h.amps) > 0]
    if not use:
        return None
    geometry = geometry if geometry is not None else getattr(manifold, "geometry", None)
    n = use[0].amps.size
    Rp = np.zeros((n, n), dtype=np.complex128)
    snr_sum = 0.0
    for h in use:
        v = h.amps / np.linalg.norm(h.amps)
        w = h.snr_post
        Rp += w * np.outer(v, v.conj())
        snr_sum += w
    Rp /= max(snr_sum, 1e-30)
    grid, A = _doa.steering_grid(manifold, f_hz, step_deg)
    spec = _doa.bartlett_spectrum(Rp, A)
    peaks = _doa.find_peaks(spec, grid, within_db=_doa.AMBIGUITY_DB)
    th0 = peaks[0][0]
    kind = getattr(manifold, "kind", "ideal")
    crb = (_doa.cramer_rao_deg(snr_sum, 1, geometry, f_hz, th0)
           if geometry is not None else 2.0)
    sig = math.hypot(crb, _doa.MODEL_SIGMA_DEG.get(kind, 4.0))
    lik = _doa.build_likelihood([(t, v) for t, v, _ in peaks], sig)
    peak_db = 10 * math.log10(spec.max() / max(float(np.median(spec)), 1e-300))
    confirmed = [h for h in use if h.confirmed]
    n_clean = sum(1 for h in use if h.clean)
    n_periodic = sum(1 for h in use if h.periodic)
    names = sorted({h.name for h in use if h.name})
    good = max(len(confirmed), n_periodic, n_clean)
    conf = min(1.0, (good + 0.25 * (len(use) - good)) / 3.0)
    caveats = list(manifold.caveats(f_hz))
    if not (n_clean or n_periodic or confirmed):
        caveats.append("pilot-aided: no sync word read cleanly or recurred on time "
                       "(all hits partial) — still this emitter's signal, at less SNR")
    q = {"method": "pilot", "confidence": float(conf),
         "n_hits": len(use), "n_confirmed": len(confirmed),
         "n_clean": int(n_clean), "n_periodic": int(n_periodic),
         "protocol": use[0].protocol if len({h.protocol for h in use}) == 1 else "mixed",
         "sync": names, "mean_score": round(float(np.mean([h.score for h in use])), 3),
         "cfo_hz": round(float(np.median([h.cfo_hz for h in use])), 1),
         "snr_post_db": round(10 * math.log10(max(snr_sum, 1e-12)), 1),
         "peak_ratio_db": round(peak_db, 1), "n_snapshots": len(use)}
    return _doa.DoaResult(theta_deg=float(th0), sigma_deg=float(sig), estimator="pilot",
                          manifold=kind, n_sources=1, all_theta_deg=[th0],
                          grid_deg=grid, spectrum=spec, likelihood=lik, quality=q,
                          caveats=caveats)


def source_tag(hits) -> str:
    """The sync type (bs / ms / direct) that at least two CLEAN DMR reads agree
    on, unanimously — or "". What the engine uses for the path tag when the
    decoder gave none. Strict on purpose: a wrong "direct" would fuse a
    repeater as a handset."""
    votes: dict = {}
    for h in select(hits):
        if h.protocol == "dmr" and h.clean and h.source:
            k = h.source.lower()
            votes[k] = votes.get(k, 0) + 1
    if len(votes) != 1:
        return ""
    (k, n), = votes.items()
    return k if n >= 2 else ""
