"""The scenario simulator: a FrameSource with a planted truth, plus fakes of
every port. This is how the engine is built before the KrakenSDR arrives and
tested in every session after, without the hardware.

WHAT IT PLANTS, and what the chain must give back:

  * emitters at known bearings from the receiver (static site or a track),
    each with a burst schedule, so `doa` must return the bearing and
    `identity` the ID;
  * a receiver perturbation -- per-channel gain, phase and fractional delay,
    exactly the thing atkdaq measures on its noise source -- and the header it
    writes to INVERT it, so `calibration.apply` must leave the steering vector
    alone;
  * optional coherent multipath (a reflector), so ML and FBSS have something
    only they can resolve;
  * optional manifold error (a perturbed array), so the measured-manifold
    path has something to learn back;
  * the frame flags a real run has: NOISE_ON while calibrating, SYNTHETIC on
    every frame, SEGMENT_START at a retune;
  * (2026-10-29) a real 4FSK waveform with real sync words when an emitter
    asks for one (`Emitter.waveform`): P25 Phase 1 C4FM with the frame sync
    every 180 ms LDU, or DMR with BS- or MS-sourced voice syncs, phase-
    continuous across frames — so atkdsp's sync correlator, and the pilot-
    aided bearing built on it, have something true to find.

THE MODEL. For a narrowband emitter at relative bearing theta, element m sees
a_m(theta) (the manifold), times the emitter's complex symbols. Receiver m
then multiplies by g_m e^{j phi_m} and delays by tau_m samples (a phase slope
across the band). The samples handed to a frame are that, quantised to int8
the way atkdaq quantises. The header carries w_m = 1/(g_m e^{j phi_m}) and
frac_m = tau_m, which is what a perfect atkdaq calibration would report, so
the engine's calibration is exact and any residual is the manifold's, not the
receiver's -- which is the point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import atkq
from .atkq import CalState, Device, Flag, Frame, Header
from .manifold import IdealManifold, Manifold, wavelength
from .ports import BurstEvent, Fix2D


@dataclass
class Emitter:
    """A transmitter. `bearing_deg` is its RELATIVE bearing from the array
    (sim v0: a fixed bearing; a moving emitter is a function of time later).
    `f0_hz` is its baseband offset from the tuner. `on`/`period_s`/`duty`
    make the burst schedule; `protocol`/`system`/`id`/`slot` give it identity
    on the side channel."""

    bearing_deg: float
    f0_hz: float = 0.0
    snr_db: float = 20.0
    bw_hz: float = 12_500.0
    protocol: str = ""
    system: str = ""
    id: str = ""
    slot: int | None = None
    sync_type: str = ""
    period_s: float = 0.0            # 0 = always on
    duty: float = 1.0
    phase0: float = 0.0              # fraction of a period to offset the schedule
    #: "" = the band-limited random-symbol stream (as before); "p25" = C4FM at
    #: 4800 Bd with the TIA-102 frame sync every 864 symbols (one LDU, 180 ms);
    #: "dmr_ms" = DMR handset bursts (132 symbols, MS voice sync in the middle,
    #: one slot: every 60 ms); "dmr_bs" = DMR repeater outbound (a burst with
    #: the BS voice sync every 30 ms). See `fsk4_waveform`.
    waveform: str = ""

    def active(self, t0: float, t1: float) -> bool:
        if self.period_s <= 0:
            return True
        # on for the first `duty` of each period, offset by phase0 -- so two
        # TDMA slots (duty 0.5, phase0 0 and 0.5) never transmit at once
        a = ((t0 / self.period_s) - self.phase0) % 1.0
        return a < self.duty


@dataclass
class Reflector:
    """A coherent multipath image of an emitter: same signal, another bearing,
    lower and delayed. `of` is the emitter's index."""

    of: int
    bearing_deg: float
    gain_db: float = -6.0
    delay_samples: int = 3


@dataclass
class Scenario:
    """Everything a run needs. Receiver gains/phases/fracs default to a
    plausible spread; give them to plant exact values."""

    emitters: list
    fc_hz: float = 446_000_000.0
    fs_hz: int = 2_400_000
    n_ch: int = 5
    frame_ms: float = 50.0
    manifold: Manifold | None = None            # None -> ideal UCA-5 r=0.18
    reflectors: list = field(default_factory=list)
    rx_gain_db: list | None = None
    rx_phase_deg: list | None = None
    rx_frac: list | None = None
    floor_db: float = -40.0          # kept for compatibility; SNR is emitter-vs-noise
    adc_rms: float = 0.10            # target RMS as a fraction of int8 full scale
    seed: int = 0
    site_lat: float = 38.70
    site_lon: float = -77.58
    heading_deg: float = 0.0


# ---------------------------------------------------------------------------
# 4FSK waveforms with real sync words (P25 Phase 1 C4FM, DMR)
# ---------------------------------------------------------------------------

#: TIA-102.BAAA frame sync and ETSI TS 102 361-1 Table 9.2 voice syncs (the
#: same 48-bit words atkdsp's dmr_sync correlates against).
P25_FRAME_SYNC = 0x5575F5FF77FF
DMR_BS_VOICE = 0x755FD7DF75F7
DMR_MS_VOICE = 0x7F7D5DD57DFD
#: dibit -> symbol (both standards): 01 +3, 00 +1, 10 -1, 11 -3
_DIBIT = {1: 3.0, 0: 1.0, 2: -1.0, 3: -3.0}
SYMBOL_RATE = 4800.0


def sync_symbols(word48: int) -> np.ndarray:
    """A 48-bit sync word as its 24 symbols, most significant dibit first."""
    return np.array([_DIBIT[(int(word48) >> (46 - 2 * i)) & 3] for i in range(24)])


def _raised_cosine(sps: int, alpha: float = 0.2, span: int = 4) -> np.ndarray:
    """A raised-cosine pulse, peak 1, over ±span symbols (Nyquist: a constant
    symbol run sits exactly on its deviation)."""
    t = np.arange(-span * sps, span * sps + 1) / float(sps)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.sinc(t) * np.cos(np.pi * alpha * t) / (1.0 - (2.0 * alpha * t) ** 2)
    sing = np.isclose(np.abs(2.0 * alpha * t), 1.0)
    h[sing] = (np.pi / 4.0) * np.sinc(1.0 / (2.0 * alpha))
    return h


def fsk4_waveform(kind: str, n_out: int, fs_out: float, seed: int = 0,
                  phase0: float = 0.0) -> np.ndarray:
    """`n_out` samples at `fs_out` of a continuous 4FSK transmission with real
    sync words, unit power where keyed. kind: p25 | dmr_ms | dmr_bs.

    Frequency pulses are raised cosine (alpha 0.2) — P25's C4FM Nyquist pulse
    without its inverse-sinc shaping, DMR's RRC pair approximated by its
    product; the correlator's reference is RRC, so the mismatch costs a little
    SNR and no bearing. Generated at 48 kHz and resampled to `fs_out`."""
    from scipy.signal import resample_poly
    rng = np.random.default_rng(seed)
    fs_gen = 48_000.0
    sps = int(round(fs_gen / SYMBOL_RATE))
    n_sym = int(math.ceil(n_out / fs_out * SYMBOL_RATE)) + 900
    sym = rng.choice(np.array([3.0, 1.0, -1.0, -3.0]), size=n_sym)
    keyed = np.ones(n_sym, dtype=bool)
    off = int(round(phase0 * 864))
    if kind == "p25":
        dev, fs_word, period = 1800.0, sync_symbols(P25_FRAME_SYNC), 864
        for s0 in range(off % period, n_sym - 24, period):
            sym[s0:s0 + 24] = fs_word
    elif kind in ("dmr_ms", "dmr_bs"):
        dev = 1944.0
        word = sync_symbols(DMR_MS_VOICE if kind == "dmr_ms" else DMR_BS_VOICE)
        period = 288 if kind == "dmr_ms" else 144          # 60 ms (one slot) / 30 ms
        burst = 132
        keyed[:] = kind == "dmr_bs"
        for b0 in range(off % period, n_sym - burst, period):
            sym[b0 + 54:b0 + 78] = word
            keyed[b0:b0 + burst] = True
    else:
        raise ValueError(f"unknown waveform {kind!r} (p25, dmr_ms, dmr_bs)")
    up = np.zeros(n_sym * sps)
    up[::sps] = sym / 3.0 * dev
    f = np.convolve(up, _raised_cosine(sps), mode="same")
    ph = 2.0 * np.pi * np.cumsum(f) / fs_gen
    gate = np.repeat(keyed.astype(float), sps)
    iq = np.exp(1j * ph) * gate
    ratio = fs_out / fs_gen
    if abs(ratio - round(ratio)) < 1e-9:
        out = resample_poly(iq, int(round(ratio)), 1)
    else:
        from fractions import Fraction
        fr = Fraction(ratio).limit_denominator(1000)
        out = resample_poly(iq, fr.numerator, fr.denominator)
    return out[:n_out].astype(np.complex64)


def _payload(h: Header, x) -> bytes:
    """The payload bytes of the frame `x` encodes to.

    FIXED 2026-10-06: this used to be `encode_frame(h, x)[h.hdr_bytes:]`, but
    a Header built here has hdr_bytes 0 (it is only filled in when a header
    is PARSED), so every simulated payload began with the 256 header bytes:
    each channel was shifted 128 samples into the one before it and channel
    0 opened with the header read as samples. Every channel moved by the
    same amount, so bearings survived — which is why nothing failed."""
    full = atkq.encode_frame(h, x)
    return full[atkq.FIXED_BYTES + h.n_channels * atkq.CH_STRIDE:]


class Simulator:
    """A `Scenario` as a FrameSource (`frames()`, `control()`) plus a matching
    DecoderEvents (`events()`) and Positioning (`positioning`)."""

    def __init__(self, scenario: Scenario, frames: int = 40, calibrate_first: bool = True):
        self.sc = scenario
        self.n_frames = int(frames)
        self.calibrate_first = calibrate_first
        self.commands: list = []
        self.rng = np.random.default_rng(scenario.seed)
        n = scenario.n_ch
        from .manifold import uca
        self.geometry = getattr(scenario.manifold, "geometry", None) or uca(n, 0.18)
        self.manifold = scenario.manifold or IdealManifold(self.geometry)
        # receiver perturbation, and the header numbers that invert it
        self.g = (np.asarray(scenario.rx_gain_db, float) if scenario.rx_gain_db is not None
                  else self.rng.normal(0, 0.6, n))
        self.ph = (np.asarray(scenario.rx_phase_deg, float) if scenario.rx_phase_deg is not None
                   else self.rng.uniform(-180, 180, n))
        self.tau = (np.asarray(scenario.rx_frac, float) if scenario.rx_frac is not None
                    else self.rng.uniform(-0.5, 0.5, n))
        self.g[0], self.ph[0], self.tau[0] = 0.0, 0.0, 0.0     # channel 0 is the reference
        self.rx = 10 ** (self.g / 20) * np.exp(1j * np.radians(self.ph))
        self.samples_per = int(round(scenario.fs_hz * scenario.frame_ms / 1000.0))
        self.t0 = 1_790_000_000.0
        self._events: list = []
        self._waves: dict = {}          # emitter index -> continuous 4FSK waveform
        self._build_events()

    # -- ports ---------------------------------------------------------------
    @property
    def positioning(self):
        from .fakes import StaticPositioning
        return StaticPositioning(self.sc.site_lat, self.sc.site_lon,
                                 self.sc.heading_deg, sigma_m=5.0)

    def events(self):
        yield from self._events

    def control(self, cmd: str) -> None:
        self.commands.append(str(cmd))

    # -- the truth, for tests ------------------------------------------------
    def truth(self) -> dict:
        return {"weights": (1.0 / self.rx).tolist(), "frac": self.tau.tolist(),
                "bearings": [e.bearing_deg for e in self.sc.emitters],
                "geometry": self.geometry.id}

    def _build_events(self):
        dt = self.samples_per / self.sc.fs_hz
        for k in range(self.n_frames):
            rel = k * dt                       # relative time; never t0 +/- rel (cancellation)
            t = self.t0 + rel
            for e in self.sc.emitters:
                if e.protocol and e.active(rel, rel + dt):
                    n0 = k * self.samples_per
                    self._events.append(BurstEvent(
                        protocol=e.protocol, system=e.system, id=e.id, t_utc=t,
                        span_s=dt * e.duty if e.period_s else dt,
                        n0=n0, n1=n0 + self.samples_per, segment=0, slot=e.slot,
                        sync_type=e.sync_type, fc_hz=self.sc.fc_hz + e.f0_hz, bw_hz=e.bw_hz))

    # -- signal --------------------------------------------------------------
    def _emitter_samples(self, e: Emitter, n: int, k: int | None = None) -> np.ndarray:
        """One emitter's contribution at the elements: (n_ch, n) complex.
        `k` (the frame index) places a continuous `waveform` emitter's samples."""
        fs = self.sc.fs_hz
        if e.waveform and k is not None:
            return self._waveform_samples(e, n, k)
        t = np.arange(n) / fs
        # a band-limited baseband symbol stream at offset f0
        sps = max(2, int(round(fs / e.bw_hz)))
        nsym = n // sps + 2
        syms = (self.rng.integers(0, 4, nsym) * 2 - 3).astype(float)
        syms = syms + 1j * (self.rng.integers(0, 4, nsym) * 2 - 3)
        up = np.zeros(nsym * sps, dtype=complex)
        up[::sps] = syms
        # simple RC-ish smoothing
        k = np.hanning(sps * 2 + 1)
        base = np.convolve(up, k / k.sum(), mode="same")[:n]
        base = base / np.sqrt(np.mean(np.abs(base) ** 2) + 1e-30)
        s = base * np.exp(2j * np.pi * e.f0_hz * t) * math.sqrt(10 ** (e.snr_db / 10))
        # a signal at baseband offset f0 is physically at RF fc+f0, so its
        # steering vector uses that frequency, not the tuner centre
        a = self.manifold.response(e.bearing_deg, self.sc.fc_hz + e.f0_hz)[:, 0]
        return np.outer(a, s)

    def _waveform_samples(self, e: Emitter, n: int, k: int) -> np.ndarray:
        idx = self.sc.emitters.index(e)
        w = self._waves.get(idx)
        if w is None:
            total = self.n_frames * self.samples_per
            w = fsk4_waveform(e.waveform, total, float(self.sc.fs_hz),
                              seed=self.sc.seed * 101 + idx, phase0=e.phase0)
            self._waves[idx] = w
        s = w[k * self.samples_per:k * self.samples_per + n].astype(np.complex128)
        t = (k * self.samples_per + np.arange(n)) / float(self.sc.fs_hz)
        s = s * np.exp(2j * np.pi * e.f0_hz * t) * math.sqrt(10 ** (e.snr_db / 10))
        a = self.manifold.response(e.bearing_deg, self.sc.fc_hz + e.f0_hz)[:, 0]
        return np.outer(a, s)

    def _apply_reflectors(self, base_of, x: np.ndarray, e_idx: int) -> np.ndarray:
        for r in self.sc.reflectors:
            if r.of != e_idx:
                continue
            src = base_of[e_idx]
            emit = self.sc.emitters[e_idx]
            f_rf = self.sc.fc_hz + emit.f0_hz
            a = self.manifold.response(r.bearing_deg, f_rf)[:, 0]
            g = 10 ** (r.gain_db / 20)
            d = int(r.delay_samples)
            # rebuild at the reflector's bearing from the emitter's symbol stream
            sym = src[0] / (self.manifold.response(emit.bearing_deg, f_rf)[0, 0])
            shifted = np.zeros_like(sym)
            if d < sym.size:
                shifted[d:] = sym[:sym.size - d]
            x = x + g * np.outer(a, shifted)
        return x

    def _receiver(self, x: np.ndarray) -> np.ndarray:
        """Apply per-channel gain/phase (flat) and fractional delay (a phase
        slope across the band), the impairments atkdaq calibrates out."""
        n = x.shape[1]
        f = np.fft.fftfreq(n, 1.0 / self.sc.fs_hz)
        X = np.fft.fft(x, axis=1)
        # tau is in SAMPLES, so the receiver's phase slope is -2pi f tau / fs
        slope = np.exp(-2j * np.pi * np.outer(self.tau, f) / self.sc.fs_hz)
        X = X * self.rx[:, None] * slope
        return np.fft.ifft(X, axis=1)

    def _noise(self, shape) -> np.ndarray:
        """Unit-power complex white noise: SNR is defined emitter-power vs this,
        and the whole frame is scaled into the ADC range afterwards."""
        return (self.rng.standard_normal(shape)
                + 1j * self.rng.standard_normal(shape)) / math.sqrt(2)

    def _to_adc(self, x: np.ndarray) -> np.ndarray:
        """Scale the whole frame so its RMS sits at `adc_rms` of full scale, the
        way an AGC-off receiver's gain is set: comfortably inside the 8-bit
        range with headroom. A common real scale changes neither the weights
        (per-element ratios) nor a bearing, so it is invisible to the engine —
        it only keeps a strong emitter from clipping the int8 payload, which is
        a nonlinearity that would destroy the array's rank-1 structure."""
        rms = float(np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-30)
        return x * (self.sc.adc_rms / rms)

    def _header(self, k: int, cal_state: int, flags: int) -> Header:
        n = self.sc.n_ch
        dt = self.samples_per / self.sc.fs_hz
        chans = []
        w = 1.0 / self.rx
        for i in range(n):
            chans.append(atkq.Channel(delay=0, frac_delay=float(self.tau[i]),
                                     weight=complex(w[i]), sample_count=k * self.samples_per))
        return Header(n_channels=n, fs_hz=self.sc.fs_hz, samples_per_ch=self.samples_per,
                      channels=tuple(chans), fc_hz=int(self.sc.fc_hz), seq=k,
                      t_utc_ns=int((self.t0 + k * dt) * 1e9), flags=flags | int(Flag.SYNTHETIC),
                      cal_state=cal_state, device_kind=int(Device.SYNTH),
                      passport=1 if cal_state == CalState.CALIBRATED else 0,
                      cal_age_ms=0, spread_cdeg=16 if cal_state == CalState.CALIBRATED else 0xFFFF,
                      gain_tenths_db=280, stream_sample0=k * self.samples_per,
                      seg_sample0=k * self.samples_per)

    def frames(self):
        n = self.samples_per
        cal_frames = 3 if self.calibrate_first else 0
        for k in range(self.n_frames):
            if k < cal_frames:
                # noise-source frames: not usable, the receivers seeing the ONE
                # common noise source (strong, as the switch delivers it)
                src = np.tile(self._noise((1, n)), (self.sc.n_ch, 1)) * 30.0
                x = self._to_adc(self._receiver(src) + self._noise((self.sc.n_ch, n)))
                h = self._header(k, CalState.CALIBRATING, int(Flag.NOISE_ON) |
                                 (int(Flag.SEGMENT_START) if k == 0 else 0))
                yield Frame(h, _payload(h, x))
                continue
            base_of = {}
            x = np.zeros((self.sc.n_ch, n), dtype=complex)
            for i, e in enumerate(self.sc.emitters):
                if not e.active((k * n) / self.sc.fs_hz, ((k + 1) * n) / self.sc.fs_hz):
                    continue
                xe = self._emitter_samples(e, n, k)
                base_of[i] = xe
                x = x + xe
            for i in base_of:
                x = self._apply_reflectors(base_of, x, i)
            x = self._to_adc(self._receiver(x) + self._noise((self.sc.n_ch, n)))
            h = self._header(k, CalState.CALIBRATED,
                             int(Flag.SEGMENT_START) if k == 0 else 0)
            payload = _payload(h, x)
            yield Frame(h, payload)

    def write(self, path) -> str:
        """Record the run to a .atkq file, exactly as atkdaq would."""
        from pathlib import Path
        with Path(path).open("wb") as f:
            for fr in self.frames():
                f.write(atkq.pack_header(fr.header))
                f.write(bytes(fr.payload))
        return str(path)


# ---------------------------------------------------------------------------
# element-level snapshots for any array (2026-10-30, plan §11.4.8)
# ---------------------------------------------------------------------------

@dataclass
class PlaneWave:
    """A narrowband plane wave at the array: relative bearing (clockwise from
    the array's forward axis), elevation (up), per-element SNR in dB against
    unit noise. `group` > 0 makes waves with the same group share one symbol
    stream (coherent multipath), each with its own `gain_db` and `phase_deg`."""

    az_deg: float
    el_deg: float = 0.0
    snr_db: float = 20.0
    group: int = 0
    gain_db: float = 0.0
    phase_deg: float = 0.0


def element_snapshots(manifold: Manifold, waves, f_hz: float, n: int = 1024,
                      seed: int = 0, noise: bool = True) -> np.ndarray:
    """(n_elements, n) complex baseband at the ELEMENTS: Σ a(θ, el)·s(t) + unit
    noise per element. Float, no ADC — for arrays the Kraken's 8-bit frame
    format cannot carry (16+ elements, analog combiners, elevation). Each
    independent wave is a unit-power complex Gaussian stream scaled to its
    SNR; waves in one `group` > 0 share a stream."""
    rng = np.random.default_rng(seed)
    geom = getattr(manifold, "geometry", None)
    n_el = geom.n if geom is not None else manifold.response(0.0, f_hz).shape[0]
    x = np.zeros((n_el, n), dtype=complex)
    streams: dict = {}
    for w in waves:
        if w.group > 0 and w.group in streams:
            s = streams[w.group]
        else:
            s = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / math.sqrt(2)
            if w.group > 0:
                streams[w.group] = s
        amp = math.sqrt(10 ** (w.snr_db / 10)) * 10 ** (w.gain_db / 20) \
            * np.exp(1j * math.radians(w.phase_deg))
        if abs(w.el_deg) > 1e-12:
            a = manifold.response2(np.array([w.az_deg]), np.array([w.el_deg]), f_hz)[:, 0]
        else:
            a = manifold.response(np.array([w.az_deg]), f_hz)[:, 0]
        x += np.outer(a, amp * s)
    if noise:
        x += (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)) / math.sqrt(2)
    return x


class SteerableArraySim:
    """A simulated analog / hybrid array behind a `ports.BeamSteerer`
    (plan §11.4.8): the elements see `waves` through `manifold`; `apply(W)`
    quantises the weights to the phase-shifter / attenuator steps and returns
    that (what the controller believes it set); the ELEMENTS actually get the
    quantised weights times a fixed per-element error (`shifter_rms_deg`,
    `shifter_gain_rms_db` — "no channel-to-channel match spec", lit2 §1.7),
    and each OUTPUT has its own fixed gain/phase mismatch. `capture(n)` is
    fresh element snapshots, combined. `elapsed_s` adds settle and capture
    time so a sweep can be checked against a burst."""

    def __init__(self, manifold: Manifold, waves, f_hz: float, n_outputs: int = 1,
                 halves=None, fs_hz: float = 48_000.0, phase_step_deg: float | None = None,
                 gain_step_db: float | None = None, shifter_rms_deg: float = 0.0,
                 shifter_gain_rms_db: float = 0.0, output_mismatch_db: float = 0.0,
                 output_mismatch_deg: float = 0.0, settle_s: float = 1e-6, seed: int = 0,
                 on_until_s: float | None = None):
        from .architectures import quantise_weights as _q
        self._q = _q
        self.manifold = manifold
        self.waves = list(waves)
        self.f_hz = float(f_hz)
        self.n_outputs = int(n_outputs)
        self.halves = halves
        self.fs_hz = float(fs_hz)
        self.settle_s = float(settle_s)
        self.phase_step_deg = phase_step_deg
        self.gain_step_db = gain_step_db
        self.rng = np.random.default_rng(seed)
        n = manifold.geometry.n
        self.element_error = (10 ** (self.rng.normal(0, shifter_gain_rms_db, n) / 20)
                              * np.exp(1j * np.radians(self.rng.normal(0, shifter_rms_deg, n))))
        self.output_error = (10 ** (self.rng.normal(0, output_mismatch_db, n_outputs) / 20)
                             * np.exp(1j * np.radians(self.rng.normal(0, output_mismatch_deg,
                                                                      n_outputs))))
        self.on_until_s = on_until_s
        self.elapsed_s = 0.0
        self.applied = None
        self._true = None
        self.n_applies = 0
        self._seed = int(seed) * 7919

    def apply(self, weights) -> np.ndarray:
        W = np.asarray(weights, dtype=complex)
        if W.ndim == 1:
            W = W[:, None]
        Wq = self._q(W, self.phase_step_deg, self.gain_step_db)
        self.applied = Wq
        self._true = Wq * self.element_error[:, None]
        self.elapsed_s += self.settle_s
        self.n_applies += 1
        return Wq

    def capture(self, n: int) -> np.ndarray:
        if self._true is None:
            raise RuntimeError("apply() weights before capture()")
        self._seed += 1
        on = self.on_until_s is None or self.elapsed_s < self.on_until_s
        x = element_snapshots(self.manifold, self.waves if on else [], self.f_hz, int(n),
                              seed=self._seed)
        y = self._true.conj().T @ x
        y = y * self.output_error[: y.shape[0], None]
        self.elapsed_s += int(n) / self.fs_hz
        return y


class SwitchedArraySim:
    """A simulated switched array: `rf_chains` coherent channels, channel
    `reference_channel` always on `reference_element`, the others switched
    through `schedule` (one list per slot of the element on each
    non-reference channel). Each channel has a fixed gain/phase (`rx_*`),
    the whole receiver a random LO phase per retune (common to the channels
    — a shared LO), and each slot is a fresh piece of the emitter's signal.
    `off_after_slot` ends the transmission part-way through the sequence."""

    def __init__(self, manifold: Manifold, waves, f_hz: float, reference_element: int,
                 schedule, rf_chains: int = 2, reference_channel: int = 0,
                 dwell_samples: int = 2048, rx_gain_db=None, rx_phase_deg=None,
                 off_after_slot: int | None = None, seed: int = 0):
        self.manifold = manifold
        self.waves = list(waves)
        self.f_hz = float(f_hz)
        self.ref_el = int(reference_element)
        self.ref_ch = int(reference_channel)
        self.schedule = [list(s) for s in schedule]
        self.ch = int(rf_chains)
        self.dwell = int(dwell_samples)
        self.rng = np.random.default_rng(seed)
        g = (np.asarray(rx_gain_db, float) if rx_gain_db is not None
             else self.rng.normal(0, 1.0, self.ch))
        p = (np.asarray(rx_phase_deg, float) if rx_phase_deg is not None
             else self.rng.uniform(-180, 180, self.ch))
        self.rx = 10 ** (g / 20) * np.exp(1j * np.radians(p))
        self.off_after = off_after_slot
        self._seed = int(seed) * 104729

    def retune(self) -> None:
        """A new random LO phase, common to every channel."""
        self.rx = self.rx * np.exp(1j * self.rng.uniform(-np.pi, np.pi))

    def capture_sequence(self) -> list:
        out = []
        for i, slot in enumerate(self.schedule):
            self._seed += 1
            waves = self.waves if (self.off_after is None or i <= self.off_after) else []
            x = element_snapshots(self.manifold, waves, self.f_hz, self.dwell, seed=self._seed)
            elements = []
            k = 0
            for c in range(self.ch):
                if c == self.ref_ch:
                    elements.append(self.ref_el)
                else:
                    elements.append(int(slot[k]))
                    k += 1
            Y = x[elements] * self.rx[:, None]
            out.append((elements, Y))
        return out


class ReceiverBank:
    """Receiver channels grouped into LO domains (one per device, or one for
    a shared LO), behaving like a hardware class on retune and power-up
    (lit2 §3.18, §4.6; plan §11.4.8):

      on_retune      keep    — LO shared and deterministic (TwinRX, N321 star)
                     random  — independent LOs: a new phase per domain (Kraken,
                               two B210s)
                     flip180 — N310 external LO: 0° or 180° per domain
      on_power_cycle keep    — the table survives a reboot (TwinRX, NI-2955)
                     random  — new phases per domain (N310/N32x class)

    Each channel also has a fixed cable/gain offset. `apply(x)` multiplies
    element-ordered samples by the current channel response; `truth()` is
    the receiver table that would undo it (channel 0 = 1)."""

    def __init__(self, n: int, domains=None, on_retune: str = "keep",
                 on_power_cycle: str = "keep", seed: int = 0):
        self.n = int(n)
        self.domains = list(domains) if domains is not None else [0] * self.n
        self.on_retune = on_retune
        self.on_power_cycle = on_power_cycle
        self.rng = np.random.default_rng(seed)
        self.fixed = (10 ** (self.rng.normal(0, 0.5, self.n) / 20)
                      * np.exp(1j * self.rng.uniform(-np.pi, np.pi, self.n)))
        nd = max(self.domains) + 1
        self.lo = np.exp(1j * self.rng.uniform(-np.pi, np.pi, nd))

    def _new(self, how: str) -> None:
        nd = self.lo.size
        if how == "random":
            self.lo = np.exp(1j * self.rng.uniform(-np.pi, np.pi, nd))
        elif how == "flip180":
            self.lo = self.lo * np.where(self.rng.random(nd) < 0.5, -1.0, 1.0)

    def retune(self) -> None:
        self._new(self.on_retune)

    def power_cycle(self) -> None:
        self._new(self.on_power_cycle)

    def response(self) -> np.ndarray:
        return self.fixed * self.lo[self.domains]

    def apply(self, x) -> np.ndarray:
        return np.asarray(x) * self.response()[:, None]

    def truth(self) -> np.ndarray:
        r = self.response()
        return r[0] / r
