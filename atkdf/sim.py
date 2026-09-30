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
    every frame, SEGMENT_START at a retune.

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
    def _emitter_samples(self, e: Emitter, n: int) -> np.ndarray:
        """One emitter's contribution at the elements: (n_ch, n) complex."""
        fs = self.sc.fs_hz
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
                yield Frame(h, atkq.encode_frame(h, x)[h.hdr_bytes:])
                continue
            base_of = {}
            x = np.zeros((self.sc.n_ch, n), dtype=complex)
            for i, e in enumerate(self.sc.emitters):
                if not e.active((k * n) / self.sc.fs_hz, ((k + 1) * n) / self.sc.fs_hz):
                    continue
                xe = self._emitter_samples(e, n)
                base_of[i] = xe
                x = x + xe
            for i in base_of:
                x = self._apply_reflectors(base_of, x, i)
            x = self._to_adc(self._receiver(x) + self._noise((self.sc.n_ch, n)))
            h = self._header(k, CalState.CALIBRATED,
                             int(Flag.SEGMENT_START) if k == 0 else 0)
            payload = atkq.encode_frame(h, x)[h.hdr_bytes:]
            yield Frame(h, payload)

    def write(self, path) -> str:
        """Record the run to a .atkq file, exactly as atkdaq would."""
        from pathlib import Path
        with Path(path).open("wb") as f:
            for fr in self.frames():
                f.write(atkq.pack_header(fr.header))
                f.write(bytes(fr.payload))
        return str(path)
