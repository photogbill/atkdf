"""DF everything: every active channel in the window, at once, without a retune.

ATKDF_PLAN §14 / CUED_DF_PLAN §5.5 / atkdsp's CHANNELIZER_DF_REQUIREMENTS: the
cheapest bearing is the one that never retunes (every Kraken retune is a
fresh noise-source calibration, ~0.2–0.6 s blind). So the five calibrated
element streams go through ONE coherent polyphase filterbank — 192 channels
of 12.5 kHz at 2.4 MS/s — and each channel that lights up gets its own 5×5
covariance, accumulated over the whole transmission, and one bearing when it
ends, through the same `doa.estimate` every other bearing goes through (so
the estimator, the source count, the likelihood and the record are exactly
the band path's). Channelising first is also the ≈23 dB in-channel gain the
sensitivity budget counts on.

THE BACKEND IS A PORT. `GridBackend` is four calls — channelize a block,
that block's per-channel power, accumulate that block into the covariance of
the gated channels, take (and clear) some channels' covariance. atkdf ships
`CpuGridBackend` on atkdsp's PFB (`atkdsp_pfb_process_n`,
`atkdsp_pfb_cov_accumulate`, `atkdsp_pfb_channel_power`; native, else its
numpy twin). ATK supplies a GPU one (atk_gpu: the grid and the covariance
stay on the card, only R comes back — `atk.core.dsp_gpu.GpuGridBackend`) and
the `accel` object that decides when to use it. atkdf never imports either.

GATING. Per block, each channel's power (mean over elements) against the
grid's noise floor — that block's median channel, which is the floor as long
as fewer than half the channels are busy. A channel OPENS at
`open_db` over the floor (unless it is `adjacent_db` under a neighbour — that
neighbour's spill) and CLOSES after `gap_s` below `close_db` (the hysteresis
keeps a fading handheld one burst) or after `max_s`. Only open
channels accumulate. The channels either side of DC are skipped by default
(the LO leak). A closed channel with at least `min_hops` hops becomes a
`GridBurst`; the Engine turns it into a BearingRecord.

SNAPSHOTS. The bank is 2× oversampled, so adjacent hops of one channel are
correlated; the count handed to the estimator (MDL, the Cramér-Rao sigma) is
hops / oversample — the honest number of independent snapshots.

SWITCHING BACKENDS (the card, when ATK offers one). The covariance lives in
the backend, so a switch closes every open burst first (each gets its
bearing from what it has), then the new backend starts clean; its first
L = nchan × taps samples read a zero history, so that one block is
channelized but not accumulated. Retunes (`reset`) are free switch points.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("atkdf.channel_grid")

OVERSAMPLE = 2
TAPS_PER_POLY = 12
ATTEN_DB = 80.0
SPACING_HZ = 12_500.0


def fast_count(n: int, oversample: int = OVERSAMPLE) -> int:
    """The nearest channel count to n that is a multiple of `oversample` and
    has no prime factor above 7 (pocketfft and cuFFT have passes for those;
    a prime-size FFT per hop costs several times more — atk_gpu §7B)."""
    def ok(c):
        for p in (2, 3, 5, 7):
            while c > 1 and c % p == 0:
                c //= p
        return c == 1
    n0 = max(2 * oversample, int(n) - int(n) % oversample)
    for d in range(0, n0, oversample):
        for c in (n0 - d, n0 + d):
            if c >= 2 * oversample and c % oversample == 0 and ok(c):
                return c
    return n0


# ---- the port ---------------------------------------------------------------------------
class GridBackend:
    """What ChannelGrid needs from a filterbank. Attributes: nchan, nel,
    out_rate, max_block (most samples per element one `process` takes), name."""

    def process(self, x_el) -> int:            # channelize one block; hops made
        raise NotImplementedError

    def power(self) -> np.ndarray:             # [nchan] mean |y|^2 over elements, this block
        raise NotImplementedError

    def accumulate(self, gate) -> None:        # this block's hops into R, gated channels
        raise NotImplementedError

    def take(self, chans) -> tuple:            # (R [k][nel][nel] summed, counts [k]); clears them
        raise NotImplementedError


def _kernels(prefer_native: bool = True):
    """(Pfb class, cov_accumulate, channel_power, description) from atkdsp —
    native when it loads, else the numpy twin; or raises."""
    from . import _atkdsp
    mod = _atkdsp.module()
    if mod is None:
        raise RuntimeError(f"DF everything needs atkdsp's channelizer: {_atkdsp.why()}")
    if prefer_native:
        try:
            if bool(getattr(mod, "available", lambda: False)()) and hasattr(mod, "Pfb"):
                return (mod.Pfb, mod.pfb_cov_accumulate, mod.pfb_channel_power,
                        f"atkdsp native {getattr(mod, 'version', lambda: '?')()}")
        except Exception as exc:                          # noqa: BLE001
            log.info("atkdsp native PFB unavailable: %s", exc)
    import importlib
    ref = importlib.import_module("atkdsp.reference")
    return ref.Pfb, ref.pfb_cov_accumulate, ref.pfb_channel_power, "atkdsp numpy twin (slow)"


class CpuGridBackend(GridBackend):
    """atkdsp's PFB on the CPU; the covariance is a numpy array here."""

    def __init__(self, fs: float, nchan: int, nel: int, prefer_native: bool = True):
        Pfb, self._acc, self._pow, why = _kernels(prefer_native)
        self.nchan, self.nel = int(nchan), int(nel)
        self._p = Pfb(float(fs), self.nchan, OVERSAMPLE, TAPS_PER_POLY, ATTEN_DB, self.nel)
        self.out_rate = OVERSAMPLE * float(fs) / self.nchan
        self.max_block = 1 << 40
        self.name = f"CPU ({why})"
        self.R = np.zeros((self.nchan, self.nel, self.nel), np.complex64)
        self.counts = np.zeros(self.nchan, np.uint32)
        self._y = None

    def process(self, x_el) -> int:
        x = np.asarray(x_el, np.complex64)
        self._y = self._p.process_n([x[e] for e in range(self.nel)], self.nel)
        return int(self._y.shape[2])

    def power(self) -> np.ndarray:
        if self._y is None or self._y.shape[2] == 0:
            return np.zeros(self.nchan, np.float32)
        _pw, mean = self._pow(self._y, self.nel, self.nchan)
        return np.asarray(mean, np.float32)

    def accumulate(self, gate) -> None:
        if self._y is None or self._y.shape[2] == 0:
            return
        self._acc(self._y, self.nel, self.nchan, np.ascontiguousarray(gate, np.uint8), self.R, self.counts)

    def take(self, chans) -> tuple:
        ch = np.asarray(list(chans), np.intp)
        R = self.R[ch].astype(np.complex128)
        c = self.counts[ch].copy()
        self.R[ch] = 0
        self.counts[ch] = 0
        return R, c


# ---- the grid -----------------------------------------------------------------------------
@dataclass
class GridBurst:
    """One channel's transmission, ready for `doa.estimate`."""
    chan: int
    f_off_hz: float                 # channel centre from the tuned centre
    bw_hz: float
    n0: int                         # stream samples it spans
    n1: int
    hops: int
    snapshots: int                  # independent: hops / oversample
    R: np.ndarray                   # (nel, nel) complex128, normalised by hops
    power_db: float                 # mean in-channel power over the floor while open
    end_reason: str
    backend: str = ""


@dataclass
class _Open:
    n0: int
    last_hot: int
    n_last: int
    pwr_sum: float = 0.0
    pwr_n: int = 0


@dataclass
class ChannelGrid:
    """Feed `process(x_el, stream_sample0)` element-ordered, calibrated blocks
    (nel, n); it returns the GridBursts that ended. `reset()` on a retune.
    `accel`: an optional object with `poll(now, load, fs, nchan, nel, block,
    current_backend)` -> a ready GridBackend to switch to, the string "cpu"
    (back to atkdsp's), or None (stay); and `failed(why)`. ATK's GPU is one."""

    fs: float
    nel: int
    spacing_hz: float = SPACING_HZ
    open_db: float = 10.0
    close_db: float = 6.0
    gap_s: float = 0.25
    max_s: float = 5.0
    min_hops: int = 32
    #: A channel this far below a neighbour does not OPEN: it is that
    #: neighbour's spill (a 12.5 kHz signal in 12.5 kHz channels leaks into
    #: both neighbours ~14 dB down) — one transmitter, one bearing. A real
    #: adjacent-channel signal more than this much weaker is the price.
    adjacent_db: float = 10.0
    skip_dc: int = 1                 # channels either side of DC left out
    prefer_native: bool = True
    accel: object | None = None
    backend: GridBackend | None = None
    nchan: int = 0
    floor: float | None = None
    n_blocks: int = 0
    _open: dict = field(default_factory=dict)
    _skip_next: bool = False
    _load: float | None = None

    def __post_init__(self):
        self._req_spacing = float(self.spacing_hz)       # asked for; spacing_hz becomes fs / nchan
        self.nchan = self.nchan or fast_count(round(self.fs / self.spacing_hz))
        self.spacing_hz = self.fs / self.nchan
        if self.backend is None:
            self.backend = CpuGridBackend(self.fs, self.nchan, self.nel, self.prefer_native)
        log.info("DF everything: %d channels of %.2f kHz at %.4g MS/s, %d elements, %s",
                 self.nchan, self.spacing_hz / 1e3, self.fs / 1e6, self.nel, self.backend.name)

    # -- frequency bookkeeping ------------------------------------------------------
    def offset_hz(self, chan: int) -> float:
        k = int(chan)
        if k >= self.nchan // 2:
            k -= self.nchan
        return k * self.spacing_hz

    def _dc_mask(self) -> np.ndarray:
        m = np.ones(self.nchan, bool)
        for d in range(-self.skip_dc, self.skip_dc + 1):
            m[d % self.nchan] = False
        return m

    # -- the work ---------------------------------------------------------------------
    def process(self, x_el, stream_sample0: int) -> list:
        x = np.asarray(x_el)
        if x.ndim != 2 or x.shape[0] != self.nel:
            raise ValueError(f"ChannelGrid needs ({self.nel}, n) samples, got {x.shape}")
        out = []
        out += self._maybe_switch(x.shape[1])
        step = max(1, int(getattr(self.backend, "max_block", x.shape[1])))
        for s in range(0, x.shape[1], step):
            out += self._block(x[:, s:s + step], int(stream_sample0) + s)
        return out

    def _block(self, x, s0: int) -> list:
        b = self.backend
        t0 = time.perf_counter()
        try:
            hops = b.process(x)
            p = np.asarray(b.power(), np.float64) if hops else None
            gate = np.zeros(self.nchan, np.uint8)
            out = []
            if p is not None:
                out = self._gate(p, gate, s0, s0 + x.shape[1])
                if self._skip_next:
                    self._skip_next = False   # a fresh bank's first block reads zero history
                elif gate.any():
                    b.accumulate(gate)
        except Exception as exc:                              # noqa: BLE001
            return self._backend_failed(exc)
        self.n_blocks += 1
        dt = time.perf_counter() - t0
        sig = x.shape[1] / self.fs
        if sig > 0:
            a = 1.0 - math.exp(-sig / 1.0)
            x_ = dt / sig
            self._load = x_ if self._load is None else self._load + a * (x_ - self._load)
        return out

    def _gate(self, p, gate, n0, n1) -> list:
        usable = self._dc_mask()
        med = float(np.median(p[usable])) if usable.any() else float(np.median(p))
        # THIS block's median channel, not a smoothed one: the receiver's
        # gain and the DAQ's scaling move the whole floor from one frame to
        # the next (a strong carrier keying changes the scaling), and a slow
        # floor then calls the whole band busy. 192 channels make the
        # per-block median steady enough on its own.
        self.floor = max(med, 1e-30)
        hi = self.floor * 10 ** (self.open_db / 10.0)
        lo = self.floor * 10 ** (self.close_db / 10.0)
        gap = int(self.gap_s * self.fs)
        mx = int(self.max_s * self.fs)
        ended = []
        nb = np.maximum(np.roll(p, 1), np.roll(p, -1))
        spill = nb > p * 10 ** (self.adjacent_db / 10.0)
        for k in np.nonzero(usable & (p > lo))[0]:
            k = int(k)
            o = self._open.get(k)
            if o is None and (p[k] <= hi or spill[k]):
                continue
            if o is None:
                o = self._open[k] = _Open(n0=n0, last_hot=n1, n_last=n1)
            o.last_hot = n1
            o.n_last = n1
            o.pwr_sum += float(p[k] / self.floor)
            o.pwr_n += 1
            gate[k] = 1
        for k, o in list(self._open.items()):
            if gate[k]:
                if n1 - o.n0 >= mx:
                    ended.append((k, "max"))
                continue
            if n1 - o.last_hot >= gap:
                ended.append((k, "gap"))
            else:
                gate[k] = 1                     # inside the hang time: keep integrating
                o.n_last = n1
        return self._close(ended) if ended else []

    def _close(self, ended) -> list:
        chans = [k for k, _ in ended]
        R, counts = self.backend.take(chans)
        out = []
        for j, (k, why) in enumerate(ended):
            o = self._open.pop(k)
            hops = int(counts[j])
            if hops < self.min_hops:
                continue
            out.append(GridBurst(
                chan=k, f_off_hz=self.offset_hz(k), bw_hz=self.spacing_hz, n0=o.n0, n1=o.n_last,
                hops=hops, snapshots=max(1, hops // OVERSAMPLE), R=R[j] / float(hops),
                power_db=10.0 * math.log10(max(o.pwr_sum / max(o.pwr_n, 1), 1e-30)),
                end_reason=why, backend=self.backend.name))
        return out

    def flush(self, reason: str = "flush") -> list:
        """End every open transmission now (a retune, the end of a file)."""
        if not self._open:
            return []
        try:
            return self._close([(k, reason) for k in list(self._open)])
        except Exception as exc:                              # noqa: BLE001
            log.warning("DF everything: could not read the open channels' covariance: %s", exc)
            self._open.clear()
            return []

    def reset(self) -> None:
        """A retune: what was open belonged to another frequency."""
        self._open.clear()
        self.floor = None
        b = self.backend
        try:
            b.take(range(self.nchan))
        except Exception:                                     # noqa: BLE001
            log.debug("reset: backend take failed", exc_info=True)
        self._skip_next = True

    # -- the card (ATK's accel) ------------------------------------------------------------
    @property
    def load(self) -> float | None:
        """Processing time / signal time of the grid, smoothed over ~1 s."""
        return self._load

    def _maybe_switch(self, block: int) -> list:
        acc = self.accel
        if acc is None:
            return []
        try:
            new = acc.poll(time.monotonic(), self._load, self.fs, self.nchan, self.nel, block,
                           self.backend)
        except Exception:                                     # noqa: BLE001
            log.warning("DF everything: the accelerator's poll failed", exc_info=True)
            return []
        if new is None or new is self.backend:
            return []
        if isinstance(new, str) and new == "cpu":            # "back to your own"
            if self.backend.name.startswith("CPU"):
                return []
            new = CpuGridBackend(self.fs, self.nchan, self.nel, self.prefer_native)
        out = self.flush("switch")
        log.info("DF everything: backend %s -> %s", self.backend.name, new.name)
        self.backend = new
        self._skip_next = True
        self._load = None
        return out

    def _backend_failed(self, exc) -> list:
        why = f"{type(exc).__name__}: {exc}"
        log.warning("DF everything: %s failed (%s) — back to the CPU", self.backend.name, why)
        self._open.clear()
        if self.accel is not None:
            try:
                self.accel.failed(why)
            except Exception:                                 # noqa: BLE001
                pass
        self.backend = CpuGridBackend(self.fs, self.nchan, self.nel, self.prefer_native)
        self._skip_next = True
        return []

    def rebuild(self, fs: float) -> "ChannelGrid":
        """The same grid settings (and accelerator) at another sample rate:
        a new channel count, a new bank, nothing carried over."""
        return ChannelGrid(fs=float(fs), nel=self.nel, spacing_hz=self._req_spacing,
                           open_db=self.open_db, close_db=self.close_db, gap_s=self.gap_s, max_s=self.max_s,
                           min_hops=self.min_hops, adjacent_db=self.adjacent_db, skip_dc=self.skip_dc,
                           prefer_native=self.prefer_native, accel=self.accel)

    def describe(self) -> str:
        return (f"DF everything: {self.nchan} × {self.spacing_hz / 1e3:.2f} kHz, {len(self._open)} open, "
                f"{self.backend.name}")
