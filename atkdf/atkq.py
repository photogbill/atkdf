"""Reading atkdaq's stream: wire format v1 of `atkdaq/include/atkdaq_frame.h`.

**Written from that header, not from atkdaq's own Python package.** atkdaq is
GPL-2.0 because it links librtlsdr, and its `python/atkdaq` is part of it.
The engine keeps its distance the way ATK already does with `rtl_sdr.exe`:
it starts `atkdaq.exe` as a separate program (`ProcessSource`) and reads the
documented bytes. The header file is the contract; where this module and the
header disagree, the header wins and this module is the bug.
`tests/test_atkq.py` reads a recording atkdaq itself wrote
(`tests/fixtures/synth_240k.atkq`), which is what holds this reader to that
writer.

WHAT A FRAME IS. A 96-byte fixed header, then one 32-byte record per channel
(found through `ch_offset` / `ch_stride`, never assumed), then the payload:
`n_channels × samples_per_ch` int8 I/Q pairs, CHANNEL-MAJOR. The DAQ has
already applied each channel's integer delay; what it REPORTS and does not
apply is each channel's complex weight `w` and its fractional delay `frac`.
`calibration.py` applies them. Every header carries a CRC-32; a reader that
loses its place skips to the next "ATKQ" whose CRC checks, and never hands
back a frame that did not validate.

THE SAMPLE VALUE. fmt 0 stores `raw_u8 − 128`. The RTL2832U's midpoint is
127.5, so the centred value is `v + 0.5`; `Frame.iq()` returns
`(v + 0.5) / 128`. The scale is common to every channel and does not move a
bearing; the half-LSB offset is what keeps a DC spike from appearing that the
ADC did not produce.
"""

from __future__ import annotations

import io
import json
import os
import shlex
import shutil
import struct
import subprocess
import threading
import time
import zlib
from dataclasses import dataclass, field, replace
from enum import IntEnum, IntFlag
from pathlib import Path
from typing import BinaryIO, Callable, Iterator

import numpy as np

MAGIC = b"ATKQ"
VERSION = 1
FIXED_BYTES = 96
CH_STRIDE = 32
MAX_CHANNELS = 16


class Flag(IntFlag):
    """`flags`. Anything set here except SEGMENT_START, CLIPPED, OVERRUN and
    SYNTHETIC means the frame must not be direction-found as it stands."""

    MISALIGNED = 1 << 0
    NOISE_ON = 1 << 1
    SEGMENT_START = 1 << 2
    CAL_STALE = 1 << 3
    OVERRUN = 1 << 4
    RETUNE = 1 << 5
    CLIPPED = 1 << 6
    DISCONTINUITY = 1 << 7
    SYNTHETIC = 1 << 8


class CalState(IntEnum):
    NONE = 0
    SYNCED = 1
    CALIBRATED = 2
    MISALIGNED = 3
    CALIBRATING = 4


class Device(IntEnum):
    KRAKEN = 1
    SYNTH = 2


class AtkqError(ValueError):
    """A header that does not validate. `code` follows atkdaq_hdr_unpack:
    -1 need more bytes, -2 bad magic, -3 unknown version, -4 CRC mismatch,
    -5 inconsistent sizes."""

    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code


# The fixed part, field by field: (name, struct code, offset). Spelled out
# rather than packed into one format string so each offset can be checked
# against the header file by eye — and `tests/test_atkq.py` checks them.
_FIELDS = (
    ("magic", "4s", 0), ("version", "H", 4), ("hdr_bytes", "H", 6),
    ("ch_offset", "H", 8), ("ch_stride", "H", 10), ("n_channels", "B", 12),
    ("fmt", "B", 13), ("cal_state", "B", 14), ("device_kind", "B", 15),
    ("flags", "I", 16), ("seq", "I", 20), ("t_utc_ns", "q", 24),
    ("fc_hz", "Q", 32), ("fs_hz", "I", 40), ("samples_per_ch", "I", 44),
    ("segment", "I", 48), ("passport", "I", 52), ("cal_age_ms", "I", 56),
    ("spread_cdeg", "H", 60), ("sync_conf", "H", 62),
    ("gain_tenths_db", "h", 64), ("mode", "H", 66), ("payload_bytes", "I", 68),
    ("stream_sample0", "Q", 72), ("seg_sample0", "Q", 80),
    ("hdr_crc32", "I", 88), ("reserved", "I", 92),
)
_CH_FIELDS = (("delay", "i", 0), ("frac_delay", "f", 4), ("w_re", "f", 8),
              ("w_im", "f", 12), ("drops", "I", 16), ("clip", "I", 20),
              ("sample_count", "Q", 24))


@dataclass(frozen=True)
class Channel:
    """One channel's record: what the DAQ applied and what it reports."""

    delay: int              # integer samples APPLIED; 0 on channel 0
    frac_delay: float       # residual delay behind channel 0, samples, NOT applied
    weight: complex         # calibration weight, NOT applied; 1 on channel 0
    drops: int = 0          # samples believed lost since the last sync
    clip: int = 0           # samples in THIS frame at the ADC rail
    sample_count: int = 0   # raw samples received since the stream started


@dataclass(frozen=True)
class Header:
    """A validated header. Field names are the header file's."""

    n_channels: int
    fs_hz: int
    samples_per_ch: int
    channels: tuple
    fc_hz: int = 0
    seq: int = 0
    t_utc_ns: int = 0
    flags: int = 0
    cal_state: int = CalState.NONE
    device_kind: int = Device.KRAKEN
    segment: int = 0
    passport: int = 0
    cal_age_ms: int = 0
    spread_cdeg: int = 0xFFFF
    sync_conf: int = 0
    gain_tenths_db: int = 0
    mode: int = 0
    stream_sample0: int = 0
    seg_sample0: int = 0
    fmt: int = 0
    version: int = VERSION
    hdr_bytes: int = 0
    payload_bytes: int = 0

    # -- derived -----------------------------------------------------------
    def has(self, flag: int) -> bool:
        return bool(self.flags & int(flag))

    @property
    def t_utc(self) -> float:
        """UTC of channel 0's first payload sample, Unix seconds."""
        return self.t_utc_ns * 1e-9

    @property
    def duration_s(self) -> float:
        return self.samples_per_ch / float(self.fs_hz)

    @property
    def spread_deg(self) -> float | None:
        """Post-calibration residual phase spread on the noise source."""
        return None if self.spread_cdeg == 0xFFFF else self.spread_cdeg / 100.0

    @property
    def sync_ptn_db(self) -> float:
        """The sync's worst peak-to-next ratio (sync_conf is centi-dB)."""
        return self.sync_conf / 100.0

    @property
    def delays(self) -> np.ndarray:
        return np.array([c.delay for c in self.channels], dtype=np.int64)

    @property
    def weights(self) -> np.ndarray:
        return np.array([c.weight for c in self.channels], dtype=np.complex128)

    @property
    def frac_delays(self) -> np.ndarray:
        return np.array([c.frac_delay for c in self.channels], dtype=np.float64)

    @property
    def session_ns(self) -> int:
        """When the stream started (UTC ns): the same number on every frame
        of one atkdaq run, a different one on the next. A passport number
        restarts at 1 with every run, so (session, passport) is what names a
        calibration uniquely."""
        return int(self.t_utc_ns - round(self.stream_sample0 * 1e9 / self.fs_hz))

    def why_not_usable(self) -> list:
        """The reasons this frame must not be direction-found, in words.
        Empty means usable."""
        out = []
        if self.cal_state != CalState.CALIBRATED:
            try:
                name = CalState(self.cal_state).name.lower()
            except ValueError:
                name = str(self.cal_state)
            out.append(f"calibration state is {name}")
        for flag, why in ((Flag.MISALIGNED, "the channels are not sample-aligned"),
                          (Flag.NOISE_ON, "the payload is the noise source"),
                          (Flag.RETUNE, "a retune or gain change has not settled"),
                          (Flag.CAL_STALE, "the calibration is stale")):
            if self.has(flag):
                out.append(why)
        if self.fmt != 0:
            out.append(f"unknown sample format {self.fmt}")
        return out

    @property
    def usable_for_df(self) -> bool:
        return not self.why_not_usable()


@dataclass
class Frame:
    """A header and its payload bytes."""

    header: Header
    payload: bytes | memoryview | np.ndarray

    @property
    def ci8(self) -> np.ndarray:
        """(n_channels, samples, 2) int8 — the bytes as sent."""
        h = self.header
        a = np.frombuffer(self.payload, dtype=np.int8,
                          count=h.n_channels * h.samples_per_ch * 2)
        return a.reshape(h.n_channels, h.samples_per_ch, 2)

    def iq(self) -> np.ndarray:
        """(n_channels, samples) complex64, centred: (v + 0.5) / 128.
        Delays already applied by the DAQ; weights NOT."""
        h = self.header
        f = np.frombuffer(self.payload, dtype=np.int8,
                          count=h.n_channels * h.samples_per_ch * 2).astype(np.float32)
        f += 0.5
        f *= 1.0 / 128.0
        return f.view(np.complex64).reshape(h.n_channels, h.samples_per_ch)

    @property
    def usable_for_df(self) -> bool:
        return self.header.usable_for_df


# ---------------------------------------------------------------------------
# parse / pack
# ---------------------------------------------------------------------------

def _crc(buf, hdr_bytes: int) -> int:
    """CRC-32 (zlib's) of the header with the CRC field (bytes 88..91) zero."""
    c = zlib.crc32(bytes(buf[:88]))
    c = zlib.crc32(b"\0\0\0\0", c)
    return zlib.crc32(bytes(buf[92:hdr_bytes]), c) & 0xFFFFFFFF


def parse_header(buf) -> Header:
    """Validate and decode one header at the start of `buf`.

    Order matters and is the header file's: magic, sizes, CRC, THEN the
    version — so a damaged version byte reads as damage (-4), and only a
    header that checks can claim to be a newer format (-3).
    """
    mv = memoryview(buf)
    if len(mv) < FIXED_BYTES:
        raise AtkqError(-1, f"need {FIXED_BYTES} bytes for a header, have {len(mv)}")
    if bytes(mv[:4]) != MAGIC:
        raise AtkqError(-2, "not an atkdaq frame (no ATKQ magic)")
    hdr_bytes, ch_off, ch_stride = struct.unpack_from("<HHH", mv, 6)
    nch = mv[12]
    if not (1 <= nch <= MAX_CHANNELS) or ch_off < FIXED_BYTES \
            or ch_stride < CH_STRIDE or hdr_bytes < ch_off + nch * ch_stride:
        raise AtkqError(-5, f"inconsistent header sizes (hdr_bytes {hdr_bytes}, "
                            f"channels {nch} at {ch_off}+{ch_stride})")
    if len(mv) < hdr_bytes:
        raise AtkqError(-1, f"need {hdr_bytes} header bytes, have {len(mv)}")
    stored = struct.unpack_from("<I", mv, 88)[0]
    if _crc(mv, hdr_bytes) != stored:
        raise AtkqError(-4, "header CRC mismatch (damaged, or not a header)")
    version = struct.unpack_from("<H", mv, 4)[0]
    if version != VERSION:
        raise AtkqError(-3, f"wire format version {version}; this reader knows "
                            f"version {VERSION} — update atkdf, or run the atkdaq "
                            f"that matches it")
    f = {name: struct.unpack_from("<" + code, mv, off)[0] for name, code, off in _FIELDS}
    chans = []
    for i in range(nch):
        base = ch_off + i * ch_stride
        c = {name: struct.unpack_from("<" + code, mv, base + off)[0]
             for name, code, off in _CH_FIELDS}
        chans.append(Channel(delay=int(c["delay"]), frac_delay=float(c["frac_delay"]),
                             weight=complex(c["w_re"], c["w_im"]), drops=int(c["drops"]),
                             clip=int(c["clip"]), sample_count=int(c["sample_count"])))
    if f["payload_bytes"] != nch * f["samples_per_ch"] * 2:
        raise AtkqError(-5, f"payload_bytes {f['payload_bytes']} is not "
                            f"{nch} × {f['samples_per_ch']} × 2")
    if f["fs_hz"] <= 0:
        raise AtkqError(-5, "sample rate 0")
    return Header(n_channels=nch, fs_hz=f["fs_hz"], samples_per_ch=f["samples_per_ch"],
                  channels=tuple(chans), fc_hz=f["fc_hz"], seq=f["seq"],
                  t_utc_ns=f["t_utc_ns"], flags=f["flags"], cal_state=f["cal_state"],
                  device_kind=f["device_kind"], segment=f["segment"],
                  passport=f["passport"], cal_age_ms=f["cal_age_ms"],
                  spread_cdeg=f["spread_cdeg"], sync_conf=f["sync_conf"],
                  gain_tenths_db=f["gain_tenths_db"], mode=f["mode"],
                  stream_sample0=f["stream_sample0"], seg_sample0=f["seg_sample0"],
                  fmt=f["fmt"], version=version, hdr_bytes=hdr_bytes,
                  payload_bytes=f["payload_bytes"])


def pack_header(h: Header) -> bytes:
    """Serialise `h` (sizes and CRC computed here). For the simulator and the
    tests; the engine never writes to atkdaq."""
    n = h.n_channels
    if len(h.channels) != n:
        raise ValueError(f"{n} channels declared, {len(h.channels)} records")
    total = FIXED_BYTES + n * CH_STRIDE
    buf = bytearray(total)
    vals = dict(magic=MAGIC, version=VERSION, hdr_bytes=total, ch_offset=FIXED_BYTES,
                ch_stride=CH_STRIDE, n_channels=n, fmt=h.fmt, cal_state=int(h.cal_state),
                device_kind=int(h.device_kind), flags=int(h.flags), seq=h.seq,
                t_utc_ns=h.t_utc_ns, fc_hz=h.fc_hz, fs_hz=h.fs_hz,
                samples_per_ch=h.samples_per_ch, segment=h.segment, passport=h.passport,
                cal_age_ms=h.cal_age_ms, spread_cdeg=h.spread_cdeg, sync_conf=h.sync_conf,
                gain_tenths_db=h.gain_tenths_db, mode=h.mode,
                payload_bytes=n * h.samples_per_ch * 2, stream_sample0=h.stream_sample0,
                seg_sample0=h.seg_sample0, hdr_crc32=0, reserved=0)
    for name, code, off in _FIELDS:
        struct.pack_into("<" + code, buf, off, vals[name])
    for i, c in enumerate(h.channels):
        base = FIXED_BYTES + i * CH_STRIDE
        w = complex(c.weight)
        for (name, code, off), v in zip(_CH_FIELDS, (c.delay, c.frac_delay, w.real, w.imag,
                                                     c.drops, c.clip, c.sample_count)):
            struct.pack_into("<" + code, buf, base + off, v)
    struct.pack_into("<I", buf, 88, _crc(buf, total))
    return bytes(buf)


def encode_frame(h: Header, iq_or_ci8) -> bytes:
    """Header + payload. `iq_or_ci8` is (n, N) complex (quantised here the
    way the DAQ quantises: round((x·128) − 0.5), clipped to int8) or
    (n, N, 2) int8 already."""
    a = np.asarray(iq_or_ci8)
    if np.iscomplexobj(a):
        q = np.stack([a.real, a.imag], axis=-1) * 128.0 - 0.5
        a = np.clip(np.rint(q), -128, 127).astype(np.int8)
    a = np.ascontiguousarray(a, dtype=np.int8)
    if a.shape != (h.n_channels, h.samples_per_ch, 2):
        raise ValueError(f"payload shape {a.shape}, header says "
                         f"({h.n_channels}, {h.samples_per_ch}, 2)")
    return pack_header(h) + a.tobytes()


def read_frames(stream: BinaryIO, skipped: list | None = None,
                chunk: int = 1 << 20) -> Iterator[Frame]:
    """Every valid frame in `stream`, in order.

    A frame that fails to validate is never returned. The reader scans
    forward for the next "ATKQ" whose header checks and carries on; the
    count of bytes it had to skip is appended to `skipped` (a list, if given)
    each time it resynchronises, so a caller can say how much was lost.
    """
    buf = bytearray()
    eof = False

    def fill(n: int) -> bool:
        nonlocal eof
        while len(buf) < n and not eof:
            got = stream.read(max(chunk, n - len(buf)))
            if not got:
                eof = True
                break
            buf.extend(got)
        return len(buf) >= n

    pos_skipped = 0
    while True:
        if not fill(FIXED_BYTES):
            if buf and skipped is not None:
                skipped.append(pos_skipped + len(buf))
            return
        at = buf.find(MAGIC)
        if at < 0:
            pos_skipped += len(buf) - 3
            del buf[:len(buf) - 3]
            continue
        if at:
            pos_skipped += at
            del buf[:at]
            continue
        if not fill(FIXED_BYTES):
            continue
        hdr_bytes = struct.unpack_from("<H", buf, 6)[0]
        try:
            if not fill(max(FIXED_BYTES, hdr_bytes)):
                raise AtkqError(-1, "truncated header")
            h = parse_header(bytes(buf[:max(FIXED_BYTES, hdr_bytes)]))
        except AtkqError as e:
            if e.code == -1 and eof:
                if skipped is not None:
                    skipped.append(pos_skipped + len(buf))
                return
            if e.code == -3:
                raise                # a valid header of a format this reader does not know
            pos_skipped += 1
            del buf[:1]
            continue
        total = h.hdr_bytes + h.payload_bytes
        if not fill(total):
            if skipped is not None:
                skipped.append(pos_skipped + len(buf))
            return
        if pos_skipped and skipped is not None:
            skipped.append(pos_skipped)
        pos_skipped = 0
        payload = bytes(buf[h.hdr_bytes:total])
        del buf[:total]
        yield Frame(h, payload)


# ---------------------------------------------------------------------------
# status lines on atkdaq's stderr
# ---------------------------------------------------------------------------

def parse_status(line: str) -> dict | None:
    """`ATKDAQ event=sync passport=3 delays=0,-17,5 msg="a b"` → a dict of
    strings; None for any other line (atkdaq prints nothing else there, but
    a consumer must not choke on a stray line)."""
    line = line.strip()
    if not line.startswith("ATKDAQ "):
        return None
    try:
        parts = shlex.split(line[7:], posix=True)
    except ValueError:
        return None
    out = {}
    for p in parts:
        k, sep, v = p.partition("=")
        if sep:
            out[k] = v
    return out if "event" in out else None


# ---------------------------------------------------------------------------
# frame sources (ports.FrameSource)
# ---------------------------------------------------------------------------

class FileSource:
    """A `.atkq` recording as a FrameSource. `control()` has nothing to
    control and records what it was asked, so a test can check the engine
    asked for the right thing."""

    def __init__(self, path, loop: bool = False, realtime: bool = False,
                 speed: float = 1.0):
        self.path = Path(path)
        self.loop = loop
        self.realtime = realtime
        self.speed = float(speed)
        self.commands: list = []
        self.skipped: list = []

    def frames(self) -> Iterator[Frame]:
        while True:
            t0 = None
            with self.path.open("rb") as f:
                for fr in read_frames(f, self.skipped):
                    if self.realtime:
                        s = fr.header.stream_sample0 / fr.header.fs_hz / self.speed
                        if t0 is None:
                            t0 = time.monotonic() - s
                        lag = t0 + s - time.monotonic()
                        if lag > 0:
                            time.sleep(lag)
                    yield fr
            if not self.loop:
                return

    def control(self, cmd: str) -> None:
        self.commands.append(str(cmd))


def find_atkdaq(explicit=None) -> Path | None:
    """atkdaq.exe: an explicit path, then $ATKDAQ_EXE, then a sibling
    checkout's bin\\, then PATH. ATK passes the path its binary finder
    chose; this order is for everyone else."""
    cands = []
    if explicit:
        cands.append(Path(explicit))
    if os.environ.get("ATKDAQ_EXE"):
        cands.append(Path(os.environ["ATKDAQ_EXE"]))
    here = Path(__file__).resolve().parents[2]
    for name in ("atkdaq.exe", "atkdaq"):
        cands.append(here / "atkdaq" / "bin" / name)
    for c in cands:
        if c.is_file():
            return c
    w = shutil.which("atkdaq")
    return Path(w) if w else None


class ProcessSource:
    """atkdaq.exe as a FrameSource: frames from its stdout, status lines from
    its stderr, commands to its stdin.

    The program is run, not imported — see the module docstring for why.
    `events` holds every parsed status line in order; `status` the latest
    `event=status` line; `on_event` (if given) is called from the stderr
    thread for each one. Closing sends `quit` and waits; atkdaq exits 0 on
    quit, on EOF on stdin, or when the reader goes away.
    """

    def __init__(self, args=(), exe=None, on_event: Callable | None = None,
                 bufsize: int = 4 << 20):
        path = find_atkdaq(exe)
        if path is None:
            raise FileNotFoundError(
                "atkdaq not found: set ATKDAQ_EXE, or build it (atkdaq\\build.bat)")
        self.exe = path
        self.argv = [str(path), "run", *[str(a) for a in args]]
        self.on_event = on_event
        self.events: list = []
        self.status: dict = {}
        self.skipped: list = []
        self._lock = threading.Lock()
        self.proc = subprocess.Popen(self.argv, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     bufsize=bufsize)
        self._err = threading.Thread(target=self._read_stderr, daemon=True)
        self._err.start()

    def _read_stderr(self) -> None:
        for raw in iter(self.proc.stderr.readline, b""):
            ev = parse_status(raw.decode("utf-8", "replace"))
            if ev is None:
                continue
            with self._lock:
                self.events.append(ev)
                if ev.get("event") == "status":
                    self.status = ev
            if self.on_event is not None:
                try:
                    self.on_event(ev)
                except Exception:                           # noqa: BLE001
                    pass

    def frames(self) -> Iterator[Frame]:
        yield from read_frames(self.proc.stdout, self.skipped)

    def control(self, cmd: str) -> None:
        try:
            self.proc.stdin.write((str(cmd).strip() + "\n").encode())
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass

    def wait_event(self, name: str, timeout: float = 10.0, after: int = 0) -> dict | None:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            with self._lock:
                for ev in self.events[after:]:
                    if ev.get("event") == name:
                        return ev
            time.sleep(0.02)
        return None

    def close(self, timeout: float = 5.0) -> int | None:
        self.control("quit")
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            return self.proc.wait(timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            return self.proc.wait(timeout)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
