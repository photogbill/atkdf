"""Opening a recording for direction finding: SigMF and Midas BLUE, one shape.

The moved `df_file` and `ephemeris` called ATK's `recording.kind`,
`recording.header` and `recording.load`. The engine cannot import ATK, so
this is those three functions — and the pieces of `iq_recorder.load_meta`
and `dsp.iq_to_complex` they stand on — reproduced with the SAME vocabulary
and the same refusals. ATK's `tests/test_df_recio_matches_recording.py`
opens the same files through both doors and requires the same answer, so
the two cannot drift apart without a red test.

THE VOCABULARY IS BLUE'S, because the DF path already speaks it (this is
ATK's `recording.header` docstring, kept because it is the contract):

    subsize        channels in a coherent capture; 1 for an ordinary one
    ydelta         seconds between frames, i.e. 1 / per-channel rate
    xdelta         the BLUE x spacing, kept only so `_time_step` can see
                   which of the two a producer used
    center_freq    hertz, or 0 when nobody said
    timecode_unix  seconds, or 0 — never `now`, see `epoch_of`
    keywords       the format's own metadata, upper-cased

WHY THE SAMPLE RATE IS TREATED AS SACRED (ATK's rule, and the engine's):
every number downstream is samples divided by the rate. A missing rate is an
error here, never a default.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import bluefile

SIGMF_EXT = {".sigmf-data", ".sigmf-meta"}
BLUE_EXT = {".tmp", ".prm", ".blue", ".cdif", ".det"}

#: Bytes per complex sample, both spellings of each format (SigMF sidecars in
#: the wild carry either; ATK learned that the hard way — see its `dsp`).
BYTES_PER = {"cu8": 2, "ci8": 2, "cs8": 2,
             "ci16_le": 4, "ci16": 4, "cs16": 4, "ci16q11": 4,
             "cf32": 8, "cf32_le": 8}

#: Guessed from the extension when a raw file has no sidecar (ATK's table).
_RAW_DTYPE = {".cu8": "cu8", ".complex16u": "cu8", ".cs8": "ci8",
              ".ci8": "ci8", ".ci16": "ci16_le", ".cs16": "ci16_le",
              ".cf32": "cf32", ".iq": "cu8"}

DEFAULT_MAX_SAMPLES = 8_000_000

#: The RTL-SDR's unsigned-8-bit midpoint.
_RTL_DC = 127.5


# ---------------------------------------------------------------------------
# sample decoding — ATK's `dsp.iq_to_complex`, the same scalings
# ---------------------------------------------------------------------------

def _pairs(b: np.ndarray) -> np.ndarray:
    if b.size < 2:
        return b[:0]
    return b[:-1] if b.size % 2 else b


def iq_to_complex(raw, fmt: str = "cu8") -> np.ndarray:
    """Interleaved I/Q bytes of a SigMF datatype → complex64.

    cu8 is centred on 127.5 and scaled by it; ci8 by 128; ci16 by 32768;
    ci16q11 (bladeRF) by 2048; cf32 as stored. An unknown name is refused
    here — ATK's decoder falls back to cu8 for history's sake, but every
    caller in this package validates the name against `BYTES_PER` first, so
    a silent fallback would only ever hide a mistake.
    """
    buf = bytes(raw) if not isinstance(raw, (bytes, bytearray)) else raw
    if fmt == "cu8":
        b = _pairs(np.frombuffer(buf, dtype=np.uint8))
        f = (b.astype(np.float32) - _RTL_DC) / _RTL_DC
    elif fmt in ("ci8", "cs8"):
        f = _pairs(np.frombuffer(buf, dtype=np.int8)).astype(np.float32) / 128.0
    elif fmt in ("ci16_le", "ci16", "cs16"):
        f = _pairs(np.frombuffer(buf, dtype="<i2")).astype(np.float32) / 32768.0
    elif fmt == "ci16q11":
        f = _pairs(np.frombuffer(buf, dtype="<i2")).astype(np.float32) / 2048.0
    elif fmt in ("cf32", "cf32_le"):
        f = _pairs(np.frombuffer(buf, dtype="<f4")).astype(np.float32)
    else:
        raise ValueError(f"unknown sample format {fmt!r}")
    if f.size < 2:
        return np.zeros(0, dtype=np.complex64)
    return (f[0::2] + 1j * f[1::2]).astype(np.complex64)


# ---------------------------------------------------------------------------
# metadata — ATK's `iq_recorder.load_meta`, SigMF part
# ---------------------------------------------------------------------------

def _meta_base(p: Path) -> str:
    """The sidecar's base name as `load_meta` finds it: strip `.sigmf-data`
    or `.cu8` only, as ATK does. A `.sigmf-meta` path is deliberately NOT
    stripped here: `load` would then read the JSON as samples, and in ATK
    that path fails cleanly as "unknown sample format" instead."""
    base = str(p)
    for suf in (".sigmf-data", ".cu8"):
        if base.endswith(suf):
            return base[: -len(suf)]
    return base


def _doc_base(p: Path) -> str:
    """The base name `header` reads the global keywords from."""
    base = str(p)
    for suf in (".sigmf-data", ".sigmf-meta"):
        if base.lower().endswith(suf):
            return base[: -len(suf)]
    return base


def load_meta(data_path) -> dict:
    """A recording's parameters from its SigMF sidecar, as a flat dict.

    Honours ATK's `atk:datatype` hint over `core:datatype` (the bladeRF's
    ci16q11 is written as ci16 for other tools and marked for ATK), reads
    `core:num_channels`, and reads `core:geolocation` in GeoJSON order —
    [lon, lat, alt] — or every recording comes home transposed into the sea.
    Also understands the legacy `.cu8.json` sidecar. `{}` when there is none.
    """
    p = Path(data_path)
    meta_p = Path(_meta_base(p) + ".sigmf-meta")
    if meta_p.exists():
        try:
            d = json.loads(meta_p.read_text(encoding="utf-8"))
            g = d.get("global", {})
            cap = (d.get("captures") or [{}])[0]
            out = {"datatype": g.get("atk:datatype",
                                     g.get("core:datatype", "cu8")),
                   "sample_rate": g.get("core:sample_rate", 0),
                   "center_freq": cap.get("core:frequency", 0.0),
                   "hw": g.get("core:hw", ""),
                   "channels": max(1, int(g.get("core:num_channels", 1) or 1)),
                   "clock": g.get("atk:clock", ""),
                   "datetime": cap.get("core:datetime", "")}
            geo = g.get("core:geolocation") or {}
            coords = geo.get("coordinates") or []
            if len(coords) >= 2:
                out["lon"] = float(coords[0])
                out["lat"] = float(coords[1])
                if len(coords) > 2:
                    out["altitude_m"] = float(coords[2])
                out["geolocation_note"] = g.get("atk:geolocation_note", "")
            if g.get("atk:geotrack"):
                out["track"] = str(meta_p.parent / g["atk:geotrack"])
                out["track_points"] = int(g.get("atk:geotrack_points", 0))
            return out
        except (ValueError, OSError):
            pass
    legacy = Path(str(data_path) + ".json")
    if legacy.exists():
        try:
            d = json.loads(legacy.read_text(encoding="utf-8"))
            d.setdefault("datatype", d.get("format", "cu8"))
            return d
        except (ValueError, OSError):
            pass
    return {}


_DATETIME_FORMATS = ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ",
                     "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S")


def epoch_of(stamp: str) -> float:
    """An RFC 3339 stamp → Unix seconds. 0.0 when it cannot be read — never
    `now`, which would let two captures pair as though simultaneous."""
    import calendar
    import time as _t
    for fmt in _DATETIME_FORMATS:
        try:
            return float(calendar.timegm(_t.strptime(str(stamp), fmt)))
        except (ValueError, TypeError):
            continue
    return 0.0


# ---------------------------------------------------------------------------
# the three doors
# ---------------------------------------------------------------------------

def kind(path) -> str:
    """'sigmf' | 'blue' | 'raw' — by extension first, then by content."""
    p = Path(path)
    suf = p.suffix.lower()
    if suf in SIGMF_EXT:
        return "sigmf"
    if bluefile.looks_like_blue(p):
        return "blue"
    if suf in BLUE_EXT:
        return "blue"
    if Path(str(p).rsplit(".sigmf-data", 1)[0] + ".sigmf-meta").exists():
        return "sigmf"
    return "raw"


def header(path) -> dict:
    """A capture's parameters in ONE shape, whatever format it is in.

    Raises the way `bluefile.read_header` does for a bad BLUE header, so a
    caller that already handles one keeps working.
    """
    p = Path(path)
    k = kind(p)
    if k == "blue":
        hdr = dict(bluefile.read_header(p))
        hdr.setdefault("path", str(p))
        hdr["format"] = "blue"
        return hdr

    m = load_meta(p)
    rate = float(m.get("sample_rate") or 0.0)
    n_ch = max(1, int(m.get("channels", 1) or 1))
    kw = {}
    try:
        doc = json.loads(Path(_doc_base(p) + ".sigmf-meta").read_text(
            encoding="utf-8"))
        # A producer's own global keys are where a DF system puts its array
        # geometry, so they are offered under BLUE's keyword name: namespace
        # stripped, upper-cased, so DF_TYPE finds `atk:df_type`.
        for key, val in (doc.get("global") or {}).items():
            kw[str(key).split(":", 1)[-1].upper()] = val
    except (OSError, ValueError, UnicodeDecodeError, KeyError):
        pass

    return {
        "path": str(p),
        "format": k,
        "type": 2000 if n_ch > 1 else 1000,
        "subsize": n_ch,
        "sample_rate": rate,
        "center_freq": float(m.get("center_freq") or 0.0),
        # the FRAME interval: 1/rate whether or not the frame holds several
        # channels — dividing by the channel count would stretch every
        # collection's timeline by that factor
        "ydelta": (1.0 / rate) if rate > 0 else 0.0,
        "xdelta": 0.0,
        "xstart": 0.0,
        "timecode_unix": epoch_of(m.get("datetime", "")),
        "clock": m.get("clock", ""),
        "complex": True,
        "datatype": m.get("datatype", ""),
        "keywords": kw,
        "ext_keywords": {},
    }


def load(path, sample_rate: float = 0.0, datatype: str = "",
         max_samples: int | None = DEFAULT_MAX_SAMPLES,
         offset: int = 0) -> dict:
    """Open a recording → {ok, iq, sample_rate, center_freq, …}. Never raises
    on a bad file: returns ok=False with a `problem` a person can read.

    `max_samples` and `offset` count INTERLEAVED samples, exactly as in ATK:
    for a multi-channel SigMF dataset the caller multiplies by the channel
    count (`df_file.read_array` does).
    """
    p = Path(path)
    if not p.is_file():
        return {"ok": False, "problem": f"{p.name} is not a file",
                "iq": np.zeros(0, dtype=np.complex64)}
    k = kind(p)
    info = {"ok": True, "kind": k, "path": str(p), "name": p.name,
            "bytes": p.stat().st_size, "sample_rate": 0.0, "center_freq": 0.0,
            "datatype": "", "samples": 0, "rate_source": "", "notes": ""}
    try:
        if k == "blue":
            hdr = bluefile.read_header(p)
            info.update(sample_rate=hdr.get("sample_rate", 0.0),
                        center_freq=hdr.get("center_freq", 0.0),
                        samples=hdr.get("samples", 0),
                        rate_source="BLUE header")
            samples, hdr = bluefile.read(p, max_samples=max_samples,
                                         offset=offset)
            iq = (samples.astype(np.complex64) if not hdr.get("complex")
                  else samples)
        else:
            m = load_meta(p)
            info.update(sample_rate=float(m.get("sample_rate") or 0.0),
                        center_freq=float(m.get("center_freq") or 0.0),
                        timecode_unix=epoch_of(m.get("datetime", "")),
                        rate_source="SigMF metadata" if k == "sigmf"
                        else "sidecar")
            dt = datatype or m.get("datatype") or (
                _RAW_DTYPE.get(p.suffix.lower(), "") if k == "raw" else "")
            if dt not in BYTES_PER:
                return {**info, "ok": False,
                        "iq": np.zeros(0, dtype=np.complex64),
                        "problem": (f"Unknown sample format "
                                    f"{dt or '(none given)'} — pick one of "
                                    f"{', '.join(sorted(BYTES_PER))}.")}
            per = BYTES_PER[dt]
            want = None if max_samples is None else int(max_samples) * per
            with p.open("rb") as fh:
                fh.seek(int(offset) * per)
                raw = fh.read(-1 if want is None else want)
            usable = (len(raw) // per) * per
            iq = iq_to_complex(raw[:usable], dt)
            info["datatype"] = dt
    except Exception as exc:                              # noqa: BLE001
        return {**info, "ok": False, "iq": np.zeros(0, dtype=np.complex64),
                "problem": f"Could not read samples: {exc}"}

    rate = float(sample_rate or info.get("sample_rate") or 0.0)
    if iq.size == 0:
        return {**info, "ok": False, "iq": iq,
                "problem": "File contains no samples."}
    if rate <= 0:
        return {**info, "ok": False, "iq": iq, "sample_rate": 0.0,
                "problem": ("Sample rate unknown. Everything downstream is "
                            "samples ÷ rate, so a guess here would produce "
                            "confident, wrong answers.")}
    info.update(iq=iq, sample_rate=rate, samples=int(iq.size),
                duration_s=float(iq.size) / rate)
    return info
