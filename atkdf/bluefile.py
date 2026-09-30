"""Midas BLUE files — the format the radar/EW world actually hands you.

SigMF is the open-source SDR standard and ATK already writes it. But X-Midas /
NeXtMidas BLUE is what circulates in radar and EW work, so a pulse-analysis
bench that can't open a `.tmp`/`.prm`/`.blue` file is a bench you can't use on
the recordings you're most likely to be given.

FORMAT, in the parts that matter (512-byte header, then data):

    off  size  field
      0     4  version        'BLUE'
      4     4  head_rep       'EEEI' little-endian, 'IEEE' big-endian
      8     4  data_rep       same codes, for the DATA section
     24     4  ext_start      extended header offset, in 512-byte blocks
     28     4  ext_size       extended header size, bytes
     32     8  data_start     double, normally 512
     40     8  data_size      double, bytes of data
     48     4  type           1000 = single stream, 2000 = framed
     52     2  format         mode char + type char, e.g. 'CI', 'CF', 'SF'
     56     8  timecode       double, seconds since 1 Jan 1950 (J1950)
    160     4  keylength
    164    92  keywords       'KEY=value' pairs, NUL/newline separated
    256   256  adjunct        type-specific; for 1000: xstart, xdelta, xunits

Format codes are two characters: **mode** then **type**.
    mode  S scalar · C complex · V vector(3) · Q quad(4) · M magnitude
    type  B int8 · I int16 · L int32 · X int64 · F float32 · D float64

The two that bite people, and that this module handles explicitly:
  * **head_rep and data_rep are independent.** A file can have a little-endian
    header describing big-endian data. Reading both with one endianness is the
    classic BLUE bug and it produces plausible-looking garbage rather than an
    error.
  * **Timecode is J1950, not Unix.** Off by 631,152,000 seconds. A recording
    dated 1970 is this mistake every time.

Stdlib + numpy only. Written from the published X-Midas header layout — the
adjunct union varies by type, so 1000 (a plain stream, which is what an I/Q
capture is) is complete and 2000 is read far enough to be useful.

COPIED FROM ATK, 2026-09-30 (ATK c52b913, `atk/core/bluefile.py`), verbatim. ATK keeps
its own `atk/core/bluefile.py`, because ATK's own recorder and every signals-bench
module stand on it and ATK must start without the DF engine; the
engine needs one of its own because it never imports ATK. The two are
expected to diverge only by additions; ATK's tests read files written
by one through the other.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

HEADER_SIZE = 512
# Seconds between the J1950 epoch BLUE uses and the Unix epoch. Getting this
# wrong silently dates every recording to 1970.
J1950_TO_UNIX = 631_152_000

# format type char → (numpy dtype base, bytes per element)
#
# **`O` IS UNSIGNED AND `B` IS SIGNED, WHICH IS THE OPPOSITE OF WHAT THE
# LETTERS SUGGEST.** `O` is "offset byte" — an unsigned 8-bit sample. It was
# missing from this table entirely, which meant `read_header` REFUSED every
# file using it, and 8-bit unsigned is one of the commonest ways an I/Q
# capture is stored. Getting the two the wrong way round instead would be
# worse: the file would open and every sample would be off by 128.
_TYPES = {
    "O": ("u1", 1), "B": ("i1", 1), "I": ("i2", 2), "L": ("i4", 4),
    "X": ("i8", 8), "F": ("f4", 4), "D": ("f8", 8),
    "P": ("u1", 1), "A": ("u1", 1),
}
#: Signed integer codes — the ones `read` rescales to ±1 by dividing. `O`
#: needs its offset removed first and so is handled separately.
_SIGNED_INT = ("B", "I", "L", "X")
#: Formats that are not data types at all: they mean "the elements of this
#: file are records, described by the adjunct". See `read_header`.
_SENTINEL_FORMATS = ("NH", "KW")
# mode char → values per sample
_MODES = {"S": 1, "C": 2, "V": 3, "Q": 4, "M": 1, "X": 3, "T": 4}


class BlueError(ValueError):
    """Not a BLUE file, or a BLUE file this reader can't honestly handle."""


def _endian(rep: str) -> str:
    """'EEEI' → little, 'IEEE' → big. Anything else is unknown."""
    rep = (rep or "").strip().upper()
    if rep.startswith("EEEI"):
        return "<"
    if rep.startswith("IEEE"):
        return ">"
    raise BlueError(f"unknown data representation {rep!r}")


def read_header(path) -> dict:
    """Parse the 512-byte header. Raises BlueError on anything unexpected."""
    p = Path(path)
    raw = p.open("rb").read(HEADER_SIZE)
    if len(raw) < HEADER_SIZE:
        raise BlueError(f"{p.name}: shorter than a BLUE header")
    if raw[0:4] != b"BLUE":
        raise BlueError(f"{p.name}: no BLUE magic (got {raw[0:4]!r})")

    head_rep = raw[4:8].decode("ascii", "replace")
    data_rep = raw[8:12].decode("ascii", "replace")
    he = _endian(head_rep)

    def i4(off):
        return struct.unpack(he + "i", raw[off:off + 4])[0]

    def f8(off):
        return struct.unpack(he + "d", raw[off:off + 8])[0]

    fmt = raw[52:54].decode("ascii", "replace").upper()
    mode, typ = (fmt[0] if fmt else "S"), (fmt[1] if len(fmt) > 1 else "F")
    # `NH` and `KW` are SENTINELS, not data types: they mean "this file's
    # elements are records, so bytes-per-atom is not a meaningful question,
    # ask the adjunct instead". Rejecting them as an unsupported type turns
    # every record file — every ephemeris track — into an unreadable one.
    sentinel = fmt in _SENTINEL_FORMATS
    if not sentinel and typ not in _TYPES:
        raise BlueError(f"{p.name}: unsupported BLUE data type {fmt!r}")

    hdr = {
        "path": str(p),
        "head_rep": head_rep, "data_rep": data_rep,
        "header_endian": he, "data_endian": _endian(data_rep),
        "ext_start": i4(24), "ext_size": i4(28),
        "data_start": f8(32), "data_size": f8(40),
        "type": i4(48), "format": fmt, "mode": mode, "dtype_char": typ,
        "format_is_sentinel": sentinel,
        "complex": (mode == "C") and not sentinel,
        "values_per_sample": 1 if sentinel else _MODES.get(mode, 1),
        "bytes_per_value": 0 if sentinel else _TYPES[typ][1],
        "timecode_j1950": f8(56),
        "keylength": i4(160),
        "keywords": _parse_keywords(raw[164:256], i4(160)),
    }
    hdr["timecode_unix"] = (hdr["timecode_j1950"] - J1950_TO_UNIX
                            if hdr["timecode_j1950"] else 0.0)
    # THE ADJUNCT IS DISPATCHED ON CLASS, NOT ON TYPE. 1001, 2001, 5001 and
    # 5010 are all real file types in the wild and each uses its family's
    # adjunct layout; keying off the exact type number rejects them.
    hdr["class"] = hdr["type"] // 1000
    hdr.update(_adjunct(raw, hdr["class"], he))
    hdr["bytes_per_sample"] = (hdr["values_per_sample"]
                               * hdr["bytes_per_value"])
    # For a record file the stride is the RECORD, not the sample.
    stride = (hdr.get("record_length") or hdr.get("vrecord_length")
              or hdr["bytes_per_sample"])
    hdr["bytes_per_element"] = int(stride)
    if hdr["class"] in (3, 5, 6) and stride > 0:
        hdr["records"] = int(hdr["data_size"] // stride)
    if hdr["bytes_per_sample"]:
        hdr["samples"] = int(hdr["data_size"] // hdr["bytes_per_sample"])
    else:
        hdr["samples"] = 0
    hdr.update(_extended_keywords(p, hdr))
    if "center_freq" not in hdr:
        hdr.update(_freq_from(hdr.get("keywords", {})))
    _apply_tc_prec(hdr)
    if hdr["class"] == 6:
        _t6_subrecords(hdr)
    return hdr


def _apply_tc_prec(hdr: dict) -> None:
    """Fold the `TC_PREC` keyword back into the timecode.

    **A DOUBLE CANNOT HOLD J1950 SECONDS TO BETTER THAN ABOUT A MICROSECOND**,
    so the convention is to truncate `timecode` at 1 µs and carry the residual
    — down to a picosecond — as an uppercase ASCII keyword in the 92-byte main
    header. Dropping it on read costs up to a microsecond, and a microsecond
    is three hundred metres of TDOA position error. It is one keyword; there
    is no excuse for not reading it.
    """
    raw = (hdr.get("keywords", {}).get("TC_PREC")
           or hdr.get("ext_keywords", {}).get("TC_PREC"))
    if raw in (None, ""):
        return
    try:
        extra = float(raw)
    except (TypeError, ValueError):
        return
    if not np.isfinite(extra) or abs(extra) >= 1e-6:
        return                      # not a sub-microsecond residual
    hdr["timecode_precision_s"] = extra
    if hdr.get("timecode_j1950"):
        hdr["timecode_j1950"] += extra
        hdr["timecode_unix"] += extra


def _t6_subrecords(hdr: dict) -> None:
    """Class 6 keeps its real column table in the `SUBREC_DEF` keyword.

    The inline adjunct table is present but not authoritative, and the
    extended header therefore has to be read BEFORE the adjunct means
    anything — the opposite of every other class. Descriptors are 96 bytes
    and entirely ASCII, which is why they can hold 24-character names and
    per-column min/max where class 3 gets four characters and nothing else.
    """
    blob = hdr.get("ext_keywords", {}).get("SUBREC_DEF")
    if not isinstance(blob, str) or len(blob) < 96:
        return
    n = len(blob) // 96
    out = []
    for k in range(n):
        s = blob[k * 96:(k + 1) * 96]

        def num(txt, cast, default=0):
            try:
                return cast(txt.strip())
            except (TypeError, ValueError):
                return default

        out.append({"name": s[0:24].strip(),
                    "minval": num(s[24:48], float, float("nan")),
                    "maxval": num(s[48:72], float, float("nan")),
                    "offset": num(s[72:80], int),
                    "num_elts": num(s[80:84], int, 1),
                    "units": num(s[84:88], int),
                    "format": s[88:90].strip(),
                    "uprefix": num(s[90:93], int)})
    if out:
        hdr["subr"] = out
        hdr["subrecords"] = len(out)
        hdr["subr_authoritative"] = True
        # It is a parsing artefact, not metadata — surfacing 96 bytes per
        # column of ASCII to the operator as a "keyword" helps nobody.
        hdr["ext_keywords"].pop("SUBREC_DEF", None)


#: Unit codes, from the two published tables that agree. Only the ones worth
#: naming — the point is that a column labelled 60 is DEGREES OF LATITUDE and
#: a reader that ignores it will happily treat radians as degrees.
UNITS = {
    0: "none", 1: "s", 2: "s", 3: "Hz", 4: "Hz", 5: "m", 6: "m/s",
    7: "m/s²", 11: "kg", 21: "amplitude", 24: "phase (rad)",
    25: "phase (deg)", 31: "dB", 33: "rad", 34: "deg", 52: "rad/s",
    60: "latitude (deg)", 61: "longitude (deg)", 62: "altitude (ft)",
    63: "altitude (m)",
}

#: Type-5000 component coordinate systems. GEODETIC is the one that matters
#: for us, and its ordering is a trap — see `T5_STATE_COMPONENTS`.
COMP_TYPES = {0: "none", 1: "scalar", 2: "cartesian", 3: "spherical",
              4: "cylindric", 5: "ellipsoid", 6: "geodetic", 10: "matrix"}

#: The standard type-5000 state-vector component names.
#:
#: **`POS` UNDER GEODETIC IS (ALTITUDE, LATITUDE, LONGITUDE).** Altitude
#: FIRST. Every instinct says lat/lon/alt and every instinct is wrong here;
#: reading it in the obvious order puts an aircraft at 3,000° north.
T5_STATE_COMPONENTS = {
    "NAME": "point name", "NOTE": "free text",
    "POS": "position — (altitude, latitude, longitude) under GEODETIC, "
           "(x, y, z) under CARTESIAN",
    "VEL": "velocity (x, y, z)", "ACC": "acceleration (x, y, z)",
    "POSP": "secondary position — same ordering as POS",
    "ELPS": "ellipse parameters (azimuth, elevation, roll)",
    "CONN": "1 = draw a line to the previous point",
    "HDG": "heading for the symbol, 0–360",
    "FLAG": "plotting flags",
    "TIME": "time of this state vector, seconds since 1 Jan 1950",
}

_FREQ_KEYS = ("CENTER_FREQ", "CENTERFREQ", "RF_CENTER", "FREQ", "COL_RF",
              "CHAN_RF")


def _adjunct(raw: bytes, klass: int, he: str) -> dict:
    """The 256-byte adjunct at offset 256, by file class.

    Classes 1 and 2 are the ones an I/Q capture uses. Classes 3, 5 and 6
    describe RECORDS with named columns, which is how BLUE carries a table —
    a state-vector track, a PDW list, an ephemeris. Class 4 is variable-length
    keyword records. Reading only 1 and 2 means every one of those files looks
    like an unsupported type when the data is right there and described.
    """
    def f8(off):
        return struct.unpack(he + "d", raw[off:off + 8])[0]

    def i4(off):
        return struct.unpack(he + "i", raw[off:off + 4])[0]

    A = 256
    out: dict = {}
    if klass in (1, 2):
        out["xstart"] = f8(A + 0)
        out["xdelta"] = f8(A + 8)
        out["xunits"] = i4(A + 16)
        # xdelta is the sample interval, so its reciprocal is the rate — the
        # single most useful number in the file for anything downstream
        out["sample_rate"] = (1.0 / out["xdelta"]) if out["xdelta"] else 0.0
        if klass == 2:
            out["subsize"] = i4(A + 20)     # elements per frame / channels
            out["ystart"] = f8(A + 24)
            out["ydelta"] = f8(A + 32)
            out["yunits"] = i4(A + 40)      # 40, not 36 — one published
            #                                 table has this wrong, and it
            #                                 cannot be right: ydelta is a
            #                                 double at 32 and ends at 40.
    elif klass in (3, 6):
        out["rstart"] = f8(A + 0)
        out["rdelta"] = f8(A + 8)
        out["runits"] = i4(A + 16)
        out["subrecords"] = i4(A + 20)
        out["r2start"] = f8(A + 24)
        out["r2delta"] = f8(A + 32)
        out["r2units"] = i4(A + 40)
        out["record_length"] = i4(A + 44)
        subr = []
        for k in range(max(0, min(26, out["subrecords"]))):
            o = A + 48 + 8 * k
            subr.append({
                "name": raw[o:o + 4].decode("ascii", "replace").strip(
                    "\x00").strip(),
                "format": raw[o + 4:o + 6].decode("ascii", "replace").strip(),
                "offset": struct.unpack(he + "h", raw[o + 6:o + 8])[0]})
        # For class 6 this inline table is explicitly NOT authoritative — the
        # real one is the SUBREC_DEF extended keyword, parsed later, because
        # class 6 needs the extended header read before its adjunct means
        # anything. That inverts the usual order and is worth saying out loud.
        out["subr"] = subr
        out["subr_authoritative"] = (klass == 3)
    elif klass == 4:
        out["vrstart"] = f8(A + 0)
        out["vrdelta"] = f8(A + 8)
        out["vrunits"] = i4(A + 16)
        out["nrecords"] = i4(A + 20)
        out["vr2start"] = f8(A + 24)
        out["vr2delta"] = f8(A + 32)
        out["vr2units"] = i4(A + 40)
        out["vrecord_length"] = i4(A + 44)
    elif klass == 5:
        out["tstart"] = f8(A + 0)
        out["tdelta"] = f8(A + 8)
        out["tunits"] = i4(A + 16)
        out["components"] = i4(A + 20)
        out["t2start"] = f8(A + 24)
        out["t2delta"] = f8(A + 32)
        out["t2units"] = i4(A + 40)
        out["record_length"] = i4(A + 44)
        comp = []
        for k in range(max(0, min(14, out["components"]))):
            o = A + 48 + 8 * k
            comp.append({
                "name": raw[o:o + 4].decode("ascii", "replace").strip(
                    "\x00").strip(),
                "format": raw[o + 4:o + 6].decode("ascii", "replace").strip(),
                "type": struct.unpack(he + "b", raw[o + 6:o + 7])[0],
                "units": struct.unpack(he + "b", raw[o + 7:o + 8])[0]})
        out["comp"] = comp
        out["quadwords"] = _quadwords(raw, he)
    return out


def _quadwords(raw: bytes, he: str) -> dict:
    """The type-5000 platform block — 96 bytes at file offset **416**.

    **THIS IS THE STANDARD ANSWER TO "WHERE WAS THE ANTENNA AND HOW WAS IT
    POINTING".** It is a fixed-offset binary block in a header every BLUE
    reader already parses, it predates all of us, and it means a DF collection
    does not need a site-local keyword convention at all.

    `frame_of_ref` selects how the doubles are read; the same slots carry
    topocentric position and attitude, or an ECI epoch. The published
    documentation names the frames but not their literal spellings, so the
    string is passed through verbatim rather than mapped to a guess.
    """
    o = 256 + 160

    def f8(k):
        return struct.unpack(he + "d", raw[o + k:o + k + 8])[0]

    return {
        "frame_of_ref": raw[o:o + 8].decode("ascii", "replace").strip(
            "\x00").strip(),
        "altitude": f8(8), "latitude": f8(16), "longitude": f8(24),
        "azimuth": f8(32), "elevation": f8(40), "roll": f8(48),
        "epoch_year": f8(72), "epoch_seconds": f8(80),
        "hour_angle": f8(88),
    }


def _freq_from(kw: dict) -> dict:
    """Tuned frequency out of a keyword dict, whichever spelling was used.

    The BLUE header has no field for it, so it lives in keywords — and every
    tool spells it differently. Missing is normal, not an error.
    """
    for key in _FREQ_KEYS:
        if key in kw:
            try:
                return {"center_freq": float(kw[key])}
            except (TypeError, ValueError):
                continue
    return {}


def _parse_keywords(blob: bytes, length: int) -> dict:
    """The 92-byte inline keyword area: KEY=value pairs."""
    text = blob[:max(0, min(length, len(blob)))].decode("ascii", "replace")
    out = {}
    for part in text.replace("\x00", "\n").split("\n"):
        if "=" in part:
            k, v = part.split("=", 1)
            k = k.strip()
            if k:
                out[k] = v.strip()
    return out


def _decode_keyword_value(raw: bytes, code: str, endian: str):
    """One extended-header keyword value, decoded by its own type code.

    THE TYPE BYTE IS NOT DECORATION. An 8-byte double read as ASCII is eight
    bytes of mojibake, and that is what this module used to return for every
    binary keyword — which is every keyword in a file carrying a real
    ephemeris.
    """
    if code in ("A", "P") or code not in _TYPES:
        # ASCII in a KEYWORD is a plain byte string of the value's own
        # length. In the DATA section the same code means 8 bytes per
        # scalar. The two readings are genuinely different and both correct.
        return raw.decode("ascii", "replace").strip("\x00").strip()
    base, width = _TYPES[code]
    if width <= 0 or len(raw) < width:
        return raw.decode("ascii", "replace").strip("\x00").strip()
    arr = np.frombuffer(raw[: (len(raw) // width) * width],
                        dtype=np.dtype(endian + base))
    if arr.size == 1:
        return float(arr[0]) if base.startswith("f") else int(arr[0])
    return [float(v) if base.startswith("f") else int(v) for v in arr]


def _parse_ext_block(blob: bytes, endian: str) -> tuple:
    """`(pairs, style)` from an extended-header block.

    **THE TAG FOLLOWS THE VALUE.** One entry is

        +0  lkey   uint32  total length including padding, a multiple of 8
        +4  lext   uint16  everything that is NOT the value: 8 + ltag + pad
        +6  ltag   uint8   tag length
        +7  type   char    the value's format code
        +8  value  lkey − lext bytes
            tag    ltag bytes, immediately after the value
            pad    to lkey — NOT zeroed, never read it

    This module previously read the tag first and the value second, and
    decoded every value as ASCII regardless of type. Both are wrong, and
    together they meant that on any file carrying binary keywords the tags
    were garbage AND the values were garbage — silently, since a keyword dict
    full of mojibake looks like a file that simply had odd keywords.

    Confirmed against four independent implementations (REDHAWK's reader and
    its writer, LGS SigPlot, a Kaitai struct, Geontech's MATLAB reader) and
    against a real keyword test file whose ten entries tile its extended
    header exactly under this layout and cannot be made to tile under any
    other.

    All three lengths are read UNSIGNED. `ltag` is the one that matters: a
    tag of 128 bytes or more reads negative otherwise, and three of those
    four implementations have that bug.

    Returns PAIRS rather than a dict because duplicate tags are real — the
    reference file carries `STRING_TEST` twice with different values, and a
    dict silently keeps one of them.
    """
    pairs: list = []
    n = len(blob)
    if n < 8:
        return pairs, "empty"

    # The legacy layout put the tag FIRST, padded to a multiple of four, with
    # an ABSOLUTE offset to the next entry rather than a length, and every
    # value ASCII. Detected the way REDHAWK detects it: in the modern layout
    # these bytes carry the format character in the top byte, which makes the
    # word large.
    if struct.unpack(endian + "i", blob[4:8])[0] <= 128:
        off = 0
        while off + 12 <= n:
            nxt, ltag, ldata = struct.unpack(endian + "iii",
                                             blob[off:off + 12])
            if nxt <= off + 12 or nxt > n or ltag < 0 or ldata < 0:
                break
            itag = off + 12
            idata = itag + 4 * ((ltag + 3) // 4)
            if idata + ldata > n:
                break
            tag = blob[itag:itag + ltag].decode("ascii", "replace").strip()
            val = blob[idata:idata + ldata].decode(
                "ascii", "replace").strip("\x00").strip()
            if tag:
                pairs.append((tag, val))
            off = nxt
        return pairs, "legacy"

    off = 0
    while off + 8 <= n:
        lkey = struct.unpack(endian + "I", blob[off:off + 4])[0]
        lext = struct.unpack(endian + "H", blob[off + 4:off + 6])[0]
        ltag = blob[off + 6]
        code = chr(blob[off + 7]) if blob[off + 7] else "A"
        ldata = lkey - lext
        # Every guard here is load-bearing. A negative ldata spins this loop
        # forever, an lkey past the end walks off the block, and an lkey of
        # zero never advances.
        if lkey <= 8 or ldata < 0 or off + lkey > n:
            break
        if 8 + ldata + ltag > lkey:
            break
        raw = blob[off + 8: off + 8 + ldata]
        tag = blob[off + 8 + ldata: off + 8 + ldata + ltag].decode(
            "ascii", "replace").strip("\x00").strip()
        if tag:
            pairs.append((tag, _decode_keyword_value(raw, code, endian)))
        off += lkey
    return pairs, "standard"


def _extended_keywords(path: Path, hdr: dict) -> dict:
    """Extended header keywords, when present.

    Where a real capture keeps centre frequency — the header proper has no
    field for it, so `xdelta` gives the rate and this gives where in the
    spectrum it sat — and, on a collection platform, where a lot of tools
    keep the ephemeris. Missing or malformed extended headers are ignored
    rather than fatal: the samples are still readable and useful.

    Returns the pairs in order AND a last-wins dict, because callers want
    both: the dict for `kw["COL_RF"]`, the pairs for anything that repeats.
    """
    blank = {"ext_keywords": {}, "ext_keyword_pairs": []}
    if not hdr.get("ext_size") or hdr.get("ext_start", 0) <= 0:
        return {**blank, "ext_keyword_style": "absent"}
    start = int(hdr["ext_start"]) * 512
    # BLUE allows the extended header before OR after the data, so the
    # overlap check differs by which. A block overlapping the samples is a
    # corrupt header, and parsing it produces keywords made out of signal.
    d0 = float(hdr.get("data_start") or 0.0)
    d1 = d0 + float(hdr.get("data_size") or 0.0)
    if (d0 < start < d1) or (start <= d0 < start + int(hdr["ext_size"])):
        return {**blank, "ext_keyword_style": "overlaps the data — not read"}
    try:
        with path.open("rb") as fh:
            fh.seek(start)
            blob = fh.read(int(hdr["ext_size"]))
    except OSError:
        return {**blank, "ext_keyword_style": "unreadable"}
    try:
        pairs, style = _parse_ext_block(blob, hdr["header_endian"])
    except (struct.error, ValueError, IndexError):
        return {**blank, "ext_keyword_style": "malformed"}
    kw = {t: v for t, v in pairs}
    out = {"ext_keywords": kw, "ext_keyword_pairs": pairs,
           "ext_keyword_style": style}
    out.update(_freq_from(kw))
    return out


def read(path, max_samples: int | None = None, offset: int = 0):
    """(samples, header). Complex files come back complex64.

    ``max_samples`` bounds the read so a multi-gigabyte capture can be
    inspected without loading all of it — a pulse bench wants a slice, not
    the whole file.
    """
    hdr = read_header(path)
    if hdr["samples"] == 0:
        return np.zeros(0, dtype=np.complex64 if hdr["complex"]
                        else np.float32), hdr

    base, width = _TYPES[hdr["dtype_char"]]
    dtype = np.dtype(hdr["data_endian"] + base)
    per = hdr["values_per_sample"]
    want = hdr["samples"] if max_samples is None else min(hdr["samples"],
                                                          int(max_samples))
    start = int(hdr["data_start"]) + int(offset) * hdr["bytes_per_sample"]
    with Path(path).open("rb") as fh:
        fh.seek(start)
        raw = fh.read(want * hdr["bytes_per_sample"])
    values = np.frombuffer(raw, dtype=dtype)
    usable = (values.size // per) * per
    values = values[:usable]

    # RESCALE THE REAL COMPONENTS, NOT THE COMPLEX NUMBER. Raw counts become
    # ±1 so downstream DSP sees the amplitudes it would from any other
    # source, and `O` is UNSIGNED — its zero sits at mid-scale, so the offset
    # has to come off first. Doing that to an already-complex array subtracts
    # 128 from the REAL PART ONLY, which leaves Q sitting at +1.0 and puts a
    # constant quadrature term through everything downstream. Scaling before
    # the components are combined is the only place it is unambiguous.
    scaled = values.astype(np.float32)
    code = hdr["dtype_char"]
    if code in _SIGNED_INT:
        scaled = scaled / np.float32(2 ** (8 * hdr["bytes_per_value"] - 1))
    elif code == "O":
        scaled = (scaled - np.float32(128.0)) / np.float32(128.0)

    if hdr["complex"]:
        iq = scaled.reshape(-1, 2)
        return (iq[:, 0] + 1j * iq[:, 1]).astype(np.complex64), hdr
    if per > 1:
        scaled = scaled.reshape(-1, per)
    return scaled, hdr


def write(path, samples, sample_rate: float, center_freq: float = 0.0,
          timecode_unix: float = 0.0, ext_keywords=None) -> str:
    """Write complex samples as a type-1000 BLUE file ('CF', little-endian).

    Float32 rather than int16 on purpose: a pulse bench measures amplitudes,
    and re-quantising to integers on the way out would throw away precision
    for a saving nobody asked for.

    `ext_keywords` is `[(tag, value), …]` and is **where a position goes**.
    The inline field at offset 160 holds 92 bytes and is already spent on the
    centre frequency and sample rate; latitude, longitude, altitude, fix
    quality and satellite count do not fit and would silently truncate. Until
    this existed, type-1000 was the only writer in this module that could not
    carry an extended header at all — it built its own header inline and
    hardcoded `ext_start`/`ext_size` to zero — so a BLUE capture from ATK
    could carry a timestamp and never a place.
    """
    data = np.asarray(samples)
    is_complex = np.iscomplexobj(data)
    if is_complex:
        inter = np.empty(data.size * 2, dtype="<f4")
        inter[0::2] = data.real
        inter[1::2] = data.imag
        payload = inter.tobytes()
        fmt = b"CF"
        per = 2
    else:
        payload = np.asarray(data, dtype="<f4").tobytes()
        fmt = b"SF"
        per = 1

    ext = pack_keywords(ext_keywords or [])
    hdr = _blank_header(1000, fmt, len(payload), timecode_unix, ext)
    struct.pack_into("<d", hdr, 256, 0.0)              # xstart
    struct.pack_into("<d", hdr, 264,
                     1.0 / float(sample_rate) if sample_rate else 0.0)
    struct.pack_into("<i", hdr, 272, 1)                # xunits = seconds

    keys = f"CENTER_FREQ={center_freq:.0f}\nSAMPLE_RATE={sample_rate:.0f}"
    kb = keys.encode("ascii")[:92]
    struct.pack_into("<i", hdr, 160, len(kb))
    hdr[164:164 + len(kb)] = kb

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as fh:
        fh.write(bytes(hdr))
        fh.write(payload)
        if ext:
            # the extended block starts on the next 512-byte boundary after
            # the data, which is what `_blank_header` wrote into ext_start
            pad = (-(HEADER_SIZE + len(payload))) % 512
            fh.write(b"\x00" * pad)
            fh.write(ext)
    _ = per
    return str(p)


#: Extended-header keyword names for a platform fix, and the vocabulary
#: `siga.ephemeris.KEYWORD_MAP` already recognises on the way back in.
#:
#: **FIX QUALITY AND SATELLITE COUNT TRAVEL WITH THE COORDINATE.** Bill's own
#: note: *"a 2D fix with three satellites and a 12-satellite 3D fix are not
#: the same input to a direction-finding solution, and a recording that hides
#: the difference will produce a confident wrong bearing later."* They are
#: what `ephemeris.position_sigma_from` turns into a σ in metres, and that σ
#: is what stops the estimator treating the antenna as a mathematical point.
FIX_KEYWORDS = ("LAT", "LON", "ALT", "HEADING", "COURSE", "SPEED",
                "FIX_QUALITY", "SATELLITES", "FIX_EPOCH")


def fix_keywords(fix: dict, epoch: float = 0.0) -> list:
    """A GPS fix in `atk.core.gps`'s shape → extended-header pairs.

    Empty when there is no usable position, because writing a placeholder
    would be worse than writing nothing: a reader cannot tell an invented
    zero from a real one on the equator.
    """
    if not isinstance(fix, dict):
        return []
    try:
        lat, lon = float(fix["lat"]), float(fix["lon"])
    except (KeyError, TypeError, ValueError):
        return []
    out = [("LAT", lat), ("LON", lon)]
    for tag, key in (("ALT", "altitude_m"), ("COURSE", "course_deg"),
                     ("HEADING", "heading_deg")):
        v = fix.get(key)
        if v not in (None, ""):
            try:
                out.append((tag, float(v)))
            except (TypeError, ValueError):
                pass
    if fix.get("speed_knots") not in (None, ""):
        try:
            out.append(("SPEED", float(fix["speed_knots"]) * 0.514444))
        except (TypeError, ValueError):
            pass
    for tag, key in (("FIX_QUALITY", "quality"), ("SATELLITES", "satellites")):
        v = fix.get(key)
        if v not in (None, ""):
            try:
                out.append((tag, int(v)))
            except (TypeError, ValueError):
                pass
    t = epoch or fix.get("received_at")
    if t:
        try:
            out.append(("FIX_EPOCH", float(t)))
        except (TypeError, ValueError):
            pass
    return out


_PACK_CODE = {"f8": "D", "f4": "F", "i8": "X", "i4": "L", "i2": "I",
              "i1": "B", "u1": "O"}


def pack_keywords(pairs, endian: str = "<") -> bytes:
    """Build an extended-header block from `(tag, value)` pairs.

    The inverse of `_parse_ext_block`, and it exists for three reasons: the
    tests need to construct files whose bytes are known rather than trusting
    the reader against itself, ATK should be able to write an ephemeris back
    onto a capture, and a writer is the cheapest possible check that the
    reader's understanding of the layout is right — round-tripping a value
    through both is a claim about the FORMAT, not about one function.

    Types follow the Python value: float → D, int → L, str → A, and a numpy
    array keeps its own dtype. `lkey` is `8 + len(value) + len(tag)` rounded
    up to a multiple of eight.
    """
    out = bytearray()
    for tag, value in pairs:
        tb = str(tag).encode("ascii", "replace")
        if isinstance(value, (bytes, bytearray)):
            vb, code = bytes(value), "A"
        elif isinstance(value, str):
            vb, code = value.encode("ascii", "replace"), "A"
        elif isinstance(value, (np.ndarray, np.generic)):
            # numpy scalars are NOT instances of int or float — np.int8(1) is
            # neither — so they have to be caught before the Python branches
            # or every one of them falls through to str() and is written as
            # ASCII. Which is exactly the class of bug this module just fixed
            # on the reading side.
            arr = np.asarray(value)
            code = _PACK_CODE.get(arr.dtype.base.str[1:], "D")
            vb = arr.astype(arr.dtype.newbyteorder(endian)).tobytes()
        elif isinstance(value, bool):
            vb, code = struct.pack(endian + "i", int(value)), "L"
        elif isinstance(value, int):
            vb, code = struct.pack(endian + "i", value), "L"
        elif isinstance(value, float):
            vb, code = struct.pack(endian + "d", value), "D"
        elif isinstance(value, (list, tuple)):
            arr = np.asarray(value, dtype=endian + "f8")
            vb, code = arr.tobytes(), "D"
        else:
            vb, code = str(value).encode("ascii", "replace"), "A"
        lkey = -(-(8 + len(vb) + len(tb)) // 8) * 8
        lext = lkey - len(vb)
        out += struct.pack(endian + "IHBc", lkey, lext, len(tb),
                           code.encode("ascii"))
        out += vb + tb + b"\x00" * (lkey - 8 - len(vb) - len(tb))
    return bytes(out)


def record_dtype(hdr: dict) -> np.dtype:
    """A numpy structured dtype for a class 3 / 5 / 6 record.

    The whole column table becomes one dtype with explicit offsets and an
    explicit itemsize, so the records read as a zero-copy structured array
    rather than a Python loop over `struct.unpack`. Explicit offsets are not
    a nicety: class 3 columns carry their own byte offset and are allowed to
    leave gaps, and numpy's automatic packing would silently shift every
    column after the first gap.
    """
    cols = hdr.get("subr") or hdr.get("comp") or []
    if not cols:
        raise BlueError("this file describes no record columns")
    names, formats, offsets = [], [], []
    running = 0
    for c in cols:
        fmt = (c.get("format") or "SF").upper()
        mode = _MODES.get(fmt[0], 1) * int(c.get("num_elts", 1) or 1)
        code = fmt[1] if len(fmt) > 1 else "F"
        if code not in _TYPES:
            raise BlueError(f"column {c.get('name')!r} has an unsupported "
                            f"format {fmt!r}")
        base, width = _TYPES[code]
        off = c.get("offset")
        # Class 5 components have no offset field at all — they are packed
        # contiguously — so a running total is the only truth there.
        if off is None or off < 0:
            off = running
        name = (c.get("name") or f"col{len(names)}").strip().lower()
        while name in names:
            name += "_"
        names.append(name)
        if code == "A":
            formats.append(f"S{8 * mode}")
        else:
            formats.append(np.dtype(hdr["data_endian"] + base)
                           if mode == 1 else
                           (np.dtype(hdr["data_endian"] + base), mode))
        offsets.append(int(off))
        running = int(off) + width * mode
    itemsize = int(hdr.get("record_length") or running) or running
    if itemsize < running:
        raise BlueError(f"record_length {itemsize} is shorter than the "
                        f"{running} bytes its own columns describe")
    return np.dtype({"names": names, "formats": formats,
                     "offsets": offsets, "itemsize": itemsize})


def read_records(path, max_records: int | None = None, offset: int = 0):
    """`(structured array, header)` for a class 3 / 5 / 6 record file.

    This is how BLUE carries a TABLE — a state-vector track, a PDW list, an
    ephemeris — and it is the reason a DF collection can arrive as one file
    with its own ephemeris inside it rather than as a capture plus a
    spreadsheet somebody has to keep alongside it.
    """
    hdr = read_header(path)
    if hdr["class"] not in (3, 5, 6):
        raise BlueError(f"{Path(path).name}: type {hdr['type']} is not a "
                        f"record file (need class 3, 5 or 6)")
    dt = record_dtype(hdr)
    total = int(hdr.get("records") or 0)
    want = total if max_records is None else min(total, int(max_records))
    want = max(0, want - int(offset))
    if want <= 0:
        return np.zeros(0, dtype=dt), hdr
    with Path(path).open("rb") as fh:
        fh.seek(int(hdr["data_start"]) + int(offset) * dt.itemsize)
        raw = fh.read(want * dt.itemsize)
    usable = (len(raw) // dt.itemsize) * dt.itemsize
    return np.frombuffer(raw[:usable], dtype=dt), hdr


def read_keyword_records(path, max_records: int | None = None) -> list:
    """Class 4: one independent keyword set per record.

    Each record is an 8-byte prefix — block size, then valid size — followed
    by a keyword block in exactly the extended-header encoding. Which makes a
    type 4000 file the format's own answer to "a list of things, each with
    arbitrary named fields", and therefore the natural container for a PDW
    list or a per-intercept metadata track.
    """
    hdr = read_header(path)
    if hdr["class"] != 4:
        raise BlueError(f"{Path(path).name}: type {hdr['type']} is not a "
                        f"variable-record file (need class 4)")
    he = hdr["header_endian"]
    out: list = []
    end = int(hdr["data_start"]) + int(hdr["data_size"])
    with Path(path).open("rb") as fh:
        fh.seek(int(hdr["data_start"]))
        while fh.tell() + 8 <= end:
            if max_records is not None and len(out) >= int(max_records):
                break
            prefix = fh.read(8)
            if len(prefix) < 8:
                break
            block, valid = struct.unpack(he + "ii", prefix)
            if valid < 0 or block < valid or block <= 0:
                break
            pairs, _style = _parse_ext_block(fh.read(valid), he)
            out.append(dict(pairs))
            if block > valid:
                fh.seek(block - valid, 1)
    return out


def write_channels(path, channels, sample_rate: float,
                   center_freq: float = 0.0, timecode_unix: float = 0.0,
                   ext_keywords=None, keywords: str = "") -> str:
    """Write a multi-channel coherent capture as a type-2000 BLUE file.

    `channels` is shaped (elements, samples). The elements are interleaved
    sample by sample, which is what `subsize` describes and what every reader
    expects — an array file written channel-after-channel instead is a file
    whose bearings are wrong in a structured, plausible-looking way.

    The frame interval goes in `ydelta` (the frame IS one time step across the
    elements) and `xdelta` carries the same value, because producers disagree
    about which axis is time for this case and writing both removes the
    guess for whoever reads it next.
    """
    a = np.asarray(channels)
    if a.ndim != 2:
        raise BlueError("channels must be (elements, samples)")
    n_el, n = a.shape
    inter = np.empty(n * n_el * 2, dtype="<f4")
    inter[0::2] = np.real(a).T.reshape(-1)
    inter[1::2] = np.imag(a).T.reshape(-1)
    payload = inter.tobytes()
    ext = pack_keywords(ext_keywords or [])
    hdr = _blank_header(2000, b"CF", len(payload), timecode_unix, ext)
    dt = 1.0 / float(sample_rate) if sample_rate else 0.0
    A = 256
    struct.pack_into("<d", hdr, A + 0, 0.0)          # xstart
    struct.pack_into("<d", hdr, A + 8, dt)           # xdelta
    struct.pack_into("<i", hdr, A + 16, 1)           # xunits = seconds
    struct.pack_into("<i", hdr, A + 20, int(n_el))   # subsize = elements
    struct.pack_into("<d", hdr, A + 24, 0.0)         # ystart
    struct.pack_into("<d", hdr, A + 32, dt)          # ydelta = frame interval
    struct.pack_into("<i", hdr, A + 40, 1)           # yunits = seconds
    kb = (keywords or
          f"CENTER_FREQ={center_freq:.0f}\nSAMPLE_RATE={sample_rate:.0f}"
          ).encode("ascii")[:92]
    struct.pack_into("<i", hdr, 160, len(kb))
    hdr[164:164 + len(kb)] = kb
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as fh:
        fh.write(bytes(hdr))
        fh.write(payload)
        if ext:
            fh.write(b"\x00" * (-len(payload) % 512))
            fh.write(ext)
    return str(p)


def _blank_header(type_no: int, fmt: bytes, payload_len: int,
                  timecode_unix: float = 0.0, ext: bytes = b"") -> bytearray:
    """A little-endian 512-byte header with the common fields filled in."""
    hdr = bytearray(b"\x00" * HEADER_SIZE)
    hdr[0:4] = b"BLUE"
    hdr[4:8] = b"EEEI"
    hdr[8:12] = b"EEEI"
    data_start = float(HEADER_SIZE)
    if ext:
        # The extended header goes AFTER the data, on the next 512-byte
        # boundary, because ext_start is counted in blocks and the data is
        # not required to end on one.
        blocks = -(-(int(data_start) + payload_len) // 512)
        struct.pack_into("<i", hdr, 24, blocks)
        struct.pack_into("<i", hdr, 28, len(ext))
    struct.pack_into("<d", hdr, 32, data_start)
    struct.pack_into("<d", hdr, 40, float(payload_len))
    struct.pack_into("<i", hdr, 48, int(type_no))
    hdr[52:54] = fmt
    tc = (timecode_unix + J1950_TO_UNIX) if timecode_unix else 0.0
    struct.pack_into("<d", hdr, 56, float(tc))
    return hdr


def write_records(path, columns: list, rows, type_no: int = 3000,
                  rstart: float = 0.0, rdelta: float = 1.0,
                  runits: int = 1, timecode_unix: float = 0.0,
                  ext_keywords=None, quadwords: dict | None = None) -> str:
    """Write a class 3 or class 5 record file.

    `columns` is `[{"name", "format", "units"?, "type"?}, …]` and `rows` is a
    sequence of dicts or tuples. This is how a table goes into BLUE — a
    state-vector track, a PDW list — and `write_state_vectors` below is the
    convenience wrapper for the case this bench actually needs.
    """
    cols, offsets, running = [], [], 0
    for c in columns:
        fmt = str(c.get("format", "SD")).upper()
        mode = _MODES.get(fmt[0], 1)
        code = fmt[1] if len(fmt) > 1 else "D"
        if code not in _TYPES:
            raise BlueError(f"unsupported column format {fmt!r}")
        base, width = _TYPES[code]
        offsets.append(running)
        cols.append({**c, "format": fmt, "mode": mode, "base": base,
                     "width": width})
        running += width * mode
    record_length = running

    klass = int(type_no) // 1000
    if klass not in (3, 5):
        raise BlueError("write_records handles class 3 and class 5 only")

    buf = bytearray()
    for row in rows:
        rec = bytearray(b"\x00" * record_length)
        for c, off in zip(cols, offsets):
            v = (row.get(c["name"]) if isinstance(row, dict)
                 else row[cols.index(c)])
            if v is None:
                continue
            if c["format"][1] == "A":
                s = str(v).encode("ascii", "replace")[: c["width"] * c["mode"]]
                rec[off:off + len(s)] = s
            else:
                arr = np.asarray(v, dtype="<" + c["base"]).ravel()
                if arr.size < c["mode"]:
                    arr = np.concatenate(
                        [arr, np.zeros(c["mode"] - arr.size,
                                       dtype="<" + c["base"])])
                rec[off:off + c["width"] * c["mode"]] = \
                    arr[: c["mode"]].tobytes()
        buf += rec

    ext = pack_keywords(ext_keywords or [])
    hdr = _blank_header(type_no, b"NH", len(buf), timecode_unix, ext)
    A = 256
    struct.pack_into("<d", hdr, A + 0, float(rstart))
    struct.pack_into("<d", hdr, A + 8, float(rdelta))
    struct.pack_into("<i", hdr, A + 16, int(runits))
    struct.pack_into("<i", hdr, A + 20, len(cols))
    struct.pack_into("<i", hdr, A + 44, record_length)
    limit = 26 if klass == 3 else 14
    if len(cols) > limit:
        raise BlueError(f"a class-{klass} record holds at most {limit} "
                        f"columns in its adjunct; {len(cols)} were given")
    for k, (c, off) in enumerate(zip(cols[:limit], offsets)):
        o = A + 48 + 8 * k
        name = str(c["name"]).upper()
        # FOUR CHARACTERS, AND SILENTLY TRUNCATING IS NOT ACCEPTABLE. A
        # column written as SIGMA comes back as SIGM, every reader looking
        # for SIGMA misses it, and the data is simply gone with no error
        # anywhere. (Class 6 exists partly to lift this to 24 characters,
        # keeping its real column table in a SUBREC_DEF keyword instead.)
        if len(name) > 4:
            raise BlueError(
                f"class-{klass} column names are four characters and "
                f"{name!r} is {len(name)} — truncating it to {name[:4]!r} "
                f"would lose the column silently. Shorten the name, or use a "
                f"type-6000 file, whose columns may be 24 characters.")
        hdr[o:o + 4] = name.encode("ascii", "replace").ljust(4, b" ")
        hdr[o + 4:o + 6] = c["format"].encode("ascii")[:2]
        if klass == 3:
            struct.pack_into("<h", hdr, o + 6, off)
        else:
            struct.pack_into("<b", hdr, o + 6, int(c.get("type", 0)))
            struct.pack_into("<b", hdr, o + 7, int(c.get("units", 0)))
    if klass == 5 and quadwords:
        q = A + 160
        hdr[q:q + 8] = str(quadwords.get("frame_of_ref", "")).encode(
            "ascii", "replace")[:8].ljust(8, b" ")
        for k, key in ((8, "altitude"), (16, "latitude"), (24, "longitude"),
                       (32, "azimuth"), (40, "elevation"), (48, "roll"),
                       (72, "epoch_year"), (80, "epoch_seconds"),
                       (88, "hour_angle")):
            struct.pack_into("<d", hdr, q + k,
                             float(quadwords.get(key, 0.0) or 0.0))

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as fh:
        fh.write(bytes(hdr))
        fh.write(bytes(buf))
        if ext:
            fh.write(b"\x00" * (-len(buf) % 512))
            fh.write(ext)
    return str(p)


#: The columns of a standard state-vector file. `POS` is a 3-vector of
#: doubles and its GEODETIC ordering is (altitude, latitude, longitude).
STATE_VECTOR_COLUMNS = [
    {"name": "TIME", "format": "SD", "units": 1},
    {"name": "POS", "format": "VD", "type": 6, "units": 63},
    {"name": "VEL", "format": "VD", "type": 2, "units": 6},
    {"name": "HDG", "format": "SF", "units": 34},
    {"name": "ELPS", "format": "VD", "type": 1, "units": 34},
]


def write_state_vectors(path, states, timecode_unix: float = 0.0,
                        frame_of_ref: str = "TOPO") -> str:
    """Write an ephemeris track as a type-5001 state-vector file.

    `states` is a sequence of dicts with `epoch` (Unix seconds), `lat`, `lon`,
    `alt_m`, and optionally `heading_deg`, `pitch_deg`, `roll_deg` and a
    velocity triple. ATK can therefore hand a collection back with its
    ephemeris attached in the format the rest of the world already reads,
    rather than in a sidecar of its own invention.
    """
    rows = []
    for s in states:
        rows.append({
            "TIME": float(s.get("epoch", 0.0)) + J1950_TO_UNIX,
            "POS": [float(s.get("alt_m", 0.0)), float(s.get("lat", 0.0)),
                    float(s.get("lon", 0.0))],
            "VEL": [float(s.get("vx", 0.0)), float(s.get("vy", 0.0)),
                    float(s.get("vz", 0.0))],
            "HDG": float(s.get("heading_deg", 0.0)),
            "ELPS": [float(s.get("heading_deg", 0.0)),
                     float(s.get("pitch_deg", 0.0)),
                     float(s.get("roll_deg", 0.0))],
        })
    return write_records(path, STATE_VECTOR_COLUMNS, rows, type_no=5001,
                         runits=1, timecode_unix=timecode_unix,
                         quadwords={"frame_of_ref": frame_of_ref})


def looks_like_blue(path) -> bool:
    """Cheap sniff so a file picker can route by content, not extension.

    BLUE files use .tmp, .prm, .blue, .cdf and plenty besides, so extension is
    not a reliable signal.
    """
    try:
        with Path(path).open("rb") as fh:
            return fh.read(4) == b"BLUE"
    except OSError:
        return False


def describe(hdr: dict) -> str:
    """One line an operator can read."""
    import time
    bits = [f"BLUE type {hdr.get('type')}", hdr.get("format", "??")]
    if hdr.get("sample_rate"):
        bits.append(f"{hdr['sample_rate'] / 1e6:.4g} Msps")
    if hdr.get("center_freq"):
        bits.append(f"@ {hdr['center_freq'] / 1e6:.4g} MHz")
    if hdr.get("class") in (3, 4, 5, 6):
        cols = hdr.get("subr") or hdr.get("comp") or []
        bits.append(f"{hdr.get('records', hdr.get('nrecords', 0)):,} records"
                    + (f" × {len(cols)} columns" if cols else ""))
        if cols:
            bits.append("[" + ", ".join(c["name"] for c in cols[:8]) + "]")
    else:
        bits.append(f"{hdr.get('samples', 0):,} samples")
    if hdr.get("subsize", 0) > 1:
        bits.append(f"{hdr['subsize']} channels")
    if hdr.get("timecode_unix"):
        stamp = time.strftime("%Y-%m-%d %H:%M:%SZ",
                              time.gmtime(hdr["timecode_unix"]))
        if hdr.get("timecode_precision_s"):
            stamp += f" (+{hdr['timecode_precision_s']:.3g} s from TC_PREC)"
        bits.append(stamp)
    q = hdr.get("quadwords") or {}
    if q.get("latitude") or q.get("longitude"):
        bits.append(f"platform {q['latitude']:.5f}, {q['longitude']:.5f} "
                    f"@ {q.get('altitude', 0.0):,.0f} m "
                    f"[{q.get('frame_of_ref') or 'frame unstated'}]")
    if hdr.get("data_endian") != hdr.get("header_endian"):
        bits.append("header/data endianness DIFFER (handled)")
    style = hdr.get("ext_keyword_style")
    if style and style not in ("absent", "standard"):
        bits.append(f"extended keywords: {style}")
    return " · ".join(bits)
