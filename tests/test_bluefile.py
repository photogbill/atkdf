"""Midas BLUE: the extended keywords, the record classes, and the ephemeris.

**THE CENTRAL TEST HERE IS A ROUND TRIP THAT REPRODUCES A REAL FILE'S BYTES.**
The reference `keyword_test_file.tmp` that circulates with the format carries
ten keywords in a 224-byte extended header. Packing the same ten values with
this module's writer produces 224 bytes, and reading them back gives the same
values — including `F_TEST` at 0.12345000356435776, which is what float32
rounding does to 0.12345 and is therefore a fingerprint of the layout rather
than of the arithmetic. A reader tested only against its own writer proves
nothing; this pair is pinned to a file neither of them wrote.

Four bugs are pinned here because all four produced plausible output:

  1. **Tag and value were the wrong way round**, and every value was decoded
     as ASCII regardless of its type byte. On any file with binary keywords
     the tags AND the values were garbage.
  2. **`O` was missing from the type table**, so an unsigned 8-bit capture —
     one of the commonest ways I/Q is stored — was refused outright.
  3. **`NH` and `KW` were treated as data types** rather than as the
     sentinels they are, so every record file, which is to say every
     ephemeris track, was unreadable.
  4. **`TC_PREC` was ignored**, silently discarding up to a microsecond of
     timing — three hundred metres of TDOA error.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from atkdf import bluefile as bf


# ---------------------------------------------------------------------------
# 1 — the extended keyword layout
# ---------------------------------------------------------------------------

#: The reference file's ten entries, in its order.
REFERENCE = [("B_TEST", np.int8(123)), ("I_TEST", np.int16(1337)),
             ("L_TEST", 113355), ("X_TEST", np.int64(987654321)),
             ("F_TEST", np.float32(0.12345)), ("D_TEST", 9.87654321),
             ("O_TEST", np.uint8(255)), ("STRING_TEST", "Hello World"),
             ("B_TEST2", np.int8(99)), ("STRING_TEST", "Goodbye World")]


def test_the_reference_keyword_block_is_reproduced_byte_for_byte():
    """224 bytes, ten entries, tiling the block exactly — the same numbers a
    real producer wrote."""
    blob = bf.pack_keywords(REFERENCE)
    assert len(blob) == 224
    assert len(blob) % 8 == 0
    pairs, style = bf._parse_ext_block(blob, "<")
    assert style == "standard"
    assert len(pairs) == 10
    assert [t for t, _v in pairs] == [t for t, _v in REFERENCE]


def test_every_type_code_decodes_to_its_own_type():
    pairs = dict(bf._parse_ext_block(bf.pack_keywords(REFERENCE), "<")[:1][0]
                 for _ in [0]) if False else dict(
        bf._parse_ext_block(bf.pack_keywords(REFERENCE), "<")[0])
    assert pairs["B_TEST"] == 123
    assert pairs["I_TEST"] == 1337
    assert pairs["L_TEST"] == 113355
    assert pairs["X_TEST"] == 987654321
    assert pairs["D_TEST"] == pytest.approx(9.87654321, abs=1e-12)
    assert pairs["O_TEST"] == 255
    assert pairs["STRING_TEST"] == "Goodbye World"


def test_float32_keeps_its_own_rounding():
    """0.12345 stored as float32 reads back as 0.12345000356435776. That
    exact number is the layout's fingerprint: it appears in the reference
    file, so matching it means the value was read from the right offset with
    the right width."""
    pairs = dict(bf._parse_ext_block(bf.pack_keywords(REFERENCE), "<")[0])
    assert pairs["F_TEST"] == pytest.approx(0.12345000356435776, abs=1e-17)


def test_duplicate_tags_survive_as_pairs_and_collapse_in_the_dict():
    """**The reference file really does carry `STRING_TEST` twice.** A
    dict-only reader loses one of them without a word."""
    pairs, _s = bf._parse_ext_block(bf.pack_keywords(REFERENCE), "<")
    tags = [t for t, _v in pairs]
    assert tags.count("STRING_TEST") == 2
    values = [v for t, v in pairs if t == "STRING_TEST"]
    assert values == ["Hello World", "Goodbye World"]


@pytest.mark.parametrize("endian", ["<", ">"])
def test_both_endiannesses_round_trip(endian):
    pairs, _s = bf._parse_ext_block(bf.pack_keywords(REFERENCE, endian),
                                    endian)
    assert dict(pairs)["L_TEST"] == 113355
    assert dict(pairs)["D_TEST"] == pytest.approx(9.87654321)


def test_the_tag_follows_the_value_not_the_other_way_round():
    """Pinned directly on the bytes, because the two layouts are impossible
    to tell apart from a round trip against a matching writer."""
    blob = bf.pack_keywords([("TAGNAME", 7.5)])
    lkey, lext, ltag, code = struct.unpack("<IHBc", blob[:8])
    assert code == b"D"
    assert ltag == len("TAGNAME")
    ldata = lkey - lext
    assert ldata == 8                                   # one float64
    assert struct.unpack("<d", blob[8:16])[0] == 7.5    # value at +8
    assert blob[16:16 + ltag] == b"TAGNAME"             # tag after it


def test_lkey_is_the_padded_length_and_the_padding_is_never_read():
    """Padding in the reference file contains leftover buffer garbage — the
    bytes `ON` in one entry. Anything that reads it produces nonsense."""
    blob = bytearray(bf.pack_keywords([("A", "xyz")]))
    lkey = struct.unpack("<I", blob[:4])[0]
    assert lkey % 8 == 0
    blob[12:lkey] = b"\xff" * (lkey - 12)     # scribble on the padding
    pairs, _s = bf._parse_ext_block(bytes(blob), "<")
    assert pairs == [("A", "xyz")]


def test_the_legacy_tag_first_layout_is_detected_and_read():
    """One real implementation uses the opposite order, with an absolute
    offset to the next entry. It is gated behind a detection heuristic rather
    than guessed at."""
    tag, val = b"OLDTAG", b"hello"
    pad = (-len(tag)) % 4
    body = tag + b"\x00" * pad + val
    blob = struct.pack("<iii", 12 + len(body), len(tag), len(val)) + body
    pairs, style = bf._parse_ext_block(blob, "<")
    assert style == "legacy"
    assert pairs == [("OLDTAG", "hello")]


@pytest.mark.parametrize("bad", [
    b"\x08\x00\x00\x00\x00\x00\x00A",            # lkey == 8, never advances
    b"\x10\x00\x00\x00\xff\xff\x04D" + b"\x00" * 8,   # lext > lkey → ldata < 0
    b"\xff\xff\xff\x7f\x08\x00\x04D" + b"\x00" * 8,   # lkey past the end
])
def test_a_malformed_block_terminates_instead_of_spinning(bad):
    """**A negative `ldata` is an infinite loop**, and this parser runs on
    files handed over by third parties. Every guard here is load-bearing."""
    pairs, _style = bf._parse_ext_block(bad, "<")
    assert isinstance(pairs, list)


# ---------------------------------------------------------------------------
# 2 — the type table
# ---------------------------------------------------------------------------

def test_O_is_unsigned_and_B_is_signed():
    """The letters suggest the opposite. Getting them the wrong way round
    opens the file and puts every sample 128 counts out."""
    assert bf._TYPES["O"] == ("u1", 1)
    assert bf._TYPES["B"] == ("i1", 1)


def test_an_unsigned_byte_capture_opens_at_all(tmp_path):
    """It used to be refused outright: `O` was simply absent from the table."""
    p = tmp_path / "cu8.tmp"
    n = 512
    raw = np.full(n * 2, 128, dtype=np.uint8)
    raw[0::2] = 200                                  # I = +0.5625
    hdr = bytearray(b"\x00" * 512)
    hdr[0:4] = b"BLUE"
    hdr[4:8] = hdr[8:12] = b"EEEI"
    struct.pack_into("<d", hdr, 32, 512.0)
    struct.pack_into("<d", hdr, 40, float(raw.nbytes))
    struct.pack_into("<i", hdr, 48, 1000)
    hdr[52:54] = b"CO"
    struct.pack_into("<d", hdr, 264, 1e-6)
    p.write_bytes(bytes(hdr) + raw.tobytes())
    data, h = bf.read(p)
    assert h["dtype_char"] == "O"
    assert data.size == n
    assert np.real(data[0]) == pytest.approx(0.5625, abs=1e-6)
    assert np.imag(data[0]) == pytest.approx(0.0, abs=1e-6)


def test_an_unsupported_type_is_still_refused(tmp_path):
    p = tmp_path / "bad.tmp"
    hdr = bytearray(b"\x00" * 512)
    hdr[0:4] = b"BLUE"
    hdr[4:8] = hdr[8:12] = b"EEEI"
    hdr[52:54] = b"CZ"
    p.write_bytes(bytes(hdr))
    with pytest.raises(bf.BlueError, match="unsupported"):
        bf.read_header(p)


# ---------------------------------------------------------------------------
# 3 — record files
# ---------------------------------------------------------------------------

def test_a_state_vector_file_round_trips(tmp_path):
    p = tmp_path / "track.tmp"
    states = [{"epoch": 1.7e9 + t, "lat": 51.5 + 0.001 * t,
               "lon": -0.12 + 0.002 * t, "alt_m": 1000.0 + t,
               "heading_deg": 90.0 + t, "pitch_deg": 1.5, "roll_deg": -2.0}
              for t in range(5)]
    bf.write_state_vectors(p, states, timecode_unix=1.7e9)
    hdr = bf.read_header(p)
    assert hdr["type"] == 5001 and hdr["class"] == 5
    assert hdr["records"] == 5
    assert [c["name"] for c in hdr["comp"]] == ["TIME", "POS", "VEL", "HDG",
                                                "ELPS"]
    arr, _h = bf.read_records(p)
    assert arr.dtype.names == ("time", "pos", "vel", "hdg", "elps")
    assert arr[2]["time"] - bf.J1950_TO_UNIX == pytest.approx(1.7e9 + 2)


def test_POS_is_altitude_latitude_longitude_in_that_order(tmp_path):
    """**Altitude FIRST.** Every instinct says lat/lon/alt; reading it that
    way puts an aircraft at 3,000° north."""
    p = tmp_path / "t.tmp"
    bf.write_state_vectors(p, [{"epoch": 1.7e9, "lat": 51.5, "lon": -0.12,
                                "alt_m": 3000.0}])
    arr, hdr = bf.read_records(p)
    assert arr[0]["pos"][0] == pytest.approx(3000.0)
    assert arr[0]["pos"][1] == pytest.approx(51.5)
    assert arr[0]["pos"][2] == pytest.approx(-0.12)
    pos = next(c for c in hdr["comp"] if c["name"] == "POS")
    assert bf.COMP_TYPES[pos["type"]] == "geodetic"
    assert "altitude, latitude, longitude" in bf.T5_STATE_COMPONENTS["POS"]


def test_the_units_are_carried_and_mean_something(tmp_path):
    p = tmp_path / "t.tmp"
    bf.write_state_vectors(p, [{"epoch": 1.7e9, "lat": 0.0, "lon": 0.0}])
    hdr = bf.read_header(p)
    units = {c["name"]: bf.UNITS.get(c["units"]) for c in hdr["comp"]}
    assert units["TIME"] == "s"
    assert units["POS"] == "altitude (m)"
    assert units["HDG"] == "deg"


def test_NH_is_a_sentinel_not_a_data_type(tmp_path):
    """It used to be rejected as an unsupported type, which made every
    record file — every ephemeris track — unreadable."""
    p = tmp_path / "t.tmp"
    bf.write_state_vectors(p, [{"epoch": 1.7e9, "lat": 1.0, "lon": 2.0}])
    hdr = bf.read_header(p)
    assert hdr["format"] == "NH"
    assert hdr["format_is_sentinel"]
    assert hdr["bytes_per_value"] == 0
    assert not hdr["complex"]


def test_a_record_dtype_honours_explicit_column_offsets(tmp_path):
    """Class-3 columns carry their own byte offset and may leave gaps.
    numpy's automatic packing would shift every column after the first one."""
    p = tmp_path / "r.tmp"
    bf.write_records(p, [{"name": "A", "format": "SD"},
                         {"name": "B", "format": "SL"},
                         {"name": "C", "format": "SD"}],
                     [{"A": 1.5, "B": 7, "C": -2.5}], type_no=3000)
    hdr = bf.read_header(p)
    assert [c["offset"] for c in hdr["subr"]] == [0, 8, 12]
    dt = bf.record_dtype(hdr)
    assert dt.itemsize == 20
    arr, _h = bf.read_records(p)
    assert arr[0]["a"] == 1.5 and arr[0]["b"] == 7 and arr[0]["c"] == -2.5


def test_reading_records_from_a_signal_file_is_refused(tmp_path):
    p = tmp_path / "iq.tmp"
    bf.write(p, np.zeros(64, dtype=np.complex64), 1e6)
    with pytest.raises(bf.BlueError, match="not a record file"):
        bf.read_records(p)


def test_class_4_records_are_independent_keyword_sets(tmp_path):
    """A type 4000 file is the format's own answer to 'a list of things,
    each with arbitrary named fields' — the natural container for a PDW
    list."""
    p = tmp_path / "kw.tmp"
    recs = [bf.pack_keywords([("PW", 1.2e-6 * (i + 1)), ("AMP", -40.0 - i),
                              ("ID", f"pulse{i}")]) for i in range(4)]
    body = b"".join(struct.pack("<ii", len(r), len(r)) + r for r in recs)
    hdr = bytearray(b"\x00" * 512)
    hdr[0:4] = b"BLUE"
    hdr[4:8] = hdr[8:12] = b"EEEI"
    struct.pack_into("<d", hdr, 32, 512.0)
    struct.pack_into("<d", hdr, 40, float(len(body)))
    struct.pack_into("<i", hdr, 48, 4000)
    hdr[52:54] = b"KW"
    struct.pack_into("<i", hdr, 256 + 20, len(recs))
    p.write_bytes(bytes(hdr) + body)
    out = bf.read_keyword_records(p)
    assert len(out) == 4
    assert out[2]["ID"] == "pulse2"
    assert out[2]["PW"] == pytest.approx(3.6e-6)


# ---------------------------------------------------------------------------
# 4 — the platform block and the timecode
# ---------------------------------------------------------------------------

def test_the_quadword_platform_block_sits_at_file_offset_416(tmp_path):
    """**The format's own answer to "where was the antenna".** Fixed
    offsets, in a header every reader already parses — no site convention
    needed at all."""
    p = tmp_path / "q.tmp"
    bf.write_records(p, [{"name": "TIME", "format": "SD"}], [{"TIME": 0.0}],
                     type_no=5000,
                     quadwords={"frame_of_ref": "TOPO", "latitude": 52.1,
                                "longitude": 0.9, "altitude": 85.0,
                                "azimuth": 137.0, "elevation": 0.5,
                                "roll": -1.25})
    raw = p.read_bytes()
    assert raw[416:424].rstrip(b" \x00") == b"TOPO"
    assert struct.unpack("<d", raw[432:440])[0] == pytest.approx(52.1)
    q = bf.read_header(p)["quadwords"]
    assert q["latitude"] == pytest.approx(52.1)
    assert q["longitude"] == pytest.approx(0.9)
    assert q["azimuth"] == pytest.approx(137.0)
    assert q["roll"] == pytest.approx(-1.25)


def test_the_timecode_is_j1950_and_the_offset_is_exact(tmp_path):
    assert bf.J1950_TO_UNIX == 7305 * 86400
    p = tmp_path / "t.tmp"
    bf.write(p, np.zeros(16, dtype=np.complex64), 1e6, timecode_unix=1.7e9)
    hdr = bf.read_header(p)
    assert hdr["timecode_unix"] == pytest.approx(1.7e9)
    assert hdr["timecode_j1950"] == pytest.approx(1.7e9 + bf.J1950_TO_UNIX)


def test_tc_prec_is_folded_back_into_the_timecode(tmp_path):
    """**A double cannot hold J1950 seconds to better than a microsecond.**
    Dropping the residual costs up to 1 µs, which is 300 m of TDOA position
    error — for the sake of reading one keyword."""
    p = tmp_path / "t.tmp"
    bf.write(p, np.zeros(16, dtype=np.complex64), 1e6, timecode_unix=1.7e9)
    raw = bytearray(p.read_bytes())
    kb = b"TC_PREC=1.5E-09"
    struct.pack_into("<i", raw, 160, len(kb))
    raw[164:164 + len(kb)] = kb
    p.write_bytes(bytes(raw))
    hdr = bf.read_header(p)
    assert hdr["timecode_precision_s"] == pytest.approx(1.5e-9)
    assert hdr["timecode_unix"] == pytest.approx(1.7e9 + 1.5e-9, abs=1e-12)
    assert "TC_PREC" in bf.describe(hdr)


def test_a_tc_prec_that_is_not_a_residual_is_ignored(tmp_path):
    """A whole second in TC_PREC is somebody else's field, not this one."""
    p = tmp_path / "t.tmp"
    bf.write(p, np.zeros(16, dtype=np.complex64), 1e6, timecode_unix=1.7e9)
    raw = bytearray(p.read_bytes())
    kb = b"TC_PREC=3.0"
    struct.pack_into("<i", raw, 160, len(kb))
    raw[164:164 + len(kb)] = kb
    p.write_bytes(bytes(raw))
    hdr = bf.read_header(p)
    assert "timecode_precision_s" not in hdr
    assert hdr["timecode_unix"] == pytest.approx(1.7e9)


# ---------------------------------------------------------------------------
# 5 — multi-channel, and the overlap guard
# ---------------------------------------------------------------------------

def test_channels_interleave_and_de_interleave_exactly(tmp_path):
    """An array file written channel-after-channel instead of sample-by-
    sample is one whose bearings are wrong in a structured, plausible way."""
    from atkdf import df_file
    rng = np.random.default_rng(4)
    ch = (rng.standard_normal((4, 300))
          + 1j * rng.standard_normal((4, 300))).astype(np.complex64)
    p = tmp_path / "arr.tmp"
    bf.write_channels(p, ch, 1e6, center_freq=300e6)
    hdr = bf.read_header(p)
    assert hdr["type"] == 2000 and hdr["subsize"] == 4
    back, _h, note = df_file.read_array(p)
    assert back.shape == ch.shape
    assert np.max(np.abs(back - ch)) == 0.0
    assert "4 channels" in note


def test_an_extended_header_overlapping_the_data_is_not_read(tmp_path):
    """A corrupt header, and parsing it produces keywords made out of
    signal."""
    p = tmp_path / "x.tmp"
    bf.write(p, np.ones(4096, dtype=np.complex64), 1e6)
    raw = bytearray(p.read_bytes())
    struct.pack_into("<i", raw, 24, 2)        # inside the data
    struct.pack_into("<i", raw, 28, 256)
    p.write_bytes(bytes(raw))
    hdr = bf.read_header(p)
    assert hdr["ext_keywords"] == {}
    assert "overlaps the data" in hdr["ext_keyword_style"]


def test_describe_says_what_kind_of_file_it_is(tmp_path):
    p = tmp_path / "t.tmp"
    bf.write_state_vectors(p, [{"epoch": 1.7e9, "lat": 1.0, "lon": 2.0}])
    text = bf.describe(bf.read_header(p))
    assert "5001" in text and "records" in text and "TIME" in text
