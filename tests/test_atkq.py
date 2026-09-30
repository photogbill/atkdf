"""atkdf reads atkdaq's wire format, and reads it the way the header file says.

The offsets are checked against `atkdaq/include/atkdaq_frame.h` by eye in
`atkq._FIELDS`; here they are checked against the fixture atkdaq itself wrote
(`tests/fixtures/synth_240k.atkq`). That file is atkdaq's writer; this module
is atkdf's independent reader; the two meeting on the same bytes is the whole
point of the arm's-length, no-GPL-import arrangement.
"""

from __future__ import annotations

import io
import struct

import numpy as np
import pytest

from atkdf import atkq
from conftest import FIXTURES

FIXTURE = FIXTURES / "synth_240k.atkq"


def test_the_documented_offsets_are_what_the_reader_uses():
    off = {name: o for name, _c, o in atkq._FIELDS}
    assert off["t_utc_ns"] == 24 and off["fc_hz"] == 32 and off["fs_hz"] == 40
    assert off["spread_cdeg"] == 60 and off["gain_tenths_db"] == 64
    assert off["payload_bytes"] == 68 and off["stream_sample0"] == 72
    assert off["seg_sample0"] == 80 and off["hdr_crc32"] == 88
    ch = {name: o for name, _c, o in atkq._CH_FIELDS}
    assert ch["sample_count"] == 24 and ch["frac_delay"] == 4


def test_pack_then_parse_round_trips():
    chans = tuple(atkq.Channel(delay=i * 7 - 3, frac_delay=0.125 * i,
                              weight=complex(1 - 0.1 * i, 0.05 * i), drops=i,
                              clip=10 * i, sample_count=10 ** 6 * i) for i in range(5))
    h = atkq.Header(n_channels=5, fs_hz=2_400_000, samples_per_ch=1200, channels=chans,
                    fc_hz=446_000_000, seq=77, t_utc_ns=1790000000123456789,
                    flags=int(atkq.Flag.NOISE_ON | atkq.Flag.SYNTHETIC), cal_state=2,
                    device_kind=2, segment=3, passport=9, stream_sample0=987654321,
                    gain_tenths_db=280)
    raw = atkq.pack_header(h)
    assert len(raw) == 96 + 5 * 32
    h2 = atkq.parse_header(raw)
    assert (h2.seq, h2.t_utc_ns, h2.fc_hz, h2.stream_sample0) == \
        (77, 1790000000123456789, 446_000_000, 987654321)
    assert list(h2.delays) == [i * 7 - 3 for i in range(5)]
    assert h2.channels[4].frac_delay == pytest.approx(0.5)
    assert h2.weights[1] == pytest.approx(complex(0.9, 0.05))


def test_the_crc_is_zlib_over_the_zeroed_header():
    import zlib
    h = atkq.Header(n_channels=2, fs_hz=2_400_000, samples_per_ch=100,
                    channels=(atkq.Channel(0, 0, 1 + 0j), atkq.Channel(-5, 0.2, 0.9 + 0j)))
    raw = bytearray(atkq.pack_header(h))
    stored = struct.unpack_from("<I", raw, 88)[0]
    zeroed = bytes(raw[:88]) + b"\0\0\0\0" + bytes(raw[92:])
    assert zlib.crc32(zeroed) & 0xFFFFFFFF == stored


def test_a_calibrated_payload_applies_weight_and_slope():
    # a channel with a pure weight (no frac) and one with a frac only
    n = 4096
    h = atkq.Header(n_channels=2, fs_hz=2_400_000, samples_per_ch=n,
                    channels=(atkq.Channel(0, 0.0, 1 + 0j),
                              atkq.Channel(0, 0.25, complex(0.5, 0.0))),
                    cal_state=2)
    rng = np.random.default_rng(0)
    iq = (rng.standard_normal((2, n)) + 1j * rng.standard_normal((2, n))).astype(np.complex64) * 0.1
    fr = atkq.Frame(h, atkq.encode_frame(h, iq))
    from atkdf.calibration import factors, apply
    c = factors(h, f0_hz=0.0)
    assert c[0] == pytest.approx(1 + 0j)
    assert c[1] == pytest.approx(0.5 + 0j)          # at f0=0 the slope is 1
    c2 = factors(h, f0_hz=h.fs_hz * 0.25)           # quarter-band: slope = exp(+j pi/2 * 0.25*...)
    assert abs(c2[1]) == pytest.approx(0.5)         # magnitude is the weight
    assert np.angle(c2[1]) != pytest.approx(0.0)    # phase has moved


def test_usable_for_df_is_the_documented_rule():
    def hdr(cal, flags):
        return atkq.Header(n_channels=2, fs_hz=1, samples_per_ch=1,
                           channels=(atkq.Channel(0, 0, 1 + 0j),) * 2,
                           cal_state=cal, flags=flags)
    assert hdr(atkq.CalState.CALIBRATED, 0).usable_for_df
    assert not hdr(atkq.CalState.SYNCED, 0).usable_for_df
    for fl in (atkq.Flag.NOISE_ON, atkq.Flag.MISALIGNED, atkq.Flag.RETUNE, atkq.Flag.CAL_STALE):
        assert not hdr(atkq.CalState.CALIBRATED, int(fl)).usable_for_df
    # SYNTHETIC / SEGMENT_START / CLIPPED do not by themselves make it unusable
    assert hdr(atkq.CalState.CALIBRATED, int(atkq.Flag.SYNTHETIC | atkq.Flag.SEGMENT_START)).usable_for_df


def test_the_fixture_atkdaq_wrote_reads_back():
    if not FIXTURE.is_file():
        pytest.skip("fixture not present")
    skipped = []
    with FIXTURE.open("rb") as f:
        frames = list(atkq.read_frames(f, skipped))
    assert frames and not skipped
    hs = [fr.header for fr in frames]
    assert all(h.has(atkq.Flag.SYNTHETIC) for h in hs)
    assert hs[0].has(atkq.Flag.SEGMENT_START)
    for a, b in zip(hs, hs[1:]):
        assert b.seq == a.seq + 1
        assert b.stream_sample0 == a.stream_sample0 + a.samples_per_ch
    for fr in frames:
        assert len(fr.payload) == fr.header.n_channels * fr.header.samples_per_ch * 2
    usable = [fr for fr in frames if fr.usable_for_df]
    assert usable
    y = usable[-1].iq()
    assert y.shape[0] == usable[-1].header.n_channels
    assert y.dtype == np.complex64


def test_a_damaged_stream_resynchronises_and_never_misreads():
    if not FIXTURE.is_file():
        pytest.skip("fixture not present")
    data = bytearray(FIXTURE.read_bytes())
    fb = len(data) // 20
    data[fb * 3 + 5] ^= 0xFF                              # frame 3 header: CRC fails
    data[fb * 7:fb * 7] = b"ATKQ not a header \x00\x01" * 8    # junk before frame 7
    skipped = []
    frames = list(atkq.read_frames(io.BytesIO(bytes(data)), skipped))
    assert sum(skipped) > 0
    seqs = [fr.header.seq for fr in frames]
    assert 3 not in seqs and seqs == sorted(seqs)


def test_a_newer_wire_version_is_refused_by_name():
    h = atkq.Header(n_channels=2, fs_hz=1, samples_per_ch=1,
                    channels=(atkq.Channel(0, 0, 1 + 0j),) * 2)
    raw = bytearray(atkq.pack_header(h))
    struct.pack_into("<H", raw, 4, 2)                     # version 2
    struct.pack_into("<I", raw, 88, 0)
    struct.pack_into("<I", raw, 88, atkq._crc(raw, len(raw)))   # a VALID v2 header
    with pytest.raises(atkq.AtkqError) as e:
        atkq.parse_header(bytes(raw))
    assert e.value.code == -3 and "version 2" in str(e.value)


def test_status_lines_parse():
    ev = atkq.parse_status('ATKDAQ event=sync passport=3 delays=0,-17,5 msg="a b"')
    assert ev["event"] == "sync" and ev["passport"] == "3" and ev["msg"] == "a b"
    assert atkq.parse_status("not a status line") is None


def test_the_file_source_replays_the_fixture():
    if not FIXTURE.is_file():
        pytest.skip("fixture not present")
    src = atkq.FileSource(FIXTURE)
    n = sum(1 for _ in src.frames())
    assert n == 20
    src.control("tune 450e6")           # a file has nothing to tune, but records it
    assert src.commands == ["tune 450e6"]
