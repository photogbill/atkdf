"""Where the antenna was, and the four places a BLUE file keeps it.

THE THEME OF THIS FILE IS REFUSAL. Every test that matters here is about the
module declining to invent a number:

  * no position → no line of bearing, and `load` returns None with a report
    naming which of the four sources it tried and what each one gave;
  * no heading → `bearing_ready` is False, and the DF layer refuses, because
    an array angle referenced to an unknown direction is not approximately
    right, it is unknown;
  * outside the track → None, never an extrapolation, because a straight-line
    continuation of a turning aircraft is fiction that looks like data.

The one thing it does invent is a position BETWEEN two fixes, and that is a
considered exception: at 200 m/s a one-second-old position is 200 m from the
truth, and 200 m of baseline error is a real bearing error on a real fix. So
it interpolates, marks the state, and inflates the uncertainty by what the
platform could have done in the gap.
"""

from __future__ import annotations

import csv
import struct

import numpy as np
import pytest

from atkdf import bluefile as bf
from atkdf import ephemeris as ep

T0 = 1.7e9


def track_file(path, n=10, moving=True):
    states = [{"epoch": T0 + t,
               "lat": 51.5 + (0.0005 * t if moving else 0.0),
               "lon": -0.12 + (0.001 * t if moving else 0.0),
               "alt_m": 1000.0, "heading_deg": (90.0 + 2 * t) % 360,
               "pitch_deg": 1.5, "roll_deg": -2.0} for t in range(n)]
    bf.write_state_vectors(path, states, timecode_unix=T0)
    return states


def iq_with_keywords(path, pairs):
    bf.write(path, np.zeros(64, dtype=np.complex64), 1e6, timecode_unix=T0)
    raw = path.read_bytes()
    ext = bf.pack_keywords(pairs)
    hdr, payload = bytearray(raw[:512]), raw[512:]
    struct.pack_into("<i", hdr, 24, -(-(512 + len(payload)) // 512))
    struct.pack_into("<i", hdr, 28, len(ext))
    path.write_bytes(bytes(hdr) + payload + b"\x00" * (-len(payload) % 512)
                     + ext)


# ---------------------------------------------------------------------------
# the four sources
# ---------------------------------------------------------------------------

def test_a_state_vector_track_is_read(tmp_path):
    p = tmp_path / "track.tmp"
    track_file(p)
    e, report = ep.load(p)
    assert e is not None and len(e) == 10
    assert e.has_attitude and not e.static
    assert any("USING: records" in r for r in report)


def test_the_quadword_block_is_read_with_its_attitude(tmp_path):
    """The best case: fixed offsets, nothing to configure, nothing to guess."""
    p = tmp_path / "site.tmp"
    bf.write_records(p, [{"name": "TIME", "format": "SD"}], [{"TIME": 0.0}],
                     type_no=5000,
                     quadwords={"frame_of_ref": "TOPO", "latitude": 52.1,
                                "longitude": 0.9, "altitude": 85.0,
                                "azimuth": 137.0, "elevation": 0.5})
    e, _r = ep.load(p)
    assert e is not None and e.static
    s = e.states[0]
    assert s.lat == pytest.approx(52.1) and s.lon == pytest.approx(0.9)
    assert s.heading_deg == pytest.approx(137.0)
    assert s.pitch_deg == pytest.approx(0.5)
    assert s.bearing_ready


def test_a_quadword_block_in_another_frame_is_flagged(tmp_path):
    """The same doubles mean different things per frame. Reading an ECI block
    as topocentric puts the platform at a latitude that is an orbital
    element."""
    p = tmp_path / "eci.tmp"
    bf.write_records(p, [{"name": "TIME", "format": "SD"}], [{"TIME": 0.0}],
                     type_no=5000,
                     quadwords={"frame_of_ref": "ECI", "latitude": 12.0,
                                "longitude": 34.0})
    e, _r = ep.load(p)
    assert any("ECI" in a for a in e.states[0].assumptions)


def test_the_vita49_keyword_vocabulary_is_recognised(tmp_path):
    """The names REDHAWK actually emits, so a file from that world works
    without configuration."""
    p = tmp_path / "kw.tmp"
    iq_with_keywords(p, [("GEOLOCATION_GPS::LATITUDE", 40.5),
                         ("GEOLOCATION_GPS::LONGITUDE", -74.2),
                         ("GEOLOCATION_GPS::ALTITUDE", 12.0),
                         ("GEOLOCATION_INS::HEADING_ANGLE", 275.0),
                         ("COL_RF", 162.55e6)])
    e, _r = ep.load(p)
    assert e is not None
    assert e.states[0].lat == pytest.approx(40.5)
    assert e.states[0].heading_deg == pytest.approx(275.0)
    assert bf.read_header(p)["center_freq"] == pytest.approx(162.55e6)


def test_a_caller_can_supply_its_own_keyword_names(tmp_path):
    """The point of the map being data. A producer with its own spelling
    needs a dict, not a patch."""
    p = tmp_path / "odd.tmp"
    iq_with_keywords(p, [("SHIP_POS_LAT", 12.25), ("SHIP_POS_LON", -3.5),
                         ("GYRO", 44.0)])
    assert ep.load(p)[0] is None
    e, _r = ep.load(p, keymap={"lat": "SHIP_POS_LAT", "lon": "SHIP_POS_LON",
                               "heading_deg": "GYRO"})
    assert e is not None
    assert e.states[0].lat == pytest.approx(12.25)
    assert e.states[0].heading_deg == pytest.approx(44.0)


def test_a_companion_csv_beside_the_capture_is_found(tmp_path):
    p = tmp_path / "cap.tmp"
    bf.write(p, np.zeros(64, dtype=np.complex64), 1e6, timecode_unix=T0)
    with open(tmp_path / "cap.ephem.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["epoch", "lat", "lon", "alt_m", "heading"])
        w.writeheader()
        for t in range(4):
            w.writerow({"epoch": T0 + t, "lat": 50.0 + 0.01 * t, "lon": 1.0,
                        "alt_m": 500.0, "heading": 12.0})
    e, report = ep.load(p)
    assert e is not None and len(e) == 4
    assert e.has_attitude
    assert any("companion" in r for r in report)


def test_atk_gps_track_rows_are_an_ephemeris_unchanged():
    """A drive with the GPS logger running IS an ephemeris."""
    rows = [{"epoch": T0 + t, "lat": 51.0, "lon": 0.5, "altitude_m": 30.0,
             "speed_knots": 40.0, "course_deg": 88.0} for t in range(3)]
    e = ep.from_rows(rows, "atk track")
    assert e is not None and len(e) == 3
    assert e.states[0].speed_mps == pytest.approx(20.58, abs=0.05)
    assert not e.has_attitude          # a course is not a heading


# ---------------------------------------------------------------------------
# lookup: interpolation, refusal, and the wrap
# ---------------------------------------------------------------------------

def test_a_state_between_two_fixes_is_interpolated_and_marked(tmp_path):
    p = tmp_path / "track.tmp"
    track_file(p)
    e, _r = ep.load(p)
    s = e.at(T0 + 3.5)
    assert s.interpolated and s.gap_s == pytest.approx(1.0)
    assert s.lat == pytest.approx(51.5 + 0.0005 * 3.5, abs=1e-9)
    assert s.position_sigma_m > 0


def test_heading_interpolates_the_short_way_round():
    """**359° and 1° average to 180° arithmetically**, which points the array
    backwards."""
    a = ep.PlatformState(epoch=0.0, lat=0.0, lon=0.0, heading_deg=359.0)
    b = ep.PlatformState(epoch=10.0, lat=0.0, lon=0.0, heading_deg=1.0)
    mid = ep.Ephemeris([a, b]).at(5.0)
    assert mid.heading_deg == pytest.approx(0.0, abs=1e-9)


def test_outside_the_track_the_answer_is_none(tmp_path):
    """**No extrapolation.** A straight-line continuation of a turning
    aircraft is fiction that looks like data."""
    p = tmp_path / "track.tmp"
    track_file(p)
    e, _r = ep.load(p)
    assert e.at(T0 + 500.0) is None
    assert e.at(T0 - 500.0) is None


def test_just_off_the_end_carries_its_own_staleness(tmp_path):
    p = tmp_path / "track.tmp"
    track_file(p)
    e, _r = ep.load(p)
    s = e.at(T0 + 9.0 + 5.0, max_gap_s=10.0)
    assert s is not None and s.interpolated
    assert s.gap_s == pytest.approx(5.0)
    assert s.position_sigma_m >= 5.0 * ep.ASSUMED_SPEED_MPS * 0.99
    assert any("had not moved" in a for a in s.assumptions)


def test_a_gap_too_large_to_bridge_is_refused():
    a = ep.PlatformState(epoch=0.0, lat=0.0, lon=0.0, heading_deg=0.0)
    b = ep.PlatformState(epoch=600.0, lat=1.0, lon=1.0, heading_deg=0.0)
    assert ep.Ephemeris([a, b]).at(300.0) is None


def test_nearest_without_interpolation_is_available_and_says_the_age(tmp_path):
    p = tmp_path / "track.tmp"
    track_file(p)
    e, _r = ep.load(p)
    s = e.at(T0 + 3.4, interpolate=False)
    assert s.lat == pytest.approx(51.5 + 0.0005 * 3, abs=1e-9)
    assert s.gap_s == pytest.approx(0.4, abs=1e-6)


# ---------------------------------------------------------------------------
# the refusals
# ---------------------------------------------------------------------------

def test_no_heading_means_not_ready_for_a_bearing():
    """**There is no partial credit here.** Without a heading the array's
    angle is relative to an unknown direction."""
    s = ep.PlatformState(epoch=0.0, lat=1.0, lon=2.0)
    assert not s.bearing_ready
    assert "NO HEADING" in s.describe()


def test_course_can_stand_in_for_heading_but_only_when_asked():
    """A platform crabbing in a crosswind points somewhere other than where
    it is going, and that difference is a bearing error of the same size."""
    rows = [{"epoch": T0 + t, "lat": 51.0, "lon": 0.5, "course_deg": 88.0}
            for t in range(3)]
    e = ep.from_rows(rows, "t")
    assert not e.has_attitude
    e2 = e.use_course_as_heading()
    assert e2.has_attitude
    s = e2.states[0]
    assert s.heading_deg == pytest.approx(88.0)
    assert s.heading_sigma_deg >= 10.0
    assert any("crabbing" in a for a in s.assumptions)


def test_a_file_with_nothing_reports_all_four_sources(tmp_path):
    """**"No ephemeris" tells an analyst nothing.** This says which of the
    four to go and fix."""
    p = tmp_path / "bare.tmp"
    bf.write(p, np.zeros(64, dtype=np.complex64), 1e6)
    e, report = ep.load(p)
    assert e is None
    text = " | ".join(report)
    assert "quadwords" in text and "state-vector records" in text
    assert "keywords" in text and "companion" in text


def test_a_track_beats_a_single_state(tmp_path):
    """A moving platform's position AT THE INTERCEPT is what matters, so a
    per-instant track wins over one header block."""
    p = tmp_path / "both.tmp"
    states = [{"epoch": T0 + t, "lat": 10.0 + t, "lon": 20.0,
               "heading_deg": 5.0} for t in range(4)]
    bf.write_records(
        p, bf.STATE_VECTOR_COLUMNS,
        [{"TIME": s["epoch"] + bf.J1950_TO_UNIX,
          "POS": [0.0, s["lat"], s["lon"]], "VEL": [0, 0, 0],
          "HDG": s["heading_deg"], "ELPS": [s["heading_deg"], 0.0, 0.0]}
         for s in states],
        type_no=5001, timecode_unix=T0,
        quadwords={"frame_of_ref": "TOPO", "latitude": 99.0, "longitude": 99.0,
                   "azimuth": 1.0})
    e, report = ep.load(p)
    assert len(e) == 4
    assert e.states[0].lat == pytest.approx(10.0)
    assert any("USING: records" in r for r in report)


def test_a_position_only_track_borrows_a_fixed_mounting_attitude(tmp_path):
    """A GPS track gives position; the header may give the mounting. Using
    both is right for a fixed installation and wrong the moment the platform
    turns, so the assumption is recorded on every state."""
    p = tmp_path / "mix.tmp"
    bf.write_records(p, [{"name": "TIME", "format": "SD"}], [{"TIME": 0.0}],
                     type_no=5000,
                     quadwords={"frame_of_ref": "TOPO", "latitude": 1.0,
                                "longitude": 2.0, "azimuth": 47.0})
    rows = [{"epoch": T0 + t, "lat": 51.0 + t, "lon": 0.5} for t in range(3)]
    e, report = ep.load(p, rows=rows)
    assert e.has_attitude
    assert e.states[0].heading_deg == pytest.approx(47.0)
    assert any("held constant" in a for a in e.states[0].assumptions)
    assert any("attitude filled from" in r for r in report)


# ---------------------------------------------------------------------------
# the inspector
# ---------------------------------------------------------------------------

def test_dump_shows_the_keywords_the_columns_and_the_verdict(tmp_path):
    """Point it at a real file and it tells you what the file actually is —
    which is how a layout nobody documented becomes one you know."""
    p = tmp_path / "track.tmp"
    track_file(p)
    text = ep.dump(p)
    assert "record columns" in text
    assert "TIME" in text and "POS" in text
    assert "altitude, latitude, longitude" in text
    assert "ephemeris sources" in text
    assert "RESULT" in text and "attitude present" in text


def test_dump_of_a_file_with_nothing_says_so_plainly(tmp_path):
    p = tmp_path / "bare.tmp"
    bf.write(p, np.zeros(64, dtype=np.complex64), 1e6)
    text = ep.dump(p)
    assert "no ephemeris" in text
    assert "cannot produce a georeferenced bearing" in text
