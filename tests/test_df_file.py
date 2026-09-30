"""Direction finding from a file, end to end, against a known emitter.

**THE POINT OF THIS FILE IS THAT THERE IS A TRUTH TO CHECK AGAINST.** A
synthetic collection run is built where the emitter's position is known
exactly, the array sees whatever cone angle the geometry gives, and the
ephemeris is written into the BLUE file the way a real collection system would
write it. Everything from there — de-interleaving the channels, matching each
block to a position on the track, rotating the array angle by the platform's
attitude, resolving the cone ambiguity, and fitting the fix — has to come back
to the point it started from.

It does, to a couple of metres, and getting there found four real bugs:

  * the Gauss-Newton step had the wrong sign and diverged geometrically;
  * the estimator compared geodesic bearings against chord azimuths, worth
    0.196° at 25 km and 89 m of fix error;
  * the `O` type code was missing, so unsigned captures would not open;
  * `NH` was treated as a data type, so no record file would open.

THE REFUSALS MATTER AS MUCH AS THE ANSWER. Without an ephemeris there is no
line of bearing; without a heading an array angle cannot be referenced to
north at all. Both are tested, because in both cases producing a plausible
number would be worse than producing nothing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from atkdf import bluefile as bf
from atkdf import df_file as df
from atkdf import fusion as fu
from atkdf import geodesy as geo
from atkdf import geoloc

FS = 1e6
FC = 300e6
LAM = 299_792_458.0 / FC
PER = 4096
T0 = 1.7e9
TRUE = (51.60, -0.05)

STRAIGHT = [(51.40, -0.30 + 0.10 * i) for i in range(5)]
DOGLEG = [(51.40, -0.30), (51.42, -0.20), (51.44, -0.10), (51.40, -0.02),
          (51.36, 0.04)]
DOGLEG_HDG = [70.0, 70.0, 70.0, 140.0, 140.0]


def collection(path, stations, headings, n_el=4, snr=25.0, seed=0,
               target=TRUE, alt=3000.0):
    """A synthetic run: the platform flies the station list and the array
    sees the emitter at whatever cone angle the geometry gives."""
    chans = np.zeros((n_el, PER * len(stations)), dtype=np.complex64)
    states = []
    for k, ((la, lo), hdg) in enumerate(zip(stations, headings)):
        brg = geo.bearing_between(la, lo, *target)
        rel = math.radians(geo.angle_difference(brg, hdg))
        cone = math.degrees(math.acos(math.cos(rel)))
        iq, _g = geoloc.synth_array(cone, elements=n_el, spacing_m=LAM / 2,
                                    wavelength_m=LAM, samples=PER,
                                    snr_db=snr, seed=seed + k)
        chans[:, k * PER:(k + 1) * PER] = iq
        # the state belongs at the BLOCK CENTRE, which is where the pipeline
        # asks for it
        states.append({"epoch": T0 + (k * PER + PER / 2) / FS,
                       "lat": la, "lon": lo, "alt_m": alt,
                       "heading_deg": hdg})
    bf.write_channels(path, chans, FS, center_freq=FC, timecode_unix=T0)
    return states


def rows_of(states):
    return [{"epoch": s["epoch"], "lat": s["lat"], "lon": s["lon"],
             "alt_m": s["alt_m"], "heading_deg": s["heading_deg"]}
            for s in states]


def cfg(**kw):
    return df.DFConfig(spacing_m=LAM / 2, block=PER, **kw)


# ---------------------------------------------------------------------------
# 1 — the end-to-end truth check
# ---------------------------------------------------------------------------

def test_every_bearing_comes_back_to_the_emitter(tmp_path):
    """Sub-hundredth-of-a-degree, which is what the array is actually capable
    of at 25 dB. Anything structurally wrong anywhere in the chain shows up
    here as tenths or worse."""
    p = tmp_path / "run.tmp"
    states = collection(p, DOGLEG, DOGLEG_HDG)
    lobs, _report = df.bearings(p, cfg(), rows=rows_of(states))
    assert len(lobs) == len(DOGLEG)
    resolved, _fix, _note = fu.resolve_ambiguity(lobs)
    for lob in resolved:
        truth = geo.bearing_between(lob.lat, lob.lon, *TRUE)
        assert abs(geo.angle_difference(lob.bearing_deg, truth)) < 0.05


def test_the_fix_lands_on_the_emitter(tmp_path):
    p = tmp_path / "run.tmp"
    states = collection(p, DOGLEG, DOGLEG_HDG)
    lobs, _r = df.bearings(p, cfg(), rows=rows_of(states))
    resolved, fix, _n = fu.resolve_ambiguity(lobs)
    assert geo.distance_between(fix.lat, fix.lon, *TRUE) < 500.0
    assert fix.consistent


def test_a_worse_snr_widens_the_ellipse_rather_than_moving_the_answer(
        tmp_path):
    out = {}
    for snr in (25.0, 5.0):
        p = tmp_path / f"snr{snr:.0f}.tmp"
        states = collection(p, DOGLEG, DOGLEG_HDG, snr=snr, seed=17)
        lobs, _r = df.bearings(p, cfg(), rows=rows_of(states))
        _res, fix, _n = fu.resolve_ambiguity(lobs)
        out[snr] = fix
    assert geo.distance_between(out[5.0].lat, out[5.0].lon, *TRUE) < 5_000.0
    assert out[5.0].residual_rms_deg >= out[25.0].residual_rms_deg


# ---------------------------------------------------------------------------
# 2 — the refusals
# ---------------------------------------------------------------------------

def test_no_ephemeris_means_no_bearings_and_a_report_that_says_why(tmp_path):
    """**A bearing with no origin is not a line of bearing.**"""
    p = tmp_path / "run.tmp"
    collection(p, DOGLEG, DOGLEG_HDG)
    lobs, report = df.bearings(p, cfg())
    assert lobs == []
    text = " | ".join(report)
    assert "REFUSED" in text
    assert "no origin" in text


def test_no_heading_means_no_bearings_and_the_report_names_it(tmp_path):
    """**Not approximately wrong — unknown.** An array measures an angle
    relative to itself."""
    p = tmp_path / "run.tmp"
    states = collection(p, DOGLEG, DOGLEG_HDG)
    rows = [{k: v for k, v in r.items() if k != "heading_deg"}
            for r in rows_of(states)]
    lobs, report = df.bearings(p, cfg(), rows=rows)
    assert lobs == []
    text = " | ".join(report)
    assert "REFUSED" in text and "no heading" in text
    assert "not approximately" in text


def test_course_over_ground_can_be_opted_into(tmp_path):
    p = tmp_path / "run.tmp"
    states = collection(p, DOGLEG, DOGLEG_HDG)
    rows = [{**{k: v for k, v in r.items() if k != "heading_deg"},
             "course_deg": s["heading_deg"]}
            for r, s in zip(rows_of(states), states)]
    assert df.bearings(p, cfg(), rows=rows)[0] == []
    lobs, report = df.bearings(p, cfg(), rows=rows,
                               use_course_as_heading=True)
    assert len(lobs) == len(DOGLEG)
    assert any("course over ground substituted" in r for r in report)
    assert lobs[0].sigma_deg >= 10.0        # and it costs what it costs


def test_a_single_channel_file_cannot_do_direction_finding(tmp_path):
    """And the refusal has to carry the physics AND the remedy, because this
    is the commonest file anyone will point at it — one dongle, one stream —
    and "0 bearings" tells its owner neither why nor what would work."""
    p = tmp_path / "one.tmp"
    bf.write(p, np.zeros(4096, dtype=np.complex64), FS, center_freq=FC)
    method, why = df.detect_method(bf.read_header(p))
    assert method == ""
    assert "PHASE DIFFERENCE" in why          # why it cannot
    assert "crossed-loop" in why              # what would
    assert "TDOA" in why                      # what to do with the file
    assert "two dongles will not do" in why   # and the trap


def test_an_unreadable_file_is_reported_not_raised(tmp_path):
    lobs, report = df.bearings(tmp_path / "absent.tmp", cfg())
    assert lobs == []
    assert "cannot read" in report[0]


# ---------------------------------------------------------------------------
# 3 — what the file supports
# ---------------------------------------------------------------------------

def test_four_channels_are_detected_as_an_array(tmp_path):
    p = tmp_path / "run.tmp"
    collection(p, DOGLEG, DOGLEG_HDG)
    method, why = df.detect_method(bf.read_header(p))
    assert method == "esprit"
    assert "4 coherent channels" in why


def test_two_channels_are_read_as_crossed_loops(tmp_path):
    p = tmp_path / "ww.tmp"
    n = PER * 3
    rng = np.random.default_rng(2)
    s = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    brg_rel = 35.0
    ch = np.stack([s * math.sin(math.radians(brg_rel)),
                   s * math.cos(math.radians(brg_rel))]).astype(np.complex64)
    bf.write_channels(p, ch, FS, center_freq=FC, timecode_unix=T0)
    method, why = df.detect_method(bf.read_header(p))
    assert method == "watson_watt"
    assert "crossed loops" in why
    rows = [{"epoch": T0 + (k * PER + PER / 2) / FS, "lat": 51.4,
             "lon": -0.3, "alt_m": 0.0, "heading_deg": 100.0}
            for k in range(3)]
    lobs, _r = df.bearings(p, cfg(method="watson_watt"), rows=rows)
    assert lobs
    # relative 35° on a heading of 100° is a true bearing of 135°
    assert abs(geo.angle_difference(lobs[0].bearing_deg, 135.0)) < 3.0
    # no sense channel, so it is ambiguous by 180°
    assert abs(geo.angle_difference(lobs[0].ambiguous_with, 315.0)) < 3.0


def test_bearings_already_in_the_file_are_read_not_recomputed(tmp_path):
    """Then this is bookkeeping: attach each one to the position it was
    taken from."""
    p = tmp_path / "aoa.tmp"
    # SIGM, not SIGMA: class-3 column names are FOUR characters, and
    # `write_records` now refuses the long form rather than truncating it
    # into a column every reader would miss.
    rows = [{"TIME": T0 + i + bf.J1950_TO_UNIX, "AOA": 33.0 + i,
             "SIGM": 1.5} for i in range(4)]
    bf.write_records(p, [{"name": "TIME", "format": "SD"},
                         {"name": "AOA", "format": "SD"},
                         {"name": "SIGM", "format": "SD"}],
                     rows, type_no=3000, timecode_unix=T0)
    method, why = df.detect_method(bf.read_header(p))
    assert method == "aoa"
    assert "already carries computed bearings" in why
    eph = [{"epoch": T0 + i, "lat": 51.4, "lon": -0.3, "alt_m": 0.0,
            "heading_deg": 90.0} for i in range(4)]
    lobs, report = df.bearings(p, cfg(), rows=eph)
    assert len(lobs) == 4
    assert lobs[2].bearing_deg == pytest.approx(35.0)
    assert lobs[2].sigma_deg == pytest.approx(1.5)
    assert any("AOA" in r for r in report)


def test_a_relative_aoa_column_has_the_heading_applied(tmp_path):
    """**The only evidence of which it is, is the column name**, so which
    was assumed is recorded on every bearing."""
    p = tmp_path / "rel.tmp"
    bf.write_records(p, [{"name": "TIME", "format": "SD"},
                         {"name": "RAOA", "format": "SD"}],
                     [{"TIME": T0 + bf.J1950_TO_UNIX, "RAOA": 20.0}],
                     type_no=3000, timecode_unix=T0)
    eph = [{"epoch": T0, "lat": 51.4, "lon": -0.3, "heading_deg": 100.0}]
    lobs, _r = df.bearings(p, cfg(), rows=eph)
    assert lobs[0].bearing_deg == pytest.approx(120.0)
    assert lobs[0].evidence["relative"] is True
    assert "heading applied" in lobs[0].method


# ---------------------------------------------------------------------------
# 4 — the cone, and time alignment
# ---------------------------------------------------------------------------

def test_every_array_bearing_is_ambiguous_and_says_so(tmp_path):
    p = tmp_path / "run.tmp"
    states = collection(p, DOGLEG, DOGLEG_HDG)
    lobs, report = df.bearings(p, cfg(), rows=rows_of(states))
    assert all(lob.ambiguous_with is not None for lob in lobs)
    assert any("AMBIGUOUS" in r for r in report)
    assert any("cannot separate" in r for r in report)


def test_each_block_uses_the_position_at_its_own_centre(tmp_path):
    """Off by half a block and the bearings are wrong by degrees — which is
    how this test was written in the first place, and it showed up as errors
    growing along the run."""
    p = tmp_path / "run.tmp"
    states = collection(p, DOGLEG, DOGLEG_HDG)
    lobs, _r = df.bearings(p, cfg(), rows=rows_of(states))
    for lob, s in zip(lobs, states):
        assert lob.epoch == pytest.approx(s["epoch"], abs=1e-6)
        assert lob.lat == pytest.approx(s["lat"], abs=1e-9)
        assert lob.evidence["heading_deg"] == pytest.approx(s["heading_deg"])


def test_a_file_with_no_time_step_is_refused(tmp_path):
    p = tmp_path / "notime.tmp"
    collection(p, DOGLEG, DOGLEG_HDG)
    raw = bytearray(p.read_bytes())
    import struct
    struct.pack_into("<d", raw, 256 + 8, 0.0)      # xdelta
    struct.pack_into("<d", raw, 256 + 32, 0.0)     # ydelta
    p.write_bytes(bytes(raw))
    states = collection(tmp_path / "tmp.tmp", DOGLEG, DOGLEG_HDG)
    lobs, report = df.bearings(p, cfg(), rows=rows_of(states))
    assert lobs == []
    assert any("no time step" in r for r in report)


# ---------------------------------------------------------------------------
# 5 — the sigma the bearings actually demonstrate
# ---------------------------------------------------------------------------

def test_a_stationary_dwell_can_check_its_own_sigma(tmp_path):
    p = tmp_path / "dwell.tmp"
    states = collection(p, [(51.40, -0.30)] * 6, [80.0] * 6, seed=200)
    lobs, report = df.bearings(p, cfg(), rows=rows_of(states))
    line = next(r for r in report if "claimed" in r)
    assert "one dwell" in line
    assert "conservative" in line or "consistent" in line
    cal = df.calibrate_sigma(lobs)
    assert cal[0].sigma_deg < lobs[0].sigma_deg
    assert "one dwell" in cal[0].evidence["sigma_source"]


def test_a_moving_run_defers_to_the_post_fit_residuals(tmp_path):
    """**Raw scatter on a moving platform measures the flight path.** On a
    28 km run it reads as 41° of "error" and calls a perfectly good array
    wildly optimistic."""
    p = tmp_path / "run.tmp"
    states = collection(p, STRAIGHT, [80.0] * 5)
    lobs, report = df.bearings(p, cfg(), rows=rows_of(states))
    line = next(r for r in report if "claimed" in r)
    assert "the platform moved" in line
    assert "post-fit residuals" in line
    # …and without a fix to subtract the geometry, nothing is changed
    assert df.calibrate_sigma(lobs)[0].sigma_deg == lobs[0].sigma_deg


def test_calibrating_on_residuals_says_it_is_the_random_part_only(tmp_path):
    """**Residuals cannot see a shared error** — it moves every bearing
    together and leaves them untouched. Measured: raw residuals gave σ =
    0.036°, a 25 m ellipse, and a fix 89 m from the truth."""
    p = tmp_path / "run.tmp"
    states = collection(p, DOGLEG, DOGLEG_HDG)
    lobs, _r = df.bearings(p, cfg(), rows=rows_of(states))
    resolved, fix, _n = fu.resolve_ambiguity(lobs)
    cal = df.calibrate_sigma(resolved, fix)
    assert cal[0].evidence["random_only"] is True
    assert "systematic_deg" in cal[0].evidence["note"]
    assert "corrected for two fitted parameters" in cal[0].evidence[
        "sigma_source"]


def test_too_few_bearings_to_calibrate_are_left_alone(tmp_path):
    p = tmp_path / "few.tmp"
    states = collection(p, DOGLEG[:3], DOGLEG_HDG[:3])
    lobs, _r = df.bearings(p, cfg(), rows=rows_of(states))
    assert df.calibrate_sigma(lobs) == lobs


# ---------------------------------------------------------------------------
# 6 — the array configuration, which cannot be guessed
# ---------------------------------------------------------------------------

def test_the_wavelength_comes_from_the_files_own_centre_frequency(tmp_path):
    p = tmp_path / "run.tmp"
    collection(p, DOGLEG, DOGLEG_HDG)
    hdr = bf.read_header(p)
    assert df.DFConfig().wavelength(hdr) == pytest.approx(LAM)


def test_without_a_wavelength_anywhere_the_array_refuses(tmp_path):
    """A spacing in metres cannot become an angle without one, and guessing
    produces a real-looking bearing that is wrong by whatever the ratio is
    wrong by."""
    p = tmp_path / "nofreq.tmp"
    states = collection(p, DOGLEG, DOGLEG_HDG)
    raw = bytearray(p.read_bytes())
    import struct
    struct.pack_into("<i", raw, 160, 0)          # drop the keywords
    p.write_bytes(bytes(raw))
    lobs, report = df.bearings(p, cfg(method="esprit"), rows=rows_of(states))
    assert lobs == []
    assert any("needs a wavelength" in r for r in report)


def test_an_over_long_class3_column_name_is_refused(tmp_path):
    """**Silent truncation is data loss.** SIGMA becomes SIGM, every reader
    looking for SIGMA misses it, and the uncertainty is simply gone."""
    with pytest.raises(bf.BlueError, match="four characters"):
        bf.write_records(tmp_path / "x.tmp",
                         [{"name": "SIGMA", "format": "SD"}],
                         [{"SIGMA": 1.0}], type_no=3000)


def test_the_calibrated_sigma_respects_a_floor(tmp_path):
    """**Six residuals are a very noisy estimate of their own spread**, and a
    four-element array claiming 0.02° is an artefact of the median absolute
    deviation of a small sample, not a measurement. Observed: calibration
    drove σ to 0.02° against a residual RMS of 0.053°, and the consistency
    test then rejected a fix that was 90 m from the truth — the fit was fine,
    the σ was not."""
    p = tmp_path / "run.tmp"
    states = collection(p, DOGLEG, DOGLEG_HDG)
    lobs, _r = df.bearings(p, cfg(), rows=rows_of(states))
    resolved, fix, _n = fu.resolve_ambiguity(lobs)
    free = df.calibrate_sigma(resolved, fix)
    floored = df.calibrate_sigma(resolved, fix, floor_deg=0.5)
    assert free[0].sigma_deg < 0.5
    assert floored[0].sigma_deg == pytest.approx(0.5)
    # …and with a credible sigma the fit is consistent again
    assert fu.ml_fix(floored).consistent
