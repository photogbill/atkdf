"""Phase 2: posterior grid + ageing, roster/keep-alive, tracker, references
(radio compass, mounting offset, learned error model), wideband spectrum.
All on the simulator or closed-form, no hardware."""

from __future__ import annotations

import math

import numpy as np
import pytest

from atkdf import doa, posterior, references, roster, tracker
from atkdf import geodesy as geo
from atkdf import manifold as M
from atkdf.records import BearingRecord


def _bearing_record(rx, brg_abs, sigma=3.0, t=0.0):
    """A fusable bearing record from receiver `rx`=(lat,lon) toward absolute
    bearing `brg_abs`, with a wrapped-Gaussian likelihood about it."""
    lik = np.zeros(360)
    g = np.arange(360.0)
    d = (g - (brg_abs % 360) + 180) % 360 - 180
    lik = np.exp(-0.5 * (d / sigma) ** 2)
    lik = lik / lik.sum()
    # store as a RELATIVE likelihood with heading 0 (so absolute == relative)
    return BearingRecord(t_utc=t, theta_rel_deg=brg_abs % 360, sigma_deg=sigma,
                         estimator="music", heading_deg=0.0, theta_abs_deg=brg_abs % 360,
                         likelihood=lik, lat=rx[0], lon=rx[1], path_tag="direct")


# --- posterior -------------------------------------------------------------

def test_two_crossing_bearings_locate_an_emitter():
    truth = (38.86, -77.05)               # target
    a = (38.86, -77.10)                   # west of it  -> bearing ~90 (east)
    b = (38.81, -77.05)                   # south of it -> bearing ~0  (north)
    spec = posterior.area_around(38.84, -77.07, half_size_m=8000, cell_m=25)
    post = posterior.Posterior(spec)
    for rx in (a, b):
        brg = geo.bearing_between(rx[0], rx[1], *truth)
        post.update(_bearing_record(rx, brg, sigma=2.0))
    fx = post.fix()
    d = geo.distance_between(fx["lat"], fx["lon"], *truth)
    assert d < 300, (d, fx)               # crossing fix within 300 m on a 25 m grid


def test_a_repeated_bearing_does_not_move_the_posterior():
    spec = posterior.area_around(38.84, -77.07, half_size_m=4000, cell_m=50)
    post = posterior.Posterior(spec)
    rec = _bearing_record((38.84, -77.10), 90.0)
    rec.path_tag = "repeated"
    post.update(rec)
    assert post.n_updates == 0            # refused: it points at the repeater
    assert np.allclose(post.logp, 0.0)


def test_ageing_spreads_a_silent_emitter():
    spec = posterior.area_around(38.84, -77.07, half_size_m=8000, cell_m=25)
    post = posterior.Posterior(spec)
    for rx in ((38.86, -77.10), (38.81, -77.05)):
        brg = geo.bearing_between(rx[0], rx[1], 38.86, -77.05)
        post.update(_bearing_record(rx, brg, sigma=2.0, t=0.0))
    tight = post.fix()["sigma_m"]
    post.age(600.0, motion_state="on_foot")     # ten minutes on foot
    spread = post.fix()["sigma_m"]
    assert spread > tight + 50                   # the search area grew
    # stationary does not spread
    post2 = posterior.Posterior(spec)
    for rx in ((38.86, -77.10), (38.81, -77.05)):
        brg = geo.bearing_between(rx[0], rx[1], 38.86, -77.05)
        post2.update(_bearing_record(rx, brg, sigma=2.0, t=0.0))
    s0 = post2.fix()["sigma_m"]
    post2.age(600.0, motion_state="stationary")
    assert abs(post2.fix()["sigma_m"] - s0) < 1.0


# --- roster ----------------------------------------------------------------

def test_keyup_after_silence_alerts():
    r = roster.Roster(default_silence_s=300.0)
    assert not r.observe("dmr:1:473", 0.0)
    a = r.observe("dmr:1:473", 1000.0)         # 1000 s later, past 300 s
    assert a and a[0].kind == "keyup_after_silence"


def test_the_keepalive_learner_flags_a_missed_checkin():
    r = roster.Roster(fleet_map={"dmr:1:473": {"name": "Medic 2", "role": "red"}})
    # regular check-ins every 60 s
    for k in range(6):
        r.observe("dmr:1:473", 60.0 * k)
    assert not r.check(60.0 * 5 + 30)           # 30 s after the last: on time
    late = r.check(60.0 * 5 + 200)              # 200 s later: well past 3σ
    assert late and late[0].kind == "missed_checkin"
    assert "Medic 2" in late[0].label


def test_a_silent_expected_id_is_flagged_once():
    r = roster.Roster(expected_keys=["dmr:1:9"], default_silence_s=120.0)
    r.observe("dmr:1:9", 0.0)
    a1 = r.check(200.0)
    a2 = r.check(260.0)
    assert a1 and a1[0].kind == "silent"
    assert not a2                              # not re-raised until heard again


# --- tracker ---------------------------------------------------------------

def test_two_interleaved_no_id_emitters_stay_two_tracks():
    trk = tracker.BearingTracker(gate_deg=10.0, coast_s=10.0)
    ids = set()
    # two emitters at ~40 and ~200 deg, bursts interleaved in time
    for k in range(12):
        t = k * 0.5
        ids.add(trk.update(t, 40.0 + np.random.default_rng(k).normal(0, 1.5)))
        ids.add(trk.update(t + 0.25, 200.0 + np.random.default_rng(100 + k).normal(0, 1.5)))
    live = trk.live_tracks(now=6.0)
    assert len(live) == 2, [(t.id, round(t.bearing_deg)) for t in live]
    bearings = sorted(t.bearing_deg for t in live)
    assert abs(((bearings[0] - 40 + 180) % 360) - 180) < 8
    assert abs(((bearings[1] - 200 + 180) % 360) - 180) < 8


def test_slots_never_associate():
    trk = tracker.BearingTracker(gate_deg=30.0)
    a = trk.update(0.0, 100.0, slot=1)
    b = trk.update(0.1, 100.0, slot=2)        # same bearing, different slot
    assert a != b                             # two tracks, not one


# --- references ------------------------------------------------------------

def test_radio_compass_recovers_heading():
    rx = (38.80, -77.00)
    ref = (38.90, -77.00)                     # due north of the receiver
    true_brg = geo.bearing_between(*rx, *ref)  # ~0
    heading_true = 30.0
    measured_rel = (true_brg - heading_true) % 360.0    # what the array would read
    obs = references.radio_compass(*rx, *ref, measured_rel, sigma_deg=2.0, t_utc=0.0)
    assert abs(((obs.heading_deg - 30.0 + 180) % 360) - 180) < 1.0


def test_mounting_offset_is_recovered():
    mo = references.MountingOffset()
    rng = np.random.default_rng(0)
    for _ in range(50):
        course = rng.uniform(0, 360)
        mo.observe((course + 12.0) % 360.0 + rng.normal(0, 1.0), course)   # +12 deg offset
    assert abs(((mo.offset_deg - 12.0 + 180) % 360) - 180) < 1.5


def test_error_model_learns_sigma_by_snr():
    em = references.ErrorModel(bin_db=3.0)
    rng = np.random.default_rng(1)
    for _ in range(200):
        em.observe(20.0, rng.normal(0, 1.0))     # ~1 deg residuals at 20 dB
        em.observe(0.0, rng.normal(0, 6.0))      # ~6 deg residuals at 0 dB
    assert 0.6 < em.sigma_for(20.0) < 1.6
    assert 4.0 < em.sigma_for(0.0) < 8.0


def test_catalogue_lookup_and_learn():
    from atkdf.ports import Reference
    cat = references.Catalogue([Reference(key="lte:0:PCI42", lat=38.9, lon=-77.0,
                                          kind="lte", verified=True)])
    assert cat.lookup("lte:0:PCI42") is not None
    assert cat.near(38.9, -77.0, 1000)
    cat.learn("dmr:1:repeater7", 38.8, -77.1, position_sigma_m=120.0, kind="repeater_output")
    assert cat.lookup("dmr:1:repeater7").source == "learned"


# --- wideband spectrum -----------------------------------------------------

def test_wideband_spectrum_finds_an_emitter_in_its_bin():
    from atkdf import sim, calibration
    sc = sim.Scenario(emitters=[sim.Emitter(bearing_deg=95.0, f0_hz=400_000.0, snr_db=25.0,
                                            bw_hz=25_000.0)], seed=4)
    s = sim.Simulator(sc, frames=6)
    fr = [f for f in s.frames() if f.usable_for_df][-1]
    y = s.geometry.element_order(calibration.calibrate(fr))
    out = doa.wideband_spectrum(y, sc.fs_hz, M.IdealManifold(s.geometry), sc.fc_hz,
                                nfft=256, step_deg=2.0)
    # the bins around +400 kHz should point near 95 deg
    fb = out["freqs_hz"]
    band = np.abs(fb - 400_000) < 20_000
    peaks = out["peak_bearing_per_bin"][band]
    assert peaks.size and np.median(np.abs(((peaks - 95 + 180) % 360) - 180)) < 6


# --- motion (Phase 3 start; closes the posterior-ageing loop) ---------------

def test_motion_calls_a_consistent_bearing_set_stationary():
    from atkdf import motion
    # a stationary emitter seen from several receiver positions: all bearings
    # point at ONE fixed point, so the fix residual ~ sigma -> stationary
    truth = (38.90, -77.00)
    recs = []
    for i, rx in enumerate([(38.86, -77.06), (38.85, -77.02), (38.84, -77.05),
                            (38.87, -77.08), (38.83, -77.03)]):
        brg = geo.bearing_between(rx[0], rx[1], *truth)
        recs.append(_bearing_record(rx, brg, sigma=2.0, t=float(i)))
    v = motion.kinematic_vote(recs)
    assert v.state == "stationary", v.detail


def test_motion_calls_a_wandering_bearing_set_moving():
    from atkdf import motion
    # bearings from ONE receiver that swing far more than sigma: no fixed point
    # fits -> moving
    rx = (38.85, -77.05)
    recs = [_bearing_record(rx, b, sigma=2.0, t=float(i))
            for i, b in enumerate([10, 40, 80, 130, 190, 250])]
    v = motion.kinematic_vote(recs)
    assert v.state in ("on_foot", "vehicle"), v.detail


def test_motion_fuse_prefers_the_confident_indicator():
    from atkdf.motion import MotionVote, fuse
    out = fuse([MotionVote("vehicle", 0.3, "kinematic"),
                MotionVote("stationary", 0.8, "channel")])
    assert out.state == "stationary"
