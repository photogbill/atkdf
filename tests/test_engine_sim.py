"""The spine: the simulator plants a truth and the whole engine returns it.

Every test here runs the same chain ATK will run live -- frames from a source,
decoder events on the side, calibration, DoA, bearing records -- against the
synthetic device, with nothing but numpy. This is how the engine is proven
before the KrakenSDR arrives, and the regression spine for every session
after.
"""

from __future__ import annotations

import numpy as np
import pytest

from atkdf import engine, sim
from atkdf import manifold as M
from atkdf.fakes import ListCatalogue
from atkdf.ports import Reference


def err(a, t):
    return ((a - t + 180) % 360) - 180


def run(sc, frames=10, mode=None, catalogue=None):
    s = sim.Simulator(sc, frames=frames)
    eng = engine.Engine(geometry=s.geometry, manifold=M.IdealManifold(s.geometry),
                        positioning=s.positioning, catalogue=catalogue,
                        mode=mode or engine.PlatformMode.static(), site_id="alpha")
    by = {}
    for e in s.events():
        by.setdefault(e.n0, []).append(e)
    recs = []
    for fr in s.frames():
        recs += eng.process(fr, events=by.get(fr.header.stream_sample0, []))
    return s, eng, recs


def test_the_planted_bearing_comes_back_through_the_whole_engine():
    sc = sim.Scenario(emitters=[sim.Emitter(bearing_deg=110.0, f0_hz=300_000.0, snr_db=20.0,
                                            protocol="dmr", system="1", id="473", sync_type="ms")],
                      heading_deg=30.0, seed=5)
    s, eng, recs = run(sc)
    assert recs
    r = recs[-1]
    assert r.emitter_key == "dmr:1:473"
    assert abs(err(r.theta_rel_deg, 110.0)) < 2.0
    assert abs(err(r.theta_abs_deg, 140.0)) < 2.0        # heading 30 + rel 110
    assert r.passport_uid.startswith("alpha/")
    assert r.fusable


def test_calibration_is_what_makes_the_bearing_right():
    # the same run, but comparing the calibrated bearing to the raw one proves
    # the receiver perturbation was really there and really removed
    from atkdf import calibration, doa
    sc = sim.Scenario(emitters=[sim.Emitter(bearing_deg=42.0, f0_hz=-200_000.0, snr_db=20.0)],
                      seed=8)
    s = sim.Simulator(sc, frames=6)
    man = M.IdealManifold(s.geometry)
    fr = [f for f in s.frames() if f.usable_for_df][-1]
    raw = s.geometry.element_order(fr.iq())
    cal = s.geometry.element_order(calibration.calibrate(fr))
    band = (-200_000 - 8000, -200_000 + 8000)
    def bearing(x):
        Y, _f, ne = doa.band_snapshots(x, sc.fs_hz, band[0], band[1], nfft=1024)
        cov = doa.covariance(Y, sc.fc_hz - 200_000, geometry=s.geometry, snapshots=ne)
        return doa.estimate(cov, man, sc.fc_hz - 200_000, method="music", sources=1).theta_deg
    assert abs(err(bearing(cal), 42.0)) < 2.0
    assert abs(err(bearing(raw), 42.0)) > 8.0            # uncalibrated is well off


def test_noise_source_frames_are_never_direction_found():
    sc = sim.Scenario(emitters=[sim.Emitter(bearing_deg=10.0, snr_db=20.0,
                                            protocol="dmr", system="1", id="1")], seed=1)
    s = sim.Simulator(sc, frames=8)
    frames = list(s.frames())
    assert any(f.header.has(sim.Flag.NOISE_ON) for f in frames)
    assert not any(f.usable_for_df for f in frames if f.header.has(sim.Flag.NOISE_ON))
    # and the calibrated ones carry passport 1
    assert {f.header.passport for f in frames if f.usable_for_df} == {1}


def test_two_slots_are_two_emitters_with_two_bearings():
    # DMR TDMA: one channel, two slots that ALTERNATE in time (never at once).
    # period 0.1 s = two 50 ms frames; slot 1 on the first, slot 2 on the second.
    sc = sim.Scenario(emitters=[
        sim.Emitter(bearing_deg=40.0, f0_hz=100_000.0, snr_db=20.0,
                    protocol="dmr", system="1", id="100", slot=1, sync_type="ms",
                    period_s=0.1, duty=0.5, phase0=0.0),
        sim.Emitter(bearing_deg=200.0, f0_hz=100_000.0, snr_db=20.0,
                    protocol="dmr", system="1", id="200", slot=2, sync_type="ms",
                    period_s=0.1, duty=0.5, phase0=0.5)],
        heading_deg=0.0, seed=6)
    s, eng, recs = run(sc, frames=12)
    by_id = {}
    for r in recs:
        by_id.setdefault(r.emitter_key, []).append(r)
    assert set(by_id) == {"dmr:1:100", "dmr:1:200"}
    assert abs(err(by_id["dmr:1:100"][-1].theta_rel_deg, 40.0)) < 3
    assert abs(err(by_id["dmr:1:200"][-1].theta_rel_deg, 200.0)) < 3
    assert by_id["dmr:1:100"][-1].slot == 1 and by_id["dmr:1:200"][-1].slot == 2


def test_a_repeater_outbound_burst_is_tagged_repeated_and_not_fused():
    # sync_type 'bs' (base-station sourced) means the bearing is the repeater's
    sc = sim.Scenario(emitters=[sim.Emitter(bearing_deg=90.0, f0_hz=0.0, snr_db=20.0,
                                            protocol="dmr", system="1", id="473",
                                            sync_type="bs")], heading_deg=0.0, seed=2)
    s, eng, recs = run(sc)
    assert recs and all(r.path_tag == "repeated" for r in recs)
    assert not any(r.fusable for r in recs)


def test_the_catalogue_tags_a_known_repeater_when_the_sync_is_silent():
    key = "p25:sys7:9001"
    cat = ListCatalogue([Reference(key=key, lat=38.7, lon=-77.5, kind="repeater_output",
                                   fc_hz=451_000_000.0)])
    sc = sim.Scenario(emitters=[sim.Emitter(bearing_deg=90.0, f0_hz=0.0, snr_db=20.0,
                                            protocol="p25", system="sys7", id="9001")],
                      heading_deg=0.0, seed=2)
    s, eng, recs = run(sc, catalogue=cat)
    # the emitter's fc must match the catalogue lookup path (fc_hz set on events)
    assert recs and all(r.path_tag == "repeated" for r in recs)


def test_a_survey_bearing_is_emitted_without_a_decoder_in_static_mode():
    sc = sim.Scenario(emitters=[sim.Emitter(bearing_deg=155.0, f0_hz=0.0, snr_db=20.0)],
                      heading_deg=0.0, seed=4)
    s = sim.Simulator(sc, frames=6)
    eng = engine.Engine(geometry=s.geometry, manifold=M.IdealManifold(s.geometry),
                        positioning=s.positioning, mode=engine.PlatformMode.static())
    recs = []
    for fr in s.frames():
        recs += eng.process(fr, events=[])              # no decoder events at all
    assert recs and abs(err(recs[-1].theta_rel_deg, 155.0)) < 3
    assert recs[-1].emitter_key == ""                   # a survey bearing has no identity


def test_a_recorded_run_reprocesses_to_the_same_bearing(tmp_path):
    from atkdf.atkq import FileSource
    sc = sim.Scenario(emitters=[sim.Emitter(bearing_deg=77.0, f0_hz=250_000.0, snr_db=20.0,
                                            protocol="dmr", system="1", id="55", sync_type="ms")],
                      heading_deg=15.0, seed=9)
    s = sim.Simulator(sc, frames=8)
    path = s.write(tmp_path / "run.atkq")
    # reprocess from the file, with the SAME decoder events (a sidecar would
    # carry them; here we take them from the sim's own side channel)
    eng = engine.Engine(geometry=s.geometry, manifold=M.IdealManifold(s.geometry),
                        positioning=s.positioning, mode=engine.PlatformMode.static())
    by = {}
    for e in s.events():
        by.setdefault(e.n0, []).append(e)
    recs = []
    for fr in FileSource(path).frames():
        recs += eng.process(fr, events=by.get(fr.header.stream_sample0, []))
    assert recs and abs(err(recs[-1].theta_rel_deg, 77.0)) < 2.0


def test_the_simulator_truth_matches_the_headers_it_writes():
    sc = sim.Scenario(emitters=[sim.Emitter(bearing_deg=10.0, snr_db=20.0)], seed=3)
    s = sim.Simulator(sc, frames=6)
    fr = [f for f in s.frames() if f.usable_for_df][0]
    truth = s.truth()
    # the header's weights are the inverse of the planted receiver gains
    assert np.allclose(fr.header.weights, [complex(w) for w in truth["weights"]], atol=1e-3)
    assert np.allclose(fr.header.frac_delays, truth["frac"], atol=1e-3)
