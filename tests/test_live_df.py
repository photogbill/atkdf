"""The live-DF additions (2026-10-06, for the KrakenSDR's first DF test):
the per-bin band path, the energy gate, the status a display reads, the
array heading from a fixed site, and aligning that heading on a reference.

All against the simulator's planted truth, numpy only.
"""

from __future__ import annotations

import numpy as np
import pytest

from atkdf import atkq, engine, sim, site
from atkdf import manifold as M
from atkdf.ports import BurstEvent, Fix2D


def err(a, t):
    return ((a - t + 180) % 360) - 180


def target(h, f0, bw=12_500.0):
    """What ATK sends for an operator's DF target: a band, the whole frame."""
    return BurstEvent(protocol="", system="", id="", t_utc=h.t_utc,
                      n0=h.stream_sample0, n1=h.stream_sample0 + h.samples_per_ch,
                      fc_hz=h.fc_hz + f0, bw_hz=bw)


def run(sc, frames=8, gate=None, positioning=None, f0=200_000.0):
    s = sim.Simulator(sc, frames=frames)
    eng = engine.Engine(geometry=s.geometry, manifold=M.IdealManifold(s.geometry),
                        positioning=positioning, gate_snr_db=gate,
                        mode=engine.PlatformMode.static())
    recs, states = [], []
    for fr in s.frames():
        recs += eng.process(fr, events=[target(fr.header, f0)])
        states.append(dict(eng.last_status))
    return eng, recs, states


def test_simulated_payloads_are_samples_only():
    s = sim.Simulator(sim.Scenario(emitters=[sim.Emitter(bearing_deg=10.0)]), frames=4)
    for fr in s.frames():
        h = fr.header
        assert len(fr.payload) == h.n_channels * h.samples_per_ch * 2


@pytest.mark.parametrize("truth", [0.0, 37.0, 141.5, 250.0, 333.0])
def test_a_target_band_returns_the_planted_bearing(truth):
    sc = sim.Scenario(emitters=[sim.Emitter(bearing_deg=truth, f0_hz=200_000.0,
                                            snr_db=15.0)], seed=3)
    eng, recs, states = run(sc)
    assert len(recs) == 5                         # 8 frames − 3 noise-source frames
    for r in recs:
        assert abs(err(r.theta_rel_deg, truth)) < 1.0
        assert r.bw_hz == pytest.approx(12_500.0)
        assert "band_snr_db" in r.quality and len(r.quality["channel_db"]) == 5
    assert states[0]["state"] == "unusable"       # noise source in
    assert states[-1]["state"] == "bearing"
    assert eng.last_result is not None and eng.last_result.spectrum is not None


def test_the_band_path_agrees_with_the_whole_band_calibration():
    """Per-bin calibration inside the band must give the bearing the full
    FFT calibration gives — it is the same correction, done cheaper."""
    sc = sim.Scenario(emitters=[sim.Emitter(bearing_deg=200.0, f0_hz=-450_000.0,
                                            snr_db=20.0)], seed=5)
    s = sim.Simulator(sc, frames=5)
    eng = engine.Engine(geometry=s.geometry, mode=engine.PlatformMode.static())
    fr = list(s.frames())[-1]
    band = eng.process(fr, events=[target(fr.header, -450_000.0)])
    survey = eng.process(fr)
    assert band and survey
    assert abs(err(band[0].theta_rel_deg, survey[0].theta_rel_deg)) < 1.0


def test_the_gate_squelches_an_unkeyed_channel_and_says_why():
    # the emitter is on half the time; with the gate the off frames give nothing
    sc = sim.Scenario(emitters=[sim.Emitter(bearing_deg=90.0, f0_hz=200_000.0,
                                            snr_db=15.0, period_s=0.2, duty=0.5)],
                      seed=7)
    _eng, recs, states = run(sc, frames=13, gate=6.0)
    sq = [s for s in states if s["state"] == "squelched"]
    assert sq and "gate 6.0 dB" in sq[0]["why"]
    assert all(s["band_snr_db"] < 6.0 for s in sq)
    assert recs and all(abs(err(r.theta_rel_deg, 90.0)) < 1.5 for r in recs)
    assert len(recs) + len(sq) == 10             # every usable frame accounted for


def test_without_a_gate_noise_is_not_silently_a_bearing():
    # nothing transmitting at all: no gate, so the estimator runs on noise —
    # the status must say what the band SNR was either way
    sc = sim.Scenario(emitters=[sim.Emitter(bearing_deg=90.0, f0_hz=-800_000.0,
                                            snr_db=15.0)], seed=8)
    _eng, _recs, states = run(sc, frames=5, gate=None, f0=200_000.0)
    assert all("band_snr_db" in s for s in states if s["state"] != "unusable")
    assert max(s["band_snr_db"] for s in states if "band_snr_db" in s) < 3.0


def test_a_fixed_site_heading_makes_true_bearings_with_a_small_sigma():
    sc = sim.Scenario(emitters=[sim.Emitter(bearing_deg=30.0, f0_hz=200_000.0,
                                            snr_db=20.0)], seed=9)
    pos = site.FixedSite(38.70, -77.58, heading_deg=300.0, heading_sigma_deg=2.0)
    _eng, recs, _ = run(sc, positioning=pos)
    r = recs[-1]
    assert abs(err(r.theta_abs_deg, 330.0)) < 1.0      # 300 + 30
    assert r.heading_deg == 300.0 and r.heading_sigma_deg == 2.0
    assert r.sigma_deg < 10.0                         # not the 45° "stopped" GPS sigma
    assert r.lat == pytest.approx(38.70)


def test_a_site_without_position_still_gives_true_bearings():
    sc = sim.Scenario(emitters=[sim.Emitter(bearing_deg=30.0, f0_hz=200_000.0,
                                            snr_db=20.0)], seed=9)
    _eng, recs, _ = run(sc, positioning=site.FixedSite(heading_deg=10.0))
    assert abs(err(recs[-1].theta_abs_deg, 40.0)) < 1.0
    assert recs[-1].lat is None


def test_a_gps_course_is_still_used_when_no_array_heading_is_given():
    class Moving:
        def at(self, t):
            return Fix2D(lat=38.7, lon=-77.6, t_utc=t, course_deg=90.0,
                         speed_mps=15.0)
    sc = sim.Scenario(emitters=[sim.Emitter(bearing_deg=30.0, f0_hz=200_000.0,
                                            snr_db=20.0)], seed=9)
    _eng, recs, _ = run(sc, positioning=Moving())
    assert abs(err(recs[-1].theta_abs_deg, 120.0)) < 1.0


def test_align_heading_on_a_reference():
    rng = np.random.default_rng(1)
    rel = 75.0 + rng.normal(0, 2.0, 40)              # measured relative bearings
    a = site.align_heading(rel, expected_true_deg=10.0)
    assert abs(err(a["heading_deg"], 295.0)) < 1.0   # 10 − 75
    assert a["trustworthy"] and a["n"] == 40
    wide = site.align_heading(rng.uniform(0, 360, 40), 10.0)
    assert not wide["trustworthy"] and "scatter" in wide["sentence"]
    few = site.align_heading([75.0, 76.0], 10.0)
    assert not few["trustworthy"] and "too few" in few["sentence"]
    assert site.align_heading([], 10.0)["heading_deg"] is None


def test_circular_mean_wraps():
    m, s = site.circular_mean_deg([359.0, 1.0, 0.0])
    assert abs(err(m, 0.0)) < 1e-6 and s < 2.0
