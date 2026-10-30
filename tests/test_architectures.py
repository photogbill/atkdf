"""The four architectures under one entry point (plan §11.4.2), each run on
the simulator with the imperfections the literature documents: quantised
phase shifters, unmatched channels, random LO phase on retune, a burst that
ends mid-sweep."""

from __future__ import annotations

import math

import numpy as np
import pytest

from atkdf import architectures as AR
from atkdf import arrayprofile as AP
from atkdf import manifold as M
from atkdf import ports, sim


def cd(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


# -- weights -----------------------------------------------------------------

def test_quantised_weights_sit_on_the_phase_and_gain_steps():
    rng = np.random.default_rng(0)
    W = rng.normal(size=(8, 2)) + 1j * rng.normal(size=(8, 2))
    W[3, 1] = 0.0
    Q = AR.quantise_weights(W, 2.8125, 0.5)
    ph = np.degrees(np.angle(Q[np.abs(Q) > 0]))
    assert np.allclose(np.round(ph / 2.8125) * 2.8125, ph, atol=1e-9)
    top = np.abs(Q).max()
    db = 20 * np.log10(np.abs(Q[np.abs(Q) > 0]) / top)
    assert np.allclose(np.round(db / 0.5) * 0.5, db, atol=1e-9)
    assert Q[3, 1] == 0


# -- switched ------------------------------------------------------------------

def _switched(th, snr, seed=3, **kw):
    p = AP.preset("b210_switched_uca8c")
    adf = AR.ArrayDF(p)
    s = sim.SwitchedArraySim(adf.manifold, [sim.PlaneWave(th, snr_db=snr)], 450e6,
                             p.switching["reference_element"], p.switching["schedule"],
                             seed=seed, **kw)
    return adf, s


@pytest.mark.parametrize("th", [37.0, 200.0, 311.0])
def test_switched_finds_the_bearing_and_receiver_offsets_cancel(th):
    adf, s = _switched(th, 10.0)
    r = adf.estimate(s.capture_sequence(), 450e6)
    assert abs(cd(r.theta_deg, th)) < 1.5
    assert r.estimator == "switched" and r.n_sources == 1
    assert any("SINGLE-SOURCE" in c for c in r.caveats)
    s.retune()                                # a new LO phase, common to both channels
    r2 = adf.estimate(s.capture_sequence(), 450e6)
    assert abs(cd(r2.theta_deg, r.theta_deg)) < 1.0


def test_switched_says_when_the_signal_ended_part_way():
    adf, s = _switched(80.0, 15.0, off_after_slot=4)
    r = adf.estimate(s.capture_sequence(), 450e6)
    assert any("dropped for low coherence" in c for c in r.caveats)
    assert any("never measured" in c for c in r.caveats)
    assert r.quality["elements_used"] == 6               # five ring elements + reference


def test_switched_refuses_with_too_few_elements():
    adf, s = _switched(80.0, 15.0, off_after_slot=0)
    with pytest.raises(ValueError, match="at least 3"):
        adf.estimate(s.capture_sequence(), 450e6)


# -- analog: beam-scan and monopulse ---------------------------------------------

PH = AP.preset("phaser_cn0566_xband")
F_X = 10.25e9


def _phaser(th, snr, seed=1, **kw):
    man = PH.manifold()
    st = sim.SteerableArraySim(man, [sim.PlaneWave(th, snr_db=snr)], F_X, n_outputs=2,
                               halves=PH.steering["halves"], phase_step_deg=2.8125,
                               gain_step_db=0.5, shifter_rms_deg=3.0, shifter_gain_rms_db=0.3,
                               seed=seed, **kw)
    return man, st


def test_the_simulated_steerer_is_a_beam_steerer():
    _man, st = _phaser(10.0, 10.0)
    assert isinstance(st, ports.BeamSteerer)


@pytest.mark.parametrize("th", [17.0, 0.0, 327.0])
def test_beamscan_lands_well_inside_a_beamwidth(th):
    _man, st = _phaser(th, 15.0)
    r = AR.ArrayDF(PH, steerer=st).estimate(f_hz=F_X, method="beamscan", noise_power=1.0)
    bw = r.quality["beamwidth_deg"]
    assert abs(cd(r.theta_deg, th)) < 0.25 * bw
    # the stated sigma: a third to a half of a beamwidth at 15 dB (lit2 §1.1),
    # plus the model prior in quadrature
    assert 0.3 * bw < r.sigma_deg < 0.75 * bw


def test_beamscan_sigma_grows_toward_a_full_beamwidth_at_low_snr():
    _m1, st1 = _phaser(17.0, 15.0)
    _m2, st2 = _phaser(17.0, 0.0)
    r15 = AR.ArrayDF(PH, steerer=st1).estimate(f_hz=F_X, method="beamscan", noise_power=1.0)
    r0 = AR.ArrayDF(PH, steerer=st2).estimate(f_hz=F_X, method="beamscan", noise_power=1.0)
    assert r0.sigma_deg > r15.sigma_deg
    assert r0.sigma_deg <= math.hypot(r0.quality["beamwidth_deg"], 4.0) + 1e-9


def test_a_sweep_longer_than_the_burst_is_flagged():
    _man, st = _phaser(17.0, 15.0)
    r = AR.ArrayDF(PH, steerer=st).estimate(f_hz=F_X, method="beamscan", noise_power=1.0,
                                            burst_s=0.05)
    assert any("longer than the burst" in c for c in r.caveats)


def test_monopulse_one_look_close_to_the_steer_direction():
    man, st = _phaser(8.0, 20.0)
    halves = PH.steering["halves"]
    w = AR.steer_weights(man, 5.0, F_X)
    W = np.zeros((8, 2), complex)
    W[halves[0], 0] = w[halves[0]]
    W[halves[1], 1] = w[halves[1]]
    st.apply(AR.quantise_weights(W, 2.8125, 0.5))
    y = st.capture(2048)
    r = AR.monopulse_estimate(y[0], y[1], 5.0, halves, man.geometry, F_X, manifold=man)
    assert abs(cd(r.theta_deg, 8.0)) < 0.5
    assert r.quality["calibrated"] is False
    assert any("THEORETICAL" in c for c in r.caveats)


def test_monopulse_track_closes_in_from_a_coarse_error():
    man, st = _phaser(21.0, 15.0)
    r = AR.monopulse_track(st, man, PH.steering["halves"], F_X, theta0=15.0,
                           phase_step_deg=2.8125, gain_step_db=0.5)
    assert abs(cd(r.theta_deg, 21.0)) < 0.4
    assert r.quality["looks"] >= 2


def test_monopulse_calibration_recovers_a_planted_channel_offset():
    man, st = _phaser(0.0, 25.0, seed=5)
    # only the planted output offset: the per-element shifter errors would
    # (rightly) be absorbed into δ too and blur what this test checks
    st.element_error = np.ones(8, complex)
    st.output_error = np.array([1.0, np.exp(1j * math.radians(20.0))])
    halves = PH.steering["halves"]
    recs = []
    for th, steer in [(-6.0, 0.0), (-3.0, 0.0), (3.0, 0.0), (6.0, 0.0), (10.0, 12.0), (14.0, 12.0)]:
        st.waves = [sim.PlaneWave(th % 360, snr_db=25)]
        w = AR.steer_weights(man, steer, F_X)
        W = np.zeros((8, 2), complex)
        W[halves[0], 0] = w[halves[0]]
        W[halves[1], 1] = w[halves[1]]
        st.apply(W)
        y = st.capture(4096)
        recs.append((th % 360, steer, float(np.angle(np.vdot(y[0], y[1])))))
    cal = AR.fit_monopulse(recs, man.geometry, halves, F_X)
    assert math.degrees(cal.delta_rad) == pytest.approx(20.0, abs=2.0)
    assert cal.kappa == pytest.approx(1.0, abs=0.1)
    # with the calibration the planted offset no longer biases the answer
    st.waves = [sim.PlaneWave(4.0, snr_db=25)]
    w = AR.steer_weights(man, 0.0, F_X)
    W = np.zeros((8, 2), complex)
    W[halves[0], 0] = w[halves[0]]
    W[halves[1], 1] = w[halves[1]]
    st.apply(W)
    y = st.capture(4096)
    raw = AR.monopulse_estimate(y[0], y[1], 0.0, halves, man.geometry, F_X)
    fixed = AR.monopulse_estimate(y[0], y[1], 0.0, halves, man.geometry, F_X, cal=cal)
    assert abs(cd(fixed.theta_deg, 4.0)) < 0.5 < abs(cd(raw.theta_deg, 4.0))
    assert AR.MonopulseCal.from_dict(cal.to_dict()) == cal


def test_analog_without_a_steerer_takes_a_capture():
    man, st = _phaser(12.0, 15.0)
    sweep = AR.beamscan_sweep(st, man, PH.steering["beams_deg"], F_X, phase_step_deg=2.8125,
                              gain_step_db=0.5)
    r = AR.ArrayDF(PH).estimate({"powers": sweep["powers"], "weights": sweep["weights"],
                                 "noise_power": 1.0}, F_X)
    assert abs(cd(r.theta_deg, 12.0)) < 3.0


# -- hybrid --------------------------------------------------------------------

HY = AP.preset("hybrid_ura4x4_4sub_v800")
F8 = 851e6


@pytest.mark.parametrize("th", [25.0, 320.0])
def test_hybrid_grating_lobes_are_settled_by_re_steering(th):
    man = HY.manifold()
    st = sim.SteerableArraySim(man, [sim.PlaneWave(th, snr_db=10)], F8, n_outputs=4,
                               phase_step_deg=5.625, gain_step_db=0.5, seed=2)
    r = AR.ArrayDF(HY, steerer=st).estimate(f_hz=F8, theta0=0.0, sources=1)
    assert len(r.quality["ambiguity_candidates"]) >= 2
    assert r.quality["resolved_by"] == "resteer"
    assert abs(cd(r.theta_deg, th)) < 0.5
    assert r.estimator.startswith("hybrid-")


def test_hybrid_without_a_steerer_reports_the_candidates():
    man = HY.manifold()
    st = sim.SteerableArraySim(man, [sim.PlaneWave(25.0, snr_db=10)], F8, n_outputs=4, seed=2)
    subs = [s["members"] for s in HY.subarrays]
    Wa = st.apply(AR.subarray_weights(man, subs, 0.0, F8))
    Y = st.capture(1024)
    r = AR.ArrayDF(HY).estimate({"Y": Y, "W": Wa}, F8, sources=1)
    assert r.quality["resolved_by"] == "unresolved"
    assert any(abs(cd(c, 25.0)) < 1.0 for c in r.quality["ambiguity_candidates"])
    assert any("GRATING-LOBE AMBIGUITY" in c for c in r.caveats)


def test_hybrid_clamps_the_source_count_to_channels_minus_one():
    man = HY.manifold()
    st = sim.SteerableArraySim(man, [sim.PlaneWave(5.0, snr_db=15)], F8, n_outputs=4, seed=2)
    subs = [s["members"] for s in HY.subarrays]
    Wa = st.apply(AR.subarray_weights(man, subs, 0.0, F8))
    r = AR.hybrid_estimate(st.capture(1024), Wa, man, F8, sources=6, subarrays=subs)
    assert r.n_sources <= 3
    assert any("resolve at most 3" in c for c in r.caveats)


# -- faces and the digital path --------------------------------------------------

@pytest.mark.parametrize("th,face", [(100.0, 1), (260.0, 3), (10.0, 0), (175.0, 2)])
def test_independent_faces_hand_off_by_power(th, face):
    p = AP.preset("mast4x4_independent_faces")
    adf = AR.ArrayDF(p)
    x = sim.element_snapshots(adf.manifold, [sim.PlaneWave(th, snr_db=10)], F8, n=2048, seed=4)
    r = adf.estimate(x, F8, sources=1)
    assert r.quality["face"] == face
    assert abs(cd(r.theta_deg, th)) < 1.0


def test_the_digital_path_honours_the_connector_map_and_tags_the_profile():
    p = AP.preset("kraken5_uca")
    p.elements["channel_of_element"] = [4, 3, 2, 1, 0]
    adf = AR.ArrayDF(p)
    el = sim.element_snapshots(M.IdealManifold(M.uca(5, 0.18)), [sim.PlaneWave(60.0, snr_db=15)],
                               446e6, n=2048, seed=1)
    channels = np.empty_like(el)
    channels[[4, 3, 2, 1, 0]] = el                 # element e arrives on channel 4−e
    r = adf.estimate(channels, 446e6, sources=1)
    assert abs(cd(r.theta_deg, 60.0)) < 1.0
    assert r.quality["profile"] == p.id and r.quality["timing_tier"] == "T0"
    assert r.quality["architecture"] == "digital"
