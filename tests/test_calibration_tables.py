"""Step 3 of the phased-array build (plan §11.4.4): two calibration tables
with a per-hardware persistence policy, the on-air recalibration trigger,
the antenna table that interpolates across a band, the channel-health
detector, and the forward-backward fix for even circles."""

from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pytest

from atkdf import arrayprofile as AP
from atkdf import caltables as CT
from atkdf import doa, health, sim
from atkdf import manifold as M
from atkdf.engine import Engine


def cd(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


# -- receiver tables and the persistence policy -------------------------------

def _tone(n=4, seed=0):
    rng = np.random.default_rng(seed)
    g = 10 ** (rng.normal(0, 1, n) / 20) * np.exp(1j * rng.uniform(-np.pi, np.pi, n))
    return g


def test_a_receiver_table_makes_the_channels_agree():
    g = _tone()
    rows = np.array([g * np.exp(1j * p) for p in (0.1, 1.3, -2.0)])     # common phase varies
    t = CT.RxCalTable.from_measurement("x", 851e6, rows)
    x = np.outer(g, np.ones(8))
    y = t.apply(x)
    assert np.allclose(y / y[0], 1.0)
    assert t.residual_deg < 1e-6
    assert CT.RxCalTable.from_dict(t.to_dict()).weights.tolist() == t.weights.tolist()


def test_per_retune_hardware_needs_a_calibration_after_every_retune():
    store = CT.CalStore(None, AP.preset("kraken5_uca"))
    store.begin_power_cycle("boot1")
    store.note_retune(446e6)
    store.put(CT.RxCalTable.from_measurement("kraken5_uca", 446e6, np.ones(5)))
    assert store.lookup(446e6)[0] is not None
    store.note_retune(450e6)
    store.note_retune(446e6)
    t, why = store.lookup(446e6)
    assert t is None and "retuned" in why


def test_per_power_cycle_hardware_keeps_a_table_across_retunes_not_reboots():
    store = CT.CalStore(None, AP.preset("n321x4_ura2x4_v800"))
    store.begin_power_cycle("boot1")
    store.note_retune(851e6)
    store.put(CT.RxCalTable.from_measurement("n321x4_ura2x4_v800", 851e6, np.ones(8)))
    store.note_retune(770e6)
    store.note_retune(851e6)
    assert store.lookup(851e6)[0] is not None
    store.begin_power_cycle("boot2")
    t, why = store.lookup(851e6)
    assert t is None and "power cycle" in why


def test_across_power_cycles_hardware_reloads_its_tables_from_disk(tmp_path):
    p = AP.preset("x310_twinrx_uca4")
    s1 = CT.CalStore(tmp_path, p)
    s1.begin_power_cycle("boot1")
    s1.note_retune(851e6)
    s1.put(CT.RxCalTable.from_measurement(p.id, 851e6, _tone()))
    s2 = CT.CalStore(tmp_path, p)                     # a new session, same folder
    s2.begin_power_cycle("boot2")
    s2.note_retune(851e6)
    t, why = s2.lookup(851e6)
    assert t is not None and why == "usable"
    files = list(tmp_path.glob("rxcal_x310_twinrx_uca4_*.json"))
    assert len(files) == 1 and json.loads(files[0].read_text())["session_id"] == "boot1"


def test_the_reference_monitor_invalidates_a_table_that_drifted():
    p = AP.preset("x310_twinrx_uca4")
    store = CT.CalStore(None, p, threshold_deg=5.0, persist=3)
    store.begin_power_cycle()
    store.note_retune(851e6)
    store.put(CT.RxCalTable.from_measurement(p.id, 851e6, np.ones(4)))
    v = np.exp(1j * np.array([0.0, 0.3, -0.7, 1.1]))
    assert store.observe_reference(851e6, v)["state"] == "baseline"
    for _ in range(2):
        assert store.observe_reference(851e6, v)["state"] == "steady"
    moved = v * np.exp(1j * np.radians([0, 0, 12.0, 0]))
    for _ in range(3):
        out = store.observe_reference(851e6, moved)
    assert out["state"] == "recalibrate" and out["element"] == 2
    t, why = store.lookup(851e6)
    assert t is None and "invalidated" in why
    assert store.observe_reference(851e6, v, stationary=False)["state"] == "ignored"


def test_ptfe_cable_temperature_flags_a_recalibration():
    assert CT.thermal_phase_deg(1.0, 1e9, 55.0, "ptfe") == pytest.approx(3.8)
    p = AP.preset("x310_twinrx_uca4")
    p.cable_class = "ptfe"
    store = CT.CalStore(None, p, cable_m=3.0, thermal_threshold_deg=2.0)
    store.begin_power_cycle()
    store.note_retune(851e6)
    store.note_temperature(10.0)
    store.put(CT.RxCalTable.from_measurement(p.id, 851e6, np.ones(4)))
    store.note_temperature(15.0)
    assert store.lookup(851e6)[1] == "usable"           # 0.9°: below the bound
    store.note_temperature(30.0)                         # 20 °C: 3.5°
    t, why = store.lookup(851e6)
    assert t is not None and "recalibrate when convenient" in why


# -- the antenna table across a band --------------------------------------------

def _truth(seed=1):
    rng = np.random.default_rng(seed)
    g = M.uca(8, 0.2)
    offs = np.zeros((8, 3))
    offs[:, :2] = rng.normal(0, 0.02, (8, 2))
    truth = M.PerturbedManifold.random(replace(g, phase_centre_m=offs), gain_db=0.5,
                                       phase_deg=8, coupling_db=-22, seed=3)
    return g, offs, truth


def _measure(truth, f, seed=2):
    rng = np.random.default_rng(seed)
    th = np.arange(0, 360, 1.0)
    V = truth.response(th, f).T * np.exp(1j * rng.uniform(-np.pi, np.pi, th.size))[:, None]
    return th, V


def test_phase_centres_are_recovered_from_a_turn():
    g = M.uca(8, 0.2)
    rng = np.random.default_rng(7)
    offs = np.zeros((8, 3))
    offs[:, :2] = rng.normal(0, 0.02, (8, 2))
    truth = M.IdealManifold(replace(g, phase_centre_m=offs))
    th, V = _measure(truth, 851e6)
    fit = M.fit_phase_centres(g, 851e6, th, V)
    want = offs.copy()
    want[:, :2] -= want[:, :2].mean(axis=0)              # a common shift is unobservable
    assert np.max(np.abs(fit - want)) < 5e-4             # half a millimetre


def test_the_phase_centre_table_interpolates_across_the_800_band():
    g, _offs, truth = _truth()
    th, V = _measure(truth, 851e6)
    full = M.MeasuredManifold.from_snapshots(g, 851e6, th, V)
    res = M.MeasuredManifold.from_snapshots(g, 851e6, th, V, phase_centres=True)
    assert res.mode == "residual"
    worst = {"full": {}, "residual": {}}
    for f in (769e6, 806e6, 851e6):
        for name, man in (("full", full), ("residual", res)):
            e = []
            for t in (13.0, 77.0, 141.0, 222.0, 300.0):
                x = sim.element_snapshots(truth, [sim.PlaneWave(t, snr_db=20)], f, n=2048,
                                          seed=int(t))
                e.append(abs(cd(doa.estimate(x, man, f, sources=1).theta_deg, t)))
            worst[name][f] = max(e)
    assert worst["full"][851e6] < 0.2 and worst["residual"][851e6] < 0.2
    for f in (769e6, 806e6):
        assert worst["residual"][f] < 0.5
        assert worst["residual"][f] <= worst["full"][f] + 0.05
    assert any("phase centres removed" in c for c in res.caveats(769e6))


def test_the_residual_table_round_trips_and_its_eadf_matches(tmp_path):
    g, _offs, truth = _truth()
    th, V = _measure(truth, 851e6)
    res = M.MeasuredManifold.from_snapshots(g, 851e6, th, V, phase_centres=True)
    p = res.save(tmp_path / "ant.npz")
    back = M.MeasuredManifold.load(p)
    tt = np.array([5.0, 99.0, 250.0])
    assert back.mode == "residual"
    assert np.allclose(back.response(tt, 790e6), res.response(tt, 790e6))
    G, q = res.eadf(790e6)
    assert np.allclose(G @ np.exp(1j * np.outer(q, np.radians(tt))), res.response(tt, 790e6),
                       atol=1e-6)


# -- forward-backward on an even circle ------------------------------------------

def test_forward_backward_uses_the_point_reflection_not_index_reversal():
    assert M.uca(8, 0.2).reflection_permutation() == (4, 5, 6, 7, 0, 1, 2, 3)
    assert M.uca(8, 0.2, centre=True).reflection_permutation()[-1] == 8
    assert M.ula(4, 0.2).reflection_permutation() == (3, 2, 1, 0)
    assert M.uca(5, 0.18).reflection_permutation() is None
    g = M.uca(8, 0.2)
    truth = M.PerturbedManifold.random(g, gain_db=0.3, phase_deg=3, coupling_db=-30, seed=3)
    for t in (13.0, 41.0, 300.0):
        x = sim.element_snapshots(truth, [sim.PlaneWave(t, snr_db=20)], 851e6, n=2048,
                                  seed=int(t))
        r = doa.estimate(x, M.IdealManifold(g), 851e6, sources=1)     # FB on (ideal, even)
        assert abs(cd(r.theta_deg, t)) < 2.0
    with pytest.raises(ValueError, match="point reflection"):
        doa.covariance(np.ones((5, 4), complex), 446e6, geometry=M.uca(5, 0.18), fb=True)


def test_a_measured_manifold_turns_forward_backward_off():
    g, _offs, truth = _truth()
    th, V = _measure(truth, 851e6)
    full = M.MeasuredManifold.from_snapshots(g, 851e6, th, V)
    x = sim.element_snapshots(truth, [sim.PlaneWave(77.0, snr_db=20)], 851e6, n=2048, seed=1)
    r = doa.estimate(x, full, 851e6, sources=1)
    assert abs(cd(r.theta_deg, 77.0)) < 0.2


# -- channel health ----------------------------------------------------------------

def _cov(x):
    return x @ x.conj().T / x.shape[1]


def _array_x(seed=0, n_el=8, th=40.0):
    man = M.IdealManifold(M.uca(n_el, 0.2))
    return man, sim.element_snapshots(man, [sim.PlaneWave(th, snr_db=15)], 851e6, n=4096,
                                      seed=seed)


def test_health_passes_a_good_array():
    _man, x = _array_x()
    rep = health.ChannelHealth(8).check(_cov(x))
    assert rep.ok and rep.sentence() == "all channels healthy"


def test_health_finds_a_dead_channel():
    _man, x = _array_x()
    x[5] *= 1e-3
    rep = health.ChannelHealth(8).check(_cov(x))
    assert rep.bad == [5] and rep.status[5] == "dead"


def test_health_finds_a_channel_that_sees_only_its_own_noise():
    _man, x = _array_x()
    rng = np.random.default_rng(9)
    x[2] = (rng.standard_normal(x.shape[1]) + 1j * rng.standard_normal(x.shape[1])) * 4
    rep = health.ChannelHealth(8).check(_cov(x))
    assert rep.bad == [2] and rep.status[2] == "noisy"


def test_health_finds_a_wandering_channel_against_a_parked_reference():
    man, _x = _array_x()
    mon = health.ChannelHealth(8, wander_deg=10.0, persist=3)
    for k in range(6):
        x = sim.element_snapshots(man, [sim.PlaneWave(40.0, snr_db=20)], 851e6, n=2048, seed=k)
        x[6] *= np.exp(1j * np.radians(6.0 * k))           # element 6 slides 6°/look
        rep = mon.check(_cov(x), stationary_reference=True)
    assert rep.bad == [6] and rep.status[6] == "wandering"


def test_the_engine_drops_a_dead_element_and_says_so():
    g = M.uca(8, 0.2)
    man = M.IdealManifold(g)
    eng = Engine(geometry=g, manifold=man, health=health.ChannelHealth(8))
    x = sim.element_snapshots(man, [sim.PlaneWave(123.0, snr_db=15)], 851e6, n=4096, seed=3)
    x[3] = 0.0
    cov = doa.covariance(x, 851e6, geometry=g, fb=False)
    r = eng._estimate(cov, 851e6)
    assert r.quality["dropped_elements"] == [3]
    assert abs(cd(r.theta_deg, 123.0)) < 1.5
    clean = doa.estimate(doa.covariance(x[[0, 1, 2, 4, 5, 6, 7]], 851e6, fb=False),
                         man.subset([0, 1, 2, 4, 5, 6, 7]), 851e6)
    assert r.sigma_deg > clean.sigma_deg
    assert any("CHANNEL HEALTH" in c for c in r.caveats)


def test_fragility_note_only_for_non_uniform_layouts():
    assert health.fragility_note(M.uca(8, 0.2)) == ""
    assert "nested" in health.fragility_note(M.custom([[0, 0], [0.1, 0], [0.4, 0]]))


# -- the policy against hardware that behaves like each class ---------------------

def _bearing_with(table_w, bank, man, th=30.0, seed=0):
    x = sim.element_snapshots(man, [sim.PlaneWave(th, snr_db=20)], 851e6, n=2048, seed=seed)
    y = bank.apply(x) * np.asarray(table_w)[:, None]
    return doa.estimate(y, man, 851e6, sources=1).theta_deg


@pytest.mark.parametrize("pid,domains,on_retune,on_cycle,survives", [
    ("x310_twinrx_uca4", [0, 0, 0, 0], "keep", "keep", True),
    ("n321x4_ura2x4_v800", [0, 0, 1, 1, 2, 2, 3, 3], "keep", "random", False),
])
def test_the_policy_matches_what_the_hardware_does_after_a_reboot(pid, domains, on_retune,
                                                                 on_cycle, survives):
    p = AP.preset(pid)
    man = p.manifold()
    bank = sim.ReceiverBank(man.geometry.n, domains, on_retune, on_cycle, seed=4)
    store = CT.CalStore(None, p)
    store.begin_power_cycle("boot1")
    store.note_retune(851e6)
    store.put(CT.RxCalTable(p.id, 851e6, bank.truth(), "tone"))
    assert abs(cd(_bearing_with(store.lookup(851e6)[0].weights, bank, man), 30.0)) < 1.0
    bank.power_cycle()
    store.begin_power_cycle("boot2")
    table, why = store.lookup(851e6)
    stale_err = abs(cd(_bearing_with(store.tables[CT.lo_key(851e6)].weights, bank, man), 30.0))
    if survives:
        assert table is not None and stale_err < 1.0          # the table really still works
    else:
        assert table is None and "power cycle" in why        # the store refuses it …
        assert stale_err > 2.0                               # … and it really is wrong


def test_n310_external_lo_flips_180_on_retune():
    bank = sim.ReceiverBank(4, [0, 0, 1, 1], on_retune="flip180", seed=1)
    flips = set()
    for _ in range(20):
        before = bank.response()
        bank.retune()
        ratio = bank.response() / before
        flips.update(np.round(np.degrees(np.angle(ratio))).astype(int).tolist())
    assert flips <= {0, 180, -180} and len(flips) >= 2


def test_health_still_finds_a_dead_element_through_forward_backward_averaging():
    # finding #2: the engine hands health the RAW covariance, so FB (which
    # mixes an element with its mirror) cannot hide a dead channel
    from atkdf import doa, health
    from atkdf.engine import Engine
    g = M.ura(4, 4, 0.172, plane="horizontal")           # centrosymmetric → FB on
    man = M.IdealManifold(g)
    assert g.reflection_permutation() is not None
    x = sim.element_snapshots(man, [sim.PlaneWave(40.0, 0.0, snr_db=15)], 851e6, n=4096, seed=3)
    x[5] = 0.0                                            # kill element 5 (mirror is 10)
    cov = doa.covariance(x, 851e6, geometry=g)           # FB applied
    assert cov.fb and cov.R_raw is not None
    # health on the FB matrix would miss it; on the raw matrix it must not
    assert health.ChannelHealth(16).check(cov.R).bad == []          # the trap
    assert 5 in health.ChannelHealth(16).check(cov.raw).bad         # the truth
    eng = Engine(geometry=g, manifold=man, health=health.ChannelHealth(16))
    r = eng._estimate(cov, 851e6)
    assert r.quality["dropped_elements"] == [5]
    assert abs(((r.theta_deg - 40.0 + 180) % 360) - 180) < 2.0
