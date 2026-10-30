"""Burst integration: one bearing per transmission, the whole burst in it.

Planted truth through the real engine (`PlatformMode.integrate`): a burst
spanning many frames comes back as ONE record with every frame's snapshots,
bursts end on a gap / the length cap / a turn / a retune / a flush, two TDMA
slots stay two radios, and integrating beats the per-frame bearings it
replaces at low SNR.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from atkdf import engine, integrate, sim
from atkdf import manifold as M
from atkdf.atkq import Flag, Frame
from atkdf.ports import Fix2D


def err(a, t):
    return ((a - t + 180) % 360) - 180


def _engine(s, mode=None, positioning=None, **kw):
    mode = mode or engine.PlatformMode.static()
    return engine.Engine(geometry=s.geometry, manifold=M.IdealManifold(s.geometry),
                         positioning=positioning or s.positioning, mode=mode,
                         site_id="alpha", **kw)


def _run(s, eng, flush=True):
    by = {}
    for e in s.events():
        by.setdefault(e.n0, []).append(e)
    recs, statuses = [], []
    for fr in s.frames():
        recs += eng.process(fr, events=by.get(fr.header.stream_sample0, []))
        statuses.append(dict(eng.last_status))
    if flush:
        recs += eng.flush()
    return recs, statuses


def _integrating(**kw):
    m = engine.PlatformMode.static()
    m.integrate = True
    m.survey = False                   # as ATK's DF tab runs it: no target, no bearing
    for k, v in kw.items():
        setattr(m, k, v)
    return m


def test_a_burst_over_many_frames_is_one_bearing_with_all_its_snapshots():
    sc = sim.Scenario(emitters=[sim.Emitter(bearing_deg=110.0, f0_hz=300_000.0, snr_db=0.0,
                                            protocol="dmr", system="1", id="473",
                                            sync_type="ms")], heading_deg=30.0, seed=5)
    s = sim.Simulator(sc, frames=13)               # 3 cal frames + 10 keyed (0.5 s)
    eng = _engine(s, mode=_integrating())
    recs, _ = _run(s, eng)
    assert len(recs) == 1
    r = recs[0]
    assert r.emitter_key == "dmr:1:473"
    assert abs(err(r.theta_rel_deg, 110.0)) < 1.0
    assert abs(err(r.theta_abs_deg, 140.0)) < 1.0
    assert r.quality["integration"] == "burst"
    assert r.quality["integrated_frames"] == 10
    assert r.quality["burst_end"] == "flush"
    assert 450.0 <= r.quality["integrated_ms"] <= 500.0
    # every frame's snapshots are in it: ten frames' worth, not one
    per_frame = engine.Engine(geometry=s.geometry, positioning=s.positioning,
                              mode=engine.PlatformMode.static())
    one, _ = _run(sim.Simulator(sc, frames=4), per_frame, flush=False)
    assert r.quality["n_snapshots"] >= 9 * one[-1].quality["n_snapshots"]
    assert r.fusable and r.path_tag == "direct"


def test_integration_off_keeps_one_bearing_per_frame():
    sc = sim.Scenario(emitters=[sim.Emitter(bearing_deg=70.0, f0_hz=200_000.0, snr_db=10.0,
                                            protocol="dmr", system="1", id="9")], seed=2)
    s = sim.Simulator(sc, frames=8)
    recs, _ = _run(s, _engine(s), flush=True)        # flush is harmless with nothing open
    assert len(recs) >= 3
    assert all("integration" not in r.quality for r in recs)
    assert all(r.sample_span[1] - r.sample_span[0] == s.samples_per for r in recs)


def test_a_gap_ends_the_burst_and_the_next_keyup_is_a_new_one():
    # keyed 0.3 s, silent 0.3 s, keyed again: two transmissions, two bearings
    e = sim.Emitter(bearing_deg=200.0, f0_hz=-150_000.0, snr_db=5.0, protocol="dmr",
                    system="1", id="5", period_s=0.6, duty=0.5)
    s = sim.Simulator(sim.Scenario(emitters=[e], seed=3), frames=27)
    eng = _engine(s, mode=_integrating())
    recs, _ = _run(s, eng)
    assert len(recs) >= 2
    assert recs[0].quality["burst_end"] == "gap"
    assert recs[0].sample_span[1] < recs[1].sample_span[0]
    for r in recs:
        assert abs(err(r.theta_rel_deg, 200.0)) < 2.0


def test_the_length_cap_gives_a_bearing_at_least_that_often():
    e = sim.Emitter(bearing_deg=15.0, f0_hz=100_000.0, snr_db=10.0, protocol="dmr",
                    system="1", id="7")
    s = sim.Simulator(sim.Scenario(emitters=[e], seed=4), frames=27)     # 1.2 s keyed
    eng = _engine(s, mode=_integrating(max_integration_s=0.4))
    recs, _ = _run(s, eng)
    assert len(recs) >= 3
    assert all(r.quality["integrated_ms"] <= 450.0 for r in recs)
    assert recs[0].quality["burst_end"] == "max_integration"


def test_two_slots_are_two_integrated_bearings():
    sc = sim.Scenario(emitters=[
        sim.Emitter(bearing_deg=40.0, f0_hz=300_000.0, snr_db=15.0, protocol="dmr",
                    system="1", id="473", slot=1, period_s=0.1, duty=0.5, phase0=0.0),
        sim.Emitter(bearing_deg=250.0, f0_hz=300_000.0, snr_db=15.0, protocol="dmr",
                    system="1", id="912", slot=2, period_s=0.1, duty=0.5, phase0=0.5)],
        seed=11)
    s = sim.Simulator(sc, frames=15)
    eng = _engine(s, mode=_integrating())
    recs, _ = _run(s, eng)
    by = {r.emitter_key: r for r in recs}
    assert set(by) == {"dmr:1:473", "dmr:1:912"}
    assert abs(err(by["dmr:1:473"].theta_rel_deg, 40.0)) < 2.0
    assert abs(err(by["dmr:1:912"].theta_rel_deg, 250.0)) < 2.0


class _Turning:
    """A vehicle turning at 20°/s, its array heading known (dual-antenna GNSS)."""

    def __init__(self, t0, rate_dps=20.0):
        self.t0, self.rate = t0, rate_dps

    def at(self, t):
        return Fix2D(lat=38.7, lon=-77.58, t_utc=t, course_deg=None, speed_mps=10.0,
                     heading_deg=(self.rate * (t - self.t0)) % 360.0, heading_sigma_deg=0.3)


def test_a_turn_ends_the_burst_on_a_moving_platform():
    e = sim.Emitter(bearing_deg=90.0, f0_hz=100_000.0, snr_db=15.0, protocol="dmr",
                    system="1", id="3")
    s = sim.Simulator(sim.Scenario(emitters=[e], seed=6), frames=23)       # 1 s keyed
    mode = engine.PlatformMode.mobile()
    mode.integrate = True
    mode.survey = False
    eng = _engine(s, mode=mode, positioning=_Turning(s.t0))
    recs, _ = _run(s, eng)
    ends = [r.quality["burst_end"] for r in recs]
    assert "turned" in ends
    # each bearing integrated across at most ~5° of turn (0.25 s at 20°/s)
    assert all(r.quality["integrated_ms"] <= 300.0 for r in recs if r.quality["burst_end"] == "turned")


def test_the_interim_needle_moves_while_the_burst_is_still_open():
    e = sim.Emitter(bearing_deg=300.0, f0_hz=0.0 + 250_000.0, snr_db=5.0, protocol="dmr",
                    system="1", id="8")
    s = sim.Simulator(sim.Scenario(emitters=[e], seed=7), frames=10)
    eng = _engine(s, mode=_integrating())
    recs, statuses = _run(s, eng, flush=False)
    assert recs == []                                   # nothing finished yet
    live = [st for st in statuses if st.get("interim")]
    assert len(live) == 7
    assert live[-1]["integrated_ms"] > live[0]["integrated_ms"]
    assert abs(err(live[-1]["theta_rel_deg"], 300.0)) < 2.0
    assert eng.open_bursts == 1
    assert len(eng.flush()) == 1 and eng.open_bursts == 0


def test_a_retune_finishes_the_open_burst():
    e = sim.Emitter(bearing_deg=33.0, f0_hz=100_000.0, snr_db=10.0, protocol="dmr",
                    system="1", id="4")
    s = sim.Simulator(sim.Scenario(emitters=[e], seed=8), frames=10)
    eng = _engine(s, mode=_integrating())
    by = {ev.n0: [ev] for ev in s.events()}
    frames = list(s.frames())
    recs = []
    for fr in frames[:7]:
        recs += eng.process(fr, events=by.get(fr.header.stream_sample0, []))
    assert recs == [] and eng.open_bursts == 1
    h = frames[7].header
    retuned = Frame(replace(h, flags=h.flags | int(Flag.SEGMENT_START), segment=1),
                    frames[7].payload)
    recs += eng.process(retuned, events=[])
    assert len(recs) == 1 and recs[0].quality["burst_end"] == "retune"


def test_integrating_the_burst_beats_its_per_frame_bearings_at_low_snr():
    # -18 dB wideband = about +5 dB in a 12.5 kHz channel: per-frame bearings
    # scatter; the burst's one bearing sits closer to the truth, with a
    # smaller stated sigma
    sc = sim.Scenario(emitters=[sim.Emitter(bearing_deg=123.0, f0_hz=200_000.0,
                                            snr_db=-18.0, protocol="dmr", system="1",
                                            id="1")], seed=21)
    s1 = sim.Simulator(sc, frames=23)
    per_frame, _ = _run(s1, _engine(s1), flush=False)
    s2 = sim.Simulator(sc, frames=23)
    whole, _ = _run(s2, _engine(s2, mode=_integrating()))
    assert len(whole) == 1 and len(per_frame) >= 15
    rms = float(np.sqrt(np.mean([err(r.theta_rel_deg, 123.0) ** 2 for r in per_frame])))
    assert abs(err(whole[0].theta_rel_deg, 123.0)) <= max(rms, 0.5)
    from atkdf import doa
    crb_one = np.median([doa.cramer_rao_deg(1, r.quality["n_snapshots"], s1.geometry,
                                            446e6, 123.0) for r in per_frame])
    crb_all = doa.cramer_rao_deg(1, whole[0].quality["n_snapshots"], s1.geometry, 446e6, 123.0)
    assert crb_all < crb_one / 3.0


def test_burst_keys_separate_identity_slot_and_frequency():
    from atkdf.ports import BurstEvent
    a = BurstEvent("dmr", "1", "473", 0.0, slot=1)
    b = BurstEvent("dmr", "1", "473", 0.0, slot=2)
    c = BurstEvent("", "", "", 0.0)
    assert integrate.burst_key(a, 0, 446e6, 12500) != integrate.burst_key(b, 0, 446e6, 12500)
    assert integrate.burst_key(c, 0, 446.30001e6, 12500) == integrate.burst_key(c, 0, 446.30003e6, 12500)
    assert integrate.burst_key(c, 0, 446.3e6, 12500) != integrate.burst_key(c, 1, 446.3e6, 12500)
