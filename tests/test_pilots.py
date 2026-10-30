"""Pilot-aided bearings: the steering vector from the sync word the
transmitter must send (DF_P25_IMPROVEMENT_PLAN §4.3).

The correlator is atkdsp's DmrSync (native or its numpy twin); these tests
find it through `atkdf._atkdsp` and are SKIPPED, not failed, when atkdsp is
not beside this repo — pilots are optional and everything else runs without
them. The planted truth is the simulator's real 4FSK waveform (P25 C4FM with
the TIA-102 frame sync every LDU; DMR with BS / MS voice syncs).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from atkdf import _atkdsp, calibration, engine, pilots, sim
from atkdf import manifold as M
from atkdf.ports import BurstEvent

FACTORY, WHY = _atkdsp.sync_factory()
needs_sync = pytest.mark.skipif(FACTORY is None, reason=f"no sync correlator: {WHY}")

IN_CHANNEL_DB = 10 * math.log10(2_400_000 / 12_500)       # 22.8 dB


def err(a, t):
    return ((a - t + 180) % 360) - 180


def wideband(snr_in_channel_db: float) -> float:
    """The simulator's SNR is over the whole 2.4 MHz; a 12.5 kHz channel sees
    22.8 dB more."""
    return snr_in_channel_db - IN_CHANNEL_DB


def _stream_hits(s, f0, chunks=None, skey="t", protocol=""):
    """Run a simulator's frames straight through a PilotStage."""
    st = pilots.PilotStage(FACTORY, backend=WHY)
    hits = []
    for fr in s.frames():
        h = fr.header
        if not h.usable_for_df:
            continue
        x = s.geometry.element_order(calibration.apply(fr.iq(), h, f0_hz=f0))
        n0 = int(h.stream_sample0)
        cuts = chunks or [x.shape[1]]
        i = 0
        for c in cuts:
            hits += st.feed(skey, x[:, i:i + c], n0 + i, h.fs_hz, f0, protocol=protocol)
            i += c
    hits += st.finish(skey, 2_400_000.0, protocol=protocol)
    return hits


def test_pilot_doa_returns_a_planted_steering_vector():
    geom = M.uca(5, 0.18)
    man = M.IdealManifold(geom)
    rng = np.random.default_rng(1)
    a = man.response(77.0, 446e6)[:, 0]
    hits = [pilots.PilotHit(n_center=k, t_utc=0.0, score=0.6, protocol="p25", window=120,
                            confirmed=True,
                            amps=a * np.exp(1j * rng.uniform(0, 2 * np.pi))
                            + 0.05 * (rng.standard_normal(5) + 1j * rng.standard_normal(5)))
            for k in range(4)]
    res = pilots.pilot_doa(hits, man, 446e6, geom)
    assert res.estimator == "pilot" and abs(err(res.theta_deg, 77.0)) < 1.0
    assert res.quality["n_hits"] == 4 and res.quality["n_confirmed"] == 4
    assert abs(res.likelihood.sum() - 1.0) < 1e-9
    assert pilots.pilot_doa([], man, 446e6, geom) is None


@needs_sync
def test_the_simulated_p25_carries_its_frame_sync_every_ldu():
    e = sim.Emitter(bearing_deg=50.0, f0_hz=250_000.0, snr_db=wideband(10.0), waveform="p25")
    s = sim.Simulator(sim.Scenario(emitters=[e], seed=2), frames=23)      # 1 s keyed
    hits = [h for h in _stream_hits(s, 250_000.0, protocol="p25") if h.clean]
    assert len(hits) >= 4
    spacing = np.diff([h.n_center for h in hits]) / 2_400_000.0
    assert np.allclose(spacing, 0.180, atol=0.002)
    assert all(h.protocol == "p25" and h.name == "P25 frame sync" for h in hits)


@needs_sync
def test_with_no_protocol_given_the_clean_family_wins_and_partials_are_dropped():
    # both correlators run on a keyless target; DMR finds partial matches in
    # P25 payload — `select` keeps P25, and timing confirms the real syncs
    e = sim.Emitter(bearing_deg=50.0, f0_hz=250_000.0, snr_db=wideband(10.0), waveform="p25")
    s = sim.Simulator(sim.Scenario(emitters=[e], seed=2), frames=23)
    raw = _stream_hits(s, 250_000.0)
    assert {h.protocol for h in raw} == {"p25", "dmr"}
    kept = pilots.select(raw)
    assert {h.protocol for h in kept} == {"p25"}
    assert sum(h.periodic for h in kept) >= 4
    assert pilots.source_tag(raw) == ""                 # P25 has no BS/MS word


@needs_sync
def test_how_the_stream_is_chunked_does_not_change_the_hits():
    e = sim.Emitter(bearing_deg=140.0, f0_hz=-300_000.0, snr_db=wideband(8.0), waveform="p25")
    sc = sim.Scenario(emitters=[e], seed=3)
    whole = [h for h in _stream_hits(sim.Simulator(sc, frames=15), -300_000.0) if h.confirmed]
    split = [h for h in _stream_hits(sim.Simulator(sc, frames=15), -300_000.0,
                                     chunks=[37_000, 50_000, 33_000]) if h.confirmed]
    assert len(whole) == len(split) >= 2
    for a, b in zip(whole, split):
        assert abs(a.n_center - b.n_center) <= 1
        va, vb = a.amps / a.amps[0], b.amps / b.amps[0]
        assert np.max(np.abs(va - vb)) < 2e-3


def _engine(s, est="music", stage=True, **kw):
    mode = engine.PlatformMode.static()
    mode.integrate, mode.survey, mode.estimator = True, False, est
    st = pilots.PilotStage(FACTORY, backend=WHY) if stage else None
    return engine.Engine(geometry=s.geometry, manifold=M.IdealManifold(s.geometry),
                         positioning=s.positioning, mode=mode, pilots=st, **kw)


def _band_events(s, fc_hz, protocol=""):
    """What ATK's DF tab sends: one target band per frame, no identity."""
    out = {}
    for k in range(s.n_frames):
        n0 = k * s.samples_per
        out[n0] = [BurstEvent(protocol=protocol, system="", id="", t_utc=s.t0,
                              n0=n0, n1=n0 + s.samples_per, fc_hz=fc_hz, bw_hz=12_500.0)]
    return out


def _run(s, eng, events):
    recs = []
    for fr in s.frames():
        recs += eng.process(fr, events=events.get(fr.header.stream_sample0, []))
    return recs + eng.flush()


@needs_sync
def test_a_p25_handset_at_minus_3_db_per_element_gets_a_sync_aided_bearing():
    e = sim.Emitter(bearing_deg=212.0, f0_hz=250_000.0, snr_db=wideband(-3.0), waveform="p25")
    s = sim.Simulator(sim.Scenario(emitters=[e], seed=4), frames=23)
    eng = _engine(s)
    recs = _run(s, eng, _band_events(s, s.sc.fc_hz + 250_000.0, protocol="p25"))
    assert len(recs) == 1
    r = recs[0]
    # clean channel: the whole-burst covariance is the more precise, and the
    # sync confirms it is THIS emitter
    assert r.quality["method"] == "pilot-confirmed" and r.estimator == "music"
    assert r.quality["pilot_n_hits"] >= 3 and r.quality["pilot_protocol"] == "p25"
    assert abs(err(r.theta_rel_deg, 212.0)) < 2.0
    assert abs(err(r.quality["pilot_theta_deg"], 212.0)) < 4.0
    assert r.quality["integration"] == "burst"


@needs_sync
def test_a_stronger_co_channel_signal_pulls_the_covariance_but_not_the_pilot():
    # the target: P25 at 60° (+5 dB in channel); an interferer 3 dB STRONGER on
    # the same channel at 200°, not P25 (no sync word). Bartlett — the
    # display's estimator — goes to the stronger signal; the sync does not.
    tgt = sim.Emitter(bearing_deg=60.0, f0_hz=200_000.0, snr_db=wideband(5.0), waveform="p25")
    jam = sim.Emitter(bearing_deg=200.0, f0_hz=200_000.0, snr_db=wideband(8.0))
    s = sim.Simulator(sim.Scenario(emitters=[tgt, jam], seed=5), frames=23)
    eng = _engine(s, est="bartlett")
    recs = _run(s, eng, _band_events(s, s.sc.fc_hz + 200_000.0, protocol="p25"))
    assert len(recs) == 1
    r = recs[0]
    assert r.quality["method"] in ("pilot", "pilot-selected")
    assert abs(err(r.theta_rel_deg, 60.0)) < 5.0
    assert abs(err(r.quality["covariance_theta_deg"], 60.0)) > 15.0
    assert any("sync" in n for n in r.notes)


@needs_sync
def test_music_with_two_solutions_lets_the_sync_pick_the_target_one():
    # MUSIC resolves both signals; the stronger one is its first answer, and
    # the sync word picks the other — the covariance's precision, the
    # interferer set aside
    tgt = sim.Emitter(bearing_deg=60.0, f0_hz=200_000.0, snr_db=wideband(5.0), waveform="p25")
    jam = sim.Emitter(bearing_deg=200.0, f0_hz=200_000.0, snr_db=wideband(8.0))
    s = sim.Simulator(sim.Scenario(emitters=[tgt, jam], seed=5), frames=23)
    eng = _engine(s, est="music")
    r = _run(s, eng, _band_events(s, s.sc.fc_hz + 200_000.0, protocol="p25"))[0]
    assert abs(err(r.theta_rel_deg, 60.0)) < 3.0
    assert r.quality["method"] in ("pilot-selected", "pilot-confirmed", "pilot")


@needs_sync
@pytest.mark.parametrize("wave, tag", [("dmr_bs", "repeated"), ("dmr_ms", "direct")])
def test_a_dmr_sync_word_supplies_the_path_tag_when_the_decoder_gave_none(wave, tag):
    e = sim.Emitter(bearing_deg=100.0, f0_hz=-200_000.0, snr_db=wideband(10.0), waveform=wave)
    s = sim.Simulator(sim.Scenario(emitters=[e], seed=6), frames=15)
    eng = _engine(s)
    recs = _run(s, eng, _band_events(s, s.sc.fc_hz - 200_000.0))
    assert recs
    r = recs[0]
    assert r.path_tag == tag
    assert "sync" in r.quality.get("path_evidence", "")
    assert abs(err(r.theta_rel_deg, 100.0)) < 3.0


def test_without_a_correlator_the_engine_runs_without_pilots(monkeypatch):
    monkeypatch.setattr(_atkdsp, "sync_factory", lambda *a, **k: (None, "test: none here"))
    st = pilots.PilotStage()
    assert not st.available and "none here" in st.describe()
    e = sim.Emitter(bearing_deg=30.0, f0_hz=100_000.0, snr_db=wideband(10.0), waveform="p25")
    s = sim.Simulator(sim.Scenario(emitters=[e], seed=7), frames=10)
    mode = engine.PlatformMode.static()
    mode.integrate, mode.survey = True, False
    eng = engine.Engine(geometry=s.geometry, positioning=s.positioning, mode=mode, pilots=st)
    recs = _run(s, eng, _band_events(s, s.sc.fc_hz + 100_000.0))
    assert len(recs) == 1 and recs[0].estimator == "music"
    assert recs[0].quality["method"] == "covariance"
    assert abs(err(recs[0].theta_rel_deg, 30.0)) < 2.0


@needs_sync
def test_the_sync_word_opens_the_squelch_on_a_weak_handset():
    # 0 dB per element: about 3 dB over the floor in the band, under a 6 dB
    # energy gate. Energy alone squelches it; the sync word says it is there.
    e = sim.Emitter(bearing_deg=145.0, f0_hz=200_000.0, snr_db=wideband(0.0), waveform="p25")
    sc = sim.Scenario(emitters=[e], seed=3)
    s = sim.Simulator(sc, frames=23)
    gated = _run(s, _engine(s, stage=False, gate_snr_db=6.0),
                 _band_events(s, sc.fc_hz + 200_000.0, protocol="p25"))
    assert gated == []                                   # energy squelch: nothing
    s = sim.Simulator(sc, frames=23)
    eng = _engine(s, gate_snr_db=6.0)
    recs = _run(s, eng, _band_events(s, sc.fc_hz + 200_000.0, protocol="p25"))
    assert len(recs) == 1
    assert abs(err(recs[0].theta_rel_deg, 145.0)) < 3.0
    assert recs[0].quality["method"].startswith("pilot")
    # held open between syncs: (nearly) the whole 1 s transmission counts,
    # not only the frames a sync word happened to fall in
    assert recs[0].quality["integrated_ms"] >= 800.0

