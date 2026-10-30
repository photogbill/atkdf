"""The contract ATK's df_live.configure() depends on (plan §11.4, step wiring):
build the live Engine from an array profile's geometry, manifold and timing
state, and a planted bearing comes back carrying the profile's timing tier.
If this passes, ATK's profile path works against this atkdf."""

from __future__ import annotations

import numpy as np

from atkdf import arrayprofile as AP
from atkdf import sim, timing
from atkdf import manifold as M
from atkdf.engine import Engine, PlatformMode


def err(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def _run(profile, bearing, f0=150_000.0, frames=24, snr=15.0):
    geom = profile.geometry()
    manifold = profile.manifold()
    tstate = timing.TimingState.from_profile(profile)
    if tstate.tier in ("T2", "T3", "T4", "T5"):
        tstate.update(True, True, "gpsdo", now=0.0)
    mode = PlatformMode.static()
    mode.estimator = profile.estimators()[0] if "music" not in profile.estimators() else "music"
    eng = Engine(geometry=geom, manifold=manifold, mode=mode, site_id="atk", timing=tstate)
    fc = profile.band_centre_hz
    sc = sim.Scenario(emitters=[sim.Emitter(bearing_deg=bearing, f0_hz=f0, snr_db=snr)],
                      fc_hz=fc, manifold=M.IdealManifold(geom), n_ch=geom.n)
    recs = []
    for fr in sim.Simulator(sc, frames=frames).frames():
        recs += eng.process(fr)
    recs += eng.flush()
    return recs


def test_kraken_profile_reproduces_the_legacy_bearing():
    recs = _run(AP.preset("kraken5_uca"), 70.0)
    assert recs and abs(err(recs[-1].theta_rel_deg, 70.0)) < 3.0
    assert recs[-1].timing_tier == "T0"


def test_a_usrp_profile_tags_bearings_with_its_timing_tier():
    recs = _run(AP.preset("x310_twinrx_uca4"), 40.0)
    assert recs and abs(err(recs[-1].theta_rel_deg, 40.0)) < 3.0
    r = recs[-1]
    assert r.timing_tier == "T2" and r.t_utc_sigma_ns == 50.0


def test_a_vertical_panel_profile_direction_finds_in_front():
    recs = _run(AP.preset("x310_twinrx_panel4_v800"), 20.0, f0=100_000.0, snr=18.0)
    assert recs and abs(err(recs[-1].theta_rel_deg, 20.0)) < 4.0


def test_the_profile_round_trips_through_the_presets_folder(tmp_path):
    AP.write_presets(tmp_path)
    good, bad = AP.load_dir(tmp_path)
    assert not bad and "kraken5_uca" in good
    recs = _run(good["kraken5_uca"], 125.0)
    assert recs and abs(err(recs[-1].theta_rel_deg, 125.0)) < 3.0
