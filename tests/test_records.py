"""The data model: keys, likelihood coding, JSON round-trips, the fusable rule."""

from __future__ import annotations

import numpy as np
import pytest

from atkdf.records import (BearingRecord, EmitterKey, Passport, circular_mean_deg,
                          decode_likelihood, encode_likelihood, wrap180)


def test_emitter_key_parses_and_prints():
    k = EmitterKey("dmr", "colour1", "473")
    assert str(k) == "dmr:colour1:473"
    assert EmitterKey.parse("dmr:colour1:473") == k
    with pytest.raises(ValueError):
        EmitterKey.parse("473")


def test_likelihood_encodes_360_float16_round_trip():
    rng = np.random.default_rng(0)
    lik = np.abs(rng.standard_normal(360)).astype(np.float32)
    lik /= lik.sum()
    back = decode_likelihood(encode_likelihood(lik))
    assert back.shape == (360,)
    assert np.allclose(back, lik, atol=1e-3)
    with pytest.raises(ValueError):
        encode_likelihood(np.zeros(180))


def test_a_bearing_record_round_trips_through_json():
    lik = np.ones(360) / 360
    r = BearingRecord(t_utc=1790000000.5, theta_rel_deg=73.2, sigma_deg=3.1,
                      estimator="music", emitter_key="dmr:1:473", slot=1,
                      sync_type="ms", heading_deg=30.0, theta_abs_deg=103.2,
                      likelihood=lik, lat=38.7, lon=-77.5, sample_span=(1000, 2200),
                      quality={"snr_db": 21.0}, platform_mode="mobile")
    r2 = BearingRecord.from_json(r.to_json())
    assert r2.theta_rel_deg == pytest.approx(73.2)
    assert r2.emitter_key == "dmr:1:473" and r2.slot == 1
    assert r2.sample_span == (1000, 2200)
    assert np.allclose(r2.likelihood, lik, atol=1e-3)
    assert r2.key == EmitterKey("dmr", "1", "473")


def test_a_repeated_burst_is_not_fusable():
    common = dict(t_utc=1.0, theta_rel_deg=10.0, sigma_deg=2.0, estimator="music",
                  heading_deg=0.0, theta_abs_deg=10.0, lat=38.7, lon=-77.5)
    assert BearingRecord(path_tag="direct", **common).fusable
    assert not BearingRecord(path_tag="repeated", **common).fusable
    # no heading -> no absolute bearing -> not fusable
    assert not BearingRecord(path_tag="direct", t_utc=1.0, theta_rel_deg=10.0,
                             sigma_deg=2.0, estimator="music", lat=38.7, lon=-77.5).fusable


def test_absolute_likelihood_rotates_by_heading():
    lik = np.zeros(360)
    lik[20] = 1.0
    r = BearingRecord(t_utc=1.0, theta_rel_deg=20.0, sigma_deg=2.0, estimator="music",
                      heading_deg=50.0, theta_abs_deg=70.0, likelihood=lik,
                      lat=1.0, lon=1.0)
    ab = r.absolute_likelihood()
    assert int(np.argmax(ab)) == 70


def test_passport_uid_is_unique_across_runs_and_sites():
    a = Passport(session=111, passport=1, t_utc=1.0, fc_hz=446e6, fs_hz=2.4e6,
                 gain_tenths_db=280, delays=[0] * 5, weights=[[1, 0]] * 5,
                 frac_delays=[0] * 5, spread_deg=0.2, drops=[0] * 5, site_id="alpha")
    b = Passport(session=222, passport=1, t_utc=1.0, fc_hz=446e6, fs_hz=2.4e6,
                 gain_tenths_db=280, delays=[0] * 5, weights=[[1, 0]] * 5,
                 frac_delays=[0] * 5, spread_deg=0.2, drops=[0] * 5, site_id="bravo")
    assert a.uid != b.uid                       # same passport number, different run+site
    assert Passport.from_dict(a.to_dict()).uid == a.uid


def test_circular_mean_and_wrap():
    assert abs(wrap180(circular_mean_deg([350, 10]))) < 1e-6
    assert circular_mean_deg([80, 100]) == pytest.approx(90.0, abs=1e-6)
    assert wrap180(190) == pytest.approx(-170.0)
    assert wrap180(-190) == pytest.approx(170.0)
