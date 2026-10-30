"""Timing tiers, the timing state machine and the network payload policy
(plan §11.4.6)."""

from __future__ import annotations

import math

import pytest

from atkdf import arrayprofile as AP
from atkdf import timing as T
from atkdf.records import BearingRecord


def test_tiers_and_profiles():
    assert T.tier_index("T4") == 4
    with pytest.raises(ValueError):
        T.tier_index("T9")
    st = T.TimingState.from_profile(AP.preset("x310_twinrx_uca4"))
    assert st.tier == "T2" and st.locked_sigma_ns == 50
    assert T.TimingState.from_profile(AP.preset("kraken5_uca")).state == "free"


def test_gpsdo_locks_goes_to_holdover_and_its_error_grows():
    st = T.TimingState(tier="T2", holdover_spec="x3x0_ocxo")
    st.update(True, True, "gpsdo", now=0.0)
    assert st.state == "locked" and st.sigma_ns(0.0) == pytest.approx(50.0)
    assert st.tdoa_ok(0.0)
    st.update(False, None, "gpsdo", now=100.0)
    assert st.state == "holdover"
    # X3x0 OCXO: < 20 µs over 3 h ≈ 1.85 ns/s
    rate = 20_000.0 / 10_800.0
    assert st.sigma_ns(100.0 + 600.0) == pytest.approx(math.hypot(50.0, rate * 600.0))
    assert st.tdoa_ok(100.0 + 300.0)                  # 5 min out: ≈ 0.56 µs, under 1 µs
    assert not st.tdoa_ok(100.0 + 600.0)              # 10 min out: ≈ 1.1 µs, past it
    st.update(True, True, "gpsdo", now=4000.0)
    assert st.state == "locked" and st.holdover_s(4000.0) == 0.0
    assert [t[2] for t in st.transitions] == ["locked", "holdover", "locked"]


def test_a_gpsdo_that_never_locked_is_host_time():
    st = T.TimingState(tier="T2", holdover_spec="b2x0_tcxo")
    st.update(False, None, "", now=0.0)
    assert st.state == "free" and not st.tdoa_ok(0.0)
    assert st.sigma_ns(0.0) == T.LOCKED_SIGMA_NS["T1"]


def test_b210_holdover_is_flagged_as_unmeasured():
    st = T.TimingState(tier="T2", holdover_spec="b2x0_tcxo")
    st.update(True, True, now=0.0)
    st.update(False, None, now=10.0)
    h = st.health(now=20.0)
    assert h["holdover_rate_measured"] is False and h["state"] == "holdover"
    assert h["sigma_ns"] == pytest.approx(math.hypot(50.0, 750.0), abs=0.1)


def test_payload_policy_bearings_always_iq_only_when_cued_and_timed():
    a = T.TimingState(tier="T2", holdover_spec="x3x0_ocxo")
    b = T.TimingState(tier="T2", holdover_spec="x3x0_ocxo")
    a.update(True, True, now=0.0)
    b.update(True, True, now=0.0)
    kinds, why = T.allowed_payloads(a, b, cued=False, now=0.0)
    assert kinds == {"A"}
    kinds, why = T.allowed_payloads(a, b, cued=True, now=0.0)
    assert kinds == {"A", "B"}
    kraken = T.TimingState(tier="T0")
    kinds, why = T.allowed_payloads(a, kraken, cued=True, now=0.0)
    assert kinds == {"A"} and "peer" in why
    kinds, _ = T.allowed_payloads(a, b.health(0.0), cued=True, now=0.0)
    assert kinds == {"A", "B"}
    assert "C" not in kinds


def test_payload_rates_match_the_plan():
    assert T.payload_bps("B", 1, 25_000.0, "sc16") == pytest.approx(0.8e6)
    assert T.payload_bps("B", 1, 25_000.0, "sc8") == pytest.approx(0.4e6)
    assert T.payload_bps("C", 4, 25e6, "sc16") == pytest.approx(3.2e9)
    ok, bps = T.fits_link({"A", "B"}, 20e6, n_elements=8)
    assert ok and bps == pytest.approx(8 * 0.8e6 + 20_000.0)
    assert not T.fits_link({"B"}, 5_000.0, n_elements=1)[0]    # a LoRa link: bearings only


def test_tdoa_planning_number_for_p25():
    # c/(4B) at 12.5 kHz ≈ 6 km per snapshot (lit2 §6.6)
    assert T.tdoa_range_sigma_m(12_500.0, 10.0) == pytest.approx(5996.0, rel=1e-3)
    assert T.tdoa_range_sigma_m(12_500.0, 10.0, integration_s=2.0) < 300.0


def test_records_carry_the_tier_and_old_logs_read_as_t0():
    r = BearingRecord(t_utc=1.0, theta_rel_deg=10.0, sigma_deg=3.0, estimator="music",
                      timing_tier="T2", t_utc_sigma_ns=50.0)
    back = BearingRecord.from_json(r.to_json())
    assert back.timing_tier == "T2" and back.t_utc_sigma_ns == 50.0
    old = '{"t_utc": 1.0, "theta_rel_deg": 5.0, "sigma_deg": 2.0, "estimator": "music"}'
    assert BearingRecord.from_json(old).timing_tier == "T0"
