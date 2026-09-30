"""Applying atkdaq's calibration: the weight AND the fractional-delay slope,
which is the trap the wire-format header warns about."""

from __future__ import annotations

import numpy as np
import pytest

from atkdf import atkq, calibration
from atkdf import manifold as M


def _frame(weights, fracs, iq, cal=2, fs=2_400_000, fc=446_000_000):
    chans = tuple(atkq.Channel(delay=0, frac_delay=float(f), weight=complex(w))
                  for w, f in zip(weights, fracs))
    h = atkq.Header(n_channels=len(weights), fs_hz=fs, samples_per_ch=iq.shape[1],
                    channels=chans, cal_state=cal, fc_hz=fc, device_kind=2)
    return atkq.Frame(h, atkq.encode_frame(h, iq))


def test_applying_weight_alone_is_wrong_off_centre():
    # one plane wave across two channels, channel 1 delayed by a known frac and
    # scaled by a known weight; calibration must undo BOTH across the band
    fs, n = 2_400_000, 8192
    tau = 0.4
    f0 = 500_000.0                      # 500 kHz off centre
    t = np.arange(n)
    rng = np.random.default_rng(0)
    # a NARROWBAND signal at f0 (a tone plus a little close-in noise): its
    # energy sits at f0, so leaving the slope in shows as a real phase offset
    # 2*pi*f0*tau/fs there. (A white signal would average the ramp away.)
    base = (0.15 * np.exp(2j * np.pi * 800.0 * t / fs)
            + 0.01 * (rng.standard_normal(n) + 1j * rng.standard_normal(n)))
    x0 = base * np.exp(2j * np.pi * f0 * t / fs)
    # channel 1 = channel 0 through gain 0.5<30deg and a frac delay tau
    X0 = np.fft.fft(x0)
    f = np.fft.fftfreq(n, 1.0 / fs)
    g = 0.5 * np.exp(1j * np.radians(30.0))
    x1 = np.fft.ifft(X0 * g * np.exp(-2j * np.pi * f * tau / fs))
    iq = np.stack([x0, x1]).astype(np.complex64) * 3    # into ADC range, no clip
    # header reports the INVERSE: w = 1/g, frac = tau
    fr = _frame([1 + 0j, 1 / g], [0.0, tau], iq)

    full = calibration.calibrate(fr)                    # weight + slope across band
    weight_only = fr.iq() * fr.header.weights[:, None]  # the trap
    # after the full correction the two channels match to a small residual
    err_full = np.degrees(np.angle(np.vdot(full[0], full[1])))
    err_wonly = np.degrees(np.angle(np.vdot(weight_only[0], weight_only[1])))
    assert abs(err_full) < 1.0
    assert abs(err_wonly) > 10.0            # applying w alone leaves the slope


def test_narrowband_factor_matches_the_full_correction_at_its_offset():
    fs, n = 2_400_000, 4096
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((3, n)) + 1j * rng.standard_normal((3, n))).astype(np.complex64) * 0.03
    fr = _frame([1, 0.8 - 0.2j, 1.3 + 0.1j], [0.0, 0.3, -0.25], x)
    f0 = 200_000.0
    narrow = calibration.calibrate(fr, f0_hz=f0)
    c = calibration.factors(fr.header, f0)
    assert np.allclose(narrow, fr.iq() * c[:, None], atol=1e-5)


def test_passport_is_built_from_the_header():
    x = (np.random.default_rng(2).standard_normal((5, 256))
         + 1j * np.random.default_rng(3).standard_normal((5, 256))).astype(np.complex64) * 0.03
    fr = _frame([1, 1, 1, 1, 1], [0, 0.1, 0.2, 0.3, 0.4], x)
    p = calibration.passport_of(fr.header, geometry_id="uca5", site_id="alpha")
    assert p.site_id == "alpha" and p.geometry_id == "uca5"
    assert p.frac_delays == pytest.approx([0, 0.1, 0.2, 0.3, 0.4])
    assert p.device == "synthetic"
    assert "/" in p.uid


def test_cal_age_verdict_tracks_the_mode():
    x = np.zeros((2, 128), np.complex64)
    def fr(age_ms):
        f = _frame([1, 1], [0, 0], x)
        object.__setattr__(f.header, "cal_age_ms", age_ms)
        return f.header
    assert calibration.cal_age_verdict(fr(200_000), "mobile")[0] == "aging"   # 200 s
    assert calibration.cal_age_verdict(fr(200_000), "static")[0] == "fresh"   # < 900 s
    assert calibration.cal_age_verdict(fr(10 * 60 * 1000), "mobile")[0] == "stale"  # 600 s


def test_the_coherence_monitor_flags_a_drift():
    mon = calibration.CoherenceMonitor(threshold_deg=5.0, persist=2)
    base = np.array([1, 1j, -1, -1j, 1], complex)
    assert mon.observe(base, "p1")["state"] == "baseline"
    assert mon.observe(base * np.exp(1j * 0.01), "p1")["state"] == "steady"
    drift = base.copy()
    drift[2] *= np.exp(1j * np.radians(15.0))       # element 2 walks off
    mon.observe(drift, "p1")
    out = mon.observe(drift, "p1")
    assert out["state"] == "recalibrate" and out["element"] == 2
    assert mon.needs_cal and "element 2" in mon.sentence()
    # a new passport resets the baseline
    assert mon.observe(base, "p2")["state"] == "baseline" and not mon.needs_cal
