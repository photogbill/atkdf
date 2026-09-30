"""Array geometry, the ideal manifold, its Fourier (EADF) form, and a measured
manifold fitted from snapshots and interpolated back."""

from __future__ import annotations

import numpy as np
import pytest

from atkdf import manifold as M


def test_uca5_geometry_and_spacing():
    g = M.uca(5, 0.18)
    assert g.n == 5 and g.kind == "uca"
    # adjacent spacing of a 5-point circle of r=0.18: 2 r sin(pi/5)
    assert g.adjacent_spacing_m() == pytest.approx(2 * 0.18 * np.sin(np.pi / 5), rel=1e-6)
    # 0.5 lambda at ~707 MHz
    assert g.max_unambiguous_hz() == pytest.approx(299_792_458.0 / (2 * g.adjacent_spacing_m()))
    # element 0 toward the nose (x forward), element 1 counter-clockwise (y left)
    assert g.positions[0, 0] == pytest.approx(0.18) and abs(g.positions[0, 1]) < 1e-9
    assert g.positions[1, 1] > 0


def test_a_uca5_is_not_centrosymmetric_but_a_ula_is():
    assert not M.uca(5, 0.18).centrosymmetric
    assert M.ula(5, 0.3).centrosymmetric


def test_spacing_refusal_names_the_frequency():
    g = M.uca(5, 0.18)
    assert g.check(450e6) == ""                    # fine at 450 MHz
    with pytest.raises(M.SpacingError) as e:
        g.check(1200e6)                            # aliases past ~707 MHz
    assert "MHz" in str(e.value)
    note = g.check(1200e6, override=True)          # allowed, with the caveat
    assert "λ" in note or "lambda" in note or "wavelength" in note.lower() or "λ" in note


def test_connector_map_reorders_channels():
    # element e is wired to channel (4-e): the array is cabled in reverse
    g = M.uca(5, 0.18, channel_of_element=(4, 3, 2, 1, 0))
    x = np.arange(5)[:, None] * np.ones((5, 3))    # channel c holds value c
    el = g.element_order(x)
    assert list(el[:, 0]) == [4, 3, 2, 1, 0]       # element order


def test_the_ideal_manifold_matches_its_own_fourier_series():
    g = M.uca(5, 0.18)
    man = M.IdealManifold(g)
    f = 450e6
    th = np.array([0.0, 37.0, 122.5, 250.0, 359.0])
    direct = man.response(th, f)
    G, q = man.eadf(f)
    fourier = G @ np.exp(1j * np.outer(q, np.radians(th)))
    assert np.allclose(direct, fourier, atol=1e-6)


def test_a_measured_manifold_recovers_a_perturbed_one():
    g = M.uca(5, 0.18)
    f = 450e6
    true = M.PerturbedManifold.random(g, gain_db=0.5, phase_deg=6.0, coupling_db=-22.0, seed=2)
    # dense noise-free "measurements" all the way round
    th = np.arange(0, 360, 2.0)
    V = np.stack([true.response(float(t), f)[:, 0] for t in th], axis=0)
    meas = M.MeasuredManifold.from_snapshots(g, f, th, V, bin_deg=2.0)
    assert meas.coverage_deg > 350
    # it should reproduce the perturbed response (up to the per-angle global
    # phase the array cannot observe) better than the ideal manifold does
    check = np.arange(1, 360, 11.0)
    def rel_err(man):
        e = []
        for t in check:
            a = man.response(float(t), f)[:, 0]
            b = true.response(float(t), f)[:, 0]
            a = a / a[0] * abs(a[0]); b = b / b[0] * abs(b[0])
            a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
            e.append(np.linalg.norm(a - b))
        return float(np.mean(e))
    assert rel_err(meas) < rel_err(M.IdealManifold(g)) / 3


def test_a_measured_manifold_saves_and_loads(tmp_path):
    g = M.uca(5, 0.18)
    f = 450e6
    true = M.PerturbedManifold.random(g, seed=1)
    th = np.arange(0, 360, 3.0)
    V = np.stack([true.response(float(t), f)[:, 0] for t in th], axis=0)
    meas = M.MeasuredManifold.from_snapshots(g, f, th, V, id="man_test",
                                             passport_uid="site/1/1")
    p = meas.save(tmp_path / "m.npz")
    back = M.MeasuredManifold.load(p)
    assert back.id == "man_test" and back.passport_uid == "site/1/1"
    assert np.allclose(back.response(np.array([10.0, 200.0]), f),
                       meas.response(np.array([10.0, 200.0]), f), atol=1e-9)
