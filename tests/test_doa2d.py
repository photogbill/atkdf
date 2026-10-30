"""2-D DoA: azimuth and elevation (plan §11.4.5). A vertical panel measures
both; a horizontal planar array measures azimuth and an up/down-ambiguous
elevation; a flat ring refuses elevation and says so."""

from __future__ import annotations

import numpy as np
import pytest

from atkdf import doa2d, sim
from atkdf import manifold as M

F = 851e6


def cd(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def _panel(rows=4, cols=4, b=0.0):
    g = M.ura(rows, cols, 0.172, plane="vertical", boresight_deg=b)
    return g, M.IdealManifold(g)


@pytest.mark.parametrize("az,el", [(10.0, 20.0), (330.0, 5.0), (40.0, 35.0), (-20.0 % 360, 0.0)])
def test_music2d_on_a_vertical_panel(az, el):
    g, man = _panel()
    x = sim.element_snapshots(man, [sim.PlaneWave(az, el, snr_db=20)], F, n=4096, seed=int(az + el))
    r = doa2d.estimate2d(x, man, F, method="music2d", sources=1)
    assert abs(cd(r.az_deg, az)) < 1.0 and abs(r.el_deg - el) < 1.5
    assert r.quality["elevation_capable"]


@pytest.mark.parametrize("az,el", [(10.0, 20.0), (330.0, 5.0), (40.0, 35.0)])
def test_esprit2d_is_auto_paired_on_a_ura(az, el):
    g, man = _panel()
    x = sim.element_snapshots(man, [sim.PlaneWave(az, el, snr_db=25)], F, n=4096, seed=int(az + el))
    r = doa2d.estimate2d(x, man, F, method="esprit2d", sources=1)
    assert abs(cd(r.az_deg, az)) < 1.5 and abs(r.el_deg - el) < 2.0
    assert r.estimator == "esprit2d"


def test_two_sources_resolved_in_azimuth_and_elevation():
    g, man = _panel(rows=4, cols=4)
    waves = [sim.PlaneWave(10.0, 30.0, snr_db=20), sim.PlaneWave(-25.0 % 360, 5.0, snr_db=20)]
    x = sim.element_snapshots(man, waves, F, n=8192, seed=7)
    r = doa2d.music2d(x, man, F, sources=2)
    found = sorted(zip(r.all_az_deg, r.all_el_deg))
    want = sorted([(10.0, 30.0), (335.0, 5.0)])
    for (fa, fe), (wa, we) in zip(found, want):
        assert abs(cd(fa, wa)) < 2.0 and abs(fe - we) < 3.0


def test_as_doaresult_carries_azimuth_downstream_and_elevation_in_quality():
    g, man = _panel()
    x = sim.element_snapshots(man, [sim.PlaneWave(15.0, 25.0, snr_db=20)], F, n=4096, seed=1)
    d = doa2d.music2d(x, man, F, sources=1).as_doaresult()
    assert abs(cd(d.theta_deg, 15.0)) < 1.0
    assert abs(d.quality["elevation_deg"] - 25.0) < 1.5
    assert d.likelihood is not None and d.likelihood.sum() == pytest.approx(1.0)


def test_a_horizontal_planar_array_is_up_down_ambiguous():
    g = M.ura(4, 4, 0.172, plane="horizontal")
    man = M.IdealManifold(g)
    can, note = doa2d.elevation_capability(g)
    assert can and "up/down ambiguity" in note
    x = sim.element_snapshots(man, [sim.PlaneWave(123.0, 30.0, snr_db=20)], F, n=4096, seed=3)
    r = doa2d.music2d(x, man, F, sources=1)
    assert abs(cd(r.az_deg, 123.0)) < 1.5
    assert abs(abs(r.el_deg) - 30.0) < 2.0           # magnitude right, sign ambiguous
    assert any("mirror" in c for c in r.caveats)


def test_a_flat_ring_refuses_elevation():
    g = M.uca(8, 0.2)
    man = M.IdealManifold(g)
    assert doa2d.elevation_capability(g)[0] is False
    x = sim.element_snapshots(man, [sim.PlaneWave(77.0, 0.0, snr_db=20)], F, n=4096, seed=1)
    r = doa2d.music2d(x, man, F, sources=1)
    assert abs(cd(r.az_deg, 77.0)) < 1.0 and r.el_deg == 0.0
    assert r.sigma_el_deg >= 90.0
    assert any("no elevation resolution" in c for c in r.caveats)


def test_music2d_respects_the_azimuth_field_of_view():
    g, man = _panel(b=0.0)
    # an emitter behind the panel should not be found in front
    x = sim.element_snapshots(man, [sim.PlaneWave(15.0, 10.0, snr_db=20)], F, n=4096, seed=2)
    r = doa2d.music2d(x, man, F, sources=1)
    lo, hi = g.fov_az_deg
    assert lo <= ((r.az_deg + 180) % 360 - 180) <= hi


def test_music2d_works_on_a_measured_panel_manifold():
    g, _m = _panel(rows=2, cols=4)
    truth = M.PerturbedManifold.random(g, gain_db=0.4, phase_deg=6, coupling_db=-24, seed=5)
    az = np.arange(-60, 61, 3.0)
    el = np.arange(-10, 46, 5.0)
    AZ, EL = np.meshgrid(az, el, indexing="ij")
    rng = np.random.default_rng(1)
    V = truth.response2(AZ.ravel(), EL.ravel(), F).T
    V = V * np.exp(1j * rng.uniform(-np.pi, np.pi, V.shape[0]))[:, None]
    # a 2-D measured manifold is future work; here just confirm music2d runs on
    # the perturbed (measured-like) manifold and still finds the angle
    x = sim.element_snapshots(truth, [sim.PlaneWave(20.0, 15.0, snr_db=20)], F, n=4096, seed=9)
    r = doa2d.music2d(x, truth, F, sources=1)
    assert abs(cd(r.az_deg, 20.0)) < 2.0 and abs(r.el_deg - 15.0) < 3.0


# -- review fixes (2026-10-30) --------------------------------------------------

def test_a_single_row_vertical_panel_is_not_elevation_capable():
    # finding #1: a 1×N vertical panel is one horizontal line — no elevation,
    # and music2d must not place the source off-horizon
    g = M.ura(1, 4, 0.172, plane="vertical", boresight_deg=0.0)
    man = M.IdealManifold(g)
    assert doa2d.elevation_capability(g)[0] is False
    x = sim.element_snapshots(man, [sim.PlaneWave(10.0, 0.0, snr_db=20)], F, n=4096, seed=1)
    r = doa2d.music2d(x, man, F, sources=1)
    assert abs(cd(r.az_deg, 10.0)) < 1.0 and r.el_deg == 0.0


def test_music2d_handles_a_sub_horizon_elevation_fov_without_crashing():
    # finding #3: an elevation FOV entirely below the old -10° default
    from dataclasses import replace
    g = M.ura(4, 4, 0.172, plane="vertical", boresight_deg=0.0)
    g = replace(g, fov_el_deg=(-60.0, -20.0))
    man = M.IdealManifold(g)
    x = sim.element_snapshots(man, [sim.PlaneWave(10.0, -40.0, snr_db=20)], F, n=4096, seed=2)
    r = doa2d.music2d(x, man, F, sources=1)              # must not raise
    assert -60.0 <= r.el_deg <= -20.0 and abs(cd(r.az_deg, 10.0)) < 3.0


def test_esprit2d_refuses_a_single_line_ura():
    g = M.ura(1, 4, 0.172, plane="vertical")
    with pytest.raises(ValueError, match="rows ≥ 2"):
        doa2d.esprit2d(sim.element_snapshots(M.IdealManifold(g),
                       [sim.PlaneWave(10.0, 0.0, 20)], F, n=1024), g, F, sources=1)
