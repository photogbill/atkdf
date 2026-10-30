"""3-D geometry, element patterns, field of view, subsets and the beamspace
wrapper (plan §11.4.3). The azimuth-only paths must read exactly what they
always read; the new paths must be honest about what a panel cannot see."""

from __future__ import annotations

import math

import numpy as np
import pytest

from atkdf import doa, sim
from atkdf import manifold as M

F = 851e6


def test_three_column_positions_split_into_xy_and_z():
    g = M.Geometry(np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.2], [0.0, 0.1, -0.1]]))
    assert g.positions.shape == (3, 2)
    assert np.allclose(g.z, [0.0, 0.2, -0.1])
    assert np.allclose(g.positions3[:, 2], g.z)
    d = g.to_dict()
    assert len(d["positions_m"][0]) == 3
    assert np.allclose(M.Geometry.from_dict(d).positions3, g.positions3)


def test_a_flat_array_still_writes_two_column_positions():
    d = M.uca(5, 0.18).to_dict()
    assert len(d["positions_m"][0]) == 2 and "pattern" not in d


def test_vertical_panel_layout_faces_its_boresight():
    g = M.ura(2, 4, 0.172, plane="vertical", boresight_deg=0.0)
    assert g.n == 8 and np.ptp(g.z) == pytest.approx(0.172)
    # facing forward (x), the face runs across y: every element at x = 0
    assert np.allclose(g.positions[:, 0], 0.0)
    # columns run to the RIGHT looking out: y decreases along a row
    assert g.positions[1, 1] < g.positions[0, 1]
    assert g.fov_az_deg == (-60.0, 60.0) and g.hemisphere == "front"
    assert g.boresight_common_deg() == 0.0


def test_azimuth_response_equals_the_3d_response_at_zero_elevation():
    g = M.ura(2, 4, 0.172, plane="vertical", boresight_deg=30.0)
    man = M.IdealManifold(g)
    th = np.array([0.0, 15.0, 30.0, 77.0, 200.0])
    assert np.allclose(man.response(th, F), man.response2(th, np.zeros_like(th), F))


def test_the_patterned_eadf_matches_the_response():
    g = M.ura(1, 4, 0.172, plane="vertical", boresight_deg=20.0)
    man = M.IdealManifold(g)
    G, q = man.eadf(F)
    th = np.array([3.0, 20.0, 47.5, 133.0, 301.0])
    four = G @ np.exp(1j * np.outer(q, np.radians(th)))
    assert np.allclose(man.response(th, F), four, atol=1e-6)


def test_root_music_runs_on_a_patterned_panel():
    g = M.ura(1, 6, 0.172, plane="vertical", boresight_deg=0.0)
    man = M.IdealManifold(g)
    x = sim.element_snapshots(man, [sim.PlaneWave(25.0, snr_db=15)], F, n=2048, seed=3)
    r = doa.estimate(x, man, F, method="root", sources=1)
    assert abs(r.theta_deg - 25.0) < 1.0


def test_fov_resolves_the_front_back_mirror_of_a_row():
    """A row across a panel's face sees θ and 180°−θ as one answer; the FOV
    (and the cardioid pattern) pick the front one."""
    g = M.ura(1, 4, 0.172, plane="vertical", boresight_deg=0.0)
    man = M.IdealManifold(g)
    x = sim.element_snapshots(man, [sim.PlaneWave(30.0, snr_db=20)], F, n=2048, seed=1)
    r = doa.estimate(x, man, F, method="music", sources=1)
    assert abs(r.theta_deg - 30.0) < 1.0
    assert r.quality["in_fov"] is True


def test_outside_the_fov_the_likelihood_is_flat_at_the_back_lobe_level():
    g = M.ura(1, 4, 0.172, plane="vertical", boresight_deg=0.0)
    man = M.IdealManifold(g)
    x = sim.element_snapshots(man, [sim.PlaneWave(10.0, snr_db=20)], F, n=2048, seed=2)
    r = doa.estimate(x, man, F, method="music", sources=1)
    lik = r.likelihood
    inside = g.in_fov(np.arange(360.0))
    behind = lik[~inside]
    assert np.ptp(behind) < 1e-12                       # flat: no information
    assert lik.sum() == pytest.approx(1.0)
    # the back lobe is −20 dB → 1 % of power; level per degree is that /360,
    # renormalised — far below the peak, far above zero
    assert behind[0] < lik.max() / 50 and behind[0] > 0


def test_a_source_at_the_fov_edge_says_so():
    g = M.ura(1, 6, 0.172, plane="vertical", boresight_deg=0.0)
    man = M.IdealManifold(g)
    x = sim.element_snapshots(man, [sim.PlaneWave(58.5, snr_db=25)], F, n=4096, seed=4)
    r = doa.estimate(x, man, F, method="music", sources=1)
    assert r.quality["fov_edge_deg"] < 3.0
    assert any("EDGE OF THE FIELD OF VIEW" in c for c in r.caveats)


def test_model_sigma_grows_off_a_panels_boresight_but_not_on_a_circle():
    panel = M.ura(1, 4, 0.172, plane="vertical", boresight_deg=0.0)
    man = M.IdealManifold(panel)
    s0 = doa.model_sigma_deg(man, panel, 0.0)
    s50 = doa.model_sigma_deg(man, panel, 50.0)
    assert s50 == pytest.approx(s0 / math.cos(math.radians(50.0)))
    circle = M.uca(5, 0.18)
    m2 = M.IdealManifold(circle)
    assert doa.model_sigma_deg(m2, circle, 0.0) == doa.model_sigma_deg(m2, circle, 50.0)


def test_subset_drops_an_element_and_the_manifold_follows():
    g = M.uca(5, 0.18)
    man = M.PerturbedManifold.random(g, seed=4)
    sub = man.subset([0, 1, 3, 4])
    assert sub.geometry.parent_elements == (0, 1, 3, 4)
    th = np.array([10.0, 100.0])
    # dropping a row of a coupled array is an approximation, but the gains and
    # the ideal part must be exactly the parent's rows when there is no coupling
    m0 = M.PerturbedManifold(g, gains=man.gains)
    assert np.allclose(m0.subset([0, 1, 3, 4]).response(th, 450e6),
                       m0.response(th, 450e6)[[0, 1, 3, 4]])
    assert sub.response(th, 450e6).shape == (4, 2)


def test_beamspace_manifold_is_w_hermitian_a_and_its_eadf_agrees():
    g = M.ura(4, 4, 0.172, plane="vertical", boresight_deg=0.0)
    base = M.IdealManifold(g)
    W = np.zeros((16, 4), complex)
    subs = [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]
    for c, mem in enumerate(subs):
        W[mem, c] = 0.5
    bm = M.BeamspaceManifold(base, W)
    th = np.array([0.0, 20.0, -35.0 % 360])
    assert np.allclose(bm.response(th, F), W.conj().T @ base.response(th, F))
    G, q = bm.eadf(F)
    four = G @ np.exp(1j * np.outer(q, np.radians(th)))
    assert np.allclose(bm.response(th, F), four, atol=1e-6)
    assert bm.geometry.n == 4 and bm.geometry.fov_az_deg == g.fov_az_deg


def test_one_coherent_mast_hands_off_between_faces_through_its_patterns():
    g = M.multi_face(faces=4, cols=4, spacing_m=0.172, standoff_m=0.25)
    assert g.n == 16 and set(g.groups) == {0, 1, 2, 3}
    man = M.IdealManifold(g)
    x = sim.element_snapshots(man, [sim.PlaneWave(100.0, snr_db=15)], F, n=4096, seed=5)
    r = doa.estimate(x, man, F, method="music", sources=1)
    assert abs(((r.theta_deg - 100.0) + 180) % 360 - 180) < 1.0


def test_element_snapshots_put_coherent_multipath_on_one_stream():
    man = M.IdealManifold(M.uca(8, 0.2))
    x = sim.element_snapshots(man, [sim.PlaneWave(10, snr_db=30, group=1),
                                    sim.PlaneWave(80, snr_db=30, group=1, gain_db=-3)],
                              450e6, n=4096, seed=0, noise=False)
    ev = np.linalg.eigvalsh(x @ x.conj().T / x.shape[1])[::-1]
    assert ev[1] / ev[0] < 1e-9                          # rank one: coherent
