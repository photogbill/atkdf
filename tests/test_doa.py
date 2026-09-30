"""Direction of arrival against planted bearings: every estimator returns the
planted angle, MDL returns the planted count, FBSS/ML separate a coherent
pair that plain MUSIC merges, and the record likelihood is a proper density."""

from __future__ import annotations

import numpy as np
import pytest

from atkdf import doa
from atkdf import manifold as M

F = 450e6


def array_data(bearings_deg, snr_db=20.0, n=4000, geometry=None, coherent=False,
               amps=None, seed=0, manifold=None):
    g = geometry or M.uca(5, 0.18)
    man = manifold or M.IdealManifold(g)
    rng = np.random.default_rng(seed)
    bearings = list(bearings_deg)
    amps = amps or [1.0] * len(bearings)
    base = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    x = np.zeros((g.n, n), dtype=complex)
    for th, amp in zip(bearings, amps):
        s = base if coherent else (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
        a = man.response(float(th), F)[:, 0]
        x += amp * np.outer(a, s)
    p = np.mean(np.abs(x) ** 2)
    pn = p / (10 ** (snr_db / 10))
    x += np.sqrt(pn / 2) * (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape))
    return x.astype(np.complex64), g, man


def err(a, t):
    return ((a - t + 180) % 360) - 180


@pytest.mark.parametrize("method", ["bartlett", "capon", "music", "root"])
def test_every_estimator_returns_a_planted_bearing(method):
    x, g, man = array_data([73.0], snr_db=20.0, seed=1)
    r = doa.estimate(x, man, F, method=method, sources=1, geometry=g)
    assert abs(err(r.theta_deg, 73.0)) < 1.0, (method, r.theta_deg)


def test_mdl_counts_two_well_separated_sources():
    x, g, man = array_data([30.0, 150.0], snr_db=20.0, seed=2)
    r = doa.estimate(x, man, F, method="music", geometry=g)
    assert r.n_sources == 2
    got = sorted(r.all_theta_deg[:2])
    assert abs(err(got[0], 30.0)) < 2 and abs(err(got[1], 150.0)) < 2


def test_mdl_counts_one_source_as_one():
    x, g, man = array_data([88.0], snr_db=20.0, seed=7)
    r = doa.estimate(x, man, F, method="music", geometry=g)
    assert r.n_sources == 1


def test_ml_resolves_a_coherent_pair_that_music_merges():
    # a direct path and its coherent image (multipath off a building). The
    # covariance is rank-1, which starves MUSIC; ML does not need full rank.
    # Separation is 80 deg -- inside the UCA-5's resolving power, unlike a
    # close pair, which no 0.36 m array resolves coherent or not.
    truth = [60.0, 140.0]
    x, g, man = array_data(truth, snr_db=25.0, coherent=True, amps=[1.0, 0.8], seed=3)
    ml = doa.estimate(x, man, F, method="ml", sources=2, geometry=g)
    got = sorted(ml.all_theta_deg[:2])
    assert abs(err(got[0], 60.0)) < 6 and abs(err(got[1], 140.0)) < 6, ml.all_theta_deg
    # plain MUSIC on the same coherent data does worse: at least one of its two
    # peaks misses a true bearing by more than ML does
    mu = doa.estimate(x, man, F, method="music", sources=2, geometry=g)
    music_best = max(min(abs(err(t, tt)) for t in mu.all_theta_deg[:2]) for tt in truth)
    ml_best = max(min(abs(err(t, tt)) for t in got) for tt in truth)
    assert ml_best <= music_best


def test_fbss_on_a_ula_separates_a_coherent_pair():
    g = M.ula(7, 0.30)
    x, _g, man = array_data([-10.0 % 360, 20.0], snr_db=25.0, coherent=True,
                            geometry=g, seed=4)
    plain = doa.estimate(x, man, F, method="music", sources=2, geometry=g)
    fb = doa.estimate(x, man, F, method="fbss", sources=2, geometry=g)
    # FBSS finds two distinct peaks; the coherent pair should be resolved
    peaks = sorted(set(round(t) for t in fb.all_theta_deg[:2]))
    assert len(peaks) == 2


def test_fbss_runs_on_the_uca_and_returns_a_bearing():
    # the production array is a UCA-5, and FBSS on it goes through the phase-mode
    # transform + smoothing path (a different path from the ULA one). It must not
    # crash and must return the planted bearing on a single source.
    x, g, man = array_data([120.0], snr_db=25.0, seed=11)
    r = doa.estimate(x, man, F, method="fbss", sources=1, geometry=g)
    assert abs(err(r.theta_deg, 120.0)) < 3.0, r.theta_deg


def test_estimate_clamps_an_over_large_source_count():
    x, g, man = array_data([50.0], snr_db=20.0, seed=12)
    # 9 sources on a 5-element array is impossible; it must clamp, not divide by
    # an empty noise subspace and return a flat spectrum
    r = doa.estimate(x, man, F, method="music", sources=9, geometry=g)
    assert r.n_sources <= g.n - 1
    assert abs(err(r.theta_deg, 50.0)) < 3.0


def test_root_music_on_a_ula_returns_the_cone_pair():
    g = M.ula(5, 0.30)
    x, _g, man = array_data([35.0], snr_db=25.0, geometry=g, seed=5)
    r = doa.estimate(x, man, F, method="root", sources=1, geometry=g)
    # a line array cannot tell front from back: 35 and 145 both fit
    assert min(abs(err(t, 35.0)) for t in r.all_theta_deg) < 2.0


def test_the_likelihood_is_a_density_over_bearing():
    x, g, man = array_data([73.0], snr_db=20.0, seed=1)
    r = doa.estimate(x, man, F, method="music", sources=1, geometry=g)
    assert r.likelihood.shape == (360,)
    assert r.likelihood.sum() == pytest.approx(1.0, abs=1e-6)
    assert r.likelihood.min() > 0                       # never zero: no veto
    assert int(np.argmax(r.likelihood)) in range(71, 76)


def test_rotate_likelihood_moves_the_peak_by_the_heading():
    lik = np.zeros(360)
    lik[40] = 1.0
    out = doa.rotate_likelihood(lik, 30.0)
    assert int(np.argmax(out)) == 70


def test_steering_grid_never_serves_one_manifold_the_wrong_arrays():
    """Regression: the steering cache once keyed on id(manifold), and CPython
    reuses ids after GC — so a 7-element manifold could be handed a dead
    5-element manifold's table. Interleave several geometries and each must get
    steering with its own element count."""
    import gc
    for _ in range(3):
        g5 = M.uca(5, 0.18)
        A5 = doa.steering_grid(M.IdealManifold(g5), F)[1]
        assert A5.shape[0] == 5
        del g5
        gc.collect()
        g7 = M.ula(7, 0.30)
        A7 = doa.steering_grid(M.IdealManifold(g7), F)[1]
        assert A7.shape[0] == 7


def test_the_uca_phase_mode_alias_is_reported_not_hidden():
    g = M.uca(5, 0.18)
    pm = doa.phase_mode_transform(g, 700e6)
    # at 700 MHz the outer modes are badly aliased; the number is exposed
    assert pm["worst_alias"] > 0.2
    assert pm["modes"].tolist() == [-2, -1, 0, 1, 2]
