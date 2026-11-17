"""The shadow likelihood (umbrella §8.6; propagation build plan §6.1, built
2026-11-16): heard from here, so cells from which an emitter of its class could
not have been heard are less likely — judged not by the region shrinking but
by the region shrinking AND still holding the emitter at its stated rate."""

from __future__ import annotations

import math

import numpy as np
import pytest

from atkdf import geodesy as geo
from atkdf import posterior as P
from atkdf.fakes import FreeSpaceField
from atkdf.records import BearingRecord

F = 450e6


def _rec(rx, brg, sigma=3.0, tag="direct", bw=12_500.0, f=F):
    g = np.arange(360.0)
    d = (g - (brg % 360) + 180) % 360 - 180
    lik = np.exp(-0.5 * (d / sigma) ** 2)
    lik /= lik.sum()
    return BearingRecord(t_utc=0.0, theta_rel_deg=brg % 360, sigma_deg=sigma,
                         estimator="music", heading_deg=0.0, theta_abs_deg=brg % 360,
                         likelihood=lik, lat=rx[0], lon=rx[1], path_tag=tag,
                         fc_hz=f, bw_hz=bw)


def test_the_heard_probability_falls_with_loss_and_is_never_zero():
    cls = P.POWER_CLASSES["handheld"]
    m = P.ShadowModel()
    s = m.sensitivity_dbm(12_500.0)
    assert s == pytest.approx(-174 + 10 * math.log10(12_500) + 6 + 6 + 1, abs=0.01)
    p = P.heard_probability(np.array([80.0, 140.0, 160.0, 250.0]), cls, s, 8.0)
    assert p[0] > 0.999 and p[0] > p[1] > p[2] > p[3] and p[3] < 1e-6
    lik = m.floor + (1 - m.floor) * p
    assert lik.min() >= m.floor                        # down-weighted, never zeroed


def test_refusals_and_once_per_place():
    spec = P.area_around(38.8, -77.5, half_size_m=4000, cell_m=100)
    post = P.Posterior(spec)
    rx = (38.8, -77.55)
    fld = FreeSpaceField()
    assert not post.terrain_update(_rec(rx, 90, tag="repeated"), fld)["applied"]
    assert not post.terrain_update(_rec(rx, 90, f=0.0), fld)["applied"]
    r = _rec(rx, 90)
    r.lat = None
    assert "position" in post.terrain_update(r, fld)["why"]
    assert post.terrain_update(_rec(rx, 90), fld)["applied"]
    again = post.terrain_update(_rec((38.8001, -77.55), 90), fld)    # ~11 m away
    assert not again["applied"] and "already" in again["why"]
    assert post.terrain_update(_rec((38.8, -77.52), 90), fld)["places"] == 2


def test_a_wall_trims_the_far_end_of_a_bearing():
    """One bearing east from a receiver whose view east is walled off beyond
    5 km by 60 dB: a handheld out there is rarely heard (a 25–50 W mobile
    still often is), so along the bearing the belief moves in toward the
    wall, and the near ground, in the clear, keeps its weight.

    (Judged along the bearing, not by the 95 % region: with ONE bearing the
    bearing's own floor spreads a few per cent over the whole grid, and taking
    mass off the far end pushes the 95 % region out into that floor. Two
    bearings — the acceptance test below — is where regions are judged.)"""
    spec = P.area_around(38.8, -77.5, half_size_m=12_000, cell_m=100)
    rx = (38.8, -77.5)
    plain, shadow = P.Posterior(spec), P.Posterior(spec)
    for post in (plain, shadow):
        post.update(_rec(rx, 90.0, sigma=2.0))
    note = shadow.terrain_update(_rec(rx, 90.0, sigma=2.0),
                                 FreeSpaceField(wall_bearing_deg=90.0, wall_range_m=5000,
                                                wall_width_deg=40, wall_loss_db=60.0))
    assert note["applied"] and note["belief_moved"] > 0.15
    e, n = spec.axes()
    E, N = np.meshgrid(e, n)
    strip = (np.abs(N) < 600) & (E > 0)                  # the cells along the bearing
    near, far = strip & (E < 4500), strip & (E > 6000)
    r0 = plain.prob()[far].sum() / plain.prob()[near].sum()
    r1 = shadow.prob()[far].sum() / shadow.prob()[near].sum()
    assert r1 < 0.4 * r0, (r0, r1)
    # the clear ground in front of the wall is not penalised at all
    pn0 = plain.prob()[near].sum() / plain.prob()[strip].sum()
    pn1 = shadow.prob()[near].sum() / shadow.prob()[strip].sum()
    assert pn1 > pn0
    pt = geo.direct(rx[0], rx[1], 90.0, 3000.0)
    assert shadow.contains(pt[0], pt[1], 0.95)


def test_heard_from_two_places_the_power_is_counted_once():
    """The joint over places shares one EIRP: heard from two walled places is
    stronger evidence than one, but not two independent ones."""
    spec = P.area_around(38.8, -77.5, half_size_m=6000, cell_m=100)
    fld = FreeSpaceField(wall_bearing_deg=0.0, wall_range_m=1000, wall_width_deg=360,
                         wall_loss_db=55.0)
    post = P.Posterior(spec)
    post.terrain_update(_rec((38.79, -77.52), 0.0), fld)
    post.terrain_update(_rec((38.79, -77.48), 0.0), fld)
    joint = post.prob()
    # the same two places, each multiplied in as if alone (the mistake)
    m = P.ShadowModel()
    cls = P.POWER_CLASSES
    naive = np.zeros(post.logp.shape)
    for rx in ((38.79, -77.52), (38.79, -77.48)):
        fr = fld.field(rx, spec, F, m.h_rx_m, 1.5)
        ps = []
        for name in ("handheld", "mobile"):
            fr2 = fld.field(rx, spec, F, m.h_rx_m, cls[name].h_tx_m)
            ps.append(P.heard_probability(fr2.loss_db, cls[name], m.sensitivity_dbm(12_500),
                                          m.sigma_db(fr.terrain_res_m), m.eirp_points))
        naive += np.log(m.floor + (1 - m.floor) * np.mean(ps, axis=0))
    pn = np.exp(naive - naive.max())
    pn /= pn.sum()
    # the naive product is the more confident one (smaller region)
    def area(p):
        q = np.sort(p.ravel())[::-1]
        return int(np.searchsorted(np.cumsum(q), 0.95)) + 1
    assert area(pn) < area(joint)


def test_the_stated_region_holds_the_emitter_at_the_stated_rate():
    """THE ACCEPTANCE TEST (build plan §6.1). Emitters uniform over the area;
    each heard or not from two receivers by the SAME model the likelihood
    assumes (one EIRP per emitter, shadowing independent per place); only the
    heard ones get bearings (σ 3°). With the shadow term the 95 % region is
    smaller on average, and it still holds the truth 95 % of the time —
    within a Clopper–Pearson interval."""
    rng = np.random.default_rng(20261116)
    spec = P.area_around(38.8, -77.5, half_size_m=8000, cell_m=100)
    m = P.ShadowModel()
    cls = P.POWER_CLASSES["handheld"]
    eirps = np.linspace(cls.eirp_dbm[0], cls.eirp_dbm[1], m.eirp_points)
    sens = m.sensitivity_dbm(12_500.0)
    rxs = [geo.direct(38.8, -77.5, 225.0, 6000.0), geo.direct(38.8, -77.5, 135.0, 6000.0)]
    walls = [FreeSpaceField(wall_bearing_deg=30.0, wall_range_m=3000, wall_width_deg=50,
                            wall_loss_db=60.0),
             FreeSpaceField(wall_bearing_deg=330.0, wall_range_m=3000, wall_width_deg=50,
                            wall_loss_db=60.0)]
    fields = [w.field(rx, spec, F, m.h_rx_m, cls.h_tx_m) for w, rx in zip(walls, rxs)]
    sig = m.sigma_db(fields[0].terrain_res_m)
    e_axis, n_axis = spec.axes()
    hits_t = hits_p = trials = 0
    area_t, area_p = [], []
    while trials < 160:
        r, c = rng.integers(0, spec.n, 2)
        eirp = rng.choice(eirps)
        heard = all(eirp - f.loss_db[r, c] + rng.normal(0, sig) >= sens for f in fields)
        if not heard:
            continue
        lat, lon, _ = geo.enu_to_geodetic(e_axis[c], n_axis[r], 0.0, (38.8, -77.5, 0.0))
        lat, lon = float(lat), float(lon)
        recs = []
        for rx in rxs:
            brg = geo.bearing_between(rx[0], rx[1], lat, lon) + rng.normal(0, 3.0)
            recs.append(_rec(rx, brg, sigma=3.0))
        plain, terr = P.Posterior(spec), P.Posterior(spec)
        for rec in recs:
            plain.update(rec)
            terr.update(rec)
        for rec, w in zip(recs, walls):
            terr.terrain_update(rec, w, model=m, classes=("handheld",))
        trials += 1
        hits_p += plain.contains(lat, lon)
        hits_t += terr.contains(lat, lon)
        area_p.append(plain.region_area_m2())
        area_t.append(terr.region_area_m2())
    lo, hi = P.coverage_interval(hits_t, trials, 0.99)
    assert lo <= 0.95 <= hi, (hits_t, trials, lo, hi)
    lo_p, hi_p = P.coverage_interval(hits_p, trials, 0.99)
    assert lo_p <= 0.95 <= hi_p, (hits_p, trials)
    assert np.mean(area_t) < 0.9 * np.mean(area_p), (np.mean(area_t), np.mean(area_p))


def test_the_region_outline_is_closed_and_encloses_the_region():
    spec = P.area_around(38.8, -77.5, half_size_m=5000, cell_m=50)
    post = P.Posterior(spec)
    post.update(_rec((38.78, -77.52), 45.0, sigma=2.0))
    post.update(_rec((38.78, -77.48), 315.0, sigma=2.0))
    rings = post.region_outline(0.95)
    assert rings and all(r[0] == r[-1] and len(r) >= 4 for r in rings)
    lats = [p[0] for r in rings for p in r]
    pk = post.peak()
    assert min(lats) < pk[0] < max(lats)


def test_ageing_keeps_the_belief_and_starts_the_places_afresh():
    spec = P.area_around(38.8, -77.5, half_size_m=4000, cell_m=100)
    post = P.Posterior(spec)
    post.update(_rec((38.8, -77.55), 90.0))
    post.terrain_update(_rec((38.8, -77.55), 90.0), FreeSpaceField())
    post.age(600.0, motion_state="on_foot")
    assert post.terrain_update(_rec((38.8, -77.55), 90.0), FreeSpaceField())["applied"]


def test_coverage_interval():
    lo, hi = P.coverage_interval(95, 100, 0.95)
    assert lo < 0.95 < hi and 0.88 < lo < 0.9 and hi > 0.98
    assert P.coverage_interval(0, 10)[0] == 0.0 and P.coverage_interval(10, 10)[1] == 1.0


def test_a_shared_bias_is_carried_so_the_region_still_holds_the_truth():
    """Twenty bearings from one array, all off by the same alignment error
    (uniform in ±3°, as fusion assumes) on top of 2° noise. Multiplied as if
    independent, the region shrinks past the truth; with the bias carried
    (`bias_bound_deg=3`) it holds the truth at about the stated rate."""
    rng = np.random.default_rng(42)
    spec = P.area_around(38.8, -77.5, half_size_m=6000, cell_m=50)
    places = [geo.direct(38.8, -77.5, b, 7000.0)[:2] for b in (200.0, 230.0, 260.0)]
    hits_ind = hits_bias = 0
    n = 60
    for _ in range(n):
        tgt = geo.direct(38.8, -77.5, rng.uniform(0, 360), rng.uniform(0, 2500))[:2]
        bias = rng.uniform(-3.0, 3.0)
        ind, bia = P.Posterior(spec), P.Posterior(spec, bias_bound_deg=3.0)
        for k in range(21):
            rx = places[k % 3]
            brg = geo.bearing_between(rx[0], rx[1], *tgt) + bias + rng.normal(0, 2.0)
            r = _rec(rx, brg, sigma=2.0)
            ind.update(r)
            bia.update(r)
        hits_ind += ind.contains(tgt[0], tgt[1], 0.9)
        hits_bias += bia.contains(tgt[0], tgt[1], 0.9)
    lo, hi = P.coverage_interval(hits_bias, n, 0.99)
    assert lo <= 0.9 <= hi, (hits_bias, n)
    assert hits_ind < hits_bias, (hits_ind, hits_bias)
    assert hits_ind / n < 0.8                       # the independent product under-covers
