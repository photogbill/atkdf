"""The road-constrained particle filter (umbrella §8.5, built 2026-11-16) on a
synthetic town: eleven by eleven streets a kilometre apart."""

from __future__ import annotations

import numpy as np
import pytest

from atkdf import geodesy as geo
from atkdf import posterior as P
from atkdf.fakes import GridRoads, NoRoads
from atkdf.roadfilter import RoadParticleFilter
from atkdf.records import BearingRecord

LAT0, LON0 = 38.70, -77.60


def _rec(rx, brg, t, sigma=2.0):
    g = np.arange(360.0)
    d = (g - (brg % 360) + 180) % 360 - 180
    lik = np.exp(-0.5 * (d / sigma) ** 2)
    lik /= lik.sum()
    return BearingRecord(t_utc=t, theta_rel_deg=brg % 360, sigma_deg=sigma, estimator="music",
                         heading_deg=0.0, theta_abs_deg=brg % 360, likelihood=lik,
                         lat=rx[0], lon=rx[1], path_tag="direct", fc_hz=450e6, bw_hz=12_500)


def _enu(e, n):
    la, lo, _ = geo.enu_to_geodetic(e, n, 0.0, (LAT0, LON0, 0.0))
    return float(la), float(lo)


def test_the_street_grid_fake_behaves_like_roads():
    roads = GridRoads(LAT0, LON0)
    e = roads.nearest_edge(*_enu(2400.0, 5030.0))
    assert e.id == "h5_2"                                   # on the 5 km street, east-bound
    nb = {x.id for x in roads.neighbours(e)}
    assert nb == {"h5_3", "v5_3", "v4_3r"}                  # straight on, left, right; no U-turn
    assert roads.nearest_edge(*_enu(-3000.0, 5000.0)) is None
    corner = roads.neighbours(roads._edge("h", 10, 9, False))
    assert {x.id for x in corner} == {"v9_10r"}             # the north-east corner: one way on


def test_one_bearing_across_a_town_is_one_candidate_per_road():
    """A bearing that cuts three roads is three candidates, not a wedge: every
    candidate holding real weight is ON the bearing (within its beam), and they
    sit on several different streets."""
    roads = GridRoads(LAT0, LON0)
    rx = _enu(-4000.0, 5500.0)
    pf = RoadParticleFilter(roads, n=4000, seed=1)
    pf.seed_area(*_enu(5000.0, 5000.0), half_size_m=5200.0, n_points=4000)
    pf.update(_rec(rx, 90.0, 0.0, sigma=1.0))               # due east, between the 5 and 6 km streets
    big = [c for c in pf.candidates(k=30) if c["share"] > 0.02]
    streets = set()
    for c in big:
        off = geo.angle_difference(geo.bearing_between(rx[0], rx[1], c["lat"], c["lon"]), 90.0)
        assert abs(off) < 3.0, (off, c)                       # inside the beam (3 sigma)
        e, n, _ = geo.geodetic_to_enu(c["lat"], c["lon"], 0.0, (LAT0, LON0, 0.0))
        streets.add(round(float(e) / 1000.0))
    assert len(streets) >= 3


def _drive(pf, rx, legs, rng, every_s=10.0, sigma=2.0):
    """Drive the truth along `legs` [(e0, n0, e1, n1, speed)], a bearing every
    `every_s`; returns the last true position."""
    t, truth = 0.0, None
    for (e0, n0, e1, n1, v) in legs:
        L = float(np.hypot(e1 - e0, n1 - n0))
        T = L / v
        tt = 0.0
        while tt <= T + 1e-9:
            f = tt / T
            truth = _enu(e0 + f * (e1 - e0), n0 + f * (n1 - n0))
            brg = geo.bearing_between(rx[0], rx[1], *truth) + rng.normal(0.0, sigma)
            assert pf.update(_rec(rx, brg, t + tt, sigma=sigma))
            tt += every_s
        t += T + every_s
    return truth


def test_a_straight_drive_seen_from_one_parked_receiver_stays_ambiguous():
    """THE OPERATOR'S LESSON, pinned. A car driving straight, seen from one
    receiver that does not move, cannot be ranged: a car on a street twice as
    far, driving twice as fast, draws the same bearings (bearings-only target
    motion analysis needs the observer to manoeuvre). The filter must SAY so —
    several candidates — not pick one."""
    rng = np.random.default_rng(7)
    roads = GridRoads(LAT0, LON0)
    rx = _enu(-5000.0, 2000.0)
    pf = RoadParticleFilter(roads, n=4000, seed=3)
    pf.seed_area(*_enu(5000.0, 5000.0), half_size_m=5200.0, n_points=4000)
    _drive(pf, rx, [(1000.0, 5000.0, 5500.0, 5000.0, 15.0)], rng)
    assert pf.estimate()["share"] < 0.6
    assert len([c for c in pf.candidates(10) if c["share"] > 0.05]) >= 2


def test_two_receivers_follow_a_turning_car_on_its_roads_better_than_the_grid():
    """A car drives east on the 5 km street, turns north on the 4 km street;
    two parked receivers (west and south) take bearings in turn, σ 2°. The
    road filter ends on the car's street (sometimes a block behind it); the
    free grid — ageing the belief at vehicle speed between bearings, the best
    it can do for a mover — ends several times further off."""
    errs_road, errs_grid = [], []
    rxs = [_enu(-5300.0, 2300.0), _enu(2500.0, -4000.0)]
    # ends mid-block (7.5 km north), not on a junction, so "its street" is one street
    legs = [(1000.0, 5000.0, 4000.0, 5000.0, 15.0), (4000.0, 5000.0, 4000.0, 7500.0, 15.0)]
    for seed in range(3):
        rng = np.random.default_rng(100 + seed)
        pf = RoadParticleFilter(GridRoads(LAT0, LON0), n=4000, seed=seed)
        pf.seed_area(*_enu(5000.0, 5000.0), half_size_m=5200.0, n_points=4000)
        post = P.Posterior(P.area_around(*_enu(5000.0, 5000.0), half_size_m=7000, cell_m=50))
        t, k, truth = 0.0, 0, None
        for (e0, n0, e1, n1, v) in legs:
            T = float(np.hypot(e1 - e0, n1 - n0)) / v
            tt = 0.0
            while tt <= T + 1e-9:
                f = tt / T
                truth = _enu(e0 + f * (e1 - e0), n0 + f * (n1 - n0))
                rx = rxs[k % 2]
                rec = _rec(rx, geo.bearing_between(rx[0], rx[1], *truth) + rng.normal(0, 2.0),
                           t + tt)
                assert pf.update(rec)
                post.age(t + tt, "vehicle")
                post.update(rec)
                tt += 10.0
                k += 1
            t += T + 10.0
        est = pf.estimate()
        errs_road.append(geo.distance_between(est["lat"], est["lon"], *truth))
        fx = post.fix()
        errs_grid.append(geo.distance_between(fx["peak_lat"], fx["peak_lon"], *truth))
        # the car's own street is among the segments the best candidate holds
        assert any(x.startswith("v") and x.rstrip("r").endswith("_4") for x in est["edges"]), est
        assert est["share"] > 0.25
    assert max(errs_road) < 800.0, errs_road          # about two beam-widths at 9 km
    assert np.mean(errs_road) < 0.6 * np.mean(errs_grid), (errs_road, errs_grid)


def test_with_no_roads_the_filter_says_so():
    pf = RoadParticleFilter(NoRoads(), n=100, seed=0)
    assert pf.seed_area(LAT0, LON0, 2000.0, 50) == 0
    assert not pf.update(_rec((LAT0, LON0), 45.0, 0.0))
    assert "no particles" in pf.describe() and pf.candidates() == []


def test_a_repeated_burst_does_not_move_it():
    roads = GridRoads(LAT0, LON0)
    pf = RoadParticleFilter(roads, n=500, seed=2)
    pf.seed_area(*_enu(5000.0, 5000.0), 5000.0, 600)
    r = _rec(_enu(-4000.0, 5000.0), 90.0, 0.0)
    r.path_tag = "repeated"
    w0 = pf.weights().copy()
    assert not pf.update(r)
    assert np.allclose(pf.weights(), w0)


def test_seeding_from_a_grid_posterior_keeps_to_its_belief():
    roads = GridRoads(LAT0, LON0)
    spec = P.area_around(*_enu(5000.0, 5000.0), half_size_m=6000, cell_m=100)
    post = P.Posterior(spec)
    rx1, rx2 = _enu(-3000.0, 0.0), _enu(13000.0, 0.0)
    tgt = _enu(3000.0, 7000.0)
    for rx in (rx1, rx2):
        post.update(_rec(rx, geo.bearing_between(rx[0], rx[1], *tgt), 0.0, sigma=1.5))
    pf = RoadParticleFilter(roads, n=2000, seed=4)
    assert pf.seed_from_grid(post, 1500) == 2000
    lat, lon = pf.positions()
    d = np.array([geo.distance_between(a, b, *tgt) for a, b in zip(lat, lon)])
    assert np.median(d) < 1200.0
