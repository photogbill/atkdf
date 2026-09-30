"""WGS-84 geodesy, and the rotation that turns an array angle into a bearing.

Everything in the geolocation layer rests on this file, so the tests here are
mostly about the three ways it can be plausibly wrong:

  * **A spherical earth is 0.3 % wrong**, which over a DF baseline is
    hundreds of metres. The difference is measured rather than asserted.
  * **A linear array measures a CONE**, which crosses the horizon twice.
    Both crossings are returned, and the tests pin the geometry: roll does
    not move a nose-forward array's bearing and pitch does.
  * **Attitude is the measurement, not a correction.** The rotation order is
    the aerospace 3-2-1, and for a level platform every order agrees — so
    the mistake is invisible until the aircraft banks, which is what
    `test_the_rotation_order_matters_once_the_platform_banks` is for.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from atkdf import geodesy as g


# ---------------------------------------------------------------------------
# coordinate conversions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lat,lon,alt", [
    (51.5, -0.12, 100.0), (-33.9, 151.2, 0.0), (0.0, 0.0, 0.0),
    (89.9, 0.0, 50.0), (-89.9, 179.0, -20.0), (45.0, -179.99, 8848.0),
])
def test_geodetic_ecef_round_trips_including_the_poles(lat, lon, alt):
    """The pole is where the obvious `p / cos(lat) − N` divides by zero, so
    it gets its own path and therefore its own test."""
    x, y, z = g.geodetic_to_ecef(lat, lon, alt)
    la, lo, al = g.ecef_to_geodetic(x, y, z)
    assert la == pytest.approx(lat, abs=1e-9)
    assert lo == pytest.approx(lon, abs=1e-9)
    assert al == pytest.approx(alt, abs=1e-6)


def test_the_ellipsoid_is_not_a_sphere():
    """A degree of latitude is longer near the poles. If this passes on a
    sphere the constants are wrong."""
    eq = g.distance_between(0.0, 0.0, 1.0, 0.0)
    pole = g.distance_between(89.0, 0.0, 90.0, 0.0)
    assert pole > eq
    assert (pole - eq) == pytest.approx(1_000.0, abs=300.0)


def test_enu_round_trips_and_is_a_local_metre_frame():
    ref = (51.5, -0.12, 0.0)
    e, n, u = g.geodetic_to_enu(51.51, -0.10, 150.0, ref)
    assert n == pytest.approx(1112.8, abs=1.0)      # ~0.01° of latitude
    assert u == pytest.approx(149.75, abs=0.5)      # curvature drop included
    la, lo, al = g.enu_to_geodetic(e, n, u, ref)
    assert la == pytest.approx(51.51, abs=1e-9)
    assert lo == pytest.approx(-0.10, abs=1e-9)
    assert al == pytest.approx(150.0, abs=1e-6)


# ---------------------------------------------------------------------------
# geodesics
# ---------------------------------------------------------------------------

def test_direct_and_inverse_are_exact_inverses():
    lat2, lon2 = g.direct(51.5, -0.12, 45.0, 100_000.0)
    back = g.inverse(51.5, -0.12, lat2, lon2)
    assert back.distance_m == pytest.approx(100_000.0, abs=1e-3)
    assert back.bearing_deg == pytest.approx(45.0, abs=1e-9)
    assert back.method == "Vincenty"


def test_the_spherical_approximation_costs_a_hundred_metres():
    """**The reason this module does not use haversine.** 0.18 % over 62 km
    is 114 m, which is the same order as the DF accuracy the whole exercise
    is trying to reach."""
    a = g.inverse(51.5, -0.12, 51.9, 0.5)
    s = g._spherical_inverse(51.5, -0.12, 51.9, 0.5, 0)
    assert a.method == "Vincenty"
    err = abs(s.distance_m - a.distance_m)
    assert 50.0 < err < 300.0, err
    assert err / a.distance_m == pytest.approx(0.0018, abs=0.001)


def test_a_near_antipodal_pair_falls_back_and_says_so():
    """Vincenty genuinely does not converge here. The failure is reported
    rather than returned as a number."""
    r = g.inverse(0.0, 0.0, 0.5, 179.7)
    assert r.method != "Vincenty"
    assert "did not converge" in r.method
    assert r.distance_m > 19_000_000.0


def test_coincident_points_are_not_a_division_by_zero():
    r = g.inverse(51.5, -0.12, 51.5, -0.12)
    assert r.distance_m == 0.0
    assert "coincident" in r.method


def test_angle_difference_wraps_the_short_way():
    """A residual computed without this is 359° when the truth is −1°, and a
    least-squares fit handed that number goes somewhere interesting."""
    assert g.angle_difference(359.0, 1.0) == pytest.approx(-2.0)
    assert g.angle_difference(1.0, 359.0) == pytest.approx(2.0)
    # exactly opposite comes back as −180: the interval is half-open, and
    # ±180 are the same direction anyway
    assert abs(g.angle_difference(180.0, 0.0)) == pytest.approx(180.0)
    assert np.allclose(g.angle_difference([10.0, 350.0], [350.0, 10.0]),
                       [20.0, -20.0])


# ---------------------------------------------------------------------------
# attitude and the cone
# ---------------------------------------------------------------------------

def test_a_direction_in_body_axes_becomes_a_true_bearing():
    assert g.body_to_bearing((1, 0, 0), 45.0)[0] == pytest.approx(45.0)
    assert g.body_to_bearing((0, 1, 0), 45.0)[0] == pytest.approx(135.0)
    az, el = g.body_to_bearing((1, 0, 0), 0.0, pitch_deg=30.0)
    assert az == pytest.approx(0.0, abs=1e-9)
    assert el == pytest.approx(30.0)


def test_the_rotation_order_matters_once_the_platform_banks():
    """**The mistake that is invisible on the bench.** Yaw-pitch-roll and
    roll-pitch-yaw give the same matrix for a level platform, so applying
    them in the wrong order costs nothing until the aircraft turns."""
    level_a = g.rotation_body_to_ned(30.0, 0.0, 0.0)
    level_b = g.rotation_body_to_ned(30.0, 0.0, 0.0)
    assert np.allclose(level_a, level_b)
    banked = g.rotation_body_to_ned(30.0, 10.0, 40.0)
    wrong_order = (g.rotation_body_to_ned(0.0, 0.0, 40.0)
                   @ g.rotation_body_to_ned(0.0, 10.0, 0.0)
                   @ g.rotation_body_to_ned(30.0, 0.0, 0.0))
    assert not np.allclose(banked, wrong_order, atol=1e-3)


def test_the_rotation_is_orthonormal():
    for h, p, r in [(0, 0, 0), (37, 12, -45), (200, -30, 80)]:
        m = g.rotation_body_to_ned(h, p, r)
        assert np.allclose(m @ m.T, np.eye(3), atol=1e-12)
        assert float(np.linalg.det(m)) == pytest.approx(1.0)


@pytest.mark.parametrize("cone,heading,expect", [
    (90.0, 0.0, [90.0, 270.0]),      # broadside from a north-facing array
    (0.0, 0.0, [0.0]),               # endfire — the cone degenerates
    (180.0, 0.0, [180.0]),           # the other endfire
    (60.0, 0.0, [60.0, 300.0]),      # ±60 about the axis
    (90.0, 90.0, [0.0, 180.0]),      # everything rotates with the heading
    (60.0, 90.0, [30.0, 150.0]),
])
def test_the_cone_crosses_the_horizon_where_the_geometry_says(cone, heading,
                                                              expect):
    got = g.cone_bearings(cone, heading)
    assert got == pytest.approx(expect, abs=1e-6)


def test_a_linear_array_usually_gives_TWO_bearings():
    """**Not a defect — the measurement really is ambiguous.** Returning one
    of them and a caveat is how an analyst ends up 40° off with no way to
    notice."""
    assert len(g.cone_bearings(60.0, 0.0)) == 2
    assert len(g.cone_bearings(0.0, 0.0)) == 1      # tangent: one solution


def test_roll_does_not_move_a_nose_forward_arrays_bearing():
    """The cone is symmetric about the array axis, and rolling about that
    same axis moves nothing. A rotation implemented wrongly usually breaks
    exactly this."""
    assert g.cone_bearings(60.0, 0.0, roll_deg=30.0) == pytest.approx(
        [60.0, 300.0], abs=1e-6)


def test_but_pitch_does_because_the_cone_tilts():
    got = g.cone_bearings(60.0, 0.0, pitch_deg=20.0)
    assert len(got) == 2
    assert abs(g.angle_difference(got[0], 60.0)) > 1.0


def test_a_cone_that_never_reaches_the_horizon_returns_nothing():
    """Itself a finding: the emitter cannot be where the geometry would need
    it to be."""
    assert g.cone_bearings(10.0, 0.0, pitch_deg=45.0) == []


def test_a_vertical_array_prefers_no_azimuth_at_all():
    assert g.cone_bearings(45.0, 0.0, axis_body=(0.0, 0.0, 1.0)) == []


def test_cutting_the_cone_at_a_different_elevation_moves_the_answer():
    horizon = g.cone_bearings(60.0, 0.0, target_elevation_deg=0.0)
    below = g.cone_bearings(60.0, 0.0, target_elevation_deg=-10.0)
    assert horizon != below


# ---------------------------------------------------------------------------
# the line of bearing
# ---------------------------------------------------------------------------

def test_a_lob_walks_out_along_its_own_bearing():
    lob = g.LOB(lat=51.5, lon=-0.12, bearing_deg=42.0, sigma_deg=1.0)
    lat, lon = lob.point_at(30_000.0)
    back = g.inverse(51.5, -0.12, lat, lon)
    assert back.distance_m == pytest.approx(30_000.0, abs=1e-3)
    assert back.bearing_deg == pytest.approx(42.0, abs=1e-9)


def test_a_lob_renders_as_geojson_in_lon_lat_order():
    """GeoJSON is longitude first. Every mapping bug of this shape looks like
    an emitter in the sea off Somalia."""
    lob = g.LOB(lat=51.5, lon=-0.12, bearing_deg=90.0, sigma_deg=1.0)
    gj = lob.to_geojson(10_000.0)
    a, b = gj["geometry"]["coordinates"]
    assert a == pytest.approx([-0.12, 51.5], abs=1e-9)
    assert b[0] > a[0]                         # due east: longitude increases
    assert b[1] == pytest.approx(51.5, abs=0.01)


def test_an_ambiguous_lob_says_so_in_words():
    lob = g.LOB(lat=51.5, lon=-0.12, bearing_deg=60.0, sigma_deg=1.0,
                ambiguous_with=300.0)
    assert "300.00" in lob.describe()
    assert "crosses the horizon twice" in lob.describe()
