"""Many observations of one emitter, combined into one fix.

**THE CENTREPIECE IS THE BIAS TABLE**, because it is the failure that looks
like success: a DF system with a shared alignment error produces bearings that
agree beautifully with each other and a fix that is confidently in the wrong
place, with an ellipse that gets tighter the more data you take. The
measurement below is on a west-to-east collection run, 1° random bearing
error, and the number is how often the truth actually fell inside the stated
95 % ellipse. With sixteen bearings and an unmodelled 3° bias it is ZERO.

Everything else here is the ordinary care an estimator like this needs, and
each test names the wrong answer it prevents:

  * two lines always cross, so an intersection is not evidence;
  * fusing two different emitters produces a confident point between them;
  * the mirror set of a linear array's cones is self-consistent too, so
    "resolved" has to mean the alternative was REJECTED, not merely beaten;
  * a bearing is a geodesic azimuth and the local chord azimuth is a
    different angle — 0.196° different at 25 km, which biased the fit by 89 m
    on data whose bearings were accurate to 0.008°.
"""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from atkdf import fusion as fu
from atkdf import geodesy as geo

TRUE = (51.60, -0.05)
OTHER = (51.30, 0.40)


def lobs_at(stations, target=TRUE, sigma=1.0, bias=0.0, seed=0,
            source="rx"):
    r = np.random.default_rng(seed)
    out = []
    for i, (la, lo) in enumerate(stations):
        b = geo.bearing_between(la, lo, *target)
        out.append(geo.LOB(lat=la, lon=lo,
                           bearing_deg=(b + bias + r.normal(0, sigma)) % 360,
                           sigma_deg=sigma, source=source, epoch=1.7e9 + i))
    return out


GOOD = [(51.50, -0.20), (51.52, 0.10), (51.45, -0.02), (51.70, -0.25)]
RUN = [(51.42, -0.30 + 0.03 * i) for i in range(16)]


def err(fix, target=TRUE):
    return geo.distance_between(fix.lat, fix.lon, *target)


# ---------------------------------------------------------------------------
# 1 — the bias table
# ---------------------------------------------------------------------------

#: The table in `fusion`'s own docstring: coverage of the stated 95 % ellipse.
#: Asserted here so the two cannot drift apart.
def coverage(n, bias, modelled, trials=40):
    hits = 0
    for s in range(trials):
        f = fu.ml_fix(lobs_at(RUN[:n], bias=bias, seed=100 + s),
                      systematic_deg=modelled)
        hits += err(f) <= f.ellipse(0.95)[0]
    return hits / trials


@pytest.mark.parametrize("n", [4, 8, 16])
def test_with_no_bias_the_ellipse_means_what_it_says(n):
    assert coverage(n, bias=0.0, modelled=0.0) >= 0.85


def test_an_unmodelled_shared_bias_destroys_the_ellipse():
    """**THE FAILURE THIS MODULE EXISTS FOR.** Sixteen bearings, a 3 °
    alignment error, and the truth falls inside the stated 95 % ellipse in
    NONE of forty trials — while every internal consistency check passes,
    because the bearings do agree with each other. They are wrong together.
    """
    assert coverage(16, bias=3.0, modelled=0.0) <= 0.10


def test_modelling_the_same_bias_repairs_it():
    """Identical data, one number supplied. The ellipse grows to something
    honest and the truth is back inside it."""
    assert coverage(16, bias=3.0, modelled=3.0) >= 0.85


def test_more_bearings_shrink_a_random_ellipse():
    a = fu.ml_fix(lobs_at(RUN[:4], seed=11)).ellipse(0.95)[0]
    b = fu.ml_fix(lobs_at(RUN[:16], seed=11)).ellipse(0.95)[0]
    assert b < a / 1.5


def test_but_not_a_bias_limited_one():
    """The whole point: a shared error does not average down, so the ellipse
    stops shrinking however much of the same data arrives."""
    a = fu.ml_fix(lobs_at(RUN[:8], bias=3.0, seed=11),
                  systematic_deg=3.0).ellipse(0.95)[0]
    b = fu.ml_fix(lobs_at(RUN[:16], bias=3.0, seed=11),
                  systematic_deg=3.0).ellipse(0.95)[0]
    assert b > 0.5 * a


def test_a_bias_limited_fix_says_so_and_says_what_would_help():
    f = fu.ml_fix(lobs_at(RUN[:16], bias=3.0, seed=11), systematic_deg=3.0)
    assert f.bias_limited
    text = " ".join(f.caveats)
    assert "BIAS-LIMITED" in text
    assert "average down" in text
    assert "independently calibrated" in text


def test_bias_groups_are_tracked_per_source():
    """Two separately calibrated systems do not share one error, so they get
    their own group and their own contribution to the floor."""
    half = len(RUN) // 2
    mixed = fu.ml_fix(lobs_at(RUN[:half], seed=11, source="rx-A")
                      + lobs_at(RUN[half:], seed=12, source="rx-B"),
                      systematic_deg=3.0)
    assert mixed.evidence["bias_groups"] == {"rx-A": half, "rx-B": half}


def test_the_systematic_covariance_is_rank_one_per_group():
    """**That rank is the physics.** A uniform rotation of every bearing
    displaces the fix along ONE direction — it does not spread it in a
    circle — so one bias group contributes a rank-one term, and two
    contribute rank two."""
    one = fu.ml_fix(lobs_at(RUN, seed=11, source="A"), systematic_deg=3.0)
    w = np.linalg.eigvalsh(one.systematic_cov)
    assert max(w) > 0 and min(w) / max(w) < 1e-9
    half = len(RUN) // 2
    two = fu.ml_fix(lobs_at(RUN[:half], seed=11, source="A")
                    + lobs_at(RUN[half:], seed=12, source="B"),
                    systematic_deg=3.0)
    w2 = np.linalg.eigvalsh(two.systematic_cov)
    assert min(w2) / max(w2) > 1e-9


def test_the_floor_never_exceeds_the_total_it_floors():
    """**A floor larger than the thing it floors is a model that has stopped
    meaning anything**, and the one-piece inversion produced exactly that:
    a 1,250 m systematic floor under a 690 m total."""
    for n in (4, 8, 12, 16):
        f = fu.ml_fix(lobs_at(RUN[:n], seed=11), systematic_deg=3.0)
        assert np.trace(f.systematic_cov) <= np.trace(f.cov) * 1.000001


# ---------------------------------------------------------------------------
# 2 — geometry
# ---------------------------------------------------------------------------

def test_a_good_crossing_finds_the_emitter():
    f = fu.ml_fix(lobs_at(GOOD, seed=3))
    assert err(f) < 1_000.0
    assert f.consistent


def test_gdop_is_dimensionless_and_range_invariant():
    """Metres-per-radian grows with range on perfect geometry, which is not
    a dilution of precision — it is being far away. Dividing by the mean
    range removes it."""
    near = fu.ml_fix(lobs_at([(51.60, -0.20), (51.50, -0.05)], sigma=0.001))
    far = fu.ml_fix(lobs_at([(51.60, -0.68), (51.20, -0.05)], sigma=0.001))
    assert near.gdop == pytest.approx(1.41, abs=0.15)
    assert far.gdop == pytest.approx(near.gdop, rel=0.1)


def test_near_parallel_bearings_report_a_huge_gdop_and_say_what_to_do():
    f = fu.ml_fix(lobs_at([(51.50, -0.20), (51.499, -0.199)], sigma=0.001))
    assert f.gdop > 50
    assert any("close to parallel" in c for c in f.caveats)
    assert any("Moving the receiver" in c for c in f.caveats)


def test_two_bearings_always_cross_and_the_fix_says_so():
    """**An intersection is not evidence.** With two there is no redundancy
    and no consistency check is possible at all."""
    f = fu.ml_fix(lobs_at(GOOD[:2], seed=3))
    assert f.n_obs == 2
    assert math.isnan(f.consistency_p)
    assert any("TWO BEARINGS ALWAYS CROSS" in c for c in f.caveats)


def test_one_bearing_is_not_a_fix():
    with pytest.raises(ValueError, match="at least two"):
        fu.ml_fix(lobs_at(GOOD[:1]))


def test_the_predicted_bearing_is_geodesic_not_a_chord_azimuth():
    """**The bug this pins cost 89 m on bearings accurate to 0.008°.** At
    51° N over 25 km the two definitions differ by 0.196°, which is four
    times a good array's error and varies systematically across a run."""
    lobs = lobs_at(GOOD, sigma=1e-6, seed=1)
    f = fu.ml_fix(lobs)
    assert err(f) < 5.0
    assert max(abs(np.array(f.evidence["residuals_deg"]))) < 0.02
    ref = f.evidence["reference"]
    chord = math.degrees(math.atan2(
        *reversed(list(geo.geodetic_to_enu(TRUE[0], TRUE[1], 0.0,
                                           (ref[0], ref[1], 0.0))[:2])))) % 360
    gd = geo.bearing_between(ref[0], ref[1], *TRUE)
    assert abs(geo.angle_difference(gd, chord)) > 0.05


# ---------------------------------------------------------------------------
# 3 — consistency, outliers, and refusal
# ---------------------------------------------------------------------------

def test_bearings_from_two_emitters_are_reported_as_inconsistent():
    lobs = lobs_at(GOOD[:2], seed=1) + lobs_at(GOOD[2:], target=OTHER, seed=2)
    f = fu.ml_fix(lobs, reject_outliers=False)
    assert not f.consistent
    assert any("NOT CONSISTENT WITH ONE POSITION" in c for c in f.caveats)


def test_one_wild_bearing_is_rejected_and_named():
    """A least-squares fit is dragged an arbitrary distance by one outlier,
    and the ellipse stays small while it happens."""
    lobs = lobs_at(RUN[:8], seed=5)
    lobs[3].bearing_deg = (lobs[3].bearing_deg + 60.0) % 360
    lobs[3].source = "rx-bad"
    f = fu.ml_fix(lobs)
    assert len(f.dropped) == 1
    assert f.dropped[0]["source"] == "rx-bad"
    assert any("rejected as" in c for c in f.caveats)
    assert err(f) < err(fu.ml_fix(lobs, reject_outliers=False))


def test_rejection_stops_before_it_eats_the_whole_set():
    """A flat 3σ bar would eventually drop every bearing. The bar rises with
    the number of chances taken."""
    f = fu.ml_fix(lobs_at(RUN[:8], sigma=3.0, seed=7))
    assert f.n_obs >= 3


def test_a_span_too_wide_for_one_tangent_plane_is_refused():
    """A geolocation that has silently left the plane it was computed in is
    worse than no geolocation."""
    with pytest.raises(fu.SpanError, match="span"):
        fu.ml_fix(lobs_at([(51.5, -0.2), (-14.9, 63.0)]))


# ---------------------------------------------------------------------------
# 4 — fusing fixes
# ---------------------------------------------------------------------------

def test_fusing_two_fixes_beats_either_of_them():
    a = fu.ml_fix(lobs_at(GOOD[:2], seed=1))
    b = fu.ml_fix(lobs_at(GOOD[2:], seed=2))
    f = fu.fuse_fixes([a, b])
    assert f.ellipse(0.95)[0] < min(a.ellipse(0.95)[0], b.ellipse(0.95)[0])
    assert err(f) < max(err(a), err(b))


def test_fusing_different_emitters_excludes_rather_than_averages():
    """**A naive average lands confidently between two things.** Here that
    would be 23 km from either."""
    a = fu.ml_fix(lobs_at(GOOD[:2], seed=1))
    b = fu.ml_fix(lobs_at(GOOD[2:], seed=2))
    c = fu.ml_fix(lobs_at([(51.20, 0.20), (51.35, 0.70)], target=OTHER,
                          seed=3))
    f = fu.fuse_fixes([a, b, c])
    assert f.evidence["fixes_used"] == 2
    assert len(f.dropped) == 1
    assert err(f) < 2_000.0
    assert any("too far apart" in cv for cv in f.caveats)


def test_a_shared_systematic_component_survives_fusion():
    """Summing it away is what makes repeated observations of one biased
    system look far better than they are."""
    a = fu.ml_fix(lobs_at(RUN[:6], bias=3.0, seed=1), systematic_deg=3.0)
    b = fu.ml_fix(lobs_at(RUN[6:12], bias=3.0, seed=2), systematic_deg=3.0)
    f = fu.fuse_fixes([a, b])
    assert f.systematic_cov is not None
    assert any("does not average down" in c for c in f.caveats)
    floor = math.sqrt(np.trace(f.systematic_cov))
    assert math.sqrt(np.trace(f.cov)) >= floor * 0.99


def test_fusing_one_fix_returns_it_unchanged():
    a = fu.ml_fix(lobs_at(GOOD, seed=1))
    assert fu.fuse_fixes([a]) is a


def test_fusing_nothing_is_an_error():
    with pytest.raises(ValueError, match="nothing to fuse"):
        fu.fuse_fixes([])


# ---------------------------------------------------------------------------
# 5 — refine: a prior plus new evidence
# ---------------------------------------------------------------------------

def test_a_prior_fix_and_one_new_bearing_sharpen_each_other():
    prior = fu.ml_fix(lobs_at(GOOD[:2], seed=1))
    new = lobs_at([(51.75, -0.05)], sigma=0.5, seed=9)
    out = fu.refine(new, prior=prior)
    assert out.ellipse(0.95)[0] <= prior.ellipse(0.95)[0]
    assert err(out) <= err(prior) + 50.0


def test_one_bearing_and_no_prior_is_a_direction_not_a_geolocation():
    with pytest.raises(ValueError, match="is not a geolocation"):
        fu.refine(lobs_at(GOOD[:1]))


def test_a_single_bearing_constrains_across_itself_and_not_along():
    """Written as a covariance with one small eigenvalue and one enormous
    one, so the ordinary fusion machinery uses it correctly."""
    prior = fu.ml_fix(lobs_at(GOOD, seed=1))
    lob = lobs_at([(51.75, -0.05)], sigma=0.5, seed=9)[0]
    as_fix = fu._lob_as_fix(lob, prior, 0.0)
    w = np.linalg.eigvalsh(as_fix.cov)
    assert max(w) / min(w) > 1e4


def test_refine_takes_bearings_and_fixes_together():
    prior = fu.ml_fix(lobs_at(GOOD[:2], seed=1))
    out = fu.refine(lobs_at(GOOD[2:], seed=2), prior=prior)
    assert out.n_obs >= 3
    assert err(out) < 2_000.0


# ---------------------------------------------------------------------------
# 6 — the cone ambiguity
# ---------------------------------------------------------------------------

def ambiguous_set(stations, headings, sigma=0.05, seed=1):
    """LOBs as a linear array would produce them: the true bearing plus its
    mirror about the array axis."""
    out = []
    for (la, lo), h in zip(stations, headings):
        t = geo.bearing_between(la, lo, *TRUE)
        rel = geo.angle_difference(t, h)
        mirror = (h - rel) % 360.0
        lob = geo.LOB(lat=la, lon=lo, bearing_deg=t, sigma_deg=sigma,
                      ambiguous_with=mirror, source="array",
                      epoch=1.7e9 + len(out))
        # present them in an arbitrary order, as `cone_bearings` does
        if sorted([t % 360, mirror])[0] != t % 360:
            lob.bearing_deg, lob.ambiguous_with = mirror, t % 360
        out.append(lob)
    return out


def test_a_manoeuvre_resolves_the_cone_ambiguity():
    """**This is why a moving platform can geolocate with one linear
    array.** The mirror set is mirrored about an axis that MOVES, so only the
    true side stays self-consistent."""
    st = [(51.40, -0.30), (51.42, -0.20), (51.44, -0.10), (51.40, -0.02),
          (51.36, 0.04)]
    hd = [70.0, 70.0, 70.0, 140.0, 140.0]
    res, fix, note = fu.resolve_ambiguity(ambiguous_set(st, hd))
    assert "RESOLVED" in note
    for r in res:
        truth = geo.bearing_between(r.lat, r.lon, *TRUE)
        assert abs(geo.angle_difference(r.bearing_deg, truth)) < 1.0
    assert err(fix) < 500.0


def test_a_straight_run_cannot_and_reports_BOTH_positions():
    """**The negative control, and the reason a χ² margin is the wrong
    test.** Reflecting every bearing about a constant array axis reflects the
    whole picture about the flight line, so the emitter and its image both
    fit. Measured: 0.0024 against 0.2655 — a gap that looks decisive and
    means nothing, with the second answer 38 km away."""
    st = [(51.42, -0.30 + 0.02 * i) for i in range(6)]
    res, fix, note = fu.resolve_ambiguity(ambiguous_set(st, [90.0] * 6))
    assert "NOT RESOLVED" in note
    assert "or at" in note
    assert "mirror_fix" in fix.evidence
    assert fix.evidence["mirror_fix"]["separation_m"] > 1_000.0
    assert any("dog-leg" in c or "across the bearing" in c
               for c in fix.caveats)


def test_a_resolved_ambiguity_always_beat_a_materially_different_rival():
    """The invariant behind the alarm: when the note says RESOLVED, the
    rejected alternative was outside the fix's own ellipse. A mirror closer
    than that is reported as immaterial instead, because raising an alarm
    about a 600 m difference trains people to ignore it for the case where
    the mirror is 40 km away."""
    st = [(51.40, -0.30), (51.42, -0.20), (51.44, -0.10), (51.40, -0.02)]
    hd = [70.0, 70.0, 140.0, 140.0]
    _res, fix, note = fu.resolve_ambiguity(ambiguous_set(st, hd, sigma=3.0))
    if "RESOLVED" in note and "NOT RESOLVED" not in note:
        km = float(note.split("emitter ")[1].split(" km")[0].replace(",", ""))
        assert km * 1000.0 > fix.ellipse(0.95)[0]
    else:
        assert "not meaningfully different" in note or "NOT RESOLVED" in note


def test_two_ambiguous_bearings_cannot_be_resolved_at_all():
    st = [(51.40, -0.30), (51.44, -0.10)]
    res, fix, note = fu.resolve_ambiguity(ambiguous_set(st, [70.0, 140.0]))
    assert fix is None
    assert "at least three" in note


def test_unambiguous_bearings_pass_straight_through():
    lobs = lobs_at(GOOD, seed=1)
    res, fix, note = fu.resolve_ambiguity(lobs)
    assert res == lobs and fix is None
    assert "no ambiguous bearings" in note


# ---------------------------------------------------------------------------
# 7 — reporting
# ---------------------------------------------------------------------------

def test_the_ellipse_orientation_is_a_bearing_not_a_maths_angle():
    """An eigenvector gives the angle anticlockwise from east, which is what
    everybody plots by mistake. A map wants degrees true."""
    f = fu.Fix(lat=51.0, lon=0.0, cov=np.diag([100.0 ** 2, 1.0 ** 2]))
    a, b, ang = f.ellipse(0.95)
    assert a > b
    assert ang == pytest.approx(90.0, abs=1e-6)      # major axis runs east


def test_the_ellipse_scales_with_confidence():
    f = fu.Fix(lat=51.0, lon=0.0, cov=np.eye(2) * 100.0)
    assert f.ellipse(0.95)[0] > f.ellipse(0.5)[0]
    assert f.ellipse(0.95)[0] / f.ellipse(0.39)[0] == pytest.approx(2.4477,
                                                                    abs=1e-3)


def test_geojson_carries_the_point_and_a_closed_ellipse():
    f = fu.ml_fix(lobs_at(GOOD, seed=1))
    gj = f.to_geojson()
    kinds = [x["geometry"]["type"] for x in gj["features"]]
    assert kinds == ["Point", "Polygon"]
    ring = gj["features"][1]["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1]
    assert gj["features"][0]["geometry"]["coordinates"] == [f.lon, f.lat]


def test_a_fix_exports_as_dossier_measurements():
    f = fu.ml_fix(lobs_at(GOOD, seed=1))
    rows = fu.fix_dossier(f, lobs_at(GOOD, seed=1))
    names = {m.name for m in rows}
    assert {"fix_latitude", "fix_longitude", "fix_semi_major_95",
            "bearings_used"} <= names
    semi = next(m for m in rows if m.name == "fix_semi_major_95")
    assert semi.unit == "m"
    assert "gdop" in semi.evidence


def test_chi2_matches_its_closed_form_at_two_degrees_of_freedom():
    for x in (0.5, 2.0, 6.0, 20.0):
        assert fu.chi2_sf(x, 2) == pytest.approx(math.exp(-x / 2), rel=1e-12)
    assert fu.chi2_sf(3.84, 1) == pytest.approx(0.05, abs=0.002)
    assert fu.chi2_sf(11.07, 5) == pytest.approx(0.05, abs=0.002)


# ---------------------------------------------------------------------------
# 9 — the region is a PROTECTION region, not a probability contour
# ---------------------------------------------------------------------------
#
# A bound is a statement that the error is somewhere in ±a and nothing more.
# Turning it into σ = a/√3 assumes a rectangular distribution, which is right
# on average and wrong at the edge — measured, with the real bias sitting AT
# the bound the 95 % ellipse contained the emitter 77 % of the time on weak
# geometry. Two dimensions are what does it: √3 standard uncertainties along
# one axis costs 3.0 of the 5.99 the χ² budget has, and the random error in
# the other direction spends the rest.

def test_the_bound_is_carried_at_full_width_not_as_a_sigma():
    """The exact construction, checked against the closed form: along the
    direction the bias pushes, the region is k·σ_random PLUS the whole
    bound — added, not combined in quadrature."""
    f = fu.Fix(lat=51.6, lon=-0.05, cov=np.diag([1000.0 ** 2, 500.0 ** 2]),
               bound_axes={"g": [0.0, 2000.0]})
    a, b, _ang = f.ellipse(0.95)
    k = fu.ELLIPSE_P[0.95]
    assert min(a, b) == pytest.approx(k * 500.0 + 2000.0, rel=1e-9)


def test_the_outer_ellipse_never_cuts_the_true_region_short():
    """The reported ellipse has to CONTAIN the ellipse-swept-along-a-segment
    that the maths actually produces, in every direction and not only along
    the two it was built from."""
    rng = np.random.default_rng(4)
    for _ in range(20):
        c = rng.normal(size=(2, 2)) * 800.0
        c = c @ c.T + np.eye(2) * 1e4
        u = rng.normal(size=2) * 1500.0
        f = fu.Fix(lat=51.6, lon=-0.05, cov=c, bound_axes={"g": list(u)})
        a, b, ang = f.ellipse(0.95)
        r = math.radians(ang)
        # semi-axis directions of the reported ellipse, as east/north
        e_major = np.array([math.sin(r), math.cos(r)])
        e_minor = np.array([math.cos(r), -math.sin(r)])
        rep = np.outer(e_major, e_major) * a ** 2 + \
            np.outer(e_minor, e_minor) * b ** 2
        k = fu.ELLIPSE_P[0.95]
        for th in np.linspace(0, math.pi, 37):
            w = np.array([math.cos(th), math.sin(th)])
            true_support = k * math.sqrt(w @ c @ w) + abs(float(w @ u))
            assert math.sqrt(w @ rep @ w) >= true_support - 1e-6


def test_a_measured_sigma_is_not_carried_rigidly():
    """A measured figure IS a σ and belongs in quadrature. Only an assumed
    bound gets the rigid treatment, and supplying a measurement is the way
    to get the smaller region."""
    lobs = lobs_at(RUN[:8], seed=3)
    assumed = fu.ml_fix(lobs)
    measured = fu.ml_fix(lobs, systematic_deg=fu.bias_from_bound(3.0))
    assert not measured.bound_axes
    assert assumed.bound_axes
    assert measured.ellipse(0.95)[0] < assumed.ellipse(0.95)[0]


def test_the_region_holds_when_the_bias_sits_at_the_bound():
    """**THE MEASUREMENT THAT DROVE THE DESIGN.** Statistically the same
    data covers 77 %; carrying the bound rigidly puts it back above 90 %."""
    bound = fu.DEFAULT_BIAS_BOUND_DEG
    hits = 0
    for s in range(60):
        f = fu.ml_fix(lobs_at(RUN[:12], bias=bound, seed=600 + s))
        hits += err(f) <= f.r95_m
    assert hits / 60 >= 0.90


def test_the_provenance_says_which_of_the_three_ways_it_came_in():
    """'1.7°' means something quite different measured than assumed, and a
    fix that does not say which cannot be argued with."""
    sig, bound, note = fu.resolve_systematic()
    assert bound == fu.DEFAULT_BIAS_BOUND_DEG and "ASSIGNED" in note
    sig, bound, note = fu.resolve_systematic(bias_bound_deg=5.0)
    assert bound == 5.0 and sig == pytest.approx(5.0 / math.sqrt(3.0))
    sig, bound, note = fu.resolve_systematic(systematic_deg=0.4)
    assert bound == 0.0 and sig == 0.4 and "supplied" in note


def test_a_bound_splits_linearly_and_a_sigma_in_quadrature():
    """Variances add; bounds add at their limits. Using √ for both made one
    receiver's two parallel bound axes sum to 1.414 × the bound they came
    from — a 41 % over-statement dressed up as caution."""
    f = fu.ml_fix(lobs_at(RUN[:8], seed=9))
    tot = np.zeros(2)
    for u in f.bound_axes.values():
        tot = tot + np.abs(np.asarray(u))
    one = fu.ml_fix(lobs_at(RUN[:8], seed=9))
    # the two groups (this source, and the part common to all sources) must
    # reconstitute exactly one bound's worth of displacement
    assert len(one.bound_axes) == 2
    parts = [np.asarray(v) for v in one.bound_axes.values()]
    assert np.allclose(parts[0] / np.linalg.norm(parts[0]),
                       parts[1] / np.linalg.norm(parts[1]), atol=1e-9)
    assert (np.linalg.norm(parts[0]) + np.linalg.norm(parts[1])) == \
        pytest.approx(np.linalg.norm(tot), rel=1e-9)


# ---------------------------------------------------------------------------
# 10 — the sign of a shared bias, which a covariance throws away
# ---------------------------------------------------------------------------

def test_a_fix_records_which_way_a_shared_bias_would_push_it():
    f = fu.ml_fix(lobs_at(RUN[:8], seed=10))
    assert f.systematic_axes
    rebuilt = np.zeros((2, 2))
    for u in f.systematic_axes.values():
        u = np.asarray(u)
        rebuilt = rebuilt + np.outer(u, u)
    assert np.allclose(rebuilt, f.systematic_cov, atol=1e-6)


def test_the_sign_actually_changes_the_fused_answer():
    """v vᵀ is identical for v and −v, so a covariance cannot distinguish
    two fixes a shared bias pushes APART from two it pushes the same way.
    The fused uncertainty is different in the two cases and the store keeps
    the signs so it stays different."""
    f1 = fu.ml_fix(lobs_at(GOOD[:2], seed=11))
    f2 = fu.ml_fix(lobs_at(GOOD[2:], seed=12, source="rx2"))
    same = fu.fuse_fixes([f1, f2])
    flipped = fu.ml_fix(lobs_at(GOOD[2:], seed=12, source="rx2"))
    flipped.systematic_axes = {k: [-v[0], -v[1]]
                               for k, v in flipped.systematic_axes.items()}
    flipped.bound_axes = {k: [-v[0], -v[1]]
                          for k, v in flipped.bound_axes.items()}
    opposed = fu.fuse_fixes([f1, flipped])
    assert same.ellipse(0.95)[0] != pytest.approx(opposed.ellipse(0.95)[0],
                                                  rel=1e-3)


def test_a_fix_with_no_axes_is_assumed_to_reinforce_rather_than_cancel():
    """An older stored fix has a covariance and no signs. Guessing the
    cancellation would be claiming a benefit nobody demonstrated."""
    f1 = fu.ml_fix(lobs_at(GOOD[:2], seed=13))
    bare = fu.ml_fix(lobs_at(GOOD[2:], seed=14))
    bare.systematic_axes = {}
    bare.bound_axes = {}
    out = fu.fuse_fixes([f1, bare])
    assert out.systematic_cov is not None
    assert np.trace(out.systematic_cov) > 0


def test_two_receivers_beat_two_visits_from_one():
    """The reason a second receiver is worth deploying: its alignment error
    is a DIFFERENT error and partly cancels, while one receiver's is the
    same error twice and does not."""
    a = fu.ml_fix(lobs_at(RUN[:6], seed=15, source="rx-A"))
    a2 = fu.ml_fix(lobs_at(RUN[6:12], seed=16, source="rx-A"))
    b = fu.ml_fix(lobs_at(RUN[6:12], seed=16, source="rx-B"))
    one = fu.fuse_fixes([a, a2])
    two = fu.fuse_fixes([a, b])
    assert two.ellipse(0.95)[0] < one.ellipse(0.95)[0]


def test_a_pair_of_receivers_is_not_told_to_go_and_find_a_second_receiver():
    """The advice has to depend on what is already in play. Telling an
    analyst holding two independently calibrated systems to go and get one
    is how a caveat gets ignored for good."""
    a = fu.ml_fix(lobs_at(RUN[:6], seed=17, source="rx-A"))
    b = fu.ml_fix(lobs_at(RUN[6:12], seed=18, source="rx-B"))
    out = fu.fuse_fixes([a, b])
    if out.bias_limited:
        text = " ".join(out.caveats)
        assert "second, independently calibrated receiver" not in text
        assert "calibrating them against a known emitter" in text


def test_more_evidence_never_makes_the_answer_worse():
    """A precise fix from a badly aligned receiver deserves less weight than
    its scatter suggests. Weighting on the random part alone made adding one
    good bearing to a good prior turn a 5.1 km region into 6.8 km."""
    prior = fu.ml_fix(lobs_at(RUN[:6], seed=19, source="rx-A"))
    one = lobs_at(GOOD[:1], seed=20, source="rx-B")
    out = fu.refine([prior] + one)
    assert out.r95_m <= prior.r95_m * 1.001


# ---------------------------------------------------------------------------
# 11 — a shared bias and outlier rejection do not mix
# ---------------------------------------------------------------------------

def test_rejection_still_finds_the_culprit_under_a_correlated_model():
    """**A REAL FAILURE, FOUND BY TURNING THE BIAS TERM ON BY DEFAULT.**
    R = diag(σ²) + σ_sys²·11ᵀ is nearly singular along the all-ones
    direction, which hands the fit a nearly free global rotation. It then
    absorbs one wild bearing by swinging everything, spreads the residual
    over the innocent bearings, and the rejector drops FIVE of eight and
    keeps the culprit. Detection therefore runs uncorrelated."""
    lobs = lobs_at(RUN[:8], seed=21)
    lobs[3].bearing_deg = (lobs[3].bearing_deg + 45.0) % 360
    lobs[3].source = "rx-bad"
    f = fu.ml_fix(lobs)
    assert [d["index"] for d in f.dropped] == [3]
    assert err(f) < 2_000


def test_the_answer_still_uses_the_correlated_model():
    """Detection is uncorrelated; the fix and its covariance are not. Losing
    the correlated model here would lose the bias floor entirely."""
    f = fu.ml_fix(lobs_at(RUN[:8], seed=22))
    assert f.systematic_cov is not None
    assert float(np.trace(f.systematic_cov)) > 0


def test_the_branch_choice_is_scored_without_the_shared_term_too():
    """The same trap as outlier rejection, one layer up. A bias common to
    every bearing cannot distinguish one branch of a cone from the other, so
    scoring under a near-singular R just hands the fit a free rotation. It
    made the straight-run negative control — where BOTH answers genuinely
    fit — come back as confidently resolved."""
    src = inspect.getsource(fu.resolve_ambiguity)
    assert "systematic_deg=0.0" in src
    # ...and the answer that comes back is nonetheless the correlated fit
    st = [(51.42, -0.30), (51.44, -0.20), (51.46, -0.10), (51.42, -0.02),
          (51.38, 0.04)]
    hdg = [70.0, 70.0, 70.0, 140.0, 140.0]
    res, fix, note = fu.resolve_ambiguity(ambiguous_set(st, hdg))
    if fix is not None:
        assert fix.bound_axes, "the returned fix lost its error model"
