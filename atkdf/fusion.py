"""Many observations of one emitter, combined into one fix — honestly.

Two jobs that are really the same job. Triangulating a fix from several lines
of bearing, and tightening an existing fix with new evidence, are both "given
observations with uncertainties, find the position and say how well you know
it". So they share an estimator and an error model, and a new bearing can
sharpen a fix taken last week.

**THE FAILURE THIS MODULE IS BUILT AROUND: MORE OBSERVATIONS ALWAYS MAKE THE
ELLIPSE SMALLER, AND THE ELLIPSE IS NOT ALWAYS WHERE THE EMITTER IS.**

Take a DF system with a 3° antenna alignment error — a calibration that was
done once, on the ground, and has been slightly wrong ever since. Every
bearing it produces is 3° off in the same direction. Fuse ten of them the
obvious way and the random scatter averages down by √10 while the 3° does not
average down at all, because it is the same 3° every time. The estimator
reports an ellipse √10 times tighter, centred on a point that is
systematically displaced, and every consistency check passes — the bearings
agree beautifully with each other. They are all wrong together.

So every observation here belongs to a **bias group**, and errors within a
group are modelled as correlated. The random part averages down; the shared
part goes straight through to the answer and stays there. When the shared part
dominates, `Fix.bias_limited` is True and `Fix.caveats` says that collecting
more of the same will not help — which is the actionable finding, because the
fix is telling you to go and move the receiver rather than take more data.

Measured on a west-to-east collection run against an emitter 20 km off track,
1° random bearing error, 40 trials per row — the fraction is how often the
truth actually fell inside the stated 95 % ellipse:

    bearings   no bias      3° shared bias      3° shared bias
               (baseline)   NOT modelled        modelled
    ---------  ----------   -----------------   ------------------
        2       90 %         90 %                 90 %
        4       95 %         90 %                 90 %
        8       98 %         90 %                100 %
       16       98 %          0 %                100 %

Read the third column, and read it downwards: the more data arrives, the worse
it gets, ending at **not once in forty trials**. Every internal check passes
while that happens, because the bearings genuinely do agree with each other.

The fourth column is the same data with one number supplied. The ellipse stops
shrinking — 5,992 m at four bearings, 3,286 at eight, 3,003 at sixteen,
converging on the floor the bias imposes rather than running past it — the
truth stays inside it, and the fix reports itself `bias_limited` so the
analyst knows that another hour of the same collection buys nothing.

**Which is why the shared term is never zero by default.** The third column
is what "we have not measured the alignment, so leave it out" actually
produces. The default is a Type B assignment from a ±3° bound — see
`DEFAULT_BIAS_BOUND_DEG` — and the bound is then carried RIGIDLY rather than
statistically, for a reason that took a measurement to find. Folding
σ = 3/√3 into the covariance covers correctly when the real bias is drawn
from that rectangular law, and covered only **77 %** with the real bias
sitting AT the bound. A bound is not a promise that the error is usually
small. `Fix.ellipse` explains the construction; `coverage.py` measures it,
and this is that measurement, 200 trials a row, the fraction being how often
the truth was inside the region the bench reported:

    bearings   no shared error   bias from the law   bias AT the bound
    ---------  ---------------   -----------------   -----------------
        2       94 %              92 %                94 %
        4       98 %              98 %                96 %
        8      100 %             100 %                99 %
       16      100 %             100 %                99 %

    two fixes fused, one receiver       99 %                96 %
    two fixes fused, two receivers     100 %                99 %

Every row holds at the edge of the assumption, not only in its middle, which
is what "err on the side of accuracy" has to mean for a number somebody
drives towards. The cost is a larger region when the equipment happens to be
well aligned, and the cure for that is a measurement: boresight against a
known emitter, pass `systematic_deg`, and the rigid term disappears.

**The receiver's own position is the second term nobody models.** A bearing
taken from a point known to ±500 m is not a 1° bearing, whatever the DF
hardware says — at 22 km that position error alone subtends 1.3°, and it is
independent of everything the array is doing. `angular_sigma_deg` converts it
at the RANGE of the current estimate, which is why the solver rebuilds its
weight matrix every iteration. 200 trials at 22 km, eight bearings:

    receiver position error   ignored    declared
    -----------------------   -------    --------
        100 m                 100 %       100 %
        250 m                 100 %       100 %
        500 m                  96 %       100 %
      1 000 m                  78 %        99 %

The alignment bound above is carrying the first two rows: with it removed as
well — a system claimed to be perfectly aligned AND perfectly located, which
is what "leave out what you have not measured" produces in full — the same
1 km case covers 53 %. `ephemeris.py` supplies the position figure per state,
and it grows with the extrapolation when a state has to be interpolated or
held.

**Two more findings, both of which look like bugs in something else.**

  * A shared bias and outlier rejection do not mix. R = diag(σ²) + σ_sys²·11ᵀ
    is nearly singular along the all-ones direction, so the fit gets a nearly
    free global rotation, absorbs one wild bearing by swinging everything,
    and the rejector then removes the innocent bearings and keeps the
    culprit — five of eight, on a measured case. Detection therefore runs on
    the uncorrelated model and the answer on the correlated one; see
    `ml_fix`.
  * A rank-one covariance cannot say which WAY a shared bias pushed a fix,
    because v vᵀ is the same for v and −v. Fusing fixes needs that sign, so
    `Fix.systematic_axes` keeps it. Without it, two fixes displaced in
    opposite directions look exactly like two displaced the same way.

The rest of the module is the ordinary care this kind of estimator needs:

  * **Two lines always cross.** An intersection is not evidence of anything.
    `gdop` says whether the geometry supports a fix at all, and near-parallel
    bearings produce a long thin ellipse rather than a confident point.
  * **Fusing observations of DIFFERENT emitters produces a confident fix
    between them.** `consistency` is a chi-squared test that asks whether
    these observations can be explained by any single position, and the
    answer is allowed to be no.
  * **Stansfield's estimator is biased towards the receivers.** It is the
    classic and it is here, but the default is the iterated maximum-likelihood
    form, and `test_siga_fusion.py` measures the difference rather than
    asserting the textbook claim.

MOVED FROM ATK, 2026-09-30 (ATK c52b913, `atk/core/siga/fusion.py`), with its tests
and this docstring. ATK's `atk.core.siga.fusion` is now an alias of this
module — the same module object, reached through `atk/core/df_engine.py`
— so the Geolocation tab and ATK's own tests run this code. Edit it here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import geodesy as geo
from .dossier import DIMENSIONLESS, Measurement

#: Confidence multipliers for a 2-D error ellipse: the ellipse scales by
#: sqrt(-2 ln(1-p)), which for two degrees of freedom is exact rather than
#: approximate. 0.5 is the CEP-equivalent, 0.95 the one to brief.
ELLIPSE_P = {0.39: 1.0, 0.5: 1.177_410, 0.9: 2.145_966, 0.95: 2.447_747,
             0.99: 3.034_854}

#: Below this, the answer is limited by what is common to every observation
#: rather than by how many there are — the ratio of the systematic-only
#: ellipse to the full one.
BIAS_LIMITED_RATIO = 0.7

#: A fit whose consistency p-value falls below this is reporting that no
#: single position explains these observations.
CONSISTENCY_ALPHA = 0.01

#: The confidence every headline number is reported at, unless asked
#: otherwise. **95 %, not 50 %.** CEP — Circular Error Probable — is the
#: 50 % radius by definition, and quoting one without saying so invites the
#: reader to hear "the emitter is within this" when the honest reading is
#: "there is a one-in-two chance the emitter is within this". Every figure
#: this module produces carries its level.
REPORT_P = 0.95

#: Half-width of the bound assumed for a DF system's shared alignment error
#: when nobody has measured it.
#:
#: **ASSUMING ZERO IS NOT THE NEUTRAL CHOICE — IT IS AN UNSUPPORTED CLAIM
#: THAT THE EQUIPMENT IS PERFECT**, and it is the setting that produced 0 %
#: coverage in the table above. The standard metrological answer to "this
#: term exists, nobody has measured it" is a Type B evaluation: assign a
#: distribution from what is known about the instrument, say that you did,
#: and let a real measurement replace it. Refusing to state an uncertainty
#: because it has not been measured is not more rigorous, it is just silent.
#:
#: 3° is the bound for an installation nobody has calibrated: a vehicle
#: compass is routinely 1–2° out from magnetic deviation alone, and an
#: antenna bolted to a mast by eye adds its own. Well-installed systems do
#: much better — see `BIAS_BOUNDS` — and any measured figure should be
#: passed instead.
DEFAULT_BIAS_BOUND_DEG = 3.0

#: How much of an assumed alignment bound is COMMON TO EVERY RECEIVER rather
#: than particular to one of them, as a fraction of the VARIANCE.
#:
#: A second receiver is worth having because its alignment error is a
#: different error — but only part of it is. Two vehicles fitted by the same
#: unit, running the same magnetic model, working to the same map datum and
#: the same convention for which way "zero" points, are wrong together in all
#: of those ways however independently their antennas were bolted on.
#: Treating a second source as fully independent is the optimistic assumption
#: and it under-covered badly when measured: with both receivers sitting at
#: the ±3° bound the truth fell inside the stated 95 % region 59 % of the
#: time. Treating it as fully shared is the pessimistic one and makes a
#: second receiver worthless, which is both wrong and the kind of wrong that
#: teaches an analyst to ignore the ellipse.
#:
#: So the bound is split: half the variance common, half particular. The
#: split is an assignment like the bound itself, it is stated rather than
#: hidden, and `coverage.py` measures what it buys.
COMMON_BIAS_FRACTION = 0.5

#: The bias-group key for the part every source shares. Not a source name, so
#: it can never collide with one.
COMMON_GROUP = "*"

#: Bounds by how the system was installed, for `bias_from_bound`. These are
#: assignments, not measurements: they are the honest starting point when the
#: alternative is claiming zero, and the first thing to replace once anybody
#: has run a calibration against a known emitter.
BIAS_BOUNDS = {
    "surveyed": 0.5,       # boresighted against a known emitter, INS-aligned
    "installed": 1.5,      # professionally fitted, alignment recorded
    "field": 3.0,          # bolted on and pointed by eye; the default
    "unknown": 5.0,        # provenance unknown — assume the worst plausible
}


def bias_from_bound(bound_deg: float = DEFAULT_BIAS_BOUND_DEG) -> float:
    """A bound on a systematic error → a standard uncertainty, GUM Type B.

    "The alignment is somewhere within ±3°" is a statement about a
    DISTRIBUTION, not a σ. With no reason to prefer any value inside the
    bound, the distribution is rectangular, and a rectangular distribution of
    half-width a has standard deviation a/√3. Feeding the bound in directly
    as a σ would overstate the uncertainty by 73 %, which is its own kind of
    wrong — an ellipse nobody believes gets switched off.
    """
    return float(bound_deg) / math.sqrt(3.0)


def resolve_systematic(systematic_deg=None, bias_bound_deg=None) -> tuple:
    """`(sigma_deg, bound_deg, provenance)` — how the shared term was arrived at.

    Three ways in, and the answer says which was used, because "1.7°" means
    something quite different when it was measured than when it was assumed.

    **The bound comes back separately from the σ**, and the difference decides
    how the reported region is built. A MEASURED σ describes scatter and
    combines in quadrature with everything else. A BOUND is a statement that
    the error is somewhere in ±a and nothing more; converting it to a σ via
    /√3 assumes a rectangular distribution, and that assumption is fine on
    average and wrong at the edge — with the real bias sitting at the bound
    the 95 % ellipse contained the emitter 77 % of the time on weak geometry.
    `Fix.ellipse` therefore carries the bound rigidly rather than statistically.
    See `Fix.ellipse` for the construction.
    """
    if systematic_deg is not None:
        return float(systematic_deg), 0.0, (
            f"σ = {float(systematic_deg):.2f}° as supplied — a measured or "
            f"otherwise known figure, so it is combined in quadrature like "
            f"any other statistical term")
    if bias_bound_deg is not None:
        b = float(bias_bound_deg)
        s = bias_from_bound(b)
        return s, b, (
            f"±{b:.2f}° bound as supplied → σ = {s:.2f}° (rectangular, /√3) "
            f"for the statistics, and the full ±{b:.2f}° is carried rigidly "
            f"in the reported region so it holds wherever in the bound the "
            f"true alignment error actually sits")
    b = float(DEFAULT_BIAS_BOUND_DEG)
    s = bias_from_bound(b)
    return s, b, (
        f"σ = {s:.2f}° ASSIGNED, not measured: no alignment figure was given, "
        f"so a Type B uncertainty was taken from a ±{b:g}° bound typical of "
        f"an uncalibrated DF installation. This is a deliberate assumption "
        f"and it inflates the region — assuming zero instead would be the "
        f"claim that the equipment is perfectly aligned, which is what makes "
        f"an ellipse that never contains the emitter. Boresight against a "
        f"known emitter and pass the measured figure.")

#: How far the inputs may spread around the tangent point before the local
#: east/north plane stops being a fair description of the earth.
#:
#: Every estimator here does its covariance algebra in a LOCAL FLAT FRAME.
#: The conversion into it is exact, but treating that frame as Cartesian is
#: not, and the error grows as the square of the distance from the tangent
#: point. At 500 km it is already kilometres; at a few thousand it is
#: nonsense that still returns a number and an ellipse. Observed while
#: testing: one absurd fix at 63° E pulled the tangent point 3,700 km from
#: the others and the fused answer landed in Switzerland with a 600 m
#: ellipse. So the span is checked and the answer refused, because a
#: geolocation that has silently left the plane it was computed in is worse
#: than no geolocation.
MAX_TANGENT_SPAN_M = 500_000.0


# ---------------------------------------------------------------------------
# small statistics, so the thresholds are computed rather than looked up
# ---------------------------------------------------------------------------

def _gammap(a: float, x: float) -> float:
    """Regularised lower incomplete gamma P(a, x). Series and continued
    fraction, the standard split at x < a+1."""
    if x <= 0:
        return 0.0
    if x < a + 1.0:
        term = 1.0 / a
        total = term
        n = a
        for _ in range(500):
            n += 1.0
            term *= x / n
            total += term
            if abs(term) < abs(total) * 1e-14:
                break
        return total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    # Lentz's continued fraction for Q(a, x)
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    q = math.exp(-x + a * math.log(x) - math.lgamma(a)) * h
    return 1.0 - q


def chi2_sf(x: float, dof: int) -> float:
    """P(χ² > x). For 2 dof this is exactly exp(−x/2)."""
    if dof <= 0:
        return 1.0
    if x <= 0:
        return 1.0
    if dof == 2:
        return math.exp(-x / 2.0)
    return max(0.0, min(1.0, 1.0 - _gammap(dof / 2.0, x / 2.0)))


# ---------------------------------------------------------------------------
# the fix
# ---------------------------------------------------------------------------

@dataclass
class Fix:
    """A position, its covariance, and everything needed to disbelieve it."""
    lat: float
    lon: float
    cov: np.ndarray                       # 2×2 east/north metres², ENU
    n_obs: int = 0
    method: str = ""
    epoch: float = 0.0
    gdop: float = float("nan")
    residual_rms_deg: float = float("nan")
    consistency_p: float = float("nan")
    systematic_cov: np.ndarray | None = None    # the part that will not shrink
    #: Bias group → the SIGNED displacement of this fix, in east/north
    #: metres, per one standard uncertainty of that group's shared error.
    #: `systematic_cov` is the sum of their outer products and throws the
    #: signs away; fusing correctly needs them, because two fixes whose
    #: sensitivities point opposite ways cancel each other's bias and two
    #: that point the same way reinforce it. Recovering that from a rank-one
    #: covariance is impossible — the outer product is identical either way.
    systematic_axes: dict = field(default_factory=dict)
    #: Bias group → the SIGNED displacement, east/north metres, at the FULL
    #: assumed bound rather than at one standard uncertainty. Present only
    #: when the shared term was ASSIGNED from a bound; a measured σ leaves
    #: this empty and is handled statistically. `ellipse` carries these
    #: rigidly — see there for why.
    bound_axes: dict = field(default_factory=dict)
    #: Which installation produced it — the key that decides whether two
    #: fixes share an alignment error or carry independent ones. Two fixes
    #: from one receiver cannot average its calibration away; two fixes from
    #: genuinely different receivers can, and that is the whole reason a
    #: second receiver is worth having.
    source: str = ""
    dropped: list = field(default_factory=list)
    caveats: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)

    # -- the ellipse ----------------------------------------------------
    def statistical_cov(self) -> np.ndarray:
        """`cov` with an ASSIGNED bound's statistical stand-in taken back out.

        `cov` is the full statistical covariance and includes σ = bound/√3
        for an assigned alignment bound. `ellipse` carries that bound rigidly
        instead, so the statistical version has to come out first or the same
        error is counted twice. A MEASURED systematic stays in — it is a real
        σ and belongs in quadrature with everything else.
        """
        c = np.asarray(self.cov, dtype=float).reshape(2, 2)
        if not self.bound_axes or self.systematic_cov is None:
            return c
        out = c - np.asarray(self.systematic_cov, dtype=float).reshape(2, 2)
        w, v = np.linalg.eigh(0.5 * (out + out.T))
        return (v * np.maximum(w, 0.0)) @ v.T

    def ellipse(self, p: float = 0.95) -> tuple:
        """`(semi_major_m, semi_minor_m, orientation_deg)` at confidence `p`.

        Orientation is the bearing of the major axis, degrees true, which is
        what a map wants — not the mathematical convention measured
        anticlockwise from east, which is what an eigenvector gives you and
        what everybody plots by mistake.

        **THIS IS A PROTECTION REGION, NOT A PURE PROBABILITY CONTOUR**,
        whenever the shared alignment error was ASSIGNED from a bound rather
        than measured. The random part is scaled by the usual
        k = √(−2 ln(1−p)) and each bias group's bound is then added RIGIDLY,
        as a displacement of up to its full half-width in the direction it
        would actually push this fix.

        The reason is a measurement. Treating a ±3° bound as a rectangular
        distribution and folding σ = 3/√3 into the covariance is correct on
        average — with the bias drawn from that distribution, coverage came
        out at 96–100 %. But a bound is not a promise that the error is
        usually small, and with the true alignment sitting AT the bound,
        coverage on weak geometry fell to **77 %**. Two dimensions are unkind
        here: a displacement of √3 standard uncertainties along one axis eats
        3.0 of the 5.99 χ² budget a 95 % ellipse has, and the random error in
        the other direction spends the rest. Carrying the bound rigidly holds
        wherever inside it the truth sits, which is the only thing a bound
        actually entitles you to claim.

        The cost is a larger region when the real alignment error happens to
        be small. That is the direction to be wrong in, and there is a cure:
        boresight against a known emitter and pass `systematic_deg`, and the
        rigid term disappears entirely in favour of the measured σ.

        Ellipse ⊕ segment is folded in one segment at a time by the standard
        outer-ellipsoid bound with t = |b| / σ_C(b̂), which is EXACT along the
        segment's own direction — the direction that matters, since that is
        where the bias pushes — and conservative elsewhere.
        """
        k = ELLIPSE_P.get(round(float(p), 2), math.sqrt(-2.0 * math.log(
            max(1e-12, 1.0 - float(p)))))
        c = (k ** 2) * self.statistical_cov()
        for u in (self.bound_axes or {}).values():
            u = np.asarray(u, dtype=float).reshape(2)
            nb = float(np.linalg.norm(u))
            if nb <= 0:
                continue
            hat = u / nb
            sc = math.sqrt(max(float(hat @ c @ hat), 0.0))
            if sc <= 0:
                c = c + np.outer(u, u)
                continue
            # t = σ_C(b̂) / |b| minimises (1+1/t)σ² + (1+t)|b|², whose
            # minimum is exactly (σ + |b|)². Inverting the ratio looks just
            # as plausible and inflates the axis by 12 % here — checked
            # against the closed form in `test_siga_fusion.py`.
            t = sc / nb
            c = (1.0 + 1.0 / t) * c + (1.0 + t) * np.outer(u, u)
        w, v = np.linalg.eigh(0.5 * (c + c.T))
        w = np.maximum(w, 0.0)
        order = np.argsort(w)[::-1]
        major, minor = math.sqrt(w[order[0]]), math.sqrt(w[order[1]])
        e, n = v[0, order[0]], v[1, order[0]]
        return major, minor, math.degrees(math.atan2(e, n)) % 180.0

    @property
    def cep_m(self) -> float:
        """Circular error probable — the radius containing half the
        probability.

        The single number people ask for, and it throws away the shape, which
        for a bearings-only fix is usually the important part: a 500 m CEP
        can be a 500 m circle or a 4 km × 60 m sliver, and only one of those
        tells you which way to drive. `ellipse` is the honest output; this is
        here because reports ask for CEP.

        The standard approximation from the 1σ semi-axes, which is what
        `ellipse(0.39)` returns — the ellipse whose scale factor is exactly 1.
        Any assigned alignment bound is inside it on the same rigid footing
        as everywhere else, so this is a 50 % figure that holds wherever in
        the bound the truth sits, rather than a 50 % figure that assumes the
        equipment is average. It is therefore larger than a textbook CEP
        computed from the same random errors, and deliberately so.
        """
        a, b, _ = self.ellipse(0.39)
        return 0.589 * (a + b)

    @property
    def r95_m(self) -> float:
        """The radius of a CIRCLE about the fix that contains the emitter at
        least `REPORT_P` of the time.

        The semi-major axis, not the geometric-mean radius, so the circle
        encloses the whole 95 % ellipse rather than cutting its ends off.
        For an elongated fix that is pessimistic — a 4 km × 60 m sliver gets
        a 4 km circle — which is the correct direction to be wrong in, and
        why `ellipse` is still what gets plotted.
        """
        a, b, _ = self.ellipse(REPORT_P)
        return max(a, b)

    @property
    def bias_limited(self) -> bool:
        """True when more of the same observations will not help.

        Measured on the REPORTED REGION rather than on the covariance,
        because that is the thing an analyst acts on: take the shared terms
        away and see how much of the region survives. If most of it does not,
        the answer is limited by calibration and not by how much data there
        is, and another hour of collection buys a smaller ellipse in the same
        wrong place.
        """
        a, _b, _ang = self.ellipse(REPORT_P)
        if a <= 0:
            return False
        bare = Fix(lat=self.lat, lon=self.lon,
                   cov=self.statistical_cov() if self.bound_axes else
                   (np.asarray(self.cov, dtype=float)
                    - (np.zeros((2, 2)) if self.systematic_cov is None
                       else np.asarray(self.systematic_cov, dtype=float))))
        try:
            a0, _b0, _a0 = bare.ellipse(REPORT_P)
        except (np.linalg.LinAlgError, ValueError):
            return False
        return bool(a0 / a <= 1.0 - BIAS_LIMITED_RATIO ** 2)

    @property
    def consistent(self) -> bool:
        """Whether any single position explains these observations.

        NaN means the test could not be run — two bearings have no spare
        degrees of freedom — and that is reported as consistent because
        "untested" must not read as "failed". The two-bearing caveat says so
        in words.
        """
        return (math.isnan(self.consistency_p)
                or self.consistency_p >= CONSISTENCY_ALPHA)

    def describe(self) -> str:
        a, b, ang = self.ellipse(REPORT_P)
        bits = [f"{self.lat:.6f}, {self.lon:.6f}",
                f"{REPORT_P:.0%} ellipse {a:,.0f} × {b:,.0f} m at {ang:.0f}°",
                f"{self.n_obs} observations"]
        if self.method:
            bits.append(self.method)
        return " · ".join(bits)

    def to_geojson(self, p: float = 0.95, points: int = 72) -> dict:
        """The fix and its ellipse, ready for the map."""
        a, b, ang = self.ellipse(p)
        th = np.linspace(0, 2 * np.pi, int(points))
        # ellipse in east/north, then rotated to the major-axis bearing
        ex, ny = a * np.sin(th), b * np.cos(th)
        r = math.radians(ang)
        e = ex * math.cos(r) + ny * math.sin(r)
        n = -ex * math.sin(r) + ny * math.cos(r)
        lat, lon, _ = geo.enu_to_geodetic(e, n, np.zeros_like(e),
                                          (self.lat, self.lon, 0.0))
        ring = [[float(lo), float(la)] for la, lo in zip(np.atleast_1d(lat),
                                                         np.atleast_1d(lon))]
        ring.append(ring[0])
        return {"type": "FeatureCollection", "features": [
            {"type": "Feature",
             "geometry": {"type": "Point", "coordinates": [self.lon, self.lat]},
             "properties": {"method": self.method, "n_obs": self.n_obs,
                            "cep50_m": round(self.cep_m, 1),
                            "cep_note": "CEP is the 50 % radius; the polygon "
                                        "is the "
                                        f"{REPORT_P:.0%} contour",
                            "bias_limited": self.bias_limited}},
            {"type": "Feature",
             "geometry": {"type": "Polygon", "coordinates": [ring]},
             "properties": {"confidence": p,
                            "semi_major_m": round(a, 1),
                            "semi_minor_m": round(b, 1),
                            "orientation_deg": round(ang, 1)}}]}

    def to_measurements(self) -> list:
        """The fix as dossier rows, so it exports with everything else."""
        a, b, ang = self.ellipse(REPORT_P)
        out = [
            Measurement("fix_latitude", float(self.lat), "deg",
                        confidence=0.8 if self.consistent else 0.2,
                        method=self.method, evidence=dict(self.evidence)),
            Measurement("fix_longitude", float(self.lon), "deg",
                        confidence=0.8 if self.consistent else 0.2,
                        method=self.method),
            Measurement("fix_semi_major_95", float(a), "m", 0.8,
                        self.method,
                        evidence={"confidence": REPORT_P,
                                  "semi_minor_m": round(b, 1),
                                  "orientation_deg": round(ang, 1),
                                  "cep50_m": round(self.cep_m, 1),
                                  "gdop": round(self.gdop, 3),
                                  "observations": self.n_obs}),
        ]
        for c in self.caveats:
            out[0].add_caveat(c)
        return out


# ---------------------------------------------------------------------------
# the error model
# ---------------------------------------------------------------------------

def _bias_groups(lobs) -> dict:
    """Which observations share a systematic error, and how much of one.

    Default grouping is by `source` — one receiver, one calibration, one
    alignment, one shared bias. A caller who knows better can set
    `lob.evidence["bias_group"]`.
    """
    groups: dict = {}
    for i, lob in enumerate(lobs):
        key = (lob.evidence.get("bias_group") if isinstance(lob.evidence, dict)
               else None) or lob.source or lob.method or "default"
        groups.setdefault(key, []).append(i)
    return groups


def _selector(n: int, ids) -> np.ndarray:
    e = np.zeros(int(n))
    e[list(ids)] = 1.0
    return e


def _source_key(lobs) -> str:
    """One string naming the installation(s) behind a set of observations.

    It is what `fuse_fixes` groups on later, so it has to survive being
    written to disk and read back next month. Several groups are joined
    rather than reduced to the first: a fix built from two receivers shares
    its bias with any other fix from EITHER of them, and the joined key keeps
    that visible instead of quietly claiming independence.
    """
    keys = sorted(_bias_groups(lobs).keys()) if lobs else []
    return "+".join(str(k) for k in keys)


def angular_sigma_deg(lobs, ranges_m=None) -> np.ndarray:
    """Per-bearing σ in degrees, INCLUDING the receiver's own position error.

    **A BEARING FROM A PLACE YOU ONLY KNOW TO 500 METRES IS NOT A 1° BEARING.**
    A receiver uncertain by s metres, observing a target at range d, produces
    a bearing uncertain by a further s/d radians — the component of the
    position error across the line of sight. The component along it changes
    nothing, which is why this is s/d and not something larger.

    The ephemeris layer has always computed that s — interpolation gaps,
    stale fixes, a track that ran out — carried it on every state, and the
    estimator ignored it completely. Measured over 60 trials at a 22 km mean
    range, with the platform position uncertain by 500 m: coverage of the
    stated 95 % ellipse was **80 %** with the term dropped and 95–100 % with
    it folded in. At 22 km, 500 m is 1.3° — larger than a good array's own
    error, and previously invisible.

    It depends on RANGE, so it cannot be computed until there is an estimate
    to measure the range to; `ml_fix` recomputes it every iteration.
    """
    base = np.array([max(float(l.sigma_deg), 1e-6) for l in lobs])
    if ranges_m is None:
        return base
    d = np.maximum(np.asarray(ranges_m, dtype=float), 1.0)
    pos = np.array([max(float(getattr(l, "position_sigma_m", 0.0) or 0.0), 0.0)
                    for l in lobs])
    return np.hypot(base, np.degrees(pos / d))


def measurement_covariance(lobs, systematic_deg: float = 0.0,
                           ranges_m=None) -> np.ndarray:
    """The N×N angular covariance, in radians², INCLUDING shared bias.

    Diagonal is each bearing's own random error, plus its share of the
    platform's position uncertainty. The off-diagonal blocks are what makes
    this module different from the textbook version: two bearings from the
    same DF system with the same alignment error are not independent
    observations, and treating them as independent is what produces a tight
    ellipse around the wrong point.
    """
    sig = np.radians(angular_sigma_deg(lobs, ranges_m))
    r = np.diag(sig ** 2)
    if systematic_deg and systematic_deg > 0:
        s2 = math.radians(float(systematic_deg)) ** 2
        for _key, idx in _bias_groups(lobs).items():
            for i in idx:
                for j in idx:
                    r[i, j] += s2
    return r


# ---------------------------------------------------------------------------
# triangulation
# ---------------------------------------------------------------------------

class SpanError(ValueError):
    """The inputs are too far apart for one local tangent plane."""


def _check_span(points, ref, what: str) -> None:
    d = max((geo.distance_between(ref[0], ref[1], la, lo)
             for la, lo in points), default=0.0)
    if d > MAX_TANGENT_SPAN_M:
        raise SpanError(
            f"these {what} span {d / 1000:,.0f} km around their centre, and "
            f"every estimator here works in ONE local east/north plane, "
            f"which stops describing the earth past about "
            f"{MAX_TANGENT_SPAN_M / 1000:,.0f} km. Split them into groups "
            f"that belong together, or fix whichever one is wildly out — a "
            f"result computed across a span like this is arithmetic on a "
            f"plane that is not there.")


def _setup(lobs, ref=None):
    if len(lobs) < 2:
        raise ValueError("a fix needs at least two lines of bearing")
    if ref is None:
        ref = (float(np.mean([l.lat for l in lobs])),
               float(np.mean([l.lon for l in lobs])), 0.0)
    _check_span([(l.lat, l.lon) for l in lobs], ref, "receiver positions")
    p = np.zeros((len(lobs), 2))
    for i, l in enumerate(lobs):
        e, n, _u = geo.geodetic_to_enu(l.lat, l.lon, 0.0, ref)
        p[i] = (float(e), float(n))
    th = np.radians(np.array([l.bearing_deg for l in lobs], dtype=float))
    # bearing is clockwise from north, so the unit vector is (sin, cos) in
    # (east, north) and the left normal is (cos, −sin)
    u = np.stack([np.sin(th), np.cos(th)], axis=1)
    nrm = np.stack([np.cos(th), -np.sin(th)], axis=1)
    return ref, p, u, nrm


def _predicted(lobs, ref, x) -> np.ndarray:
    """Geodesic bearings from each origin to an ENU point. Not chord azimuths.

    **THE PREDICTION HAS TO BE THE SAME KIND OF ANGLE AS THE MEASUREMENT.** A
    LOB's `bearing_deg` is a geodesic azimuth — that is what `direct` walks
    along and what a compass means. The obvious `atan2(Δeast, Δnorth)` in the
    local frame is a CHORD azimuth, and the two differ by the convergence of
    the meridians: measured at 51° N over 25 km, 0.196°.

    That is not a rounding difference. It is four times the bearing error of a
    good array, it varies systematically with position across the collection,
    and it biased the fit by 89 m on synthetic data where every bearing was
    accurate to 0.008°. The residuals looked plausible — tens of a degree —
    which is exactly why it survived until the per-bearing errors were
    compared against the fitted residuals and turned out to be ten times
    smaller.
    """
    lat, lon, _a = geo.enu_to_geodetic(float(x[0]), float(x[1]), 0.0, ref)
    lat = float(np.atleast_1d(lat)[0])
    lon = float(np.atleast_1d(lon)[0])
    return np.array([geo.bearing_between(l.lat, l.lon, lat, lon)
                     for l in lobs], dtype=float)


def _linear_intersection(p, nrm, w=None) -> np.ndarray:
    """Weighted least-squares crossing point. The starting estimate."""
    w = np.ones(len(p)) if w is None else np.asarray(w, dtype=float)
    a = nrm * w[:, None]
    ata = nrm.T @ a
    atb = a.T @ np.einsum("ij,ij->i", nrm, p)
    return np.linalg.solve(ata + 1e-12 * np.eye(2), atb)


def stansfield(lobs, ref=None, systematic_deg=None,
               bias_bound_deg=None) -> Fix:
    """The classic bearings-only estimator. One pass, closed form.

    Stansfield's 1947 formulation: weight each bearing by the reciprocal of
    its linear miss distance at the target range, which turns an angular
    uncertainty into a positional one. Fast, well understood, and **biased
    towards the receivers** — it minimises perpendicular distances, and the
    ranges used for weighting come from an estimate that is itself pulled in.
    The bias grows with σ and with the number of bearings, which is a
    disconcerting property: taking more data makes it worse, not better.

    Kept because it is the reference everybody compares against, and because
    on good geometry with small σ the difference from the ML form is
    negligible and it is one matrix solve. For anything else use `ml_fix`.
    """
    ref, p, _u, nrm = _setup(lobs, ref)
    x = _linear_intersection(p, nrm)
    d = np.maximum(np.linalg.norm(x[None, :] - p, axis=1), 1.0)
    sig = np.radians(angular_sigma_deg(lobs, d))
    w = 1.0 / (d * sig) ** 2
    x = _linear_intersection(p, nrm, w)
    return _finish(lobs, ref, p, nrm, x, "Stansfield (one pass, biased "
                                         "towards the receivers)",
                   *resolve_systematic(systematic_deg, bias_bound_deg))


def _solve(lobs, ref, p, nrm, sys_sigma: float, max_iter: int) -> np.ndarray:
    """Gauss-Newton in angle space. The position only — `_finish` does the rest.

    `sys_sigma` of zero gives the uncorrelated model, which is what outlier
    detection wants; the real figure gives the correlated one, which is what
    the answer wants. Both go through here so the two passes cannot drift
    apart.
    """
    x = _linear_intersection(p, nrm)
    for _ in range(int(max_iter)):
        d = np.maximum(np.linalg.norm(x[None, :] - p, axis=1), 1.0)
        # geodesic bearing from each station TO the current estimate — see
        # `_predicted` for why this is not atan2 of the ENU offsets
        pred = _predicted(lobs, ref, x)
        resid = np.radians(geo.angle_difference(
            [l.bearing_deg for l in lobs], pred))
        # THE WEIGHTS ARE REBUILT EVERY ITERATION, because the platform's
        # position uncertainty becomes an ANGULAR one only at a range, and
        # the range is what is being solved for.
        r_inv = np.linalg.pinv(measurement_covariance(lobs, sys_sigma, d))
        # ∂θ/∂p is the left normal over the range: one metre sideways at
        # range d turns the bearing by 1/d radians
        jac = nrm / d[:, None]
        h = jac.T @ r_inv @ jac
        g = jac.T @ r_inv @ resid
        try:
            step = np.linalg.solve(h + 1e-12 * np.eye(2), g)
        except np.linalg.LinAlgError:
            break
        # PLUS, not minus. `resid` is measured − predicted and the Jacobian
        # maps a movement to the CHANGE IN THE PREDICTION, so the step that
        # reduces the residual adds. Getting this backwards does not fail to
        # converge — it diverges geometrically, and the fix ends up thousands
        # of kilometres away with an ellipse to match, which at least is
        # obvious. A sign error that merely halved the step would have been
        # far harder to see.
        x = x + step
        if float(np.linalg.norm(step)) < 1e-4:
            break
    return x


def ml_fix(lobs, ref=None, systematic_deg=None, bias_bound_deg=None,
           max_iter: int = 50, reject_outliers: bool = True) -> Fix:
    """Iterated maximum-likelihood fix from lines of bearing. The default.

    Gauss-Newton in ANGLE space rather than in miss-distance space, which is
    where the measurement noise actually lives: the residual for bearing i is
    the angle between where it points and where the current estimate is, and
    its sensitivity to a metre of movement is 1/range. That makes the
    normal equations naturally range-weighted without a separate weighting
    step, and it makes the correlated-bias covariance drop straight in.

    Outlier rejection is on by default and is a real decision, not a
    convenience: one bearing that belongs to a different emitter — or that
    caught a multipath reflection — drags a least-squares fix an arbitrary
    distance, and it does so while the ellipse stays small. Rejection is
    reported: what was dropped, and why.

    **REJECTION RUNS WITH THE SHARED-BIAS TERM SWITCHED OFF, and it has to.**
    R = diag(σ²) + σ_sys²·11ᵀ is nearly singular in the all-ones direction —
    at σ_sys = 1.73° over 1° bearings the common mode is down-weighted
    twenty-five-fold — which is correct for the covariance and catastrophic
    for outlier detection, because it hands the fit an almost free global
    rotation. One bearing 60° out is then absorbed by swinging the whole
    solution, the residual is spread across every bearing instead of landing
    on the culprit, and the rejector removes the five INNOCENT bearings while
    keeping the wild one. Measured: five of eight dropped and the answer
    24 km from a truth the unrejected fit reached to 14 km.
    An outlier is by definition a bearing that disagrees with its FELLOWS,
    and a bias shared by all of them cannot explain one of them being wrong,
    so the differential model is the right one to detect it with. The
    surviving set is then re-fitted with the full correlated model, which is
    the right one for the answer and its covariance.
    """
    ref, p_all, _u, nrm_all = _setup(lobs, ref)
    sys_sigma, sys_bound, sys_note = resolve_systematic(systematic_deg,
                                                        bias_bound_deg)
    keep = list(range(len(lobs)))
    dropped: list = []
    while True:
        idx = np.array(keep)
        p, nrm = p_all[idx], nrm_all[idx]
        sub = [lobs[i] for i in keep]
        # detection pass: uncorrelated. See the docstring.
        x = _solve(sub, ref, p, nrm, 0.0, max_iter)
        if not reject_outliers or len(keep) <= 3:
            break
        pred = _predicted(sub, ref, x)
        resid = geo.angle_difference([l.bearing_deg for l in sub], pred)
        sig = angular_sigma_deg(
            sub, np.maximum(np.linalg.norm(x[None, :] - p, axis=1), 1.0))
        z = np.abs(resid) / sig
        worst = int(np.argmax(z))
        # Bonferroni: the largest of N standard normals is expected to reach
        # about 3σ for N of ten or so, and dropping every bearing that
        # exceeds a flat 3σ would eventually drop them all.
        bar = max(3.0, math.sqrt(2.0 * math.log(max(2, len(keep)))) + 2.0)
        if z[worst] <= bar:
            break
        dropped.append({"index": keep[worst],
                        "residual_deg": round(float(resid[worst]), 3),
                        "sigmas": round(float(z[worst]), 2),
                        "source": sub[worst].source,
                        "bearing_deg": sub[worst].bearing_deg})
        keep.pop(worst)
    # the answer, with the correlated model back on
    idx = np.array(keep)
    x = _solve([lobs[i] for i in keep], ref, p_all[idx], nrm_all[idx],
               sys_sigma, max_iter)
    fix = _finish([lobs[i] for i in keep], ref, p_all[idx], nrm_all[idx], x,
                  "maximum likelihood (iterated, angle-space)",
                  sys_sigma, sys_bound, sys_note)
    fix.dropped = dropped
    if dropped:
        fix.caveats.append(
            f"{len(dropped)} of {len(lobs)} bearings were rejected as "
            f"outliers ("
            + "; ".join(f"{d['source'] or 'unnamed'} at {d['bearing_deg']:.1f}°"
                        f", {d['sigmas']:.1f}σ off" for d in dropped)
            + "). A bearing that far out is a different emitter, a "
              "reflection, or a bad ephemeris — worth knowing which before "
              "accepting the fix that remains.")
    return fix


def _finish(lobs, ref, p, nrm, x, method, systematic_deg,
            systematic_bound_deg: float = 0.0,
            systematic_note: str = "") -> Fix:
    """Covariance, geometry, consistency — everything after the position."""
    n = len(lobs)
    delta = x[None, :] - p
    d = np.maximum(np.linalg.norm(delta, axis=1), 1.0)
    pred = _predicted(lobs, ref, x)
    resid_deg = geo.angle_difference([l.bearing_deg for l in lobs], pred)
    jac = nrm / d[:, None]

    # THE COVARIANCE IS DECOMPOSED, NOT INVERTED IN ONE PIECE.
    #
    # The obvious route — build R = diag(σ²) + σ_sys²·11ᵀ and take
    # pinv(JᵀR⁻¹J) — is correct in theory and useless in practice, because
    # 11ᵀ is rank one and singular. Regularising it makes the inverse enormous
    # in the directions orthogonal to 1, the pseudo-inverse then truncates on
    # rcond, and the "systematic floor" that comes out is not a floor at all:
    # measured with sixteen bearings it was 1,250 m while the total it was
    # supposed to bound came out at 690 m. A floor larger than the thing it
    # floors is a model that has stopped meaning anything.
    #
    # So the two parts are computed separately and added, which is both
    # numerically stable and easier to reason about:
    #
    #   random      pinv(Jᵀ D⁻¹ J), the usual weighted least-squares result,
    #               and the part that shrinks as more bearings arrive
    #   systematic  σ_sys² · v vᵀ per bias group, where v is how far the fix
    #               MOVES per radian of bias common to that group. It is
    #               RANK ONE, and that is the physics: a uniform rotation of
    #               every bearing displaces the answer along one direction,
    #               not in a circle.
    #
    # Adding them guarantees total ≥ systematic. It also gets the multi-group
    # case right: two independently calibrated receivers contribute
    # σ²(v_A v_Aᵀ + v_B v_Bᵀ), which is smaller than the σ² v vᵀ of one shared
    # calibration — their errors are not the same error, so they partly
    # cancel, which is exactly why a second receiver helps. The part they DO
    # still share — datum, magnetic model, which way "zero" points — is its
    # own group, `COMMON_GROUP`, spanning every observation. See
    # `COMMON_BIAS_FRACTION` for why that is not zero.
    #
    # THE SIGNED SENSITIVITY IS KEPT, not just its outer product. v and −v
    # give the same v vᵀ, so a covariance cannot say whether two fixes are
    # displaced the same way by a shared bias or opposite ways. Fusing them
    # needs to know, and reconstructing it from the covariance afterwards is
    # not possible even in principle.
    sig_deg = angular_sigma_deg(lobs, d)
    d_inv = np.diag(1.0 / np.radians(sig_deg) ** 2)
    fisher = jac.T @ d_inv @ jac
    cov_rand = np.linalg.pinv(fisher)

    sys_cov, axes, b_axes = None, {}, {}
    if systematic_deg and systematic_deg > 0:
        f_c = min(max(float(COMMON_BIAS_FRACTION), 0.0), 1.0)
        gain = cov_rand @ jac.T @ d_inv        # fix movement per radian
        sel = [(str(k), _selector(n, ids))
               for k, ids in _bias_groups(lobs).items()]
        sel.append((COMMON_GROUP, np.ones(n)))
        sys_cov = np.zeros((2, 2))
        # THE TWO SPLITS ARE DIFFERENT, and getting that wrong costs 41 %.
        # Variances add, so a σ splits as √f and √(1−f) between the common
        # and the particular part. BOUNDS add linearly — the worst case is
        # both parts at their limit at once — so a bound splits as f and
        # (1−f). Using √ for both made a single source's two parallel bound
        # axes sum to 1.414 × the bound it came from.
        for scale, store, quad in (
                (math.radians(float(systematic_deg)), axes, True),
                (math.radians(float(systematic_bound_deg or 0.0)), b_axes,
                 False)):
            if scale <= 0:
                continue
            for k, e_g in sel:
                share = f_c if k == COMMON_GROUP else 1.0 - f_c
                part = math.sqrt(share) if quad else share
                u = (scale * part) * (gain @ e_g)
                if not np.any(u):
                    continue
                store[k] = [float(u[0]), float(u[1])]
                if quad:
                    sys_cov = sys_cov + np.outer(u, u)
    cov = cov_rand if sys_cov is None else cov_rand + sys_cov

    # GDOP: how much the GEOMETRY alone multiplies angular error into
    # position error, independent of how good the bearings are.
    #
    # It has to be made DIMENSIONLESS to mean anything. A bearings-only fit
    # gives a position covariance in metres² from measurements in radians²,
    # so sqrt(trace) is metres per radian — a number that grows with range
    # even on perfect geometry, which is not a dilution of precision, it is
    # just being far away. Dividing by the mean range removes it. One good
    # crossing is then ≈1, and near-parallel bearings run to the hundreds.
    try:
        cov_unit = np.linalg.pinv(jac.T @ jac)
        gdop = float(math.sqrt(max(np.trace(cov_unit), 0.0))
                     / max(float(np.mean(d)), 1e-9))
    except (np.linalg.LinAlgError, ValueError):
        gdop = float("nan")

    dof = max(1, n - 2)
    chi2 = float(np.sum((resid_deg / sig_deg) ** 2))
    p_val = chi2_sf(chi2, dof) if n > 2 else float("nan")

    lat, lon, _alt = geo.enu_to_geodetic(float(x[0]), float(x[1]), 0.0, ref)
    fix = Fix(lat=float(np.atleast_1d(lat)[0]),
              lon=float(np.atleast_1d(lon)[0]),
              cov=cov, n_obs=n, method=method, source=_source_key(lobs),
              epoch=float(np.mean([l.epoch for l in lobs])),
              gdop=gdop,
              residual_rms_deg=float(np.sqrt(np.mean(resid_deg ** 2))),
              consistency_p=p_val, systematic_cov=sys_cov,
              systematic_axes=axes, bound_axes=b_axes,
              evidence={"reference": [round(ref[0], 8), round(ref[1], 8)],
                        "residuals_deg": [round(float(v), 3)
                                          for v in resid_deg],
                        "ranges_m": [round(float(v), 1) for v in d],
                        "chi2": round(chi2, 3), "dof": dof,
                        "systematic_deg": float(systematic_deg),
                        "systematic_bound_deg": float(systematic_bound_deg
                                                      or 0.0),
                        "systematic_provenance": systematic_note,
                        "sigma_used_deg": [round(float(v), 4)
                                           for v in sig_deg],
                        "position_sigma_m": [
                            round(float(getattr(l, "position_sigma_m", 0.0)
                                        or 0.0), 1) for l in lobs],
                        "bias_groups": {k: len(v) for k, v in
                                        _bias_groups(lobs).items()}})
    _annotate(fix)
    return fix


def _annotate(fix: Fix) -> None:
    """The caveats that turn a number into something an analyst can act on."""
    if fix.n_obs == 2:
        fix.caveats.append(
            "TWO BEARINGS ALWAYS CROSS. With two there is no redundancy, no "
            "consistency check is possible, and the ellipse describes the "
            "stated uncertainties rather than anything that was verified. A "
            "third bearing is worth more than tightening either of these.")
    if fix.gdop == fix.gdop and fix.gdop > 10:
        fix.caveats.append(
            f"GDOP is {fix.gdop:.0f} — the bearings are close to parallel, so "
            f"the geometry multiplies their angular error into a long thin "
            f"ellipse. Moving the receiver to open the crossing angle will "
            f"do more than any improvement to the bearings themselves.")
    if fix.consistency_p == fix.consistency_p and \
            fix.consistency_p < CONSISTENCY_ALPHA:
        fix.caveats.append(
            f"THESE OBSERVATIONS ARE NOT CONSISTENT WITH ONE POSITION "
            f"(χ² = {fix.evidence.get('chi2')}, {fix.evidence.get('dof')} "
            f"dof, p = {fix.consistency_p:.2g}). Either they are not all the "
            f"same emitter, or a bearing is worse than its stated σ, or the "
            f"ephemeris is wrong for one of them. The position below is the "
            f"best compromise between observations that disagree, which is "
            f"not the same as a fix.")
    if fix.bias_limited:
        # the same measure `bias_limited` uses: how much of the REPORTED
        # region survives when the shared terms are taken away
        a, _b, _ang = fix.ellipse(REPORT_P)
        bare = Fix(lat=fix.lat, lon=fix.lon,
                   cov=(fix.statistical_cov() if fix.bound_axes else
                        np.asarray(fix.cov, dtype=float)
                        - (np.zeros((2, 2)) if fix.systematic_cov is None
                           else np.asarray(fix.systematic_cov, dtype=float))))
        share = 100.0 * (1.0 - bare.ellipse(REPORT_P)[0] / max(a, 1e-9))
        n_src = len(fix.evidence.get("bias_groups", {})) or 1
        remedy = ("What helps is a second, independently calibrated "
                  "receiver, or fixing the calibration."
                  if n_src < 2 else
                  f"It already draws on {n_src} sources, so the remedy is "
                  f"not another one of the same kind — it is calibrating "
                  f"them against a known emitter, which replaces an assumed "
                  f"bound with a measured figure and is the only thing that "
                  f"lowers this floor.")
        fix.caveats.append(
            f"THIS FIX IS BIAS-LIMITED: {share:.0f} % of the reported region "
            f"comes from error shared by every observation, which does NOT "
            f"average down. More bearings from the same system will shrink "
            f"the ellipse on paper and not move the truth. " + remedy)


# ---------------------------------------------------------------------------
# fusing fixes
# ---------------------------------------------------------------------------

def fuse_fixes(fixes, systematic_shared: bool = True,
               gate: bool = True) -> Fix:
    """Combine several completed fixes into one. Information-filter fusion.

    The inverse-covariance weighted mean: each fix contributes in proportion
    to how well it is known, and the fused information is the sum of the
    parts. That is the right answer when the fixes are INDEPENDENT.

    **THEY USUALLY ARE NOT.** Fixes of the same emitter from the same DF
    system share its calibration, and that shared part must not be summed
    away. With `systematic_shared` the common component is removed before
    fusion and added back afterwards, so it passes through unchanged — which
    is the correct behaviour and the one that stops ten fixes from a
    three-degree-biased system reporting a hundred-metre ellipse in the wrong
    place.

    Shared **per source**, though, not across the whole set. Fixes carry the
    installation that produced them in `Fix.source`, and two receivers that
    were calibrated separately have independent alignment errors that DO
    partly cancel. Holding one floor out of everything punished exactly the
    thing worth rewarding — a second, independent receiver — and told the
    analyst to go and get one they were already using.

    `gate` runs the pairwise consistency test first. Two fixes 40 km apart
    with 500 m ellipses are not two observations of one emitter, and averaging
    them produces a confident point 20 km from either.
    """
    fixes = [f for f in fixes if f is not None]
    if not fixes:
        raise ValueError("nothing to fuse")
    if len(fixes) == 1:
        return fixes[0]
    # THE TANGENT POINT IS WEIGHTED BY INFORMATION, not by count. A plain
    # mean lets one hopeless fix — the kind a two-bearing near-parallel
    # crossing produces, whose own ellipse is thousands of kilometres —
    # drag the plane away from every fix that actually knows something, and
    # then the good fixes are the ones evaluated far from the tangent point.
    w = np.array([1.0 / max(float(np.trace(np.asarray(f.cov))), 1e-6)
                  for f in fixes])
    w = w / w.sum()
    ref = (float(np.dot(w, [f.lat for f in fixes])),
           float(np.dot(w, [f.lon for f in fixes])), 0.0)
    _check_span([(f.lat, f.lon) for f in fixes], ref, "fixes")

    xs, covs = [], []
    for f in fixes:
        e, n, _u = geo.geodetic_to_enu(f.lat, f.lon, 0.0, ref)
        xs.append([float(e), float(n)])
        covs.append(np.asarray(f.cov, dtype=float))
    xs = np.asarray(xs)

    rejected = []
    if gate:
        keep = _consistency_gate(xs, covs)
        rejected = [i for i in range(len(fixes)) if i not in keep]
        if len(keep) >= 1:
            xs = xs[keep]
            covs = [covs[i] for i in keep]
            fixes_kept = [fixes[i] for i in keep]
        else:
            fixes_kept = fixes
    else:
        fixes_kept = fixes

    # THE WEIGHTS AND THE COVARIANCE ARE COMPUTED SEPARATELY, and that is
    # the whole design. An information filter normally gets both from one
    # inverse, which forces a single answer to two different questions:
    # "how much should I trust this fix?" and "how wrong is the result?"
    #
    #   * Trust has to use each fix's FULL uncertainty, bias included. A
    #     precise fix from a badly aligned receiver deserves less weight than
    #     its random scatter suggests, and weighting on the random part alone
    #     made adding one such bearing to a good prior make the answer WORSE —
    #     a 5.1 km region became 6.8 km on strictly more evidence.
    #   * The result's error has to be built from the SIGNED sensitivities,
    #     because a shared bias reinforces or cancels depending on which way
    #     it pushed each fix, and v vᵀ cannot tell those apart.
    #
    # So: weights from the full covariances, then
    #     x̂ − truth = Σ Wᵢ εᵢ + Σ_g (Σ Wᵢ u_ig) b_g
    # evaluated term by term. Exact for ANY weights that sum to the identity,
    # which frees the weighting to be chosen for robustness.
    per_fix_sys = [_systematic_of(f) for f in fixes_kept]
    axes_of = [_axes_of(f, sysc) for f, sysc in zip(fixes_kept, per_fix_sys)]
    bounds_of = [dict(getattr(f, "bound_axes", None) or {})
                 for f in fixes_kept]

    info = np.zeros((2, 2))
    p_invs = []
    for c in covs:
        p_inv = np.linalg.pinv(c)
        p_invs.append(p_inv)
        info = info + p_inv
    base = np.linalg.pinv(info)
    ws = [base @ p_inv for p_inv in p_invs]       # Σ wᵢ = I by construction
    x = sum(w @ xi for w, xi in zip(ws, xs))

    # random part: the fixes' own scatter, which is independent between them
    cov = np.zeros((2, 2))
    for w, c, sysc in zip(ws, covs, per_fix_sys):
        c_rand = c if sysc is None else c - sysc
        ev, _v = np.linalg.eigh(0.5 * (c_rand + c_rand.T))
        if np.min(ev) < 0:
            c_rand = c                 # all systematic; keep it whole
        cov = cov + w @ c_rand @ w.T

    def _combine(per_fix):
        """Each group's signed sensitivity, propagated through the weights."""
        out, acc = {}, np.zeros((2, 2))
        for k in sorted({key for a in per_fix for key in a}):
            u = np.zeros(2)
            for w, a in zip(ws, per_fix):
                if k in a:
                    u = u + w @ np.asarray(a[k], dtype=float)
            if np.any(u):
                out[k] = [float(u[0]), float(u[1])]
                acc = acc + np.outer(u, u)
        return out, 0.5 * (acc + acc.T)

    shared, fused_axes, fused_bounds = None, {}, {}
    if systematic_shared:
        fused_axes, acc = _combine(axes_of)
        fused_bounds, _acc_b = _combine(bounds_of)
        if fused_axes:
            shared = acc
            cov = cov + shared
    cov = 0.5 * (cov + cov.T)

    lat, lon, _a = geo.enu_to_geodetic(float(x[0]), float(x[1]), 0.0, ref)
    n_obs = int(sum(max(1, f.n_obs) for f in fixes_kept))
    out = Fix(lat=float(np.atleast_1d(lat)[0]),
              lon=float(np.atleast_1d(lon)[0]), cov=cov,
              n_obs=n_obs,
              method=f"information-filter fusion of {len(fixes_kept)} fixes",
              epoch=float(np.mean([f.epoch for f in fixes_kept])),
              systematic_cov=shared, systematic_axes=fused_axes,
              bound_axes=fused_bounds,
              # the fused fix inherits EVERY source that fed it, so fusing it
              # again next month still knows which biases it already carries
              source="+".join(sorted({str(getattr(f, "source", "") or "")
                                      for f in fixes_kept} - {""})),
              evidence={"fixes_in": len(fixes), "fixes_used": len(fixes_kept),
                        "shared_systematic": shared is not None,
                        "bias_groups": sorted(k for k in fused_axes
                                              if k != COMMON_GROUP),
                        "independent_sources": len([k for k in fused_axes
                                                    if k != COMMON_GROUP]),
                        "separations_m": _separations(fixes_kept)})
    if rejected:
        out.dropped = [{"index": i, "lat": fixes[i].lat, "lon": fixes[i].lon,
                        "method": fixes[i].method} for i in rejected]
        out.caveats.append(
            f"{len(rejected)} of {len(fixes)} fixes were excluded: they are "
            f"too far apart, relative to their own stated uncertainties, to "
            f"be the same emitter. Fusing them anyway would produce a "
            f"confident position between two things.")
    if shared is not None:
        n_ind = len([k for k in fused_axes if k != COMMON_GROUP])
        if n_ind > 1:
            out.caveats.append(
                f"these fixes come from {n_ind} separately calibrated "
                f"sources. The part of each system's alignment error that is "
                f"its own partly cancels — which is why this is better than "
                f"any one of them alone — but the part they share, from a "
                f"common datum, magnetic model or convention, does not. Both "
                f"are in the ellipse; only a calibration against a known "
                f"emitter removes the second.")
        else:
            out.caveats.append(
                "a systematic component common to these fixes was held out "
                "of the fusion and added back afterwards — it does not "
                "average down, and summing it away is what makes repeated "
                "observations of one biased system look far better than they "
                "are.")
    _annotate(out)
    return out


def _systematic_of(fix):
    """A fix's systematic covariance, rebuilt from its axes when it has them.

    The axes are the authority: `systematic_cov` is derived from them and a
    fix that came back from disk may carry one and not the other.
    """
    axes = getattr(fix, "systematic_axes", None) or {}
    if axes:
        out = np.zeros((2, 2))
        for u in axes.values():
            u = np.asarray(u, dtype=float).reshape(2)
            out = out + np.outer(u, u)
        return out
    if fix.systematic_cov is None:
        return None
    return np.asarray(fix.systematic_cov, dtype=float).reshape(2, 2)


def _axes_of(fix, sysc) -> dict:
    """Signed sensitivities for a fix, inventing them if it has none.

    A fix from an older store, or one built by hand, has a systematic
    covariance and no axes. The rank-one square root recovers the magnitude
    and direction but NOT the sign, so the sign is chosen to make the
    contributions reinforce rather than cancel — the conservative reading,
    since the alternative silently claims a cancellation nobody demonstrated.
    The group is the fix's own source, which is right when it has one and
    conservative when it does not, because unnamed sources group together.
    """
    axes = getattr(fix, "systematic_axes", None) or {}
    if axes:
        return {str(k): [float(v[0]), float(v[1])] for k, v in axes.items()}
    if sysc is None:
        return {}
    w, v = np.linalg.eigh(np.asarray(sysc, dtype=float))
    i = int(np.argmax(w))
    if w[i] <= 0:
        return {}
    u = v[:, i] * math.sqrt(float(w[i]))
    if u[0] < 0 or (u[0] == 0 and u[1] < 0):
        u = -u                              # a fixed sign, so they reinforce
    return {str(getattr(fix, "source", "") or ""): [float(u[0]), float(u[1])]}


def _separations(fixes) -> list:
    out = []
    for i in range(len(fixes)):
        for j in range(i + 1, len(fixes)):
            out.append(round(geo.distance_between(
                fixes[i].lat, fixes[i].lon, fixes[j].lat, fixes[j].lon), 1))
    return out


def _consistency_gate(xs, covs, alpha: float = CONSISTENCY_ALPHA) -> list:
    """Which fixes can plausibly be the same emitter.

    Pairwise Mahalanobis distance against the sum of the two covariances,
    two degrees of freedom. Keeps the largest mutually-consistent group; ties
    go to the group whose members are individually better known, because a
    cluster of three vague fixes should not outvote two precise ones.
    """
    n = len(xs)
    ok = np.ones((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            d = xs[i] - xs[j]
            s = np.asarray(covs[i]) + np.asarray(covs[j])
            try:
                m = float(d @ np.linalg.pinv(s) @ d)
            except np.linalg.LinAlgError:
                m = 0.0
            ok[i, j] = ok[j, i] = chi2_sf(m, 2) >= alpha
    best: list = []
    best_info = -1.0
    for i in range(n):
        group = [j for j in range(n) if ok[i, j] or j == i]
        group = [j for j in group
                 if all(ok[j, k] for k in group if k != j)]
        info = sum(1.0 / max(float(np.trace(covs[j])), 1e-9) for j in group)
        if len(group) > len(best) or (len(group) == len(best)
                                      and info > best_info):
            best, best_info = group, info
    return sorted(best)


# ---------------------------------------------------------------------------
# the combined path
# ---------------------------------------------------------------------------

def _with_bearing(lob, deg: float):
    out = geo.LOB(**{**lob.__dict__})
    out.bearing_deg = float(deg) % 360.0
    out.evidence = dict(lob.evidence or {})
    return out


def resolve_ambiguity(lobs, systematic_deg=None, bias_bound_deg=None,
                      max_seeds: int = 80) -> tuple:
    """`(resolved, fix, report)` — pick the right side of each array cone.

    **THIS IS WHY A MOVING PLATFORM CAN GEOLOCATE WITH ONE LINEAR ARRAY.** A
    single measurement gives two candidate bearings mirrored about the array
    axis, and nothing in that measurement can separate them. But the mirror
    solutions are mirrored about an axis that MOVES as the platform turns, so
    a sequence of them is only mutually consistent for the true side. Take the
    wrong branch of any block and it stops pointing where the others point.

    The search is a small RANSAC: seed a fix from two blocks and one of their
    four sign combinations, use it to choose every other block's branch by
    which one points nearer, refit, and score by how well the whole selection
    agrees. The best-scoring selection wins. Seeds are ordered by baseline
    length, because a widely separated pair gives the sharpest seed fix and
    usually settles it on the first try.

    It CAN fail, and it says so: a platform that flew a straight line towards
    a target barely changes the geometry, both branches stay consistent, and
    the honest answer is that the collection could not resolve it. The report
    carries the margin between the best and second-best selections so that a
    marginal call is visible rather than silently taken.
    """
    lobs = list(lobs)
    cand = [[l.bearing_deg] if l.ambiguous_with is None
            else [l.bearing_deg, float(l.ambiguous_with)] for l in lobs]
    if not any(len(c) > 1 for c in cand):
        return lobs, None, "no ambiguous bearings to resolve"
    n = len(lobs)
    if n < 3:
        return lobs, None, (
            "resolving a cone ambiguity needs at least three bearings from "
            "different places — with two, both branches fit equally well by "
            "construction")

    pairs = sorted(((i, j) for i in range(n) for j in range(i + 1, n)),
                   key=lambda ij: -geo.distance_between(
                       lobs[ij[0]].lat, lobs[ij[0]].lon,
                       lobs[ij[1]].lat, lobs[ij[1]].lon))[:max_seeds]
    scored = []
    seen = set()
    for i, j in pairs:
        for bi in cand[i]:
            for bj in cand[j]:
                try:
                    seed = ml_fix([_with_bearing(lobs[i], bi),
                                   _with_bearing(lobs[j], bj)],
                                  reject_outliers=False)
                except (ValueError, np.linalg.LinAlgError, SpanError):
                    continue
                sel = []
                for k, l in enumerate(lobs):
                    pred = geo.bearing_between(l.lat, l.lon, seed.lat,
                                               seed.lon)
                    sel.append(min(cand[k],
                                   key=lambda b: abs(geo.angle_difference(
                                       b, pred))))
                key = tuple(round(b, 6) for b in sel)
                if key in seen:
                    continue
                seen.add(key)
                trial = [_with_bearing(l, b) for l, b in zip(lobs, sel)]
                try:
                    # SCORED WITHOUT THE SHARED TERM, for the same reason
                    # `ml_fix` detects outliers without it: a bias common to
                    # every bearing cannot tell one branch from another, and
                    # a near-singular R gives the fit a nearly free global
                    # rotation that lands the mirror set somewhere it does
                    # not belong. Measured, it made the straight-run negative
                    # control — where BOTH answers genuinely fit and the
                    # honest report is "here or here, 38 km apart" — come
                    # back as confidently RESOLVED.
                    f = ml_fix(trial, systematic_deg=0.0,
                               reject_outliers=False)
                except (ValueError, np.linalg.LinAlgError, SpanError):
                    continue
                sig = np.array([max(l.sigma_deg, 1e-6) for l in trial])
                r = np.array(f.evidence.get("residuals_deg", []))
                if r.size != sig.size:
                    continue
                scored.append((float(np.sum((r / sig) ** 2)), sel, f))
    if not scored:
        return lobs, None, "no self-consistent selection of branches was found"
    scored.sort(key=lambda s: s[0])
    best, sel, _seed_fix = scored[0]
    resolved = [_with_bearing(l, b) for l, b in zip(lobs, sel)]
    # the branch is chosen; the ANSWER is refitted with the real error model
    try:
        fix = ml_fix(resolved, systematic_deg=systematic_deg,
                     bias_bound_deg=bias_bound_deg, reject_outliers=False)
    except (ValueError, np.linalg.LinAlgError, SpanError):
        fix = _seed_fix
    for r, l in zip(resolved, lobs):
        other = [c for c in ([l.bearing_deg, l.ambiguous_with]
                             if l.ambiguous_with is not None else [])
                 if abs(geo.angle_difference(c, r.bearing_deg)) > 1e-6]
        r.ambiguous_with = float(other[0]) if other else None
        r.evidence["branch_resolved_by"] = "mutual consistency across blocks"
    flipped = sum(1 for r, l in zip(resolved, lobs)
                  if abs(geo.angle_difference(r.bearing_deg,
                                              l.bearing_deg)) > 1e-6)

    # **AN AMBIGUITY IS RESOLVED ONLY WHEN THE ALTERNATIVE IS REJECTED**, not
    # when it merely scores worse. On a straight collection run the ENTIRE
    # mirror set is self-consistent too — reflecting every bearing about a
    # constant array axis reflects the whole picture about the flight line,
    # so the emitter and its image across the track fit equally well. A χ²
    # margin cannot see that: measured on such a run the two selections
    # scored 0.0024 and 0.2655, a gap that looks decisive and means nothing,
    # while the second answer was 38 km away. So the runner-up gets the same
    # consistency test the winner does, and if it survives, BOTH positions
    # are reported, because "it is here or here, 38 km apart" is the true
    # finding and is actionable — one more leg across the bearing settles it.
    rival = None
    for score, s2, f2 in scored[1:]:
        if any(abs(geo.angle_difference(a, b)) > 1e-6 for a, b in zip(s2, sel)):
            rival = (score, f2)
            break
    if rival is not None:
        r_score, r_fix = rival
        sep = geo.distance_between(fix.lat, fix.lon, r_fix.lat, r_fix.lon)
        # An "ambiguity" smaller than the ellipse is not an ambiguity. When
        # the mirror lands inside the fix's own 95 % contour the two answers
        # are the same answer, and raising an alarm about it trains people to
        # ignore the alarm for the case where the mirror is 40 km away.
        if sep <= fix.ellipse(0.95)[0]:
            return resolved, fix, (
                f"the mirror branch puts the emitter {sep:,.0f} m away, "
                f"inside this fix's own 95 % ellipse — the two solutions are "
                f"not meaningfully different here and the choice does not "
                f"change the answer.")
        if r_fix.consistent:
            fix.evidence["mirror_fix"] = {
                "lat": round(r_fix.lat, 6), "lon": round(r_fix.lon, 6),
                "chi2": round(r_score, 4), "separation_m": round(sep, 1)}
            note = (f"AMBIGUITY NOT RESOLVED — TWO ANSWERS FIT. The emitter "
                    f"is at {fix.lat:.5f}, {fix.lon:.5f} or at "
                    f"{r_fix.lat:.5f}, {r_fix.lon:.5f}, {sep / 1000:,.1f} km "
                    f"apart, and this collection cannot say which: the mirror "
                    f"set about the array axis is just as self-consistent. "
                    f"That is what a straight run gives you. One leg across "
                    f"the bearing — a dog-leg, a turn, anything that changes "
                    f"the array's orientation relative to the target — "
                    f"separates them immediately.")
            fix.caveats.append(note)
            return resolved, fix, note
        note = (f"cone ambiguity RESOLVED: {flipped} of {n} bearings were "
                f"flipped to their mirror branch. The alternative selection "
                f"puts the emitter {sep / 1000:,.1f} km away and is rejected "
                f"by its own consistency test (χ² {r_score:,.0f} against "
                f"{best:.2f}). The platform's change of geometry is what "
                f"separates them; a straight run would not have.")
        return resolved, fix, note
    return resolved, fix, (f"one branch selection was found; {flipped} of {n} "
                           f"bearings were flipped")


def refine(observations, prior: Fix | None = None,
           systematic_deg=None, bias_bound_deg=None, gate: bool = True,
           **kw) -> Fix:
    """Everything about one emitter, in one answer.

    `observations` may hold lines of bearing and completed fixes together —
    which is the shape the question usually arrives in. Yesterday's fix plus
    two bearings taken this morning is three pieces of evidence about one
    thing, and the point of this module is that they combine rather than
    competing.

    `systematic_deg` defaults to None, not to zero, and None means the Type B
    assignment described at `DEFAULT_BIAS_BOUND_DEG` — the same default
    `ml_fix` uses. A zero default here would have quietly reintroduced the
    perfect-equipment claim on the one path most likely to be fed a long
    history of observations from a single installation, which is precisely
    where an unmodelled shared bias does the most damage.
    """
    lobs = [o for o in observations if isinstance(o, geo.LOB)]
    fixes = [o for o in observations if isinstance(o, Fix)]
    if prior is not None:
        fixes.append(prior)
    if lobs and len(lobs) >= 2:
        fixes.append(ml_fix(lobs, systematic_deg=systematic_deg,
                            bias_bound_deg=bias_bound_deg, **kw))
    elif lobs and fixes:
        # One bearing cannot fix anything alone, but it can sharpen a fix
        # that already exists — and along one axis only, which is exactly
        # what the covariance below says.
        fixes.append(_lob_as_fix(lobs[0], fixes[0], systematic_deg,
                                 bias_bound_deg))
    elif lobs:
        raise ValueError("one line of bearing and no prior fix is not a "
                         "geolocation — it is a direction")
    if not fixes:
        raise ValueError("nothing to refine")
    out = fuse_fixes(fixes, systematic_shared=True, gate=gate)
    if lobs and prior is not None:
        out.caveats.append(
            f"{len(lobs)} new bearing(s) combined with a prior fix. The "
            f"prior contributes its own uncertainty and is not treated as "
            f"truth; if the new evidence disagrees with it, the consistency "
            f"test above is what says so.")
    return out


def _lob_as_fix(lob, near: Fix, systematic_deg=None,
                bias_bound_deg=None) -> Fix:
    """One bearing expressed as a (very) elongated fix around a prior.

    A single LOB constrains position across the bearing and not at all along
    it, which is a covariance with one small eigenvalue and one enormous one.
    Writing it that way lets the ordinary fusion machinery use it correctly
    instead of needing a special case.

    Three error terms go into the across-track width, and all three have to,
    because this is the one place a bearing's uncertainty gets converted into
    metres without passing through `_finish`:

      * the bearing's own σ, at this range;
      * the shared alignment error, likewise — carried separately in
        `systematic_cov` so `fuse_fixes` holds it out rather than averaging
        it down over a long history from one installation;
      * the receiver's own position uncertainty, which is already in metres
        and does not scale with range at all.
    """
    sys_deg, sys_bound, sys_note = resolve_systematic(systematic_deg,
                                                      bias_bound_deg)
    ref = (near.lat, near.lon, 0.0)
    e, n, _u = geo.geodetic_to_enu(lob.lat, lob.lon, 0.0, ref)
    d = max(math.hypot(float(e), float(n)), 1.0)
    pos = max(float(getattr(lob, "position_sigma_m", 0.0) or 0.0), 0.0)
    rand = math.hypot(d * math.radians(max(lob.sigma_deg, 1e-6)), pos)
    syst = d * math.radians(max(float(sys_deg), 0.0))
    across = max(math.hypot(rand, syst), 1.0)
    along = max(d * 10.0, 1e6)          # effectively unconstrained
    th = math.radians(lob.bearing_deg)
    # rotate diag(across², along²) from bearing-aligned axes into east/north
    r = np.array([[math.cos(th), math.sin(th)],
                  [-math.sin(th), math.cos(th)]])
    cov = r.T @ np.diag([across ** 2, along ** 2]) @ r
    # the across-track unit vector, in east/north: the direction a shared
    # bias actually pushes this constraint
    across_en = np.array([math.cos(th), -math.sin(th)])
    sys_cov, axes, b_axes = None, {}, {}
    if syst > 0:
        sys_cov = r.T @ np.diag([syst ** 2, 0.0]) @ r
        key = str(getattr(lob, "source", "") or "")
        axes[key] = [float(syst * across_en[0]), float(syst * across_en[1])]
    if sys_bound and sys_bound > 0:
        bm = d * math.radians(float(sys_bound))
        key = str(getattr(lob, "source", "") or "")
        b_axes[key] = [float(bm * across_en[0]), float(bm * across_en[1])]
    lat, lon = lob.point_at(d)
    return Fix(lat=lat, lon=lon, cov=cov, n_obs=1,
               source=str(getattr(lob, "source", "") or ""),
               method="single bearing, as a constraint across it only",
               epoch=lob.epoch, systematic_cov=sys_cov,
               systematic_axes=axes, bound_axes=b_axes,
               evidence={"note": "along-track extent is deliberately "
                                 "unbounded: one bearing says nothing about "
                                 "range",
                         "systematic_deg": float(sys_deg),
                         "systematic_bound_deg": float(sys_bound or 0.0),
                         "systematic_provenance": sys_note,
                         "position_sigma_m": round(pos, 1)})


def fix_dossier(fix: Fix, lobs=None) -> list:
    """Everything about a fix, as dossier measurements."""
    out = fix.to_measurements()
    if lobs:
        out.append(Measurement(
            "bearings_used", float(len(lobs)), DIMENSIONLESS, 0.9,
            "geolocation", evidence={
                "bearings": [{"lat": round(l.lat, 6), "lon": round(l.lon, 6),
                              "deg": round(l.bearing_deg, 2),
                              "sigma": round(l.sigma_deg, 2),
                              "source": l.source} for l in lobs[:32]]}))
    return out
