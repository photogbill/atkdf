"""WGS-84 geodesy, and the rotation that turns an array angle into a bearing.

Everything in the geolocation layer rests on this file, and almost every way
of getting a confident wrong answer passes through it. Three of them are worth
naming before the code starts.

**A SPHERICAL EARTH IS 0.3 % WRONG AND 0.3 % IS 150 METRES.** The haversine
formula every codebase reaches for assumes a sphere; the earth is an ellipsoid
flattened by about one part in 298. Over a 50 km DF baseline that is a
150-metre error in the range and up to 0.2° in the bearing, which is the same
order as the DF accuracy the whole exercise is trying to achieve. So the
geodesic here is Vincenty's, on the ellipsoid, and it says when it fell back.

**A LINEAR ARRAY MEASURES A CONE, NOT A BEARING.** The angle a uniform linear
array recovers is the angle to its own AXIS, so the set of directions
consistent with it is a cone about that axis. On the ground that cone
generally cuts the horizon in TWO places. Rotating it into geographic
coordinates does not fix that — it just moves the cone. `cone_bearings` below
returns both crossings, because returning one of them and a caveat is how an
analyst ends up 40° off with no way to notice.

**ATTITUDE IS NOT A CORRECTION, IT IS THE MEASUREMENT.** A bearing from a
moving platform is the array angle plus the platform's heading, and if the
heading is 5° out so is the bearing. Pitch and roll matter too, and they
matter more the further the target is from the horizon. There is no way to
recover from a wrong attitude downstream, which is why `rotate` takes all
three and why the ephemeris layer refuses to invent any of them.

MOVED FROM ATK, 2026-09-30 (ATK c52b913, `atk/core/siga/geodesy.py`), with its tests
and this docstring. ATK's `atk.core.siga.geodesy` is now an alias of this
module — the same module object, reached through `atk/core/df_engine.py`
— so the Geolocation tab and ATK's own tests run this code. Edit it here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# WGS-84
A_WGS84 = 6_378_137.0                    # semi-major axis, metres
F_WGS84 = 1.0 / 298.257_223_563          # flattening
B_WGS84 = A_WGS84 * (1.0 - F_WGS84)      # semi-minor axis
E2 = F_WGS84 * (2.0 - F_WGS84)           # first eccentricity squared
EP2 = E2 / (1.0 - E2)                    # second eccentricity squared

#: Mean radius, for the spherical fallback only. Never the primary path.
R_SPHERE = 6_371_008.8


# ---------------------------------------------------------------------------
# coordinate conversions
# ---------------------------------------------------------------------------

def geodetic_to_ecef(lat_deg, lon_deg, alt_m=0.0) -> tuple:
    """(x, y, z) in earth-centred earth-fixed metres."""
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    alt = np.asarray(alt_m, dtype=float)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    n = A_WGS84 / np.sqrt(1.0 - E2 * sin_lat ** 2)
    x = (n + alt) * cos_lat * np.cos(lon)
    y = (n + alt) * cos_lat * np.sin(lon)
    z = (n * (1.0 - E2) + alt) * sin_lat
    return x, y, z


def ecef_to_geodetic(x, y, z) -> tuple:
    """(lat_deg, lon_deg, alt_m). Bowring's method — closed form.

    Closed form rather than the obvious iteration because the iteration is
    slow to converge near the poles and this runs inside a fit.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    theta = np.arctan2(z * A_WGS84, p * B_WGS84)
    lat = np.arctan2(z + EP2 * B_WGS84 * np.sin(theta) ** 3,
                     p - E2 * A_WGS84 * np.cos(theta) ** 3)
    n = A_WGS84 / np.sqrt(1.0 - E2 * np.sin(lat) ** 2)
    # cos(lat) → 0 at the poles, so the usual p/cos(lat) − N is a division by
    # zero there. The z-based form is well conditioned exactly where the
    # other one is not, so each is used where it behaves.
    alt = np.where(np.abs(np.cos(lat)) > 1e-9,
                   p / np.maximum(np.abs(np.cos(lat)), 1e-30) - n,
                   z / np.maximum(np.abs(np.sin(lat)), 1e-30)
                   - n * (1.0 - E2))
    return np.degrees(lat), np.degrees(lon), alt


def enu_matrix(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Rotation from ECEF to the local east/north/up frame at a point."""
    lat, lon = math.radians(float(lat_deg)), math.radians(float(lon_deg))
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    return np.array([[-so, co, 0.0],
                     [-sl * co, -sl * so, cl],
                     [cl * co, cl * so, sl]])


def geodetic_to_enu(lat_deg, lon_deg, alt_m, ref: tuple) -> tuple:
    """Local east/north/up metres relative to `ref` = (lat, lon, alt).

    The frame every estimator in `fusion.py` works in. Curvature is exact
    here — the flat-earth approximation is only in treating the ENU frame as
    Cartesian, which costs about a metre at 10 km and grows as the square of
    the range.
    """
    x, y, z = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
    x0, y0, z0 = geodetic_to_ecef(ref[0], ref[1],
                                  ref[2] if len(ref) > 2 else 0.0)
    d = np.stack([np.asarray(x) - x0, np.asarray(y) - y0,
                  np.asarray(z) - z0])
    e, n, u = enu_matrix(ref[0], ref[1]) @ d
    return e, n, u


def enu_to_geodetic(e, n, u, ref: tuple) -> tuple:
    """The inverse of `geodetic_to_enu`."""
    x0, y0, z0 = geodetic_to_ecef(ref[0], ref[1],
                                  ref[2] if len(ref) > 2 else 0.0)
    d = enu_matrix(ref[0], ref[1]).T @ np.stack(
        [np.asarray(e, dtype=float), np.asarray(n, dtype=float),
         np.asarray(u, dtype=float)])
    return ecef_to_geodetic(d[0] + x0, d[1] + y0, d[2] + z0)


# ---------------------------------------------------------------------------
# geodesics on the ellipsoid
# ---------------------------------------------------------------------------

@dataclass
class Geodesic:
    distance_m: float
    bearing_deg: float          # initial bearing at point 1, degrees true
    back_bearing_deg: float     # bearing at point 2 back towards point 1
    method: str                 # "Vincenty" or the fallback, named
    iterations: int = 0


def inverse(lat1, lon1, lat2, lon2, max_iter: int = 200) -> Geodesic:
    """Distance and bearing between two points, on the ellipsoid.

    Vincenty's inverse solution. It is accurate to well under a millimetre
    and it does NOT converge for near-antipodal points — a real limitation,
    not a theoretical one — so the failure is detected and reported rather
    than returned as a number. The spherical fallback is good to about 0.3 %,
    which is fine for deciding whether two fixes are the same emitter and not
    fine for the fix itself, so the caller can see which it got.
    """
    lat1, lon1 = float(lat1), float(lon1)
    lat2, lon2 = float(lat2), float(lon2)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    u1 = math.atan((1 - F_WGS84) * math.tan(phi1))
    u2 = math.atan((1 - F_WGS84) * math.tan(phi2))
    ll = math.radians(lon2 - lon1)
    sin_u1, cos_u1 = math.sin(u1), math.cos(u1)
    sin_u2, cos_u2 = math.sin(u2), math.cos(u2)

    lam = ll
    sin_sig = cos_sig = sigma = sin_alpha = cos2_alpha = cos_2sm = 0.0
    for it in range(int(max_iter)):
        sin_lam, cos_lam = math.sin(lam), math.cos(lam)
        sin_sig = math.hypot(
            cos_u2 * sin_lam,
            cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lam)
        if sin_sig == 0.0:
            return Geodesic(0.0, 0.0, 0.0, "coincident points", it)
        cos_sig = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lam
        sigma = math.atan2(sin_sig, cos_sig)
        sin_alpha = cos_u1 * cos_u2 * sin_lam / sin_sig
        cos2_alpha = 1.0 - sin_alpha ** 2
        cos_2sm = (cos_sig - 2 * sin_u1 * sin_u2 / cos2_alpha
                   if cos2_alpha != 0.0 else 0.0)     # 0 on the equator
        c = (F_WGS84 / 16 * cos2_alpha
             * (4 + F_WGS84 * (4 - 3 * cos2_alpha)))
        lam_prev = lam
        lam = ll + (1 - c) * F_WGS84 * sin_alpha * (
            sigma + c * sin_sig
            * (cos_2sm + c * cos_sig * (-1 + 2 * cos_2sm ** 2)))
        if abs(lam - lam_prev) < 1e-12:
            break
    else:
        return _spherical_inverse(lat1, lon1, lat2, lon2, max_iter)

    u2_ = cos2_alpha * EP2
    a_ = 1 + u2_ / 16384 * (4096 + u2_ * (-768 + u2_ * (320 - 175 * u2_)))
    b_ = u2_ / 1024 * (256 + u2_ * (-128 + u2_ * (74 - 47 * u2_)))
    d_sigma = b_ * sin_sig * (
        cos_2sm + b_ / 4 * (
            cos_sig * (-1 + 2 * cos_2sm ** 2)
            - b_ / 6 * cos_2sm * (-3 + 4 * sin_sig ** 2)
            * (-3 + 4 * cos_2sm ** 2)))
    s = B_WGS84 * a_ * (sigma - d_sigma)
    sin_lam, cos_lam = math.sin(lam), math.cos(lam)
    a1 = math.atan2(cos_u2 * sin_lam,
                    cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lam)
    a2 = math.atan2(cos_u1 * sin_lam,
                    -sin_u1 * cos_u2 + cos_u1 * sin_u2 * cos_lam)
    return Geodesic(float(s), math.degrees(a1) % 360.0,
                    math.degrees(a2) % 360.0, "Vincenty", it + 1)


def _spherical_inverse(lat1, lon1, lat2, lon2, it) -> Geodesic:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    dp = p2 - p1
    a = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    d = 2 * R_SPHERE * math.asin(min(1.0, math.sqrt(a)))
    y = math.sin(dl) * math.cos(p2)
    x = (math.cos(p1) * math.sin(p2)
         - math.sin(p1) * math.cos(p2) * math.cos(dl))
    fwd = math.degrees(math.atan2(y, x)) % 360.0
    return Geodesic(d, fwd, (fwd + 180.0) % 360.0,
                    "spherical fallback — Vincenty did not converge "
                    "(near-antipodal); this range is good to about 0.3 %", it)


def direct(lat1, lon1, bearing_deg, distance_m) -> tuple:
    """`(lat2, lon2)` a given distance along a given bearing. Vincenty direct.

    The other half of a line of bearing: this is what draws it on a map, and
    what `fusion.py` uses to walk out along a LOB.
    """
    phi1 = math.radians(float(lat1))
    alpha1 = math.radians(float(bearing_deg))
    s = float(distance_m)
    sin_a1, cos_a1 = math.sin(alpha1), math.cos(alpha1)
    tan_u1 = (1 - F_WGS84) * math.tan(phi1)
    cos_u1 = 1 / math.sqrt(1 + tan_u1 ** 2)
    sin_u1 = tan_u1 * cos_u1
    sigma1 = math.atan2(tan_u1, cos_a1)
    sin_alpha = cos_u1 * sin_a1
    cos2_alpha = 1 - sin_alpha ** 2
    u2_ = cos2_alpha * EP2
    a_ = 1 + u2_ / 16384 * (4096 + u2_ * (-768 + u2_ * (320 - 175 * u2_)))
    b_ = u2_ / 1024 * (256 + u2_ * (-128 + u2_ * (74 - 47 * u2_)))

    sigma = s / (B_WGS84 * a_)
    for _ in range(200):
        cos_2sm = math.cos(2 * sigma1 + sigma)
        sin_sig, cos_sig = math.sin(sigma), math.cos(sigma)
        d_sigma = b_ * sin_sig * (
            cos_2sm + b_ / 4 * (
                cos_sig * (-1 + 2 * cos_2sm ** 2)
                - b_ / 6 * cos_2sm * (-3 + 4 * sin_sig ** 2)
                * (-3 + 4 * cos_2sm ** 2)))
        prev, sigma = sigma, s / (B_WGS84 * a_) + d_sigma
        if abs(sigma - prev) < 1e-12:
            break
    sin_sig, cos_sig = math.sin(sigma), math.cos(sigma)
    cos_2sm = math.cos(2 * sigma1 + sigma)
    tmp = sin_u1 * sin_sig - cos_u1 * cos_sig * cos_a1
    phi2 = math.atan2(sin_u1 * cos_sig + cos_u1 * sin_sig * cos_a1,
                      (1 - F_WGS84) * math.hypot(sin_alpha, tmp))
    lam = math.atan2(sin_sig * sin_a1,
                     cos_u1 * cos_sig - sin_u1 * sin_sig * cos_a1)
    c = F_WGS84 / 16 * cos2_alpha * (4 + F_WGS84 * (4 - 3 * cos2_alpha))
    ll = lam - (1 - c) * F_WGS84 * sin_alpha * (
        sigma + c * sin_sig
        * (cos_2sm + c * cos_sig * (-1 + 2 * cos_2sm ** 2)))
    lon2 = (float(lon1) + math.degrees(ll) + 540.0) % 360.0 - 180.0
    return math.degrees(phi2), lon2


def bearing_between(lat1, lon1, lat2, lon2) -> float:
    """Initial true bearing from 1 to 2, degrees. The common shorthand."""
    return inverse(lat1, lon1, lat2, lon2).bearing_deg


def distance_between(lat1, lon1, lat2, lon2) -> float:
    """Geodesic distance in metres. The common shorthand."""
    return inverse(lat1, lon1, lat2, lon2).distance_m


def angle_difference(a_deg, b_deg):
    """`a − b` wrapped into [−180, 180). Every bearing residual needs this.

    A residual computed without it is 359° when the truth is −1°, and a
    least-squares fit handed that number goes somewhere interesting.

    Exactly opposite comes back as −180 rather than +180. They are the same
    direction and nothing here distinguishes them; the half-open interval is
    stated so that a test asserting +180 is a test of the convention rather
    than of the maths.
    """
    return (np.asarray(a_deg, dtype=float)
            - np.asarray(b_deg, dtype=float) + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# attitude
# ---------------------------------------------------------------------------

def rotation_body_to_ned(heading_deg: float, pitch_deg: float = 0.0,
                         roll_deg: float = 0.0) -> np.ndarray:
    """Body → north/east/down rotation, the aerospace 3-2-1 sequence.

    Body axes are x forward through the nose, y out the right wing, z down.
    The sequence is yaw about z, then pitch about the new y, then roll about
    the new x — which is what every INS on earth reports and what "heading,
    pitch, roll" means without further qualification.

    ORDER IS NOT A DETAIL. Applying roll before yaw gives a different matrix,
    and for a level platform the two agree exactly, so the mistake is
    invisible on the bench and appears only when the aircraft banks.
    """
    y, p, r = (math.radians(float(heading_deg)), math.radians(float(pitch_deg)),
               math.radians(float(roll_deg)))
    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]])


def body_to_bearing(vec_body, heading_deg: float, pitch_deg: float = 0.0,
                    roll_deg: float = 0.0) -> tuple:
    """`(azimuth_deg_true, elevation_deg)` for a direction in body axes."""
    v = np.asarray(vec_body, dtype=float).reshape(3)
    n, e, d = rotation_body_to_ned(heading_deg, pitch_deg, roll_deg) @ v
    az = math.degrees(math.atan2(float(e), float(n))) % 360.0
    el = math.degrees(math.atan2(-float(d), math.hypot(float(n), float(e))))
    return az, el


def cone_bearings(cone_deg: float, heading_deg: float, pitch_deg: float = 0.0,
                  roll_deg: float = 0.0, axis_body=(1.0, 0.0, 0.0),
                  target_elevation_deg: float = 0.0) -> list:
    """True bearings where a linear array's cone crosses a given elevation.

    **THIS IS THE HONEST OUTPUT OF A LINEAR ARRAY AND IT USUALLY HAS TWO
    ELEMENTS.** The array measures `cone_deg`, the angle between the arriving
    wavefront and the array axis. Every direction at that angle to the axis is
    consistent with the measurement, and they form a cone. Intersecting the
    cone with the plane at `target_elevation_deg` — the horizon, for a ground
    emitter — leaves at most two directions, mirrored about the array axis.

    Returns them sorted, `[]` when the cone does not reach that elevation at
    all (which is itself a finding: the emitter is not where the geometry
    would need it to be), and ONE bearing when the cone is tangent.

    Resolving between two returned bearings needs something else: a second
    baseline that is not collinear, platform motion, or an a-priori. The
    bench's job is to say there are two, not to pick.
    """
    r = rotation_body_to_ned(heading_deg, pitch_deg, roll_deg)
    axis = np.asarray(axis_body, dtype=float).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm <= 0:
        return []
    axis_ned = r @ (axis / norm)
    ct = math.cos(math.radians(float(cone_deg)))
    el = math.radians(float(target_elevation_deg))
    # A direction at elevation `el` and azimuth φ is
    #   u = (cos el cos φ, cos el sin φ, −sin el)   in NED
    # and the constraint is  axis · u = cos(cone).  Expanding gives
    #   P cos φ + Q sin φ = R,  which is the standard one-unknown form.
    p = float(axis_ned[0]) * math.cos(el)
    q = float(axis_ned[1]) * math.cos(el)
    rr = ct + float(axis_ned[2]) * math.sin(el)
    amp = math.hypot(p, q)
    if amp < 1e-12:
        return []                       # axis is vertical: no azimuth is
        #                                 preferred, the cone IS the horizon
    if abs(rr) > amp + 1e-12:
        return []                       # the cone never reaches that elevation
    phase = math.atan2(q, p)
    inner = max(-1.0, min(1.0, rr / amp))
    delta = math.acos(inner)
    out = sorted({round((math.degrees(phase + delta)) % 360.0, 9),
                  round((math.degrees(phase - delta)) % 360.0, 9)})
    return out


# ---------------------------------------------------------------------------
# the line of bearing
# ---------------------------------------------------------------------------

@dataclass
class LOB:
    """One bearing, from one place, at one time, with its uncertainty.

    The common currency of the whole geolocation layer: whatever produced it —
    a phase interferometer, crossed loops, a bearing somebody typed in — it
    arrives here in the same shape, and `fusion.py` never needs to know which.

    `sigma_deg` is not optional and must not be invented. It is what decides
    how much this bearing moves the answer, and a fix built from bearings with
    made-up uncertainties has a made-up error ellipse.
    """
    lat: float
    lon: float
    bearing_deg: float
    sigma_deg: float
    alt_m: float = 0.0
    epoch: float = 0.0
    source: str = ""
    method: str = ""
    quality: float = 1.0
    ambiguous_with: float | None = None     # the mirrored cone solution
    #: How well the RECEIVER's own position was known, in metres, one sigma.
    #:
    #: A first-class field rather than a note, because it is an error in the
    #: measurement and has to reach the estimator. A position uncertain by s
    #: metres, seen from a target at range d, is a bearing uncertain by s/d
    #: radians — which at 22 km makes 500 m of platform error worth 1.3°,
    #: comparable to the array's own error and previously invisible to the
    #: covariance. `fusion` folds it in at the range it finds.
    position_sigma_m: float = 0.0
    evidence: dict = field(default_factory=dict)

    def point_at(self, range_m: float) -> tuple:
        """`(lat, lon)` this far out along the bearing."""
        return direct(self.lat, self.lon, self.bearing_deg, float(range_m))

    def to_geojson(self, range_m: float = 50_000.0) -> dict:
        a = (self.lat, self.lon)
        b = self.point_at(range_m)
        return {"type": "Feature",
                "geometry": {"type": "LineString",
                             "coordinates": [[a[1], a[0]], [b[1], b[0]]]},
                "properties": {"bearing_deg": round(self.bearing_deg, 3),
                               "sigma_deg": round(self.sigma_deg, 3),
                               "epoch": self.epoch, "source": self.source,
                               "method": self.method}}

    def describe(self) -> str:
        bits = [f"{self.bearing_deg:.2f}° ±{self.sigma_deg:.2f}°",
                f"from {self.lat:.6f}, {self.lon:.6f}"]
        if self.ambiguous_with is not None:
            bits.append(f"OR {self.ambiguous_with:.2f}° — the array's cone "
                        f"crosses the horizon twice and nothing here "
                        f"separates them")
        if self.method:
            bits.append(self.method)
        return " · ".join(bits)
