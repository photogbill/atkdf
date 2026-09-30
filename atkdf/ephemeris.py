"""Where the antenna was and how it was pointing, read out of the file.

A bearing without a position is not a line of bearing, it is a number. This
module is the part that turns a capture into something geolocation can use,
and it reads the four places a BLUE file actually keeps that information:

    quadwords    the 96-byte platform block at file offset 416 in every
                 type-5000 header — position AND attitude, fixed offsets,
                 no site convention needed. The format's own answer.
    records      a type-5000 state-vector track, or a type-3000 record file
                 with recognisable columns: a position per instant
    keywords     extended-header keywords, by a CONFIGURABLE map, because
                 this is where every tool that did not know about the
                 quadwords put it, and each of them spelled it differently
    companion    a CSV, a GPX, or one of ATK's own GPS track files beside
                 the capture

WHAT THIS MODULE WILL NOT DO, and the reason is the same in each case: it will
not invent a number that changes the answer.

  * **It will not default the attitude.** With no heading, an array-frame
    angle cannot become a true bearing at all — not approximately, not with a
    caveat. `bearing_ready` says so and the DF layer refuses. Course over
    ground can be used INSTEAD of heading, but only if asked for explicitly,
    because a platform that is crabbing in a crosswind points somewhere other
    than where it is going and the difference is a bearing error.
  * **It will not extrapolate past the ends of the track.** Beyond the last
    fix there is no information, and a linear continuation of a turning
    aircraft is fiction that looks like data.
  * **It interpolates BETWEEN fixes, and says by how much.** ATK's GPS track
    reader deliberately returns the nearest fix instead, on the grounds that
    an interpolated position is a guess. That is right for plotting a track
    and wrong here: at 200 m/s a one-second-old position is 200 m from the
    truth, and 200 m of baseline error is a real bearing error on a real fix.
    So this interpolates, marks the state `interpolated`, records the gap it
    spanned, and inflates the position uncertainty by what the platform could
    have done in that gap — so the error arrives in the fix rather than being
    quietly absorbed.

MOVED FROM ATK, 2026-09-30 (ATK c52b913, `atk/core/siga/ephemeris.py`), with its tests
and this docstring. ATK's `atk.core.siga.ephemeris` is now an alias of this
module — the same module object, reached through `atk/core/df_engine.py`
— so the Geolocation tab and ATK's own tests run this code. Edit it here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import bluefile

#: Where each field might be found, by the names tools actually use. The
#: first group is the VITA-49 vocabulary REDHAWK emits, which is the closest
#: thing to a standard; the rest is what turns up in practice. Callers can
#: pass their own map — that is the point of it being data.
KEYWORD_MAP = {
    "lat": ["GEOLOCATION_GPS::LATITUDE", "GEOLOCATION_INS::LATITUDE",
            "LATITUDE", "LAT", "PLATFORM_LAT", "ANT_LAT", "RX_LAT"],
    "lon": ["GEOLOCATION_GPS::LONGITUDE", "GEOLOCATION_INS::LONGITUDE",
            "LONGITUDE", "LON", "LNG", "PLATFORM_LON", "ANT_LON", "RX_LON"],
    "alt_m": ["GEOLOCATION_GPS::ALTITUDE", "GEOLOCATION_INS::ALTITUDE",
              "ALTITUDE", "ALT", "ALT_M", "HEIGHT", "PLATFORM_ALT"],
    "heading_deg": ["GEOLOCATION_INS::HEADING_ANGLE",
                    "GEOLOCATION_GPS::HEADING_ANGLE", "HEADING",
                    "HEADING_DEG", "YAW", "PLATFORM_HEADING", "ANT_HEADING"],
    "course_deg": ["GEOLOCATION_GPS::TRACK_ANGLE",
                   "GEOLOCATION_INS::TRACK_ANGLE", "TRACK_ANGLE", "COURSE",
                   "COG"],
    "pitch_deg": ["PITCH", "PITCH_DEG", "ELEVATION_ANGLE", "PLATFORM_PITCH"],
    "roll_deg": ["ROLL", "ROLL_DEG", "PLATFORM_ROLL"],
    "speed_mps": ["GEOLOCATION_GPS::GROUND_SPEED", "GROUND_SPEED", "SPEED",
                  "SOG"],
    "epoch": ["GEOLOCATION_GPS::TIME_SECONDS", "GEOLOCATION_INS::TIME_SECONDS",
              "EPHEMERIS_ECEF::TIME_SECONDS", "TIME", "TIMESTAMP", "EPOCH",
              "FIX_EPOCH", "FIX_TIME"],
    # HOW WELL THE RECEIVER KNEW WHERE IT WAS. Without these a keyword
    # ephemeris claims a perfect position, which propagates into the fix
    # covariance as though the antenna were a mathematical point — the same
    # silent zero that took measured coverage from 99 % to 53 %.
    "quality": ["FIX_QUALITY", "GPS_QUALITY", "QUALITY", "FIX_TYPE"],
    "position_sigma_m": ["POSITION_SIGMA_M", "POS_SIGMA_M", "HACC_M",
                         "ACCURACY_M", "HACC", "EPH"],
}

#: VITA-49 carries attitude as ALPHA/BETA/PHI. The mapping to
#: heading/pitch/roll is the conventional reading and is NOT stated in any
#: source I could confirm, so it is applied only when nothing better exists
#: and the state it produces is marked with the assumption.
VITA_ATTITUDE = {"heading_deg": "EPHEMERIS_ECEF::ATTITUDE_ALPHA",
                 "pitch_deg": "EPHEMERIS_ECEF::ATTITUDE_BETA",
                 "roll_deg": "EPHEMERIS_ECEF::ATTITUDE_PHI"}

#: How far apart two fixes may be before interpolating between them is a
#: fabrication rather than an approximation.
MAX_GAP_S = 30.0

#: Speed assumed when a track carries no velocity, for inflating the position
#: uncertainty across an interpolation gap. Deliberately generous.
ASSUMED_SPEED_MPS = 250.0


@dataclass
class PlatformState:
    """Where the antenna was, and how it was pointing, at one instant."""
    epoch: float
    lat: float
    lon: float
    alt_m: float = 0.0
    heading_deg: float | None = None
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    speed_mps: float | None = None
    course_deg: float | None = None
    position_sigma_m: float = 0.0
    heading_sigma_deg: float = 0.0
    interpolated: bool = False
    gap_s: float = 0.0
    source: str = ""
    assumptions: list = field(default_factory=list)

    @property
    def bearing_ready(self) -> bool:
        """Can an array-frame angle become a true bearing from this state?

        Only with a heading. There is no partial credit and no sensible
        default: without it the array's angle is relative to an unknown
        direction, and every bearing derived from it is unknown by the same
        amount.
        """
        return self.heading_deg is not None

    def describe(self) -> str:
        bits = [f"{self.lat:.6f}, {self.lon:.6f} @ {self.alt_m:,.0f} m"]
        if self.heading_deg is None:
            bits.append("NO HEADING — bearings cannot be referenced to true "
                        "north from this state")
        else:
            bits.append(f"hdg {self.heading_deg:.1f}°")
            if self.pitch_deg or self.roll_deg:
                bits.append(f"pitch {self.pitch_deg:.1f}° "
                            f"roll {self.roll_deg:.1f}°")
        if self.interpolated:
            bits.append(f"interpolated across {self.gap_s:.2f} s "
                        f"(±{self.position_sigma_m:,.0f} m from that alone)")
        for a in self.assumptions:
            bits.append(f"ASSUMED: {a}")
        return " · ".join(bits)


class Ephemeris:
    """A time-ordered platform track, with honest lookup.

    `at(t)` is the whole point: an intercept carries a timestamp and this
    turns it into a position and an attitude. Everything else here exists to
    fill this in from whatever the file happened to carry.
    """

    def __init__(self, states=None, source: str = "",
                 note: str = "") -> None:
        self.states = sorted(states or [], key=lambda s: s.epoch)
        self.source = source
        self.note = note
        self._t = np.array([s.epoch for s in self.states], dtype=float)

    def __len__(self) -> int:
        return len(self.states)

    def __bool__(self) -> bool:
        return bool(self.states)

    @property
    def span_s(self) -> float:
        return (self._t[-1] - self._t[0]) if len(self._t) > 1 else 0.0

    @property
    def static(self) -> bool:
        """One state and no time span — a fixed site, or a header block."""
        return len(self.states) == 1

    @property
    def has_attitude(self) -> bool:
        return any(s.heading_deg is not None for s in self.states)

    def at(self, epoch: float, max_gap_s: float = MAX_GAP_S,
           interpolate: bool = True) -> PlatformState | None:
        """The platform state at `epoch`, or None if the track cannot say.

        None is a real answer and callers must handle it: outside the track
        there is no information, and the alternative — extrapolating — is a
        straight line through wherever the aircraft was going to turn.
        """
        if not self.states:
            return None
        if self.static:
            s = self.states[0]
            # A single state with no time is a FIXED SITE and applies at any
            # time. A single state WITH a time is one observation, and using
            # it an hour later is a choice the caller has to make knowingly.
            if s.epoch and abs(float(epoch) - s.epoch) > max_gap_s:
                return None
            return s
        t = float(epoch)
        if t < self._t[0] - max_gap_s or t > self._t[-1] + max_gap_s:
            return None
        k = int(np.searchsorted(self._t, t))
        if k <= 0:
            return self._edge(self.states[0], t, max_gap_s)
        if k >= len(self.states):
            return self._edge(self.states[-1], t, max_gap_s)
        a, b = self.states[k - 1], self.states[k]
        gap = b.epoch - a.epoch
        if gap <= 0:
            return a
        if not interpolate:
            near = a if (t - a.epoch) <= (b.epoch - t) else b
            out = PlatformState(**{**near.__dict__})
            out.gap_s = abs(t - near.epoch)
            out.position_sigma_m = max(
                near.position_sigma_m,
                out.gap_s * (near.speed_mps or ASSUMED_SPEED_MPS))
            return out
        if gap > max_gap_s:
            return None
        return self._blend(a, b, (t - a.epoch) / gap, t, gap)

    def _edge(self, s: PlatformState, t: float,
              max_gap_s: float) -> PlatformState | None:
        gap = abs(t - s.epoch)
        if gap > max_gap_s:
            return None
        out = PlatformState(**{**s.__dict__})
        out.epoch = t
        out.gap_s = gap
        out.interpolated = gap > 0
        # Off the end of the track the position is only as good as what the
        # platform could have done since — no extrapolation, but no pretence
        # that the last fix is still current either.
        out.position_sigma_m = math.hypot(
            s.position_sigma_m, gap * (s.speed_mps or ASSUMED_SPEED_MPS))
        if gap > 0:
            out.assumptions = list(s.assumptions) + [
                f"the platform had not moved since the last fix "
                f"{gap:.2f} s earlier"]
        return out

    @staticmethod
    def _blend(a: PlatformState, b: PlatformState, f: float, t: float,
               gap: float) -> PlatformState:
        def lin(x, y):
            return None if (x is None or y is None) else x + (y - x) * f

        def ang(x, y):
            """Angles interpolate the SHORT way round. Interpolating 359° and
            1° arithmetically gives 180°, which points the array backwards."""
            if x is None or y is None:
                return None
            d = (float(y) - float(x) + 180.0) % 360.0 - 180.0
            return (float(x) + d * f) % 360.0

        out = PlatformState(
            epoch=t, lat=lin(a.lat, b.lat), lon=lin(a.lon, b.lon),
            alt_m=lin(a.alt_m, b.alt_m) or 0.0,
            heading_deg=ang(a.heading_deg, b.heading_deg),
            pitch_deg=lin(a.pitch_deg, b.pitch_deg) or 0.0,
            roll_deg=lin(a.roll_deg, b.roll_deg) or 0.0,
            speed_mps=lin(a.speed_mps, b.speed_mps),
            course_deg=ang(a.course_deg, b.course_deg),
            heading_sigma_deg=max(a.heading_sigma_deg, b.heading_sigma_deg),
            interpolated=True, gap_s=gap,
            source=a.source, assumptions=list(a.assumptions))
        # A straight line between two fixes is wrong by however much the
        # platform turned in between, and the worst case is bounded by how
        # far it could travel: a quarter of the chord is the standard rough
        # bound for a constant-rate turn and it is not tight, which is the
        # right direction to be wrong in.
        #
        # BUT IT IS ZERO AT THE ENDS. A turn bows the path away from the
        # chord in the MIDDLE and not at the fixes, which are on the path by
        # definition — so the term is shaped 4f(1−f), peaking at a quarter of
        # the chord halfway across and vanishing at either fix. A flat 0.25
        # charged the full mid-gap error to a block whose time landed exactly
        # on a recorded position: measured, 1,826 m of invented uncertainty
        # on a state known to 10 m, which then propagated into every bearing
        # taken from it.
        chord = 0.0
        try:
            from .geodesy import distance_between
            chord = distance_between(a.lat, a.lon, b.lat, b.lon)
        except Exception:                                   # noqa: BLE001
            chord = gap * (a.speed_mps or ASSUMED_SPEED_MPS)
        bow = chord * max(0.0, min(1.0, f)) * max(0.0, 1.0 - min(1.0, f))
        out.position_sigma_m = math.hypot(
            max(a.position_sigma_m, b.position_sigma_m), bow)
        out.interpolated = 0.0 < f < 1.0
        return out

    def use_course_as_heading(self) -> "Ephemeris":
        """Substitute course over ground for a missing heading. Opt-in.

        **A CRABBING PLATFORM POINTS SOMEWHERE OTHER THAN WHERE IT IS
        GOING**, and the difference goes straight into every bearing. In a
        20 kt crosswind at 120 kt that is about 10°. Sometimes course is all
        there is and a 10° bearing is better than none — but it is the
        caller's decision, it is recorded on every state it touches, and it
        carries an uncertainty that says so.
        """
        out = []
        for s in self.states:
            if s.heading_deg is None and s.course_deg is not None:
                s = PlatformState(**{**s.__dict__})
                s.heading_deg = s.course_deg
                s.heading_sigma_deg = max(s.heading_sigma_deg, 10.0)
                s.assumptions = list(s.assumptions) + [
                    "course over ground used as heading — a platform "
                    "crabbing in a crosswind points elsewhere, and that "
                    "difference is a bearing error of the same size"]
            out.append(s)
        return Ephemeris(out, self.source,
                         (self.note + " · course substituted for heading"
                          ).strip(" ·"))

    def describe(self) -> str:
        if not self.states:
            return "no ephemeris"
        if self.static:
            return f"fixed platform · {self.states[0].describe()}"
        a, b = self.states[0], self.states[-1]
        d = ""
        try:
            from .geodesy import distance_between
            d = f" · moved {distance_between(a.lat, a.lon, b.lat, b.lon):,.0f} m"
        except Exception:                                   # noqa: BLE001
            pass
        att = ("attitude present" if self.has_attitude else
               "NO ATTITUDE — bearings cannot be referenced to true north")
        return (f"{len(self.states):,} states over {self.span_s:,.1f} s{d} · "
                f"{att}" + (f" · {self.note}" if self.note else ""))


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------

def _first(kw: dict, names, default=None):
    for n in names:
        if n in kw:
            v = kw[n]
            if isinstance(v, str):
                try:
                    v = float(v.strip())
                except (TypeError, ValueError):
                    continue
            if isinstance(v, (int, float)) and np.isfinite(v):
                return float(v), n
    return default, ""   # default is a SCALAR; the caller unpacks a pair


def from_quadwords(hdr: dict) -> Ephemeris | None:
    """The type-5000 platform block: position and attitude at fixed offsets.

    The best case in the whole module, because there is nothing to configure
    and nothing to guess. `azimuth` in that block is the platform's heading
    and `elevation` its pitch; the names come from the topocentric reading of
    the frame, which is what `frame_of_ref` selects.
    """
    q = hdr.get("quadwords") or {}
    lat, lon = q.get("latitude"), q.get("longitude")
    if not lat and not lon:
        return None
    if not np.isfinite(lat or 0.0) or not np.isfinite(lon or 0.0):
        return None
    frame = (q.get("frame_of_ref") or "").upper()
    st = PlatformState(
        epoch=float(hdr.get("timecode_unix") or 0.0),
        lat=float(lat), lon=float(lon), alt_m=float(q.get("altitude") or 0.0),
        heading_deg=(float(q["azimuth"]) if q.get("azimuth") is not None
                     else None),
        pitch_deg=float(q.get("elevation") or 0.0),
        roll_deg=float(q.get("roll") or 0.0),
        source=f"BLUE type-5000 quadwords ({frame or 'frame unstated'})")
    if frame and "TOPO" not in frame:
        # The same doubles mean different things per frame. Reading an ECI
        # block as topocentric puts the platform at a plausible-looking
        # latitude that is actually an orbital element.
        st.assumptions.append(
            f"the quadword block declares frame {frame!r}, which is not a "
            f"topocentric frame — these values were read as latitude, "
            f"longitude and altitude anyway and may be orbital elements")
    if not frame:
        st.assumptions.append(
            "the quadword block states no frame of reference; a topocentric "
            "reading was assumed")
    return Ephemeris([st], hdr.get("path", ""), "from the type-5000 header")


def from_records(path) -> Ephemeris | None:
    """A state-vector track: a type-5000 or type-3000 record file.

    `POS` under a GEODETIC component is **(altitude, latitude, longitude)** —
    altitude FIRST. Every instinct says lat/lon/alt and every instinct is
    wrong; reading it in the obvious order puts the platform at 3,000° north,
    which at least fails loudly. What fails quietly is a file whose altitude
    happens to be small.
    """
    try:
        arr, hdr = bluefile.read_records(path)
    except (bluefile.BlueError, OSError, ValueError):
        return None
    if arr.size == 0:
        return None
    names = set(arr.dtype.names or ())
    cols = {c["name"].upper(): c for c in (hdr.get("comp")
                                           or hdr.get("subr") or [])}
    states = []

    def col(row, key, idx=None):
        if key not in names:
            return None
        v = row[key]
        return float(v[idx]) if idx is not None else float(v)

    geodetic = (cols.get("POS", {}).get("type") == 6) or ("POS" in names)
    for row in arr:
        if "pos" in names and geodetic:
            alt, lat, lon = (float(row["pos"][0]), float(row["pos"][1]),
                             float(row["pos"][2]))
        else:
            lat = col(row, "lat") or col(row, "latitude")
            lon = col(row, "lon") or col(row, "longitude")
            alt = col(row, "alt") or col(row, "altitude") or 0.0
        if lat is None or lon is None or not (np.isfinite(lat)
                                              and np.isfinite(lon)):
            continue
        t = col(row, "time")
        if t is not None and t > bluefile.J1950_TO_UNIX / 2:
            t -= bluefile.J1950_TO_UNIX          # J1950 → Unix
        elif t is None:
            t = float(hdr.get("timecode_unix") or 0.0)
        hdg = col(row, "hdg")
        pitch = roll = 0.0
        if "elps" in names:
            # ELPS is (azimuth, elevation, roll) — the platform's attitude
            # when the file is carrying one.
            hdg = hdg if hdg is not None else float(row["elps"][0])
            pitch, roll = float(row["elps"][1]), float(row["elps"][2])
        vel = row["vel"] if "vel" in names else None
        states.append(PlatformState(
            epoch=float(t), lat=lat, lon=lon, alt_m=alt,
            heading_deg=hdg, pitch_deg=pitch, roll_deg=roll,
            speed_mps=(float(np.linalg.norm(vel)) if vel is not None
                       and np.any(vel) else None),
            source=f"BLUE type-{hdr.get('type')} state vectors"))
    if not states:
        return None
    return Ephemeris(states, str(path),
                     f"{len(states):,} state vectors from the record data")


def from_keywords(hdr: dict, keymap: dict | None = None,
                  allow_vita_attitude: bool = True) -> Ephemeris | None:
    """Platform state out of header keywords, by a configurable map.

    This is the fallback for every tool that wrote an ephemeris somewhere of
    its own choosing. Which spelling was found is recorded, because the next
    person to open a file from the same producer will want to know.
    """
    kw = dict(hdr.get("keywords") or {})
    kw.update(hdr.get("ext_keywords") or {})
    if not kw:
        return None
    m = dict(KEYWORD_MAP)
    if keymap:
        for k, v in keymap.items():
            m[k] = ([v] if isinstance(v, str) else list(v)) + m.get(k, [])
    found = {}
    lat, lat_k = _first(kw, m["lat"])
    lon, lon_k = _first(kw, m["lon"])
    if lat is None or lon is None:
        return None
    found["lat"], found["lon"] = lat_k, lon_k
    alt, found["alt_m"] = _first(kw, m["alt_m"], 0.0)
    hdg, found["heading_deg"] = _first(kw, m["heading_deg"], None)
    pitch, found["pitch_deg"] = _first(kw, m["pitch_deg"], 0.0)
    roll, found["roll_deg"] = _first(kw, m["roll_deg"], 0.0)
    course, found["course_deg"] = _first(kw, m["course_deg"], None)
    speed, found["speed_mps"] = _first(kw, m["speed_mps"], None)
    epoch, found["epoch"] = _first(kw, m["epoch"], None)
    qual, found["quality"] = _first(kw, m.get("quality", []), None)
    psig, found["position_sigma_m"] = _first(
        kw, m.get("position_sigma_m", []), None)

    assumptions = []
    if hdg is None and allow_vita_attitude:
        hdg, k = _first(kw, [VITA_ATTITUDE["heading_deg"]], None)
        if hdg is not None:
            found["heading_deg"] = k
            pitch = _first(kw, [VITA_ATTITUDE["pitch_deg"]], 0.0)[0]
            roll = _first(kw, [VITA_ATTITUDE["roll_deg"]], 0.0)[0]
            assumptions.append(
                "VITA-49 ATTITUDE_ALPHA/BETA/PHI read as heading/pitch/roll "
                "— the conventional reading, not one I could confirm against "
                "the standard. If bearings come out rotated, this is the "
                "first thing to check")
    st = PlatformState(
        epoch=float(epoch if epoch is not None
                    else (hdr.get("timecode_unix") or 0.0)),
        lat=lat, lon=lon, alt_m=alt or 0.0, heading_deg=hdg,
        pitch_deg=pitch or 0.0, roll_deg=roll or 0.0,
        speed_mps=speed, course_deg=course,
        position_sigma_m=position_sigma_from(
            {"quality": qual, "position_sigma_m": psig}),
        source="BLUE header keywords", assumptions=assumptions)
    return Ephemeris([st], hdr.get("path", ""),
                     "keywords: " + ", ".join(f"{k}←{v}" for k, v in
                                              found.items() if v))


def from_rows(rows, source: str = "") -> Ephemeris | None:
    """A companion track: dicts with lat/lon and, ideally, epoch and heading.

    Accepts ATK's own GPS track rows unchanged, which is the point — a drive
    with the GPS logger running IS an ephemeris.
    """
    states = []
    for r in rows or []:
        try:
            lat, lon = float(r["lat"]), float(r["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        hdg = r.get("heading_deg", r.get("heading"))
        course = r.get("course_deg", r.get("course"))
        spd = r.get("speed_mps")
        if spd is None and r.get("speed_knots") is not None:
            spd = float(r["speed_knots"]) * 0.514444
        states.append(PlatformState(
            epoch=float(r.get("epoch") or 0.0), lat=lat, lon=lon,
            alt_m=float(r.get("altitude_m") or r.get("alt_m") or 0.0),
            # HOW WELL THE RECEIVER KNEW ITS OWN POSITION travels with the
            # position. At range it becomes an angular error on every bearing
            # taken from here, and it is the term that took measured coverage
            # from 53 % to 99 % once it stopped being silently zero.
            position_sigma_m=position_sigma_from(r),
            heading_deg=(float(hdg) if hdg is not None else None),
            pitch_deg=float(r.get("pitch_deg") or 0.0),
            roll_deg=float(r.get("roll_deg") or 0.0),
            speed_mps=(float(spd) if spd is not None else None),
            course_deg=(float(course) if course is not None else None),
            source=source or "companion track"))
    return Ephemeris(states, source, "companion track") if states else None


def from_csv(path) -> Ephemeris | None:
    """A CSV beside the capture. Column names are matched case-insensitively
    against the same vocabulary as the keywords."""
    import csv
    p = Path(path)
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    except (OSError, UnicodeDecodeError, csv.Error):
        return None
    alias = {"lat": ("lat", "latitude"), "lon": ("lon", "lng", "longitude"),
             "alt_m": ("alt", "alt_m", "altitude", "altitude_m", "height"),
             "epoch": ("epoch", "time", "timestamp", "utc"),
             "heading_deg": ("heading", "heading_deg", "hdg", "yaw"),
             "pitch_deg": ("pitch", "pitch_deg"),
             "roll_deg": ("roll", "roll_deg"),
             "course_deg": ("course", "course_deg", "cog", "track"),
             "speed_mps": ("speed_mps", "speed"),
             "speed_knots": ("speed_knots", "sog"),
             # HOW WELL THE RECEIVER KNEW ITS OWN POSITION. Dropping these
             # on the way in is not a small loss: without them every state
             # falls back to the blanket default and a surveyed or RTK
             # position is treated no better than an unknown one.
             "quality": ("quality", "fix_quality", "gps_quality"),
             "position_sigma_m": ("position_sigma_m", "pos_sigma_m",
                                  "hacc_m", "accuracy_m", "hacc", "eph")}
    out = []
    for r in rows:
        low = {str(k).strip().lower(): v for k, v in r.items() if k}
        rec = {}
        for key, names in alias.items():
            for n in names:
                if low.get(n) not in (None, ""):
                    try:
                        rec[key] = float(low[n])
                    except (TypeError, ValueError):
                        pass
                    break
        if "lat" in rec and "lon" in rec:
            out.append(rec)
    return from_rows(out, str(p))


#: Position uncertainty, one sigma in metres, by NMEA GGA fix quality.
#:
#: **THESE ARE ASSIGNMENTS, NOT MEASUREMENTS**, in the same spirit as
#: `fusion.BIAS_BOUNDS`: a receiver that reports its fix quality is telling
#: you something real about how well it knows where it is, and turning that
#: into a number is better than the silent zero that says "the antenna
#: position is exact". At 22 km a 10 m position error is only 0.03° and
#: matters to nobody; a 100 m one is 0.26° and matters a great deal, and the
#: point of carrying the figure at all is that the estimator finds out which
#: case it is in. Conservative by intent — a standalone GPS routinely does
#: better than 10 m and is occasionally much worse.
GPS_SIGMA_M = {
    0: 100.0,      # no fix / invalid — should not appear, treated as bad
    1: 10.0,       # standalone GPS
    2: 3.0,        # DGPS / SBAS
    3: 10.0,       # PPS
    4: 0.05,       # RTK fixed
    5: 0.5,        # RTK float
    6: 50.0,       # dead reckoning
}

#: When nothing says how good the fix was. A position of unknown provenance
#: is not a good position.
DEFAULT_GPS_SIGMA_M = 15.0


def position_sigma_from(row) -> float:
    """One sigma in metres for a track row, from whatever it admits to.

    An explicit figure wins; otherwise the NMEA fix quality; otherwise the
    default. Never zero — a stated zero is a claim of exactness that no GPS
    receiver on earth can make, and it propagates straight into a fix's
    covariance as though the antenna position were a mathematical point.
    """
    for key in ("position_sigma_m", "pos_sigma_m", "hacc_m", "accuracy_m"):
        v = row.get(key) if hasattr(row, "get") else None
        if v not in (None, ""):
            try:
                return max(float(v), 0.0)
            except (TypeError, ValueError):
                pass
    q = row.get("quality") if hasattr(row, "get") else None
    try:
        return GPS_SIGMA_M[int(q)]
    except (TypeError, ValueError, KeyError):
        return DEFAULT_GPS_SIGMA_M


def from_sigmf(path) -> Ephemeris | None:
    """A SigMF sidecar's `core:geolocation`, which is where ATK's own
    recorder puts the position of a capture that did not move.

    One state, no attitude: a GeoJSON Point says where the antenna was and
    nothing about which way it faced, so an array bearing still cannot be
    referenced to north from this alone. That is the honest outcome and the
    reason this is a source rather than an answer — combined with a mounting
    angle from the header, or with a track that carries a course, it becomes
    usable.
    """
    import json
    p = Path(path)
    base = str(p)
    for suf in (".sigmf-data", ".sigmf-meta", ".cu8"):
        if base.lower().endswith(suf):
            base = base[: -len(suf)]
            break
    meta = Path(base + ".sigmf-meta")
    if not meta.exists():
        return None
    try:
        doc = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    g = doc.get("global") or {}
    coords = ((g.get("core:geolocation") or {}).get("coordinates") or [])
    if len(coords) < 2:
        return None
    try:
        # GeoJSON is [lon, lat, alt] — longitude FIRST. Reading it as
        # lat/lon puts every British capture in the Gulf of Guinea.
        lon, lat = float(coords[0]), float(coords[1])
    except (TypeError, ValueError):
        return None
    alt = 0.0
    if len(coords) > 2:
        try:
            alt = float(coords[2])
        except (TypeError, ValueError):
            alt = 0.0
    epoch = 0.0
    cap = (doc.get("captures") or [{}])[0]
    stamp = cap.get("core:datetime") or ""
    if stamp:
        import calendar
        import time as _t
        for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
            try:
                epoch = calendar.timegm(_t.strptime(stamp, fmt))
                break
            except ValueError:
                continue
    note = g.get("atk:geolocation_note") or ""
    st = PlatformState(
        epoch=float(epoch), lat=lat, lon=lon, alt_m=alt,
        position_sigma_m=DEFAULT_GPS_SIGMA_M,
        source="SigMF core:geolocation",
        assumptions=([note] if note else []) + [
            "a GeoJSON Point carries no heading, so an array angle cannot be "
            "referenced to true north from this source alone"])
    return Ephemeris([st], str(meta), "SigMF core:geolocation")


#: Suffixes tried beside a capture, in order, when hunting for a companion.
COMPANION_SUFFIXES = (".ephem.csv", ".track.csv", ".csv", ".gps.csv")


def find_companion(path) -> Path | None:
    p = Path(path)
    for suf in COMPANION_SUFFIXES:
        for cand in (p.with_suffix(suf), p.parent / (p.stem + suf),
                     p.parent / (p.name + suf)):
            if cand.exists() and cand != p:
                return cand
    return None


def load(path, keymap: dict | None = None, rows=None,
         prefer: str = "auto") -> tuple:
    """`(Ephemeris|None, report)` — everything tried, and what each gave.

    The report is the deliverable when the answer is None. "No ephemeris
    found" tells an analyst nothing; "the quadword block is all zeros, no
    keyword in the map matched, and there is no companion file beside it"
    tells them exactly which of the four to go and fix.
    """
    report: list = []
    hdr = None
    try:
        hdr = bluefile.read_header(path)
    except (bluefile.BlueError, OSError, ValueError) as exc:
        # A SigMF capture has no BLUE header and that is not a fault. Saying
        # "unreadable" about a perfectly good file sends an analyst hunting
        # for corruption instead of reading the next three lines, which are
        # where the answer is.
        from . import recio as _rec
        try:
            kind = _rec.kind(path)
        except (OSError, ValueError):
            kind = ""
        report.append("header: not a BLUE file — this is SigMF, whose "
                      "position lives in the sidecar (see below)"
                      if kind == "sigmf" else f"header: unreadable ({exc})")

    found: dict = {}
    if hdr is not None:
        e = from_quadwords(hdr)
        report.append("quadwords: " + (e.describe() if e else
                                       "absent or all zero"))
        if e:
            found["quadwords"] = e
        e = from_records(path)
        report.append("state-vector records: " + (e.describe() if e else
                                                  "none in this file"))
        if e:
            found["records"] = e
        e = from_keywords(hdr, keymap)
        report.append("keywords: " + (e.describe() if e else
                                      "no position keyword matched the map"))
        if e:
            found["keywords"] = e

    e = from_sigmf(path)
    report.append("SigMF geolocation: " + (e.describe() if e else
                                           "no sidecar, or none in it"))
    if e:
        found["sigmf"] = e

    if rows:
        e = from_rows(rows, "caller-supplied track")
        report.append("supplied track: " + (e.describe() if e else "empty"))
        if e:
            found["supplied"] = e
    else:
        comp = find_companion(path)
        if comp is not None:
            e = from_csv(comp)
            report.append(f"companion {comp.name}: "
                          + (e.describe() if e else "unreadable"))
            if e:
                found["companion"] = e
        else:
            report.append("companion file: none beside the capture")

    if not found:
        return None, report
    # A TRACK BEATS A SINGLE STATE, because a moving platform's position at
    # the moment of the intercept is the thing that matters. Among tracks,
    # prefer the one carrying attitude — a position without a heading cannot
    # produce a bearing at all.
    order = ["records", "supplied", "companion", "quadwords", "keywords",
             "sigmf"]
    if prefer in found:
        order = [prefer] + [o for o in order if o != prefer]
    def rank(e):
        # A TRACK BEATS A STATIC STATE EVEN WHEN THE STATIC ONE HAS THE
        # ATTITUDE. Where the platform was AT THE INTERCEPT is the thing a
        # bearing cannot do without, and a fixed mounting angle can be
        # borrowed afterwards — the reverse cannot: no attitude anywhere lets
        # you invent a track. Preferring attitude outright made a one-state
        # header block win over a ten-state flight path, which pinned the
        # whole collection to a single point.
        return (0 if e.static else 1, 1 if e.has_attitude else 0, len(e))

    best = None
    for key in order:
        e = found.get(key)
        if e is None:
            continue
        if best is None or rank(e) > rank(best[1]):
            best = (key, e)
    key, chosen = best
    # Fill a missing attitude from another source that has one — a companion
    # GPS track gives position, the header quadwords may give the mounting.
    if not chosen.has_attitude:
        for other_key, other in found.items():
            if other_key != key and other.has_attitude and other.static:
                mount = other.states[0]
                filled = []
                for s in chosen.states:
                    s2 = PlatformState(**{**s.__dict__})
                    s2.heading_deg = mount.heading_deg
                    s2.pitch_deg = mount.pitch_deg
                    s2.roll_deg = mount.roll_deg
                    s2.assumptions = list(s.assumptions) + [
                        f"attitude taken from {other_key} and held constant "
                        f"across the whole track — correct for a fixed "
                        f"mounting, wrong the moment the platform turns"]
                    filled.append(s2)
                chosen = Ephemeris(filled, chosen.source,
                                   chosen.note + f" + attitude from "
                                                 f"{other_key}")
                report.append(f"attitude filled from {other_key}")
                break
    report.append(f"USING: {key} — {chosen.describe()}")
    return chosen, report


def dump(path, keymap: dict | None = None) -> str:
    """The inspector. Point it at a real file and read what it says.

    Prints the header, every keyword with its decoded type, every record
    column, the quadword block, and then the four ephemeris sources and what
    each one yielded. This is how a file whose layout nobody documented
    becomes a file whose layout you know.
    """
    lines = [f"=== {Path(path).name} ==="]
    try:
        hdr = bluefile.read_header(path)
    except (bluefile.BlueError, OSError, ValueError) as exc:
        # SigMF has none of the structure below — no keyword block, no record
        # columns, no quadwords — so there is nothing to inspect except the
        # sidecar and the companion, and those come out of `load` in a
        # moment. Reporting "cannot read" about a file ATK itself wrote would
        # send its owner hunting for corruption.
        from . import recio as _rec
        try:
            if _rec.kind(path) == "sigmf":
                h = _rec.header(path)
                lines.append(f"  SigMF · {h.get('datatype') or '?'} · "
                             f"{float(h.get('sample_rate') or 0) / 1e6:,.3g} "
                             f"Msps · {int(h.get('subsize') or 1)} channel(s)")
                lines.append("  no BLUE header, keyword block, record columns "
                             "or quadwords — SigMF keeps none of those; "
                             "position lives in the sidecar and the companion")
                lines.append("  ephemeris sources:")
                _e, rep = load(path, keymap)
                lines.extend(f"    {r}" for r in rep)
                return "\n".join(lines)
        except (OSError, ValueError, KeyError):
            pass
        return "\n".join(lines + [f"  cannot read: {exc}"])
    lines.append("  " + bluefile.describe(hdr))
    lines.append(f"  class {hdr.get('class')} · header endian "
                 f"{hdr.get('head_rep')} · data endian {hdr.get('data_rep')}")

    kws = hdr.get("keywords") or {}
    if kws:
        lines.append("  main-header keywords:")
        for k, v in kws.items():
            lines.append(f"    {k:28s} = {v!r}")
    pairs = hdr.get("ext_keyword_pairs") or []
    lines.append(f"  extended keywords ({hdr.get('ext_keyword_style')}): "
                 f"{len(pairs)}")
    for k, v in pairs:
        shown = v if not isinstance(v, list) else (
            f"[{len(v)} values] {v[:4]}{'…' if len(v) > 4 else ''}")
        lines.append(f"    {k:28s} = {shown!r}  ({type(v).__name__})")

    cols = hdr.get("subr") or hdr.get("comp") or []
    if cols:
        lines.append("  record columns:")
        for c in cols:
            u = bluefile.UNITS.get(c.get("units"), c.get("units"))
            t = bluefile.COMP_TYPES.get(c.get("type"), c.get("type"))
            lines.append(f"    {c['name']:8s} {c['format']:3s} "
                         f"units={u} type={t} offset={c.get('offset')}")
            hint = bluefile.T5_STATE_COMPONENTS.get(c["name"].upper())
            if hint:
                lines.append(f"             {hint}")

    q = hdr.get("quadwords")
    if q:
        lines.append("  type-5000 quadwords (file offset 416):")
        for k, v in q.items():
            lines.append(f"    {k:16s} = {v!r}")

    lines.append("  ephemeris sources:")
    eph, report = load(path, keymap)
    for r in report:
        lines.append(f"    {r}")
    if eph:
        lines.append(f"  RESULT: {len(eph)} state(s) · "
                     f"{'attitude present' if eph.has_attitude else 'NO ATTITUDE'}")
        for s in eph.states[:5]:
            lines.append(f"    {s.describe()}")
        if len(eph) > 5:
            lines.append(f"    … {len(eph) - 5} more")
    else:
        lines.append("  RESULT: no ephemeris — direction finding from this "
                     "file cannot produce a georeferenced bearing")
    return "\n".join(lines)
