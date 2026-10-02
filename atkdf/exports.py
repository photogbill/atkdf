"""Live sharing: the DF picture as KML, GeoJSON, CoT and a record feed
(ATK docs/CUED_DF_PLAN.md §12A).

Everything here is a pure function from a `Picture` — a plain, frozen view
of what the engine knows right now — to text. ATK's share server calls
these per request; the Operator view draws the same `Picture`; a test
builds one by hand. Nothing in here opens a socket or a file.

Three rules, from the plan:

  * **The label is the person** (§12A.3). When the fleet map knows the key,
    the placemark's name and the CoT callsign are the person; the RF detail
    is underneath. A key the fleet map does not know is shown as what it is
    — never a guessed name.
  * **The map cannot be shown something ATK would not show its own
    operator.** A bearing-only track is a wedge, not a point; an aged search
    area is drawn the size it has grown to; the ellipse is the honest one.
  * **Time fades** (§12). A bearing is opaque when it lands and fades to a
    floor over `fade_s`; after `keep_s` it leaves the live views, never the
    log. Transparency says how old; size says how sure.
"""

from __future__ import annotations

import datetime as _dt
import html
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from xml.sax.saxutils import escape as _x

from . import geodesy as geo

# ---------------------------------------------------------------------------
# the picture
# ---------------------------------------------------------------------------

STATES = ("located", "bearing_only", "aged", "silent", "heard_via_repeater", "unknown")


@dataclass
class Vehicle:
    lat: float
    lon: float
    t_utc: float
    heading_deg: float | None = None
    speed_mps: float = 0.0
    site_id: str = "site"
    callsign: str = "ATK"
    #: what the array is doing, in the Operator view's words
    status: str = ""
    window_lo_hz: float = 0.0
    window_hi_hz: float = 0.0


@dataclass
class Lob:
    t_utc: float
    lat: float
    lon: float
    bearing_deg: float
    sigma_deg: float
    key: str = ""
    label: str = ""
    range_m: float = 5_000.0
    site_id: str = ""
    quality: float = 1.0


@dataclass
class Track:
    key: str
    label: str                  # the person when known, else the key / signature description
    state: str = "unknown"      # one of STATES
    name: str = ""
    role: str = ""
    team: str = ""
    priority: int = 0           # higher sorts first everywhere
    photo: str = ""
    lat: float | None = None
    lon: float | None = None
    semi_major_m: float = 0.0
    semi_minor_m: float = 0.0
    orientation_deg: float = 0.0
    containment: float = 0.9    # the ellipse's stated probability
    sigma_m: float = 0.0        # the 1σ radius, for the sentence
    n_bearings: int = 0
    first_heard: float = 0.0
    last_heard: float = 0.0
    last_lob: Lob | None = None
    motion_state: str = "unknown"
    protocol: str = ""
    fc_hz: float = 0.0
    cal_tier: str = ""
    in_box: bool = False
    trend: str = ""             # tightening | growing | steady | ""
    sentence: str = ""          # plain words, written by `plain_sentence`
    silence_threshold_s: float = 600.0
    extra: dict = field(default_factory=dict)

    @property
    def located(self) -> bool:
        return self.lat is not None and self.lon is not None and self.semi_major_m > 0


@dataclass
class Box:
    name: str
    polygon: list               # [(lat, lon), ...]
    inverted: bool = False


@dataclass
class RefMark:
    key: str
    lat: float
    lon: float
    name: str = ""
    verified: bool = False
    residual_deg: float | None = None


@dataclass
class Picture:
    t_utc: float
    site_id: str = "site"
    vehicle: Vehicle | None = None
    lobs: list = field(default_factory=list)
    tracks: list = field(default_factory=list)
    boxes: list = field(default_factory=list)
    references: list = field(default_factory=list)
    notices: list = field(default_factory=list)
    fade_s: float = 120.0
    keep_s: float = 900.0
    policy: str = "everything"
    sequence: int = 0
    title: str = "ATK live DF"


# ---------------------------------------------------------------------------
# building a picture from the engine's objects
# ---------------------------------------------------------------------------

def plain_sentence(t: Track, now: float) -> str:
    """One sentence a searcher can act on, first in every balloon."""
    bits = []
    if t.last_heard:
        bits.append(f"Last heard {_ago(now - t.last_heard)}.")
    if t.state == "heard_via_repeater":
        bits.append("Heard through a repeater, so no position yet — the ID and "
                    "the status are known, the place is not.")
    elif t.state == "bearing_only":
        bits.append("One direction only so far; the vehicle has to move to get "
                    "a position.")
    elif t.located:
        size = 2 * t.semi_major_m
        word = ("tightening" if t.trend == "tightening" else
                "growing" if t.trend == "growing" else "")
        bits.append(f"Search area about {_metres(size)} across"
                    + (f", {word}." if word else "."))
    if t.motion_state == "on_foot":
        bits.append("Probably on foot.")
    elif t.motion_state == "vehicle":
        bits.append("Probably in a vehicle.")
    elif t.motion_state == "stationary":
        bits.append("Not moving.")
    if t.state == "silent":
        bits.append(f"Silent past the {int(t.silence_threshold_s // 60)}-minute threshold.")
    return " ".join(bits) or "Nothing heard yet."


def label_for(key: str, name: str = "", role: str = "", protocol: str = "",
              fc_hz: float = 0.0, signature: dict | None = None) -> str:
    """The person when known; otherwise what the emitter IS, never a guess."""
    if name:
        return f"{name} ({role})" if role else name
    if key.startswith(("sig:", "freq:")):
        what = (signature or {}).get("modulation_hint") or protocol or "unknown"
        return f"{what} {fc_hz / 1e6:.3f} MHz".strip() if fc_hz else f"{what} emitter"
    if key.startswith("track:"):
        return f"unidentified {protocol or 'emitter'}" + (f" {fc_hz / 1e6:.3f} MHz" if fc_hz else "")
    return key


def picture_from(registry, *, now: float, vehicle: Vehicle | None = None,
                 roster=None, posteriors: dict | None = None, boxes: Iterable = (),
                 references: Iterable = (), in_box=None, fleet_map: dict | None = None,
                 fade_s: float = 120.0, keep_s: float = 900.0, lob_range_m: float = 5_000.0,
                 containment: float = 0.9, site_id: str = "", motion: dict | None = None,
                 cal_tier: dict | None = None, history: dict | None = None,
                 notices: Iterable = ()) -> Picture:
    """The picture from a `Registry` (plus whatever else is available: the
    `Roster` for names and silence, per-key `Posterior`s for aged search
    areas, the box test, the references). Every argument but the registry
    is optional, so a bench with only bearings still produces a picture."""
    fleet = dict(fleet_map or {}) or dict(getattr(registry, "fleet", {}) or {})
    site = site_id or getattr(registry, "site_id", "") or "site"
    pic = Picture(t_utc=now, site_id=site, vehicle=vehicle, boxes=list(boxes),
                  references=list(references), fade_s=fade_s, keep_s=keep_s,
                  notices=list(notices))
    hist = history if history is not None else {}
    for key, em in getattr(registry, "emitters", {}).items():
        fm = fleet.get(key, {})
        name = em.name or fm.get("name", "")
        role = em.role or fm.get("role", "")
        team = em.team or fm.get("team", "")
        tr = Track(key=key, label=label_for(key, name, role, em.protocol,
                                             getattr(em.last_bearing, "fc_hz", 0.0) or 0.0),
                   name=name, role=role, team=team,
                   priority=int(fm.get("priority", 0) or 0), photo=fm.get("photo", ""),
                   n_bearings=em.n_bearings, first_heard=em.first_heard,
                   last_heard=em.last_heard, protocol=em.protocol,
                   fc_hz=float(getattr(em.last_bearing, "fc_hz", 0.0) or 0.0),
                   motion_state=(motion or {}).get(key, em.motion_state),
                   cal_tier=(cal_tier or {}).get(key, ""))
        # the fix: fusion's ellipse when there are enough bearings, else the
        # posterior's honest radius, else bearing-only
        fix = None
        try:
            fix = registry.fix(key)
        except Exception:                                    # noqa: BLE001
            fix = None
        post = (posteriors or {}).get(key)
        if fix is not None:
            a, b, orient = fix.ellipse(containment)
            tr.lat, tr.lon = float(fix.lat), float(fix.lon)
            tr.semi_major_m, tr.semi_minor_m, tr.orientation_deg = float(a), float(b), float(orient)
            tr.sigma_m = float(max(a, b)) / _k(containment)
            tr.state = "located"
        elif post is not None:
            f = post.fix()
            tr.lat, tr.lon = float(f["lat"]), float(f["lon"])
            r = float(f["sigma_m"]) * _k(containment)
            tr.semi_major_m = tr.semi_minor_m = r
            tr.sigma_m = float(f["sigma_m"])
            tr.state = "located" if tr.n_bearings >= 2 else "bearing_only"
        else:
            tr.state = "bearing_only"
        # the last FUSABLE line of bearing, for the wedge and the balloon — a
        # bearing taken through a repeater points at the repeater and is
        # never drawn as if it pointed at the radio
        fusable = registry.fusable_bearings(key)
        last = fusable[-1] if fusable else None
        if last is not None:
            tr.last_lob = Lob(t_utc=last.t_utc, lat=float(last.lat), lon=float(last.lon),
                              bearing_deg=float(last.theta_abs_deg),
                              sigma_deg=float(last.sigma_deg), key=key, label=tr.label,
                              range_m=lob_range_m, site_id=last.site_id or site)
        if not fusable and em.path_tags.get("repeated", 0) > 0:
            tr.state = "heard_via_repeater"
        # silence, from the roster when there is one
        entry = getattr(roster, "entries", {}).get(key) if roster is not None else None
        if entry is not None:
            tr.silence_threshold_s = entry.silence_s or getattr(roster, "default_silence", 600.0)
            if entry.silent_flagged or (entry.last_heard >= 0 and
                                        now - entry.last_heard >= tr.silence_threshold_s):
                tr.state = "silent" if tr.located or tr.state == "bearing_only" else tr.state
        elif tr.last_heard and now - tr.last_heard >= tr.silence_threshold_s and tr.located:
            tr.state = "aged"
        # trend from the caller's history of (t, sigma_m)
        h = hist.get(key)
        if h and len(h) >= 2:
            tr.trend = ("tightening" if h[-1][1] < h[0][1] * 0.8 else
                        "growing" if h[-1][1] > h[0][1] * 1.25 else "steady")
        tr.in_box = bool(in_box(key)) if in_box else False
        tr.sentence = plain_sentence(tr, now)
        pic.tracks.append(tr)
        # recent lines of bearing (fusable only; a repeater's bearing points at the repeater)
        for rec in fusable:
            if now - rec.t_utc > keep_s:
                continue
            pic.lobs.append(Lob(t_utc=rec.t_utc, lat=float(rec.lat), lon=float(rec.lon),
                                bearing_deg=float(rec.theta_abs_deg),
                                sigma_deg=float(rec.sigma_deg), key=key, label=tr.label,
                                range_m=lob_range_m, site_id=rec.site_id or site,
                                quality=float(rec.quality.get("confidence", 1.0))))
    pic.tracks.sort(key=lambda t: (-(t.priority), 0 if t.name else 1,
                                  -(t.last_heard or 0.0)))
    pic.lobs.sort(key=lambda l: l.t_utc)
    return pic


# ---------------------------------------------------------------------------
# the share policy
# ---------------------------------------------------------------------------

POLICIES = ("everything", "box_only", "selected", "fixes_only")


def apply_policy(pic: Picture, policy: str, selected: Iterable[str] = ()) -> Picture:
    """What leaves ATK. `box_only` keeps tracks in a box; `selected` keeps
    the given keys (named people by default when none are given);
    `fixes_only` keeps positions but strips every identity string — keys,
    names, callsigns — for a map that must not carry who is who."""
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, not {policy!r}")
    out = Picture(t_utc=pic.t_utc, site_id=pic.site_id, vehicle=pic.vehicle,
                  boxes=list(pic.boxes), references=list(pic.references),
                  notices=list(pic.notices), fade_s=pic.fade_s, keep_s=pic.keep_s,
                  policy=policy, sequence=pic.sequence, title=pic.title)
    if policy == "everything":
        keep = {t.key for t in pic.tracks}
    elif policy == "box_only":
        keep = {t.key for t in pic.tracks if t.in_box}
    elif policy == "selected":
        sel = set(selected) or {t.key for t in pic.tracks if t.name}
        keep = {t.key for t in pic.tracks if t.key in sel}
    else:
        keep = {t.key for t in pic.tracks}
    out.tracks = [t for t in pic.tracks if t.key in keep]
    out.lobs = [l for l in pic.lobs if l.key in keep]
    if policy == "fixes_only":
        anon = []
        for i, t in enumerate(out.tracks, 1):
            a = Track(key=f"anon:{i}", label=f"emitter {i}", state=t.state, lat=t.lat, lon=t.lon,
                      semi_major_m=t.semi_major_m, semi_minor_m=t.semi_minor_m,
                      orientation_deg=t.orientation_deg, containment=t.containment,
                      sigma_m=t.sigma_m, n_bearings=t.n_bearings, first_heard=t.first_heard,
                      last_heard=t.last_heard, motion_state=t.motion_state, trend=t.trend,
                      silence_threshold_s=t.silence_threshold_s)
            a.sentence = plain_sentence(a, pic.t_utc)
            if t.last_lob is not None:
                a.last_lob = Lob(t.last_lob.t_utc, t.last_lob.lat, t.last_lob.lon,
                                 t.last_lob.bearing_deg, t.last_lob.sigma_deg, key=a.key,
                                 label=a.label, range_m=t.last_lob.range_m)
            anon.append(a)
        rename = {t.key: a.key for t, a in zip(out.tracks, anon)}
        out.tracks = anon
        out.lobs = [Lob(l.t_utc, l.lat, l.lon, l.bearing_deg, l.sigma_deg,
                        key=rename.get(l.key, "anon"), label="", range_m=l.range_m,
                        quality=l.quality) for l in out.lobs]
        out.references = []
        out.notices = []
        if pic.vehicle is not None:
            v = pic.vehicle
            out.vehicle = Vehicle(v.lat, v.lon, v.t_utc, v.heading_deg, v.speed_mps,
                                  site_id="site", callsign="DF vehicle")
        out.site_id = "site"
    return out


# ---------------------------------------------------------------------------
# fade, ellipses, wedges
# ---------------------------------------------------------------------------

def fade_alpha(age_s: float, fade_s: float, floor: float = 0.15) -> float:
    """1.0 when it lands, linearly to `floor` at `fade_s`, flat after."""
    if fade_s <= 0:
        return 1.0
    f = 1.0 - max(0.0, age_s) / fade_s
    return max(floor, min(1.0, f))


def ellipse_ring(lat: float, lon: float, semi_major_m: float, semi_minor_m: float,
                 orientation_deg: float, n: int = 36) -> list:
    """`[(lat, lon), ...]` closed ring; orientation is the bearing of the
    major axis, degrees true (the convention `fusion.Fix.ellipse` gives)."""
    pts = []
    a = max(semi_major_m, 1.0)
    b = max(semi_minor_m, 1.0)
    for i in range(n + 1):
        th = 2 * math.pi * i / n
        # along-major (north-ish before rotation) and along-minor components
        x = a * math.cos(th)            # along the major axis
        y = b * math.sin(th)            # along the minor axis
        dist = math.hypot(x, y)
        ang = math.degrees(math.atan2(y, x))      # from the major axis
        brg = (orientation_deg + ang) % 360.0
        pts.append(geo.direct(lat, lon, brg, dist))
    return pts


def wedge_ring(lob: Lob, k_sigma: float = 2.0, n: int = 12) -> list:
    """A bearing-only emitter's honest shape: a wedge from the receiver,
    ±k·σ wide, out to the range prior."""
    half = max(0.5, k_sigma * lob.sigma_deg)
    pts = [(lob.lat, lob.lon)]
    for i in range(n + 1):
        brg = lob.bearing_deg - half + 2 * half * i / n
        pts.append(geo.direct(lob.lat, lob.lon, brg, lob.range_m))
    pts.append((lob.lat, lob.lon))
    return pts


# ---------------------------------------------------------------------------
# KML
# ---------------------------------------------------------------------------

#: Colours per state, as rgb hex; KML wants aabbggrr and the alpha is the fade.
STATE_RGB = {
    "located": "54c98a",            # green
    "bearing_only": "e6b34a",       # amber
    "aged": "9aa4b2",               # grey
    "silent": "ef6f6c",             # red
    "heard_via_repeater": "4c8dff", # blue — known, not located
    "unknown": "9aa4b2",
}
LOB_RGB = "4c8dff"
VEHICLE_RGB = "e7ecf3"
BOX_RGB = "b98cf0"
REF_RGB = "4fd6d0"


def _kml_colour(rgb: str, alpha: float) -> str:
    r, g, b = rgb[0:2], rgb[2:4], rgb[4:6]
    a = f"{round(max(0.0, min(1.0, alpha)) * 255):02x}"
    return f"{a}{b}{g}{r}"


def _iso(t: float) -> str:
    return _dt.datetime.fromtimestamp(float(t), tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _coords(ring: list) -> str:
    return " ".join(f"{lon:.6f},{lat:.6f},0" for lat, lon in ring)


def network_link_kml(url: str, refresh_s: float = 2.0, name: str = "ATK live DF") -> str:
    """The one-line document a client opens once; it polls `url` after."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2"><NetworkLink>'
        f'<name>{_x(name)}</name><open>1</open>'
        f'<Link><href>{_x(url)}</href><refreshMode>onInterval</refreshMode>'
        f'<refreshInterval>{float(refresh_s):g}</refreshInterval></Link>'
        '</NetworkLink></kml>\n')


def to_kml(pic: Picture, *, now: float | None = None) -> str:
    """The live document. Folders: vehicle, lines of bearing, people &
    emitters (person-first), boxes, references. Every bearing and fix has a
    `TimeStamp`, so a client's time slider replays the sortie from the live
    document alone."""
    now = pic.t_utc if now is None else now
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
           f'<name>{_x(pic.title)}</name>',
           f'<description>{_x(_doc_description(pic, now))}</description>']
    # styles per state (icon) — alpha is per placemark, so styles are inline
    # -- vehicle ---------------------------------------------------------------
    v = pic.vehicle
    if v is not None:
        out.append('<Folder><name>Own vehicle</name>')
        out.append(
            f'<Placemark><name>{_x(v.callsign)}</name>'
            f'<description>{_x(v.status or "")}</description>'
            f'<TimeStamp><when>{_iso(v.t_utc)}</when></TimeStamp>'
            f'<Style><IconStyle><color>{_kml_colour(VEHICLE_RGB, 1.0)}</color>'
            f'<heading>{(v.heading_deg or 0.0):.0f}</heading>'
            '<Icon><href>http://maps.google.com/mapfiles/kml/shapes/arrow.png</href></Icon>'
            '</IconStyle></Style>'
            + _extended({"site": v.site_id, "heading_deg": v.heading_deg,
                         "speed_mps": v.speed_mps,
                         "window_mhz": (f"{v.window_lo_hz/1e6:.3f}-{v.window_hi_hz/1e6:.3f}"
                                        if v.window_hi_hz else "")})
            + f'<Point><coordinates>{v.lon:.6f},{v.lat:.6f},0</coordinates></Point>'
            '</Placemark></Folder>')
    # -- lines of bearing --------------------------------------------------------
    out.append('<Folder><name>Lines of bearing</name>')
    for l in pic.lobs:
        age = now - l.t_utc
        if age > pic.keep_s:
            continue
        alpha = fade_alpha(age, pic.fade_s)
        width = 2.5 if l.sigma_deg <= 3 else 1.8 if l.sigma_deg <= 8 else 1.2
        end = geo.direct(l.lat, l.lon, l.bearing_deg, l.range_m)
        out.append(
            f'<Placemark><name>{_x(l.label or l.key)}</name>'
            f'<TimeStamp><when>{_iso(l.t_utc)}</when></TimeStamp>'
            f'<Style><LineStyle><color>{_kml_colour(LOB_RGB, alpha)}</color>'
            f'<width>{width}</width></LineStyle></Style>'
            + _extended({"key": l.key, "bearing_deg": round(l.bearing_deg, 1),
                         "sigma_deg": round(l.sigma_deg, 1), "age_s": round(age),
                         "site": l.site_id})
            + '<LineString><tessellate>1</tessellate><coordinates>'
            f'{l.lon:.6f},{l.lat:.6f},0 {end[1]:.6f},{end[0]:.6f},0'
            '</coordinates></LineString></Placemark>')
    out.append('</Folder>')
    # -- people and emitters -----------------------------------------------------------
    out.append('<Folder><name>People &amp; emitters</name>')
    for t in pic.tracks:
        out.append(_track_kml(t, pic, now))
    out.append('</Folder>')
    # -- boxes -----------------------------------------------------------------------
    if pic.boxes:
        out.append('<Folder><name>Boxes</name>')
        for b in pic.boxes:
            ring = list(b.polygon) + [b.polygon[0]] if b.polygon and b.polygon[0] != b.polygon[-1] else list(b.polygon)
            out.append(
                f'<Placemark><name>{_x(b.name)}{" (everything except)" if b.inverted else ""}</name>'
                f'<Style><LineStyle><color>{_kml_colour(BOX_RGB, 0.9)}</color><width>2</width></LineStyle>'
                f'<PolyStyle><color>{_kml_colour(BOX_RGB, 0.08)}</color></PolyStyle></Style>'
                '<Polygon><outerBoundaryIs><LinearRing><coordinates>'
                f'{_coords(ring)}</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>')
        out.append('</Folder>')
    # -- references ------------------------------------------------------------------
    if pic.references:
        out.append('<Folder><name>References</name>')
        for r in pic.references:
            tick = "✓ " if r.verified else ""
            out.append(
                f'<Placemark><name>{_x(tick + (r.name or r.key))}</name>'
                f'<Style><IconStyle><color>{_kml_colour(REF_RGB, 0.9)}</color><scale>0.7</scale>'
                '<Icon><href>http://maps.google.com/mapfiles/kml/shapes/triangle.png</href></Icon>'
                '</IconStyle></Style>'
                + _extended({"key": r.key, "verified": r.verified,
                             "residual_deg": r.residual_deg})
                + f'<Point><coordinates>{r.lon:.6f},{r.lat:.6f},0</coordinates></Point></Placemark>')
        out.append('</Folder>')
    out.append('</Document></kml>')
    return "\n".join(out) + "\n"


def _track_kml(t: Track, pic: Picture, now: float) -> str:
    age = now - t.last_heard if t.last_heard else 0.0
    alpha = fade_alpha(age, pic.fade_s, floor=0.35)
    rgb = STATE_RGB.get(t.state, STATE_RGB["unknown"])
    name = t.label
    if t.state == "silent":
        name = f"⚠ {name} — silent {_ago(age, short=True)}"
    desc = _balloon(t, now)
    parts = [f'<Placemark><name>{_x(name)}</name>',
             f'<description><![CDATA[{desc}]]></description>']
    if t.last_heard:
        parts.append(f'<TimeStamp><when>{_iso(t.last_heard)}</when></TimeStamp>')
    parts.append(
        f'<Style><IconStyle><color>{_kml_colour(rgb, alpha)}</color>'
        f'<scale>{1.3 if t.priority or t.state == "silent" else 1.0}</scale>'
        '<Icon><href>http://maps.google.com/mapfiles/kml/paddle/wht-blank.png</href></Icon>'
        '</IconStyle>'
        f'<LineStyle><color>{_kml_colour(rgb, min(1.0, alpha + 0.2))}</color><width>2</width></LineStyle>'
        f'<PolyStyle><color>{_kml_colour(rgb, 0.18 * alpha)}</color></PolyStyle>'
        f'<LabelStyle><scale>{1.1 if t.name else 0.8}</scale></LabelStyle></Style>')
    parts.append(_extended({
        "key": t.key, "name": t.name, "role": t.role, "team": t.team, "state": t.state,
        "protocol": t.protocol, "fc_mhz": round(t.fc_hz / 1e6, 4) if t.fc_hz else "",
        "last_heard_utc": _iso(t.last_heard) if t.last_heard else "",
        "bearings": t.n_bearings, "ellipse_m": round(2 * t.semi_major_m) if t.located else "",
        "containment": t.containment if t.located else "", "trend": t.trend,
        "motion": t.motion_state, "cal_tier": t.cal_tier, "in_box": t.in_box,
        "priority": t.priority}))
    geoms = []
    if t.located:
        geoms.append(f'<Point><coordinates>{t.lon:.6f},{t.lat:.6f},0</coordinates></Point>')
        ring = ellipse_ring(t.lat, t.lon, t.semi_major_m, t.semi_minor_m, t.orientation_deg)
        geoms.append('<Polygon><tessellate>1</tessellate><outerBoundaryIs><LinearRing><coordinates>'
                     f'{_coords(ring)}</coordinates></LinearRing></outerBoundaryIs></Polygon>')
    elif t.last_lob is not None and t.state != "heard_via_repeater":
        ring = wedge_ring(t.last_lob)
        geoms.append('<Polygon><tessellate>1</tessellate><outerBoundaryIs><LinearRing><coordinates>'
                     f'{_coords(ring)}</coordinates></LinearRing></outerBoundaryIs></Polygon>')
        # the label sits a third of the way out along the bearing
        mid = geo.direct(t.last_lob.lat, t.last_lob.lon, t.last_lob.bearing_deg,
                         t.last_lob.range_m / 3)
        geoms.append(f'<Point><coordinates>{mid[1]:.6f},{mid[0]:.6f},0</coordinates></Point>')
    if not geoms:
        return ""           # heard via a repeater and nothing else: nothing to draw
    parts.append('<MultiGeometry>' + "".join(geoms) + '</MultiGeometry></Placemark>')
    return "".join(parts)


def _balloon(t: Track, now: float) -> str:
    """The sentence first; the RF underneath a rule."""
    rows = [("Protocol", t.protocol or "—"), ("Key", t.key),
            ("Channel", f"{t.fc_hz/1e6:.4f} MHz" if t.fc_hz else "—"),
            ("Bearings", str(t.n_bearings)),
            ("Ellipse", f"{2*t.semi_major_m:.0f} × {2*t.semi_minor_m:.0f} m at "
                        f"{int(t.containment*100)} %" if t.located else "—"),
            ("Motion", t.motion_state), ("Calibration", t.cal_tier or "—"),
            ("First heard", _iso(t.first_heard) if t.first_heard else "—"),
            ("Last heard", _iso(t.last_heard) if t.last_heard else "—")]
    table = "".join(f"<tr><td style='color:#888;padding-right:8px'>{html.escape(k)}</td>"
                    f"<td>{html.escape(str(v))}</td></tr>" for k, v in rows)
    photo = (f'<img src="{html.escape(t.photo)}" style="max-width:160px;float:right;'
             f'margin-left:8px"/>' if t.photo else "")
    who = html.escape(t.label) + (f" · team {html.escape(t.team)}" if t.team else "")
    return (f"{photo}<p style='font-size:1.1em'><b>{who}</b></p>"
            f"<p>{html.escape(t.sentence or plain_sentence(t, now))}</p>"
            f"<hr/><table style='font-size:0.85em'>{table}</table>")


def _doc_description(pic: Picture, now: float) -> str:
    n_loc = sum(1 for t in pic.tracks if t.state == "located")
    n_sil = sum(1 for t in pic.tracks if t.state == "silent")
    return (f"{pic.site_id} · {_iso(now)} · {len(pic.tracks)} emitters, {n_loc} located, "
            f"{n_sil} silent · {len(pic.lobs)} bearings in the last {int(pic.keep_s//60)} min · "
            f"policy: {pic.policy}" + ("".join(f" · {n}" for n in pic.notices)))


def _extended(fields: dict) -> str:
    items = "".join(f'<Data name="{_x(str(k))}"><value>{_x(str(v))}</value></Data>'
                    for k, v in fields.items() if v not in (None, ""))
    return f"<ExtendedData>{items}</ExtendedData>" if items else ""


# ---------------------------------------------------------------------------
# GeoJSON
# ---------------------------------------------------------------------------

def to_geojson(pic: Picture, *, now: float | None = None) -> str:
    """The same picture with the numbers as properties — for web maps and
    for a second ATK that wants σ, tier and bearing count as fields rather
    than as polygons."""
    now = pic.t_utc if now is None else now
    feats = []
    v = pic.vehicle
    if v is not None:
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [v.lon, v.lat]},
                      "properties": {"kind": "vehicle", "callsign": v.callsign, "site": v.site_id,
                                     "heading_deg": v.heading_deg, "t_utc": v.t_utc,
                                     "status": v.status}})
    for l in pic.lobs:
        age = now - l.t_utc
        if age > pic.keep_s:
            continue
        end = geo.direct(l.lat, l.lon, l.bearing_deg, l.range_m)
        feats.append({"type": "Feature",
                      "geometry": {"type": "LineString",
                                   "coordinates": [[l.lon, l.lat], [end[1], end[0]]]},
                      "properties": {"kind": "lob", "key": l.key, "label": l.label,
                                     "bearing_deg": l.bearing_deg, "sigma_deg": l.sigma_deg,
                                     "t_utc": l.t_utc, "age_s": age,
                                     "alpha": fade_alpha(age, pic.fade_s), "site": l.site_id}})
    for t in pic.tracks:
        props = {"kind": "track", "key": t.key, "label": t.label, "name": t.name, "role": t.role,
                 "team": t.team, "state": t.state, "priority": t.priority,
                 "sentence": t.sentence, "protocol": t.protocol, "fc_hz": t.fc_hz,
                 "n_bearings": t.n_bearings, "first_heard": t.first_heard,
                 "last_heard": t.last_heard, "motion": t.motion_state, "cal_tier": t.cal_tier,
                 "in_box": t.in_box, "trend": t.trend, "containment": t.containment,
                 "semi_major_m": t.semi_major_m, "semi_minor_m": t.semi_minor_m,
                 "orientation_deg": t.orientation_deg, "sigma_m": t.sigma_m}
        if t.located:
            ring = ellipse_ring(t.lat, t.lon, t.semi_major_m, t.semi_minor_m, t.orientation_deg)
            geom = {"type": "GeometryCollection", "geometries": [
                {"type": "Point", "coordinates": [t.lon, t.lat]},
                {"type": "Polygon", "coordinates": [[[lon, lat] for lat, lon in ring]]}]}
        elif t.last_lob is not None:
            ring = wedge_ring(t.last_lob)
            geom = {"type": "Polygon", "coordinates": [[[lon, lat] for lat, lon in ring]]}
        else:
            geom = None
        feats.append({"type": "Feature", "geometry": geom, "properties": props})
    for b in pic.boxes:
        ring = list(b.polygon) + ([b.polygon[0]] if b.polygon and b.polygon[0] != b.polygon[-1] else [])
        feats.append({"type": "Feature",
                      "geometry": {"type": "Polygon", "coordinates": [[[lon, lat] for lat, lon in ring]]},
                      "properties": {"kind": "box", "name": b.name, "inverted": b.inverted}})
    for r in pic.references:
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [r.lon, r.lat]},
                      "properties": {"kind": "reference", "key": r.key, "name": r.name,
                                     "verified": r.verified, "residual_deg": r.residual_deg}})
    doc = {"type": "FeatureCollection", "features": feats,
           "properties": {"site": pic.site_id, "t_utc": now, "policy": pic.policy,
                          "fade_s": pic.fade_s, "keep_s": pic.keep_s, "sequence": pic.sequence,
                          "notices": list(pic.notices)}}
    return json.dumps(doc, separators=(",", ":"), default=_jsonable)


# ---------------------------------------------------------------------------
# the record feed (merge between ATKs)
# ---------------------------------------------------------------------------

def records_since(log_path, offset: int = 0, max_bytes: int = 4_000_000) -> tuple:
    """New complete JSONL lines from the registry's log after byte `offset`:
    `(text, next_offset)`. A second ATK polls this and fuses what it gets
    (`Registry.from_logs` reads the same lines). A partial last line is
    left for next time."""
    try:
        with open(log_path, "rb") as f:
            f.seek(offset)
            data = f.read(max_bytes)
    except FileNotFoundError:
        return "", offset
    if not data:
        return "", offset
    cut = data.rfind(b"\n")
    if cut < 0:
        return "", offset
    chunk = data[:cut + 1]
    return chunk.decode("utf-8", errors="replace"), offset + len(chunk)


# ---------------------------------------------------------------------------
# Cursor-on-Target
# ---------------------------------------------------------------------------

#: CoT types used, in one place so a change after testing on a real ATAK is
#: one edit. Friendly ground: the vehicle; the tracks as "other" ground
#: units unless the fleet map says friendly; a range-and-bearing line for a
#: LOB; a shape for a box.
COT_TYPES = {
    "vehicle": "a-f-G-E-V-C",
    "track": "a-f-G-U-C",          # a radio on the roster: friendly
    "unknown": "a-u-G",            # an emitter nobody has named
    "lob": "u-rb-a",
    "box": "u-d-f",
}


def to_cot(pic: Picture, *, now: float | None = None, stale_s: float | None = None,
           lob_stale_s: float | None = None, max_lobs_per_key: int = 3) -> list:
    """CoT events for a TAK device, one XML string each. UIDs are stable per
    key, so a radio is one object that moves. `stale` is the roster's
    silence threshold for a track (the marker fades when ATK greys it) and
    the fade time for a line."""
    now = pic.t_utc if now is None else now
    events = []
    v = pic.vehicle
    if v is not None:
        events.append(_cot_event(
            uid=f"atk.{pic.site_id}.vehicle", typ=COT_TYPES["vehicle"], t=now,
            stale=now + 30.0, lat=v.lat, lon=v.lon, ce=10.0,
            detail=(f'<contact callsign="{_x(v.callsign)}"/>'
                    f'<track course="{(v.heading_deg or 0.0):.1f}" speed="{v.speed_mps:.1f}"/>'
                    f'<remarks>{_x(v.status or "")}</remarks>')))
    for t in pic.tracks:
        if not (t.located or t.last_lob is not None):
            continue
        stale = (t.last_heard + (stale_s if stale_s is not None else t.silence_threshold_s)
                 if t.last_heard else now + 300.0)
        if t.located:
            lat, lon, ce = t.lat, t.lon, t.semi_major_m
        else:
            mid = geo.direct(t.last_lob.lat, t.last_lob.lon, t.last_lob.bearing_deg,
                             t.last_lob.range_m / 3)
            lat, lon, ce = mid[0], mid[1], t.last_lob.range_m
        typ = COT_TYPES["track"] if t.name else COT_TYPES["unknown"]
        callsign = t.label + (" — SILENT" if t.state == "silent" else "")
        remarks = t.sentence or plain_sentence(t, now)
        colour = "-65536" if t.state == "silent" else ("-16711936" if t.state == "located" else "-256")
        events.append(_cot_event(
            uid=f"atk.{pic.site_id}.{_uid(t.key)}", typ=typ, t=t.last_heard or now,
            stale=stale, lat=lat, lon=lon, ce=ce,
            detail=(f'<contact callsign="{_x(callsign)}"/>'
                    f'<remarks>{_x(remarks)}</remarks>'
                    f'<color argb="{colour}"/>'
                    f'<__atk key="{_x(t.key)}" state="{t.state}" bearings="{t.n_bearings}" '
                    f'ellipse_m="{2*t.semi_major_m:.0f}" containment="{t.containment}" '
                    f'trend="{_x(t.trend)}" tier="{_x(t.cal_tier)}" motion="{t.motion_state}"/>')))
    # lines: the last few per key, each stale at the fade time
    per_key: dict = {}
    for l in sorted(pic.lobs, key=lambda l: -l.t_utc):
        if now - l.t_utc > pic.keep_s:
            continue
        lst = per_key.setdefault(l.key, [])
        if len(lst) >= max_lobs_per_key:
            continue
        lst.append(l)
    for key, lobs in per_key.items():
        for i, l in enumerate(lobs):
            events.append(_cot_event(
                uid=f"atk.{pic.site_id}.{_uid(key)}.lob{i}", typ=COT_TYPES["lob"], t=l.t_utc,
                stale=l.t_utc + (lob_stale_s if lob_stale_s is not None else pic.fade_s),
                lat=l.lat, lon=l.lon, ce=9999999.0,
                detail=(f'<range value="{l.range_m:.0f}"/><bearing value="{l.bearing_deg:.1f}"/>'
                        '<inclination value="0"/><rangeUnits value="1"/>'
                        '<bearingUnits value="0"/><northRef value="1"/>'
                        f'<contact callsign="{_x(l.label or key)} LOB"/>'
                        f'<remarks>σ {l.sigma_deg:.1f}°</remarks>')))
    return events


def _cot_event(*, uid: str, typ: str, t: float, stale: float, lat: float, lon: float,
               ce: float, detail: str, how: str = "m-g") -> str:
    return (f'<?xml version="1.0" encoding="UTF-8"?>'
            f'<event version="2.0" uid="{_x(uid)}" type="{typ}" time="{_iso(t)}" '
            f'start="{_iso(t)}" stale="{_iso(stale)}" how="{how}">'
            f'<point lat="{lat:.6f}" lon="{lon:.6f}" hae="9999999.0" ce="{ce:.1f}" le="9999999.0"/>'
            f'<detail>{detail}</detail></event>')


def _uid(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", key)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _k(p: float) -> float:
    """k·σ for a 2-D Gaussian containment p."""
    p = min(max(p, 0.01), 0.999)
    return math.sqrt(-2.0 * math.log(1.0 - p))


def _ago(seconds: float, short: bool = False) -> str:
    s = max(0, int(seconds))
    if s < 60:
        return f"{s} s" if short else f"{s} seconds ago"
    m = s // 60
    if m < 60:
        return f"{m} min" if short else f"{m} minute{'s' * (m != 1)} ago"
    h = m // 60
    return f"{h} h {m % 60} min" if short else f"{h} hour{'s' * (h != 1)} {m % 60} min ago"


def _metres(m: float) -> str:
    if m < 1000:
        return f"{int(round(m / 10.0) * 10)} m"
    return f"{m / 1000:.1f} km"


def _jsonable(v):
    """json.dumps default: numpy scalars and arrays become plain numbers."""
    try:
        import numpy as np
    except ImportError:                                      # pragma: no cover
        np = None
    if np is not None:
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
    raise TypeError(f"not JSON: {type(v).__name__}")
