"""The data model: what a bearing is, what a calibration was, who an emitter is.

Everything the engine concludes is one of these, and every one of them can be
written to a line of JSON and read back unchanged — the log beside a clip is
the evidence, and "how good is this fix?" is answered by the records, not by
anyone's memory of the day (umbrella plan §3.5).

MERGE-READY FROM DAY ONE. Every record carries `site_id` and the identity of
the calibration it was measured under. A second vehicle's log merges by file,
later, with no schema migration — and two logs that both say "passport 1" are
still distinguishable, because a passport number restarts with every atkdaq
run and is always stored with the run (`session`) it belongs to.
"""

from __future__ import annotations

import base64
import json
import math
from dataclasses import asdict, dataclass, field, fields
from typing import NamedTuple

import numpy as np

#: Bearing likelihoods are stored at 1° resolution: 360 float16 values, the
#: pseudo-spectrum normalised to a density (sum × 1° = 1). The estimators work
#: finer (0.5°); the record keeps what fusion and the posterior need.
LIKELIHOOD_POINTS = 360

#: Where a bearing came from, in the terms that decide whether it may be fused
#: as the emitter's position (umbrella plan §7.1).
PATH_TAGS = ("direct", "repeated", "unknown")

PLATFORM_MODES = ("static", "mobile")


class EmitterKey(NamedTuple):
    """Who transmitted: (protocol, system, id).

    `system` is the DMR network / colour code, the P25 system ID or NAC, or
    TETRA MCC-MNC — so the same numeric ID on two networks is two radios.
    An emitter with no identity has no key; the tracker gives it a
    `track_id` instead.
    """

    protocol: str
    system: str
    id: str

    def __str__(self) -> str:
        return f"{self.protocol}:{self.system}:{self.id}"

    @classmethod
    def parse(cls, text: str) -> "EmitterKey":
        parts = str(text).split(":", 2)
        if len(parts) != 3:
            raise ValueError(f"an emitter key is protocol:system:id, not {text!r}")
        return cls(*parts)


def encode_likelihood(lik) -> str:
    """360 float16, base64 — 960 characters instead of ~2 KB of decimals."""
    a = np.asarray(lik, dtype=np.float16).reshape(-1)
    if a.size != LIKELIHOOD_POINTS:
        raise ValueError(f"a likelihood has {LIKELIHOOD_POINTS} points, not {a.size}")
    return base64.b64encode(a.astype("<f2").tobytes()).decode("ascii")


def decode_likelihood(text: str) -> np.ndarray:
    a = np.frombuffer(base64.b64decode(text), dtype="<f2").astype(np.float32)
    if a.size != LIKELIHOOD_POINTS:
        raise ValueError(f"a stored likelihood has {a.size} points")
    return a


@dataclass
class Passport:
    """One calibration, and the conditions it was measured under.

    Built from the atkdaq frame header (`calibration.passport_of`): the
    receivers' integer delays, weights and fractional delays and the spread
    they left on the noise source; plus what the engine knows that the DAQ
    does not — the array geometry, the manifold, and the reference residual
    at the time (the radio compass's opinion of how true the array was).
    """

    session: int                    # atkdaq run, as UTC ns of stream start
    passport: int                   # atkdaq's counter within that run
    t_utc: float                    # when the frame that introduced it was taken
    fc_hz: float
    fs_hz: float
    gain_tenths_db: int
    delays: list
    weights: list                   # [re, im] pairs — JSON has no complex
    frac_delays: list
    spread_deg: float | None
    drops: list
    segment: int = 0
    device: str = "kraken"
    dither_off: bool = True         # atkdaq always turns it off; recorded, not assumed
    geometry_id: str = ""
    manifold_id: str = ""           # "" = ideal
    ref_residual_deg: float | None = None
    site_id: str = ""

    @property
    def uid(self) -> str:
        """Unique across runs and sites: site/session/passport."""
        return f"{self.site_id or 'site'}/{self.session}/{self.passport}"

    def weights_complex(self) -> np.ndarray:
        return np.array([complex(a, b) for a, b in self.weights])

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Passport":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class BearingRecord:
    """One bearing: one burst (or one window) by one estimator.

    theta_rel_deg is in the ARRAY frame: clockwise from the array's forward
    axis (element 0's direction; the vehicle's nose or the plate's marked
    north). theta_abs_deg = heading_deg + theta_rel_deg, a compass bearing,
    and sigma_deg already includes heading_sigma_deg — a bearing taken mid-
    turn is a wide one, not a discarded one.
    """

    t_utc: float
    theta_rel_deg: float
    sigma_deg: float
    estimator: str
    segment: int = 0
    sample_span: tuple = (0, 0)     # (n0, n1): channel-0 stream indices, end exclusive
    fc_hz: float = 0.0              # the signal's RF centre, not the tuner's
    bw_hz: float = 0.0
    emitter_key: str = ""           # str(EmitterKey) or ""
    track_id: str = ""
    slot: int | None = None
    sync_type: str = ""
    path_tag: str = "unknown"
    heading_deg: float | None = None
    heading_sigma_deg: float | None = None
    theta_abs_deg: float | None = None
    quality: dict = field(default_factory=dict)
    likelihood: np.ndarray | None = None      # LIKELIHOOD_POINTS, array frame, 1°
    lat: float | None = None
    lon: float | None = None
    alt_m: float | None = None
    fix_age_s: float | None = None
    platform_mode: str = "static"
    site_id: str = ""
    passport_uid: str = ""
    manifold: str = "ideal"
    notes: list = field(default_factory=list)
    #: The site's timing tier (T0 … T5, plan §11.4.6) and its 1-σ UTC error
    #: when the bearing was taken: what decides whether this bearing's
    #: timestamp may pair with another site's for TDOA. Old logs read as T0.
    timing_tier: str = "T0"
    t_utc_sigma_ns: float | None = None

    @property
    def key(self) -> EmitterKey | None:
        return EmitterKey.parse(self.emitter_key) if self.emitter_key else None

    @property
    def fusable(self) -> bool:
        """May this bearing locate its emitter? Not if it was heard via a
        repeater (it points at the repeater), not without an absolute
        bearing, and not without a position to draw it from."""
        return (self.path_tag != "repeated" and self.theta_abs_deg is not None
                and self.lat is not None and self.lon is not None)

    def absolute_likelihood(self) -> np.ndarray | None:
        """The likelihood rotated into compass bearings (index = degrees
        clockwise from true north), or None without a heading."""
        if self.likelihood is None or self.heading_deg is None:
            return None
        from .doa import rotate_likelihood
        return rotate_likelihood(np.asarray(self.likelihood, dtype=float), self.heading_deg)

    def to_json(self) -> str:
        d = asdict(self)
        d["likelihood"] = (None if self.likelihood is None
                           else encode_likelihood(self.likelihood))
        d["sample_span"] = list(self.sample_span)
        return json.dumps(d, separators=(",", ":"), default=_jsonable)

    @classmethod
    def from_json(cls, line: str) -> "BearingRecord":
        d = json.loads(line)
        if d.get("likelihood"):
            d["likelihood"] = decode_likelihood(d["likelihood"])
        d["sample_span"] = tuple(d.get("sample_span") or (0, 0))
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


    def to_lob(self, position_sigma_m: float = 0.0):
        """This bearing as a geodesy.LOB for fusion — only if it is fusable
        (absolute bearing, a position, and not heard via a repeater). None
        otherwise, so a caller cannot fuse a bearing that points at a
        repeater or has no origin."""
        if not self.fusable:
            return None
        from .geodesy import LOB
        return LOB(lat=float(self.lat), lon=float(self.lon),
                   bearing_deg=float(self.theta_abs_deg), sigma_deg=float(self.sigma_deg),
                   epoch=float(self.t_utc), source=self.site_id or "site",
                   method=self.estimator, quality=float(self.quality.get("confidence", 1.0)),
                   position_sigma_m=float(position_sigma_m),
                   evidence={"emitter": self.emitter_key or self.track_id,
                             "passport": self.passport_uid})


@dataclass
class Emitter:
    """A registry entry: who a radio is, and everything heard from it.

    Keyed by (protocol, system, id), or by a tracker's `track_id` when there
    is no identity. `aliases` carries the fleet-map name/role/team so an alert
    reads 'Medic 2 - Rodriguez', not '473'. The bearing history is kept in
    full; the posterior and the current fix are the engine's summary of it.
    """

    key: str                              # str(EmitterKey) or "track:<id>"
    name: str = ""
    role: str = ""
    team: str = ""
    protocol: str = ""
    system: str = ""
    first_heard: float = 0.0
    last_heard: float = 0.0
    n_bearings: int = 0
    last_bearing: BearingRecord | None = None
    motion_state: str = "unknown"
    motion_confidence: float = 0.0
    path_tags: dict = field(default_factory=dict)     # tag -> count
    slots: set = field(default_factory=set)
    notes: list = field(default_factory=list)

    def observe(self, rec: BearingRecord) -> None:
        if self.first_heard == 0.0:
            self.first_heard = rec.t_utc
        self.last_heard = max(self.last_heard, rec.t_utc)
        self.n_bearings += 1
        self.last_bearing = rec
        self.path_tags[rec.path_tag] = self.path_tags.get(rec.path_tag, 0) + 1
        if rec.slot is not None:
            self.slots.add(rec.slot)

    @property
    def label(self) -> str:
        """What an operator should read: the fleet name if known, else the key."""
        if self.name:
            return f"{self.name}" + (f" ({self.role})" if self.role else "")
        return self.key

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "last_bearing"}
        d["slots"] = sorted(self.slots)
        d["last_bearing"] = self.last_bearing.to_json() if self.last_bearing else None
        return d


def _jsonable(v):
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, complex):
        return [v.real, v.imag]
    raise TypeError(f"not JSON: {type(v).__name__}")


def wrap180(deg):
    """Angle difference to (−180, 180]."""
    return (np.asarray(deg, dtype=float) + 180.0) % 360.0 - 180.0 if np.ndim(deg) \
        else ((float(deg) + 180.0) % 360.0) - 180.0


def wrap360(deg):
    return np.asarray(deg, dtype=float) % 360.0 if np.ndim(deg) else float(deg) % 360.0


def circular_mean_deg(deg, weights=None) -> float:
    a = np.radians(np.asarray(deg, dtype=float))
    w = np.ones_like(a) if weights is None else np.asarray(weights, dtype=float)
    return float(np.degrees(math.atan2(float(np.sum(w * np.sin(a))),
                                       float(np.sum(w * np.cos(a))))) % 360.0)
