"""The ports: everything the engine needs from the outside world, as small
protocols. ATK implements them against its real subsystems; tests implement
fakes; the simulator (`sim.py`) implements all of them. The engine imports
none of those -- it takes the ports it needs as arguments (dependency
injection), which is what lets the whole thing run on numpy alone against
fakes, before the KrakenSDR exists and in every session after.

A Protocol here is a shape, not a base class: anything with the right methods
satisfies it, so ATK's objects and the fakes need not inherit anything. The
dataclasses (Fix2D, BurstEvent, Profile, Edge, Reference) ARE shared, because
they are the vocabulary the two sides speak.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Protocol, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# the values that cross the boundary
# ---------------------------------------------------------------------------

@dataclass
class Fix2D:
    """Where the receiver was, and how sure. `course_deg` is direction of
    travel (GPS course over ground); `sigma_m` the position uncertainty;
    `fix_age_s` how old the underlying GPS fix is (a stale fix is a wide
    one, not a wrong one)."""

    lat: float
    lon: float
    t_utc: float
    course_deg: float | None = None
    speed_mps: float = 0.0
    alt_m: float | None = None
    fix_age_s: float = 0.0
    sigma_m: float = 10.0
    #: The ARRAY's forward heading (degrees true) when something other than
    #: the GPS course knows it — a tripod set by compass, a heading aligned
    #: on a reference, a vehicle's mounting offset applied. When set, the
    #: engine uses it (with `heading_sigma_deg`) instead of `course_deg`,
    #: whose sigma is "undefined when stopped".
    heading_deg: float | None = None
    heading_sigma_deg: float | None = None


@dataclass
class BurstEvent:
    """A decoder said a known emitter transmitted. `t_utc`/`span_s` locate it
    in time; `n0`/`n1` (if the decoder is sample-exact) locate it in the
    stream; `segment` ties those indices to a tuning. Identity is
    (protocol, system, id) with `slot` and `sync_type` where the protocol has
    them."""

    protocol: str
    system: str
    id: str
    t_utc: float
    span_s: float = 0.0
    n0: int | None = None
    n1: int | None = None
    segment: int | None = None
    slot: int | None = None
    sync_type: str = ""
    fc_hz: float = 0.0
    bw_hz: float = 0.0


@dataclass
class Detection:
    """The LISTENER heard energy (ATK's cued-DF plan, §5.2). A detection is
    the primary object of cued DF; an identity is an optional field on it,
    because most RF energy carries none.

    `t0` is when it began; `t1` is None while it is still going. `key` is
    the emitter key when a decoder supplied one (``dmr:1:473``), else a
    signature key (``sig:…``, see `scheduler.signature_key`), else empty —
    the scheduler falls back to the frequency. `grant` marks a trunking
    grant: a look-ahead cue naming the traffic channel BEFORE the voice
    starts. `signature` carries what the detector measured about the
    burst's shape (bw_hz, rise_ms, duration_s, periodicity_s,
    modulation_hint) for emitters with no identity.
    """

    t0: float
    fc_hz: float
    bw_hz: float = 12_500.0
    t1: float | None = None
    level_db: float = 0.0
    snr_db: float = 0.0
    key: str = ""
    protocol: str = ""
    grant: bool = False
    signature: dict = field(default_factory=dict)
    source: str = "listener"

    @property
    def open(self) -> bool:
        return self.t1 is None

    def is_open(self, now: float) -> bool:
        """Still transmitting at `now` (a scripted cue may carry its end
        time from the start; it is open until then)."""
        return self.t1 is None or self.t1 > now

    def duration(self, now: float) -> float:
        """How long it has been going AS OF `now` — never credits a scripted
        end time that has not happened yet."""
        end = now if self.t1 is None else min(self.t1, now)
        return max(0.0, end - self.t0)


@dataclass
class Profile:
    """A terrain profile between two points: distance and height along it,
    and which DEM resolution answered (so downstream confidence can carry
    it)."""

    dist_m: np.ndarray
    height_m: np.ndarray
    resolution_m: float = 30.0


@dataclass
class Edge:
    """A road-graph edge: an id, its endpoints, and a length. `neighbours`
    are the ids reachable from its far end."""

    id: str
    a: tuple
    b: tuple
    length_m: float = 0.0


@dataclass
class Reference:
    """A catalogued fixed emitter: an LTE PCI/eNB, a DMR/P25 repeater.
    `key` matches an emitter key when the reference is heard; `position_sigma_m`
    and `verified` say how much to trust its coordinates."""

    key: str
    lat: float
    lon: float
    kind: str = ""
    name: str = ""
    fc_hz: float = 0.0
    position_sigma_m: float = 50.0
    verified: bool = False
    source: str = ""


# ---------------------------------------------------------------------------
# the protocols
# ---------------------------------------------------------------------------

@runtime_checkable
class FrameSource(Protocol):
    """Coherent multi-channel frames: atkdaq.exe, a .atkq file, or the
    simulator. `frames()` yields atkq.Frame; `control()` tunes/cals/etc. (a
    no-op for a file)."""

    def frames(self) -> Iterator: ...
    def control(self, cmd: str) -> None: ...


@runtime_checkable
class CueSource(Protocol):
    """Detections from the listening receiver (ATK's bladeRF CFAR detector
    with its decoder taps), or the simulator's scripted cues. The scheduler
    consumes them; nothing waits for a decode."""

    def cues(self) -> Iterator[Detection]: ...


@runtime_checkable
class DecoderEvents(Protocol):
    """Burst spans with identity: ATK's DSD tap, dmr_sync, or the
    simulator's side channel."""

    def events(self) -> Iterator[BurstEvent]: ...


@runtime_checkable
class Positioning(Protocol):
    """Where the receiver is at a UTC instant. ATK's interpolating gps_track,
    or a simulator track. Everything is matched by time, never by 'latest'."""

    def at(self, t_utc: float) -> Fix2D: ...


@runtime_checkable
class Elevation(Protocol):
    """Terrain heights. ATK's elevation service, or a flat-earth fake."""

    def profile(self, a: tuple, b: tuple, step_m: float = 30.0) -> Profile: ...


@runtime_checkable
class RoadGraph(Protocol):
    """The offline road graph a vehicle posterior is constrained to."""

    def nearest_edge(self, lat: float, lon: float) -> "Edge | None": ...
    def neighbours(self, edge: Edge) -> list: ...
    def route_time_s(self, a: tuple, b: tuple) -> float: ...


@runtime_checkable
class Catalogue(Protocol):
    """Reference emitters, looked up by key or by area."""

    def lookup(self, key: str) -> "Reference | None": ...
    def near(self, lat: float, lon: float, radius_m: float) -> list: ...


@runtime_checkable
class PathLoss(Protocol):
    """Propagation loss over a profile: atkdsp's ITM, or free space."""

    def loss_db(self, profile: Profile, f_hz: float, h_tx: float, h_rx: float) -> float: ...
