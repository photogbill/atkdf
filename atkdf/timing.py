"""Timing tiers, the per-site timing state machine, and what may travel
between sites (plan §11.4.6).

THE TIERS (`arrayprofile.TIER_WORDS` has them in words):

  T0  none (Kraken over USB, ms-class)          → AOA fusion only — today
  T1  host PPS, software timestamps (µs-class)  → better time tags; no TDOA
  T2  GPSDO per receiver (±50 ns RMS class)     → wideband TDOA; for 12.5 kHz
                                                  P25 a km-class hyperbola per
                                                  snapshot (c/4B ≈ 6 km) that
                                                  breaks sidedness/collinearity,
                                                  not an ellipse tightener
  T3  cabled 10 MHz + PPS on one platform       → sample-aligned devices; LO
                                                  phase still random per device
  T4  shared / distributed LO                   → the only tier at which several
                                                  receivers are ONE array
  T5  White Rabbit over fibre (fixed sites)     → cross-site coherence, VHF only

THE STATE MACHINE. A tier is what the hardware CAN do; the state is what it
is doing now. A GPSDO that has lost GPS is in holdover and its time error
grows: ≈ 1.9 ns/s on an X3x0 OCXO (< 20 µs over 3 h), unspecified on a B210
TCXO (taken here as its 75 ppb frequency accuracy: 75 ns/s), unmeasured on
N3xx (taken as the X3x0 figure, flagged) (lit2 §6.2, §6.11). `sigma_ns()`
is the 1-σ time error now; TDOA is offered only while it is small enough.

WHAT TRAVELS (lit2 §6.1, §6.4):

  A  bearing records — always (tens of kbit/s; ATK's share server /records)
  B  short timestamped CHANNELISED I/Q snapshots — only when cued AND both
     sites can time-tag them (≥ T2, locked or briefly in holdover):
     ≈ 0.8 Mbit/s per element at 25 kS/s sc16, 0.4 at sc8 — for cross-site
     correlation, calibration-on-opportunity, listen-on-array
  C  raw wideband I/Q — never over the network (4 × 25 MS/s ≈ 3.2 Gbit/s)

One host owns one UHD multi_usrp (up to 16 synchronised devices); separate
hosts coordinate above UHD — in ATK's peers layer — with these rules.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

log = logging.getLogger("atkdf.timing")

TIERS = ("T0", "T1", "T2", "T3", "T4", "T5")
C_M_PER_NS = 0.299792458

#: 1-σ time error when locked, ns, by tier or holdover spec
LOCKED_SIGMA_NS = {"T0": 1e6, "T1": 2e3, "T2": 50.0, "T3": 50.0, "T4": 50.0, "T5": 1.0}
#: holdover drift, ns per second, and whether the number is a measured spec
HOLDOVER = {
    "x3x0_ocxo": (20_000.0 / (3 * 3600.0), True),     # < 20 µs over 3 h
    "b2x0_tcxo": (75.0, False),                        # 75 ppb, holdover unspecified
    "n3xx_gpsdo": (20_000.0 / (3 * 3600.0), False),    # unmeasured: X3x0 figure assumed
    "": (75.0, False),
}
#: TDOA is offered while the time error is under this (300 m of range difference)
TDOA_MAX_SIGMA_NS = 1000.0

PAYLOADS = ("A", "B", "C")
PAYLOAD_WORDS = {"A": "bearing records", "B": "channelised I/Q snapshots (cued)",
                 "C": "raw wideband I/Q"}
BYTES_PER_COMPLEX = {"sc8": 2, "sc16": 4, "cf32": 8}


def tier_index(tier: str) -> int:
    if tier not in TIERS:
        raise ValueError(f"timing tier {tier!r} is not one of {', '.join(TIERS)}")
    return TIERS.index(tier)


@dataclass
class TimingState:
    """One site's time source, now. Feed it what UHD's sensors say
    (`gps_locked`, `gps_servo`, `using_ref`) via `update`; count overruns
    ("O": the host too slow) and drops ("D": NIC/PCIe/MTU) as they come."""

    tier: str = "T0"
    holdover_spec: str = ""
    locked_sigma_ns: float | None = None
    state: str = "unknown"            # unknown | locked | holdover | free
    holdover_since: float | None = None
    gps_locked: bool | None = None
    servo_ok: bool | None = None
    using_ref: str = ""
    overruns: int = 0
    drops: int = 0
    transitions: list = field(default_factory=list)

    def __post_init__(self):
        tier_index(self.tier)
        if self.locked_sigma_ns is None:
            self.locked_sigma_ns = LOCKED_SIGMA_NS[self.tier]
        if self.tier in ("T0", "T1"):
            self.state = "free" if self.tier == "T0" else "locked"

    @classmethod
    def from_profile(cls, profile) -> "TimingState":
        t = dict(profile.timing or {})
        return cls(tier=t.get("tier", "T0"), holdover_spec=t.get("holdover_spec", "") or "",
                   locked_sigma_ns=t.get("t_utc_sigma_ns"))

    def _go(self, new: str, now: float) -> None:
        if new != self.state:
            self.transitions.append((now, self.state, new))
            log.info("timing %s: %s → %s", self.tier, self.state, new)
            self.state = new

    def update(self, gps_locked: bool, servo_ok: bool | None = None, using_ref: str = "",
               now: float | None = None) -> str:
        now = time.time() if now is None else float(now)
        self.gps_locked, self.servo_ok, self.using_ref = bool(gps_locked), servo_ok, using_ref
        if self.tier in ("T0", "T1"):
            return self.state
        good = bool(gps_locked) and servo_ok is not False
        if good:
            self.holdover_since = None
            self._go("locked", now)
        elif self.state in ("locked", "holdover"):
            if self.holdover_since is None:
                self.holdover_since = now
            self._go("holdover", now)
        else:
            self._go("free", now)
        return self.state

    def note_overrun(self, n: int = 1) -> None:
        self.overruns += int(n)

    def note_drop(self, n: int = 1) -> None:
        self.drops += int(n)

    def holdover_s(self, now: float | None = None) -> float:
        if self.holdover_since is None:
            return 0.0
        now = time.time() if now is None else float(now)
        return max(0.0, now - self.holdover_since)

    def sigma_ns(self, now: float | None = None) -> float:
        """1-σ time error now: the locked figure, plus holdover drift."""
        if self.state == "free" and self.tier not in ("T0", "T1"):
            return LOCKED_SIGMA_NS["T1"]           # a GPSDO that never locked: host time
        base = float(self.locked_sigma_ns)
        if self.state != "holdover":
            return base
        rate, _measured = HOLDOVER.get(self.holdover_spec, HOLDOVER[""])
        return math.hypot(base, rate * self.holdover_s(now))

    def tdoa_ok(self, now: float | None = None, max_sigma_ns: float = TDOA_MAX_SIGMA_NS) -> bool:
        return (tier_index(self.tier) >= tier_index("T2")
                and self.state in ("locked", "holdover")
                and self.sigma_ns(now) <= max_sigma_ns)

    def health(self, now: float | None = None) -> dict:
        """What a site publishes on the peers layer (§11.4.6 Health)."""
        rate, measured = HOLDOVER.get(self.holdover_spec, HOLDOVER[""])
        return {"tier": self.tier, "state": self.state,
                "sigma_ns": round(self.sigma_ns(now), 1),
                "holdover_s": round(self.holdover_s(now), 1),
                "holdover_spec": self.holdover_spec or "unknown",
                "holdover_rate_measured": measured,
                "gps_locked": self.gps_locked, "servo_ok": self.servo_ok,
                "using_ref": self.using_ref, "overruns": self.overruns, "drops": self.drops,
                "tdoa_ok": self.tdoa_ok(now)}

    def sentence(self, now: float | None = None) -> str:
        s = f"timing {self.tier}, {self.state}, ±{self.sigma_ns(now):.0f} ns"
        if self.state == "holdover":
            s += f" ({self.holdover_s(now):.0f} s in holdover)"
        if self.overruns or self.drops:
            s += f"; {self.overruns} overrun(s), {self.drops} drop(s)"
        return s


def allowed_payloads(local: TimingState, peer: TimingState | dict | None,
                     cued: bool = False, now: float | None = None) -> tuple:
    """({payload kinds allowed}, reason). A is always allowed; B only when
    cued and both ends can time-tag; C never."""
    allowed = {"A"}
    if not cued:
        return allowed, "bearing records only (no cue: no I/Q leaves the site)"
    if peer is None:
        return allowed, "bearing records only (peer timing unknown)"
    peer_ok = peer.tdoa_ok(now) if isinstance(peer, TimingState) else bool(peer.get("tdoa_ok"))
    if not local.tdoa_ok(now):
        return allowed, f"bearing records only (this site: {local.sentence(now)})"
    if not peer_ok:
        return allowed, "bearing records only (the peer cannot time-tag I/Q)"
    allowed.add("B")
    return allowed, "bearing records + cued channelised I/Q snapshots (both sites ≥ T2)"


def payload_bps(kind: str, n_elements: int = 1, fs_hz: float = 25_000.0,
                fmt: str = "sc16", records_per_s: float = 10.0) -> float:
    """Bits per second for a payload kind: A ≈ 2 kbit per record; B = elements
    × rate × bytes; C the same at the wideband rate."""
    if kind == "A":
        return 2_000.0 * float(records_per_s)
    if kind not in ("B", "C"):
        raise ValueError(f"payload kind {kind!r} is not A, B or C")
    return float(n_elements) * float(fs_hz) * BYTES_PER_COMPLEX[fmt] * 8.0


def fits_link(kinds, link_bps: float, **kw) -> tuple:
    """(fits, total_bps) for a set of payload kinds on a link (Wi-Fi, LTE,
    a LoRa link — which carries A only, and only compressed)."""
    total = sum(payload_bps(k, **kw) for k in kinds)
    return total <= float(link_bps), total


def tdoa_range_sigma_m(bandwidth_hz: float, snr_db: float, integration_s: float = 0.0,
                       timing_sigma_ns: float = 0.0) -> float:
    """A planning number, not a measurement: single-snapshot TDOA range
    error ≈ c/(4B) (lit2 §6.6), improving as 1/√(time-bandwidth·SNR) with
    integration, plus the two sites' timing error in quadrature."""
    base = 299_792_458.0 / (4.0 * float(bandwidth_hz))
    if integration_s > 0:
        tb = max(1.0, float(bandwidth_hz) * float(integration_s))
        base = base / math.sqrt(tb * 10 ** (float(snr_db) / 10.0))
    t = math.sqrt(2.0) * float(timing_sigma_ns) * C_M_PER_NS
    return math.hypot(base, t)
