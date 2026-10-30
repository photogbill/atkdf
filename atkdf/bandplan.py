"""Which way a frequency points: subscriber (inbound) or repeater (outbound).

THE TRAP THIS EXISTS FOR. A repeater re-transmits a handset's burst with the
handset's identity on it, from the REPEATER's mast. A bearing on that outbound
burst points at the repeater; fused as the handset's position it puts the
handset on the hill where the repeater is (umbrella §7.1). DMR's sync word
says which side sent it (BS / MS / direct) and `identity.path_tag_of` already
uses it. **P25 Phase 1 does not**: inbound and outbound share one frame sync
(TIA-102.BAAA), so the evidence has to come from somewhere else — and the
cheapest, most certain evidence is the frequency itself.

Three sources of that evidence, most specific first:

  1. A site's IDEN table. A P25 trunked control channel broadcasts, per band
     identifier, the base frequency, the channel spacing and the subscriber
     TRANSMIT OFFSET (the IDEN_UP / IDEN_UP_VU / IDEN_UP_TDMA messages,
     TIA-102.AABC). Every channel number then resolves to an outbound and an
     inbound frequency, exactly. `IdenTable` holds those entries in
     engineering units; decoding the TSBK bit fields into them is the
     decoder's job (atkdsp's LMR_DECODE_PLAN), not this module's.
  2. A regional band plan. Where a band has ONE duplex split by rule — US
     800 MHz (806–824 subscriber / 851–869 base, −45 MHz), 700 MHz public
     safety narrowband (799–805 / 769–775, +30 MHz), 900 MHz (896–901 /
     935–940, −39 MHz) — the frequency alone decides. VHF and UHF have no
     single split (pairs vary by licence), so there the plan says "unknown"
     and says why, rather than guessing.
  3. The catalogue (`references`), for a known repeater output on a band with
     no rule. `identity.path_tag_of` consults it.

TALK-AROUND. Some channels in an outbound range are ALSO used handset-to-
handset on the base frequency (US 800 MHz mutual-aid 8CALL90 / 8TAC91–94).
On those the frequency cannot decide, and the plan says "unknown" — a bearing
that will be fused, carrying a note — never "repeated", which would refuse a
real handset bearing, and never "direct" with false confidence.

REGION. A band plan is a regional fact. A US rule applied abroad is not
merely useless, it is dangerous: a band that is subscriber-transmit in the US
may be base-transmit elsewhere, and a repeater's bearing would then be fused
as a handset's. So a plan is built for a named region (`for_region`), an
unknown region gets an EMPTY plan that answers "unknown" everywhere, and the
region travels in every answer.

What a classification is used for: `identity.path_tag_of` (and the engine
through it) turns ``inbound`` into ``direct`` and ``outbound`` into
``repeated``; ``talkaround`` and ``unknown`` stay ``unknown``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger("atkdf.bandplan")

#: Channel-centre tolerance when matching a frequency to a channel: half the
#: narrowest P25 / NXDN raster (6.25 kHz), so a channel is never confused with
#: its neighbour, while a handset's ±1.5 ppm offset (±1.3 kHz at 860 MHz)
#: still matches.
DEFAULT_TOL_HZ = 3_125.0

DIRECTIONS = ("inbound", "outbound", "talkaround", "unknown")


@dataclass(frozen=True)
class Classification:
    """The answer, with the reason in words (it goes on the bearing)."""

    direction: str                 # inbound | outbound | talkaround | unknown
    source: str = ""               # iden | bandplan | catalogue | ""
    band: str = ""
    why: str = ""
    region: str = ""

    @property
    def path_tag(self) -> str:
        """direct / repeated / unknown, in `records.PATH_TAGS` terms."""
        return {"inbound": "direct", "outbound": "repeated"}.get(self.direction, "unknown")


@dataclass(frozen=True)
class SplitBand:
    """A band with ONE duplex rule: base (outbound) transmits in `outbound`,
    subscribers (inbound) in `inbound`, and inbound = outbound + `offset_hz`.
    `talkaround_hz` lists outbound channels also used simplex on the base
    frequency."""

    name: str
    outbound: tuple
    inbound: tuple
    offset_hz: float
    talkaround_hz: tuple = ()
    note: str = ""

    def contains(self, f_hz: float) -> str:
        lo, hi = self.outbound
        if lo <= f_hz <= hi:
            return "outbound"
        lo, hi = self.inbound
        if lo <= f_hz <= hi:
            return "inbound"
        return ""


@dataclass(frozen=True)
class IdenEntry:
    """One band identifier from a P25 site's IDEN_UP / IDEN_UP_VU /
    IDEN_UP_TDMA, in engineering units.

    `base_hz` is channel 0's OUTBOUND (base transmit) frequency;
    `spacing_hz` the channel step; `tx_offset_hz` the subscriber transmit
    offset, so inbound = outbound + tx_offset_hz (negative at 800 MHz);
    `slots` 1 for FDMA, 2 for Phase 2 TDMA (a channel number then names a
    carrier AND a slot: carrier = channel // slots, slot = channel % slots).
    """

    iden: int
    base_hz: float
    spacing_hz: float
    tx_offset_hz: float
    bw_hz: float = 12_500.0
    slots: int = 1

    def __post_init__(self):
        if not 0 <= int(self.iden) <= 15:
            raise ValueError(f"a P25 band identifier is 0..15, not {self.iden}")
        if self.spacing_hz <= 0:
            raise ValueError("channel spacing must be > 0 Hz")
        if int(self.slots) not in (1, 2):
            raise ValueError("slots is 1 (FDMA) or 2 (Phase 2 TDMA)")


class IdenTable:
    """A site's band identifiers. `channel(ch16)` resolves a 16-bit P25
    channel number (4-bit identifier, 12-bit channel) to its outbound and
    inbound frequencies; `classify(f)` answers which side a frequency is."""

    def __init__(self, entries=(), name: str = ""):
        self.name = name
        self._by_iden: dict = {}
        self._channels: set = set()        # channel numbers seen in grants
        for e in entries:
            self.add(e)

    def add(self, entry: IdenEntry) -> None:
        self._by_iden[int(entry.iden)] = entry
        log.debug("iden %s: base %.4f MHz, step %.2f kHz, tx offset %+.4f MHz, slots %d",
                  entry.iden, entry.base_hz / 1e6, entry.spacing_hz / 1e3,
                  entry.tx_offset_hz / 1e6, entry.slots)

    def note_channel(self, ch16: int) -> None:
        """Remember a channel number a grant named, so `classify` can match
        traffic channels and not only the control channel."""
        self._channels.add(int(ch16) & 0xFFFF)

    @property
    def entries(self) -> list:
        return list(self._by_iden.values())

    def channel(self, ch16: int):
        """(outbound_hz, inbound_hz, slot or None), or None for an unknown
        identifier."""
        ch16 = int(ch16) & 0xFFFF
        e = self._by_iden.get(ch16 >> 12)
        if e is None:
            return None
        ch = ch16 & 0x0FFF
        carrier, slot = (ch // e.slots, ch % e.slots) if e.slots > 1 else (ch, None)
        out = e.base_hz + carrier * e.spacing_hz
        return out, out + e.tx_offset_hz, slot

    def classify(self, f_hz: float, tol_hz: float = DEFAULT_TOL_HZ) -> "Classification | None":
        """Inbound/outbound from the table: an exact match against a channel a
        grant named (or the control channel, noted the same way), else a
        raster match on any identifier — which decides only when the
        frequency sits on ONE sense's grid. In practice it rarely can: a
        transmit offset is a whole number of channel steps (5 MHz is 400 ×
        12.5 kHz), so a frequency on the inbound grid is usually on the
        outbound grid too, and the answer is honestly None. The named
        channels are what make a site table decisive — `note_channel` every
        channel number a grant names. None when the table cannot say."""
        for ch16 in sorted(self._channels):
            r = self.channel(ch16)
            if r is None:
                continue
            out, inb, _slot = r
            if abs(f_hz - out) <= tol_hz:
                return Classification("outbound", "iden", self.name,
                                      f"{f_hz / 1e6:.4f} MHz is channel {ch16:#06x}'s "
                                      f"base-transmit frequency (site IDEN table)")
            if abs(f_hz - inb) <= tol_hz:
                return Classification("inbound", "iden", self.name,
                                      f"{f_hz / 1e6:.4f} MHz is channel {ch16:#06x}'s "
                                      f"subscriber-transmit frequency (site IDEN table)")
        hits = set()
        for e in self._by_iden.values():
            for sense, base in (("outbound", e.base_hz), ("inbound", e.base_hz + e.tx_offset_hz)):
                k = (f_hz - base) / e.spacing_hz
                if k >= -0.5 and abs(k - round(k)) * e.spacing_hz <= tol_hz \
                        and round(k) < 4096 // e.slots:
                    hits.add((sense, e.iden))
        senses = {s for s, _ in hits}
        if len(senses) == 1:
            sense = senses.pop()
            idens = sorted(i for _, i in hits)
            return Classification(sense, "iden", self.name,
                                  f"{f_hz / 1e6:.4f} MHz sits on band identifier "
                                  f"{idens}'s {'base' if sense == 'outbound' else 'subscriber'}"
                                  f"-transmit raster (site IDEN table)")
        if len(senses) > 1:
            log.debug("iden raster ambiguous at %.4f MHz: %s", f_hz / 1e6, sorted(hits))
        return None


# ---------------------------------------------------------------------------
# regional plans
# ---------------------------------------------------------------------------

#: US 800 MHz mutual-aid calling / tactical channels (outbound frequencies),
#: after rebanding and before it (some areas still carry the old ones). These
#: are repeater pairs that are ALSO used talk-around on the base frequency, so
#: the frequency cannot decide. Verify for your region before relying on it.
_US_800_TALKAROUND = (851.0125e6, 851.5125e6, 852.0125e6, 852.5125e6, 853.0125e6,
                      866.0125e6, 866.5125e6, 867.0125e6, 867.5125e6, 868.0125e6)

_REGIONS = {
    "us": (
        SplitBand("US 800 MHz", (851.0e6, 869.0e6), (806.0e6, 824.0e6), -45.0e6,
                  _US_800_TALKAROUND,
                  "806–824 MHz subscriber / 851–869 MHz base (rebanded); "
                  "8CALL90/8TAC91–94 also talk-around"),
        SplitBand("US 700 MHz public safety", (769.0e6, 775.0e6), (799.0e6, 805.0e6), 30.0e6,
                  (), "799–805 MHz subscriber / 769–775 MHz base; talk-around on base "
                      "frequencies happens here too — mark known ones in the catalogue"),
        SplitBand("US 900 MHz", (935.0e6, 940.0e6), (896.0e6, 901.0e6), -39.0e6,
                  (), "896–901 MHz subscriber / 935–940 MHz base"),
    ),
    # European public-safety TETRA: 380–385 MHz mobile transmit, 390–395 MHz
    # base transmit. Other TETRA / PMR bands in Europe have no single rule.
    "eu": (
        SplitBand("EU TETRA public safety", (390.0e6, 395.0e6), (380.0e6, 385.0e6), -10.0e6,
                  (), "380–385 MHz mobile / 390–395 MHz base"),
    ),
}

#: Bands with no single duplex rule, named so the answer can say so.
_NO_RULE = (
    ("VHF high band", 136.0e6, 174.0e6),
    ("UHF 380–512 MHz", 380.0e6, 512.0e6),
    ("220 MHz", 216.0e6, 225.0e6),
)


class BandPlan:
    """Classifies a frequency as inbound / outbound / talkaround / unknown,
    from the site IDEN tables first, then the region's split bands. Built by
    `for_region`; an unknown region is an empty plan (everything "unknown",
    with the reason)."""

    def __init__(self, region: str = "", bands=(), idens=(), tol_hz: float = DEFAULT_TOL_HZ):
        self.region = (region or "").lower()
        self.bands = tuple(bands)
        self.idens: list = list(idens)
        self.tol_hz = float(tol_hz)

    @classmethod
    def for_region(cls, region: str, tol_hz: float = DEFAULT_TOL_HZ) -> "BandPlan":
        key = (region or "").strip().lower()
        bands = _REGIONS.get(key)
        if bands is None:
            log.warning("no band plan for region %r: every frequency will be "
                        "'unknown' (known: %s)", region, ", ".join(sorted(_REGIONS)))
            return cls(key, (), tol_hz=tol_hz)
        return cls(key, bands, tol_hz=tol_hz)

    @staticmethod
    def regions() -> list:
        return sorted(_REGIONS)

    def add_site(self, table: IdenTable) -> None:
        """A site's IDEN table, consulted before the regional rules."""
        self.idens.append(table)

    def classify(self, f_hz: float) -> Classification:
        f = float(f_hz)
        if f <= 0:
            return Classification("unknown", why="no frequency", region=self.region)
        for t in self.idens:
            c = t.classify(f, self.tol_hz)
            if c is not None:
                return Classification(c.direction, c.source, c.band, c.why, self.region)
        for b in self.bands:
            side = b.contains(f)
            if not side:
                continue
            if side == "outbound" and any(abs(f - t) <= self.tol_hz for t in b.talkaround_hz):
                return Classification("talkaround", "bandplan", b.name,
                                      f"{f / 1e6:.4f} MHz is a {b.name} base frequency also "
                                      f"used talk-around — the frequency cannot say which",
                                      self.region)
            words = "subscriber-transmit" if side == "inbound" else "base-transmit"
            return Classification(side, "bandplan", b.name,
                                  f"{f / 1e6:.4f} MHz is {b.name} {words} ({b.note})",
                                  self.region)
        for name, lo, hi in _NO_RULE:
            if lo <= f <= hi:
                return Classification("unknown", "bandplan", name,
                                      f"{name} has no single duplex split — the site's "
                                      f"IDEN table or the catalogue must say which side "
                                      f"{f / 1e6:.4f} MHz is", self.region)
        if not self.bands and not self.idens:
            why = (f"no band plan for region {self.region!r}" if self.region
                   else "no band plan configured")
        else:
            why = f"{f / 1e6:.4f} MHz is outside every band the {self.region or 'given'} plan knows"
        return Classification("unknown", "", "", why, self.region)

    def describe(self) -> str:
        if not self.bands and not self.idens:
            return f"band plan: none ({self.region or 'no region'})"
        names = [b.name for b in self.bands]
        if self.idens:
            names.append(f"{len(self.idens)} site IDEN table(s)")
        return f"band plan {self.region}: " + ", ".join(names)
