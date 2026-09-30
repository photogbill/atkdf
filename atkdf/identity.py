"""Turning decoder events into burst spans the engine can direction-find.

Registrations, radio checks and keep-alives are tens of milliseconds and
unannounced — nobody has a window open in time, but the ring does. The decoder
reports AFTER THE FACT that ID 473 keyed at samples n0..n1 of segment S, and
the DoA is computed over exactly those samples. One mechanism serves every
protocol; the decoder supplies the schedule.

This module normalises a `BurstEvent` (from ATK's DSD tap, dmr_sync, or the
simulator's side channel) into a `Span`: sample range, identity, slot, sync
type, and the PATH TAG. The path tag is the trap every naive DF app falls into
— a repeater's outbound burst carries the handset's ID with the REPEATER's
bearing, and must never be fused as the handset's position. DMR's sync pattern
says BS- or MS-sourced; the catalogue's repeater plan is the fallback.

Timing can be sample-exact (the decoder gave n0/n1) or time-only (it gave a
UTC instant and a span). A time-only event is widened by `slack_s` on each
side, because a decoder console's latency can be coarser than a burst and it
is better to integrate a little noise than to miss the burst.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ports import BurstEvent
from .records import EmitterKey

#: sync-type spellings that mean the burst came FROM a base station / repeater
#: (outbound: the handset's ID, the repeater's bearing) or FROM a mobile
#: (inbound: the handset itself).
_BS = {"bs", "base", "bs_voice", "bs_data", "outbound", "repeater"}
_MS = {"ms", "mobile", "ms_voice", "ms_data", "direct", "dmo", "inbound"}


@dataclass
class Span:
    """A burst located in the stream and in identity."""

    n0: int                       # channel-0 stream index, inclusive
    n1: int                       # exclusive
    t_utc: float
    key: str = ""                 # str(EmitterKey), or "" with no identity
    slot: int | None = None
    sync_type: str = ""
    path_tag: str = "unknown"
    fc_hz: float = 0.0
    bw_hz: float = 0.0
    segment: int | None = None
    exact: bool = True            # False when widened from a time-only event

    @property
    def n_samples(self) -> int:
        return max(0, self.n1 - self.n0)


def path_tag_of(event: BurstEvent, catalogue=None) -> str:
    """direct / repeated / unknown, from the sync type, then the catalogue."""
    st = (event.sync_type or "").lower()
    if st in _BS:
        return "repeated"
    if st in _MS:
        return "direct"
    if catalogue is not None:
        ref = catalogue.lookup(str(EmitterKey(event.protocol, event.system, event.id)))
        if ref is not None and ref.kind in ("repeater_output", "bs", "repeater"):
            return "repeated"
    return "unknown"


def span_of(event: BurstEvent, fs_hz: float, stream_sample0_at_t0: int = 0,
            t0_utc: float = 0.0, slack_s: float = 0.010, catalogue=None) -> Span:
    """One event -> a Span. If the event is sample-exact (n0/n1 set) those are
    used; otherwise the span is derived from the UTC instant and widened by
    `slack_s`, mapping time to sample index with (t0_utc -> stream_sample0)."""
    key = str(EmitterKey(event.protocol, event.system, event.id)) if event.id else ""
    if event.n0 is not None and event.n1 is not None:
        n0, n1, exact = int(event.n0), int(event.n1), True
    else:
        span = event.span_s or 0.0
        start = event.t_utc - slack_s
        end = event.t_utc + span + slack_s
        n0 = stream_sample0_at_t0 + int(round((start - t0_utc) * fs_hz))
        n1 = stream_sample0_at_t0 + int(round((end - t0_utc) * fs_hz))
        exact = False
    return Span(n0=n0, n1=n1, t_utc=event.t_utc, key=key, slot=event.slot,
                sync_type=event.sync_type, path_tag=path_tag_of(event, catalogue),
                fc_hz=event.fc_hz, bw_hz=event.bw_hz, segment=event.segment, exact=exact)


def spans(events, fs_hz: float, catalogue=None, slack_s: float = 0.010) -> list:
    """A batch of events -> Spans, in time order. Two IDs in the two slots of
    one channel become two Spans with two keys — never merged (umbrella §7.1)."""
    out = [span_of(e, fs_hz, catalogue=catalogue, slack_s=slack_s)
           for e in sorted(events, key=lambda e: e.t_utc)]
    return out
