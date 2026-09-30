"""`Measurement` and `Dossier` — the shape every instrument answers in, and
the running record the analyst, the exports and the persona all read.

THE SUMMARY LAYER IS A DELIVERABLE, NOT DECORATION. A bi-frequency correlation
surface is beautiful and unreadable to a person at hour eleven. This module is
where surfaces become *"symbol rate ≈ 9600 Bd, carrier offset ≈ −1.2 kHz,
conjugate features present → likely BPSK"*, with the surface one click away.
Every instrument in `siga/` returns `Measurement`s and nothing else, which is
what makes one summary function possible instead of nine.

FOUR PROPERTIES ARE LOAD-BEARING, and each one is a mistake this codebase has
already made somewhere else:

* **`unit` is mandatory and never empty for a number.** A bare `9600` that
  might be Hz or Bd is exactly the confusion O.W.L. enforces units to prevent.
  `Measurement.__post_init__` raises on a numeric value with no unit — at
  construction, in the instrument's own test, not in a report six weeks later.

* **`UNKNOWN` and `AMBIGUOUS` are values, not exceptions and not None.** An
  instrument that cannot decide must say so *in the dossier*, where it is
  visible, exportable and countable. `emitters.py`'s rule — a wrong name stops
  the search — applies to every estimator here: refusing is a result.

* **Confidence comes from the method's own statistic.** Peak-to-floor in dB,
  spectral coherence, eigenvalue ratio, cluster separation. Never a constant
  somebody typed to make the bar look encouraging.

* **Provenance travels with the number.** `chain` is the preprocessing history
  of the `SigCtx` the measurement was made on. A symbol rate measured after a
  matched filter and a de-rotation is a different claim from one measured on
  the raw channel, and an exported CSV that cannot tell them apart is an
  exported CSV that cannot be defended.

COPIED FROM ATK, 2026-09-30 (ATK c52b913, `atk/core/siga/dossier.py`), verbatim. ATK keeps
its own `atk/core/siga/dossier.py`, because ATK's own recorder and every signals-bench
module stand on it and ATK must start without the DF engine; the
engine needs one of its own because it never imports ATK. The two are
expected to diverge only by additions; ATK's tests read files written
by one through the other.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

UNKNOWN = "UNKNOWN"
AMBIGUOUS = "AMBIGUOUS"

# Units the bench uses. Not a closed set — an instrument may invent one — but
# an empty unit on a number is refused, and "" is legal only for a label.
DIMENSIONLESS = "—"


@dataclass
class Measurement:
    """One measured quantity, with everything needed to disbelieve it."""

    name: str
    value: float | str | None
    unit: str
    confidence: float = 0.0
    method: str = ""
    evidence: dict = field(default_factory=dict)
    caveats: list = field(default_factory=list)
    source: str = ""                 # the SigCtx.source it was measured on
    chain: tuple = ()                # the SigCtx.history at measurement time

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.method = str(self.method)
        self.unit = str(self.unit or "")
        # A CAVEAT IS ALWAYS A STRING. `add_caveat` coerces, but the
        # constructors — `unknown`, `ambiguous`, and every instrument passing
        # `caveats=[...]` — do not, so a list or a number slipped in here
        # survives as far as `" · ".join(m.caveats)` in the panel and raises
        # in front of the analyst, at display time rather than at the point
        # the mistake was made.
        self.caveats = [str(c) for c in (self.caveats or [])]
        self.evidence = dict(self.evidence or {})
        try:
            self.confidence = float(self.confidence)
        except (TypeError, ValueError):
            self.confidence = 0.0
        self.confidence = min(1.0, max(0.0, self.confidence))
        if isinstance(self.value, (int, float, np.floating, np.integer)):
            self.value = float(self.value)
            if not self.unit:
                raise ValueError(
                    f"measurement {self.name!r} carries a number with no "
                    f"unit. A bare value that might be Hz or Bd is the "
                    f"confusion units exist to prevent — pass "
                    f"unit={DIMENSIONLESS!r} if it really is dimensionless.")
            if not math.isfinite(self.value):
                self.value = UNKNOWN
                self.caveats.append("estimator produced a non-finite value")

    # -- interrogation ----------------------------------------------------
    @property
    def known(self) -> bool:
        return isinstance(self.value, float)

    @property
    def undecided(self) -> bool:
        return self.value in (UNKNOWN, AMBIGUOUS)

    def add_caveat(self, text: str) -> "Measurement":
        if text and text not in self.caveats:
            self.caveats.append(str(text))
        return self

    def bind(self, ctx) -> "Measurement":
        """Stamp the context this was measured on. Called by the instruments;
        a measurement that skipped it exports with an empty chain, which is
        visibly wrong rather than quietly wrong."""
        self.source = getattr(ctx, "source", "") or self.source
        self.chain = tuple(getattr(ctx, "history", ()) or ())
        return self

    # -- presentation -----------------------------------------------------
    def format_value(self) -> str:
        if not self.known:
            return str(self.value if self.value is not None else UNKNOWN)
        v = float(self.value)
        if self.unit == "Hz" and abs(v) >= 1e6:
            return f"{v / 1e6:,.6f} MHz"
        if self.unit == "Hz" and abs(v) >= 1e3:
            return f"{v / 1e3:,.3f} kHz"
        if self.unit == "s" and abs(v) < 1e-3:
            return f"{v * 1e6:,.3f} µs"
        if self.unit == "s" and abs(v) < 1.0:
            return f"{v * 1e3:,.3f} ms"
        if self.unit in ("Bd", "Hz", "sym"):
            return f"{v:,.1f} {self.unit}"
        if self.unit == DIMENSIONLESS:
            return f"{v:,.4g}"
        return f"{v:,.4g} {self.unit}"

    def line(self) -> str:
        bits = [f"{self.name}: {self.format_value()}"]
        if self.confidence:
            bits.append(f"confidence {self.confidence:.0%}")
        if self.method:
            bits.append(f"via {self.method}")
        text = " · ".join(bits)
        if self.caveats:
            text += "  [" + "; ".join(self.caveats) + "]"
        return text

    def as_dict(self) -> dict:
        return {"name": self.name, "value": self.value, "unit": self.unit,
                "confidence": round(self.confidence, 4),
                "method": self.method,
                "evidence": {k: _plain(v) for k, v in self.evidence.items()},
                "caveats": list(self.caveats), "source": self.source,
                "chain": list(self.chain)}


def _plain(v):
    """numpy scalars/arrays → JSON-able. Arrays are summarised, never dumped:
    an evidence field that inlines a 4096-point spectrum makes the export
    unreadable and the CSV unopenable."""
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    if isinstance(v, np.ndarray):
        if v.size <= 8:
            return [_plain(x) for x in v.tolist()]
        return {"array": list(v.shape), "min": float(np.nanmin(v)),
                "max": float(np.nanmax(v))}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    return v


# ---------------------------------------------------------------------------
# helpers every instrument uses, so "I don't know" looks the same everywhere
# ---------------------------------------------------------------------------

def unknown(name: str, unit: str, method: str, why: str,
            evidence: dict | None = None) -> Measurement:
    """The honest-failure constructor. Used ~40 times across the bench."""
    return Measurement(name=name, value=UNKNOWN, unit=unit or DIMENSIONLESS,
                       confidence=0.0, method=method,
                       evidence=evidence or {}, caveats=[why])


def ambiguous(name: str, unit: str, method: str, why: str,
              evidence: dict | None = None) -> Measurement:
    """Distinct from `unknown` on purpose: AMBIGUOUS means several estimators
    answered and disagreed, which is a different situation from nothing
    working, and the analyst treats them differently."""
    return Measurement(name=name, value=AMBIGUOUS, unit=unit or DIMENSIONLESS,
                       confidence=0.0, method=method,
                       evidence=evidence or {}, caveats=[why])


def consensus(name: str, unit: str, candidates: list, tol: float = 0.02,
              method: str = "consensus") -> Measurement:
    """Fuse several independent estimates of ONE quantity.

    `candidates` is a list of `Measurement`. The rule, and it is the whole
    reason this function exists rather than `max(..., key=confidence)`:

      * two or more estimates agreeing within `tol` (relative) is strong
        evidence, and the agreement itself becomes the confidence — a value
        no single estimator could have justified;
      * exactly one estimate is that estimate, at ITS confidence, with a
        caveat saying it stood alone;
      * estimates that disagree return **AMBIGUOUS with the spread**, not the
        most confident one. Picking a winner from disagreement is how a bench
        produces a number nobody can reproduce.

    Agreement is clustered rather than averaged, because two estimators on
    9600 and one on 1200 is a 9600 signal with a failed estimator, and the
    mean of the three is 6800 Bd, which is nothing.
    """
    known = [m for m in candidates if m is not None and m.known
             and float(m.value) > 0]
    if not known:
        return unknown(name, unit, method,
                       "no estimator produced a usable value",
                       {"tried": [m.method for m in candidates if m]})
    vals = np.array([float(m.value) for m in known], dtype=float)

    # cluster on relative distance, largest cluster wins
    best: list[int] = []
    for i, v in enumerate(vals):
        group = [j for j, w in enumerate(vals)
                 if abs(w - v) <= tol * max(abs(v), abs(w), 1e-12)]
        if len(group) > len(best):
            best = group
    members = [known[j] for j in best]
    value = float(np.median(vals[best]))
    spread = (float(np.max(vals[best]) - np.min(vals[best])) / value
              if value else 0.0)
    ev = {"estimates": {m.method: float(m.value) for m in known},
          "agreed": [m.method for m in members],
          "spread_pct": round(spread * 100, 3)}

    if len(members) == 1 and len(known) > 1:
        return ambiguous(
            name, unit, method,
            "estimators disagree — "
            + ", ".join(f"{m.method} {float(m.value):,.4g} {unit}"
                        for m in known)
            + ". The spread is reported rather than one of them chosen.",
            ev)
    if len(members) == 1:
        m = members[0]
        out = Measurement(name=name, value=value, unit=unit,
                          confidence=min(0.6, m.confidence),
                          method=f"{method} ({m.method}, alone)", evidence=ev)
        return out.add_caveat("only one estimator produced a value — "
                              "confirm it on the eye diagram before relying "
                              "on it")
    # agreement: confidence rises with how many agreed and how tightly
    tightness = 1.0 - min(1.0, spread / max(tol, 1e-9))
    base = float(np.mean([m.confidence for m in members]))
    conf = min(0.98, 0.5 * base + 0.5 * (0.6 + 0.1 * len(members)) * (
        0.5 + 0.5 * tightness))
    out = Measurement(name=name, value=value, unit=unit, confidence=conf,
                      method=f"{method} ({len(members)} of {len(known)} "
                             f"agree)", evidence=ev)
    if len(members) < len(known):
        out.add_caveat(
            f"{len(known) - len(members)} estimator(s) disagreed and were "
            f"outvoted, not averaged in: "
            + ", ".join(f"{m.method} {float(m.value):,.4g}"
                        for m in known if m not in members))
    return out


# ---------------------------------------------------------------------------
# resolution honesty
# ---------------------------------------------------------------------------

@dataclass
class Resolution:
    """What a record and a window can and cannot resolve.

    Δf comes from the analysis WINDOW; Δα from the whole RECORD. They are
    different numbers and confusing them is the standard way to over-read a
    cyclic surface, so both are computed here and printed together.
    """

    fs: float
    record_samples: int
    window_samples: int

    @property
    def df_hz(self) -> float:
        return self.fs / max(1, self.window_samples)

    @property
    def dalpha_hz(self) -> float:
        return self.fs / max(1, self.record_samples)

    @property
    def record_s(self) -> float:
        return self.record_samples / self.fs

    @property
    def valid(self) -> bool:
        """A valid spectral-correlation estimate needs Δf comfortably COARSER
        than Δα — the reliability condition. Equal resolutions give a surface
        that looks fine and means nothing.

        Named `valid` to match `csp.caf.ScfSurface.valid`, which asks exactly
        the same question: two names for one condition is how one of them
        ends up unchecked.
        """
        return self.df_hz >= 4.0 * self.dalpha_hz

    #: Kept so nothing that spelled it the old way breaks.
    @property
    def valid_scf(self) -> bool:
        return self.valid

    def note(self) -> str:
        bits = [f"record {self.record_s * 1e3:,.1f} ms",
                f"Δf ≈ {self.df_hz:,.1f} Hz (window)",
                f"Δα ≈ {self.dalpha_hz:,.2f} Hz (record)"]
        if not self.valid:
            bits.append("Δf is NOT comfortably coarser than Δα — cyclic "
                        "estimates from this record are unreliable; lengthen "
                        "the record or shorten the window")
        return " · ".join(bits)


def resolution_note(record_samples: int, window_samples: int,
                    fs: float) -> Resolution:
    return Resolution(fs=float(fs), record_samples=int(record_samples),
                      window_samples=int(window_samples))


def guard_resolution(m: Measurement, resolution: float,
                     what: str = "estimate") -> Measurement:
    """Refuse an estimate finer than the record can support.

    Not a caveat — a REFUSAL. Reporting 9601.3 Bd from a record whose cyclic
    resolution is 40 Hz is reporting four significant figures of arithmetic
    and one of measurement, and the analyst has no way to tell from the
    number which it was. The value is replaced by UNKNOWN and the attempted
    value is kept in evidence so nothing is hidden.
    """
    if resolution <= 0:
        return m
    #: RECORDED EVEN WHEN THERE IS NOTHING TO GUARD, 2026-09-02. The early
    #: return used to skip this, so a measurement that had ALREADY been
    #: refused for some other reason came back with no record of what this
    #: capture could have resolved — and "no estimator produced a usable
    #: value" is exactly the verdict where an analyst needs to know whether
    #: the record was ever long enough. Centralising the rule is only worth
    #: doing if the centre carries everything the copies did.
    m.evidence.setdefault("resolution_hz", float(resolution))
    if not m.known:
        return m
    if abs(float(m.value)) < resolution:
        m.evidence["refused_value"] = float(m.value)
        m.evidence["resolution_hz"] = float(resolution)
        m.value = UNKNOWN
        m.confidence = 0.0
        m.add_caveat(
            f"{what} fell below this record's resolution "
            f"({resolution:,.3g} {m.unit}) — refused rather than reported")
    else:
        m.evidence.setdefault("resolution_hz", float(resolution))
        if abs(float(m.value)) < 4.0 * resolution:
            m.add_caveat(f"within 4× the resolution limit "
                         f"({resolution:,.3g} {m.unit}) — treat the last "
                         f"digits as noise")
    return m


# ---------------------------------------------------------------------------
# the Dossier
# ---------------------------------------------------------------------------

@dataclass
class Dossier:
    """Every measurement of one signal, plus the sentence at the top of it."""

    title: str = ""
    source: str = ""
    provenance: str = ""
    measurements: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    surfaces: dict = field(default_factory=dict, repr=False)

    # -- building ---------------------------------------------------------
    def add(self, m, ctx=None) -> "Dossier":
        """Add one measurement or an iterable of them, binding provenance."""
        if m is None:
            return self
        items = list(m) if isinstance(m, (list, tuple)) else [m]
        for item in items:
            if item is None:
                continue
            if ctx is not None:
                item.bind(ctx)
            self.measurements.append(item)
        return self

    def note(self, text: str) -> "Dossier":
        if text:
            self.notes.append(str(text))
        return self

    def bind_context(self, ctx) -> "Dossier":
        self.source = getattr(ctx, "source", "") or self.source
        self.provenance = ctx.provenance() if hasattr(ctx, "provenance") \
            else self.provenance
        return self

    # -- reading ----------------------------------------------------------
    def get(self, name: str) -> Measurement | None:
        """The most confident measurement of `name`, or None.

        Most confident rather than most recent: the bench runs cheap
        estimators first and expensive ones later, and "later" is not a
        reason to believe something.
        """
        hits = [m for m in self.measurements if m.name == name]
        if not hits:
            return None
        known = [m for m in hits if m.known]
        pool = known or hits
        return max(pool, key=lambda m: m.confidence)

    def value(self, name: str, default=None):
        """The value of `name`, numeric OR label, or `default`.

        Label measurements — modulation, line code, detection verdict — are
        not `known` (that property means "is a number"), and an earlier
        version returned `default` for them. The visible symptom was a SigMF
        export with no `core:label` on a signal the bench had confidently
        classified: the annotation was written, and the one field another
        tool would actually read was silently missing.
        """
        m = self.get(name)
        if m is None:
            return default
        if m.known:
            return m.value
        if isinstance(m.value, str) and not m.undecided:
            return m.value
        return default

    def all_of(self, name: str) -> list:
        return [m for m in self.measurements if m.name == name]

    @property
    def caveats(self) -> list:
        seen, out = set(), []
        for m in self.measurements:
            for c in m.caveats:
                if c not in seen:
                    seen.add(c)
                    out.append(c)
        return out

    # -- the sentence -----------------------------------------------------
    def sentence(self) -> str:
        """The plain-language line a tired operator can act on.

        Composed from whichever measurements exist, in the order an analyst
        reads them: what it is, how fast, where, and then what is doubtful.
        Deliberately says "nothing measured yet" rather than an empty string —
        a blank summary reads as a working instrument that found nothing.
        """
        if not self.measurements:
            return ("Nothing measured yet — pick a signal and run an "
                    "instrument.")
        parts: list[str] = []

        # FIRST, BECAUSE IT IS WHAT ORIENTS EVERYTHING ELSE. A summary that
        # opens "likely OFDM, symbol rate 4,272 Bd" without saying WHERE is a
        # summary you cannot act on or hand to anybody — and every other
        # frequency in the line is an offset from this one.
        at = self.get("signal_frequency") or self.get("capture_centre")
        if at is not None and at.known:
            parts.append(f"at {float(at.value) / 1e6:,.4f} MHz")

        cls = self.get("modulation")
        if cls is not None:
            if cls.known or cls.value not in (UNKNOWN, AMBIGUOUS):
                parts.append(f"likely {cls.value}")
            elif cls.value == AMBIGUOUS:
                parts.append("modulation AMBIGUOUS (estimators disagree)")
            else:
                parts.append("modulation not identified")

        baud = self.get("symbol_rate")
        if baud is not None and baud.known:
            parts.append(f"symbol rate ≈ {float(baud.value):,.1f} Bd"
                         + (f" ({baud.confidence:.0%})" if baud.confidence
                            else ""))
        elif baud is not None:
            parts.append(f"symbol rate {baud.value}")

        off = self.get("carrier_offset")
        if off is not None and off.known:
            parts.append(f"carrier offset ≈ {float(off.value):+,.1f} Hz")

        order = self.get("psk_order")
        if order is not None and order.known:
            parts.append(f"{int(float(order.value))}-ary phase structure")

        bw = self.get("occupied_bw")
        if bw is not None and bw.known:
            parts.append(f"occupied bandwidth ≈ {float(bw.value) / 1e3:,.2f} "
                         f"kHz")

        det = self.get("detection")
        if det is not None:
            parts.append(f"detection: {det.value}"
                         if not det.known else
                         f"detection statistic {float(det.value):,.3g}")

        prf = self.get("prf")
        if prf is not None and prf.known:
            parts.append(f"pulse repetition ≈ {float(prf.value):,.1f} Hz")

        if not parts:
            parts.append(f"{len(self.measurements)} measurements taken, none "
                         f"conclusive")
        head = "; ".join(parts) + "."
        # Only the HEADLINE measurements are named as undecided. The per-power
        # line probes are UNKNOWN on almost every signal — that is what they
        # are for — and listing them turns the one sentence a tired operator
        # reads into a list of things that were never expected to answer.
        undecided = [m.name for m in self.measurements
                     if m.undecided and m.name in self.HEADLINE]
        if undecided:
            head += (f" Undecided: {', '.join(sorted(set(undecided)))} — "
                     f"reported as such rather than guessed.")
        return head

    # -- exports ----------------------------------------------------------
    def rows(self) -> list:
        """One row per measurement — the UI table and the CSV share this, so
        what is exported is exactly what was on screen."""
        return [[m.name, m.format_value(), m.unit,
                 f"{m.confidence:.0%}" if m.confidence else "",
                 m.method, "; ".join(m.caveats),
                 " → ".join(m.chain) or "(as received)"]
                for m in self.measurements]

    #: The measurements the one-line summary is allowed to call "undecided".
    HEADLINE = ("modulation", "symbol_rate", "carrier_offset", "psk_order",
                "occupied_bw", "detection", "prf")

    COLUMNS = ["measurement", "value", "unit", "confidence", "method",
               "caveats", "preprocessing"]

    def to_csv(self) -> str:
        import csv
        import io
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(self.COLUMNS)
        w.writerows(self.rows())
        return buf.getvalue()

    def to_markdown(self) -> str:
        out = [f"# {self.title or 'Signal dossier'}", ""]
        out.append(f"**{self.sentence()}**")
        out.append("")
        if self.provenance:
            out.append(f"*Source:* {self.provenance}")
            out.append("")
        out.append("| " + " | ".join(self.COLUMNS) + " |")
        out.append("|" + "|".join(["---"] * len(self.COLUMNS)) + "|")
        for r in self.rows():
            out.append("| " + " | ".join(str(c).replace("|", "\\|")
                                         for c in r) + " |")
        if self.notes:
            out += ["", "## Notes", ""] + [f"- {n}" for n in self.notes]
        if self.caveats:
            out += ["", "## Caveats carried by these measurements", ""]
            out += [f"- {c}" for c in self.caveats]
        return "\n".join(out) + "\n"

    def to_json(self) -> str:
        return json.dumps({
            "title": self.title, "source": self.source,
            "provenance": self.provenance, "summary": self.sentence(),
            "measurements": [m.as_dict() for m in self.measurements],
            "notes": list(self.notes)}, indent=2)

    def to_sigmf_annotations(self, sample_start: int = 0,
                             sample_count: int | None = None) -> list:
        """SigMF `annotations` entries describing this signal.

        Written back onto the recording so a marked-up capture is portable —
        IQEngine, inspectrum and anything else that reads SigMF see the
        bench's conclusions without ATK. Only measurements SigMF has a home
        for become core fields; everything else rides in a namespaced
        `atk:` block rather than being dropped, because a dossier that loses
        its caveats on export is a dossier that gets over-trusted elsewhere.
        """
        ann = {"core:sample_start": int(sample_start)}
        if sample_count:
            ann["core:sample_count"] = int(sample_count)
        label = self.value("modulation")
        if isinstance(label, str) and label not in (UNKNOWN, AMBIGUOUS):
            ann["core:label"] = label
        bw = self.value("occupied_bw")
        if bw:
            ann["core:freq_lower_edge"] = -float(bw) / 2.0
            ann["core:freq_upper_edge"] = float(bw) / 2.0
        ann["core:description"] = self.sentence()
        ann["atk:measurements"] = [m.as_dict() for m in self.measurements]
        ann["atk:provenance"] = self.provenance
        return [ann]
