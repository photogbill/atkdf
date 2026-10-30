"""Array profiles: one JSON file per physical DF array (schema v2).

DF_P25_IMPROVEMENT_PLAN §11.4.1. A profile says everything ATK needs to know
about an array to direction-find with it: how the elements are laid out, what
the receiver sees of them (every element, a switched subset, analog
combinations), whether its channels share an LO (and so whether a calibration
survives a retune or a power cycle), what time source it has, which
calibration tables it keeps, and its field of view. The engine gets a
`Geometry`, a `Manifold` and an estimator policy out of it; ATK's DF tab gets
a sentence about what the array can and cannot do.

(Called `arrayprofile`, not `profile`: a module named `profile` shadows the
standard library's, the same reason `manifold` is not called `array`.)

THE FIELDS THE LITERATURE MADE FIRST-CLASS (lit2 = DF_PHASED_ARRAY_LITERATURE_2020_2026):

  architecture        digital | switched | analog | hybrid (§11.4.2)
  lo_sharing          shared_within_device | distributed | independent —
                      LO sharing, not a 10 MHz clock, decides whether N
                      channels stay one array across retunes (lit2 §3.18, §4.6)
  rx_cal_persistence  across_power_cycles | per_power_cycle | per_retune —
                      how long a receiver calibration is good for, per
                      hardware class (X310+TwinRX tables persist across power
                      cycles; N310-class do not; independent LOs never)
  timing.tier         T0 none … T5 White Rabbit (§11.4.6); gates TDOA
  rf_chains           receiver channels: = elements for digital, fewer otherwise
  calibration         TWO tables: rx_table (per LO frequency) and
                      antenna_table (per band, interpolable) (lit2 §3.13)
  cable_class         ptfe | phase_stable | unknown — PTFE cable moves
                      ≈ 3.8°/m over 0–55 °C at 1 GHz (lit2 §4.7)

Conservative defaults: a profile that does not say how long its calibration
lasts is assumed to need one every power cycle (shared or distributed LO) or
every retune (independent LOs).

Everything here is configuration: no hardware is touched. A profile for
hardware ATK cannot stream from yet (a USRP before its FrameSource exists)
loads, validates and simulates; the DF tab says it is not live.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import manifold as _m

log = logging.getLogger("atkdf.arrayprofile")

SCHEMA = 2
ARCHITECTURES = ("digital", "switched", "analog", "hybrid")
LO_SHARING = ("shared_within_device", "distributed", "independent")
PERSISTENCE = ("across_power_cycles", "per_power_cycle", "per_retune")
TIMING_TIERS = ("T0", "T1", "T2", "T3", "T4", "T5")
CABLE_CLASSES = ("ptfe", "phase_stable", "unknown")
RX_CAL_METHODS = ("noise_source", "tone", "known_emitter", "reference_channel", "none")
ANT_CAL_METHODS = ("none", "known_emitter_drive", "tripod", "opportunity", "chamber")
PRESET_KINDS = ("uca", "ula", "ura", "l", "multi_face")

#: what each timing tier means, in the words the DF tab shows (§11.4.6)
TIER_WORDS = {
    "T0": "no time source — AOA fusion only",
    "T1": "host PPS (software timestamps) — better time tags, no TDOA",
    "T2": "GPSDO per receiver (±50 ns class) — wideband TDOA; for 12.5 kHz P25 a "
          "km-class hyperbola that breaks sidedness, not an ellipse tightener",
    "T3": "cabled 10 MHz + PPS on one platform — sample-aligned, LO phase still "
          "random per device per retune unless shared",
    "T4": "shared / distributed LO — the only tier at which several receivers are "
          "one array",
    "T5": "White Rabbit over fibre between fixed sites — cross-site coherence at "
          "VHF only",
}

#: Kraken / TwinRX / N321 facts the presets and the checks rely on (lit2 §4.6,
#: §3.17, §3.18): the frequency range over which each can SHARE its LO.
LO_SHARE_BAND_MHZ = {
    "usrp_x310_twinrx": (10.0, 6000.0),
    "usrp_n321": (450.0, 6000.0),
    "usrp_n320": (450.0, 6000.0),
    "usrp_n310": (300.0, 6000.0),
}

_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")


class ProfileError(ValueError):
    """A profile that cannot be used: every problem, listed."""

    def __init__(self, profile_id: str, problems: list):
        self.problems = list(problems)
        super().__init__(f"array profile {profile_id!r}: " + "; ".join(self.problems))


def _default_persistence(lo_sharing: str) -> str:
    return "per_retune" if lo_sharing == "independent" else "per_power_cycle"


@dataclass
class ArrayProfile:
    """One array. Build with `ArrayProfile.from_dict` / `load`, or a preset
    (`preset("kraken5_uca")`); `validate()` lists what is wrong and
    `geometry()` / `manifold()` give the engine what it needs."""

    id: str
    description: str = ""
    architecture: str = "digital"
    receiver: str = ""
    lo_sharing: str = "shared_within_device"
    rx_cal_persistence: str | None = None
    timing: dict = field(default_factory=lambda: {"tier": "T0", "t_utc_sigma_ns": None,
                                                  "holdover_spec": ""})
    band_mhz: tuple = (24.0, 1766.0)
    elements: dict = field(default_factory=lambda: {"preset": "uca", "n": 5, "radius_m": 0.18})
    mount: dict = field(default_factory=lambda: {"plane": "horizontal", "boresight_deg": 0.0,
                                                 "tilt_deg": 0.0})
    field_of_view_deg: dict = field(default_factory=dict)
    pattern: dict | None = None
    rf_chains: int | None = None
    switching: dict | None = None
    subarrays: list | None = None
    steering: dict | None = None
    calibration: dict = field(default_factory=lambda: {
        "rx_table": {"method": "noise_source", "cache_key": "lo_hz"},
        "antenna_table": {"method": "none", "per": "band"}})
    cable_class: str = "unknown"
    notes: str = ""
    schema: int = SCHEMA
    #: where the profile was loaded from (not saved)
    source_path: str = ""

    # -- construction ----------------------------------------------------------
    def __post_init__(self):
        if self.rx_cal_persistence in (None, ""):
            self.rx_cal_persistence = _default_persistence(self.lo_sharing)
        self.band_mhz = tuple(float(x) for x in self.band_mhz)
        t = {"tier": "T0", "t_utc_sigma_ns": None, "holdover_spec": ""}
        t.update(self.timing or {})
        self.timing = t
        self.mount = {"plane": "horizontal", "boresight_deg": 0.0, "tilt_deg": 0.0,
                      **(self.mount or {})}
        self.field_of_view_deg = dict(self.field_of_view_deg or {})
        cal = {"rx_table": {"method": "noise_source", "cache_key": "lo_hz"},
               "antenna_table": {"method": "none", "per": "band"}}
        for k, v in (self.calibration or {}).items():
            cal[k] = {**cal.get(k, {}), **(v or {})}
        self.calibration = cal

    @classmethod
    def from_dict(cls, d: dict, source_path: str = "") -> "ArrayProfile":
        d = dict(d or {})
        if "schema" in d and int(d["schema"]) != SCHEMA:
            raise ProfileError(d.get("id", "?"), [f"schema {d['schema']} is not {SCHEMA}"])
        known = {f for f in cls.__dataclass_fields__ if f != "source_path"}
        unknown = sorted(set(d) - known)
        if unknown:
            log.warning("array profile %r: ignoring unknown keys %s", d.get("id"), unknown)
        kw = {k: copy.deepcopy(v) for k, v in d.items() if k in known}
        if "id" not in kw:
            raise ProfileError("?", ["a profile needs an id"])
        if "band_mhz" in kw:
            kw["band_mhz"] = tuple(kw["band_mhz"])
        return cls(source_path=str(source_path), **kw)

    def to_dict(self) -> dict:
        out = {}
        for k in self.__dataclass_fields__:
            if k == "source_path":
                continue
            v = getattr(self, k)
            out[k] = list(v) if isinstance(v, tuple) else copy.deepcopy(v)
        # schema first, id second, for a human reading the file
        return {"schema": out.pop("schema"), "id": out.pop("id"), **out}

    @classmethod
    def load(cls, path) -> "ArrayProfile":
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            d = json.load(f)
        prof = cls.from_dict(d, source_path=str(p))
        log.info("array profile %s loaded from %s", prof.id, p)
        return prof

    def save(self, path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")
        log.info("array profile %s saved to %s", self.id, p)
        return p

    # -- what the engine needs ---------------------------------------------------
    @property
    def plane(self) -> str:
        return str(self.mount.get("plane", "horizontal"))

    @property
    def boresight_deg(self) -> float:
        return float(self.mount.get("boresight_deg", 0.0) or 0.0)

    @property
    def band_centre_hz(self) -> float:
        lo, hi = self.band_mhz
        return 0.5 * (lo + hi) * 1e6

    def geometry(self) -> "_m.Geometry":
        """The element Geometry: preset or explicit positions, then the
        mount, pattern, field of view, connector map, phase centres and
        amplitudes the profile gives."""
        e = dict(self.elements or {})
        preset = str(e.get("preset", "")).lower()
        pat = _m.ElementPattern.from_dict(self.pattern) if self.pattern else None
        cmap = e.get("channel_of_element")
        cmap = tuple(int(c) for c in cmap) if cmap else None
        b = self.boresight_deg
        if preset == "uca":
            g = _m.uca(int(e.get("n", 5)), float(e.get("radius_m", 0.18)),
                       channel_of_element=cmap, centre=bool(e.get("centre", False)))
        elif preset == "ula":
            g = _m.ula(int(e.get("n", 4)), float(e.get("spacing_m", 0.18)),
                       channel_of_element=cmap)
        elif preset == "ura":
            g = _m.ura(int(e.get("rows", 2)), int(e.get("cols", 2)), float(e["spacing_m"]),
                       plane=self.plane, boresight_deg=b, pattern=pat,
                       channel_of_element=cmap, row_spacing_m=e.get("row_spacing_m"))
        elif preset == "l":
            g = _m.l_array(int(e["n_a"]), int(e["n_b"]), float(e["spacing_m"]),
                           plane=self.plane, boresight_deg=b, pattern=pat,
                           channel_of_element=cmap)
        elif preset == "multi_face":
            g = _m.multi_face(int(e.get("faces", 4)), int(e.get("cols", 4)),
                              float(e.get("spacing_m", 0.18)), float(e.get("standoff_m", 0.25)),
                              rows=int(e.get("rows", 1)), pattern=pat, first_face_deg=b)
        elif "positions_m" in e:
            g = _m.Geometry(np.asarray(e["positions_m"], dtype=float), id=f"{self.id}_custom",
                            kind="custom", description=f"custom positions for {self.id}",
                            channel_of_element=cmap, pattern=pat,
                            boresight_deg=(np.asarray(e["boresight_deg"], float)
                                           if e.get("boresight_deg") is not None
                                           else (b if pat is not None and pat.kind != "omni"
                                                 else None)),
                            groups=tuple(e["groups"]) if e.get("groups") else None,
                            hemisphere=str(e.get("hemisphere") or "up"))
        else:
            raise ProfileError(self.id, ["elements needs a preset "
                                         f"({', '.join(PRESET_KINDS)}) or positions_m"])
        changes = {}
        if pat is not None and preset in ("uca", "ula"):
            changes["pattern"] = pat
            changes["boresight_deg"] = np.full(g.n, b)
        fov = self.field_of_view_deg or {}
        if fov.get("az") is not None:
            changes["fov_az_deg"] = tuple(float(x) for x in fov["az"])
        if fov.get("el") is not None:
            changes["fov_el_deg"] = tuple(float(x) for x in fov["el"])
        if e.get("phase_centre_m") is not None:
            changes["phase_centre_m"] = np.asarray(e["phase_centre_m"], dtype=float)
        if e.get("amplitude") is not None:
            changes["amplitude"] = np.asarray(e["amplitude"], dtype=float)
        if self.mount.get("hemisphere"):
            changes["hemisphere"] = str(self.mount["hemisphere"])
        if self.mount.get("tilt_deg"):
            changes["tilt_deg"] = float(self.mount["tilt_deg"])
        if self.architecture == "analog" and g.fov_az_deg is None and "fov_az_deg" not in changes:
            # an analog array's FOV is a hard limit (lit2 §1.8): ±60° unless told
            changes["fov_az_deg"] = (b - 60.0, b + 60.0)
        if changes:
            from dataclasses import replace
            g = replace(g, **changes)
        return replace_id(g, f"{self.id}:{g.id}")

    def manifold(self, antenna_table: str | Path | None = None) -> "_m.Manifold":
        """The element manifold: the measured antenna table when one is
        given (or named in the profile and present), else the ideal one."""
        geom = self.geometry()
        path = antenna_table or (self.calibration.get("antenna_table") or {}).get("file")
        if path:
            p = Path(path)
            if not p.is_absolute() and self.source_path:
                p = Path(self.source_path).parent / p
            if p.exists():
                man = _m.MeasuredManifold.load(p)
                if man.geometry.n != geom.n:
                    raise ProfileError(self.id, [f"antenna table {p.name} is for "
                                                 f"{man.geometry.n} elements, the profile "
                                                 f"has {geom.n}"])
                log.info("profile %s: measured antenna table %s", self.id, p)
                return man
            log.warning("profile %s: antenna table %s not found — ideal manifold", self.id, p)
        return _m.IdealManifold(geom)

    # -- what the array can do ------------------------------------------------------
    def capabilities(self) -> dict:
        g = self.geometry()
        planar_vertical = bool(np.ptp(g.z) > 1e-6)
        planar_horizontal = (g.kind in ("ura", "l") and self.plane == "horizontal")
        n_ch = self.channels()
        return {
            "elements": g.n,
            "channels": n_ch,
            "multi_source": self.architecture in ("digital", "hybrid") and n_ch >= 3,
            "max_sources": max(n_ch - 1, 1) if self.architecture in ("digital", "hybrid") else 1,
            "single_source_only": self.architecture in ("switched", "analog"),
            "elevation": planar_vertical or planar_horizontal,
            "elevation_note": ("measured well near the horizon" if planar_vertical else
                               "up/down ambiguous and weak near the horizon"
                               if planar_horizontal else "none: one horizontal plane"),
            "tdoa": self.timing.get("tier") in ("T2", "T3", "T4", "T5"),
            "one_array_across_devices": self.lo_sharing != "independent"
                                        or self.timing.get("tier") == "T4",
            "fov_az_deg": g.fov_az_deg,
            "estimators": self.estimators(),
        }

    def channels(self) -> int:
        if self.rf_chains:
            return int(self.rf_chains)
        return self.geometry().n if self.architecture == "digital" else 1

    def estimators(self) -> tuple:
        return {"digital": ("music", "root", "capon", "bartlett", "ml", "fbss"),
                "hybrid": ("music", "root", "capon", "bartlett", "ml"),
                "switched": ("switched",),
                "analog": ("beamscan", "monopulse") if self.channels() >= 2 else ("beamscan",),
                }.get(self.architecture, ())

    def summary(self) -> list:
        """What the DF tab says about this array, line by line."""
        cap = self.capabilities()
        lines = [f"{self.id}: {self.description or self.architecture}",
                 f"{cap['elements']} elements, {cap['channels']} receiver channel(s), "
                 f"{self.architecture}; band {self.band_mhz[0]:.0f}–{self.band_mhz[1]:.0f} MHz"]
        if cap["single_source_only"]:
            lines.append("SINGLE-SOURCE: one emitter per look; co-channel multipath "
                         "resolution needs a digital array")
        else:
            lines.append(f"up to {cap['max_sources']} simultaneous source(s) per channel")
        lines.append(f"elevation: {cap['elevation_note']}")
        lines.append(f"LO: {self.lo_sharing.replace('_', ' ')}; receiver calibration "
                     f"good {self.rx_cal_persistence.replace('_', ' ')}")
        lines.append(f"timing {self.timing.get('tier')}: {TIER_WORDS.get(self.timing.get('tier'), '?')}")
        if cap["fov_az_deg"] is not None:
            lo, hi = cap["fov_az_deg"]
            lines.append(f"field of view {lo:.0f}° to {hi:.0f}° relative")
        return lines

    # -- validation --------------------------------------------------------------
    def problems(self) -> tuple:
        """(errors, warnings): errors make the profile unusable."""
        err, warn = [], []
        if int(self.schema) != SCHEMA:
            err.append(f"schema {self.schema} is not {SCHEMA}")
        if not _ID_RE.match(str(self.id or "")):
            err.append("id must be 1–64 letters, digits, '_', '-', '.'")
        for name, val, allowed in (("architecture", self.architecture, ARCHITECTURES),
                                   ("lo_sharing", self.lo_sharing, LO_SHARING),
                                   ("rx_cal_persistence", self.rx_cal_persistence, PERSISTENCE),
                                   ("timing.tier", self.timing.get("tier"), TIMING_TIERS),
                                   ("cable_class", self.cable_class, CABLE_CLASSES)):
            if val not in allowed:
                err.append(f"{name} {val!r} is not one of {', '.join(allowed)}")
        lo, hi = self.band_mhz if len(self.band_mhz) == 2 else (0, 0)
        if not 0 < lo < hi:
            err.append("band_mhz must be (low, high) with 0 < low < high")
        rx_m = (self.calibration.get("rx_table") or {}).get("method", "none")
        ant_m = (self.calibration.get("antenna_table") or {}).get("method", "none")
        if rx_m not in RX_CAL_METHODS:
            err.append(f"calibration.rx_table.method {rx_m!r} is not one of {', '.join(RX_CAL_METHODS)}")
        if ant_m not in ANT_CAL_METHODS:
            err.append(f"calibration.antenna_table.method {ant_m!r} is not one of "
                       f"{', '.join(ANT_CAL_METHODS)}")
        try:
            g = self.geometry()
        except Exception as exc:                              # noqa: BLE001
            err.append(f"elements: {exc}")
            return err, warn
        n = g.n
        ch = int(self.rf_chains) if self.rf_chains else None

        # LO sharing and persistence (lit2 §3.18)
        if self.lo_sharing == "independent" and self.rx_cal_persistence != "per_retune":
            err.append("independent LOs: the receiver calibration cannot outlive a retune "
                       "(rx_cal_persistence must be per_retune)")
        if (self.lo_sharing == "independent" and self.architecture != "switched"
                and rx_m in ("none",)):
            err.append("independent LOs need a per-retune receiver calibration source "
                       "(noise_source, tone or known_emitter)")
        if self.timing.get("tier") == "T4" and self.lo_sharing == "independent":
            err.append("timing tier T4 IS a shared LO; lo_sharing cannot be independent")
        band = LO_SHARE_BAND_MHZ.get(self.receiver)
        if band and self.lo_sharing != "independent" and lo > 0 and (lo < band[0] or hi > band[1]):
            warn.append(f"{self.receiver} can share its LO only over {band[0]:.0f}–"
                        f"{band[1]:.0f} MHz; outside that its channels are independent")
        if self.timing.get("tier") == "T5":
            warn.append("T5 (White Rabbit) gives cross-site coherence at VHF only, for "
                        "fixed fibre-linked sites (lit2 §4.8)")

        # architecture rules (§11.4.1)
        a = self.architecture
        if a == "digital":
            if ch is not None and ch != n:
                err.append(f"a digital array has one channel per element: rf_chains {ch} ≠ {n}")
        elif a == "switched":
            sw = self.switching or {}
            if ch is None or ch >= n:
                err.append("a switched array has fewer channels than elements (rf_chains < elements)")
            ref = sw.get("reference_element")
            refch = sw.get("reference_channel", 0)
            sched = sw.get("schedule")
            if ref is None or not 0 <= int(ref) < n:
                err.append("switching.reference_element must be an element index")
            if ch is not None and not 0 <= int(refch) < ch:
                err.append("switching.reference_channel must be a channel index")
            if not sched:
                err.append("switching.schedule must list, per slot, the element on each "
                           "non-reference channel")
            elif ch is not None and ref is not None:
                seen = set()
                for i, slot in enumerate(sched):
                    if len(slot) != ch - 1:
                        err.append(f"switching.schedule slot {i} lists {len(slot)} elements; "
                                   f"{ch - 1} non-reference channels need one each")
                        break
                    seen.update(int(x) for x in slot)
                missing = set(range(n)) - seen - {int(ref)}
                if missing:
                    err.append(f"switching.schedule never visits elements {sorted(missing)}")
            if rx_m not in ("reference_channel", "noise_source", "tone", "known_emitter"):
                err.append("a switched array calibrates against its reference channel "
                           "(calibration.rx_table.method reference_channel)")
            warn.append("switched: single-source, and the signal must outlast the switch "
                        "sequence — a fixed-site / repeater-watch mode, not a handset-burst "
                        "mode (lit2 §2.14)")
        elif a == "analog":
            st = self.steering or {}
            if ch not in (1, 2):
                err.append("an analog array has one combined output (beam-scan) or two "
                           "(halves, for monopulse): rf_chains 1 or 2")
            if not st:
                err.append("an analog array needs steering (phase_step_deg, gain_step_db, "
                           "switch_time_us)")
            if ch == 2:
                halves = st.get("halves")
                if not halves or len(halves) != 2:
                    err.append("two analog outputs need steering.halves: the two element "
                               "groups the outputs combine")
                else:
                    h0, h1 = set(map(int, halves[0])), set(map(int, halves[1]))
                    if h0 & h1:
                        err.append("steering.halves overlap")
                    if (h0 | h1) - set(range(n)):
                        err.append("steering.halves name elements that do not exist")
        elif a == "hybrid":
            subs = self.subarrays or []
            if ch is None:
                err.append("a hybrid array must set rf_chains (the number of coherent "
                           "channels its subarrays feed)")
            if not subs:
                err.append("a hybrid array needs subarrays: the elements feeding each channel")
            else:
                members = [set(int(x) for x in (s.get("members") if isinstance(s, dict) else s))
                           for s in subs]
                if ch is not None and len(members) != ch:
                    err.append(f"{len(members)} subarrays for {ch} channels")
                allm = set()
                for mset in members:
                    if mset & allm:
                        err.append("subarrays overlap (each element feeds one channel)")
                        break
                    allm |= mset
                if allm - set(range(n)):
                    err.append("subarrays name elements that do not exist")
                if len(members) < 2:
                    err.append("a hybrid array needs at least two subarrays (one is analog)")
            if not self.steering:
                err.append("a hybrid array needs steering (phase_step_deg, gain_step_db)")

        # geometry sense
        if hi > 0:
            note = g.check(hi * 1e6, override=True)
            if note:
                warn.append(note)
        if g.kind == "multi_face" and self.lo_sharing == "independent":
            warn.append("independent faces: each face is DF'd on its own, the face hand-off "
                        "by received power (no cross-face coherence)")
        if a == "analog" and not (self.field_of_view_deg or {}).get("az"):
            warn.append("analog array: field of view defaulted to ±60° about boresight")
        return err, warn

    def validate(self) -> list:
        """Raise ProfileError listing every error; return the warnings."""
        err, warn = self.problems()
        if err:
            raise ProfileError(self.id, err)
        for w in warn:
            log.info("array profile %s: %s", self.id, w)
        return warn


def replace_id(g: "_m.Geometry", new_id: str) -> "_m.Geometry":
    from dataclasses import replace
    return replace(g, id=new_id)


# ---------------------------------------------------------------------------
# the Kraken's existing settings → a profile
# ---------------------------------------------------------------------------

def from_legacy_cfg(cfg: dict, n: int = 5) -> ArrayProfile:
    """ATK's flat settings["df"] (array, radius_m, spacing_m, channel map —
    the Kraken today) as a schema-v2 profile, so the Kraken goes through the
    same path as everything else. The channel map passed in is already the
    element→channel map (ATK applies its cw/ccw numbering first)."""
    cmap = cfg.get("channel_of_element")
    if str(cfg.get("array", "uca")).lower() == "ula":
        elements = {"preset": "ula", "n": n, "spacing_m": float(cfg.get("spacing_m") or 0.30)}
    else:
        elements = {"preset": "uca", "n": n, "radius_m": float(cfg.get("radius_m") or 0.18)}
    if cmap:
        elements["channel_of_element"] = [int(c) for c in cmap]
    p = preset("kraken5_uca")
    p.elements = elements
    p.id = "kraken5_" + elements["preset"]
    return p


# ---------------------------------------------------------------------------
# presets — each a starting point, not a measured array
# ---------------------------------------------------------------------------

def _p(**kw) -> ArrayProfile:
    return ArrayProfile(**kw)


def _presets() -> dict:
    lam800 = 299_792_458.0 / 870e6                 # half-wavelength at the top of the band
    d800 = math.floor(lam800 / 2.0 * 1000) / 1000  # 0.172 m: ≤ λ/2 at 870 MHz
    out = {}
    out["kraken5_uca"] = _p(
        id="kraken5_uca", description="KrakenSDR, five whips on a 0.18 m circle (ATK's default)",
        architecture="digital", receiver="krakensdr", lo_sharing="independent",
        rx_cal_persistence="per_retune", timing={"tier": "T0"}, band_mhz=(24.0, 1766.0),
        elements={"preset": "uca", "n": 5, "radius_m": 0.18},
        calibration={"rx_table": {"method": "noise_source", "cache_key": "lo_hz"},
                     "antenna_table": {"method": "known_emitter_drive", "per": "band"}},
        cable_class="unknown",
        notes="Five R820T2 tuners on one reference: phase is re-measured on the noise "
              "source after every retune (independent LOs). The calibration ends at the "
              "switch; cables, whips and roof are the antenna table's job.")
    out["x310_twinrx_uca4"] = _p(
        id="x310_twinrx_uca4",
        description="USRP X310 + 2×TwinRX, four whips on a circle (vehicle, 360°)",
        architecture="digital", receiver="usrp_x310_twinrx", lo_sharing="shared_within_device",
        rx_cal_persistence="across_power_cycles",
        timing={"tier": "T2", "t_utc_sigma_ns": 50, "holdover_spec": "x3x0_ocxo"},
        band_mhz=(380.0, 870.0), elements={"preset": "uca", "n": 4, "radius_m": 0.12},
        calibration={"rx_table": {"method": "tone", "cache_key": "lo_hz"},
                     "antenna_table": {"method": "known_emitter_drive", "per": "band"}},
        cable_class="phase_stable",
        notes="TwinRX shares its LO across both daughterboards: ≤ ±0.08° between "
              "channels, stable across power cycles (lit2 §3.18), so a cached "
              "per-frequency receiver table is viable. Four elements: 3 sources at most.")
    out["x310_twinrx_panel4_v800"] = _p(
        id="x310_twinrx_panel4_v800",
        description="USRP X310 + 2×TwinRX, a 1×4 vertical panel facing forward, 700/800 MHz",
        architecture="digital", receiver="usrp_x310_twinrx", lo_sharing="shared_within_device",
        rx_cal_persistence="across_power_cycles",
        timing={"tier": "T2", "t_utc_sigma_ns": 50, "holdover_spec": "x3x0_ocxo"},
        band_mhz=(764.0, 870.0),
        elements={"preset": "ura", "rows": 1, "cols": 4, "spacing_m": d800},
        mount={"plane": "vertical", "boresight_deg": 0.0, "tilt_deg": 0.0},
        field_of_view_deg={"az": [-60.0, 60.0]},
        pattern={"kind": "cardioid", "exponent": 1.0, "back_db": -20.0},
        calibration={"rx_table": {"method": "tone", "cache_key": "lo_hz"},
                     "antenna_table": {"method": "tripod", "per": "band"}},
        cable_class="phase_stable",
        notes="A row across the face measures azimuth in front only (±60°); no elevation.")
    out["n321x4_ura2x4_v800"] = _p(
        id="n321x4_ura2x4_v800",
        description="Four USRP N321 on the LO-distribution star, a 2×4 vertical panel, 700/800 MHz",
        architecture="digital", receiver="usrp_n321", lo_sharing="distributed",
        rx_cal_persistence="per_power_cycle",
        timing={"tier": "T4", "t_utc_sigma_ns": 8, "holdover_spec": "n3xx_gpsdo"},
        band_mhz=(764.0, 870.0),
        elements={"preset": "ura", "rows": 2, "cols": 4, "spacing_m": d800},
        mount={"plane": "vertical", "boresight_deg": 0.0, "tilt_deg": 0.0},
        field_of_view_deg={"az": [-60.0, 60.0], "el": [-10.0, 45.0]},
        pattern={"kind": "cardioid", "exponent": 1.0, "back_db": -20.0},
        calibration={"rx_table": {"method": "tone", "cache_key": "lo_hz"},
                     "antenna_table": {"method": "tripod", "per": "band"}},
        cable_class="phase_stable",
        notes="N321 LO distribution: < 0.1° between modules at 25 °C, deterministic on "
              "retune — but only 450–6000 MHz (no LO-shared VHF on N3xx, lit2 §4.6). "
              "Two rows give elevation near the horizon.")
    out["b210_switched_uca8c"] = _p(
        id="b210_switched_uca8c",
        description="USRP B210, eight-element ring + centre reference through a 1:8 switch",
        architecture="switched", receiver="usrp_b210", lo_sharing="shared_within_device",
        rx_cal_persistence="per_retune",
        timing={"tier": "T2", "t_utc_sigma_ns": 50, "holdover_spec": "b2x0_tcxo"},
        band_mhz=(380.0, 870.0),
        elements={"preset": "uca", "n": 8, "radius_m": 0.17, "centre": True},
        rf_chains=2,
        switching={"reference_element": 8, "reference_channel": 0,
                   "schedule": [[0], [1], [2], [3], [4], [5], [6], [7]],
                   "dwell_samples": 2048, "settle_us": 5.0},
        calibration={"rx_table": {"method": "reference_channel", "cache_key": "lo_hz"},
                     "antenna_table": {"method": "known_emitter_drive", "per": "band"}},
        notes="The ring element on channel 1 is measured against the centre element on "
              "channel 0 in every slot, so receiver offsets cancel. ≈ 5° class "
              "(lit2 §2.14); single-source; the signal must outlast the eight slots.")
    out["phaser_cn0566_xband"] = _p(
        id="phaser_cn0566_xband",
        description="ADI CN0566 'Phaser' (8 elements, ADAR1000, two outputs into a Pluto) — "
                    "X-band learning kit, not P25",
        architecture="analog", receiver="pluto_cn0566", lo_sharing="shared_within_device",
        rx_cal_persistence="per_power_cycle", timing={"tier": "T0"},
        band_mhz=(10000.0, 10500.0),
        elements={"preset": "ura", "rows": 1, "cols": 8, "spacing_m": 0.0142},
        mount={"plane": "vertical", "boresight_deg": 0.0, "tilt_deg": 0.0},
        field_of_view_deg={"az": [-60.0, 60.0]},
        pattern={"kind": "cardioid", "exponent": 2.0, "back_db": -20.0},
        rf_chains=2,
        steering={"phase_step_deg": 2.8125, "gain_step_db": 0.5, "switch_time_us": 1.0,
                  "halves": [[0, 1, 2, 3], [4, 5, 6, 7]],
                  "beams_deg": [float(x) for x in range(-60, 61, 5)]},
        calibration={"rx_table": {"method": "tone", "cache_key": "lo_hz"},
                     "antenna_table": {"method": "tripod", "per": "band"}},
        notes="No channel-to-channel match spec: gain normalised to the weakest element, "
              "phase calibrated pairwise (ADI's own procedure, lit2 §1.7). Beam-scan and "
              "monopulse; single-source.")
    out["hybrid_ura4x4_4sub_v800"] = _p(
        id="hybrid_ura4x4_4sub_v800",
        description="Example hybrid: 4×4 vertical panel, four 2×2 analog subarrays into four "
                    "coherent channels, 700/800 MHz",
        architecture="hybrid", receiver="usrp_x310_twinrx", lo_sharing="shared_within_device",
        rx_cal_persistence="across_power_cycles",
        timing={"tier": "T2", "t_utc_sigma_ns": 50, "holdover_spec": "x3x0_ocxo"},
        band_mhz=(764.0, 870.0),
        elements={"preset": "ura", "rows": 4, "cols": 4, "spacing_m": d800},
        mount={"plane": "vertical", "boresight_deg": 0.0, "tilt_deg": 0.0},
        field_of_view_deg={"az": [-60.0, 60.0], "el": [-10.0, 45.0]},
        pattern={"kind": "cardioid", "exponent": 1.0, "back_db": -20.0},
        rf_chains=4,
        subarrays=[{"members": [0, 1, 4, 5]}, {"members": [2, 3, 6, 7]},
                   {"members": [8, 9, 12, 13]}, {"members": [10, 11, 14, 15]}],
        steering={"phase_step_deg": 5.625, "gain_step_db": 0.5, "switch_time_us": 1.0},
        calibration={"rx_table": {"method": "tone", "cache_key": "lo_hz"},
                     "antenna_table": {"method": "tripod", "per": "band"}},
        notes="Subarray phase centres 2 spacings apart: grating lobes the subarray "
              "pattern only partly suppresses — resolved by re-steering to the candidates "
              "(lit2 §1.11). At most 3 sources per look.")
    out["mast4x4_independent_faces"] = _p(
        id="mast4x4_independent_faces",
        description="Four vertical 1×4 faces round a mast, each on its own receiver (no "
                    "shared LO): face hand-off by power",
        architecture="digital", receiver="usrp_x310_twinrx", lo_sharing="independent",
        rx_cal_persistence="per_retune",
        timing={"tier": "T2", "t_utc_sigma_ns": 50, "holdover_spec": "x3x0_ocxo"},
        band_mhz=(764.0, 870.0),
        elements={"preset": "multi_face", "faces": 4, "rows": 1, "cols": 4,
                  "spacing_m": d800, "standoff_m": 0.25},
        mount={"plane": "vertical", "boresight_deg": 0.0, "tilt_deg": 0.0},
        pattern={"kind": "cardioid", "exponent": 1.0, "back_db": -20.0},
        calibration={"rx_table": {"method": "tone", "cache_key": "lo_hz"},
                     "antenna_table": {"method": "tripod", "per": "band"}},
        notes="Each face is coherent within itself only; faces are compared by power and "
              "the strongest face's bearing is used.")
    return out


def preset_ids() -> tuple:
    return tuple(_presets().keys())


def preset(name: str) -> ArrayProfile:
    ps = _presets()
    if name not in ps:
        raise KeyError(f"no array preset {name!r}; one of {', '.join(ps)}")
    return ps[name]


def write_presets(folder, overwrite: bool = False) -> list:
    """Drop every preset as <id>.json into `folder` (ATK: config/df_arrays,
    beside the code — never C:\\ or AppData). Existing files are left alone
    unless `overwrite`."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for pid, prof in _presets().items():
        p = folder / f"{pid}.json"
        if p.exists() and not overwrite:
            continue
        prof.save(p)
        written.append(p)
    return written


def load_dir(folder) -> tuple:
    """({id: profile}, {filename: error}) for every *.json in `folder`.
    A broken file is reported and skipped, never fatal."""
    good, bad = {}, {}
    folder = Path(folder)
    if not folder.is_dir():
        return good, bad
    for p in sorted(folder.glob("*.json")):
        try:
            prof = ArrayProfile.load(p)
            prof.validate()
            if prof.id in good:
                raise ProfileError(prof.id, [f"id already used by {Path(good[prof.id].source_path).name}"])
            good[prof.id] = prof
        except Exception as exc:                              # noqa: BLE001
            bad[p.name] = str(exc)
            log.warning("array profile %s skipped: %s", p.name, exc)
    return good, bad
