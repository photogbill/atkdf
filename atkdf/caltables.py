"""Two calibration tables, a persistence policy, and the recalibration trigger
(plan §11.4.4).

THE TWO TABLES (lit2 §3.13, §3.17). A 4-channel USRP ULA went 151° → 58° max
RMSE with the receiver step alone and → 3.35° with the receiver AND antenna
steps: they are different errors with different lifetimes, so they are
different files.

  * the RECEIVER table — one complex weight per channel at one LO frequency,
    from a noise source, a tone, a known emitter or (switched arrays) the
    reference channel. Its lifetime is the hardware's (below).
  * the ANTENNA table — the measured manifold, per band
    (`manifold.MeasuredManifold`, with phase centres removed so it
    interpolates across the band). Its lifetime is the installation's: it
    changes when the antennas, cables or roof change, not when the radio
    retunes.

THE PERSISTENCE POLICY (lit2 §3.18, §4.6, §6.3) — `rx_cal_persistence`:

  across_power_cycles   X310 + TwinRX, NI-2955: LO shared in the box, the
                        table survives a reboot and a retune back to that
                        frequency. Cached per LO frequency until the on-air
                        monitor says it moved.
  per_power_cycle       N320 / N321 class, retune-DETERMINISTIC LO: good for
                        this power-up only, but stable across retunes within
                        it, so a table taken at an LO frequency is reused when
                        the radio returns to it in the same session. (An LO
                        that does NOT come back the same on retune — the N310
                        external-LO path, which can flip 180° — is per_retune,
                        not this; do not mark such hardware per_power_cycle.)
  per_retune            independent LOs (the Kraken, two B210s): every retune
                        needs a calibration. "Calibration caching across
                        retunes" (CUED_DF_PLAN) is honoured exactly where the
                        hardware allows it, and nowhere else.

THE TRIGGER. No paper gives a drift rate in °/h or °/°C for any USRP array;
the measured windows are 20–40 s (lit2 §3.18, §4.13). So the trigger is the
on-air `calibration.CoherenceMonitor` bound — the reference's steering vector
moved — not a timer, plus a thermal warning for PTFE cable (≈ 3.8°/m over
0–55 °C at 1 GHz, lit2 §4.7) when the temperature has moved enough to matter.

Files live beside the code (ATK: `D:\\Analyst_Toolkit\\ATK\\calibration\\…`),
never C:\\ or AppData; every table names the session it was taken in.
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .calibration import CoherenceMonitor

log = logging.getLogger("atkdf.caltables")

#: PTFE coax phase: 3.8° per metre over a 55 °C swing at 1 GHz (lit2 §4.7),
#: as degrees per metre per °C per GHz. Phase-stable cable is taken as a
#: tenth of it; "unknown" as PTFE (the conservative assumption).
CABLE_DEG_PER_M_C_GHZ = {"ptfe": 3.8 / 55.0, "phase_stable": 0.38 / 55.0,
                         "unknown": 3.8 / 55.0}

LO_KEY_HZ = 1000.0          # tables are keyed to the nearest kHz of LO


def lo_key(lo_hz: float) -> int:
    return int(round(float(lo_hz) / LO_KEY_HZ))


def thermal_phase_deg(cable_m: float, f_hz: float, delta_c: float,
                      cable_class: str = "unknown") -> float:
    """Worst-case phase a cable run of `cable_m` moves for a `delta_c`
    temperature change at `f_hz` — the differential between two cables can
    be up to this when they do not see the same temperature."""
    c = CABLE_DEG_PER_M_C_GHZ.get(cable_class, CABLE_DEG_PER_M_C_GHZ["unknown"])
    return abs(c * float(cable_m) * (float(f_hz) / 1e9) * float(delta_c))


@dataclass
class RxCalTable:
    """One receiver calibration: `weights[c]` multiplies channel c so the
    channels agree (channel 0 = 1 by convention)."""

    profile_id: str
    lo_hz: float
    weights: np.ndarray
    method: str = "tone"
    created_utc: float = 0.0
    session_id: str = ""             # the power cycle it was taken in
    retune_seq: int = 0              # the store's retune counter when taken
    passport_uid: str = ""
    residual_deg: float | None = None
    temperature_c: float | None = None
    uid: str = ""

    def __post_init__(self):
        self.weights = np.asarray(self.weights, dtype=complex).reshape(-1)
        self.created_utc = float(self.created_utc or time.time())
        self.uid = self.uid or uuid.uuid4().hex[:12]

    @classmethod
    def from_measurement(cls, profile_id: str, lo_hz: float, vectors, method: str = "tone",
                         **kw) -> "RxCalTable":
        """From the channels' response to ONE common stimulus (a tone split to
        every channel, the noise source, or a known emitter's steering vector
        divided by the antenna table's): each row of `vectors` (or the single
        vector) is what the channels measured; the weight makes every channel
        read like channel 0. Several rows are averaged after aligning their
        common phase."""
        V = np.atleast_2d(np.asarray(vectors, dtype=complex))
        V = V / np.where(np.abs(V[:, :1]) > 0, V[:, :1], 1.0)          # channel 0 = 1
        mean = V.mean(axis=0)
        resid = float(np.degrees(np.sqrt(np.mean(np.angle(V / mean) ** 2)))) if V.shape[0] > 1 else None
        w = 1.0 / np.where(np.abs(mean) > 1e-12, mean, 1e-12)
        return cls(profile_id, float(lo_hz), w, method, residual_deg=resid, **kw)

    def apply(self, x) -> np.ndarray:
        """Calibrated channels: (channels, N) × weights."""
        x = np.asarray(x)
        if x.shape[0] != self.weights.size:
            raise ValueError(f"table has {self.weights.size} channels, data {x.shape[0]}")
        return x * self.weights[:, None]

    def to_dict(self) -> dict:
        return {"profile_id": self.profile_id, "lo_hz": self.lo_hz,
                "weights": [[float(w.real), float(w.imag)] for w in self.weights],
                "method": self.method, "created_utc": self.created_utc,
                "session_id": self.session_id, "retune_seq": self.retune_seq,
                "passport_uid": self.passport_uid, "residual_deg": self.residual_deg,
                "temperature_c": self.temperature_c, "uid": self.uid, "kind": "rx_table"}

    @classmethod
    def from_dict(cls, d: dict) -> "RxCalTable":
        w = np.array([complex(a, b) for a, b in d["weights"]])
        return cls(d["profile_id"], float(d["lo_hz"]), w, d.get("method", "tone"),
                   float(d.get("created_utc", 0.0)), d.get("session_id", ""),
                   int(d.get("retune_seq", 0)), d.get("passport_uid", ""),
                   d.get("residual_deg"), d.get("temperature_c"), d.get("uid", ""))

    def filename(self) -> str:
        return f"rxcal_{self.profile_id}_{lo_key(self.lo_hz)}kHz_{self.uid}.json"


class CalStore:
    """The receiver tables of one array, with its persistence policy.

        store = CalStore(folder, profile)
        store.begin_power_cycle()          # once per device power-up / session
        store.note_retune(lo_hz)           # every retune
        table, why = store.lookup(lo_hz)   # None → calibrate now
        store.put(RxCalTable.from_measurement(...))
        store.observe_reference(lo_hz, vector)   # on-air monitor: may invalidate

    `folder` = where tables are written (one JSON per table, never
    overwritten — the audit trail); None keeps them in memory only."""

    def __init__(self, folder, profile, threshold_deg: float = 5.0, persist: int = 3,
                 thermal_threshold_deg: float = 2.0, cable_m: float | None = None):
        self.folder = Path(folder) if folder else None
        self.profile = profile
        self.policy = profile.rx_cal_persistence
        self.session_id = ""
        self.retune_seq = 0
        self.lo_hz = None
        self.tables: dict = {}                 # lo_key -> RxCalTable (latest)
        self.invalid: dict = {}                # lo_key -> reason
        self.monitors: dict = {}               # lo_key -> CoherenceMonitor
        self.threshold_deg = float(threshold_deg)
        self.persist = int(persist)
        self.thermal_threshold_deg = float(thermal_threshold_deg)
        rx = (profile.calibration or {}).get("rx_table") or {}
        self.cable_m = float(cable_m if cable_m is not None else rx.get("cable_m", 3.0))
        self.temperature_c = None
        if self.folder is not None:
            self.folder.mkdir(parents=True, exist_ok=True)
            self._load_disk()

    # -- lifecycle -------------------------------------------------------------
    def begin_power_cycle(self, session_id: str | None = None) -> str:
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.retune_seq = 0
        log.info("calstore %s: power cycle %s (policy %s)", self.profile.id, self.session_id,
                 self.policy)
        return self.session_id

    def note_retune(self, lo_hz: float) -> None:
        if self.lo_hz is None or lo_key(lo_hz) != lo_key(self.lo_hz):
            self.retune_seq += 1
            log.debug("calstore %s: retune %d → %.6f MHz", self.profile.id, self.retune_seq,
                      lo_hz / 1e6)
        self.lo_hz = float(lo_hz)

    def note_temperature(self, celsius: float) -> None:
        self.temperature_c = float(celsius)

    # -- tables ------------------------------------------------------------------
    def put(self, table: RxCalTable) -> RxCalTable:
        table.session_id = table.session_id or self.session_id
        table.retune_seq = self.retune_seq
        if table.temperature_c is None:
            table.temperature_c = self.temperature_c
        k = lo_key(table.lo_hz)
        self.tables[k] = table
        self.invalid.pop(k, None)
        self.monitors.pop(k, None)
        if self.folder is not None:
            p = self.folder / table.filename()
            with p.open("w", encoding="utf-8", newline="\n") as f:
                json.dump(table.to_dict(), f, indent=1)
        log.info("calstore %s: rx table %s at %.6f MHz (%s, residual %s)", self.profile.id,
                 table.uid, table.lo_hz / 1e6, table.method, table.residual_deg)
        return table

    def lookup(self, lo_hz: float) -> tuple:
        """(table | None, why). None means: calibrate before direction-finding
        at this LO."""
        k = lo_key(lo_hz)
        t = self.tables.get(k)
        if t is None:
            return None, f"no receiver table at {lo_hz / 1e6:.3f} MHz"
        if k in self.invalid:
            return None, f"the table at {lo_hz / 1e6:.3f} MHz was invalidated: {self.invalid[k]}"
        if self.policy == "per_retune" and t.retune_seq != self.retune_seq:
            return None, ("independent LOs: the receivers have retuned since that table "
                          "was taken — calibrate again")
        if self.policy in ("per_retune", "per_power_cycle") and t.session_id != self.session_id:
            return None, ("the table is from another power cycle and this hardware does not "
                          "keep its LO phase across one")
        due, phase = self.thermal_due(t, lo_hz)
        if due:
            return t, (f"usable, but the temperature has moved enough to shift cable phase "
                       f"by up to {phase:.1f}° — recalibrate when convenient")
        return t, "usable"

    def thermal_due(self, table: RxCalTable, f_hz: float) -> tuple:
        if table.temperature_c is None or self.temperature_c is None:
            return False, 0.0
        ph = thermal_phase_deg(self.cable_m, f_hz, self.temperature_c - table.temperature_c,
                               self.profile.cable_class)
        return ph > self.thermal_threshold_deg, ph

    def invalidate(self, lo_hz: float, reason: str) -> None:
        self.invalid[lo_key(lo_hz)] = reason
        log.warning("calstore %s: table at %.6f MHz invalidated: %s", self.profile.id,
                    lo_hz / 1e6, reason)

    # -- the on-air trigger --------------------------------------------------------
    def observe_reference(self, lo_hz: float, vector, stationary: bool = True) -> dict:
        """Feed a parked reference's steering vector (after the receiver
        table). When it moves past the bound, the table at this LO is
        invalidated — recalibrate when the reference says so, not on a timer."""
        if not stationary:
            return {"state": "ignored", "why": "the platform is moving"}
        k = lo_key(lo_hz)
        t = self.tables.get(k)
        mon = self.monitors.setdefault(k, CoherenceMonitor(self.threshold_deg, self.persist))
        out = mon.observe(vector, t.uid if t else "")
        if mon.needs_cal and k not in self.invalid:
            self.invalidate(lo_hz, mon.sentence())
        return out

    # -- disk ------------------------------------------------------------------------
    def _load_disk(self) -> None:
        n = 0
        for p in sorted(self.folder.glob(f"rxcal_{self.profile.id}_*.json")):
            try:
                with p.open("r", encoding="utf-8") as f:
                    t = RxCalTable.from_dict(json.load(f))
            except Exception as exc:                          # noqa: BLE001
                log.warning("calstore: unreadable table %s: %s", p.name, exc)
                continue
            k = lo_key(t.lo_hz)
            if k not in self.tables or t.created_utc > self.tables[k].created_utc:
                self.tables[k] = t
                n += 1
        if n:
            log.info("calstore %s: %d receiver table(s) from %s", self.profile.id, n, self.folder)
