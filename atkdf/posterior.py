"""The per-emitter posterior: a probability grid over the operating area.

A bearing does not give a point; it gives a line, and a handful of bearings
from different places give an overlap. The posterior is that overlap made
explicit: a log-probability grid where each bearing multiplies in its FULL
360-point likelihood (not just its peak), so multipath images and cone
ambiguities survive as competing modes and no single confident-but-wrong
bearing can zero out the truth (umbrella §8.4).

Each cell has a true bearing from the receiver's position; the record's
likelihood — already rotated into absolute bearings and already widened by the
heading σ — is sampled at that bearing and added in the log domain. Terrain and
range likelihoods (Phase 3) multiply in the same way.

Ageing (umbrella §8.5): when a radio goes quiet its posterior is not frozen, it
DIFFUSES by the emitter's motion state — stationary stays put, on foot spreads
at a walking pace over free ground, a vehicle spreads faster — so "473 was here
forty minutes ago on foot" becomes a search area of the right size that shrinks
the moment 473 keys up again.

numpy, log-domain float32; a 50 km area at 25 m is 4M cells ≈ 16 MB, and a
hundred emitters is what the RAM is for. `fusion` summarises a grid to the
ellipse; this is the multimodal view under it.

TERRAIN: THE SHADOW LIKELIHOOD (umbrella §8.6; propagation build plan §6.1,
built 2026-11-16). We HEARD the emitter from here. So every cell from which an
emitter of its power class could not have been heard is less likely. For each
cell the loss field (`ports.LossField`: atkprop over real terrain in ATK, or a
fake) gives the path loss to the receiver; the chance an emitter there is heard
is the chance its EIRP, less that loss, clears the receiver's sensitivity —
marginalised over the class's EIRP range, with location shadowing (and the
terrain's own coarseness) as a normal spread in dB. It needs NO calibrated
received level, which is why it comes before the range likelihood. It is
applied once per place heard from (`ShadowModel.repeat_m`): twenty bearings
from one parked spot are twenty bearings, but only one "heard from here".
A shadowed cell is DOWN-WEIGHTED, never zeroed (`floor`): at 450 MHz a ridge
is 10–30 dB, not silence, and the model can be wrong.

HOW IT IS JUDGED. Not by the region shrinking: by the region shrinking AND
the stated region still holding the emitter at the stated rate
(`region`, `contains`, `coverage_interval`). A likelihood that tightens the
region and breaks containment is a regression.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from . import geodesy as geo
from .records import BearingRecord

#: metres a target of each motion state can move per second, for ageing.
DIFFUSE_MPS = {"stationary": 0.0, "on_foot": 1.2, "vehicle": 13.4, "unknown": 3.0}


@dataclass(frozen=True)
class PowerClass:
    """What kind of transmitter it might be: its EIRP range in dBm (transmitter
    power plus antenna gain, less body or feeder loss) and the height of its
    antenna above the ground."""

    name: str
    eirp_dbm: tuple
    h_tx_m: float


#: Umbrella §8.7. Handheld: 1–5 W (30–37 dBm) through an antenna and a body
#: that lose 0–10 dB. Mobile: 25–50 W (44–47 dBm) on a roof antenna, 0–3 dBi.
#: Base / repeater: 25–100 W (44–50 dBm), 3–9 dBi, 3 dB of feeder, on a mast.
POWER_CLASSES = {
    "handheld": PowerClass("handheld 1–5 W", (20.0, 37.0), 1.5),
    "mobile": PowerClass("vehicle mobile 25–50 W", (44.0, 50.0), 1.8),
    "base": PowerClass("base / repeater 25–100 W", (44.0, 56.0), 30.0),
}

#: Which classes a bearing's path tag allows. A DIRECT burst is a subscriber
#: (a handheld or a mobile); UNKNOWN could be anything. REPEATED never reaches
#: here: it points at the repeater, not the emitter.
PATH_TAG_CLASSES = {"direct": ("handheld", "mobile"),
                    "unknown": ("handheld", "mobile", "base")}


@dataclass
class ShadowModel:
    """The receiver side of "could it have been heard from there".

    `noise_figure_db`, `snr_needed_db`  the sensitivity, with the burst's
                    bandwidth: S = −174 + 10·log10(B) + NF + SNR − G + L dBm
    `rx_gain_dbi`, `rx_loss_db`  the array element and its cable
    `shadow_sigma_db`  location variability around the predicted loss (VHF/UHF
                    land mobile: 5.6–8 dB; 8 is the cautious end)
    `res_sigma_db`  extra spread at 1 km terrain posts, scaled by the field's
                    post spacing — DTED0's 900 m cannot place a shadow edge
    `floor`         the least a cell's likelihood may become (never zero)
    `h_rx_m`        the receiving antenna above the ground
    `repeat_m`      a place heard from counts once; a new one must be this
                    far from every earlier one
    `eirp_points`   how finely the class's EIRP range is integrated
    """

    noise_figure_db: float = 6.0
    snr_needed_db: float = 6.0
    rx_gain_dbi: float = 0.0
    rx_loss_db: float = 1.0
    shadow_sigma_db: float = 8.0
    res_sigma_db: float = 6.0
    floor: float = 0.02
    h_rx_m: float = 2.0
    repeat_m: float = 250.0
    eirp_points: int = 9

    def sensitivity_dbm(self, bw_hz: float) -> float:
        b = max(float(bw_hz or 12_500.0), 100.0)
        return (-174.0 + 10.0 * math.log10(b) + self.noise_figure_db + self.snr_needed_db
                - self.rx_gain_dbi + self.rx_loss_db)

    def sigma_db(self, terrain_res_m: float) -> float:
        r = min(max(float(terrain_res_m or 30.0), 0.0), 1000.0) / 1000.0
        return math.hypot(self.shadow_sigma_db, self.res_sigma_db * r)


#: the shadow evidence is kept on a grid no finer than this many cells across
#: (memory: classes × EIRP points × side² float32 per emitter — 3 × 9 × 512² is
#: 28 MB); the field is upsampled from 250 m anyway, so nothing real is lost
TERRAIN_MAX_SIDE = 512


def _fit(a: np.ndarray, side: int) -> np.ndarray:
    """Crop or edge-pad a square array to `side` (stride slicing can be one short)."""
    a = a[:side, :side]
    if a.shape[0] < side or a.shape[1] < side:
        a = np.pad(a, ((0, side - a.shape[0]), (0, side - a.shape[1])), mode="edge")
    return a


def _expand(a: np.ndarray, stride: int, n: int) -> np.ndarray:
    """The stored grid back to the posterior's n x n (nearest)."""
    if stride == 1:
        return _fit(a, n)
    return _fit(np.repeat(np.repeat(a, stride, axis=0), stride, axis=1), n)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    from scipy.special import ndtr
    return ndtr(x)


def heard_probability(loss_db: np.ndarray, cls: PowerClass, sens_dbm: float,
                      sigma_db: float, points: int = 9) -> np.ndarray:
    """P(an emitter of class `cls` behind `loss_db` clears `sens_dbm`), its EIRP
    uniform in dB over the class's range and the shadowing normal in dB."""
    lo, hi = (float(v) for v in cls.eirp_dbm)
    eirps = np.linspace(lo, hi, max(1, int(points))) if hi > lo else np.array([lo])
    margin0 = -np.asarray(loss_db, dtype=np.float32) - np.float32(sens_dbm)
    acc = np.zeros(np.shape(margin0), dtype=np.float64)
    for e in eirps:
        acc += _norm_cdf((margin0 + e) / max(sigma_db, 1e-3))
    return (acc / len(eirps)).astype(np.float32)


@dataclass
class GridSpec:
    """A local ENU grid: a reference lat/lon, a half-size, and a cell size."""

    ref_lat: float
    ref_lon: float
    half_size_m: float = 25_000.0
    cell_m: float = 25.0

    @property
    def n(self) -> int:
        return int(2 * round(self.half_size_m / self.cell_m)) + 1

    def axes(self):
        h = (self.n // 2) * self.cell_m
        e = np.linspace(-h, h, self.n)
        return e, e.copy()          # east, north (metres from ref)


class Posterior:
    """One emitter's log-posterior over a GridSpec. Start uniform; `update`
    with each bearing; `age` between them; `fix` for the summary.

    A SHARED BEARING BIAS (`bias_bound_deg`, 2026-11-16). Every bearing from
    one array carries the same alignment error — compass deviation, a mount
    bolted on by eye — and multiplying bearings as if their errors were
    independent shrinks the region past the truth: twenty bearings each 2°
    wide do not make a 0.4° line when all twenty are 2.5° off together.
    `fusion.ml_fix` carries this as a bias bound (±3° for a field
    installation, `fusion.DEFAULT_BIAS_BOUND_DEG`). With `bias_bound_deg` set,
    the posterior does the same in its own way: `bias_points` hypotheses
    spread evenly over ±bound (a rectangular distribution, as fusion assumes),
    each bias applied to EVERY bearing, mixed with equal weight. It costs
    `bias_points` grids of memory; leave it None for a huge grid. One array
    = one bias; ageing a mover folds the hypotheses back into one belief.
    """

    def __init__(self, spec: GridSpec, bias_bound_deg: float | None = None,
                 bias_points: int = 7):
        self.spec = spec
        n = spec.n
        self.logp = np.zeros((n, n), dtype=np.float32)      # north (row) × east (col)
        if bias_bound_deg and bias_bound_deg > 0 and bias_points > 1:
            self.bias = np.linspace(-float(bias_bound_deg), float(bias_bound_deg),
                                    int(bias_points))
            self._logk = np.zeros((len(self.bias), n, n), dtype=np.float32)
        else:
            self.bias = None
            self._logk = None
        e, nth = spec.axes()
        self._E, self._N = np.meshgrid(e, nth)              # (n,n) metres from ref
        self.last_t = None
        self.n_updates = 0
        self._rx_cache: dict = {}
        self._heard_from: list = []          # ENU of the places a shadow update used
        self.terrain_updates = 0
        self.terrain_notes: list = []
        self._t_classes = None               # the shadow evidence (terrain_update)
        self._t_eirps = None
        self._t_logr = None
        self._t_term = None
        self._t_stride = 1

    def _cell_bearings(self, rx_lat: float, rx_lon: float) -> np.ndarray:
        """Absolute bearing (deg, 0..360, clockwise from north) from the
        receiver to every cell centre. Cached per rounded receiver position."""
        key = (round(rx_lat, 6), round(rx_lon, 6))
        hit = self._rx_cache.get(key)
        if hit is not None:
            return hit
        re, rn, _ = geo.geodetic_to_enu(rx_lat, rx_lon, 0.0,
                                        (self.spec.ref_lat, self.spec.ref_lon, 0.0))
        de = self._E - re
        dn = self._N - rn
        brg = (np.degrees(np.arctan2(de, dn))) % 360.0      # atan2(east, north)
        if len(self._rx_cache) > 32:
            self._rx_cache.clear()
        self._rx_cache[key] = brg
        return brg

    def update(self, rec: BearingRecord, min_lik: float = 1e-4) -> None:
        """Multiply in one bearing's absolute likelihood (log-add)."""
        lik = rec.absolute_likelihood()
        if lik is None or rec.lat is None or rec.lon is None:
            return
        if rec.path_tag == "repeated":
            return                       # points at the repeater, not the emitter
        brg = self._cell_bearings(rec.lat, rec.lon)
        if self.bias is None:
            idx = np.round(brg).astype(np.int64) % 360
            cell_lik = np.maximum(lik[idx], min_lik)
            self.logp += np.log(cell_lik).astype(np.float32)
            self.logp -= self.logp.max()     # keep it bounded; it is unnormalised
        else:
            # the same bias b on every bearing: the cell's true bearing seen as
            # (true + b), so the likelihood is read at (true + b)
            for k, b in enumerate(self.bias):
                idx = np.round(brg + b).astype(np.int64) % 360
                self._logk[k] += np.log(np.maximum(lik[idx], min_lik)).astype(np.float32)
            self._logk -= self._logk.max()       # one common shift keeps the mixture
            self._remix()
        self.last_t = rec.t_utc
        self.n_updates += 1

    # -- terrain ----------------------------------------------------------------
    def _remix(self) -> None:
        """logp = log mean_k exp(logk) (+ the terrain term) — the bias mixture."""
        mx = self._logk.max(axis=0)
        lse = mx + np.log(np.exp(self._logk - mx).mean(axis=0))
        self.logp = lse.astype(np.float32)
        if self._t_term is not None:
            self.logp += self._t_term
        self.logp -= self.logp.max()

    def terrain_update(self, rec: BearingRecord, field_port, model: ShadowModel | None = None,
                       classes: tuple | None = None, f_hz: float | None = None) -> dict:
        """Multiply in the shadow likelihood for one more place the emitter was
        heard from: the receiver position and frequency on `rec`, a field from
        `field_port` (`ports.LossField`). Returns what was done, in words —
        `applied` False with the reason when it was not (no position, no
        frequency, a repeated burst, a place already used, a field that failed).

        THE EIRP IS THE SAME EMITTER'S EVERY TIME. Heard from three places, a
        weak handheld must have cleared the sensitivity three times; treating
        each place on its own would count its unknown power three times and
        over-state the result. So the evidence is kept per class and per EIRP
        point and combined jointly: P(heard from every place | cell) =
        Σ_class w · mean_EIRP Π_place Φ(...). The shadowing is independent
        from place to place; the power is not.

        `classes` names entries of POWER_CLASSES (or gives PowerClass objects);
        by default the FIRST place's path tag decides, and the set is kept for
        the life of this evidence. Classes mix with equal weight, each with its
        own transmit height."""
        m = model or ShadowModel()
        if rec.path_tag == "repeated":
            return {"applied": False, "why": "a repeated burst points at the repeater"}
        if rec.lat is None or rec.lon is None:
            return {"applied": False, "why": "no receiver position"}
        f = float(f_hz or rec.fc_hz or 0.0)
        if f <= 0:
            return {"applied": False, "why": "no frequency"}
        re, rn, _ = geo.geodetic_to_enu(rec.lat, rec.lon, 0.0,
                                        (self.spec.ref_lat, self.spec.ref_lon, 0.0))
        for (pe, pn) in self._heard_from:
            if math.hypot(float(re) - pe, float(rn) - pn) < m.repeat_m:
                return {"applied": False,
                        "why": f"heard from here already (within {m.repeat_m:.0f} m)"}
        if self._t_classes is None:
            names = classes or PATH_TAG_CLASSES.get(rec.path_tag, PATH_TAG_CLASSES["unknown"])
            self._t_classes = [c if isinstance(c, PowerClass) else POWER_CLASSES[c] for c in names]
            self._t_eirps = [np.linspace(c.eirp_dbm[0], c.eirp_dbm[1], max(1, m.eirp_points))
                             if c.eirp_dbm[1] > c.eirp_dbm[0] else np.array([c.eirp_dbm[0]])
                             for c in self._t_classes]
            n = self.spec.n
            self._t_stride = max(1, int(math.ceil(n / float(TERRAIN_MAX_SIDE))))
            ns = len(range(0, n, self._t_stride))
            self._t_logr = [np.zeros((len(e), ns, ns), dtype=np.float32) for e in self._t_eirps]
            self._t_term = None
        sens = m.sensitivity_dbm(rec.bw_hz)
        fields: dict = {}
        res, model_name = [], ""
        new_terms = []
        for ci, c in enumerate(self._t_classes):
            hk = round(float(c.h_tx_m), 1)
            try:
                fr = fields.get(hk)
                if fr is None:
                    fr = fields[hk] = field_port.field((float(rec.lat), float(rec.lon)),
                                                       self.spec, f, m.h_rx_m, float(c.h_tx_m))
            except Exception as e:                        # noqa: BLE001
                return {"applied": False, "why": f"the loss field failed ({e})"}
            loss = np.asarray(fr.loss_db, dtype=np.float32)
            if loss.shape != self.logp.shape:
                return {"applied": False,
                        "why": f"field shape {loss.shape} is not the grid's {self.logp.shape}"}
            st = self._t_stride
            ls = loss[st // 2::st, st // 2::st] if st > 1 else loss
            ls = _fit(ls, self._t_logr[ci].shape[1])
            sig = max(m.sigma_db(fr.terrain_res_m), 1e-3)
            margin0 = -ls - np.float32(sens)
            terms = np.stack([np.log(np.maximum(_norm_cdf((margin0 + e) / sig), 1e-30))
                              for e in self._t_eirps[ci]]).astype(np.float32)
            new_terms.append(terms)
            res.append(float(fr.terrain_res_m))
            model_name = model_name or str(getattr(fr, "model", ""))
        for ci, terms in enumerate(new_terms):           # commit only when every field came
            self._t_logr[ci] += terms
        # P(heard from every place | cell): mixture over classes of the mean over EIRP
        w = 1.0 / len(self._t_classes)
        joint = np.zeros(self._t_logr[0].shape[1:], dtype=np.float64)
        for lr in self._t_logr:
            mx = lr.max(axis=0)
            joint += w * np.exp(mx) * np.exp(lr - mx).mean(axis=0)
        term_s = np.log(m.floor + (1.0 - m.floor) * joint).astype(np.float32)
        term = _expand(term_s, self._t_stride, self.spec.n)
        before = self.prob()
        if self._t_term is not None:
            self.logp -= self._t_term
        self.logp += term
        self.logp -= self.logp.max()
        self._t_term = term
        after = self.prob()
        self._heard_from.append((float(re), float(rn)))
        self.terrain_updates += 1
        moved = float(0.5 * np.abs(after - before).sum())      # total-variation distance
        out = {"applied": True, "places": len(self._heard_from),
               "classes": [c.name for c in self._t_classes],
               "sensitivity_dbm": round(sens, 1),
               "sigma_db": round(m.sigma_db(max(res) if res else 30.0), 1),
               "terrain_res_m": max(res) if res else None,
               "unlikely_share": round(float((joint < 0.5).mean()), 3),
               "belief_moved": round(moved, 3), "model": model_name, "why": ""}
        self.terrain_notes.append(out)
        return out

    def age(self, t_utc: float, motion_state: str = "unknown",
            terrain_mask: np.ndarray | None = None) -> None:
        """Diffuse the posterior forward to `t_utc` by the motion state. A
        terrain mask (0 where impassable, e.g. water) is applied after."""
        if self.last_t is None:
            self.last_t = t_utc
            return
        dt = max(0.0, t_utc - self.last_t)
        self.last_t = t_utc
        speed = DIFFUSE_MPS.get(motion_state, DIFFUSE_MPS["unknown"])
        sigma_m = speed * dt
        if sigma_m > 0:
            # the shadow evidence was about where it WAS; the diffused belief
            # keeps it, and the places heard from start afresh
            self._t_term = None
            self._heard_from = []
            self._t_classes = self._t_eirps = self._t_logr = None
            sig_cells = sigma_m / self.spec.cell_m
            # diffuse in probability space, then back to log
            p = np.exp(self.logp - self.logp.max())
            p = gaussian_filter(p, sigma=min(sig_cells, self.spec.n / 4.0), mode="constant")
            p = np.maximum(p, 1e-30)
            self.logp = np.log(p).astype(np.float32)
            self.logp -= self.logp.max()
        if terrain_mask is not None:
            self.logp = np.where(terrain_mask > 0, self.logp, -1e30).astype(np.float32)
        if self._logk is not None and sigma_m > 0:
            # the mover's belief has been diffused as one: fold the bias
            # hypotheses back into it (later bearings start them afresh)
            self._logk[:] = self.logp

    def prob(self) -> np.ndarray:
        """The normalised probability grid (sums to 1)."""
        p = np.exp(self.logp - self.logp.max())
        s = p.sum()
        return (p / s) if s > 0 else p

    def peak(self) -> tuple:
        """(lat, lon) of the most probable cell."""
        r, c = np.unravel_index(int(np.argmax(self.logp)), self.logp.shape)
        e, nth = self._E[r, c], self._N[r, c]
        lat, lon, _ = geo.enu_to_geodetic(e, nth, 0.0,
                                          (self.spec.ref_lat, self.spec.ref_lon, 0.0))
        return float(lat), float(lon)

    def fix(self) -> dict:
        """A summary: the peak, the probability-weighted centroid, and a 1σ
        equivalent radius from the spread — the honest size of the belief."""
        p = self.prob()
        tot = float(p.sum()) or 1.0
        ec = float((p * self._E).sum() / tot)
        nc = float((p * self._N).sum() / tot)
        var = float((p * ((self._E - ec) ** 2 + (self._N - nc) ** 2)).sum() / tot)
        lat, lon, _ = geo.enu_to_geodetic(ec, nc, 0.0,
                                          (self.spec.ref_lat, self.spec.ref_lon, 0.0))
        plat, plon = self.peak()
        # concentration: how peaked the belief is (1 = one cell, →0 = diffuse)
        conc = float((p ** 2).sum())      # inverse participation ratio proxy
        return {"lat": float(lat), "lon": float(lon), "peak_lat": plat, "peak_lon": plon,
                "sigma_m": math.sqrt(max(var, 0.0)), "n_updates": self.n_updates,
                "concentration": conc}


    # -- the region it is in --------------------------------------------------
    def region(self, containment: float = 0.95) -> np.ndarray:
        """The smallest set of cells holding `containment` of the belief (the
        highest-density region), as a boolean grid."""
        p = self.prob().ravel()
        order = np.argsort(p)[::-1]
        c = np.cumsum(p[order])
        k = int(np.searchsorted(c, float(containment))) + 1
        mask = np.zeros(p.size, dtype=bool)
        mask[order[:min(k, p.size)]] = True
        return mask.reshape(self.logp.shape)

    def region_area_m2(self, containment: float = 0.95) -> float:
        return float(self.region(containment).sum()) * self.spec.cell_m ** 2

    def cell_of(self, lat: float, lon: float):
        """(row, col) of the cell holding a point, or None outside the grid."""
        e, n, _ = geo.geodetic_to_enu(lat, lon, 0.0, (self.spec.ref_lat, self.spec.ref_lon, 0.0))
        h = (self.spec.n // 2) * self.spec.cell_m
        c = int(round((float(e) + h) / self.spec.cell_m))
        r = int(round((float(n) + h) / self.spec.cell_m))
        if 0 <= r < self.spec.n and 0 <= c < self.spec.n:
            return r, c
        return None

    def contains(self, lat: float, lon: float, containment: float = 0.95) -> bool:
        """Is a point inside the `containment` region? (False outside the grid.)"""
        rc = self.cell_of(lat, lon)
        return bool(rc is not None and self.region(containment)[rc])

    def region_outline(self, containment: float = 0.95, max_cells: int = 160) -> list:
        """The region's outline as rings of (lat, lon), each ring closed —
        for a map layer. Traced on a grid no finer than `max_cells` across."""
        mask = self.region(containment)
        n = mask.shape[0]
        step = max(1, int(math.ceil(n / float(max_cells))))
        if step > 1:
            m = n // step
            mask = mask[:m * step, :m * step].reshape(m, step, m, step).any(axis=(1, 3))
        rings = _trace_rings(mask)
        h = (self.spec.n // 2) * self.spec.cell_m
        size = self.spec.cell_m * step
        out = []
        for ring in rings:
            e = np.array([-h - self.spec.cell_m / 2 + c * size for r, c in ring])
            nn = np.array([-h - self.spec.cell_m / 2 + r * size for r, c in ring])
            lat, lon, _ = geo.enu_to_geodetic(e, nn, np.zeros(e.size),
                                              (self.spec.ref_lat, self.spec.ref_lon, 0.0))
            out.append([(float(a), float(b)) for a, b in zip(np.atleast_1d(lat),
                                                              np.atleast_1d(lon))])
        return out


def _trace_rings(mask: np.ndarray) -> list:
    """Closed outlines of the True cells, as rings of (row, col) CORNER
    coordinates (row 0 = south), each walked counter-clockwise with the region
    on its left. Every boundary edge is used once; where two cells touch only
    at a corner the corner has two ways out, and either keeps the ring closed."""
    m = np.pad(np.asarray(mask, dtype=bool), 1)
    out_edges: dict = {}

    def add(a, b):
        out_edges.setdefault(a, []).append(b)

    rows, cols = m.shape
    rr_idx, cc_idx = np.nonzero(m[1:-1, 1:-1])
    for rr, cc in zip(rr_idx.tolist(), cc_idx.tolist()):
        r, c = rr + 1, cc + 1
        if not m[r - 1, c]:
            add((rr, cc), (rr, cc + 1))                 # south edge, west -> east
        if not m[r, c + 1]:
            add((rr, cc + 1), (rr + 1, cc + 1))         # east edge, south -> north
        if not m[r + 1, c]:
            add((rr + 1, cc + 1), (rr + 1, cc))         # north edge, east -> west
        if not m[r, c - 1]:
            add((rr + 1, cc), (rr, cc))                 # west edge, north -> south
    rings = []
    while out_edges:
        start = next(iter(out_edges))
        ring = [start]
        cur = start
        while True:
            outs = out_edges.get(cur)
            if not outs:
                break
            nxt = outs.pop()
            if not outs:
                del out_edges[cur]
            if nxt == start:
                break
            ring.append(nxt)
            cur = nxt
        ring.append(start)
        if len(ring) >= 4:
            rings.append(ring)
    return rings


#: public name: outline any boolean grid (row 0 = south) as closed corner rings
trace_rings = _trace_rings


def coverage_interval(hits: int, n: int, level: float = 0.99) -> tuple:
    """Clopper–Pearson interval for a containment rate: `hits` of `n` trials
    held the truth. A stated 95 % region is honest when 0.95 is inside it."""
    from scipy.stats import beta
    a = (1.0 - float(level)) / 2.0
    lo = 0.0 if hits == 0 else float(beta.ppf(a, hits, n - hits + 1))
    hi = 1.0 if hits == n else float(beta.ppf(1.0 - a, hits + 1, n - hits))
    return lo, hi


def area_around(lat: float, lon: float, half_size_m: float = 25_000.0,
                cell_m: float = 25.0) -> GridSpec:
    """A grid centred on a point — the operating area around the receiver."""
    return GridSpec(ref_lat=lat, ref_lon=lon, half_size_m=half_size_m, cell_m=cell_m)
