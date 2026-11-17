"""The road-constrained particle filter: a vehicle's posterior lives on the
road graph (umbrella §8.5; built 2026-11-16).

WHY. A free grid treats a bearing as a wedge: every cell along it is a
candidate. A car is not anywhere along the wedge; it is on a road. So a
bearing that cuts three roads is THREE candidates, and a car that keeps
driving is followed along its roads instead of being smeared into a growing
blob — the grid can only age a mover by diffusing it at vehicle speed. This is
map-matching, as a phone's navigator does, with bearings in place of GPS.

WHAT IT CANNOT DO, said plainly (and pinned by a test). One receiver that
does not move cannot range a car on a straight road: a car on a road twice as
far, driving twice as fast, draws the same bearings. Bearings-only motion
analysis needs geometry — a second receiver, the DF vehicle moving across the
line of sight, or the target turning onto a road a scaled copy of its track
could not follow. In a dense, regular street grid even a turn is often not
enough from one parked receiver. The filter does not pretend: it keeps several
candidates, and `candidates()` shows them.

HOW. Each particle is a position on a directed road segment (an `Edge` from
`ports.RoadGraph`: travelled from `a` to `b`), a distance along it, and a
speed. `predict(dt)` moves every particle along its road at its speed, turning
onto a random onward segment at the end (a dead end stops it); the speed takes
a random walk between 0 and `max_speed_mps`. `update(rec)` weights each
particle by the bearing's full 360-point likelihood at the particle's bearing
from the receiver — the same likelihood the grid posterior uses — floored, so
one wild bearing cannot kill the truth. A terrain shadow term can be multiplied
in from a field (`update_shadow`). Resampling is systematic, when the effective
sample size falls below half.

SEEDING. The port offers `nearest_edge(lat, lon)`, so the filter is seeded by
sampling points (from a grid posterior's belief, or uniformly over an area)
and snapping each to its nearest road. A point far from any road (more than
the graph's snap distance) gives no particle. The snap favours roads with
empty country around them a little; that is said, not hidden.

WHAT IT REPORTS. `candidates()`: the particles grouped by road segment, the
heaviest first, each with its share of the belief and its weighted position —
"three candidates on three roads" as the operator should hear it. `estimate()`:
the heaviest candidate's position with the spread of its particles. `grid()`:
the particles as a probability grid on a `posterior.GridSpec`, for the same
map display as the free posterior.

Receive-only, numpy only, nothing from ATK: the road graph is a port (ATK's is
`atk.core.df_terrain.StoreRoadGraph`; the tests use a synthetic grid of streets).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import geodesy as geo
from .records import BearingRecord

MIN_LIK = 1e-4


@dataclass
class _P:
    """The filter's particles as parallel arrays (fast to weight and resample)."""
    edge_idx: np.ndarray      # int: index into RoadParticleFilter._edges
    s: np.ndarray             # metres along the edge from its `a` end
    v: np.ndarray             # metres per second, along the edge's direction
    logw: np.ndarray          # log weights


class RoadParticleFilter:
    """A vehicle emitter's belief, on the roads. See the module text.

    `roads`          a `ports.RoadGraph`
    `n`              particles
    `max_speed_mps`  the fastest a target is believed to drive (default 35, ~78 mph)
    `accel_mps2`     how fast its speed may change: the random walk's scale,
                     m/s per √s (0.5 → about ±1.6 m/s over 10 s; a loose walk
                     lets every particle in the beam keep up with it, a tight one
                     cannot follow real traffic)
    `stop_prob`      chance per second that a moving particle stops (lights,
                     junctions) — a parked car must stay representable
    """

    def __init__(self, roads, n: int = 3000, max_speed_mps: float = 35.0,
                 accel_mps2: float = 0.5, stop_prob: float = 0.003, seed: int | None = None):
        self.roads = roads
        self.n = int(n)
        self.max_speed = float(max_speed_mps)
        self.accel = float(accel_mps2)
        self.stop_prob = float(stop_prob)
        self.rng = np.random.default_rng(seed)
        self._edges: list = []
        self._edge_ix: dict = {}
        self._nbrs: dict = {}
        self.p: _P | None = None
        self.last_t: float | None = None
        self.n_updates = 0
        self.resamples = 0
        self.notes: list = []

    # -- edges ---------------------------------------------------------------
    def _add_edge(self, e) -> int:
        k = str(e.id)
        i = self._edge_ix.get(k)
        if i is None:
            i = len(self._edges)
            self._edges.append(e)
            self._edge_ix[k] = i
        return i

    def _neighbours(self, i: int) -> list:
        hit = self._nbrs.get(i)
        if hit is None:
            try:
                hit = [self._add_edge(e) for e in (self.roads.neighbours(self._edges[i]) or [])]
            except Exception:                                 # noqa: BLE001
                hit = []
            self._nbrs[i] = hit
        return hit

    def _length(self, i: int) -> float:
        e = self._edges[i]
        L = float(getattr(e, "length_m", 0.0) or 0.0)
        if L <= 0:
            L = geo.distance_between(e.a[0], e.a[1], e.b[0], e.b[1])
        return max(L, 1.0)

    # -- seeding ---------------------------------------------------------------
    def seed_points(self, lats, lons, speed_mps=None) -> int:
        """Start from candidate points: each is snapped to its nearest road (a
        point with none near it is dropped). Particles are spread over the
        snapped set; both directions of travel are equally likely when the
        graph offers them. Returns how many particles were placed."""
        lats = np.atleast_1d(np.asarray(lats, float))
        lons = np.atleast_1d(np.asarray(lons, float))
        idx, svals = [], []
        for la, lo in zip(lats, lons):
            try:
                e = self.roads.nearest_edge(float(la), float(lo))
            except Exception:                                 # noqa: BLE001
                e = None
            if e is None:
                continue
            i = self._add_edge(e)
            L = self._length(i)
            # the point's place along the edge (projected in a local flat frame)
            ae, an, _ = geo.geodetic_to_enu(e.b[0], e.b[1], 0.0, (e.a[0], e.a[1], 0.0))
            pe, pn, _ = geo.geodetic_to_enu(la, lo, 0.0, (e.a[0], e.a[1], 0.0))
            seg2 = float(ae) ** 2 + float(an) ** 2
            frac = 0.5 if seg2 <= 0 else (float(pe) * float(ae) + float(pn) * float(an)) / seg2
            idx.append(i)
            svals.append(min(max(frac, 0.0), 1.0) * L)
            # the other direction too, when the graph has it (a two-way road)
            try:
                for j in self._neighbours(i):
                    ej = self._edges[j]
                    if (abs(ej.a[0] - e.b[0]) < 1e-9 and abs(ej.a[1] - e.b[1]) < 1e-9
                            and abs(ej.b[0] - e.a[0]) < 1e-9 and abs(ej.b[1] - e.a[1]) < 1e-9):
                        idx.append(j)
                        svals.append((1.0 - min(max(frac, 0.0), 1.0)) * self._length(j))
                        break
            except Exception:                                 # noqa: BLE001
                pass
        if not idx:
            self.p = None
            self.notes.append("no road near any candidate point — nothing to filter")
            return 0
        pick = self.rng.integers(0, len(idx), self.n)
        ei = np.asarray(idx, dtype=np.int64)[pick]
        s = np.asarray(svals, dtype=float)[pick]
        if speed_mps is None:
            v = self.rng.uniform(0.0, 0.6 * self.max_speed, self.n)
        else:
            v = np.full(self.n, float(speed_mps))
        self.p = _P(edge_idx=ei, s=s, v=v, logw=np.zeros(self.n))
        return self.n

    def seed_from_grid(self, post, n_points: int = 1500) -> int:
        """Seed from a grid posterior's belief: points drawn in proportion to
        it (with a little jitter inside each cell), then snapped to roads."""
        pr = post.prob().ravel()
        k = self.rng.choice(pr.size, size=int(n_points), p=pr / pr.sum())
        r, c = np.unravel_index(k, post.logp.shape)
        e_ax, n_ax = post.spec.axes()
        jit = post.spec.cell_m / 2.0
        e = e_ax[c] + self.rng.uniform(-jit, jit, k.size)
        nn = n_ax[r] + self.rng.uniform(-jit, jit, k.size)
        lat, lon, _ = geo.enu_to_geodetic(e, nn, np.zeros(k.size),
                                          (post.spec.ref_lat, post.spec.ref_lon, 0.0))
        return self.seed_points(np.atleast_1d(lat), np.atleast_1d(lon))

    def seed_area(self, lat: float, lon: float, half_size_m: float, n_points: int = 1500) -> int:
        """Seed uniformly over a square area, snapped to its roads."""
        e = self.rng.uniform(-half_size_m, half_size_m, int(n_points))
        nn = self.rng.uniform(-half_size_m, half_size_m, int(n_points))
        la, lo, _ = geo.enu_to_geodetic(e, nn, np.zeros(e.size), (lat, lon, 0.0))
        return self.seed_points(np.atleast_1d(la), np.atleast_1d(lo))

    # -- positions -------------------------------------------------------------
    def positions(self) -> tuple:
        """(lat, lon) arrays of every particle (linear along its segment)."""
        if self.p is None:
            return np.array([]), np.array([])
        A = np.array([self._edges[i].a for i in range(len(self._edges))], dtype=float)
        Bv = np.array([self._edges[i].b for i in range(len(self._edges))], dtype=float)
        L = np.array([self._length(i) for i in range(len(self._edges))], dtype=float)
        ei = self.p.edge_idx
        f = np.clip(self.p.s / L[ei], 0.0, 1.0)
        lat = A[ei, 0] + f * (Bv[ei, 0] - A[ei, 0])
        lon = A[ei, 1] + f * (Bv[ei, 1] - A[ei, 1])
        return lat, lon

    def weights(self) -> np.ndarray:
        if self.p is None:
            return np.array([])
        w = np.exp(self.p.logw - self.p.logw.max())
        return w / w.sum()

    def ess(self) -> float:
        w = self.weights()
        return float(1.0 / np.sum(w ** 2)) if w.size else 0.0

    # -- predict -----------------------------------------------------------------
    def predict(self, t_utc: float) -> None:
        """Drive every particle forward to `t_utc` along its roads."""
        if self.p is None:
            self.last_t = t_utc
            return
        if self.last_t is None:
            self.last_t = t_utc
            return
        dt = max(0.0, float(t_utc) - self.last_t)
        self.last_t = float(t_utc)
        if dt <= 0:
            return
        p = self.p
        # speed: a random walk, with a chance to stop and to start again
        p.v = p.v + self.rng.normal(0.0, self.accel * math.sqrt(dt), self.n)
        stop = self.rng.random(self.n) < (1.0 - (1.0 - self.stop_prob) ** dt)
        p.v[stop] = 0.0
        p.v = np.clip(p.v, 0.0, self.max_speed)
        p.s = p.s + p.v * dt
        # carry over the ends of segments (a few hops at most per step)
        for _ in range(50):
            L = np.array([self._length(int(i)) for i in p.edge_idx])
            over = p.s > L
            if not over.any():
                break
            for k in np.nonzero(over)[0]:
                i = int(p.edge_idx[k])
                nb = self._neighbours(i)
                if not nb:
                    p.s[k] = L[k]                          # a dead end: stop there
                    p.v[k] = 0.0
                    continue
                j = nb[int(self.rng.integers(0, len(nb)))]
                p.s[k] -= L[k]
                p.edge_idx[k] = j
        else:
            L = np.array([self._length(int(i)) for i in p.edge_idx])
            p.s = np.minimum(p.s, L)

    # -- update -------------------------------------------------------------------
    def update(self, rec: BearingRecord, min_lik: float = MIN_LIK) -> bool:
        """Weight the particles by one bearing (drives them to its time first).
        False when the bearing cannot be used (repeated, no heading, no position)."""
        if self.p is None:
            return False
        lik = rec.absolute_likelihood()
        if lik is None or rec.lat is None or rec.lon is None or rec.path_tag == "repeated":
            return False
        self.predict(rec.t_utc)
        lat, lon = self.positions()
        e, n, _ = geo.geodetic_to_enu(lat, lon, np.zeros(lat.size), (rec.lat, rec.lon, 0.0))
        brg = np.degrees(np.arctan2(e, n)) % 360.0
        cell = np.maximum(lik[np.round(brg).astype(np.int64) % 360], min_lik)
        self.p.logw = self.p.logw + np.log(cell)
        self.p.logw -= self.p.logw.max()
        self.n_updates += 1
        if self.ess() < self.n / 2.0:
            self._resample()
        return True

    def update_shadow(self, heard_prob_at) -> None:
        """Multiply in a terrain shadow term: `heard_prob_at(lat, lon)` returns
        P(heard) at each particle (from `posterior.heard_probability` over a
        field), floored as the grid's is."""
        if self.p is None:
            return
        lat, lon = self.positions()
        ph = np.asarray(heard_prob_at(lat, lon), dtype=float)
        self.p.logw = self.p.logw + np.log(np.maximum(ph, 0.02))
        self.p.logw -= self.p.logw.max()

    def _resample(self) -> None:
        w = self.weights()
        u = (self.rng.random() + np.arange(self.n)) / self.n
        k = np.minimum(np.searchsorted(np.cumsum(w), u), self.n - 1)
        p = self.p
        L = np.array([self._length(int(i)) for i in p.edge_idx[k]])
        s = np.clip(p.s[k] + self.rng.normal(0.0, 5.0, self.n), 0.0, L)
        v = np.clip(p.v[k] + self.rng.normal(0.0, 0.5, self.n), 0.0, self.max_speed)
        self.p = _P(edge_idx=p.edge_idx[k].copy(), s=s, v=v, logw=np.zeros(self.n))
        self.resamples += 1

    # -- what it says --------------------------------------------------------------
    def candidates(self, k: int = 5, merge_m: float = 150.0) -> list:
        """The belief grouped by road: the heaviest first, each with its share,
        its weighted position, its spread (m), its speed, and the segment id.
        Segments whose weighted positions lie within `merge_m` of each other (the
        two directions of one road, or a junction) are one candidate."""
        if self.p is None:
            return []
        w = self.weights()
        lat, lon = self.positions()
        groups: dict = {}
        for i in np.unique(self.p.edge_idx):
            m = self.p.edge_idx == i
            ww = w[m]
            if ww.sum() <= 0:
                continue
            groups[int(i)] = (float(ww.sum()), float((lat[m] * ww).sum() / ww.sum()),
                              float((lon[m] * ww).sum() / ww.sum()), m)
        order = sorted(groups.items(), key=lambda kv: -kv[1][0])
        out: list = []
        for i, (share, la, lo, m) in order:
            for c in out:
                if geo.distance_between(c["lat"], c["lon"], la, lo) < merge_m:
                    tot = c["share"] + share
                    c["lat"] = (c["lat"] * c["share"] + la * share) / tot
                    c["lon"] = (c["lon"] * c["share"] + lo * share) / tot
                    c["share"] = tot
                    c["_mask"] = c["_mask"] | m
                    c["edges"].append(str(self._edges[i].id))
                    break
            else:
                out.append({"lat": la, "lon": lo, "share": share, "_mask": m,
                            "edges": [str(self._edges[i].id)]})
        out.sort(key=lambda c: -c["share"])
        for c in out:
            m = c.pop("_mask")
            ww = w[m]
            d = np.array([geo.distance_between(c["lat"], c["lon"], a, b)
                          for a, b in zip(lat[m], lon[m])])
            c["spread_m"] = float(np.sqrt((ww * d ** 2).sum() / ww.sum())) if ww.sum() else 0.0
            c["speed_mps"] = float((self.p.v[m] * ww).sum() / ww.sum()) if ww.sum() else 0.0
            c["share"] = round(c["share"], 4)
        return out[:k]

    def estimate(self) -> dict | None:
        """The heaviest candidate, with how much of the belief it holds."""
        c = self.candidates(1)
        return c[0] if c else None

    def grid(self, spec) -> np.ndarray:
        """The particles as a normalised probability grid on `spec`."""
        g = np.zeros((spec.n, spec.n))
        if self.p is None:
            return g
        lat, lon = self.positions()
        e, n, _ = geo.geodetic_to_enu(lat, lon, np.zeros(lat.size),
                                      (spec.ref_lat, spec.ref_lon, 0.0))
        h = (spec.n // 2) * spec.cell_m
        c = np.round((np.asarray(e) + h) / spec.cell_m).astype(int)
        r = np.round((np.asarray(n) + h) / spec.cell_m).astype(int)
        ok = (r >= 0) & (r < spec.n) & (c >= 0) & (c < spec.n)
        np.add.at(g, (r[ok], c[ok]), self.weights()[ok])
        s = g.sum()
        return g / s if s > 0 else g

    def describe(self) -> str:
        cs = self.candidates(3)
        if not cs:
            return "road filter: no particles (no road near the search area)"
        bits = [f"{c['share'] * 100:.0f} % at {c['lat']:.5f}, {c['lon']:.5f} "
                f"(±{c['spread_m']:.0f} m, {c['speed_mps'] * 2.237:.0f} mph)" for c in cs]
        return (f"road filter, {self.n_updates} bearings: " + "; ".join(bits)
                + f" — {len(self._edges)} road segments in play")
