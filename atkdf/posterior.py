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
    with each bearing; `age` between them; `fix` for the summary."""

    def __init__(self, spec: GridSpec):
        self.spec = spec
        n = spec.n
        self.logp = np.zeros((n, n), dtype=np.float32)      # north (row) × east (col)
        e, nth = spec.axes()
        self._E, self._N = np.meshgrid(e, nth)              # (n,n) metres from ref
        self.last_t = None
        self.n_updates = 0
        self._rx_cache: dict = {}

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
        idx = np.round(brg).astype(np.int64) % 360
        cell_lik = np.maximum(lik[idx], min_lik)
        self.logp += np.log(cell_lik).astype(np.float32)
        self.logp -= self.logp.max()     # keep it bounded; it is unnormalised
        self.last_t = rec.t_utc
        self.n_updates += 1

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
            sig_cells = sigma_m / self.spec.cell_m
            # diffuse in probability space, then back to log
            p = np.exp(self.logp - self.logp.max())
            p = gaussian_filter(p, sigma=min(sig_cells, self.spec.n / 4.0), mode="constant")
            p = np.maximum(p, 1e-30)
            self.logp = np.log(p).astype(np.float32)
            self.logp -= self.logp.max()
        if terrain_mask is not None:
            self.logp = np.where(terrain_mask > 0, self.logp, -1e30).astype(np.float32)

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


def area_around(lat: float, lon: float, half_size_m: float = 25_000.0,
                cell_m: float = 25.0) -> GridSpec:
    """A grid centred on a point — the operating area around the receiver."""
    return GridSpec(ref_lat=lat, ref_lon=lon, half_size_m=half_size_m, cell_m=cell_m)
