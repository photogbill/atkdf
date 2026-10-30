"""Channel health: dead, noisy or wandering channels, found from the data and
dropped before they spoil a bearing (plan §11.4.4).

Large arrays lose elements; a phased array of 16 is more likely to have one
bad element on any given day than a Kraken is. Three signatures, all from the
spatial covariance the estimator computes anyway:

  dead       the channel's row of |R| (its absolute row sum) is far below the
             array's — nothing is arriving on it (Sun et al., Sensors 2020,
             lit2 §3.16a — PRC flag: NUDT; identifiable as such per the
             provenance rule).
  noisy      it has power but no correlation with the other channels while
             they correlate with each other — a disconnected antenna or a
             failed LNA passing only its own noise.
  wandering  with a parked reference in view, its phase against the reference
             channel moves while the others stay put — a channel losing lock
             or a connector working loose. (Only judged when the caller says
             the reference is stationary.)

On a positive, `drop_bad` gives the manifold and geometry without those
elements (`Manifold.subset`); the bearing carries the dropped list and a
model sigma raised by half again in quadrature, because a dead element's
termination changes the coupling a measured table assumed. Nobody has
published the degradation per lost element on 8–16-element arrays (lit2
gaps): that number is ATK's to measure. Fragility: a uniform array survives
one interior loss gracefully; a nested or coprime layout may not (lit2
§3.16b) — `fragility_note` says so.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("atkdf.health")

PROVENANCE = {"dead": "Sun et al. 2020 (NUDT) — PRC-flagged method"}


@dataclass
class HealthReport:
    status: list                       # per element: ok | dead | noisy | wandering
    bad: list
    numbers: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.bad

    def sentence(self) -> str:
        if self.ok:
            return "all channels healthy"
        parts = [f"element {i} {self.status[i]}" for i in self.bad]
        return "; ".join(parts)


class ChannelHealth:
    """Per-array health monitor. `check(R)` judges one covariance (element
    order); with `stationary_reference=True` it also tracks each channel's
    phase against `ref_element` for the wandering test."""

    def __init__(self, n: int, ref_element: int = 0, dead_ratio: float = 0.2,
                 noisy_ratio: float = 0.3, min_coherence: float = 0.3,
                 wander_deg: float = 10.0, persist: int = 3):
        self.n = int(n)
        self.ref = int(ref_element)
        self.dead_ratio = float(dead_ratio)
        self.noisy_ratio = float(noisy_ratio)
        self.min_coherence = float(min_coherence)
        self.wander_deg = float(wander_deg)
        self.persist = int(persist)
        self._baseline = None
        self._over = np.zeros(self.n, dtype=int)
        self.last: HealthReport | None = None

    def reset_baseline(self) -> None:
        self._baseline = None
        self._over[:] = 0

    def check(self, R, stationary_reference: bool = False) -> HealthReport:
        R = np.asarray(R, dtype=complex)
        if R.shape != (self.n, self.n):
            raise ValueError(f"expected a {self.n}×{self.n} covariance, got {R.shape}")
        A = np.abs(R)
        rows = A.sum(axis=1)
        status = ["ok"] * self.n
        diag = np.maximum(np.real(np.diag(R)), 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            C = A / np.maximum(np.sqrt(np.outer(diag, diag)), 1e-300)
        np.fill_diagonal(C, np.nan)
        coh = np.nanmean(C, axis=1)                 # each channel's mean coherence
        med_diag = float(np.median(diag)) or 1e-300
        # Sun et al.'s absolute row sum against the array mean flags a channel
        # that contributes nothing correlated; its own power says which kind:
        # almost none = dead, normal = noisy (its own noise only)
        for i in range(self.n):
            if rows[i] < self.dead_ratio * float(np.mean(rows)):
                status[i] = "dead" if diag[i] < 0.1 * med_diag else "noisy"
        # noisy: no correlation with the others while they correlate with each other
        alive = [i for i in range(self.n) if status[i] == "ok"]
        if len(alive) >= 3:
            med = float(np.median(coh[alive]))
            if med >= self.min_coherence:
                for i in alive:
                    if coh[i] < self.noisy_ratio * med:
                        status[i] = "noisy"
        # wandering: phase against the reference element, from a baseline
        drift = np.zeros(self.n)
        if stationary_reference and status[self.ref] == "ok":
            ph = np.angle(R[:, self.ref])
            if self._baseline is None:
                self._baseline = ph
            else:
                d = np.degrees(np.angle(np.exp(1j * (ph - self._baseline))))
                common = float(np.median(d))
                drift = np.abs(d - common)
                for i in range(self.n):
                    if i == self.ref or status[i] != "ok":
                        continue
                    self._over[i] = self._over[i] + 1 if drift[i] > self.wander_deg else 0
                    if self._over[i] >= self.persist:
                        status[i] = "wandering"
        bad = [i for i, s in enumerate(status) if s != "ok"]
        rep = HealthReport(status, bad, {"row_sum": rows.round(4).tolist(),
                                         "coherence": np.nan_to_num(coh).round(3).tolist(),
                                         "drift_deg": drift.round(2).tolist()})
        if bad:
            log.warning("channel health: %s", rep.sentence())
        self.last = rep
        return rep


def drop_bad(manifold, report: HealthReport, geometry=None) -> tuple:
    """(manifold, geometry, keep) without the bad elements."""
    geometry = geometry if geometry is not None else manifold.geometry
    keep = [i for i in range(geometry.n) if i not in set(report.bad)]
    if len(keep) < 2:
        raise ValueError("fewer than two healthy elements: no bearing")
    sub = manifold.subset(keep)
    return sub, sub.geometry, keep


def fragility_note(geometry) -> str:
    """lit2 §3.16b: ULA/UCA/URA keep working with an interior element gone; a
    sparse (nested/coprime) layout may lose its unique lags."""
    if geometry.kind in ("ula", "uca", "uca_centre", "ura", "l", "multi_face"):
        return ""
    return ("this layout is not uniform: losing an element may remove lags the "
            "estimator needs (nested/coprime arrays are maximally fragile) — verify "
            "before trusting bearings with an element dropped")


def inflate_sigma(sigma_deg: float, model_sigma_deg: float) -> float:
    """The bearing sigma with half the model sigma again in quadrature."""
    return math.hypot(float(sigma_deg), 0.5 * float(model_sigma_deg))
