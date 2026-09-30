"""Array geometry and the array manifold: what the elements should see from θ.

(Called `manifold` rather than the plan's `array` because a module named
`array` shadows the standard library's whenever a file in this package is
run directly — the same reason `dossier` is not called `types`.)

THE ARRAY FRAME, fixed here once and used everywhere:

    x forward     toward element 0 of a UCA: the vehicle's nose, or the
                  plate's marked north on a tripod
    y left        so x, y, z(up) is right-handed; counter-clockwise viewed
                  from above goes from x toward y
    θ (theta_rel) the RELATIVE BEARING of the source, in degrees CLOCKWISE
                  from x — the way a compass reads — so a source dead ahead
                  is 0°, on the right is 90°, and the absolute bearing is
                  heading + θ with no sign flip anywhere downstream

A UCA's elements are numbered COUNTER-CLOCKWISE viewed from above, element 0
toward the nose (umbrella plan §5.1), so element m sits at +m·360°/n from x
toward y — i.e. at relative bearing −m·360°/n. The mounting offset between
the plate's forward mark and the vehicle's true axis is measured (the radio
compass, `references`), never assumed.

THE STEERING VECTOR. A plane wave from unit direction u(θ) = (cos θ, −sin θ)
reaches element m at position p_m earlier by p_m·u / c, so its baseband
arrives with phase exp(+j·k·p_m·u), k = 2π/λ — the same convention as the
moved `geoloc.steering_vector`. If a real array's bearings ever come back
mirrored (θ ↔ −θ), the receivers' I/Q convention is inverted; the measured
manifold absorbs it, and T3 (a tripod on a known reference) is where it
would show first.

THE FOURIER FORM (EADF). Every planar array's response is a Fourier series
in θ: with element m at polar (ρ_m, φ_m), p_m·u = ρ_m·cos(θ + φ_m), and by
the Jacobi–Anger expansion

    a_m(θ) = Σ_q  j^q · J_q(k ρ_m) · e^{j q φ_m} · e^{j q θ}

— a matrix G (elements × 2Q+1) times the vector d(θ) = [e^{jqθ}]. A MEASURED
manifold has the same form with G fitted to the measurements. That shared
form is what lets root-MUSIC run on a circle or on a measured table
(`doa.root_music`, the manifold-separation technique) and what makes a
measured table interpolate smoothly between the angles it was measured at.

WHAT THE NOISE SOURCE DOES NOT SEE. atkdaq's calibration ends at the switch;
cables, connectors, whips and the roof are upstream of it. An ideal manifold
therefore carries a caveat on every bearing, and the measured manifold
(Phase 3: a drive past, or a tripod turned in front of, a catalogued
reference) is the single largest accuracy gain available on a vehicle.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.special import jv

C = 299_792_458.0

#: The estimators' grid: 0.5° (umbrella §5, "MUSIC's grid is 0.5°").
GRID_STEP_DEG = 0.5


def grid(step_deg: float = GRID_STEP_DEG) -> np.ndarray:
    """Relative bearings 0 ≤ θ < 360, in degrees."""
    n = int(round(360.0 / float(step_deg)))
    return np.arange(n) * (360.0 / n)


def wavelength(f_hz: float) -> float:
    if f_hz <= 0:
        raise ValueError("a manifold needs the RF frequency (> 0 Hz)")
    return C / float(f_hz)


class SpacingError(ValueError):
    """The geometry is too sparse for the frequency: bearings would alias."""


@dataclass
class Geometry:
    """Element positions in the array frame, metres, shape (n, 2) as (x, y).

    `channel_of_element[e]` is the receiver channel wired to element e (the
    connector map: umbrella §14, `config/kraken.json` on the ATK side).
    Identity unless the cabling says otherwise; the engine reorders channels
    into element order before any estimator sees them.
    """

    positions: np.ndarray
    id: str = "custom"
    description: str = ""
    channel_of_element: tuple | None = None
    kind: str = "custom"            # uca | ula | custom
    radius_m: float = 0.0           # UCA only
    spacing_m: float = 0.0          # ULA only

    def __post_init__(self):
        p = np.asarray(self.positions, dtype=float)
        if p.ndim != 2 or p.shape[1] != 2 or p.shape[0] < 2:
            raise ValueError("positions must be (elements ≥ 2, 2) metres")
        self.positions = p
        n = p.shape[0]
        if self.channel_of_element is None:
            self.channel_of_element = tuple(range(n))
        m = tuple(int(c) for c in self.channel_of_element)
        if sorted(m) != list(range(n)):
            raise ValueError(f"the connector map must use each of the {n} channels "
                             f"exactly once, not {m}")
        self.channel_of_element = m

    @property
    def n(self) -> int:
        return int(self.positions.shape[0])

    @property
    def polar(self) -> tuple:
        """(ρ, φ): element radius, and angle counter-clockwise from x."""
        x, y = self.positions[:, 0], self.positions[:, 1]
        return np.hypot(x, y), np.arctan2(y, x)

    @property
    def aperture_m(self) -> float:
        d = self.positions[:, None, :] - self.positions[None, :, :]
        return float(np.max(np.hypot(d[..., 0], d[..., 1])))

    def adjacent_spacing_m(self) -> float:
        """The largest nearest-neighbour distance — the spacing the
        half-wavelength rule is about."""
        d = self.positions[:, None, :] - self.positions[None, :, :]
        dist = np.hypot(d[..., 0], d[..., 1])
        np.fill_diagonal(dist, np.inf)
        return float(np.max(np.min(dist, axis=1)))

    def max_unambiguous_hz(self) -> float:
        return C / (2.0 * self.adjacent_spacing_m())

    @property
    def centrosymmetric(self) -> bool:
        """Is the array its own point reflection? Forward-backward averaging
        is only valid when it is — a ULA is, a 5-element UCA is NOT (an odd
        circle has no element opposite each element), so for a UCA the
        engine smooths in phase-mode space instead (`doa`)."""
        p = self.positions - self.positions.mean(axis=0)
        q = -p
        d = np.hypot(p[:, None, 0] - q[None, :, 0], p[:, None, 1] - q[None, :, 1])
        return bool(np.all(np.min(d, axis=1) < 1e-6))

    def check(self, f_hz: float, override: bool = False) -> str:
        """Refuse a geometry too sparse for `f_hz` (umbrella §5.1), saying by
        how much. Returns a note (empty when fine); raises SpacingError unless
        `override`, in which case the note is the caveat to carry."""
        lam = wavelength(f_hz)
        d = self.adjacent_spacing_m()
        if d <= lam / 2.0 + 1e-12:
            return ""
        note = (f"adjacent element spacing {d:.3f} m is {d / lam:.2f} λ at "
                f"{f_hz / 1e6:.1f} MHz — beyond the 0.5 λ that keeps a bearing "
                f"unambiguous; this geometry is good to "
                f"{self.max_unambiguous_hz() / 1e6:.0f} MHz")
        if not override:
            raise SpacingError(note + ". Pass override=True to use it anyway "
                               "(bearings will have aliases).")
        return note

    def element_order(self, x_channels: np.ndarray) -> np.ndarray:
        """Receiver channels (n, …) → element order (n, …)."""
        return np.asarray(x_channels)[list(self.channel_of_element)]

    def to_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "description": self.description,
                "positions_m": self.positions.tolist(),
                "channel_of_element": list(self.channel_of_element),
                "radius_m": self.radius_m, "spacing_m": self.spacing_m}

    @classmethod
    def from_dict(cls, d: dict) -> "Geometry":
        return cls(positions=np.asarray(d["positions_m"], dtype=float), id=d.get("id", "custom"),
                   description=d.get("description", ""),
                   channel_of_element=tuple(d.get("channel_of_element") or ()) or None,
                   kind=d.get("kind", "custom"), radius_m=float(d.get("radius_m", 0.0)),
                   spacing_m=float(d.get("spacing_m", 0.0)))


def uca(n: int = 5, radius_m: float = 0.18, channel_of_element=None) -> Geometry:
    """Uniform circular array: element 0 toward the nose, counter-clockwise.
    The default is THE preset for this hardware (umbrella §5.1): r = 0.18 m,
    adjacent spacing 0.212 m = 0.32 λ at 450 MHz, 0.5 λ at 707 MHz."""
    phi = 2.0 * np.pi * np.arange(n) / n
    pos = np.stack([radius_m * np.cos(phi), radius_m * np.sin(phi)], axis=1)
    return Geometry(pos, id=f"uca{n}_r{radius_m:.3f}", kind="uca", radius_m=float(radius_m),
                    description=f"UCA, {n} elements, radius {radius_m:.3f} m, element 0 "
                                f"forward, counter-clockwise from above",
                    channel_of_element=channel_of_element)


def ula(n: int = 5, spacing_m: float = 0.3, channel_of_element=None) -> Geometry:
    """Uniform linear array ACROSS the forward axis (along y), broadside
    forward: element 0 on the right, numbering toward the left. It measures
    sin θ only, so θ and 180° − θ (front and back) are one answer — the cone
    ambiguity `fusion`'s mirror logic already handles."""
    y = (np.arange(n) - (n - 1) / 2.0) * spacing_m
    pos = np.stack([np.zeros(n), y], axis=1)
    return Geometry(pos, id=f"ula{n}_d{spacing_m:.3f}", kind="ula", spacing_m=float(spacing_m),
                    description=f"ULA, {n} elements, spacing {spacing_m:.3f} m, across "
                                f"the forward axis, element 0 on the right",
                    channel_of_element=channel_of_element)


def custom(xy, id: str = "custom", description: str = "", channel_of_element=None) -> Geometry:
    return Geometry(np.asarray(xy, dtype=float), id=id, description=description,
                    channel_of_element=channel_of_element)


def direction(theta_deg) -> np.ndarray:
    """(K, 2) unit vectors toward relative bearings θ (clockwise from x)."""
    t = np.radians(np.asarray(theta_deg, dtype=float).reshape(-1))
    return np.stack([np.cos(t), -np.sin(t)], axis=1)


def eadf_order(geometry: Geometry, f_hz: float) -> int:
    """How many Fourier terms each side represent the ideal manifold to
    ~1e-9: |J_q(z)| falls off like (z/2)^q / q! past q ≈ z."""
    z = 2.0 * np.pi / wavelength(f_hz) * float(np.max(geometry.polar[0]))
    return int(math.ceil(z)) + 12


class Manifold:
    """What the array sees from θ. `response(theta, f)` is (n, K)."""

    kind = "ideal"
    id = ""

    def response(self, theta_deg, f_hz: float) -> np.ndarray:
        raise NotImplementedError

    def eadf(self, f_hz: float, order: int | None = None) -> tuple:
        """(G, q): the Fourier coefficients with a(θ) = G @ exp(j q θ)."""
        raise NotImplementedError

    def caveats(self, f_hz: float) -> list:
        return []


class IdealManifold(Manifold):
    """Free-space point elements at the geometry's positions."""

    kind = "ideal"

    def __init__(self, geometry: Geometry):
        self.geometry = geometry
        self.id = ""

    def response(self, theta_deg, f_hz: float) -> np.ndarray:
        k = 2.0 * np.pi / wavelength(f_hz)
        return np.exp(1j * k * (self.geometry.positions @ direction(theta_deg).T))

    def eadf(self, f_hz: float, order: int | None = None) -> tuple:
        Q = eadf_order(self.geometry, f_hz) if order is None else int(order)
        q = np.arange(-Q, Q + 1)
        rho, phi = self.geometry.polar
        k = 2.0 * np.pi / wavelength(f_hz)
        # a_m(θ) = Σ_q j^q J_q(kρ_m) e^{jqφ_m} e^{jqθ}, θ in RADIANS clockwise
        G = (1j ** q)[None, :] * jv(q[None, :], k * rho[:, None]) * np.exp(1j * q[None, :] * phi[:, None])
        return G, q

    def caveats(self, f_hz: float) -> list:
        return ["IDEAL MANIFOLD: the steering vectors are free-space geometry. "
                "atkdaq's calibration ends at the noise-source switch, so cables, "
                "connectors, whips and the roof are not in this answer; a measured "
                "manifold is what removes their error."]


class PerturbedManifold(Manifold):
    """An ideal manifold seen through per-element gain/phase errors and
    mutual coupling: a(θ) = C · diag(g) · a_ideal(θ). The simulator's way of
    making the array wrong the way a roof makes it wrong — and the thing a
    measured manifold must learn back."""

    kind = "perturbed"

    def __init__(self, geometry: Geometry, gains=None, coupling=None, id: str = "perturbed"):
        self.geometry = geometry
        self.ideal = IdealManifold(geometry)
        n = geometry.n
        self.gains = np.ones(n, complex) if gains is None else np.asarray(gains, complex)
        self.coupling = np.eye(n, dtype=complex) if coupling is None else np.asarray(coupling, complex)
        self.id = id

    @classmethod
    def random(cls, geometry: Geometry, gain_db: float = 0.5, phase_deg: float = 5.0,
               coupling_db: float = -20.0, seed: int = 0) -> "PerturbedManifold":
        """Errors of the size a vehicle roof produces: a fraction of a dB and
        a few degrees per element, neighbours coupled at `coupling_db`."""
        rng = np.random.default_rng(seed)
        n = geometry.n
        g = 10 ** (rng.normal(0, gain_db, n) / 20) * np.exp(1j * np.radians(rng.normal(0, phase_deg, n)))
        Cm = np.eye(n, dtype=complex)
        if coupling_db is not None and coupling_db > -200:
            c = 10 ** (coupling_db / 20)
            d = geometry.positions[:, None, :] - geometry.positions[None, :, :]
            dist = np.hypot(d[..., 0], d[..., 1])
            near = dist <= geometry.adjacent_spacing_m() * 1.01
            np.fill_diagonal(near, False)
            ph = np.exp(1j * rng.uniform(-np.pi, np.pi, (n, n)))
            Cm = Cm + c * near * ph
        return cls(geometry, g, Cm, id=f"perturbed_seed{seed}")

    def response(self, theta_deg, f_hz: float) -> np.ndarray:
        return self.coupling @ (self.gains[:, None] * self.ideal.response(theta_deg, f_hz))

    def eadf(self, f_hz: float, order: int | None = None) -> tuple:
        G, q = self.ideal.eadf(f_hz, order)
        return self.coupling @ (self.gains[:, None] * G), q


class MeasuredManifold(Manifold):
    """A manifold measured on the real array, per band (umbrella §5.2).

    Stored as the Fourier coefficients G (elements × 2Q+1) at the band's
    centre frequency, fitted from measured steering vectors normalised to
    element 0 — the absolute phase of a measurement is the emitter's, not the
    array's, so only the ratios are the array's. Saved as
    `manifold_<geometry>_<band>.npz` with the passport of the run.
    """

    kind = "measured"

    def __init__(self, geometry: Geometry, f_hz: float, G: np.ndarray, q: np.ndarray,
                 id: str = "", passport_uid: str = "", source: str = "",
                 band_hz: tuple | None = None, residual_deg: float | None = None,
                 coverage_deg: float = 360.0):
        self.geometry = geometry
        self.f_hz = float(f_hz)
        self.G = np.asarray(G, complex)
        self.q = np.asarray(q, int)
        self.id = id or f"manifold_{geometry.id}_{self.f_hz / 1e6:.0f}MHz"
        self.passport_uid = passport_uid
        self.source = source
        self.band_hz = band_hz or (self.f_hz * 0.98, self.f_hz * 1.02)
        self.residual_deg = residual_deg
        self.coverage_deg = coverage_deg

    def response(self, theta_deg, f_hz: float) -> np.ndarray:
        t = np.radians(np.asarray(theta_deg, dtype=float).reshape(-1))
        return self.G @ np.exp(1j * np.outer(self.q, t))

    def eadf(self, f_hz: float, order: int | None = None) -> tuple:
        if order is None or order >= self.q.max():
            return self.G, self.q
        keep = np.abs(self.q) <= int(order)
        return self.G[:, keep], self.q[keep]

    def caveats(self, f_hz: float) -> list:
        out = []
        lo, hi = self.band_hz
        if not lo <= f_hz <= hi:
            out.append(f"the measured manifold {self.id} was measured for "
                       f"{lo / 1e6:.1f}–{hi / 1e6:.1f} MHz and is being used at "
                       f"{f_hz / 1e6:.1f} MHz")
        if self.coverage_deg < 359.0:
            out.append(f"the measured manifold covers only {self.coverage_deg:.0f}° of "
                       f"bearing; outside that it is extrapolated")
        return out

    # -- building one --------------------------------------------------------
    @classmethod
    def from_snapshots(cls, geometry: Geometry, f_hz: float, theta_deg, vectors,
                       bin_deg: float = 2.0, order: int | None = None,
                       weights=None, **kw) -> "MeasuredManifold":
        """Fit from (true relative bearing, measured steering vector) pairs —
        a drive calibration's or a tripod's output.

        Each vector is normalised to element 0 and to unit norm (its absolute
        phase and power are the emitter's), averaged in `bin_deg` bins (the
        umbrella's 2°), and the Fourier series is fitted to the bin means by
        least squares with a small ridge toward the ideal manifold — so a
        sector the drive never saw falls back to geometry rather than to
        noise, and `coverage_deg` says how much was seen.
        """
        th = np.asarray(theta_deg, dtype=float).reshape(-1) % 360.0
        V = np.asarray(vectors, dtype=complex)
        if V.shape != (th.size, geometry.n):
            raise ValueError(f"vectors must be ({th.size}, {geometry.n})")
        ideal = IdealManifold(geometry)
        # A measured vector's absolute phase is the emitter's. Re-reference each
        # one so element 0 carries the IDEAL element-0 phase for its true
        # bearing: the table is then in the same (array-centre) reference as
        # the ideal manifold, so the two can be compared and the ideal can be
        # the fit's prior. A common factor per θ is invisible to every
        # estimator, so this loses nothing the array can measure.
        a0 = ideal.response(th, f_hz)[0]
        ref = (V[:, 0] / np.maximum(np.abs(V[:, 0]), 1e-30)).conj() * (a0 / np.abs(a0))
        V = V * ref[:, None]
        V = V * (math.sqrt(geometry.n) / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-30))
        w = np.ones(th.size) if weights is None else np.asarray(weights, float)
        nb = int(round(360.0 / bin_deg))
        idx = np.floor(th / (360.0 / nb)).astype(int) % nb
        centres, means = [], []
        for b in range(nb):
            sel = idx == b
            if not np.any(sel):
                continue
            m = np.sum(V[sel] * w[sel, None], axis=0) / np.sum(w[sel])
            centres.append(np.sum(th[sel] * w[sel]) / np.sum(w[sel]))
            means.append(m)
        centres = np.asarray(centres)
        means = np.asarray(means)
        Q = eadf_order(geometry, f_hz) if order is None else int(order)
        q = np.arange(-Q, Q + 1)
        D = np.exp(1j * np.outer(np.radians(centres), q))            # (B, 2Q+1)
        G0, _q0 = ideal.eadf(f_hz, Q)
        # Ridge toward the ideal coefficients: with bins everywhere it is
        # negligible; over a sector nobody drove through, it is the answer.
        lam = 1e-2
        A = D.conj().T @ D + lam * len(centres) / q.size * np.eye(q.size)
        G = np.linalg.solve(A, D.conj().T @ means + (lam * len(centres) / q.size) * G0.T).T
        fit = D @ G.T
        resid = float(np.degrees(np.sqrt(np.mean(np.angle(
            fit * np.conj(means)) ** 2))))
        covered = len(set(idx.tolist())) * (360.0 / nb)
        return cls(geometry, f_hz, G, q, residual_deg=resid, coverage_deg=covered, **kw)

    def save(self, path) -> Path:
        path = Path(path)
        np.savez_compressed(path, G=self.G, q=self.q, f_hz=self.f_hz,
                            band_hz=np.asarray(self.band_hz),
                            meta=json.dumps({"id": self.id, "passport_uid": self.passport_uid,
                                             "source": self.source,
                                             "residual_deg": self.residual_deg,
                                             "coverage_deg": self.coverage_deg,
                                             "geometry": self.geometry.to_dict()}))
        return path if path.suffix == ".npz" else path.with_suffix(path.suffix + ".npz")

    @classmethod
    def load(cls, path) -> "MeasuredManifold":
        z = np.load(Path(path), allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        geom = Geometry.from_dict(meta["geometry"])
        return cls(geom, float(z["f_hz"]), z["G"], z["q"], id=meta.get("id", ""),
                   passport_uid=meta.get("passport_uid", ""), source=meta.get("source", ""),
                   band_hz=tuple(z["band_hz"].tolist()), residual_deg=meta.get("residual_deg"),
                   coverage_deg=float(meta.get("coverage_deg", 360.0)))
