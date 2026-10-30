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
from dataclasses import dataclass, field, replace
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
class ElementPattern:
    """One element's voltage pattern, the same for every element (each element
    has its own BORESIGHT on the Geometry).

    ``omni``: 1 everywhere — whips, the Kraken's case.
    ``cardioid``: g(ψ) = b + (1 − b)·((1 + cos ψ)/2)^q, ψ the angle between the
    arrival direction and the element's boresight, b = 10^(back_db/20). A patch
    or a dipole in front of a ground plane is roughly q = 1–2 with a back lobe
    of −15 to −25 dB. Smooth everywhere (an integer q keeps its Fourier series
    finite), which is what lets the EADF and root-MUSIC run on a panel. A
    measured manifold replaces this with what the elements actually do.
    """

    kind: str = "omni"               # omni | cardioid
    exponent: float = 1.0
    back_db: float = -20.0

    def __post_init__(self):
        if self.kind not in ("omni", "cardioid"):
            raise ValueError(f"element pattern kind must be omni or cardioid, not {self.kind!r}")
        if self.back_db > 0:
            raise ValueError("back_db is a level below the front lobe (≤ 0 dB)")
        self.exponent = float(self.exponent)

    @property
    def back_amplitude(self) -> float:
        return 10 ** (float(self.back_db) / 20.0)

    def gain(self, cos_psi) -> np.ndarray:
        c = np.asarray(cos_psi, dtype=float)
        if self.kind == "omni":
            return np.ones_like(c)
        b = self.back_amplitude
        return b + (1.0 - b) * np.clip((1.0 + c) / 2.0, 0.0, 1.0) ** self.exponent

    def to_dict(self) -> dict:
        return {"kind": self.kind, "exponent": self.exponent, "back_db": self.back_db}

    @classmethod
    def from_dict(cls, d) -> "ElementPattern | None":
        if not d:
            return None
        return cls(kind=d.get("kind", "omni"), exponent=float(d.get("exponent", 1.0)),
                   back_db=float(d.get("back_db", -20.0)))


def _wrap_rel(a):
    return np.mod(np.asarray(a, dtype=float), 360.0)


@dataclass
class Geometry:
    """Element positions in the array frame, metres.

    `positions` is (n, 2) — x forward, y left — and stays (n, 2) so every
    azimuth-only path (the EADF, root-MUSIC, the CRB) reads what it always
    read. Height is `z` (n,), metres up. A (n, 3) array passed as
    `positions` is split into the two. `positions3` gives (n, 3).

    `channel_of_element[e]` is the receiver channel wired to element e (the
    connector map: umbrella §14, `config/kraken.json` on the ATK side).
    Identity unless the cabling says otherwise; the engine reorders channels
    into element order before any estimator sees them.

    Phased-array additions (DF_P25_IMPROVEMENT_PLAN §11.4.3, lit2 §2.9,
    §2.12, §2.23), every one defaulting to "as before":

    * `phase_centre_m` (n, 3): each element's ELECTRICAL phase centre offset
      from its mechanical position. The ideal manifold uses position + offset;
      the measured manifold fits and removes it before storing its Fourier
      coefficients, which is what makes a table interpolable across a band.
    * `amplitude` (n,): a static per-element gain (a shadowed element).
    * `pattern` + `boresight_deg` (n,): the element pattern and where each
      element points (relative bearing, clockwise). A flat panel's elements all
      point one way; a multi-face mast's point four ways.
    * `fov_az_deg` / `fov_el_deg`: the field of view, relative degrees. Outside
      it the array sees only through its back lobe, so the estimator searches
      inside it and a bearing's likelihood is flat outside it at the back-lobe
      level — "no information", not a confident ghost.
    * `groups` (n,): a face / sub-array index per element (multi-face masts).
    * `grid`: for a uniform rectangular array, the row/column axes and
      spacings (2-D ESPRIT needs the shift structure).
    * `hemisphere`: which side of a planar array the emitters are on, for the
      sign of the out-of-plane direction cosine (2-D DoA): "up" / "down" for
      a horizontal array, "front" for a vertical one.
    """

    positions: np.ndarray
    id: str = "custom"
    description: str = ""
    channel_of_element: tuple | None = None
    kind: str = "custom"            # uca | uca_centre | ula | ura | l | multi_face | custom
    radius_m: float = 0.0           # UCA only
    spacing_m: float = 0.0          # ULA / URA / L
    z: np.ndarray | None = None
    phase_centre_m: np.ndarray | None = None
    amplitude: np.ndarray | None = None
    pattern: ElementPattern | None = None
    boresight_deg: np.ndarray | None = None
    tilt_deg: float = 0.0
    fov_az_deg: tuple | None = None
    fov_el_deg: tuple | None = None
    groups: tuple | None = None
    grid: dict | None = None
    hemisphere: str = "up"
    parent_elements: tuple | None = None     # set by subset(): which original elements these are

    def __post_init__(self):
        p = np.asarray(self.positions, dtype=float)
        if p.ndim != 2 or p.shape[1] not in (2, 3) or p.shape[0] < 2:
            raise ValueError("positions must be (elements ≥ 2, 2) or (elements ≥ 2, 3) metres")
        n = p.shape[0]
        if p.shape[1] == 3:
            if self.z is not None and not np.allclose(np.asarray(self.z, float), p[:, 2]):
                raise ValueError("z given twice and the two disagree")
            self.z = p[:, 2].copy()
            p = p[:, :2].copy()
        self.positions = p
        self.z = np.zeros(n) if self.z is None else np.asarray(self.z, dtype=float).reshape(n)
        self.phase_centre_m = (np.zeros((n, 3)) if self.phase_centre_m is None
                               else np.asarray(self.phase_centre_m, dtype=float).reshape(n, 3))
        self.amplitude = (np.ones(n) if self.amplitude is None
                          else np.asarray(self.amplitude, dtype=float).reshape(n))
        if np.any(self.amplitude < 0):
            raise ValueError("element amplitudes are magnitudes (≥ 0)")
        if self.boresight_deg is not None:
            b = np.asarray(self.boresight_deg, dtype=float).reshape(-1)
            self.boresight_deg = np.full(n, b[0]) if b.size == 1 else b.reshape(n)
        if self.pattern is not None and self.pattern.kind != "omni" and self.boresight_deg is None:
            raise ValueError("a directional element pattern needs boresight_deg")
        if self.channel_of_element is None:
            self.channel_of_element = tuple(range(n))
        m = tuple(int(c) for c in self.channel_of_element)
        if sorted(m) != list(range(n)):
            raise ValueError(f"the connector map must use each of the {n} channels "
                             f"exactly once, not {m}")
        self.channel_of_element = m
        for name in ("fov_az_deg", "fov_el_deg"):
            v = getattr(self, name)
            if v is not None:
                lo, hi = (float(v[0]), float(v[1]))
                if name == "fov_el_deg" and not -90.0 <= lo < hi <= 90.0:
                    raise ValueError("fov_el_deg must be (lo, hi) with −90 ≤ lo < hi ≤ 90")
                if name == "fov_az_deg" and not 0.0 < (hi - lo) <= 360.0:
                    raise ValueError("fov_az_deg must be (lo, hi) with 0 < hi − lo ≤ 360")
                setattr(self, name, (lo, hi))
        if self.groups is not None:
            g = tuple(int(x) for x in self.groups)
            if len(g) != n:
                raise ValueError("groups needs one entry per element")
            self.groups = g
        if self.hemisphere not in ("up", "down", "front"):
            raise ValueError("hemisphere is up, down or front")

    # -- shape ----------------------------------------------------------------
    @property
    def n(self) -> int:
        return int(self.positions.shape[0])

    @property
    def positions3(self) -> np.ndarray:
        """(n, 3) mechanical positions: x forward, y left, z up."""
        return np.column_stack([self.positions, self.z])

    @property
    def electrical3(self) -> np.ndarray:
        """(n, 3) phase centres: mechanical position + the fitted offset."""
        return self.positions3 + self.phase_centre_m

    @property
    def is_planar_horizontal(self) -> bool:
        return bool(np.ptp(self.z) < 1e-9)

    @property
    def has_pattern(self) -> bool:
        return self.pattern is not None and self.pattern.kind != "omni"

    @property
    def polar(self) -> tuple:
        """(ρ, φ) of the ELECTRICAL horizontal positions: element radius, and
        angle counter-clockwise from x (the EADF's coordinates)."""
        e = self.electrical3
        x, y = e[:, 0], e[:, 1]
        return np.hypot(x, y), np.arctan2(y, x)

    def _pairwise(self) -> np.ndarray:
        p = self.positions3
        d = p[:, None, :] - p[None, :, :]
        return np.sqrt(np.sum(d ** 2, axis=-1))

    @property
    def aperture_m(self) -> float:
        return float(np.max(self._pairwise()))

    def adjacent_spacing_m(self) -> float:
        """The largest nearest-neighbour distance (3-D) — the spacing the
        half-wavelength rule is about."""
        dist = self._pairwise()
        np.fill_diagonal(dist, np.inf)
        return float(np.max(np.min(dist, axis=1)))

    def max_unambiguous_hz(self) -> float:
        return C / (2.0 * self.adjacent_spacing_m())

    def reflection_permutation(self) -> tuple | None:
        """perm with element perm[m] at the point reflection of element m
        (through the array centre), or None if the array is not its own
        point reflection.

        Forward-backward averaging pairs each element with its MIRROR
        element. For a ULA or a URA numbered row by row that is plain index
        reversal; for a circle it is NOT — on an 8-element ring the mirror of
        element m is m + 4, while reversal (7 − m) is a reflection about an
        axis, and averaging with it mixes in a ghost array (fixed 2026-10-30;
        it never ran on the Kraken's odd 5-element circle). Directional
        elements pointing different ways, unequal amplitudes, or combined
        (beamspace) channels are never mirror images."""
        if self.kind == "beamspace":
            return None
        if self.has_pattern and np.ptp(np.mod(self.boresight_deg, 360.0)) > 1e-6:
            return None
        if np.ptp(self.amplitude) > 1e-9:
            return None
        p = self.electrical3 - self.electrical3.mean(axis=0)
        d = np.sqrt(np.sum((p[:, None, :] + p[None, :, :]) ** 2, axis=-1))
        perm = np.argmin(d, axis=1)
        if not np.all(d[np.arange(self.n), perm] < 1e-6):
            return None
        if sorted(perm.tolist()) != list(range(self.n)):
            return None
        return tuple(int(i) for i in perm)

    @property
    def centrosymmetric(self) -> bool:
        """Is the array its own point reflection? Forward-backward averaging
        is only valid when it is — a ULA is, a 5-element UCA is NOT (an odd
        circle has no element opposite each element), so for a UCA the
        engine smooths in phase-mode space instead (`doa`). The averaging
        must use `reflection_permutation`, not index reversal."""
        return self.reflection_permutation() is not None

    # -- field of view ----------------------------------------------------------
    def in_fov(self, theta_deg, el_deg=None) -> np.ndarray:
        """Boolean mask: which relative bearings (and elevations) are inside
        the field of view. No FOV = everything."""
        th = _wrap_rel(theta_deg)
        ok = np.ones(th.shape, dtype=bool)
        if self.fov_az_deg is not None:
            lo, hi = self.fov_az_deg
            span = hi - lo
            ok &= (span >= 360.0) | (np.mod(th - lo, 360.0) <= span + 1e-9)
        if el_deg is not None and self.fov_el_deg is not None:
            e = np.asarray(el_deg, dtype=float)
            ok &= (e >= self.fov_el_deg[0] - 1e-9) & (e <= self.fov_el_deg[1] + 1e-9)
        return ok

    def fov_edge_distance_deg(self, theta_deg: float) -> float:
        """Degrees from `theta_deg` to the nearer azimuth FOV edge (inf if
        there is no azimuth FOV)."""
        if self.fov_az_deg is None or (self.fov_az_deg[1] - self.fov_az_deg[0]) >= 360.0:
            return math.inf
        lo, hi = self.fov_az_deg
        d = lambda a, b: abs((a - b + 180.0) % 360.0 - 180.0)
        return float(min(d(theta_deg, lo), d(theta_deg, hi)))

    def boresight_common_deg(self) -> float | None:
        """The one boresight a flat panel has (None for omni / multi-face)."""
        if self.boresight_deg is None:
            return None
        b = np.mod(self.boresight_deg, 360.0)
        return float(b[0]) if np.ptp(b) < 1e-6 else None

    def back_lobe_power(self) -> float:
        """Power fraction an emitter outside the FOV couples through: the
        element pattern's back lobe, or −20 dB for an FOV with no pattern."""
        if self.has_pattern:
            return float(self.pattern.back_amplitude ** 2)
        return 0.01

    # -- checks and reorders ------------------------------------------------------
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

    def subset(self, keep) -> "Geometry":
        """The same array with only elements `keep` (element indices, in
        order) — a dead element dropped (§11.4.4), or one face of a mast.
        The result works on ELEMENT-ORDERED data with those rows kept, so its
        connector map is the identity; `parent_elements` says which they were."""
        keep = tuple(int(k) for k in keep)
        if len(keep) < 2 or len(set(keep)) != len(keep) or min(keep) < 0 or max(keep) >= self.n:
            raise ValueError(f"subset needs ≥ 2 distinct element indices in 0…{self.n - 1}")
        idx = list(keep)
        parent = (tuple(self.parent_elements[i] for i in idx) if self.parent_elements
                  else keep)
        return replace(self, positions=self.positions[idx].copy(), z=self.z[idx].copy(),
                       phase_centre_m=self.phase_centre_m[idx].copy(),
                       amplitude=self.amplitude[idx].copy(),
                       boresight_deg=(None if self.boresight_deg is None
                                      else self.boresight_deg[idx].copy()),
                       groups=(None if self.groups is None
                               else tuple(self.groups[i] for i in idx)),
                       channel_of_element=None, grid=None,
                       id=f"{self.id}_sub{len(idx)}",
                       description=f"{self.description} [elements {list(parent)} of the parent]",
                       parent_elements=parent)

    # -- persistence ---------------------------------------------------------------
    def to_dict(self) -> dict:
        has_z = bool(np.any(np.abs(self.z) > 0))
        d = {"id": self.id, "kind": self.kind, "description": self.description,
             "positions_m": (self.positions3 if has_z else self.positions).tolist(),
             "channel_of_element": list(self.channel_of_element),
             "radius_m": self.radius_m, "spacing_m": self.spacing_m}
        if np.any(self.phase_centre_m != 0):
            d["phase_centre_m"] = self.phase_centre_m.tolist()
        if np.any(self.amplitude != 1):
            d["amplitude"] = self.amplitude.tolist()
        if self.pattern is not None:
            d["pattern"] = self.pattern.to_dict()
        if self.boresight_deg is not None:
            d["boresight_deg"] = self.boresight_deg.tolist()
        if self.tilt_deg:
            d["tilt_deg"] = self.tilt_deg
        if self.fov_az_deg is not None:
            d["fov_az_deg"] = list(self.fov_az_deg)
        if self.fov_el_deg is not None:
            d["fov_el_deg"] = list(self.fov_el_deg)
        if self.groups is not None:
            d["groups"] = list(self.groups)
        if self.grid is not None:
            d["grid"] = {k: (list(v) if isinstance(v, (tuple, list, np.ndarray)) else v)
                         for k, v in self.grid.items()}
        if self.hemisphere != "up":
            d["hemisphere"] = self.hemisphere
        if self.parent_elements is not None:
            d["parent_elements"] = list(self.parent_elements)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Geometry":
        def opt(key):
            v = d.get(key)
            return None if v is None else np.asarray(v, dtype=float)
        return cls(positions=np.asarray(d["positions_m"], dtype=float), id=d.get("id", "custom"),
                   description=d.get("description", ""),
                   channel_of_element=tuple(d.get("channel_of_element") or ()) or None,
                   kind=d.get("kind", "custom"), radius_m=float(d.get("radius_m", 0.0)),
                   spacing_m=float(d.get("spacing_m", 0.0)),
                   phase_centre_m=opt("phase_centre_m"), amplitude=opt("amplitude"),
                   pattern=ElementPattern.from_dict(d.get("pattern")),
                   boresight_deg=opt("boresight_deg"), tilt_deg=float(d.get("tilt_deg", 0.0)),
                   fov_az_deg=tuple(d["fov_az_deg"]) if d.get("fov_az_deg") else None,
                   fov_el_deg=tuple(d["fov_el_deg"]) if d.get("fov_el_deg") else None,
                   groups=tuple(d["groups"]) if d.get("groups") else None,
                   grid=dict(d["grid"]) if d.get("grid") else None,
                   hemisphere=d.get("hemisphere", "up"),
                   parent_elements=tuple(d["parent_elements"]) if d.get("parent_elements") else None)


# ---------------------------------------------------------------------------
# presets
# ---------------------------------------------------------------------------

def uca(n: int = 5, radius_m: float = 0.18, channel_of_element=None,
        centre: bool = False) -> Geometry:
    """Uniform circular array: element 0 toward the nose, counter-clockwise.
    The default is THE preset for this hardware (umbrella §5.1): r = 0.18 m,
    adjacent spacing 0.212 m = 0.32 λ at 450 MHz, 0.5 λ at 707 MHz.

    `centre=True` adds one more element at the centre, LAST (index n): the
    phase reference of a switched array (lit2 §2.14) and the "reference
    antenna" that cancels common phase jitter (lit. §4.10)."""
    phi = 2.0 * np.pi * np.arange(n) / n
    pos = np.stack([radius_m * np.cos(phi), radius_m * np.sin(phi)], axis=1)
    if centre:
        pos = np.vstack([pos, [0.0, 0.0]])
    tag = "c" if centre else ""
    return Geometry(pos, id=f"uca{n}{tag}_r{radius_m:.3f}",
                    kind="uca_centre" if centre else "uca", radius_m=float(radius_m),
                    description=(f"UCA, {n} elements{' + centre' if centre else ''}, radius "
                                 f"{radius_m:.3f} m, element 0 forward, counter-clockwise "
                                 f"from above"),
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


def _unit_h(bearing_deg: float) -> np.ndarray:
    """Horizontal unit vector toward a relative bearing (clockwise from x)."""
    t = math.radians(bearing_deg)
    return np.array([math.cos(t), -math.sin(t), 0.0])


def ura(rows: int, cols: int, spacing_m: float, plane: str = "horizontal",
        boresight_deg: float = 0.0, pattern: ElementPattern | None = None,
        fov_half_deg: float | None = None, channel_of_element=None,
        row_spacing_m: float | None = None) -> Geometry:
    """Uniform rectangular array, centred on the origin. Element index is
    r·cols + c.

    ``plane="horizontal"``: lying flat (a roof plate). Columns run along y
    (left), rows along x (forward). Azimuth all round; elevation with an
    up/down ambiguity (`hemisphere`) and poor sensitivity near the horizon
    (lit2 §2.8) — say so rather than promise it.

    ``plane="vertical"``: a panel standing up, facing `boresight_deg`.
    Columns run horizontally across the face (to the right looking out),
    rows run up. Elevation is measured well near the horizon; azimuth only in
    front, so a vertical panel gets a cardioid element pattern and a ±60°
    field of view unless told otherwise."""
    if rows < 1 or cols < 1 or rows * cols < 2:
        raise ValueError("a URA needs at least two elements")
    dr = float(row_spacing_m if row_spacing_m is not None else spacing_m)
    dc = float(spacing_m)
    if plane == "horizontal":
        col_axis = np.array([0.0, 1.0, 0.0])
        row_axis = np.array([1.0, 0.0, 0.0])
        hemi = "up"
    elif plane == "vertical":
        col_axis = _unit_h(boresight_deg + 90.0)          # to the right, looking out
        row_axis = np.array([0.0, 0.0, 1.0])
        hemi = "front"
        if pattern is None:
            pattern = ElementPattern("cardioid", 1.0, -20.0)
        if fov_half_deg is None:
            fov_half_deg = 60.0
    else:
        raise ValueError("plane is horizontal or vertical")
    rr, cc = np.meshgrid(np.arange(rows) - (rows - 1) / 2.0,
                         np.arange(cols) - (cols - 1) / 2.0, indexing="ij")
    pos = (rr.reshape(-1, 1) * dr * row_axis[None, :]
           + cc.reshape(-1, 1) * dc * col_axis[None, :])
    n = rows * cols
    fov = None
    if fov_half_deg is not None:
        fov = (boresight_deg - float(fov_half_deg), boresight_deg + float(fov_half_deg))
    return Geometry(pos, id=f"ura{rows}x{cols}_{plane[0]}_d{dc:.3f}", kind="ura",
                    spacing_m=dc,
                    description=(f"URA {rows}×{cols}, {plane}, spacing {dc:.3f} m"
                                 + (f", facing {boresight_deg:.0f}°" if plane == "vertical" else "")),
                    pattern=pattern,
                    boresight_deg=(np.full(n, float(boresight_deg)) if pattern is not None else None),
                    fov_az_deg=fov, channel_of_element=channel_of_element,
                    grid={"rows": int(rows), "cols": int(cols),
                          "row_axis": row_axis.tolist(), "col_axis": col_axis.tolist(),
                          "row_spacing_m": dr, "col_spacing_m": dc},
                    hemisphere=hemi)


def l_array(n_a: int, n_b: int, spacing_m: float, plane: str = "horizontal",
            boresight_deg: float = 0.0, pattern: ElementPattern | None = None,
            channel_of_element=None) -> Geometry:
    """L-shaped array sharing element 0 at the corner: arm A has n_a elements
    (incl. the corner), arm B n_b − 1 more. Horizontal: A along x, B along y.
    Vertical (facing `boresight_deg`): A across the face, B up — the classic
    2-D DoA layout (lit2 §2.4)."""
    if plane == "horizontal":
        ax_a, ax_b, hemi = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), "up"
    elif plane == "vertical":
        ax_a, ax_b, hemi = _unit_h(boresight_deg + 90.0), np.array([0.0, 0.0, 1.0]), "front"
        if pattern is None:
            pattern = ElementPattern("cardioid", 1.0, -20.0)
    else:
        raise ValueError("plane is horizontal or vertical")
    pts = [ax_a * spacing_m * i for i in range(n_a)] + [ax_b * spacing_m * j for j in range(1, n_b)]
    pos = np.asarray(pts)
    n = pos.shape[0]
    fov = (boresight_deg - 60.0, boresight_deg + 60.0) if plane == "vertical" else None
    return Geometry(pos, id=f"l{n_a}x{n_b}_{plane[0]}_d{spacing_m:.3f}", kind="l",
                    spacing_m=float(spacing_m),
                    description=f"L-array {n_a}+{n_b - 1}, {plane}, spacing {spacing_m:.3f} m",
                    pattern=pattern,
                    boresight_deg=np.full(n, float(boresight_deg)) if pattern is not None else None,
                    fov_az_deg=fov, channel_of_element=channel_of_element, hemisphere=hemi)


def multi_face(faces: int = 4, cols: int = 4, spacing_m: float = 0.18, standoff_m: float = 0.25,
               rows: int = 1, pattern: ElementPattern | None = None,
               first_face_deg: float = 0.0) -> Geometry:
    """Vertical panels round a mast, face f looking at first_face_deg +
    f·360/faces, each `standoff_m` out from the mast. Elements numbered face
    by face; `groups` says which face. One coherent array (one manifold whose
    element patterns do the hand-off) only if every face shares an LO;
    otherwise each face is its own array and `architectures` picks the face
    by power (§11.4.2)."""
    pattern = pattern or ElementPattern("cardioid", 1.0, -20.0)
    pts, bs, grp = [], [], []
    for f in range(faces):
        b = first_face_deg + f * 360.0 / faces
        out = _unit_h(b) * standoff_m
        right = _unit_h(b + 90.0)
        for r in range(rows):
            for c in range(cols):
                pts.append(out + right * (c - (cols - 1) / 2.0) * spacing_m
                           + np.array([0.0, 0.0, (r - (rows - 1) / 2.0) * spacing_m]))
                bs.append(b)
                grp.append(f)
    return Geometry(np.asarray(pts), id=f"mast{faces}x{rows}x{cols}_d{spacing_m:.3f}",
                    kind="multi_face", spacing_m=float(spacing_m),
                    description=f"{faces} vertical faces of {rows}×{cols}, spacing "
                                f"{spacing_m:.3f} m, standoff {standoff_m:.2f} m",
                    pattern=pattern, boresight_deg=np.asarray(bs), groups=tuple(grp),
                    hemisphere="front")


def direction(theta_deg) -> np.ndarray:
    """(K, 2) unit vectors toward relative bearings θ (clockwise from x)."""
    t = np.radians(np.asarray(theta_deg, dtype=float).reshape(-1))
    return np.stack([np.cos(t), -np.sin(t)], axis=1)


def direction3(theta_deg, el_deg=0.0) -> np.ndarray:
    """(K, 3) unit vectors toward relative bearing θ (clockwise from x) and
    elevation (up from the horizontal plane)."""
    t = np.radians(np.asarray(theta_deg, dtype=float).reshape(-1))
    e = np.radians(np.broadcast_to(np.asarray(el_deg, dtype=float), t.shape).reshape(-1))
    ce = np.cos(e)
    return np.stack([ce * np.cos(t), -ce * np.sin(t), np.sin(e)], axis=1)


def eadf_order(geometry: Geometry, f_hz: float) -> int:
    """How many Fourier terms each side represent the ideal manifold to
    ~1e-9: |J_q(z)| falls off like (z/2)^q / q! past q ≈ z. A directional
    element pattern adds its own order on top."""
    z = 2.0 * np.pi / wavelength(f_hz) * float(np.max(geometry.polar[0]))
    extra = 0
    if geometry.has_pattern:
        extra = int(math.ceil(geometry.pattern.exponent)) + 8
    return int(math.ceil(z)) + 12 + extra


def eadf_numeric(response_fn, Q: int) -> tuple:
    """(G, q) by FFT of `response_fn(theta_deg) -> (n, K)` sampled on a fine
    circle: for any manifold whose response is smooth in θ (patterns,
    wrappers) when the closed form does not apply."""
    M_ = max(64, 4 * (2 * Q + 1))
    th = np.arange(M_) * 360.0 / M_
    A = response_fn(th)                                   # (n, M)
    q = np.arange(-Q, Q + 1)
    E = np.exp(-1j * np.outer(np.radians(th), q))         # (M, 2Q+1)
    return (A @ E) / M_, q


class Manifold:
    """What the array sees from θ. `response(theta, f)` is (n, K);
    `response2(az, el, f)` is (n, K) for paired azimuth/elevation."""

    kind = "ideal"
    id = ""

    def response(self, theta_deg, f_hz: float) -> np.ndarray:
        raise NotImplementedError

    def response2(self, az_deg, el_deg, f_hz: float) -> np.ndarray:
        """Default: an azimuth-only manifold is valid at elevation 0 only."""
        el = np.asarray(el_deg, dtype=float)
        if np.any(np.abs(el) > 1e-9):
            raise NotImplementedError(f"the {self.kind} manifold is azimuth-only; "
                                      f"it has no elevation response")
        return self.response(az_deg, f_hz)

    def eadf(self, f_hz: float, order: int | None = None) -> tuple:
        """(G, q): the Fourier coefficients with a(θ) = G @ exp(j q θ)."""
        raise NotImplementedError

    def caveats(self, f_hz: float) -> list:
        return []

    def subset(self, keep) -> "Manifold":
        raise NotImplementedError(f"{type(self).__name__} cannot drop elements")


class IdealManifold(Manifold):
    """Free-space elements at the geometry's phase centres, with the element
    pattern and static amplitudes the geometry carries."""

    kind = "ideal"

    def __init__(self, geometry: Geometry):
        self.geometry = geometry
        self.id = ""

    def _pattern_gain(self, U: np.ndarray) -> np.ndarray:
        """(n, K) element voltage gain for unit directions U (K, 3)."""
        g = self.geometry
        amp = g.amplitude[:, None]
        if not g.has_pattern:
            return np.broadcast_to(amp, (g.n, U.shape[0]))
        B = direction3(g.boresight_deg, g.tilt_deg)       # (n, 3)
        return amp * g.pattern.gain(B @ U.T)

    def response2(self, az_deg, el_deg, f_hz: float) -> np.ndarray:
        k = 2.0 * np.pi / wavelength(f_hz)
        U = direction3(az_deg, el_deg)
        a = np.exp(1j * k * (self.geometry.electrical3 @ U.T))
        return a * self._pattern_gain(U)

    def response(self, theta_deg, f_hz: float) -> np.ndarray:
        return self.response2(theta_deg, 0.0, f_hz)

    def eadf(self, f_hz: float, order: int | None = None) -> tuple:
        Q = eadf_order(self.geometry, f_hz) if order is None else int(order)
        g = self.geometry
        if g.has_pattern:
            return eadf_numeric(lambda th: self.response(th, f_hz), Q)
        q = np.arange(-Q, Q + 1)
        rho, phi = g.polar
        k = 2.0 * np.pi / wavelength(f_hz)
        # a_m(θ) = A_m e^{jk z_m·0} Σ_q j^q J_q(kρ_m) e^{jqφ_m} e^{jqθ}, θ in
        # RADIANS clockwise. (z does not enter at elevation 0.)
        G = (1j ** q)[None, :] * jv(q[None, :], k * rho[:, None]) * np.exp(1j * q[None, :] * phi[:, None])
        return g.amplitude[:, None] * G, q

    def caveats(self, f_hz: float) -> list:
        return ["IDEAL MANIFOLD: the steering vectors are free-space geometry. "
                "atkdaq's calibration ends at the noise-source switch, so cables, "
                "connectors, whips and the roof are not in this answer; a measured "
                "manifold is what removes their error."]

    def subset(self, keep) -> "IdealManifold":
        return IdealManifold(self.geometry.subset(keep))


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
            p3 = geometry.positions3
            dist = np.sqrt(np.sum((p3[:, None, :] - p3[None, :, :]) ** 2, axis=-1))
            near = dist <= geometry.adjacent_spacing_m() * 1.01
            np.fill_diagonal(near, False)
            ph = np.exp(1j * rng.uniform(-np.pi, np.pi, (n, n)))
            Cm = Cm + c * near * ph
        return cls(geometry, g, Cm, id=f"perturbed_seed{seed}")

    def response(self, theta_deg, f_hz: float) -> np.ndarray:
        return self.coupling @ (self.gains[:, None] * self.ideal.response(theta_deg, f_hz))

    def response2(self, az_deg, el_deg, f_hz: float) -> np.ndarray:
        return self.coupling @ (self.gains[:, None] * self.ideal.response2(az_deg, el_deg, f_hz))

    def eadf(self, f_hz: float, order: int | None = None) -> tuple:
        G, q = self.ideal.eadf(f_hz, order)
        return self.coupling @ (self.gains[:, None] * G), q

    def subset(self, keep) -> "PerturbedManifold":
        k = list(int(i) for i in keep)
        return PerturbedManifold(self.geometry.subset(k), self.gains[k],
                                 self.coupling[np.ix_(k, k)], id=f"{self.id}_sub{len(k)}")


class BeamspaceManifold(Manifold):
    """What the RECEIVER CHANNELS of a hybrid (or switched-beam) array see:
    each channel is a weighted sum of elements, so the channel response is
    A_b(θ) = Wᴴ a(θ) (lit2 §1.9 — the literature's own model). `weights` is
    (n_elements, n_channels); column c holds the analog weights feeding
    channel c, zero for elements not on it. Pass the weights ACTUALLY applied
    — quantised by the phase shifters, times their calibration — not the
    ones asked for (lit2 §1.7).

    Every estimator that takes a Manifold runs on this unchanged, including
    root-MUSIC (the EADF of a weighted sum is the weighted sum of EADFs).
    The `geometry` is a stand-in the size of the channel count, so code that
    asks the geometry for n sees the channels; the element geometry is
    `element_geometry`."""

    def __init__(self, base: Manifold, weights, id: str = "", phase_centres_m=None):
        self.base = base
        W = np.asarray(weights, dtype=complex)
        eg = getattr(base, "geometry", None)
        if eg is None or W.ndim != 2 or W.shape[0] != eg.n:
            raise ValueError("weights must be (n_elements, n_channels) for the base manifold")
        if W.shape[1] < 1:
            raise ValueError("a beamspace manifold needs at least one channel")
        self.W = W
        self.element_geometry = eg
        self.kind = f"beamspace({getattr(base, 'kind', 'ideal')})"
        self.id = id or f"beamspace_{getattr(base, 'id', '') or eg.id}_{W.shape[1]}ch"
        # channel "positions": the power-weighted centroid of each channel's
        # elements, so sizes, CRB and caveats have something sensible to read
        if phase_centres_m is None:
            w2 = np.abs(W) ** 2
            tot = np.maximum(w2.sum(axis=0), 1e-30)
            phase_centres_m = (w2.T @ eg.positions3) / tot[:, None]
        pc = np.asarray(phase_centres_m, dtype=float).reshape(W.shape[1], 3)
        if W.shape[1] >= 2:
            self.geometry = Geometry(pc, id=f"{eg.id}_beams{W.shape[1]}", kind="beamspace",
                                     description=f"{W.shape[1]} combined channels of {eg.id}",
                                     fov_az_deg=eg.fov_az_deg, fov_el_deg=eg.fov_el_deg,
                                     hemisphere=eg.hemisphere)
        else:
            self.geometry = None                   # one channel: no subspace, beam-scan only

    @property
    def model_kind(self) -> str:
        return getattr(self.base, "kind", "ideal")

    def response(self, theta_deg, f_hz: float) -> np.ndarray:
        return self.W.conj().T @ self.base.response(theta_deg, f_hz)

    def response2(self, az_deg, el_deg, f_hz: float) -> np.ndarray:
        return self.W.conj().T @ self.base.response2(az_deg, el_deg, f_hz)

    def eadf(self, f_hz: float, order: int | None = None) -> tuple:
        G, q = self.base.eadf(f_hz, order)
        return self.W.conj().T @ G, q

    def caveats(self, f_hz: float) -> list:
        return list(self.base.caveats(f_hz)) + [
            f"HYBRID/BEAMSPACE: {self.W.shape[1]} combined channel(s) from "
            f"{self.element_geometry.n} elements; at most {max(self.W.shape[1] - 1, 0)} "
            f"source(s) per look, and the answer is only as good as the applied "
            f"weights' calibration"]


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
                 coverage_deg: float = 360.0, mode: str = "full"):
        if mode not in ("full", "residual"):
            raise ValueError("mode is full or residual")
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
        #: "full": G is the whole response at f_hz (as before). "residual":
        #: G is the element pattern AFTER the geometric phase of the fitted
        #: phase centres (`geometry.phase_centre_m`) is taken out; the
        #: response at any f is that pattern times the ideal phase at f —
        #: the Lund "enhanced EADF" form that makes a table interpolate
        #: across a band (lit2 §2.23).
        self.mode = mode
        self._ideal = IdealManifold(replace(geometry, pattern=None, boresight_deg=None,
                                            amplitude=None)) if mode == "residual" else None

    def response(self, theta_deg, f_hz: float) -> np.ndarray:
        t = np.radians(np.asarray(theta_deg, dtype=float).reshape(-1))
        a = self.G @ np.exp(1j * np.outer(self.q, t))
        if self.mode == "residual":
            a = a * self._ideal.response(theta_deg, f_hz)
        return a

    def eadf(self, f_hz: float, order: int | None = None) -> tuple:
        if self.mode == "residual":
            Gi, qi = self._ideal.eadf(f_hz)
            Gt = np.stack([np.convolve(self.G[m], Gi[m]) for m in range(self.G.shape[0])])
            qt = np.arange(self.q.min() + qi.min(), self.q.max() + qi.max() + 1)
            if order is not None and order < qt.max():
                keep = np.abs(qt) <= int(order)
                return Gt[:, keep], qt[keep]
            return Gt, qt
        if order is None or order >= self.q.max():
            return self.G, self.q
        keep = np.abs(self.q) <= int(order)
        return self.G[:, keep], self.q[keep]

    def caveats(self, f_hz: float) -> list:
        out = []
        lo, hi = self.band_hz
        if not lo <= f_hz <= hi:
            if self.mode == "residual":
                out.append(f"the measured manifold {self.id} was measured at "
                           f"{self.f_hz / 1e6:.1f} MHz and is used at {f_hz / 1e6:.1f} MHz "
                           f"with its phase centres removed (lit2 §2.23); how far in "
                           f"frequency that holds is not yet measured")
            else:
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
                       weights=None, phase_centres: bool = False, residual_order: int = 6,
                       **kw) -> "MeasuredManifold":
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
        if phase_centres:
            return cls._fit_residual(geometry, f_hz, th, V, bin_deg, residual_order,
                                     weights, **kw)
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

    @classmethod
    def _fit_residual(cls, geometry: Geometry, f_hz: float, th, V, bin_deg: float,
                      residual_order: int, weights, **kw) -> "MeasuredManifold":
        """`phase_centres=True`: fit each element's horizontal phase-centre
        offset, then a LOW-order Fourier series of what is left (the element
        pattern, coupling, cable phase) — smooth in θ and close to
        frequency-independent, so the table interpolates across the band."""
        offs = fit_phase_centres(geometry, f_hz, th, V)
        g2 = replace(geometry, phase_centre_m=geometry.phase_centre_m + offs)
        ideal2 = IdealManifold(replace(g2, pattern=None, boresight_deg=None, amplitude=None))
        A = ideal2.response(th, f_hz).T                     # (K, n)
        Rr = V / np.where(np.abs(A) > 1e-12, A, 1e-12)
        # the emitter's phase per measurement: take out the common phase
        c = np.angle(np.sum(Rr, axis=1))
        Rr = Rr * np.exp(-1j * c)[:, None]
        Rr = Rr * (math.sqrt(geometry.n) / np.maximum(np.linalg.norm(Rr, axis=1, keepdims=True), 1e-30))
        w = np.ones(th.size) if weights is None else np.asarray(weights, float)
        nb = int(round(360.0 / bin_deg))
        idx = np.floor(th / (360.0 / nb)).astype(int) % nb
        centres, means = [], []
        for b in range(nb):
            sel = idx == b
            if not np.any(sel):
                continue
            means.append(np.sum(Rr[sel] * w[sel, None], axis=0) / np.sum(w[sel]))
            centres.append(np.sum(th[sel] * w[sel]) / np.sum(w[sel]))
        centres = np.asarray(centres)
        means = np.asarray(means)
        Q = int(residual_order)
        q = np.arange(-Q, Q + 1)
        D = np.exp(1j * np.outer(np.radians(centres), q))
        G0 = np.zeros((geometry.n, q.size), complex)
        G0[:, Q] = 1.0                                     # prior: an omni element
        lam = 1e-2
        Am = D.conj().T @ D + lam * len(centres) / q.size * np.eye(q.size)
        G = np.linalg.solve(Am, D.conj().T @ means + (lam * len(centres) / q.size) * G0.T).T
        fit = D @ G.T
        resid = float(np.degrees(np.sqrt(np.mean(np.angle(fit * np.conj(means)) ** 2))))
        covered = len(set(idx.tolist())) * (360.0 / nb)
        return cls(g2, f_hz, G, q, residual_deg=resid, coverage_deg=covered, mode="residual",
                   **kw)

    def save(self, path) -> Path:
        path = Path(path)
        np.savez_compressed(path, G=self.G, q=self.q, f_hz=self.f_hz,
                            band_hz=np.asarray(self.band_hz),
                            meta=json.dumps({"id": self.id, "passport_uid": self.passport_uid,
                                             "source": self.source,
                                             "residual_deg": self.residual_deg,
                                             "coverage_deg": self.coverage_deg,
                                             "mode": self.mode,
                                             "geometry": self.geometry.to_dict()}))
        return path if path.suffix == ".npz" else path.with_suffix(path.suffix + ".npz")

    def subset(self, keep) -> "MeasuredManifold":
        k = list(int(i) for i in keep)
        return MeasuredManifold(self.geometry.subset(k), self.f_hz, self.G[k], self.q,
                                id=f"{self.id}_sub{len(k)}", passport_uid=self.passport_uid,
                                source=self.source, band_hz=self.band_hz,
                                residual_deg=self.residual_deg, coverage_deg=self.coverage_deg,
                                mode=self.mode)

    @classmethod
    def load(cls, path) -> "MeasuredManifold":
        z = np.load(Path(path), allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        geom = Geometry.from_dict(meta["geometry"])
        return cls(geom, float(z["f_hz"]), z["G"], z["q"], id=meta.get("id", ""),
                   passport_uid=meta.get("passport_uid", ""), source=meta.get("source", ""),
                   band_hz=tuple(z["band_hz"].tolist()), residual_deg=meta.get("residual_deg"),
                   coverage_deg=float(meta.get("coverage_deg", 360.0)),
                   mode=meta.get("mode", "full"))


def fit_phase_centres(geometry: Geometry, f_hz: float, theta_deg, vectors) -> np.ndarray:
    """Each element's horizontal phase-centre offset (n, 3; z = 0), relative
    to the array, from measured steering vectors at known bearings.

    With the ideal geometric phase taken out, element m's residual phase is
    c_m + k(δx_m·cos θ − δy_m·sin θ) + (pattern phase, small and smooth): a
    straight-line fit in [1, cos θ, −sin θ] per element after removing each
    measurement's common (emitter) phase. A translation of the whole array is
    common to every element and unobservable, so the offsets are returned
    with zero mean. Elevation-0 measurements cannot see z."""
    th = np.asarray(theta_deg, dtype=float).reshape(-1)
    V = np.asarray(vectors, dtype=complex)
    k = 2.0 * np.pi / wavelength(f_hz)
    ideal = IdealManifold(replace(geometry, pattern=None, boresight_deg=None, amplitude=None))
    A = ideal.response(th, f_hz).T
    R = V / np.where(np.abs(A) > 1e-12, A, 1e-12)
    R = R * np.exp(-1j * np.angle(np.sum(R, axis=1)))[:, None]
    ph = np.angle(R)                                    # (K, n), small after the ideal is out
    t = np.radians(th)
    X = np.column_stack([np.ones_like(t), k * np.cos(t), -k * np.sin(t)])
    sol, *_ = np.linalg.lstsq(X, ph, rcond=None)        # (3, n)
    offs = np.zeros((geometry.n, 3))
    offs[:, 0] = sol[1]
    offs[:, 1] = sol[2]
    offs[:, :2] -= offs[:, :2].mean(axis=0)
    return offs
