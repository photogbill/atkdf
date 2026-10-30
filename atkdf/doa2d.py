"""Two-dimensional DoA: azimuth AND elevation (plan §11.4.5).

An azimuth-only bearing is enough for a handset on the ground seen from a
vehicle. It is not enough for a drone overhead, an aircraft calibrator, or a
handset seen from a panel high on a mast — there the emitter is off the
horizontal plane and an azimuth-only manifold places it wrong. A planar array
(a URA, an L, a vertical panel) can measure both angles; a horizontal ring
can measure elevation only weakly and with an up/down ambiguity, so this
module says which case it is in rather than promising elevation it cannot
give (lit2 §2.8).

Two estimators, both on the real (measured or ideal) manifold:

  music2d   the pseudo-spectrum over an (azimuth, elevation) grid, searched
            COARSE then FINE so a full fine grid is never built — the
            reduced-dimension idea of lit2 §2.1 without a URA's special
            structure, so it runs on any geometry, including a measured
            manifold and a UCA+centre. This is the default.

  esprit2d  closed-form, auto-paired, for a uniform rectangular array on the
            IDEAL manifold: the row-shift and column-shift invariances share
            eigenvectors, so the two angles come out already paired, with no
            search (lit2 §2.3, the Kronecker/ESPRIT route). Fast at N ≥ 16.

Angles: azimuth is the relative bearing (clockwise from the array's forward
axis, as everywhere in atkdf); elevation is degrees above the horizontal
plane. `direction3` turns (az, el) into the unit vector the manifold uses.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from . import doa as _doa
from .manifold import Geometry, Manifold, direction3, wavelength

log = logging.getLogger("atkdf.doa2d")

METHODS_2D = ("music2d", "esprit2d")


@dataclass
class Doa2DResult:
    az_deg: float
    el_deg: float
    sigma_az_deg: float
    sigma_el_deg: float
    estimator: str
    manifold: str
    n_sources: int
    all_az_deg: list = field(default_factory=list)
    all_el_deg: list = field(default_factory=list)
    quality: dict = field(default_factory=dict)
    caveats: list = field(default_factory=list)

    def as_doaresult(self) -> "_doa.DoaResult":
        """The azimuth as an ordinary DoaResult, so the fusion / records /
        map path downstream is unchanged; elevation rides in `quality`."""
        q = dict(self.quality)
        q.update({"elevation_deg": round(self.el_deg, 2),
                  "sigma_el_deg": round(self.sigma_el_deg, 2),
                  "all_el_deg": [round(e, 2) for e in self.all_el_deg]})
        g = _doa.make_grid()
        lik = _doa.build_likelihood([(self.az_deg, 1.0)], self.sigma_az_deg)
        return _doa.DoaResult(theta_deg=self.az_deg, sigma_deg=self.sigma_az_deg,
                              estimator=self.estimator, manifold=self.manifold,
                              n_sources=self.n_sources, all_theta_deg=list(self.all_az_deg),
                              grid_deg=g, spectrum=None, likelihood=lik, quality=q,
                              caveats=list(self.caveats))


def elevation_capability(geometry: Geometry) -> tuple:
    """(can_elevation, note). Elevation needs a two-dimensional aperture:
    vertical extent measures it well; a horizontal array that spans two
    dimensions measures it weakly and up/down-ambiguous; a single line of
    elements (a 1-row panel, a straight ULA) cannot measure it at all,
    whichever way it faces — it has only one spatial dimension, so (az, el)
    lie on one direction cosine and cannot be separated.

    A `hemisphere == "front"` panel is NOT automatically capable: a 1×N
    vertical panel is still one horizontal line of elements with no vertical
    aperture. The test is the aperture's real dimensionality, not the mount."""
    if np.ptp(geometry.z) > 1e-6:
        return True, "vertical aperture: elevation measured"
    xy = geometry.positions - geometry.positions.mean(axis=0)
    sv = np.linalg.svd(xy, compute_uv=False)
    two_d = sv.size >= 2 and sv[1] > 1e-6 * max(float(sv[0]), 1e-30)
    if two_d and geometry.kind in ("ura", "l"):
        return True, ("horizontal planar array: elevation has an up/down ambiguity and "
                      "is weak near the horizon (lit2 §2.8)")
    return False, ("this array is a single line (or a ring the engine does not trust for "
                   "elevation): essentially no elevation resolution; elevation is pinned "
                   "to 0°")


def _el_range(geometry: Geometry, el_lo: float, el_hi: float) -> tuple:
    """The elevation search window. When the array has an elevation FOV, that
    IS the window (a panel aimed below the horizon searches below the
    horizon) rather than an intersection with the default, which could be
    empty and crash the grid. Always returns lo < hi."""
    can, _note = elevation_capability(geometry)
    if not can:
        return 0.0, 0.0
    if geometry.fov_el_deg is not None:
        el_lo, el_hi = float(geometry.fov_el_deg[0]), float(geometry.fov_el_deg[1])
    if el_hi <= el_lo:
        el_lo, el_hi = el_lo, el_lo + 1e-6
    return el_lo, el_hi


def _grid_1d(lo: float, hi: float, step: float) -> np.ndarray:
    """An inclusive [lo, hi] grid that is never empty (at least the one point
    lo when hi <= lo)."""
    if hi <= lo:
        return np.array([float(lo)])
    return np.arange(lo, hi + 1e-9, step)


def _az_grid(geometry: Geometry, step: float) -> np.ndarray:
    lo, hi = geometry.fov_az_deg or (0.0, 360.0)
    if hi - lo >= 360.0:
        n = int(round(360.0 / step))
        return np.arange(n) * (360.0 / n)
    return np.arange(lo, hi + 1e-9, step)


def _music_metric(En, A):
    """1/‖Eₙᴴ a‖² for steering matrix A (n, K), columns a."""
    p = En.conj().T @ A
    return 1.0 / np.maximum(np.real(np.sum(p.conj() * p, axis=0)), 1e-300)


def music2d(x, manifold: Manifold, f_hz: float, sources: int | None = None,
            geometry: Geometry | None = None, coarse_az: float = 3.0, coarse_el: float = 3.0,
            fine: float = 0.25, el_lo: float = -10.0, el_hi: float = 80.0,
            snapshots: int | None = None, fb: bool | None = None) -> Doa2DResult:
    """2-D MUSIC, coarse grid then a local fine grid around each coarse peak.

    The coarse grid is (field-of-view azimuth × elevation) at `coarse_*`
    degrees; the `sources` strongest coarse peaks are each refined on a
    ±`coarse` window at `fine` degrees. A full 0.25° × 0.25° grid over a
    hemisphere is ~1M points; this builds tens of thousands."""
    geometry = geometry if geometry is not None else manifold.geometry
    x = np.asarray(x)
    if isinstance(x, _doa.Covariance):
        cov = x
    else:
        kind = getattr(manifold, "kind", "ideal")
        perm = geometry.reflection_permutation()
        usefb = (kind == "ideal" and perm is not None) if fb is None else bool(fb)
        R = (x @ x.conj().T) / x.shape[1]
        if usefb:
            R = _doa.forward_backward(R, perm)
        cov = _doa.Covariance(R=R, snapshots=int(snapshots or x.shape[1]), f_hz=float(f_hz),
                              fb=usefb)
    R = cov.R
    n = R.shape[0]
    _s, _ns, vals = _doa.noise_subspace(R, 1)
    enum = _doa.enumerate_sources(vals, cov.snapshots, fb=cov.fb)
    k = max(1, min(int(sources) if sources else enum["mdl"], n - 1))
    En = _doa.noise_subspace(R, k)[1]

    can_el, el_note = elevation_capability(geometry)
    e_lo, e_hi = _el_range(geometry, el_lo, el_hi)
    az_c = _az_grid(geometry, coarse_az)
    el_c = np.array([0.0]) if not can_el else _grid_1d(e_lo, e_hi, coarse_el)
    AZ, EL = np.meshgrid(az_c, el_c, indexing="ij")
    A = manifold.response2(AZ.ravel(), EL.ravel(), f_hz)
    A = A / np.linalg.norm(A, axis=0, keepdims=True)
    spec = _music_metric(En, A).reshape(AZ.shape)

    # coarse peaks: local maxima, az wrapped if the grid is full 360
    full = (geometry.fov_az_deg is None) or (geometry.fov_az_deg[1] - geometry.fov_az_deg[0] >= 360)
    peaks = _local_maxima_2d(spec, wrap_axis0=full)
    peaks = peaks[:max(k, 1)]
    refined = []
    for ia, ie in peaks:
        a0, e0 = az_c[ia], el_c[ie]
        az_f = _grid_1d(a0 - coarse_az, a0 + coarse_az, fine)
        el_f = (np.array([0.0]) if not can_el
                else _grid_1d(max(e0 - coarse_el, e_lo), min(e0 + coarse_el, e_hi), fine))
        AZf, ELf = np.meshgrid(az_f, el_f, indexing="ij")
        Af = manifold.response2(AZf.ravel() % 360.0, ELf.ravel(), f_hz)
        Af = Af / np.linalg.norm(Af, axis=0, keepdims=True)
        sf = _music_metric(En, Af).reshape(AZf.shape)
        j = np.unravel_index(int(np.argmax(sf)), sf.shape)
        refined.append((float(AZf[j] % 360.0), float(ELf[j]), float(sf[j])))
    refined.sort(key=lambda r: -r[2])
    if not refined:
        refined = [(0.0, 0.0, 1.0)]
    az0, el0, _p = refined[0]

    snr = _doa._snr_from_eigs(vals, k)
    crb_az = _doa.cramer_rao_deg(snr, cov.snapshots, geometry, f_hz, az0)
    sig_az = math.hypot(crb_az, _doa.model_sigma_deg(manifold, geometry, az0))
    sig_el = _elevation_sigma(geometry, f_hz, snr, cov.snapshots, az0, el0, can_el)
    caveats = list(manifold.caveats(f_hz))
    caveats.append("2-D: " + el_note)
    if can_el and not (np.ptp(geometry.z) > 1e-6 or geometry.hemisphere == "front"):
        # a horizontal planar array: report the mirror elevation too
        caveats.append(f"elevation {el0:.0f}° has a mirror at {-el0:.0f}° (an array in one "
                       f"horizontal plane cannot tell up from down)")
    quality = {"architecture": "digital", "method": "music2d",
               "confidence": float(enum["confidence"]), "snr_db": round(10 * math.log10(snr), 1),
               "n_snapshots": cov.snapshots, "source_count_mdl": enum["mdl"],
               "elevation_capable": can_el}
    _doa._fov_note(geometry, az0, quality, caveats)
    log.debug("music2d: az=%.2f el=%.2f (σ %.2f/%.2f) k=%d", az0, el0, sig_az, sig_el, k)
    return Doa2DResult(az_deg=az0, el_deg=el0, sigma_az_deg=sig_az, sigma_el_deg=sig_el,
                       estimator="music2d", manifold=getattr(manifold, "kind", "ideal"),
                       n_sources=k, all_az_deg=[r[0] for r in refined],
                       all_el_deg=[r[1] for r in refined], quality=quality, caveats=caveats)


def _local_maxima_2d(spec: np.ndarray, wrap_axis0: bool) -> list:
    """(i, j) of local maxima, strongest first. Axis 0 is azimuth (wrapped if
    the whole circle is covered), axis 1 elevation (not wrapped)."""
    s = np.asarray(spec, dtype=float)
    na, ne = s.shape
    out = []
    for i in range(na):
        for j in range(ne):
            v = s[i, j]
            ok = True
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ii = (i + di) % na if wrap_axis0 else i + di
                    jj = j + dj
                    if not wrap_axis0 and not (0 <= ii < na):
                        continue
                    if not (0 <= jj < ne):
                        continue
                    if s[ii, jj] > v:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                out.append((i, j, v))
    out.sort(key=lambda t: -t[2])
    return [(i, j) for i, j, _v in out] or [tuple(np.unravel_index(int(np.argmax(s)), s.shape))]


def _elevation_sigma(geometry, f_hz, snr, snapshots, az, el, can_el) -> float:
    """CRB-like elevation sigma from the array's vertical aperture, plus the
    model prior; large when there is little vertical extent."""
    if not can_el:
        return 90.0
    lam = wavelength(f_hz)
    p = geometry.positions3
    u = direction3(az, el)[0]
    # derivative of the array phase w.r.t. elevation
    d_el = np.array([-math.sin(math.radians(el)) * math.cos(math.radians(az)),
                     math.sin(math.radians(el)) * math.sin(math.radians(az)),
                     math.cos(math.radians(el))])
    k = 2 * np.pi / lam
    dphase = k * (p @ d_el)
    dphase -= dphase.mean()
    fisher = 2.0 * max(1, snapshots) * max(snr, 1e-6) * float(np.sum(dphase ** 2))
    crb = math.degrees(1.0 / math.sqrt(fisher)) if fisher > 0 else 90.0
    return math.hypot(max(crb, 0.1), _doa.MODEL_SIGMA_DEG.get("ideal", 4.0))


# ---------------------------------------------------------------------------
# 2-D unitary ESPRIT for a uniform rectangular array (ideal manifold)
# ---------------------------------------------------------------------------

def esprit2d(x, geometry: Geometry, f_hz: float, sources: int | None = None,
             snapshots: int | None = None) -> Doa2DResult:
    """Closed-form paired (az, el) for a URA, ideal manifold. The row and
    column shift invariances of a rectangular grid give two sets of
    eigenvalues from ONE joint eigendecomposition (Kronecker ESPRIT), so the
    pairing is automatic — no 2-D search. Needs the geometry's `grid`."""
    if not geometry.grid or geometry.kind != "ura":
        raise ValueError("esprit2d needs a uniform rectangular array (geometry.grid)")
    g = geometry.grid
    rows, cols = int(g["rows"]), int(g["cols"])
    if rows < 2 or cols < 2:
        raise ValueError(f"2-D ESPRIT needs a filled grid (rows ≥ 2 and cols ≥ 2); this URA "
                         f"is {rows}×{cols}, a single line — use music2d, or a 1-D method")
    row_axis = np.asarray(g["row_axis"], dtype=float)
    col_axis = np.asarray(g["col_axis"], dtype=float)
    dr, dc = float(g["row_spacing_m"]), float(g["col_spacing_m"])
    x = np.asarray(x)
    R = (x @ x.conj().T) / x.shape[1]
    _s, _ns, vals = _doa.noise_subspace(R, 1)
    enum = _doa.enumerate_sources(vals, int(snapshots or x.shape[1]))
    k = max(1, min(int(sources) if sources else enum["mdl"], rows * cols - 1))
    Es = _doa.noise_subspace(R, k)[0]              # (n, k) signal subspace, element order

    idx = np.arange(rows * cols).reshape(rows, cols)      # element r*cols + c
    # row selection matrices (shift along rows), stacked over columns
    sel_r0 = idx[:-1, :].ravel()
    sel_r1 = idx[1:, :].ravel()
    sel_c0 = idx[:, :-1].ravel()
    sel_c1 = idx[:, 1:].ravel()
    Psi_r = np.linalg.pinv(Es[sel_r0]) @ Es[sel_r1]
    Psi_c = np.linalg.pinv(Es[sel_c0]) @ Es[sel_c1]
    # joint diagonalisation: eigvecs of Psi_r + j Psi_c pair the two sets
    w, V = np.linalg.eig(Psi_r + 1j * Psi_c)
    Vi = np.linalg.inv(V)
    mu_r = np.angle(np.diag(Vi @ Psi_r @ V))       # phase per row step
    mu_c = np.angle(np.diag(Vi @ Psi_c @ V))       # phase per column step
    lam = wavelength(f_hz)
    # mu = k * d * (u · axis); solve the 2×2 for the horizontal unit vector,
    # then (az, el). row/col axes are orthonormal in the array frame.
    kk = 2 * np.pi / lam
    B = np.array([row_axis, col_axis])             # (2, 3)
    az_list, el_list = [], []
    for i in range(k):
        proj = np.array([mu_r[i] / (kk * dr), mu_c[i] / (kk * dc)])   # u·row, u·col
        # u = proj_r*row + proj_c*col + w*normal; |u|=1 fixes |w|
        n_axis = np.cross(row_axis, col_axis)
        par = proj[0] * row_axis + proj[1] * col_axis
        perp2 = 1.0 - float(par @ par)
        w_perp = math.sqrt(perp2) if perp2 > 0 else 0.0
        u = par + w_perp * n_axis                   # the hemisphere the array faces
        u = u / max(np.linalg.norm(u), 1e-12)
        az = math.degrees(math.atan2(-u[1], u[0])) % 360.0
        el = math.degrees(math.asin(max(-1.0, min(1.0, u[2]))))
        az_list.append(az)
        el_list.append(el)
    snr = _doa._snr_from_eigs(vals, k)
    sig_az = math.hypot(_doa.cramer_rao_deg(snr, int(snapshots or x.shape[1]), geometry, f_hz,
                                            az_list[0]), _doa.MODEL_SIGMA_DEG["ideal"])
    sig_el = _elevation_sigma(geometry, f_hz, snr, int(snapshots or x.shape[1]),
                              az_list[0], el_list[0], True)
    caveats = ["2-D unitary ESPRIT on the ideal URA manifold, auto-paired (lit2 §2.3); "
               "for a measured manifold use music2d"]
    quality = {"architecture": "digital", "method": "esprit2d",
               "confidence": float(enum["confidence"]), "snr_db": round(10 * math.log10(snr), 1),
               "source_count_mdl": enum["mdl"]}
    return Doa2DResult(az_deg=az_list[0], el_deg=el_list[0], sigma_az_deg=sig_az,
                       sigma_el_deg=sig_el, estimator="esprit2d", manifold="ideal",
                       n_sources=k, all_az_deg=az_list, all_el_deg=el_list,
                       quality=quality, caveats=caveats)


def estimate2d(x, manifold: Manifold, f_hz: float, method: str = "music2d",
               sources: int | None = None, geometry: Geometry | None = None,
               **kw) -> Doa2DResult:
    geometry = geometry if geometry is not None else manifold.geometry
    if method == "esprit2d":
        return esprit2d(x, geometry, f_hz, sources=sources, snapshots=kw.get("snapshots"))
    if method == "music2d":
        return music2d(x, manifold, f_hz, sources=sources, geometry=geometry, **kw)
    raise ValueError(f"unknown 2-D method {method!r}; one of {METHODS_2D}")
