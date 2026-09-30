"""Direction of arrival: covariance, estimators, source count, quality, likelihood.

Takes element-ordered, calibrated samples (or their spatial covariance), a
manifold and the RF frequency; returns a DoaResult: bearing(s), the
pseudo-spectrum on the 0.5 deg grid, the 360-point likelihood a bearing
record carries, quality numbers, and a dossier Measurement. Angles are
RELATIVE bearings in the array frame (manifold: clockwise from the forward
axis).

Estimators: bartlett (robust, low resolution, the display default), capon
(MVDR, sharper, needs diagonal loading), music (needs the source count, which
mdl supplies), root (root-MUSIC; ULA polynomial or the manifold-separation
form on a circle/measured table), ml (deterministic ML by alternating
projection, for coherent multipath), fbss (MUSIC after spatial smoothing).

Forward-backward averaging needs a centro-symmetric array: a ULA is, a
5-element UCA is NOT, so covariance() asks the geometry and the circle is
smoothed in phase-mode space instead.

The record likelihood is a mixture of wrapped Gaussians (one per significant
Bartlett peak, each at the bearing's sigma) over a small never-zero floor, so
every mode survives and no single bearing can veto a posterior cell. The
pseudo-spectrum stays in the result for the polar display.
"""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass, field

import numpy as np
from scipy.special import jv

from .dossier import Measurement, unknown
from .manifold import (GRID_STEP_DEG, Geometry, IdealManifold, Manifold,
                       grid as make_grid, wavelength)

#: sigma added in quadrature per manifold kind: the error the data cannot see
#: (priors until `references` measures them, umbrella 8.8).
MODEL_SIGMA_DEG = {"ideal": 4.0, "perturbed": 4.0, "measured": 1.5}
LIKELIHOOD_FLOOR = 0.02
AMBIGUITY_DB = 3.0
METHODS = ("bartlett", "capon", "music", "root", "ml", "fbss")


@dataclass
class Covariance:
    R: np.ndarray
    snapshots: int
    f_hz: float
    bw_hz: float = 0.0
    fb: bool = False

    @property
    def n(self) -> int:
        return int(self.R.shape[0])


def exchange(n: int) -> np.ndarray:
    return np.eye(n)[::-1]


def forward_backward(R: np.ndarray) -> np.ndarray:
    J = exchange(R.shape[0])
    return 0.5 * (R + J @ R.conj() @ J)


def covariance(x, f_hz: float, geometry: Geometry | None = None,
               fb: bool | None = None, bw_hz: float = 0.0,
               snapshots: int | None = None) -> Covariance:
    """R = x xH / N from element-ordered samples (n, N). fb=None means
    'if the geometry is centro-symmetric'."""
    x = np.asarray(x, dtype=np.complex128)
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError("covariance needs (elements, samples)")
    R = (x @ x.conj().T) / x.shape[1]
    use_fb = bool(fb) if fb is not None else bool(
        geometry is not None and geometry.centrosymmetric)
    if use_fb:
        R = forward_backward(R)
    return Covariance(R=R, snapshots=int(snapshots or x.shape[1]),
                      f_hz=float(f_hz), bw_hz=float(bw_hz), fb=use_fb)


def band_snapshots(x, fs: float, f_lo: float, f_hi: float, nfft: int = 256,
                   overlap: float = 0.5, window: str = "hann", header=None) -> tuple:
    """Frequency-domain snapshots of the band [f_lo, f_hi] (baseband Hz):
    (Y (n, K), freqs, n_eff). Every STFT bin in the band, from every segment,
    is an interchangeable narrowband snapshot for an array this small. With
    `header`, each bin is calibrated with its own phase slope."""
    x = np.asarray(x)
    n, N = x.shape
    nfft = int(nfft)
    if N < nfft:
        nfft = max(8, 1 << int(math.floor(math.log2(max(8, N)))))
    hop = max(1, int(round(nfft * (1.0 - overlap))))
    w = np.hanning(nfft) if window == "hann" else np.ones(nfft)
    starts = np.arange(0, N - nfft + 1, hop)
    segs = np.stack([x[:, s:s + nfft] for s in starts], axis=1) * w
    X = np.fft.fft(segs, axis=2)
    freqs = np.fft.fftfreq(nfft, 1.0 / fs)
    sel = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(sel):
        sel = np.zeros(nfft, bool)
        sel[int(np.argmin(np.abs(freqs - 0.5 * (f_lo + f_hi))))] = True
    Xb = X[:, :, sel]
    if header is not None:
        from .calibration import band_factors
        Xb = Xb * band_factors(header, freqs[sel])[:, None, :]
    Y = Xb.reshape(n, -1)
    eff = len(starts) * (1.0 if overlap <= 0 else 0.7) * int(np.count_nonzero(sel))
    return Y, freqs[sel], max(1, int(eff))


#: Steering grids are cached PER MANIFOLD OBJECT, in a WeakKeyDictionary. The
#: cache must NOT be keyed on id(manifold): CPython reuses an object's id after
#: it is garbage-collected, so a new ULA manifold could inherit a dead UCA
#: manifold's id and be handed its 5-row steering table for a 7-element array (a
#: matmul size mismatch that appears only when GC happens to have run — it hid
#: on one machine and fired on another). A weak key holds the actual object, so
#: its id cannot be reused while the cache entry lives, and the entry is dropped
#: when the manifold is.
_STEER_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def steering_grid(manifold: Manifold, f_hz: float, step_deg: float = GRID_STEP_DEG) -> tuple:
    """(grid_deg, unit-norm steering vectors (n, K)); cached per manifold/freq."""
    key = (round(float(f_hz), 1), float(step_deg))
    per = _STEER_CACHE.get(manifold)
    if per is not None and key in per:
        return per[key]
    g = make_grid(step_deg)
    A = manifold.response(g, f_hz)
    A = A / np.linalg.norm(A, axis=0, keepdims=True)
    if per is None:
        per = {}
        try:
            _STEER_CACHE[manifold] = per
        except TypeError:
            return g, A                    # unhashable/unweakreffable: skip caching
    per[key] = (g, A)
    return g, A


def enumerate_sources(eigvals, n_snapshots: int, fb: bool = False) -> dict:
    """MDL and AIC on descending eigenvalues (Wax & Kailath 1985); FB halves
    the free-parameter count (Xu, Roy & Kailath). Returns both counts, the
    score vectors, and a confidence from MDL's margin."""
    lam = np.sort(np.maximum(np.asarray(eigvals, dtype=float), 1e-300))[::-1]
    n = lam.size
    N = max(2, int(n_snapshots))
    mdl, aic = [], []
    for k in range(n):
        tail = lam[k:]
        geo = float(np.exp(np.mean(np.log(tail))))
        ari = float(np.mean(tail))
        ll = -N * (n - k) * math.log(max(geo / ari, 1e-300))
        p = k * (2 * n - k + 1) / 2.0 if fb else k * (2 * n - k)
        mdl.append(ll + 0.5 * p * math.log(N))
        aic.append(2 * ll + 2 * p)
    mdl, aic = np.asarray(mdl), np.asarray(aic)
    s = np.sort(mdl)
    margin = float(s[1] - s[0]) if s.size > 1 else 0.0
    return {"mdl": int(np.argmin(mdl)), "aic": int(np.argmin(aic)),
            "mdl_scores": mdl.tolist(), "aic_scores": aic.tolist(),
            "confidence": float(1.0 - math.exp(-max(0.0, margin) / 4.0))}


def bartlett_spectrum(R, A) -> np.ndarray:
    return np.maximum(np.real(np.sum(A.conj() * (R @ A), axis=0)), 1e-300)


def capon_spectrum(R, A, loading: float = 1e-3) -> np.ndarray:
    n = R.shape[0]
    d = loading * float(np.real(np.trace(R))) / n
    Ri = np.linalg.inv(R + d * np.eye(n))
    return 1.0 / np.maximum(np.real(np.sum(A.conj() * (Ri @ A), axis=0)), 1e-300)


def music_spectrum(En, A) -> np.ndarray:
    p = En.conj().T @ A
    return 1.0 / np.maximum(np.real(np.sum(p.conj() * p, axis=0)), 1e-300)


def find_peaks(spec, grid_deg, k: int | None = None, within_db: float | None = None) -> list:
    """Local maxima of a CIRCULAR spectrum, strongest first, each refined by a
    parabola on the log spectrum. [(theta_deg, value, index)]."""
    s = np.asarray(spec, dtype=float)
    left, right = np.roll(s, 1), np.roll(s, -1)
    idx = np.nonzero((s > left) & (s >= right))[0]
    if idx.size == 0:
        idx = np.array([int(np.argmax(s))])
    idx = idx[np.argsort(s[idx])[::-1]]
    step = float(grid_deg[1] - grid_deg[0])
    out = []
    top = float(s[idx[0]])
    for i in idx:
        v = float(s[i])
        if within_db is not None and 10 * math.log10(top / v) > within_db:
            break
        a, b, c = (math.log(s[(i - 1) % s.size]), math.log(v), math.log(s[(i + 1) % s.size]))
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0
        off = max(-0.5, min(0.5, off))
        out.append(((float(grid_deg[i]) + off * step) % 360.0, v, int(i)))
        if k is not None and len(out) >= k:
            break
    return out


def _roots_from_coeffs(c) -> np.ndarray:
    r = np.roots(np.asarray(c)[::-1])
    return r[np.isfinite(r)]


def root_music(En, manifold: Manifold, f_hz: float, k: int) -> list:
    """Grid-free bearings from the noise subspace: [(theta_deg, |z|)]. ULA:
    the classic polynomial (both theta and 180-theta returned). Otherwise the
    manifold-separation form a(theta)=G[e^{jq theta}], whose roots nearest the
    unit circle are the bearings on a circle or a measured table alike."""
    C = En @ En.conj().T
    geom = getattr(manifold, "geometry", None)
    if isinstance(manifold, IdealManifold) and geom is not None and geom.kind == "ula":
        n = C.shape[0]
        coeffs = np.array([np.sum(np.diag(C, l)) for l in range(-(n - 1), n)])
        roots = _roots_from_coeffs(coeffs)
        roots = roots[np.abs(roots) <= 1.0 + 1e-9]
        roots = roots[np.argsort(np.abs(np.abs(roots) - 1.0))][:k]
        kd = 2 * np.pi / wavelength(f_hz) * geom.spacing_m
        out = []
        for z in roots:
            s = float(np.clip(-np.angle(z) / kd, -1.0, 1.0))
            th = math.degrees(math.asin(s)) % 360.0
            out.append((th, float(abs(z))))
            out.append(((180.0 - th) % 360.0, float(abs(z))))
        return out
    G, q = manifold.eadf(f_hz)
    M = G.conj().T @ C @ G
    Q = int(q.max())
    L = 2 * Q
    coeffs = np.zeros(2 * L + 1, complex)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            coeffs[(q[j] - q[i]) + L] += M[i, j]
    roots = _roots_from_coeffs(coeffs)
    roots = roots[(np.abs(roots) <= 1.0 + 1e-9) & (np.abs(roots) > 1e-6)]
    roots = roots[np.argsort(np.abs(np.abs(roots) - 1.0))]
    out, seen = [], []
    for z in roots:
        th = math.degrees(float(np.angle(z))) % 360.0
        if any(abs(((th - s) + 180) % 360 - 180) < 1.0 for s in seen):
            continue
        seen.append(th)
        out.append((th, float(abs(z))))
        if len(out) >= k:
            break
    return out


def ml_alternating(R, A, grid_deg, k: int, sweeps: int = 6) -> list:
    """Deterministic ML (Ziskind & Wax 1988): maximise tr(P_A(theta) R) over k
    bearings, one at a time. Coherent sources leave R rank-deficient, which
    starves the subspace methods; this criterion does not care. Returns
    [(theta_deg, grid index)]."""
    n, K = A.shape
    chosen: list = []

    def gain(sel):
        if not sel:
            return np.real(np.sum(A.conj() * (R @ A), axis=0))
        B = A[:, sel]
        Aperp = A - (B @ np.linalg.pinv(B)) @ A
        nrm = np.maximum(np.real(np.sum(Aperp.conj() * Aperp, axis=0)), 1e-12)
        return np.real(np.sum(Aperp.conj() * (R @ Aperp), axis=0)) / nrm

    for _ in range(k):
        g = gain(chosen)
        g[chosen] = -np.inf
        chosen.append(int(np.argmax(g)))
    for _ in range(sweeps):
        moved = False
        for i in range(k):
            others = chosen[:i] + chosen[i + 1:]
            g = gain(others)
            g[others] = -np.inf
            j = int(np.argmax(g))
            if j != chosen[i]:
                chosen[i], moved = j, True
        if not moved:
            break
    return [(float(grid_deg[j]), j) for j in chosen]


# ---------------------------------------------------------------------------
# phase-mode transform (UCA -> virtual ULA) and spatial smoothing
# ---------------------------------------------------------------------------

def phase_mode_transform(geometry: Geometry, f_hz: float, M: int | None = None) -> dict:
    """Davies transform for a UCA: T such that T @ x is a virtual ULA of modes
    -M..M (M < n/2; M=2 for five elements). Mode m is scaled by j^m J_m(kr)
    and ALIASED by mode m -/+ n, so the outer modes are contaminated -- the
    weight ratios are returned so a caller can state the bias rather than
    hide it. Valid for a UCA only."""
    if geometry.kind != "uca":
        raise ValueError("the phase-mode transform is for a UCA")
    n = geometry.n
    if M is None:
        M = (n - 1) // 2
    rho, phi = geometry.polar
    r = float(np.mean(rho))
    kr = 2 * np.pi / wavelength(f_hz) * r
    modes = np.arange(-M, M + 1)
    # F: element m0 -> mode; V: mode weights j^m J_m(kr)
    Fh = np.exp(-1j * np.outer(modes, phi)) / n
    Jm = (1j ** modes) * jv(modes, kr)
    with np.errstate(divide="ignore"):
        Vinv = np.where(np.abs(Jm) > 1e-9, 1.0 / Jm, 0.0)
    T = (Vinv[:, None] * Fh) * n
    alias = np.abs(jv(modes + n, kr) / np.where(np.abs(jv(modes, kr)) > 1e-12,
                                                jv(modes, kr), np.inf))
    return {"T": T, "modes": modes, "kr": float(kr), "alias_ratio": alias,
            "worst_alias": float(np.max(alias))}


def spatial_smooth(R: np.ndarray, sub: int) -> np.ndarray:
    """Forward-backward spatial smoothing over `sub`-element subarrays of a
    ULA (or a virtual ULA): decorrelates coherent sources by averaging the
    covariance of every overlapping subarray. Rank restored up to sub."""
    n = R.shape[0]
    L = n - sub + 1
    if L < 1:
        raise ValueError("subarray larger than the array")
    Rs = np.zeros((sub, sub), complex)
    J = exchange(sub)
    for i in range(L):
        blk = R[i:i + sub, i:i + sub]
        Rs += blk + J @ blk.conj() @ J
    return Rs / (2 * L)


def noise_subspace(R: np.ndarray, k: int) -> tuple:
    """(signal eigenvecs, noise eigenvecs, eigenvalues desc). Hermitian."""
    vals, vecs = np.linalg.eigh(R)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    return vecs[:, :k], vecs[:, k:], vals


# ---------------------------------------------------------------------------
# the likelihood a bearing record carries
# ---------------------------------------------------------------------------

def rotate_likelihood(rel_likelihood, heading_deg: float) -> np.ndarray:
    """A relative-bearing likelihood (index = deg clockwise from the array
    axis) -> absolute (index = deg clockwise from true north), by rolling it
    heading_deg. Both are 360-long, 1 deg."""
    a = np.asarray(rel_likelihood, dtype=float)
    return np.roll(a, int(round(heading_deg)) % a.size)


def cramer_rao_deg(snr_lin: float, snapshots: int, geometry: Geometry,
                   f_hz: float, theta_deg: float) -> float:
    """The stochastic CRB on one bearing, in degrees: the array's own
    statement of how well THIS data could place it. Uses the aperture's
    beam-derivative; a floor keeps a huge-SNR burst from claiming 0.001 deg
    the manifold cannot honour (the manifold sigma is added on top)."""
    n = geometry.n
    lam = wavelength(f_hz)
    p = geometry.positions
    u = np.array([math.cos(math.radians(theta_deg)), -math.sin(math.radians(theta_deg))])
    du = np.array([-math.sin(math.radians(theta_deg)), -math.cos(math.radians(theta_deg))])
    k = 2 * np.pi / lam
    d_phase = k * (p @ du)                       # d(phase)/d(theta_rad) per element
    d_phase -= d_phase.mean()
    fisher = 2.0 * max(1, snapshots) * max(snr_lin, 1e-6) * float(np.sum(d_phase ** 2))
    if fisher <= 0:
        return 90.0
    sig_rad = 1.0 / math.sqrt(fisher)
    return max(math.degrees(sig_rad), 0.05)


def _wrapped_gaussian(centre_deg: float, sigma_deg: float, grid360: np.ndarray) -> np.ndarray:
    d = (grid360 - centre_deg + 180.0) % 360.0 - 180.0
    s = max(float(sigma_deg), 0.5)
    return np.exp(-0.5 * (d / s) ** 2)


def build_likelihood(peaks, sigma_deg: float, floor: float = LIKELIHOOD_FLOOR) -> np.ndarray:
    """A 360-point (1 deg) likelihood over relative bearing: a wrapped Gaussian
    per peak (weighted by its linear power), plus a uniform floor. Normalised
    to a density (sum = 1). `peaks` is [(theta_deg, weight)]."""
    g = np.arange(360.0)
    dens = np.zeros(360)
    wsum = sum(max(w, 0.0) for _, w in peaks) or 1.0
    for th, w in peaks:
        dens += (max(w, 0.0) / wsum) * _wrapped_gaussian(th, sigma_deg, g)
    if dens.sum() <= 0:
        dens[:] = 1.0
    dens = dens / dens.sum()
    dens = (1.0 - floor) * dens + floor / 360.0
    return (dens / dens.sum()).astype(np.float64)


@dataclass
class DoaResult:
    theta_deg: float
    sigma_deg: float
    estimator: str
    manifold: str
    n_sources: int
    all_theta_deg: list = field(default_factory=list)
    grid_deg: np.ndarray | None = None
    spectrum: np.ndarray | None = None          # pseudo-spectrum on the grid (polar display)
    likelihood: np.ndarray | None = None        # 360-pt, relative bearing, 1 deg
    quality: dict = field(default_factory=dict)
    caveats: list = field(default_factory=list)

    def measurement(self) -> Measurement:
        m = Measurement(name="doa", value=float(self.theta_deg), unit="deg",
                        confidence=float(self.quality.get("confidence", 0.5)),
                        method=f"{self.estimator} ({self.manifold} manifold, "
                               f"{self.n_sources} source(s))",
                        evidence={k: v for k, v in self.quality.items()
                                  if k != "confidence"})
        for c in self.caveats:
            m.add_caveat(c)
        return m


def _snr_from_eigs(vals: np.ndarray, k: int) -> float:
    """Rough per-source SNR (linear) from the eigenvalues: signal power over
    the noise-eigenvalue floor."""
    vals = np.asarray(vals, dtype=float)
    noise = float(np.mean(vals[k:])) if k < vals.size else float(vals[-1])
    sig = float(np.mean(vals[:k])) - noise
    return max(sig / max(noise, 1e-12), 1e-3)


def estimate(x, manifold: Manifold, f_hz: float, method: str = "music",
             sources: int | None = None, geometry: Geometry | None = None,
             snapshots: int | None = None, step_deg: float = GRID_STEP_DEG,
             coherent: bool | None = None) -> DoaResult:
    """One bearing measurement from element-ordered calibrated samples x (n, N)
    (or a Covariance). `sources=None` lets MDL decide. `method` in METHODS;
    `coherent=True` forces the ML/smoothing path for multipath."""
    geometry = geometry if geometry is not None else getattr(manifold, "geometry", None)
    if geometry is None:
        raise ValueError("estimate needs a geometry (on the manifold or passed in)")
    if isinstance(x, Covariance):
        cov = x
    else:
        cov = covariance(x, f_hz, geometry=geometry, snapshots=snapshots)
    R = cov.R
    n = R.shape[0]
    grid_deg, A = steering_grid(manifold, f_hz, step_deg)
    _sig, _noise, vals = noise_subspace(R, 1)
    enum = enumerate_sources(vals, cov.snapshots, fb=cov.fb)
    # a subspace method can resolve at most n-1 sources; clamp an explicit
    # request too, or the noise subspace is empty and MUSIC returns a flat
    # (meaningless) spectrum instead of a bearing
    k = max(1, min(int(sources) if sources else enum["mdl"], n - 1))
    kind = getattr(manifold, "kind", "ideal")
    caveats = list(manifold.caveats(f_hz))
    En = noise_subspace(R, k)[1]

    if method == "bartlett":
        spec = bartlett_spectrum(R, A)
    elif method == "capon":
        spec = capon_spectrum(R, A)
    elif method in ("music", "fbss"):
        if method == "fbss" and geometry.kind == "uca":
            pm = phase_mode_transform(geometry, f_hz)
            Rv = pm["T"] @ R @ pm["T"].conj().T
            sub = Rv.shape[0] - 1                       # smoothing drops one mode
            Rv = spatial_smooth(Rv, sub)
            caveats.append(f"UCA phase-mode smoothing: outer modes are aliased "
                           f"{100 * pm['worst_alias']:.0f}% at this frequency "
                           f"(kr={pm['kr']:.2f}); coherent pairs separate but a "
                           f"bias is expected -- see tests/test_doa.py")
            Env = noise_subspace(Rv, min(k, sub - 1))[1]
            # After smoothing the virtual array is `sub` CONSECUTIVE modes, so
            # its steering is `sub` mode rows, not all 2M+1 (mixing 2M+1 rows
            # into a `sub`-row subspace is a shape mismatch, and the crash the
            # UCA-FBSS test now covers). The absolute mode offset is a per-angle
            # global phase MUSIC ignores.
            steer = np.exp(1j * np.outer(np.arange(sub), np.radians(grid_deg)))
            steer = steer / np.linalg.norm(steer, axis=0, keepdims=True)
            spec = music_spectrum(Env, steer)
        elif method == "fbss" and geometry.kind == "ula":
            Rs = spatial_smooth(R, max(2, n - 1))
            gsub, Asub = steering_grid(manifold, f_hz, step_deg)
            Asub = Asub[:Rs.shape[0]]
            Asub = Asub / np.linalg.norm(Asub, axis=0, keepdims=True)
            Ens = noise_subspace(Rs, min(k, Rs.shape[0] - 1))[1]
            spec = music_spectrum(Ens, Asub)
        else:
            spec = music_spectrum(En, A)
    elif method == "root":
        roots = root_music(En, manifold, f_hz, k)
        spec = bartlett_spectrum(R, A)
        if roots:
            th0 = roots[0][0]
            snr = _snr_from_eigs(vals, k)
            sig = math.hypot(cramer_rao_deg(snr, cov.snapshots, geometry, f_hz, th0),
                             MODEL_SIGMA_DEG.get(kind, 4.0))
            peaks = find_peaks(spec, grid_deg, within_db=AMBIGUITY_DB)
            lik = build_likelihood([(t, v) for t, v, _ in peaks], sig)
            return DoaResult(theta_deg=th0, sigma_deg=sig, estimator="root-music",
                             manifold=kind, n_sources=k,
                             all_theta_deg=[t for t, _ in roots], grid_deg=grid_deg,
                             spectrum=spec, likelihood=lik,
                             quality={"confidence": float(enum["confidence"]),
                                      "snr_db": round(10 * math.log10(snr), 1),
                                      "n_snapshots": cov.snapshots,
                                      "source_count_mdl": enum["mdl"],
                                      "root_radius": round(roots[0][1], 4)},
                             caveats=caveats)
    elif method == "ml":
        picks = ml_alternating(R, A, grid_deg, k)
        spec = bartlett_spectrum(R, A)
        th0 = picks[0][0] if picks else float(grid_deg[int(np.argmax(spec))])
        snr = _snr_from_eigs(vals, k)
        sig = math.hypot(cramer_rao_deg(snr, cov.snapshots, geometry, f_hz, th0),
                         MODEL_SIGMA_DEG.get(kind, 4.0))
        peaks = [(t, float(spec[j])) for t, j in picks]
        lik = build_likelihood(peaks or [(th0, 1.0)], sig)
        caveats.append("deterministic ML: chosen for coherent sources; the "
                       "bearing set is joint, not independent peaks")
        return DoaResult(theta_deg=th0, sigma_deg=sig, estimator="ml", manifold=kind,
                         n_sources=k, all_theta_deg=[t for t, _ in picks],
                         grid_deg=grid_deg, spectrum=spec, likelihood=lik,
                         quality={"confidence": float(enum["confidence"]),
                                  "snr_db": round(10 * math.log10(snr), 1),
                                  "n_snapshots": cov.snapshots,
                                  "source_count_mdl": enum["mdl"]},
                         caveats=caveats)
    else:
        raise ValueError(f"unknown estimator {method!r}; one of {METHODS}")

    peaks = find_peaks(spec, grid_deg, k=max(k, 1), within_db=None)
    amb = find_peaks(spec, grid_deg, within_db=AMBIGUITY_DB)
    th0 = peaks[0][0]
    snr = _snr_from_eigs(vals, k)
    sig = math.hypot(cramer_rao_deg(snr, cov.snapshots, geometry, f_hz, th0),
                     MODEL_SIGMA_DEG.get(kind, 4.0))
    lik = build_likelihood([(t, v) for t, v, _ in amb], sig)
    peak_db = 10 * math.log10(spec.max() / max(np.median(spec), 1e-300))
    eig_gap = float(vals[k - 1] / max(vals[k], 1e-300)) if k < vals.size else float("inf")
    return DoaResult(theta_deg=th0, sigma_deg=sig, estimator=method, manifold=kind,
                     n_sources=k, all_theta_deg=[t for t, _, _ in peaks[:k]],
                     grid_deg=grid_deg, spectrum=spec, likelihood=lik,
                     quality={"confidence": float(enum["confidence"]),
                              "peak_ratio_db": round(peak_db, 1),
                              "eigen_gap": round(eig_gap, 2),
                              "snr_db": round(10 * math.log10(snr), 1),
                              "n_snapshots": cov.snapshots,
                              "source_count_mdl": enum["mdl"],
                              "source_count_aic": enum["aic"]},
                     caveats=caveats)


# ---------------------------------------------------------------------------
# the wideband spatial spectrum (the DF waterfall)
# ---------------------------------------------------------------------------

def wideband_spectrum(x, fs: float, manifold: Manifold, f_center_hz: float,
                      nfft: int = 256, overlap: float = 0.5, method: str = "bartlett",
                      step_deg: float = 2.0, header=None) -> dict:
    """A (bins, angles) likelihood image over the whole band: for each FFT bin,
    a spatial spectrum across bearing. Two peaks in one bin is a multipath
    image or a co-channel emitter and shows as two (umbrella §6.6).

    x is element-ordered (n, N) calibrated samples. Each bin's RF frequency is
    f_center + baseband offset, so the steering uses the right wavelength per
    bin. Bartlett (robust, the display default) or MUSIC per bin; the 5×5
    per-bin covariance is cheap on CPU. Returns {freqs_hz, angles_deg, image,
    peak_bearing_per_bin}."""
    x = np.asarray(x)
    n, N = x.shape
    nfft = int(min(nfft, 1 << int(math.floor(math.log2(max(8, N))))))
    hop = max(1, int(round(nfft * (1.0 - overlap))))
    w = np.hanning(nfft)
    starts = np.arange(0, N - nfft + 1, hop)
    segs = np.stack([x[:, s:s + nfft] for s in starts], axis=1) * w      # (n, S, nfft)
    X = np.fft.fft(segs, axis=2)                                          # (n, S, nfft)
    freqs = np.fft.fftfreq(nfft, 1.0 / fs)
    order = np.argsort(freqs)
    freqs = freqs[order]
    X = X[:, :, order]
    if header is not None:
        from .calibration import band_factors
        X = X * band_factors(header, freqs)[:, None, :]
    grid = make_grid(step_deg)
    image = np.empty((nfft, grid.size), dtype=np.float32)
    peak = np.empty(nfft, dtype=np.float32)
    for b in range(nfft):
        Xi = X[:, :, b]                       # (n, S) snapshots at this bin
        R = (Xi @ Xi.conj().T) / Xi.shape[1]
        f_rf = f_center_hz + freqs[b]
        A = manifold.response(grid, f_rf)
        A = A / np.linalg.norm(A, axis=0, keepdims=True)
        if method == "music":
            spec = music_spectrum(noise_subspace(R, 1)[1], A)
        else:
            spec = bartlett_spectrum(R, A)
        image[b] = spec
        peak[b] = grid[int(np.argmax(spec))]
    return {"freqs_hz": freqs, "angles_deg": grid, "image": image,
            "peak_bearing_per_bin": peak}
