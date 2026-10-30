"""One estimator entry point, four array architectures (plan §11.4.2).

`doa.estimate` is the digital path: every element has its own coherent
receiver channel, and the subspace methods see the whole array. Three other
kinds of array need something before (or instead of) that:

  switched  N coherent channels + one reference channel + an RF switch over
            M > N elements (lit2 §2.14). Each slot measures the switched
            element AGAINST the reference element, which removes the
            receivers' own gain and phase; the slots together give one
            steering-vector estimate, matched to the manifold. Single-source,
            and the signal must last the whole sequence.

  analog    phase shifters combine the elements into one output (beam-scan)
            or two halves (monopulse). Beam-scan: steer through a set of
            beams, read the power in each, and match that power vector to the
            STORED beam patterns — the beams the applied (quantised) weights
            actually form — with a quadratic peak fit (lit2 §1.1, §1.3–§1.5).
            Monopulse: the phase between the two halves' outputs is
            k·D·(sin-like term) off the steer direction, one look gives the
            angle, and steering to it and looking again closes in (lit2 §1.8).
            Both single-source; expect a fraction of a beamwidth, not a
            fraction of a degree.

  hybrid    several analog subarrays, each into its own coherent channel. The
            channels' manifold is A_b(θ) = Wᴴa(θ) with the weights actually
            applied (`manifold.BeamspaceManifold`), and the digital
            estimators run on it unchanged (lit2 §1.9). Subarray phase
            centres far apart make grating lobes; candidates within 3 dB are
            settled by re-steering the subarrays to each and keeping the one
            that lights them up (lit2 §1.11). At most channels − 1 sources.

A multi-face mast whose faces do NOT share an LO is a digital array per face:
the face with the most power gets the bearing (`face_handoff_estimate`).

Everything returns a `doa.DoaResult`, so the engine, the records and the map
treat these bearings like any other; `quality["architecture"]` and the
caveats say how the bearing was made and what it cannot do.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace

import numpy as np

from . import doa as _doa
from .manifold import (GRID_STEP_DEG, BeamspaceManifold, Geometry, IdealManifold, Manifold,
                       direction3, grid as make_grid, wavelength)

log = logging.getLogger("atkdf.architectures")

SINGLE_SOURCE_CAVEAT = ("SINGLE-SOURCE: this array resolves one emitter per look; a second "
                        "co-channel signal or strong multipath biases the bearing instead of "
                        "appearing as a second one")


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------

def quantise_weights(W, phase_step_deg: float | None = None, gain_step_db: float | None = None,
                     gain_range_db: float = 31.5) -> np.ndarray:
    """What a phase shifter / attenuator chain can actually set: phase rounded
    to `phase_step_deg`, magnitude rounded to `gain_step_db` below the
    strongest weight and clipped to `gain_range_db` of attenuation. Zeros stay
    zero (an element not on that channel). The ADAR1000-class numbers are
    2.8125° and 0.5 dB (lit2 §1.7)."""
    W = np.asarray(W, dtype=complex)
    mag = np.abs(W)
    ph = np.angle(W)
    if phase_step_deg:
        st = math.radians(float(phase_step_deg))
        ph = np.round(ph / st) * st
    if gain_step_db:
        nz = mag > 0
        top = float(mag[nz].max()) if np.any(nz) else 1.0
        mdb = np.zeros_like(mag)
        mdb[nz] = 20.0 * np.log10(mag[nz] / top)
        mdb = np.clip(np.round(mdb / gain_step_db) * gain_step_db, -gain_range_db, 0.0)
        mag = np.where(nz, top * 10 ** (mdb / 20.0), 0.0)
    return mag * np.exp(1j * ph)


def steer_weights(manifold: Manifold, theta_deg: float, f_hz: float, members=None,
                  el_deg: float = 0.0) -> np.ndarray:
    """Phase-only weights (n_elements,) that point `members` (all elements
    when None) at θ: w = e^{j∠a(θ)} / √m, zero off the members. y = wᴴx then
    adds the members in phase for an emitter at θ."""
    if abs(el_deg) > 1e-12:
        a = manifold.response2(np.array([theta_deg]), np.array([el_deg]), f_hz)[:, 0]
    else:
        a = manifold.response(np.array([float(theta_deg)]), f_hz)[:, 0]
    n = a.size
    mem = np.arange(n) if members is None else np.asarray(list(members), dtype=int)
    w = np.zeros(n, dtype=complex)
    w[mem] = np.exp(1j * np.angle(a[mem])) / math.sqrt(len(mem))
    return w


def subarray_weights(manifold: Manifold, subarrays, theta_deg: float, f_hz: float,
                     el_deg: float = 0.0) -> np.ndarray:
    """(n_elements, n_subarrays): each subarray steered to θ on its own
    channel."""
    return np.stack([steer_weights(manifold, theta_deg, f_hz, mem, el_deg) for mem in subarrays],
                    axis=1)


def _members(subarrays) -> list:
    out = []
    for s in subarrays or []:
        m = s.get("members") if isinstance(s, dict) else s
        out.append([int(x) for x in m])
    return out


def _sigma_and_likelihood(theta: float, sigma: float, geometry: Geometry | None,
                          extra_peaks=()) -> np.ndarray:
    peaks = [(theta, 1.0)] + [(t, w) for t, w in extra_peaks]
    return _doa.fov_likelihood(_doa.build_likelihood(peaks, sigma), geometry)


def _circ_diff(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def _parabolic(c: np.ndarray, i: int) -> float:
    n = c.size
    a, b, d = c[(i - 1) % n], c[i], c[(i + 1) % n]
    den = a - 2 * b + d
    off = 0.5 * (a - d) / den if abs(den) > 1e-15 else 0.0
    return float(max(-0.5, min(0.5, off)))


# ---------------------------------------------------------------------------
# switched
# ---------------------------------------------------------------------------

def switched_estimate(slots, manifold: Manifold, f_hz: float, reference_element: int,
                      reference_channel: int = 0, channel_weights=None,
                      min_coherence: float = 0.2, step_deg: float = GRID_STEP_DEG,
                      geometry: Geometry | None = None) -> _doa.DoaResult:
    """Bearing from a switched sequence.

    `slots` is a list of (elements, Y): `elements[c]` is the element on
    channel c during the slot (the reference channel's entry is the
    reference element), Y is (channels, N). Each non-reference channel's
    element is measured as the complex ratio ⟨y_c, y_ref⟩ / ⟨y_ref, y_ref⟩ —
    its response RELATIVE to the reference element, so the emitter's own
    phase and amplitude, and any gain/phase common to the slot, cancel. With
    more than one switched channel, `channel_weights` (a receiver table,
    complex per channel) removes the channels' differences; with one switched
    channel its offset is common to every element and invisible.

    A slot whose coherence with the reference falls below `min_coherence`
    (the transmission ended, or never reached that element) is dropped and
    said so; elements never measured are left out of the match."""
    geometry = geometry if geometry is not None else manifold.geometry
    n_el = geometry.n
    sums = np.zeros(n_el, dtype=complex)
    count = np.zeros(n_el)
    coh_all, dropped, n_snap = [], [], []
    cw = None if channel_weights is None else np.asarray(channel_weights, dtype=complex)
    for i, (elements, Y) in enumerate(slots):
        Y = np.asarray(Y, dtype=complex)
        if cw is not None:
            Y = Y * cw[:, None]
        ref = Y[reference_channel]
        pref = float(np.real(ref @ ref.conj()))
        for c, e in enumerate(elements):
            if c == reference_channel:
                continue
            yc = Y[c]
            cross = yc @ ref.conj()
            pc = float(np.real(yc @ yc.conj()))
            coh = abs(cross) / math.sqrt(max(pref * pc, 1e-300))
            coh_all.append(coh)
            if coh < min_coherence:
                dropped.append((i, int(e), round(coh, 3)))
                continue
            sums[int(e)] += cross / max(pref, 1e-300)
            count[int(e)] += 1
            n_snap.append(Y.shape[1])
    sums[int(reference_element)] += 1.0
    count[int(reference_element)] += 1
    have = np.nonzero(count > 0)[0]
    caveats = [SINGLE_SOURCE_CAVEAT,
               "SWITCHED: one element at a time against the reference; the emitter must "
               "be steady (on, not moving much) across the whole sequence"]
    if dropped:
        caveats.append(f"{len(dropped)} slot measurement(s) dropped for low coherence "
                       f"with the reference (signal off or too weak): {dropped[:6]}")
    if have.size < 3:
        raise ValueError(f"switched DF needs at least 3 measured elements, got {have.size}")
    if have.size < n_el:
        caveats.append(f"elements {sorted(set(range(n_el)) - set(have.tolist()))} were "
                       f"never measured; the bearing uses the other {have.size}")
    v = sums[have] / count[have]
    g = make_grid(step_deg)
    A = manifold.response(g, f_hz)
    Aref = A[int(reference_element)]
    Mm = A[have] / np.where(np.abs(Aref) > 1e-12, Aref, 1e-12)[None, :]
    num = np.abs(Mm.conj().T @ v) ** 2
    den = np.sum(np.abs(Mm) ** 2, axis=0) * float(np.real(v @ v.conj()))
    c = num / np.maximum(den, 1e-300)
    fov = geometry.in_fov(g)
    cm = np.where(fov, c, 0.0)
    i = int(np.argmax(cm))
    theta = (float(g[i]) + _parabolic(cm, i) * step_deg) % 360.0
    rho = float(np.median(coh_all)) if coh_all else 0.0
    gamma = max(rho / max(1.0 - rho, 1e-6), 1e-3)            # per-element SNR from coherence
    N = int(np.median(n_snap)) if n_snap else 1
    sub_geom = geometry.subset(have.tolist()) if have.size < n_el else geometry
    crb = _doa.cramer_rao_deg(gamma, N, sub_geom, f_hz, theta)
    sigma = math.hypot(crb, _doa.model_sigma_deg(manifold, geometry, theta))
    quality = {"architecture": "switched", "confidence": float(min(1.0, max(0.0, rho))),
               "match": round(float(cm[i]), 4), "snr_db": round(10 * math.log10(gamma), 1),
               "n_slots": len(slots), "elements_used": int(have.size),
               "coherence_median": round(rho, 3), "n_snapshots": N}
    _doa._fov_note(geometry, theta, quality, caveats)
    caveats.extend(manifold.caveats(f_hz))
    log.debug("switched: θ=%.2f σ=%.2f match=%.3f coh=%.3f elements=%d dropped=%d",
              theta, sigma, cm[i], rho, have.size, len(dropped))
    return _doa.DoaResult(theta_deg=theta, sigma_deg=sigma, estimator="switched",
                          manifold=getattr(manifold, "kind", "ideal"), n_sources=1,
                          all_theta_deg=[theta], grid_deg=g, spectrum=np.maximum(c, 1e-300),
                          likelihood=_sigma_and_likelihood(theta, sigma, geometry),
                          quality=quality, caveats=caveats)


# ---------------------------------------------------------------------------
# analog: beam-scan
# ---------------------------------------------------------------------------

def _sum_weights(W) -> np.ndarray:
    W = np.asarray(W, dtype=complex)
    return W if W.ndim == 1 else W.sum(axis=1)


def beam_patterns(weights_list, manifold: Manifold, f_hz: float,
                  step_deg: float = GRID_STEP_DEG) -> tuple:
    """(grid, B): B[b, θ] = |w_bᴴ a(θ)|², the power each APPLIED beam
    collects from θ — the stored beam manifold beam-scan matches against."""
    g = make_grid(step_deg)
    A = manifold.response(g, f_hz)
    Ws = np.stack([_sum_weights(W) for W in weights_list], axis=0)      # (beams, n)
    return g, np.abs(Ws.conj() @ A) ** 2


def half_power_width_deg(pattern: np.ndarray, grid_deg: np.ndarray) -> float:
    """Width of the main lobe at half power around the pattern's maximum."""
    p = np.asarray(pattern, dtype=float)
    i = int(np.argmax(p))
    half = 0.5 * p[i]
    n = p.size
    step = float(grid_deg[1] - grid_deg[0])
    r = 0
    while r < n // 2 and p[(i + r + 1) % n] >= half:
        r += 1
    l = 0
    while l < n // 2 and p[(i - l - 1) % n] >= half:
        l += 1
    return float((r + l + 1) * step)


def beamscan_estimate(powers, weights_list, manifold: Manifold, f_hz: float,
                      noise_power: float | None = None, geometry: Geometry | None = None,
                      step_deg: float = GRID_STEP_DEG, sweep_s: float | None = None,
                      burst_s: float | None = None, refine: str = "match") -> _doa.DoaResult:
    """Bearing from a beam sweep: `powers[b]` is the mean power out of beam
    b, whose applied weights are `weights_list[b]` ((n,) or (n, outputs);
    two outputs are summed, as the sum beam).

    refine="match" (default): normalised correlation of the measured power
    vector with the stored beam patterns at every θ in the field of view,
    peak refined by a parabola (lit2 §1.3, the Gdańsk stored-pattern form).
    refine="quadratic": the strongest beam and its neighbours, a parabola in
    steer angle on log power (lit2 §1.1) — needs `beams_deg` order.

    `noise_power` is the per-element noise power (the receiver's measured
    floor); each beam's share (Σ|w|²·noise) is taken off before matching.
    Without it the match runs on raw power and says so."""
    geometry = geometry if geometry is not None else manifold.geometry
    p = np.asarray(powers, dtype=float)
    g, B = beam_patterns(weights_list, manifold, f_hz, step_deg)
    wn = np.array([float(np.sum(np.abs(_sum_weights(W)) ** 2)) for W in weights_list])
    caveats = [SINGLE_SOURCE_CAVEAT,
               "BEAM-SCAN: power only; expect a third to a half of a beamwidth at 15 dB, "
               "a full beamwidth near 0 dB (lit2 §1.1) — the cheap tier, not the P25 tier"]
    if noise_power is not None:
        ps = np.maximum(p - wn * float(noise_power), 0.0)
        noise_beam = wn * float(noise_power)
    else:
        ps = p.copy()
        noise_beam = None
        caveats.append("no noise floor given: matched on raw beam power (biased toward "
                       "broad beams at low SNR)")
    fov = geometry.in_fov(g)
    Bn = B / np.maximum(np.linalg.norm(B, axis=0, keepdims=True), 1e-300)
    pn = ps / max(float(np.linalg.norm(ps)), 1e-300)
    c = pn @ Bn
    cm = np.where(fov, c, -1.0)
    i = int(np.argmax(cm))
    theta = (float(g[i]) + _parabolic(cm, i) * step_deg) % 360.0
    best = int(np.argmax(ps))
    if refine == "quadratic" and 0 < best < p.size - 1:
        # the steer angles are where each beam's own pattern peaks
        steer = [float(g[int(np.argmax(B[b]))]) for b in range(B.shape[0])]
        lp = np.log(np.maximum(ps[best - 1:best + 2], 1e-300))
        den = lp[0] - 2 * lp[1] + lp[2]
        off = 0.5 * (lp[0] - lp[2]) / den if abs(den) > 1e-15 else 0.0
        off = max(-1.0, min(1.0, off))
        d_lo = _circ_diff(steer[best], steer[best - 1])
        d_hi = _circ_diff(steer[best + 1], steer[best])
        theta = (steer[best] + off * (d_hi if off > 0 else d_lo)) % 360.0
    bw = half_power_width_deg(B[best], g)
    if noise_beam is not None:
        snr = max(float(ps[best]) / max(float(noise_beam[best]), 1e-300), 1e-3)
    else:
        snr = max(float(p.max()) / max(float(np.median(p)), 1e-300), 1e-3)
    noise_sig = 0.4 * bw * math.sqrt(10 ** 1.5 / snr)
    noise_sig = min(max(noise_sig, 0.05 * bw), bw)
    sigma = math.hypot(noise_sig, _doa.model_sigma_deg(manifold, geometry, theta))
    if sweep_s is not None and burst_s is not None and sweep_s > burst_s:
        caveats.append(f"the sweep ({sweep_s * 1e3:.1f} ms) is longer than the burst "
                       f"({burst_s * 1e3:.1f} ms): beams after the burst saw nothing")
    quality = {"architecture": "analog", "method": f"beamscan-{refine}",
               "confidence": float(max(0.0, min(1.0, cm[i]))),
               "match": round(float(cm[i]), 4), "beamwidth_deg": round(bw, 2),
               "snr_db": round(10 * math.log10(snr), 1), "n_beams": int(p.size),
               "best_beam": best}
    if sweep_s is not None:
        quality["sweep_ms"] = round(sweep_s * 1e3, 3)
    _doa._fov_note(geometry, theta, quality, caveats)
    caveats.extend(manifold.caveats(f_hz))
    log.debug("beamscan: θ=%.2f σ=%.2f bw=%.1f snr=%.1f dB match=%.3f beams=%d",
              theta, sigma, bw, 10 * math.log10(snr), cm[i], p.size)
    return _doa.DoaResult(theta_deg=theta, sigma_deg=sigma, estimator="beamscan",
                          manifold=getattr(manifold, "kind", "ideal"), n_sources=1,
                          all_theta_deg=[theta], grid_deg=g,
                          spectrum=np.maximum(c - c.min() + 1e-6, 1e-300),
                          likelihood=_sigma_and_likelihood(theta, sigma, geometry),
                          quality=quality, caveats=caveats)


def beamscan_sweep(steerer, manifold: Manifold, beams_deg, f_hz: float, n: int = 512,
                   members=None, phase_step_deg: float | None = None,
                   gain_step_db: float | None = None) -> dict:
    """Steer through `beams_deg`, capture `n` samples each, and return the
    powers, the weights the steerer says it applied, and the time taken.
    Two outputs (halves) are both steered to the beam and summed."""
    powers, applied = [], []
    ch = None
    t = 0.0
    for b in beams_deg:
        w = steer_weights(manifold, float(b), f_hz, members)
        if ch is None:
            ch = int(getattr(steerer, "n_outputs", 1))
        W = np.repeat(w[:, None], ch, axis=1) if ch > 1 else w[:, None]
        if ch > 1 and getattr(steerer, "halves", None):
            W = np.zeros((w.size, ch), complex)
            for c, half in enumerate(steerer.halves):
                W[list(half), c] = w[list(half)]
        Wa = np.asarray(steerer.apply(quantise_weights(W, phase_step_deg, gain_step_db)))
        y = steerer.capture(n)
        s = y.sum(axis=0)
        powers.append(float(np.mean(np.abs(s) ** 2)))
        applied.append(Wa)
        t += float(steerer.settle_s) + n / float(steerer.fs_hz)
    return {"powers": np.asarray(powers), "weights": applied, "sweep_s": t,
            "beams_deg": list(map(float, beams_deg))}


# ---------------------------------------------------------------------------
# analog: phase-comparison monopulse
# ---------------------------------------------------------------------------

@dataclass
class MonopulseCal:
    """The measured monopulse curve at one frequency: Δφ = δ + κ·k·D·(s(θ) −
    s(θ_steer)). No paper gives a slope for a communications waveform and no
    beamformer has a channel-match spec (lit2 §1.7, §1.8), so κ and δ are
    measured on the known-good emitter, per frequency."""

    f_hz: float
    delta_rad: float = 0.0
    kappa: float = 1.0
    residual_deg: float | None = None
    n: int = 0

    def to_dict(self) -> dict:
        return {"f_hz": self.f_hz, "delta_rad": self.delta_rad, "kappa": self.kappa,
                "residual_deg": self.residual_deg, "n": self.n}

    @classmethod
    def from_dict(cls, d) -> "MonopulseCal":
        return cls(float(d["f_hz"]), float(d.get("delta_rad", 0.0)), float(d.get("kappa", 1.0)),
                   d.get("residual_deg"), int(d.get("n", 0)))


def halves_axis(geometry: Geometry, halves) -> tuple:
    """(e, D, cL, cR): the unit vector from the first half's phase centre to
    the second's, their distance, and the two centres (element means)."""
    P = geometry.electrical3
    cL = P[list(halves[0])].mean(axis=0)
    cR = P[list(halves[1])].mean(axis=0)
    d = cR - cL
    D = float(np.linalg.norm(d))
    if D < 1e-9:
        raise ValueError("the two halves share a phase centre: no monopulse baseline")
    return d / D, D, cL, cR


def _s_of(e: np.ndarray, theta_deg) -> np.ndarray:
    return direction3(theta_deg, 0.0) @ e


def _theta_from_s(e: np.ndarray, s: float, near_deg: float, geometry: Geometry) -> float | None:
    R = math.hypot(e[0], e[1])
    if R < 1e-9:
        raise ValueError("the halves are stacked vertically: that is an elevation monopulse")
    r = s / R
    if abs(r) > 1.0:
        return None
    alpha = math.atan2(e[1], e[0])
    a = math.acos(r)
    cands = [math.degrees(-alpha + a) % 360.0, math.degrees(-alpha - a) % 360.0]
    cands.sort(key=lambda t: (not bool(geometry.in_fov(np.array([t]))[0]),
                              abs(_circ_diff(t, near_deg))))
    return cands[0]


def monopulse_estimate(yL, yR, steer_deg: float, halves, geometry: Geometry, f_hz: float,
                       cal: MonopulseCal | None = None, manifold: Manifold | None = None
                       ) -> _doa.DoaResult:
    """One look: the two halves' outputs, both steered to `steer_deg`.
    Δφ = ∠Σ y_R y_L* gives s(θ) = s(θ_steer) + (Δφ − δ)/(κ k D), unambiguous
    while |Δφ − δ| < π — about ±λ/(2D) in s around the steer direction, so a
    coarse beam-scan (or the track loop) puts the beam on the emitter first.
    Also returns the Σ/Δ ratio the radar books use, for display."""
    yL = np.asarray(yL, dtype=complex)
    yR = np.asarray(yR, dtype=complex)
    e, D, _cL, _cR = halves_axis(geometry, halves)
    k = 2.0 * math.pi / wavelength(f_hz)
    cross = np.vdot(yL, yR)                      # Σ yR · conj(yL)
    dphi = float(np.angle(cross))
    delta = cal.delta_rad if cal is not None else 0.0
    kappa = cal.kappa if cal is not None else 1.0
    d_adj = ((dphi - delta + math.pi) % (2 * math.pi)) - math.pi
    s_hat = float(_s_of(e, steer_deg)[0]) + d_adj / (kappa * k * D)
    caveats = [SINGLE_SOURCE_CAVEAT,
               "MONOPULSE: phase comparison of two half-array outputs; the slope is "
               + ("measured (calibrated)" if cal is not None else
                  "THEORETICAL — uncalibrated, no beamformer has a channel-match spec")]
    theta = _theta_from_s(e, s_hat, steer_deg, geometry)
    if theta is None:
        raise ValueError("monopulse: the phase difference points outside visible space")
    pL = float(np.real(np.vdot(yL, yL)))
    pR = float(np.real(np.vdot(yR, yR)))
    rho = abs(cross) / math.sqrt(max(pL * pR, 1e-300))
    gamma = max(rho / max(1.0 - rho, 1e-6), 1e-3)
    N = yL.size
    sig_phi = math.sqrt(1.0 / (N * gamma) + 1.0 / (2.0 * N * gamma ** 2))
    R = math.hypot(e[0], e[1])
    alpha = math.atan2(e[1], e[0])
    ds = abs(R * math.sin(math.radians(theta) + alpha))
    sig_noise = math.degrees(sig_phi / (kappa * k * D * max(ds, 0.05)))
    # the model error: a measured slope earns the measured-manifold prior,
    # an uncalibrated one the manifold's own; both grow off boresight
    kind = "measured" if cal is not None else (_doa._kind_key(manifold) if manifold else "ideal")
    model = _doa.MODEL_SIGMA_DEG[kind] * _doa.scan_factor(geometry, theta)
    sigma = math.hypot(sig_noise, model)
    if abs(d_adj) > 0.8 * math.pi:
        caveats.append("the phase difference is near ±π: the emitter is at the edge of the "
                       "unambiguous span — re-steer toward it and look again")
    S = yL + yR
    Dd = yL - yR
    ratio = complex(np.vdot(S, Dd) / max(float(np.real(np.vdot(S, S))), 1e-300))
    quality = {"architecture": "analog", "method": "monopulse",
               "confidence": float(max(0.0, min(1.0, rho))),
               "dphi_deg": round(math.degrees(dphi), 3), "steer_deg": round(steer_deg, 3),
               "snr_db": round(10 * math.log10(gamma), 1), "n_snapshots": int(N),
               "baseline_m": round(D, 4), "delta_over_sigma": [round(ratio.real, 4),
                                                               round(ratio.imag, 4)],
               "calibrated": cal is not None}
    _doa._fov_note(geometry, theta, quality, caveats)
    if manifold is not None:
        caveats.extend(manifold.caveats(f_hz))
    log.debug("monopulse: steer=%.2f Δφ=%.2f° θ=%.3f σ=%.2f coh=%.3f", steer_deg,
              math.degrees(dphi), theta, sigma, rho)
    return _doa.DoaResult(theta_deg=theta, sigma_deg=sigma, estimator="monopulse",
                          manifold=getattr(manifold, "kind", "ideal") if manifold else "ideal",
                          n_sources=1, all_theta_deg=[theta],
                          likelihood=_sigma_and_likelihood(theta, sigma, geometry),
                          quality=quality, caveats=caveats)


def fit_monopulse(records, geometry: Geometry, halves, f_hz: float) -> MonopulseCal:
    """Fit δ and κ from known-angle looks: `records` = [(θ_true, θ_steer,
    Δφ_rad)], from the known-good emitter (a tripod turn or a drive past a
    surveyed site). Linear least squares on Δφ = δ + κ·k·D·(s(θ) − s(θ_s))."""
    e, D, _a, _b = halves_axis(geometry, halves)
    k = 2.0 * math.pi / wavelength(f_hz)
    rows, rhs = [], []
    for th, st, dphi in records:
        x = k * D * float(_s_of(e, th)[0] - _s_of(e, st)[0])
        rows.append([1.0, x])
        rhs.append(float(dphi))
    A = np.asarray(rows)
    b = np.asarray(rhs)
    if A.shape[0] < 2 or np.ptp(A[:, 1]) < 1e-9:
        raise ValueError("fitting a monopulse slope needs looks at ≥ 2 different offsets")
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    delta, kappa = float(sol[0]), float(sol[1])
    res = b - A @ sol
    resid_s = np.sqrt(np.mean(res ** 2)) / max(abs(kappa) * k * D, 1e-12)
    resid_deg = math.degrees(float(resid_s))
    log.info("monopulse cal at %.3f MHz: δ=%.2f° κ=%.4f residual≈%.3f° from %d looks",
             f_hz / 1e6, math.degrees(delta), kappa, resid_deg, len(records))
    return MonopulseCal(f_hz=float(f_hz), delta_rad=delta, kappa=kappa,
                        residual_deg=round(resid_deg, 4), n=len(records))


def monopulse_track(steerer, manifold: Manifold, halves, f_hz: float, theta0: float,
                    n: int = 1024, iterations: int = 4, tol_deg: float = 0.1,
                    cal: MonopulseCal | None = None, phase_step_deg: float | None = None,
                    gain_step_db: float | None = None) -> _doa.DoaResult:
    """Steer both halves to θ0, look, re-steer to the answer, look again —
    until it moves less than `tol_deg` (the classic tracking loop: one look
    per step, a burst-length budget rather than a full sweep)."""
    geometry = manifold.geometry
    theta = float(theta0)
    path = [theta]
    res = None
    t = 0.0
    for it in range(int(iterations)):
        w = steer_weights(manifold, theta, f_hz)
        W = np.zeros((w.size, 2), complex)
        W[list(halves[0]), 0] = w[list(halves[0])]
        W[list(halves[1]), 1] = w[list(halves[1])]
        steerer.apply(quantise_weights(W, phase_step_deg, gain_step_db))
        y = steerer.capture(n)
        t += float(steerer.settle_s) + n / float(steerer.fs_hz)
        res = monopulse_estimate(y[0], y[1], theta, halves, geometry, f_hz, cal, manifold)
        move = abs(_circ_diff(res.theta_deg, theta))
        theta = res.theta_deg
        path.append(theta)
        if move < tol_deg:
            break
    res.quality["track_path_deg"] = [round(x, 3) for x in path]
    res.quality["looks"] = len(path) - 1
    res.quality["track_ms"] = round(t * 1e3, 3)
    log.info("monopulse track: %s (%d looks, %.2f ms)", res.quality["track_path_deg"],
             res.quality["looks"], t * 1e3)
    return res


# ---------------------------------------------------------------------------
# hybrid
# ---------------------------------------------------------------------------

def hybrid_estimate(Y, W_applied, element_manifold: Manifold, f_hz: float,
                    method: str = "music", sources: int | None = None, subarrays=None,
                    steerer=None, n: int | None = None, resteer: bool = True,
                    phase_step_deg: float | None = None, gain_step_db: float | None = None
                    ) -> _doa.DoaResult:
    """Digital DoA on the subarray outputs Y (channels, N) through
    A_b = Wᴴa. Ambiguity (several peaks within 3 dB inside the FOV, more
    than the source count): with a `steerer`, re-steer every subarray to
    each candidate, keep the candidate that collects the most power, and
    estimate again on that look; without one, report the candidates."""
    W_applied = np.asarray(W_applied, dtype=complex)
    S = W_applied.shape[1]
    if S < 2:
        raise ValueError("hybrid DoA needs at least two subarray channels")
    k_max = S - 1
    k = None if sources is None else max(1, min(int(sources), k_max))
    bm = BeamspaceManifold(element_manifold, W_applied)
    res = _doa.estimate(Y, bm, f_hz, method=method, sources=k, geometry=bm.geometry)
    geom = bm.geometry
    caveats = list(res.caveats)
    if sources is not None and int(sources) > k_max:
        caveats.append(f"asked for {sources} sources; {S} channels resolve at most {k_max}")
    fov = geom.in_fov(res.grid_deg)
    spec = np.where(fov, res.spectrum, res.spectrum.min())
    amb = _doa.find_peaks(spec, res.grid_deg, within_db=_doa.AMBIGUITY_DB)
    cands = [float(t) for t, _v, _i in amb]
    resolved = "none-needed"
    if len(cands) > res.n_sources:
        resolved = "unresolved"
        if steerer is not None and resteer and subarrays is not None:
            mem = _members(subarrays)
            nn = int(n or (Y.shape[1] if hasattr(Y, "shape") else 1024))
            scores = []
            looks = {}
            for c in cands:
                Wc = subarray_weights(element_manifold, mem, c, f_hz)
                Wa = np.asarray(steerer.apply(quantise_weights(Wc, phase_step_deg, gain_step_db)))
                Yc = steerer.capture(nn)
                pw = float(np.mean(np.abs(Yc) ** 2) * Yc.shape[0])
                scores.append(pw)
                looks[c] = (Yc, Wa)
            best = cands[int(np.argmax(scores))]
            log.info("hybrid: %d candidates %s, re-steer powers %s → %.2f", len(cands),
                     [round(c, 1) for c in cands], [round(s, 2) for s in scores], best)
            Yb, Wb = looks[best]
            bm2 = BeamspaceManifold(element_manifold, Wb)
            res2 = _doa.estimate(Yb, bm2, f_hz, method=method, sources=k, geometry=bm2.geometry)
            # the re-steered look may still alias; keep its solution nearest the winner
            fov2 = bm2.geometry.in_fov(res2.grid_deg)
            spec2 = np.where(fov2, res2.spectrum, res2.spectrum.min())
            pk2 = _doa.find_peaks(spec2, res2.grid_deg, within_db=_doa.AMBIGUITY_DB)
            pick = min(pk2, key=lambda p: abs(_circ_diff(p[0], best)))[0] if pk2 else best
            res2.theta_deg = float(pick)
            res2.likelihood = _sigma_and_likelihood(pick, res2.sigma_deg, bm2.geometry)
            res2.caveats = list(res2.caveats)
            res = res2
            caveats = list(res2.caveats)
            resolved = "resteer"
            res.quality["resteer_power"] = [round(s, 3) for s in scores]
        else:
            caveats.append(f"GRATING-LOBE AMBIGUITY: {len(cands)} candidates within "
                           f"{_doa.AMBIGUITY_DB:.0f} dB {[round(c, 1) for c in cands]}; "
                           f"re-steer the subarrays to settle it")
    res.quality.update({"architecture": "hybrid", "channels": S,
                        "ambiguity_candidates": [round(c, 2) for c in cands],
                        "resolved_by": resolved})
    res.estimator = f"hybrid-{res.estimator}"
    res.caveats = caveats
    return res


# ---------------------------------------------------------------------------
# multi-face masts with independent faces
# ---------------------------------------------------------------------------

def face_handoff_estimate(x, manifold: Manifold, f_hz: float, method: str = "music",
                          sources: int | None = None, half_fov_deg: float = 60.0
                          ) -> _doa.DoaResult:
    """Element-ordered samples from a mast whose faces are NOT one coherent
    array: pick the face with the most power and direction-find on it alone,
    inside its own ±`half_fov_deg`."""
    geom = manifold.geometry
    if geom.groups is None:
        raise ValueError("face hand-off needs a geometry with groups (faces)")
    x = np.asarray(x)
    faces = sorted(set(geom.groups))
    power = {f: float(np.mean(np.abs(x[[i for i, g in enumerate(geom.groups) if g == f]]) ** 2))
             for f in faces}
    best = max(power, key=power.get)
    mem = [i for i, g in enumerate(geom.groups) if g == best]
    sub = manifold.subset(mem)
    b = float(np.mean(geom.boresight_deg[mem])) if geom.boresight_deg is not None else 0.0
    sub_geom = replace(sub.geometry, fov_az_deg=(b - half_fov_deg, b + half_fov_deg))
    if isinstance(sub, IdealManifold):
        sub = IdealManifold(sub_geom)
    res = _doa.estimate(x[mem], sub, f_hz, method=method, sources=sources, geometry=sub_geom)
    res.quality.update({"architecture": "digital", "face": int(best),
                        "face_power_db": {int(f): round(10 * math.log10(p + 1e-300), 1)
                                          for f, p in power.items()}})
    res.caveats = list(res.caveats) + [f"FACE HAND-OFF: face {best} (boresight {b:.0f}°) had "
                                       f"the most power; the faces are not one coherent array"]
    log.debug("face hand-off: face %d of %s, θ=%.2f", best, faces, res.theta_deg)
    return res


# ---------------------------------------------------------------------------
# the dispatcher
# ---------------------------------------------------------------------------

class ArrayDF:
    """A profile's DF, whatever its architecture.

        adf = ArrayDF(profile)                    # validates; builds geometry + manifold
        res = adf.estimate(data, f_hz)            # digital: (channels, N) receiver order
                                                  # switched: [(elements, Y), …]
        res = ArrayDF(profile, steerer=s).estimate(f_hz=f)   # analog / hybrid live

    For analog without a steerer, `data` is {"powers", "weights"} (a
    beam-scan already taken) or {"yL", "yR", "steer_deg"} (one monopulse
    look); for hybrid, {"Y", "W"}."""

    def __init__(self, profile, manifold: Manifold | None = None, steerer=None,
                 monopulse_cal: MonopulseCal | None = None):
        self.profile = profile
        self.warnings = profile.validate()
        self.geometry = profile.geometry()
        self.manifold = manifold or profile.manifold()
        self.steerer = steerer
        self.monopulse_cal = monopulse_cal
        st = profile.steering or {}
        self.phase_step = st.get("phase_step_deg")
        self.gain_step = st.get("gain_step_db")
        log.info("ArrayDF %s: %s, %d elements, %d channel(s)", profile.id,
                 profile.architecture, self.geometry.n, profile.channels())

    def beams(self) -> list:
        st = self.profile.steering or {}
        if st.get("beams_deg"):
            return [float(b) for b in st["beams_deg"]]
        g = self.geometry
        lo, hi = g.fov_az_deg or (self.profile.boresight_deg - 60.0, self.profile.boresight_deg + 60.0)
        # half a beamwidth apart: the broadside beam's width from the stored pattern
        f = self.profile.band_centre_hz
        _g, B = beam_patterns([steer_weights(self.manifold, self.profile.boresight_deg, f)],
                              self.manifold, f)
        step = max(1.0, 0.5 * half_power_width_deg(B[0], _g))
        return list(np.arange(lo, hi + 1e-9, step))

    def estimate(self, data=None, f_hz: float | None = None, method: str | None = None,
                 sources: int | None = None, **kw) -> _doa.DoaResult:
        f = float(f_hz if f_hz is not None else self.profile.band_centre_hz)
        lo, hi = self.profile.band_mhz
        arch = self.profile.architecture
        if not lo * 1e6 <= f <= hi * 1e6:
            log.warning("ArrayDF %s: %.3f MHz is outside the profile's band %.0f–%.0f MHz",
                        self.profile.id, f / 1e6, lo, hi)
        if arch == "digital":
            x = self.geometry.element_order(np.asarray(data))
            if self.geometry.kind == "multi_face" and self.profile.lo_sharing == "independent":
                res = face_handoff_estimate(x, self.manifold, f, method or "music", sources)
            elif kw.get("two_d") or method in ("music2d", "esprit2d"):
                from . import doa2d as _doa2d
                m2 = method if method in _doa2d.METHODS_2D else "music2d"
                res = _doa2d.estimate2d(x, self.manifold, f, method=m2, sources=sources,
                                        geometry=self.geometry).as_doaresult()
            else:
                res = _doa.estimate(x, self.manifold, f, method=method or "music",
                                    sources=sources, geometry=self.geometry)
                res.quality["architecture"] = "digital"
        elif arch == "switched":
            sw = self.profile.switching or {}
            res = switched_estimate(data, self.manifold, f, int(sw["reference_element"]),
                                    int(sw.get("reference_channel", 0)),
                                    channel_weights=kw.get("channel_weights"),
                                    geometry=self.geometry)
        elif arch == "analog":
            res = self._analog(data, f, method, **kw)
        elif arch == "hybrid":
            res = self._hybrid(data, f, method, sources, **kw)
        else:
            raise ValueError(f"unknown architecture {arch!r}")
        res.quality.setdefault("architecture", arch)
        res.quality["profile"] = self.profile.id
        res.quality["timing_tier"] = self.profile.timing.get("tier")
        return res

    def _analog(self, data, f, method, **kw):
        halves = (self.profile.steering or {}).get("halves")
        if self.steerer is not None:
            sweep = beamscan_sweep(self.steerer, self.manifold, kw.get("beams_deg") or self.beams(),
                                   f, n=int(kw.get("n", 512)), phase_step_deg=self.phase_step,
                                   gain_step_db=self.gain_step)
            coarse = beamscan_estimate(sweep["powers"], sweep["weights"], self.manifold, f,
                                       noise_power=kw.get("noise_power"), geometry=self.geometry,
                                       sweep_s=sweep["sweep_s"], burst_s=kw.get("burst_s"))
            if halves and method != "beamscan" and self.profile.channels() == 2:
                res = monopulse_track(self.steerer, self.manifold, halves, f, coarse.theta_deg,
                                      n=int(kw.get("n", 1024)), cal=self.monopulse_cal,
                                      phase_step_deg=self.phase_step, gain_step_db=self.gain_step)
                res.quality["coarse_beamscan_deg"] = round(coarse.theta_deg, 3)
                res.quality["sweep_ms"] = coarse.quality.get("sweep_ms")
                return res
            return coarse
        if data is None:
            raise ValueError("an analog array needs a steerer or a capture")
        if "powers" in data:
            return beamscan_estimate(data["powers"], data["weights"], self.manifold, f,
                                     noise_power=data.get("noise_power"), geometry=self.geometry)
        return monopulse_estimate(data["yL"], data["yR"], float(data["steer_deg"]), halves,
                                  self.geometry, f, self.monopulse_cal, self.manifold)

    def _hybrid(self, data, f, method, sources, **kw):
        subs = _members(self.profile.subarrays)
        if self.steerer is not None:
            th0 = float(kw.get("theta0", self.profile.boresight_deg))
            W0 = subarray_weights(self.manifold, subs, th0, f)
            Wa = np.asarray(self.steerer.apply(quantise_weights(W0, self.phase_step, self.gain_step)))
            Y = self.steerer.capture(int(kw.get("n", 1024)))
            return hybrid_estimate(Y, Wa, self.manifold, f, method or "music", sources, subs,
                                   steerer=self.steerer, n=int(kw.get("n", 1024)),
                                   phase_step_deg=self.phase_step, gain_step_db=self.gain_step)
        if data is None:
            raise ValueError("a hybrid array needs a steerer or {'Y', 'W'}")
        return hybrid_estimate(data["Y"], data["W"], self.manifold, f, method or "music", sources,
                               subs)
