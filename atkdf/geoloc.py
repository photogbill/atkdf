"""Layer 9 — TDOA / FDOA / direction finding. **Built, and NOT wired to radios.**

The arithmetic is here and it is tested against synthetic arrays with known
geometry, because that is possible today and costs nothing. What is NOT here
is any path from a real receiver into these functions, and that omission is
deliberate rather than unfinished: multi-receiver geolocation needs a shared
time and frequency reference, and `diversity.py` already documents what ATK's
free-running crystals actually provide, which is not that.

WHAT A FREE-RUNNING CRYSTAL COSTS YOU, in the units that matter:

  * TDOA resolves position by time difference, and light travels 300 m per
    microsecond. A 1 ppm oscillator drifts 1 µs per second — so 300 m of
    position error per second of unsynchronised operation, growing linearly.
    Without a common clock (GPSDO, a shared reference, or a cooperative
    beacon in view of every receiver) TDOA is a demonstration, not a fix.
  * FDOA resolves velocity by frequency difference, and needs the receivers'
    frequency references to agree far better than the Doppler being measured.
    A 1 ppm difference at 400 MHz is 400 Hz, which is larger than most of the
    Doppler shifts anybody wants to measure.
  * Array DF (MUSIC, ESPRIT, Watson-Watt) needs COHERENT sampling across
    elements — one clock, phase-locked — which is a different hardware
    problem again, and not one that separate USB dongles solve.

So these functions take arrays and geometry and return answers, every one of
them carries the reference requirement in its own docstring, and the UI does
not offer them. When the multi-SDR path is live (`receiver_manager.py`,
`diversity.py`, serial-pinned slots), wiring is a UI job and not a maths one —
which is exactly the state a deferred layer should be left in.

KEPT DISTINCT FROM POLARISATION DIVERSITY COMBINING, which `diversity.py`
does: that is two MATCHED receivers fused on one weak signal. This is
direction-finding ACROSS receivers. They share hardware and share nothing else.

MOVED FROM ATK, 2026-09-30 (ATK c52b913, `atk/core/siga/geoloc.py`), with its tests
and this docstring. ATK's `atk.core.siga.geoloc` is now an alias of this
module — the same module object, reached through `atk/core/df_engine.py`
— so the Geolocation tab and ATK's own tests run this code. Edit it here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _backend as be
from . import _prep as prep
from ._prep import SigCtx
from .dossier import DIMENSIONLESS, Measurement, unknown

#: Metres per second. Used to turn a time difference into a distance
#: difference, which is the only place a physical constant enters this file.
C = 299_792_458.0

REFERENCE_CAVEAT = (
    "This measurement assumes the receivers share a time and frequency "
    "reference. ATK's radios do not, by default: a 1 ppm oscillator drifts "
    "1 µs per second, which is 300 m of TDOA position error per second and "
    "grows linearly. Treat the number as correct arithmetic on possibly "
    "wrong inputs until a GPSDO or a common reference is in the chain.")


# ---------------------------------------------------------------------------
# TDOA
# ---------------------------------------------------------------------------

@dataclass
class TdoaResult:
    delay_s: float
    range_difference_m: float
    correlation_peak: float
    peak_to_floor_db: float
    lags_s: np.ndarray
    correlation: np.ndarray


def tdoa(ctx_a: SigCtx, ctx_b: SigCtx, max_lag_s: float | None = None,
         interpolate: bool = True) -> TdoaResult:
    """Time difference of arrival between two receivers, by cross-correlation.

    Sub-sample interpolation is on by default and matters more than it looks:
    at 2.4 Msps one sample is 417 ns, which is 125 m of range difference — so
    an integer-sample TDOA quantises the position solution into 125-metre
    steps whatever the SNR. Parabolic interpolation on the correlation peak
    typically recovers a tenth of that.
    """
    a = np.asarray(ctx_a.iq, dtype=np.complex128)
    b = np.asarray(ctx_b.iq, dtype=np.complex128)
    if a.size < 16 or b.size < 16:
        return TdoaResult(0.0, 0.0, 0.0, 0.0, np.zeros(0), np.zeros(0))
    fs = float(ctx_a.fs)
    n = min(a.size, b.size)
    a, b = a[:n] - np.mean(a[:n]), b[:n] - np.mean(b[:n])
    corr = be.correlate_full(a, b)
    lags = (np.arange(corr.size) - (b.size - 1)) / fs
    mag = np.abs(corr)
    if max_lag_s:
        mask = np.abs(lags) <= float(max_lag_s)
        mag = np.where(mask, mag, 0.0)
    k = int(np.argmax(mag))
    kf = prep.interpolate_peak(mag, k) if interpolate else float(k)
    delay = float((kf - (b.size - 1)) / fs)
    floor = float(np.median(mag)) or 1e-30
    ptf = float(20.0 * np.log10(mag[k] / floor))
    return TdoaResult(delay_s=delay, range_difference_m=delay * C,
                      correlation_peak=float(mag[k]), peak_to_floor_db=ptf,
                      lags_s=lags, correlation=mag)


def tdoa_measurement(ctx_a: SigCtx, ctx_b: SigCtx, **kw) -> Measurement:
    r = tdoa(ctx_a, ctx_b, **kw)
    if r.correlation.size == 0:
        return unknown("tdoa", "s", "cross-correlation", "empty input")
    # THE BAR IS THE SEARCH FLOOR, NOT A ROUND NUMBER. Cross-correlating two
    # INDEPENDENT noise records over 40,000 lags puts the tallest of those
    # lags around 13 dB above the median by chance alone, so the flat 10 dB
    # threshold that used to be here called two receivers hearing unrelated
    # noise a usable fix. The correction is the one the spectral estimators
    # use: the expected height of the largest of N draws.
    floor = float(20.0 * np.log10(max(2.0, np.log(max(2, r.correlation.size)))))
    m = Measurement(
        name="tdoa", value=r.delay_s, unit="s",
        confidence=float(min(0.9, max(0.0,
                                      (r.peak_to_floor_db - floor) / 24))),
        method="cross-correlation between receivers",
        evidence={"range_difference_m": round(r.range_difference_m, 2),
                  "peak_to_floor_db": round(r.peak_to_floor_db, 2),
                  "search_floor_db": round(floor, 2),
                  "lags_searched": int(r.correlation.size),
                  "sample_period_m": round(C / ctx_a.fs, 3)})
    m.add_caveat(REFERENCE_CAVEAT)
    if r.peak_to_floor_db < floor + 3.0:
        m.add_caveat(
            f"the correlation peak stands {r.peak_to_floor_db:.1f} dB over "
            f"its floor, against the {floor:.1f} dB that the tallest of "
            f"{r.correlation.size:,} lags reaches on unrelated noise alone — "
            f"the two receivers may not be hearing the same signal")
    return m


def fdoa(ctx_a: SigCtx, ctx_b: SigCtx, max_hz: float | None = None,
         points: int = 4096) -> Measurement:
    """Frequency difference of arrival — the Doppler difference, in Hz.

    Measured as the peak of the cross-ambiguity in frequency: multiply one
    signal by the conjugate of the other and find the line in the product. If
    both receivers hear the same emitter, the product's line is at the
    difference of their Doppler shifts — plus the difference of their
    oscillator errors, which is the whole problem.
    """
    a = np.asarray(ctx_a.iq, dtype=np.complex128)
    b = np.asarray(ctx_b.iq, dtype=np.complex128)
    n = min(a.size, b.size)
    if n < 256:
        return unknown("fdoa", "Hz", "cross-ambiguity", "records too short")
    prod = a[:n] * np.conj(b[:n])
    prod = prod - np.mean(prod)
    m = int(max(points, be.next_fast_len(n)))
    w = prep.window("blackmanharris", n)
    mag = np.abs(be.fftshift(be.fft(prod * w, m)))
    freqs = np.fft.fftshift(be.fftfreq(m, 1.0 / ctx_a.fs))
    if max_hz:
        mag = np.where(np.abs(freqs) <= float(max_hz), mag, 0.0)
    k = int(np.argmax(mag))
    lobe = max(2, int(6 * m / n))
    centroid, ptf, _fl = prep.peak_stats(mag, k, lobe)
    hz = float(prep.bin_to_hz(centroid, ctx_a.fs, m))
    out = Measurement(
        name="fdoa", value=hz, unit="Hz",
        confidence=float(min(0.9, max(0.0, (ptf - 8) / 20))),
        method="cross-ambiguity peak",
        evidence={"peak_over_local_floor_db": round(ptf, 2),
                  "resolution_hz": round(ctx_a.fs / n, 4)})
    out.add_caveat(REFERENCE_CAVEAT)
    out.add_caveat("an FDOA measurement is the difference of two Doppler "
                   "shifts PLUS the difference of two oscillator errors, and "
                   "nothing in the measurement separates them. At 400 MHz a "
                   "1 ppm reference difference is 400 Hz — larger than most "
                   "Doppler worth measuring.")
    return out


def rssi_heatmap(points: list, decay_exponent: float = 2.0) -> dict:
    """Power-difference DF — the poor-man's fix, GPS-stamped.

    `points` = [{lat, lon, power_db}, …] collected while moving. The emitter
    is estimated as the power-weighted centroid, with a spread that says how
    little that means. It is here because it needs no shared reference at all
    and works with one receiver and a car, which makes it the only member of
    this module usable today.

    **It is a centroid, not a fix.** Terrain, multipath and antenna pattern
    all bias it, and it cannot distinguish a distant strong emitter from a
    near weak one. The returned `spread_m` is the honest error bar.
    """
    if not points:
        return {}
    lat = np.array([p["lat"] for p in points], dtype=float)
    lon = np.array([p["lon"] for p in points], dtype=float)
    pdb = np.array([p["power_db"] for p in points], dtype=float)
    w = np.power(10.0, (pdb - pdb.max()) / (10.0 * float(decay_exponent)))
    tot = float(w.sum()) or 1e-30
    clat, clon = float(np.dot(w, lat) / tot), float(np.dot(w, lon) / tot)
    # metres per degree, locally flat
    m_lat = 111_320.0
    m_lon = 111_320.0 * np.cos(np.radians(clat))
    d = np.hypot((lat - clat) * m_lat, (lon - clon) * m_lon)
    spread = float(np.sqrt(np.dot(w, d ** 2) / tot))
    return {"lat": clat, "lon": clon, "spread_m": spread,
            "samples": len(points),
            "strongest": {"lat": float(lat[int(np.argmax(pdb))]),
                          "lon": float(lon[int(np.argmax(pdb))]),
                          "power_db": float(pdb.max())},
            "note": ("power-weighted centroid, NOT a fix: terrain, multipath "
                     "and antenna pattern all bias it, and it cannot tell a "
                     "distant strong emitter from a near weak one. The "
                     "spread is the error bar, and it is usually large.")}


# ---------------------------------------------------------------------------
# array direction finding
# ---------------------------------------------------------------------------

def steering_vector(geometry: np.ndarray, theta_rad: float,
                    wavelength_m: float) -> np.ndarray:
    """Array response to a plane wave from `theta`. Geometry in metres,
    shape (elements, 2) as (x, y)."""
    g = np.asarray(geometry, dtype=float)
    k = 2 * np.pi / float(wavelength_m)
    direction = np.array([np.cos(theta_rad), np.sin(theta_rad)])
    return np.exp(1j * k * (g @ direction))


def music(array_iq: np.ndarray, geometry, wavelength_m: float,
          sources: int = 1, grid: int = 721) -> Measurement:
    """MUSIC direction of arrival. `array_iq` is (elements, samples).

    Eigen-decompose the spatial covariance, take the eigenvectors BELOW the
    `sources` largest as the noise subspace, and sweep the steering vector:
    where it is orthogonal to the noise subspace, the pseudo-spectrum spikes.
    Super-resolution — two sources closer than a beamwidth are separable — at
    the price of needing the source COUNT in advance, which the caller rarely
    has.

    **Requires coherent sampling across elements.** Separate dongles sampling
    the same signal are not coherent; their phases wander independently, the
    covariance is not the array covariance, and the peak means nothing.
    """
    a = np.asarray(array_iq, dtype=np.complex128)
    if a.ndim != 2 or a.shape[0] < 2 or a.shape[1] < 16:
        return unknown("doa", "deg", "MUSIC",
                       "need at least 2 elements and 16 samples")
    m_el = a.shape[0]
    k = int(max(1, min(sources, m_el - 1)))
    r = (a @ a.conj().T) / a.shape[1]
    vals, vecs = np.linalg.eigh(r)
    noise = vecs[:, : m_el - k]                # ascending eigenvalues
    thetas = np.linspace(-np.pi, np.pi, int(grid))
    spec = np.empty(thetas.size)
    for i, t in enumerate(thetas):
        s = steering_vector(geometry, float(t), wavelength_m)
        proj = noise.conj().T @ s
        spec[i] = 1.0 / (float(np.vdot(proj, proj).real) + 1e-30)
    j = int(np.argmax(spec))
    db = prep.db10(spec)
    lobe = max(2, grid // 64)
    _c, ptf, _f = prep.peak_stats(np.sqrt(spec), j, lobe)
    out = Measurement(
        name="doa", value=float(np.degrees(thetas[j])), unit="deg",
        confidence=float(min(0.9, max(0.0, ptf / 30.0))),
        method=f"MUSIC ({m_el} elements, {k} source(s))",
        evidence={"peak_over_local_floor_db": round(ptf, 2),
                  "eigenvalues": [round(float(v), 6) for v in vals[::-1]],
                  "grid_points": int(grid),
                  "spectrum_db_range": [round(float(db.min()), 2),
                                        round(float(db.max()), 2)]})
    out.add_caveat("MUSIC needs the NUMBER OF SOURCES in advance. Give it too "
                   "few and a real source joins the noise subspace and "
                   "disappears; too many and noise eigenvectors become "
                   "'sources'. Compare the eigenvalues — a clear gap is where "
                   "the true count is.")
    out.add_caveat("A LINEAR array measures only the angle to its own AXIS: "
                   "+θ and −θ give identical steering vectors, so the "
                   "pseudo-spectrum is symmetric and the true solution is a "
                   "CONE about the array axis. Read the magnitude of the "
                   "angle; resolving its sign needs a non-collinear baseline.")
    out.add_caveat("requires coherent sampling across elements: one clock, "
                   "phase-locked. Independent receivers do not qualify.")
    return out


def esprit(array_iq: np.ndarray, spacing_m: float, wavelength_m: float,
           sources: int = 1) -> Measurement:
    """ESPRIT direction of arrival for a uniform linear array.

    No grid search at all: the rotational invariance between two overlapping
    sub-arrays turns the angle into an eigenvalue, so the answer is exact
    rather than the finest point of a sweep. Faster and more accurate than
    MUSIC, at the cost of requiring a UNIFORM array — the invariance is a
    property of the geometry, not of the algorithm.
    """
    a = np.asarray(array_iq, dtype=np.complex128)
    if a.ndim != 2 or a.shape[0] < 3:
        return unknown("doa", "deg", "ESPRIT",
                       "need at least 3 uniformly spaced elements")
    m_el = a.shape[0]
    k = int(max(1, min(sources, m_el - 2)))
    r = (a @ a.conj().T) / a.shape[1]
    _vals, vecs = np.linalg.eigh(r)
    sig = vecs[:, -k:]
    s1, s2 = sig[:-1, :], sig[1:, :]
    psi = np.linalg.pinv(s1) @ s2
    ev = np.linalg.eigvals(psi)
    ang = np.angle(ev)
    # The array response built by `steering_vector` for a ULA along x is
    # exp(j·k·n·d·COS θ) — the projection of the direction onto the element
    # axis. So the eigenvalue phase inverts with arccos, not arcsin. Using
    # arcsin returns 65° for a true 25° source (arcsin of cos 25°), which is
    # a plausible-looking bearing and completely wrong.
    cos_t = np.clip(ang * float(wavelength_m)
                    / (2 * np.pi * float(spacing_m)), -1.0, 1.0)
    thetas = np.degrees(np.arccos(cos_t))
    out = Measurement(
        name="doa", value=float(thetas[0]), unit="deg", confidence=0.6,
        method=f"ESPRIT ({m_el}-element ULA, {k} source(s))",
        evidence={"all_angles_deg": [round(float(t), 3) for t in thetas],
                  "spacing_m": float(spacing_m),
                  "spacing_wavelengths": round(float(spacing_m)
                                               / float(wavelength_m), 4)})
    if float(spacing_m) > float(wavelength_m) / 2:
        out.add_caveat(f"element spacing is "
                       f"{float(spacing_m) / float(wavelength_m):.2f} "
                       f"wavelengths — beyond half a wavelength the mapping "
                       f"from phase to angle is ambiguous and this answer has "
                       f"aliases")
    out.add_caveat("A LINEAR array measures only the angle to its own AXIS, "
                   "so +θ and −θ produce identical phases and are not "
                   "separable — the true solution is a CONE about the array "
                   "axis, not a bearing. Resolving it needs a second, "
                   "non-collinear baseline.")
    out.add_caveat("ESPRIT assumes a UNIFORM linear array; on any other "
                   "geometry the rotational invariance it exploits does not "
                   "exist and the answer is meaningless rather than "
                   "approximate.")
    out.add_caveat("requires coherent sampling across elements.")
    return out


def watson_watt(sin_ch: np.ndarray, cos_ch: np.ndarray,
                omni_ch: np.ndarray | None = None) -> Measurement:
    """Watson-Watt DF from crossed loops (+ optional sense antenna).

    The classic: two orthogonal loops give amplitudes proportional to sin θ
    and cos θ, and the arctangent is the bearing. Cheap, wideband,
    instantaneous — and ambiguous by 180° without the omnidirectional sense
    channel, which is what resolves which end of the line the signal came
    from.
    """
    s = np.asarray(sin_ch, dtype=np.complex128)
    c = np.asarray(cos_ch, dtype=np.complex128)
    n = min(s.size, c.size)
    if n < 16:
        return unknown("doa", "deg", "Watson-Watt", "channels too short")
    num = float(np.real(np.vdot(c[:n], s[:n])))
    den = float(np.real(np.vdot(c[:n], c[:n])))
    bearing = float(np.degrees(np.arctan2(num, den)))
    ev = {"sin_power": float(np.mean(np.abs(s[:n]) ** 2)),
          "cos_power": float(np.mean(np.abs(c[:n]) ** 2))}
    resolved = False
    if omni_ch is not None and np.size(omni_ch) >= n:
        o = np.asarray(omni_ch, dtype=np.complex128)[:n]
        ref = s[:n] * np.sin(np.radians(bearing)) \
            + c[:n] * np.cos(np.radians(bearing))
        if float(np.real(np.vdot(o, ref))) < 0:
            bearing = (bearing + 180.0) % 360.0
        resolved = True
        ev["sense_channel"] = "used"
    m = Measurement(name="doa", value=bearing % 360.0, unit="deg",
                    confidence=0.5 if resolved else 0.35,
                    method="Watson-Watt crossed loops"
                           + (" + sense" if resolved else ""),
                    evidence=ev)
    if not resolved:
        m.add_caveat("AMBIGUOUS BY 180°: without an omnidirectional sense "
                     "channel, crossed loops cannot tell the front of the "
                     "bearing line from the back. Both ends are equally "
                     "consistent with this measurement.")
    return m


def beamform(array_iq: np.ndarray, geometry, wavelength_m: float,
             theta_rad: float) -> np.ndarray:
    """Delay-and-sum beamforming toward `theta` — gain there, nulls elsewhere.

    The other use of an array: not finding a signal, but hearing one better,
    or nulling an interferer by steering toward the wanted signal and letting
    the array pattern do the rest.
    """
    a = np.asarray(array_iq, dtype=np.complex128)
    w = steering_vector(geometry, float(theta_rad), float(wavelength_m))
    return (w.conj() @ a) / max(1, w.size)


def synth_array(theta_deg: float, elements: int = 8, spacing_m: float = 0.5,
                wavelength_m: float = 1.0, samples: int = 2048,
                snr_db: float = 20.0, seed: int = 0) -> tuple:
    """A synthetic uniform linear array observing one source. `(iq, geometry)`.

    The test fixture, and the reason this module is testable at all without
    hardware: a known bearing in, and MUSIC/ESPRIT must return it.
    """
    rng = np.random.default_rng(int(seed))
    g = np.stack([np.arange(elements) * float(spacing_m),
                  np.zeros(elements)], axis=1)
    s = (rng.standard_normal(samples)
         + 1j * rng.standard_normal(samples)) / np.sqrt(2)
    a = steering_vector(g, np.radians(float(theta_deg)), wavelength_m)
    x = np.outer(a, s)
    p = float(np.mean(np.abs(x) ** 2))
    pn = p / (10.0 ** (float(snr_db) / 10.0))
    x = x + np.sqrt(pn / 2) * (rng.standard_normal(x.shape)
                               + 1j * rng.standard_normal(x.shape))
    return x, g
