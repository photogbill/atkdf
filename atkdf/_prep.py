"""The unit and peak helpers the moved `geoloc` needs, and no more.

Copied VERBATIM from ATK's `siga/prep.py` (2026-09-30, ATK c52b913) when
`geoloc` moved here — the rest of `prep` (the SigCtx preparation chain) is
the signals bench's and stays in ATK. The docstrings came too, because they
are why the functions are shaped the way they are.

`SigCtx` is the one thing `geoloc` reads from ATK's signal context, stated
as what it is: anything with `.iq` and `.fs`. ATK's `SigCtx` qualifies, and
so does `Samples` below, which is what the engine and the tests build.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class SigCtx(Protocol):
    """What `geoloc.tdoa` / `fdoa` read from a signal: the samples and the
    rate. ATK's `siga.context.SigCtx` satisfies this without knowing it."""

    iq: np.ndarray
    fs: float


@dataclass
class Samples:
    """The smallest thing that satisfies `SigCtx`: samples, rate, a label."""

    iq: np.ndarray
    fs: float
    source: str = ""


def window(kind: str, n: int) -> np.ndarray:
    """One of hann, hamming, blackmanharris, flattop, rectangular; length
    `n`, as float64."""
    n = int(n)
    k = (kind or "hann").lower()
    if k == "hann":
        return np.hanning(n)
    if k == "hamming":
        return np.hamming(n)
    if k == "rectangular":
        return np.ones(n)
    idx = np.arange(n)
    if k == "blackmanharris":
        a = (0.35875, 0.48829, 0.14128, 0.01168)
    elif k == "flattop":
        a = (0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368)
    else:
        return np.hanning(n)
    w = np.zeros(n)
    for i, ai in enumerate(a):
        w += ((-1) ** i) * ai * np.cos(2 * np.pi * i * idx / max(1, n - 1))
    return w


def bin_to_hz(bins, fs: float, nfft: int, shifted: bool = True):
    """FFT bin index → Hz. `shifted=True` means bin 0 is −fs/2 (the layout
    every display in ATK uses); `False` means numpy's raw ordering."""
    b = np.asarray(bins, dtype=float)
    if shifted:
        return (b - nfft / 2.0) * (fs / nfft)
    return np.where(b < nfft / 2.0, b, b - nfft) * (fs / nfft)


def db10(p) -> np.ndarray:
    return 10.0 * np.log10(np.abs(np.asarray(p)) + 1e-30)


def peak_stats(mag: np.ndarray, k: int, lobe: int,
               span: int | None = None) -> tuple:
    """`(centroid_bin, peak_over_LOCAL_floor_db, local_floor_power)`.

    Two corrections to the naive "tallest bin over the median of everything",
    each one a false positive it was caught producing:

    **The floor must be LOCAL.** A hard-limited 8PSK signal occupies ±1.6 kHz
    of a ±24 kHz span, so the global median is the empty part of the receiver
    and the signal's own passband measures 37 dB above it. That is not a
    line, it is a signal — and read as a line it says "residual carrier" about
    a suppressed-carrier PSK. The floor here is the median power in a band
    around the peak, excluding the peak's own main lobe, which asks the
    question that was meant: *how far does this stand above its own
    neighbourhood.*

    **The centroid must have the pedestal removed.** A line sitting on the
    shoulder of a continuum has its main-lobe centroid dragged toward the
    shoulder — measured at 3.5 Hz of pull on a 64-QAM fourth-power line,
    which is enough phase drift across a 0.3 s record to collapse the
    coherence test. Subtracting the local floor before taking the centroid
    removes the pull.
    """
    p = np.asarray(mag, dtype=float) ** 2
    n = p.size
    k = int(k)
    lobe = max(1, int(lobe))
    span = int(span) if span else max(4 * lobe, n // 32)
    lo, hi = max(0, k - span), min(n, k + span + 1)
    keep = np.ones(hi - lo, dtype=bool)
    a = max(0, max(0, k - lobe) - lo)
    b = max(0, min(n, k + lobe + 1) - lo)
    keep[a:b] = False
    floor = float(np.median(p[lo:hi][keep])) if keep.any() \
        else float(np.median(p))
    ptf = 10.0 * np.log10((p[k] + 1e-30) / (floor + 1e-30))
    l0, l1 = max(0, k - lobe), min(n, k + lobe + 1)
    seg = np.maximum(p[l0:l1] - floor, 0.0)
    total = float(seg.sum())
    centroid = (float(np.dot(seg, np.arange(l0, l1)) / total) if total > 0
                else float(k))
    return centroid, float(ptf), floor


def interpolate_peak(mag: np.ndarray, k: int) -> float:
    """Sub-bin peak position by parabolic interpolation on log magnitude.

    Worth the six lines: without it every frequency the bench reports is
    quantised to a bin, and the M-th-power estimate is then divided by M,
    which divides the error by M as well but never removes the bias.
    """
    k = int(k)
    if k <= 0 or k >= mag.size - 1:
        return float(k)
    a, b, c = (np.log(float(mag[k - 1]) + 1e-30),
               np.log(float(mag[k]) + 1e-30),
               np.log(float(mag[k + 1]) + 1e-30))
    den = a - 2 * b + c
    if abs(den) < 1e-20:
        return float(k)
    return float(k) + 0.5 * (a - c) / den
