"""The FFT and correlation helpers the moved `geoloc` needs, and no more.

Copied VERBATIM from ATK's `siga/backend.py` (2026-09-30, ATK c52b913) when
`geoloc` moved here, so a moved function computes exactly what it computed
in ATK. ATK's backend also routes large transforms to CuPy when asked; the
engine does not need that (its own batched work is in `doa`), so these are
the numpy paths, with scipy's `next_fast_len` when scipy is present, as in
ATK. The result is the same number either way; only where it ran differs.
"""

from __future__ import annotations

import numpy as np

try:                                                      # pragma: no cover
    import scipy.fft as _sfft                             # noqa: F401
    HAVE_SCIPY = True
except Exception:                                         # noqa: BLE001
    _sfft = None
    HAVE_SCIPY = False


def fft(x, n: int | None = None, axis: int = -1) -> np.ndarray:
    return np.fft.fft(np.asarray(x), n=n, axis=axis)


def ifft(x, n: int | None = None, axis: int = -1) -> np.ndarray:
    return np.fft.ifft(np.asarray(x), n=n, axis=axis)


def fftshift(x, axes=None) -> np.ndarray:
    return np.fft.fftshift(np.asarray(x), axes=axes)


def fftfreq(n: int, d: float) -> np.ndarray:
    return np.fft.fftfreq(int(n), d=float(d))


def next_fast_len(n: int) -> int:
    """A length the FFT likes (5-smooth). scipy has a real one; the fallback
    walks up to the next power of two, which is never wrong, only bigger."""
    n = max(1, int(n))
    if HAVE_SCIPY:                                        # pragma: no cover
        return int(_sfft.next_fast_len(n))
    return 1 << (n - 1).bit_length()


def correlate_full(a, b) -> np.ndarray:
    """Full cross-correlation of `a` with `b` by FFT — `a ⋆ b`, peak at the
    lag where b best matches a. numpy on both paths: this is one FFT pair and
    scipy would not do it differently."""
    a = np.asarray(a)
    b = np.asarray(b)
    n = next_fast_len(a.size + b.size - 1)
    A = np.fft.fft(a, n)
    B = np.fft.fft(np.conj(b[::-1]), n)
    out = np.fft.ifft(A * B)[: a.size + b.size - 1]
    return out if (np.iscomplexobj(a) or np.iscomplexobj(b)) else out.real
