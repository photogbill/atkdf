"""Reference signals for the moved `geoloc` tests — ATK's `siga/synth.py`, the
two kinds those tests use (qpsk, noise), copied so the engine's tests make
exactly the samples ATK's tests made from the same seeds.

Copied from ATK c52b913 on 2026-09-30. The random draws are in the same
order as in ATK (`_linear`'s symbol indices, then `add_noise`'s two normal
draws), so a seed gives the same record here as there; a change to either
copy that alters the draw order makes these tests measure a different
signal, which is why this file says where it came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from atkdf._prep import Samples


@dataclass
class Synth:
    """A generated signal and the answers it was generated from."""

    iq: np.ndarray
    fs: float
    truth: dict = field(default_factory=dict)

    def ctx(self) -> Samples:
        return Samples(iq=self.iq, fs=self.fs,
                       source=f"synth:{self.truth.get('kind', '?')}")

    def __len__(self) -> int:
        return int(self.iq.size)


def rrc_taps(beta: float, sps: int, span: int = 10) -> np.ndarray:
    """Root-raised-cosine taps, unit energy (ATK's `synth.rrc_taps`)."""
    sps = max(1, int(sps))
    beta = float(beta)
    n = np.arange(-span * sps / 2.0, span * sps / 2.0 + 1)
    t = n / float(sps)
    h = np.empty_like(t)
    if beta <= 0:
        h = np.sinc(t)
    else:
        sing = np.isclose(np.abs(t), 1.0 / (4.0 * beta), atol=1e-9)
        zero = np.isclose(t, 0.0, atol=1e-12)
        ok = ~(sing | zero)
        tt = t[ok]
        num = (np.sin(np.pi * tt * (1 - beta))
               + 4 * beta * tt * np.cos(np.pi * tt * (1 + beta)))
        den = np.pi * tt * (1 - (4 * beta * tt) ** 2)
        h[ok] = num / den
        h[zero] = 1.0 - beta + 4.0 * beta / np.pi
        if sing.any():
            h[sing] = (beta / np.sqrt(2.0)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
    e = np.sqrt(np.sum(h ** 2))
    return (h / e) if e > 0 else h


_ALPHABETS = {
    "bpsk": np.array([1.0, -1.0], dtype=complex),
    "qpsk": np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2),
}


def add_noise(x: np.ndarray, snr_db: float | None,
              rng: np.random.Generator) -> tuple:
    """AWGN at `snr_db` over the SAMPLED bandwidth. `(y, noise_power)`."""
    if snr_db is None:
        return x, 0.0
    ps = float(np.mean(np.abs(x) ** 2))
    if ps <= 0:
        return x, 0.0
    pn = ps / (10.0 ** (float(snr_db) / 10.0))
    noise = np.sqrt(pn / 2.0) * (rng.standard_normal(x.size)
                                 + 1j * rng.standard_normal(x.size))
    return (x + noise).astype(np.complex128), pn


def generate(kind: str, fs: float, duration_s: float = 0.05, *,
             baud: float = 9600.0, snr_db: float | None = 20.0,
             rolloff: float = 0.35, seed: int = 0) -> Synth:
    """ATK's `synth.generate` for kind in {bpsk, qpsk, noise}."""
    fs = float(fs)
    n_total = max(16, int(round(fs * float(duration_s))))
    rng = np.random.default_rng(int(seed))
    truth: dict = {"kind": kind, "fs": fs, "snr_db": snr_db, "seed": int(seed)}
    if kind in _ALPHABETS:
        sps = max(2, int(round(fs / max(1.0, float(baud)))))
        alpha = _ALPHABETS[kind]
        nsym = int(np.ceil(n_total / sps)) + 24
        idx = rng.integers(0, alpha.size, nsym)
        up = np.zeros(nsym * sps, dtype=complex)
        up[::sps] = alpha[idx]
        taps = rrc_taps(rolloff, sps)
        x = np.convolve(up, taps, mode="full")
        lead = (taps.size - 1) // 2
        x = x[lead: lead + n_total]
        p = float(np.mean(np.abs(x) ** 2))
        if p > 0:
            x = x / np.sqrt(p)
        truth.update(baud=fs / sps, sps=sps)
    elif kind == "noise":
        x = (rng.standard_normal(n_total)
             + 1j * rng.standard_normal(n_total)) / np.sqrt(2)
        snr_db = None
    else:
        raise ValueError(f"_signals makes bpsk, qpsk and noise, not {kind!r}")
    x = np.asarray(x, dtype=complex)[:n_total]
    if x.size < n_total:
        x = np.concatenate([x, np.zeros(n_total - x.size, dtype=complex)])
    x, _pn = add_noise(x, snr_db, rng)
    return Synth(iq=x.astype(np.complex64), fs=fs, truth=truth)


def noise_only(fs: float, duration_s: float, seed: int = 0) -> Synth:
    """The control every detector must NOT fire on."""
    return generate("noise", fs, duration_s, seed=seed, snr_db=None)
