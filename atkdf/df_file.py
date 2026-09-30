"""Direction finding from a recorded file, with the ephemeris that came with it.

**THIS IS THE PATH THAT WORKS TODAY, AND THE REASON IS COHERENCE.** ATK's live
array direction finding is deliberately unwired: separate dongles sampling the
same signal are not phase-coherent, their phases wander independently, the
spatial covariance is not the array covariance, and MUSIC's peak means
nothing. None of that applies to a file. A recording made by a real DF system
is coherent by construction — the samples were digitised together off one
clock — and the ephemeris in the header supplies the position and attitude
reference the live path has no way to get. So the maths that has been sitting
in `geoloc.py` waiting for hardware can run now, on a file.

Three producers, one output. Whatever the file contains, what comes out is a
list of `geodesy.LOB` — a bearing, from a place, at a time, with a σ — which
is the only thing `fusion.py` ever needs to see:

    array          N coherent channels. MUSIC or ESPRIT per time block gives
                   the angle to the array AXIS, which is a CONE, and the
                   attitude rotates that cone into geographic coordinates.
                   Usually yields TWO candidate bearings — see below.
    watson_watt    crossed loops plus a sense channel. Gives a body-frame
                   azimuth directly, unambiguous once the sense channel has
                   resolved the 180°, and needs only the heading.
    aoa            the bearings were computed by the collection system and
                   written into the file. Then this is bookkeeping: attach
                   each one to the position it was taken from.

WHAT COMES OUT IS OFTEN TWO BEARINGS AND THAT IS NOT A DEFECT. A linear array
cannot distinguish a signal at +θ from its axis from one at −θ; the two give
identical phases across the elements. Both are reported, in one LOB with
`ambiguous_with` set, and resolving between them takes something this file
cannot supply: platform motion, a second non-collinear baseline, or an
a-priori. The bench's job is to say there are two.

AND IT WILL REFUSE WITHOUT AN ATTITUDE. An array angle is measured relative to
the array, so without a heading there is no way to express it relative to
north — not approximately, not with a caveat, not at all. `bearings` returns
nothing and says why, which is a better outcome than a plausible number
referenced to an unknown direction.

MOVED FROM ATK, 2026-09-30 (ATK c52b913, `atk/core/siga/df_file.py`), with its tests
and this docstring. ATK's `atk.core.siga.df_file` is now an alias of this
module — the same module object, reached through `atk/core/df_engine.py`
— so the Geolocation tab and ATK's own tests run this code. Edit it here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import bluefile
from . import ephemeris as eph
from . import geodesy as geo
from . import geoloc

#: Column and keyword names that mean "somebody already computed the bearing".
AOA_NAMES = ("AOA", "AZ", "AZIM", "AZIMUTH", "BEARING", "BRG", "DOA",
             "ANGLE_OF_ARRIVAL")
#: …and the ones that mean it is relative to the platform rather than to north.
RELATIVE_AOA_NAMES = ("RAOA", "REL_AZ", "RELATIVE_BEARING", "AOA_REL")


@dataclass
class DFConfig:
    """How the array was built. None of this can be guessed from the samples.

    `spacing_m` and `wavelength_m` set the scale: with the wrong ratio ESPRIT
    returns a real-looking angle that is wrong by however much the ratio is
    wrong. `wavelength_m` is derived from the file's centre frequency when the
    file states one, which is the usual case and the one to prefer.
    """
    elements: int = 0
    spacing_m: float = 0.5
    wavelength_m: float = 0.0            # 0 → derive from the centre frequency
    axis_body: tuple = (1.0, 0.0, 0.0)   # array axis in body coordinates
    method: str = "auto"                 # auto | esprit | music | watson_watt | aoa
    block: int = 4096                    # samples per bearing
    overlap: float = 0.0
    sigma_floor_deg: float = 0.5         # never claim better than this
    target_elevation_deg: float = 0.0    # where the cone is cut: the horizon
    geometry: np.ndarray | None = None   # explicit element positions, metres
    sense_channel: int | None = None     # Watson-Watt omni, if present

    #: Whether a failed coherence test stops the run.
    #:
    #: True by default and it is the right default: a bearing from channels
    #: that are not phase-coherent is not a weak bearing, it is a measurement
    #: of two oscillators. Turning it off is for looking at what the failure
    #: produces, and what it produces looks entirely normal.
    require_coherence: bool = True

    #: WHICH INSTALLATION PRODUCED THESE BEARINGS — the key that decides
    #: whether two collections share an alignment error or carry independent
    #: ones. It defaults to a single unnamed receiver, deliberately, because
    #: that is the CONSERVATIVE reading: two files from one aircraft on two
    #: days share its boresight error completely, and treating them as
    #: independent lets fusion average away an error that is not averaging.
    #: Naming the file here instead would make every capture look like its
    #: own independently calibrated system. Set it per receiver when you
    #: genuinely have more than one.
    source: str = "receiver"

    def wavelength(self, hdr: dict) -> float:
        if self.wavelength_m > 0:
            return float(self.wavelength_m)
        f = float(hdr.get("center_freq") or 0.0)
        return (299_792_458.0 / f) if f > 0 else 0.0


def read_array(path, max_samples: int | None = None) -> tuple:
    """`(iq, hdr, note)` with `iq` shaped (elements, samples).

    **BLUE AND SigMF BOTH, because ATK records SigMF.** For a long time this
    read Midas BLUE and nothing else, which meant a capture ATK had made
    itself — with the GPS track sitting beside it — could not be opened in
    the Geolocation tab at all. `recording.header` normalises the two into
    one vocabulary; the sample layout is genuinely the same in both, so there
    is one de-interleave rather than two.

    A multi-channel capture is a type-2000 BLUE file whose `subsize` is the
    element count, or a SigMF dataset declaring `core:num_channels`, and in
    both the samples arrive interleaved element-by-element. Getting the
    de-interleave wrong transposes the array — which does not error, it just
    produces bearings that are wrong in a structured way — so the shape and
    the element count are reported for the caller to sanity-check against the
    hardware they think they have.
    """
    from . import recio as _rec
    hdr = _rec.header(path)
    if hdr.get("format") == "blue":
        data, hdr = bluefile.read(path, max_samples=max_samples)
        hdr.setdefault("path", str(path))
        hdr.setdefault("format", "blue")
    else:
        n_ch = max(1, int(hdr.get("subsize") or 1))
        # max_samples is PER CHANNEL here, as it is for BLUE — asking for a
        # million samples of a four-element array means a million per
        # element, or the block arithmetic downstream silently quarters.
        want = None if max_samples is None else int(max_samples) * n_ch
        info = _rec.load(path, max_samples=want)
        if not info.get("ok"):
            raise bluefile.BlueError(info.get("problem", "unreadable"))
        data = info["iq"]
        hdr["samples"] = int(data.size)
        hdr["center_freq"] = hdr.get("center_freq") or info.get(
            "center_freq", 0.0)
    n_el = int(hdr.get("subsize") or 1)
    fmt = str(hdr.get("format") or "blue")
    if n_el <= 1:
        return np.atleast_2d(data), hdr, f"single channel ({fmt})"
    usable = (data.size // n_el) * n_el
    iq = np.asarray(data[:usable]).reshape(-1, n_el).T
    return iq, hdr, (
        f"{n_el} channels de-interleaved from "
        + (f"a type-{hdr.get('type')} frame of subsize {n_el}"
           if fmt == "blue" else
           f"a SigMF dataset declaring core:num_channels = {n_el}"))


#: How far past what the geometry could possibly explain the inter-channel
#: phase may wander before the "coherent" claim stops being believable.
#: Generous — the point is to catch free-running receivers, whose phase runs
#: away without bound, not to quibble with a well-behaved array.
COHERENCE_SLACK = 2.0

#: And how straight that wander has to be. A clock offset is a RAMP; geometry
#: is a wiggle. Below this correlation the excursion is not evidence of a
#: clock even when it is large.
COHERENCE_STRAIGHTNESS = 0.9


def coherence_check(iq, cfg: "DFConfig", hdr: dict, blocks: int = 16) -> dict:
    """Is this really one clock, or several receivers pretending?

    **A FILE CANNOT BE TAKEN AT ITS WORD ABOUT THIS.** `core:num_channels`
    and a BLUE `subsize` are claims about hardware, and interleaving four
    free-running dongles produces a file that is byte-for-byte the shape of a
    phase-coherent array. Every estimator downstream will accept it, return a
    real-looking angle, and the fix will be confidently somewhere else. It is
    the same class of failure as the unmodelled alignment bias: the internal
    checks all pass because the data is self-consistent, it is just not
    measuring what it says.

    The discriminator is physical rather than statistical. For an array of
    element spacing d and wavelength λ, the inter-element phase can only ever
    span ±2π·d/λ — that is the entire range the geometry has to work with,
    whatever the platform does. Two independent oscillators differing by Δf
    produce a phase that ramps at 2πΔf and does not stop. So: a monotonic
    excursion beyond what the aperture allows is a clock, and nothing else.

    **THE TEST CAN CATCH INCOHERENCE AND CAN NEVER CERTIFY COHERENCE**, and
    it says so rather than returning a comfortable True. An offset small
    enough to keep the phase inside the aperture over THIS capture produces a
    file indistinguishable from a coherent one, and the shorter the capture
    the larger that offset may be. So the outcome is False (caught) or None
    (not disproved), never True, and the note carries the detection floor in
    ppm — which is the number that says whether the answer means anything and
    the number that a longer dwell improves.

    The offset it reports is an ESTIMATE good to tens of percent, not a
    measurement: the platform's own geometry changes the true inter-element
    phase too, and that rides on top of the ramp being fitted. Enough to say
    which receiver is wrong and roughly by how much, which is what the number
    is for.
    """
    out = {"coherent": None, "note": "", "drift_hz": [], "ppm": [],
           "excursion_rad": [], "allowed_rad": 0.0, "floor_hz": None,
           "floor_ppm": None, "span_s": 0.0}
    iq = np.atleast_2d(np.asarray(iq))
    n_el, n = iq.shape
    lam = cfg.wavelength(hdr)
    if n_el < 2:
        out["note"] = "one channel — nothing to be coherent with"
        return out
    if lam <= 0 or float(cfg.spacing_m) <= 0:
        out["note"] = ("no wavelength or spacing, so the phase the geometry "
                       "could explain is unknown and the claim cannot be "
                       "tested")
        return out
    nb = max(4, int(blocks))
    size = n // nb
    if size < 64:
        out["note"] = f"capture too short to test coherence ({n} samples)"
        return out

    # the whole range the aperture can produce, end to end
    allowed = 2.0 * math.pi * float(cfg.spacing_m) / lam * 2.0
    out["allowed_rad"] = float(allowed)
    dt_block = size * float(_time_step(hdr)[0] or 0.0)
    worst = 0.0
    notes = []
    for k in range(1, n_el):
        ph = []
        for b in range(nb):
            seg0 = iq[0, b * size:(b + 1) * size]
            segk = iq[k, b * size:(b + 1) * size]
            # np.vdot conjugates its FIRST argument, so this is Σ x0* · xk
            # and its angle is φk − φ0: channel k's phase RELATIVE TO
            # channel 0, which is the sign an operator expects when the
            # report says which receiver is running fast.
            c = np.vdot(seg0, segk)
            ph.append(math.atan2(c.imag, c.real))
        ph = np.unwrap(np.asarray(ph, dtype=float))
        exc = float(np.max(ph) - np.min(ph))
        out["excursion_rad"].append(round(exc, 3))
        x = np.arange(nb, dtype=float)
        slope, _icept = np.polyfit(x, ph, 1)
        resid = ph - (slope * x + _icept)
        var = float(np.var(ph))
        straight = 0.0 if var <= 0 else 1.0 - float(np.var(resid)) / var
        drift = (float(slope) / (2.0 * math.pi * dt_block)
                 if dt_block > 0 else float("nan"))
        out["drift_hz"].append(None if math.isnan(drift) else round(drift, 6))
        fc = float(hdr.get("center_freq") or 0.0)
        ppm = (abs(drift) / fc * 1e6) if (fc > 0 and not math.isnan(drift))             else float("nan")
        out["ppm"].append(None if math.isnan(ppm) else round(ppm, 4))
        if exc > allowed * COHERENCE_SLACK and straight >=                 COHERENCE_STRAIGHTNESS:
            worst = max(worst, exc / max(allowed, 1e-9))
            notes.append(
                f"channel {k}: phase relative to channel 0 ramps through "
                f"{exc:.1f} rad, where this aperture can only account for "
                f"{allowed:.1f}"
                + (f" — about {abs(drift):,.3g} Hz of clock offset"
                   f"{f', roughly {ppm:,.2g} ppm' if not math.isnan(ppm) else ''}"
                   if not math.isnan(drift) else ""))

    # THE DETECTION FLOOR, which is the number that says what a pass is
    # worth. Anything slower than this keeps the phase inside the aperture
    # for the length of THIS capture and is invisible here however wrong it
    # is — so the floor is reported, in ppm, alongside the verdict.
    span_s = float(nb * dt_block)
    floor_hz = (allowed * COHERENCE_SLACK / (2.0 * math.pi * span_s)
                if span_s > 0 else float("nan"))
    fc = float(hdr.get("center_freq") or 0.0)
    floor_ppm = (floor_hz / fc * 1e6) if (fc > 0 and not math.isnan(floor_hz)) \
        else float("nan")
    out["floor_hz"] = None if math.isnan(floor_hz) else round(floor_hz, 4)
    out["floor_ppm"] = None if math.isnan(floor_ppm) else round(floor_ppm, 5)
    out["span_s"] = round(span_s, 6)
    floor_txt = ""
    if not math.isnan(floor_hz):
        floor_txt = (f" over {span_s:.3g} s this would have caught an offset "
                     f"above {floor_hz:,.3g} Hz"
                     + (f" ({floor_ppm:,.3g} ppm)"
                        if not math.isnan(floor_ppm) else ""))

    if notes:
        out["coherent"] = False
        out["note"] = (
            "THESE CHANNELS ARE NOT PHASE-COHERENT, whatever the file says. "
            + "; ".join(notes)
            + ". Independent oscillators produce a phase that ramps without "
              "bound; an array's cannot leave its aperture. Every bearing "
              "below is measuring the clocks, not the emitter — the numbers "
              "will look entirely normal and the fix will be somewhere else.")
        return out

    # NOT True. NOT-DISPROVED. The distinction is the whole point: a slow
    # offset over a short capture looks exactly like a coherent array, so
    # "nothing detected" is a statement about this capture's sensitivity and
    # not about the hardware.
    out["coherent"] = None
    out["note"] = (
        f"no clock offset large enough to leave the {allowed:.1f} rad this "
        f"aperture allows"
        + (f" (worst excursion {max(out['excursion_rad']):.1f} rad)"
           if out["excursion_rad"] else "")
        + ". That RULES OUT free-running receivers and does not establish "
          "coherence — a slower offset produces an identical file at this "
          "capture length, and the floor falls as the dwell grows."
        + floor_txt + ".")
    return out


def _timeline_mismatch(hdr: dict, track, duration_s: float) -> str:
    """`""` when the capture and its ephemeris agree about when this was.

    A refusal string otherwise, naming both times and, where it can, the
    likely cause — because "your capture says 2026 and your GPS says 2023"
    is an answer somebody can act on and "0 bearings" is not.
    """
    if track is None or not len(track):
        return ""
    t0 = float(hdr.get("timecode_unix") or 0.0) + float(hdr.get("xstart") or 0)
    first = float(track.states[0].epoch)
    last = float(track.states[-1].epoch)
    if t0 <= 0:
        return ("REFUSED: this capture carries no timecode, so there is no "
                "way to say which part of the track it belongs to. The "
                "ephemeris covers "
                f"{_stamp(first)} to {_stamp(last)}; the samples could be "
                "from any of it. A capture without a time is not a capture "
                "that can be direction-found from a moving platform.")
    # a static ephemeris has no span to fall outside of
    if track.static:
        return ""
    slack = max(duration_s, 1.0)
    if first - slack <= t0 <= last + slack:
        return ""
    off = t0 - (first if t0 < first else last)
    clock = str(hdr.get("clock") or "")
    hint = ""
    if abs(off) > 86_400:
        hint = (" — that is a whole-day scale difference, which is what an "
                "unsynchronised PC clock looks like beside a GPS")
    elif abs(round(off / 3600.0) - off / 3600.0) < 0.02:
        hint = (" — a whole number of hours, which is what a local-time "
                "stamp read as UTC looks like")
    return (
        f"REFUSED: the capture and its ephemeris are not the same event. The "
        f"samples are stamped {_stamp(t0)}"
        + (f" (by the {clock})" if clock else "")
        + f" and the track runs {_stamp(first)} to {_stamp(last)}, "
          f"{abs(off):,.0f} s away{hint}. Matching blocks to positions across "
          f"that gap would produce a full set of ordinary-looking bearings "
          f"attached to the wrong places, so it is refused instead. Fix the "
          f"clock, or supply the right track.")


def _stamp(t: float) -> str:
    import time as _t
    return _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime(float(t)))


def _describe_sigmf(hdr: dict) -> str:
    """The one-line banner for a SigMF capture, in the same shape as the BLUE
    one — so a report says what was actually opened rather than leaving the
    reader to guess which reader ran."""
    n = int(hdr.get("subsize") or 1)
    rate = float(hdr.get("sample_rate") or 0.0)
    bits = [f"SigMF · {hdr.get('datatype') or 'unknown format'}"]
    if rate:
        bits.append(f"{rate / 1e6:,.3g} Msps")
    if hdr.get("center_freq"):
        bits.append(f"@ {float(hdr['center_freq']) / 1e6:,.4g} MHz")
    bits.append(f"{n} channel{'s' if n != 1 else ''}"
                + (" (declared coherent)" if n > 1 else ""))
    t = float(hdr.get("timecode_unix") or 0.0)
    if t:
        import time as _t
        bits.append(_t.strftime("%Y-%m-%d %H:%M:%SZ", _t.gmtime(t)))
    else:
        bits.append("NO TIMECODE — cannot be paired with another capture")
    return "    " + " · ".join(bits)


def _time_step(hdr: dict) -> tuple:
    """Seconds per frame, and which field it came from.

    Type 2000 has two axes and producers disagree about which one carries
    time for a multi-channel recording: `ydelta` is the frame interval by the
    letter of the format, but plenty of tools write the sample interval into
    `xdelta` and leave `ydelta` at its default. Both are reported so a wrong
    guess is visible as a timeline that does not match the collection.
    """
    yd = float(hdr.get("ydelta") or 0.0)
    xd = float(hdr.get("xdelta") or 0.0)
    if yd > 0 and (xd <= 0 or abs(math.log10(yd / max(xd, 1e-30))) < 6):
        return yd, "ydelta (the frame interval)"
    if xd > 0:
        return xd, "xdelta — no usable ydelta, so the x axis was read as time"
    return 0.0, "no time step in the header; blocks are numbered, not timed"


# ---------------------------------------------------------------------------
# the producers
# ---------------------------------------------------------------------------

def _blocks(n: int, size: int, overlap: float):
    step = max(1, int(size * (1.0 - float(overlap))))
    for s in range(0, max(0, n - size + 1), step):
        yield s, s + size


def _array_bearings(iq, hdr, cfg: DFConfig, track, t0, dt) -> tuple:
    """Cone angles per block, rotated into geographic bearings."""
    lam = cfg.wavelength(hdr)
    if lam <= 0:
        return [], ("the array needs a wavelength: the file states no centre "
                    "frequency and none was supplied, so an element spacing "
                    "in metres cannot be turned into an angle")
    n_el = iq.shape[0]
    if n_el < 2:
        return [], "array direction finding needs at least two channels"
    geometry = cfg.geometry
    if geometry is None:
        geometry = np.stack([np.arange(n_el) * float(cfg.spacing_m),
                             np.zeros(n_el)], axis=1)
    use_esprit = (cfg.method in ("auto", "esprit")) and n_el >= 3 \
        and cfg.geometry is None
    out, notes = [], []
    for k, (a, b) in enumerate(_blocks(iq.shape[1], int(cfg.block),
                                       cfg.overlap)):
        seg = iq[:, a:b]
        if use_esprit:
            m = geoloc.esprit(seg, spacing_m=float(cfg.spacing_m),
                              wavelength_m=lam)
        else:
            m = geoloc.music(seg, geometry, wavelength_m=lam)
        if not m.known:
            continue
        t = t0 + (a + (b - a) / 2.0) * dt
        st = track.at(t) if track else None
        if st is None:
            notes.append(f"block {k}: no ephemeris at {t:.3f}")
            continue
        if not st.bearing_ready:
            notes.append(f"block {k}: the ephemeris has no heading")
            continue
        # `cone_deg` is the angle to the array AXIS, which is what both
        # estimators return for a uniform linear array.
        cone = float(m.value)
        bearings = geo.cone_bearings(
            cone, st.heading_deg, st.pitch_deg, st.roll_deg,
            axis_body=cfg.axis_body,
            target_elevation_deg=float(cfg.target_elevation_deg))
        if not bearings:
            notes.append(f"block {k}: the cone at {cone:.1f}° never reaches "
                         f"the target elevation from this attitude")
            continue
        sigma = _sigma_from(m, cfg, st)
        out.append(geo.LOB(
            lat=st.lat, lon=st.lon, alt_m=st.alt_m,
            bearing_deg=bearings[0], sigma_deg=sigma, epoch=t,
            # the platform's own position uncertainty travels WITH the
            # bearing: at range it becomes an angular error the estimator
            # has to know about, and the ephemeris is the only thing that
            # knows how stale or interpolated this state was
            position_sigma_m=st.position_sigma_m,
            source=cfg.source or str(hdr.get("path") or ""),
            method=m.method,
            ambiguous_with=(bearings[1] if len(bearings) > 1 else None),
            evidence={"cone_deg": round(cone, 3),
                      "block": [a, b],
                      "heading_deg": round(st.heading_deg, 3),
                      "interpolated": st.interpolated,
                      "position_sigma_m": round(st.position_sigma_m, 1),
                      "estimator": m.method,
                      **{k2: v for k2, v in (m.evidence or {}).items()
                         if k2 in ("peak_over_local_floor_db",
                                   "spacing_wavelengths")}}))
    return out, "; ".join(notes[:6])


def _sigma_from(m, cfg: DFConfig, st) -> float:
    """A bearing's uncertainty, from the estimator and from the ephemeris.

    THREE TERMS, and leaving any of them out understates the answer:
    the estimator's own confidence, the heading uncertainty — which passes
    straight into the bearing one for one — and a floor, because no bench
    should claim a hundredth of a degree from a four-element array.
    """
    conf = float(getattr(m, "confidence", 0.0) or 0.0)
    est = max(float(cfg.sigma_floor_deg), 3.0 * (1.0 - conf))
    return float(math.sqrt(est ** 2 + float(st.heading_sigma_deg) ** 2))


def _watson_watt_bearings(iq, hdr, cfg: DFConfig, track, t0, dt) -> tuple:
    """Crossed loops: a body-frame azimuth per block, plus the heading."""
    if iq.shape[0] < 2:
        return [], "Watson-Watt needs at least a sine and a cosine channel"
    sense = cfg.sense_channel
    if sense is None and iq.shape[0] >= 3:
        sense = 2
    out, notes = [], []
    for k, (a, b) in enumerate(_blocks(iq.shape[1], int(cfg.block),
                                       cfg.overlap)):
        m = geoloc.watson_watt(iq[0, a:b], iq[1, a:b],
                               iq[sense, a:b] if sense is not None else None)
        if not m.known:
            continue
        t = t0 + (a + (b - a) / 2.0) * dt
        st = track.at(t) if track else None
        if st is None or not st.bearing_ready:
            notes.append(f"block {k}: no usable ephemeris")
            continue
        rel = float(m.value)
        # A crossed-loop pair measures azimuth in the platform's own frame, so
        # the true bearing is that plus the heading — with pitch and roll
        # applied properly rather than added, which matters as soon as the
        # platform is not level.
        vec = (math.cos(math.radians(rel)), math.sin(math.radians(rel)), 0.0)
        az, _el = geo.body_to_bearing(vec, st.heading_deg, st.pitch_deg,
                                      st.roll_deg)
        ambiguous = None
        if sense is None:
            ambiguous = (az + 180.0) % 360.0
        out.append(geo.LOB(
            lat=st.lat, lon=st.lon, alt_m=st.alt_m, bearing_deg=az,
            sigma_deg=_sigma_from(m, cfg, st), epoch=t,
            position_sigma_m=st.position_sigma_m,
            source=cfg.source or str(hdr.get("path") or ""),
            method=m.method,
            ambiguous_with=ambiguous,
            evidence={"relative_deg": round(rel, 3), "block": [a, b],
                      "heading_deg": round(st.heading_deg, 3),
                      "sense_channel": sense}))
    return out, "; ".join(notes[:6])


def _aoa_bearings(path, hdr, cfg: DFConfig, track) -> tuple:
    """Bearings the collection system already computed.

    Read from a record file's columns, or from header keywords for a single
    intercept. The only real question is whether they are TRUE or RELATIVE to
    the platform, and the column name is the only evidence — so a relative
    name gets the heading applied and a true one does not, and which was
    assumed is recorded on every bearing.
    """
    rows, names = [], set()
    try:
        arr, hdr2 = bluefile.read_records(path)
        if arr.size:
            names = {n.upper() for n in (arr.dtype.names or ())}
            rows = arr
            hdr = hdr2
    except (bluefile.BlueError, OSError, ValueError):
        rows = []
    kw = {**(hdr.get("keywords") or {}), **(hdr.get("ext_keywords") or {})}

    def pick(pool):
        for n in AOA_NAMES:
            if n in pool:
                return n, False
        for n in RELATIVE_AOA_NAMES:
            if n in pool:
                return n, True
        return None, False

    out = []
    col, relative = pick(names)
    if col is not None and len(rows):
        low = col.lower()
        tcol = "time" if "TIME" in names else None
        for i, row in enumerate(rows):
            t = float(row[tcol]) if tcol else 0.0
            if t > bluefile.J1950_TO_UNIX / 2:
                t -= bluefile.J1950_TO_UNIX
            st = track.at(t) if track else None
            if st is None:
                continue
            val = float(row[low])
            if relative:
                if not st.bearing_ready:
                    continue
                val = (val + st.heading_deg) % 360.0
            # SIGM as well as SIGMA: class-3 column names are four
            # characters, so a producer writing "SIGMA" gets "SIGM" and a
            # reader looking only for the long form silently loses the
            # uncertainty — and then invents one, which is worse.
            sig = max(cfg.sigma_floor_deg, 2.0)
            for cand in ("sigma", "sigm", "err", "unc"):
                if cand.upper() in names:
                    sig = float(row[cand])
                    break
            out.append(geo.LOB(
                lat=st.lat, lon=st.lon, alt_m=st.alt_m, bearing_deg=val % 360,
                sigma_deg=sig, epoch=t, source=cfg.source or str(path),
                position_sigma_m=st.position_sigma_m,
                method=f"AOA column {col!r} "
                       f"({'relative, heading applied' if relative else 'true'})",
                evidence={"column": col, "relative": relative, "row": i}))
        return out, f"{len(out)} bearings from the {col!r} column"

    kcol, krel = pick(set(kw))
    if kcol is not None:
        try:
            val = float(kw[kcol])
        except (TypeError, ValueError):
            return [], f"the {kcol!r} keyword is not a number"
        t = float(hdr.get("timecode_unix") or 0.0)
        st = track.at(t) if track else None
        if st is None:
            return [], f"the {kcol!r} keyword has no ephemeris to attach to"
        if krel:
            if not st.bearing_ready:
                return [], (f"{kcol!r} is a RELATIVE bearing and the "
                            f"ephemeris has no heading to reference it to")
            val = (val + st.heading_deg) % 360.0
        return [geo.LOB(lat=st.lat, lon=st.lon, alt_m=st.alt_m,
                        bearing_deg=val % 360,
                        sigma_deg=max(cfg.sigma_floor_deg, 2.0), epoch=t,
                        source=cfg.source or str(path),
                        position_sigma_m=st.position_sigma_m,
                        method=f"AOA keyword {kcol!r}",
                        evidence={"keyword": kcol, "relative": krel})], \
            f"one bearing from the {kcol!r} keyword"
    return [], "no pre-computed bearing column or keyword"


# ---------------------------------------------------------------------------
# the entry point
# ---------------------------------------------------------------------------

def detect_method(hdr: dict, iq_shape=None) -> tuple:
    """`(method, why)` — what this file can actually support."""
    n_el = int(hdr.get("subsize") or (iq_shape[0] if iq_shape else 1) or 1)
    kw = {**(hdr.get("keywords") or {}), **(hdr.get("ext_keywords") or {})}
    cols = {c["name"].upper() for c in (hdr.get("subr") or hdr.get("comp")
                                        or [])}
    if (cols | set(kw)) & (set(AOA_NAMES) | set(RELATIVE_AOA_NAMES)):
        return "aoa", "the file already carries computed bearings"
    if n_el == 2:
        return "watson_watt", ("two channels — read as crossed loops. Two "
                               "channels of a linear array would need the "
                               "method set explicitly, because two elements "
                               "cannot resolve anything on their own")
    if n_el == 3 and (hdr.get("ext_keywords") or {}).get("DF_TYPE", "") \
            .upper().startswith("WATSON"):
        return "watson_watt", "three channels and DF_TYPE says Watson-Watt"
    if n_el >= 3:
        return "esprit", f"{n_el} coherent channels — a uniform linear array"
    # ONE CHANNEL IS THE COMMON CASE AND IT DESERVES THE FULL SENTENCE.
    # This is what an ordinary ATK recording looks like — one dongle, one
    # stream — and "0 bearings" tells its owner nothing about why or what to
    # do instead. Direction finding is not a property of the software.
    return "", (
        f"{n_el} channel: a single receiver cannot produce a bearing at all — "
        f"the direction is in the PHASE DIFFERENCE between antennas, and with "
        f"one antenna there is no difference to measure. This file is fine; "
        f"it is the wrong kind of file for this question. What works: a "
        f"coherent multi-channel capture (an array, or a crossed-loop pair) "
        f"digitised off ONE clock — two dongles will not do, they free-run "
        f"and their phase relationship is meaningless; or a file from a DF "
        f"receiver that already carries computed bearings, which this reads "
        f"straight out of the header. Failing that, a single-receiver "
        f"collection still fixes an emitter by TDOA against a SECOND "
        f"recording of the same signal from somewhere else."
        + (" (This capture declares one channel in its SigMF sidecar; if the "
           "hardware really was coherent, core:num_channels is what says so.)"
           if str(hdr.get("format", "")) == "sigmf" else ""))


def bearings(path, cfg: DFConfig | None = None, keymap: dict | None = None,
             rows=None, max_samples: int | None = 1 << 22,
             use_course_as_heading: bool = False) -> tuple:
    """`(lobs, report)` — every line of bearing this file supports.

    The report is the deliverable when the list is empty. "No bearings" is
    useless; "four channels and a good ephemeris, but the ephemeris carries
    no heading, so an array angle cannot be referenced to north" tells an
    analyst exactly what to go and find.
    """
    cfg = cfg or DFConfig()
    report: list = []
    from . import recio as _rec
    try:
        hdr = _rec.header(path)
    except (bluefile.BlueError, OSError, ValueError) as exc:
        return [], [f"cannot read {path}: {exc}"]
    report.append(bluefile.describe(hdr) if hdr.get("format") == "blue"
                  else _describe_sigmf(hdr))

    track, ephem_report = eph.load(path, keymap=keymap, rows=rows)
    report.extend(ephem_report)
    if track is None:
        report.append("REFUSED: without an ephemeris a bearing has no origin, "
                      "and a bearing with no origin is not a line of bearing.")
        return [], report
    if use_course_as_heading and not track.has_attitude:
        track = track.use_course_as_heading()
        report.append("course over ground substituted for heading, on request")

    method = cfg.method
    if method == "auto":
        method, why = detect_method(hdr)
        report.append(f"method: {method or 'none available'} — {why}")
        if not method:
            return [], report

    if method == "aoa":
        lobs, note = _aoa_bearings(path, hdr, cfg, track)
        report.append(note)
        return lobs, report

    iq, hdr2, note = read_array(path, max_samples=max_samples)
    hdr.update({k: v for k, v in hdr2.items() if k not in hdr})
    report.append(note)
    dt, tnote = _time_step(hdr)
    report.append(f"time step: {dt:g} s from {tnote}")

    # BEFORE ANY BEARING IS COMPUTED, and before the ephemeris refusals, so
    # one look at a file surfaces EVERY blocker rather than the first one — a
    # capture with no heading AND incoherent channels needs both sentences,
    # not two round trips. A file that interleaves free-running receivers has
    # exactly the shape of a coherent array and produces entirely
    # normal-looking angles, so the claim is tested against physics rather
    # than taken from the header.
    coh = coherence_check(iq, cfg, hdr)
    if coh["note"]:
        report.append(coh["note"])
    incoherent = coh["coherent"] is False
    t0 = float(hdr.get("timecode_unix") or 0.0) + float(hdr.get("xstart")
                                                        or 0.0)
    if dt <= 0:
        report.append("REFUSED: with no time step, no block can be matched to "
                      "a position on the track.")
        return [], report

    # DO THE SAMPLES AND THE POSITIONS SHARE A TIMELINE? A bearing is matched
    # to where the platform was by ABSOLUTE time, so a capture stamped from a
    # drifted PC clock and a track stamped from GPS describe the same event at
    # two different moments and every block looks for a position that is not
    # there. The symptom — "no ephemeris at 1787342772.002", five times — says
    # nothing about the cause, so the cause is named here. Sliding one
    # timeline onto the other would be worse than refusing: it produces a full
    # set of ordinary-looking bearings attached to the wrong positions.
    span = _timeline_mismatch(hdr, track, iq.shape[1] * dt)
    if span:
        report.append(span)
        return [], report
    if not track.has_attitude:
        report.append(
            "REFUSED: the ephemeris carries no heading. An array measures an "
            "angle relative to ITSELF, so without knowing where the array was "
            "pointing there is no way to express that relative to north — not "
            "approximately, not with a caveat. Supply a heading, or pass "
            "use_course_as_heading if course over ground is genuinely all "
            "there is.")
        return [], report

    if incoherent and cfg.require_coherence:
        report.append(
            "REFUSED: bearings from channels that are not phase-coherent are "
            "not bearings. Set require_coherence=False to see them anyway — "
            "they will look completely ordinary and they will be wrong.")
        return [], report

    if method == "watson_watt":
        lobs, note = _watson_watt_bearings(iq, hdr, cfg, track, t0, dt)
    else:
        lobs, note = _array_bearings(iq, hdr, cfg, track, t0, dt)
    if note:
        report.append(note)
    if incoherent:
        for lob in lobs:
            lob.evidence["coherence"] = coh["note"]
    report.append(f"{len(lobs)} bearings from "
                  f"{len(list(_blocks(iq.shape[1], cfg.block, cfg.overlap)))} "
                  f"blocks")
    if lobs:
        report.append(_scatter_note(lobs))
        amb = sum(1 for l in lobs if l.ambiguous_with is not None)
        if amb:
            report.append(
                f"{amb} of {len(lobs)} bearings are AMBIGUOUS: a linear array "
                f"cannot separate +θ from −θ about its own axis, so each of "
                f"these has a mirror solution. If the platform moved during "
                f"the collection the true one stays self-consistent and the "
                f"mirror does not — which is what the fix's consistency test "
                f"will show.")
    return lobs, report


#: How far the platform may move before block-to-block scatter stops being a
#: measurement of the bearing error and starts being a measurement of the
#: platform's own motion.
DWELL_SPAN_M = 500.0


def _spread(lobs) -> float:
    lat = [l.lat for l in lobs]
    lon = [l.lon for l in lobs]
    c = (float(np.mean(lat)), float(np.mean(lon)))
    return max((geo.distance_between(c[0], c[1], a, b)
                for a, b in zip(lat, lon)), default=0.0)


def _scatter_note(lobs) -> str:
    """Compare the claimed σ against the observed block-to-block scatter.

    A CLAIMED UNCERTAINTY THAT DOES NOT MATCH THE OBSERVED SCATTER IS THE ONE
    NUMBER MOST WORTH CHECKING, because every ellipse downstream is built from
    it: if consecutive blocks of the same signal disagree by four times the
    stated σ, the σ is wrong and every fix will be overconfident.

    **But it only means anything for a STATIONARY dwell.** On a collection run
    the true bearing changes from block to block because the platform moved,
    and comparing raw scatter against σ then measures the flight path — which
    on a 28 km run reads as 41° of "error" and reports a perfectly good array
    as wildly optimistic. So the platform's own movement is checked first, and
    when it has moved the honest answer is to defer to the post-fit residuals,
    which subtract the geometry.
    """
    if len(lobs) < 3:
        return "too few bearings to compare the claimed σ against the scatter"
    spread = _spread(lobs)
    claimed = float(np.median([l.sigma_deg for l in lobs]))
    if spread > DWELL_SPAN_M:
        return (f"claimed σ {claimed:.2f}° · the platform moved "
                f"{spread:,.0f} m during this collection, so raw scatter "
                f"between blocks is mostly the changing geometry and says "
                f"nothing about the bearing error. The post-fit residuals "
                f"are the number to look at instead.")
    b = np.array([l.bearing_deg for l in lobs])
    scatter = float(np.median(np.abs(geo.angle_difference(
        b, np.median(b)))) * 1.4826)
    ratio = scatter / max(claimed, 1e-9)
    verdict = ("consistent" if 0.5 <= ratio <= 2.0 else
               "OPTIMISTIC — the bearings scatter more than they claim to"
               if ratio > 2.0 else
               "conservative — the bearings agree better than they claim to")
    return (f"claimed σ {claimed:.2f}° vs observed scatter {scatter:.2f}° "
            f"across {len(lobs)} blocks from one dwell: {verdict}")


def calibrate_sigma(lobs, fix=None, floor_deg: float = 0.0) -> list:
    """Replace claimed σ with the σ the bearings actually demonstrate.

    Two ways, and which one applies is decided by whether the platform moved:

      * **stationary dwell** — the true bearing was the same for every block,
        so the spread between them IS the error.
      * **moving** — the true bearing changed, so the spread is geometry.
        Pass a `fix` and the POST-FIT RESIDUALS are used instead, which have
        the geometry subtracted. Without one, nothing is changed, because
        calibrating against the platform's own flight path would inflate
        every σ to the width of the run and produce ellipses kilometres wide
        that would then look reassuringly conservative.

    Either way this is usually a better number than an estimator's
    self-reported confidence, which measures how peaked a spectrum was rather
    than how right it was.
    """
    if len(lobs) < 5:
        return list(lobs)
    n = len(lobs)
    if _spread(lobs) > DWELL_SPAN_M:
        if fix is None:
            return list(lobs)
        resid = np.array(fix.evidence.get("residuals_deg", []), dtype=float)
        if resid.size != n or n <= 3:
            return list(lobs)
        # THE FIT ATE TWO DEGREES OF FREEDOM. Residuals about a position that
        # was itself chosen to minimise them are smaller than the true errors
        # by sqrt((n−2)/n), and skipping that correction makes a six-bearing
        # fix look 22 % better than it is.
        scatter = float(np.median(np.abs(resid)) * 1.4826) \
            * math.sqrt(n / max(1.0, n - 2.0))
        why = "post-fit residuals, corrected for two fitted parameters"
    else:
        b = np.array([l.bearing_deg for l in lobs])
        scatter = float(np.median(np.abs(geo.angle_difference(
            b, np.median(b)))) * 1.4826)
        why = "block-to-block scatter within one dwell"

    # **RESIDUALS CANNOT SEE A SHARED ERROR.** Anything common to every
    # bearing — a heading offset, an antenna alignment, a mounting angle —
    # moves them all together and leaves the residuals untouched. So this
    # calibration measures the RANDOM part only, and driving σ below the
    # heading uncertainty would claim a precision that the platform's own
    # attitude cannot support. Measured on a six-block run: raw residuals
    # gave σ = 0.036°, a 25 m ellipse, and a fix 89 m from the truth — the
    # ellipse excluded the answer because it excluded the error that put it
    # there. Whatever remains shared must still be passed to the estimator as
    # `systematic_deg`; that is the other half of this and it is not
    # something residuals can supply.
    # …and a FLOOR, because six residuals produce a very noisy estimate of
    # their own spread and a four-element array claiming 0.02° is not a
    # measurement, it is an artefact of the median absolute deviation of a
    # small sample. Observed: calibration drove σ to 0.02° against a residual
    # RMS of 0.053°, and the consistency test then correctly rejected a fix
    # that was actually 90 m from the truth — the fit was fine, the σ was not.
    floor = max(float(floor_deg),
                float(np.median([l.evidence.get("heading_sigma_deg", 0.0)
                                 or 0.0 for l in lobs])), 1e-3)
    scatter = max(scatter, floor)
    out = []
    for l in lobs:
        m = geo.LOB(**{**l.__dict__})
        m.sigma_deg = scatter
        m.evidence = {**(l.evidence or {}), "sigma_source": why,
                      "sigma_claimed": round(l.sigma_deg, 3),
                      "random_only": True,
                      "note": "this sigma is the RANDOM error only — a shared "
                              "bias leaves residuals unchanged and must be "
                              "given to the fit as systematic_deg"}
        out.append(m)
    return out
