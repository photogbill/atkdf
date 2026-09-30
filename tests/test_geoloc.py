"""Layer 9 — geolocation and direction finding: the maths, not the fielding.

The build scope for this layer is **the arithmetic and its caveats**, with the
panel held back until multi-SDR capture is coherent. That decision is the
subject of the first test in this file, and the rest test the maths against
synthetic arrays and known delays.

TWO THINGS THIS FILE EXISTS TO PIN DOWN, both of which are ways to produce a
confident bearing that is wrong:

  1. **ESPRIT's angle convention.** `steering_vector` builds a ULA response as
     exp(j·k·n·d·COS θ) — the projection onto the element axis — so inverting
     it takes arccos. Using arcsin returns 65° for a true 25° source, which is
     a plausible bearing on a plausible-looking display.
  2. **The cone ambiguity.** A linear array cannot separate +θ from −θ: the
     steering vectors are identical. Every routine here says so, and MUSIC's
     returned sign is therefore not evidence of anything.

And the standing one, in every measurement this module produces: ATK's radios
do not share a time or frequency reference, so a TDOA is correct arithmetic on
inputs that drift 300 m of position error per second of elapsed time.
"""

from __future__ import annotations

import numpy as np
import pytest

import _signals as synth
from atkdf import geoloc
from atkdf._prep import Samples as SigCtx
from atkdf.dossier import UNKNOWN

FS = 2_000_000.0


# ---------------------------------------------------------------------------
# the standing caveat
# ---------------------------------------------------------------------------

def test_every_two_receiver_measurement_carries_the_reference_caveat():
    """The one that decides whether the number means anything. It is not
    optional, and it is not phrased as a footnote."""
    a = synth.generate("qpsk", FS, 0.01, baud=100_000.0, snr_db=20.0, seed=1)
    ctx_a = a.ctx()
    ctx_b = SigCtx(iq=np.roll(np.asarray(a.iq), 40), fs=FS,
                   source="synth:second receiver")
    for m in (geoloc.tdoa_measurement(ctx_a, ctx_b),
              geoloc.fdoa(ctx_a, ctx_b)):
        assert any("time and frequency reference" in c for c in m.caveats)
        assert any("300 m" in c for c in m.caveats)


def test_the_caveat_names_the_drift_rate_rather_than_saying_be_careful():
    assert "1 ppm" in geoloc.REFERENCE_CAVEAT
    assert "per second" in geoloc.REFERENCE_CAVEAT


# ---------------------------------------------------------------------------
# TDOA
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shift", [0, 17, 40, -25])
def test_a_known_sample_delay_is_recovered(shift):
    s = synth.generate("qpsk", FS, 0.01, baud=200_000.0, snr_db=25.0, seed=2)
    x = np.asarray(s.iq)
    a = SigCtx(iq=x, fs=FS, source="synth:rx A")
    b = SigCtx(iq=np.roll(x, shift), fs=FS, source="synth:rx B")
    r = geoloc.tdoa(a, b)
    assert r.delay_s * FS == pytest.approx(-shift, abs=0.5)
    assert r.range_difference_m == pytest.approx(-shift / FS * geoloc.C,
                                                 rel=0.02, abs=20.0)


def test_sub_sample_interpolation_beats_the_sample_grid():
    """At 2 Msps one sample is 150 m of range difference, so an
    integer-sample TDOA quantises the fix into 150-metre steps at any SNR."""
    s = synth.generate("qpsk", FS, 0.02, baud=200_000.0, snr_db=30.0, seed=3)
    x = np.asarray(s.iq, dtype=np.complex128)
    # a half-sample delay, applied in the frequency domain
    n = x.size
    f = np.fft.fftfreq(n)
    y = np.fft.ifft(np.fft.fft(x) * np.exp(-2j * np.pi * f * 0.5))
    a = SigCtx(iq=x, fs=FS, source="synth:rx A")
    b = SigCtx(iq=y, fs=FS, source="synth:rx B")
    fine = geoloc.tdoa(a, b, interpolate=True).delay_s * FS
    coarse = geoloc.tdoa(a, b, interpolate=False).delay_s * FS
    assert coarse == pytest.approx(round(coarse), abs=1e-9)   # on the grid
    assert abs(fine - (-0.5)) < abs(coarse - (-0.5))


def test_two_receivers_hearing_unrelated_noise_are_told_so():
    """**The negative control, and it needed the multiple-comparisons
    correction to pass.** Correlating two independent noise records over
    40,000 lags puts the tallest lag ~13 dB above the median by chance, so
    the flat 10 dB bar that was here called this a usable fix."""
    a = synth.noise_only(FS, 0.01, seed=1).ctx()
    b = synth.noise_only(FS, 0.01, seed=2).ctx()
    m = geoloc.tdoa_measurement(a, b)
    assert any("may not be hearing the same signal" in c for c in m.caveats)
    assert m.confidence < 0.3
    assert m.evidence["peak_to_floor_db"] < (
        m.evidence["search_floor_db"] + 3.0)


def test_a_real_common_signal_clears_that_same_bar():
    """The other half — a bar nothing can clear is not a bar."""
    s = synth.generate("qpsk", FS, 0.01, baud=200_000.0, snr_db=20.0, seed=7)
    x = np.asarray(s.iq)
    a = SigCtx(iq=x, fs=FS, source="synth:rx A")
    b = SigCtx(iq=np.roll(x, 31), fs=FS, source="synth:rx B")
    m = geoloc.tdoa_measurement(a, b)
    assert m.evidence["peak_to_floor_db"] > (
        m.evidence["search_floor_db"] + 3.0)
    assert m.confidence > 0.3


def test_tdoa_reports_what_one_sample_is_worth_in_metres():
    s = synth.generate("qpsk", FS, 0.01, baud=200_000.0, seed=4)
    m = geoloc.tdoa_measurement(s.ctx(), s.ctx())
    assert m.evidence["sample_period_m"] == pytest.approx(
        geoloc.C / FS, abs=0.01)


def test_tdoa_declines_an_empty_record():
    e = SigCtx(iq=np.zeros(4, dtype=complex), fs=FS, source="synth:stub")
    assert geoloc.tdoa(e, e).correlation.size == 0
    assert geoloc.tdoa_measurement(e, e).value == UNKNOWN


# ---------------------------------------------------------------------------
# FDOA
# ---------------------------------------------------------------------------

def test_a_known_frequency_offset_between_receivers_is_recovered():
    s = synth.generate("qpsk", FS, 0.05, baud=200_000.0, snr_db=25.0, seed=5)
    x = np.asarray(s.iq, dtype=np.complex128)
    t = np.arange(x.size) / FS
    offset = 320.0
    a = SigCtx(iq=x * np.exp(2j * np.pi * offset * t), fs=FS, source="A")
    b = SigCtx(iq=x, fs=FS, source="B")
    m = geoloc.fdoa(a, b, max_hz=5000.0)
    assert m.value == pytest.approx(offset, abs=3.0), m.value


def test_fdoa_says_that_doppler_and_oscillator_error_are_not_separable():
    s = synth.generate("qpsk", FS, 0.02, baud=200_000.0, seed=6)
    m = geoloc.fdoa(s.ctx(), s.ctx())
    assert any("nothing in the measurement separates them" in c
               for c in m.caveats)
    assert any("400 Hz" in c for c in m.caveats)


# ---------------------------------------------------------------------------
# MUSIC and ESPRIT against a synthetic array
# ---------------------------------------------------------------------------

BEARINGS = [25.0, 40.0, 70.0, 115.0]


@pytest.mark.parametrize("truth", BEARINGS)
def test_esprit_returns_the_bearing_it_was_given(truth):
    """**The arccos test.** With arcsin this returns 65° for a true 25° —
    a bearing that looks entirely reasonable and is wrong by 40 degrees."""
    iq, _g = geoloc.synth_array(truth, elements=8, spacing_m=0.5,
                                wavelength_m=1.0, samples=4096, snr_db=25.0,
                                seed=1)
    m = geoloc.esprit(iq, spacing_m=0.5, wavelength_m=1.0)
    assert m.value == pytest.approx(truth, abs=1.0), (m.value, truth)


@pytest.mark.parametrize("truth", BEARINGS)
def test_music_finds_the_same_bearing_up_to_the_cone_ambiguity(truth):
    """MUSIC's grid covers ±180°, and for a linear array +θ and −θ have
    IDENTICAL steering vectors. So the magnitude is the measurement and the
    sign is not — which is exactly what the caveat says."""
    iq, g = geoloc.synth_array(truth, elements=8, spacing_m=0.5,
                               wavelength_m=1.0, samples=4096, snr_db=25.0,
                               seed=1)
    m = geoloc.music(iq, g, wavelength_m=1.0, sources=1)
    assert abs(m.value) == pytest.approx(truth, abs=1.0), (m.value, truth)


def test_both_array_methods_state_the_cone_ambiguity_and_the_coherence_need():
    iq, g = geoloc.synth_array(30.0, samples=2048, seed=2)
    for m in (geoloc.music(iq, g, wavelength_m=1.0),
              geoloc.esprit(iq, spacing_m=0.5, wavelength_m=1.0)):
        assert any("CONE" in c for c in m.caveats)
        assert any("coherent sampling" in c for c in m.caveats)


def test_music_states_that_it_needs_the_source_count_in_advance():
    iq, g = geoloc.synth_array(30.0, samples=2048, seed=2)
    m = geoloc.music(iq, g, wavelength_m=1.0, sources=1)
    assert any("NUMBER OF SOURCES" in c for c in m.caveats)
    assert len(m.evidence["eigenvalues"]) == 8
    ev = m.evidence["eigenvalues"]
    assert ev[0] > 4 * ev[1]          # one source: one dominant eigenvalue


def test_esprit_warns_when_the_spacing_aliases():
    """Beyond half a wavelength the phase-to-angle map is many-to-one, and
    the answer has aliases the display will not show."""
    iq, _g = geoloc.synth_array(30.0, spacing_m=0.9, wavelength_m=1.0,
                                samples=2048, seed=3)
    m = geoloc.esprit(iq, spacing_m=0.9, wavelength_m=1.0)
    assert any("ambiguous" in c and "aliases" in c for c in m.caveats)


def test_esprit_declines_an_array_too_small_to_have_sub_arrays():
    iq, _g = geoloc.synth_array(30.0, elements=2, samples=1024, seed=4)
    assert geoloc.esprit(iq, spacing_m=0.5, wavelength_m=1.0).value == UNKNOWN


def test_music_declines_too_few_elements_or_samples():
    assert geoloc.music(np.zeros((1, 1024), dtype=complex),
                        np.zeros((1, 2)), 1.0).value == UNKNOWN
    iq, g = geoloc.synth_array(30.0, samples=8, seed=5)
    assert geoloc.music(iq, g, wavelength_m=1.0).value == UNKNOWN


def test_two_sources_are_separated_when_music_is_told_there_are_two():
    """The super-resolution claim, and the reason to accept the source-count
    burden: two sources inside one beamwidth."""
    a, g = geoloc.synth_array(50.0, elements=8, samples=4096, snr_db=30.0,
                              seed=6)
    b, _g = geoloc.synth_array(62.0, elements=8, samples=4096, snr_db=30.0,
                               seed=7)
    m = geoloc.music(a + b, g, wavelength_m=1.0, sources=2)
    ev = m.evidence["eigenvalues"]
    assert ev[1] > 4 * ev[2]           # two dominant eigenvalues, not one


# ---------------------------------------------------------------------------
# Watson-Watt and beamforming
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("truth", [0.0, 35.0, 120.0, -95.0])
def test_watson_watt_recovers_the_bearing_from_crossed_loops(truth):
    n = 4096
    rng = np.random.default_rng(1)
    s = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    sin_ch = s * np.sin(np.radians(truth))
    cos_ch = s * np.cos(np.radians(truth))
    m = geoloc.watson_watt(sin_ch, cos_ch, omni_ch=s)
    got = float(m.value)
    assert min(abs(got - truth), abs(abs(got - truth) - 360.0)) < 2.0, got


def test_watson_watt_without_a_sense_channel_says_it_is_ambiguous_by_180():
    n = 2048
    rng = np.random.default_rng(2)
    s = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    m = geoloc.watson_watt(s * np.sin(np.radians(35.0)),
                           s * np.cos(np.radians(35.0)))
    assert any("180" in c for c in m.caveats)


def test_beamforming_towards_the_source_beats_beamforming_elsewhere():
    iq, g = geoloc.synth_array(40.0, elements=8, samples=4096, snr_db=25.0,
                               seed=8)
    on = geoloc.beamform(iq, g, 1.0, np.radians(40.0))
    off = geoloc.beamform(iq, g, 1.0, np.radians(110.0))
    assert np.mean(np.abs(on) ** 2) > 4 * np.mean(np.abs(off) ** 2)


def test_the_cone_ambiguity_is_a_measured_fact_and_not_only_a_caveat():
    """**+θ and −θ are not merely hard to separate; they are the SAME
    input.** Beamforming towards −40° on a source at +40° returns bit-for-bit
    the beam towards +40°, because for a linear array along x the response
    depends on cos θ. Any display offering a signed bearing from one linear
    baseline is offering half of it for free."""
    iq, g = geoloc.synth_array(40.0, elements=8, samples=2048, snr_db=25.0,
                               seed=9)
    plus = geoloc.beamform(iq, g, 1.0, np.radians(40.0))
    minus = geoloc.beamform(iq, g, 1.0, np.radians(-40.0))
    assert np.allclose(plus, minus)


# ---------------------------------------------------------------------------
# the RSSI centroid, and its honesty about being a centroid
# ---------------------------------------------------------------------------

def test_the_rssi_centroid_leans_towards_the_strongest_samples():
    pts = [{"lat": 51.50, "lon": -0.12, "power_db": -80.0},
           {"lat": 51.51, "lon": -0.12, "power_db": -60.0},
           {"lat": 51.52, "lon": -0.12, "power_db": -85.0}]
    r = geoloc.rssi_heatmap(pts)
    assert 51.505 < r["lat"] < 51.515
    assert r["strongest"]["power_db"] == -60.0
    assert r["spread_m"] > 0


def test_the_rssi_result_calls_itself_not_a_fix():
    r = geoloc.rssi_heatmap([{"lat": 0.0, "lon": 0.0, "power_db": -70.0},
                             {"lat": 0.01, "lon": 0.0, "power_db": -75.0}])
    assert "NOT a fix" in r["note"]
    assert "spread" in r["note"]


def test_the_rssi_heatmap_declines_an_empty_track():
    assert geoloc.rssi_heatmap([]) == {}


# ---------------------------------------------------------------------------
# the scope decision itself
# ---------------------------------------------------------------------------

def test_this_maths_reaches_arrays_by_the_FILE_path_only():
    """**The scope decision, restated after the move.** In ATK this test also
    checked the Geolocation tab (`siga_view`); that half stays in ATK's copy
    of this test, because the tab is ATK's. What moved with the maths is the
    other half: the file path says why it is allowed to exist, and it
    reaches no live receiver. The live, COHERENT path — the KrakenSDR through
    atkdaq — is `atkdf.doa` and `atkdf.engine`, which read a stream that
    declares its coherence in every frame header; this module still does not
    read receivers at all."""
    import pathlib
    from atkdf import df_file
    src = pathlib.Path(df_file.__file__).read_text(encoding="utf-8")
    assert "coherent by construction" in src
    assert "receiver_manager" not in src
