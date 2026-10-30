"""The path tag by frequency: P25 has no BS/MS sync word, so the band plan
(a site's IDEN table, then the region's duplex rule) decides — and refuses to
guess where it cannot."""

from __future__ import annotations

import pytest

from atkdf import engine, sim
from atkdf import manifold as M
from atkdf.bandplan import BandPlan, IdenEntry, IdenTable
from atkdf.fakes import ListCatalogue
from atkdf.identity import path_evidence, path_tag_of
from atkdf.ports import BurstEvent, Reference


@pytest.mark.parametrize("f_mhz, direction", [
    (815.3, "inbound"), (860.3, "outbound"),            # 800 MHz, -45 MHz split
    (802.5, "inbound"), (772.5, "outbound"),            # 700 MHz public safety
    (898.0, "inbound"), (937.0, "outbound"),            # 900 MHz
])
def test_us_split_bands_decide_by_frequency(f_mhz, direction):
    c = BandPlan.for_region("us").classify(f_mhz * 1e6)
    assert c.direction == direction
    assert c.path_tag == {"inbound": "direct", "outbound": "repeated"}[direction]
    assert c.why and c.region == "us"


def test_a_talkaround_channel_is_unknown_not_repeated():
    c = BandPlan.for_region("us").classify(851.0125e6)        # 8CALL90
    assert c.direction == "talkaround" and c.path_tag == "unknown"
    assert "talk-around" in c.why


@pytest.mark.parametrize("f_mhz", [155.0, 460.0, 220.5])
def test_bands_without_a_rule_say_so(f_mhz):
    c = BandPlan.for_region("us").classify(f_mhz * 1e6)
    assert c.direction == "unknown" and "no single duplex split" in c.why


def test_an_unknown_region_guesses_nothing():
    c = BandPlan.for_region("atlantis").classify(860.3e6)
    assert c.direction == "unknown" and "atlantis" in c.why


def test_the_iden_table_resolves_channels_both_ways():
    t = IdenTable([IdenEntry(1, 851.00625e6, 6_250.0, -45e6),
                   IdenEntry(2, 769.00625e6, 12_500.0, 30e6, slots=2)], name="site 1-1")
    out, inb, slot = t.channel((1 << 12) | 100)
    assert out == pytest.approx(851.63125e6) and inb == pytest.approx(806.63125e6)
    assert slot is None
    out, inb, slot = t.channel((2 << 12) | 7)               # TDMA: carrier 3, slot 1
    assert out == pytest.approx(769.04375e6) and slot == 1
    t.note_channel((1 << 12) | 100)
    assert t.classify(806.63125e6 + 900).direction == "inbound"   # a handset 900 Hz off
    assert t.classify(851.63125e6).direction == "outbound"
    with pytest.raises(ValueError):
        IdenEntry(16, 851e6, 6250, -45e6)


def test_the_site_table_is_consulted_before_the_region():
    # a UHF trunked site: the region has no rule there, the site's table does —
    # for the channels its grants have named
    bp = BandPlan.for_region("us")
    assert bp.classify(453.5e6).direction == "unknown"
    t = IdenTable([IdenEntry(3, 451.0e6, 12_500.0, 5e6)], name="uhf site")
    bp.add_site(t)
    ch = (3 << 12) | 200
    out, inb, _ = t.channel(ch)                     # 453.5 / 458.5 MHz
    # before any grant: 458.5 MHz is on BOTH rasters (5 MHz is 400 channel
    # steps — true of real tables), so the table honestly cannot say
    assert bp.classify(inb).direction == "unknown"
    t.note_channel(ch)
    assert bp.classify(out).direction == "outbound"
    assert bp.classify(inb).direction == "inbound"


def test_evidence_order_sync_then_modulation_then_frequency_then_catalogue():
    bp = BandPlan.for_region("us")
    ev = BurstEvent("p25", "BEE00-1A2", "1234567", 0.0, fc_hz=860.3e6)
    assert path_evidence(ev, None, bp)[0] == "repeated"
    # the decoder's sync type wins over the band
    assert path_tag_of(BurstEvent("dmr", "1", "9", 0.0, sync_type="ms", fc_hz=860.3e6),
                       None, bp) == "direct"
    # P25 Phase 2: outbound H-DQPSK, inbound H-CPM; trunking ISP / OSP
    assert path_tag_of(BurstEvent("p25", "", "", 0.0, sync_type="h-cpm"), None, None) == "direct"
    assert path_tag_of(BurstEvent("p25", "", "", 0.0, sync_type="H-DQPSK"), None, None) == "repeated"
    assert path_tag_of(BurstEvent("p25", "", "", 0.0, sync_type="osp"), None, None) == "repeated"
    # a UHF repeater output known only to the catalogue
    cat = ListCatalogue([Reference(key="ref:county", lat=0, lon=0, kind="repeater_output",
                                   name="county P25", fc_hz=460.125e6)])
    tag, why = path_evidence(BurstEvent("p25", "", "", 0.0, fc_hz=460.1255e6), cat, bp)
    assert tag == "repeated" and "county P25" in why
    tag, why = path_evidence(BurstEvent("p25", "", "", 0.0, fc_hz=851.5125e6), None, bp)
    assert tag == "unknown" and "talk-around" in why
    # no plan at all: the old behaviour
    assert path_tag_of(ev) == "unknown"


def _p25_run(fc_hz, bandplan):
    # a 0.12 m UCA keeps 860 MHz under half a wavelength
    geom = M.uca(5, 0.12)
    sc = sim.Scenario(emitters=[sim.Emitter(bearing_deg=75.0, f0_hz=300_000.0, snr_db=10.0,
                                            protocol="p25", system="BEE00-1A2", id="1234567")],
                      fc_hz=fc_hz, manifold=M.IdealManifold(geom), seed=9)
    s = sim.Simulator(sc, frames=6)
    eng = engine.Engine(geometry=s.geometry, manifold=s.manifold, positioning=s.positioning,
                        bandplan=bandplan)
    by = {}
    for e in s.events():
        by.setdefault(e.n0, []).append(e)
    recs = []
    for fr in s.frames():
        recs += eng.process(fr, events=by.get(fr.header.stream_sample0, []))
    return recs


def test_a_p25_outbound_bearing_is_tagged_and_never_fused():
    recs = _p25_run(860.0e6, BandPlan.for_region("us"))
    assert recs and all(r.path_tag == "repeated" and not r.fusable for r in recs)
    assert "base-transmit" in recs[-1].quality["path_evidence"]


def test_a_p25_inbound_bearing_is_a_handset():
    recs = _p25_run(815.0e6, BandPlan.for_region("us"))
    assert recs and all(r.path_tag == "direct" and r.fusable for r in recs)


def test_without_a_band_plan_p25_stays_unknown_as_before():
    recs = _p25_run(860.0e6, None)
    assert recs and all(r.path_tag == "unknown" for r in recs)
