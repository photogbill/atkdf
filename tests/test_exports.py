"""Live sharing (`atkdf.exports`): the picture, the policy, KML, GeoJSON,
CoT and the record feed. Everything here is built by hand or from a small
registry; nothing opens a socket."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import numpy as np

from atkdf import exports as ex
from atkdf import geodesy as geo
from atkdf.records import BearingRecord
from atkdf.registry import Registry
from atkdf.roster import Roster

T0 = 1_790_000_000.0
TRUTH = (38.860, -77.050)
RX_A = (38.860, -77.100)        # west of the truth
RX_B = (38.810, -77.050)        # south of the truth
FLEET = {"dmr:1:473": {"name": "Rodriguez", "role": "Medic 2", "team": "Alpha",
                       "priority": 2}}


def _rec(rx, brg_abs, *, t=T0, key="", track_id="", sigma=2.5, path_tag="direct",
         fc_hz=462.5625e6, site_id="van-1"):
    g = np.arange(360.0)
    d = (g - (brg_abs % 360) + 180) % 360 - 180
    lik = np.exp(-0.5 * (d / sigma) ** 2)
    return BearingRecord(t_utc=t, theta_rel_deg=brg_abs % 360, sigma_deg=sigma,
                         estimator="music", heading_deg=0.0, theta_abs_deg=brg_abs % 360,
                         likelihood=lik / lik.sum(), lat=rx[0], lon=rx[1],
                         path_tag=path_tag, emitter_key=key, track_id=track_id,
                         fc_hz=fc_hz, site_id=site_id)


def _registry(now=T0 + 60.0):
    """Rodriguez located from two sites; an unnamed track heard once; a
    repeater-only ID; a silent one heard an hour ago."""
    reg = Registry(site_id="van-1", fleet_map=FLEET)
    for rx, t in ((RX_A, T0), (RX_B, T0 + 20.0)):
        reg.add(_rec(rx, geo.bearing_between(*rx, *TRUTH), t=t, key="dmr:1:473"))
    reg.add(_rec(RX_A, 45.0, t=T0 + 30.0, track_id="t7", fc_hz=155.160e6))
    reg.add(_rec(RX_A, 200.0, t=T0 + 40.0, key="dmr:1:900", path_tag="repeated"))
    reg.add(_rec(RX_B, 300.0, t=T0 - 3600.0, key="dmr:1:12"))
    return reg, now


def _picture(**kw):
    reg, now = _registry()
    veh = ex.Vehicle(lat=RX_B[0], lon=RX_B[1], t_utc=now, heading_deg=10.0,
                     site_id="van-1", callsign="ATK van 1", status="parked on 462.5625")
    kw.setdefault("vehicle", veh)
    return ex.picture_from(reg, now=now, **kw), now


def _by_key(pic):
    return {t.key: t for t in pic.tracks}


# --- the picture -----------------------------------------------------------

def test_the_picture_names_the_person_and_draws_an_honest_ellipse():
    pic, _now = _picture()
    t = _by_key(pic)["dmr:1:473"]
    assert t.label == "Rodriguez (Medic 2)" and t.team == "Alpha" and t.priority == 2
    assert t.state == "located" and t.located
    assert t.semi_major_m > 0 and t.semi_minor_m > 0
    assert geo.distance_between(t.lat, t.lon, *TRUTH) < t.semi_major_m
    assert "Search area about" in t.sentence and "Last heard" in t.sentence
    assert pic.tracks[0].key == "dmr:1:473"            # priority sorts first


def test_one_bearing_is_bearing_only_and_a_repeater_is_not_a_place():
    pic, _now = _picture()
    by = _by_key(pic)
    assert by["track:t7"].state == "bearing_only"
    assert by["track:t7"].label.startswith("unidentified")
    assert by["track:t7"].last_lob is not None and not by["track:t7"].located
    assert by["dmr:1:900"].state == "heard_via_repeater"
    assert "repeater" in by["dmr:1:900"].sentence
    assert not [l for l in pic.lobs if l.key == "dmr:1:900"]   # not fusable, not drawn


def test_silence_from_the_roster_and_box_membership():
    roster = Roster(fleet_map=FLEET, default_silence_s=600.0)
    reg, now = _registry()
    for key, em in reg.emitters.items():
        roster.observe(key, em.last_heard)
    roster.check(now)
    pic = ex.picture_from(reg, now=now, roster=roster, in_box=lambda k: k == "dmr:1:473")
    by = _by_key(pic)
    assert by["dmr:1:12"].state == "silent"
    assert "Silent past the 10-minute threshold" in by["dmr:1:12"].sentence
    assert by["dmr:1:473"].in_box and not by["track:t7"].in_box


def test_an_aged_posterior_stands_in_when_there_is_no_fix():
    class Post:
        def fix(self):
            return {"lat": 38.85, "lon": -77.06, "sigma_m": 400.0}
    reg = Registry(site_id="s")
    reg.add(_rec(RX_A, 90.0, key="dmr:1:5"))
    pic = ex.picture_from(reg, now=T0 + 5, posteriors={"dmr:1:5": Post()}, containment=0.9)
    t = pic.tracks[0]
    assert t.state == "bearing_only"                   # one bearing: still only a direction
    assert abs(t.semi_major_m - 400.0 * ex._k(0.9)) < 1e-6
    assert t.sigma_m == 400.0


def test_label_for_never_guesses_a_name():
    assert ex.label_for("dmr:1:473", "Rodriguez", "Medic 2") == "Rodriguez (Medic 2)"
    assert ex.label_for("sig:ab12", protocol="", fc_hz=155.16e6,
                        signature={"modulation_hint": "NFM"}) == "NFM 155.160 MHz"
    assert ex.label_for("freq:462562500", fc_hz=462.5625e6) == "unknown 462.562 MHz"
    assert ex.label_for("track:9", protocol="dmr") == "unidentified dmr"
    assert ex.label_for("p25:1:77") == "p25:1:77"


def test_plain_sentence_variants():
    t = ex.Track(key="k", label="k", state="located", lat=1, lon=1, semi_major_m=250,
                 semi_minor_m=100, last_heard=T0 - 90, trend="tightening",
                 motion_state="on_foot")
    s = ex.plain_sentence(t, T0)
    assert s.startswith("Last heard 1 minute ago.")
    assert "about 500 m across, tightening." in s and "Probably on foot." in s
    assert ex.plain_sentence(ex.Track(key="k", label="k"), T0) == "Nothing heard yet."


# --- fade and shapes -------------------------------------------------------

def test_fade_is_opaque_on_arrival_and_rests_on_the_floor():
    assert ex.fade_alpha(0, 120) == 1.0
    assert abs(ex.fade_alpha(60, 120) - 0.5) < 1e-9
    assert ex.fade_alpha(120, 120) == 0.15 and ex.fade_alpha(9999, 120) == 0.15
    assert ex.fade_alpha(500, 0) == 1.0                # fade off


def test_ellipse_ring_has_the_stated_axes_and_closes():
    ring = ex.ellipse_ring(38.85, -77.05, 800.0, 200.0, 30.0, n=36)
    assert len(ring) == 37 and ring[0] == ring[-1]
    # the first point is on the major axis, along the orientation
    d = geo.inverse(38.85, -77.05, *ring[0])
    assert abs(d.distance_m - 800.0) < 1.0 and abs(d.bearing_deg - 30.0) < 0.5
    # a quarter of the way round is the minor axis
    d2 = geo.inverse(38.85, -77.05, *ring[9])
    assert abs(d2.distance_m - 200.0) < 1.0


def test_wedge_ring_is_a_closed_fan_from_the_receiver():
    lob = ex.Lob(T0, 38.85, -77.05, bearing_deg=90.0, sigma_deg=3.0, range_m=3000.0)
    ring = ex.wedge_ring(lob, k_sigma=2.0, n=12)
    assert ring[0] == ring[-1] == (38.85, -77.05)
    far = [geo.inverse(38.85, -77.05, *p) for p in ring[1:-1]]
    assert all(abs(f.distance_m - 3000.0) < 1.0 for f in far)
    assert abs(far[0].bearing_deg - 84.0) < 0.1 and abs(far[-1].bearing_deg - 96.0) < 0.1


def test_kml_colour_is_aabbggrr():
    assert ex._kml_colour("112233", 1.0) == "ff332211"
    assert ex._kml_colour("112233", 0.0) == "00332211"


# --- KML -------------------------------------------------------------------

NS = {"k": "http://www.opengis.net/kml/2.2"}


def _names(root, folder):
    f = [x for x in root.iterfind(".//k:Folder", NS) if x.find("k:name", NS).text == folder]
    assert f, folder
    return [p.find("k:name", NS).text for p in f[0].iterfind("k:Placemark", NS)]


def test_kml_is_well_formed_and_person_first():
    pic, now = _picture()
    text = ex.to_kml(pic, now=now)
    root = ET.fromstring(text)
    people = _names(root, "People & emitters")
    assert people[0] == "Rodriguez (Medic 2)"
    assert any(n.startswith("unidentified") for n in people)
    assert not any("dmr:1:900" in n for n in people)       # repeater-only: nothing to draw
    assert _names(root, "Own vehicle") == ["ATK van 1"]
    lobs = _names(root, "Lines of bearing")
    assert lobs.count("Rodriguez (Medic 2)") == 2          # both sites' bearings
    # every bearing and every heard track carries a TimeStamp for the slider
    for pm in root.iterfind(".//k:Placemark", NS):
        if pm.find("k:LineString", NS) is not None:
            assert pm.find("k:TimeStamp/k:when", NS) is not None
    # the balloon leads with the sentence, the RF under the rule
    desc = root.find(".//k:Folder/k:Placemark/k:description", NS)
    assert desc is not None
    body = next(p for p in root.iterfind(".//k:Placemark", NS)
            if p.find("k:name", NS).text == "Rodriguez (Medic 2)"
            and p.find("k:MultiGeometry", NS) is not None)
    html = body.find("k:description", NS).text
    assert html.index("Search area") < html.index("<hr/>") < html.index("Protocol")
    assert "<Polygon>" in text and "<coordinates>" in text


def test_kml_fades_old_bearings_and_drops_them_after_keep():
    pic, now = _picture(fade_s=100.0, keep_s=1000.0)
    old = ex.Lob(now - 1500.0, *RX_A, 10.0, 3.0, key="dmr:1:473", label="old")
    mid = ex.Lob(now - 50.0, *RX_A, 20.0, 3.0, key="dmr:1:473", label="mid")
    pic.lobs += [old, mid]
    text = ex.to_kml(pic, now=now)
    assert "<name>old</name>" not in text
    i = text.index("<name>mid</name>")
    colour = text[i:].split("<color>")[1].split("</color>")[0]
    assert colour[:2] == "80"                               # alpha 0.5 at half the fade
    gj = json.loads(ex.to_geojson(pic, now=now))
    labels = [f["properties"].get("label") for f in gj["features"]
              if f["properties"]["kind"] == "lob"]
    assert "mid" in labels and "old" not in labels


def test_a_silent_track_is_flagged_in_its_name():
    pic, now = _picture()
    t = _by_key(pic)["dmr:1:12"]
    t.state = "silent"
    text = ex.to_kml(pic, now=now)
    assert "⚠ dmr:1:12 — silent 1 h 1 min" in text


def test_boxes_and_references_get_their_own_folders():
    pic, now = _picture()
    pic.boxes = [ex.Box("Sector 2", [(38.80, -77.10), (38.80, -77.00), (38.90, -77.00)])]
    pic.references = [ex.RefMark("nws:kec83", 38.9, -77.1, name="NOAA 162.550", verified=True)]
    root = ET.fromstring(ex.to_kml(pic, now=now))
    assert _names(root, "Boxes") == ["Sector 2"]
    assert _names(root, "References") == ["✓ NOAA 162.550"]
    ring = root.find(".//k:Folder[k:name='Boxes']//k:coordinates", NS).text.split()
    assert len(ring) == 4 and ring[0] == ring[-1]            # closed for the client


def test_network_link_document():
    text = ex.network_link_kml("http://10.0.0.5:8765/live.kml", refresh_s=2.0)
    root = ET.fromstring(text)
    link = root.find(".//k:NetworkLink/k:Link", NS)
    assert link.find("k:href", NS).text == "http://10.0.0.5:8765/live.kml"
    assert link.find("k:refreshMode", NS).text == "onInterval"
    assert link.find("k:refreshInterval", NS).text == "2"


def test_text_is_escaped():
    pic = ex.Picture(t_utc=T0, title="A <b> & B")
    pic.tracks = [ex.Track(key="x<y", label="Tom & Jerry <3", state="located", lat=38.8,
                           lon=-77.0, semi_major_m=100, semi_minor_m=50, last_heard=T0)]
    root = ET.fromstring(ex.to_kml(pic))                  # would raise if unescaped
    assert root.find(".//k:Document/k:name", NS).text == "A <b> & B"


# --- GeoJSON ---------------------------------------------------------------

def test_geojson_carries_the_numbers():
    pic, now = _picture()
    gj = json.loads(ex.to_geojson(pic, now=now))
    kinds = {f["properties"]["kind"] for f in gj["features"]}
    assert kinds == {"vehicle", "lob", "track"}
    tr = next(f for f in gj["features"] if f["properties"].get("key") == "dmr:1:473"
          and f["properties"]["kind"] == "track")
    assert tr["geometry"]["type"] == "GeometryCollection"
    assert tr["properties"]["semi_major_m"] > 0 and tr["properties"]["n_bearings"] == 2
    wedge = next(f for f in gj["features"] if f["properties"].get("key") == "track:t7"
             and f["properties"]["kind"] == "track")
    assert wedge["geometry"]["type"] == "Polygon"
    assert gj["properties"]["policy"] == "everything"


# --- policy ----------------------------------------------------------------

def test_policies_decide_what_leaves():
    pic, _now = _picture(in_box=lambda k: k == "track:t7")
    assert {t.key for t in ex.apply_policy(pic, "box_only").tracks} == {"track:t7"}
    assert {t.key for t in ex.apply_policy(pic, "selected").tracks} == {"dmr:1:473"}
    sel = ex.apply_policy(pic, "selected", selected=["dmr:1:12"])
    assert [t.key for t in sel.tracks] == ["dmr:1:12"]
    assert all(l.key == "dmr:1:12" for l in sel.lobs)
    try:
        ex.apply_policy(pic, "everybody")
    except ValueError as e:
        assert "policy" in str(e)
    else:
        raise AssertionError("bad policy accepted")


def test_fixes_only_strips_every_identity_from_every_format():
    pic, now = _picture()
    pic.references = [ex.RefMark("nws:kec83", 38.9, -77.1, name="NOAA")]
    pic.notices = ["Rodriguez overdue"]
    anon = ex.apply_policy(pic, "fixes_only")
    assert len(anon.tracks) == len(pic.tracks)
    assert anon.tracks[0].located and anon.tracks[0].semi_major_m == pic.tracks[0].semi_major_m
    for text in (ex.to_kml(anon, now=now), ex.to_geojson(anon, now=now),
                 "".join(ex.to_cot(anon, now=now))):
        for secret in ("Rodriguez", "Medic", "Alpha", "dmr:1:473", "dmr_1_473", "track:t7",
                       "track_t7", "van-1", "van_1", "ATK van 1", "NOAA", "overdue",
                       "462.5625", "parked on"):
            assert secret not in text, (secret, text[:200])
        assert "emitter 1" in text


# --- CoT -------------------------------------------------------------------

def test_cot_events_parse_and_keep_stable_uids():
    pic, now = _picture()
    events = ex.to_cot(pic, now=now)
    roots = [ET.fromstring(e) for e in events]
    assert all(r.tag == "event" for r in roots)
    uids = [r.get("uid") for r in roots]
    assert len(uids) == len(set(uids))
    assert "atk.van-1.vehicle" in uids
    assert "atk.van-1.dmr_1_473" in uids
    assert "atk.van-1.dmr_1_900" not in uids              # repeater-only: no place
    # a second call for the same picture yields the same uids (the radio moves,
    # the object does not multiply)
    assert uids == [ET.fromstring(e).get("uid") for e in ex.to_cot(pic, now=now)]
    for r in roots:
        assert r.get("stale") > r.get("time")
        assert r.find("point").get("lat")


def test_cot_callsign_is_the_person_and_silent_is_loud():
    pic, now = _picture()
    _by_key(pic)["dmr:1:12"].state = "silent"
    roots = [ET.fromstring(e) for e in ex.to_cot(pic, now=now)]
    by_uid = {r.get("uid"): r for r in roots}
    rod = by_uid["atk.van-1.dmr_1_473"]
    assert rod.find("detail/contact").get("callsign") == "Rodriguez (Medic 2)"
    assert rod.get("type") == ex.COT_TYPES["track"]
    assert float(rod.find("point").get("ce")) > 0
    assert "Search area" in rod.find("detail/remarks").text
    sil = by_uid["atk.van-1.dmr_1_12"]
    assert sil.find("detail/contact").get("callsign").endswith("— SILENT")
    assert sil.find("detail/color").get("argb") == "-65536"
    unk = by_uid["atk.van-1.track_t7"]
    assert unk.get("type") == ex.COT_TYPES["unknown"]


def test_cot_lobs_are_range_and_bearing_lines_capped_per_key():
    pic, now = _picture()
    for i in range(6):
        pic.lobs.append(ex.Lob(now - i, *RX_A, 10.0 + i, 3.0, key="dmr:1:473", label="R"))
    roots = [ET.fromstring(e) for e in ex.to_cot(pic, now=now, max_lobs_per_key=2)]
    lobs = [r for r in roots if r.get("type") == ex.COT_TYPES["lob"]]
    per_key = {}
    for r in lobs:
        per_key[r.get("uid").rsplit(".lob", 1)[0]] = per_key.get(r.get("uid").rsplit(".lob", 1)[0], 0) + 1
    assert max(per_key.values()) == 2
    r = lobs[0]
    assert r.find("detail/range").get("value") == "5000"
    assert r.find("detail/bearing").get("value") is not None
    assert "LOB" in r.find("detail/contact").get("callsign")


# --- the record feed -------------------------------------------------------

def test_records_since_returns_complete_lines_and_an_offset(tmp_path):
    p = tmp_path / "bearings.jsonl"
    p.write_text('{"a":1}\n{"b":2}\n{"c":', encoding="utf-8")
    text, off = ex.records_since(p, 0)
    assert text == '{"a":1}\n{"b":2}\n' and off == len('{"a":1}\n{"b":2}\n')
    text2, off2 = ex.records_since(p, off)
    assert text2 == "" and off2 == off                       # partial line waits
    with open(p, "a", encoding="utf-8") as f:
        f.write('3}\n')
    text3, off3 = ex.records_since(p, off2)
    assert text3 == '{"c":3}\n' and off3 == p.stat().st_size
    assert ex.records_since(tmp_path / "missing.jsonl", 7) == ("", 7)


def test_the_feed_round_trips_through_a_second_registry(tmp_path):
    log = tmp_path / "van1.jsonl"
    reg = Registry(site_id="van-1", log_path=log)
    for rx, t in ((RX_A, T0), (RX_B, T0 + 20.0)):
        reg.add(_rec(rx, geo.bearing_between(*rx, *TRUTH), t=t, key="dmr:1:473"))
    reg.close()
    text, _ = ex.records_since(log, 0)
    other = tmp_path / "from_van1.jsonl"
    other.write_text(text, encoding="utf-8")
    merged = Registry.from_logs([other], site_id="van-2", fleet_map=FLEET)
    pic = ex.picture_from(merged, now=T0 + 60)
    assert pic.tracks[0].located and pic.tracks[0].label == "Rodriguez (Medic 2)"
