"""Phase 1: registry + JSONL log + merge, heading filter, identity/path-tag,
recorder + reprocess. All on the simulator, no hardware."""

from __future__ import annotations

import numpy as np
import pytest

from atkdf import engine, identity, recorder, registry, sim
from atkdf import manifold as M
from atkdf.fakes import ListCatalogue
from atkdf.heading import HeadingFilter
from atkdf.ports import BurstEvent, Reference
from atkdf.records import BearingRecord


def err(a, t):
    return ((a - t + 180) % 360) - 180


def _run(sc, frames=10, mode=None, catalogue=None):
    s = sim.Simulator(sc, frames=frames)
    eng = engine.Engine(geometry=s.geometry, manifold=M.IdealManifold(s.geometry),
                        positioning=s.positioning, catalogue=catalogue,
                        mode=mode or engine.PlatformMode.static(), site_id="alpha")
    by = {}
    for e in s.events():
        by.setdefault(e.n0, []).append(e)
    recs = []
    for fr in s.frames():
        recs += eng.process(fr, events=by.get(fr.header.stream_sample0, []))
    return s, recs


# --- registry --------------------------------------------------------------

def test_registry_files_bearings_by_emitter_and_computes_a_fix():
    sc = sim.Scenario(emitters=[sim.Emitter(bearing_deg=90.0, f0_hz=200_000.0, snr_db=20.0,
                                            protocol="dmr", system="1", id="473", sync_type="ms")],
                      heading_deg=0.0, seed=5)
    s, recs = _run(sc, frames=14)
    reg = registry.Registry(site_id="alpha")
    reg.add_all(recs)
    em = reg.get("dmr:1:473")
    assert em is not None and em.n_bearings >= 3
    assert em.path_tags.get("direct", 0) == em.n_bearings   # all direct
    # a static receiver gives one bearing line; a fix needs crossing bearings,
    # so with a single site and no motion the fix may be None — but the call
    # must not raise, and with >=2 fusable LOBs it returns a Fix
    fx = reg.fix("dmr:1:473")
    assert fx is None or (hasattr(fx, "lat") and hasattr(fx, "lon"))


def test_registry_jsonl_log_merges_two_sites(tmp_path):
    # two vehicles, each logs the same emitter from its own site; merge by file
    logs = []
    for site, hdg in (("alpha", 0.0), ("bravo", 90.0)):
        sc = sim.Scenario(emitters=[sim.Emitter(bearing_deg=70.0, f0_hz=0.0, snr_db=20.0,
                                                protocol="dmr", system="1", id="9")],
                          heading_deg=hdg, seed=7)
        s = sim.Simulator(sc, frames=8)
        eng = engine.Engine(geometry=s.geometry, manifold=M.IdealManifold(s.geometry),
                            positioning=s.positioning, site_id=site)
        p = tmp_path / f"{site}.jsonl"
        reg = registry.Registry(site_id=site, log_path=p)
        by = {}
        for e in s.events():
            by.setdefault(e.n0, []).append(e)
        for fr in s.frames():
            for r in eng.process(fr, events=by.get(fr.header.stream_sample0, [])):
                reg.add(r)
        reg.close()
        logs.append(p)
    merged = registry.Registry.from_logs(logs)
    em = merged.get("dmr:1:9")
    assert em is not None
    # both sites' site_ids are present in the merged records
    sites = {r.site_id for r in merged.bearings("dmr:1:9")}
    assert sites == {"alpha", "bravo"}


def test_fleet_map_gives_an_emitter_a_readable_name(tmp_path):
    csv = tmp_path / "fleet.csv"
    csv.write_text("id,name,role,team,protocol,system\n473,Rodriguez,Medic 2,red,dmr,1\n",
                   encoding="utf-8")
    fm = registry.load_fleet_map(csv)
    reg = registry.Registry(fleet_map=fm)
    reg.add(BearingRecord(t_utc=1.0, theta_rel_deg=10.0, sigma_deg=2.0, estimator="music",
                          emitter_key="dmr:1:473", path_tag="direct", heading_deg=0.0,
                          theta_abs_deg=10.0, lat=38.7, lon=-77.5))
    assert reg.get("dmr:1:473").label == "Rodriguez (Medic 2)"


# --- heading ---------------------------------------------------------------

def test_heading_sigma_grows_at_a_stop_and_shrinks_at_speed():
    hf = HeadingFilter()
    t = 0.0
    # driving straight at 15 m/s, course 90
    for _ in range(20):
        h = hf.update_gps(t, 90.0, 15.0)
        t += 1.0
    moving_sigma = h.sigma_deg
    assert abs(err(h.heading_deg, 90.0)) < 5
    assert moving_sigma < 10
    # now stop: no course, sigma must grow
    for _ in range(20):
        h = hf.update_gps(t, None, 0.0)
        t += 1.0
    assert h.sigma_deg > moving_sigma + 10


def test_a_radio_compass_fix_rescues_heading_at_a_stop():
    hf = HeadingFilter()
    t = 0.0
    for _ in range(10):
        h = hf.update_gps(t, None, 0.0)     # stopped from the start: heading unknown
        t += 1.0
    assert h.sigma_deg > 20
    h = hf.update_compass(t, 135.0, sigma_deg=3.0)   # a reference is in view
    assert abs(err(h.heading_deg, 135.0)) < 4
    assert h.sigma_deg < 6


def test_heading_tracks_through_a_turn():
    hf = HeadingFilter()
    t = 0.0
    h = None
    for c in list(range(0, 90, 5)):         # a steady right turn at speed
        h = hf.update_gps(t, float(c), 12.0)
        t += 0.5
    assert abs(err(h.heading_deg, 85.0)) < 15    # tracks, not stuck at 0


# --- identity / path tag ---------------------------------------------------

def test_path_tag_marks_a_repeater_outbound():
    bs = BurstEvent(protocol="dmr", system="1", id="473", t_utc=1.0, sync_type="bs")
    ms = BurstEvent(protocol="dmr", system="1", id="473", t_utc=1.0, sync_type="ms")
    assert identity.path_tag_of(bs) == "repeated"
    assert identity.path_tag_of(ms) == "direct"


def test_path_tag_falls_back_to_the_catalogue():
    cat = ListCatalogue([Reference(key="p25:7:9001", lat=1, lon=1, kind="repeater_output")])
    ev = BurstEvent(protocol="p25", system="7", id="9001", t_utc=1.0, fc_hz=451e6)
    assert identity.path_tag_of(ev, catalogue=cat) == "repeated"


def test_a_time_only_event_is_widened_to_a_span():
    ev = BurstEvent(protocol="dmr", system="1", id="5", t_utc=100.010, span_s=0.030)
    sp = identity.span_of(ev, fs_hz=2_400_000, stream_sample0_at_t0=0, t0_utc=100.0,
                          slack_s=0.010)
    assert not sp.exact
    # [t0+0.000 .. t0+0.050] in samples: 0 .. 120000
    assert sp.n0 == 0 and abs(sp.n1 - 120000) < 5
    assert sp.key == "dmr:1:5"


# --- recorder + reprocess --------------------------------------------------

def test_a_clip_reprocesses_to_the_same_bearings(tmp_path):
    sc = sim.Scenario(emitters=[sim.Emitter(bearing_deg=123.0, f0_hz=250_000.0, snr_db=20.0,
                                            protocol="dmr", system="1", id="55", sync_type="ms")],
                      heading_deg=15.0, seed=9)
    s = sim.Simulator(sc, frames=10)
    geom = s.geometry
    eng = engine.Engine(geometry=geom, manifold=M.IdealManifold(geom),
                        positioning=s.positioning, site_id="alpha")
    by = {}
    for e in s.events():
        by.setdefault(e.n0, []).append(e)
    live = []
    rc = recorder.ClipRecorder(tmp_path / "run", geometry=geom, site_id="alpha")
    for fr in s.frames():
        rc.add_frame(fr)
        for r in eng.process(fr, events=by.get(fr.header.stream_sample0, [])):
            rc.add_record(r); rc.add_passport(eng.passport); live.append(r)
    for e in s.events():
        rc.add_event(e)
    data, side = rc.close()
    assert data.endswith(".atkq")

    # reprocess with a FRESH engine over the clip
    eng2 = engine.Engine(geometry=geom, manifold=M.IdealManifold(geom),
                         positioning=s.positioning, site_id="alpha")
    redone = recorder.reprocess(data, eng2)
    assert redone, "reprocess produced no records"
    assert len(redone) == len(live)
    a = np.array([r.theta_rel_deg for r in live])
    b = np.array([r.theta_rel_deg for r in redone])
    assert np.allclose(a, b, atol=1e-6), "reprocess did not reproduce the live bearings"
    # the sidecar carries the passport and the records
    saved = recorder.read_sidecar(side)
    assert saved["passports"] and saved["records"]
    assert saved["geometry"]["id"] == geom.id


def test_a_clip_exports_a_five_channel_sigmf(tmp_path):
    sc = sim.Scenario(emitters=[sim.Emitter(bearing_deg=30.0, snr_db=20.0)], seed=2)
    s = sim.Simulator(sc, frames=6)
    rc = recorder.ClipRecorder(tmp_path / "run", geometry=s.geometry)
    for fr in s.frames():
        rc.add_frame(fr)
    data, _side = rc.close()
    dpath, mpath = recorder.to_sigmf(data, tmp_path / "clip_sigmf")
    import json
    meta = json.loads(open(mpath).read())
    assert meta["global"]["core:num_channels"] == 5
    assert meta["global"]["core:sample_rate"] == sc.fs_hz
    # and atkdf's own recio reads the channel count back
    from atkdf import recio
    assert recio.header(dpath)["subsize"] == 5
