"""Array profiles (schema v2, plan §11.4.1): presets validate, files round
trip, the validation rules the literature made catch what they should, and
the Kraken's existing settings map onto a profile without changing its
geometry."""

from __future__ import annotations

import json

import numpy as np
import pytest

from atkdf import arrayprofile as AP
from atkdf import manifold as M


@pytest.mark.parametrize("pid", AP.preset_ids())
def test_every_preset_validates_and_builds(pid):
    p = AP.preset(pid)
    errors, _warnings = p.problems()
    assert errors == []
    g = p.geometry()
    man = p.manifold()
    assert man.response(np.array([0.0, 90.0]), p.band_centre_hz).shape == (g.n, 2)
    assert p.summary()[0].startswith(pid)


def test_the_kraken_preset_is_atks_default_circle():
    g = AP.preset("kraken5_uca").geometry()
    ref = M.uca(5, 0.18)
    assert np.allclose(g.positions, ref.positions)
    assert AP.preset("kraken5_uca").rx_cal_persistence == "per_retune"


def test_legacy_settings_map_to_the_same_geometry_and_connector_map():
    cfg = {"array": "uca", "radius_m": 0.21, "channel_of_element": (4, 3, 2, 1, 0)}
    p = AP.from_legacy_cfg(cfg)
    g = p.geometry()
    ref = M.uca(5, 0.21, channel_of_element=(4, 3, 2, 1, 0))
    assert np.allclose(g.positions, ref.positions)
    assert g.channel_of_element == ref.channel_of_element
    p2 = AP.from_legacy_cfg({"array": "ula", "spacing_m": 0.25})
    assert np.allclose(p2.geometry().positions, M.ula(5, 0.25).positions)


def test_profiles_round_trip_through_json(tmp_path):
    for pid in AP.preset_ids():
        p = AP.preset(pid)
        path = p.save(tmp_path / f"{pid}.json")
        q = AP.ArrayProfile.load(path)
        assert q.to_dict() == p.to_dict()
        assert np.allclose(q.geometry().positions3, p.geometry().positions3)


def test_write_presets_never_overwrites_unless_asked(tmp_path):
    written = AP.write_presets(tmp_path)
    assert len(written) == len(AP.preset_ids())
    f = tmp_path / "kraken5_uca.json"
    d = json.loads(f.read_text())
    d["notes"] = "edited by Bill"
    f.write_text(json.dumps(d))
    assert AP.write_presets(tmp_path) == []
    assert json.loads(f.read_text())["notes"] == "edited by Bill"
    AP.write_presets(tmp_path, overwrite=True)
    assert json.loads(f.read_text())["notes"] != "edited by Bill"


def test_load_dir_reports_a_broken_file_and_keeps_the_rest(tmp_path):
    AP.write_presets(tmp_path)
    (tmp_path / "broken.json").write_text("{not json")
    bad_rules = AP.preset("kraken5_uca").to_dict()
    bad_rules["id"] = "bad_rules"
    bad_rules["rx_cal_persistence"] = "across_power_cycles"     # independent LOs cannot
    (tmp_path / "bad_rules.json").write_text(json.dumps(bad_rules))
    good, bad = AP.load_dir(tmp_path)
    assert set(good) == set(AP.preset_ids())
    assert set(bad) == {"broken.json", "bad_rules.json"}
    assert "per_retune" in bad["bad_rules.json"]


def _errs(p):
    return " | ".join(p.problems()[0])


def test_validation_rules_from_the_literature():
    # independent LOs: a receiver table cannot outlive a retune (lit2 §3.18)
    p = AP.preset("kraken5_uca")
    p.rx_cal_persistence = "across_power_cycles"
    assert "per_retune" in _errs(p)
    # T4 IS a shared LO
    p = AP.preset("kraken5_uca")
    p.timing["tier"] = "T4"
    assert "T4" in _errs(p)
    # digital: one channel per element
    p = AP.preset("x310_twinrx_uca4")
    p.rf_chains = 3
    assert "one channel per element" in _errs(p)
    # switched: every element must be visited
    p = AP.preset("b210_switched_uca8c")
    p.switching["schedule"] = [[0], [1], [2]]
    assert "never visits" in _errs(p)
    # analog: halves must not overlap
    p = AP.preset("phaser_cn0566_xband")
    p.steering["halves"] = [[0, 1, 2, 3], [3, 4, 5, 6]]
    assert "overlap" in _errs(p)
    # hybrid: subarrays must not overlap and must match the channel count
    p = AP.preset("hybrid_ura4x4_4sub_v800")
    p.subarrays = [{"members": [0, 1]}, {"members": [1, 2]}]
    e = _errs(p)
    assert "overlap" in e and "channels" in e
    # ids are file-safe
    p = AP.preset("kraken5_uca")
    p.id = "bad id/with slash"
    assert "id must be" in _errs(p)


def test_lo_share_band_warning_for_n3xx_at_vhf():
    p = AP.preset("n321x4_ura2x4_v800")
    p.band_mhz = (150.0, 174.0)
    p.elements["spacing_m"] = 0.86
    _e, w = p.problems()
    assert any("share its LO only over 450" in x for x in w)


def test_capabilities_say_what_each_array_can_and_cannot_do():
    sw = AP.preset("b210_switched_uca8c").capabilities()
    assert sw["single_source_only"] and sw["max_sources"] == 1
    panel = AP.preset("n321x4_ura2x4_v800").capabilities()
    assert panel["elevation"] and panel["max_sources"] == 7 and panel["tdoa"]
    row = AP.preset("x310_twinrx_panel4_v800").capabilities()
    assert not row["elevation"]
    kr = AP.preset("kraken5_uca").capabilities()
    assert not kr["tdoa"] and kr["max_sources"] == 4
    ph = AP.preset("phaser_cn0566_xband").capabilities()
    assert ph["estimators"] == ("beamscan", "monopulse")


def test_defaults_are_conservative_when_persistence_is_not_given():
    d = AP.preset("x310_twinrx_uca4").to_dict()
    d.pop("rx_cal_persistence")
    assert AP.ArrayProfile.from_dict(d).rx_cal_persistence == "per_power_cycle"
    d["lo_sharing"] = "independent"
    assert AP.ArrayProfile.from_dict(d).rx_cal_persistence == "per_retune"


def test_an_analog_array_without_a_field_of_view_gets_plus_minus_60():
    p = AP.preset("phaser_cn0566_xband")
    p.field_of_view_deg = {}
    g = p.geometry()
    assert g.fov_az_deg == (-60.0, 60.0)


def test_a_hybrid_profile_must_set_rf_chains():
    p = AP.preset("hybrid_ura4x4_4sub_v800")
    p.rf_chains = None
    assert "must set rf_chains" in _errs(p)
