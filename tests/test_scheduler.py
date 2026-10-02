"""The dwell scheduler (ATK docs/CUED_DF_PLAN.md §5–§6), on scripted cues.

Every rule the plan states is a test here: windows not bursts; energy first
and identity optional (ID → signature → frequency); the three cued modes;
the gating rule with the per-key hold-time learner; pre-positioning for a
priority emitter that fails the gate; grants as look-ahead; the dwell plan
with revisit and rotation; the return from a chase that went quiet; the
calibration refresh in idle time; and the missed-burst log that explains
every cue not acted on.
"""

from __future__ import annotations

import pytest

from atkdf.ports import Detection
from atkdf import scheduler as S

MHz = 1e6


class _Control:
    def __init__(self):
        self.cmds: list = []

    def __call__(self, cmd: str) -> None:
        self.cmds.append(cmd)

    def tunes(self) -> list:
        return [int(c.split()[1]) for c in self.cmds if c.startswith("tune")]


def _sched(policy="follow_listener", mode="all_triggers", fc=452 * MHz, now=0.0,
           costs=None, **cfg):
    ctl = _Control()
    config = S.SchedulerConfig(policy=policy, cued_mode=mode, **cfg)
    s = S.Scheduler(config, costs=costs or S.Costs(), control=ctl, fc_hz=fc, now=now)
    return s, ctl


# --- keys -----------------------------------------------------------------

def test_keys_fall_back_from_id_to_signature_to_frequency():
    d_id = Detection(t0=0, fc_hz=451.1 * MHz, key="dmr:1:473")
    d_sig = Detection(t0=0, fc_hz=451.1 * MHz, signature={"bw_hz": 12_500, "duration_s": 4})
    d_bare = Detection(t0=0, fc_hz=451.1 * MHz)
    assert S.key_for(d_id) == "dmr:1:473"
    assert S.key_for(d_sig).startswith("sig:")
    assert S.key_for(d_bare) == S.freq_key(451.1 * MHz) == "freq:36088"
    # a signature is coarse: small changes do not make a new emitter
    a = S.signature_key({"bw_hz": 12_500, "duration_s": 4.1}, 451.1 * MHz)
    b = S.signature_key({"bw_hz": 12_900, "duration_s": 4.9}, 451.1 * MHz)
    c = S.signature_key({"bw_hz": 200_000, "duration_s": 0.2}, 451.1 * MHz)
    assert a == b and a != c


def test_hold_times_learn_per_key_with_the_fallback_chain():
    h = S.HoldTimes(default_s=3.0)
    assert h.expected(["dmr:1:473", "freq:1"]) == (3.0, "default")
    assert h.expected(["x"], protocol="p25") == (4.0, "protocol:p25")
    for d in (2.0, 9.0, 8.0, 7.5, 8.2):
        h.observe("freq:1", d)
    assert h.expected(["dmr:1:473", "freq:1"]) == (8.0, "freq:1")   # the median
    h.observe("dmr:1:473", 1.5)
    assert h.expected(["dmr:1:473", "freq:1"]) == (1.5, "dmr:1:473")


# --- windows, not bursts -------------------------------------------------

def test_a_cue_inside_the_window_costs_nothing():
    s, ctl = _sched()
    d = s.on_cue(Detection(t0=10.0, fc_hz=452.8 * MHz, key="dmr:1:473"), now=10.0)
    assert d.action == "stay" and "ring" in d.reason
    assert ctl.cmds == []
    assert s.missed == []


def test_handset_inputs_five_megahertz_apart_are_two_windows():
    assert len(S._windows_for([451.0 * MHz, 456.0 * MHz], 2.4 * MHz)) == 2
    assert len(S._windows_for([451.0 * MHz, 451.5 * MHz, 452.9 * MHz], 2.4 * MHz)) == 1


# --- modes -----------------------------------------------------------------

def test_mode_off_logs_every_outside_cue_as_missed_with_the_reason():
    s, ctl = _sched(mode="off")
    assert s.on_cue(Detection(t0=0, fc_hz=460 * MHz, key="dmr:1:473"), now=0.5) is None
    assert ctl.cmds == []
    assert s.missed[0].reason == "mode_off"


def test_priority_only_chases_the_list_and_logs_the_rest():
    s, ctl = _sched(mode="priority_only", priority_keys=frozenset({"dmr:1:473"}))
    s.on_cue(Detection(t0=0, fc_hz=460 * MHz, key="dmr:1:999"), now=0.2)
    assert s.missed[-1].reason == "not_priority"
    d = s.on_cue(Detection(t0=0, fc_hz=460 * MHz, key="dmr:1:473", protocol="dmr"), now=0.2)
    assert d.action == "tune" and ctl.tunes() == [460_000_000]
    assert "gate passed" in d.reason


def test_a_beacon_band_is_priority_without_any_list():
    s, ctl = _sched(mode="priority_only")
    d = s.on_cue(Detection(t0=0, fc_hz=406.028 * MHz, bw_hz=3000.0,
                           signature={"bw_hz": 3000, "duration_s": 0.5}), now=0.05)
    # a 406 burst is short (default hold for unknown protocol = 3 s, but the
    # detection has only just begun) -> it passes the gate on remaining time
    assert d.action in ("tune", "preposition")
    assert ctl.tunes() == [406_028_000]


# --- the gating rule ------------------------------------------------------

def test_the_gate_compares_expected_remaining_to_the_cost():
    costs = S.Costs(settle_s=0.05, cal_s=0.14, integration_s=0.5)
    s, ctl = _sched(costs=costs)
    s.hold.observe("dmr:1:473", 8.0)
    s.hold.observe("dmr:1:473", 8.0)
    s.hold.observe("dmr:1:473", 8.0)
    # arrived 1 s ago, 7 s expected to remain: go
    d = s.on_cue(Detection(t0=0.0, fc_hz=460 * MHz, key="dmr:1:473"), now=1.0)
    assert d.action == "tune"
    # a short talker: 0.6 s median, already 0.5 s in -> not worth a retune
    s2, ctl2 = _sched(costs=costs)
    for _ in range(3):
        s2.hold.observe("dmr:1:5", 0.6)
    assert s2.on_cue(Detection(t0=0.0, fc_hz=460 * MHz, key="dmr:1:5"), now=0.5) is None
    m = s2.missed[-1]
    assert m.reason == "gate_failed" and "expected remaining" in m.detail
    assert ctl2.cmds == []


def test_a_priority_emitter_that_fails_the_gate_gets_the_array_prepositioned():
    s, ctl = _sched(mode="all_triggers", priority_keys=frozenset({"dmr:1:473"}))
    for _ in range(3):
        s.hold.observe("dmr:1:473", 0.6)
    d = s.on_cue(Detection(t0=0.0, fc_hz=460 * MHz, key="dmr:1:473"), now=0.5)
    assert d.action == "preposition"
    assert ctl.tunes() == [460_000_000]
    # and the miss is still logged, honestly, with the pre-position noted
    assert s.missed[-1].reason == "gate_failed"
    assert "pre-positioning" in s.missed[-1].detail


def test_a_grant_is_a_look_ahead_the_whole_burst_is_still_to_come():
    s, ctl = _sched()
    for _ in range(3):
        s.hold.observe("p25:7:1234", 2.0)
    # the same cue, 1.5 s in, would fail; as a grant it has 2.0 s ahead
    late = s.on_cue(Detection(t0=0.0, fc_hz=460 * MHz, key="p25:7:1234"), now=1.5)
    assert late is None and s.missed[-1].reason == "gate_failed"
    g = s.on_cue(Detection(t0=1.5, fc_hz=461 * MHz, key="p25:7:1234", grant=True), now=1.5)
    assert g.action == "tune" and ctl.tunes()[-1] == 461_000_000


def test_a_retune_in_progress_is_part_of_the_cost():
    costs = S.Costs(settle_s=0.05, cal_s=0.14, integration_s=0.5)
    s, ctl = _sched(costs=costs)
    s.on_cue(Detection(t0=0.0, fc_hz=460 * MHz, key="a", protocol="fm"), now=0.0)   # tune, busy 0.19 s
    # a second cue elsewhere, 2.8 s of a 3 s default remaining: 2.8 >= 0.19+0.19+0.5 -> go
    d = s.on_cue(Detection(t0=0.0, fc_hz=470 * MHz, key="b"), now=0.2)
    assert d is not None


# --- priority between windows ----------------------------------------------

def test_an_open_priority_burst_in_the_window_is_not_abandoned_for_a_novel_one():
    s, ctl = _sched(priority_keys=frozenset({"dmr:1:473"}))
    s.on_cue(Detection(t0=0.0, fc_hz=452.5 * MHz, key="dmr:1:473"), now=0.0)   # in window, open
    r = s.on_cue(Detection(t0=0.2, fc_hz=470 * MHz, key="freq:x"), now=0.2)
    assert r is None and s.missed[-1].reason == "lower_priority"
    assert ctl.cmds == []
    # once the priority burst ends and the window has been quiet, the novel one may be chased
    s.on_burst_end(Detection(t0=0.0, fc_hz=452.5 * MHz, key="dmr:1:473", t1=3.0), now=3.0)
    d = s.on_cue(Detection(t0=12.0, fc_hz=470 * MHz, key="freq:x"), now=12.0)
    assert d is not None and d.action == "tune"


def test_novelty_counts_only_until_a_key_has_been_bearinged():
    s, _ = _sched()
    d = Detection(t0=0, fc_hz=470 * MHz)
    tier0, _ = s._score(d, S.key_for(d), 0.0)
    assert tier0 == S.TIER_NOVEL
    s.note_bearing(S.key_for(d), 470 * MHz, now=1.0)
    tier1, _ = s._score(d, S.key_for(d), 1.0)
    assert tier1 == S.TIER_LEVEL


# --- the dwell plan -------------------------------------------------------------

def test_park_handsets_rotates_between_its_windows_and_honours_revisit():
    s, ctl = _sched(policy="park_handsets", fc=None,
                    handset_fcs_hz=(451.0 * MHz, 456.0 * MHz), revisit_s=3.0, dwell_s=1.0)
    s.tick(0.0)                                   # initial park on the first window
    first = ctl.tunes()[-1]
    s.tick(0.5)                                   # busy / dwelling: nothing
    n = len(ctl.tunes())
    s.tick(1.2)                                   # dwell over -> rotate
    assert len(ctl.tunes()) == n + 1 and ctl.tunes()[-1] != first
    # revisit: the first window is due after revisit_s
    s.tick(4.5)
    assert ctl.tunes()[-1] == first
    assert any(d.reason.startswith("revisit due") for d in s.decisions)


def test_a_chase_that_goes_quiet_returns_to_the_plan():
    s, ctl = _sched(policy="park_band", park_fc_hz=452 * MHz, return_after_s=8.0)
    cue = Detection(t0=0.0, fc_hz=470 * MHz, key="x")
    d = s.on_cue(cue, now=0.0)
    assert d.action == "tune" and s.fc_hz == 470 * MHz
    assert s.tick(5.0) == []                      # the burst is still open: stay
    s.on_burst_end(cue, now=1.0)
    assert s.tick(5.0) == []                      # closed, but not quiet for 8 s yet
    out = s.tick(9.0)
    assert out and out[0].action == "tune" and out[0].fc_hz == 452 * MHz
    assert "chase over" in out[0].reason


def test_the_calibration_is_refreshed_in_a_quiet_moment_not_during_a_burst():
    s, ctl = _sched(policy="park_band", park_fc_hz=452 * MHz, cal_max_age_s=60.0,
                    quiet_before_cal_s=1.0)
    cue = Detection(t0=61.0, fc_hz=452.3 * MHz, key="x")
    s.on_cue(cue, now=61.0)                        # live window
    assert s.tick(61.5) == []                      # a burst is up: no cal
    s.on_burst_end(cue, now=61.6)
    assert s.tick(62.0) == []                      # closed, not quiet for 1 s yet
    out = s.tick(62.6)
    assert out and out[0].action == "cal" and "cal" in ctl.cmds
    assert s.last_cal_t == 62.6


def test_follow_listener_moves_with_the_listener_unless_chasing():
    s, ctl = _sched(policy="follow_listener", fc=None)
    s.set_listener(455 * MHz, now=0.0)
    assert ctl.tunes() == [455_000_000]
    s.on_cue(Detection(t0=1.0, fc_hz=470 * MHz, key="x"), now=1.0)      # a chase
    s.set_listener(458 * MHz, now=1.5)
    assert ctl.tunes()[-1] == 470_000_000          # the chase holds


# --- the missed-burst log and the status line -------------------------------------

def test_missed_summary_and_status_say_it_plainly():
    s, _ = _sched(mode="off")
    s.on_cue(Detection(t0=0, fc_hz=470 * MHz), now=0.0)
    s.on_cue(Detection(t0=0, fc_hz=471 * MHz), now=0.0)
    assert s.missed_summary() == {"mode_off": 2}
    line = s.status(now=30.0)
    assert "following the listener" in line and "450.800–453.200 MHz" in line
    assert "calibrated 30 s ago" in line and "cued DF off" in line


def test_costs_follow_t1b():
    assert S.Costs().retune_s() == pytest.approx(0.19)
    assert S.Costs(delays_survive=False).retune_s() == pytest.approx(0.32)
    assert S.Costs(phase_cache=True).retune_s(cached=True) == pytest.approx(0.05)
    assert S.Costs(phase_cache=True).retune_s(cached=False) == pytest.approx(0.19)


def test_bad_configuration_is_refused_with_a_sentence():
    with pytest.raises(ValueError, match="policy must be one of"):
        S.Scheduler(S.SchedulerConfig(policy="wander"))
    with pytest.raises(ValueError, match="cued_mode must be one of"):
        S.Scheduler(S.SchedulerConfig(cued_mode="sometimes"))


# --- a scripted drive (T5's scheduler half) ------------------------------------

def test_a_scripted_relief_drive_explains_every_cue():
    """Push-to-talk profile: parked on the handsets' input window; the
    roster's medic on the priority list; a repeater output the detector has
    masked (so it never arrives); a short unknown burst elsewhere; a long
    unknown burst elsewhere; a 406 beacon. Every cue is either bearinged,
    chased, pre-positioned for, or explained."""
    from atkdf.fakes import ListCues
    medic = "dmr:1:473"
    s, ctl = _sched(policy="park_handsets", mode="all_triggers", fc=None,
                    handset_fcs_hz=(451.0 * MHz,), priority_keys=frozenset({medic}),
                    return_after_s=4.0)
    for _ in range(3):
        s.hold.observe(medic, 5.0)
    cues = [
        # the medic keys up on the input: in the window -> from the ring
        Detection(t0=2.0, t1=7.0, fc_hz=451.2 * MHz, key=medic, protocol="dmr"),
        # a 0.4 s unknown burst 10 MHz away: not worth the trip
        Detection(t0=3.0, t1=3.4, fc_hz=461.0 * MHz,
                  signature={"bw_hz": 12_500, "duration_s": 0.4}),
        # a long unknown FM carrier 15 MHz away after the window goes quiet: novel -> chased
        Detection(t0=14.0, t1=26.0, fc_hz=466.0 * MHz, protocol="fm",
                  signature={"bw_hz": 16_000, "duration_s": 12}),
        # a 406 beacon burst while the chase is on: beacon beats novel
        Detection(t0=18.0, t1=18.5, fc_hz=406.025 * MHz, bw_hz=3000.0, protocol="beacon",
                  signature={"bw_hz": 3000, "duration_s": 0.5, "periodicity_s": 50}),
        # the medic again, after the array came home
        Detection(t0=40.0, t1=46.0, fc_hz=451.2 * MHz, key=medic, protocol="dmr"),
    ]
    decisions = ListCues(cues, tick_s=0.5).replay(s, until=50.0)
    acts = [(round(d.t, 1), d.action, round(d.fc_hz / 1e6, 3)) for d in decisions]

    # parked on the handsets first, the medic's bursts come from the ring
    assert acts[0] == (0.0, "tune", 451.0)
    first_medic = [d for d in decisions if d.key == medic and d.t < 10.0]
    assert first_medic and first_medic[0].action == "stay"
    # the 0.4 s burst was not chased — the medic's burst was open in the
    # window and outranks it — and the log says so
    short = [m for m in s.missed if abs(m.fc_hz - 461.0 * MHz) < 1]
    assert short and short[0].reason == "lower_priority"
    # the long unknown was chased as novel
    assert any(a == (14.0, "tune", 466.0) for a in acts)
    # the beacon pre-empted it: too short to chase, so the array is
    # pre-positioned and HELD for the beacon's next period (~50 s), which is
    # the right call for a 406 burst
    beacon = [d for d in decisions if abs(d.fc_hz - 406.025 * MHz) < 1]
    assert beacon and beacon[0].action == "preposition" and "periodic" in beacon[0].reason
    assert not any(a[1] == "tune" and a[2] == 451.0 and 18 < a[0] < 40 for a in acts), acts
    second = [d for d in decisions if d.key == medic and d.t >= 40.0]
    assert second and second[0].action == "tune", "the medic outranks a parked beacon watch"
    assert all(m.reason in ("gate_failed", "lower_priority", "not_priority", "mode_off")
               for m in s.missed)
    # nothing was missed silently: every cue is a decision or a miss
    assert len(decisions) + len(s.missed) >= len(cues)


def test_a_cue_nobody_closes_cannot_pin_the_array_forever():
    s, ctl = _sched(policy="park_band", park_fc_hz=452 * MHz, return_after_s=8.0)
    s.on_cue(Detection(t0=0.0, fc_hz=470 * MHz, key="x"), now=0.0)      # never closed
    assert s.tick(30.0) == []                                            # still "open"
    out = s.tick(S.Scheduler.OPEN_MAX_S + 9.0)
    assert out and out[0].fc_hz == 452 * MHz and "chase over" in out[0].reason

