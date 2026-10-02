"""The dwell scheduler: where the array is tuned, and why (cued-DF plan §6).

One Kraken hears 2.4 MHz at a time. The listening receiver hears much more
and produces DETECTIONS (`ports.Detection`); this module decides, for each
one, whether the array stays where it is (the burst is already in the
window and the ring has it), retunes to it (the burst is worth the cost),
pre-positions for its next key-up, or lets it go — and in every case says
why, in a log the engineer can read after the drive.

Design rules carried from the plan, stated here so the code is checkable
against them:

  * **Energy first, identity optional** (§0A). Every per-emitter mechanism
    is keyed by the decoder's key when there is one, a signature key when
    there is not, and the frequency when there is neither. The code never
    asks which kind of key it has.
  * **Windows, not bursts** (§6.1). The decision is which 2.4 MHz window to
    sit in. A burst inside the current window costs nothing: the engine
    computes it from the ring, energy-clocked (§7).
  * **The gating rule** (§6.4): retune only if the burst is expected to
    outlast the cost of getting there plus the integration. Expected
    remaining comes from the per-key hold-time learner. A burst that fails
    the gate is not chased — but a priority emitter gets the array
    pre-positioned for its next key-up.
  * **Costs are measured, not assumed** (§13). `Costs` is filled from T1b;
    the defaults are atkdaq's built numbers. The phase cache is a flag that
    T1b opens or closes.
  * **Missing is allowed; missing silently is not** (§6.7). Every cue not
    acted on is logged with a reason.

Pure Python, no I/O: `control` is a callable (``FrameSource.control``) and
time is passed in, so the whole thing runs against scripted cues in tests.
"""

from __future__ import annotations

import bisect
from collections import deque
from dataclasses import dataclass
import hashlib
from typing import Callable, Iterable

from .ports import Detection

# ---------------------------------------------------------------------------
# keys
# ---------------------------------------------------------------------------

#: Channel raster used to turn a bare frequency into a key: 12.5 kHz is the
#: narrowest common land-mobile channel; anything finer is noise.
CHANNEL_HZ = 12_500.0


def freq_key(fc_hz: float) -> str:
    """``freq:<channel index>`` — the key of last resort."""
    return f"freq:{int(round(fc_hz / CHANNEL_HZ))}"


def signature_key(sig: dict, fc_hz: float) -> str:
    """``sig:<hash>`` from what the detector measured about a burst with no
    identity: the channel, the bandwidth class, the burst-length class, the
    periodicity class and the modulation hint. Coarse on purpose — a
    signature is for keeping unknowns apart from each other over minutes,
    not for proving identity (umbrella §7.5's caveat)."""
    bw = float(sig.get("bw_hz", 0.0) or 0.0)
    dur = float(sig.get("duration_s", 0.0) or 0.0)
    per = float(sig.get("periodicity_s", 0.0) or 0.0)
    hint = str(sig.get("modulation_hint", "") or "")
    bw_class = int(round(bw / 2_500.0))            # 2.5 kHz steps
    dur_class = 0 if dur <= 0 else min(9, int(dur).bit_length())   # log-ish
    per_class = 0 if per <= 0 else min(9, int(per).bit_length())
    raw = f"{int(round(fc_hz / CHANNEL_HZ))}|{bw_class}|{dur_class}|{per_class}|{hint}"
    return "sig:" + hashlib.sha1(raw.encode()).hexdigest()[:10]


def key_for(det: Detection) -> str:
    """ID → signature → frequency, one interface."""
    if det.key:
        return det.key
    if det.signature:
        return signature_key(det.signature, det.fc_hz)
    return freq_key(det.fc_hz)


# ---------------------------------------------------------------------------
# costs (T1b) and hold times
# ---------------------------------------------------------------------------

@dataclass
class Costs:
    """What a retune costs, in seconds. Defaults are atkdaq as built
    (§11.6: 50 ms retune settle, 0.14 s cal-only, 0.27 s sync+cal). T1b
    replaces them with the measured numbers; `phase_cache` is True only if
    T1b showed the PLL phase repeats at the same frequency."""

    settle_s: float = 0.05
    cal_s: float = 0.14
    sync_cal_s: float = 0.27
    integration_s: float = 0.5
    phase_cache: bool = False
    delays_survive: bool = True

    def retune_s(self, *, cached: bool = False) -> float:
        if not self.delays_survive:
            return self.settle_s + self.sync_cal_s
        if self.phase_cache and cached:
            return self.settle_s
        return self.settle_s + self.cal_s


class HoldTimes:
    """Per-key median burst length, with the ID → signature → frequency
    fallback built into `expected()`. Bounded memory per key."""

    def __init__(self, default_s: float = 3.0, keep: int = 50):
        self.default_s = default_s
        self.keep = keep
        self._samples: dict[str, list] = {}
        #: typical PTT lengths by protocol, used before a key has history
        self.by_protocol: dict[str, float] = {
            "dmr": 4.0, "p25": 4.0, "tetra": 3.0, "nxdn": 4.0, "fm": 6.0,
            "beacon": 0.5, "lora": 0.3,
        }

    def observe(self, key: str, duration_s: float) -> None:
        if duration_s <= 0 or not key:
            return
        xs = self._samples.setdefault(key, [])
        bisect.insort(xs, float(duration_s))
        if len(xs) > self.keep:
            # drop the extremes alternately so the median stays honest
            del xs[0 if len(xs) % 2 else -1]

    def n(self, key: str) -> int:
        return len(self._samples.get(key, ()))

    def median(self, key: str) -> float | None:
        xs = self._samples.get(key)
        if not xs:
            return None
        return xs[len(xs) // 2]

    def expected(self, keys: Iterable[str], protocol: str = "") -> tuple:
        """(expected total length, basis) for the first key with history,
        else the protocol's typical, else the default."""
        for k in keys:
            m = self.median(k)
            if m is not None:
                return m, k
        if protocol and protocol.lower() in self.by_protocol:
            return self.by_protocol[protocol.lower()], f"protocol:{protocol}"
        return self.default_s, "default"


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

POLICIES = ("follow_listener", "park_band", "park_handsets", "manual")
CUED_MODES = ("off", "priority_only", "all_triggers")

#: Built-in priority: distress beacons. (lo, hi) in Hz.
BEACON_BANDS: tuple = (
    (406.0e6, 406.1e6),       # 406 MHz PLB / EPIRB / ELT
    (121.45e6, 121.55e6),     # 121.5 MHz
    (242.9e6, 243.1e6),       # 243.0 MHz
)


@dataclass
class SchedulerConfig:
    window_hz: float = 2.4e6
    policy: str = "follow_listener"
    cued_mode: str = "priority_only"
    #: park_band: the window centre; park_handsets: the input-side centres
    #: (one or more); manual: wherever the operator put it.
    park_fc_hz: float | None = None
    handset_fcs_hz: tuple = ()
    #: operator lists — keys (ID, sig: or freq:), and frequency ranges
    priority_keys: frozenset = frozenset()
    priority_bands: tuple = ()
    beacon_bands: tuple = BEACON_BANDS
    #: a callable key -> bool for the map box(es); None = no box
    in_box: Callable[[str], bool] | None = None
    #: the dwell plan
    revisit_s: float = 3.0          # guaranteed revisit for priority windows
    dwell_s: float = 1.0            # time in a non-priority window per visit
    idle_drop_s: float = 120.0      # a window with nothing heard leaves the rotation
    return_after_s: float = 8.0     # after a chase: go back when quiet this long
    #: calibration refresh (§4.3): in idle time, when older than this
    cal_max_age_s: float = 600.0
    quiet_before_cal_s: float = 1.0
    #: a cue is "novel" if its key was never bearinged this session
    novelty_bonus: bool = True


# ---------------------------------------------------------------------------
# what comes out
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    t: float
    action: str            # stay | tune | preposition | cal | skip
    fc_hz: float
    reason: str
    key: str = ""
    cue: Detection | None = None
    score: float = 0.0


@dataclass
class Missed:
    """A cue the scheduler did not act on. 'Heard, not bearinged'."""

    t: float
    key: str
    fc_hz: float
    reason: str
    detail: str = ""


# ---------------------------------------------------------------------------
# the scheduler
# ---------------------------------------------------------------------------

#: Priority tiers, highest first. The score is tier-major, level-minor, so a
#: priority emitter at low level always beats a loud unknown.
TIER_PRIORITY_KEY = 6
TIER_PRIORITY_BAND = 5
TIER_BEACON = 4
TIER_GRANT = 3
TIER_NOVEL = 2
TIER_BOX = 1
TIER_LEVEL = 0


class Scheduler:
    """Feed it cues (`on_cue`), burst ends (`on_burst_end`), bearings as the
    engine produces them (`note_bearing`), and the clock (`tick`). It issues
    ``tune <hz>`` / ``cal`` through `control` and keeps two logs:
    `decisions` and `missed`."""

    def __init__(self, config: SchedulerConfig, costs: Costs | None = None,
                 control: Callable[[str], None] | None = None,
                 fc_hz: float | None = None, now: float = 0.0,
                 hold_times: HoldTimes | None = None):
        if config.policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}, not {config.policy!r}")
        if config.cued_mode not in CUED_MODES:
            raise ValueError(f"cued_mode must be one of {CUED_MODES}, not {config.cued_mode!r}")
        self.config = config
        self.costs = costs or Costs()
        self._control = control or (lambda _cmd: None)
        self.hold = hold_times or HoldTimes()
        #: array state
        self.fc_hz: float | None = fc_hz
        self.busy_until: float = now           # retune/cal in progress until
        self.last_cal_t: float = now
        self.parked_since: float = now
        self.listener_fc_hz: float | None = None
        #: where the policy wants the array when nothing is being chased
        self._home_index = 0
        self._chasing: str = ""                # key being chased, if any
        self._chase_fc: float | None = None
        self._chase_until: float = 0.0         # hold a pre-position at least until
        self._last_heard_in_window: float = now
        self._last_cue_t: dict[str, float] = {}    # key -> last cue time
        self._visited: dict[float, float] = {}     # window fc -> last visit
        self._bearinged: set = set()               # keys bearinged this session
        self._cached_fcs: set = set()              # frequencies calibrated before
        self._open: dict[str, Detection] = {}      # open cues by key
        self._preposition_fc: float | None = None
        self.decisions: list[Decision] = []
        self.missed: list[Missed] = []
        self._recent: deque = deque(maxlen=200)

    # -- external facts ------------------------------------------------------
    def set_listener(self, fc_hz: float, now: float) -> None:
        """The bladeRF's centre moved (follow_listener)."""
        self.listener_fc_hz = fc_hz
        if self.config.policy == "follow_listener" and not self._chasing:
            self._go(self._listener_window(fc_hz), now, "tune",
                     "following the listener", key="")

    def note_bearing(self, key: str, fc_hz: float, now: float) -> None:
        """The engine produced a bearing for `key` — it is no longer novel,
        and the window is live."""
        self._bearinged.add(key)
        self._last_heard_in_window = now

    def on_burst_end(self, det: Detection, now: float) -> None:
        """Feeds the hold-time learner under every key the burst had."""
        key = key_for(det)
        dur = det.duration(now)
        for k in self._keys(det):
            self.hold.observe(k, dur)
        self._open.pop(key, None)
        if self.in_window(det.fc_hz):
            # the end of a burst is the last moment the window was heard
            self._last_heard_in_window = max(self._last_heard_in_window,
                                             det.t1 if det.t1 is not None else now)

    # -- the decision ----------------------------------------------------------
    def on_cue(self, det: Detection, now: float) -> Decision | None:
        key = key_for(det)
        self._last_cue_t[key] = now
        if det.is_open(now):
            self._open[key] = det
        cfg = self.config

        # 1. already in the window: the engine takes it from the ring (§7)
        if self.in_window(det.fc_hz):
            self._last_heard_in_window = now
            d = Decision(now, "stay", self.fc_hz or det.fc_hz,
                         "already in the window — bearinged from the ring", key, det)
            self.decisions.append(d)
            return d

        # 2. mode gate
        if cfg.cued_mode == "off":
            return self._miss(now, key, det, "mode_off",
                              "cued DF is off; the array stays parked")
        tier, score = self._score(det, key, now)
        if cfg.cued_mode == "priority_only" and tier < TIER_BEACON:
            return self._miss(now, key, det, "not_priority",
                              f"outside the window and not on a priority list "
                              f"(tier {tier})")

        # 3. is something better already being chased or heard here?
        current = self._current_tier(now)
        if current > tier and not self._window_quiet(now, cfg.return_after_s):
            return self._miss(now, key, det, "lower_priority",
                              f"the current window holds tier {current}; this is tier {tier}")

        # 4. the gating rule
        cost = self._cost_to(det.fc_hz)
        if now < self.busy_until:
            cost += self.busy_until - now
        expected, basis = self.hold.expected(self._keys(det), det.protocol)
        if det.grant:
            # a grant arrives BEFORE the voice: the whole burst is ahead of us
            remaining = expected
        else:
            remaining = expected - det.duration(now)
        # you integrate over what exists: a 0.5 s burst never needs 0.5 s
        # of integration on top of itself
        integ = min(self.costs.integration_s, max(0.1, expected))
        need = cost + integ
        if remaining >= need:
            return self._go(self._target_for(det.fc_hz), now, "tune",
                            f"gate passed: expected remaining {remaining:.1f} s "
                            f"≥ cost {cost:.2f} s + integration "
                            f"{integ:.1f} s (hold time from {basis})",
                            key=key, cue=det, score=score, chase=True)

        # 5. failed the gate: pre-position for the next key-up if it matters.
        # A periodic emitter (a beacon every 50 s, a radio's keep-alive) says
        # when that is; the pre-position holds until then.
        if tier >= TIER_BEACON or (cfg.in_box and cfg.in_box(key)):
            period = float((det.signature or {}).get("periodicity_s", 0.0) or 0.0)
            self._miss(now, key, det, "gate_failed",
                       f"expected remaining {remaining:.1f} s < cost {cost:.2f} s "
                       f"+ integration {integ:.1f} s "
                       f"(hold time from {basis}); pre-positioning for the next key-up"
                       + (f" expected in ~{period:.0f} s" if period else ""))
            d = self._go(self._target_for(det.fc_hz), now, "preposition",
                         "parked on this channel for the next key-up"
                         + (f" (periodic, ~{period:.0f} s)" if period else ""),
                         key=key, cue=det, score=score, chase=True)
            if period:
                self._chase_until = max(self._chase_until, det.t0 + period + 2.0)
            return d
        return self._miss(now, key, det, "gate_failed",
                          f"expected remaining {remaining:.1f} s < cost {cost:.2f} s "
                          f"+ integration {integ:.1f} s (hold time from {basis})")

    # -- the clock ---------------------------------------------------------------
    def tick(self, now: float) -> list:
        """The dwell plan between cues: return from a chase that went quiet,
        honour revisit intervals, rotate, refresh the calibration in idle
        moments. Returns the decisions made."""
        out: list = []
        if now < self.busy_until:
            return out
        cfg = self.config

        # a chase that has gone quiet ends (a periodic pre-position waits
        # for its expected key-up first)
        if self._chasing and now >= self._chase_until \
                and self._window_quiet(now, cfg.return_after_s):
            home = self._home_windows()
            self._chasing = ""
            self._chase_fc = None
            if home and not self.in_window(home[0]):
                out.append(self._go(home[0], now, "tune",
                                    f"chase over — nothing heard for "
                                    f"{cfg.return_after_s:.0f} s; back to the plan",
                                    key=""))
                return out

        # the plan: the guaranteed revisit first, then the rotation
        if not self._chasing:
            windows = self._home_windows()
            if windows and self.fc_hz is None:
                self._home_index = 0
                out.append(self._go(windows[0], now, "tune", "initial park", key=""))
                return out
            if len(windows) > 1:
                # a window visited before whose revisit interval has run out
                # pre-empts the dwell: that is what "guaranteed" means
                due = [w for w in windows
                       if w in self._visited
                       and now - self._visited[w] >= cfg.revisit_s
                       and not self.in_window(w)]
                if due:
                    self._home_index = windows.index(due[0])
                    out.append(self._go(due[0], now, "tune",
                                        f"revisit due ({cfg.revisit_s:.0f} s plan)",
                                        key=""))
                    return out
                if now - self.parked_since >= cfg.dwell_s:
                    nxt = windows[(self._home_index + 1) % len(windows)]
                    if not self.in_window(nxt):
                        self._home_index = (self._home_index + 1) % len(windows)
                        out.append(self._go(nxt, now, "tune", "rotation", key=""))
                        return out

        # calibration refresh in a quiet moment (§4.3 / §6.6)
        if (now - self.last_cal_t >= cfg.cal_max_age_s
                and self._window_quiet(now, cfg.quiet_before_cal_s)):
            self._control("cal")
            self.last_cal_t = now
            self.busy_until = now + self.costs.cal_s
            d = Decision(now, "cal", self.fc_hz or 0.0,
                         f"calibration older than {cfg.cal_max_age_s:.0f} s and the "
                         f"window has been quiet {cfg.quiet_before_cal_s:.0f} s")
            self.decisions.append(d)
            out.append(d)
        return out

    # -- answers ------------------------------------------------------------------
    def in_window(self, fc_hz: float) -> bool:
        if self.fc_hz is None:
            return False
        return abs(fc_hz - self.fc_hz) <= self.config.window_hz / 2.0

    def window_bounds(self) -> tuple | None:
        if self.fc_hz is None:
            return None
        h = self.config.window_hz / 2.0
        return (self.fc_hz - h, self.fc_hz + h)

    def status(self, now: float) -> str:
        """One line for the Operator view."""
        b = self.window_bounds()
        where = (f"{b[0]/1e6:.3f}–{b[1]/1e6:.3f} MHz" if b else "not tuned")
        what = (f"chasing {self._chasing}" if self._chasing
                else {"follow_listener": "following the listener",
                      "park_band": "parked on the band",
                      "park_handsets": "parked on the handsets",
                      "manual": "parked where you put it"}[self.config.policy])
        age = now - self.last_cal_t
        return f"{what} ({where}); calibrated {age:.0f} s ago; cued DF {self.config.cued_mode}"

    def missed_summary(self) -> dict:
        out: dict = {}
        for m in self.missed:
            out[m.reason] = out.get(m.reason, 0) + 1
        return out

    # -- internals ------------------------------------------------------------------
    def _keys(self, det: Detection) -> list:
        """The fallback chain for one detection: its key, its signature key,
        its frequency key — deduplicated, in that order."""
        ks = []
        for k in (det.key,
                  signature_key(det.signature, det.fc_hz) if det.signature else "",
                  freq_key(det.fc_hz)):
            if k and k not in ks:
                ks.append(k)
        return ks

    def _score(self, det: Detection, key: str, now: float) -> tuple:
        cfg = self.config
        tier = TIER_LEVEL
        if any(k in cfg.priority_keys for k in self._keys(det)):
            tier = TIER_PRIORITY_KEY
        elif any(lo <= det.fc_hz <= hi for lo, hi in cfg.priority_bands):
            tier = TIER_PRIORITY_BAND
        elif any(lo <= det.fc_hz <= hi for lo, hi in cfg.beacon_bands):
            tier = TIER_BEACON
        elif det.grant:
            tier = TIER_GRANT
        elif cfg.novelty_bonus and not any(k in self._bearinged for k in self._keys(det)):
            tier = TIER_NOVEL
        elif cfg.in_box and cfg.in_box(key):
            tier = TIER_BOX
        # box relevance lifts anything one tier short of the top
        if cfg.in_box and cfg.in_box(key) and tier < TIER_PRIORITY_KEY:
            tier = max(tier, TIER_BOX)
        score = tier * 1000.0 + max(0.0, min(999.0, det.level_db + 200.0))
        return tier, score

    def _current_tier(self, now: float) -> int:
        """The best tier among cues still open in the current window."""
        best = -1
        for d in self._open_now(now):
            if self.in_window(d.fc_hz):
                t, _ = self._score(d, key_for(d), now)
                best = max(best, t)
        return best

    def _window_quiet(self, now: float, for_s: float) -> bool:
        """Nothing heard in the window for `for_s`, and no cue in it is
        still open (a 12 s carrier is one detection, not twelve)."""
        if any(self.in_window(d.fc_hz) for d in self._open_now(now)):
            return False
        return (now - self._last_heard_in_window) >= for_s

    #: A cue nobody closed is treated as closed after this long: a detector
    #: that forgets to end a burst must not pin the array forever.
    OPEN_MAX_S = 60.0

    def _open_now(self, now: float) -> list:
        out = []
        for k, d in list(self._open.items()):
            if not d.is_open(now) or now - d.t0 > self.OPEN_MAX_S:
                self._open.pop(k, None)
                continue
            out.append(d)
        return out

    def _target_for(self, fc_hz: float) -> float:
        """Where to tune for a cue: a home window that covers it, so the
        rest of that window stays covered; else the cue's own frequency."""
        for w in self._home_windows():
            if abs(fc_hz - w) <= self.config.window_hz / 2.0:
                return w
        return fc_hz

    def _cost_to(self, fc_hz: float) -> float:
        cached = fc_hz in self._cached_fcs
        return self.costs.retune_s(cached=cached)

    def _listener_window(self, fc_hz: float) -> float:
        return fc_hz

    def _home_windows(self) -> list:
        cfg = self.config
        if cfg.policy == "follow_listener":
            return [self.listener_fc_hz] if self.listener_fc_hz is not None else []
        if cfg.policy == "park_band":
            return [cfg.park_fc_hz] if cfg.park_fc_hz is not None else []
        if cfg.policy == "park_handsets":
            # cluster the input frequencies into as few windows as needed
            return _windows_for(sorted(cfg.handset_fcs_hz), cfg.window_hz)
        return [self.fc_hz] if self.fc_hz is not None else []

    def _go(self, fc_hz: float, now: float, action: str, reason: str, *,
            key: str = "", cue: Detection | None = None, score: float = 0.0,
            chase: bool = False) -> Decision:
        if self.in_window(fc_hz) and action != "preposition":
            d = Decision(now, "stay", self.fc_hz, f"{reason} — already here", key, cue, score)
            self.decisions.append(d)
            return d
        cost = self._cost_to(fc_hz)
        self._control(f"tune {int(round(fc_hz))}")
        self.fc_hz = float(fc_hz)
        self.busy_until = max(self.busy_until, now) + cost
        self.last_cal_t = now if not (self.costs.phase_cache and fc_hz in self._cached_fcs) \
            else self.last_cal_t
        self._cached_fcs.add(float(fc_hz))
        self.parked_since = now
        self._visited[float(fc_hz)] = now
        self._last_heard_in_window = now          # give the new window a chance
        if chase:
            self._chasing = key
            self._chase_fc = float(fc_hz)
            self._chase_until = now
        d = Decision(now, action, float(fc_hz), reason, key, cue, score)
        self.decisions.append(d)
        return d

    def _miss(self, now: float, key: str, det: Detection, reason: str,
              detail: str) -> None:
        self.missed.append(Missed(now, key, det.fc_hz, reason, detail))
        return None


def _windows_for(fcs: list, window_hz: float) -> list:
    """Greedy cover of sorted frequencies by windows of `window_hz`: each
    window centred on the span it covers. Repeater inputs 5 MHz apart become
    two windows; a cluster within 2.4 MHz becomes one."""
    out: list = []
    i = 0
    while i < len(fcs):
        lo = fcs[i]
        j = i
        while j + 1 < len(fcs) and fcs[j + 1] - lo <= window_hz * 0.9:
            j += 1
        out.append((lo + fcs[j]) / 2.0)
        i = j + 1
    return out
