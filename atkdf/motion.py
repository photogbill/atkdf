"""Is the target moving? — a per-emitter motion state from the evidence to hand.

The posterior ages a silent emitter by its motion state (stationary stays put,
on-foot diffuses at a walk, vehicle along roads), so the state has to come from
somewhere. Five indicators were specified (umbrella §8.3); the always-available
one, and the only one that needs nothing but the bearing stream, is:

  * Kinematic. Are the emitter's bearings consistent with a SINGLE fixed point,
    given where the receiver was for each? If a stationary point explains them
    within their own σ, the target may be stationary; if the best fixed point
    leaves large, structured residuals, the target moved. This reuses
    `fusion.ml_fix` — the fix's residual against the bearings IS the test.

The others (channel dynamics — Doppler spread / gait modulation, strongest when
the receiver is stopped; carrier tracking; protocol site-changes; RSSI trend)
attach through `fuse`, which combines whatever indicators a caller can supply
into a state with a confidence. Operating rule that shaped this: to answer "is
473 moving?", stop for thirty seconds — then the channel indicator is clean.
`fuse` weights indicators by the caller-supplied confidence so a clean stopped-
receiver reading dominates a noisy moving one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import fusion

STATES = ("stationary", "on_foot", "vehicle", "unknown")


@dataclass
class MotionVote:
    state: str
    confidence: float
    source: str
    detail: str = ""


def kinematic_vote(records, min_bearings: int = 4) -> MotionVote:
    """Fixed-point consistency of an emitter's fusable bearings.

    A single fix is fitted; if it explains the bearings within their σ
    (consistency high, residual ~ σ) the target is consistent with stationary;
    a large residual means no fixed point fits — the target moved. Speed is not
    resolved here (that needs an observer manoeuvre), so a moving target is
    reported as 'vehicle' vs 'on_foot' only weakly, via the residual size."""
    lobs = [r.to_lob(position_sigma_m=r.quality.get("position_sigma_m", 0.0))
            for r in records if r.fusable]
    lobs = [l for l in lobs if l is not None]
    if len(lobs) < min_bearings:
        return MotionVote("unknown", 0.0, "kinematic", "too few bearings")
    fx = fusion.ml_fix(lobs)
    if fx is None:
        return MotionVote("unknown", 0.0, "kinematic", "no fix")
    resid = float(getattr(fx, "residual_rms_deg", float("nan")))
    sig = float(np.median([l.sigma_deg for l in lobs])) or 1.0
    if not math.isfinite(resid):
        return MotionVote("unknown", 0.0, "kinematic", "no residual")
    ratio = resid / sig
    if ratio < 1.5:
        # bearings fit one point within their noise -> looks stationary
        conf = float(min(0.9, 0.4 + 0.5 * (1.5 - ratio)))
        return MotionVote("stationary", conf, "kinematic",
                          f"residual {resid:.1f} deg ~ sigma {sig:.1f} deg")
    # no fixed point fits: it moved. Bigger miss -> faster (weak).
    state = "vehicle" if ratio > 4.0 else "on_foot"
    conf = float(min(0.8, 0.3 + 0.1 * ratio))
    return MotionVote(state, conf, "kinematic",
                      f"residual {resid:.1f} deg >> sigma {sig:.1f} deg")


def fuse(votes) -> MotionVote:
    """Combine indicator votes into one state. Each state's score is the sum of
    the confidences voting for it; the winner's confidence is its share of the
    total. An empty set is 'unknown'."""
    votes = [v for v in votes if v and v.state != "unknown" and v.confidence > 0]
    if not votes:
        return MotionVote("unknown", 0.0, "fused", "no usable indicator")
    score: dict = {}
    for v in votes:
        score[v.state] = score.get(v.state, 0.0) + v.confidence
    total = sum(score.values())
    state = max(score, key=score.get)
    return MotionVote(state, float(score[state] / total), "fused",
                      " + ".join(f"{v.source}:{v.state}({v.confidence:.2f})" for v in votes))
