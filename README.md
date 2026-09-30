# atkdf

The Analyst Toolkit's direction-finding and geolocation engine: coherent
multi-channel samples → bearings → fixes → situational awareness, and the
simulator that tests all of it against a planted truth. Pure Python
(numpy/scipy, optional torch), **no Qt, no imports from ATK, nothing GPL**.
Frames come from [atkdaq](../atkdaq), a `.atkq` recording, a SigMF clip, or
the simulator — the same `FrameSource`.

**Status (2026-09-30):** Phase 0 built and green — the `siga` maths moved in,
the DF core, the simulator spine, and the engine end to end. Nothing here
needs the KrakenSDR. The plan is `docs/ATKDF_PLAN.md`; ATK's
`docs/KRAKEN_DF_PLAN.md` is the umbrella.

## What's built

```
atkdf/
  dossier.py geodesy.py fusion.py geoloc.py ephemeris.py df_file.py bluefile.py
                         MOVED from ATK's siga, with their tests. ATK now
                         reaches these through re-export shims (its own tests
                         still pass). Edit them HERE.
  recio.py               open a recording (SigMF + Midas BLUE) the way ATK's
                         recording.py does, without importing ATK
  _prep.py _backend.py   the few helpers geoloc needs from ATK's prep/backend
  atkq.py                atkdaq's wire format v1, read from the header file —
                         NOT from atkdaq's GPL package. FileSource / ProcessSource.
  manifold.py            geometry (UCA-5 r=0.18, ULA, custom), ideal manifold,
                         its Fourier (EADF) form, and a measured manifold
  calibration.py         apply atkdaq's weight AND fractional-delay slope;
                         the passport; the on-air coherence monitor
  doa.py                 covariance, Bartlett/Capon/MUSIC/root-MUSIC/ML/FBSS,
                         MDL/AIC source count, the UCA phase-mode transform,
                         quality metrics, the 360-point likelihood
  records.py             BearingRecord, Passport, EmitterKey — JSON round-trip
  ports.py fakes.py      the injected protocols, and fakes for tests/sim
  engine.py              the per-frame loop: calibrate → per-burst DoA → record
  sim.py                 the scenario simulator: a FrameSource with a planted
                         truth plus fakes of every port — the test spine
  tests/                 246 tests, numpy only, no hardware, no ATK, no Qt
```

## Testing

```
python -m pytest tests -q
```

246 tests: the moved suites (geodesy, ephemeris, fusion, geoloc, df_file,
bluefile), the wire-format reader against a fixture atkdaq itself wrote, the
manifold, the estimators against planted bearings, calibration, the records,
and the engine end-to-end on the simulator.

## The boundaries that make it a separate repo

1. **Pure Python, ports not dependencies.** Everything the engine needs from
   outside — frames, decoder events, positioning, elevation, a road graph, a
   catalogue, a path-loss model — arrives through a small protocol
   (`ports.py`). ATK implements them; tests fake them; the simulator provides
   them all. `atkdf` never imports ATK.
2. **No GPL.** `atkq.py` reads atkdaq's wire format from the documented header,
   and the live source starts `atkdaq.exe` as a separate process. atkdaq's own
   GPL Python package is never imported.
3. **Known-answer first.** The simulator plants the truth and the whole chain
   must return it; a plausible residual is not a pass.

## Untested until the KrakenSDR arrives

The engine is proven on the synthetic device. What only the hardware answers:
the real array manifold on a vehicle roof (the ideal manifold carries a caveat
on every bearing until a drive/tripod calibration replaces it), real coherent
multipath in a town, decoder timing latency, and the T3/T4 field numbers.
Those are Phase 3 and the umbrella plan's §12.
"# atkdf" 
