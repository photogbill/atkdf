# atkdf — Direction finding and geolocation engine: Build Plan (v1)

> **Status (2026-09-30):** Phases 0–2 built and green, plus the start of
> Phase 3 (276 tests, numpy only, no hardware, no ATK, no Qt). Phase 0: the
> `siga` move, the DF core, the simulator spine, the engine. Phase 1:
> registry + JSONL merge, heading filter, identity/path-tag, recorder +
> reprocess + SigMF export. Phase 2: posterior grid + ageing, roster +
> keep-alive learner, bearing tracker, references (radio compass, mounting
> offset, learned error model), wideband spatial spectrum. Phase 3 so far:
> the measured manifold and the learned error model (the two biggest accuracy
> items) and the kinematic motion indicator. §11 records what was built.
> KrakenSDR not in hand; none of this needs it. What remains (§11.12): pilots
> (needs atkdsp), the road-constrained particle filter and terrain/range
> likelihoods (need ATK's road-graph and elevation ports), the planner,
> fingerprints, and RTS smoothing.
> This is the plan for the **engine**: everything that turns coherent
> multi-channel samples into bearings, bearings into fixes, and fixes into
> situational awareness — plus the simulator that tests all of it with a
> planted truth. The DAQ is `atkdaq` (its own repo, GPL); the ATK-side wiring
> (the `IqSource`, the DF panel, map layers, terrain tiles, the DSD tap) is in
> ATK's `docs/KRAKEN_DF_PLAN.md`, the umbrella. The umbrella holds the
> rationale for every algorithm here; this document holds the package, its
> boundaries, and its build order. Where they disagree about what is *inside*
> `atkdf`, this one wins.
>
> **Design target (Bill, 2026-09-30):** one KrakenSDR, five omnis, static and
> mobile, GPS-only heading, DMR 400–450 MHz and an LTE tower to test on;
> bearings keyed by radio ID for humanitarian relief; *"computational
> complexity shouldn't scare us."*

---

## 0. Boundaries — the rules that make this a separate repo

1. **Pure Python; numpy/scipy; optional torch backend.** No Qt. No imports
   from ATK. Nothing GPL. Licence: the author's choice — the code links
   nothing that constrains it.
2. **Ports, not dependencies.** Everything the engine needs from the
   outside world arrives as an injected object with a tiny protocol
   (§2). ATK implements the ports; tests implement fakes; the simulator
   implements them all.
3. **Every computing function is pure**: arrays and dataclasses in,
   dataclasses out, reproducible from a file. The dossier conventions from
   ATK's `siga` (a `Measurement` with value, unit, confidence, method,
   evidence, caveats) come with the code that moves (§1.1).
4. **Known-answer testing.** The simulator plants the truth and the whole
   chain must return it; per-bearing errors are compared against fitted
   residuals, not residuals alone. Fixtures from ATK's hardware tests
   (T1–T4) are committed small and are the regression spine.
5. **Nothing here needs the hardware.** Frames come from `atkdaq`'s Python
   client, from a `.atkq` file, from a SigMF clip, or from the simulator —
   all four are the same `FrameSource` port.

## 1. Layout

```
atkdf/
  atkdf/
    __init__.py            version; the public API surface
    types.py               Measurement/dossier (moved), BearingRecord, Passport, Emitter, Fix
    ports.py               the injected protocols (§2)
    geodesy.py             MOVED from ATK siga: WGS-84, Vincenty, body→NED, cone_bearings
    ephemeris.py           MOVED: the four ephemeris sources + dump
    fusion.py              MOVED: LOBs + priors → Fix, bias groups, ambiguity reporting
    sigmf_io.py            MOVED from siga/df_file.py + the 5-ch writer (§7)
    geoloc.py              MOVED from siga/geoloc.py: TDOA/FDOA/RSSI heatmap, steering_vector,
                           music, esprit, watson_watt, beamform, synth_array
    array.py               geometries, connector map, ideal manifold, measured manifold
    calibration.py         Passport handling; applying atkdaq weights; cal-age policy; on-air monitor
    doa.py                 covariance policy, Bartlett/Capon/MUSIC/root-MUSIC/UCA phase-mode,
                           MDL/AIC, FBSS, quality metrics, wideband spatial spectrum
    pilots.py              pilot-aided steering vectors: LTE PSS (via atkdsp when present), DMR sync, P25 FS
    identity.py            burst spans + IDs from decoder events; slot; sync type; path tag
    registry.py            emitters keyed by (protocol, system, id); bearing records; state
    roster.py              expected IDs, fleet-map names, silence thresholds, keep-alive learner, alerts
    tracker.py             GNN → JPDA bearing tracker for emitters without identity
    heading.py             constant-turn-rate filter on GPS + radio-compass observations, with σ
    motion.py              the five motion indicators → motion state per emitter
    posterior.py           per-emitter grid (log-domain), road-constrained particle filter,
                           ageing of silent emitters, terrain and range likelihoods, ellipse via fusion
    planner.py             next-observation planning over the road graph
    references.py          reference-emitter catalogue, radio compass, error-model learning
    fingerprint.py         specific-emitter features (Phase 3)
    recorder.py            clip writer (5-ch SigMF + sidecar), reprocessing driver
    sim.py                 the scenario simulator (a FrameSource + fake ports)
    backends/
      numpy_backend.py     batched eigh, correlations, grid ops
      torch_backend.py     the same, on the GPU, only if torch imports
    _atkdsp.py             optional binding to atkdsp (lte.c PSS/SSS, dmr_sync.c, ITM); numpy fallbacks
  tests/
    test_geodesy.py … test_fusion.py … (MOVED with the code, 645 checks)
    test_array.py test_doa.py test_pilots.py test_identity.py test_registry.py test_roster.py
    test_tracker.py test_heading.py test_motion.py test_posterior.py test_planner.py
    test_references.py test_recorder.py test_sim.py (the spine) test_ports_fakes.py
    fixtures/  T1_noise.atkq T2_splitter.atkq T3_static.sigmf T4_drive.sigmf (+ sidecars, tracks)
  bench/bench_wideband.py bench_posterior.py
  docs/ATKDF_PLAN.md · README.md · LICENSE · pyproject.toml
```

### 1.1 The move from ATK's `siga`

`geodesy.py`, `ephemeris.py`, `fusion.py`, `df_file.py` and `geoloc.py`
are pure, tested, and exactly what this engine is made of; an engine that
ATK vendors cannot import them from ATK. They move here **with their tests
and their docstrings intact**, and ATK's `siga` keeps one-line re-export
shims (`from atkdf.fusion import *`) so the Geolocation tab, `siga_probe`
and every existing test keep working unchanged. The `dossier` types they
depend on move too (`types.py`); `siga.dossier` becomes a shim. `context.py`
(`SigCtx`) and `prep.py` are used by `geoloc.tdoa/fdoa`: the minimal
pieces they need (`interpolate_peak`, `db10`, `peak_stats`) come along as
`atkdf/_prep.py`; the rest of `siga` stays in ATK.

**Decision needed (umbrella §18):** confirm the move. Alternative — atkdf
depends on ATK — inverts the vendoring and is rejected here.

## 2. Ports — `ports.py`

```python
class FrameSource(Protocol):      # atkdaq client | .atkq file | SigMF clip | simulator
    def frames(self) -> Iterator[Frame]: ...   # Frame: header fields + (n_ch, N) complex64 view
    def control(self, cmd: str) -> None: ...   # tune/gain/cal/sync/mode/quiet — no-op for files

class DecoderEvents(Protocol):    # ATK's dsd tap | dmr_sync correlator | simulator side channel
    def events(self) -> Iterator[BurstEvent]: ...  # protocol, system, id, slot, sync_type, t_utc, span

class Positioning(Protocol):      # ATK's gps_track (interpolating) | simulator track
    def at(self, t_utc: float) -> Fix2D: ...       # lat, lon, course, speed, fix_age_s, sigma_m

class Elevation(Protocol):        # ATK terrain service | flat-earth fake
    def profile(self, a, b, step_m) -> Profile: ...  # dist, height, resolution_m

class RoadGraph(Protocol):        # ATK's offline road graph | none
    def nearest_edge(self, lat, lon) -> Edge | None: ...
    def neighbours(self, edge) -> list[Edge]: ...
    def route_time_s(self, a, b) -> float: ...

class Catalogue(Protocol):        # references: PCI/eNB, repeaters
    def lookup(self, key) -> Reference | None: ...
    def near(self, lat, lon, radius_m) -> list[Reference]: ...

class PathLoss(Protocol):         # atkdsp ITM | free-space fallback
    def loss_db(self, profile: Profile, f_hz, h_tx, h_rx) -> float: ...
```

Every module takes the ports it needs as constructor arguments. `sim.py`
implements all of them so the whole engine runs with nothing but numpy.

## 3. The core loop (what `atkdf.Engine` does per frame)

```
frame ──▶ calibration.apply(weights)          (skip frames flagged noise_on / misaligned)
      ──▶ ring (the engine keeps its own window of recent frames; ATK's DVR ring is upstream)
      ──▶ doa.wideband_spectrum(frame)         → (bins × 360) likelihood image, every block
      ──▶ identity.spans(events)               → burst spans with IDs / slot / sync type / path tag
      ──▶ for each span: doa.estimate(ring[span], manifold, estimator)  → Measurement + likelihood
              heading.at(t)  → θ_abs, σ            positioning.at(t) → lat, lon
              registry.add(BearingRecord)          posterior[key].update(record, terrain, range)
              motion[key].update(record, channel stats, carrier, protocol)
              roster.observe(key, t)               tracker (if no key)
      ──▶ references: radio compass, coherence monitor, error-model residuals
      ──▶ planner (on demand) ; recorder (on demand / on trigger)
```

Static vs mobile is a `PlatformMode` object holding window length, heading
σ policy, quality gates, recal/dwell cadence and the posterior prior; the
estimators are the same.

## 4. Module specifications (short form — the umbrella has the rationale)

| module | in | out | acceptance |
|---|---|---|---|
| `array` | preset or `(x,y)` list, connector map, tuned f | ideal steering vectors; refuses >0.5 λ spacing without override | UCA-5 at 0.18 m: 0.212 m spacing; manifold load/interp/apply round-trips |
| `calibration` | `Frame` header weights, passport, cal_age | calibrated `(n_ch, N)`; Passport dataclass; "stale" verdict | T1 fixture: spread reported = spread recomputed |
| `doa` | `(n_ch, N)`, geometry/manifold, λ, estimator, `sources` or MDL | `Measurement` with 360-pt likelihood, quality dict | planted bearing ±1° at 20 dB for every estimator; MDL = planted count; FBSS separates 15° coherent pair |
| `pilots` | element streams, pattern/PSS id, CFO bank | per-element complex amplitude, timing, pattern type | LTE at −3 dB: planted bearing where MUSIC fails; DMR: BS/MS type right |
| `identity` | `BurstEvent`s, segment/time map, catalogue | spans with key, slot, sync type, `path_tag` | outbound tagged `repeated`; two slots → two keys |
| `registry` | records | emitters, histories, JSONL log with `site_id` + passport | merge of two site logs yields one registry |
| `roster` | expected IDs, fleet map, observations | states, alerts, keep-alive model | scripted silence-then-key-up raises alert with bearing; 3σ-late check-in flagged |
| `tracker` | bearings without key | tracks | two interleaved no-ID emitters stay two tracks over a T4-like sequence |
| `heading` | GPS fixes, radio-compass obs | heading, σ(t) | σ grows at stop, shrinks on compass obs; turn error bounded |
| `motion` | records, channel stats, carrier track, protocol events | state + confidence | sim: static / walking / driving classified with receiver stopped |
| `posterior` | records, terrain, range, motion state, road graph | grid or particles; `Fix` via fusion | ellipse coverage ≈ stated on sim with learned model; vehicle fix lands on its road |
| `planner` | posterior, road graph, coverage | ranked waypoints | chosen waypoint reduces sim posterior entropy more than a random one |
| `references` | catalogue, records to references, positioning | mounting offset, compass obs, residual model | offset recovered ±1° on sim drive; residual model reproduces planted σ |
| `recorder` | ring window, everything above | 5-ch SigMF + sidecar; reprocess(clip) → same records | reprocess of a live sim session ≥ live accuracy |
| `sim` | scenario | all ports + frames | the spine: T-like scenarios with planted truth |

## 5. Estimator notes that belong to the code, not the umbrella

- Covariance over a span uses forward-backward averaging by default; the
  UCA→virtual-ULA phase-mode transform (Davies) is `array.uca_to_vula(order)`
  and is where root-MUSIC, ESPRIT and FBSS attach for the circle.
- MUSIC's grid is 0.5° by default (720), refined by parabolic interpolation
  on the peak; the 360-pt likelihood in the record is the pseudo-spectrum
  normalised to a density, `float16`.
- Quality dict: `peak_ratio_db`, `eigen_gap`, `snr_db`, `n_snapshots`,
  `subband_agreement_deg` (wideband emitters), `manifold: ideal|measured`.
- The wideband spatial spectrum uses the FFT frames the caller already has
  when it passes them (ATK's engine computes them per block); otherwise it
  computes its own. `(bins, 5, 5)` batched `eigh` is numpy; torch only
  when `bins × windows` says so.
- Grid posterior is log-domain float32, 25 m default, bounds from the
  operating area, renormalised lazily; each update is a rotate-and-convolve
  of the 360-pt likelihood by heading σ, then a gather over the cell→vehicle
  angle map (precomputed per vehicle position bucket).
- Ageing: stationary — none; on foot — Gaussian diffusion 1.2 m/s,
  terrain-weighted; vehicle — particles advance along edges at the speed
  prior; `unknown` — mixture.

## 6. The simulator — `sim.py`

A `Scenario`: emitters (position or a path on the road graph or a walking
track; protocol; ID; slot; burst schedule; power class; carrier offset),
reflectors (for coherent multipath), own-vehicle track (or a static site),
array geometry and a manifold perturbation (per-element phase/gain error,
mutual-coupling matrix), receiver noise figure, optional USB drop events,
optional reference emitters. It produces `Frame`s with atkdaq headers,
`BurstEvent`s on the side channel (identity without a decoder), a
`Positioning` track, an `Elevation` (flat or from a tile the test supplies),
a `RoadGraph` (a small synthetic graph, or none), a `Catalogue`, and a
`PathLoss` (free-space or ITM if atkdsp is present). `test_sim.py` runs the
scenarios that mirror T1–T4 and asserts the umbrella's acceptance numbers.
This is how the engine is built before the Kraken arrives and how every
later session tests without Bill's time.

## 7. Files

- **Clip:** 5-channel SigMF (`core:num_channels`, per-channel
  `core:sample_rate` — the trap ATK's `test_siga_sigmf_df.py` documents),
  sidecar JSON with geometry, connector map, passport, manifold id, the
  positioning track and heading estimates for the window, references seen,
  platform mode, and the bearing/identity records for the window.
- **Log:** JSONL of `BearingRecord`s with `site_id`, beside the clip.
- **Manifold:** `manifold_<geometry>_<band>.npz` with the passport of the
  calibration run.
- **Catalogue:** CSV/JSON with a documented schema; `references.import_*`.
- **Fleet map:** CSV `id, name, role, team, protocol, system`.

## 8. Phases

| phase | builds | done when |
|---|---|---|
| **0 — skeleton and the move** | package, `types`, `ports`, fakes; move `geodesy/ephemeris/fusion/df_file/geoloc` + tests from ATK with shims left behind; `array`, `calibration`; `sim` v0 (ideal manifold, static emitters, no propagation); `doa` = the moved estimators + Bartlett/Capon + MDL | ATK's suite still green through the shims; sim plants a bearing and MUSIC returns it through `Engine` |
| **1 — records and files** | `registry`, `recorder` (SigMF + sidecar), `heading` (GPS only), `identity` from side-channel events, JSONL log; `atkdaq` client as a `FrameSource`; `.atkq` and clip sources | a T1/T2 fixture and a sim session both round-trip through recorder → reprocess |
| **2 — relief mode** | `roster` (fleet map, silence, keep-alive learner); path tag; `tracker`; `posterior` grid + ageing; `references` (catalogue, radio compass, coherence monitor); wideband spatial spectrum; FBSS; UCA phase-mode + root-MUSIC; `PlatformMode` | T3/T4 fixtures pass the umbrella's Phase 2 numbers; sim silence-then-key-up alerts with a fix; two slots → two fixes |
| **3 — accuracy and motion** | measured manifold (drive + tripod tools); `pilots` (LTE PSS, DMR sync); learned error model; `motion` (all five indicators); road-constrained particle filter; terrain + range likelihoods; `planner`; `fingerprint`; MVDR output for a listener; ML/sparse for static | error vs truth halves against Phase 2 on the same fixtures; a driving sim emitter's fix sits on its road; motion state right with the receiver stopped |
| **4 — reprocessing** | RTS smoothing; multi-day static-site accumulation; quality classifier from fixtures; exports | a whole T4 clip reprocesses to the Phase 3 numbers with no live source |

Phase 0 needs no hardware and no `atkdaq`. Start it now.

## 9. Testing rules

- Every module has a fake-port test and a sim test; nothing requires a
  device, a GUI, ATK, or atkdsp (the `_atkdsp` binding has numpy fallbacks
  and tests run both paths when the DLL is present).
- Known-answer first: a planted truth and the chain returning it. Residual
  plausibility is not a pass.
- Fixtures are small and committed; a fixture is never edited to make a
  test pass (ATK's stale-needles rule).
- The simulator's own correctness is tested against closed-form cases
  (a ULA broadside source; a stationary emitter and a circling receiver
  give a constant fix).

## 10. Risks and limits

- **The `siga` move** touches ATK's tests; do it in one commit on both
  sides with the shims, and run ATK's suite before and after.
- **Decoder timing latency** (ATK's dsd tap) may be coarser than a burst;
  `identity` accepts either sample-exact spans or time-only events and
  widens the span accordingly, and `pilots`/`dmr_sync` are the sample-exact
  source when it matters.
- **GPS-only heading** limits mobile accuracy until references or a dwell
  fix it; `heading` reports σ honestly and the posterior widens rather than
  the record being dropped.
- **torch is optional** and must stay so — the engine runs on numpy alone.
- **Not in scope:** the DAQ, Qt, map rendering, terrain tiles and
  downloads, the DSD process, navigation UI. Those are ATK or atkdaq.

---

## 11. Build notes (2026-09-30) — Phase 0

What was built, and where it departs from §0–§10. Each note names the file.

### 11.1 The `siga` move as executed

`dossier.py`, `geodesy.py`, `fusion.py`, `geoloc.py`, `ephemeris.py`,
`df_file.py` and `bluefile.py` moved from ATK verbatim, each with a header note
saying where it came from. Their ATK tests moved too (renamed `test_*`), and
all pass here.

Departures from §1's file plan, each for a reason that only appears when you
run a package file directly:

- **`dossier.py`, not `types.py`.** A module named `types` shadows the standard
  library's `types` whenever a package file is run as a script — a trap that
  costs an afternoon. The dossier `Measurement` lives in `dossier.py`.
- **`manifold.py`, not `array.py`.** Same reason: `array` shadows the stdlib.
- **`df_file.py` keeps its name; `recio.py` is new.** ATK's `df_file` called
  `recording.kind/header/load`. The engine cannot import ATK, so `recio.py`
  reproduces those three (and the SigMF metadata reader and the sample
  decoders they stand on) with the SAME BLUE-key vocabulary and the same
  refusals. `sigmf_io.py` for writing 5-channel clips is Phase 1.
- **`_prep.py` and `_backend.py`** carry only the handful of helpers `geoloc`
  needs (`interpolate_peak`, `db10`, `peak_stats`, `window`, `bin_to_hz`;
  `fft`, `fftshift`, `fftfreq`, `next_fast_len`, `correlate_full`). The rest of
  ATK's `prep`/`backend` is the signals bench's and stayed in ATK.
- **`SigCtx` is duck-typed.** `geoloc` needs only `.iq` and `.fs`; `_prep`
  defines that as a Protocol and a tiny `Samples` dataclass, so the engine
  does not drag ATK's signal-context object across.

### 11.2 The wire-format reader (`atkq.py`)

Written from `atkdaq/include/atkdaq_frame.h`, **not** from atkdaq's GPL Python
package (§0's no-GPL rule). It parses the located, CRC-checked header (CRC
before version, so a damaged version byte reads as damage), exposes
`Frame.iq()` (delays applied, weights not), resynchronises past garbage, and
provides `FileSource` and `ProcessSource` (starts `atkdaq.exe`, reads stdout,
parses stderr status, sends stdin commands). `tests/test_atkq.py` holds it to a
fixture atkdaq itself wrote.

### 11.3 The manifold and the array frame (`manifold.py`)

The array frame is fixed once: x forward (toward element 0), y left, θ
clockwise from x so absolute = heading + θ with no sign flip. The ideal
manifold and its Fourier (EADF) form agree to 1e-6; the EADF is what lets
root-MUSIC run on a circle or a measured table. `MeasuredManifold.from_snapshots`
fits a real array's response and recovers a perturbed manifold better than the
ideal one does (`tests/test_manifold.py`). The UCA-5 refuses a frequency past
0.5 λ adjacent spacing (≈707 MHz for r = 0.18) unless overridden, saying by
how much.

### 11.4 Calibration applies the slope, not just the weight (`calibration.py`)

The header's fractional delay is a phase SLOPE across the band; applying the
weight alone is right only at band centre (22° of error 500 kHz out at
2.4 MSPS). `calibration.apply` does both — a narrowband factor at one offset,
or the full per-bin correction by FFT. A test shows the weight-only path
leaving >10° on a signal off centre that the full path removes.

### 11.5 DoA (`doa.py`) — what departs from §5/§6

- **Forward-backward averaging is conditional.** §5 said "on by default"; it is
  valid only for a centro-symmetric array. A ULA is; a 5-element UCA is not, so
  `covariance` asks the geometry, and the circle is smoothed in phase-mode
  space instead. The phase-mode transform's aliasing is measured and reported
  (`worst_alias`), not hidden — at 700 MHz the outer modes on r = 0.18 are
  badly aliased, and the caveat says so.
- **The record's likelihood is a density, not a pseudo-spectrum.** A MUSIC
  spectrum is 40 dB tall and, multiplied into the posterior, would let one
  confident wrong bearing veto the truth. The 360-point likelihood is a
  mixture of wrapped Gaussians (one per significant peak, at the bearing's σ)
  over a never-zero floor, so every mode survives and none can veto a cell.
- **σ is the Cramér–Rao bound on the actual data, root-summed with a
  per-manifold model error.** The model error is a prior (4° ideal, 1.5°
  measured) until `references` measures the real one (§8.8).
- **The source count is MDL/AIC**, but a decoder-clocked burst defaults to the
  one emitter the decoder named; coherent multipath is handled by asking for
  ML or FBSS with a small known count.

### 11.6 The engine (`engine.py`)

Per frame: skip unless usable (`atkq.Header.why_not_usable`), calibrate,
reorder channels into element order (the connector map), then for each decoder
burst that lands in the frame take band-limited snapshots in the burst's
channel and estimate; or, with no burst and in a survey mode, one bearing on
the window. `PlatformMode` sets the window, estimator, quality gate and
whether a heading is required. A BS-sourced (repeater outbound) burst is tagged
`repeated` and never fused as the handset's position (§7.1's trap).

### 11.7 The simulator (`sim.py`) — the spine

A `Scenario` → a FrameSource with a planted truth, plus a matching
DecoderEvents and Positioning. It plants emitter bearings, receiver
gain/phase/fractional-delay (and the header that inverts them, so calibration
is exact and any residual is the manifold's), coherent multipath, manifold
error, and the frame flags a real run has. **Two lessons the build cost:**

- **Levels are set like an AGC-off receiver.** A first cut set emitter
  amplitude from `snr_db` as absolute power; at 20 dB the signal was 30× the
  int8 full scale and clipped hard, and clipping is a nonlinearity that
  destroys the rank-1 spatial structure MUSIC and MDL depend on — MDL then
  counted four sources where there was one. The fix scales each frame to a
  target ADC fill (a common real scale changes no bearing), so nothing clips.
- **Never `t0 ± rel`.** The burst schedule was computed as `(t0 + k·dt) − t0`
  with `t0 ≈ 1.79e9`; the cancellation lost the millisecond and flipped TDMA
  slot boundaries at random. The schedule is computed from relative time only.

### 11.8 SNR and the processing gain, so the bench is not misread

`bench/bench_doa.py` shows ~0.01° error even at 0 dB. That is real: `snr_db` is
the WIDEBAND SNR over the 2.4 MHz span, and band-limiting a 12.5 kHz emitter
out of it is ~23 dB of processing gain, so 0 dB wideband is ~23 dB in-channel.
The synthetic emitter is also a clean rank-1 plane wave; the roof, the cables
and real multipath are what the measured manifold and the T3/T4 fixtures add,
and they are Phase 3.

### 11.9 Open until the hardware / later phases

Phases 1–4 (registry, recorder and reprocessing, heading filter, roster,
tracker, posterior with ageing, references and the learned error model, pilots,
motion, road-constrained particle filter, terrain/range likelihoods, planner)
are built on this Phase-0 core and need no hardware to write — only the T1–T4
fixtures to tune against. The measured manifold, real multipath and decoder
timing are what the KrakenSDR and Bill's drives answer.


---

## 11.10 Build notes — Phase 1 (records, files, heading, identity)

- **registry.py** — emitters keyed by `protocol:system:id` (or `track:<id>`);
  every bearing filed under its key and, optionally, appended to a JSONL log
  beside the clip. `Registry.from_logs([...])` merges two vehicles' logs by
  concatenation — the merge-ready-from-day-one design (§8.10) — and a fix is
  computed on demand from an emitter's FUSABLE bearings through the moved
  `fusion.ml_fix`. `load_fleet_map` reads the `id,name,role,team,...` CSV so an
  alert reads a name.
- **heading.py** — a constant-turn-rate filter on GPS course with a σ that
  GROWS at a stop and in a turn and SHRINKS on a radio-compass fix, so a
  mid-turn bearing comes out wide, not dropped (§8.2). Course is treated as
  unobservable below a stop speed (only the prediction runs); a compass
  observation is a direct heading measurement valid at a stop.
- **identity.py** — a `BurstEvent` → a `Span` (sample range + identity + slot
  + sync type + path tag). Sample-exact when the decoder gave n0/n1, else
  widened from a UTC instant by a slack. The path tag marks a BS-sourced
  (repeater outbound) burst `repeated` so it is never fused as the handset's
  position (§7.1); the catalogue is the fallback when the sync type is silent.
- **recorder.py** — a clip is the atkdaq stream written verbatim (`.atkq`,
  which keeps every per-frame header) plus a sidecar JSON (geometry, connector
  map, passports, track, headings, references, events, records). `reprocess`
  runs the SAME `Engine` over the clip — the live and file paths are one
  (§11.2) — and a test proves a reprocessed run reproduces the live bearings.
  `to_sigmf` exports a 5-channel SigMF pair (`core:num_channels`, per-channel
  `core:sample_rate`) for ATK's Geolocation tab, the sidecar carrying the
  calibration SigMF cannot.

## 11.11 Build notes — Phase 2 (relief mode)

- **posterior.py** — a per-emitter log-probability grid over a local ENU area
  (25 m default). Each bearing multiplies in its FULL 360-point likelihood
  (rotated to absolute, already widened by heading σ), sampled at each cell's
  bearing from the receiver, so ambiguities and multipath survive as competing
  modes and no one confident-wrong bearing can veto the truth. A `repeated`
  bearing is refused. `age` diffuses a silent emitter by its motion state
  (§8.5); `fix` gives the peak, the centroid and an honest 1σ size.
- **roster.py** — expected IDs, silence thresholds, and a keep-alive learner
  (online mean/variance of each ID's check-in interval) that flags a MISSED
  periodic check-in at 3σ late, and a keys-up-after-silence alert with the
  fleet name. The life-or-death case from the umbrella intro (§DF intro).
- **tracker.py** — a global-nearest-neighbour bearing tracker for emitters
  with no identity, keeping slots apart, so two handsets alternating on one
  channel stay two tracks (§7.4).
- **references.py** — the radio compass (array angle to a catalogued emitter −
  its true bearing = heading + mounting offset), the mounting-offset estimator,
  a per-SNR learned error model (residuals to references become the σ handed to
  `fusion`, replacing the assumption that gave 0% ellipse coverage, §8.8), and
  a catalogue that can learn a repeatedly-seen emitter's position.
- **doa.wideband_spectrum** — the DF waterfall: a (bins × bearing) image, a
  spatial spectrum per FFT bin, each bin steered at its own RF frequency (§6.6).

## 11.12 What remains (Phases 3–4)

Built already from Phase 3: the **measured manifold** (`manifold.MeasuredManifold`,
the single largest vehicle accuracy gain) and the **learned error model**
(`references.ErrorModel`), plus the **kinematic motion indicator**
(`motion.py`). Not yet built, and each waits on something this session does not
have: **pilots** (LTE PSS / DMR sync — needs `atkdsp`), the
**road-constrained particle filter** and **terrain/range likelihoods** (need
ATK's `RoadGraph`/`Elevation`/`PathLoss` ports and DEM tiles), the **planner**,
**specific-emitter fingerprints**, and **RTS smoothing** for reprocessing.
These are Bill-directed once the hardware, the terrain data, and atkdsp's
pilots are in place; the ports they plug into are already defined in `ports.py`
and faked in `fakes.py`.

## 12. Asks from the cued-DF plan (ATK `docs/CUED_DF_PLAN.md`, 2026-10-02)

The two-receiver configuration adds the following to this engine; all
pure Python, all against the simulator first, each a port or a module in
the existing shape.

1. **`ports.CueSource`** — **built 2026-10-02.** `ports.Detection` is the
   cue record (`t0`, `t1 | None` while open, `fc_hz`, `bw_hz`, level, snr,
   optional `key`, `protocol`, `grant`, `signature{…}`), `CueSource.cues()`
   the iterator, `fakes.ListCues` the scripted source (its `replay()` drives
   a scheduler through cues, burst ends and ticks without a clock). ATK
   implements it over the bladeRF's CFAR detector and decoder taps.
   (`CUED_DF_PLAN.md` §5)
2. **`scheduler.py`** — **built 2026-10-02**, 22 tests including a scripted
   relief drive. `Scheduler(config, costs, control)`: `on_cue`,
   `on_burst_end`, `note_bearing`, `set_listener`, `tick`; `Costs` (atkdaq's
   built numbers as defaults, T1b to replace them; `phase_cache` a flag
   T1b opens); `HoldTimes` (per-key median burst length with the ID →
   signature → frequency → protocol → default chain); keys by `key_for`
   (`signature_key`, `freq_key`). Decisions: `stay` (in the window — the
   engine takes it from the ring), `tune`, `preposition` (held until a
   periodic emitter's next burst), `cal`; every cue not acted on is a
   `Missed` with a reason (`mode_off`, `not_priority`, `lower_priority`,
   `gate_failed`). Four parking policies, three cued modes, the gating rule
   with integration bounded by the burst, revisit guarantees that pre-empt
   the dwell, the idle-time cal refresh, a cue nobody closes expiring after
   `OPEN_MAX_S`. Not yet done: wiring to a live `FrameSource.control` and
   to the engine's bearings in `Engine`; reference dwells as scheduled
   actions (§6.6) beyond the cal refresh. (§6)
3. **Energy-clocked spans** in `engine.py`/`identity.py` — a detection's
   `(t0, t1, fc)` addresses the ring exactly as a decoder burst does; DF
   everything is the same path run on the array's own window. (§7)
4. **Provenance tiers** in `calibration.py`/`references.py` — `in_band |
   out_of_band | cached | cached_previous | uncalibrated` from
   `cal_state`, passport age and the reference residuals; the tier widens
   σ through the error model and travels in the passport. (§4.4)
5. **Bias as a track parameter** — `fusion` solves (position, bias) per
   window+passport bias group once a track has geometry; agreement across
   tracks marks a cache entry stale. (§8.3)
6. **Signature association** — `sig:<hash>` keys from the detection
   signature for emitters with no ID, between the registry's ID keys and
   the tracker. (§8.1)
7. **`boxes.py`** — line-of-bearing/box intersection, then posterior mass
   inside; a priority contribution, never a discard; inversion; several
   named boxes. (§9)
8. **Catalogue roles** in `references.py` — `reference | ignore | target`
   with the defaults in `CUED_DF_PLAN.md` §10.
9. **Simulator scenarios** — a two-receiver timing model (listener cue →
   array span), planted per-burst SNR and PTT lengths, a planted window
   bias, references in and out of band — so T5 runs without hardware.
   (§15)
10. **`exports.py`** — **built 2026-10-02**, 24 tests. Live sharing as
    pure functions from a `Picture` (a frozen view: `Vehicle`, `Lob`,
    `Track`, `Box`, `RefMark`) to text: `picture_from(registry, …)` (the
    fix from `fusion`'s ellipse, a one-bearing track as a wedge, a
    repeater-only ID as *heard via repeater* with no place, silence from
    the roster, aged areas from a posterior, the fleet map's `priority`
    and `photo` — two new optional CSV columns in `registry.load_fleet_map`);
    `plain_sentence`, `label_for` (the person, else what the emitter is,
    never a guess); `apply_policy` (`everything | box_only | selected |
    fixes_only` — the last strips every identity string, callsign and
    site id included); `fade_alpha`, `ellipse_ring`, `wedge_ring`;
    `to_kml` (folders, per-placemark fade, `TimeStamp`s, person-first
    names, the sentence above the rule in the balloon), `network_link_kml`,
    `to_geojson`, `records_since(log, offset)` for the merge feed, and
    `to_cot` (stable uids per key, callsign = the person, `stale` from the
    silence threshold, LOBs as range-and-bearing lines). ATK serves and
    pushes these from `atk/core/share_server.py` and `atk/core/cot.py`.
    Not drawn yet: the posterior as a heat image, the coverage footprint.
    (§12A)

## 13. Additions 2026-10-28 — nulling, self-calibration, aircraft calibrators (planned)

Bill, 2026-10-28, after asking whether SVD nulling, CLEAN and spectral
kurtosis belonged on the Kraken, and which radio-astronomy methods could
join them: *"I think the ones that you recommend, we should add. Even if
they are only for specific scenarios, just make sure that when they hover
over that option, that text appears explaining when and how it should be
used."* **Planned, not built.** Each operator option below is off by
default and carries the hover text drafted with it; the DF panel itself
is ATK's (`KRAKEN_DF_PLAN.md`). The detection-side methods are in the
Diffusion Toolkit's `DETECTION_DESIGN.md` §4.6, their kernels in atkdsp's
`DSP_EXPANSION_PLAN.md` Part C.

**Already here — do not duplicate.** `doa.py`'s eigendecomposition-based
estimators (MUSIC, root-MUSIC, Capon/MVDR, ML, FBSS, MDL/AIC counting);
`calibration.CoherenceMonitor` (drift against a reference while parked);
SCORE in the Diffusion Toolkit (`cyclo/filters.py`), which nulls an
interferer that does not share the target's symbol rate.

**Not added: CLEAN for DF.** MUSIC, Capon and ML do not have the wide-beam
sidelobes CLEAN exists to remove, and `ml_alternating` already fits every
source jointly. CLEAN belongs to passive radar (`ATKPR_PLAN.md` §6A.4).

**Not added: closure phase.** Considered and dropped. Around three
antennas each channel's own phase error cancels by construction, so a
closure phase cannot see a stale calibration — the very error it was meant
to catch — and for one emitter it is zero whatever the calibration. What
does carry over from interferometry is self-calibration (§13.2).

### 13.1 Null the strongest, then DF again — `doa.null_and_look`

Estimate the strongest source and project it out: P = I − U_s·U_sᴴ over
its signal-subspace eigenvector(s), or P = I − a·aᴴ/(aᴴa) from its steering
vector when its bearing is known (a repeater, a tower, a jammer the
operator marks). Run the chosen estimator on P·R·P with steering vectors
P·a(θ). Repeat to at most two nulls. Out: the weak bearings, the nulled
bearings, and the null depth measured on the data.

- **Limits, said in the result.** Five elements give four nulls at most,
  one or two useful. Null depth is capped by calibration — a few degrees of
  phase error holds it near 20–30 dB — so the depth is measured and
  reported, and the provenance tier (`cached`, `uncalibrated`) warns.
  Nothing digital recovers a clipped ADC; when atkdaq reports overload the
  result says so. The null removes everything on that bearing, so a grey
  "blind" wedge marks where a weak target behind the interferer cannot be
  seen.
- **Uses.** A weak handset near a strong repeater or tower; a jammer in
  band (GNSS_DENIED_AID_PLAN); DF everything.
- **Accept (sim).** A source 30 dB below a strong one, well separated,
  invisible in the plain spectrum, found after the null with its error
  inside the Cramér–Rao spread; negative control — no weak source, no false
  bearing above the configured rate; a phase-error sweep where measured
  null depth follows the model.

> **Hover:** "Hides the strongest signal so you can find a weaker one on
> the same frequency. **Use it when** a strong transmitter (a repeater, a
> tower, a jammer) swamps the bearing and you think something weaker is
> also there. **How:** tick the box, or right-click the strong bearing →
> *Null this*. ATK removes that direction and DFs again; the new bearings
> appear beside a grey 'blind' wedge where the strong one was — anything
> right behind it cannot be seen. **Works best** soon after a calibration;
> the result says how deep the null got."

### 13.2 Self-calibration from emitters of unknown position — `calibration.self_cal`

Radio astronomy's self-calibration, in its array-processing form
(Weiss–Friedlander blind calibration): from single-source snapshots (high
`eigen_gap`, MDL = 1) of several emitters at different bearings in the same
band — or one emitter as the vehicle turns — solve the per-channel phase
(and gain) errors and the bearings together, by alternating least squares
against the manifold. No known positions needed. It extends
`CoherenceMonitor` to a moving platform and to bands with no catalogued
reference.

- **What it is for.** Checking a cached calibration after a retune or on
  the move: the solved per-channel phases are compared with the passport's,
  and the difference feeds the provenance tier (§12 item 4) and the
  cache-staleness check (§12 item 5). It **offers** a correction; it never
  applies one silently.
- **What it does not solve.** Heading — a common rotation of all bearings
  is still the GPS track's and AUTOCAL's job. A wrong manifold is partly
  absorbed into the solved phases; the measured manifold limits that.
- **Limits.** Needs at least two, better three or more, well-separated
  emitters in the band; coherent multipath breaks the single-source screen
  (`eigen_gap` and the manifold fit catch it).
- **Accept (sim).** Planted per-channel phase errors recovered from emitters
  at unknown bearings to the accuracy the Cramér–Rao bound predicts;
  negative control — a single emitter at one bearing returns "not
  solvable", never a fabricated solution.

> **Hover:** "Checks — and can correct — the Kraken's calibration using
> whatever signals are around, with no known tower needed. **Use it when**
> you are driving, far from known towers, or have retuned and wonder if
> the calibration still holds. **How:** tick the box. It needs at least two
> or three signals from different directions on the band. The DF panel
> shows how far each antenna has drifted and offers a correction if it is
> past the limit. **Cannot help** on an empty band."

### 13.3 Aircraft as moving calibrators — `references.aircraft`

Astronomers map antenna beams with satellites; ADS-B (1090 MHz) and, in
the US, UAT (978 MHz) aircraft broadcast where they are. With the
vehicle's GPS, every position report is a true azimuth **and elevation**,
and aircraft cross every bearing and elevation within minutes near a
flight path. Each burst DF'd against its decoded position measures the
antenna-side manifold — roof, coupling and the elevation response towers
near the horizon cannot give — and checks `MeasuredManifold`. Position is
interpolated to the burst time; low-elevation bursts carry a ground-multipath
flag.

- **Honest scope.** The manifold changes with frequency: this is exact only
  near 1 GHz and a partial check at 450 MHz. It also improves DF of ADS-B
  itself and the Drone Discovery air picture.
- **Research note, not planned.** GPS L1 satellites are known-position
  calibrators everywhere, but sit below the noise and need C/A code
  correlation per satellite; they would calibrate L1 for the GNSS-jammer
  DF case.
- **Accept (sim).** Planted manifold errors recovered from a synthetic
  traffic pattern; a real parked session's residuals reported against the
  model.

> **Hover:** "Uses aircraft as free calibration targets. They broadcast
> exactly where they are, so ATK knows the true direction to each one and
> can measure how your antennas really behave — vehicle roof and all.
> **Use it when** you are parked where aircraft pass (near an airport is
> best), after mounting the array on a new vehicle, or before DF work near
> 1 GHz. **How:** tick the box and leave it running; the panel counts the
> aircraft used and says when there are enough. **Best** around 1 GHz; at
> other frequencies it is a partial check."

### 13.4 Spectral-kurtosis screening of snapshots

Before the covariance is formed, frames whose bins `atkdsp_sk` (C1) flags
as impulsive are dropped — one click or spark can dominate a 5×5
covariance and pull the bearing. The count dropped goes into the quality
dict (`n_snapshots_dropped_sk`).

> **Hover:** "Throws away the moments when a click or spark hits, so they
> do not drag the bearing. **Use it when** bearings jump around near
> engines, power lines or electrical noise. **How:** tick the box; the
> result says how many moments were dropped."

### 13.5 Bearings from the correlation radiometer

The Kraken correlation radiometer (`DETECTION_DESIGN.md` §4.6 item 7) finds
noise-like emitters by their pair cross-correlations; the pair phases are a
rough bearing. `atkdf` takes it as a wide, low-weight line of bearing from a
`ports.Detection` with `signature{"xcorr": …}`, never as a fix on its own.
No separate option — it follows the radiometer's box.

## 14. 2026-10-29 — the P25 DF improvement plan (pointer)

A prospective P25 DF job (3dB Labs' Sceptre as the benchmark; phased arrays
possible) prompted a code review of this engine, atkdaq and ATK's live path
plus a five-year literature sweep. Both live in ATK: `ATK\docs\DF_P25_IMPROVEMENT_PLAN.md`
(findings, the P25-aware chain, validation campaign, what to wire, estimator
and tracking upgrades, the phased-array port, the Sceptre comparison, build
order) and `ATK\docs\DF_LITERATURE_2021_2026.md` (the sweep, with a verdict
per item for this hardware). What that plan asks of THIS repo, in order:
`pilots.py` wired to atkdsp's `DmrSync` (§6.5 here, never written);
burst-level covariance integration across frames in `engine.py`; the P25
path tag by frequency/modulation in `identity.py`; `MeasuredManifold` and
`ErrorModel` on the live path; the channel-grid "DF everything" source once
atkdsp's channelizer lands (`atkdsp\docs\CHANNELIZER_DF_REQUIREMENTS.md`);
then `doa.iaa`, the kr gate on the phase-mode path, the circulant
parametrisation in `calibration.self_cal`, robust mixture likelihoods in
`fusion`, and `timing_source` on the frame header for non-Kraken sources.
Nothing in §0–§13 above changes.

## 15. Build notes — 2026-10-29: Tier 1 of the P25 DF plan (built)

Bill, 2026-10-29: *"Build Tier 1 if you can."* — items 1–4 of
`ATK\docs\DF_P25_IMPROVEMENT_PLAN.md`'s "what we can do now" list. Built and
green: **381 tests** here (343 before + 38 new), and ATK's `test_df_live.py`
(22, 5 new), `test_gnss_heading.py` (7, new) and 311 Qt-free ATK tests that
reach this engine. Not built in atkdsp: nothing there was touched (the
channelizer is being built in parallel; `atkdsp\docs\CHANNELIZER_DF_REQUIREMENTS.md`).

### 15.1 Burst integration — `integrate.py`, `engine.py`

`PlatformMode.integrate` (off by default: the per-frame contract and every
earlier test are unchanged). Each frame's band-limited, calibrated snapshots
(every frame by its OWN header) are summed per transmitter into one Y·Yᴴ;
one record per transmission when it ends — gap (`burst_gap_s`, 0.25 s: a
0.14 s noise-source cal inside a burst does not split it), length cap
(`max_integration_s`: static 5 s, mobile 1 s), a turn (`max_heading_change_deg`:
mobile 5°; integration is in the array frame), a retune, or `Engine.flush()`.
Keys: emitter key + slot (two TDMA slots never share a covariance), else
the RF centre on a 100 Hz grid + bandwidth; the segment is in every key.
While open, `last_status` carries an interim estimate (`interim: True`,
`integrated_ms`, `n_snapshots`) and no record. Records carry `integration`,
`integrated_ms`, `integrated_frames`, `n_contributions`, `burst_end`, the
passports if more than one, band SNR median/max. Measured (sim): a 0.5 s
burst is one bearing with ≥ 9× one frame's snapshots; at +5 dB in channel
the burst's CRB is under a third of a frame's.

### 15.2 The path tag by frequency — `bandplan.py`, `identity.path_evidence`

P25 Phase 1's frame sync is the same both ways, so the evidence chain is now:
sync type (incl. P25 Phase 2 `h-cpm`/`h-dqpsk`, trunking `isp`/`osp`, DMR
direct-mode names) → catalogue by key → `BandPlan` by frequency (a site
`IdenTable`, then the region's split: US 800 −45 MHz, 700 +30 MHz, 900
−39 MHz; EU TETRA 380/390) → catalogue by frequency (`at_frequency`, new on
`references.Catalogue` and `fakes.ListCatalogue`). Talk-around channels
(800 MHz 8CALL90/8TAC91–94, pre- and post-rebanding) answer `unknown`, never
`repeated`. VHF/UHF answer `unknown` with the reason. An unknown region is an
EMPTY plan (a US rule abroad could fuse a repeater as a handset). The reason
is written on the bearing (`quality.path_evidence`). Lesson the test taught:
an `IdenTable`'s raster alone almost never decides — a transmit offset is a
whole number of channel steps (5 MHz = 400 × 12.5 kHz), so a frequency on the
inbound grid is on the outbound grid too; the channels a grant names
(`note_channel`) are what make a site table decisive.

### 15.3 Pilots — `pilots.py`, `_atkdsp.py` (plan §6.5, finally written)

`PilotStage` channelises each target (calibrated at its offset, element
order, mixed on the ABSOLUTE stream index, FFT low-pass overlap-save with
history + lookahead, decimated on a fixed grid — chunk-invariant, tested) to
24 kHz and streams it through atkdsp's `DmrSync` (injected by ATK; found by
`_atkdsp` otherwise — native, else the twin; nothing duplicated). `pilot_doa`
is the known-waveform ML bearing: Bartlett on Σ wₖ v̂ₖ v̂ₖᴴ, weighted by
post-correlation SNR, σ from the CRB ⊕ the manifold prior.

**Two findings the tests forced, both now in the code:**

1. *Partial matches.* Inside a strong 4FSK burst the correlator also reports
   payload partial matches, and with no protocol given the DMR correlator
   finds them in P25 (one read 3 errors — "confirmed"). Harmless to the
   bearing (same signal), dangerous to a LABEL. So `select` keeps the family
   whose syncs read cleanly (≤ 1 error), `mark_periodic` confirms by timing
   (true syncs recur at 180 ms / 30 ms; partials do not — this is what still
   recognises real syncs at low SNR), and the DMR path tag needs two clean
   reads that agree.
2. *Pilot vs covariance.* Measured on clean P25 over 1 s: at −3/−6/−9 dB per
   element the whole-burst covariance was 0.5–1.9° off, the pilot 0.2–6.9°
   — the pilot uses ~25 ms of sync, the covariance all 1000 ms. So the pilot
   is OWNERSHIP, not precision: `Engine._choose` keeps the covariance bearing
   when they agree (`method: pilot-confirmed`), takes the covariance's other
   solution when it sits on the pilot (`pilot-selected` — with a co-channel
   signal 3 dB stronger, Bartlett's first answer was the interferer at 201°
   and the sync picked its solution at 55.6°, truth 60°), and falls back to
   the pilot bearing only when no covariance solution fits (`pilot`).
   `quality.method` is what ATK colours bearings by.

3. *The sync squelch.* An energy gate (ATK's 6 dB squelch) threw away a
   0 dB-per-element handset before the sync word was consulted — exactly
   where the pilot matters. Now a frame whose sync reads cleanly (or
   confirmed) opens the squelch, and it is HELD open `SYNC_HOLD_S` (0.4 s, two
   LDUs) past the last sync, as a P25 receiver holds it: the frames between
   syncs are the same transmission. Measured: energy gate alone, no bearing;
   with the sync squelch, one bearing within 3° over ≥ 800 ms of the 1 s
   transmission (450 ms before the hold). Payload partials do not open it.

### 15.4 The simulator's 4FSK — `sim.fsk4_waveform`, `Emitter.waveform`

`"p25"` (C4FM, the TIA-102 frame sync every 864 symbols / 180 ms LDU),
`"dmr_ms"` (132-symbol bursts with the MS voice sync, one slot, every 60 ms),
`"dmr_bs"` (BS voice sync every 30 ms); raised-cosine frequency pulses,
generated at 48 kHz and resampled, phase-continuous across frames. Checked
against the correlator: P25 syncs every 4 320 samples at 24 kHz, score 0.97,
0 errors; DMR labelled MS/BS correctly. Simplifications, said: no NID/LC/
voice structure, no Phase 2, P25's inverse-sinc shaping omitted (costs SNR in
the correlator, no bearing).

### 15.5 What is still open from the plan's Tier 1–2

Verified only on the simulator: everything above. Real P25/DMR recordings
through the pilot path (`dmr_sync_check.bat` in ATK for the kernel; then a
clip through `reprocess`) are the next check. The channel-grid DF-everything
path waits on atkdsp's channelizer; `MeasuredManifold` / `ErrorModel` on the
live path, the kr gate, IAA, CEP on the map and baseline thinning are Tier 2.

