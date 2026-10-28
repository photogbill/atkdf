# atkdf — Automated calibration sequence: design note

> **Status (2026-09-30):** design only, no code yet. This is the orchestration
> layer over machinery that already exists and is green: it does not replace
> `references.py`, `calibration.py`, `heading.py` or `identity.py` — it drives
> them. New work is the bundled reference catalogue (cell towers, multi-RAT
> identity, provenance tiers), the multi-reference *consensus* solve across
> several bands at once, and the frequency-dependent manifold table that falls
> out of it. Sim-first like the rest of the engine (§7); needs no Kraken to
> build. Companion to `ATKDF_PLAN.md`; where they disagree about what is
> *inside* atkdf, the plan wins.
>
> **Scope correction (Bill, 2026-09-30):** DF calibration references are bounded
> by the KrakenSDR's tuning range (~24 MHz–1.7 GHz, R820T2). The 3.5 GHz band is
> beyond the array and is **survey-only** on wider hardware — never a DF cal
> reference. An earlier sketch put references at 3.5 GHz; that was wrong.

---

## 1. What this builds on (already in the tree — do not duplicate)

| symbol | file | what it already does |
|---|---|---|
| `radio_compass()` | `references.py` | array angle to a catalogued emitter − true bearing = heading + mounting offset; folds the reference's `position_sigma_m` into σ |
| `MountingOffset` | `references.py` | array-forward vs platform axis, circular mean of (compass − GPS course) while moving straight |
| `ErrorModel` | `references.py` | per-SNR RMS residual to references → the σ handed to `fusion` (replaces the assumption that gave 0% ellipse coverage) |
| `Catalogue.learn()` | `references.py` | promotes a repeatedly-seen unknown into an entry once bearings fix it (§5.3, "grows from your drives") |
| `CoherenceMonitor` | `calibration.py` | parked reference's steering vector drifts ⇒ the receivers' cal has drifted ⇒ recal **when the reference says so, not on a timer** |
| `calibration.apply()` | `calibration.py` | applies atkdaq's per-channel weight **and** fractional-delay phase slope — the receiver-chain frequency dependence is already handled here |
| `cal_age_verdict()` | `calibration.py` | fresh/aging/stale by platform mode; the thermal rationale ("tuner phases drift with temperature") is already the reason for the thresholds |
| `HeadingFilter.update_compass()` | `heading.py` | fuses a compass heading at a stop or in a turn; σ grows off it, shrinks on it |
| `path_tag_of()` | `identity.py` | tags a repeater's outbound burst `repeated` so it is never fused as the handset's position |
| `Catalogue` / `Reference` | `ports.py` | `lookup(key)`, `near(lat,lon,r)`; `Reference(key, lat, lon, kind, fc_hz, position_sigma_m, verified, source)` |

The single-reference radio compass is done. What is missing is the **automation
and the numbers**: many references at once, across bands, from a catalogue dense
enough to be everywhere, with the receiver cal separated cleanly from the
antenna/platform manifold.

## 2. The idea, in one line

ATK is GPS-synced and holds the tower catalogue, so at any instant it knows the
*true* bearing (and range, and band) to every known emitter around it. The
apparent AOA minus the true bearing, stacked over many towers and several bands,
solves heading + mounting offset + the antenna-side manifold — continuously,
automatically, with no turntable and no staged single emitter. The receiver
electrical path is atkdaq's job (`calibration.apply`); this sequence calibrates
everything *outboard* of the connector — the antennas, the mutual coupling, the
roof — plus heading, and it uses `CoherenceMonitor` as the trigger.

## 3. The reference catalogue

A `Catalogue` (the port already exists) populated from a **bundled, offline,
provenance-tiered** cell-tower database, pre-loaded per region exactly like the
terrain stack: a coarse world base, fine data where it was pulled down.

- **Sources.** US dev/test: FCC ASR (surveyed structure coordinates, meters,
  bulk-downloadable). Overseas (the real mission): OpenCelliD global CSV, plus
  in-country regulator data where it exists. New `references.import_fcc` /
  `import_opencellid` produce `Reference` rows keyed by cell identity.
- **Provenance maps onto fields that already exist.** Surveyed ⇒ `verified=True`,
  small `position_sigma_m`. Crowdsourced centroid ⇒ `verified=False`, large
  `position_sigma_m`, `source="opencellid"`. Self-fixed ⇒ `source="learned"`
  (via `Catalogue.learn`). `radio_compass` already inflates σ by
  `position_sigma_m` at range, so a crowdsourced reference *automatically*
  counts for less — the provenance tier needs no new mechanism, only honest
  `position_sigma_m` at import. **A crowdsourced location is a candidate, never
  a truth to calibrate against.**
- **Association = cell identity.** `Catalogue.lookup` takes an `EmitterKey`
  (`identity.py`); the decode supplies it. LTE identity is done. Add GSM (BSIC,
  ARFCN, CI/LAC), UMTS (PSC, UARFCN, CID/LAC), 5G NR (PCI, SSB/GSCN, NCGI/TAC).
  This decode is **dual-use** with the passive cell-identity survey — one
  decoder, two features, all in the clear.
- **A tower is not an emitter.** One structure hosts many carriers, bands and
  three sectors; horizontally they are co-located on the same mast, so the
  structure location is a valid *azimuth* reference for any sector once the
  signal is associated to it.

## 4. Band scope — hardware-bounded

DF cal references live inside the Kraken's tuning range (~24 MHz–1.7 GHz):

- **Usable as DF cal references (sub-1.7 GHz):** GSM-450, GSM-850, E-GSM-900;
  UMTS 850/900; LTE 600–960 and the ~1.5 GHz band (incl. the China
  frequency-shift repeaters Bill flagged); low-band 5G NR (n5/n8/n28/n71).
- **Survey-only (above ~1.7 GHz), on wider hardware (bladeRF to ~3.8 GHz):**
  DCS-1800 (DL 1805–1880), PCS-1900 (DL 1930–1990), UMTS/LTE 2100, mid-band
  5G NR (3.3–3.8 GHz). Decode for the survey catalogue; **do not** treat as cal
  references — the array cannot hear them.
- **Why multi-band matters for cal.** The receiver-chain frequency dependence is
  already corrected (atkdaq weight + frac-delay slope, `calibration.apply`).
  What multi-band references add is the **antenna-side** manifold across
  frequency — mutual coupling and element patterns change with band, and a
  single-band cal only trues the array at that band. References spread across
  450–1700 MHz build the manifold cal across the span actually DF'd. The older
  continuously-broadcast RATs (GSM BCCH, UMTS CPICH) are the *cleaner* cal
  references; 5G is TDD and beam-swept (DF the SSB burst) and is weighted below
  them.

## 5. The sequence

**Trigger** (event-driven, not a fixed loop):
- cold start / arrival — one pass to seed the manifold from whatever is in view;
- opportunistic — a trusted reference already in the current passband folds in
  for free (no retune);
- scheduled excursion — a brief retune to an under-covered band, only when
  `PlatformMode` can spare the dwell;
- drift — `CoherenceMonitor.needs_cal`, or `cal_age_verdict` past "aging";
- geometry — enough new azimuth spread from the platform's motion to re-solve
  better-conditioned.

**Solve.** For each in-view, LOS (terrain-gated via the `Elevation`/`PathLoss`
ports when present), *trusted* reference: a `radio_compass` observation (heading
+ offset) and a steering-vector residual (manifold). Stack across azimuth and
band, weight by `position_sigma_m` and by `ErrorModel.sigma_for(snr)`, and feed
`heading.update_compass` + the manifold/error model. Identifiability: ≥2–3
azimuth-spread references separate heading from a common azimuth bias; a single
reference gives heading only against the last good manifold (which is exactly
how `radio_compass` + `MountingOffset` behave today).

**Outlier rejection.** Robust consensus (RANSAC / M-estimator): the cal the
majority agree on; a reference that dissents — wrong DB location, NLOS, or a
mis-association — is down-weighted or dropped. Verified references anchor the
consensus and can promote a crowdsourced one that agrees (and refine its
location, `Catalogue.learn`). **On a leash:** the solution is bounded relative
to the trusted manifold; a correction that wants to be huge is a hardware fault
or a bad catalogue, and is raised, not applied.

**Frequency table.** Maintain a per-band antenna-manifold correction; interpolate
between observed bands (the receiver frac-delay slope is already parametric in
atkdaq, so only the smooth antenna/coupling residual is tabled). Refresh entries
on drift. Always leave a usable manifold at the *target* frequency even with no
reference sitting exactly on it.

## 6. Thermal

Two separate things; keep them apart.

- **RF-path / Kraken thermal sensing — deferred (Bill, later).** Note it is
  already handled *reactively*: `cal_age_verdict` ages a cal on the
  temperature-drift rationale, and `CoherenceMonitor` catches the drift directly
  off a parked reference and asks for recal. The *feedforward* thermal model
  (learn dφ/dT, dΔτ/dT to pre-compensate between reference opportunities) is
  future work, gated on the added tuner/cable temperature sensors. Until then,
  the on-air monitor is the drift authority.
- **CPU/GPU platform temperature — an ATK-side operator feature (not this
  engine).** Read CPU and GPU temperature, surface it to the operator, warn on
  throttle and shutdown risk. Field heat kills laptops (Iraq); the operator has
  to see the machine cooking. Engine-relevant hook: if the GPU throttles, the
  optional torch backend slows and the low-latency path degrades — the operator
  should see *why* real-time DF slowed. Recorded here as a decision; the
  implementation belongs in ATK (`docs/KRAKEN_DF_PLAN.md`), which is not part of
  this repo.

## 7. Acceptance (sim-first, known-answer)

`sim.py` already plants reference emitters, a manifold perturbation (per-element
phase/gain, mutual-coupling matrix) and a `Catalogue` — the spine is in place.

- **Recovery.** A scenario with N references across azimuth and 2–3 bands, a
  planted manifold perturbation, and a planted heading: the sequence recovers
  heading to the plan's Phase-2 number and the recovered manifold beats the
  ideal on the target bearing.
- **Provenance.** A reference with a large `position_sigma_m` (crowdsourced)
  does not pull the solve; a `verified` one does.
- **Outlier.** A planted wrong-location decoy reference is rejected by consensus;
  the fix does not move to it.
- **Cross-band.** A cal built on band A improves a target bearing in band A; the
  interpolated manifold is usable in band B with no reference exactly on it.
- Known-answer first, residual plausibility is not a pass (the plan's rule).

## 8. Where it plugs into the core loop

In `ATKDF_PLAN.md` §3, the step *"references: radio compass, coherence monitor,
error-model residuals"* becomes *"references: automated multi-reference cal
sequence"* — `Catalogue.near()` for the in-view towers, identity decode for
association, the consensus solve, then the same `heading` / manifold / ErrorModel
sinks it already feeds. Nothing downstream changes; the sequence is a richer
producer for ports that already exist.

## 9. Open / Bill-directed

- Multi-RAT identity decode: GSM and UMTS first (overseas, and the cleanest cal
  references), then 5G NR; CDMA2000 skipped (WLL niche).
- RF thermal sensors and the feedforward thermal model — after the Kraken is in
  hand.
- The excursion scheduler (when to step off-target to a reference band) is a
  `PlatformMode` policy; default to opportunistic-only until measured.
- Moving and unknown-position references (2026-10-28, planned): aircraft
  (ADS-B 1090 / UAT 978) as calibrators with true azimuth and elevation, and
  self-calibration from emitters of unknown position — `ATKDF_PLAN.md`
  §13.2–13.3. They feed the same sinks as this sequence; nothing here
  changes.
