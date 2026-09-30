"""How fast a bearing is, and how accurate vs SNR — the numbers to watch when
the estimators change. Runs on the simulator, no hardware."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from atkdf import sim, engine, manifold as M

def err(a, t): return ((a - t + 180) % 360) - 180

def accuracy():
    print("SNR(dB)  RMS error(deg) over 20 frames, UCA-5, MUSIC, band snapshots")
    for snr in (0, 5, 10, 20, 30):
        errs = []
        sc = sim.Scenario(emitters=[sim.Emitter(bearing_deg=73.0, f0_hz=300_000.0,
                                                snr_db=snr, protocol="dmr", system="1",
                                                id="1", sync_type="ms")], seed=1)
        s = sim.Simulator(sc, frames=24)
        eng = engine.Engine(geometry=s.geometry, manifold=M.IdealManifold(s.geometry),
                            positioning=s.positioning)
        by = {}
        for e in s.events(): by.setdefault(e.n0, []).append(e)
        for fr in s.frames():
            for r in eng.process(fr, events=by.get(fr.header.stream_sample0, [])):
                errs.append(err(r.theta_rel_deg, 73.0))
        rms = float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")
        print(f"  {snr:3d}      {rms:6.2f}   (n={len(errs)})")

def throughput():
    sc = sim.Scenario(emitters=[sim.Emitter(bearing_deg=73.0, f0_hz=300_000.0, snr_db=20.0,
                                            protocol="dmr", system="1", id="1", sync_type="ms")], seed=1)
    s = sim.Simulator(sc, frames=40)
    eng = engine.Engine(geometry=s.geometry, manifold=M.IdealManifold(s.geometry),
                        positioning=s.positioning)
    frames = list(s.frames())
    by = {}
    for e in s.events(): by.setdefault(e.n0, []).append(e)
    t0 = time.monotonic()
    n = 0
    for fr in frames:
        n += len(eng.process(fr, events=by.get(fr.header.stream_sample0, [])))
    dt = time.monotonic() - t0
    print(f"\n{len(frames)} frames -> {n} bearings in {dt*1000:.0f} ms "
          f"({dt/len(frames)*1000:.1f} ms/frame; a frame is 50 ms of air)")

if __name__ == "__main__":
    accuracy(); throughput()
