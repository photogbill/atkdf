"""DF everything (channel_grid.py): every channel that lights up in the window
gets one bearing per transmission, through the same estimator and record as
a band burst — no decoder, no retune.

Runs on the simulator. The CPU backend is atkdsp's PFB (native, else its
numpy twin), so these need atkdsp beside this repo; ATK's GPU backend is a
drop-in for the same port, tested in ATK (tests/test_df_grid_gpu.py).
"""

from __future__ import annotations

import pytest

from atkdf import _atkdsp, engine, sim
from atkdf import channel_grid as G
from atkdf import manifold as M

pytestmark = pytest.mark.skipif(_atkdsp.module() is None, reason="needs atkdsp beside this repo")


def err(a, t):
    return ((a - t + 180) % 360) - 180


def scenario(seed=3):
    return sim.Scenario(emitters=[
        # two handhelds on the 12.5 kHz raster, keyed in turn
        sim.Emitter(bearing_deg=70.0, f0_hz=300_000.0, snr_db=15.0, period_s=1.0, duty=0.4),
        sim.Emitter(bearing_deg=215.0, f0_hz=-412_500.0, snr_db=15.0, period_s=1.0, duty=0.4, phase0=0.5),
    ], seed=seed)


def run(sc, frames=40, grid_kw=None, accel=None):
    s = sim.Simulator(sc, frames=frames)
    grid = G.ChannelGrid(fs=float(sc.fs_hz), nel=sc.n_ch, accel=accel, **(grid_kw or {}))
    eng = engine.Engine(geometry=s.geometry, manifold=M.IdealManifold(s.geometry),
                        positioning=s.positioning, mode=engine.PlatformMode.static(),
                        site_id="alpha", grid=grid)
    recs = []
    for fr in s.frames():
        recs += eng.process(fr)
    recs += eng.flush("end")
    return s, eng, grid, recs


def by_channel(recs, fc, f_off, bw=12_500.0):
    return [r for r in recs if abs(r.fc_hz - (fc + f_off)) < bw / 2]


def test_the_channel_count_is_fast_and_on_the_raster():
    g = G.ChannelGrid(fs=2_400_000.0, nel=5)
    assert g.nchan == 192 and g.spacing_hz == pytest.approx(12_500.0)
    assert g.offset_hz(24) == pytest.approx(300_000.0)
    assert g.offset_hz(192 - 33) == pytest.approx(-412_500.0)
    assert G.fast_count(614) in (600, 616, 630) or G.fast_count(614) == 600


def test_every_keyed_channel_gets_its_own_bearing_with_no_decoder():
    sc = scenario()
    s, eng, grid, recs = run(sc)
    a = by_channel(recs, sc.fc_hz, 300_000.0)
    b = by_channel(recs, sc.fc_hz, -412_500.0)
    assert a and b, [round(r.fc_hz - sc.fc_hz) for r in recs]
    for r in a:
        assert abs(err(r.theta_rel_deg, 70.0)) < 3.0
        assert r.quality["grid_channel"] == 24 and r.quality["method"].startswith("DF everything")
    for r in b:
        assert abs(err(r.theta_rel_deg, 215.0)) < 3.0
    # one record per transmission, not one per frame
    assert len(a) <= 4 and len(b) <= 4
    # nothing anywhere else
    assert len(recs) == len(a) + len(b)


def test_a_quiet_band_gives_no_bearings():
    sc = sim.Scenario(emitters=[], seed=4)
    s, eng, grid, recs = run(sc, frames=20)
    assert recs == []


class _SecondCpu:
    """An `accel` that swaps in another CPU backend after a while — the same
    port ATK's GPU backend plugs into — and records what it was told."""

    def __init__(self, after_blocks=12):
        self.after = after_blocks
        self.polls = 0
        self.made = None
        self.failures = []

    def poll(self, now, load, fs, nchan, nel, block, current):
        self.polls += 1
        if self.made is None and self.polls > self.after:
            self.made = G.CpuGridBackend(fs, nchan, nel)
            self.made.name = "second CPU"
            return self.made
        return None

    def failed(self, why):
        self.failures.append(why)


def test_switching_backends_mid_run_keeps_the_bearings():
    sc = scenario(seed=6)
    acc = _SecondCpu()
    s, eng, grid, recs = run(sc, accel=acc)
    assert grid.backend is acc.made
    a = by_channel(recs, sc.fc_hz, 300_000.0)
    b = by_channel(recs, sc.fc_hz, -412_500.0)
    assert a and b
    assert all(abs(err(r.theta_rel_deg, 70.0)) < 3.0 for r in a)
    assert all(abs(err(r.theta_rel_deg, 215.0)) < 3.0 for r in b)
    assert any(r.quality["grid_backend"] == "second CPU" for r in recs)


class _Broken(G.CpuGridBackend):
    def __init__(self, *a, fail_at=5, **k):
        super().__init__(*a, **k)
        self.name = "broken card"
        self.n = 0
        self.fail_at = fail_at

    def process(self, x_el):
        self.n += 1
        if self.n >= self.fail_at:
            raise RuntimeError("card fell over")
        return super().process(x_el)


class _GivesBroken(_SecondCpu):
    def poll(self, now, load, fs, nchan, nel, block, current):
        self.polls += 1
        if self.made is None and self.polls > 2:
            self.made = _Broken(fs, nchan, nel)
            return self.made
        return None


def test_a_failing_backend_falls_back_to_the_cpu_and_never_raises():
    sc = scenario(seed=7)
    acc = _GivesBroken()
    s, eng, grid, recs = run(sc, accel=acc)
    assert acc.failures and "card fell over" in acc.failures[0]
    assert grid.backend.name.startswith("CPU")
    assert by_channel(recs, sc.fc_hz, 300_000.0) or by_channel(recs, sc.fc_hz, -412_500.0)


def test_a_new_sample_rate_rebuilds_the_grid_and_keeps_its_settings():
    acc = _SecondCpu(after_blocks=10 ** 9)
    g = G.ChannelGrid(fs=2_400_000.0, nel=5, open_db=12.0, accel=acc)
    h = g.rebuild(1_200_000.0)
    assert h.nchan == 96 and h.spacing_hz == pytest.approx(12_500.0)
    assert h.open_db == 12.0 and h.accel is acc


def test_back_to_the_cpu_when_the_accelerator_says_so():
    class Flip(_SecondCpu):
        def poll(self, now, load, fs, nchan, nel, block, current):
            self.polls += 1
            if self.polls == 5:
                self.made = G.CpuGridBackend(fs, nchan, nel)
                self.made.name = "the card"
                return self.made
            return "cpu" if self.polls == 15 else None
    sc = scenario(seed=9)
    acc = Flip()
    s, eng, grid, recs = run(sc, frames=30, accel=acc)
    assert grid.backend is not acc.made and grid.backend.name.startswith("CPU")
    assert all(abs(err(r.theta_rel_deg, 70.0)) < 3.0 for r in by_channel(recs, sc.fc_hz, 300_000.0))


def test_the_independent_snapshot_count_is_half_the_hops():
    sc = scenario(seed=8)
    s, eng, grid, recs = run(sc, frames=24)
    assert recs
    for r in recs:
        hops = int(r.notes[-1].split(" hops")[0].rsplit(" ", 1)[-1]) if any(
            "hops" in n for n in r.notes) else None
        if hops:
            assert r.quality["n_snapshots"] == hops // 2
