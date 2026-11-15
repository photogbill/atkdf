"""The LossField port (added 2026-11-14): terrain reaches the posterior as a
field over the whole grid, not one path at a time."""

from __future__ import annotations

import math

import numpy as np

from atkdf import fakes, ports
from atkdf.posterior import GridSpec, Posterior


def test_the_fake_satisfies_the_port_and_matches_the_grid():
    f = fakes.FreeSpaceField()
    assert isinstance(f, ports.LossField)
    spec = GridSpec(ref_lat=38.7, ref_lon=-77.6, half_size_m=5_000.0, cell_m=100.0)
    r = f.field((38.7, -77.6), spec, 450e6, 2.0, 1.5)
    assert isinstance(r, ports.FieldResult)
    assert r.loss_db.shape == Posterior(spec).logp.shape == (spec.n, spec.n)
    assert r.visible.all() and r.loss_db.dtype == np.float32
    # the cell 3 km due east of the receiver: 20 log10(4 pi d f / c)
    c = spec.n // 2
    want = 20 * math.log10(4 * math.pi * 3000.0 * 450e6 / 299_792_458.0)
    assert abs(float(r.loss_db[c, c + 30]) - want) < 0.01
    # rows run north: the cell 3 km NORTH is the same distance, in row c + 30
    assert abs(float(r.loss_db[c + 30, c]) - want) < 0.01


def test_the_wall_hides_one_sector_beyond_its_range():
    f = fakes.FreeSpaceField(wall_bearing_deg=0.0, wall_range_m=2_000.0,
                             wall_width_deg=40.0, wall_loss_db=30.0)
    spec = GridSpec(ref_lat=38.7, ref_lon=-77.6, half_size_m=5_000.0, cell_m=100.0)
    r = f.field((38.7, -77.6), spec, 450e6, 2.0, 1.5)
    c = spec.n // 2
    assert not r.visible[c + 40, c]            # 4 km north: behind the wall
    assert r.visible[c + 10, c]                # 1 km north: in front of it
    assert r.visible[c - 40, c] and r.visible[c, c + 40]     # south and east: clear
    clear = fakes.FreeSpaceField().field((38.7, -77.6), spec, 450e6, 2.0, 1.5)
    assert abs(float(r.loss_db[c + 40, c] - clear.loss_db[c + 40, c]) - 30.0) < 1e-3
