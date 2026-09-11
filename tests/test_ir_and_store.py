"""M2: IR sampling/building and the Zarr v3 deformation store."""

import numpy as np
import pytest
import torch

from cets_nonrigid.ir.build import build_ir_tilt_series
from cets_nonrigid.ir.core import IRMeta
from cets_nonrigid.ir.sampling import (
    boundary_ramp_weights,
    heldout_points,
    image_grid,
    volume_grid,
)
from cets_nonrigid.models.aretomo_ts import AreTomoLocalField, AretomoTsModel

RNG = np.random.default_rng(7)


def _model(t=5, p=4):
    coord = torch.tensor(RNG.uniform(-300, 300, size=(t, p, 2)))
    shift = torch.tensor(RNG.uniform(-5, 5, size=(t, p, 2)))
    good = torch.ones(t, p)
    local = AreTomoLocalField(coord, shift, good, torch.tensor([800, 700]))
    return AretomoTsModel(
        rot_deg=torch.tensor(RNG.uniform(-10, 10, size=t)),
        tilt_deg=torch.linspace(-60, 60, t),
        shifts_px=torch.tensor(RNG.uniform(-20, 20, size=(t, 2))),
        raw_size_px=(800, 700),
        pixel_size_a=2.0,
        volume_dims_a=(1600.0, 1400.0, 600.0),
        local=local,
    )


def _meta(t=5):
    return IRMeta(
        kind="tilt_series",
        series_name="test_series",
        pixel_size_image_a=2.0,
        image_dims_px=(800, 700),
        volume_dims_px=(800, 700, 300),
        pixel_size_volume_a=2.0,
        projection_index=list(range(t)),
        projection_valid=[True] * t,
        projection_order=list(range(t)),
        projection_dose=[3.0 * i for i in range(t)],
        projection_angle_deg=list(np.linspace(-60, 60, t)),
        projection_sec=[i + 1 for i in range(t)],
        projection_dark=[False] * t,
        source_tool="aretomo3",
    )


def test_volume_grid_covers_borders():
    dims = torch.tensor([100.0, 200.0, 50.0])
    pts = volume_grid(dims, (3, 4, 2))
    assert pts.shape == (24, 3)
    for ax, hi in enumerate([100.0, 200.0, 50.0]):
        assert pts[:, ax].min().item() == 0.0
        assert pts[:, ax].max().item() == hi


def test_heldout_is_not_nested_in_regular_grid():
    dims = torch.tensor([100.0, 100.0, 100.0])
    regular = volume_grid(dims, (5, 5, 5))
    held = heldout_points(dims, 100, seed=1)
    dmin = torch.cdist(held, regular).min()
    assert dmin > 1e-3  # Sobol points don't coincide with grid nodes
    held2 = heldout_points(dims, 100, seed=1)
    torch.testing.assert_close(held, held2)  # deterministic per seed


def test_boundary_ramp():
    dims = torch.tensor([100.0, 100.0])
    pos = torch.tensor([[50.0, 50.0], [0.0, 0.0], [-2.5, 50.0], [-5.0, 50.0], [-20.0, 50.0]])
    w = boundary_ramp_weights(pos, dims, margin_frac=0.05)
    assert w[0] == 1.0 and w[1] == 1.0  # inside incl. the edge
    assert 0.0 < w[2] < 1.0  # inside the outside-margin band
    assert w[3] == pytest.approx(0.0, abs=1e-6)  # margin end
    assert w[4] == 0.0  # far outside


def test_build_ir():
    model = _model()
    ir = build_ir_tilt_series(
        model,
        volume_dims_a=model.volume_dims_a,
        image_dims_a=model.raw_size_px * model.pixel_size_a,
        meta=_meta(),
        grid_shape=(7, 7, 3),
    )
    assert ir.points.shape == (147, 3)
    assert ir.source_projected.shape == (5, 147, 2)
    assert ir.heldout_points.shape[0] >= 64
    # Full vs global-only projections differ (locals nonzero).
    assert (ir.source_projected - ir.source_projected_global).abs().max() > 0.01
    assert torch.isfinite(ir.source_projected).all()












def test_image_grid():
    pts = image_grid(torch.tensor([400.0, 300.0]), (4, 3))
    assert pts.shape == (12, 2)
    assert pts[:, 0].max() == 400.0 and pts[:, 1].max() == 300.0
