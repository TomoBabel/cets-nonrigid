"""M4: fitting Warp movement grids to an IR.

Self-consistency gate (scoped to representable models): a movement-only Warp
source, fitted back into a same-dims fresh movement grid, must reproduce the
source projections to < 1e-2 px on the held-out set.
"""

import copy

import pytest
import torch

from cets_nonrigid.fit.warp_ts_fit import fit_warp_movement, movement_grid_dims
from cets_nonrigid.io.warp_xml import load_warp_tiltseries
from cets_nonrigid.ir.build import build_ir_tilt_series
from cets_nonrigid.ir.core import IRMeta


def _meta(t):
    return IRMeta(
        kind="tilt_series",
        series_name="ts1",
        pixel_size_image_a=1.7,
        image_dims_px=(2826, 2008),
        volume_dims_px=(1963, 2796, 491),
        pixel_size_volume_a=1.7,
        source_tool="warp",
    )


@pytest.fixture(scope="module")
def ts1_ir_and_series(ts1_xml_path_module):
    series = load_warp_tiltseries(ts1_xml_path_module)
    ir = build_ir_tilt_series(
        series.model,
        volume_dims_a=series.model.volume_dims_a,
        image_dims_a=series.model.image_dims_a,
        meta=_meta(series.n_tilts),
    )
    return series, ir


def test_movement_grid_defaults():
    assert movement_grid_dims((4, 4)) == (4, 4)
    assert movement_grid_dims((2, 3)) == (4, 4)
    assert movement_grid_dims((9, 6)) == (8, 6)


def test_self_consistency_same_dims(ts1_ir_and_series):
    series, ir = ts1_ir_and_series
    src_dims = series.ts.grid_movement_x.dimensions  # (6, 4, 41) in TS_1
    target = copy.deepcopy(series.ts)

    result = fit_warp_movement(
        ir, target, movement_grid=(src_dims[0], src_dims[1]), lam=0.0
    )
    # Same family, same dims: exact linear LSQ recovers the model up to
    # float32 forward noise. 1e-2 px at 1.7 A/px = 0.017 A.
    assert result.rms_a_heldout < 0.017, result.rms_a_heldout
    assert result.rms_a_train < 0.017
    assert result.min_rank == src_dims[0] * src_dims[1]
    assert result.coverage_heldout > 0.5

    # Recovered node values match the source grids (same gauge: identical
    # globals), up to spline-fit noise.
    src_x = series.ts.grid_movement_x.values
    fit_x = result.ts.grid_movement_x.values
    assert (src_x - fit_x).abs().max() < 0.1  # Angstrom


def test_smaller_grid_is_lossy_but_reported(ts1_ir_and_series):
    series, ir = ts1_ir_and_series
    target = copy.deepcopy(series.ts)
    result = fit_warp_movement(ir, target, movement_grid=(4, 4), lam=1e-3)
    # (6,4) source into (4,4) target: representation error appears and the
    # report captures it; still far below the raw field magnitude (~46 A).
    assert 0.017 < result.rms_a_heldout < 10.0
    assert result.p95_a_heldout >= result.rms_a_heldout * 0.5
    assert torch.isfinite(result.per_tilt_rms_a_heldout).all()
    assert result.max_condition > 0 and result.regularization_norm >= 0


def test_disabled_tilt_slice_stays_zero(ts1_ir_and_series):
    series, ir = ts1_ir_and_series
    target = copy.deepcopy(series.ts)
    target.use_tilt = target.use_tilt.clone()
    target.use_tilt[5] = False
    ir2 = copy.deepcopy(ir)
    ir2.projection_valid[5] = False

    result = fit_warp_movement(ir2, target, movement_grid=(4, 4))
    n = 16
    assert (result.ts.grid_movement_x.values[5 * n : 6 * n] == 0).all()
