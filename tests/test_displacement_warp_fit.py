"""Step 3 of the 3D-displacement plan: the Warp volume-warp fit.

Numbers quoted in the plan and reproduced here: golden 3x3x2x4 recovery ~1e-6 A,
L = T = 41 full rank (738 parameters), 2D held-out 1.22 A movement-only vs
~1e-4 A with the volume warp; the reviewer's irregular-dose rank-4-of-5 case
and the two-node midpoint case are REJECTED; the 41-dose dark-row case passes
through the temporal extension.
"""

from __future__ import annotations

import copy
import warnings

import pytest
import test_displacement_ir_store as h
import torch
from warpylib import CubicGrid, LinearGrid4D, TiltSeries

from cets_nonrigid.fit.coverage import node_support_3d
from cets_nonrigid.fit.warp_ts_fit import fit_warp_locals, fit_warp_movement, fit_warp_volume_warp
from cets_nonrigid.io.warp_xml import load_warp_tiltseries
from cets_nonrigid.ir.build import build_ir_tilt_series, build_ir_tilt_series_from_points
from cets_nonrigid.ir.core import IRMeta
from cets_nonrigid.models.warp_ts import WarpTiltSeriesModel

F64 = torch.float64


def _reset_locals(ts):
    ts = copy.deepcopy(ts)
    ts.grid_movement_x = CubicGrid((1, 1, 1))
    ts.grid_movement_y = CubicGrid((1, 1, 1))
    ts.grid_volume_warp_x = LinearGrid4D((1, 1, 1, 1))
    ts.grid_volume_warp_y = LinearGrid4D((1, 1, 1, 1))
    ts.grid_volume_warp_z = LinearGrid4D((1, 1, 1, 1))
    return ts


@pytest.fixture(scope="module")
def golden():
    s, ir = h._grid_ir(h.GOLDEN_XML, grid=(15, 15, 5))
    return s, ir, _reset_locals(s.ts)


# ---------------------------------------------------------------------------
# golden self-consistency
# ---------------------------------------------------------------------------


def test_golden_fit_recovers_source_nodes_and_observables(golden):
    s, ir, ts_target = golden
    ts_fit, vw = fit_warp_volume_warp(ir, ts_target, grid=(3, 3, 2, 4))
    assert vw.data_rank == vw.n_params_supported == 72 and vw.n_unsupported_slices == 0
    assert vw.dose_collisions == 0 and vw.node_support == 1.0
    for name in ("x", "y", "z"):
        a = getattr(ts_fit, f"grid_volume_warp_{name}").values.to(F64)
        b = getattr(s.ts, f"grid_volume_warp_{name}").values.to(F64)
        assert (a - b).abs().max() <= 1e-3, f"node recovery {name}: {(a - b).abs().max():.3e} A"  # float32 XML values
    assert vw.heldout_status == "evaluated"
    assert vw.rms_a_heldout <= 1e-4 and vw.rms_a_heldout_inplane <= 1e-4 and vw.rms_a_heldout_beam <= 1e-4
    # Warp -> Warp with identical globals/centres/hand: the CTF-depth deviation IS the beam residual
    assert vw.ctf_depth_deviation_rms_a_heldout is not None
    assert abs(vw.ctf_depth_deviation_rms_a_heldout - vw.rms_a_heldout_beam) <= 1e-9
    assert vw.max_vw_first_plane_a > 5.0  # M would hold this nonzero plane fixed: reported, not imposed


def test_golden_fit_L_equals_T_is_full_rank(golden):
    _s, ir, ts_target = golden
    _ts, vw = fit_warp_volume_warp(ir, ts_target, grid=(3, 3, 2, 41))
    assert vw.n_params == 738 and vw.data_rank == vw.n_params_supported == 738
    assert vw.data_condition < 10
    assert vw.rms_a_heldout <= 1e-4 and vw.rms_a_heldout_beam <= 1e-4


def test_locals_with_volume_warp_beat_movement_only(golden):
    _s, ir, ts_target = golden
    full = fit_warp_locals(ir, ts_target, movement_grid=(6, 4), volume_warp_grid=(3, 3, 2, 4))
    mo = fit_warp_locals(ir, ts_target, movement_grid=(6, 4))
    assert full.volume_warp is not None and mo.volume_warp is None
    assert full.meta["volume_warp"].startswith("fitted 3x3x2x4")
    assert mo.meta["volume_warp"].startswith("zero")
    pix = ir.meta.pixel_size_image_a
    assert full.rms_a_heldout / pix <= 1e-2, f"{full.rms_a_heldout / pix:.4f} px"
    assert mo.rms_a_heldout > 100 * full.rms_a_heldout  # 1.22 A vs 1e-4 A on this source
    # the movement grids were fitted at the PREMOVEMENT positions of the fitted warp
    assert full.meta["movement_baseline"] == "premovement"


def test_movement_only_path_is_unchanged_without_volume_warp(golden):
    """fit_warp_locals(volume_warp_grid=None) == fit_warp_movement on the given target."""
    _s, ir, ts_target = golden
    a = fit_warp_locals(ir, ts_target, movement_grid=(5, 5))
    b = fit_warp_movement(ir, ts_target, movement_grid=(5, 5))
    assert torch.equal(a.ts.grid_movement_x.values, b.ts.grid_movement_x.values)
    assert torch.equal(a.ts.grid_movement_y.values, b.ts.grid_movement_y.values)
    assert a.rms_a_heldout == b.rms_a_heldout


# ---------------------------------------------------------------------------
# refusals and identifiability
# ---------------------------------------------------------------------------


def test_refuses_ir_without_displacement(golden):
    _s, _ir, _ts_target = golden
    _s0, ir0 = h._grid_ir(h.DEGENERATE_XML)
    with pytest.raises(ValueError, match="carries no 3D displacement.*zero_at_samples"):
        fit_warp_volume_warp(ir0, _reset_locals(_s0.ts), grid=(3, 3, 2, 4))


def test_refuses_equal_dose_target(golden):
    _s, ir, ts_target = golden
    ts_eq = copy.deepcopy(ts_target)
    ts_eq.dose = torch.full_like(ts_eq.dose, 7.0)
    with pytest.raises(ValueError, match="same dose on every tilt"):
        fit_warp_volume_warp(ir, ts_eq, grid=(3, 3, 2, 4))


def _synthetic_series(n_tilts, dose, *, volwarp_dims=(3, 3, 2, 4), seed=5):
    gen = torch.Generator().manual_seed(seed)
    ts = TiltSeries()
    ts.image_dimensions_physical = torch.tensor([400.0, 400.0])
    ts.volume_dimensions_physical = torch.tensor([300.0, 300.0, 120.0])
    ts.angles = torch.linspace(50.0, -50.0, n_tilts)
    ts.tilt_axis_angles = torch.full((n_tilts,), 85.0)
    ts.tilt_axis_offset_x = torch.zeros(n_tilts)
    ts.tilt_axis_offset_y = torch.zeros(n_tilts)
    ts.dose = torch.tensor(dose, dtype=torch.float32)
    ts.use_tilt = torch.ones(n_tilts, dtype=torch.bool)
    ts.level_angle_x = 0.0
    ts.level_angle_y = 0.0
    ts.grid_movement_x = CubicGrid((1, 1, 1))
    ts.grid_movement_y = CubicGrid((1, 1, 1))
    k = volwarp_dims[0] * volwarp_dims[1] * volwarp_dims[2] * volwarp_dims[3]
    for name in ("x", "y", "z"):
        setattr(ts, f"grid_volume_warp_{name}", LinearGrid4D(volwarp_dims, (torch.rand(k, generator=gen) - 0.5) * 10))
    return ts


def _meta_for(ts, sampling="grid"):
    t = ts.n_tilts
    return IRMeta(
        kind="tilt_series", series_name="syn", pixel_size_image_a=2.0, image_dims_px=(200, 200),
        volume_dims_px=(150, 150, 60), pixel_size_volume_a=2.0, projection_index=list(range(t)),
        projection_valid=[True] * t, projection_order=list(range(t)), projection_dose=[float(d) for d in ts.dose],
        projection_angle_deg=[float(a) for a in ts.angles], projection_sec=[-1] * t, projection_dark=[False] * t,
        source_tool="warp", sampling=sampling,
    )


def test_irregular_doses_rank_deficient_L_equals_T_is_rejected():
    """Reviewer's counterexample: L = T = 5, normalized doses 0, .025, .05, .625, 1 —
    every node touched, no empty interior cell pair, temporal rank 4."""
    ts = _synthetic_series(5, [0.0, 3.0, 6.0, 75.0, 120.0])
    model = WarpTiltSeriesModel(ts)
    ir = build_ir_tilt_series(model, model.volume_dims_a, model.image_dims_a, meta=_meta_for(ts), grid_shape=(6, 6, 3))
    assert ir.meta.displacement_3d == "present"
    with pytest.raises(RuntimeError, match=r"not identifiable.*data rank 72 of 90.*dependent nodes"):
        fit_warp_volume_warp(ir, _reset_locals(ts), grid=(3, 3, 2, 5))
    # same data, L = 4 (the source's own temporal resolution): identifiable and exact
    _ts_fit, vw = fit_warp_volume_warp(ir, _reset_locals(ts), grid=(3, 3, 2, 4))
    assert vw.data_rank == vw.n_params_supported == 72
    assert vw.rms_a_heldout <= 1e-4


def test_two_node_midpoint_case_is_rejected_as_dependent():
    """Two spatial x-nodes sampled only at their midpoint: [10, 10] and
    [110, -90] are both data-consistent and penalty-free -> rank 1 of 2."""
    ts = _synthetic_series(4, [0.0, 3.0, 6.0, 9.0], volwarp_dims=(1, 1, 1, 1))
    model = WarpTiltSeriesModel(ts)
    v = model.volume_dims_a.to(F64)
    n = 30
    pts = torch.stack([torch.full((n,), float(v[0]) / 2, dtype=F64), torch.linspace(0, float(v[1]), n, dtype=F64), torch.linspace(0, float(v[2]), n, dtype=F64)], -1)
    ir = build_ir_tilt_series_from_points(model, pts, model.image_dims_a, meta=_meta_for(ts, "particles"), heldout_fraction=0.0)
    # give the IR a nonzero displacement so the fit is attempted (the model's is zero)
    ir.source_displacement_3d = torch.full((4, n, 3), 10.0)
    ir.heldout_source_displacement_3d = torch.zeros(4, 0, 3)
    ir.meta = ir.meta.model_copy(update={"displacement_3d": "present"})
    with pytest.raises(RuntimeError, match=r"data rank 1 of 2.*dependent nodes: \[.*x=0.*x=1"):
        fit_warp_volume_warp(ir, _reset_locals(ts), grid=(2, 1, 1, 1))


def test_dose_collisions_are_reported_and_difference_goes_to_movement():
    ts = _synthetic_series(6, [0.0, 3.0, 3.0, 6.0, 9.0, 12.0], volwarp_dims=(2, 2, 1, 5))
    model = WarpTiltSeriesModel(ts)
    ir = build_ir_tilt_series(model, model.volume_dims_a, model.image_dims_a, meta=_meta_for(ts), grid_shape=(5, 5, 3))
    _ts_fit, vw = fit_warp_volume_warp(ir, _reset_locals(ts), grid=(2, 2, 1, 5))
    assert vw.dose_collisions == 2  # rows 1 and 2 share a target dose
    # rows 1 and 2 have identical dose coordinates -> identical fitted displacement; the source's
    # differ only if its own field does (same dose -> same VW here, so the fit is still exact)
    assert vw.rms_a_heldout <= 1e-4
    full = fit_warp_locals(ir, _reset_locals(ts), movement_grid=(4, 4), volume_warp_grid=(2, 2, 1, 5))
    assert full.rms_a_heldout <= 1e-2


def test_dark_row_uses_temporal_extension(golden):
    """41 regular doses with one inactive row: its dose slice is unsupported
    (float32 leakage ~1e-6 only), gets tied to the nearest supported slice, and
    the fit stays full rank on the 40 supported slices."""
    _s, ir, ts_target = golden
    ts_dark = copy.deepcopy(ts_target)
    ts_dark.use_tilt[2] = False
    _ts_fit, vw = fit_warp_volume_warp(ir, ts_dark, grid=(3, 3, 2, 41))
    assert vw.n_unsupported_slices == 1 and vw.n_params_supported == 18 * 40
    assert vw.data_rank == vw.n_params_supported
    assert 0 < vw.discarded_temporal_weight_max < 1e-5  # float32 leakage (2.4e-6 on these doses), far below tau
    unsupported = [i for i, ok in enumerate(vw.supported_slices) if not ok]
    dose_idx = round(float(ts_dark.dose[2] - ts_dark.dose.min()) / 3)
    assert unsupported == [dose_idx]
    assert vw.slice_occupancy[dose_idx] == 0 and min(o for i, o in enumerate(vw.slice_occupancy) if i != dose_idx) >= 1
    # active rows are still reproduced
    assert vw.rms_a_heldout <= 1e-4


def test_node_support_3d_and_sparse_particles_fail():
    v = torch.tensor([300.0, 300.0, 120.0], dtype=F64)
    full = torch.rand(500, 3, dtype=F64) * v
    assert node_support_3d(full, v, (3, 3, 2)) == 1.0
    corner = torch.rand(60, 3, dtype=F64) * v * 0.15  # all in one corner
    assert node_support_3d(corner, v, (3, 3, 2)) == 8 / 18  # nodes at 0 and 0.5 are within one spacing
    assert node_support_3d(corner, v, (4, 4, 2)) == 8 / 32 < 0.4
    ts = _synthetic_series(5, [0.0, 3.0, 6.0, 9.0, 12.0], volwarp_dims=(1, 1, 1, 1))
    model = WarpTiltSeriesModel(ts)
    ir = build_ir_tilt_series_from_points(model, corner, model.image_dims_a, meta=_meta_for(ts, "particles"), heldout_fraction=0.0)
    ir.source_displacement_3d = torch.randn(5, 60, 3) * 3
    ir.heldout_source_displacement_3d = torch.zeros(5, 0, 3)
    ir.meta = ir.meta.model_copy(update={"displacement_3d": "present"})
    with pytest.raises(RuntimeError, match="node support|condition|not identifiable"):
        fit_warp_volume_warp(ir, _reset_locals(ts), grid=(4, 4, 2, 5))


# ---------------------------------------------------------------------------
# CTF-depth deviation semantics
# ---------------------------------------------------------------------------


def test_ctf_depth_deviation_along_beam_synthetic(golden):
    """Synthetic RELION-like source: the stored source CTF depth is that of the
    STATIC position (RELION convention) while the displacement carries 10 A along
    each tilt's beam. An exact fit has beam residual ~0 but CTF-depth deviation
    ~10 A — the two metrics are different quantities."""
    _s, ir, ts_target = golden
    model = WarpTiltSeriesModel(ts_target)
    rot = model._tilt_matrices(flipped=False).to(F64)  # (T, 3, 3); beam direction = row 2
    beam = rot[:, 2, :]  # (T, 3)
    ir2 = copy.deepcopy(ir)
    d = 10.0 * beam[:, None, :].expand(-1, ir.n_points, -1).to(torch.float32)
    hd = 10.0 * beam[:, None, :].expand(-1, ir.heldout_points.shape[0], -1).to(torch.float32)
    ir2.source_displacement_3d, ir2.heldout_source_displacement_3d = d.contiguous(), hd.contiguous()
    zero = torch.zeros_like(d)
    ir2.source_ctf_depth_a = model.ctf_depth(ir.points, zero)  # static-position depth
    ir2.heldout_source_ctf_depth_a = model.ctf_depth(ir.heldout_points, torch.zeros_like(hd))
    _ts_fit, vw = fit_warp_volume_warp(ir2, ts_target, grid=(1, 1, 1, 41))  # constant-in-space field, per dose
    assert vw.rms_a_heldout_beam <= 1e-2
    assert abs(vw.ctf_depth_deviation_rms_a_heldout - 10.0) <= 1e-2
    assert abs(vw.ctf_depth_deviation_max_a_heldout - 10.0) <= 1e-2


def test_ctf_depth_deviation_not_available_without_source_depths(golden):
    _s, ir, ts_target = golden
    ir2 = copy.deepcopy(ir)
    ir2.source_ctf_depth_a = ir2.heldout_source_ctf_depth_a = None
    ir2.meta = ir2.meta.model_copy(update={"source_ctf_depth": "none"})
    _ts_fit, vw = fit_warp_volume_warp(ir2, ts_target, grid=(3, 3, 2, 4))
    assert vw.ctf_depth_deviation_rms_a_heldout is None and vw.ctf_depth_deviation_rms_a_train is None
    assert vw.meta["ctf_depth_deviation"] == "not available"
    assert vw.rms_a_heldout_beam is not None  # the displacement-fit beam residual is independent of it


def test_fitted_xml_round_trips_through_the_strict_loader(golden, tmp_path):
    from cets_nonrigid.io.warp_xml import write_alignment_into_template

    s, ir, ts_target = golden
    full = fit_warp_locals(ir, ts_target, movement_grid=(6, 4), volume_warp_grid=(3, 3, 2, 4))
    out = tmp_path / "fitted.xml"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write_alignment_into_template(s.xml_bytes, full.ts, out)
        back = load_warp_tiltseries(out)
    assert tuple(back.ts.grid_volume_warp_x.dimensions) == (3, 3, 2, 4)
    xy_a, _ = WarpTiltSeriesModel(full.ts).project_volume(ir.heldout_points)
    xy_b, _ = back.model.project_volume(ir.heldout_points)
    assert (xy_a - xy_b).abs().max() <= 1e-2  # XML value formatting only


