"""Step 1 of the 3D-displacement plan: IR arrays, flags, store 0.4, row alignment.

Serialization tests cross float32 storage, so their tolerances include
quantization; algebraic identities live in ``test_displacement_model.py``.
"""

from __future__ import annotations

import warnings

import numpy as np
import test_ir_and_store as base
import torch

from cets_nonrigid.io.warp_xml import load_warp_tiltseries
from cets_nonrigid.ir.build import build_ir_tilt_series, build_ir_tilt_series_from_points
from cets_nonrigid.ir.core import IRMeta
from cets_nonrigid.ir.rows import RowMatch, TargetRowTable, align_ir_rows

GOLDEN_XML = "tests/golden/TS_1_volwarp.xml"
DEGENERATE_XML = "tests/data/TS_1.xml"


def _warp_series(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_warp_tiltseries(path)


def _warp_meta(series, sampling="grid"):
    ts = series.ts
    t = ts.n_tilts
    pix = float(ts.image_dimensions_physical[0]) / 4096
    return IRMeta(
        kind="tilt_series", series_name="TS_1", pixel_size_image_a=pix,
        image_dims_px=(4096, 4096), volume_dims_px=(4096, 4096, 2000), pixel_size_volume_a=pix,
        projection_index=list(range(t)), projection_valid=[bool(u) for u in ts.use_tilt],
        projection_order=list(range(t)), projection_dose=[float(d) for d in ts.dose],
        projection_angle_deg=[float(a) for a in ts.angles], projection_sec=[-1] * t,
        projection_dark=[False] * t, source_tool="warp", sampling=sampling,
    )


def _grid_ir(path, grid=(4, 4, 3)):
    s = _warp_series(path)
    return s, build_ir_tilt_series(s.model, s.model.volume_dims_a, s.model.image_dims_a, meta=_warp_meta(s), grid_shape=grid)


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def test_grid_builder_records_warp_displacement_and_ctf_depth():
    s, ir = _grid_ir(GOLDEN_XML)
    ir.validate_optional_arrays()
    assert ir.meta.displacement_3d == "present" and ir.meta.source_ctf_depth == "present"
    t, n, m = ir.n_projections, ir.n_points, ir.heldout_points.shape[0]
    assert ir.source_displacement_3d.shape == (t, n, 3) and ir.source_displacement_3d.dtype == torch.float32
    assert ir.heldout_source_displacement_3d.shape == (t, m, 3)
    assert ir.source_ctf_depth_a.shape == (t, n) and ir.heldout_source_ctf_depth_a.shape == (t, m)
    # what is stored IS the model's own intermediate (no re-evaluation at deformed points)
    torch.testing.assert_close(ir.source_displacement_3d, s.model.displace_volume(ir.points))
    torch.testing.assert_close(ir.source_ctf_depth_a, s.model.ctf_depth(ir.points))
    torch.testing.assert_close(ir.heldout_source_ctf_depth_a, s.model.ctf_depth(ir.heldout_points))
    # warped points helper
    wp = ir.warped_points()
    assert wp.shape == (t, n, 3) and wp.dtype == torch.float64
    torch.testing.assert_close(wp[3] - ir.points, ir.source_displacement_3d[3].to(torch.float64))
    # the 2D projections are still the full model's (nothing removed)
    xy, _ = s.model.project_volume(ir.points)
    torch.testing.assert_close(ir.source_projected, xy)


def test_grid_builder_zero_suppression_keeps_ctf_depth():
    _s, ir = _grid_ir(DEGENERATE_XML)
    assert ir.meta.displacement_3d == "zero_at_samples" and ir.source_displacement_3d is None
    assert ir.warped_points() is None
    # CTF depth availability is independent of the displacement
    assert ir.meta.source_ctf_depth == "present" and ir.source_ctf_depth_a is not None
    ir.validate_optional_arrays()


def test_aretomo_builder_records_none():
    model = base._model()
    ir = build_ir_tilt_series(model, torch.tensor(model.volume_dims_a), torch.tensor([1600.0, 1400.0]), meta=base._meta())
    assert ir.meta.displacement_3d == "none" and ir.meta.source_ctf_depth == "none"
    assert ir.source_displacement_3d is None and ir.source_ctf_depth_a is None


def _relion_particle_ir(with_traj: bool, n=60):
    from cets_nonrigid.models.relion_ts import RelionParticleSetModel, RelionTomogramModel

    t = 7
    g = RelionTomogramModel(
        xtilt_deg=torch.zeros(t), ytilt_deg=torch.linspace(-60, 60, t), zrot_deg=torch.full((t,), 85.0),
        xshift_a=torch.zeros(t), yshift_a=torch.zeros(t), tomo_dims_px=(400, 400, 200),
        image_dims_px=(500, 500), pixel_size_a=2.0, hand=-1, defocus_slope=1.1,
    )
    gen = torch.Generator().manual_seed(3)
    pos = (torch.rand(n, 3, generator=gen, dtype=torch.float64) * 0.8 + 0.1) * g.volume_dims_a
    traj = torch.randn(t, n, 3, generator=gen, dtype=torch.float64) * 4 if with_traj else None
    model = RelionParticleSetModel(g, pos, trajectories_a=traj)
    meta = IRMeta(
        kind="tilt_series", series_name="rel", pixel_size_image_a=2.0, image_dims_px=(500, 500),
        volume_dims_px=(400, 400, 200), pixel_size_volume_a=2.0, projection_index=list(range(t)),
        projection_valid=[True] * t, projection_order=list(range(t)), projection_dose=[3.0 * i for i in range(t)],
        projection_angle_deg=list(np.linspace(-60, 60, t)), projection_sec=[-1] * t,
        projection_dark=[False] * t, source_tool="relion", sampling="particles",
    )
    names = [f"rel/{i + 1}" for i in range(n)]
    ir = build_ir_tilt_series_from_points(model, pos, g.image_dims_a, meta=meta, point_names=names)
    return model, g, pos, traj, ir


def test_particle_builder_records_trajectories_split_first():
    _model, g, pos, traj, ir = _relion_particle_ir(True)
    ir.validate_optional_arrays()
    assert ir.meta.displacement_3d == "present" and ir.meta.source_ctf_depth == "present"
    assert ir.heldout_status == "evaluated"
    # arrays are split by the same permutation as points (row correspondence)
    torch.testing.assert_close(ir.source_displacement_3d.to(torch.float64), traj[:, ir.point_index], atol=1e-5, rtol=0)
    torch.testing.assert_close(
        ir.heldout_source_displacement_3d.to(torch.float64), traj[:, ir.heldout_point_index], atol=1e-5, rtol=0
    )
    torch.testing.assert_close(ir.source_ctf_depth_a.to(torch.float64), g.ctf_depth(pos)[:, ir.point_index], atol=1e-3, rtol=0)
    # without trajectories: no 3D model, but the CTF depth model still exists
    _m, _g, _p, _t, ir0 = _relion_particle_ir(False)
    assert ir0.meta.displacement_3d == "none" and ir0.source_displacement_3d is None
    assert ir0.meta.source_ctf_depth == "present"


# ---------------------------------------------------------------------------
# store 0.4
# ---------------------------------------------------------------------------










# ---------------------------------------------------------------------------
# row alignment
# ---------------------------------------------------------------------------


def test_align_ir_rows_expands_optional_arrays_with_zero_fill():
    _s, ir = _grid_ir(GOLDEN_XML)
    t = ir.n_projections
    # target has one extra (unmatched) row at the front
    row_map = [i + 1 for i in range(t)]
    tgt_angles = [0.0] + [float(a) for a in ir.meta.projection_angle_deg]
    target = TargetRowTable(
        angle_deg=tgt_angles, angle_kind=["unknown"] * (t + 1), active=[False] + [True] * t,
        dark=[True] + [False] * t, sec=[-1] * (t + 1), dose=[0.0] + list(ir.meta.projection_dose),
        labels=None, order=list(range(t + 1)),
    )
    match = RowMatch(row_map=row_map, target_active=[False] + [True] * t, method="test")
    out = align_ir_rows(ir, match, target)
    assert out.source_displacement_3d.shape == (t + 1, ir.n_points, 3)
    assert torch.equal(out.source_displacement_3d[0], torch.zeros(ir.n_points, 3))
    torch.testing.assert_close(out.source_displacement_3d[1:], ir.source_displacement_3d)
    assert torch.equal(out.source_ctf_depth_a[0], torch.zeros(ir.n_points))
    torch.testing.assert_close(out.heldout_source_ctf_depth_a[1:], ir.heldout_source_ctf_depth_a)
    assert out.meta.displacement_3d == "present"  # flags survive the meta replacement
    out.validate_optional_arrays()


# ---------------------------------------------------------------------------
# r2w: builder flags survive the pipeline meta replacement
# ---------------------------------------------------------------------------


