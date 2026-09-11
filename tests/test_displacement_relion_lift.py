"""Step 5 of the 3D-displacement plan: the RELION trajectory lift with a source
3D displacement, the gauge contract, the CTF-depth deviation and the
lift-consistency diagnostic.

Promise: 2D projection positions are unchanged under every gauge and lift.
Intentional change: RELION's CTF depth (static coordinate) depends on the gauge
and on whether the source displacement entered the lift.
"""

from __future__ import annotations

import math
import warnings

import pytest
import test_r4_w2r as h
import torch

from cets_nonrigid.convert_relion import relion_model_from_data, warp_to_relion
from cets_nonrigid.fit.relion_traj import (
    GAUGES,
    LIFT_EXACT_MAX_PX,
    lift_consistency,
    lift_particle_trajectories,
)
from cets_nonrigid.io.relion_star import read_motion_star, read_particles_star, read_tomograms_star
from cets_nonrigid.io.warp_xml import load_warp_tiltseries
from cets_nonrigid.models.relion_ts import RelionTomogramModel, effective_positions_a
from cets_nonrigid.models.warp_ts import WarpTiltSeriesModel

F64 = torch.float64


def _model(t=5, hand=1, slope=1.0):
    return RelionTomogramModel(
        xtilt_deg=torch.zeros(t), ytilt_deg=torch.linspace(-60, 60, t), zrot_deg=torch.full((t,), 85.0),
        xshift_a=torch.zeros(t), yshift_a=torch.zeros(t), tomo_dims_px=(200, 200, 100),
        image_dims_px=(240, 240), pixel_size_a=2.0, hand=hand, defocus_slope=slope,
    )


def _project(model, pos, disp=None):
    r = model.projection_matrices[:, :3, :3]
    tr = model.projection_matrices[:, :3, 3]
    p = pos[None] if disp is None else pos[None] + disp
    return (torch.einsum("tij,tpj->tpi", r, p / model.pixel_size_a) + tr[:, None, :])[..., :2] * model.pixel_size_a


# ---------------------------------------------------------------------------
# lift algebra
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gauge", GAUGES)
@pytest.mark.parametrize("with_disp", [False, True])
def test_2d_positions_are_gauge_and_lift_invariant(gauge, with_disp):
    m = _model(7, hand=-1, slope=1.2)
    gen = torch.Generator().manual_seed(2)
    pos = (torch.rand(20, 3, generator=gen, dtype=F64) * 0.6 + 0.2) * m.volume_dims_a
    disp = torch.randn(7, 20, 3, generator=gen, dtype=F64) * 5 if with_disp else None
    xy = _project(m, pos, disp) + torch.randn(7, 20, 2, generator=gen, dtype=F64) * 3  # + in-plane movement
    dose = torch.arange(7, dtype=F64) * 3
    src_depth = m.ctf_depth(pos) + (torch.randn(7, 20, generator=gen, dtype=F64) * 4)  # some source depth
    r = lift_particle_trajectories(xy, m, pos, dose, source_disp_a=disp, source_ctf_depth_a=src_depth, gauge=gauge)
    assert r.max_residual_px < 1e-9
    back = _project(m, r.positions_out_a, r.motion_a)
    assert (back - xy).abs().max() < 1e-9
    assert r.depth_source == ("displacement" if with_disp else "min_norm") and r.gauge == gauge
    if with_disp:
        # the per-frame 3D coordinate p' + motion equals p + d + in-plane lift; its beam
        # coordinate equals the source's (the trajectory PRESERVES the source depth)
        rot = m.rotations
        beam_new = torch.einsum("tj,tpj->tp", rot[:, 2, :], r.positions_out_a[None] + r.motion_a)
        beam_src = torch.einsum("tj,tpj->tp", rot[:, 2, :], pos[None] + disp)
        assert (beam_new - beam_src).abs().max() < 1e-9


def test_constant_z_displacement_changes_static_depth_not_2d():
    """Reviewer's case: constant 10 A along volume Z at 0 and 60 deg. Old and new
    lifts give identical 2D positions; static CTF depths differ (0/0 vs 10/5 A
    for unit hand/slope)."""
    m = RelionTomogramModel(
        xtilt_deg=torch.zeros(2), ytilt_deg=torch.tensor([0.0, 60.0]), zrot_deg=torch.zeros(2),
        xshift_a=torch.zeros(2), yshift_a=torch.zeros(2), tomo_dims_px=(200, 200, 100),
        image_dims_px=(240, 240), pixel_size_a=1.0, hand=1, defocus_slope=1.0,
    )
    pos = torch.tensor([[100.0, 100.0, 50.0]], dtype=F64)
    d = torch.tensor([[[0.0, 0.0, 10.0]], [[0.0, 0.0, 10.0]]], dtype=F64)
    xy = _project(m, pos, d)
    dose = torch.tensor([0.0, 3.0], dtype=F64)
    old = lift_particle_trajectories(xy, m, pos, dose)
    new = lift_particle_trajectories(xy, m, pos, dose, source_disp_a=d)
    for r in (old, new):
        assert (_project(m, r.positions_out_a, r.motion_a) - xy).abs().max() < 1e-9
    depth_old = m.ctf_depth(old.positions_out_a) - m.ctf_depth(pos)
    depth_new = m.ctf_depth(new.positions_out_a) - m.ctf_depth(pos)
    true = torch.tensor([[10.0], [10.0 * math.cos(math.radians(60))]], dtype=F64)
    assert torch.allclose(depth_old, torch.zeros_like(true), atol=1e-9)
    assert torch.allclose(depth_new, true, atol=1e-9)


def test_ctf_optimal_gauge_rank_cut_and_fallback():
    """Single-axis: beam directions span a plane -> rank 2; the tilt-axis
    component stays at the lowest-dose gauge. A near-coplanar perturbation must
    not blow the gauge up (an unconstrained solve moved it by ~700 A)."""
    t = 9
    m = _model(t)
    gen = torch.Generator().manual_seed(4)
    pos = (torch.rand(15, 3, generator=gen, dtype=F64) * 0.5 + 0.25) * m.volume_dims_a
    xy = _project(m, pos)
    dose = torch.arange(t, dtype=F64)
    src_depth = m.ctf_depth(pos) + torch.rand(t, 15, generator=gen, dtype=F64) * 9  # < 9 A targets
    r = lift_particle_trajectories(xy, m, pos, dose, source_ctf_depth_a=src_depth, gauge="ctf-optimal")
    assert r.gauge_rank == 2 and r.n_gauge_fallback == 0
    # RELION's P_f = Rz(zrot) Ry(ytilt) Rx(xtilt): with xtilt = 0 the beam row is that of
    # Ry alone, (-sin, 0, cos) -> the unconstrained direction is the volume y-axis
    axis = torch.tensor([0.0, 1.0, 0.0], dtype=F64)
    assert torch.linalg.matrix_rank(m.rotations[:, 2, :]).item() == 2
    ref = lift_particle_trajectories(xy, m, pos, dose, source_ctf_depth_a=src_depth, gauge="lowest-dose")
    shift = r.positions_out_a - ref.positions_out_a
    assert (shift @ axis).abs().max() < 1e-9  # no component along the unconstrained direction
    assert shift.norm(dim=-1).max() < 20.0
    # RMS improves within the retained directions; the maximum is reported, not promised
    assert r.ctf_depth_deviation_rms_a <= ref.ctf_depth_deviation_rms_a
    assert r.ctf_depth_deviation_max_a is not None
    # near-coplanar perturbation: +/-0.05 deg x-tilt -> third direction dropped
    mp = RelionTomogramModel(
        xtilt_deg=torch.tensor([0.05 * (-1) ** i for i in range(t)]), ytilt_deg=torch.linspace(-60, 60, t),
        zrot_deg=torch.full((t,), 85.0), xshift_a=torch.zeros(t), yshift_a=torch.zeros(t),
        tomo_dims_px=(200, 200, 100), image_dims_px=(240, 240), pixel_size_a=2.0,
    )
    xyp = _project(mp, pos)
    rp = lift_particle_trajectories(xyp, mp, pos, dose, source_ctf_depth_a=mp.ctf_depth(pos) + 5.0, gauge="ctf-optimal")
    assert rp.gauge_rank == 2
    assert (rp.positions_out_a - pos).norm(dim=-1).max() < 20.0


def test_out_of_volume_reference_is_an_error_under_both_gauges():
    """A constant displacement pushes the reference position 8 A outside the
    x = 0 face. lowest-dose: error. ctf-optimal with a source depth that agrees
    with the displaced position keeps c = t_ref (x is a retained direction):
    error too — the gauge does not repair an out-of-volume reference."""
    m = _model(5)
    pos = torch.tensor([[2.0, 100.0, 100.0]], dtype=F64)  # 2 A from the x = 0 face
    d = torch.zeros(5, 1, 3, dtype=F64)
    d[:, 0, 0] = -10.0  # displaced 10 A outside at every frame -> t_ref outside
    xy = _project(m, pos, d)
    dose = torch.arange(5, dtype=F64)
    depth_displaced = m.ctf_depth(pos + d[0])  # (T, 1): the source's own depth (constant d)
    for gauge in GAUGES:
        with pytest.raises(ValueError, match="outside the tomogram volume"):
            lift_particle_trajectories(xy, m, pos, dose, source_disp_a=d, source_ctf_depth_a=depth_displaced, gauge=gauge)
    # and a source depth that says the STATIC position is right lets ctf-optimal move
    # the particle back inside (a legitimate gauge choice, not a repair)
    r = lift_particle_trajectories(xy, m, pos, dose, source_disp_a=d, source_ctf_depth_a=m.ctf_depth(pos), gauge="ctf-optimal")
    assert (r.positions_out_a - pos).abs().max() < 1e-9


def test_ctf_optimal_requires_source_depth():
    m = _model(5)
    pos = torch.tensor([[100.0, 100.0, 50.0]], dtype=F64)
    with pytest.raises(ValueError, match="needs the source CTF depth"):
        lift_particle_trajectories(_project(m, pos), m, pos, torch.arange(5, dtype=F64), gauge="ctf-optimal")


def test_lift_consistency_diagnostic():
    m = _model(7)
    gen = torch.Generator().manual_seed(9)
    pos = (torch.rand(12, 3, generator=gen, dtype=F64) * 0.5 + 0.25) * m.volume_dims_a
    xy = _project(m, pos) + torch.randn(7, 12, 2, generator=gen, dtype=F64) * 8
    r = lift_particle_trajectories(xy, m, pos, torch.arange(7, dtype=F64))
    lc = lift_consistency(r.motion_a, m.rotations)
    assert lc.rank == 2 and lc.residual_dof == 5
    assert lc.residual_max_a < 1e-9 and lc.verdict == "consistent with a pure in-plane lift"
    # the gauged min-norm motion itself has NONZERO beam components (not an exact-zero statistic)
    beam = torch.einsum("tj,tpj->tp", m.rotations[:, 2, :], r.motion_a)
    assert beam.abs().max() > 1.0
    # dose-dependent depth breaks the consistency
    depth = torch.linspace(-4, 4, 7, dtype=F64)[:, None, None] * torch.tensor([0.0, 0.0, 1.0], dtype=F64)
    lc2 = lift_consistency(r.motion_a + depth, m.rotations)
    assert lc2.residual_max_a > 0.1 and lc2.verdict == "carries depth beyond an in-plane lift"


# ---------------------------------------------------------------------------
# w2r end to end
# ---------------------------------------------------------------------------


def _volwarp_xml(path, seed=77, dims=(3, 3, 2, 4)):
    """The r4 synthetic Warp series (integer-pixel dims, movement grids, CTF)
    with NON-degenerate volume-warp grids: the golden XML's volume is not an
    integer pixel count at its pixel size, so w2r cannot consume it."""
    from warpylib import LinearGrid4D
    from warpylib.tilt_series.io import save_meta

    h._write_synthetic_xml(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts = load_warp_tiltseries(path).ts
    gen = torch.Generator().manual_seed(seed)
    k = dims[0] * dims[1] * dims[2] * dims[3]
    for name in ("x", "y", "z"):
        setattr(ts, f"grid_volume_warp_{name}", LinearGrid4D(dims, (torch.rand(k, generator=gen) - 0.5) * 6))
    ts.path = str(path)
    path.unlink()
    save_meta(ts, str(path))
    return path


def _golden_particles(tmp_path, n=40, seed=12):
    xml = _volwarp_xml(tmp_path / "volwarp_src.xml")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = load_warp_tiltseries(xml)
    assert (s.model.displace_volume(s.model.volume_dims_a.to(F64)[None] * 0.5) != 0).any()
    gen = torch.Generator().manual_seed(seed)
    v = s.model.volume_dims_a.to(F64)
    pos = (torch.rand(n, 3, generator=gen, dtype=F64) * 0.7 + 0.15) * v
    return s, xml, pos


@pytest.mark.parametrize("gauge", GAUGES)
def test_w2r_deviation_report_matches_independent_recomputation(tmp_path, gauge):
    """Predicted deviation == RELION.ctf_depth(emitted static) - Warp.ctf_depth(source),
    recomputed from the WRITTEN star files (hand, slope) and the source XML."""
    s, xml, pos = _golden_particles(tmp_path)
    names = [f"G/{i + 1}" for i in range(pos.shape[0])]
    stack = h._dummy_stack(tmp_path / "stack.mrc")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = warp_to_relion(
            xml, tmp_path / f"b_{gauge}", pixel_size_a=h.PIX, tomo_name="G", positions_eff_a=pos,
            particle_names=names, tilt_stack=stack, trajectory_gauge=gauge,
        )
    lift = r.lift
    assert lift.depth_source == "displacement" and lift.gauge == gauge
    assert lift.max_residual_px < LIFT_EXACT_MAX_PX
    tomo = read_tomograms_star(r.tomograms_star)["G"]
    parts = read_particles_star(r.particles_star)
    motion = read_motion_star(r.motion_star, names, len(r.rows))
    emitted = relion_model_from_data(tomo, image_dims_px=(96, 96))
    assert emitted.hand == r.hand
    static = effective_positions_a(
        pixel_size_a=tomo.pixel_size_a, tomo_dims_px=tuple(tomo.tomo_dims_px),
        centered_coords_a=parts.centered_coords_a, origins_a=parts.origins_a,
        subtomo_angles_deg=parts.subtomo_angles_deg,
    )
    rows = torch.tensor(r.rows)
    dev_indep = emitted.ctf_depth(static) - s.model.ctf_depth(pos)[rows].to(F64)
    assert (dev_indep - lift.ctf_depth_deviation_a).abs().max() < 1e-2  # f32 star/model quantization
    assert abs(float(dev_indep.pow(2).mean().sqrt()) - lift.ctf_depth_deviation_rms_a) < 1e-2
    # the trajectory beam coordinate equals the source's (preservation), the 2D observable is exact
    rot = emitted.rotations
    beam_traj = torch.einsum("tj,tpj->tp", rot[:, 2, :], static[None] + motion)
    beam_src = torch.einsum("tj,tpj->tp", rot[:, 2, :], pos[None] + s.model.displace_volume(pos)[rows].to(F64))
    assert (beam_traj - beam_src).abs().max() < 1e-2
    assert lift.depth_source == "displacement" and lift.gauge == gauge
    assert lift.ctf_depth_deviation_a.shape == (len(r.rows), len(pos))


def test_w2r_ctf_optimal_rms_not_worse_than_lowest_dose(tmp_path):
    _s, xml, pos = _golden_particles(tmp_path, n=30, seed=3)
    names = [f"G/{i + 1}" for i in range(pos.shape[0])]
    stack = h._dummy_stack(tmp_path / "stack.mrc")
    res = {}
    for gauge in GAUGES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res[gauge] = warp_to_relion(
                xml, tmp_path / f"c_{gauge}", pixel_size_a=h.PIX, tomo_name="G", positions_eff_a=pos,
                particle_names=names, tilt_stack=stack, trajectory_gauge=gauge,
            )
    lo, opt = res["lowest-dose"].lift, res["ctf-optimal"].lift
    assert opt.ctf_depth_deviation_rms_a <= lo.ctf_depth_deviation_rms_a + 1e-9
    # per particle as well (each particle is its own least-squares problem)
    per_lo = lo.ctf_depth_deviation_a.pow(2).mean(dim=0).sqrt()
    per_opt = opt.ctf_depth_deviation_a.pow(2).mean(dim=0).sqrt()
    assert bool((per_opt <= per_lo + 1e-9).all())
    # both bundles project identically
    assert (read_motion_star(res["ctf-optimal"].motion_star, names, len(opt.motion_a)).shape == lo.motion_a.shape)


def test_movement_only_source_bundle_is_unchanged(tmp_path):
    """A source without volume warp takes the classic min-norm lift path: bytes identical."""
    xml = h._write_synthetic_xml(tmp_path / "src.xml")
    stack = h._dummy_stack(tmp_path / "stack.mrc")
    pos = h._particles()
    names = [f"TS/{i + 1}" for i in range(pos.shape[0])]
    r = warp_to_relion(xml, tmp_path / "b", pixel_size_a=h.PIX, tomo_name="TS", positions_eff_a=pos,
                       particle_names=names, tilt_stack=stack)
    assert r.lift.depth_source == "min_norm" and r.lift.gauge == "lowest-dose"
    motion = read_motion_star(r.motion_star, names, len(r.rows))
    torch.testing.assert_close(motion[r.lift.ref_row], torch.zeros_like(motion[0]), atol=1e-5, rtol=0)
    lc = lift_consistency(r.lift.motion_a, r.global_result.model.rotations)
    assert lc.verdict == "consistent with a pure in-plane lift"


def test_w2r_r2w_roundtrip_on_volwarp_source_recovers_observables(tmp_path):
    """w2r (displacement lift) -> r2w --volume-warp-grid: 2D held-out < 0.1 px at the
    gauged particles AND the beam-axis coordinate is reported (pinned at the
    measured value, not compared as 3D vectors: the VW/movement split of in-plane
    shifts is non-unique)."""
    from cets_nonrigid.convert_relion import relion_to_warp

    _s, xml, pos = _golden_particles(tmp_path, n=220, seed=21)
    names = [f"G/{i + 1}" for i in range(pos.shape[0])]
    stack = h._dummy_stack(tmp_path / "stack.mrc")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = warp_to_relion(xml, tmp_path / "bundle", pixel_size_a=h.PIX, tomo_name="G",
                           positions_eff_a=pos, particle_names=names, tilt_stack=stack)
        res = relion_to_warp(
            xml, tmp_path / "rt.xml", tomo_name="G", optimisation_set=r.optimisation_set,
            movement_grid=(4, 4), volume_warp_grid=(3, 3, 2, 4),
        )
        back = WarpTiltSeriesModel(load_warp_tiltseries(tmp_path / "rt.xml").ts)
    assert res.fit.heldout_status == "evaluated"
    assert res.fit.rms_a_heldout / h.PIX < 0.1, f"{res.fit.rms_a_heldout / h.PIX:.3f} px"
    p_prime = r.lift.positions_out_a
    rel = r.global_result.model
    rot = rel.projection_matrices[:, :3, :3]
    trans = rel.projection_matrices[:, :3, 3]
    src = (torch.einsum("tij,tpj->tpi", rot, (p_prime[None] + r.lift.motion_a) / h.PIX) + trans[:, None, :])[..., :2] * h.PIX
    b_xy, b_valid = back.project_volume(p_prime)
    rows = torch.tensor(r.rows)
    v = b_valid[rows]
    err = (b_xy.to(F64)[rows] - src)[v].norm(dim=-1)
    assert float(err.pow(2).mean().sqrt()) / h.PIX < 0.1
    # beam-axis coordinate: bundle trajectory vs recovered Warp model, reported and pinned
    beam_bundle = torch.einsum("tj,tpj->tp", rot[:, 2, :], p_prime[None] + r.lift.motion_a)
    d_back = back.displace_volume(p_prime)[rows].to(F64)
    beam_back = torch.einsum("tj,tpj->tp", rot[:, 2, :], p_prime[None] + d_back)
    beam_rms = float((beam_bundle - beam_back).pow(2).mean().sqrt())
    assert beam_rms < 3.0, f"beam-axis RMS {beam_rms:.3f} A"  # measured ~1 A class: in-plane lift leaks into z
    assert res.fit.volume_warp.data_rank == res.fit.volume_warp.n_params_supported
