"""R5: from-RELION — r2w round trip + gates."""

import numpy as np
import pytest
import test_r4_w2r as w2r_helpers
import torch

from cets_nonrigid.convert_relion import relion_to_warp, warp_to_relion
from cets_nonrigid.io.warp_xml import load_warp_tiltseries
from cets_nonrigid.models.warp_ts import WarpTiltSeriesModel

RNG = np.random.default_rng(20260907)

PIX = w2r_helpers.PIX
VOL_A = w2r_helpers.VOL_A


def _many_particles(n=200):
    vol_a = torch.tensor(VOL_A, dtype=torch.float64)
    frac = torch.tensor(RNG.uniform(0.08, 0.92, (n, 3)))
    return frac * vol_a


def _bundle_from_synthetic_warp(tmp_path, n_particles=200):
    xml = w2r_helpers._write_synthetic_xml(tmp_path / "src.xml")
    stack = w2r_helpers._dummy_stack(tmp_path / "stack.mrc")
    pos = _many_particles(n_particles)
    names = [f"TS_RT/{i + 1}" for i in range(pos.shape[0])]
    r = warp_to_relion(
        xml, tmp_path / "bundle", pixel_size_a=PIX, tomo_name="TS_RT",
        positions_eff_a=pos, particle_names=names, tilt_stack=stack,
    )
    return xml, r


def test_w2r_r2w_roundtrip_movement_only(tmp_path):
    """Movement-only Warp source -> exact RELION bundle -> fitted Warp XML:
    the round trip must recover the source model (plan gate: held-out RMS
    < 0.1 px with >= 200 spread particles)."""
    xml, r = _bundle_from_synthetic_warp(tmp_path)

    out_xml = tmp_path / "roundtrip.xml"
    res = relion_to_warp(
        xml, out_xml,
        tomo_name="TS_RT",
        optimisation_set=r.optimisation_set,
        movement_grid=(4, 4),
    )
    assert res.fit.heldout_status == "evaluated"
    assert res.fit.rms_a_heldout / PIX < 0.1, (
        f"round-trip held-out RMS {res.fit.rms_a_heldout / PIX:.3f} px"
    )
    assert abs(res.level_angle_x_deg - 2.0) < 1e-6  # xtilt -> LevelAngleX exactly

    # The round-trip invariant is the OBSERVABLE, not model identity: w2r
    # gauges particle positions (p' = p + t_ref), so the recovered Warp model
    # reproduces the source projections AT THE BUNDLE'S PARTICLES; on fresh 3D
    # points it may differ by the absorbed position shifts (a few A here).
    # Verify the observable directly: recovered(p' ) vs the exact bundle
    # reconstruction P(p' + motion) at every particle and emitted row.
    back = WarpTiltSeriesModel(load_warp_tiltseries(out_xml).ts)
    p_prime = r.lift.positions_out_a
    rel = r.global_result.model
    rot = rel.projection_matrices[:, :3, :3]
    trans = rel.projection_matrices[:, :3, 3]
    src_at_particles = (
        torch.einsum("tij,tpj->tpi", rot, (p_prime[None] + r.lift.motion_a) / PIX)
        + trans[:, None, :]
    )[..., :2] * PIX  # (T_rows, P, 2) canonical A
    b_xy, b_valid = back.project_volume(p_prime)
    rows = torch.tensor(r.rows)
    v = b_valid[rows]
    err = (b_xy.to(torch.float64)[rows] - src_at_particles)[v].norm(dim=-1)
    assert float(err.pow(2).mean().sqrt()) / PIX < 0.1

    # CTF grids traveled through the round trip
    from cets_nonrigid.ctf import tiltctf_from_warp

    ctf_back = tiltctf_from_warp(load_warp_tiltseries(out_xml).ts)
    assert ctf_back is not None

    # Rank is checked on the numerical result. CETS diagnostic publication is
    # exercised through with_fit_report in test_displacement_cli.
    assert res.fit.min_data_rank > 0
    assert res.fit.meta.get("min_node_support") is None or res.fit.meta["min_node_support"] > 0.4


def test_r2w_trajectory_source_is_honored(tmp_path):
    """r2w must fit the FULL source (global + trajectories): against the exact
    bundle reconstruction, the with-motion fit matches and the without-motion
    fit misses by the trajectory magnitude."""
    xml, r = _bundle_from_synthetic_warp(tmp_path)
    out_with = tmp_path / "with_mot.xml"
    relion_to_warp(
        xml, out_with, tomo_name="TS_RT", optimisation_set=r.optimisation_set,
        movement_grid=(4, 4),
    )
    out_without = tmp_path / "without_mot.xml"
    relion_to_warp(
        xml, out_without, tomo_name="TS_RT",
        tomograms_star=r.tomograms_star, particles_star=r.particles_star,
        motion_star=None, movement_grid=(4, 4),
    )
    p_prime = r.lift.positions_out_a
    rel = r.global_result.model
    rot = rel.projection_matrices[:, :3, :3]
    trans = rel.projection_matrices[:, :3, 3]
    full_src = (
        torch.einsum("tij,tpj->tpi", rot, (p_prime[None] + r.lift.motion_a) / PIX)
        + trans[:, None, :]
    )[..., :2] * PIX
    rows = torch.tensor(r.rows)

    def rms_vs_full(path):
        m = WarpTiltSeriesModel(load_warp_tiltseries(path).ts)
        xy, valid = m.project_volume(p_prime)
        v = valid[rows]
        return float(
            (xy.to(torch.float64)[rows] - full_src)[v].norm(dim=-1).pow(2).mean() ** 0.5
        )

    with_rms = rms_vs_full(out_with)
    without_rms = rms_vs_full(out_without)
    assert with_rms / PIX < 0.1
    assert without_rms > 3 * with_rms  # trajectories carried the locals


def test_r2w_gate_fails_on_clustered_particles(tmp_path):
    xml = w2r_helpers._write_synthetic_xml(tmp_path / "src.xml")
    stack = w2r_helpers._dummy_stack(tmp_path / "stack.mrc")
    vol_a = torch.tensor(VOL_A, dtype=torch.float64)
    # 40 particles in a tiny blob: plenty of counts, no spatial support
    pos = (torch.tensor(RNG.uniform(0.48, 0.52, (40, 3)))) * vol_a
    names = [f"TS_CL/{i + 1}" for i in range(pos.shape[0])]
    r = warp_to_relion(
        xml, tmp_path / "bundle", pixel_size_a=PIX, tomo_name="TS_CL",
        positions_eff_a=pos, particle_names=names, tilt_stack=stack,
    )
    with pytest.raises(RuntimeError, match="gates failed"):
        relion_to_warp(
            xml, tmp_path / "out.xml", tomo_name="TS_CL",
            optimisation_set=r.optimisation_set, movement_grid=(5, 5),
        )


def test_relion_model_matrix_precedence_warns_on_disagreement(tmp_path):
    from cets_nonrigid.convert_relion import relion_model_from_data
    from cets_nonrigid.io.relion_star import RelionTomogramData
    from cets_nonrigid.models.relion_ts import RelionTomogramModel

    t = 3
    base = RelionTomogramModel(
        xtilt_deg=torch.zeros(t, dtype=torch.float64),
        ytilt_deg=torch.linspace(-30, 30, t, dtype=torch.float64),
        zrot_deg=torch.full((t,), 85.0, dtype=torch.float64),
        xshift_a=torch.zeros(t, dtype=torch.float64),
        yshift_a=torch.zeros(t, dtype=torch.float64),
        tomo_dims_px=(48, 48, 24), image_dims_px=(96, 96), pixel_size_a=2.0,
    )
    data = RelionTomogramData(
        name="TS_P", voltage_kv=300.0, cs_mm=2.7, amplitude_contrast=0.07, hand=1,
        pixel_size_a=2.0, tomo_dims_px=(48, 48, 24), image_dims_px=(96, 96),
        xtilt_deg=torch.zeros(t, dtype=torch.float64),
        ytilt_deg=torch.linspace(-30, 30, t, dtype=torch.float64) + 5.0,  # DISAGREES
        zrot_deg=torch.full((t,), 85.0, dtype=torch.float64),
        xshift_a=torch.zeros(t, dtype=torch.float64),
        yshift_a=torch.zeros(t, dtype=torch.float64),
        pre_exposure=torch.arange(t, dtype=torch.float64),
        nominal_stage_angle_deg=torch.linspace(-30, 30, t, dtype=torch.float64),
        matrices=base.projection_matrices,
    )
    with pytest.warns(UserWarning, match="authoritative"):
        model = relion_model_from_data(data, (96, 96))
    torch.testing.assert_close(model.projection_matrices, base.projection_matrices)


# --- r2a ----------------------------------------------------------------------


def test_a2r_r2a_roundtrip(tmp_path):
    """AreTomo .aln (with locals) -> RELION bundle -> fitted .aln: the
    recovered model must reproduce the bundle observables at held-out
    particles (plan gate analog: IDW fields within 0.5 px)."""
    import test_r4_a2r as a2r_helpers

    from cets_nonrigid.convert_relion import aretomo_to_relion, relion_to_aretomo
    from cets_nonrigid.io.aln import load_aln
    from cets_nonrigid.io.dose import raw_dose_from_tlt

    aln_path, tlt, ctf, stack, _pos3 = a2r_helpers._synthetic_inputs(tmp_path)
    vol_a = torch.tensor(a2r_helpers.TOMO, dtype=torch.float64) * a2r_helpers.PIX
    pos = torch.tensor(RNG.uniform(0.08, 0.92, (200, 3))) * vol_a
    names = [f"TS_RA/{i + 1}" for i in range(pos.shape[0])]
    r = aretomo_to_relion(
        aln_path, tmp_path / "bundle",
        pixel_size_a=a2r_helpers.PIX, tomo_name="TS_RA", tomo_dims_px=a2r_helpers.TOMO,
        voltage_kv=300.0, cs_mm=2.7, amplitude_contrast=0.07, hand=1,
        positions_eff_a=pos, particle_names=names,
        raw_dose=raw_dose_from_tlt(tlt), ctf_file=ctf, tilt_stack=stack,
    )

    out_aln = tmp_path / "roundtrip.aln"
    res = relion_to_aretomo(
        out_aln,
        tomo_name="TS_RA",
        image_dims_px=a2r_helpers.IMG,
        optimisation_set=r.optimisation_set,
        patch_grid=(2, 2),
    )
    assert res.local_fit.heldout_status == "evaluated"
    assert res.local_fit.rms_px_heldout < 0.5
    assert res.ctf_path is not None and res.ctf_path.exists()

    # recovered .aln reproduces the bundle observables at the gauged particles
    recovered = load_aln(
        out_aln, pixel_size_a=a2r_helpers.PIX, volume_dims_a=tuple(float(v) for v in vol_a)
    )
    p_prime = r.lift.positions_out_a
    rel = r.global_result.model
    rot = rel.projection_matrices[:, :3, :3]
    trans = rel.projection_matrices[:, :3, 3]
    full_src = (
        torch.einsum("tij,tpj->tpi", rot, (p_prime[None] + r.lift.motion_a) / a2r_helpers.PIX)
        + trans[:, None, :]
    )[..., :2] * a2r_helpers.PIX  # (T_bundle, P, 2), bundle row order
    xy, valid = recovered.model.project_volume(p_prime)
    # recovered aln rows are TILT-ascending == the bundle's emission order here
    v = valid & torch.isfinite(full_src).all(dim=-1)
    err = (xy.to(torch.float64) - full_src)[v].norm(dim=-1)
    assert float(err.pow(2).mean().sqrt()) / a2r_helpers.PIX < 0.5
