"""R4: w2r — Warp + particles -> RELION bundle.

Headline (binary-gated): a stack synthesized from the FULL Warp model
(nonzero movement grids + level angles) at the particles, converted by w2r,
must extract CENTERED through relion_tomo_subtomo — "polished particles
embody the source local alignments", proven executably. XY is exact; with a
nonzero GridVolumeWarp the particle-depth CTF would be approximate (beam-axis
motion is dropped by the min-norm lift).
"""

import numpy as np
import pytest
import test_r3_relion_binary as rb
import torch

from cets_nonrigid.conventions import RELION_TOMO_HAND_FOR_WARP_NOT_INVERTED
from cets_nonrigid.convert_relion import warp_to_relion
from cets_nonrigid.ctf import TiltCtf, tiltctf_to_warp
from cets_nonrigid.io.relion_star import read_motion_star, read_particles_star, read_tomograms_star
from cets_nonrigid.io.warp_xml import load_warp_tiltseries

RNG = np.random.default_rng(20260905)

T, PIX = 8, 2.0
IMG_A = (192.0, 192.0)  # 96 x 96 px
VOL_A = (96.0, 96.0, 48.0)  # 48 x 48 x 24 px
DARK_ROW = 2


def _write_synthetic_xml(path, *, inverted=False, with_ctf=True, dark=True):
    from warpylib import CubicGrid, TiltSeries
    from warpylib.linear_grid import LinearGrid4D
    from warpylib.tilt_series.io import save_meta

    gen = torch.Generator().manual_seed(11)
    ts = TiltSeries()
    ts.image_dimensions_physical = torch.tensor(IMG_A)
    ts.volume_dimensions_physical = torch.tensor(VOL_A)
    ts.angles = torch.linspace(52.5, -52.5, T)
    ts.tilt_axis_angles = torch.full((T,), 85.0) + torch.rand(T, generator=gen)
    ts.tilt_axis_offset_x = torch.rand(T, generator=gen) * 10 - 5
    ts.tilt_axis_offset_y = torch.rand(T, generator=gen) * 10 - 5
    ts.dose = torch.arange(T, dtype=torch.float32) * 3.0
    ts.use_tilt = torch.ones(T, dtype=torch.bool)
    if dark:
        ts.use_tilt[DARK_ROW] = False
    ts.level_angle_x = 2.0
    ts.level_angle_y = -1.5
    ts.are_angles_inverted = inverted
    ts.grid_movement_x = CubicGrid((3, 3, T), (torch.rand(9 * T, generator=gen) - 0.5) * 8)
    ts.grid_movement_y = CubicGrid((3, 3, T), (torch.rand(9 * T, generator=gen) - 0.5) * 8)
    ts.grid_volume_warp_x = LinearGrid4D((1, 1, 1, 1))
    ts.grid_volume_warp_y = LinearGrid4D((1, 1, 1, 1))
    ts.grid_volume_warp_z = LinearGrid4D((1, 1, 1, 1))
    if with_ctf:
        u = torch.tensor(RNG.uniform(15000, 30000, T))
        tiltctf_to_warp(
            ts,
            TiltCtf(
                defocus_u_a=u, defocus_v_a=u - 400,
                angle_deg=torch.full((T,), 30.0, dtype=torch.float64),
                phase_deg=torch.zeros(T, dtype=torch.float64),
                voltage_kv=300.0, cs_mm=2.7, amplitude_contrast=0.07,
            ),
        )
    ts.path = str(path)
    save_meta(ts, str(path))
    return path


def _particles():
    vol_a = torch.tensor(VOL_A, dtype=torch.float64)
    frac = torch.tensor(
        [[0.15, 0.20, 0.40], [0.85, 0.50, 0.60], [0.40, 0.85, 0.50]],
        dtype=torch.float64,
    )
    return frac * vol_a


def _dummy_stack(path):
    import mrcfile

    with mrcfile.new(path) as m:
        m.set_data(np.zeros((T, 96, 96), dtype=np.float32))
        m.voxel_size = PIX
    return path


def test_w2r_bundle_exactness_and_gauge(tmp_path):
    xml = _write_synthetic_xml(tmp_path / "syn.xml")
    stack = _dummy_stack(tmp_path / "stack.mrc")
    pos = _particles()
    names = [f"TS_W2R/{i + 1}" for i in range(pos.shape[0])]

    r = warp_to_relion(
        xml, tmp_path / "bundle",
        pixel_size_a=PIX, tomo_name="TS_W2R",
        positions_eff_a=pos, particle_names=names,
        tilt_stack=stack,
    )

    # exactness: closed-form global, no fallback; lift at float32-source level
    assert r.global_result.global_exact and not r.global_result.used_fallback
    assert r.global_result.rms_px < 1e-3
    assert r.lift.max_residual_px < 1e-3
    # darks dropped, order preserved
    assert DARK_ROW not in r.rows and len(r.rows) == T - 1
    assert r.hand == RELION_TOMO_HAND_FOR_WARP_NOT_INVERTED

    # bundle reads back consistently
    tomo = read_tomograms_star(r.tomograms_star)["TS_W2R"]
    assert tomo.n_tilts == T - 1
    assert tomo.ctf is not None
    parts = read_particles_star(r.particles_star)
    assert parts.particle_names == names  # every input particle, original order
    motion = read_motion_star(r.motion_star, names, T - 1)
    # gauge: zero at the lowest-dose emitted row
    ref = int(torch.argmin(torch.tensor([float(d) for d in tomo.pre_exposure])))
    assert ref == r.lift.ref_row
    torch.testing.assert_close(motion[ref], torch.zeros_like(motion[ref]), atol=1e-5, rtol=0)
    # trajectories are nonzero elsewhere (the movement grids are nonzero)
    assert motion.abs().max() > 0.5

    # Check the same exactness and trajectory shape directly; CETS payload
    # dimension names are checked in test_cets_payload.
    assert r.global_result.global_exact is True
    assert r.lift.motion_a.shape[-1] == 3
    from cets_nonrigid import api
    sampled = api.to_cets("warp", xml, pixel_size_a=PIX, positions_a=pos, names=names)
    assert sampled.samples.heldout.count == 0


def test_w2r_reconstruction_matches_source_model(tmp_path):
    """P(p' + motion) must reproduce the FULL Warp model at every particle."""
    xml = _write_synthetic_xml(tmp_path / "syn.xml")
    series = load_warp_tiltseries(xml)
    stack = _dummy_stack(tmp_path / "stack.mrc")
    pos = _particles()
    names = [f"TS_C/{i + 1}" for i in range(pos.shape[0])]
    r = warp_to_relion(
        xml, tmp_path / "bundle", pixel_size_a=PIX, tomo_name="TS_C",
        positions_eff_a=pos, particle_names=names, tilt_stack=stack,
    )

    src_xy, _ = series.model.project_volume(pos)
    rows = torch.tensor(r.rows)
    rel = r.global_result.model
    check = r.lift.positions_out_a[None] + r.lift.motion_a
    rot = rel.projection_matrices[:, :3, :3]
    trans = rel.projection_matrices[:, :3, 3]
    xy = (torch.einsum("tij,tpj->tpi", rot, check / PIX) + trans[:, None, :])[..., :2] * PIX
    err = (xy - src_xy.to(torch.float64)[rows]).norm(dim=-1)
    assert float(err.max()) < 1e-3 * PIX  # px-scale exactness incl. float32 source


def test_w2r_hand_follows_inversion(tmp_path):
    for inverted in (False, True):
        xml = _write_synthetic_xml(tmp_path / f"syn_{inverted}.xml", inverted=inverted)
        stack = _dummy_stack(tmp_path / f"stack_{inverted}.mrc")
        pos = _particles()
        r = warp_to_relion(
            xml, tmp_path / f"bundle_{inverted}", pixel_size_a=PIX, tomo_name="TS_H",
            positions_eff_a=pos, particle_names=[f"TS_H/{i + 1}" for i in range(3)],
            tilt_stack=stack,
        )
        expect = RELION_TOMO_HAND_FOR_WARP_NOT_INVERTED * (-1 if inverted else 1)
        assert r.hand == expect


def test_w2r_requires_ctf_or_flag(tmp_path):
    xml = _write_synthetic_xml(tmp_path / "syn.xml", with_ctf=False)
    stack = _dummy_stack(tmp_path / "stack.mrc")
    pos = _particles()
    with pytest.raises(ValueError, match="no per-tilt CTF"):
        warp_to_relion(
            xml, tmp_path / "b1", pixel_size_a=PIX, tomo_name="TS_N",
            positions_eff_a=pos, particle_names=["TS_N/1", "TS_N/2", "TS_N/3"],
            tilt_stack=stack,
        )
    r = warp_to_relion(
        xml, tmp_path / "b2", pixel_size_a=PIX, tomo_name="TS_N",
        positions_eff_a=pos, particle_names=["TS_N/1", "TS_N/2", "TS_N/3"],
        tilt_stack=stack, no_ctf=True,
    )
    assert r.tomograms_star.exists()


# --- headline: extraction through the real binary ------------------------------



@pytest.mark.skipif(
    rb.RELION_SUBTOMO is None,
    reason="relion_tomo_subtomo not found (set AREWARPION_RELION_BIN or module load relion/5.0.0)",
)
def test_w2r_headline_polished_extraction(tmp_path):
    import subprocess

    import mrcfile

    xml = _write_synthetic_xml(tmp_path / "syn.xml")
    series = load_warp_tiltseries(xml)
    pos = _particles()
    names = [f"TS_HL/{i + 1}" for i in range(pos.shape[0])]

    # paint the stack from the FULL Warp model (globals + movement grids)
    src_xy, _ = series.model.project_volume(pos)  # (T, P, 2) canonical A
    proj_px = src_xy.to(torch.float64) / PIX
    yy, xx = np.mgrid[0:96, 0:96]
    frames = np.zeros((T, 96, 96), dtype=np.float32)
    for f in range(T):
        for j in range(pos.shape[0]):
            x, y = float(proj_px[f, j, 0]), float(proj_px[f, j, 1])
            frames[f] += np.exp(-(((xx - x) ** 2) + ((yy - y) ** 2)) / (2 * 1.5**2)).astype(
                np.float32
            )
    stack = tmp_path / "stack.mrc"
    with mrcfile.new(stack) as m:
        m.set_data(frames)
        m.voxel_size = PIX

    r = warp_to_relion(
        xml, tmp_path / "bundle", pixel_size_a=PIX, tomo_name="TS_HL",
        positions_eff_a=pos, particle_names=names, tilt_stack=stack,
    )

    out = tmp_path / "out"
    cmd = [
        rb.RELION_SUBTOMO,
        "--p", str(r.particles_star),
        "--t", str(r.tomograms_star),
        "--mot", str(r.motion_star),
        "--o", str(out) + "/",
        "--b", "24", "--crop", "24", "--bin", "1",
        "--stack2d", "--no_ctf", "--no_ic", "--j", "2",
    ]
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
    assert run.returncode == 0, f"relion_tomo_subtomo failed:\n{run.stdout}\n{run.stderr}"

    mag = rb._centroid_offsets(out).norm(dim=-1)
    assert float(mag.mean()) <= 0.1, f"mean centroid offset {mag.mean():.3f} px"
    assert float(mag.quantile(0.95)) <= 0.2

    # sanity contrast: dropping the trajectories must de-center the impulses
    out2 = tmp_path / "out_nomot"
    run2 = subprocess.run(
        [c for c in cmd if c not in ("--mot", str(r.motion_star))][:6]
        + ["--o", str(out2) + "/", "--b", "24", "--crop", "24", "--bin", "1",
           "--stack2d", "--no_ctf", "--no_ic", "--j", "2"],
        capture_output=True, text=True, timeout=600, check=False,
    )
    assert run2.returncode == 0, run2.stderr
    mag2 = rb._centroid_offsets(out2).norm(dim=-1)
    assert float(mag2.max()) > 0.5, "local alignments had no effect on extraction?"
