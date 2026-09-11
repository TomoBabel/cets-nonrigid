"""R3: RELION star IO — write/read round trips + projection-matrix precedence."""

import numpy as np
import pytest
import torch

from cets_nonrigid.ctf import TiltCtf
from cets_nonrigid.io.relion_star import (
    RelionTomogramData,
    read_motion_star,
    read_optimisation_set,
    read_particles_star,
    read_tomograms_star,
    write_motion_star,
    write_optimisation_set,
    write_particles_star,
    write_tomograms_star,
)

RNG = np.random.default_rng(20260903)


def _tomo(t=6, with_ctf=True, **over):
    ctf = None
    if with_ctf:
        u = torch.tensor(RNG.uniform(15000, 30000, t))
        ctf = TiltCtf(
            defocus_u_a=u,
            defocus_v_a=u - 500,
            angle_deg=torch.full((t,), 45.0, dtype=torch.float64),
            phase_deg=torch.zeros(t, dtype=torch.float64),
        )
    kw = dict(  # noqa: C408 - updated via **over
        name="TS_TEST",
        voltage_kv=300.0,
        cs_mm=2.7,
        amplitude_contrast=0.07,
        hand=1,
        pixel_size_a=2.0,
        tomo_dims_px=(48, 48, 24),
        image_dims_px=(96, 96),
        xtilt_deg=torch.zeros(t, dtype=torch.float64),
        ytilt_deg=torch.linspace(-60, 60, t, dtype=torch.float64),
        zrot_deg=torch.full((t,), 85.0, dtype=torch.float64),
        xshift_a=torch.tensor(RNG.uniform(-15, 15, t)),
        yshift_a=torch.tensor(RNG.uniform(-15, 15, t)),
        pre_exposure=torch.arange(t, dtype=torch.float64) * 3.0,
        nominal_stage_angle_deg=torch.linspace(-60, 60, t, dtype=torch.float64),
        ctf=ctf,
        micrograph_names=[f"tilts/TS_TEST_{i:03d}.mrc" for i in range(t)],
    )
    kw.update(over)
    return RelionTomogramData(**kw)


def test_tomograms_star_roundtrip(tmp_path):
    tomo = _tomo()
    write_tomograms_star(tmp_path, tomo)
    back = read_tomograms_star(tmp_path / "tomograms.star")
    assert list(back) == ["TS_TEST"]
    b = back["TS_TEST"]
    tol = {"atol": 1e-5, "rtol": 0}  # starfile writes %.6f
    torch.testing.assert_close(b.ytilt_deg, tomo.ytilt_deg, **tol)
    torch.testing.assert_close(b.zrot_deg, tomo.zrot_deg, **tol)
    torch.testing.assert_close(b.xshift_a, tomo.xshift_a, **tol)
    torch.testing.assert_close(b.pre_exposure, tomo.pre_exposure, **tol)
    assert b.hand == 1 and b.tomo_dims_px == (48, 48, 24)
    assert b.micrograph_names == tomo.micrograph_names
    torch.testing.assert_close(b.ctf.defocus_u_a, tomo.ctf.defocus_u_a, atol=1e-4, rtol=0)
    assert b.matrices is None


def test_tomograms_star_requires_ctf(tmp_path):
    with pytest.raises(ValueError, match="requires per-tilt CTF"):
        write_tomograms_star(tmp_path, _tomo(with_ctf=False))


def test_projection_matrix_precedence_on_read(tmp_path):
    """rlnTomoProjX/Y/Z/W are authoritative when present (RELION-4 style)."""
    ts_dir = tmp_path / "tilt_series"
    ts_dir.mkdir()
    (tmp_path / "tomograms.star").write_text(
        "\ndata_global\n\nloop_\n"
        "_rlnTomoName #1\n_rlnVoltage #2\n_rlnSphericalAberration #3\n"
        "_rlnAmplitudeContrast #4\n_rlnTomoHand #5\n_rlnTomoTiltSeriesPixelSize #6\n"
        "_rlnTomoSizeX #7\n_rlnTomoSizeY #8\n_rlnTomoSizeZ #9\n"
        "_rlnTomoTiltSeriesStarFile #10\n"
        f"TS_M 300 2.7 0.07 1 2.0 48 48 24 {ts_dir}/TS_M.star\n"
    )
    (ts_dir / "TS_M.star").write_text(
        "\ndata_TS_M\n\nloop_\n"
        "_rlnTomoYTilt #1\n_rlnTomoZRot #2\n_rlnTomoXShiftAngst #3\n_rlnTomoYShiftAngst #4\n"
        "_rlnDefocusU #5\n_rlnDefocusV #6\n_rlnDefocusAngle #7\n_rlnMicrographPreExposure #8\n"
        "_rlnTomoProjX #9\n_rlnTomoProjY #10\n_rlnTomoProjZ #11\n_rlnTomoProjW #12\n"
        "0 0 0 0 20000 20000 0 3.0 [1,0,0,10] [0,1,0,20] [0,0,1,0] [0,0,0,1]\n"
        "30 85 5 5 21000 20500 10 6.0 [0.5,0.5,0,1] [-0.5,0.5,0,2] [0,0,1,3] [0,0,0,1]\n"
    )
    back = read_tomograms_star(tmp_path / "tomograms.star")["TS_M"]
    assert back.matrices is not None and back.matrices.shape == (2, 4, 4)
    torch.testing.assert_close(
        back.matrices[0],
        torch.tensor(
            [[1.0, 0, 0, 10], [0, 1, 0, 20], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float64
        ),
    )
    # Euler columns still read (for the disagreement warning in the pipeline)
    assert float(back.ytilt_deg[1]) == 30.0


def test_particles_star_roundtrip(tmp_path):
    p = 12
    coords = torch.tensor(RNG.uniform(-40, 40, (p, 3)))
    names = [f"TS_TEST/{i + 1}" for i in range(p)]
    path = write_particles_star(
        tmp_path / "particles.star",
        tomo_name="TS_TEST",
        particle_names=names,
        centered_coords_a=coords,
        voltage_kv=300.0,
        cs_mm=2.7,
        amplitude_contrast=0.07,
        pixel_size_a=2.0,
    )
    back = read_particles_star(path)
    assert back.particle_names == names
    assert back.tomo_names == ["TS_TEST"] * p
    torch.testing.assert_close(back.centered_coords_a, coords, atol=1e-5, rtol=0)
    torch.testing.assert_close(back.origins_a, torch.zeros(p, 3, dtype=torch.float64))
    assert back.optics_pixel_size_a == {1: 2.0}


def test_motion_star_roundtrip(tmp_path):
    t, p = 6, 5
    names = [f"TS_TEST/{i + 1}" for i in range(p)]
    motion = torch.tensor(RNG.uniform(-8, 8, (t, p, 3)))
    path = write_motion_star(tmp_path / "motion.star", names, motion)
    text = path.read_text()
    assert "_rlnParticleNumber" in text  # the verified label, NOT rlnNrOfParticles
    assert "data_TS_TEST/1" in text
    back = read_motion_star(path, names, t)
    torch.testing.assert_close(back, motion.to(torch.float64), atol=1e-5, rtol=0)

    # order-independence: shuffled name order returns matching columns
    order = list(RNG.permutation(p))
    back2 = read_motion_star(path, [names[i] for i in order], t)
    torch.testing.assert_close(back2, motion.to(torch.float64)[:, order], atol=1e-5, rtol=0)

    with pytest.raises(ValueError, match="no trajectory block"):
        read_motion_star(path, ["TS_TEST/99"], t)
    with pytest.raises(ValueError, match="rows, expected"):
        read_motion_star(path, names, t + 1)


def test_optimisation_set_roundtrip(tmp_path):
    path = write_optimisation_set(
        tmp_path / "optimisation_set.star",
        particles="particles.star",
        tomograms="tomograms.star",
        trajectories="motion.star",
    )
    back = read_optimisation_set(path)
    assert back == {
        "particles": "particles.star",
        "tomograms": "tomograms.star",
        "trajectories": "motion.star",
    }
