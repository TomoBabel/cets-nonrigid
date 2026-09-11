"""R1: RELION 5 models — golden-first equivalence tests.

Every reference implementation in this file is a LITERAL transcription of the
cited relion 5.0.1 (checkout 210f68c8) source, written in numpy and kept
independent of the torch models under test.
"""

import numpy as np
import pytest
import torch

from cets_nonrigid.models.base import FrameMotionModel, TiltProjectionModel
from cets_nonrigid.models.relion_motion import RelionMicrographMotionModel
from cets_nonrigid.models.relion_ts import (
    RelionParticleSetModel,
    RelionTomogramModel,
    angles_to_matrix3,
    effective_positions_a,
)

RNG = np.random.default_rng(20260831)


# --- literal transcriptions ---------------------------------------------------


def _gravis_rotation(axis, angle_deg):
    """t3Matrix<double>::rotation — verbatim (src/jaz/gravis/t3Matrix.h:476-496)."""
    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    angle = angle_deg * (np.pi / 180.0)
    s = np.array([[0.0, -n[2], n[1]], [n[2], 0.0, -n[0]], [-n[1], n[0], 0.0]])
    nnt = np.outer(n, n)
    return nnt + np.cos(angle) * (np.eye(3) - nnt) + np.sin(angle) * s


def _cpp_projection_matrix(w0, h0, d0, nx, ny, pix, xtilt, ytilt, zrot, xshift_a, yshift_a):
    """Tomogram::setProjectionMatrix — verbatim (src/jaz/tomography/tomogram.cpp:41-62)."""

    def trans(v):
        m = np.eye(4)
        m[:3, 3] = v
        return m

    def rot4(axis, ang):
        m = np.eye(4)
        m[:3, :3] = _gravis_rotation(axis, ang)
        return m

    s0 = trans([-float(int(w0 / 2)), -float(int(h0 / 2)), -float(int(d0 / 2))])  # :44-45
    s1 = trans([xshift_a / pix, yshift_a / pix, 0.0])  # :48-49
    s2 = trans([float(int(nx / 2)), float(int(ny / 2)), 0.0])  # :52-54
    r0 = rot4([1, 0, 0], xtilt)  # :58
    r1 = rot4([0, 1, 0], ytilt)  # :59
    r2 = rot4([0, 0, 1], zrot)  # :60
    return s1 @ s2 @ r2 @ r1 @ r0 @ s0  # :62


def _py_affine_rot(kind, deg):
    """Rx/Ry/Rz — transcription of src/tomography_python_programs/_utils/transformations.py."""
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    m = np.eye(4)
    if kind == "x":
        m[1, 1], m[1, 2], m[2, 1], m[2, 2] = c, -s, s, c
    elif kind == "y":
        m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
    else:
        m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
    return m


def _python_reference_matrix(nx, ny, w0, h0, d0, xtilt, ytilt, zrot, shift_px_xy):
    """tilt_series_alignment_parameters_to_relion_projection_matrices — transcription
    of src/tomography_python_programs/align_tilt_series/_job_utils.py:25-66
    (FLOAT centers: dims / 2)."""

    def shift(v):
        m = np.eye(4)
        m[:3, 3] = v
        return m

    tilt_image_center = np.array([nx, ny, 0.0]) / 2.0
    tilt_image_center[2] = 0.0  # promote_2d_to_3d pads z with 0
    specimen_center = np.array([w0, h0, d0]) / 2.0
    s0 = shift(-specimen_center)
    r0 = _py_affine_rot("x", xtilt)
    r1 = _py_affine_rot("y", ytilt)
    r2 = _py_affine_rot("z", zrot)
    s1 = shift([shift_px_xy[0], shift_px_xy[1], 0.0])
    s2 = shift([nx / 2.0, ny / 2.0, 0.0])
    return s2 @ s1 @ r2 @ r1 @ r0 @ s0  # :65


def _cpp_angles_to_matrix3(phi_rad, theta_rad, chi_rad):
    """Euler::anglesToMatrix3 — verbatim (src/jaz/math/Euler_angles_relion.h:38-48)."""
    sp, cp = np.sin(phi_rad), np.cos(phi_rad)
    st, ct = np.sin(theta_rad), np.cos(theta_rad)
    sc, cc = np.sin(chi_rad), np.cos(chi_rad)
    return np.array(
        [
            [cc * ct * cp - sc * sp, cc * ct * sp + sc * cp, -cc * st],
            [-sc * ct * cp - cc * sp, -sc * ct * sp + cc * cp, sc * st],
            [st * cp, st * sp, ct],
        ]
    )


def _cpp_get_position(
    coords, tomo_centre_px, pix, origins_a, subtomo_deg, legacy
):
    """ParticleSet::getParticleCoordDecenteredPixel + getPosition — verbatim
    (src/jaz/tomography/particle_set.cpp:589-619, :355-379). Returns decentered px."""
    out = np.array(coords, dtype=float)
    if not legacy:
        out = out / pix  # :602
        out = out + tomo_centre_px  # :603
    if origins_a is not None:
        a = _cpp_angles_to_matrix3(*np.deg2rad(subtomo_deg)) if subtomo_deg is not None else np.eye(3)
        out = out - (a @ np.asarray(origins_a, float)) / pix  # :365-367
    return out


def _cpp_poly_shift(coeff_x, coeff_y, z, x, y):
    """ThirdOrderPolynomialModel::getShiftAt — verbatim (src/micrograph_model.cpp:32-51)."""
    x2, y2, xy = x * x, y * y, x * y
    z2 = z * z
    z3 = z2 * z
    sx = (
        (coeff_x[0] * z + coeff_x[1] * z2 + coeff_x[2] * z3)
        + (coeff_x[3] * z + coeff_x[4] * z2 + coeff_x[5] * z3) * x
        + (coeff_x[6] * z + coeff_x[7] * z2 + coeff_x[8] * z3) * x2
        + (coeff_x[9] * z + coeff_x[10] * z2 + coeff_x[11] * z3) * y
        + (coeff_x[12] * z + coeff_x[13] * z2 + coeff_x[14] * z3) * y2
        + (coeff_x[15] * z + coeff_x[16] * z2 + coeff_x[17] * z3) * xy
    )
    sy = (
        (coeff_y[0] * z + coeff_y[1] * z2 + coeff_y[2] * z3)
        + (coeff_y[3] * z + coeff_y[4] * z2 + coeff_y[5] * z3) * x
        + (coeff_y[6] * z + coeff_y[7] * z2 + coeff_y[8] * z3) * x2
        + (coeff_y[9] * z + coeff_y[10] * z2 + coeff_y[11] * z3) * y
        + (coeff_y[12] * z + coeff_y[13] * z2 + coeff_y[14] * z3) * y2
        + (coeff_y[15] * z + coeff_y[16] * z2 + coeff_y[17] * z3) * xy
    )
    return sx, sy


NOT_OBSERVED = -9999.0


def _cpp_get_shift_at(global_px, coeffs, first_frame, width, height, frame_1, x_px, y_px):
    """Micrograph::getShiftAt with normalise=true — verbatim
    (src/micrograph_model.cpp:360-406). frame_1 is 1-indexed. Returns
    (status, sx, sy) in unbinned px."""
    x = x_px / width - 0.5  # :366
    y = y_px / height - 0.5  # :367
    gx = global_px[frame_1 - 1]
    if gx[0] == NOT_OBSERVED or gx[1] == NOT_OBSERVED:  # (:370 tests X twice; both here)
        sx = sy = 0.0  # :376
        for i in range(frame_1 - 1, -1, -1):  # :378
            if global_px[i][0] != NOT_OBSERVED and global_px[i][1] != NOT_OBSERVED:  # :380
                sx, sy = global_px[i]  # :382-383
                break
        return -1, sx, sy  # :387
    if coeffs is not None:
        sx, sy = _cpp_poly_shift(coeffs[:18], coeffs[18:36], frame_1 - first_frame, x, y)  # :393
    else:
        sx = sy = 0.0  # :397-398
    sx += global_px[frame_1 - 1][0]  # :402
    sy += global_px[frame_1 - 1][1]  # :403
    return 0, sx, sy


# --- helpers ------------------------------------------------------------------


def _random_globals(t=7):
    return {
        "xtilt_deg": torch.tensor(RNG.uniform(-4, 4, t)),
        "ytilt_deg": torch.tensor(RNG.uniform(-58, 58, t)),
        "zrot_deg": torch.tensor(RNG.uniform(-185, 185, t)),
        "xshift_a": torch.tensor(RNG.uniform(-90, 90, t)),
        "yshift_a": torch.tensor(RNG.uniform(-90, 90, t)),
    }


EVEN = {"tomo_dims_px": (64, 64, 32), "image_dims_px": (128, 96)}
ODD = {"tomo_dims_px": (63, 65, 31), "image_dims_px": (127, 95)}


# --- RelionTomogramModel ------------------------------------------------------


@pytest.mark.parametrize("dims", [EVEN, ODD], ids=["even", "odd"])
def test_projection_matrix_matches_cpp_transcription(dims):
    pix = 2.34
    g = _random_globals()
    model = RelionTomogramModel(**g, **dims, pixel_size_a=pix)
    w0, h0, d0 = dims["tomo_dims_px"]
    nx, ny = dims["image_dims_px"]
    for f in range(model.n_projections):
        ref = _cpp_projection_matrix(
            w0, h0, d0, nx, ny, pix,
            g["xtilt_deg"][f].item(), g["ytilt_deg"][f].item(), g["zrot_deg"][f].item(),
            g["xshift_a"][f].item(), g["yshift_a"][f].item(),
        )
        np.testing.assert_allclose(model.projection_matrices[f].numpy(), ref, atol=1e-12)

    # projected positions: model (canonical A) vs manual matrix application
    pts_a = torch.tensor(RNG.uniform(0, 1, (40, 3))) * model.volume_dims_a
    xy_a, valid = model.project_volume(pts_a)
    p_px = (pts_a / pix).numpy()
    for f in range(model.n_projections):
        ref = _cpp_projection_matrix(
            w0, h0, d0, nx, ny, pix,
            g["xtilt_deg"][f].item(), g["ytilt_deg"][f].item(), g["zrot_deg"][f].item(),
            g["xshift_a"][f].item(), g["yshift_a"][f].item(),
        )
        hom = np.c_[p_px, np.ones(len(p_px))]
        ref_xy = (ref @ hom.T).T[:, :2] * pix
        np.testing.assert_allclose(xy_a[f].numpy(), ref_xy, atol=1e-8)
    assert valid.shape == (model.n_projections, 40)


def test_even_dims_match_shipped_python_reference():
    pix = 1.87
    g = _random_globals()
    model = RelionTomogramModel(**g, **EVEN, pixel_size_a=pix)
    w0, h0, d0 = EVEN["tomo_dims_px"]
    nx, ny = EVEN["image_dims_px"]
    for f in range(model.n_projections):
        ref = _python_reference_matrix(
            nx, ny, w0, h0, d0,
            g["xtilt_deg"][f].item(), g["ytilt_deg"][f].item(), g["zrot_deg"][f].item(),
            (g["xshift_a"][f].item() / pix, g["yshift_a"][f].item() / pix),
        )
        np.testing.assert_allclose(model.projection_matrices[f].numpy(), ref, atol=1e-10)


def test_odd_dims_use_int_div_centers_not_float():
    # Guard: on odd dims the model must match the C++ int-div centers and
    # therefore DIFFER from the float-centre python reference.
    pix = 2.0
    g = _random_globals(t=3)
    model = RelionTomogramModel(**g, **ODD, pixel_size_a=pix)
    w0, h0, d0 = ODD["tomo_dims_px"]
    nx, ny = ODD["image_dims_px"]
    ref = _python_reference_matrix(
        nx, ny, w0, h0, d0,
        g["xtilt_deg"][0].item(), g["ytilt_deg"][0].item(), g["zrot_deg"][0].item(),
        (g["xshift_a"][0].item() / pix, g["yshift_a"][0].item() / pix),
    )
    diff = np.abs(model.projection_matrices[0].numpy() - ref).max()
    assert diff > 0.1  # ~0.5 px scale, from the centre deltas


def test_from_matrices_roundtrip_and_validation():
    pix = 2.1
    g = _random_globals(t=5)
    model = RelionTomogramModel(**g, **EVEN, pixel_size_a=pix)
    m2 = RelionTomogramModel.from_matrices(
        model.projection_matrices, pixel_size_a=pix, **EVEN
    )
    pts = torch.tensor(RNG.uniform(0, 1, (10, 3))) * model.volume_dims_a
    torch.testing.assert_close(m2.project_volume(pts)[0], model.project_volume(pts)[0])

    good = model.projection_matrices
    bad = good.clone()
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        RelionTomogramModel.from_matrices(bad, pixel_size_a=pix, **EVEN)
    bad = good.clone()
    bad[1, 3, 0] = 0.5
    with pytest.raises(ValueError, match="homogeneous"):
        RelionTomogramModel.from_matrices(bad, pixel_size_a=pix, **EVEN)
    bad = good.clone()
    bad[2, :3, :3] *= 1.02  # scaled: not orthonormal
    with pytest.raises(ValueError, match="orthonormal"):
        RelionTomogramModel.from_matrices(bad, pixel_size_a=pix, **EVEN)
    bad = good.clone()
    bad[3, :3, 0] *= -1  # reflection: orthonormal but det = -1
    with pytest.raises(ValueError, match="proper rotation"):
        RelionTomogramModel.from_matrices(bad, pixel_size_a=pix, **EVEN)


# --- Euler port + particle coordinate contract --------------------------------


def test_angles_to_matrix3_matches_transcription():
    for _ in range(10):
        rot, tilt, psi = RNG.uniform(-180, 180, 3)
        ours = angles_to_matrix3(torch.tensor(rot), torch.tensor(tilt), torch.tensor(psi))
        ref = _cpp_angles_to_matrix3(np.deg2rad(rot), np.deg2rad(tilt), np.deg2rad(psi))
        np.testing.assert_allclose(ours.numpy(), ref, atol=1e-12)
        # Guard against the anglesToMatrix4 (2,1) sign bug (-st*sp there):
        st, sp = np.sin(np.deg2rad(tilt)), np.sin(np.deg2rad(rot))
        assert ours[2, 1].item() == pytest.approx(st * sp, abs=1e-12)


def test_effective_positions_match_getposition_transcription():
    pix = 2.62
    tomo_dims = (63, 65, 31)  # odd on purpose: float centre must be used here
    p = 12
    centered_a = torch.tensor(RNG.uniform(-40, 40, (p, 3)) * pix)
    legacy_px = torch.tensor(RNG.uniform(0, 60, (p, 3)))
    origins_a = torch.tensor(RNG.uniform(-8, 8, (p, 3)))
    subtomo_deg = torch.tensor(RNG.uniform(-170, 170, (p, 3)))

    tomo_centre_px = np.array([d / 2.0 for d in tomo_dims])  # tomogram_set.cpp:314 (float)

    # centered branch, nonzero origins + nontrivial subtomogram angles
    ours = effective_positions_a(
        pixel_size_a=pix, tomo_dims_px=tomo_dims,
        centered_coords_a=centered_a, origins_a=origins_a, subtomo_angles_deg=subtomo_deg,
    )
    for i in range(p):
        ref_px = _cpp_get_position(
            centered_a[i].numpy(), tomo_centre_px, pix,
            origins_a[i].numpy(), subtomo_deg[i].numpy(), legacy=False,
        )
        np.testing.assert_allclose(ours[i].numpy(), ref_px * pix, atol=1e-9)

    # legacy branch: decentered px used as-is (no centre addition)
    ours = effective_positions_a(
        pixel_size_a=pix, tomo_dims_px=tomo_dims,
        legacy_coords_px=legacy_px, origins_a=origins_a, subtomo_angles_deg=subtomo_deg,
    )
    for i in range(p):
        ref_px = _cpp_get_position(
            legacy_px[i].numpy(), tomo_centre_px, pix,
            origins_a[i].numpy(), subtomo_deg[i].numpy(), legacy=True,
        )
        np.testing.assert_allclose(ours[i].numpy(), ref_px * pix, atol=1e-9)

    # centered columns take precedence when both exist (particle_set.cpp:593-605)
    both = effective_positions_a(
        pixel_size_a=pix, tomo_dims_px=tomo_dims,
        centered_coords_a=centered_a, legacy_coords_px=legacy_px,
        origins_a=origins_a, subtomo_angles_deg=subtomo_deg,
    )
    first = effective_positions_a(
        pixel_size_a=pix, tomo_dims_px=tomo_dims,
        centered_coords_a=centered_a, origins_a=origins_a, subtomo_angles_deg=subtomo_deg,
    )
    torch.testing.assert_close(both, first)


def test_particle_set_projection_with_trajectories():
    pix = 2.34  # non-unit on purpose: dimensional cleanliness
    g = _random_globals(t=6)
    model = RelionTomogramModel(**g, **EVEN, pixel_size_a=pix)
    p = 9
    pos_a = torch.tensor(RNG.uniform(0.2, 0.8, (p, 3))) * model.volume_dims_a
    traj_a = torch.tensor(RNG.uniform(-6, 6, (6, p, 3)))

    ps = RelionParticleSetModel(model, pos_a, trajectories_a=traj_a)
    xy_a, _valid = ps.project_volume(pos_a)
    # reference: s * (P @ ((p_eff_A + t_A) / s)) per tilt
    for f in range(6):
        m = model.projection_matrices[f].numpy()
        pos_px = ((pos_a + traj_a[f]) / pix).numpy()
        hom = np.c_[pos_px, np.ones(p)]
        ref = (m @ hom.T).T[:, :2] * pix
        np.testing.assert_allclose(xy_a[f].numpy(), ref, atol=1e-8)

    # global path: no trajectory applied
    xy_g, _ = ps.project_volume_global(pos_a)
    xy_bare, _ = model.project_volume(pos_a)
    torch.testing.assert_close(xy_g, xy_bare)
    assert (xy_a - xy_g).abs().max() > 0.1  # trajectories actually moved things

    # only defined on the bound set
    with pytest.raises(ValueError, match="bound particle set"):
        ps.project_volume(pos_a[:4])


# --- RelionMicrographMotionModel ----------------------------------------------


def _motion_fixture(f_count=9, start_frame=3, with_poly=True, mid_sentinel=None):
    g = RNG.uniform(-4, 4, (f_count, 2))
    g[: start_frame - 1] = NOT_OBSERVED  # leading truncated frames
    if mid_sentinel is not None:
        g[mid_sentinel] = NOT_OBSERVED
    coeffs = RNG.uniform(-0.05, 0.05, 36) if with_poly else None
    return g, coeffs


@pytest.mark.parametrize("with_poly", [True, False], ids=["poly", "version0"])
def test_micrograph_motion_matches_getshiftat_transcription(with_poly):
    pix = 1.7
    w, h = 4096, 4200
    start = 3
    g, coeffs = _motion_fixture(start_frame=start, with_poly=with_poly, mid_sentinel=5)
    model = RelionMicrographMotionModel(
        global_shifts_px=torch.tensor(g),
        poly_coeffs=None if coeffs is None else torch.tensor(coeffs),
        image_size_px=(w, h),
        pixel_size_a=pix,
        start_frame=start,
    )
    pts_px = RNG.uniform(0, 1, (15, 2)) * [w, h]
    raw_a, valid = model.map_image(torch.tensor(pts_px) * pix)

    for f1 in range(1, g.shape[0] + 1):
        for n, (x_px, y_px) in enumerate(pts_px):
            status, sx, sy = _cpp_get_shift_at(g, coeffs, start, w, h, f1, x_px, y_px)
            assert valid[f1 - 1, n].item() == (status == 0)
            # mapped position follows corrected -> raw: raw = x - shift
            np.testing.assert_allclose(
                raw_a[f1 - 1, n].numpy(),
                (np.array([x_px, y_px]) - np.array([sx, sy])) * pix,
                atol=1e-9,
            )


def test_micrograph_motion_global_only_and_sentinel_rules():
    g, coeffs = _motion_fixture(start_frame=2, with_poly=True)
    model = RelionMicrographMotionModel(
        global_shifts_px=torch.tensor(g), poly_coeffs=torch.tensor(coeffs),
        image_size_px=(4096, 4096), pixel_size_a=1.0, start_frame=2,
    )
    pts = torch.tensor(RNG.uniform(100, 4000, (5, 2)))
    raw_g, _ = model.map_image_global(pts)
    # global-only: spatially constant shift per frame
    per_frame_spread = (raw_g - pts[None]).std(dim=1).max()
    assert per_frame_spread < 1e-9

    # X-only sentinel marks the frame invalid (upstream tests X twice; we
    # deliberately check both axes)
    g2 = g.copy()
    g2[6] = [NOT_OBSERVED, 1.5]
    m2 = RelionMicrographMotionModel(
        global_shifts_px=torch.tensor(g2), poly_coeffs=torch.tensor(coeffs),
        image_size_px=(4096, 4096), pixel_size_a=1.0, start_frame=2,
    )
    assert not m2.frame_valid[6]
    g3 = g.copy()
    g3[6] = [1.5, NOT_OBSERVED]
    m3 = RelionMicrographMotionModel(
        global_shifts_px=torch.tensor(g3), poly_coeffs=torch.tensor(coeffs),
        image_size_px=(4096, 4096), pixel_size_a=1.0, start_frame=2,
    )
    assert not m3.frame_valid[6]


def test_micrograph_motion_rejects_binning_and_bad_coeffs():
    g = torch.zeros(4, 2)
    with pytest.raises(ValueError, match="rlnMicrographBinning"):
        RelionMicrographMotionModel(
            global_shifts_px=g, image_size_px=(100, 100), pixel_size_a=1.0, binning=2.0
        )
    with pytest.raises(ValueError, match="poly_coeffs"):
        RelionMicrographMotionModel(
            global_shifts_px=g, poly_coeffs=torch.zeros(35),
            image_size_px=(100, 100), pixel_size_a=1.0,
        )


# --- protocol conformance -----------------------------------------------------


def test_protocol_conformance():
    g = _random_globals(t=3)
    model = RelionTomogramModel(**g, **EVEN, pixel_size_a=2.0)
    assert isinstance(model, TiltProjectionModel)
    pos = torch.tensor(RNG.uniform(10, 50, (4, 3)))
    ps = RelionParticleSetModel(model, pos)
    assert isinstance(ps, TiltProjectionModel)
    motion = RelionMicrographMotionModel(
        global_shifts_px=torch.zeros(3, 2), image_size_px=(64, 64), pixel_size_a=1.0
    )
    assert isinstance(motion, FrameMotionModel)
