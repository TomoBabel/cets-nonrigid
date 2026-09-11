"""R5b: RELION 2D deformation ports — transcription goldens + read-side wiring."""

import numpy as np
import pytest
import torch

from cets_nonrigid.models.relion_deform import Deformation2D, deformation_field, fourier_frequencies

RNG = np.random.default_rng(20260910)
IMG = (96.0, 84.0)


# --- literal numpy transcriptions (independent of the torch ports) ------------


def _linear_shift_ref(pl, coeffs, image_size):
    """Linear2DDeformationModel::computeShift — verbatim."""
    axx, axy, ayy = coeffs
    r = pl - 0.5 * np.asarray(image_size)
    return np.array([axx * r[0], axy * r[0] + ayy * r[1]])


def _fourier_shift_ref(pl, coeffs, grid, image_size):
    """Fourier2DDeformationModel ctor + computeShift — verbatim."""
    gx, gy = grid
    freqs = []
    for y in range(gy):
        if y < gy // 2:
            for x in range(1 if y == 0 else 0, gx // 2 + 1):
                freqs.append((x * np.pi / image_size[0], y * np.pi / image_size[1]))
        else:
            for x in range(1, gx // 2 + 1):
                freqs.append((x * np.pi / image_size[0], (y - gy) * np.pi / image_size[1]))
    n = len(freqs)
    p = np.asarray(coeffs).reshape(2, n, 2)
    out = np.zeros(2)
    for dim in range(2):
        for i, k in enumerate(freqs):
            t = k[0] * pl[0] + k[1] * pl[1]
            out[dim] += p[dim, i, 0] * np.cos(t) + p[dim, i, 1] * np.sin(t)
    return out


def _spline_shift_ref(pl, coeffs, grid, image_size):
    """Spline2DDeformationModel::projectPoint + computeShift — verbatim."""
    gx, gy = grid
    spacing = (image_size[0] / (gx - 1), image_size[1] / (gy - 1))
    eps = 1e-10
    g = [pl[0] / spacing[0], pl[1] / spacing[1]]
    lim = (gx - 1 - eps, gy - 1 - eps)
    for d in range(2):
        g[d] = min(max(g[d], 0.0), lim[d])
    cell = (int(g[0]), int(g[1]))
    x = g[0] - cell[0]
    y = g[1] - cell[1]
    x2, x3 = x * x, x * x * x
    y2, y3 = y * y, y * y * y
    vx = np.array([1 - 3 * x2 + 2 * x3, 3 * x2 - 2 * x3, x - 2 * x2 + x3, -x2 + x3])
    vy = np.array([1 - 3 * y2 + 2 * y3, 3 * y2 - 2 * y3, y - 2 * y2 + y3, -y2 + y3])
    data = np.asarray(coeffs).reshape(2, gy, gx, 4)  # value, slope_x, slope_y, twist
    out = np.zeros(2)
    for dim in range(2):
        d00 = data[dim, cell[1], cell[0]]
        d01 = data[dim, cell[1] + 1, cell[0]]
        d10 = data[dim, cell[1], cell[0] + 1]
        d11 = data[dim, cell[1] + 1, cell[0] + 1]
        f = np.array(
            [
                [d00[0], d01[0], d00[2], d01[2]],
                [d10[0], d11[0], d10[2], d11[2]],
                [d00[1], d01[1], d00[3], d01[3]],
                [d10[1], d11[1], d10[3], d11[3]],
            ]
        )
        out[dim] = vx @ (f @ vy)
    return out


def _pts(n=40):
    p = RNG.uniform(-0.1, 1.1, (n, 2))  # incl. out-of-range (spline clamps)
    return torch.tensor(p * np.asarray(IMG))


def test_linear_transcription():
    coeffs = RNG.uniform(-0.01, 0.01, 3)
    d = Deformation2D("linear", (1, 1), IMG, torch.tensor(coeffs))
    pts = _pts()
    got = d.shift(pts).numpy()
    for i, pl in enumerate(pts.numpy()):
        np.testing.assert_allclose(got[i], _linear_shift_ref(pl, coeffs, IMG), atol=1e-12)


def test_fourier_transcription():
    grid = (4, 4)
    n = fourier_frequencies(grid, IMG).shape[0]
    coeffs = RNG.uniform(-0.5, 0.5, 4 * n)
    d = Deformation2D("Fourier", grid, IMG, torch.tensor(coeffs))
    pts = _pts()
    got = d.shift(pts).numpy()
    for i, pl in enumerate(pts.numpy()):
        np.testing.assert_allclose(got[i], _fourier_shift_ref(pl, coeffs, grid, IMG), atol=1e-12)


def test_spline_transcription():
    grid = (3, 4)
    coeffs = RNG.uniform(-1.5, 1.5, 8 * grid[0] * grid[1])
    d = Deformation2D("spline", grid, IMG, torch.tensor(coeffs))
    pts = _pts()
    got = d.shift(pts).numpy()
    for i, pl in enumerate(pts.numpy()):
        np.testing.assert_allclose(got[i], _spline_shift_ref(pl, coeffs, grid, IMG), atol=1e-12)


def test_coefficient_count_validation():
    with pytest.raises(ValueError, match="exactly 3"):
        Deformation2D("linear", (1, 1), IMG, torch.zeros(4))
    with pytest.raises(ValueError, match="exactly 96"):  # 8*3*4
        Deformation2D("spline", (3, 4), IMG, torch.zeros(90))
    n = fourier_frequencies((4, 4), IMG).shape[0]
    with pytest.raises(ValueError, match=f"exactly {4 * n}"):
        Deformation2D("Fourier", (4, 4), IMG, torch.zeros(4 * n + 1))
    with pytest.raises(ValueError, match="unknown"):
        Deformation2D("cubic", (1, 1), IMG, torch.zeros(3))


def test_particle_model_applies_deformation():
    from cets_nonrigid.models.relion_ts import RelionParticleSetModel, RelionTomogramModel

    t, pix = 4, 2.0
    model = RelionTomogramModel(
        xtilt_deg=torch.zeros(t, dtype=torch.float64),
        ytilt_deg=torch.linspace(-30, 30, t, dtype=torch.float64),
        zrot_deg=torch.full((t,), 85.0, dtype=torch.float64),
        xshift_a=torch.zeros(t, dtype=torch.float64),
        yshift_a=torch.zeros(t, dtype=torch.float64),
        tomo_dims_px=(48, 48, 24),
        image_dims_px=(96, 84),
        pixel_size_a=pix,
    )
    coeffs = [torch.tensor(RNG.uniform(-0.01, 0.01, 3)) for _ in range(t)]
    field = deformation_field("linear", (1, 1), (96, 84), coeffs)
    pos = torch.tensor(RNG.uniform(0.2, 0.8, (5, 3))) * model.volume_dims_a.to(torch.float64)

    plain = RelionParticleSetModel(model, pos)
    deformed = RelionParticleSetModel(model, pos, deformation=field)
    xy0, _ = plain.project_volume(pos)
    xy1, _ = deformed.project_volume(pos)
    # deformation acts additively in px AFTER the rigid projection
    for f in range(t):
        d = Deformation2D("linear", (1, 1), (96, 84), coeffs[f])
        expected = d.shift(xy0[f].to(torch.float64) / pix) * pix
        torch.testing.assert_close(xy1[f] - xy0[f], expected, atol=1e-8, rtol=0)


def test_r2w_consumes_deformation(tmp_path):
    """A bundle whose tomogram star declares a linear deformation: the r2w fit
    must absorb it (the fitted models with/without the columns differ by the
    deformation field at the particles)."""
    import test_r4_w2r as w2r_helpers

    from cets_nonrigid.convert_relion import relion_to_warp, warp_to_relion
    from cets_nonrigid.io.warp_xml import load_warp_tiltseries
    from cets_nonrigid.models.warp_ts import WarpTiltSeriesModel

    xml = w2r_helpers._write_synthetic_xml(tmp_path / "src.xml")
    stack = w2r_helpers._dummy_stack(tmp_path / "stack.mrc")
    vol_a = torch.tensor(w2r_helpers.VOL_A, dtype=torch.float64)
    pos = torch.tensor(RNG.uniform(0.08, 0.92, (120, 3))) * vol_a
    names = [f"TS_D/{i + 1}" for i in range(pos.shape[0])]
    r = warp_to_relion(
        xml,
        tmp_path / "bundle",
        pixel_size_a=w2r_helpers.PIX,
        tomo_name="TS_D",
        positions_eff_a=pos,
        particle_names=names,
        tilt_stack=stack,
    )

    # append the deformation labels by rewriting the two star files
    import pandas as pd
    import starfile

    gpath = r.tomograms_star
    g = starfile.read(gpath, always_dict=True)["global"]
    if isinstance(g, pd.Series):
        g = g.to_frame().T
    g["rlnTomoDeformationGridSizeX"] = 1
    g["rlnTomoDeformationGridSizeY"] = 1
    g["rlnTomoDeformationType"] = "linear"
    ts_path = tmp_path / "bundle" / "tilt_series" / "TS_D.star"
    ts_df = starfile.read(ts_path, always_dict=True)["TS_D"]
    n_rows = len(ts_df)
    coeffs = [np.array([0.004, -0.003, 0.005]) for _ in range(n_rows)]
    ts_df["rlnTomoDeformationCoefficients"] = ["[" + ",".join(f"{v:.6f}" for v in c) + "]" for c in coeffs]
    gpath.unlink()
    ts_path.unlink()
    starfile.write({"global": g}, gpath)
    starfile.write({"TS_D": ts_df}, ts_path)

    out_def = tmp_path / "with_def.xml"
    relion_to_warp(
        xml,
        out_def,
        tomo_name="TS_D",
        optimisation_set=r.optimisation_set,
        movement_grid=(4, 4),
    )
    out_plain = tmp_path / "plain.xml"
    # same bundle minus the deformation columns = the original write
    g2 = g.drop(columns=["rlnTomoDeformationGridSizeX", "rlnTomoDeformationGridSizeY", "rlnTomoDeformationType"])
    ts2 = ts_df.drop(columns=["rlnTomoDeformationCoefficients"])
    g2path = tmp_path / "plain_tomograms.star"
    ts2dir = tmp_path / "plain_ts"
    ts2dir.mkdir()
    g2["rlnTomoTiltSeriesStarFile"] = str(ts2dir / "TS_D.star")
    starfile.write({"global": g2}, g2path)
    starfile.write({"TS_D": ts2}, ts2dir / "TS_D.star")
    relion_to_warp(
        xml,
        out_plain,
        tomo_name="TS_D",
        tomograms_star=g2path,
        particles_star=r.particles_star,
        motion_star=r.motion_star,
        movement_grid=(4, 4),
    )

    m_def = WarpTiltSeriesModel(load_warp_tiltseries(out_def).ts)
    m_plain = WarpTiltSeriesModel(load_warp_tiltseries(out_plain).ts)
    xy_d, vd = m_def.project_volume(pos)
    xy_p, vp = m_plain.project_volume(pos)
    v = vd & vp
    diff = (xy_d.to(torch.float64) - xy_p.to(torch.float64)).norm(dim=-1)[v]
    # linear deformation of ~0.004 * r over a 192 A image -> few-tenths-A shifts
    assert float(diff.mean()) > 0.05, "deformation did not enter the fit"
