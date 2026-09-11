"""M1: AreTomo3 tilt-series model vs independent literal transcriptions.

References transcribed directly from:
- AreTomo3/AreTomo/PatchAlign/CFitPatchShifts.cpp (Coord columns)
- AreTomo3/AreTomo/Correct/GCorrPatchShift.cu::mGCalcLocalShift (IDW field)
- zarr-particle-tools/core/forwardprojection.py (independent global model,
  recon-frame Z convention — checks the fit-frame Z sign flip)
"""

import numpy as np
import pytest
import torch

from cets_nonrigid import frames
from cets_nonrigid.conventions import ARETOMO_FIT_Z_SIGN
from cets_nonrigid.models.aretomo_ts import AreTomoLocalField, AretomoTsModel

RNG = np.random.default_rng(20260828)


def _random_model(t=7, p=9, with_local=True, idw_mode="stable"):
    raw = (1024, 936)
    rot = RNG.uniform(-15, 15, size=t)
    tilt = np.linspace(-60, 60, t) + RNG.uniform(-1, 1, t)
    shifts = RNG.uniform(-30, 30, size=(t, 2))
    local = None
    if with_local:
        coord = RNG.uniform(-400, 400, size=(t, p, 2))
        shift = RNG.uniform(-8, 8, size=(t, p, 2))
        good = (RNG.uniform(0, 1, size=(t, p)) > 0.25).astype(np.float64)
        good[:, 0] = 1.0  # at least one good patch per tilt
        local = AreTomoLocalField(
            coord_xy=torch.tensor(coord),
            shift_xy=torch.tensor(shift),
            good=torch.tensor(good),
            raw_size_px=torch.tensor(raw),
        )
    model = AretomoTsModel(
        rot_deg=torch.tensor(rot),
        tilt_deg=torch.tensor(tilt),
        shifts_px=torch.tensor(shifts),
        raw_size_px=raw,
        pixel_size_a=1.54,
        volume_dims_a=(1024 * 1.54, 936 * 1.54, 300 * 1.54),
        local=local,
        idw_mode=idw_mode,
    )
    return model


# ---------------------------------------------------------------------------
# Literal numpy references
# ---------------------------------------------------------------------------


def ref_coord(cx, cy, cz, rot_deg, tilt_deg):
    """CFitPatchShifts::mCalcPatchLocalShifts, verbatim."""
    theta = np.deg2rad(tilt_deg)
    rho = np.deg2rad(rot_deg)
    fx = cx * np.cos(theta) - cz * np.sin(theta)
    u = fx * np.cos(rho) - cy * np.sin(rho)
    v = fx * np.sin(rho) + cy * np.cos(rho)
    return u, v


def ref_idw(u, v, coord, shift, good, nx, ny, f32=False):
    """mGCalcLocalShift, verbatim loop (single point, single tilt)."""
    cast = np.float32 if f32 else np.float64
    sx = cast(0.0)
    sy = cast(0.0)
    sw = cast(0.0)
    count = 0
    for p in range(coord.shape[0]):
        if good[p] < 0.9:
            continue
        dx = cast((u - coord[p, 0]) / nx)
        dy = cast((v - coord[p, 1]) / ny)
        w = np.exp(cast(-100.0) * (dx * dx + dy * dy), dtype=cast)
        sx += cast(shift[p, 0]) * w
        sy += cast(shift[p, 1]) * w
        sw += w
        count += 1
    if count > 0:
        return float(sx / sw), float(sy / sw)  # may be NaN on underflow
    return 0.0, 0.0


# ---------------------------------------------------------------------------
# Coord columns (machine precision)
# ---------------------------------------------------------------------------


def test_coord_frame_matches_reference():
    model = _random_model(with_local=False)
    pts = torch.tensor(RNG.uniform(-450, 450, size=(20, 3)))
    out = model.project_coord_frame(pts)  # (T, N, 2)

    for t in range(model.n_projections):
        for n in range(pts.shape[0]):
            u, v = ref_coord(
                pts[n, 0].item(), pts[n, 1].item(), pts[n, 2].item(),
                model.rot_deg[t].item(), model.tilt_deg[t].item(),
            )
            assert out[t, n, 0].item() == pytest.approx(u, abs=1e-9)
            assert out[t, n, 1].item() == pytest.approx(v, abs=1e-9)


def test_global_matches_zarr_particle_tools_convention():
    """Independent global reference: recon-frame Z (X cos + Z sin), 4x4-matrix
    style as in zarr-particle-tools forwardprojection.py. Checks that our
    fit-frame Z sign (Cz = -z_centered) composes to the same projection."""
    model = _random_model(with_local=False)
    pts_fit = torch.tensor(RNG.uniform(-300, 300, size=(15, 3)))
    out = model.project_fit_frame(pts_fit, with_local=False)

    for t in range(model.n_projections):
        theta = np.deg2rad(model.tilt_deg[t].item())
        rho = np.deg2rad(model.rot_deg[t].item())
        for n in range(pts_fit.shape[0]):
            x, y = pts_fit[n, 0].item(), pts_fit[n, 1].item()
            z_recon = ARETOMO_FIT_Z_SIGN * pts_fit[n, 2].item()  # fit -> recon Z
            px = x * np.cos(theta) + z_recon * np.sin(theta)
            u = px * np.cos(rho) - y * np.sin(rho) + model.shifts_px[t, 0].item()
            v = px * np.sin(rho) + y * np.cos(rho) + model.shifts_px[t, 1].item()
            assert out[t, n, 0].item() == pytest.approx(u, abs=1e-9)
            assert out[t, n, 1].item() == pytest.approx(v, abs=1e-9)


# ---------------------------------------------------------------------------
# IDW field
# ---------------------------------------------------------------------------


def test_idw_matches_reference_loop():
    model = _random_model(t=5, p=7)
    local = model.local
    uv = torch.tensor(RNG.uniform(-450, 450, size=(5, 12, 2)))
    out = local.evaluate(uv, mode="stable")

    nx, ny = local.raw_size_px.tolist()
    for t in range(5):
        for n in range(12):
            sx, sy = ref_idw(
                uv[t, n, 0].item(), uv[t, n, 1].item(),
                local.coord_xy[t].numpy(), local.shift_xy[t].numpy(),
                local.good[t].numpy(), nx, ny,
            )
            assert out[t, n, 0].item() == pytest.approx(sx, abs=1e-8)
            assert out[t, n, 1].item() == pytest.approx(sy, abs=1e-8)


def test_idw_good_gate():
    # Patches below the 0.9 gate contribute nothing.
    coord = torch.zeros(1, 2, 2)
    coord[0, 1] = torch.tensor([5.0, 5.0])
    shift = torch.tensor([[[1.0, 2.0], [100.0, 100.0]]])
    good = torch.tensor([[1.0, 0.89]])
    field = AreTomoLocalField(coord, shift, good, torch.tensor([100, 100]))
    out = field.evaluate(torch.zeros(1, 1, 2))
    assert out[0, 0, 0].item() == pytest.approx(1.0)
    assert out[0, 0, 1].item() == pytest.approx(2.0)


def test_idw_no_good_patch_is_zero():
    field = AreTomoLocalField(
        torch.zeros(1, 2, 2), torch.ones(1, 2, 2), torch.zeros(1, 2),
        torch.tensor([100, 100]),
    )
    out = field.evaluate(torch.rand(1, 4, 2) * 50, mode="stable")
    assert (out == 0).all()
    out_c = field.evaluate(torch.rand(1, 4, 2) * 50, mode="compat")
    assert (out_c == 0).all()


def test_idw_underflow_compat_vs_stable():
    # A point ~3 normalized units away: exponent -100*9 -> exp underflows in
    # float32. compat reproduces the CUDA NaN (0/0 with iCount > 0); stable
    # stays finite and returns the dominant patch's shift.
    coord = torch.tensor([[[0.0, 0.0]]])  # one good patch at center
    shift = torch.tensor([[[3.0, -4.0]]])
    good = torch.ones(1, 1)
    field = AreTomoLocalField(coord, shift, good, torch.tensor([100, 100]))
    far = torch.tensor([[[300.0, 0.0]]])  # 3.0 normalized

    compat = field.evaluate(far, mode="compat")
    assert torch.isnan(compat).all()

    stable = field.evaluate(far, mode="stable")
    assert stable[0, 0, 0].item() == pytest.approx(3.0)
    assert stable[0, 0, 1].item() == pytest.approx(-4.0)


def test_idw_compat_matches_stable_in_range():
    model = _random_model(t=4, p=6)
    local = model.local
    uv = torch.tensor(RNG.uniform(-200, 200, size=(4, 10, 2)))
    a = local.evaluate(uv, mode="stable")
    b = local.evaluate(uv, mode="compat")
    torch.testing.assert_close(a.to(torch.float32), b, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Full chain, canonical frame, gradients
# ---------------------------------------------------------------------------


def test_full_chain_composition():
    model = _random_model(t=5, p=7)
    pts_fit = torch.tensor(RNG.uniform(-300, 300, size=(8, 3)))
    full = model.project_fit_frame(pts_fit)
    uv0 = model.project_coord_frame(pts_fit)
    local = model.local.evaluate(uv0)
    expected = uv0 + local + model.shifts_px[:, None, :]
    torch.testing.assert_close(full, expected)


def test_canonical_frame_roundtrip():
    vol = torch.tensor([1000.0, 900.0, 400.0], dtype=torch.float64)
    pts = torch.tensor(RNG.uniform(0, 900, size=(11, 3)))
    fit = frames.canonical_volume_to_fit(pts, vol, 1.7)
    back = frames.fit_volume_to_canonical(fit, vol, 1.7)
    torch.testing.assert_close(back, pts)

    raw = torch.tensor([512, 480])
    uv = torch.tensor(RNG.uniform(-200, 200, size=(6, 2)))
    xy = frames.aretomo_image_to_canonical(uv, raw, 1.7)
    back2 = frames.canonical_image_to_aretomo(xy, raw, 1.7)
    torch.testing.assert_close(back2, uv)


def test_project_volume_consistency_and_validity():
    model = _random_model(t=5, p=7)
    pts = torch.tensor(RNG.uniform(0, 900, size=(10, 3)))
    xy, valid = model.project_volume(pts)
    assert xy.shape == (5, 10, 2) and valid.shape == (5, 10)

    # Same thing assembled by hand through the frame transforms.
    fit = frames.canonical_volume_to_fit(pts, model.volume_dims_a, model.pixel_size_a)
    uv = model.project_fit_frame(fit)
    xy2 = frames.aretomo_image_to_canonical(uv, model.raw_size_px, model.pixel_size_a)
    torch.testing.assert_close(xy, xy2)

    # Global-only differs from full wherever locals are nonzero.
    xy_g, _ = model.project_volume_global(pts)
    assert (xy - xy_g).abs().max() > 0.1


def test_gradients_flow():
    model = _random_model(t=4, p=5)
    model.shifts_px.requires_grad_(True)
    model.local.shift_xy.requires_grad_(True)
    pts = torch.tensor(RNG.uniform(0, 900, size=(6, 3)), requires_grad=True)
    xy, _ = model.project_volume(pts)
    xy.sum().backward()
    assert torch.isfinite(model.shifts_px.grad).all()
    assert torch.isfinite(model.local.shift_xy.grad).all()
    assert torch.isfinite(pts.grad).all()
