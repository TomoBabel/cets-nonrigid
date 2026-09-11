"""M6: MotionCor model vs a literal transcription of mGCorrect3D."""

import numpy as np
import pytest
import torch

from cets_nonrigid.models.aretomo_motion import AretomoMotionModel

RNG = np.random.default_rng(606)


def _model(f=6, p=9, idw_mode="stable"):
    size = (512, 448)
    centers = np.stack(
        np.meshgrid(
            (np.arange(3) + 0.5) * size[0] / 3,
            (np.arange(3) + 0.5) * size[1] / 3,
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 2)
    return AretomoMotionModel(
        global_shifts_px=torch.tensor(RNG.uniform(-15, 15, (f, 2))),
        patch_centers_px=torch.tensor(centers),
        patch_shifts_px=torch.tensor(RNG.uniform(-3, 3, (f, p, 2))),
        patch_valid=torch.tensor(RNG.uniform(0, 1, (f, p)) > 0.2),
        frame_size_px=size,
        pixel_size_a=0.77,
        idw_mode=idw_mode,
    )


def ref_local(x, y, centers, shifts, valid, nx, ny):
    """mGCorrect3D local field, verbatim loop (one frame, one point)."""
    sx = sy = sw = 0.0
    count = 0
    for p in range(centers.shape[0]):
        if not valid[p]:
            continue
        dx = (x - centers[p, 0]) / nx
        dy = (y - centers[p, 1]) / ny
        r = np.sqrt(dx * dx + dy * dy)
        if r > 0.5:  # HARD CUTOFF (unlike the tilt-series kernel)
            continue
        w = np.exp(-100.0 * r)  # LINEAR r (unlike the tilt-series kernel)
        sx += shifts[p, 0] * w
        sy += shifts[p, 1] * w
        sw += w
        count += 1
    if count > 0 and sw > 0:
        return sx / sw, sy / sw
    return 0.0, 0.0


def test_local_field_matches_reference():
    m = _model()
    pts = torch.tensor(RNG.uniform(-30, 540, (14, 2)))
    out = m.local_field_px(pts)
    nx, ny = m.frame_size_px.tolist()
    for f in range(m.n_projections):
        for n in range(pts.shape[0]):
            sx, sy = ref_local(
                pts[n, 0].item(), pts[n, 1].item(),
                m.patch_centers_px.numpy(), m.patch_shifts_px[f].numpy(),
                m.patch_valid[f].numpy(), nx, ny,
            )
            assert out[f, n, 0].item() == pytest.approx(sx, abs=1e-9)
            assert out[f, n, 1].item() == pytest.approx(sy, abs=1e-9)


def test_cutoff_is_hard():
    # A point > 0.5 normalized units from the ONLY valid patch: zero field
    # (the tilt-series kernel would extrapolate the constant instead).
    m = AretomoMotionModel(
        global_shifts_px=torch.zeros(1, 2),
        patch_centers_px=torch.tensor([[0.0, 0.0]]),
        patch_shifts_px=torch.tensor([[[5.0, -5.0]]]),
        patch_valid=torch.ones(1, 1, dtype=torch.bool),
        frame_size_px=(100, 100),
        pixel_size_a=1.0,
    )
    near = m.local_field_px(torch.tensor([[30.0, 0.0]]))  # r = 0.3
    far = m.local_field_px(torch.tensor([[60.0, 0.0]]))  # r = 0.6 > cutoff
    assert near[0, 0, 0].item() == pytest.approx(5.0)
    assert (far == 0).all()


def test_map_composition_and_sign():
    """raw = x - S_local(x) - glob: corrections are SUBTRACTED."""
    m = _model()
    pts_a = torch.tensor(RNG.uniform(50, 350, (8, 2))) * m.pixel_size_a
    raw, _ = m.map_image(pts_a)
    glob_only, _ = m.map_image_global(pts_a)

    xy_px = pts_a.to(torch.float64) / m.pixel_size_a
    local = m.local_field_px(xy_px).to(torch.float64)
    expected = (
        xy_px[None] - m.global_shifts_px.to(torch.float64)[:, None] - local
    ) * m.pixel_size_a
    torch.testing.assert_close(raw, expected)

    expected_glob = (xy_px[None] - m.global_shifts_px.to(torch.float64)[:, None]) * m.pixel_size_a
    torch.testing.assert_close(glob_only, expected_glob)


def test_compat_matches_stable_in_range():
    m = _model()
    pts = torch.tensor(RNG.uniform(100, 400, (10, 2)))
    a = m.local_field_px(pts, mode="stable")
    b = m.local_field_px(pts, mode="compat")
    torch.testing.assert_close(a.to(torch.float32), b, rtol=1e-4, atol=1e-4)
