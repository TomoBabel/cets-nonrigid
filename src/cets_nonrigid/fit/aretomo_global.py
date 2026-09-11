"""w2a global fit: best per-tilt AreTomo3 global parameters for a Warp model.

Warp's global geometry (LevelAngleX, runtime rounding factors) is not exactly
representable by .aln globals, so the conversion FITS (ROT, TILT, TX, TY) per
tilt to the Warp global-only projection of a coarse 3D grid, starting from
the closed-form mapping. LevelAngleY folds exactly into TILT; the LevelAngleX
remainder is minimized here and later baked into the local shifts.

Everything runs in float64 through the differentiable AretomoTsModel.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cets_nonrigid.ir.sampling import heldout_points, volume_grid
from cets_nonrigid.models.aretomo_ts import AretomoTsModel
from cets_nonrigid.models.warp_ts import WarpTiltSeriesModel


@dataclass
class GlobalFitResult:
    rot_deg: torch.Tensor  # (T,) over Warp file order (used tilts fitted; unused hold init)
    tilt_deg: torch.Tensor  # (T,)
    shifts_px: torch.Tensor  # (T, 2)
    per_tilt_rms_px: torch.Tensor  # (T,) held-out
    rms_px_train: float
    rms_px_heldout: float
    n_iter: int
    converged: bool


def closed_form_init(
    warp_model: WarpTiltSeriesModel, pixel_size_a: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closed-form .aln globals from Warp parameters (see docs/conventions.md):
    ROT = +TiltAxisAngle, TILT = -(Angle + LevelAngleY), TX/TY = AxisOffset / px."""
    ts = warp_model.ts
    rot = ts.tilt_axis_angles.to(torch.float64)
    tilt = -(ts.angles.to(torch.float64) + float(ts.level_angle_y))
    shifts = (
        torch.stack([ts.tilt_axis_offset_x, ts.tilt_axis_offset_y], dim=-1).to(torch.float64)
        / pixel_size_a
    )
    return rot, tilt, shifts


def fit_aretomo_globals(
    warp_model: WarpTiltSeriesModel,
    pixel_size_a: float,
    *,
    grid_shape: tuple[int, int, int] = (5, 5, 3),
    max_iter: int = 200,
    heldout_seed: int = 20260828,
) -> GlobalFitResult:
    """Warp wrapper: closed-form init, then the LBFGS core."""
    rot0, tilt0, shifts0 = closed_form_init(warp_model, pixel_size_a)
    return fit_aretomo_globals_from_init(
        warp_model, pixel_size_a, rot0, tilt0, shifts0,
        grid_shape=grid_shape, max_iter=max_iter, heldout_seed=heldout_seed,
    )


def fit_aretomo_globals_from_init(
    source_model,  # any TiltProjectionModel with volume_dims_a/image_dims_a
    pixel_size_a: float,
    rot0: torch.Tensor,
    tilt0: torch.Tensor,
    shifts0: torch.Tensor,
    *,
    grid_shape: tuple[int, int, int] = (5, 5, 3),
    max_iter: int = 200,
    heldout_seed: int = 20260828,
) -> GlobalFitResult:
    """LBFGS refinement of per-tilt AreTomo globals against ANY source model's
    global-only projections (fits on its own coarse grid + Sobol held-out set —
    never on user particles, so particle train/held-out splits stay
    independent)."""
    warp_model = source_model
    vol = warp_model.volume_dims_a.to(torch.float64)
    img_px = (warp_model.image_dims_a.to(torch.float64) / pixel_size_a).round().to(torch.int64)
    raw_size = (int(img_px[0]), int(img_px[1]))

    points = volume_grid(vol, grid_shape)
    held = heldout_points(vol, max(64, points.shape[0]), seed=heldout_seed)

    ref_xy, ref_valid = warp_model.project_volume_global(points)
    ref_xy = ref_xy.to(torch.float64)
    held_xy, held_valid = warp_model.project_volume_global(held)
    held_xy = held_xy.to(torch.float64)

    d_rot = torch.zeros_like(rot0).requires_grad_(True)
    d_tilt = torch.zeros_like(tilt0).requires_grad_(True)
    d_shift = torch.zeros_like(shifts0).requires_grad_(True)

    def build(dr, dt, ds) -> AretomoTsModel:
        return AretomoTsModel(
            rot_deg=rot0 + dr,
            tilt_deg=tilt0 + dt,
            shifts_px=shifts0 + ds,
            raw_size_px=raw_size,
            pixel_size_a=pixel_size_a,
            volume_dims_a=tuple(vol.tolist()),
            local=None,
        )

    mask = ref_valid.to(torch.float64)
    n_obs = mask.sum().clamp_min(1.0)

    opt = torch.optim.LBFGS(
        [d_rot, d_tilt, d_shift], line_search_fn="strong_wolfe", max_iter=max_iter
    )
    n_evals = 0

    def closure():
        nonlocal n_evals
        n_evals += 1
        opt.zero_grad()
        xy, _ = build(d_rot, d_tilt, d_shift).project_volume_global(points)
        loss = ((xy - ref_xy).pow(2).sum(-1) * mask).sum() / n_obs
        loss.backward()
        return loss

    opt.step(closure)

    with torch.no_grad():
        model = build(d_rot, d_tilt, d_shift)

        xy, _ = model.project_volume_global(points)
        train_sq = ((xy - ref_xy).pow(2).sum(-1) * mask).sum() / n_obs
        rms_train_px = float(train_sq.sqrt()) / pixel_size_a

        hxy, _ = model.project_volume_global(held)
        hmask = held_valid.to(torch.float64)
        h_sq_per_tilt = ((hxy - held_xy).pow(2).sum(-1) * hmask).sum(dim=1) / hmask.sum(
            dim=1
        ).clamp_min(1.0)
        per_tilt_rms_px = h_sq_per_tilt.sqrt() / pixel_size_a
        rms_held_px = float(
            (((hxy - held_xy).pow(2).sum(-1) * hmask).sum() / hmask.sum().clamp_min(1.0)).sqrt()
        ) / pixel_size_a

        return GlobalFitResult(
            rot_deg=(rot0 + d_rot).detach(),
            tilt_deg=(tilt0 + d_tilt).detach(),
            shifts_px=(shifts0 + d_shift).detach(),
            per_tilt_rms_px=per_tilt_rms_px,
            rms_px_train=rms_train_px,
            rms_px_heldout=rms_held_px,
            n_iter=n_evals,
            converged=bool(torch.isfinite(torch.tensor(rms_held_px))),
        )
