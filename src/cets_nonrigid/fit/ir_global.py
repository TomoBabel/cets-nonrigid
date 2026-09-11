"""IR-driven global fits (store-direct `fit`, plan rev. 5).

Both fitters TRAIN against the store's normative ``source_projected_global``
baseline and EVALUATE on ``heldout_source_projected_global`` — never on user
particles' full projections. The global-fit domain is all finite points on
ACTIVE rows, independent of the image FOV.

Initializer (geometry-invariant, specified): per-tilt affine LSQ of
``points -> global projections`` on CENTERED AND SCALED coordinates (centroid
subtracted, unit-RMS axes) so the rank/condition gates do not depend on volume
size or origin; the recovered affine is UNSCALED before the orthonormal
(polar/SVD) completion, which happens on the physical A->A linear part; angle
initials come from the gimbal-safe ZYX extraction of the completed rotation,
shift initials from the model residual at zero shifts. LBFGS then refines
against the baseline.

Warp gauge (pinned): only ``Angle + LevelAngleY`` enters the projection, so
``LevelAngleY = 0`` is fixed; one shared ``LevelAngleX`` is fitted (initialized
from the recovered per-row rotations); per-row ``Angle``/``AxisAngle`` and XY
offsets are fitted. Output angles are normalized deterministically (axis angle
wrapped to (-180, 180]).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from cets_nonrigid.fit.aretomo_global import GlobalFitResult
from cets_nonrigid.ir.core import IRTiltSeries
from cets_nonrigid.models.aretomo_ts import AretomoTsModel

_F64 = torch.float64

DEFAULT_INIT_MAX_CONDITION = 1e8


class HeldoutGlobalBaselineMissing(ValueError):
    """Raised for schema-0.2 stores: the held-out global baseline was never
    stored — re-dump the IR with this version of cets_nonrigid."""


@dataclass
class IrGlobalDiagnostics:
    init_condition: float  # worst per-tilt condition of the NORMALIZED design
    rms_px_train: float
    rms_px_heldout: float | None  # None when not evaluated
    global_validation_status: str  # "evaluated" | "not_evaluated"


def _active_rows(ir: IRTiltSeries) -> list[int]:
    valid = ir.meta.projection_valid or [True] * ir.n_projections
    return [t for t in range(ir.n_projections) if valid[t]]


def _require_baselines(ir: IRTiltSeries) -> None:
    if ir.heldout_source_projected_global is None:
        raise HeldoutGlobalBaselineMissing(
            "this store carries no held-out global baseline (schema 0.2) - "
            "re-dump the IR with this version of cets_nonrigid to enable "
            "independently validated global fitting"
        )


def _affine_rotations(ir: IRTiltSeries, max_condition: float) -> tuple[torch.Tensor, float]:
    """(T_active, 3, 3) completed proper rotations from normalized per-tilt
    affine LSQ, plus the worst normalized-design condition number."""
    pts = ir.points.to(_F64)
    centroid = pts.mean(dim=0)
    scale = (pts - centroid).pow(2).mean(dim=0).sqrt().clamp_min(1e-9)
    x_norm = torch.cat(
        [(pts - centroid) / scale, torch.ones(pts.shape[0], 1, dtype=_F64)], dim=1
    )  # (N, 4)

    active = _active_rows(ir)
    target = ir.source_projected_global.to(_F64)
    rotations = []
    worst_cond = 0.0
    for t in active:
        finite = torch.isfinite(target[t]).all(dim=-1)
        design = x_norm[finite]
        rank = int(torch.linalg.matrix_rank(design))
        if rank < 4:
            raise ValueError(
                f"global-fit initializer: design rank {rank} < 4 on row {t} - the "
                "sample points are geometrically degenerate (coplanar/collinear?)"
            )
        sv = torch.linalg.svdvals(design)
        cond = float(sv[0] / sv[-1].clamp_min(1e-30))
        worst_cond = max(worst_cond, cond)
        if cond > max_condition:
            raise ValueError(
                f"global-fit initializer: normalized design condition {cond:.3g} exceeds "
                f"{max_condition:.3g} on row {t}"
            )
        sol = torch.linalg.lstsq(design, target[t][finite]).solution  # (4, 2)
        a_norm = sol.T  # (2, 4): [linear_norm | offset]
        linear_phys = a_norm[:, :3] / scale  # unscale -> physical A->A
        # complete the 2x3 projection of a rotation to a proper 3x3 rotation
        r3 = torch.linalg.cross(linear_phys[0], linear_phys[1])
        m = torch.stack([linear_phys[0], linear_phys[1], r3])
        u, _s, vt = torch.linalg.svd(m)
        rot = u @ vt
        if torch.det(rot) < 0:
            u[:, -1] = -u[:, -1]
            rot = u @ vt
        rotations.append(rot)
    return torch.stack(rotations), worst_cond


def _zyx_angles(rot: torch.Tensor) -> tuple[float, float, float]:
    """Gimbal-safe ZYX extraction (R = Rz(z) @ Ry(y) @ Rx(x)), degrees."""
    if float(rot[2, 0]) < 1.0:
        if float(rot[2, 0]) > -1.0:
            x = math.atan2(float(rot[2, 1]), float(rot[2, 2]))
            y = math.asin(-float(rot[2, 0]))
            z = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
        else:
            x, y = 0.0, math.pi / 2
            z = -math.atan2(-float(rot[1, 2]), float(rot[1, 1]))
    else:
        x, y = 0.0, -math.pi / 2
        z = math.atan2(-float(rot[1, 2]), float(rot[1, 1]))
    return math.degrees(x), math.degrees(y), math.degrees(z)


def _wrap_axis(deg: torch.Tensor) -> torch.Tensor:
    """Deterministic normalization: wrap into (-180, 180]."""
    wrapped = torch.remainder(deg + 180.0, 360.0) - 180.0
    return torch.where(wrapped == -180.0, torch.full_like(wrapped, 180.0), wrapped)


def _masked_rms(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """RMS over all finite reference entries (active rows only are passed in)."""
    finite = torch.isfinite(ref).all(dim=-1)
    diff = (pred - ref)[finite]
    return (diff.pow(2).sum(-1).mean() if diff.numel() else torch.tensor(0.0, dtype=_F64)).sqrt()


# ---------------------------------------------------------------------------
# AreTomo target
# ---------------------------------------------------------------------------


def fit_aretomo_globals_from_ir(
    ir: IRTiltSeries,
    *,
    max_iter: int = 200,
    init_max_condition: float = DEFAULT_INIT_MAX_CONDITION,
) -> tuple[GlobalFitResult, IrGlobalDiagnostics]:
    """Fit per-active-row AreTomo globals (ROT, TILT, TX, TY) against the
    store's global baselines. Returns the fit over ACTIVE rows (in IR active
    order) plus diagnostics incl. ``global_validation_status``."""
    _require_baselines(ir)
    meta = ir.meta
    pix = meta.pixel_size_image_a
    active = _active_rows(ir)
    rotations, worst_cond = _affine_rotations(ir, init_max_condition)

    rot0, tilt0 = [], []
    for rot in rotations:
        _x, y, z = _zyx_angles(rot)
        rot0.append(z)
        tilt0.append(y)  # thetaY == TILT for the AreTomo chain (sign pinned by golden)
    rot0 = torch.tensor(rot0, dtype=_F64)
    tilt0 = torch.tensor(tilt0, dtype=_F64)

    vol_a = tuple(float(d) * meta.pixel_size_volume_a for d in meta.volume_dims_px)
    raw_size = tuple(meta.image_dims_px)
    points = ir.points.to(_F64)
    ref = ir.source_projected_global.to(_F64)[active]

    def build(rot_deg, tilt_deg, shifts_px):
        return AretomoTsModel(
            rot_deg=rot_deg, tilt_deg=tilt_deg, shifts_px=shifts_px,
            raw_size_px=raw_size, pixel_size_a=pix, volume_dims_a=vol_a, local=None,
        )

    with torch.no_grad():
        xy0, _ = build(rot0, tilt0, torch.zeros(len(active), 2, dtype=_F64)).project_volume_global(
            points
        )
        finite = torch.isfinite(ref).all(dim=-1, keepdim=True)
        resid = torch.where(finite, ref - xy0, torch.zeros_like(ref))
        shifts0 = resid.sum(dim=1) / finite.sum(dim=1).clamp_min(1) / pix

    d_rot = torch.zeros_like(rot0).requires_grad_(True)
    d_tilt = torch.zeros_like(tilt0).requires_grad_(True)
    d_shift = torch.zeros_like(shifts0).requires_grad_(True)
    mask = torch.isfinite(ref).all(dim=-1).to(_F64)
    n_obs = mask.sum().clamp_min(1.0)

    opt = torch.optim.LBFGS([d_rot, d_tilt, d_shift], line_search_fn="strong_wolfe", max_iter=max_iter)
    n_evals = 0

    def closure():
        nonlocal n_evals
        n_evals += 1
        opt.zero_grad()
        xy, _ = build(rot0 + d_rot, tilt0 + d_tilt, shifts0 + d_shift).project_volume_global(points)
        loss = ((xy - torch.nan_to_num(ref)).pow(2).sum(-1) * mask).sum() / n_obs
        loss.backward()
        return loss

    opt.step(closure)

    with torch.no_grad():
        model = build(rot0 + d_rot, tilt0 + d_tilt, shifts0 + d_shift)
        xy, _ = model.project_volume_global(points)
        rms_train = float(_masked_rms(xy, ref)) / pix

        held = ir.heldout_points.to(_F64)
        held_ref = ir.heldout_source_projected_global.to(_F64)[active]
        if held.shape[0] > 0:
            hxy, _ = model.project_volume_global(held)
            rms_held = float(_masked_rms(hxy, held_ref)) / pix
            status = "evaluated"
            hmask = torch.isfinite(held_ref).all(dim=-1).to(_F64)
            per_tilt = (
                ((hxy - torch.nan_to_num(held_ref)).pow(2).sum(-1) * hmask).sum(dim=1)
                / hmask.sum(dim=1).clamp_min(1.0)
            ).sqrt() / pix
        else:
            rms_held, status = None, "not_evaluated"
            per_tilt = torch.full((len(active),), float("nan"), dtype=_F64)

    fit = GlobalFitResult(
        rot_deg=_wrap_axis((rot0 + d_rot).detach()),
        tilt_deg=(tilt0 + d_tilt).detach(),
        shifts_px=(shifts0 + d_shift).detach(),
        per_tilt_rms_px=per_tilt,
        rms_px_train=rms_train,
        rms_px_heldout=rms_held if rms_held is not None else float("nan"),
        n_iter=n_evals,
        converged=math.isfinite(rms_train),
    )
    return fit, IrGlobalDiagnostics(
        init_condition=worst_cond,
        rms_px_train=rms_train,
        rms_px_heldout=rms_held,
        global_validation_status=status,
    )


# ---------------------------------------------------------------------------
# Warp target
# ---------------------------------------------------------------------------


@dataclass
class WarpGlobalsFit:
    level_angle_x_deg: float  # shared; LevelAngleY pinned to 0
    angle_deg: torch.Tensor  # (T_active,)
    axis_angle_deg: torch.Tensor  # (T_active,)
    axis_offset_x_a: torch.Tensor  # (T_active,)
    axis_offset_y_a: torch.Tensor  # (T_active,)
    n_iter: int


def _warp_global_xy(points_a, vol_a, img_a, level_x, angle, axis, off_x, off_y):
    """Warp GLOBAL projection (docs/conventions.md pinned chain), differentiable:
    R = Rz(axis) @ Ry(-(angle + levelY=0)) @ Rx(levelX); xy = (R p_c)[:2] + I/2 + offset."""
    p_c = points_a.to(_F64) - vol_a / 2  # (N, 3)
    rx = torch.deg2rad(level_x)
    ry = torch.deg2rad(-angle)  # (T,)
    rz = torch.deg2rad(axis)
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)
    # rows of R = Rz @ Ry @ Rx (only the first two are needed)
    r00 = cz * cy
    r01 = cz * sy * sx - sz * cx
    r02 = cz * sy * cx + sz * sx
    r10 = sz * cy
    r11 = sz * sy * sx + cz * cx
    r12 = sz * sy * cx - cz * sx
    x = r00[:, None] * p_c[:, 0] + r01[:, None] * p_c[:, 1] + r02[:, None] * p_c[:, 2]
    y = r10[:, None] * p_c[:, 0] + r11[:, None] * p_c[:, 1] + r12[:, None] * p_c[:, 2]
    xy = torch.stack([x, y], dim=-1)  # (T, N, 2)
    return xy + img_a / 2 + torch.stack([off_x, off_y], dim=-1)[:, None, :]


def fit_warp_globals_from_ir(
    ir: IRTiltSeries,
    *,
    max_iter: int = 200,
    init_max_condition: float = DEFAULT_INIT_MAX_CONDITION,
) -> tuple[WarpGlobalsFit, IrGlobalDiagnostics]:
    """Fit Warp globals (shared LevelAngleX; per-active-row Angle, AxisAngle,
    XY offsets; LevelAngleY = 0 pinned) against the store's global baselines."""
    _require_baselines(ir)
    meta = ir.meta
    pix = meta.pixel_size_image_a
    active = _active_rows(ir)
    rotations, worst_cond = _affine_rotations(ir, init_max_condition)

    axis0, angle0, level0 = [], [], []
    for rot in rotations:
        x, y, z = _zyx_angles(rot)
        axis0.append(z)
        angle0.append(-y)  # thetaY = -(angle + levelY), levelY = 0
        level0.append(x)
    axis0 = torch.tensor(axis0, dtype=_F64)
    angle0 = torch.tensor(angle0, dtype=_F64)
    level0 = torch.tensor(level0, dtype=_F64).mean()

    vol_a = torch.tensor(
        [float(d) * meta.pixel_size_volume_a for d in meta.volume_dims_px], dtype=_F64
    )
    img_a = torch.tensor([d * pix for d in meta.image_dims_px], dtype=_F64)
    points = ir.points.to(_F64)
    ref = ir.source_projected_global.to(_F64)[active]

    with torch.no_grad():
        zero = torch.zeros(len(active), dtype=_F64)
        xy0 = _warp_global_xy(points, vol_a, img_a, level0, angle0, axis0, zero, zero)
        finite = torch.isfinite(ref).all(dim=-1, keepdim=True)
        resid = torch.where(finite, ref - xy0, torch.zeros_like(ref))
        off0 = resid.sum(dim=1) / finite.sum(dim=1).clamp_min(1)

    d_level = torch.zeros((), dtype=_F64).requires_grad_(True)
    d_angle = torch.zeros_like(angle0).requires_grad_(True)
    d_axis = torch.zeros_like(axis0).requires_grad_(True)
    d_off = torch.zeros_like(off0).requires_grad_(True)
    mask = torch.isfinite(ref).all(dim=-1).to(_F64)
    n_obs = mask.sum().clamp_min(1.0)

    opt = torch.optim.LBFGS(
        [d_level, d_angle, d_axis, d_off], line_search_fn="strong_wolfe", max_iter=max_iter
    )
    n_evals = 0

    def closure():
        nonlocal n_evals
        n_evals += 1
        opt.zero_grad()
        xy = _warp_global_xy(
            points, vol_a, img_a, level0 + d_level, angle0 + d_angle, axis0 + d_axis,
            off0[:, 0] + d_off[:, 0], off0[:, 1] + d_off[:, 1],
        )
        loss = ((xy - torch.nan_to_num(ref)).pow(2).sum(-1) * mask).sum() / n_obs
        loss.backward()
        return loss

    opt.step(closure)

    with torch.no_grad():
        level = level0 + d_level
        angle = angle0 + d_angle
        axis = _wrap_axis(axis0 + d_axis)
        off = off0 + d_off
        xy = _warp_global_xy(points, vol_a, img_a, level, angle, axis, off[:, 0], off[:, 1])
        rms_train = float(_masked_rms(xy, ref)) / pix

        held = ir.heldout_points.to(_F64)
        held_ref = ir.heldout_source_projected_global.to(_F64)[active]
        if held.shape[0] > 0:
            hxy = _warp_global_xy(held, vol_a, img_a, level, angle, axis, off[:, 0], off[:, 1])
            rms_held = float(_masked_rms(hxy, held_ref)) / pix
            status = "evaluated"
        else:
            rms_held, status = None, "not_evaluated"

    return (
        WarpGlobalsFit(
            level_angle_x_deg=float(level),
            angle_deg=angle.detach(),
            axis_angle_deg=axis.detach(),
            axis_offset_x_a=off[:, 0].detach(),
            axis_offset_y_a=off[:, 1].detach(),
            n_iter=n_evals,
        ),
        IrGlobalDiagnostics(
            init_condition=worst_cond,
            rms_px_train=rms_train,
            rms_px_heldout=rms_held,
            global_validation_status=status,
        ),
    )
