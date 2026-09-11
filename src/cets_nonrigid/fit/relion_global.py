"""To-RELION global alignment: exact closed forms + runtime verification.

Closed-form mapping (structural proof in the plan appendix; rotations match
because R_warp = Rz(axisAngle)*Ry(-(angle+levelY))*Rx(levelX) and RELION's
linear part is Rz(zrot)*Ry(ytilt)*Rx(xtilt)):

  Warp:    xtilt = LevelAngleX, ytilt = -(Angle + LevelAngleY),
           zrot = TiltAxisAngle,
           shift_A = AxisOffset_A + s*(di - [R dc]_xy)
  AreTomo: xtilt = 0, ytilt = TILT, zrot = ROT,
           shift_A = s*(TX, TY) + s*(di - [R dc]_xy)

with the odd-dimension centre deltas dc = V_px/2 - (V_px // 2) (per axis) and
di = I_px/2 - (I_px // 2) — RELION's matrix uses INTEGER-division centers
(tomogram.cpp:44,53) while the canonical chains use float centers; the deltas
are per-tilt constants and vanish for even dims.

Every conversion VERIFIES the closed form at runtime against the torch
``RelionTomogramModel`` on a coarse grid; on failure an LBFGS fallback
refines the parameters, which must meet the SAME tolerance.
``global_exact = (achieved_rms_px <= 1e-3)`` is always computed from the
achieved residual — a relaxed user tolerance merely permits shipping a
non-exact result, it never marks an exact one non-exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cets_nonrigid.ir.sampling import volume_grid
from cets_nonrigid.models.relion_ts import RelionTomogramModel

_F64 = torch.float64

#: The exactness threshold for ``global_exact`` (px), independent of any user
#: tolerance.
GLOBAL_EXACT_RMS_PX = 1e-3


@dataclass
class RelionGlobalResult:
    model: RelionTomogramModel
    xtilt_deg: torch.Tensor  # (T,)
    ytilt_deg: torch.Tensor
    zrot_deg: torch.Tensor
    xshift_a: torch.Tensor
    yshift_a: torch.Tensor
    rms_px: float  # achieved vs the source global model, coarse grid
    global_exact: bool  # rms_px <= GLOBAL_EXACT_RMS_PX (never tolerance-derived)
    used_fallback: bool


def integer_dims_px(dims_a: torch.Tensor, pixel_size_a: float, what: str) -> tuple:
    """Physical dims -> integral pixel dims, validated (the srf==1 assumption
    is enforced here: non-integral dims mean the XML and pixel size disagree)."""
    px = dims_a.to(_F64) / pixel_size_a
    rounded = px.round()
    if (px - rounded).abs().max() > 1e-3:
        raise ValueError(
            f"{what} {dims_a.tolist()} A is not an integer pixel count at "
            f"{pixel_size_a} A/px (got {px.tolist()})"
        )
    return tuple(int(v) for v in rounded)


def _centre_deltas(dims_px: tuple) -> torch.Tensor:
    return torch.tensor([d / 2.0 - d // 2 for d in dims_px], dtype=_F64)


def _shift_corrections(
    rot: torch.Tensor,  # (T, 3, 3) RELION rotations
    tomo_dims_px: tuple,
    image_dims_px: tuple,
    pixel_size_a: float,
) -> torch.Tensor:
    """(T, 2) additive shift corrections in Angstrom (zero for even dims)."""
    dc = _centre_deltas(tomo_dims_px)  # (3,)
    di = _centre_deltas((image_dims_px[0], image_dims_px[1], 0))[:2]  # (2,)
    rdc = torch.einsum("tij,j->ti", rot, dc)[:, :2]
    return (di[None, :] - rdc) * pixel_size_a


def _params_to_model(xt, yt, zr, sx, sy, tomo_dims_px, image_dims_px, pixel_size_a):
    return RelionTomogramModel(
        xtilt_deg=xt, ytilt_deg=yt, zrot_deg=zr, xshift_a=sx, yshift_a=sy,
        tomo_dims_px=tomo_dims_px, image_dims_px=image_dims_px, pixel_size_a=pixel_size_a,
    )


def relion_global_from_warp(
    ts,  # warpylib TiltSeries
    source_model,  # TiltProjectionModel with project_volume_global (Warp)
    *,
    rows: list,  # emitted tilt indices (XML order, darks dropped)
    pixel_size_a: float,  # tilt-series pixel size (explicit, like w2a's --pix)
    tolerance_px: float = GLOBAL_EXACT_RMS_PX,
    max_fallback_iter: int = 200,
) -> RelionGlobalResult:
    """Closed-form Warp -> RELION globals for the emitted rows + verification."""
    pix = float(pixel_size_a)
    tomo_dims_px = integer_dims_px(ts.volume_dimensions_physical, pix, "VolumeDimensionsAngstrom")
    image_dims_px = integer_dims_px(ts.image_dimensions_physical, pix, "ImageDimensionsAngstrom")

    idx = torch.tensor(rows, dtype=torch.long)
    t = len(rows)
    xt = torch.full((t,), float(ts.level_angle_x), dtype=_F64)
    yt = -(ts.angles.to(_F64)[idx] + float(ts.level_angle_y))
    zr = ts.tilt_axis_angles.to(_F64)[idx]
    sx0 = ts.tilt_axis_offset_x.to(_F64)[idx]
    sy0 = ts.tilt_axis_offset_y.to(_F64)[idx]

    probe = _params_to_model(xt, yt, zr, sx0, sy0, tomo_dims_px, image_dims_px, pix)
    corr = _shift_corrections(probe.rotations, tomo_dims_px, image_dims_px, pix)
    sx, sy = sx0 + corr[:, 0], sy0 + corr[:, 1]

    return _verify_or_fallback(
        source_model, rows, xt, yt, zr, sx, sy,
        tomo_dims_px, image_dims_px, pix, tolerance_px, max_fallback_iter,
    )


def relion_global_from_aretomo(
    aretomo_model,  # AretomoTsModel (rows = emission order)
    source_model,  # TiltProjectionModel for verification (usually the same)
    *,
    tolerance_px: float = GLOBAL_EXACT_RMS_PX,
    max_fallback_iter: int = 200,
) -> RelionGlobalResult:
    """Closed-form AreTomo -> RELION globals (the inverse of RELION's own
    importer, align_tiltseries_runner.cpp:709-883) + verification."""
    pix = aretomo_model.pixel_size_a
    tomo_dims_px = integer_dims_px(aretomo_model.volume_dims_a, pix, "volume dims")
    image_dims_px = (int(aretomo_model.raw_size_px[0]), int(aretomo_model.raw_size_px[1]))

    t = aretomo_model.n_projections
    xt = torch.zeros(t, dtype=_F64)
    yt = aretomo_model.tilt_deg.to(_F64)
    zr = aretomo_model.rot_deg.to(_F64)
    sx0 = aretomo_model.shifts_px.to(_F64)[:, 0] * pix
    sy0 = aretomo_model.shifts_px.to(_F64)[:, 1] * pix

    probe = _params_to_model(xt, yt, zr, sx0, sy0, tomo_dims_px, image_dims_px, pix)
    corr = _shift_corrections(probe.rotations, tomo_dims_px, image_dims_px, pix)
    sx, sy = sx0 + corr[:, 0], sy0 + corr[:, 1]

    return _verify_or_fallback(
        source_model, list(range(t)), xt, yt, zr, sx, sy,
        tomo_dims_px, image_dims_px, pix, tolerance_px, max_fallback_iter,
    )


def _verify_or_fallback(
    source_model, rows, xt, yt, zr, sx, sy,
    tomo_dims_px, image_dims_px, pix, tolerance_px, max_fallback_iter,
):
    active = torch.zeros(source_model.n_projections, dtype=torch.bool)
    active[torch.tensor(rows, dtype=torch.long)] = True

    def expanded_rms(model) -> float:
        # compare on the emitted rows against the source's full tilt list
        vol_a = model.volume_dims_a
        pts = volume_grid(vol_a.to(_F64) * 0.8, (4, 4, 3)) + vol_a.to(_F64) * 0.1
        src_xy, _ = source_model.project_volume_global(pts)
        rel_xy, _ = model.project_volume(pts)
        diff = src_xy.to(_F64)[active] - rel_xy.to(_F64)
        return float(diff.pow(2).mean().sqrt()) / pix

    model = _params_to_model(xt, yt, zr, sx, sy, tomo_dims_px, image_dims_px, pix)
    rms = expanded_rms(model)
    used_fallback = False

    if rms > GLOBAL_EXACT_RMS_PX:
        used_fallback = True
        d = {
            "xt": torch.zeros_like(xt, requires_grad=True),
            "yt": torch.zeros_like(yt, requires_grad=True),
            "zr": torch.zeros_like(zr, requires_grad=True),
            "sx": torch.zeros_like(sx, requires_grad=True),
            "sy": torch.zeros_like(sy, requires_grad=True),
        }
        vol_a = model.volume_dims_a
        pts = volume_grid(vol_a.to(_F64) * 0.8, (4, 4, 3)) + vol_a.to(_F64) * 0.1
        src_xy, _ = source_model.project_volume_global(pts)
        target = src_xy.to(_F64)[active]
        opt = torch.optim.LBFGS(list(d.values()), max_iter=max_fallback_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            m = _params_to_model(
                xt + d["xt"], yt + d["yt"], zr + d["zr"], sx + d["sx"], sy + d["sy"],
                tomo_dims_px, image_dims_px, pix,
            )
            rel_xy, _ = m.project_volume(pts)
            loss = (rel_xy - target).pow(2).mean()
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            xt, yt, zr = xt + d["xt"], yt + d["yt"], zr + d["zr"]
            sx, sy = sx + d["sx"], sy + d["sy"]
        model = _params_to_model(
            xt.detach(), yt.detach(), zr.detach(), sx.detach(), sy.detach(),
            tomo_dims_px, image_dims_px, pix,
        )
        rms = expanded_rms(model)

    if rms > tolerance_px:
        raise RuntimeError(
            f"RELION global residual {rms:.4g} px exceeds --global-tol-px {tolerance_px} "
            "even after the LBFGS fallback — trajectories must not silently absorb a "
            "wrong global; conversion aborted"
        )

    return RelionGlobalResult(
        model=model,
        xtilt_deg=xt.detach(),
        ytilt_deg=yt.detach(),
        zrot_deg=zr.detach(),
        xshift_a=sx.detach(),
        yshift_a=sy.detach(),
        rms_px=rms,
        global_exact=rms <= GLOBAL_EXACT_RMS_PX,
        used_fallback=used_fallback,
    )
