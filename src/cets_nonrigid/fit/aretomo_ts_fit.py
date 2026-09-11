"""w2a local fit: per-(tilt, patch) .aln shifts reproducing a Warp model.

Given fitted AreTomo globals (frozen) and the IR of a Warp source, choose a
patch grid, derive each patch's 3D center (Cz via AreTomo's own
``mCalcPatchZ`` closed form applied to IR-derived residuals), compute the
per-tilt Coord columns exactly (rigid projection), and solve the per-tilt
linear system of the Gaussian-IDW field for the ShiftX/Y columns.

The IDW field S(q) = sum(w_p s_p) / sum(w_p) is linear in the shifts given
the evaluation points, so each tilt is one weighted LSQ. The target residual
is computed against the TARGET model's own global baseline (never the
source's). Everything runs in float64; reported residuals go through the
float32-compatible stable evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from cets_nonrigid import frames
from cets_nonrigid.conventions import COMPATIBILITY_DTYPE, FIT_DTYPE
from cets_nonrigid.fit.linear import solve_weighted
from cets_nonrigid.ir.core import IRTiltSeries
from cets_nonrigid.models.aretomo_ts import AreTomoLocalField, AretomoTsModel


@dataclass
class AretomoLocalFitResult:
    model: AretomoTsModel  # globals + fitted local field (rows = .aln rows)
    patch_centers_fit: torch.Tensor  # (P, 3) Cx, Cy, Cz in the fit frame
    rms_px_train: float
    # None / empty when heldout_status == "not_evaluated" (branch on it first).
    rms_px_heldout: float | None
    p95_px_heldout: float | None
    max_px_heldout: float | None
    per_tilt_rms_px_heldout: torch.Tensor | None  # (T_aln,)
    z_stratified_rms_px: dict[str, float]  # held-out, by volume-z tercile ({} if not evaluated)
    coverage_heldout: float | None
    meta: dict
    heldout_status: str = "evaluated"
    min_data_rank: int = 0  # data-only (pre-regularization), per-tilt minimum
    max_data_condition: float = float("inf")
    max_rank_deficit: int = 0  # max over solved tilts of (n_active_patches - data_rank)


def _patch_centers_xy(raw_size_px: torch.Tensor, patch_grid: tuple[int, int]) -> torch.Tensor:
    """Regular (nx, ny) patch centers, centered px (AreTomo-style bin centers)."""
    nx, ny = patch_grid
    cx = (torch.arange(nx, dtype=torch.float64) + 0.5) / nx - 0.5
    cy = (torch.arange(ny, dtype=torch.float64) + 0.5) / ny - 0.5
    gx, gy = torch.meshgrid(cx * float(raw_size_px[0]), cy * float(raw_size_px[1]), indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # (P, 2)


def _solve_shifts(
    model: AretomoTsModel,
    field_coord: torch.Tensor,  # (T, P, 2) Coord columns
    good: torch.Tensor,  # (T, P)
    q: torch.Tensor,  # (T, N, 2) coord-frame positions of IR points
    b: torch.Tensor,  # (T, N, 2) residual targets (fit-frame px)
    w: torch.Tensor,  # (T, N)
) -> tuple:
    """Per-tilt weighted LSQ for the IDW shifts (kernel: exp(-100 r^2), good
    patches only, normalized weights). Also returns the data-only solver
    diagnostics (min rank / max condition over solved tilts) — gates use them
    (the per-tilt LinearSolveResult was previously discarded)."""
    t_count, p_count = good.shape
    nx = float(model.raw_size_px[0])
    ny = float(model.raw_size_px[1])
    shifts = torch.zeros(t_count, p_count, 2, dtype=torch.float64)
    min_data_rank = None
    max_data_cond = 0.0
    max_deficit = 0

    for t in range(t_count):
        sel = w[t] > 0
        if int(sel.sum()) == 0:
            continue  # inactive tilt: not_evaluated, never a rank-zero failure
        gsel = good[t] >= 0.9
        if int(gsel.sum()) == 0:
            continue
        dq = q[t, sel][:, None, :] - field_coord[t, None, gsel, :]  # (B, Pg, 2)
        e = -100.0 * ((dq[..., 0] / nx) ** 2 + (dq[..., 1] / ny) ** 2)
        wk = torch.exp(e - e.max(dim=1, keepdim=True).values)
        a = wk / wk.sum(dim=1, keepdim=True)  # (B, Pg), rows sum to 1
        res = solve_weighted(a, b[t, sel], w[t, sel], penalty=None, lam=0.0)
        sol = res.x if res.x.ndim == 2 else res.x[:, None]
        shifts[t, gsel] = sol
        min_data_rank = res.data_rank if min_data_rank is None else min(min_data_rank, res.data_rank)
        max_data_cond = max(max_data_cond, res.data_condition)
        max_deficit = max(max_deficit, int(gsel.sum()) - res.data_rank)
    return shifts, (min_data_rank if min_data_rank is not None else 0), max_data_cond, max_deficit


def fit_aretomo_locals(
    ir: IRTiltSeries,
    model_global: AretomoTsModel,  # fitted globals, rows = .aln rows
    row_to_warp: list[int],
    *,
    patch_grid: tuple[int, int] = (5, 5),
    patch_z: Literal["lsq", "zero"] = "lsq",
    good_margin: float = 0.25,
) -> AretomoLocalFitResult:
    t_count = model_global.n_projections
    perm = torch.tensor(row_to_warp)

    # IR arrays in .aln-row order.
    points = ir.points
    target = frames.canonical_image_to_aretomo(
        ir.source_projected.to(torch.float64)[perm],
        model_global.raw_size_px,
        model_global.pixel_size_a,
    )  # (T, N, 2) fit-frame px
    w = (
        ir.weights.to(torch.float64)
        * ir.projection_valid.to(torch.float64)
        * ir.sample_valid.to(torch.float64)[None, :]
    )[perm]

    points_fit = frames.canonical_volume_to_fit(
        points, model_global.volume_dims_a, model_global.pixel_size_a
    )
    q = model_global.project_coord_frame(points_fit)  # (T, N, 2)
    b = target - q - model_global.shifts_px.to(torch.float64)[:, None, :]

    # --- patch 3D centers -------------------------------------------------
    centers_xy = _patch_centers_xy(model_global.raw_size_px, patch_grid)  # (P, 2)
    p_count = centers_xy.shape[0]
    cz = torch.zeros(p_count, dtype=torch.float64)

    def coord_columns(czs: torch.Tensor) -> torch.Tensor:
        pts = torch.cat([centers_xy, czs[:, None]], dim=1)  # (P, 3) fit frame
        return model_global.project_coord_frame(pts)  # (T, P, 2)

    good = torch.ones(t_count, p_count, dtype=torch.float64)

    if patch_z == "lsq":
        # Preliminary Cz=0 solve; then AreTomo's mCalcPatchZ closed form on
        # the field evaluated at each patch's Coord column.
        coord0 = coord_columns(cz)
        s0, _, _, _ = _solve_shifts(model_global, coord0, good, q, b, w)
        field0 = AreTomoLocalField(coord0, s0, good, model_global.raw_size_px)
        meas = coord0 + field0.evaluate(coord0, mode="stable")  # measured - T
        theta = torch.deg2rad(model_global.tilt_deg.to(torch.float64))[:, None]
        rho = torch.deg2rad(model_global.rot_deg.to(torch.float64))[:, None]
        xp = meas[..., 0] * torch.cos(rho) + meas[..., 1] * torch.sin(rho)  # (T, P)
        sin_t, cos_t = torch.sin(theta), torch.cos(theta)
        num = centers_xy[None, :, 0] * (sin_t * cos_t) - xp * sin_t
        cz = num.sum(dim=0) / (sin_t.pow(2).sum() / 1.0).clamp_min(1e-12)

    coord = coord_columns(cz)

    # Good gate: patch Coord too far outside the image contributes nothing.
    img = model_global.raw_size_px.to(torch.float64)
    half = img / 2
    outside_x = (coord[..., 0].abs() - half[0]).clamp_min(0) / img[0]
    outside_y = (coord[..., 1].abs() - half[1]).clamp_min(0) / img[1]
    good = torch.where(
        (outside_x > good_margin) | (outside_y > good_margin),
        torch.zeros_like(good),
        good,
    )

    shifts, min_data_rank, max_data_cond, max_deficit = _solve_shifts(model_global, coord, good, q, b, w)

    field = AreTomoLocalField(coord, shifts, good, model_global.raw_size_px)
    fitted = AretomoTsModel(
        rot_deg=model_global.rot_deg,
        tilt_deg=model_global.tilt_deg,
        shifts_px=model_global.shifts_px,
        raw_size_px=tuple(model_global.raw_size_px.tolist()),
        pixel_size_a=model_global.pixel_size_a,
        volume_dims_a=tuple(model_global.volume_dims_a.tolist()),
        local=field,
        idw_mode="stable",
    )

    # --- residual metrics (compatibility evaluation) ----------------------
    pix = model_global.pixel_size_a

    def _metrics(pts, ref_canonical, valid_ref, weights):
        xy, valid = fitted.project_volume(pts)
        v = valid & valid_ref[perm]
        d = (xy.to(torch.float64) - ref_canonical.to(torch.float64)[perm]).norm(dim=-1)
        ww = weights.to(torch.float64)[perm] * v.to(torch.float64)
        n = ww.sum().clamp_min(1e-30)
        rms = float(((d.pow(2) * ww).sum() / n).sqrt()) / pix
        dv = d[v & (weights[perm] > 0)] / pix
        p95 = float(torch.quantile(dv, 0.95)) if dv.numel() else float("nan")
        mx = float(dv.max()) if dv.numel() else float("nan")
        per_tilt = (((d.pow(2) * ww).sum(dim=1) / ww.sum(dim=1).clamp_min(1e-30)).sqrt()) / pix
        cov = float(v.to(torch.float64).mean())
        return rms, p95, mx, per_tilt, cov, d, ww, v

    rms_tr, _, _, _, _, _, _, _ = _metrics(
        ir.points, ir.source_projected, ir.projection_valid, ir.weights
    )
    strata: dict[str, float] = {}
    if ir.heldout_status == "evaluated":
        rms_ho, p95_ho, max_ho, per_tilt_ho, cov_ho, d_ho, w_ho, _ = _metrics(
            ir.heldout_points,
            ir.heldout_source_projected,
            ir.heldout_projection_valid,
            ir.heldout_weights,
        )

        # Z-stratified held-out residuals (the .aln model is z-independent at
        # apply time; Warp volume warp is not — this is where that loss shows).
        z = ir.heldout_points[:, 2]
        q1, q2 = torch.quantile(z, torch.tensor([1 / 3, 2 / 3], dtype=z.dtype))
        for name, mask in (
            ("z_low", z <= q1),
            ("z_mid", (z > q1) & (z <= q2)),
            ("z_high", z > q2),
        ):
            ws = w_ho[:, mask]
            ds = d_ho[:, mask] / pix
            n = ws.sum().clamp_min(1e-30)
            strata[name] = float(((ds.pow(2) * ws).sum() / n).sqrt())
    else:
        rms_ho = p95_ho = max_ho = cov_ho = None
        per_tilt_ho = None

    return AretomoLocalFitResult(
        model=fitted,
        patch_centers_fit=torch.cat([centers_xy, cz[:, None]], dim=1),
        rms_px_train=rms_tr,
        rms_px_heldout=rms_ho,
        p95_px_heldout=p95_ho,
        max_px_heldout=max_ho,
        per_tilt_rms_px_heldout=per_tilt_ho,
        z_stratified_rms_px=strata,
        coverage_heldout=cov_ho,
        heldout_status=ir.heldout_status,
        min_data_rank=min_data_rank,
        max_data_condition=max_data_cond,
        max_rank_deficit=max_deficit,
        meta={
            "fit_dtype": FIT_DTYPE,
            "compatibility_dtype": COMPATIBILITY_DTYPE,
            "patch_grid": list(patch_grid),
            "patch_z": patch_z,
            "good_margin": good_margin,
        },
    )
