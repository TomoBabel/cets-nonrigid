"""Frame-series fits (both directions). Linear solves throughout — both
models are linear in their parameters with fixed sampling positions.

m2w (fit Warp movie grids): per-frame global = weighted mean of the total
correction (goes into GridMovement (1,1,F)); GridLocal (Lx,Ly,Lt) fitted to
the residual by LSQ. The zero-mean gauge between global and local grids is
enforced: the local field's per-frame weighted mean is folded into the
global grid after the solve.

w2m (fit .mcaln shifts): per-frame global = weighted spatial mean of the
total correction; per-(patch, frame) shifts from the linear IDW-r system on
the residual; the global column is renormalized so glob[fmRef] = 0 (a pure
gauge shift of the corrected-frame origin, documented in the .mcaln spec).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
from warpylib import CubicGrid
from warpylib.movie import Movie

from cets_nonrigid.conventions import COMPATIBILITY_DTYPE, FIT_DTYPE
from cets_nonrigid.fit.linear import solve_weighted
from cets_nonrigid.ir.core import IRFrameSeries
from cets_nonrigid.models.aretomo_motion import AretomoMotionModel
from cets_nonrigid.models.warp_movie import WarpMovieModel


def _spline1d_design(n_nodes: int, t_norm: torch.Tensor) -> torch.Tensor:
    """(B, n_nodes) interpolating-spline basis along one axis."""
    from warpylib.cubic_grid import InterpolatingBSplineOperator1D

    if n_nodes == 1:
        return torch.ones(t_norm.shape[0], 1, dtype=torch.float64)
    eye = torch.eye(n_nodes, dtype=torch.float64)  # (C, M)
    return InterpolatingBSplineOperator1D()(eye, t_norm.to(torch.float64)[:, None])


def _spline3d_design(
    dims: tuple[int, int, int], coords_norm: torch.Tensor
) -> torch.Tensor:
    """(B, X*Y*Z) basis of a full 3D CubicGrid via the identity-channel trick."""
    from warpylib.cubic_grid import InterpolatingBSplineOperator3D

    gx, gy, gz = dims
    n = gx * gy * gz
    eye = torch.eye(n, dtype=torch.float64).reshape(n, gz, gy, gx)  # (C, Z, Y, X)
    return InterpolatingBSplineOperator3D()(eye, coords_norm.to(torch.float64)[:, [2, 1, 0]])


@dataclass
class MovieFitResult:
    rms_a_train: float
    # None when heldout_status == "not_evaluated" (branch on it first).
    rms_a_heldout: float | None
    p95_a_heldout: float | None
    coverage_heldout: float | None
    meta: dict
    heldout_status: str = field(default="evaluated", kw_only=True)


@dataclass
class WarpMovieFitResult(MovieFitResult):
    movie: Movie
    model: WarpMovieModel


@dataclass
class McAlnFitResult(MovieFitResult):
    model: AretomoMotionModel


def _shift_targets(ir: IRFrameSeries) -> tuple[torch.Tensor, torch.Tensor]:
    """Total subtracted correction (F, N, 2) in A + weights (F, N)."""
    total = ir.points.to(torch.float64)[None] - ir.source_projected.to(torch.float64)
    w = (
        ir.weights.to(torch.float64)
        * ir.projection_valid.to(torch.float64)
        * ir.sample_valid.to(torch.float64)[None, :]
    )
    return total, w


def fit_warp_movie(
    ir: IRFrameSeries,
    template_movie: Movie,
    *,
    n_frames: int,
    image_dims_a: tuple[float, float],
    fraction_frames: float = 1.0,
    local_grid: tuple[int, int, int] = (3, 3, 4),
    lam: float = 1e-3,
) -> WarpMovieFitResult:
    """m2w: fit GridMovement (1,1,F) + GridLocal local_grid to the IR."""
    movie = copy.deepcopy(template_movie)
    f_count = n_frames
    if ir.n_projections != f_count:
        raise ValueError("IR frame count mismatch")

    total, w = _shift_targets(ir)  # (F, N, 2)

    # Temporal sampling coordinate of frame f (see WarpMovieModel).
    step = 1.0 / max(1, f_count - 1)
    t_frac = torch.tensor(
        [f * step * fraction_frames for f in range(f_count)], dtype=torch.float64
    )

    # --- global: (1,1,F) grid. Evaluated at t_frac; per-frame decoupling
    # holds only when fraction_frames == 1, so solve the 1D spline LSQ.
    wsum = w.sum(dim=1).clamp_min(1e-30)
    mean_shift = (total * w[..., None]).sum(dim=1) / wsum[:, None]  # (F, 2)
    a_t = _spline1d_design(f_count, t_frac)  # (F, F)
    res_g = solve_weighted(a_t, mean_shift, wsum / wsum.max())
    g_nodes = res_g.x if res_g.x.ndim == 2 else res_g.x[:, None]  # (F, 2)

    global_at_f = a_t @ g_nodes  # (F, 2)
    resid = total - global_at_f[:, None, :]

    # --- local: (Lx, Ly, Lt) grid over (x/Ix, y/Iy, t_frac) ---------------
    lx, ly, lt = local_grid
    img = torch.tensor(image_dims_a, dtype=torch.float64)
    n_pts = ir.points.shape[0]
    coords = torch.empty(f_count * n_pts, 3, dtype=torch.float64)
    coords[:, 0] = (ir.points[:, 0] / img[0]).repeat(f_count)
    coords[:, 1] = (ir.points[:, 1] / img[1]).repeat(f_count)
    coords[:, 2] = t_frac.repeat_interleave(n_pts)
    a_l = _spline3d_design((lx, ly, lt), coords)  # (F*N, Lx*Ly*Lt)
    res_l = solve_weighted(
        a_l, resid.reshape(-1, 2), w.reshape(-1), lam=lam, penalty=None
    )
    l_nodes = res_l.x if res_l.x.ndim == 2 else res_l.x[:, None]

    # --- zero-mean gauge: fold the local field's per-frame mean into global.
    local_at = (a_l @ l_nodes).reshape(f_count, n_pts, 2)
    local_mean = (local_at * w[..., None]).sum(dim=1) / wsum[:, None]  # (F, 2)
    # Re-solve global on (mean_shift + local_mean)? Equivalent: add to nodes
    # via the same 1D solve.
    # The 1D solve is exact interpolation (square basis, distinct temporal
    # coordinates), so the folded amount reproduces local_mean at the frames.
    res_g2 = solve_weighted(a_t, local_mean, wsum / wsum.max())
    g2 = res_g2.x if res_g2.x.ndim == 2 else res_g2.x[:, None]
    g_nodes = g_nodes + g2

    movie.grid_movement_x = CubicGrid((1, 1, f_count), g_nodes[:, 0].to(torch.float32))
    movie.grid_movement_y = CubicGrid((1, 1, f_count), g_nodes[:, 1].to(torch.float32))
    # Refit the local grid on the residual with the folded per-frame mean
    # removed, so the local field is zero-mean per frame at the samples.
    resid2 = resid - local_mean[:, None, :]
    res_l2 = solve_weighted(a_l, resid2.reshape(-1, 2), w.reshape(-1), lam=lam)
    l2 = res_l2.x if res_l2.x.ndim == 2 else res_l2.x[:, None]
    movie.grid_local_x = CubicGrid((lx, ly, lt), l2[:, 0].to(torch.float32))
    movie.grid_local_y = CubicGrid((lx, ly, lt), l2[:, 1].to(torch.float32))
    movie.pyramid_shift_x = []
    movie.pyramid_shift_y = []

    model = WarpMovieModel(
        movie, n_frames=f_count, image_dims_a=image_dims_a, fraction_frames=fraction_frames
    )
    rms_tr, _, _, _ = _movie_metrics(model, ir, train=True)
    if ir.heldout_status == "evaluated":
        rms_ho, p95, cov, _ = _movie_metrics(model, ir, train=False)
    else:
        rms_ho = p95 = cov = None
    return WarpMovieFitResult(
        movie=movie,
        model=model,
        rms_a_train=rms_tr,
        rms_a_heldout=rms_ho,
        p95_a_heldout=p95,
        coverage_heldout=cov,
        heldout_status=ir.heldout_status,
        meta={
            "fit_dtype": FIT_DTYPE,
            "compatibility_dtype": COMPATIBILITY_DTYPE,
            "local_grid": list(local_grid),
            "lambda": lam,
        },
    )


def fit_mcaln_shifts(
    ir: IRFrameSeries,
    *,
    frame_size_px: tuple[int, int],
    pixel_size_a: float,
    patch_grid: tuple[int, int] = (5, 5),
    fm_ref: int = -1,
) -> McAlnFitResult:
    """w2m: per-frame global + per-(patch, frame) IDW-r shifts from the IR."""
    f_count = ir.n_projections
    if fm_ref < 0:
        fm_ref = f_count // 2

    total_a, w = _shift_targets(ir)
    total_px = total_a / pixel_size_a  # corrections in alignment px

    wsum = w.sum(dim=1).clamp_min(1e-30)
    glob = (total_px * w[..., None]).sum(dim=1) / wsum[:, None]  # (F, 2)
    resid = total_px - glob[:, None, :]

    # Patch centers: AreTomo-style bin centers, corner-origin px.
    nx, ny = patch_grid
    cx = (torch.arange(nx, dtype=torch.float64) + 0.5) / nx * frame_size_px[0]
    cy = (torch.arange(ny, dtype=torch.float64) + 0.5) / ny * frame_size_px[1]
    gx, gy = torch.meshgrid(cx, cy, indexing="ij")
    centers = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # (P, 2)
    p_count = centers.shape[0]

    pts_px = ir.points.to(torch.float64) / pixel_size_a  # (N, 2)
    dx = (pts_px[:, None, 0] - centers[None, :, 0]) / frame_size_px[0]
    dy = (pts_px[:, None, 1] - centers[None, :, 1]) / frame_size_px[1]
    r = torch.sqrt(dx * dx + dy * dy)
    wk = torch.where(r <= 0.5, torch.exp(-100.0 * (r - r.min(dim=1, keepdim=True).values)), torch.zeros_like(r))
    row_ok = wk.sum(dim=1) > 0
    a = torch.zeros_like(wk)
    a[row_ok] = wk[row_ok] / wk[row_ok].sum(dim=1, keepdim=True)

    shifts = torch.zeros(f_count, p_count, 2, dtype=torch.float64)
    for f in range(f_count):
        sel = (w[f] > 0) & row_ok
        if int(sel.sum()) == 0:
            continue
        res = solve_weighted(a[sel], resid[f, sel], w[f, sel])
        sol = res.x if res.x.ndim == 2 else res.x[:, None]
        shifts[f] = sol

    # Gauge: glob[fm_ref] = 0 (shifts the corrected-frame origin by a
    # constant; documented in the .mcaln spec).
    glob = glob - glob[fm_ref : fm_ref + 1]

    model = AretomoMotionModel(
        global_shifts_px=glob,
        patch_centers_px=centers,
        patch_shifts_px=shifts,
        patch_valid=torch.ones(f_count, p_count, dtype=torch.bool),
        frame_size_px=frame_size_px,
        pixel_size_a=pixel_size_a,
    )
    rms_tr, _, _, _ = _movie_metrics(model, ir, train=True, gauge_free=True)
    if ir.heldout_status == "evaluated":
        rms_ho, p95, cov, _ = _movie_metrics(model, ir, train=False, gauge_free=True)
    else:
        rms_ho = p95 = cov = None
    return McAlnFitResult(
        model=model,
        rms_a_train=rms_tr,
        rms_a_heldout=rms_ho,
        p95_a_heldout=p95,
        coverage_heldout=cov,
        heldout_status=ir.heldout_status,
        meta={
            "fit_dtype": FIT_DTYPE,
            "compatibility_dtype": COMPATIBILITY_DTYPE,
            "patch_grid": list(patch_grid),
            "fm_ref": fm_ref,
        },
    )


def _movie_metrics(model, ir: IRFrameSeries, train: bool, gauge_free: bool = False):
    if train:
        pts, ref, valid_ref, weights = (
            ir.points,
            ir.source_projected,
            ir.projection_valid,
            ir.weights,
        )
    else:
        pts, ref, valid_ref, weights = (
            ir.heldout_points,
            ir.heldout_source_projected,
            ir.heldout_projection_valid,
            ir.heldout_weights,
        )
    xy, valid = model.map_image(pts)
    diff = xy.to(torch.float64) - ref.to(torch.float64)
    if gauge_free:
        # The fm_ref renormalization moves everything by a constant per
        # series; residuals are compared modulo that global constant.
        w0 = (weights.to(torch.float64) * (valid & valid_ref).to(torch.float64))
        const = (diff * w0[..., None]).sum(dim=(0, 1)) / w0.sum().clamp_min(1e-30)
        diff = diff - const
    v = valid & valid_ref
    d = diff.norm(dim=-1)
    ww = weights.to(torch.float64) * v.to(torch.float64)
    n = ww.sum().clamp_min(1e-30)
    rms = float(((d.pow(2) * ww).sum() / n).sqrt())
    dv = d[v & (weights > 0)]
    p95 = float(torch.quantile(dv, 0.95)) if dv.numel() else float("nan")
    cov = float(v.to(torch.float64).mean())
    return rms, p95, cov, d
