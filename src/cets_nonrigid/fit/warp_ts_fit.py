"""Fit Warp GridVolumeWarpX/Y/Z and GridMovementX/Y to an IR.

Two fits, in this order (``fit_warp_locals``):

1. **Volume warp** (``fit_warp_volume_warp``, opt-in): the quadrilinear
   ``LinearGrid4D`` triple is fitted LINEARLY to the IR's per-projection 3D
   displacement ``source_displacement_3d`` (schema 0.4) — 3D data, so there is
   no projection-direction null space. Rows are every (active row, sample-valid
   point) pair regardless of the image FOV (the field lives on the volume);
   coordinates are ``(p/V, dose_norm_t)`` with the TARGET's dose exactly as
   warpylib computes it in float32. Temporal slices no active row supports
   (1-D weight < ``TEMPORAL_SUPPORT_TAU``) are tied to the nearest supported
   slice by an extension operator BEFORE solving; any remaining data-rank
   deficiency is rejected (no "penalty-determined" parameters: curvature
   penalties have their own null spaces). The reduced system comes from a
   streaming QR (never a Gram matrix) and goes through ``solve_weighted`` with
   the original observation count.
2. **Movement grids** (``fit_warp_movement``): unchanged exact per-tilt linear
   least squares, but the sampling positions AND the residual baseline are the
   PREMOVEMENT positions (volume warp on, movement off — ``TiltSeries.cs:471``),
   which is what Warp evaluates. With a zero volume warp this is bitwise what
   the global baseline gave before.

Metrics are kept apart (see the plan): the *displacement-fit beam residual*
(target frame, = ``target.ctf_depth(p, d_fit) - target.ctf_depth(p, d_src)``)
says how well the 3D field is reproduced along the beam; the *CTF-depth
deviation* (``target.ctf_depth(p, d_fit) - source_ctf_depth_a``, each tool's
own convention) says what the target's CTF would do relative to the source's,
and is reported only when the IR carries the source's depths.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field

import torch
from warpylib import CubicGrid, LinearGrid4D, TiltSeries

from cets_nonrigid.conventions import COMPATIBILITY_DTYPE, FIT_DTYPE
from cets_nonrigid.fit.coverage import (
    DEFAULT_MAX_CONDITION,
    DEFAULT_MIN_NODE_SUPPORT,
    DEFAULT_WARN_NODE_SUPPORT,
    node_support_3d,
)
from cets_nonrigid.fit.linear import (
    TEMPORAL_SUPPORT_TAU,
    StreamingQR,
    bspline2d_design_matrix,
    extension_operator,
    lineargrid4d_basis,
    second_difference_penalty_2d,
    second_difference_penalty_4d,
    solve_weighted,
    temporal_support,
    temporal_weights,
)
from cets_nonrigid.ir.core import IRTiltSeries
from cets_nonrigid.models.warp_ts import WarpTiltSeriesModel, dose_range

_F64 = torch.float64
_F32 = torch.float32

#: Rows x columns of one streaming-QR block (float64): ~64 MiB.
_QR_BLOCK_ELEMS = 8 * 1024 * 1024


@dataclass
class VolumeWarpFitResult:
    grid: tuple[int, int, int, int]
    n_params: int  # K = X*Y*Z*L
    n_params_supported: int  # K_S after the temporal extension
    supported_slices: list[bool]  # (L,)
    n_unsupported_slices: int
    discarded_temporal_weight_max: float  # largest 1-D weight of a dropped slice (~1e-7 for float32 leakage)
    discarded_temporal_weight_sum: float
    slice_occupancy: list[int]  # active rows with 1-D weight >= tau, per slice
    dose_collisions: int  # active rows sharing a target dose with another active row
    n_rows: int  # (active row, sample-valid point) pairs
    data_rank: int
    data_condition: float
    regularization_norm: float
    node_support: float  # spatial node support fraction (node_support_3d)
    # 3D displacement-fit residuals (Angstrom), target frame:
    rms_a_train: float
    rms_a_train_inplane: float
    rms_a_train_beam: float
    rms_a_heldout: float | None
    rms_a_heldout_inplane: float | None
    rms_a_heldout_beam: float | None
    p95_a_heldout: float | None
    max_a_heldout: float | None
    per_tilt_rms_a_heldout: torch.Tensor | None  # (T,)
    # source-to-target CTF-depth deviation (Angstrom; x1e-4 = micrometres): None = not available
    ctf_depth_deviation_rms_a_train: float | None
    ctf_depth_deviation_rms_a_heldout: float | None
    ctf_depth_deviation_max_a_heldout: float | None
    max_vw_first_plane_a: float  # max |VW| on the (w=0, z=0) node plane M holds fixed
    heldout_status: str
    lam: float
    meta: dict = field(default_factory=dict)


@dataclass
class WarpTsFitResult:
    ts: TiltSeries  # fitted copy (movement grids set; volume warp as fitted or as given)
    rms_a_train: float
    # Held-out metrics are None (and the per-tilt tensor absent) when
    # heldout_status == "not_evaluated" — consumers must branch on the status
    # BEFORE touching them; a conversion may succeed without held-out data,
    # but nothing may report success derived from missing metrics.
    rms_a_heldout: float | None
    p95_a_heldout: float | None
    max_a_heldout: float | None
    per_tilt_rms_a_heldout: torch.Tensor | None  # (T,)
    coverage_heldout: float | None
    min_rank: int
    max_condition: float
    regularization_norm: float
    meta: dict
    heldout_status: str = "evaluated"
    min_data_rank: int = 0  # data-only (pre-regularization) — gates use these
    max_data_condition: float = float("inf")
    volume_warp: VolumeWarpFitResult | None = None


def movement_grid_dims(patch_grid: tuple[int, int]) -> tuple[int, int]:
    """Default movement dims per axis: Gk = min(8, max(4, patch_grid_k))."""
    return (min(8, max(4, patch_grid[0])), min(8, max(4, patch_grid[1])))


# ---------------------------------------------------------------------------
# volume warp
# ---------------------------------------------------------------------------


def _active_rows(ir: IRTiltSeries, ts: TiltSeries) -> torch.Tensor:
    rows = torch.tensor(ir.meta.projection_valid or [True] * ir.n_projections, dtype=torch.bool)
    return rows & ts.use_tilt.to(torch.bool)


def fit_warp_volume_warp(
    ir: IRTiltSeries,
    ts_target: TiltSeries,
    *,
    grid: tuple[int, int, int, int],
    lam: float = 1e-3,
    max_condition: float = DEFAULT_MAX_CONDITION,
    min_node_support: float = DEFAULT_MIN_NODE_SUPPORT,
    warn_node_support: float = DEFAULT_WARN_NODE_SUPPORT,
) -> tuple[TiltSeries, VolumeWarpFitResult]:
    """Fit ``GridVolumeWarpX/Y/Z`` of a copy of ``ts_target`` to
    ``ir.source_displacement_3d``. Returns (fitted copy, result)."""
    if ir.meta.displacement_3d != "present" or ir.source_displacement_3d is None:
        raise ValueError(
            f"the IR carries no 3D displacement (displacement_3d={ir.meta.displacement_3d!r}); "
            "a volume warp can only be fitted to an IR whose source has a 3D deformation model"
        )
    ir.validate_optional_arrays()
    ts = copy.deepcopy(ts_target)
    t_count = ts.n_tilts
    if t_count != ir.n_projections:
        raise ValueError(f"target has {t_count} tilts, IR has {ir.n_projections}")
    if dose_range(ts) <= 0:
        raise ValueError(
            "the target series carries the same dose on every tilt: Warp normalizes the volume-warp "
            "dose axis by 1/(MaxDose-MinDose) (TiltSeries.cs:411) and would evaluate NaN; refusing to "
            "fit a volume warp (provide per-tilt doses)"
        )
    dims = tuple(int(g) for g in grid)
    if len(dims) != 4 or any(g < 1 for g in dims):
        raise ValueError(f"volume-warp grid must be four positive integers WxHxDxL, got {grid}")
    x, y, z, n_l = dims
    k = x * y * z * n_l

    active = _active_rows(ir, ts)
    if int(active.sum()) == 0:
        raise ValueError("no active rows")
    pts_mask = ir.sample_valid.to(torch.bool)
    pts = ir.points[pts_mask].to(_F64)  # (N, 3)
    d_src = ir.source_displacement_3d[:, pts_mask].to(_F64)  # (T, N, 3)
    n_pts = pts.shape[0]
    if n_pts == 0:
        raise ValueError("no sample-valid points")

    model_target = WarpTiltSeriesModel(ts)
    vol = ts.volume_dimensions_physical.to(_F32)
    spatial = (pts.to(_F32) / vol).to(_F64)  # float32 exactly as warpylib, then float64
    dose = model_target._dose_coords().to(_F64)  # (T,) target dose coordinates
    rows_idx = torch.nonzero(active).reshape(-1)

    # temporal numerical support on the 1-D weights of the ACTIVE rows
    supported, discarded = temporal_support(n_l, dose[rows_idx], TEMPORAL_SUPPORT_TAU)
    w1 = temporal_weights(n_l, dose[rows_idx]).abs()
    occupancy = [int((w1[:, j] >= TEMPORAL_SUPPORT_TAU).sum()) for j in range(n_l)]
    e = extension_operator(dims, supported)  # (K, K_S)
    k_s = e.shape[1]
    dose_active = dose[rows_idx]
    collisions = int(sum(int((dose_active == v).sum()) for v in dose_active.unique() if int((dose_active == v).sum()) > 1))

    # streaming QR over tilt blocks of [A_t E | d_t]
    qr = StreamingQR(k_s, 3)
    rows_per_block = max(1, _QR_BLOCK_ELEMS // (k * max(n_pts, 1)))
    for start in range(0, rows_idx.numel(), rows_per_block):
        block_rows = rows_idx[start : start + rows_per_block]
        nb = block_rows.numel()
        coords4 = torch.cat(
            [
                spatial[None, :, :].expand(nb, n_pts, 3),
                dose[block_rows][:, None, None].expand(nb, n_pts, 1),
            ],
            dim=-1,
        ).reshape(-1, 4)
        a = lineargrid4d_basis(dims, coords4) @ e  # (nb*N, K_S)
        b = d_src[block_rows].reshape(-1, 3)
        qr.add_block(a, b)
    n_rows = qr.n_rows

    # data rank / condition on the SUPPORTED parameters (exact singular values)
    svals = qr.singular_values()
    tol = svals.max() * max(n_rows, k_s) * torch.finfo(_F64).eps
    data_rank = int((svals > tol).sum())
    data_cond = float(svals.max() / svals[svals > tol].min()) if data_rank > 0 else float("inf")
    if data_rank < k_s:
        raise RuntimeError(_rank_failure_message(qr.r_data, dims, supported, e, k_s, data_rank))
    if data_cond > max_condition:
        raise RuntimeError(
            f"volume-warp data condition {data_cond:.3g} exceeds {max_condition:.3g}; use a coarser grid"
        )
    support = node_support_3d(pts, ts.volume_dimensions_physical, (x, y, z))
    if support < min_node_support:
        raise RuntimeError(
            f"volume-warp spatial node support {support:.2f} below {min_node_support}; use a coarser spatial grid"
        )
    if support < warn_node_support:
        warnings.warn(f"volume-warp spatial node support {support:.2f} below {warn_node_support}", stacklevel=2)

    penalty = second_difference_penalty_4d(dims) @ e
    res = solve_weighted(qr.r_data, qr.qtb, torch.ones(k_s, dtype=_F64), penalty=penalty, lam=lam, n_rows=n_rows)
    v_s = res.x if res.x.ndim == 2 else res.x[:, None]  # (K_S, 3)
    v = e @ v_s  # (K, 3) full grid, extension applied BEFORE evaluation
    ts.grid_volume_warp_x = LinearGrid4D(dims, v[:, 0].to(_F32))
    ts.grid_volume_warp_y = LinearGrid4D(dims, v[:, 1].to(_F32))
    ts.grid_volume_warp_z = LinearGrid4D(dims, v[:, 2].to(_F32))

    # ---- compatibility (float32) evaluation --------------------------------
    fitted = WarpTiltSeriesModel(ts)

    def _resid(points, d_ref, ctf_ref):
        d_fit = fitted.displace_volume(points).to(_F64)
        r = d_fit - d_ref.to(_F64)  # (T, N, 3)
        beam = (fitted.ctf_depth(points, d_fit.to(_F32)).to(_F64) - fitted.ctf_depth(points, d_ref.to(_F32)).to(_F64))
        rn = r.norm(dim=-1)
        inplane = (rn.pow(2) - beam.pow(2)).clamp_min(0).sqrt()
        rows = active[:, None].expand_as(rn)
        n = rows.to(_F64).sum().clamp_min(1)

        def rms(t):
            return float(((t.pow(2) * rows.to(_F64)).sum() / n).sqrt())

        sel = rn[rows]
        per_tilt = ((rn.pow(2) * rows.to(_F64)).sum(dim=1) / rows.to(_F64).sum(dim=1).clamp_min(1e-30)).sqrt()
        dev_rms = dev_max = None
        if ctf_ref is not None:
            dev = fitted.ctf_depth(points, d_fit.to(_F32)).to(_F64) - ctf_ref.to(_F64)
            dev_rms = rms(dev)
            dev_max = float(dev[rows].abs().max()) if sel.numel() else float("nan")
        return (
            rms(rn), rms(inplane), rms(beam),
            float(torch.quantile(sel, 0.95)) if sel.numel() else float("nan"),
            float(sel.max()) if sel.numel() else float("nan"),
            per_tilt, dev_rms, dev_max,
        )

    tr = _resid(pts, d_src, None if ir.source_ctf_depth_a is None else ir.source_ctf_depth_a[:, pts_mask])
    if ir.heldout_status == "evaluated":
        ho = _resid(
            ir.heldout_points.to(_F64), ir.heldout_source_displacement_3d.to(_F64), ir.heldout_source_ctf_depth_a
        )
    else:
        ho = None

    plane = x * y  # nodes (x, y, z=0, w=0) are the first X*Y flat entries
    first_plane = max(float(g.values[:plane].abs().max()) for g in (ts.grid_volume_warp_x, ts.grid_volume_warp_y, ts.grid_volume_warp_z))

    result = VolumeWarpFitResult(
        grid=dims,
        n_params=k,
        n_params_supported=k_s,
        supported_slices=[bool(s) for s in supported],
        n_unsupported_slices=int((~supported).sum()),
        discarded_temporal_weight_max=float(discarded.max()) if discarded.numel() else 0.0,
        discarded_temporal_weight_sum=float(discarded.sum()) if discarded.numel() else 0.0,
        slice_occupancy=occupancy,
        dose_collisions=collisions,
        n_rows=n_rows,
        data_rank=data_rank,
        data_condition=data_cond,
        regularization_norm=res.regularization_norm,
        node_support=support,
        rms_a_train=tr[0],
        rms_a_train_inplane=tr[1],
        rms_a_train_beam=tr[2],
        rms_a_heldout=ho[0] if ho else None,
        rms_a_heldout_inplane=ho[1] if ho else None,
        rms_a_heldout_beam=ho[2] if ho else None,
        p95_a_heldout=ho[3] if ho else None,
        max_a_heldout=ho[4] if ho else None,
        per_tilt_rms_a_heldout=ho[5] if ho else None,
        ctf_depth_deviation_rms_a_train=tr[6],
        ctf_depth_deviation_rms_a_heldout=ho[6] if ho else None,
        ctf_depth_deviation_max_a_heldout=ho[7] if ho else None,
        max_vw_first_plane_a=first_plane,
        heldout_status=ir.heldout_status,
        lam=lam,
        meta={
            "fit_dtype": FIT_DTYPE,
            "compatibility_dtype": COMPATIBILITY_DTYPE,
            "volume_warp_grid": list(dims),
            "temporal_support_tau": TEMPORAL_SUPPORT_TAU,
            "solver": "streaming-qr + gelsd (reduced system, n_rows carried)",
            "ctf_depth_deviation": "available" if ir.source_ctf_depth_a is not None else "not available",
        },
    )
    return ts, result


def _rank_failure_message(r_data, dims, supported, e, k_s, data_rank) -> str:
    """Name zero (unsupported) columns separately from dependent combinations."""
    colnorm = r_data.norm(dim=0)
    zero_cols = torch.nonzero(colnorm <= colnorm.max() * 1e-12).reshape(-1)
    # dependent directions: right singular vectors of the (near-)null space
    _u, _s, vh = torch.linalg.svd(r_data, full_matrices=True)
    null = vh[data_rank:]  # (K_S - rank, K_S)
    involved = torch.nonzero(null.abs().max(dim=0).values > 1e-6).reshape(-1)
    dependent = [int(c) for c in involved if int(c) not in set(zero_cols.tolist())]
    x, y, z, n_l = dims
    nxyz = x * y * z
    sup_idx = torch.nonzero(supported).reshape(-1).tolist()

    def name(col):
        w = sup_idx[col // nxyz]
        rem = col % nxyz
        iz, rem = divmod(rem, x * y)
        iy, ix = divmod(rem, x)
        return f"(x={ix},y={iy},z={iz},w={w})"

    return (
        f"volume warp {x}x{y}x{z}x{n_l} is not identifiable from the IR: data rank {data_rank} of "
        f"{k_s} supported parameters; unsupported (zero) nodes: "
        f"{[name(int(c)) for c in zero_cols.tolist()] or 'none'}; dependent nodes: "
        f"{[name(c) for c in dependent] or 'none'}. Use a coarser spatial grid or fewer dose nodes (L)."
    )


# ---------------------------------------------------------------------------
# movement grids
# ---------------------------------------------------------------------------


def fit_warp_movement(
    ir: IRTiltSeries,
    ts_target: TiltSeries,
    *,
    movement_grid: tuple[int, int] = (5, 5),
    lam: float = 1e-3,
) -> WarpTsFitResult:
    """Fit GridMovementX/Y of ``ts_target`` (globals and volume warp already set) to the IR.

    ts_target is not mutated; a fitted copy is returned. The sampling positions
    and the residual baseline are the PREMOVEMENT positions (volume warp on,
    movement off), i.e. exactly where Warp samples GridMovement; with a zero
    volume warp they coincide with the global baseline.
    """
    ts = copy.deepcopy(ts_target)
    t_count = ts.n_tilts
    if t_count != ir.n_projections:
        raise ValueError(f"target has {t_count} tilts, IR has {ir.n_projections}")

    gx, gy = movement_grid
    model_stub = WarpTiltSeriesModel(ts)

    points = ir.points
    q_pre, pre_valid = model_stub.project_volume_premovement(points)  # (T, N, 2) float32
    q_pre64 = q_pre.to(_F64)

    img = model_stub.image_dims_a.to(_F64)
    target = ir.source_projected.to(_F64)

    # Movement is SUBTRACTED in the forward chain: full = premovement - M  =>
    # M = premovement - target.
    b_all = q_pre64 - target  # (T, N, 2)
    w_all = (
        ir.weights.to(_F64)
        * ir.projection_valid.to(_F64)
        * pre_valid.to(_F64)
        * ir.sample_valid.to(_F64)[None, :]
    )

    penalty = second_difference_penalty_2d((gx, gy))
    n_nodes = gx * gy
    values_x = torch.zeros(t_count * n_nodes, dtype=_F64)
    values_y = torch.zeros(t_count * n_nodes, dtype=_F64)

    min_rank = n_nodes
    max_cond = 0.0
    reg_norm = 0.0
    min_data_rank = n_nodes
    max_data_cond = 0.0
    for t in range(t_count):
        sel = w_all[t] > 0
        if int(sel.sum()) == 0:
            continue  # dark/disabled tilt: slice stays zero
        coords = q_pre64[t, sel] / img  # normalized sampling positions
        a = bspline2d_design_matrix((gx, gy), coords)
        res = solve_weighted(a, b_all[t, sel], w_all[t, sel], penalty=penalty, lam=lam)
        sol = res.x if res.x.ndim == 2 else res.x[:, None]
        values_x[t * n_nodes : (t + 1) * n_nodes] = sol[:, 0]
        values_y[t * n_nodes : (t + 1) * n_nodes] = sol[:, 1]
        min_rank = min(min_rank, res.rank)
        max_cond = max(max_cond, res.condition_estimate)
        min_data_rank = min(min_data_rank, res.data_rank)
        max_data_cond = max(max_data_cond, res.data_condition)
        reg_norm = max(reg_norm, res.regularization_norm)

    ts.grid_movement_x = CubicGrid((gx, gy, t_count), values_x.to(_F32))
    ts.grid_movement_y = CubicGrid((gx, gy, t_count), values_y.to(_F32))

    # ---- compatibility (float32) evaluation --------------------------------
    fitted = WarpTiltSeriesModel(ts)

    def _metrics(pts, ref, valid_ref, weights):
        xy, valid = fitted.project_volume(pts)
        v = valid & valid_ref
        d = (xy.to(_F64) - ref.to(_F64)).norm(dim=-1)
        w = weights.to(_F64) * v.to(_F64)
        n = w.sum().clamp_min(1e-30)
        rms = float(((d.pow(2) * w).sum() / n).sqrt())
        dv = d[v & (weights > 0)]
        p95 = float(torch.quantile(dv, 0.95)) if dv.numel() else float("nan")
        mx = float(dv.max()) if dv.numel() else float("nan")
        per_tilt = ((d.pow(2) * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-30)).sqrt()
        coverage = float(v.to(_F64).mean())
        return rms, p95, mx, per_tilt, coverage

    rms_tr, _, _, _, _ = _metrics(ir.points, ir.source_projected, ir.projection_valid, ir.weights)
    if ir.heldout_status == "evaluated":
        rms_ho, p95_ho, max_ho, per_tilt_ho, cov_ho = _metrics(
            ir.heldout_points,
            ir.heldout_source_projected,
            ir.heldout_projection_valid,
            ir.heldout_weights,
        )
    else:
        rms_ho = p95_ho = max_ho = cov_ho = None
        per_tilt_ho = None

    vw_dims = tuple(ts.grid_volume_warp_x.dimensions)
    vw_zero = all(
        g.dimensions == (1, 1, 1, 1) and float(g.values.abs().max()) == 0.0
        for g in (ts.grid_volume_warp_x, ts.grid_volume_warp_y, ts.grid_volume_warp_z)
    )
    return WarpTsFitResult(
        ts=ts,
        rms_a_train=rms_tr,
        rms_a_heldout=rms_ho,
        p95_a_heldout=p95_ho,
        max_a_heldout=max_ho,
        per_tilt_rms_a_heldout=per_tilt_ho,
        coverage_heldout=cov_ho,
        heldout_status=ir.heldout_status,
        min_rank=min_rank,
        max_condition=max_cond,
        min_data_rank=min_data_rank,
        max_data_condition=max_data_cond,
        regularization_norm=reg_norm,
        meta={
            "fit_dtype": FIT_DTYPE,
            "compatibility_dtype": COMPATIBILITY_DTYPE,
            "movement_grid": [gx, gy],
            "lambda": lam,
            "movement_baseline": "premovement",
            "volume_warp": "zero" if vw_zero else "as given " + "x".join(str(d) for d in vw_dims),
        },
    )


# ---------------------------------------------------------------------------
# wrapper
# ---------------------------------------------------------------------------


def fit_warp_locals(
    ir: IRTiltSeries,
    ts_target: TiltSeries,
    *,
    movement_grid: tuple[int, int] = (5, 5),
    lam: float = 1e-3,
    volume_warp_grid: tuple[int, int, int, int] | None = None,
    volume_warp_lam: float | None = None,
    max_condition: float = DEFAULT_MAX_CONDITION,
    min_node_support: float = DEFAULT_MIN_NODE_SUPPORT,
    warn_node_support: float = DEFAULT_WARN_NODE_SUPPORT,
) -> WarpTsFitResult:
    """Volume warp (opt-in, iff ``volume_warp_grid``) then movement grids.

    Without ``volume_warp_grid`` this is exactly ``fit_warp_movement`` on
    ``ts_target`` as given (callers reset its volume warp to zero) — today's
    behaviour, byte for byte.
    """
    vw = None
    ts_after = ts_target
    if volume_warp_grid is not None:
        ts_after, vw = fit_warp_volume_warp(
            ir, ts_target, grid=volume_warp_grid,
            lam=lam if volume_warp_lam is None else volume_warp_lam,
            max_condition=max_condition, min_node_support=min_node_support, warn_node_support=warn_node_support,
        )
    fit = fit_warp_movement(ir, ts_after, movement_grid=movement_grid, lam=lam)
    fit.volume_warp = vw
    if vw is not None:
        fit.meta["volume_warp"] = "fitted " + "x".join(str(d) for d in vw.grid)
        fit.meta["volume_warp_fit"] = vw.meta
    elif ir.meta.displacement_3d in ("present", "zero_at_samples", "none", "not_recorded"):
        fit.meta["volume_warp"] = fit.meta["volume_warp"] + f" (IR displacement_3d={ir.meta.displacement_3d})"
    return fit


# ---------------------------------------------------------------------------
# helpers for callers (CLI parsing, gates, store-out, reports)
# ---------------------------------------------------------------------------


def resolve_volume_warp_grid(spec, n_rows: int) -> tuple[int, int, int, int] | None:
    """``(W, H, D, L)`` with ``L`` possibly ``None`` (the CLI's ``T``) -> concrete
    grid; ``None`` stays ``None`` (no volume-warp fit)."""
    if spec is None:
        return None
    w, h, d, n_l = spec
    return (int(w), int(h), int(d), int(n_rows) if n_l is None else int(n_l))


def volume_warp_max_norm_a(ts: TiltSeries) -> float:
    """max over grid nodes of |(VW_x, VW_y, VW_z)| — a conservative bound on the
    displacement anywhere inside the volume (in-domain quadrilinear interpolation
    is a convex combination of node values). 0 for degenerate grids."""
    gx, gy, gz = ts.grid_volume_warp_x, ts.grid_volume_warp_y, ts.grid_volume_warp_z
    if not (gx.dimensions == gy.dimensions == gz.dimensions):
        # different dims: bound by the sum of per-grid maxima
        return float(sum(float(g.values.abs().max()) for g in (gx, gy, gz)))
    v = torch.stack([gx.values.to(_F64), gy.values.to(_F64), gz.values.to(_F64)], dim=-1)
    return float(v.norm(dim=-1).max()) if v.numel() else 0.0


def volume_warp_fit_attrs(vw: VolumeWarpFitResult | None) -> dict:
    """JSON-serializable summary for store ``fit`` attrs / reports."""
    if vw is None:
        return {"volume_warp_fitted": False}
    return {
        "volume_warp_fitted": True,
        "volume_warp_grid": list(vw.grid),
        "volume_warp_n_params": vw.n_params,
        "volume_warp_n_params_supported": vw.n_params_supported,
        "volume_warp_n_unsupported_slices": vw.n_unsupported_slices,
        "volume_warp_discarded_temporal_weight_max": vw.discarded_temporal_weight_max,
        "volume_warp_dose_collisions": vw.dose_collisions,
        "volume_warp_data_rank": vw.data_rank,
        "volume_warp_data_condition": vw.data_condition,
        "volume_warp_node_support": vw.node_support,
        "volume_warp_rms_a_train": vw.rms_a_train,
        "volume_warp_rms_a_heldout": vw.rms_a_heldout,
        "volume_warp_rms_a_heldout_inplane": vw.rms_a_heldout_inplane,
        "volume_warp_rms_a_heldout_beam": vw.rms_a_heldout_beam,
        "volume_warp_p95_a_heldout": vw.p95_a_heldout,
        "volume_warp_max_a_heldout": vw.max_a_heldout,
        "ctf_depth_deviation_rms_a_heldout": vw.ctf_depth_deviation_rms_a_heldout,
        "ctf_depth_deviation_max_a_heldout": vw.ctf_depth_deviation_max_a_heldout,
        "ctf_depth_deviation_status": vw.meta.get("ctf_depth_deviation"),
        "volume_warp_max_first_plane_a": vw.max_vw_first_plane_a,
        "volume_warp_lambda": vw.lam,
    }


def volume_warp_fit_arrays(ts: TiltSeries) -> tuple[dict, dict]:
    """Fitted grids as store arrays ``fit/volume_warp_{x,y,z}`` shaped (L, D, H, W)
    with explicit dimension names (rank-4 arrays need them, io/store.py)."""
    arrays, dims = {}, {}
    for name, g in (("x", ts.grid_volume_warp_x), ("y", ts.grid_volume_warp_y), ("z", ts.grid_volume_warp_z)):
        w, h, d, n_l = g.dimensions
        arrays[f"volume_warp_{name}"] = g.values.reshape(n_l, d, h, w).to(_F32)
        dims[f"volume_warp_{name}"] = ["w", "z", "y", "x"]
    return arrays, dims
