"""Fit RELION's micrograph motion model (per-frame globals + third-order
polynomial) to a frame-series IR.

Gauge (plan appendix E): RELION's model is spatially CONSTANT at z = 0 and
RELION's own fit re-centers every patch trajectory to its frame-0 value
(motioncorr_runner.cpp:1600-1604) — a pure gauge choice. The fit therefore
targets the RE-GAUGED source field

    T~_f(x) = T_f(x) - T_{f1}(x)      (f1 = first frame with observations)

which preserves every inter-frame difference exactly; the DISCARDED STATIC
FIELD's spatial variation (a static warp of the corrected average, not a fit
residual) is reported as rms/p95/max. "Exact recovery" means exact RELATIVE
trajectories — never absolute maps.

Joint linear LSQ of per-frame globals + polynomial coefficients with the six
z-only columns removed (three per output component: X indices 0-2, Y indices
18-20) — they lie in the span of the free per-frame globals and would make
the joint system rank-deficient. Globals are zero-gauged at f1;
rlnMicrographStartFrame = f1 + 1 so the polynomial vanishes there by
construction. Frames without observations become NOT_OBSERVED sentinel rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cets_nonrigid.conventions import (
    COMPATIBILITY_DTYPE,
    FIT_DTYPE,
    RELION_NOT_OBSERVED,
    RELION_POLY_N_COEFFS,
)
from cets_nonrigid.fit.linear import solve_weighted
from cets_nonrigid.io.relion_motion_star import RelionMicrographMotion
from cets_nonrigid.ir.core import IRFrameSeries

_F64 = torch.float64


@dataclass
class RelionMotionFitResult:
    motion: RelionMicrographMotion
    rms_px_train: float  # vs the RE-GAUGED field T~
    rms_px_heldout: float | None
    p95_px_heldout: float | None
    coverage_heldout: float | None
    #: spatial variation of the discarded static field T_{f1} (px) — a gauge
    #: property of the representation, NOT a fit residual
    static_field_rms_px: float
    static_field_p95_px: float
    static_field_max_px: float
    data_rank: int
    data_condition: float
    meta: dict
    heldout_status: str = "evaluated"


def _spatial_terms(pts_px: torch.Tensor, image_size_px) -> torch.Tensor:
    """(N, 5) the non-constant spatial basis {x, x^2, y, y^2, xy} at
    normalized coords (px/width - 0.5)."""
    xn = pts_px[:, 0].to(_F64) / image_size_px[0] - 0.5
    yn = pts_px[:, 1].to(_F64) / image_size_px[1] - 0.5
    return torch.stack([xn, xn * xn, yn, yn * yn, xn * yn], dim=-1)


def fit_relion_motion(
    ir: IRFrameSeries,
    *,
    image_size_px: tuple,
    pixel_size_a: float,
    movie_name: str = "",
    dose_rate: float | None = None,
    pre_exposure: float | None = None,
    voltage_kv: float | None = None,
    eer_upsampling: int | None = None,
    eer_grouping: int | None = None,
) -> RelionMotionFitResult:
    p = float(pixel_size_a)
    f_count = ir.n_projections
    pts_px = ir.points.to(_F64) / p  # (N, 2) corrected coords, unbinned px

    # Correction field T_f(x) = x - raw(x), in px.
    t_field = (ir.points.to(_F64)[None] - ir.source_projected.to(_F64)) / p  # (F, N, 2)
    w = (
        ir.weights.to(_F64)
        * ir.projection_valid.to(_F64)
        * ir.sample_valid.to(_F64)[None, :]
    )
    frame_has_obs = w.sum(dim=1) > 0
    if not frame_has_obs.any():
        raise ValueError("no observations on any frame")
    f1 = int(torch.nonzero(frame_has_obs).flatten()[0])

    # Re-gauge: T~ = T - T_{f1}; discarded static field = T_{f1}'s spatial part.
    t_ref = t_field[f1]  # (N, 2)
    t_tilde = t_field - t_ref[None]
    w_ref = w[f1]
    mean_ref = (t_ref * w_ref[:, None]).sum(dim=0) / w_ref.sum().clamp_min(1e-30)
    static_dev = (t_ref - mean_ref[None]).norm(dim=-1)[w_ref > 0]
    static_rms = float((static_dev.pow(2).mean()).sqrt()) if static_dev.numel() else 0.0
    static_p95 = float(torch.quantile(static_dev, 0.95)) if static_dev.numel() else 0.0
    static_max = float(static_dev.max()) if static_dev.numel() else 0.0

    # --- joint linear system ------------------------------------------------
    used = torch.nonzero(frame_has_obs).flatten().tolist()
    fu = len(used)
    n = pts_px.shape[0]
    terms = _spatial_terms(pts_px, image_size_px)  # (N, 5)
    z = torch.tensor([f - f1 for f in used], dtype=_F64)  # z = frame - f1
    zpow = torch.stack([z, z * z, z * z * z], dim=-1)  # (Fu, 3)

    rows = []
    b_rows = []
    w_rows = []
    for k, f in enumerate(used):
        onehot = torch.zeros(n, fu, dtype=_F64)
        onehot[:, k] = 1.0
        # spatial-term x z-power columns (15): term t, power p -> terms[:, t] * z^p
        poly_cols = (terms[:, :, None] * zpow[k][None, None, :]).reshape(n, 15)
        rows.append(torch.cat([onehot, poly_cols], dim=1))
        b_rows.append(t_tilde[f])
        w_rows.append(w[f])
    a = torch.cat(rows, dim=0)  # (Fu*N, Fu+15)
    b = torch.cat(b_rows, dim=0)  # (Fu*N, 2)
    ww = torch.cat(w_rows, dim=0)
    res = solve_weighted(a, b, ww, penalty=None, lam=0.0)
    sol = res.x  # (Fu+15, 2)

    g_used = sol[:fu]  # (Fu, 2) px
    c15 = sol[fu:]  # (15, 2)
    g_used = g_used - g_used[used.index(f1)][None]  # exact zero gauge at f1

    coeffs = torch.zeros(RELION_POLY_N_COEFFS, dtype=_F64)
    coeffs[3:18] = c15[:, 0]  # X: slots 0-2 (pure z) stay zero
    coeffs[21:36] = c15[:, 1]  # Y: slots 18-20 stay zero
    has_poly = bool(c15.abs().max() > 1e-12)

    globals_px = torch.full((f_count, 2), float(RELION_NOT_OBSERVED), dtype=_F64)
    for k, f in enumerate(used):
        globals_px[f] = g_used[k]

    motion = RelionMicrographMotion(
        image_size_px=(int(image_size_px[0]), int(image_size_px[1])),
        n_frames=f_count,
        global_shifts_px=globals_px,
        poly_coeffs=coeffs if has_poly else None,
        pixel_size_a=p,
        start_frame=f1 + 1,
        movie_name=movie_name,
        dose_rate=dose_rate,
        pre_exposure=pre_exposure,
        voltage_kv=voltage_kv,
        eer_upsampling=eer_upsampling,
        eer_grouping=eer_grouping,
    )
    model = motion.to_model()

    # --- metrics vs the RE-GAUGED field -------------------------------------
    def metrics(points_a, source_projected, valid, weights):
        pxs = points_a.to(_F64) / p
        t_f = (points_a.to(_F64)[None] - source_projected.to(_F64)) / p
        # re-gauge the evaluation targets with THIS point set's own reference row
        t_t = t_f - t_f[f1][None]
        raw_a, model_valid = model.map_image(points_a)
        pred = pxs[None] - raw_a.to(_F64) / p  # model shift field (zero static)
        v = valid & model_valid
        wm = weights.to(_F64) * v.to(_F64)
        d = (pred - t_t).norm(dim=-1)
        nobs = wm.sum().clamp_min(1e-30)
        rms = float(((d.pow(2) * wm).sum() / nobs).sqrt())
        dv = d[v & (weights > 0)]
        p95 = float(torch.quantile(dv, 0.95)) if dv.numel() else float("nan")
        cov = float(v.to(_F64).mean())
        return rms, p95, cov

    rms_tr, _, _ = metrics(ir.points, ir.source_projected, ir.projection_valid, ir.weights)
    if ir.heldout_status == "evaluated":
        rms_ho, p95_ho, cov_ho = metrics(
            ir.heldout_points, ir.heldout_source_projected,
            ir.heldout_projection_valid, ir.heldout_weights,
        )
    else:
        rms_ho = p95_ho = cov_ho = None

    return RelionMotionFitResult(
        motion=motion,
        rms_px_train=rms_tr,
        rms_px_heldout=rms_ho,
        p95_px_heldout=p95_ho,
        coverage_heldout=cov_ho,
        static_field_rms_px=static_rms,
        static_field_p95_px=static_p95,
        static_field_max_px=static_max,
        data_rank=res.data_rank,
        data_condition=res.data_condition,
        heldout_status=ir.heldout_status,
        meta={
            "fit_dtype": FIT_DTYPE,
            "compatibility_dtype": COMPATIBILITY_DTYPE,
            "gauge": "T~_f = T_f - T_f1 (relative trajectories; static field reported)",
            "f1": f1,
        },
    )
