"""Exact per-particle trajectory lift + gauge (to-RELION).

Unit-clean contract (all Angstrom; plan appendix C, rev. 4 of the
displacement plan):

  d(f,i)      = source 3D displacement of particle i at frame f (0 without one)
  r'_A(f,i)   = source_xy_A(f,i) - relion_global_xy_A(f, p_eff + d)     # 2D remainder
  t_A(f,i)    = d(f,i) + R_f^T (r'_x, r'_y, 0)^T                         # 3D lift
  c_i         = gauge constant (per particle)
  p'_eff_A(i) = p_eff_A(i) + c_i                                        # static position
  motion_A(f,i) = t_A(f,i) - c_i                                        # -> motion.star

R_f is the pure rotation of the emitted RELION matrix (pseudo-inverse =
transpose — enforced by from_matrices/Euler construction), so the in-plane part
of the lift is exact by construction and the 2D observable
``P_f(p' + motion_f) = source_xy_f`` is GAUGE-INDEPENDENT (checked to
``LIFT_EXACT_MAX_PX``).

What RELION does with the result (verified in source):
* extraction applies ``origin + shifts_Ang[f]`` with no re-gauge
  (trajectory.cpp:172) and projects it — the beam-axis component of ``t`` is
  invisible to every RELION consumer;
* the CTF depth is that of the STATIC coordinate ``p'`` at every ``getCtf``
  call site (subtomo.cpp:780-783, ctf_refinement.cpp:591,
  reconstruct_particle.cpp:374, prediction.cpp:200/389,
  local_particle_refinement.cpp:81). RELION never uses trajectory depth for CTF.

Hence the source's per-frame 3D coordinate is PRESERVED in motion.star (for a
later r2w), while RELION's CTF depth depends on the gauge ``c``:

* ``lowest-dose`` (default, RELION's own convention): ``c = t(ref)`` at the
  lowest-cumulative-dose emitted row.
* ``ctf-optimal``: ``c`` minimizes ``sum_f (RELION.ctf_depth_f(p + c) -
  source.ctf_depth_f)^2`` with the ACTUAL emitted-target and source
  ``ctf_depth`` functions (hand, slope, globals included). RELION's depth is
  affine in ``c`` (``B_f = hand*slope*(R_f)_z``); single-axis tomography makes
  ``rank(B) = 2``, so the minimizer is taken within the retained SVD directions
  (``sigma >= 1e-2 sigma_max``) and tied to the lowest-dose gauge
  (``c = c_ref + B_r^+ (y - B c_ref)``); it falls back to lowest-dose with a
  warning when ``|c - c_ref|`` exceeds ``GAUGE_SHIFT_CAP`` times the largest
  depth deviation it corrects. It minimizes only that squared error over the
  emitted rows: RMS improves, the maximum may not.

In BOTH gauges the static position must lie inside the volume (the particle IR
builder rejects it otherwise); a fallback does not repair that — it is an error.

``t_A`` is computed from model positions for ALL frames (finite even outside
the FOV; outside-FOV alone triggers no fallback) — the nearest-chronologically-
valid fallback applies only to genuinely nonfinite source data.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import torch

from cets_nonrigid.models.relion_ts import RelionTomogramModel

_F64 = torch.float64

#: Exactness gate for the lift: max |P(p'+t') - source| over all finite
#: (frame, particle) pairs, in px (expect ~1e-10; float32 sources ~1e-3).
LIFT_EXACT_MAX_PX = 1e-3

#: Retained-direction cut of the CTF-optimal gauge's SVD pseudo-inverse: a
#: +/-0.05 deg x-tilt perturbation of a single-axis series gives sigma_3/sigma_1
#: ~ 9e-4 and an unconstrained solve then moved the gauge by ~700 A.
GAUGE_SVD_RTOL = 1e-2

#: ctf-optimal fallback: a gauge shift larger than this multiple of the largest
#: per-frame depth deviation it corrects is a leaking near-degenerate direction.
#: For single-axis geometry with the SVD cut the ratio is ~1.
GAUGE_SHIFT_CAP = 10.0

GAUGES = ("lowest-dose", "ctf-optimal")


@dataclass
class TrajectoryLiftResult:
    positions_out_a: torch.Tensor  # (P, 3) gauged static positions, canonical A
    motion_a: torch.Tensor  # (T, P, 3) additive A offsets, emitted-row order
    max_residual_px: float  # 2D exactness check (finite pairs), gauge-independent
    ref_row: int  # lowest-dose emitted row (the lowest-dose gauge reference)
    n_fallback: int  # (frame, particle) pairs healed from nonfinite source data
    depth_source: str = "min_norm"  # "displacement" when a source 3D displacement entered the lift
    gauge: str = "lowest-dose"
    n_gauge_fallback: int = 0  # particles where ctf-optimal fell back to lowest-dose
    gauge_rank: int | None = None  # rank of the beam-direction matrix B (2 for single-axis)
    # RELION.ctf_depth(static) - source.ctf_depth(p + d): None when the source depth is unknown
    ctf_depth_deviation_a: torch.Tensor | None = None  # (T, P) Angstrom
    ctf_depth_deviation_rms_a: float | None = None  # over the frames entering the gauge (NOT held-out)
    ctf_depth_deviation_max_a: float | None = None
    meta: dict = field(default_factory=dict)


def _heal_nonfinite(src: torch.Tensor, dose: torch.Tensor) -> tuple[torch.Tensor, int]:
    finite = torch.isfinite(src).all(dim=-1)  # (T, P)
    n_fallback = int((~finite).sum())
    if not n_fallback:
        return src, 0
    # heal genuinely nonfinite rows from the nearest chronologically valid
    # frame (chronology = ascending cumulative dose)
    order = torch.argsort(dose.to(_F64))
    src = src.clone()
    p_count = src.shape[1]
    for j in range(p_count):
        last = None
        for f in order.tolist():
            if finite[f, j]:
                last = src[f, j].clone()
            elif last is not None:
                src[f, j] = last
        nxt = None
        for f in reversed(order.tolist()):
            if finite[f, j]:
                nxt = src[f, j].clone()
            elif nxt is not None:
                src[f, j] = nxt
    if not torch.isfinite(src).all():
        raise ValueError("a particle has no finite source projection on any frame")
    return src, n_fallback


def lift_particle_trajectories(
    source_xy_a: torch.Tensor,  # (T, P, 2) full-model projections at the particles, canonical A
    relion_model: RelionTomogramModel,  # emitted global model (T rows), CTF convention set
    positions_eff_a: torch.Tensor,  # (P, 3) canonical corner-origin A
    dose: torch.Tensor,  # (T,) cumulative dose of the emitted rows
    *,
    source_disp_a: torch.Tensor | None = None,  # (T, P, 3) source 3D displacement at the particles
    source_ctf_depth_a: torch.Tensor | None = None,  # (T, P) source signed defocus contribution (A)
    gauge: str = "lowest-dose",
) -> TrajectoryLiftResult:
    if gauge not in GAUGES:
        raise ValueError(f"gauge must be one of {GAUGES}, got {gauge!r}")
    s = relion_model.pixel_size_a
    t_count = relion_model.n_projections
    p_count = positions_eff_a.shape[0]
    if source_xy_a.shape != (t_count, p_count, 2):
        raise ValueError(f"source projections {tuple(source_xy_a.shape)} vs ({t_count}, {p_count}, 2)")
    pos = positions_eff_a.to(_F64)
    if source_disp_a is not None:
        d = source_disp_a.to(_F64)
        if d.shape != (t_count, p_count, 3):
            raise ValueError(f"source displacement {tuple(d.shape)} vs ({t_count}, {p_count}, 3)")
        if not torch.isfinite(d).all():
            raise ValueError("source displacement contains non-finite values")
        depth_source = "displacement"
    else:
        d = torch.zeros(t_count, p_count, 3, dtype=_F64)
        depth_source = "min_norm"
    if source_ctf_depth_a is not None and source_ctf_depth_a.shape != (t_count, p_count):
        raise ValueError(f"source ctf depth {tuple(source_ctf_depth_a.shape)} vs ({t_count}, {p_count})")

    src, n_fallback = _heal_nonfinite(source_xy_a.to(_F64), dose)

    rot = relion_model.rotations  # (T, 3, 3)
    trans_px = relion_model.projection_matrices[:, :3, 3]  # (T, 3)
    # rigid projection of the DISPLACED point, in A
    warped_px = (pos[None] + d) / s  # (T, P, 3)
    rel_xy = (torch.einsum("tij,tpj->tpi", rot, warped_px) + trans_px[:, None, :])[..., :2] * s
    r_a = src - rel_xy  # (T, P, 2) in-plane remainder
    r3 = torch.cat([r_a, torch.zeros(t_count, p_count, 1, dtype=_F64)], dim=-1)
    t_a = d + torch.einsum("tji,tpj->tpi", rot, r3)  # d + R^T (r', 0)   (T, P, 3)

    ref_row = int(torch.argmin(dose.to(_F64)))
    c_ref = t_a[ref_row]  # (P, 3) lowest-dose gauge
    c = c_ref.clone()
    n_gauge_fallback = 0
    gauge_rank = None
    if gauge == "ctf-optimal":
        if source_ctf_depth_a is None:
            raise ValueError("gauge='ctf-optimal' needs the source CTF depth at every (frame, particle)")
        c, n_gauge_fallback, gauge_rank = _ctf_optimal_gauge(
            relion_model, pos, t_a, c_ref, source_ctf_depth_a.to(_F64)
        )

    positions_out = pos + c
    motion = t_a - c[None, :, :]

    # static positions must lie inside the volume under EITHER gauge (ir/build.py:115)
    vol = relion_model.volume_dims_a.to(_F64)
    outside = ((positions_out < 0) | (positions_out > vol)).any(dim=1)
    if bool(outside.any()):
        raise ValueError(
            f"{int(outside.sum())} gauged static particle position(s) fall outside the tomogram volume "
            f"{vol.tolist()} A under gauge {gauge!r}; the reference position itself is out of bounds"
        )

    # exactness: rebuild and compare (invariance of the gauge is exact by
    # linearity; this catches wiring errors, not representation error)
    check_pos = positions_out[None] + motion  # (T, P, 3)
    rcheck = torch.einsum("tij,tpj->tpi", rot, check_pos / s) + trans_px[:, None, :]
    finite = torch.isfinite(source_xy_a.to(_F64)).all(dim=-1)
    resid_px = (rcheck[..., :2] * s - src).norm(dim=-1)[finite] / s
    max_resid = float(resid_px.max()) if resid_px.numel() else 0.0

    dev = dev_rms = dev_max = None
    if source_ctf_depth_a is not None:
        dev = relion_model.ctf_depth(positions_out) - source_ctf_depth_a.to(_F64)  # (T, P)
        dev_rms = float(dev.pow(2).mean().sqrt())
        dev_max = float(dev.abs().max())

    return TrajectoryLiftResult(
        positions_out_a=positions_out,
        motion_a=motion,
        max_residual_px=max_resid,
        ref_row=ref_row,
        n_fallback=n_fallback,
        depth_source=depth_source,
        gauge=gauge,
        n_gauge_fallback=n_gauge_fallback,
        gauge_rank=gauge_rank,
        ctf_depth_deviation_a=dev,
        ctf_depth_deviation_rms_a=dev_rms,
        ctf_depth_deviation_max_a=dev_max,
        meta={
            "gauge": gauge,
            "depth_source": depth_source,
            "gauge_svd_rtol": GAUGE_SVD_RTOL,
            "ctf_depth_deviation_frames": "the emitted rows entering the gauge (not held-out)",
        },
    )


def _ctf_optimal_gauge(relion_model, pos, t_a, c_ref, source_depth):
    """Per particle: c = c_ref + B_r^+ (y - B c_ref), B_f = hand*slope*(R_f)_z,
    y_f = source_depth_f - RELION.ctf_depth_f(p). Retained directions
    sigma >= GAUGE_SVD_RTOL * sigma_max; fallback to c_ref when the shift exceeds
    the trajectory's own spread."""
    hand, slope = relion_model.hand, relion_model.defocus_slope
    b = hand * slope * relion_model.rotations[:, 2, :]  # (T, 3)
    base = relion_model.ctf_depth(pos)  # (T, P): depth of the undisplaced static position
    y = source_depth - base  # (T, P)
    u, sv, vh = torch.linalg.svd(b, full_matrices=False)
    keep = sv >= GAUGE_SVD_RTOL * sv.max()
    rank = int(keep.sum())
    b_pinv = (vh[keep].T * (1.0 / sv[keep])) @ u[:, keep].T  # (3, T)
    resid = y - torch.einsum("tj,pj->tp", b, c_ref)  # (T, P): what c_ref leaves
    delta = (b_pinv @ resid).T  # (P, 3)
    c = c_ref + delta
    # The SVD cut already bounds |delta| <= |resid| / sigma_min_retained. A shift
    # more than GAUGE_SHIFT_CAP times the largest depth deviation it corrects is a
    # degenerate direction leaking through nonetheless -> keep the lowest-dose gauge.
    cap = GAUGE_SHIFT_CAP * resid.abs().max(dim=0).values  # (P,)
    fallback = delta.norm(dim=-1) > cap
    n_fb = int(fallback.sum())
    if n_fb:
        warnings.warn(
            f"ctf-optimal gauge: {n_fb} particle(s) fell back to the lowest-dose gauge "
            f"(shift exceeds {GAUGE_SHIFT_CAP}x the depth deviation it corrects)",
            stacklevel=3,
        )
        c[fallback] = c_ref[fallback]
    return c, n_fb, rank


@dataclass
class LiftConsistency:
    rank: int  # rank of the beam-direction matrix B (2 for single-axis)
    residual_dof: int  # T_active - rank; the test is informative only if >= 1
    residual_rms_a: float  # RMS over particles of the best per-particle constant's residual
    residual_max_a: float
    verdict: str  # "consistent with a pure in-plane lift" | "carries depth beyond an in-plane lift" | "not informative"


def lift_consistency(
    motion_a: torch.Tensor,  # (T, P, 3) trajectories as read from motion.star
    rotations: torch.Tensor,  # (T, 3, 3) RELION rotations of those rows
    *,
    tol_a: float = 1e-3,
) -> LiftConsistency:
    """Does a per-particle constant c exist with (R_t (motion_t + c))_z = 0 for
    all t? True for any gauged min-norm lift (whatever its gauge), false when
    the trajectories carry dose-dependent depth. SVD solve of B c = -y with
    B_t = (R_t)_z; low sensitivity (a depth varying linearly with the beam
    direction is largely absorbed by c) — a consistency indicator, never proof.
    """
    t_count = motion_a.shape[0]
    b = rotations[:, 2, :].to(_F64)  # (T, 3)
    y = torch.einsum("tj,tpj->tp", b, motion_a.to(_F64))  # (T, P) beam components
    u, sv, vh = torch.linalg.svd(b, full_matrices=False)
    keep = sv >= GAUGE_SVD_RTOL * sv.max()
    rank = int(keep.sum())
    dof = t_count - rank
    b_pinv = (vh[keep].T * (1.0 / sv[keep])) @ u[:, keep].T  # (3, T)
    c = (b_pinv @ (-y)).T  # (P, 3)
    res = y + torch.einsum("tj,pj->tp", b, c)  # (T, P)
    rms = float(res.pow(2).mean().sqrt())
    mx = float(res.abs().max()) if res.numel() else 0.0
    if dof < 1:
        verdict = "not informative"
    elif mx <= tol_a:
        verdict = "consistent with a pure in-plane lift"
    else:
        verdict = "carries depth beyond an in-plane lift"
    return LiftConsistency(rank=rank, residual_dof=dof, residual_rms_a=rms, residual_max_a=mx, verdict=verdict)
