"""Sign calibration on a dual-processed dataset.

Given the SAME tilt series aligned independently by Warp and AreTomo3,
compare global-only projections of a coarse 3D grid under all candidate
interpretation flips of the .aln geometry (tilt-angle sign, tilt-axis sign,
volume-Z sign). The configured interpretation (identity: exactly what
:class:`AretomoTsModel` + ``conventions.py`` implement) must be the argmin
with a meaningful margin, otherwise calibration FAILS.

A wrong sign produces geometry errors of hundreds to thousands of Angstrom;
genuine alignment differences between two independent solutions are tens of
Angstrom — hence the margin criterion. Zero-shift or angle-symmetric datasets
cannot separate candidates and are rejected via the same margin.

EXACT GAUGE SYMMETRY (verified on real data): the AreTomo global projection
``fX = Cx cos(theta) - Cz sin(theta)`` is invariant under jointly flipping
``(theta, Cz) -> (-theta, -Cz)``, so the tilt-angle sign and the volume-Z
sign are jointly unidentifiable from projections — on ANY dataset. The
tilt-angle sign is pinned definitionally instead: AreTomo3's writer emits the
TILT column as the (sorted, AlphaOffset-added) stage angle by construction
(``CSaveAlignFile.cpp``), which then pins Z. Calibration therefore
enumerates only the identifiable quotient: (tilt-axis flip) x (joint
tilt·Z flip, realized as a Z flip with the tilt sign held fixed).

The half-pixel convention is deliberately NOT calibrated here (absorbable by
TX/TY; fixed once by synthetic odd/even + kernel tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import torch

from cets_nonrigid.io.aln import AlnSeries, TiltMatch
from cets_nonrigid.ir.sampling import volume_grid
from cets_nonrigid.models.aretomo_ts import AretomoTsModel
from cets_nonrigid.models.warp_ts import WarpTiltSeriesModel

Candidate = tuple[int, int, int]  # (tilt_axis_flip, tilt_axis_180_shift, joint_tilt_z_flip)
CONFIGURED: Candidate = (1, 0, 1)

DEFAULT_MARGIN = 0.5  # best RMS must be <= margin * runner-up RMS


@dataclass
class CalibrationResult:
    best: Candidate
    configured_is_best: bool
    rms_by_candidate: dict[Candidate, float]  # Angstrom
    margin: float  # best / runner-up
    n_observations: int


def _variant_model(base: AretomoTsModel, s_axis: int, axis_shift_deg: float) -> AretomoTsModel:
    return AretomoTsModel(
        rot_deg=base.rot_deg * s_axis + axis_shift_deg,
        tilt_deg=base.tilt_deg,
        shifts_px=base.shifts_px,
        raw_size_px=tuple(base.raw_size_px.tolist()),
        pixel_size_a=base.pixel_size_a,
        volume_dims_a=tuple(base.volume_dims_a.tolist()),
        local=None,  # global-only comparison
    )


def calibrate_signs(
    warp_model: WarpTiltSeriesModel,
    aln_series: AlnSeries,
    match: TiltMatch,
    *,
    grid_shape: tuple[int, int, int] = (5, 5, 3),
    margin: float = DEFAULT_MARGIN,
) -> CalibrationResult:
    vol = warp_model.volume_dims_a.to(torch.float64)
    points = volume_grid(vol, grid_shape)

    warp_xy, warp_valid = warp_model.project_volume_global(points)
    perm = torch.tensor(match.aln_to_warp)
    ref_xy = warp_xy[perm]  # (T_aln, N, 2)
    ref_valid = warp_valid[perm]

    base = aln_series.model
    rms_by: dict[Candidate, float] = {}
    # Axis candidates cover both the mirror (-ROT) and the 180-degree shift
    # (the "known handedness fix" alternative: axis - 180 vs tilt-sign flip;
    # the latter is the joint flip's projection-visible half).
    for s_axis, shift, s_joint in product((1, -1), (0, 180), (1, -1)):
        model = _variant_model(base, s_axis, float(shift))
        pts = points.clone()
        if s_joint == -1:
            pts[:, 2] = float(vol[2]) - pts[:, 2]  # flip about the volume center
        at_xy, _ = model.project_volume_global(pts)
        diff = (at_xy - ref_xy.to(at_xy))[ref_valid]
        rms_by[(s_axis, shift, s_joint)] = float(diff.pow(2).mean().sqrt())

    ranked = sorted(rms_by.items(), key=lambda kv: kv[1])
    best, best_rms = ranked[0]
    runner_rms = ranked[1][1]
    ratio = best_rms / runner_rms if runner_rms > 0 else float("inf")

    result = CalibrationResult(
        best=best,
        configured_is_best=(best == CONFIGURED),
        rms_by_candidate=rms_by,
        margin=ratio,
        n_observations=int(ref_valid.sum()),
    )

    if not result.configured_is_best:
        raise ValueError(
            f"sign calibration FAILED: best candidate {best} "
            f"(rms {best_rms:.1f} A) != configured {CONFIGURED} "
            f"(rms {rms_by[CONFIGURED]:.1f} A)"
        )
    if ratio > margin:
        raise ValueError(
            f"sign calibration ambiguous: best rms {best_rms:.1f} A vs runner-up "
            f"{runner_rms:.1f} A (ratio {ratio:.2f} > margin {margin}) - this dataset "
            "cannot separate sign candidates (symmetric/zero-shift?)"
        )
    return result
