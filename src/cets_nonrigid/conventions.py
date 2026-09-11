"""Single source of truth for every discrete convention cets_nonrigid relies on.

All sign constants, offsets, and frame conventions live here. Nothing outside
this module may hard-code a sign relating the Warp and AreTomo3 models.

Internal native-kernel frame (CETS exchange uses a centered physical frame):
  - Image: raw motion-corrected tilt image (or movie frame), Angstrom units,
    corner origin, pixel ``i`` centered at coordinate ``i * pixel_size``.
  - Volume: tomogram XYZ in Angstrom, corner origin (Warp's
    ``VolumeDimensionsAngstrom`` frame); volume center projects to image
    center under zero shifts.

Sources (verified against the staged checkouts, 2026-08-28):
  - warp/WarpLib/TiltSeries/TiltSeries.cs (GetPositionInAllTilts, line 402)
  - AreTomo3/AreTomo/Correct/GCorrPatchShift.cu (apply-side model)
  - AreTomo3/AreTomo/PatchAlign/CFitPatchShifts.cpp (fit-side model)
  - cryoet-alignment/src/cryoet_alignment/io/cryoet_data_portal/alignment.py
    (empirically validated tilt angle/axis signs)
"""

from __future__ import annotations

# --- Tilt-series angle relations (Warp <-> AreTomo3 .aln) --------------------

#: Warp tilt angle = WARP_TILT_ANGLE_SIGN * AreTomo TILT column.
#: The AreTomo TILT column already includes AlphaOffset; the Warp counterpart of
#: the composite tilt is (Angle_t + LevelAngleY). Exact relation:
#: TILT = -(Angle + LevelAngleY).
WARP_TILT_ANGLE_SIGN: int = -1

#: Warp TiltAxisAngle = WARP_TILT_AXIS_SIGN * AreTomo ROT column (degrees,
#: measured from +Y, CCW positive in both tools).
WARP_TILT_AXIS_SIGN: int = 1

#: AreTomo's patch-fit frame computes fX = Cx*cos(theta) - Cz*sin(theta), i.e.
#: its Z axis is inverted relative to the canonical (Warp) volume Z:
#: Cz_fit = ARETOMO_FIT_Z_SIGN * z_centered / s, with s the pixel size (A/px)
#: of the motion-corrected tilt-series image (the scale of all .aln pixel
#: quantities).
ARETOMO_FIT_Z_SIGN: int = -1

# --- Local-shift semantics ---------------------------------------------------

#: AreTomo tilt-series local shifts are POSITIONS: raw position = rigid
#: projection + IDW(shift) + global shift (GCorrPatchShift.cu adds them).
ARETOMO_TS_SHIFT_IS_POSITION: bool = True

#: Warp GridMovementX/Y values are SUBTRACTED from the projected position
#: (TiltSeries.cs step (d)).
WARP_MOVEMENT_IS_SUBTRACTED: bool = True

#: Frame-series (MotionCor and Warp movie) shifts are both CORRECTIONS
#: subtracted from the corrected-frame coordinate: raw = x - S(x) - glob.
#: The sign flip exists only between tilt-series locals and motion shifts,
#: never within a single conversion direction.
MOTION_SHIFT_IS_CORRECTION: bool = True

# --- Pixel-center convention -------------------------------------------------

#: Offset (in pixels) between AreTomo's centered pixel coordinates and the
#: canonical corner-origin coordinates beyond the N/2 shift:
#: x_canonical_px = u_centered + N/2 + HALF_PIXEL_OFFSET.
#: PROVISIONAL (0.0) until the synthetic odd/even image-size and kernel tests
#: pin it. Fixed convention thereafter: conversion never chooses it
#: automatically; a diagnostic override exists for investigation only.
HALF_PIXEL_OFFSET: float = 0.0

# --- IDW kernels (do not conflate!) ------------------------------------------

#: Tilt-series local field (GCorrPatchShift.cu): w = exp(-100 * r_norm^2) with
#: r_norm^2 = ((du/Nx)^2 + (dv/Ny)^2); NO distance cutoff; patches with
#: Good < 0.9 skipped.
ARETOMO_TS_IDW_EXPONENT: float = -100.0

#: MotionCor frame local field (GCorrectPatchShift.cu): w = exp(-100 * r_norm)
#: (LINEAR r), hard cutoff at r_norm > 0.5; bad patches skipped.
ARETOMO_MOTION_IDW_EXPONENT: float = -100.0
ARETOMO_MOTION_IDW_CUTOFF: float = 0.5

#: Good-flag threshold used by the apply kernel (good >= 0.9 participates).
ARETOMO_GOOD_THRESHOLD: float = 0.9

# --- RELION 5 (relion 5.0.1, checkout 210f68c8) -------------------------------

#: rlnMicrographShiftX/Y and the local polynomial are the CORRECTION (-drift):
#: corrected(x) = raw(x - shift), evaluated at the corrected coordinate — the
#: same direction as the frozen FrameMotionModel contract, NO flip
#: (motioncorr_runner.cpp:2057; micrograph_handler.cpp:543-563).
RELION_MOTION_SHIFT_IS_CORRECTION: bool = True

#: Trajectories (motion.star) are 3D Angstrom offsets ADDED to the particle
#: position before rigid projection (trajectory.cpp:164-176;
#: particle_set.cpp:680-692).
RELION_TRAJECTORY_IS_ADDITIVE: bool = True

#: Sentinel for unobserved global shifts in the micrograph motion star
#: (micrograph_model.cpp:29).
RELION_NOT_OBSERVED: float = -9999.0

#: rlnMicrographFrameNumber and rlnMicrographStartFrame are 1-indexed.
RELION_FRAME_INDEX_BASE: int = 1

#: Third-order polynomial local motion model: 18 coefficients per output
#: component, basis {1, x, x^2, y, y^2, xy} x {z, z^2, z^3}, no constant term
#: (micrograph_model.cpp:32-51).
RELION_POLY_N_COEFFS: int = 36

#: The projection matrix uses INTEGER-DIVISION volume/image centers
#: (tomogram.cpp:44,53) while the matrix inverse and the particle-coordinate
#: conversion use FLOAT centers (tomogram.cpp:101; tomogram_set.cpp:314) —
#: prefer even dims; odd dims are handled via the delta terms in the shifts.
RELION_CENTER_IS_INT_DIV: bool = True

#: Closed-form global mapping (verified structural proof; see plan appendix):
#:   Warp -> RELION: xtilt = LevelAngleX, ytilt = -(Angle + LevelAngleY),
#:                   zrot = TiltAxisAngle, shifts = AxisOffset (+ odd-dim deltas)
#:   AreTomo -> RELION: zrot = ROT, ytilt = TILT, xtilt = 0,
#:                      shifts_A = pixel_size * (TX, TY)
#: z_relion = +z_warp (no flip).
#:
#: rlnTomoHand for the converter's Euler mapping: PINNED by golden G4
#: (tests/test_r2_ctf.py) — warpylib's depth-defocus channel matches
#: dz = (+1) * pixelSize * depthOffset for AreAnglesInverted=False (and -1 for
#: inverted) when the RELION matrix is built from our xtilt/ytilt/zrot mapping.
#: Warp's own export writes -1 for not-inverted (RelionParticleSeriesExport
#: .cs:31-34) — that value belongs to Warp's rlnTomoProj* matrix convention,
#: not to ours. G11 (R4) re-validates against real WarpTools-exported data.
RELION_TOMO_HAND_FOR_WARP_NOT_INVERTED: int = 1

# --- Fit dtypes --------------------------------------------------------------

#: Fits run in float64 through pure cets_nonrigid operators; emitted results are
#: re-evaluated through the float32 compatibility model. Both recorded in fit
#: metadata.
FIT_DTYPE: str = "float64"
COMPATIBILITY_DTYPE: str = "float32"

# --- Store schema ------------------------------------------------------------

#: Still readable: 0.2 stores (no held-out global baseline, row metadata in
#: meta_json). 0.3 promotes source_projected_global to a normative fitting
#: baseline (held-out counterpart stored), makes /projection/* the single row
#: authority (meta_json carries no row fields), and adds label/angle_kind.
#: 0.4 adds OPTIONAL per-projection 3D source arrays: ir/source_displacement_3d
#: (T,N,3) f4 + held-out twin (the source's additive 3D displacement,
#: warped = points + d) and ir/source_ctf_depth_a (T,N) f4 + twin (the source
#: tool's signed defocus contribution, its conventions applied), each with an
#: ir attr saying why it is present/absent (displacement_3d: present |
#: zero_at_samples | none | not_recorded; source_ctf_depth: present | none |
#: not_recorded). Pre-0.4 stores read as not_recorded. The 0.4 writer does NOT
#: require them. Fit stores may persist rank-4 fit/volume_warp_{x,y,z} grids.


def sign_conventions() -> dict:
    """The convention set recorded into every store's root attributes."""
    return {
        "warp_tilt_angle_sign": WARP_TILT_ANGLE_SIGN,
        "warp_tilt_axis_sign": WARP_TILT_AXIS_SIGN,
        "aretomo_fit_z_sign": ARETOMO_FIT_Z_SIGN,
        "aretomo_ts_shift_is_position": ARETOMO_TS_SHIFT_IS_POSITION,
        "warp_movement_is_subtracted": WARP_MOVEMENT_IS_SUBTRACTED,
        "motion_shift_is_correction": MOTION_SHIFT_IS_CORRECTION,
        "half_pixel_offset": HALF_PIXEL_OFFSET,
        "relion_motion_shift_is_correction": RELION_MOTION_SHIFT_IS_CORRECTION,
        "relion_trajectory_is_additive": RELION_TRAJECTORY_IS_ADDITIVE,
        "relion_center_is_int_div": RELION_CENTER_IS_INT_DIV,
    }
