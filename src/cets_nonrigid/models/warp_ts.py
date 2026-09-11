"""Warp tilt-series model — thin wrapper around warpylib's TiltSeries.

warpylib's ``get_position_in_all_tilts`` is the golden-tested forward model;
this wrapper adapts it to the :class:`TiltProjectionModel` protocol (drops the
defocus channel, adds validity) and provides the global-only variant with all
local deformation grids zeroed.

It additionally exposes the two intermediates Warp computes and discards
(``warp/WarpLib/TiltSeries/TiltSeries.cs`` ``GetPositionInAllTilts``,
lines 401-503; warpylib ``tilt_series/positions.py:176-349``):

* :meth:`displace_volume` — the additive 3D volume-warp displacement per tilt
  (``TiltSeries.cs:410-426`` coordinates, ``:454-457`` application; the 4th
  grid axis is normalized dose, so the displacement is a per-projection
  quantity).  Implements :class:`VolumeDeformingModel`.
* :meth:`ctf_depth` — the signed defocus contribution in Angstrom,
  ``Z_t = (R_t (p - V/2 + d_t))_z`` with the ``AreAnglesInverted`` flip
  (``TiltSeries.cs:459-480``), which Warp adds as ``1e-4 * Z`` micrometres to
  the per-tilt defocus (``:500``; ``SizeRoundingFactors.Z`` is hard-coded 1,
  ``:1494``).  Implements :class:`CtfDepthModel`.
* :meth:`project_volume_premovement` — the pre-correction image position at
  which ``GridMovementX/Y`` are sampled (``TiltSeries.cs:471``): volume warp on,
  movement grids off.

Dose convention (deliberate divergence from Warp, documented in the plan):
Warp's ``DoseStep = 1f / (MaxDose - MinDose)`` (``TiltSeries.cs:411``) is +Inf
for an equal-dose series and every position becomes NaN through the native
lerp; warpylib substitutes ``dose_step = 0`` (``positions.py:217-218``) so every
tilt samples dose node 0.  ``displace_volume`` follows warpylib so that
``rigid(p + d) - M == project_volume`` holds for every series cets_nonrigid can
load; the strict loader refuses equal-dose series whose volume warp is
dose-dependent (see ``io/warp_xml.py``).
"""

from __future__ import annotations

import copy

import torch
from warpylib import CubicGrid, LinearGrid4D, TiltSeries
from warpylib.euler import euler_to_matrix, rotate_x
from warpylib.tilt_series.positions import get_position_in_all_tilts_single

_F32 = torch.float32


def volume_warp_is_dose_dependent(ts: TiltSeries) -> bool:
    """True iff any ``GridVolumeWarp*`` grid has more than one dose node."""
    return any(
        g.dimensions[3] > 1 for g in (ts.grid_volume_warp_x, ts.grid_volume_warp_y, ts.grid_volume_warp_z)
    )


def dose_range(ts: TiltSeries) -> float:
    """``MaxDose - MinDose`` over ALL tilts (``TiltSeries.cs:320-321``)."""
    return float(ts.max_dose - ts.min_dose)


#: Why an equal-dose series is invalid Warp metadata, and what to do about it.
#: Warp normalizes the volume-warp dose axis by 1/(MaxDose-MinDose)
#: (TiltSeries.cs:411): with equal doses every position is NaN and WarpTools
#: dies in GetCTFsForOneParticle (TiltSeries.cs:908) — verified with WarpTools
#: 2.0.0 ts_reconstruct on real data, 2026-09-09. Dose is Warp's time axis; real
#: acquisitions never produce this, only synthesized metadata does.
EQUAL_DOSE_NUDGE = (
    "every tilt carries the same cumulative dose - Warp normalizes its volume-warp time axis by "
    "1/(MaxDose-MinDose) and evaluates NaN positions for such a series (verified: WarpTools 2.0.0 "
    "ts_reconstruct fails). Supply the acquisition order AND the per-tilt dose: the _TLT.txt "
    "companion or the mdoc (--mdoc-dir) carry both; --dose-per-tilt D|file sets the per-image dose "
    "on top of that order (a constant alone cannot be accumulated without it); RELION sources carry "
    "rlnMicrographPreExposure"
)


class WarpTiltSeriesModel:
    """Canonical-frame projection model backed by a warpylib TiltSeries."""

    def __init__(self, ts: TiltSeries):
        img = ts.image_dimensions_physical
        vol = ts.volume_dimensions_physical
        if img is None or vol is None or float(img.min()) <= 0 or float(vol.min()) <= 0:
            raise ValueError(
                "TiltSeries has missing/zero physical dimensions - load it through "
                "cets_nonrigid.io.warp_xml.load_warp_tiltseries (old Warp XML formats "
                "without ImageDimensionsAngstrom/VolumeDimensionsAngstrom are not supported)"
            )
        self.ts = ts
        self._ts_global = self._strip_local_grids(ts, volume_warp=True, movement=True)
        self._ts_premovement = self._strip_local_grids(ts, volume_warp=False, movement=True)

    @staticmethod
    def _strip_local_grids(ts: TiltSeries, *, volume_warp: bool, movement: bool) -> TiltSeries:
        stripped = copy.deepcopy(ts)
        if movement:
            stripped.grid_movement_x = CubicGrid((1, 1, 1))
            stripped.grid_movement_y = CubicGrid((1, 1, 1))
        if volume_warp:
            stripped.grid_volume_warp_x = LinearGrid4D((1, 1, 1, 1))
            stripped.grid_volume_warp_y = LinearGrid4D((1, 1, 1, 1))
            stripped.grid_volume_warp_z = LinearGrid4D((1, 1, 1, 1))
        return stripped

    @property
    def n_projections(self) -> int:
        return self.ts.n_tilts

    @property
    def image_dims_a(self) -> torch.Tensor:
        return self.ts.image_dimensions_physical

    @property
    def volume_dims_a(self) -> torch.Tensor:
        return self.ts.volume_dimensions_physical

    def _project(self, ts: TiltSeries, points_3d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # warpylib computes in float32; (N, 3) -> (N, T, 3) -> (T, N, 2)
        out = get_position_in_all_tilts_single(ts, points_3d.to(_F32))
        xy = out[..., :2].permute(1, 0, 2)  # (T, N, 2)

        img = ts.image_dimensions_physical.to(xy)
        valid = (
            (xy[..., 0] >= 0)
            & (xy[..., 0] <= img[0])
            & (xy[..., 1] >= 0)
            & (xy[..., 1] <= img[1])
            & torch.isfinite(xy).all(dim=-1)
        )
        # A tilt disabled in Warp contributes no observations.
        use = ts.use_tilt.to(torch.bool)
        valid = valid & use[:, None]
        return xy, valid

    def project_volume(self, points_3d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._project(self.ts, points_3d)

    def project_volume_global(self, points_3d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._project(self._ts_global, points_3d)

    def project_volume_premovement(self, points_3d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Volume warp ON, movement grids OFF: the position at which Warp samples
        ``GridMovementX/Y`` (``TiltSeries.cs:471``, ``positions.py:318-322``)."""
        return self._project(self._ts_premovement, points_3d)

    # ------------------------------------------------------------------
    # 3D intermediates (VolumeDeformingModel / CtfDepthModel)
    # ------------------------------------------------------------------

    def _dose_coords(self) -> torch.Tensor:
        """(T,) float32 normalized dose exactly as warpylib (``positions.py:216-218, 236``)."""
        ts = self.ts
        dr = ts.max_dose - ts.min_dose
        dose_step = 1.0 / dr if dr > 0 else 0.0
        return ((ts.dose - ts.min_dose) * dose_step).to(_F32)

    def displace_volume(self, points_3d: torch.Tensor) -> torch.Tensor:
        """Additive volume-warp displacement (T, N, 3) in float32 Angstrom.

        Transcribes ``TiltSeries.cs:410-426`` (coordinates ``(x/Vx, y/Vy, z/Vz,
        (Dose[t]-MinDose)*DoseStep)``, three ``LinearGrid4D`` lookups) as
        evaluated by warpylib ``positions.py:235-257``; the result is the
        ``SampleWarping`` added at ``TiltSeries.cs:457``.
        """
        ts = self.ts
        pts = points_3d.to(_F32)
        n = pts.shape[0]
        t = ts.n_tilts
        normalized = pts / ts.volume_dimensions_physical.to(_F32)  # (N, 3)
        dose = self._dose_coords()  # (T,)
        coords4 = torch.cat(
            [
                normalized[None, :, :].expand(t, n, 3),
                dose[:, None, None].expand(t, n, 1),
            ],
            dim=-1,
        ).reshape(-1, 4)
        dx = ts.grid_volume_warp_x.get_interpolated(coords4)
        dy = ts.grid_volume_warp_y.get_interpolated(coords4)
        dz = ts.grid_volume_warp_z.get_interpolated(coords4)
        return torch.stack([dx, dy, dz], dim=-1).reshape(t, n, 3).to(_F32)

    def _tilt_matrices(self, flipped: bool) -> torch.Tensor:
        """(T, 3, 3) float32 as ``positions.py:263-275`` / ``:299-308``."""
        ts = self.ts
        deg_to_rad = torch.pi / 180.0
        sign = -1.0 if flipped else 1.0
        euler = torch.stack(
            [
                torch.zeros(ts.n_tilts, dtype=_F32),
                sign * (ts.angles + ts.level_angle_y) * deg_to_rad,
                -ts.tilt_axis_angles * deg_to_rad,
            ],
            dim=-1,
        )
        m = euler_to_matrix(euler)
        lx = rotate_x(torch.tensor([sign * ts.level_angle_x * deg_to_rad])).squeeze(0)
        return torch.matmul(m, lx)

    def ctf_depth(self, points_3d: torch.Tensor, displacement: torch.Tensor | None = None) -> torch.Tensor:
        """Signed defocus contribution (T, N) in Angstrom for UNDEFORMED ``points_3d``.

        ``Z_t = (R_t (p - V/2 + d_t))_z`` (``TiltSeries.cs:454-467``); when
        ``AreAnglesInverted`` the z of the warped centred point is negated and
        the flipped rotation used (``:472-480``). Warp adds ``1e-4 * Z_t``
        micrometres to the tilt defocus (``:500``). ``displacement=None``
        evaluates this model's own :meth:`displace_volume`; passing one (e.g. a
        fitted target's) evaluates the depth Warp would use with that field.
        The volume warp is never re-evaluated at deformed positions.
        """
        ts = self.ts
        assert float(ts.size_rounding_factors[2]) == 1.0, "Warp hard-codes SizeRoundingFactors.Z = 1"
        pts = points_3d.to(_F32)
        d = self.displace_volume(pts) if displacement is None else displacement.to(_F32)
        if d.shape != (ts.n_tilts, pts.shape[0], 3):
            raise ValueError(f"displacement has shape {tuple(d.shape)}, expected ({ts.n_tilts}, {pts.shape[0]}, 3)")
        centered = pts[None, :, :] - (ts.volume_dimensions_physical.to(_F32) / 2) + d  # (T, N, 3)
        if bool(getattr(ts, "are_angles_inverted", False)):
            centered = centered.clone()
            centered[..., 2] *= -1
            rot = self._tilt_matrices(flipped=True)
        else:
            rot = self._tilt_matrices(flipped=False)
        z = torch.einsum("tji,tni->tnj", rot, centered)[..., 2]
        return z.to(_F32)
