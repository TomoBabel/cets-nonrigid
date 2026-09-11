"""Model protocols.

Tilt-series and frame-series maps are deliberately separate protocols with
distinct method names and point ranks, so a 3D volume point can never be fed
to a 2D motion map (or vice versa) without a type error.

Both return explicit validity masks alongside positions; validity is never
encoded through weights (NaN * 0 == NaN in torch).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class TiltProjectionModel(Protocol):
    """Maps 3D tomogram points to 2D positions on each raw tilt image.

    Coordinates are canonical: volume points in Angstrom, corner origin;
    image positions in Angstrom, corner origin of the raw motion-corrected
    tilt image.
    """

    n_projections: int

    def project_volume(self, points_3d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full model. points_3d (N, 3) -> positions (T, N, 2), valid (T, N) bool."""
        ...

    def project_volume_global(self, points_3d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Global-only (rigid) model, local deformation fields zeroed."""
        ...


@runtime_checkable
class FrameMotionModel(Protocol):
    """Maps 2D corrected/reference-frame points to positions in each raw frame.

    Map direction (frozen contract): for a point ``x`` in the corrected frame,
    ``map_image`` returns its source sample position in raw frame ``f``
    (``out(x) = in(x - S(x))`` semantics in both tools). Coordinates are
    canonical: Angstrom, corner origin.
    """

    n_projections: int

    def map_image(self, points_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full model. points_2d (N, 2) -> positions (F, N, 2), valid (F, N) bool."""
        ...

    def map_image_global(self, points_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Global-only (per-frame rigid shift) model."""
        ...


@runtime_checkable
class VolumeDeformingModel(Protocol):
    """A tilt-series model whose 3D deformation is evaluated BEFORE projection.

    ``displace_volume`` returns the additive per-projection 3D displacement of
    each sample in the canonical volume frame (Angstrom):
    ``warped_t = points + displacement[t]``. Warp implements it through its
    quadrilinear ``GridVolumeWarp`` grids (dose axis, so the displacement is a
    per-projection quantity); RELION's particle-bound model through its
    per-particle trajectories. Models without a 3D deformation (AreTomo3, the
    bare RELION tomogram model) do not implement it.
    """

    n_projections: int

    def displace_volume(self, points_3d: torch.Tensor) -> torch.Tensor:
        """points_3d (N, 3) -> displacement (T, N, 3), Angstrom, canonical frame."""
        ...


@runtime_checkable
class CtfDepthModel(Protocol):
    """A tilt-series model with a per-particle CTF depth convention.

    ``ctf_depth`` returns the SIGNED defocus contribution in Angstrom that the
    tool itself adds to the per-tilt defocus for a particle at ``points_3d``,
    with the tool's own conventions (handedness, defocus slope, centre, angle
    inversion) already applied. Callers only subtract two models' outputs and
    scale by 1e-4 to micrometres; no further sign or slope factor is legal.

    ``displacement`` is the per-projection 3D displacement of the UNDEFORMED
    ``points_3d`` (see ``VolumeDeformingModel``); ``None`` means "use the
    model's own deformation" (Warp) or "not applicable" (RELION, whose CTF
    always uses the static coordinate — ``subtomo.cpp:780-783``).
    """

    n_projections: int

    def ctf_depth(self, points_3d: torch.Tensor, displacement: torch.Tensor | None = None) -> torch.Tensor:
        """points_3d (N, 3) [+ displacement (T, N, 3)] -> signed defocus contribution (T, N), Angstrom."""
        ...
