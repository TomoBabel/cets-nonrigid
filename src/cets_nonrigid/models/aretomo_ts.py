"""AreTomo3 tilt-series projection model (independent torch implementation).

Replicates the model implied by the ``.aln`` file and applied by
``AreTomo3/AreTomo/Correct/GCorrPatchShift.cu`` (verified against the kernel
source, see docs/conventions.md):

Global (fit-convention frame, centered tilt-series px)::

    fX = Cx cos(TILT) - Cz sin(TILT)
    u0 = fX cos(ROT) - Cy sin(ROT)      # the ".aln Coord frame"
    v0 = fX sin(ROT) + Cy cos(ROT)
    raw = (u0, v0) + S_local(u0, v0) + (TX, TY)

Local field ``S_local`` (``mGCalcLocalShift``): Gaussian-IDW over patches with
``Good >= 0.9``::

    w_p = exp(-100 * (((u - CoordX_p)/Nx)^2 + ((v - CoordY_p)/Ny)^2))
    S   = sum(w_p * shift_p) / sum(w_p)     if any good patch, else (0, 0)

evaluated at the rotated PRE-global-shift coordinate; no distance cutoff.
Note the CUDA kernel divides whenever ANY good patch exists — if every weight
underflows in float32, the result is NaN. ``idw_mode="compat"`` reproduces
that faithfully (float32 arithmetic); ``idw_mode="stable"`` (default for
fitting) subtracts the per-point max exponent so the ratio is exact and
finite everywhere. The mode used is recorded in fit metadata.

TILT values already include AlphaOffset; ROT is degrees from +Y, CCW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from cets_nonrigid import frames

IdwMode = Literal["stable", "compat"]

#: exp(-100 * r_norm^2); r_norm normalized by the raw image dims.
_IDW_EXPONENT = -100.0
_GOOD_THRESHOLD = 0.9


@dataclass
class AreTomoLocalField:
    """Per-tilt Gaussian-IDW residual field (the ``.aln`` local section).

    coord_xy: (T, P, 2) per-tilt projected patch-center positions, centered px
              (the .aln CoordX/CoordY columns — these change per tilt).
    shift_xy: (T, P, 2) residual shifts, px (.aln ShiftX/ShiftY).
    good:     (T, P) float (.aln Good column; kernel gate is ``>= 0.9``).
    raw_size_px: (2,) int tensor — the UNPADDED raw image size used to
              normalize distances (kernel args iInImgX / giInSize[1]).
    """

    coord_xy: torch.Tensor
    shift_xy: torch.Tensor
    good: torch.Tensor
    raw_size_px: torch.Tensor

    def __post_init__(self) -> None:
        t, p, two = self.coord_xy.shape
        if two != 2 or self.shift_xy.shape != (t, p, 2) or self.good.shape != (t, p):
            raise ValueError(
                f"inconsistent local field shapes: coord {tuple(self.coord_xy.shape)}, "
                f"shift {tuple(self.shift_xy.shape)}, good {tuple(self.good.shape)}"
            )

    @property
    def n_tilts(self) -> int:
        return self.coord_xy.shape[0]

    @property
    def n_patches(self) -> int:
        return self.coord_xy.shape[1]

    def evaluate(self, uv: torch.Tensor, mode: IdwMode = "stable") -> torch.Tensor:
        """Evaluate the field at uv (T, N, 2) centered px -> shifts (T, N, 2) px.

        The evaluation coordinate is the rotated, PRE-global-shift position
        (the same frame the Coord columns live in).
        """
        if uv.dim() != 3 or uv.shape[0] != self.n_tilts or uv.shape[-1] != 2:
            raise ValueError(f"uv must be (T={self.n_tilts}, N, 2), got {tuple(uv.shape)}")

        dtype = torch.float32 if mode == "compat" else uv.dtype
        uv = uv.to(dtype)
        coord = self.coord_xy.to(dtype)  # (T, P, 2)
        shift = self.shift_xy.to(dtype)
        nx = float(self.raw_size_px[0])
        ny = float(self.raw_size_px[1])

        dx = (uv[..., None, 0] - coord[:, None, :, 0]) / nx  # (T, N, P)
        dy = (uv[..., None, 1] - coord[:, None, :, 1]) / ny
        e = _IDW_EXPONENT * (dx * dx + dy * dy)  # (T, N, P)

        good = (self.good >= _GOOD_THRESHOLD)[:, None, :]  # (T, 1, P)
        any_good = good.any(dim=-1)  # (T, 1) -> broadcasts over N

        if mode == "stable":
            # Max over good patches only; the shift ratio is invariant to the
            # subtraction, so this equals the kernel result wherever the
            # kernel's float32 sums do not underflow — and stays finite where
            # they do.
            e_masked = torch.where(good, e, torch.full_like(e, -math.inf))
            e_max = e_masked.max(dim=-1, keepdim=True).values
            e_max = torch.where(torch.isfinite(e_max), e_max, torch.zeros_like(e_max))
            w = torch.where(good, torch.exp(e - e_max), torch.zeros_like(e))
        else:
            # CUDA-faithful float32: plain expf; with >=1 good patch the
            # division happens even if all weights underflowed -> NaN (0/0),
            # exactly like mGCalcLocalShift.
            w = torch.where(good, torch.exp(e), torch.zeros_like(e))

        num = torch.einsum("tnp,tpc->tnc", w, shift)  # (T, N, 2)
        den = w.sum(dim=-1, keepdim=True)  # (T, N, 1)

        result = num / den
        zero = torch.zeros_like(result)
        return torch.where(any_good[..., None], result, zero)


class AretomoTsModel:
    """AreTomo3 tilt-series model over the dark-removed, tilt-sorted list.

    Native math happens in the AreTomo fit frame; ``project_volume`` speaks
    the canonical (Warp) frame via :mod:`cets_nonrigid.frames`.
    """

    def __init__(
        self,
        rot_deg: torch.Tensor,  # (T,)
        tilt_deg: torch.Tensor,  # (T,) incl. AlphaOffset
        shifts_px: torch.Tensor,  # (T, 2) TX, TY
        raw_size_px: tuple[int, int],
        pixel_size_a: float,
        volume_dims_a: tuple[float, float, float],
        local: AreTomoLocalField | None = None,
        idw_mode: IdwMode = "stable",
    ):
        t = rot_deg.shape[0]
        if tilt_deg.shape != (t,) or shifts_px.shape != (t, 2):
            raise ValueError("rot_deg (T,), tilt_deg (T,), shifts_px (T,2) required")
        if local is not None and local.n_tilts != t:
            raise ValueError(f"local field has {local.n_tilts} tilts, model has {t}")
        if pixel_size_a <= 0:
            raise ValueError("pixel_size_a must be positive")

        self.rot_deg = rot_deg
        self.tilt_deg = tilt_deg
        self.shifts_px = shifts_px
        self.raw_size_px = torch.tensor(raw_size_px)
        self.pixel_size_a = float(pixel_size_a)
        self.volume_dims_a = torch.tensor(volume_dims_a, dtype=torch.float64)
        self.local = local
        self.idw_mode: IdwMode = idw_mode

    @property
    def n_projections(self) -> int:
        return self.rot_deg.shape[0]

    # ------------------------------------------------------------------
    # Native (fit-frame) math
    # ------------------------------------------------------------------

    def project_coord_frame(self, points_fit: torch.Tensor) -> torch.Tensor:
        """Rigid projection of fit-frame points (N, 3) -> (T, N, 2) centered px,
        EXCLUDING global shift and locals — exactly the .aln Coord columns."""
        dtype = points_fit.dtype
        theta = torch.deg2rad(self.tilt_deg.to(dtype))[:, None]  # (T, 1)
        rho = torch.deg2rad(self.rot_deg.to(dtype))[:, None]

        cx = points_fit[None, :, 0]  # (1, N)
        cy = points_fit[None, :, 1]
        cz = points_fit[None, :, 2]

        fx = cx * torch.cos(theta) - cz * torch.sin(theta)  # (T, N)
        u0 = fx * torch.cos(rho) - cy * torch.sin(rho)
        v0 = fx * torch.sin(rho) + cy * torch.cos(rho)
        return torch.stack([u0, v0], dim=-1)  # (T, N, 2)

    def project_fit_frame(
        self, points_fit: torch.Tensor, with_local: bool = True
    ) -> torch.Tensor:
        """Full model in the fit frame: (N, 3) -> raw positions (T, N, 2),
        centered px."""
        uv0 = self.project_coord_frame(points_fit)
        out = uv0
        if with_local and self.local is not None:
            out = out + self.local.evaluate(uv0, mode=self.idw_mode).to(uv0.dtype)
        return out + self.shifts_px.to(uv0.dtype)[:, None, :]

    # ------------------------------------------------------------------
    # Canonical-frame protocol (TiltProjectionModel)
    # ------------------------------------------------------------------

    def _project(self, points_3d: torch.Tensor, with_local: bool) -> tuple[torch.Tensor, torch.Tensor]:
        points_fit = frames.canonical_volume_to_fit(
            points_3d, self.volume_dims_a, self.pixel_size_a
        )
        uv = self.project_fit_frame(points_fit, with_local=with_local)
        xy_a = frames.aretomo_image_to_canonical(uv, self.raw_size_px, self.pixel_size_a)

        img_a = self.raw_size_px.to(xy_a) * self.pixel_size_a
        valid = (
            (xy_a[..., 0] >= 0)
            & (xy_a[..., 0] <= img_a[0])
            & (xy_a[..., 1] >= 0)
            & (xy_a[..., 1] <= img_a[1])
            & torch.isfinite(xy_a).all(dim=-1)
        )
        return xy_a, valid

    def project_volume(self, points_3d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Canonical points (N, 3) A -> positions (T, N, 2) A + valid (T, N)."""
        return self._project(points_3d, with_local=True)

    def project_volume_global(self, points_3d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Global-only model (rigid; locals ignored)."""
        return self._project(points_3d, with_local=False)
