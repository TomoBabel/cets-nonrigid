"""AreTomo3 MotionCor frame-motion model (independent torch implementation).

Replicates ``MotionCor/Correct/GCorrectPatchShift.cu::mGCorrect3D``:

    r_p  = sqrt(((x - cx_p)/Nx)^2 + ((y - cy_p)/Ny)^2)   # LINEAR r
    w_p  = exp(-100 * r_p)   for r_p <= 0.5 and patch not bad, else 0
    S    = sum(w_p s_{f,p}) / sum(w_p)      (0 if no contributing patch)
    raw(x, f) = x - S(x, f) - glob_f

Shifts are CORRECTIONS subtracted from the corrected-frame coordinate — the
same sign convention as Warp's movie chain (the sign flip in this codebase
exists only between tilt-series locals and motion shifts). This is a
deliberately separate module from the tilt-series IDW: different kernel
(linear r vs r^2), hard 0.5 cutoff vs none, per-frame vs per-tilt semantics.

Composition order (S evaluated at the corrected coordinate, global applied
after) is PROVISIONAL until the M6 correction-kernel contract test pins it
(docs/mcaln_format.md).
"""

from __future__ import annotations

import math
from typing import Literal

import torch

_EXPONENT = -100.0
_CUTOFF = 0.5

IdwMode = Literal["stable", "compat"]


class AretomoMotionModel:
    """Frame-series motion model in the alignment-image frame.

    global_shifts_px: (F, 2) per-frame corrections (aligned-frame indexing).
    patch_centers_px: (P, 2) corner-origin patch centers (constant over frames).
    patch_shifts_px:  (F, P, 2) local corrections.
    patch_valid:      (F, P) bool (kernel skips invalid patches).
    frame_size_px:    (Nx, Ny) alignment-image size (distance normalization).
    pixel_size_a:     A/px of the alignment image (canonical-frame interface).
    """

    def __init__(
        self,
        global_shifts_px: torch.Tensor,
        patch_centers_px: torch.Tensor,
        patch_shifts_px: torch.Tensor,
        patch_valid: torch.Tensor,
        frame_size_px: tuple[int, int],
        pixel_size_a: float,
        idw_mode: IdwMode = "stable",
    ):
        f_count = global_shifts_px.shape[0]
        p_count = patch_centers_px.shape[0]
        if global_shifts_px.shape != (f_count, 2):
            raise ValueError("global_shifts_px must be (F, 2)")
        if patch_shifts_px.shape != (f_count, p_count, 2) or patch_valid.shape != (f_count, p_count):
            raise ValueError("inconsistent patch array shapes")
        if pixel_size_a <= 0:
            raise ValueError("pixel_size_a must be positive")

        self.global_shifts_px = global_shifts_px
        self.patch_centers_px = patch_centers_px
        self.patch_shifts_px = patch_shifts_px
        self.patch_valid = patch_valid
        self.frame_size_px = torch.tensor(frame_size_px)
        self.pixel_size_a = float(pixel_size_a)
        self.idw_mode: IdwMode = idw_mode

    @property
    def n_projections(self) -> int:
        return self.global_shifts_px.shape[0]

    def local_field_px(self, xy_px: torch.Tensor, mode: IdwMode | None = None) -> torch.Tensor:
        """Evaluate the local correction field at xy_px (N, 2) -> (F, N, 2)."""
        mode = mode or self.idw_mode
        dtype = torch.float32 if mode == "compat" else (
            xy_px.dtype if xy_px.dtype.is_floating_point else torch.float64
        )
        xy = xy_px.to(dtype)
        centers = self.patch_centers_px.to(dtype)
        nx = float(self.frame_size_px[0])
        ny = float(self.frame_size_px[1])

        dx = (xy[:, None, 0] - centers[None, :, 0]) / nx  # (N, P)
        dy = (xy[:, None, 1] - centers[None, :, 1]) / ny
        r = torch.sqrt(dx * dx + dy * dy)
        in_range = r <= _CUTOFF  # hard cutoff (differs from the TS kernel!)
        e = _EXPONENT * r  # LINEAR r (differs from the TS kernel!)

        gate = in_range[None, :, :] & self.patch_valid[:, None, :]  # (F, N, P)
        if mode == "stable":
            e_masked = torch.where(gate, e[None].expand_as(gate).to(dtype), torch.full_like(gate, -math.inf, dtype=dtype))
            e_max = e_masked.max(dim=-1, keepdim=True).values
            e_max = torch.where(torch.isfinite(e_max), e_max, torch.zeros_like(e_max))
            w = torch.where(gate, torch.exp(e[None] - e_max), torch.zeros(1, dtype=dtype))
        else:
            w = torch.where(gate, torch.exp(e)[None].expand_as(gate).to(dtype), torch.zeros(1, dtype=dtype))

        num = torch.einsum("fnp,fpc->fnc", w, self.patch_shifts_px.to(dtype))
        den = w.sum(dim=-1, keepdim=True)
        any_p = gate.any(dim=-1)
        out = torch.where(any_p[..., None], num / den.clamp_min(torch.finfo(dtype).tiny), torch.zeros_like(num))
        return out

    def _map(self, points_2d_a: torch.Tensor, with_local: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if points_2d_a.shape[-1] != 2:
            raise ValueError(f"points must be (N, 2), got {tuple(points_2d_a.shape)}")
        xy_px = points_2d_a.to(torch.float64) / self.pixel_size_a

        shift = self.global_shifts_px.to(torch.float64)[:, None, :].expand(
            -1, xy_px.shape[0], -1
        ).clone()
        if with_local:
            shift = shift + self.local_field_px(xy_px).to(torch.float64)

        raw_px = xy_px[None, :, :] - shift
        raw_a = raw_px * self.pixel_size_a

        dims_a = self.frame_size_px.to(torch.float64) * self.pixel_size_a
        valid = (
            (raw_a[..., 0] >= 0)
            & (raw_a[..., 0] <= dims_a[0])
            & (raw_a[..., 1] >= 0)
            & (raw_a[..., 1] <= dims_a[1])
            & torch.isfinite(raw_a).all(dim=-1)
        )
        return raw_a, valid

    def map_image(self, points_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full model: canonical (N, 2) A -> raw positions (F, N, 2) A."""
        return self._map(points_2d, with_local=True)

    def map_image_global(self, points_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._map(points_2d, with_local=False)
