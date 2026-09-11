"""RELION per-micrograph motion model (implements ``FrameMotionModel``).

Exact torch port of ``Micrograph::getShiftAt`` + ``ThirdOrderPolynomialModel``
(relion 5.0.1 src/micrograph_model.cpp):

  total_shift(f, x, y) = poly(z_f, xn, yn) + global[f]        (:360-406)
  z_f = f_1indexed - rlnMicrographStartFrame                   (:392-393)
  xn  = x_px / width - 0.5,  yn = y_px / height - 0.5          (micrograph_model.h:49)
  poly basis {1, x, x^2, y, y^2, xy} x {z, z^2, z^3},
  idx = 3*term + (power-1), X coeffs 0-17, Y coeffs 18-35, no
  constant term (:32-51).

Sentinel semantics (NOT_OBSERVED = -9999, :29): a frame whose global shift is
the sentinel maps through the backward-fallback rule — the nearest *preceding*
observed global shift (zero before the first observed frame), with the local
polynomial NOT evaluated (:370-388) — and is reported invalid so fitting
excludes it.  Upstream tests only X (twice, a bug at :370); an X **or** Y
sentinel marks the frame invalid here.

The shift is the correction (corrected -> raw, evaluated at the corrected
coordinate): ``raw(x) = x - shift(x)``, matching the frozen
``FrameMotionModel`` contract — no sign flip (motioncorr_runner.cpp:2057,
micrograph_handler.cpp:543-563).

Units: unbinned original-movie pixels internally; the public contract is
canonical Angstrom via ``rlnMicrographOriginalPixelSize``.
``rlnMicrographBinning != 1`` is refused: upstream stores the polynomial
coefficients in *binned working* pixels while the globals are unbinned
(motioncorr_runner.cpp:1642), an inconsistency v1 does not model.
"""

from __future__ import annotations

import torch

from ..conventions import RELION_NOT_OBSERVED, RELION_POLY_N_COEFFS

_F64 = torch.float64


def _poly_shift(coeffs: torch.Tensor, z: torch.Tensor, xn: torch.Tensor, yn: torch.Tensor) -> torch.Tensor:
    """ThirdOrderPolynomialModel::getShiftAt (micrograph_model.cpp:32-51).

    coeffs (36,); z scalar-per-frame broadcast against xn/yn (N,); returns
    (..., 2) shifts for the frame(s).
    """
    x2, y2, xy = xn * xn, yn * yn, xn * yn
    z2 = z * z
    z3 = z2 * z
    terms = torch.stack([torch.ones_like(xn), xn, x2, yn, y2, xy], dim=-1)  # (N, 6)
    zpow = torch.stack([z, z2, z3], dim=-1)  # (..., 3)
    cx = coeffs[:18].reshape(6, 3)
    cy = coeffs[18:36].reshape(6, 3)
    # shift = sum_term sum_power c[term, power] * term * z^power
    sx = torch.einsum("...p,tp,nt->...n", zpow, cx, terms)
    sy = torch.einsum("...p,tp,nt->...n", zpow, cy, terms)
    return torch.stack([sx, sy], dim=-1)


class RelionMicrographMotionModel:
    """Global per-frame shifts + optional third-order polynomial local model."""

    def __init__(
        self,
        *,
        global_shifts_px: torch.Tensor,  # (F, 2) unbinned movie px; NOT_OBSERVED = -9999
        poly_coeffs: torch.Tensor | None = None,  # (36,) RELION order, None = model version 0
        image_size_px: tuple[int, int],  # rlnImageSizeX/Y (unbinned)
        pixel_size_a: float,  # rlnMicrographOriginalPixelSize
        start_frame: int = 1,  # rlnMicrographStartFrame, 1-indexed
        binning: float = 1.0,  # rlnMicrographBinning
    ) -> None:
        if abs(binning - 1.0) > 1e-6:
            raise ValueError(
                f"rlnMicrographBinning = {binning}: unsupported — upstream stores polynomial "
                "coefficients in binned working px but globals in unbinned px "
                "(motioncorr_runner.cpp:1642)"
            )
        g = torch.as_tensor(global_shifts_px, dtype=_F64)
        if g.ndim != 2 or g.shape[1] != 2:
            raise ValueError(f"global_shifts_px has shape {tuple(g.shape)}, expected (F, 2)")
        if poly_coeffs is not None:
            poly_coeffs = torch.as_tensor(poly_coeffs, dtype=_F64)
            if poly_coeffs.shape != (RELION_POLY_N_COEFFS,):
                raise ValueError(
                    f"poly_coeffs has shape {tuple(poly_coeffs.shape)}, expected ({RELION_POLY_N_COEFFS},)"
                )
            if not torch.isfinite(poly_coeffs).all():
                raise ValueError("poly_coeffs contain non-finite values")
        self.poly_coeffs = poly_coeffs
        self.image_size_px = (int(image_size_px[0]), int(image_size_px[1]))
        self.pixel_size_a = float(pixel_size_a)
        self.start_frame = int(start_frame)

        # Backward-fallback (micrograph_model.cpp:370-388): a sentinel frame
        # maps through the nearest preceding observed shift (zero before the
        # first observed frame); either-axis sentinel marks the frame invalid.
        sentinel = (g[:, 0] == RELION_NOT_OBSERVED) | (g[:, 1] == RELION_NOT_OBSERVED)
        mapped = torch.zeros_like(g)
        last = torch.zeros(2, dtype=_F64)
        have_last = False
        for f in range(g.shape[0]):
            if sentinel[f]:
                mapped[f] = last if have_last else torch.zeros(2, dtype=_F64)
            else:
                mapped[f] = g[f]
                last = g[f]
                have_last = True
        self.global_shifts_px = g
        self._mapped_global_px = mapped
        self._frame_valid = ~sentinel

    @property
    def n_projections(self) -> int:
        return self.global_shifts_px.shape[0]

    @property
    def frame_valid(self) -> torch.Tensor:
        """(F,) bool — False on sentinel frames."""
        return self._frame_valid

    def _shift_px(self, points_px: torch.Tensor, use_local: bool) -> torch.Tensor:
        """(N, 2) corrected px -> (F, N, 2) total shift in px."""
        f_count = self.n_projections
        n = points_px.shape[0]
        shift = self._mapped_global_px[:, None, :].expand(f_count, n, 2).clone()
        if use_local and self.poly_coeffs is not None:
            xn = points_px[:, 0] / self.image_size_px[0] - 0.5
            yn = points_px[:, 1] / self.image_size_px[1] - 0.5
            frames_1 = torch.arange(1, f_count + 1, dtype=_F64)
            z = frames_1 - float(self.start_frame)  # micrograph_model.cpp:392-393
            local = _poly_shift(self.poly_coeffs, z, xn, yn)  # (F, N, 2)
            # The local polynomial is NOT evaluated on sentinel frames (:370-388).
            shift = shift + local * self._frame_valid[:, None, None].to(_F64)
        return shift

    def _map(self, points_2d: torch.Tensor, use_local: bool) -> tuple[torch.Tensor, torch.Tensor]:
        pts_px = points_2d.to(_F64) / self.pixel_size_a
        shift = self._shift_px(pts_px, use_local)
        raw_a = (pts_px[None, :, :] - shift) * self.pixel_size_a
        valid = self._frame_valid[:, None].expand(self.n_projections, pts_px.shape[0])
        valid = valid & torch.isfinite(raw_a).all(dim=-1)
        return raw_a, valid

    def map_image(self, points_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._map(points_2d, use_local=True)

    def map_image_global(self, points_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._map(points_2d, use_local=False)
