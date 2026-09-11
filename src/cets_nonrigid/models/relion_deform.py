"""RELION 2D image deformations — exact torch ports (read-side).

Applied ADDITIVELY to the projected position, AFTER the rigid projection, in
tilt-image pixels (Tomogram::projectPoint, tomogram.cpp:119-131). Three
models (relion 5.0.1, src/jaz/tomography/motion/[modular_alignment/]):

* ``linear``  — 3 coefficients (axx, axy, ayy); ``r = pl - 0.5*imageSize``
  (FLOAT centre, linear_2D_deformation_model.cpp:11);
  ``def = (axx*rx, axy*rx + ayy*ry)``.
* ``spline``  — 8*gx*gy coefficients: per node and per output dim a
  ``{value, slope_x, slope_y, twist}`` quadruple laid out as
  RawImage<DataPoint>(gx, gy, 2); bicubic Hermite with grid spacing
  ``imageSize/(grid-1)`` and clamped cell coordinates
  (spline_2D_deformation_model.{h,cpp}).
* ``Fourier`` — 4*|freqs| coefficients (complex per frequency and output
  dim, laid out [dim][freq] as computeShift reads them);
  ``def[dim] = sum Re*cos(k.pl) + Im*sin(k.pl)`` with the half-plane
  frequency enumeration skipping DC
  (Fourier_2D_deformation_model.{h,cpp}).

Coefficient counts are validated exactly on construction.
"""

from __future__ import annotations

import torch

_F64 = torch.float64


def fourier_frequencies(grid_size: tuple, image_size_px: tuple) -> torch.Tensor:
    """(K, 2) spatial frequencies — verbatim enumeration
    (Fourier_2D_deformation_model.cpp:15-38)."""
    gx, gy = int(grid_size[0]), int(grid_size[1])
    w, h = float(image_size_px[0]), float(image_size_px[1])
    freqs = []
    for y in range(gy):
        if y < gy // 2:
            for x in range(1 if y == 0 else 0, gx // 2 + 1):
                freqs.append((x * torch.pi / w, y * torch.pi / h))
        else:
            for x in range(1, gx // 2 + 1):
                freqs.append((x * torch.pi / w, (y - gy) * torch.pi / h))
    return torch.tensor(freqs, dtype=_F64)


class Deformation2D:
    """One tilt's deformation: ``apply(pl_px (N, 2)) -> shifted pl_px``."""

    def __init__(self, kind: str, grid_size: tuple, image_size_px: tuple, coeffs: torch.Tensor):
        kind_l = str(kind)
        if kind_l not in ("linear", "spline", "Fourier"):
            raise ValueError(f"unknown rlnTomoDeformationType {kind!r}")
        self.kind = kind_l
        self.grid_size = (int(grid_size[0]), int(grid_size[1]))
        self.image_size_px = (float(image_size_px[0]), float(image_size_px[1]))
        c = torch.as_tensor(coeffs, dtype=_F64).flatten()
        if not torch.isfinite(c).all():
            raise ValueError("non-finite deformation coefficients")

        gx, gy = self.grid_size
        if kind_l == "linear":
            expected = 3
        elif kind_l == "spline":
            expected = 8 * gx * gy
        else:
            self._freqs = fourier_frequencies(self.grid_size, self.image_size_px)
            expected = 4 * self._freqs.shape[0]
        if c.numel() != expected:
            raise ValueError(
                f"{kind_l} deformation needs exactly {expected} coefficients for grid "
                f"{gx}x{gy}, got {c.numel()}"
            )
        self.coeffs = c

    def shift(self, pl_px: torch.Tensor) -> torch.Tensor:
        """(N, 2) tilt-image px -> (N, 2) additive deformation shift (px)."""
        pl = pl_px.to(_F64)
        if self.kind == "linear":
            axx, axy, ayy = self.coeffs.tolist()
            centre = torch.tensor(
                [0.5 * self.image_size_px[0], 0.5 * self.image_size_px[1]], dtype=_F64
            )
            r = pl - centre
            return torch.stack([axx * r[:, 0], axy * r[:, 0] + ayy * r[:, 1]], dim=-1)
        if self.kind == "Fourier":
            k = self._freqs  # (K, 2)
            n_f = k.shape[0]
            p = self.coeffs.reshape(2, n_f, 2)  # [dim][freq][(re, im)]
            t = pl @ k.T  # (N, K)
            ct, st = torch.cos(t), torch.sin(t)
            out = torch.empty(pl.shape[0], 2, dtype=_F64)
            for dim in range(2):
                out[:, dim] = ct @ p[dim, :, 0] + st @ p[dim, :, 1]
            return out
        return self._spline_shift(pl)

    def _spline_shift(self, pl: torch.Tensor) -> torch.Tensor:
        gx, gy = self.grid_size
        spacing = torch.tensor(
            [self.image_size_px[0] / (gx - 1), self.image_size_px[1] / (gy - 1)], dtype=_F64
        )
        eps = 1e-10
        grid = pl / spacing
        lim = torch.tensor([gx - 1 - eps, gy - 1 - eps], dtype=_F64)
        grid = torch.clamp(grid, min=torch.zeros(2, dtype=_F64), max=lim)
        cell = grid.floor().to(torch.long)  # (N, 2)
        frc = grid - cell.to(_F64)
        x, y = frc[:, 0], frc[:, 1]
        x2, x3 = x * x, x * x * x
        y2, y3 = y * y, y * y * y
        vx = torch.stack(
            [1.0 - 3.0 * x2 + 2.0 * x3, 3.0 * x2 - 2.0 * x3, x - 2.0 * x2 + x3, -x2 + x3], dim=-1
        )  # (N, 4)
        vy = torch.stack(
            [1.0 - 3.0 * y2 + 2.0 * y3, 3.0 * y2 - 2.0 * y3, y - 2.0 * y2 + y3, -y2 + y3], dim=-1
        )

        # RawImage<DataPoint>(gx, gy, 2): flat index = ((dim*gy + yy)*gx + xx)*4
        data = self.coeffs.reshape(2, gy, gx, 4)  # [dim][y][x][value, slope_x, slope_y, twist]
        out = torch.empty(pl.shape[0], 2, dtype=_F64)
        cx, cy = cell[:, 0], cell[:, 1]
        for dim in range(2):
            d00 = data[dim, cy, cx]  # (N, 4)
            d01 = data[dim, cy + 1, cx]
            d10 = data[dim, cy, cx + 1]
            d11 = data[dim, cy + 1, cx + 1]
            # F rows (d4Matrix row-major, spline_2D_deformation_model.h):
            f = torch.stack(
                [
                    torch.stack([d00[:, 0], d01[:, 0], d00[:, 2], d01[:, 2]], dim=-1),
                    torch.stack([d10[:, 0], d11[:, 0], d10[:, 2], d11[:, 2]], dim=-1),
                    torch.stack([d00[:, 1], d01[:, 1], d00[:, 3], d01[:, 3]], dim=-1),
                    torch.stack([d10[:, 1], d11[:, 1], d10[:, 3], d11[:, 3]], dim=-1),
                ],
                dim=1,
            )  # (N, 4, 4)
            out[:, dim] = torch.einsum("ni,nij,nj->n", vx, f, vy)
        return out

    def apply(self, pl_px: torch.Tensor) -> torch.Tensor:
        return pl_px.to(_F64) + self.shift(pl_px)


def deformation_field(
    kind: str,
    grid_size: tuple,
    image_size_px: tuple,
    coeffs_per_tilt: list,  # T entries, each a coefficient vector
):
    """-> callable (T, N, 2) px -> (T, N, 2) px for RelionParticleSetModel."""
    defs = [Deformation2D(kind, grid_size, image_size_px, c) for c in coeffs_per_tilt]

    def apply(xy_px: torch.Tensor) -> torch.Tensor:
        out = xy_px.to(_F64).clone()
        for t, d in enumerate(defs):
            out[t] = d.apply(xy_px[t])
        return out

    return apply
