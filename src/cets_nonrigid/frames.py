"""Canonical-frame transforms.

Canonical frame (= Warp's): image positions in Angstrom, corner origin, on the
raw motion-corrected tilt image; volume points in Angstrom, corner origin.

AreTomo3 fit frame: positions in motion-corrected tilt-series pixels, centered
origin; volume Z sign inverted (``fX = Cx cos(theta) - Cz sin(theta)``).

All conversions in one place; constants from :mod:`cets_nonrigid.conventions`.
See docs/conventions.md.
"""

from __future__ import annotations

import torch

from cets_nonrigid.conventions import ARETOMO_FIT_Z_SIGN, HALF_PIXEL_OFFSET


def canonical_volume_to_fit(
    points_a: torch.Tensor,
    volume_dims_a: torch.Tensor,
    pixel_size_a: float,
) -> torch.Tensor:
    """Canonical volume points (N, 3) [A, corner] -> AreTomo fit frame (N, 3)
    [centered tilt-series px, fit-convention Z]."""
    centered = points_a - volume_dims_a.to(points_a) / 2
    out = centered / pixel_size_a
    return torch.stack(
        [out[..., 0], out[..., 1], ARETOMO_FIT_Z_SIGN * out[..., 2]], dim=-1
    )


def fit_volume_to_canonical(
    points_fit: torch.Tensor,
    volume_dims_a: torch.Tensor,
    pixel_size_a: float,
) -> torch.Tensor:
    """Inverse of :func:`canonical_volume_to_fit`."""
    unsigned = torch.stack(
        [points_fit[..., 0], points_fit[..., 1], ARETOMO_FIT_Z_SIGN * points_fit[..., 2]],
        dim=-1,
    )
    return unsigned * pixel_size_a + volume_dims_a.to(points_fit) / 2


def aretomo_image_to_canonical(
    uv_centered_px: torch.Tensor,
    raw_size_px: torch.Tensor,
    pixel_size_a: float,
) -> torch.Tensor:
    """AreTomo image positions (..., 2) [centered px] -> canonical (..., 2) [A, corner].

    ``x_canonical_px = u_centered + N/2 + HALF_PIXEL_OFFSET`` (docs/conventions.md).
    """
    return (uv_centered_px + raw_size_px.to(uv_centered_px) / 2 + HALF_PIXEL_OFFSET) * pixel_size_a


def canonical_image_to_aretomo(
    xy_a: torch.Tensor,
    raw_size_px: torch.Tensor,
    pixel_size_a: float,
) -> torch.Tensor:
    """Inverse of :func:`aretomo_image_to_canonical`."""
    return xy_a / pixel_size_a - raw_size_px.to(xy_a) / 2 - HALF_PIXEL_OFFSET


# --- RELION 5 -----------------------------------------------------------------
# RELION's tilt-image and bin-1 tomogram-voxel frames are corner-origin pixel
# coordinates at rlnTomoTiltSeriesPixelSize (the frame-series analog uses
# rlnMicrographOriginalPixelSize) — canonical is the same frame in Angstrom,
# so the transforms are pure scales. The int-vs-float centre subtleties live
# INSIDE the RELION projection matrix (models/relion_ts.py), not here.


def relion_px_to_canonical(points_px: torch.Tensor, pixel_size_a: float) -> torch.Tensor:
    """RELION corner-origin pixel coordinates (image, tomogram voxel, or movie)
    -> canonical corner-origin Angstrom."""
    return points_px * pixel_size_a


def canonical_to_relion_px(points_a: torch.Tensor, pixel_size_a: float) -> torch.Tensor:
    """Inverse of :func:`relion_px_to_canonical`."""
    return points_a / pixel_size_a
