"""IR sample-point generation.

Training points: regular grids, borders inclusive. Held-out points: scrambled
Sobol sequences — non-nested with any regular grid by construction, so
held-out metrics detect overfitting to the training nodes.
"""

from __future__ import annotations

import torch


def volume_grid(
    volume_dims_a: torch.Tensor,
    shape: tuple[int, int, int] = (15, 15, 5),
) -> torch.Tensor:
    """Regular grid of 3D points (N, 3) float64, Angstrom, corner origin,
    borders inclusive (linspace(0, V, n) per axis)."""
    axes = [
        torch.linspace(0.0, float(volume_dims_a[i]), shape[i], dtype=torch.float64)
        for i in range(3)
    ]
    gx, gy, gz = torch.meshgrid(*axes, indexing="ij")
    return torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)


def image_grid(
    image_dims_a: torch.Tensor,
    shape: tuple[int, int] = (11, 11),
) -> torch.Tensor:
    """Regular grid of 2D points (N, 2) float64, Angstrom, corner origin."""
    axes = [
        torch.linspace(0.0, float(image_dims_a[i]), shape[i], dtype=torch.float64)
        for i in range(2)
    ]
    gx, gy = torch.meshgrid(*axes, indexing="ij")
    return torch.stack([gx, gy], dim=-1).reshape(-1, 2)


def heldout_points(
    dims_a: torch.Tensor,
    n: int,
    seed: int,
) -> torch.Tensor:
    """Scrambled Sobol points (n, D) float64 spanning [0, dims] per axis."""
    d = int(dims_a.numel())
    engine = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=seed)
    unit = engine.draw(n).to(torch.float64)
    return unit * dims_a.to(torch.float64)


def boundary_ramp_weights(
    positions_a: torch.Tensor,
    image_dims_a: torch.Tensor,
    margin_frac: float = 0.05,
) -> torch.Tensor:
    """Cosine ramp weights for projected positions (..., 2) -> (...) float32.

    1 inside the image; cos^2 ramp to 0 over a ``margin_frac``-of-width band
    OUTSIDE each edge; 0 beyond. Weights are fit weights only — validity is a
    separate boolean mask (never encode validity through weights).
    """
    dims = image_dims_a.to(positions_a)
    w = torch.ones(positions_a.shape[:-1], dtype=positions_a.dtype, device=positions_a.device)
    for axis in range(2):
        x = positions_a[..., axis]
        span = dims[axis]
        margin = margin_frac * span
        outside = torch.clamp(torch.maximum(-x, x - span), min=0.0)
        t = torch.clamp(outside / margin, max=1.0)
        ramp = torch.where(t >= 1.0, torch.zeros_like(t), torch.cos(t * torch.pi / 2) ** 2)
        w = w * ramp
    return w.to(torch.float32)
