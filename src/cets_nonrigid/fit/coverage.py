"""Node-support diagnostics for scattered-particle fits.

Support is computed per tilt from TRAINING particles projected into the
target fit's 2D coordinate frame, using the SAME validity/weight mask as the
design matrix — never from canonical 3D IR points and never from held-out
particles. Gates are evaluated only over ACTIVE tilts and ACTIVE target
nodes: dark/disabled/unmatched tilts contribute ``not_evaluated``, never a
support-zero failure.

Default gate actions (CLI-overridable, recorded in fit metadata):
  * data-only rank deficiency        -> failure
  * data-only condition > 1e8        -> failure   (--max-condition)
  * node_support < 0.4               -> failure   (--min-node-support, failure
                                                   threshold ONLY)
  * node_support < 0.7               -> warning   (--warn-node-support)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_F64 = torch.float64

DEFAULT_MAX_CONDITION = 1e8
DEFAULT_MIN_NODE_SUPPORT = 0.4
DEFAULT_WARN_NODE_SUPPORT = 0.7


def node_support(
    projected_train_xy: torch.Tensor,  # (T, N, 2) target-frame coords of TRAIN points
    active_weights: torch.Tensor,  # (T, N) the fit's own weight*validity mask
    target_nodes_xy: torch.Tensor,  # (K, 2) node/patch positions, same frame
    spacing: torch.Tensor,  # (2,) node spacing per axis, same units
    *,
    node_active: torch.Tensor | None = None,  # (T, K) bool: nodes REACHABLE per tilt
    min_count: int = 1,
) -> torch.Tensor:
    """(T,) fraction of ACTIVE target nodes with >= min_count active training
    points within one node spacing (Chebyshev), per tilt.

    ``node_active`` restricts to nodes the projected volume can reach at all —
    image-corner nodes outside every projection are constrained only by
    regularization for ANY sampling and must not count against support.
    Tilts with no active weights (or no active nodes) return NaN
    (not_evaluated) — callers exclude them from gating."""
    t_count = projected_train_xy.shape[0]
    k = target_nodes_xy.shape[0]
    out = torch.full((t_count,), float("nan"), dtype=_F64)
    nodes = target_nodes_xy.to(_F64)
    sp = spacing.to(_F64).clamp_min(1e-9)
    for t in range(t_count):
        w = active_weights[t] > 0
        if int(w.sum()) == 0:
            continue  # not_evaluated for this tilt
        mask = node_active[t] if node_active is not None else torch.ones(k, dtype=torch.bool)
        if int(mask.sum()) == 0:
            continue
        pts = projected_train_xy[t, w].to(_F64)  # (B, 2)
        d = (pts[:, None, :] - nodes[None, mask, :]).abs() / sp  # (B, Ka, 2)
        near = (d.max(dim=-1).values <= 1.0).sum(dim=0)  # (Ka,)
        out[t] = float((near >= min_count).to(_F64).mean())
    return out


def nodes_reachable_by_volume(
    target_nodes_xy: torch.Tensor,  # (K, 2)
    volume_corners_xy: torch.Tensor,  # (T, 8, 2) projected volume corners
    spacing: torch.Tensor,  # (2,)
    *,
    inflate_a: float = 0.0,
) -> torch.Tensor:
    """(T, K) bool: node within one spacing of the projected volume's
    bounding box (cheap superset of the true footprint).

    ``inflate_a`` widens the box by the maximum 3D displacement magnitude: the
    corners bound only the UNDEFORMED volume; with a volume warp of node norms
    <= m every projection moves by <= m, so the deformed footprint lies within
    the undeformed box grown by m (any frame)."""
    lo = volume_corners_xy.min(dim=1).values - spacing[None, :] - float(inflate_a)
    hi = volume_corners_xy.max(dim=1).values + spacing[None, :] + float(inflate_a)
    n = target_nodes_xy.to(_F64)[None, :, :]
    return ((n >= lo[:, None, :]) & (n <= hi[:, None, :])).all(dim=-1)


@dataclass
class GateReport:
    node_support_per_tilt: torch.Tensor  # (T,) NaN = not evaluated
    min_node_support: float | None  # over evaluated tilts (None if none)
    min_data_rank: int
    max_data_condition: float
    failures: list
    warnings: list


def evaluate_gates(
    support: torch.Tensor,  # (T,) from node_support (NaN = not evaluated)
    min_data_rank: int,
    n_params: int,
    max_data_condition: float,
    *,
    max_condition: float = DEFAULT_MAX_CONDITION,
    min_node_support: float = DEFAULT_MIN_NODE_SUPPORT,
    warn_node_support: float = DEFAULT_WARN_NODE_SUPPORT,
) -> GateReport:
    failures, warns = [], []
    evaluated = support[torch.isfinite(support)]
    worst = float(evaluated.min()) if evaluated.numel() else None
    if min_data_rank < n_params:
        failures.append(
            f"data-only rank deficiency: {min_data_rank} < {n_params} parameters "
            "(the observations alone do not support the fit)"
        )
    if max_data_condition > max_condition:
        failures.append(
            f"data-only condition {max_data_condition:.3g} exceeds {max_condition:.3g}"
        )
    if worst is not None:
        if worst < min_node_support:
            failures.append(
                f"node support {worst:.2f} below the failure threshold {min_node_support}"
            )
        elif worst < warn_node_support:
            warns.append(
                f"node support {worst:.2f} below the warning threshold {warn_node_support}"
            )
    return GateReport(
        node_support_per_tilt=support,
        min_node_support=worst,
        min_data_rank=min_data_rank,
        max_data_condition=max_data_condition,
        failures=failures,
        warnings=warns,
    )


def node_support_3d(
    points_a: torch.Tensor,  # (N, 3) TRAINING sample positions, canonical A
    volume_dims_a: torch.Tensor,  # (3,)
    spatial_dims: tuple[int, int, int],  # (X, Y, Z) volume-warp nodes
    *,
    min_count: int = 1,
) -> float:
    """Fraction of the X*Y*Z spatial volume-warp nodes with >= ``min_count``
    training points within one node spacing (Chebyshev, normalized volume
    coordinates). Every spatial node is reachable (the field lives on the
    volume), so no reachability mask applies; axes with a single node are
    always satisfied. The same points are present on every active row, so
    this is one number, not a per-tilt vector."""
    x, y, z = (int(d) for d in spatial_dims)
    p = points_a.to(_F64) / volume_dims_a.to(_F64)  # (N, 3) in [0, 1]
    axes = []
    for n in (x, y, z):
        axes.append(torch.linspace(0.0, 1.0, n, dtype=_F64) if n > 1 else torch.tensor([0.5], dtype=_F64))
    nodes = torch.cartesian_prod(*axes).reshape(-1, 3)  # (K, 3)
    spacing = torch.tensor([1.0 / (n - 1) if n > 1 else float("inf") for n in (x, y, z)], dtype=_F64)
    diff = (p[None, :, :] - nodes[:, None, :]).abs() <= spacing[None, None, :] * (1 + 1e-9)
    counts = diff.all(dim=-1).sum(dim=1)  # (K,)
    return float((counts >= min_count).to(_F64).mean())
