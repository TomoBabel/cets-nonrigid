"""Float64 linear least-squares machinery for grid fits.

Design matrices are built through warpylib's channel-batched spline operator
(identity-channel trick), so the basis is EXACTLY the einspline interpolating
B-spline that Warp evaluates — including extrapolation outside [0, 1] — while
the solve runs in float64 on CPU.

Solver: ``torch.linalg.lstsq(driver="gelsd")`` (CPU LAPACK, rank-revealing)
with an explicit SVD-based fallback. Regularization uses spatial DIFFERENCE
penalties (curvature), not an identity ridge — shrinking deformation values
toward zero is not the desired prior; smoothing them is.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from warpylib.cubic_grid import InterpolatingBSplineOperator2D

_OP2D = InterpolatingBSplineOperator2D()


def bspline2d_design_matrix(
    grid_xy: tuple[int, int],
    coords_norm_xy: torch.Tensor,
) -> torch.Tensor:
    """Basis matrix A (B, Gx*Gy) of the 2D interpolating einspline.

    grid_xy: (Gx, Gy) control-point counts.
    coords_norm_xy: (B, 2) normalized (x, y) coordinates; values outside
        [0, 1] extrapolate exactly like CubicGrid.
    Column c corresponds to CubicGrid flat index c = y * Gx + x within one
    temporal slice.
    """
    gx, gy = grid_xy
    n = gx * gy
    eye = torch.eye(n, dtype=torch.float64).reshape(n, gy, gx)  # (C, Y, X)
    coords = coords_norm_xy.to(torch.float64)
    return _OP2D(eye, coords[:, [1, 0]])  # operator wants (y, x)


def second_difference_penalty_2d(grid_xy: tuple[int, int]) -> torch.Tensor:
    """Rows of second differences along x and y over the (Gx, Gy) node grid,
    columns in CubicGrid flat order (y * Gx + x). Empty (0, N) if no axis has
    >= 3 nodes."""
    gx, gy = grid_xy
    n = gx * gy
    rows = []
    for y in range(gy):
        for x in range(1, gx - 1):
            r = torch.zeros(n, dtype=torch.float64)
            r[y * gx + x - 1] = 1.0
            r[y * gx + x] = -2.0
            r[y * gx + x + 1] = 1.0
            rows.append(r)
    for x in range(gx):
        for y in range(1, gy - 1):
            r = torch.zeros(n, dtype=torch.float64)
            r[(y - 1) * gx + x] = 1.0
            r[y * gx + x] = -2.0
            r[(y + 1) * gx + x] = 1.0
            rows.append(r)
    if not rows:
        return torch.zeros(0, n, dtype=torch.float64)
    return torch.stack(rows)


@dataclass
class LinearSolveResult:
    x: torch.Tensor  # (N_params,) or (N_params, K)
    rank: int  # of the regularization-AUGMENTED matrix (solver reporting)
    condition_estimate: float  # augmented
    # of the weighted DATA block alone — gates use these (regularization can
    # conceal unsupported parameters):
    data_rank: int
    data_condition: float
    regularization_norm: float
    effective_samples: float
    used_fallback: bool


def solve_weighted(
    a: torch.Tensor,  # (B, N)
    b: torch.Tensor,  # (B,) or (B, K)
    weights: torch.Tensor,  # (B,)
    penalty: torch.Tensor | None = None,  # (R, N)
    lam: float = 1e-3,
    *,
    n_rows: int | None = None,
) -> LinearSolveResult:
    """Weighted LSQ with optional difference penalty, via augmented gelsd.

    ``lam`` is relative: the penalty block is scaled by
    ``lam * sqrt(mean weighted row norm^2)`` so its strength is comparable
    across problems.

    ``n_rows`` is the number of OBSERVATION rows of the original problem. Pass
    it when ``a`` is a QR-compressed data block (``StreamingQR.r_data``, K x N
    instead of B x N): it enters the lambda scaling (mean over the original
    rows) and every dimension-based rank tolerance (``max(n_rows, N)``), so
    compression changes neither the regularization strength nor the rank
    decisions of the equivalent dense solve. Penalty rows are never counted
    as observations.
    """
    a = a.to(torch.float64)
    b = b.to(torch.float64)
    if b.ndim == 1:
        b = b[:, None]
    w = weights.to(torch.float64).clamp_min(0.0)
    sw = w.sqrt()[:, None]
    n_obs = int(n_rows) if n_rows is not None else a.shape[0]
    if n_obs <= 0:
        raise ValueError("n_rows must be positive")

    aw = a * sw
    bw = b * sw

    if penalty is not None and penalty.shape[0] > 0 and lam > 0:
        scale = lam * (aw.pow(2).sum() / n_obs).sqrt()
        aug_a = torch.cat([aw, penalty.to(torch.float64) * scale], dim=0)
        aug_b = torch.cat([bw, torch.zeros(penalty.shape[0], b.shape[1], dtype=torch.float64)], dim=0)
    else:
        aug_a = aw
        aug_b = bw

    used_fallback = False
    try:
        rcond = float(max(n_obs + (aug_a.shape[0] - aw.shape[0]), aug_a.shape[1]) * torch.finfo(torch.float64).eps)
        sol = torch.linalg.lstsq(aug_a, aug_b, rcond=rcond, driver="gelsd")
        x = sol.solution
        rank = int(sol.rank)
    except Exception:  # noqa: BLE001 - any lstsq failure routes to the SVD fallback
        # Explicit SVD-based fallback (svd is not an lstsq driver).
        used_fallback = True
        u, s, vh = torch.linalg.svd(aug_a, full_matrices=False)
        tol = s.max() * max(n_obs + (aug_a.shape[0] - aw.shape[0]), aug_a.shape[1]) * torch.finfo(torch.float64).eps
        rank = int((s > tol).sum())
        s_inv = torch.where(s > tol, 1.0 / s, torch.zeros_like(s))
        x = vh.mH @ (s_inv[:, None] * (u.mH @ aug_b))

    svals = torch.linalg.svdvals(aug_a)
    cond = float(svals.max() / svals.min()) if float(svals.min()) > 0 else float("inf")
    # data-only diagnostics (pre-augmentation): what the observations alone support
    dvals = torch.linalg.svdvals(aw)
    dtol = (dvals.max() * max(n_obs, aw.shape[1]) * torch.finfo(torch.float64).eps) if dvals.numel() else 0.0
    data_rank = int((dvals > dtol).sum()) if dvals.numel() else 0
    data_cond = (
        float(dvals.max() / dvals[dvals > dtol].min()) if data_rank > 0 else float("inf")
    )
    reg_norm = 0.0
    if penalty is not None and penalty.shape[0] > 0 and lam > 0:
        reg_norm = float((penalty.to(torch.float64) @ x).norm())

    return LinearSolveResult(
        x=x.squeeze(-1) if x.shape[-1] == 1 else x,
        rank=rank,
        condition_estimate=cond,
        data_rank=data_rank,
        data_condition=data_cond,
        regularization_norm=reg_norm,
        effective_samples=float(w.sum() / w.max().clamp_min(1e-30)),
        used_fallback=used_fallback,
    )


# ---------------------------------------------------------------------------
# Quadrilinear (Warp GridVolumeWarp / LinearGrid4D) machinery
# ---------------------------------------------------------------------------


def _unit_grids(dims: tuple[int, int, int, int]):
    from warpylib import LinearGrid4D

    k = int(dims[0] * dims[1] * dims[2] * dims[3])
    eye = torch.eye(k, dtype=torch.float64)
    return [LinearGrid4D(tuple(int(d) for d in dims), eye[j]) for j in range(k)]


def lineargrid4d_basis(dims: tuple[int, int, int, int], coords4: torch.Tensor) -> torch.Tensor:
    """Basis matrix A (B, K) of Warp's quadrilinear ``LinearGrid4D``.

    Column j is the grid with unit value at flat node j (layout
    ``((w*Z + z)*Y + y)*X + x``, x fastest) evaluated at ``coords4`` (B, 4) in
    the nominal [0, 1]^4 domain — built by evaluating warpylib's golden-pinned
    operator itself, so truncation toward zero, clamping, the negative-fraction
    extrapolation below 0 and the collapse above 1 are reproduced without a
    second transcription. Float64.
    """
    coords = coords4.to(torch.float64)
    cols = [g.get_interpolated(coords) for g in _unit_grids(dims)]
    return torch.stack(cols, dim=-1)


def temporal_weights(n_dose_nodes: int, dose_coords: torch.Tensor) -> torch.Tensor:
    """(T, L) one-dimensional temporal interpolation weights of ``LinearGrid4D``
    along its dose axis, with Warp's boundary rules (evaluated through the
    operator itself on a 1x1x1xL grid; NOT a bare hat function — a float32
    coordinate can overshoot 1)."""
    c = dose_coords.to(torch.float64).reshape(-1)
    coords4 = torch.stack([torch.zeros_like(c), torch.zeros_like(c), torch.zeros_like(c), c], dim=-1)
    return lineargrid4d_basis((1, 1, 1, int(n_dose_nodes)), coords4)  # (T, L)


#: A dose slice is numerically supported when some ACTIVE row gives it at least
#: this much one-dimensional temporal weight. Float32 leakage of a row that sits
#: on a neighbouring node is ~1e-7; a genuinely interpolating row gives >= 1e-3
#: to both neighbours for every realistic dose schedule.
TEMPORAL_SUPPORT_TAU = 1e-3


def temporal_support(
    n_dose_nodes: int,
    dose_coords_active: torch.Tensor,
    tau: float = TEMPORAL_SUPPORT_TAU,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(supported (L,) bool, discarded (L,) max 1-D weight of each UNsupported slice).

    Support is decided on the one-dimensional temporal weights only — the 4D
    coefficients are these times spatial weights (down to 1/8 at a cell centre)
    and would misclassify a supported slice.
    """
    if dose_coords_active.numel() == 0:
        raise ValueError("no active rows")
    w = temporal_weights(n_dose_nodes, dose_coords_active).abs()  # (T, L)
    wmax = w.max(dim=0).values
    supported = wmax >= tau
    discarded = torch.where(supported, torch.zeros_like(wmax), wmax)
    return supported, discarded


def extension_operator(dims: tuple[int, int, int, int], supported_slices: torch.Tensor) -> torch.Tensor:
    """E (K, K_S): full-grid node values from the SUPPORTED-slice node values.

    Every node of an unsupported dose slice takes the value of the same spatial
    node on the nearest supported slice (ties -> lower index): a constant
    extension along the dose axis, applied BEFORE solving (the penalty is
    composed as ``P @ E``), so no post-hoc fill exists. Rows have a single 1.
    """
    x, y, z, dose_nodes = (int(d) for d in dims)
    sup = supported_slices.to(torch.bool)
    if sup.shape != (dose_nodes,):
        raise ValueError(f"supported_slices has shape {tuple(sup.shape)}, expected ({dose_nodes},)")
    if not bool(sup.any()):
        raise ValueError("no supported dose slice")
    sup_idx = torch.nonzero(sup).reshape(-1)
    nxyz = x * y * z
    # column index of node (spatial s, slice w) in the reduced parameter vector
    col_of_slice = torch.full((dose_nodes,), -1, dtype=torch.long)
    col_of_slice[sup_idx] = torch.arange(sup_idx.numel())
    e = torch.zeros(nxyz * dose_nodes, nxyz * sup_idx.numel(), dtype=torch.float64)
    for w in range(dose_nodes):
        if bool(sup[w]):
            src = w
        else:
            dist = (sup_idx - w).abs()
            src = int(sup_idx[int(torch.argmin(dist))])  # argmin returns the first minimum -> lower index
        cs = int(col_of_slice[src])
        rows = torch.arange(nxyz) + w * nxyz
        cols = torch.arange(nxyz) + cs * nxyz
        e[rows, cols] = 1.0
    return e


def second_difference_penalty_4d(dims: tuple[int, int, int, int]) -> torch.Tensor:
    """Second differences along every axis with >= 3 nodes, over the flat
    LinearGrid4D layout ``((w*Z + z)*Y + y)*X + x``. (R, K) float64; (0, K)
    when no axis qualifies."""
    x, y, z, dose_nodes = (int(d) for d in dims)
    k = x * y * z * dose_nodes

    def flat(ix, iy, iz, iw):
        return ((iw * z + iz) * y + iy) * x + ix

    rows = []
    for iw in range(dose_nodes):
        for iz in range(z):
            for iy in range(y):
                for ix in range(x):
                    for axis, n, idx in ((0, x, ix), (1, y, iy), (2, z, iz), (3, dose_nodes, iw)):
                        if n < 3 or idx == 0 or idx == n - 1:
                            continue
                        r = torch.zeros(k, dtype=torch.float64)
                        lo = [ix, iy, iz, iw]
                        hi = [ix, iy, iz, iw]
                        lo[axis] -= 1
                        hi[axis] += 1
                        r[flat(*lo)] = 1.0
                        r[flat(ix, iy, iz, iw)] = -2.0
                        r[flat(*hi)] = 1.0
                        rows.append(r)
    if not rows:
        return torch.zeros(0, k, dtype=torch.float64)
    return torch.stack(rows)


class StreamingQR:
    """Tall-skinny QR accumulator: ``R = qr([R; block]).R`` over blocks of
    ``[sqrt(w) A | sqrt(w) b]``.

    Keeps only a (K + n_rhs) x (K + n_rhs) triangle, never the (rows x K) matrix,
    and — unlike a Gram matrix — preserves the singular values of ``sqrt(W) A``
    to working precision (the Gram route squares the condition number, which at
    the 1e8 failure threshold sits at float64's limit). ``r_data`` (K x K) and
    ``qtb`` (K x n_rhs) form the reduced least-squares problem
    ``min ||r_data x - qtb||`` equivalent to the weighted original.
    """

    def __init__(self, n_cols: int, n_rhs: int):
        self.n_cols = int(n_cols)
        self.n_rhs = int(n_rhs)
        self._r = torch.zeros(0, self.n_cols + self.n_rhs, dtype=torch.float64)
        self.n_rows = 0

    def add_block(self, a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor | None = None) -> None:
        a = a.to(torch.float64)
        b = b.to(torch.float64)
        if b.ndim == 1:
            b = b[:, None]
        if a.shape[1] != self.n_cols or b.shape != (a.shape[0], self.n_rhs):
            raise ValueError(f"block shapes {tuple(a.shape)} / {tuple(b.shape)} do not match ({self.n_cols}, {self.n_rhs})")
        if a.shape[0] == 0:
            return
        if weights is not None:
            sw = weights.to(torch.float64).clamp_min(0.0).sqrt()[:, None]
            a, b = a * sw, b * sw
        stacked = torch.cat([self._r, torch.cat([a, b], dim=1)], dim=0)
        self._r = torch.linalg.qr(stacked, mode="r").R
        self.n_rows += int(a.shape[0])

    @property
    def r_data(self) -> torch.Tensor:
        """(K, K) upper-triangular factor of sqrt(W) A (zero-padded if fewer rows)."""
        r = self._r[:, : self.n_cols]
        if r.shape[0] < self.n_cols:
            r = torch.cat([r, torch.zeros(self.n_cols - r.shape[0], self.n_cols, dtype=torch.float64)], dim=0)
        return r[: self.n_cols]

    @property
    def qtb(self) -> torch.Tensor:
        """(K, n_rhs) — Q^T (sqrt(W) b) restricted to the range of A."""
        q = self._r[:, self.n_cols :]
        if q.shape[0] < self.n_cols:
            q = torch.cat([q, torch.zeros(self.n_cols - q.shape[0], self.n_rhs, dtype=torch.float64)], dim=0)
        return q[: self.n_cols]

    def singular_values(self) -> torch.Tensor:
        return torch.linalg.svdvals(self.r_data)
