"""Step 2 of the 3D-displacement plan: quadrilinear basis, temporal support,
extension operator, 4D penalty, streaming QR and the n_rows solver contract.

Tolerances are conditioning-aware (absolute vs sigma_max), not per-value
relative; the lam = 0 exactness statements are solver unit tests, not CLI
behaviour.
"""

from __future__ import annotations

import torch
from warpylib import LinearGrid4D

from cets_nonrigid.fit.linear import (
    TEMPORAL_SUPPORT_TAU,
    StreamingQR,
    extension_operator,
    lineargrid4d_basis,
    second_difference_penalty_4d,
    solve_weighted,
    temporal_support,
    temporal_weights,
)

EPS = torch.finfo(torch.float64).eps
F64 = torch.float64


def test_basis_reproduces_lineargrid4d_incl_boundaries_and_dim1_axes():
    gen = torch.Generator().manual_seed(1)
    for dims in ((3, 3, 2, 4), (2, 1, 3, 5), (1, 1, 1, 1), (4, 2, 1, 2)):
        k = dims[0] * dims[1] * dims[2] * dims[3]
        values = torch.randn(k, generator=gen, dtype=torch.float64) * 10
        coords = torch.rand(200, 4, generator=gen, dtype=torch.float64) * 1.4 - 0.2  # includes < 0 and > 1
        coords[:5] = 1.0  # exactly at the top (collapse)
        coords[5:10] = 0.0
        ref = LinearGrid4D(dims, values).get_interpolated(coords)
        got = lineargrid4d_basis(dims, coords) @ values
        assert (got - ref).abs().max() <= 1e-12 * max(1.0, ref.abs().max().item())


def test_temporal_weights_follow_warp_boundary_rules():
    # in-domain: hat weights summing to 1; below 0: extrapolation (weight > 1 on node 0,
    # negative on node 1); above 1 (float32 overshoot): all weight on the last node
    w = temporal_weights(5, torch.tensor([0.3, -0.1, 1.0 + 3e-7], dtype=F64))
    assert torch.allclose(w.sum(dim=1), torch.ones(3, dtype=F64))
    assert abs(w[0, 1] - 0.8) < 1e-12 and abs(w[0, 2] - 0.2) < 1e-12
    assert w[1, 0] > 1.0 and w[1, 1] < 0.0
    assert abs(w[2, 4] - 1.0) < 1e-12 and w[2, :4].abs().max() == 0.0


def _dose_coords_float32(n, disabled=()):
    dose = torch.arange(n, dtype=torch.float32) * 3
    step = torch.tensor(1.0 / float(dose.max() - dose.min()), dtype=torch.float32)
    c = ((dose - dose.min()) * step).to(torch.float64)  # exactly as warpylib
    active = torch.ones(n, dtype=torch.bool)
    for i in disabled:
        active[i] = False
    return c, active


def test_temporal_support_float32_leakage_case():
    """41 regularly spaced doses, row 2 disabled: node 2 keeps a ~1e-7 column
    from its float32-off-node neighbours, so 41 nodes are 'touched' by 40 rows
    (rank 40); numerical support on the 1-D weights removes exactly that node."""
    c, active = _dose_coords_float32(41, disabled=(2,))
    w = temporal_weights(41, c[active])
    touched = (w.abs().max(dim=0).values > 0).sum().item()
    assert touched == 41 and torch.linalg.matrix_rank(w).item() == 40
    supported, discarded = temporal_support(41, c[active])
    assert supported.sum().item() == 40 and not bool(supported[2])
    leak = discarded[2].item()
    assert 0 < leak < 1e-6, f"leakage {leak:.3e}"
    assert torch.linalg.matrix_rank(w[:, supported]).item() == 40


def test_temporal_support_uses_1d_weights_not_4d_coefficients():
    """A row with 1-D temporal weight 4e-3 on a slice supports it even though at
    a spatial cell centre every 4-D coefficient is 4e-3 / 8 = 5e-4 < tau."""
    L = 5
    c = torch.tensor([0.0, 0.25 + 0.001, 0.5, 0.75, 1.0], dtype=F64)  # row 1 sits 0.004 (scaled) past node 1
    scaled = c * (L - 1)
    w = temporal_weights(L, c)
    assert abs(w[1, 2] - (scaled[1] - 1)) < 1e-12 and 0.003 < w[1, 2] < 0.005
    # only row 1 touches node 2? No — row 2 sits on node 2. Use a subset: rows {0, 1, 3, 4}
    sub = torch.tensor([0, 1, 3, 4])
    supported, _ = temporal_support(L, c[sub])
    assert bool(supported[2]), "1-D weight 4e-3 >= tau must count as support"
    # the 4-D coefficients at a cell centre would have been below tau
    coords4 = torch.tensor([[0.5 / 2, 0.5 / 2, 0.5, float(c[1])]], dtype=F64)  # cell centre of a 3x3x2 spatial grid
    a = lineargrid4d_basis((3, 3, 2, L), coords4)
    slice2 = a.reshape(1, L, 2, 3, 3)[0, 2]
    assert slice2.abs().max() < TEMPORAL_SUPPORT_TAU


def test_extension_operator_nearest_supported_slice():
    dims = (2, 1, 1, 6)
    sup = torch.tensor([False, True, False, False, True, False])
    e = extension_operator(dims, sup)
    assert e.shape == (12, 4) and torch.equal(e.sum(dim=1), torch.ones(12))
    v = torch.tensor([1.0, 2.0, 10.0, 20.0], dtype=F64)  # slice1 -> (1,2), slice4 -> (10,20)
    full = (e @ v).reshape(6, 2)
    assert torch.equal(full[0], full[1]) and torch.equal(full[1], torch.tensor([1.0, 2.0], dtype=F64))
    assert torch.equal(full[2], torch.tensor([1.0, 2.0], dtype=F64))  # nearer to slice 1
    assert torch.equal(full[3], torch.tensor([10.0, 20.0], dtype=F64))  # nearer to slice 4
    assert torch.equal(full[5], torch.tensor([10.0, 20.0], dtype=F64))
    # equidistant (slice 2 of a 1..3 pair) -> lower index
    e2 = extension_operator((1, 1, 1, 3), torch.tensor([True, False, True]))
    assert torch.equal((e2 @ torch.tensor([5.0, 9.0], dtype=F64)), torch.tensor([5.0, 5.0, 9.0], dtype=F64))


def test_penalty_row_count_and_composition_with_extension():
    dims = (3, 3, 2, 4)
    p = second_difference_penalty_4d(dims)
    x, y, z, dose_nodes = dims
    expect = (x - 2) * y * z * dose_nodes + x * (y - 2) * z * dose_nodes + 0 + x * y * z * (dose_nodes - 2)  # z has 2 nodes -> none
    assert p.shape == (expect, 72)
    # a linear field along each axis has zero second differences
    grid = torch.stack(torch.meshgrid(*[torch.arange(n, dtype=torch.float64) for n in dims], indexing="ij"), -1)
    lin = (grid @ torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=F64)).permute(3, 2, 1, 0).reshape(-1)  # flat layout w,z,y,x
    assert (p @ lin).abs().max() < 1e-12
    # P @ E acts on the reduced vector without dropping rows
    sup = torch.tensor([True, True, False, True])
    e = extension_operator(dims, sup)
    assert (p @ e).shape == (expect, 54)


def _rank_deficient(cond_target=1e8, rank=4, n=5, rows=60):
    gen = torch.Generator().manual_seed(7)
    u, _ = torch.linalg.qr(torch.randn(rows, n, generator=gen, dtype=torch.float64))
    v, _ = torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=torch.float64))
    s = torch.tensor([1.0, 1e-2, 1e-4, 1.0 / cond_target, 0.0], dtype=F64)[:n]
    return u @ torch.diag(s) @ v.T


def test_streaming_qr_matches_dense_svd_conditioning_aware():
    m = _rank_deficient()
    w = torch.rand(60, dtype=torch.float64) + 0.5
    b = torch.randn(60, 3, dtype=torch.float64)
    q = StreamingQR(5, 3)
    for blk in range(0, 60, 13):
        q.add_block(m[blk : blk + 13], b[blk : blk + 13], w[blk : blk + 13])
    assert q.n_rows == 60
    sw = w.sqrt()[:, None]
    dense = torch.linalg.svdvals(m * sw)
    got = q.singular_values()
    assert (got - dense).abs().max() <= 1e-8 * dense.max()
    tol = dense.max() * max(60, 5) * EPS
    assert int((got > tol).sum()) == int((dense > tol).sum()) == 4
    # the reduced normal-equations are those of the weighted problem
    assert torch.allclose(q.r_data.T @ q.r_data, (m * sw).T @ (m * sw), atol=1e-12)
    assert torch.allclose(q.r_data.T @ q.qtb, (m * sw).T @ (b * sw), atol=1e-12)


def test_gram_route_is_not_equivalent_at_cond_1e8():
    """Documents why the Gram matrix was rejected: its eigenvalues distort the
    small singular values that the rank decision depends on."""
    m = _rank_deficient()
    dense = torch.linalg.svdvals(m)
    gram = torch.linalg.eigvalsh(m.T @ m).clamp_min(0).sqrt().flip(0)
    assert (gram - dense).abs().max() > 1e-8 * dense.max()  # fails the tolerance streaming QR meets


def test_reduced_solve_matches_dense_solve_and_rank_decisions():
    gen = torch.Generator().manual_seed(11)
    a = torch.randn(80, 6, generator=gen, dtype=torch.float64)
    a[:, 5] = a[:, 0] + a[:, 1]  # dependent column -> data rank 5
    b = torch.randn(80, 3, generator=gen, dtype=torch.float64)
    w = torch.rand(80, generator=gen, dtype=torch.float64) + 0.2
    pen = torch.zeros(2, 6, dtype=torch.float64)
    pen[0, 0], pen[0, 1], pen[0, 2] = 1, -2, 1
    pen[1, 3], pen[1, 4], pen[1, 5] = 1, -2, 1
    dense = solve_weighted(a, b, w, penalty=pen, lam=1e-3)
    q = StreamingQR(6, 3)
    for blk in range(0, 80, 17):
        q.add_block(a[blk : blk + 17], b[blk : blk + 17], w[blk : blk + 17])
    reduced = solve_weighted(q.r_data, q.qtb, torch.ones(6, dtype=torch.float64), penalty=pen, lam=1e-3, n_rows=80)
    # eps * cond(augmented)^2 sensitivity of a least-squares solution (~1e3^2 here): 1e-8, not 1e-10
    assert (reduced.x - dense.x).abs().max() <= 1e-8 * dense.x.abs().max()
    assert reduced.data_rank == dense.data_rank == 5
    assert reduced.rank == dense.rank
    assert abs(reduced.data_condition - dense.data_condition) <= 1e-6 * dense.data_condition
    assert abs(reduced.regularization_norm - dense.regularization_norm) <= 1e-10
    # without n_rows the lambda scaling (mean over rows) would differ by sqrt(80/6): guard the
    # contract at a lambda where the regularization visibly matters
    dense_strong = solve_weighted(a, b, w, penalty=pen, lam=0.3)
    reduced_strong = solve_weighted(q.r_data, q.qtb, torch.ones(6, dtype=torch.float64), penalty=pen, lam=0.3, n_rows=80)
    naive_strong = solve_weighted(q.r_data, q.qtb, torch.ones(6, dtype=torch.float64), penalty=pen, lam=0.3)
    assert (reduced_strong.x - dense_strong.x).abs().max() <= 1e-8 * dense_strong.x.abs().max()
    assert (naive_strong.x - dense_strong.x).abs().max() > 1e-3 * dense_strong.x.abs().max()


def test_lam_zero_reproduces_consistent_observations_despite_rank_deficiency():
    """Solver property (not a CLI mode): L = T = 5 with normalized doses
    0, .025, .05, .625, 1 has temporal rank 4 of 5 (rows 1-3 span one segment,
    row 4 constrains only the mean of two nodes); consistent data is still
    reproduced exactly at the observed rows while node values are not
    identifiable."""
    c = torch.tensor([0.0, 0.025, 0.05, 0.625, 1.0], dtype=F64)
    a = temporal_weights(5, c)
    assert torch.linalg.matrix_rank(a).item() == 4
    v_true = torch.tensor([1.0, -2.0, 3.0, 0.5, -1.0], dtype=F64)
    b = a @ v_true
    res = solve_weighted(a, b, torch.ones(5, dtype=torch.float64), penalty=None, lam=0.0)
    assert (a @ res.x - b).abs().max() <= 1e-12
    assert (res.x - v_true).abs().max() > 0.5
    assert res.data_rank == 4
