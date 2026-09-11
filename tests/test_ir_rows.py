"""ir/rows.py: row identity between pipeline-agnostic IRs and fit targets."""

import pytest
import torch

from cets_nonrigid.ir.build import build_ir_tilt_series
from cets_nonrigid.ir.core import IRMeta
from cets_nonrigid.ir.rows import RowMatch, TargetRowTable, align_ir_rows, match_rows


def _meta(t=4, kind="effective", labels=None, angles=None, valid=None):
    return IRMeta(
        kind="tilt_series",
        series_name="R",
        pixel_size_image_a=2.0,
        image_dims_px=(96, 96),
        volume_dims_px=(48, 48, 24),
        pixel_size_volume_a=2.0,
        projection_index=list(range(t)),
        projection_valid=valid or [True] * t,
        projection_order=list(range(t)),
        projection_dose=[float(i) for i in range(t)],
        projection_angle_deg=angles or [-30.0 + 20.0 * i for i in range(t)],
        projection_sec=[i + 1 for i in range(t)],
        projection_dark=[False] * t,
        source_tool="aretomo3",
        projection_angle_kind=[kind] * t,
        projection_label=labels,
    )


def _target(t=4, kind="effective", labels=None, angles=None, active=None):
    return TargetRowTable(
        angle_deg=angles or [-30.0 + 20.0 * i for i in range(t)],
        angle_kind=[kind] * t,
        active=active or [True] * t,
        dark=[not a for a in (active or [True] * t)],
        sec=[i + 10 for i in range(t)],
        dose=[5.0 * i for i in range(t)],
        labels=labels,
        order=list(range(t)),
    )


def test_row_map_file(tmp_path):
    f = tmp_path / "map.txt"
    f.write_text("# target source (zero-based)\n0 1\n1 0\n2 2\n3 3\n")
    m = match_rows(_meta(), _target(), row_map_file=f)
    assert m.row_map == [1, 0, 2, 3]
    assert m.method == "row_map"

    f.write_text("0 1\n0 2\n")  # duplicate target
    with pytest.raises(ValueError, match="duplicate"):
        match_rows(_meta(), _target(), row_map_file=f)
    f.write_text("9 0\n")  # out of range
    with pytest.raises(ValueError, match="ZERO-based"):
        match_rows(_meta(), _target(), row_map_file=f)


def test_label_matching_and_participation():
    labels = [f"img_{i}.mrc" for i in range(4)]
    m = match_rows(_meta(labels=list(reversed(labels))), _target(labels=labels))
    assert m.method == "labels"
    assert m.row_map == [3, 2, 1, 0]

    # duplicate labels on one side -> labels do not participate; angles run
    dup = ["a.mrc", "a.mrc", "c.mrc", "d.mrc"]
    m2 = match_rows(_meta(labels=dup), _target(labels=[f"x{i}" for i in range(4)]))
    assert m2.method in ("angles", "identity")

    # missing counterpart is a hard error, never a silent drop
    with pytest.raises(ValueError, match="never silently dropped|no target counterpart"):
        match_rows(
            _meta(labels=["a", "b", "c", "MISSING"]),
            _target(labels=["a", "b", "c", "d"]),
        )


def test_angle_kind_matrix():
    # unknown on either side: never eligible
    with pytest.raises(ValueError, match="unknown"):
        match_rows(_meta(kind="unknown"), _target(kind="effective"))
    with pytest.raises(ValueError, match="unknown"):
        match_rows(_meta(kind="effective"), _target(kind="unknown"))
    # cross kinds: not eligible
    with pytest.raises(ValueError, match="not eligible"):
        match_rows(_meta(kind="nominal"), _target(kind="effective"))
    # same non-unknown kinds: eligible
    assert match_rows(_meta(kind="nominal"), _target(kind="nominal")).method == "identity"
    assert match_rows(_meta(kind="effective"), _target(kind="effective")).method == "identity"


def test_duplicate_angles_demand_labels_or_map(tmp_path):
    angles = [-30.0, -30.0, 10.0, 30.0]
    with pytest.raises(ValueError, match="row labels or an explicit"):
        match_rows(_meta(angles=angles), _target(angles=angles))
    # an explicit row map resolves it
    f = tmp_path / "map.txt"
    f.write_text("0 0\n1 1\n2 2\n3 3\n")
    m = match_rows(_meta(angles=angles), _target(angles=angles), row_map_file=f)
    assert m.row_map == [0, 1, 2, 3]


def test_unmatched_active_target_rows():
    tgt = _target(t=5, angles=[-30.0, -10.0, 10.0, 30.0, 50.0])
    with pytest.raises(ValueError, match="deactivate_unmatched"):
        match_rows(_meta(), tgt)
    m = match_rows(_meta(), tgt, deactivate_unmatched=True)
    assert m.target_active == [True, True, True, True, False]


def _grid_ir(meta):
    from cets_nonrigid.models.aretomo_ts import AretomoTsModel

    t = len(meta.projection_index)
    model = AretomoTsModel(
        rot_deg=torch.full((t,), 85.0, dtype=torch.float64),
        tilt_deg=torch.tensor(meta.projection_angle_deg, dtype=torch.float64),
        shifts_px=torch.zeros(t, 2, dtype=torch.float64),
        raw_size_px=tuple(meta.image_dims_px),
        pixel_size_a=meta.pixel_size_image_a,
        volume_dims_a=tuple(d * 2.0 for d in meta.volume_dims_px),
        local=None,
    )
    return build_ir_tilt_series(
        model,
        torch.tensor([d * 2.0 for d in meta.volume_dims_px]),
        torch.tensor([d * 2.0 for d in meta.image_dims_px]),
        meta=meta,
        grid_shape=(4, 4, 3),
    )


def test_align_identity_still_replaces_metadata():
    ir = _grid_ir(_meta())
    tgt = _target()
    m = RowMatch(row_map=[0, 1, 2, 3], target_active=list(tgt.active), method="identity")
    aligned = align_ir_rows(ir, m, tgt)
    # arrays untouched
    assert aligned.source_projected is ir.source_projected
    # metadata REPLACED by the target row table
    assert aligned.meta.projection_sec == [10, 11, 12, 13]
    assert aligned.meta.projection_dose == [0.0, 5.0, 10.0, 15.0]


def test_align_permutation_and_expansion():
    ir = _grid_ir(_meta())
    # 6-row target; IR rows land at 5,3,1,0; rows 2 and 4 unmatched (inactive)
    tgt = TargetRowTable(
        angle_deg=[30.0, 10.0, 99.0, -10.0, 98.0, -30.0],
        angle_kind=["effective"] * 6,
        active=[True, True, False, True, False, True],
        dark=[False, False, True, False, True, False],
        sec=list(range(1, 7)),
        dose=[0.0] * 6,
    )
    m = RowMatch(row_map=[5, 3, 1, 0], target_active=list(tgt.active), method="angles")
    aligned = align_ir_rows(ir, m, tgt)
    assert aligned.n_projections == 6
    torch.testing.assert_close(aligned.source_projected[5], ir.source_projected[0])
    torch.testing.assert_close(aligned.source_projected[0], ir.source_projected[3])
    # unmatched rows invalid everywhere, incl. held-out and global baselines
    assert not aligned.projection_valid[2].any()
    assert not aligned.heldout_projection_valid[4].any()
    torch.testing.assert_close(
        aligned.heldout_source_projected_global[5], ir.heldout_source_projected_global[0]
    )
    assert aligned.meta.projection_valid == [True, True, False, True, False, True]


def test_row_mapped_model_equivalence(tmp_path):
    """Phase B retirement proof: the align_ir_rows path reproduces the retired
    _RowMappedModel-built IR array-for-array on the r2w fixture. The single
    intended difference: `weights` on unmapped (projection_valid=False) rows —
    the old path stored the boundary-ramp value of a zero coordinate (1.0),
    the new path stores 0; every consumer masks by projection_valid."""
    import numpy as np
    import test_r4_w2r as w2r_helpers

    from cets_nonrigid.convert_relion import (
        load_relion_source,
        match_relion_rows_to_template,
        relion_model_from_data,
        warp_to_relion,
    )
    from cets_nonrigid.io.warp_xml import load_warp_tiltseries
    from cets_nonrigid.ir.build import build_ir_tilt_series_from_points
    from cets_nonrigid.models.relion_ts import RelionParticleSetModel

    class RowMappedModelReference:
        """Verbatim copy of the retired convert_relion._RowMappedModel."""

        def __init__(self, model, row_map, n_target):
            self._m = model
            self._map = row_map
            self.n_projections = n_target

        def _expand(self, xy, valid):
            t_src, n, _ = xy.shape
            out = torch.zeros(self.n_projections, n, 2, dtype=xy.dtype)
            ov = torch.zeros(self.n_projections, n, dtype=torch.bool)
            for i in range(t_src):
                w = int(self._map[i])
                out[w] = xy[i]
                ov[w] = valid[i]
            return out, ov

        def project_volume(self, points_3d):
            return self._expand(*self._m.project_volume(points_3d))

        def project_volume_global(self, points_3d):
            return self._expand(*self._m.project_volume_global(points_3d))

    rng = np.random.default_rng(20260912)
    xml = w2r_helpers._write_synthetic_xml(tmp_path / "src.xml")
    stack = w2r_helpers._dummy_stack(tmp_path / "stack.mrc")
    vol_a = torch.tensor(w2r_helpers.VOL_A, dtype=torch.float64)
    pos = torch.tensor(rng.uniform(0.1, 0.9, (100, 3))) * vol_a
    names = [f"EQ/{i + 1}" for i in range(100)]
    r = warp_to_relion(
        xml, tmp_path / "bundle", pixel_size_a=w2r_helpers.PIX, tomo_name="EQ",
        positions_eff_a=pos, particle_names=names, tilt_stack=stack,
    )
    template = load_warp_tiltseries(xml)
    ts = template.ts
    data, model, positions, pnames, trajectories, _d = load_relion_source(
        optimisation_set=r.optimisation_set, tomo_name="EQ", image_dims_px=(0, 0),
    )
    pix = data.pixel_size_a
    image_dims_px = tuple(
        round(float(v) / pix) for v in ts.image_dimensions_physical
    )
    model = relion_model_from_data(data, image_dims_px)
    row_map, _active = match_relion_rows_to_template(
        ts, data.nominal_stage_angle_deg, deactivate_unmatched=True
    )
    source = RelionParticleSetModel(model, positions, trajectories_a=trajectories)

    meta = _meta(t=ts.n_tilts, kind="nominal")
    meta = meta.model_copy(update={"sampling": "particles",
                                   "volume_dims_px": tuple(int(d) for d in data.tomo_dims_px),
                                   "pixel_size_volume_a": pix,
                                   "pixel_size_image_a": pix})

    # old path: mapped model, template-row IR
    mapped = RowMappedModelReference(source, row_map, ts.n_tilts)
    ir_old = build_ir_tilt_series_from_points(
        mapped, positions.to(torch.float64), ts.image_dimensions_physical, meta=meta,
        point_names=pnames,
    )
    # new path: star-order IR + align
    t_star = int(data.nominal_stage_angle_deg.shape[0])
    star_meta = meta.model_copy(update={
        "projection_index": list(range(t_star)), "projection_valid": [True] * t_star,
        "projection_order": list(range(t_star)), "projection_dose": [0.0] * t_star,
        "projection_angle_deg": [float(a) for a in data.nominal_stage_angle_deg],
        "projection_sec": [-1] * t_star, "projection_dark": [False] * t_star,
    })
    ir_star = build_ir_tilt_series_from_points(
        source, positions.to(torch.float64), ts.image_dimensions_physical, meta=star_meta,
        point_names=pnames,
    )
    table = TargetRowTable(
        angle_deg=[float(a) for a in ts.angles], angle_kind=["unknown"] * ts.n_tilts,
        active=[True] * ts.n_tilts, dark=[False] * ts.n_tilts,
        sec=[-1] * ts.n_tilts, dose=[0.0] * ts.n_tilts,
    )
    ir_new = align_ir_rows(
        ir_star,
        RowMatch(row_map=[int(m) for m in row_map],
                 target_active=[True] * ts.n_tilts, method="pipeline"),
        table,
    )

    torch.testing.assert_close(ir_new.source_projected, ir_old.source_projected)
    torch.testing.assert_close(ir_new.source_projected_global, ir_old.source_projected_global)
    torch.testing.assert_close(ir_new.heldout_source_projected, ir_old.heldout_source_projected)
    torch.testing.assert_close(
        ir_new.heldout_source_projected_global, ir_old.heldout_source_projected_global
    )
    assert torch.equal(ir_new.projection_valid, ir_old.projection_valid)
    assert torch.equal(ir_new.heldout_projection_valid, ir_old.heldout_projection_valid)
    assert torch.equal(ir_new.point_index, ir_old.point_index)
    assert ir_new.point_names == ir_old.point_names
    # weights: identical on valid rows; the documented inert-row difference
    valid_rows = ir_old.projection_valid.any(dim=1)
    torch.testing.assert_close(ir_new.weights[valid_rows], ir_old.weights[valid_rows])
    assert (ir_new.weights[~valid_rows] == 0).all()
