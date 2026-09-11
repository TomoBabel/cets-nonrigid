"""R3: scattered-particle IR builder, store round trip, heldout_status."""

import copy
import json

import numpy as np
import pytest
import torch
import zarr

from cets_nonrigid.ir.build import build_ir_tilt_series_from_points
from cets_nonrigid.ir.core import IRMeta
from cets_nonrigid.models.relion_ts import RelionTomogramModel

RNG = np.random.default_rng(20260902)

DIMS = {"tomo_dims_px": (64, 64, 32), "image_dims_px": (128, 96)}
PIX = 2.0


def _model(t=6):
    return RelionTomogramModel(
        xtilt_deg=torch.zeros(t, dtype=torch.float64),
        ytilt_deg=torch.linspace(-60, 60, t, dtype=torch.float64),
        zrot_deg=torch.full((t,), 85.0, dtype=torch.float64),
        xshift_a=torch.tensor(RNG.uniform(-20, 20, t)),
        yshift_a=torch.tensor(RNG.uniform(-20, 20, t)),
        pixel_size_a=PIX,
        **DIMS,
    )


def _meta(t=6, sampling="particles"):
    return IRMeta(
        kind="tilt_series",
        series_name="TS_R3",
        pixel_size_image_a=PIX,
        image_dims_px=DIMS["image_dims_px"],
        volume_dims_px=DIMS["tomo_dims_px"],
        pixel_size_volume_a=PIX,
        projection_index=list(range(t)),
        projection_valid=[True] * t,
        projection_order=list(range(t)),
        projection_dose=[3.0 * i for i in range(t)],
        projection_angle_deg=list(np.linspace(-60, 60, t)),
        projection_sec=[i + 1 for i in range(t)],
        projection_dark=[False] * t,
        source_tool="relion5",
        sampling=sampling,
    )


def _particles(p):
    vol_a = torch.tensor(DIMS["tomo_dims_px"], dtype=torch.float64) * PIX
    pts = torch.tensor(RNG.uniform(0.1, 0.9, (p, 3))) * vol_a
    names = [f"TS_R3/{i + 1}" for i in range(p)]
    return pts, names


def test_builder_split_and_projection_consistency():
    model = _model()
    pts, names = _particles(80)
    ir = build_ir_tilt_series_from_points(
        model, pts, model.image_dims_a, meta=_meta(), point_names=names,
        heldout_fraction=0.2, heldout_seed=7,
    )
    assert ir.grid_shape is None
    assert ir.heldout_status == "evaluated"
    n_tr, n_ho = ir.points.shape[0], ir.heldout_points.shape[0]
    assert n_tr + n_ho == 80 and 16 <= n_ho <= 80 // 3

    # index partition covers the original order exactly once
    all_idx = torch.cat([ir.point_index, ir.heldout_point_index]).sort().values
    assert torch.equal(all_idx, torch.arange(80))
    # names travel with their rows
    assert ir.point_names == [names[int(i)] for i in ir.point_index]
    assert ir.heldout_point_names == [names[int(i)] for i in ir.heldout_point_index]

    # rows are the full-set projections at the split indices (projected ONCE)
    full_proj, _ = model.project_volume(pts)
    torch.testing.assert_close(
        ir.source_projected.to(torch.float64),
        full_proj[:, ir.point_index].to(torch.float32).to(torch.float64),
    )
    torch.testing.assert_close(ir.points, pts[ir.point_index])

    # deterministic per seed
    ir2 = build_ir_tilt_series_from_points(
        model, pts, model.image_dims_a, meta=_meta(), point_names=names,
        heldout_fraction=0.2, heldout_seed=7,
    )
    assert torch.equal(ir.point_index, ir2.point_index)


def test_builder_too_few_particles_reports_not_evaluated():
    model = _model()
    pts, names = _particles(10)
    with pytest.warns(UserWarning, match="not_evaluated"):
        ir = build_ir_tilt_series_from_points(
            model, pts, model.image_dims_a, meta=_meta(), point_names=names
        )
    assert ir.heldout_status == "not_evaluated"
    assert ir.heldout_points.shape[0] == 0
    assert ir.points.shape[0] == 10  # nothing withheld


def test_builder_validation():
    model = _model()
    pts, _names = _particles(20)
    meta = _meta()
    with pytest.raises(ValueError, match="sampling"):
        build_ir_tilt_series_from_points(model, pts, model.image_dims_a, meta=_meta(sampling="grid"))
    with pytest.raises(ValueError, match="duplicate particle name"):
        build_ir_tilt_series_from_points(
            model, pts, model.image_dims_a, meta=meta, point_names=["a"] * 20
        )
    with pytest.raises(ValueError, match="STAR-safe"):
        build_ir_tilt_series_from_points(
            model, pts, model.image_dims_a, meta=meta,
            point_names=[f"p {i}" for i in range(20)],
        )
    bad = pts.clone()
    bad[3, 2] = -5.0
    with pytest.raises(ValueError, match="outside the tomogram"):
        build_ir_tilt_series_from_points(model, bad, model.image_dims_a, meta=meta)
    with pytest.raises(ValueError, match="non-finite"):
        nan_pts = pts.clone()
        nan_pts[0, 0] = float("nan")
        build_ir_tilt_series_from_points(model, nan_pts, model.image_dims_a, meta=meta)


def _cets_particle_bundle(tmp_path, count):
    from cets_nonrigid import api
    points = torch.rand((count, 3), generator=torch.Generator().manual_seed(count), dtype=torch.float64)
    points = (points * .7 + .15) * torch.tensor([3336., 4753.8, 834.])
    path = tmp_path / "particles.cets.json"
    bundle = api.to_cets("warp", "tests/golden/TS_1_volwarp.xml", pixel_size_a=.834,
        positions_a=points, names=[f"point-{i}" for i in range(count)], output=path)
    return bundle, api.read_bundle(path)


def test_store_roundtrip_particle_sampling(tmp_path):
    from cets_nonrigid.runtime import runtime_ir
    bundle, restored = _cets_particle_bundle(tmp_path, 64)
    ir, back = runtime_ir(bundle), runtime_ir(restored)
    assert back.grid_shape is None and back.meta.sampling == "particles"
    assert back.heldout_status == "evaluated"
    assert back.point_names == ir.point_names and back.heldout_point_names == ir.heldout_point_names
    assert torch.equal(back.point_index, ir.point_index)
    assert torch.equal(back.heldout_point_index, ir.heldout_point_index)
    group = restored.context.owner.non_rigid_alignment.payload_group
    metadata = json.loads((tmp_path / "particles.nonrigid.zarr" / group / "point_ids/zarr.json").read_text())
    assert metadata["data_type"] == "string"
    assert "vlen-utf8" in [codec["name"] for codec in metadata["codecs"]]
    assert metadata["dimension_names"] == ["sample"]


def test_store_roundtrip_not_evaluated(tmp_path):
    from cets_nonrigid.runtime import runtime_ir
    _, restored = _cets_particle_bundle(tmp_path, 8)
    ir = runtime_ir(restored)
    assert restored.samples.heldout.count == 0
    assert ir.heldout_points.shape[0] == 0
    assert ir.heldout_status == "not_evaluated"


def test_target_fit_explicit_dimension_names(tmp_path):
    from cets_nonrigid import api
    _, restored = _cets_particle_bundle(tmp_path, 40)
    group_name = restored.context.owner.non_rigid_alignment.payload_group
    root = zarr.open_group(str(tmp_path / "particles.nonrigid.zarr"), mode="r")
    assert root[group_name + "/displacement_3d"].metadata.dimension_names == ("tilt_image", "sample", "coordinate")
    metadata_path = tmp_path / "particles.nonrigid.zarr" / group_name / "displacement_3d/zarr.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["dimension_names"] = ["tilt_image", "sample", "wrong_axis"]
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="dimension names"):
        api.read_bundle(tmp_path / "particles.cets.json")


def test_fit_reports_not_evaluated_on_empty_heldout(ts1_xml_path_module):
    """fit_warp_movement propagates heldout_status instead of NaN metrics."""
    from cets_nonrigid.fit.warp_ts_fit import fit_warp_movement
    from cets_nonrigid.io.warp_xml import load_warp_tiltseries
    from cets_nonrigid.models.warp_ts import WarpTiltSeriesModel

    series = load_warp_tiltseries(ts1_xml_path_module)
    model = WarpTiltSeriesModel(series.ts)
    t = series.ts.n_tilts
    vol_a = model.volume_dims_a.to(torch.float64)
    pts = torch.tensor(RNG.uniform(0.2, 0.8, (12, 3))) * vol_a

    meta = IRMeta(
        kind="tilt_series",
        series_name="TS_1",
        pixel_size_image_a=float(model.image_dims_a[0]) / 1000.0,
        image_dims_px=(1000, 1000),
        volume_dims_px=tuple(int(v) for v in (vol_a / (vol_a[0] / 1000.0)).round()),
        pixel_size_volume_a=float(vol_a[0]) / 1000.0,
        projection_index=list(range(t)),
        projection_valid=[True] * t,
        projection_order=list(range(t)),
        projection_dose=[float(d) for d in series.ts.dose],
        projection_angle_deg=[float(a) for a in series.ts.angles],
        projection_sec=[-1] * t,
        projection_dark=[False] * t,
        source_tool="warp",
        sampling="particles",
    )
    with pytest.warns(UserWarning):
        ir = build_ir_tilt_series_from_points(model, pts, model.image_dims_a, meta=meta)
    assert ir.heldout_status == "not_evaluated"

    ts_target = copy.deepcopy(series.ts)
    fit = fit_warp_movement(ir, ts_target, movement_grid=(4, 4))
    assert fit.heldout_status == "not_evaluated"
    assert fit.rms_a_heldout is None
    assert fit.per_tilt_rms_a_heldout is None
    assert np.isfinite(fit.rms_a_train)
