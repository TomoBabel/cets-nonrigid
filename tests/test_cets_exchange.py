"""CETS boundary gates independent of the transferred numerical unit tests."""

import json
from pathlib import Path
import shutil
from xml.etree import ElementTree as ET

import numpy as np
import pytest
import torch
import zarr

from cets_nonrigid.api import (
    load_native,
    sample,
    attach_deformation,
    read_bundle,
    write_bundle,
    fit,
    export_native,
    to_cets,
)
from cets_nonrigid.context import CetsContext
from cets_nonrigid.runtime import runtime_ir


@pytest.fixture(scope="module")
def warp_source():
    return load_native("warp", "tests/golden/TS_1_volwarp.xml", pixel_size_a=0.834, grid_shape=(5, 5, 3))


@pytest.fixture(scope="module")
def warp_bundle(warp_source):
    return attach_deformation(warp_source.context, sample(warp_source))


def test_residual_quantization_and_no_double_counting(warp_source, warp_bundle):
    context = warp_bundle.context
    for block in (warp_bundle.samples.training, warp_bundle.samples.heldout):
        points = block.points + torch.tensor(context.reference_center_a)
        native, _ = warp_source.ir.native_model.project_volume(points)
        expected = native.to(torch.float64) - torch.tensor(context.image_centers_a)[:, None]
        restored = context.evaluate_global(block.points) + block.projected_residual.to(torch.float64)
        r = block.projected_residual
        ulp = (torch.nextafter(r, torch.full_like(r, float("inf"))) - r).abs().to(torch.float64)
        rounding = 8 * torch.finfo(torch.float64).eps * torch.maximum(torch.ones_like(expected), expected.abs())
        assert ((restored - expected).abs()[block.observation_valid] <= (ulp + rounding)[block.observation_valid]).all()
        assert block.displacement_3d.abs().max() > 1
        assert block.ctf_depth.abs().max() > 1
    assert not warp_source.context.owner.has_non_rigid_alignment
    assert warp_bundle.context.owner.has_non_rigid_alignment


def test_outside_image_observations_survive(warp_bundle):
    block = warp_bundle.samples.training
    outside = block.observation_valid & ~block.projection_valid
    assert outside.any()
    assert torch.isfinite(block.observations(warp_bundle.context)[outside]).all()
    assert block.projected_residual[outside].abs().max() > 0


def test_store_roundtrip_relocation_and_geometry_tampering(warp_bundle, tmp_path):
    path = tmp_path / "first" / "exchange.cets.json"
    write_bundle(warp_bundle, path)
    original = path.read_bytes()
    restored = read_bundle(path)
    torch.testing.assert_close(
        restored.samples.training.projected_residual, warp_bundle.samples.training.projected_residual, rtol=0, atol=0
    )
    store = zarr.open_group(str(path.parent / "exchange.nonrigid.zarr"), mode="r")
    group = store[restored.context.owner.non_rigid_alignment.payload_group]
    assert "source_projected_global" not in group and "projection" not in group
    assert "spacing" not in group.attrs and "volume_dims_px" not in group.attrs
    with pytest.raises(FileExistsError):
        write_bundle(warp_bundle, path)
    assert path.read_bytes() == original
    shutil.move(path.parent, tmp_path / "moved")
    moved = tmp_path / "moved" / path.name
    read_bundle(moved)
    document = json.loads(moved.read_text())
    document["regions"][0]["alignments"][0]["projection_alignments"][0]["sequence"][-1]["translation"][0] += 0.5
    moved.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="context digest mismatch"):
        read_bundle(moved)


def test_safe_attachment_refuses_changed_global(warp_source):
    data = sample(warp_source)
    changed = warp_source.context.document.model_copy(deep=True)
    ctx = CetsContext(changed, warp_source.context.alignment_id)
    ctx.owner.projection_alignments[0].sequence[-1].translation[0] += 5
    with pytest.raises(ValueError, match="context changed"):
        attach_deformation(ctx, data)
    with pytest.raises(ValueError, match="native rigid baseline"):
        sample(warp_source, ctx)


@pytest.mark.parametrize("count", [10, 36])
def test_particle_identity_and_heldout_roundtrip(count, tmp_path):
    points = torch.rand((count, 3), generator=torch.Generator().manual_seed(count), dtype=torch.float64) * torch.tensor(
        [3000, 4000, 800]
    )
    bundle = to_cets(
        "warp",
        "tests/golden/TS_1_volwarp.xml",
        pixel_size_a=0.834,
        positions_a=points,
        names=[f"p-{i}" for i in range(count)],
    )
    path = tmp_path / "particles.cets.json"
    write_bundle(bundle, path)
    restored = read_bundle(path)
    assert restored.samples.training.point_ids == bundle.samples.training.point_ids
    assert restored.samples.heldout.point_ids == bundle.samples.heldout.point_ids
    assert restored.samples.heldout.count == (0 if count < 32 else 12)
    ir = runtime_ir(restored)
    assert ir.source_projected_global.dtype == torch.float64
    torch.testing.assert_close(ir.points, points[ir.point_index], rtol=0, atol=1e-9)


def test_warp_x_tilt_representability_is_enforced(warp_bundle):
    with pytest.raises(ValueError, match="global fit residual"):
        fit(warp_bundle, "aretomo3", patch_grid=(3, 3))


def test_aretomo_to_warp_to_aretomo_cets(tmp_path):
    from cryoet_alignment.io.aretomo3.aln import AreTomo3ALN
    from cets_nonrigid.io.aln import raw_tilts_from_aln
    from cets_nonrigid.io.dose import raw_dose_assume_stack_order

    source = "tests/data/test.aln"
    parsed = AreTomo3ALN.from_file(source)
    dose = raw_dose_assume_stack_order(raw_tilts_from_aln(parsed), dose_per_tilt=3)
    a = to_cets("aretomo3", source, pixel_size_a=2, tomo_size_px=(2032, 2032, 400), raw_dose=dose, grid_shape=(5, 5, 3))
    assert len(a.context.row_ids) == 8
    assert np.count_nonzero(a.context.operators()[2]) == 6
    assert not a.samples.training.observation_valid[~torch.tensor(a.context.operators()[2])].any()
    wpath = tmp_path / "warp.xml"
    export_native(fit(a, "warp", movement_grid=(3, 3)), wpath)
    w = to_cets("warp", wpath, pixel_size_a=2, grid_shape=(5, 5, 3))
    apath = tmp_path / "output.aln"
    result = fit(w, "aretomo3", patch_grid=(3, 3))
    export_native(result, apath)
    assert apath.is_file()
    assert result.metrics["global_rms_px_heldout"] < 1e-3
    restored = to_cets("aretomo3", apath, pixel_size_a=2, tomo_size_px=(2032, 2032, 400), grid_shape=(5, 5, 3))
    assert len(restored.context.row_ids) == 6


def test_warp_constants_remain_payload_and_total_matches_folded(tmp_path):
    root = ET.parse("tests/golden/TS_1_volwarp.xml").getroot()
    root.set("LevelAngleX", "0")
    for name in ("GridVolumeWarpX", "GridVolumeWarpY", "GridVolumeWarpZ", "GridMovementX", "GridMovementY"):
        old = root.find(name)
        if old is not None:
            root.remove(old)
        grid = ET.SubElement(
            root, name, Width="1", Height="1", Depth="1", **({"Duration": "1"} if "Volume" in name else {})
        )
        ET.SubElement(
            grid,
            "Node",
            X="0",
            Y="0",
            Z="0",
            Value="7" if name == "GridMovementX" else "0",
            **({"W": "0"} if "Volume" in name else {}),
        )
    path = tmp_path / "constant.xml"
    path.write_bytes(ET.tostring(root))
    native = load_native("warp", path, pixel_size_a=0.834, grid_shape=(3, 3, 2))
    data = sample(native)
    assert (data.training.projected_residual[..., 0] + 7).abs().max() < 0.002
    assert data.training.projected_residual[..., 1].abs().max() < 0.002
    from cryoet_alignment.io.warp.alignment import WarpAlignment
    from cryoet_alignment.io.cryoet_data_portal.alignment import Alignment
    from cryoet_alignment.io.cets.alignment import alignment_to_cets, ReferenceVolume, fold_projection
    from cryoet_alignment.io.cets.frames import image_frame

    parsed = WarpAlignment.from_string(path.read_text(), pixel_size_a=0.834)
    folded = alignment_to_cets(
        Alignment.from_warp(parsed, pixel_size_a=0.834),
        tilt_series_id=native.context.parent.id,
        alignment_name="folded",
        image=image_frame(native.context.rows[0]),
        reference=ReferenceVolume.from_tomogram(native.context.reference_entity),
    )
    point = data.training.points.numpy()
    expected = torch.tensor(
        np.stack(
            [(point @ fold_projection(p)[0].T)[:, :2] + fold_projection(p)[1] for p in folded.projection_alignments]
        )
    )
    actual = data.training.observations(native.context)
    assert (expected - actual).abs().max() < 0.002


def test_movie_cycle_and_global_classification(tmp_path):
    golden = json.loads(Path("tests/golden/movie_positions.json").read_text())
    size = tuple(int(v) for v in golden["image_dims"])
    movie = to_cets(
        "warp-movie",
        "tests/golden/movie_synthetic.xml",
        image_size_px=size,
        pixel_size_a=1,
        n_frames=golden["n_frames"],
        fraction_frames=golden["fraction_frames"],
        grid_shape=(5, 5),
    )
    assert np.abs(movie.context.operators()[1]).max() > 0
    path = tmp_path / "movie.cets.json"
    write_bundle(movie, path)
    read_bundle(path)
    for target, suffix in (("mcaln", ".mcaln"), ("relion-motion", ".star")):
        result = fit(movie, target)
        out = tmp_path / (target + suffix)
        export_native(result, out)
        source = to_cets(target, out, grid_shape=(5, 5))
        assert source.context.row_ids and source.context.kind == "movie-frame-residual"
        returned = fit(source, "warp-movie", local_grid=(3, 3, 4))
        assert returned.files


def test_relion_trajectory_export_uses_all_particles_and_cets_ctf(tmp_path):
    count = 36
    points = torch.rand((count, 3), generator=torch.Generator().manual_seed(2), dtype=torch.float64) * torch.tensor(
        [3000, 4000, 800]
    )
    b = to_cets(
        "warp",
        "tests/golden/TS_1_volwarp.xml",
        pixel_size_a=0.834,
        positions_a=points,
        names=[f"p{i}" for i in range(count)],
    )
    result = fit(
        b,
        "relion",
        micrograph_names=[f"{i + 1}@tilts.mrcs" for i in range(len(b.context.rows))],
        trajectory_gauge="ctf-optimal",
    )
    assert result.metrics["n_particles"] == count
    assert result.metrics["max_residual_px"] < 1e-9
    assert result.metrics["depth_source"] == "displacement"
    out = tmp_path / "relion"
    export_native(result, out)
    assert all(str(tmp_path) not in v.decode(errors="ignore") for k, v in result.files.items() if k.endswith(".star"))
    r = to_cets(
        "relion",
        out / "optimisation_set.star",
        tomo_name=b.context.parent.id,
        image_size_px=b.context.image_frames[0].size_px,
    )
    assert r.samples.training.count + r.samples.heldout.count == count
    assert r.context.parent.defocus_handedness == b.context.parent.defocus_handedness


def test_grid_cannot_invent_particle_trajectories(warp_bundle):
    with pytest.raises(ValueError, match="particle-bound"):
        fit(warp_bundle, "relion")


def test_row_reordering_permutates_all_channels(warp_bundle):
    from cets_nonrigid.api import reorder_rows

    ids = list(reversed(warp_bundle.context.row_ids))
    reordered = reorder_rows(warp_bundle, ids)
    assert reordered.context.row_ids == ids
    for first, second in (
        (warp_bundle.samples.training, reordered.samples.training),
        (warp_bundle.samples.heldout, reordered.samples.heldout),
    ):
        for name in (
            "projected_residual",
            "observation_valid",
            "projection_valid",
            "weights",
            "displacement_3d",
            "ctf_depth",
        ):
            assert torch.equal(getattr(first, name).flip(0), getattr(second, name))
    assert warp_bundle.context.row_ids != ids


def test_plain_rigid_document_requires_no_payload_or_processing_context(tmp_path):
    from cets_data_model.models import models as m
    from cets_nonrigid.api import AlignmentBundle

    data = m.Dataset(
        regions=[
            m.Region(
                id="region", tilt_series=[m.TiltSeries(id="ts")], alignments=[m.Alignment(id="a", tilt_series_id="ts")]
            )
        ]
    )
    path = tmp_path / "rigid.cets.json"
    write_bundle(AlignmentBundle(data), path)
    restored = read_bundle(path)
    assert not restored.context.owner.has_non_rigid_alignment
    assert not restored.deformations


def test_converter_supplied_context_and_rows(warp_source):
    document = warp_source.context.document.model_copy(deep=True)
    context = CetsContext(document, warp_source.context.alignment_id)
    context.parent.images.reverse()
    data = sample(warp_source, context)
    attached = attach_deformation(context, data)
    assert not context.owner.has_non_rigid_alignment
    assert attached.context.row_ids == list(reversed(warp_source.context.row_ids))
    original = sample(warp_source)
    torch.testing.assert_close(
        data.training.projected_residual, original.training.projected_residual.flip(0), rtol=0, atol=0
    )


def test_target_template_geometry_gate(warp_bundle, tmp_path):
    root = ET.parse("tests/golden/TS_1_volwarp.xml").getroot()
    root.set("ImageDimensionsAngstrom", "100, 100")
    path = tmp_path / "different.xml"
    ET.ElementTree(root).write(path)
    with pytest.raises(ValueError, match="template geometry differs"):
        fit(warp_bundle, "warp", template_xml=path)


def test_fit_report_is_bound_to_context(warp_bundle):
    from cets_nonrigid.api import FitResult, with_fit_report

    wrong = FitResult("warp", {"fit.xml": b"result"}, "fit.xml", {"source_context_digest": "other"})
    with pytest.raises(ValueError, match="different CETS context"):
        with_fit_report(warp_bundle, wrong)
