"""Native readers and the CETS boundary; numerical kernels keep their tested frames."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import torch
from cets_data_model.models import models as m
from cryoet_alignment.io.cets.alignment import ReferenceVolume, alignment_to_cets
from cryoet_alignment.io.cets.annotations import point_set_entity
from cryoet_alignment.io.cets.frames import attach_frames, image_frame
from cryoet_alignment.io.cryoet_data_portal.alignment import Alignment as HubAlignment

from cets_nonrigid.context import CetsContext


@dataclass
class NativeAlignment:
    source: str
    ir: Any  # validated native tilt-series or movie runtime container
    context: CetsContext
    native_files: dict[str, bytes] = field(default_factory=dict)
    options: dict = field(default_factory=dict)
    model_rows: list[int] = field(default_factory=list)


def _frame(entity, size, spacing):
    for name, value in zip(("width", "height", "depth"), size):
        setattr(entity, name, int(value))
    attach_frames(entity, tuple(size), [spacing] * len(size))
    return entity


def _projection(matrix, shift, row_id, alignment_id):
    from cryoet_alignment.io.cets.rotation import check_rotation, decompose, tilt_matrix, in_plane_matrix

    check_rotation(matrix, "native projection")
    rot, tilt, xrot = decompose(matrix)
    return m.ProjectionAlignment(
        id=f"{alignment_id}_{row_id}",
        tilt_image_id=row_id,
        name="tomogram_to_projection",
        input="physical",
        output="physical",
        sequence=[
            m.Affine(name="tilt", affine=tilt_matrix(tilt, xrot).tolist()),
            m.Affine(name="in_plane_rotation", affine=in_plane_matrix(rot).tolist()),
            m.Translation(name="shift", translation=list(map(float, shift))),
        ],
    )


def _provenance(source, files, options):
    parameters = []
    for key in ("fraction_frames",):
        if key in options:
            parameters.append(m.NativeParameter(name=key, value_json=json.dumps(options[key], allow_nan=False)))
    return m.ProcessingProvenance(
        software_name=source,
        parameters=parameters,
        artifacts=[
            m.NativeArtifact(role=key, uri=f"snapshot:{key}", sha256=hashlib.sha256(value).hexdigest())
            for key, value in files.items()
        ],
    )


def _tilt_document(source, ir, files, options, series_id, alignment_id, region_id):
    meta, model = ir.meta, ir.native_model
    pix = meta.pixel_size_image_a
    if source == "warp":
        from cets_nonrigid.fit.relion_global import integer_dims_px

        integer_dims_px(model.image_dims_a, pix, "native image extent")
        integer_dims_px(model.volume_dims_a, meta.pixel_size_volume_a, "native volume extent")
    ids = [f"{series_id}_{i}" for i in range(ir.n_projections)]
    images = []
    dose = meta.projection_dose
    known_dose = len(set(dose)) > 1
    order = np.argsort(np.argsort(dose, kind="stable"), kind="stable") if known_dose else None
    kinds = meta.projection_angle_kind or ["unknown"] * len(ids)
    for i, key in enumerate(ids):
        image = m.TiltImage(
            id=key,
            section=i,
            path=(
                ir.native_data.micrograph_names[i] if source == "relion" and ir.native_data.micrograph_names else None
            ),
            accumulated_dose=float(dose[i]) if known_dose else None,
            acquisition_order=int(order[i]) if order is not None else None,
            nominal_tilt_angle=meta.projection_angle_deg[i] if kinds[i] == "nominal" else None,
        )
        images.append(_frame(image, meta.image_dims_px, pix))
    series = m.TiltSeries(id=series_id, images=images)
    reference = _frame(
        m.Tomogram(id=f"{series_id}_volume", path=None, tilt_series_id=series_id),
        meta.volume_dims_px,
        meta.pixel_size_volume_a,
    )
    if source == "warp":
        from cryoet_alignment.io.warp.alignment import WarpAlignment

        # Zero source-defined deformation terms in a private XML tree. The shared rigid
        # reader otherwise folds constant movement/volume grids, violating this profile.
        root = ET.fromstring(files["source_xml"])
        for name in ("GridMovementX", "GridMovementY", "GridVolumeWarpX", "GridVolumeWarpY", "GridVolumeWarpZ"):
            grid = root.find(name)
            if grid is not None:
                for node in grid.iter("Node"):
                    node.set("Value", "0")
        parsed = WarpAlignment.from_string(
            ET.tostring(root, encoding="unicode"),
            pixel_size_a=pix,
            image_dims_a=model.image_dims_a.tolist(),
            volume_dims_a=model.volume_dims_a.tolist(),
        )
        hub = HubAlignment.from_warp(parsed, pixel_size_a=pix)
        owner = alignment_to_cets(
            hub,
            tilt_series_id=series_id,
            alignment_name=alignment_id,
            image=image_frame(images[0]),
            reference=ReferenceVolume.from_tomogram(reference),
            tilt_image_ids=dict(enumerate(ids)),
        )
        series.defocus_handedness = -1 if model.ts.are_angles_inverted else 1
        series.defocus_slope = 1.0
        series.nominal_tilt_axis_angle = float(torch.median(model.ts.tilt_axis_angles))
    elif source == "aretomo3":
        parsed = model._aln.aln
        hub = HubAlignment.from_aretomo3(parsed, vol_size_px=meta.volume_dims_px, pixel_size_a=pix)
        row_ids = {entry.z_index: ids[model._perm[i]] for i, entry in enumerate(hub.per_section_alignment_parameters)}
        owner = alignment_to_cets(
            hub,
            tilt_series_id=series_id,
            alignment_name=alignment_id,
            image=image_frame(images[0]),
            reference=ReferenceVolume.from_tomogram(reference),
            tilt_image_ids=row_ids,
        )
    elif source == "relion":
        bare = model.global_model
        owner = m.Alignment(tilt_series_id=series_id)
        centre = np.asarray(meta.volume_dims_px) // 2 * meta.pixel_size_volume_a
        image_centre = np.asarray(meta.image_dims_px) // 2 * pix
        matrices = bare.projection_matrices.detach().cpu().numpy()
        owner.projection_alignments = [
            _projection(p[:3, :3], (p[:3, :3] @ centre)[:2] + p[:2, 3] * pix - image_centre, ids[i], alignment_id)
            for i, p in enumerate(matrices)
        ]
        series.defocus_handedness = int(bare.hand)
        series.defocus_slope = float(bare.defocus_slope)
    else:
        raise ValueError(f"unsupported tilt-series source {source}")
    owner.id, owner.name, owner.reference_volume_id = alignment_id, alignment_id, reference.id
    owner.provenance = _provenance(source, files, options)
    if meta.projection_label is not None:
        owner.provenance.parameters.append(
            m.NativeParameter(name="native_image_labels", value_json=json.dumps(dict(zip(ids, meta.projection_label))))
        )
    if source == "aretomo3":
        owner.provenance.parameters.extend(
            [
                m.NativeParameter(name=name, value_json=json.dumps(float(getattr(parsed, name) or 0)))
                for name in ("AlphaOffset", "BetaOffset")
            ]
        )
    owner.tilt_angle_observations = [
        m.TiltAngleObservation(
            tilt_image_id=key, value_degrees=float(meta.projection_angle_deg[i]), angle_kind=kinds[i], source=source
        )
        for i, key in enumerate(ids)
    ]
    active = {p.tilt_image_id for p in owner.projection_alignments}
    owner.exclusions = [
        m.ProjectionExclusion(tilt_image_id=key, reason="excluded in native alignment")
        for key in ids
        if key not in active
    ]
    region = m.Region(id=region_id, tilt_series=[series], tomograms=[reference], alignments=[owner])
    document = m.Dataset(name=series_id, regions=[region])
    return document, list(range(len(ids)))


def _movie_document(source, ir, files, options, series_id, alignment_id, region_id):
    meta = ir.meta
    count = ir.n_projections
    ids = [f"{series_id}_{i}" for i in range(count)]
    rows = list(range(count))
    frames = [
        m.MovieFrame(id=key, section=i, acquisition_order=i, source_start_index=i, source_frame_count=1)
        for i, key in enumerate(ids)
    ]
    gauge, reference, raw_count = "native-origin", None, count
    if source == "mcaln":
        from cets_nonrigid.io.motion_txt import McAln

        parsed = McAln.from_string(files["mcaln"].decode())
        ids = [f"{series_id}_{fr.integrated_index}" for fr in parsed.frames]
        frames = [
            m.MovieFrame(
                id=ids[i],
                section=fr.integrated_index,
                acquisition_order=i,
                source_start_index=fr.source_start,
                source_frame_count=fr.source_count,
            )
            for i, fr in enumerate(parsed.frames)
        ]
        rows = [
            next(i for i, fr in enumerate(parsed.frames) if fr.aligned_index == f and fr.included) for f in range(count)
        ]
        gauge, reference, raw_count = "reference-frame", ids[rows[parsed.fm_ref]], parsed.raw_frame_count
    stack = _frame(
        m.MovieStack(id=series_id, images=frames, raw_frame_count=raw_count),
        meta.image_dims_px,
        meta.pixel_size_image_a,
    )
    native_global, _ = ir.native_model.map_image_global(torch.zeros((1, 2), dtype=torch.float64))
    frame_available = getattr(ir.native_model, "frame_valid", torch.ones(count, dtype=torch.bool))
    owner = m.MovieAlignment(
        id=alignment_id,
        name=alignment_id,
        profile_version="cets-nonrigid/0.1",
        movie_stack_id=series_id,
        frame_ids=ids,
        gauge=gauge,
        reference_frame_id=reference,
        provenance=_provenance(source, files, options),
        frame_alignments=[
            m.FrameAlignment(
                frame_id=ids[rows[i]],
                transform=m.Translation(translation=native_global[i, 0].to(torch.float64).tolist()),
            )
            for i in range(count)
            if bool(frame_available[i])
        ],
    )
    stack.alignments = [owner]
    region = m.Region(
        id=region_id,
        movie_stack_collection=m.MovieStackCollection(
            movie_stacks=[m.MovieStackSeries(id=series_id + "_series", stacks=[stack])]
        ),
    )
    return m.Dataset(name=series_id, regions=[region]), rows


def _particle_annotation(context, ir):
    # Native indices retain original ordering through the held-out split.
    total = ir.n_points + len(ir.heldout_points)
    points = torch.empty((total, 3), dtype=torch.float64)
    names = [None] * total
    for p, ix, labels in (
        (ir.points, ir.point_index, ir.point_names),
        (ir.heldout_points, ir.heldout_point_index, ir.heldout_point_names),
    ):
        points[ix] = p.to(torch.float64)
        for i, index in enumerate(ix.tolist()):
            names[index] = str(labels[i]) if labels is not None else f"point_{index}"
    if len(set(names)) != total:
        raise ValueError("particle identifiers must be unique")
    annotation = point_set_entity(
        annotation_id=context.alignment_id + "_particles",
        tomogram_id=context.reference_entity.id,
        points_a=points.numpy() - context.reference_center_a,
    )
    annotation.point_ids = names
    if ir.meta.source_tool == "relion5" and ir.native_data is not None:
        for name, values in getattr(ir.native_data, "point_attributes", {}).items():
            annotation.point_attributes.append(m.PointAttribute(name=name, numeric_values=values))
    context.resolve()[0].annotations.append(annotation)


def load_native(
    source: str, path=None, *, series_id=None, alignment_id=None, region_id="region_0", **options
) -> NativeAlignment:
    """Load one native alignment. Source options match the validated native readers.

    Geometry may come from native files or explicit arguments; missing required
    context is an error. The returned CETS context already owns the rigid baseline.
    """
    from cets_nonrigid import convert_store as loaders

    source = {"aretomo": "aretomo3", "movie-xml": "warp-movie"}.get(source, source)
    functions = {
        "warp": loaders.dump_ir_warp,
        "aretomo3": loaders.dump_ir_aretomo,
        "relion": loaders.dump_ir_relion,
        "warp-movie": loaders.dump_ir_warp_movie,
        "mcaln": loaders.dump_ir_mcaln,
        "relion-motion": loaders.dump_ir_relion_motion,
    }
    if source not in functions:
        raise ValueError(f"unknown source {source!r}; choose {', '.join(functions)}")
    from cets_nonrigid.inputs import prepare_native_options, apply_discovered_metadata

    options, discovered, input_options = prepare_native_options(source, path, options)
    extra = {
        key: options.pop(key)
        for key in ("ctf_file", "voltage_kv", "cs_mm", "amplitude_contrast", "defocus_handedness", "defocus_slope")
        if key in options
    }
    if source == "relion":
        if isinstance(options.get("overrides"), dict):
            from cets_nonrigid.convert_relion import RelionSourceOverrides

            options["overrides"] = RelionSourceOverrides(**options["overrides"])
        if path is not None:
            options.setdefault("optimisation_set", path)
        ir, files = functions[source](**options)
    else:
        if path is None:
            raise ValueError(f"{source} requires a native input path")
        ir, files = functions[source](path, **options)
    if input_options.get("particles") is not None:
        if source not in {"warp", "aretomo3"}:
            raise ValueError(
                "independent particle picks are supported for Warp and AreTomo3 sources; RELION uses particles_star"
            )
        from cets_nonrigid.inputs import load_series_particles
        from cets_nonrigid.cli.specs import parse_particles_spec, parse_particles_voxel

        spec = input_options["particles"]
        spec = parse_particles_spec(spec) if isinstance(spec, (str, Path)) else spec
        voxel = input_options.get("particles_voxel")
        voxel = parse_particles_voxel(str(voxel)) if voxel is not None else None
        positions, names = load_series_particles(
            stem=Path(path).stem,
            tomo_name=series_id or Path(path).stem,
            tomo_dims_px=ir.meta.volume_dims_px,
            pixel_size_a=ir.meta.pixel_size_image_a,
            raw_x_px=ir.meta.image_dims_px[0],
            spec=spec,
            voxel=voxel,
            no_particles=False,
            n_sources=1,
            portal=input_options.get("portal_context"),
        )
        options.update(positions_a=positions, names=names)
        ir, files = functions[source](path, **options)
    series_id = (
        series_id
        or (Path(path).stem if path is not None and source != "relion" else ir.meta.series_name)
        or options.get("tomo_name", "series")
    )
    alignment_id = alignment_id or f"{series_id}_{source}"
    builder = _tilt_document if ir.meta.kind == "tilt_series" else _movie_document
    document, rows = builder(source, ir, files, options, series_id, alignment_id, region_id)
    context = CetsContext(document, alignment_id, region_id)
    if ir.meta.sampling == "particles":
        _particle_annotation(context, ir)
    result = NativeAlignment(source, ir, context, files, dict(options), rows)
    from cets_nonrigid.metadata import discover_native_metadata

    discover_native_metadata(result, {**input_options, **extra})
    apply_discovered_metadata(result, discovered)
    context.validate()
    return result
