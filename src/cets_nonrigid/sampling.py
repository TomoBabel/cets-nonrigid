"""Sample complete native observations relative to document-evaluated globals."""

from __future__ import annotations

import hashlib
import numpy as np
import torch
from cets_data_model.models import models as m

from cets_nonrigid.context import CetsContext, canonical_json
from cets_nonrigid.native import NativeAlignment
from cets_nonrigid.samples import AlignmentBundle, DeformationSamples, SampleBlock


def geometry_digest(context):
    record = context.digest_record()
    for key in ("sampling", "heldout", "channels", "particle_binding"):
        record.pop(key, None)
    return hashlib.sha256(canonical_json(record)).hexdigest()


def _native_observations(native):
    ir, model = native.ir, native.ir.native_model
    full = model.project_volume if native.context.ndim == 3 else model.map_image
    global_only = model.project_volume_global if native.context.ndim == 3 else model.map_image_global
    if ir.meta.sampling == "particles":
        positions = torch.empty((ir.n_points + len(ir.heldout_points), 3), dtype=torch.float64)
        positions[ir.point_index] = ir.points
        positions[ir.heldout_point_index] = ir.heldout_points
        q, valid = full(positions)
        g, _ = global_only(positions)
        return [(q[:, ix], valid[:, ix], g[:, ix]) for ix in (ir.point_index, ir.heldout_point_index)]
    observations = []
    for points in (ir.points, ir.heldout_points):
        q, valid = full(points)
        observations.append((q, valid, global_only(points)[0]))
    return observations


def sample(
    native: NativeAlignment,
    context: CetsContext | None = None,
    *,
    row_map=None,
    baseline_tolerance_a: float | None = None,
    annotation_id: str | None = None,
) -> DeformationSamples:
    """Sample training and held-out observations, checking the native rigid split.

    ``row_map`` maps native CETS image/frame IDs to supplied context IDs. A converter
    may supply its own existing context; folded Warp constants fail the baseline gate.
    """
    context = context or native.context
    context.validate()
    if context.ndim != native.context.ndim:
        raise ValueError("native and CETS alignment kinds differ")
    if context.reference_frame.size_px != native.context.reference_frame.size_px or not np.allclose(
        context.reference_frame.spacing_a, native.context.reference_frame.spacing_a, atol=1e-12, rtol=0
    ):
        raise ValueError("native and CETS reference geometry disagree")
    mapping = row_map or {key: key for key in native.context.row_ids}
    if set(mapping) != set(native.context.row_ids) or set(mapping.values()) != set(context.row_ids):
        raise ValueError("row_map must bijectively identify every native and CETS image")
    destination = [context.row_ids.index(mapping[native.context.row_ids[i]]) for i in native.model_rows]
    for src_i, dest_i in enumerate(destination):
        a, b = native.context.image_frames[native.model_rows[src_i]], context.image_frames[dest_i]
        if a != b:
            raise ValueError("native and CETS image frames disagree")
    native_active = native.context.operators()[2]
    expected_active: np.ndarray = np.zeros(len(context.row_ids), dtype=bool)
    for i, key in enumerate(native.context.row_ids):
        expected_active[context.row_ids.index(mapping[key])] = native_active[i]
    if not np.array_equal(expected_active, context.operators()[2]):
        raise ValueError("native and CETS alignment activity disagree")
    ir = native.ir
    channels = m.NonRigidChannels(
        displacement_3d=ir.meta.displacement_3d if context.ndim == 3 else "none",
        ctf_depth=ir.meta.source_ctf_depth if context.ndim == 3 else "none",
    )
    t = len(context.row_ids)
    blocks, baseline_errors = [], []
    for held, (native_q, native_valid, native_g) in zip((False, True), _native_observations(native)):
        prefix = "heldout_" if held else ""
        native_points = getattr(ir, prefix + "points")
        points = native_points.to(torch.float64) - torch.tensor(context.reference_center_a, dtype=torch.float64)
        g = context.evaluate_global(points)
        centre = torch.tensor(context.image_centers_a[destination], dtype=torch.float64)[:, None]
        q = native_q.to(torch.float64) - centre
        global_q = native_g.to(torch.float64) - centre
        active = torch.tensor(context.operators()[2][destination])
        available = torch.isfinite(q).all(-1) & active[:, None]
        baseline_available = torch.isfinite(global_q).all(-1) & active[:, None]
        if not torch.equal(baseline_available, active[:, None].expand_as(baseline_available)):
            raise ValueError("native rigid baseline is nonfinite on active samples")
        errors = (global_q - g[destination]).abs()[baseline_available]
        max_error = float(errors.max()) if errors.numel() else 0.0
        magnitude = max(
            1.0,
            float(native_g.abs().max()) if native_g.numel() else 1.0,
            float(native_points.abs().max()) if native_points.numel() else 1.0,
        )
        tolerance = (
            baseline_tolerance_a
            if baseline_tolerance_a is not None
            else max(1e-6, 8 * torch.finfo(torch.float32).eps * magnitude)
        )
        if max_error > tolerance:
            raise ValueError(
                f"native rigid baseline differs from CETS G by {max_error:.6g} Å (bound {tolerance:.6g}); "
                "use a new alignment with the native rigid baseline; do not attach to folded Warp constants"
            )
        baseline_errors.append(
            {
                "status": "evaluated" if len(points) else "not_evaluated",
                "maximum_a": max_error if len(points) else None,
                "tolerance_a": tolerance if len(points) else None,
            }
        )
        count = len(points)
        residual = torch.zeros((t, count, 2), dtype=torch.float32)
        observation_valid = torch.zeros((t, count), dtype=torch.bool)
        fit_valid = torch.zeros_like(observation_valid)
        weights = torch.zeros((t, count), dtype=torch.float32)
        residual[destination] = torch.where(available[..., None], q - g[destination], 0).to(torch.float32)
        observation_valid[destination] = available
        fit_valid[destination] = native_valid & available
        weights[destination] = torch.nan_to_num(getattr(ir, prefix + "weights").to(torch.float32))
        sample_valid = torch.ones(count, dtype=torch.bool) if held else ir.sample_valid.clone()
        observation_valid &= sample_valid[None]
        fit_valid &= observation_valid
        residual[~observation_valid] = 0
        block = SampleBlock(points, residual, sample_valid, observation_valid, fit_valid, weights)
        for name, old_name, state, mask_name, ndim in (
            ("displacement_3d", "source_displacement_3d", channels.displacement_3d, "displacement_valid", 3),
            ("ctf_depth", "source_ctf_depth_a", channels.ctf_depth, "ctf_depth_valid", 1),
        ):
            if state == "none":
                continue
            mask = torch.zeros((t, count), dtype=torch.bool)
            if state == "zero_at_samples":
                mask[destination] = sample_valid[None].expand(len(destination), -1)
            else:
                value = getattr(ir, prefix + old_name)
                valid = torch.isfinite(value).all(-1) if ndim == 3 else torch.isfinite(value)
                mask[destination] = valid & sample_valid[None]
                array = torch.zeros((t, count, 3) if ndim == 3 else (t, count), dtype=torch.float32)
                array[destination] = torch.nan_to_num(value).to(torch.float32)
                array[~mask] = 0
                setattr(block, name, array)
            setattr(block, mask_name, mask)
        if ir.meta.sampling == "particles":
            annotation = next(
                a for a in native.context.resolve()[0].annotations if a.id == native.context.alignment_id + "_particles"
            )
            indices = ir.heldout_point_index if held else ir.point_index
            block.point_ids = [annotation.point_ids[i] for i in indices.tolist()]
        blocks.append(block)
    is_particles = ir.meta.sampling == "particles"
    descriptor = (
        m.ParticleSampling(annotation_id=annotation_id or native.context.alignment_id + "_particles")
        if is_particles
        else m.GridSampling(grid_shape=list(ir.grid_shape))
    )
    heldout = m.HeldoutSampling(
        method="particle-subset" if is_particles else "scrambled-sobol",
        seed=native.options.get("heldout_seed", 20260828),
        count=blocks[1].count,
        fraction=native.options.get("heldout_fraction", 0.2) if is_particles else None,
        min_count=16 if is_particles else None,
    )
    result = DeformationSamples(
        blocks[0],
        blocks[1],
        descriptor,
        heldout,
        channels,
        context_fingerprint=geometry_digest(context),
        diagnostics={"native_global_baseline": dict(zip(("training", "heldout"), baseline_errors))},
    )
    result.validate(context)
    return result


def attach_deformation(
    context: CetsContext, data: DeformationSamples, *, bundle: AlignmentBundle | None = None
) -> AlignmentBundle:
    """Return a new document version, leaving the supplied document and payload intact."""
    if data.context_fingerprint != geometry_digest(context):
        raise ValueError("sample context changed before attachment; resample against the intended CETS alignment")
    if context.owner.non_rigid_alignment is not None:
        raise ValueError("alignment already has a non-rigid component; create a new alignment instance")
    data.validate(context)
    document = context.document.model_copy(deep=True)
    target = CetsContext(document, context.alignment_id, context.region_id)
    target.owner.non_rigid_alignment = m.NonRigidAlignment(
        profile_version="cets-nonrigid/0.1",
        kind=context.kind,
        tilt_image_ids=context.row_ids if context.ndim == 3 else [],
        payload_uri="pending.nonrigid.zarr",
        payload_group=("alignments/" if context.ndim == 3 else "movies/") + "pending",
        sampling=data.sampling.model_copy(deep=True),
        heldout=data.heldout_sampling.model_copy(deep=True),
        channels=data.channels.model_copy(deep=True),
        context_digest="0" * 64,
        digest_version=1,
    )
    target.owner.non_rigid_alignment.context_digest = target.digest(
        point_ids=data.training.point_ids, heldout_point_ids=data.heldout.point_ids
    )
    deformations = dict(bundle.deformations) if bundle is not None else {}
    reports = dict(bundle.reports) if bundle is not None else {}
    deformations[target.key] = data
    reports["/".join(target.key)] = data.diagnostics
    result = AlignmentBundle(
        document,
        deformations,
        reports,
        snapshots=dict(bundle.snapshots) if bundle is not None else {},
        selected_alignment=target.key,
    )
    result.validate()
    return result
