"""Transient native-frame views for the ported fitters; never a wire format."""

from __future__ import annotations

import torch
from cets_nonrigid.ir.core import IRMeta, IRTiltSeries, IRFrameSeries
from cets_nonrigid.samples import AlignmentBundle


def runtime_ir(bundle: AlignmentBundle, *, alignment_id=None, region_id=None, compress_movie_rows=False):
    bundle.validate()
    context = bundle.get_context(alignment_id, region_id)
    data = bundle.deformations.get(context.key)
    if data is None:
        raise ValueError("alignment has no sampled non-rigid component")
    frames = context.image_frames
    if any(f != frames[0] for f in frames):
        raise ValueError("native fitters require common image dimensions and spacing")
    owner, parent = context.owner, context.parent
    rows = context.rows
    _, _, active = context.operators()
    if context.ndim == 2 and compress_movie_rows:
        indices = [i for i, a in enumerate(active) if a]
    else:
        indices = list(range(len(rows)))
    chosen = [rows[i] for i in indices]
    import json

    native_labels = {}
    for parameter in owner.provenance.parameters if owner.provenance else []:
        if parameter.name == "native_image_labels":
            native_labels = json.loads(parameter.value_json)
    labels = [native_labels.get(row.id, row.path or row.id) for row in chosen]
    angles = {a.tilt_image_id: a for a in (owner.tilt_angle_observations or [])} if context.ndim == 3 else {}
    from cryoet_alignment.io.cets.rotation import decompose

    operators = context.operators()[0]
    angle_values, angle_kinds = [], []
    for i in indices:
        row = rows[i]
        if row.nominal_tilt_angle is not None:
            angle_values.append(row.nominal_tilt_angle)
            angle_kinds.append("nominal")
        elif row.id in angles:
            angle_values.append(angles[row.id].value_degrees)
            angle_kinds.append(str(getattr(angles[row.id].angle_kind, "value", angles[row.id].angle_kind)))
        else:
            angle_values.append(decompose(operators[i])[1] if active[i] and context.ndim == 3 else 0.0)
            angle_kinds.append("effective" if active[i] else "unknown")
    metadata = IRMeta(
        kind="tilt_series" if context.ndim == 3 else "frame_series",
        series_name=parent.id,
        frame="native-corner-runtime",
        pixel_size_image_a=frames[0].isotropic_spacing,
        image_dims_px=tuple(frames[0].size_px),
        volume_dims_px=tuple(context.reference_frame.size_px) if context.ndim == 3 else None,
        pixel_size_volume_a=context.reference_frame.isotropic_spacing if context.ndim == 3 else None,
        projection_index=list(range(len(indices))),
        projection_valid=[bool(active[i]) for i in indices],
        projection_order=[r.acquisition_order if r.acquisition_order is not None else i for i, r in enumerate(chosen)],
        projection_dose=[r.accumulated_dose if r.accumulated_dose is not None else 0.0 for r in chosen],
        projection_angle_deg=angle_values if context.ndim == 3 else None,
        projection_sec=[(r.section + 1) if r.section is not None else -1 for r in chosen]
        if context.ndim == 3
        else None,
        projection_dark=[not bool(active[i]) for i in indices] if context.ndim == 3 else None,
        source_tool=owner.provenance.software_name if owner.provenance else "cets",
        sampling=data.sampling.kind,
        projection_label=labels,
        projection_angle_kind=angle_kinds if context.ndim == 3 else None,
        displacement_3d=data.channels.displacement_3d,
        source_ctf_depth=data.channels.ctf_depth,
    )
    available_channels = {}
    for channel, mask_name, metadata_name in (
        ("displacement_3d", "displacement_valid", "displacement_3d"),
        ("ctf_depth", "ctf_depth_valid", "source_ctf_depth"),
    ):
        state = getattr(data.channels, channel)
        available = state != "present" or all(
            getattr(block, mask_name)[[i for i in indices if active[i]]][:, block.sample_valid].all()
            for block in (data.training, data.heldout)
        )
        available_channels[channel] = bool(available)
        if not available:
            # Partial channels remain in CETS. These legacy numerical kernels
            # require complete support; operations using the channel gate it explicitly.
            setattr(metadata, metadata_name, "none")
    values = dict(meta=metadata, grid_shape=tuple(data.sampling.grid_shape) if data.sampling.kind == "grid" else None)
    for prefix, block in (("", data.training), ("heldout_", data.heldout)):
        centres = torch.tensor(context.image_centers_a, dtype=torch.float64)[:, None]
        values[prefix + "points"] = block.points + torch.tensor(context.reference_center_a, dtype=torch.float64)
        # Complete observations retain unavailable NaNs only where the native fitters
        # already use validity. Inactive rows use zero placeholders, matching their contract.
        q = block.observations(context) + centres
        q[~torch.tensor(active)] = 0
        values[prefix + "source_projected"] = q[indices]
        values[prefix + "source_projected_global"] = (context.evaluate_global(block.points) + centres)[indices]
        values[prefix + "projection_valid"] = block.projection_valid[indices].clone()
        values[prefix + "weights"] = block.weights[indices].clone()
        if context.ndim == 3:
            for name, old_name in (("displacement_3d", "source_displacement_3d"), ("ctf_depth", "source_ctf_depth_a")):
                array = getattr(block, name)
                values[prefix + old_name] = (
                    array[indices].clone() if array is not None and available_channels[name] else None
                )
        if data.sampling.kind == "particles":
            annotation = next(a for a in context.resolve()[0].annotations if a.id == data.sampling.annotation_id)
            assert block.point_ids is not None  # particle binding validated by the bundle
            values[prefix + "point_names"] = list(block.point_ids)
            values[prefix + "point_index"] = torch.tensor(
                [annotation.point_ids.index(key) for key in block.point_ids], dtype=torch.int64
            )
    values["sample_valid"] = data.training.sample_valid.clone()
    cls = IRTiltSeries if context.ndim == 3 else IRFrameSeries
    return cls(**values)
