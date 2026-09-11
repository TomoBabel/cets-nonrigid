"""RELION project fitting from CETS observations and CETS scientific metadata."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import tempfile
import numpy as np
import torch
from cryoet_alignment.io.cets.rotation import decompose

from cets_nonrigid.ctf import TiltCtf
from cets_nonrigid.fit.relion_traj import lift_particle_trajectories, LIFT_EXACT_MAX_PX
from cets_nonrigid.io.relion_star import RelionTomogramData
from cets_nonrigid.metadata import get_ctf, get_optics
from cets_nonrigid.models.relion_ts import RelionTomogramModel
from cets_nonrigid.project.relion import RelionProject


@dataclass
class RelionFitArtifacts:
    tomo: object
    particle_names: list[str] | None
    centered_coords_a: torch.Tensor | None
    motion_a: torch.Tensor | None
    tilt_series_uri: str | None
    lift: object | None
    point_attributes: dict | None = None


def fit_relion_bundle(
    bundle,
    *,
    tomo_name=None,
    no_particles=False,
    no_ctf=False,
    hand=None,
    trajectory_gauge="lowest-dose",
    micrograph_names=None,
    tilt_stack=None,
    tilt_image_list=None,
    voltage_kv=None,
    cs_mm=None,
    amplitude_contrast=None,
    tilt_series_uri=None,
    random_subset="hash",
    placeholder_stack=False,
):
    from cets_nonrigid.api import FitResult

    context = bundle.context
    if tilt_series_uri is None and context.owner.provenance is not None:
        import json

        for parameter in context.owner.provenance.parameters:
            if parameter.name == "tilt_series_uri":
                tilt_series_uri = json.loads(parameter.value_json)
    name = tomo_name or context.parent.id
    if Path(name).name != name or "\\" in name or name in {".", ".."}:
        raise ValueError("RELION tomogram name must be a single filename component")
    if context.ndim != 3:
        raise ValueError("RELION tomography requires a tilt-series alignment")
    data = bundle.deformations.get(context.key)
    if not no_particles and (data is None or data.sampling.kind != "particles"):
        raise ValueError(
            "RELION trajectories require a particle-bound payload: supply particles at to-cets time; "
            "a sampled grid cannot define trajectories at unsampled particles"
        )
    rotations, shifts, active = context.operators()
    rows = np.flatnonzero(active).tolist()
    if not rows:
        raise ValueError("no active projections to export")
    if any(context.rows[i].accumulated_dose is None for i in rows):
        raise ValueError("RELION export requires known per-image pre-exposure dose")
    dose = torch.tensor([context.rows[i].accumulated_dose for i in rows], dtype=torch.float64)
    pix = context.image_frames[0].isotropic_spacing
    image_dims = tuple(context.image_frames[0].size_px)
    extent = np.asarray(context.reference_frame.size_px) * context.reference_frame.isotropic_spacing / pix
    if not np.allclose(extent, np.rint(extent), atol=1e-3, rtol=0):
        raise ValueError("reference volume extent must be an integer number of RELION tilt-image pixels")
    volume_dims = tuple(np.rint(extent).astype(int).tolist())
    if any(f != context.image_frames[0] for f in context.image_frames):
        raise ValueError("RELION export requires common image frames")
    hand = context.parent.defocus_handedness if hand is None else hand
    if hand not in (-1, 1):
        raise ValueError("RELION export requires known defocus handedness or an explicit hand override")
    angle = torch.tensor([decompose(rotations[i]) for i in rows], dtype=torch.float64)
    zr, yt, xt = angle.T
    # RELION's projection matrices use integer centers. CETS reference spacing may differ.
    delta = np.asarray(volume_dims) // 2 * pix - context.reference_center_a
    target_shift = torch.tensor(
        shifts[rows] + np.einsum("tij,j->ti", rotations[rows], delta)[:, :2], dtype=torch.float64
    )
    slope = context.parent.defocus_slope if context.parent.defocus_slope is not None else 1.0
    model = RelionTomogramModel(
        xtilt_deg=xt,
        ytilt_deg=yt,
        zrot_deg=zr,
        xshift_a=target_shift[:, 0],
        yshift_a=target_shift[:, 1],
        tomo_dims_px=volume_dims,
        image_dims_px=image_dims,
        pixel_size_a=pix,
        hand=hand,
        defocus_slope=slope,
    )
    optics = get_optics(context)
    for key, value in (("voltage_kv", voltage_kv), ("cs_mm", cs_mm), ("amplitude_contrast", amplitude_contrast)):
        if value is not None:
            optics[key] = value
    if any(value is None for value in optics.values()):
        raise ValueError(
            "RELION export requires voltage, spherical aberration and amplitude contrast in CETS acquisition metadata or explicit options"
        )
    ctf = get_ctf(context, rows)
    if no_ctf:
        ctf = TiltCtf(
            torch.full((len(rows),), 20000.0),
            torch.full((len(rows),), 20000.0),
            torch.zeros(len(rows)),
            torch.zeros(len(rows)),
            **optics,
        )
    elif ctf is None:
        raise ValueError(
            "RELION export requires complete per-image CTF; no_ctf explicitly creates geometry-only placeholder CTF"
        )
    lift, names, point_attributes = None, None, {}
    if not no_particles:
        annotation = next(a for a in context.resolve()[0].annotations if a.id == data.sampling.annotation_id)
        names = list(annotation.point_ids)
        point_attributes = {
            a.name: list(a.numeric_values)
            for a in annotation.point_attributes
            if a.name in {"half_set", "class_number"}
        }
        count = len(names)
        points = torch.empty((count, 3), dtype=torch.float64)
        observations = torch.empty((len(rows), count, 2), dtype=torch.float64)
        displacement = (
            torch.empty((len(rows), count, 3), dtype=torch.float64)
            if data.channels.displacement_3d == "present"
            else None
        )
        depth = torch.empty((len(rows), count), dtype=torch.float64) if data.channels.ctf_depth == "present" else None
        centres = torch.tensor(context.image_centers_a[rows], dtype=torch.float64)[:, None]
        for block in (data.training, data.heldout):
            ix = [names.index(key) for key in block.point_ids]
            points[ix] = block.points + torch.tensor(context.reference_center_a, dtype=torch.float64)
            observations[:, ix] = block.observations(context)[rows] + centres
            if displacement is not None:
                if not block.displacement_valid[rows].all():
                    raise ValueError("trajectory lift requires available 3D displacement at every emitted particle")
                displacement[:, ix] = block.displacement_3d[rows].to(torch.float64)
            if depth is not None:
                if not block.ctf_depth_valid[rows].all():
                    raise ValueError("CTF-depth comparison requires available depth at every emitted particle")
                depth[:, ix] = block.ctf_depth[rows].to(torch.float64)
        lift = lift_particle_trajectories(
            observations,
            model,
            points,
            dose,
            source_disp_a=displacement,
            source_ctf_depth_a=depth,
            gauge=trajectory_gauge,
        )
        if lift.max_residual_px > LIFT_EXACT_MAX_PX:
            raise ValueError(
                f"trajectory lift residual {lift.max_residual_px} px exceeds validated bound {LIFT_EXACT_MAX_PX}"
            )
    observed_angles = {a.tilt_image_id: a.value_degrees for a in context.owner.tilt_angle_observations}
    nominal = [
        context.rows[i].nominal_tilt_angle
        if context.rows[i].nominal_tilt_angle is not None
        else observed_angles.get(context.rows[i].id, float(yt[k]))
        for k, i in enumerate(rows)
    ]
    assets = {}
    workspace = tempfile.TemporaryDirectory(prefix="cets-relion-fit-")
    try:
        root = Path(workspace.name)
        if micrograph_names is None and tilt_image_list is not None:
            from cets_nonrigid.convert_relion import _read_tilt_image_list

            micrograph_names = _read_tilt_image_list(Path(tilt_image_list), len(rows))
        if micrograph_names is None and tilt_stack is not None:
            from cets_nonrigid.convert_relion import _slice_tilt_stack

            absolute = _slice_tilt_stack(Path(tilt_stack), rows, root / "images" / name, pix)
            micrograph_names = [str(Path(name).relative_to(root)) for name in absolute]
        if placeholder_stack:
            if not tilt_series_uri:
                raise ValueError("placeholder stacks require an explicit tilt_series_uri for streaming")
            placeholder = RelionProject(root).placeholder_stack(
                nx=image_dims[0], ny=image_dims[1], nz=len(rows), pixel_size_a=pix
            )
            placeholder = placeholder.rename(placeholder.with_name(name + "_placeholder.mrcs"))
            micrograph_names = [f"{i + 1}@{placeholder.relative_to(root)}" for i in range(len(rows))]
        if micrograph_names is None:
            micrograph_names = [context.rows[i].path for i in rows]
            if any(name is None for name in micrograph_names):
                raise ValueError("RELION extraction requires real tilt-image paths, tilt_stack or micrograph_names")
        # Bind an existing CETS stack into the project under a RELION-compatible
        # .mrcs name. This avoids copying image data while fitting.
        stack_refs = [str(value).split("@", 1) for value in micrograph_names]
        if stack_refs and all(len(value) == 2 for value in stack_refs):
            paths = {value[1] for value in stack_refs}
            if len(paths) == 1 and Path(next(iter(paths))).is_file():
                asset_name = f"tilt_series/{name}.mrcs"
                assets[asset_name] = Path(next(iter(paths))).resolve()
                micrograph_names = [f"{value[0]}@{asset_name}" for value in stack_refs]
        if len(micrograph_names) != len(rows):
            raise ValueError("micrograph_names must cover the emitted active rows")
        tomo = RelionTomogramData(
            name=tomo_name or context.parent.id,
            hand=hand,
            pixel_size_a=pix,
            tomo_dims_px=volume_dims,
            image_dims_px=image_dims,
            xtilt_deg=xt,
            ytilt_deg=yt,
            zrot_deg=zr,
            xshift_a=target_shift[:, 0],
            yshift_a=target_shift[:, 1],
            pre_exposure=dose,
            nominal_stage_angle_deg=torch.tensor(nominal, dtype=torch.float64),
            ctf=ctf,
            micrograph_names=micrograph_names,
            defocus_slope=slope,
            **optics,
        )
        project = RelionProject(root, random_subset=random_subset)
        project.add_series(
            tomo,
            particle_names=names,
            centered_coords_a=lift.positions_out_a - torch.tensor(volume_dims, dtype=torch.float64) * pix / 2
            if lift is not None
            else None,
            motion_a=lift.motion_a if lift is not None else None,
            tilt_series_uri=tilt_series_uri,
            point_attributes=point_attributes,
        )
        project.flush()
        files = {}
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if path.suffix.lower() in {".mrc", ".mrcs"}:
                assets[relative] = path
            else:
                files[relative] = path.read_bytes()
    except BaseException:
        workspace.cleanup()
        raise
    metrics = {
        "source_context_digest": context.digest(),
        "nominal_angle_fallback_ids": [context.rows[i].id for i in rows if context.rows[i].nominal_tilt_angle is None],
        "n_particles": len(names) if names else 0,
        "n_tilts": len(rows),
        "hand": hand,
        "ctf_disposition": "geometry-only placeholders" if no_ctf else "CETS per-image CTF",
        "heldout_status": "evaluated" if data and data.heldout.count else "not_evaluated",
    }
    if lift is not None:
        metrics.update(
            {
                key: getattr(lift, key)
                for key in (
                    "max_residual_px",
                    "n_fallback",
                    "depth_source",
                    "gauge",
                    "ctf_depth_deviation_rms_a",
                    "ctf_depth_deviation_max_a",
                    "n_gauge_fallback",
                )
            }
        )
    return FitResult(
        "relion",
        files,
        "optimisation_set.star",
        metrics,
        RelionFitArtifacts(
            tomo,
            names,
            lift.positions_out_a - torch.tensor(volume_dims, dtype=torch.float64) * pix / 2
            if lift is not None
            else None,
            lift.motion_a if lift is not None else None,
            tilt_series_uri,
            lift,
            point_attributes,
        ),
        layout="project",
        assets=assets,
        workspace=workspace,
    )
