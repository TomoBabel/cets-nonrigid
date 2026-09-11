"""Public CETS-native conversion API. Core wire types come from cets_data_model."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import tempfile
import torch

from cets_nonrigid.bundle_io import read_bundle, write_bundle
from cets_nonrigid.context import CetsContext
from cets_nonrigid.native import NativeAlignment, load_native
from cets_nonrigid.samples import AlignmentBundle, DeformationSamples, SampleBlock
from cets_nonrigid.sampling import attach_deformation, sample

__all__ = [
    "CetsContext",
    "NativeAlignment",
    "DeformationSamples",
    "SampleBlock",
    "AlignmentBundle",
    "FitResult",
    "load_native",
    "sample",
    "attach_deformation",
    "fit",
    "read_bundle",
    "write_bundle",
    "export_native",
    "to_cets",
    "convert",
    "inspect_alignment",
    "validate_bundle",
    "prepare_project",
    "merge_projects",
    "reorder_rows",
    "with_fit_report",
]


@dataclass
class FitResult:
    target: str
    files: dict[str, bytes]
    primary_file: str
    metrics: dict = field(default_factory=dict)
    native_result: object | None = field(default=None, repr=False)
    layout: str = "file"
    assets: dict[str, Path] = field(default_factory=dict, repr=False)
    workspace: object | None = field(default=None, repr=False)


def to_cets(source, path=None, *, output=None, include_native_files=True, **options):
    native = load_native(source, path, **options)
    bundle = attach_deformation(native.context, sample(native))
    if include_native_files:
        bundle.snapshots[native.context.key] = dict(native.native_files)
    if output is not None:
        write_bundle(bundle, output)
    return bundle


def fit(
    bundle: AlignmentBundle | str | Path, target: str, *, alignment_id=None, region_id=None, **options
) -> FitResult:
    """Fit in scratch space and return reviewable native files without publishing them."""
    from cets_nonrigid import convert_store as fitting

    if not isinstance(bundle, AlignmentBundle):
        bundle = read_bundle(bundle)
    bundle.validate()
    if alignment_id is not None or region_id is not None:
        from dataclasses import replace

        selection = bundle.get_context(alignment_id, region_id)
        bundle = replace(bundle, selected_alignment=selection.key)
    context = bundle.context
    targets = {
        "warp": (fitting.fit_store_to_warp, ".xml", 3),
        "aretomo3": (fitting.fit_store_to_aretomo, ".aln", 3),
        "warp-movie": (fitting.fit_store_to_movie_xml, ".xml", 2),
        "mcaln": (fitting.fit_store_to_mcaln, ".mcaln", 2),
        "relion-motion": (fitting.fit_store_to_relion_motion, ".star", 2),
    }
    if target == "relion":
        from cets_nonrigid.relion_export import fit_relion_bundle

        return fit_relion_bundle(bundle, **options)
    if target not in targets:
        raise ValueError(f"unsupported target {target!r}")
    function, suffix, ndim = targets[target]
    if target == "warp":
        if options.get("template_xml") is not None:
            from cets_nonrigid.io.warp_xml import load_warp_tiltseries

            template = load_warp_tiltseries(options["template_xml"]).ts
            frames = (template.image_dimensions_physical, template.volume_dimensions_physical)
            expected = (context.image_frames[0], context.reference_frame)
            for actual, frame in zip(frames, expected):
                extent = torch.tensor(frame.size_px, dtype=torch.float64) * frame.isotropic_spacing
                if not torch.allclose(actual.to(torch.float64), extent, atol=1e-3, rtol=1e-7):
                    raise ValueError(
                        "Warp target template geometry differs from CETS; resampling into a different frame is outside profile 0.1"
                    )
        if "angles_inverted" not in options and context.parent.defocus_handedness is not None:
            options["angles_inverted"] = context.parent.defocus_handedness == -1
        if options.get("volume_warp_grid") is not None:
            data = bundle.deformations[context.key]
            active = torch.as_tensor(context.operators()[2])
            for block in (data.training, data.heldout):
                if (
                    data.channels.displacement_3d != "present"
                    or block.displacement_valid is None
                    or not block.displacement_valid[active][:, block.sample_valid].all()
                ):
                    raise ValueError(
                        "volume-warp fitting requires available 3D displacement on active training and held-out samples"
                    )
    if context.ndim != ndim:
        raise ValueError("target and CETS alignment kinds differ")
    if context.ndim == 2:
        if target == "relion-motion":
            from cets_nonrigid.metadata import get_optics
            import json

            doses = [row.exposure_dose for row in context.rows]
            if doses and all(d == doses[0] for d in doses) and doses[0] is not None:
                options.setdefault("dose_rate", doses[0])
            if context.rows and context.rows[0].accumulated_dose is not None:
                options.setdefault("pre_exposure", context.rows[0].accumulated_dose)
            if get_optics(context)["voltage_kv"] is not None:
                options.setdefault("voltage_kv", get_optics(context)["voltage_kv"])
            if context.parent.path:
                options.setdefault("movie_name", context.parent.path)
            for parameter in context.owner.provenance.parameters if context.owner.provenance else []:
                if parameter.name in {"eer_grouping", "eer_upsampling"}:
                    options.setdefault(parameter.name, json.loads(parameter.value_json))
        if target == "mcaln" and context.owner.reference_frame_id is not None:
            active_ids = [key for key, valid in zip(context.row_ids, context.operators()[2]) if valid]
            options.setdefault("fm_ref", active_ids.index(context.owner.reference_frame_id))
    if "store_out" in options:
        raise ValueError("fit returns files and metrics; use write_bundle for CETS exchange")
    with tempfile.TemporaryDirectory(prefix="cets-nonrigid-fit-") as scratch:
        name = context.parent.id + suffix
        if target == "warp-movie" and options.get("movie_path"):
            name = Path(options["movie_path"]).stem + suffix
        if Path(name).name != name or "\\" in name:
            raise ValueError("native output identity must be a single filename component")
        output = Path(scratch) / name
        native_result = function(bundle, output, **options)
        from cets_nonrigid.native_output import apply_scientific_metadata

        metadata_metrics = apply_scientific_metadata(bundle, target, native_result, output, options)
        files = {p.relative_to(scratch).as_posix(): p.read_bytes() for p in Path(scratch).rglob("*") if p.is_file()}
        metrics = dict(metadata_metrics)
        metrics["source_context_digest"] = context.digest()
        for attr in (
            "global_check_rms_px",
            "global_rms_px_train",
            "global_rms_px_heldout",
            "global_validation_status",
            "template_source",
        ):
            value = getattr(native_result, attr, None)
            if value is not None:
                metrics[attr] = value
        numerical_fit = getattr(native_result, "fit", getattr(native_result, "local_fit", None))
        for attr in (
            "rms_a_train",
            "rms_a_heldout",
            "rms_px_train",
            "rms_px_heldout",
            "coverage_heldout",
            "heldout_status",
        ):
            if numerical_fit is not None and hasattr(numerical_fit, attr):
                metrics[attr] = getattr(numerical_fit, attr)
        if target == "warp":
            from cets_nonrigid.fit.warp_ts_fit import volume_warp_fit_attrs

            metrics.update(volume_warp_fit_attrs(native_result.fit.volume_warp))
            from cets_nonrigid.native_output import depth_comparison

            metrics.update(depth_comparison(bundle, native_result))
        return FitResult(target, files, name, metrics, native_result)


def export_native(result: FitResult, output) -> Path:
    """Publish a new native file or project directory, refusing existing outputs."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if result.layout == "file":
        primary = Path(result.primary_file)
        targets = {}
        for name, data in result.files.items():
            path = Path(name)
            if path.parent != Path("."):
                raise ValueError("file output cannot contain project subdirectories")
            if name == result.primary_file:
                target_path = output
            elif path.name.startswith(primary.stem):
                target_path = output.with_name(output.stem + path.name[len(primary.stem) :])
            else:
                raise ValueError("companion output has no unambiguous primary-file association")
            if target_path.exists():
                raise FileExistsError(f"refusing to replace {target_path}")
            targets[target_path] = data
        published = []
        try:
            # Publish companions first and the primary file last, using no-overwrite links.
            for target_path in sorted(targets, key=lambda p: p == output):
                fd, temporary = tempfile.mkstemp(prefix="." + target_path.name, dir=output.parent)
                try:
                    with os.fdopen(fd, "wb") as handle:
                        handle.write(targets[target_path])
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.link(temporary, target_path)
                    published.append(target_path)
                finally:
                    Path(temporary).unlink(missing_ok=True)
        except Exception:
            for path in published:
                path.unlink()
            raise
    else:
        for name in set(result.files) | set(result.assets):
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("native output file must remain within its project")
        output.mkdir(exist_ok=False)
        try:
            for name, data in result.files.items():
                path = output / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as handle:
                    handle.write(data)
            import shutil

            for name, source in result.assets.items():
                path = output / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with Path(source).open("rb") as src, path.open("xb") as dst:
                    shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        except Exception:
            import shutil

            shutil.rmtree(output)
            raise
    return output


def convert(source, target, path=None, *, output=None, source_options=None, target_options=None):
    """Every native-to-native exchange passes through a complete CETS bundle."""
    bundle = to_cets(source, path, **(source_options or {}))
    result = fit(bundle, target, **(target_options or {}))
    if output is not None:
        export_native(result, output)
    return result


def inspect_alignment(path, *, source=None, **options):
    """Return a JSON-compatible description from CETS-owned context."""
    bundle = to_cets(source, path, **options) if source is not None else read_bundle(path)
    from cets_nonrigid.context import alignment_contexts

    alignments = []
    for context in alignment_contexts(bundle.document):
        descriptor = context.owner.non_rigid_alignment
        entry = {
            "region_id": context.key[0],
            "alignment_id": context.alignment_id,
            "has_non_rigid_alignment": descriptor is not None,
            "kind": context.kind,
            "parent_id": context.parent.id,
        }
        if descriptor is not None:
            data = bundle.deformations[context.key]
            entry.update(
                {
                    "row_ids": context.row_ids,
                    "active_rows": int(context.operators()[2].sum()),
                    "training_count": data.training.count,
                    "heldout_count": data.heldout.count,
                    "heldout_status": "evaluated" if data.heldout.count else "not_evaluated",
                    "sampling": data.sampling.kind,
                    "channels": data.channels.model_dump(mode="json"),
                    "available_observations": int(data.training.observation_valid.sum()),
                    "fit_valid_observations": int(data.training.projection_valid.sum()),
                    "context_digest": descriptor.context_digest,
                }
            )
        alignments.append(entry)
    return {"profile_version": "cets-nonrigid/0.1", "alignments": alignments}


def validate_bundle(path):
    bundle = path if isinstance(path, AlignmentBundle) else read_bundle(path)
    bundle.validate()
    return {
        "status": "pass",
        "non_rigid_alignments": len(bundle.deformations),
        "checks": ["core references", "canonical geometry", "array shapes and availability", "context digests"],
        "native_binary_validation": "not_evaluated",
    }


def report_bundle(path):
    bundle = path if isinstance(path, AlignmentBundle) else read_bundle(path)
    bundle.validate()
    return {"validation": validate_bundle(bundle), "alignments": bundle.reports}


def prepare_project(bundle, result, **options):
    from cets_nonrigid.project_export import prepare_project as prepare

    return prepare(bundle, result, **options)


def merge_projects(results):
    from cets_nonrigid.project_export import merge_projects as merge

    return merge(results)


def reorder_rows(bundle, row_ids, *, alignment_id=None, region_id=None):
    """Return a new bundle with every row-indexed array permuted consistently."""
    from dataclasses import replace
    import torch
    from cets_nonrigid.sampling import geometry_digest

    bundle.validate()
    context = bundle.get_context(alignment_id, region_id)
    if len(row_ids) != len(context.row_ids) or set(row_ids) != set(context.row_ids):
        raise ValueError("new row order must contain every image identity exactly once")
    data = bundle.deformations[context.key]
    order = torch.tensor([context.row_ids.index(key) for key in row_ids], dtype=torch.int64)
    blocks = []
    for block in (data.training, data.heldout):
        updates = {}
        for name in (
            "projected_residual",
            "observation_valid",
            "projection_valid",
            "weights",
            "displacement_3d",
            "displacement_valid",
            "ctf_depth",
            "ctf_depth_valid",
        ):
            value = getattr(block, name)
            if value is not None:
                updates[name] = value[order].clone()
        blocks.append(replace(block, **updates))
    document = bundle.document.model_copy(deep=True)
    target = CetsContext(document, context.alignment_id, context.region_id)
    descriptor = target.owner.non_rigid_alignment
    if target.ndim == 3:
        descriptor.tilt_image_ids = list(row_ids)
    else:
        target.owner.frame_ids = list(row_ids)
    data = replace(data, training=blocks[0], heldout=blocks[1], context_fingerprint=geometry_digest(target))
    descriptor.context_digest = target.digest(
        point_ids=data.training.point_ids, heldout_point_ids=data.heldout.point_ids
    )
    deformations = dict(bundle.deformations)
    deformations[target.key] = data
    result = replace(bundle, document=document, deformations=deformations, selected_alignment=target.key)
    result.validate()
    return result


def with_fit_report(bundle, result, *, alignment_id=None, region_id=None):
    """Return a new CETS bundle with fit diagnostics and output artifact hashes."""
    import copy
    import hashlib
    import math
    from dataclasses import replace

    context = bundle.get_context(alignment_id, region_id)
    if result.metrics.get("source_context_digest") != context.digest():
        raise ValueError("fit report belongs to a different CETS context")
    reports = copy.deepcopy(bundle.reports)

    def clean(value):
        if isinstance(value, dict):
            return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, torch.Tensor):
            return clean(value.detach().cpu().tolist())
        if hasattr(value, "item"):
            return clean(value.item())
        return value

    entry = reports.setdefault("/".join(context.key), {})
    entry.setdefault("fits", []).append(
        {
            "target": result.target,
            "metrics": clean(result.metrics),
            "artifacts": [
                {"name": name, "sha256": hashlib.sha256(data).hexdigest()} for name, data in result.files.items()
            ],
        }
    )
    updated = replace(bundle, document=bundle.document.model_copy(deep=True), reports=reports)
    updated.validate()
    return updated
