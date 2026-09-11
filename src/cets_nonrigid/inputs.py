"""Shared source/picks grammar and validated metadata discovery workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import click

from cets_nonrigid.cli.specs import (
    parse_portal_source,
    PARTICLE_FORMS,
    PICK_SUFFIXES,
    find_by_stem,
)


@dataclass
class SourceItem:
    path: Path
    stem: str
    portal: object | None = None
    tomo_name: str | None = None


def expand_inputs(source, tokens, *, cache_root, need_stack=False, tomo_name=None, project_root=None):
    """Resolve local files/directories and Portal sources without touching source files."""
    patterns = {
        "aretomo3": "*.aln",
        "warp": "*.xml",
        "relion": "optimisation_set.star",
        "mcaln": "*.mcaln",
        "warp-movie": "*.xml",
        "relion-motion": "*.star",
    }
    result, seen = [], set()
    for token in tokens:
        if str(token).startswith("portal:"):
            if source != "aretomo3":
                raise ValueError("Portal native alignment sources use the aretomo3 reader")
            from types import SimpleNamespace
            from cets_nonrigid.meta.portal import (
                portal_client,
                find_runs,
                fetch_run_data,
                download_alignment,
                download_stack,
            )

            spec = parse_portal_source(str(token))
            client = portal_client()
            for run in find_runs(
                client, dataset_id=spec.dataset_id, run_names=(spec.run_name,) if spec.run_name else ()
            ):
                data = fetch_run_data(client, run, alignment_id=spec.alignment_id)
                cache = Path(cache_root) / data.run_name
                path = download_alignment(data, cache)
                stack = download_stack(data, cache, data.run_name) if need_stack else None
                portal = SimpleNamespace(client=client, data=data, cache_dir=cache, spec=spec, stack_path=stack)
                result.append(SourceItem(path, data.run_name, portal))
            continue
        path = Path(token)
        paths = sorted(path.glob(patterns[source])) if path.is_dir() else [path]
        if not paths:
            raise ValueError(f"no {patterns[source]} sources in {path}")
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.resolve() not in seen:
                seen.add(path.resolve())
                if source == "relion":
                    from cets_nonrigid.convert_relion import list_relion_tomograms

                    names = list_relion_tomograms(optimisation_set=path, project_root=project_root)
                    if tomo_name is not None:
                        if tomo_name not in names:
                            raise ValueError(f"tomogram {tomo_name!r} not in {path}")
                        names = [tomo_name]
                    result.extend(SourceItem(path, name, tomo_name=name) for name in names)
                else:
                    result.append(SourceItem(path, path.stem))
    if not result:
        raise ValueError("no input sources found")
    return result


def single_only(name, value, n_sources):
    if value is not None and n_sources != 1:
        raise ValueError(f"{name} requires a single input source")


def load_series_particles(
    *,
    stem: str,
    tomo_name: str,
    tomo_dims_px,
    pixel_size_a: float,
    raw_x_px: int | None,
    spec,
    voxel,
    no_particles: bool,
    n_sources: int,
    portal=None,
):
    """(positions, names) for one series from the ``--particles`` spec and
    ``--particles-voxel``; (None, None) with ``--no-particles``."""
    from cets_nonrigid.io.particles import load_copick_uri_particles, load_particles, voxel_size_from_tomogram

    if no_particles:
        return None, None
    if spec is None:
        raise click.UsageError(f"{stem}: no particles — pass --particles ({PARTICLE_FORMS}) or --no-particles")
    if spec.kind == "copick":
        return load_copick_uri_particles(
            spec.config,
            spec.uri,
            run_name=stem,
            tomo_name=tomo_name,
            tomo_dims_px=tomo_dims_px,
            pixel_size_a=pixel_size_a,
        )
    if spec.kind == "portal":
        if portal is None:
            raise click.UsageError(f"{stem}: --particles {spec} needs a portal:<dataset>/<run> source")
        from cets_nonrigid.meta.portal import download_annotation_points

        ndjson, vs, implied = download_annotation_points(
            portal.client,
            portal.data,
            portal.cache_dir,
            object_name=spec.object_name,
            annotation_id=spec.annotation_id,
            voxel_spacing=portal.spec.voxel_spacing,
        )
        click.echo(f"  picks: {ndjson.name} at {vs} A voxels (implied {implied:.5f} A)")
        return load_particles(
            ndjson,
            tomo_name=tomo_name,
            tomo_dims_px=tomo_dims_px,
            pixel_size_a=pixel_size_a,
            unit="voxel",
            voxel_size_a=implied,
        )
    if spec.kind == "file":
        single_only("--particles FILE", spec.path, n_sources)
        path = spec.path
    else:
        path = find_by_stem(spec.path, stem, PICK_SUFFIXES)
        if path is None:
            raise click.UsageError(f"{stem}: no picks <stem>.star|.ndjson|.txt in {spec.path}")
    suffix = Path(path).suffix.lower()
    voxel_a = None
    if voxel is not None:
        if suffix == ".star":
            raise click.UsageError("--particles-voxel does not apply to .star picks (RELION coordinates are Angstrom)")
        if isinstance(voxel, Path):
            tomo = find_by_stem(voxel, stem, (".mrc", ".mrcs", ".rec"))
            if tomo is None:
                raise click.UsageError(f"{stem}: no tomogram <stem>*.mrc in {voxel}")
            if raw_x_px is None:
                raise click.UsageError("--particles-voxel DIR needs the raw image width (RawSize)")
            voxel_a = voxel_size_from_tomogram(tomo, pixel_size_a=pixel_size_a, raw_x_px=raw_x_px)
        else:
            voxel_a = float(voxel)
    if suffix == ".ndjson" and voxel_a is None:
        raise click.UsageError(f"{path}: .ndjson picks are voxel units — pass --particles-voxel A|DIR")
    return load_particles(
        path,
        tomo_name=tomo_name,
        tomo_dims_px=tomo_dims_px,
        pixel_size_a=pixel_size_a,
        unit="voxel" if voxel_a is not None else "angstrom",
        voxel_size_a=voxel_a,
    )


_DISCOVERY_KEYS = {
    "mdoc",
    "mdoc_dir",
    "tilt_stack",
    "tilt_stack_dir",
    "tlt",
    "settings",
    "frames_dir",
    "dose_per_tilt",
    "dose_convention",
    "discover_adjacent",
    "no_ctf",
    "particles",
    "particles_voxel",
    "portal_context",
    "dose_rate",
    "pre_exposure",
    "eer_grouping",
    "eer_upsampling",
    "raw_frames_per_aligned",
}


def prepare_native_options(source, path, options):
    """Return native-reader arguments and resolved metadata, retaining conflict rules."""
    from cets_nonrigid.meta.resolve import resolve_series

    values = dict(options)
    extra = {key: values.pop(key) for key in list(values) if key in _DISCOVERY_KEYS}
    discovery, metadata = None, None
    portal = extra.get("portal_context")
    if source == "aretomo3":
        if portal is not None:
            from cets_nonrigid.meta.portal import discover_portal_series

            discovery = discover_portal_series(
                portal.data,
                Path(path),
                cache_dir=portal.cache_dir,
                voxel_spacing=portal.spec.voxel_spacing,
                stack=portal.stack_path,
            )
        else:
            from cets_nonrigid.meta.aretomo_run import discover_aretomo_series

            arguments = {
                k: extra[k]
                for k in ("mdoc", "mdoc_dir", "tilt_stack", "tilt_stack_dir", "tlt", "no_ctf", "discover_adjacent")
                if k in extra
            }
            if "ctf_file" in values:
                arguments["ctf"] = values["ctf_file"]
            discovery = discover_aretomo_series(path, **arguments)
    elif source == "warp":
        from cets_nonrigid.meta.warp_project import discover_warp_series

        arguments = {
            k: extra[k]
            for k in ("settings", "frames_dir", "tilt_stack", "tilt_stack_dir", "discover_adjacent")
            if k in extra
        }
        discovery = discover_warp_series(path, **arguments)
    if discovery is not None:
        overrides = {
            key: values[key] for key in ("pixel_size_a", "voltage_kv", "cs_mm", "amplitude_contrast") if key in values
        }
        if values.get("tomo_size_px") is not None:
            overrides["tomo_dims_px"] = values["tomo_size_px"]
        if values.get("defocus_handedness") is not None:
            overrides["defocus_hand"] = values["defocus_handedness"]
        dose = extra.get("dose_per_tilt")
        if dose is not None and dose != "file":
            overrides["dose_per_tilt"] = float(dose)
        metadata = resolve_series(
            discovery,
            overrides,
            dose_from="file" if dose == "file" else None,
            dose_convention=extra.get("dose_convention", "exclusive"),
        )
        for name, meta_name in (
            ("pixel_size_a", "pixel_size_a"),
            ("voltage_kv", "voltage_kv"),
            ("cs_mm", "cs_mm"),
            ("amplitude_contrast", "amplitude_contrast"),
            ("defocus_handedness", "defocus_hand"),
            ("ctf_file", "ctf_path"),
        ):
            value = getattr(metadata, meta_name)
            if value is not None:
                values.setdefault(name, value)
        if source == "aretomo3":
            if metadata.tomo_dims_px is not None:
                values.setdefault("tomo_size_px", metadata.tomo_dims_px)
            if metadata.raw_dose is not None:
                values.setdefault("raw_dose", metadata.raw_dose)
        elif metadata.image_dims_px is not None or metadata.tomo_dims_px is not None:
            from cets_nonrigid.io.warp_xml import DimsOverride
            from xml.etree import ElementTree as ET

            root = ET.parse(path).getroot()

            def missing(name):
                return not any(float(x) > 0 for x in (root.get(name) or "0").split(","))

            image = metadata.image_dims_px if missing("ImageDimensionsAngstrom") else None
            volume = metadata.tomo_dims_px if missing("VolumeDimensionsAngstrom") else None
            if image is not None or volume is not None:
                values["dims_override"] = DimsOverride(
                    image_a=tuple(d * metadata.pixel_size_a for d in image) if image else None,
                    volume_a=tuple(d * metadata.pixel_size_a for d in volume) if volume else None,
                )
        if extra.get("no_ctf"):
            values.pop("ctf_file", None)
    return values, metadata, extra


def apply_discovered_metadata(native, metadata):
    if metadata is None:
        return
    context = native.context
    context.owner.provenance.warnings.extend(metadata.warnings)
    import json
    from cets_data_model.models import models as m

    for name, provenance in metadata.provenance.items():
        context.owner.provenance.parameters.append(
            m.NativeParameter(name="metadata_source:" + name, value_json=json.dumps(provenance.model_dump(mode="json")))
        )
    if metadata.aux.get("tilt_series_uri_s3"):
        context.owner.provenance.parameters.append(
            m.NativeParameter(name="tilt_series_uri", value_json=json.dumps(metadata.aux["tilt_series_uri_s3"]))
        )
    context.parent.nominal_tilt_axis_angle = metadata.tilt_axis_deg
    for i, row in enumerate(context.rows):
        source_index = i
        if native.source == "aretomo3":
            sec = native.ir.meta.projection_sec[i]
            if sec > 0:
                source_index = sec - 1
            else:
                # Bind dark rows only where the remaining native section is unique.
                used = {v - 1 for v in native.ir.meta.projection_sec if v > 0}
                remaining = set(range(metadata.n_raw_sections or len(context.rows))) - used
                if len(remaining) != 1:
                    row.section = None
                    continue
                source_index = remaining.pop()
        row.section = source_index
        if metadata.acq_order_1b is not None:
            row.acquisition_order = int(metadata.acq_order_1b[source_index]) - 1
        if metadata.dose_per_section is not None:
            row.exposure_dose = float(metadata.dose_per_section[source_index])
        elif metadata.dose_per_tilt is not None:
            row.exposure_dose = float(metadata.dose_per_tilt)
        if metadata.stage_tilt_deg is not None:
            row.nominal_tilt_angle = float(metadata.stage_tilt_deg[source_index])
        if metadata.stack_path:
            row.path = f"{source_index + 1}@{Path(metadata.stack_path).resolve()}"
        elif metadata.tilt_image_names is not None and metadata.frames_dir:
            row.path = str(
                Path(metadata.frames_dir).resolve()
                / "average"
                / (Path(metadata.tilt_image_names[source_index]).stem + ".mrc")
            )
    if metadata.aux.get("mdoc"):
        context.parent.collection_metadata_path = metadata.aux["mdoc"]
