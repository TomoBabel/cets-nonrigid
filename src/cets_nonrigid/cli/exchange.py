"""Per-format and per-pair CETS commands; all execution uses the public API."""

from __future__ import annotations

import json
from pathlib import Path
import click

SOURCE_FORMATS = ("aretomo3", "warp", "relion", "mcaln", "warp-movie", "relion-motion")
TILT_FORMATS = SOURCE_FORMATS[:3]
MOVIE_FORMATS = SOURCE_FORMATS[3:]


class Grid(click.ParamType):
    name = "dimensions"

    def __init__(self, ndim):
        self.ndim = ndim

    def convert(self, value, param, ctx):
        if isinstance(value, (tuple, list)):
            return tuple(value)
        try:
            values = tuple(None if v.lower() == "t" else int(v) for v in str(value).lower().split("x"))
        except ValueError:
            self.fail("use positive dimensions separated by x", param, ctx)
        if (
            len(values) != self.ndim
            or any(v is not None and v < 1 for v in values)
            or any(v is None for v in values[:-1])
        ):
            self.fail(f"expected {self.ndim} positive dimensions", param, ctx)
        if self.ndim != 4 and None in values:
            self.fail("T is allowed only for the last volume-warp axis", param, ctx)
        return values


def option(flag, name, kind=str, help="", *, is_flag=False):
    return click.Option(
        [flag, name], type=bool if is_flag else kind, default=False if is_flag else None, is_flag=is_flag, help=help
    )


def source_options(source):
    def o(flag, name, kind=str, help=""):
        return option(flag, "source_" + name, kind, help)

    common = [
        o(
            "--grid",
            "grid_shape",
            Grid(3 if source in TILT_FORMATS else 2),
            "Training sampling grid; held-out samples are separate.",
        )
    ]
    if source in {"aretomo3", "warp"}:
        common += [
            o("--pix", "pixel_size_a", float, "Tilt-image pixel size in Å; otherwise discover it."),
            o("--particles", "particles", str, "STAR/text/ndjson, picks directory, copick: or portal: spec."),
            o("--particles-voxel", "particles_voxel", str, "Picks voxel spacing in Å or tomogram directory."),
        ]
    if source == "aretomo3":
        common += [
            o("--tomo-size", "tomo_size_px", Grid(3), "Reference-volume dimensions in tilt-image pixels."),
            o("--mdoc-dir", "mdoc_dir", click.Path(exists=True, path_type=Path), "Acquisition metadata directory."),
            o("--dose-per-tilt", "dose_per_tilt", str, "Exposure per image, or file; requires acquisition order."),
            o("--ctf-file", "ctf_file", click.Path(exists=True, path_type=Path), "AreTomo3 per-tilt CTF companion."),
        ]
    elif source == "warp":
        common += [
            o(
                "--settings",
                "settings",
                click.Path(exists=True, path_type=Path),
                "Warp settings used for metadata discovery.",
            )
        ]
    elif source == "relion":
        common = [
            o("--tomo-name", "tomo_name", str, "Tomogram to read from the optimisation set."),
            o("--image-size", "image_size_px", Grid(2), "Raw tilt-image dimensions in pixels."),
            o(
                "--particles-star",
                "particles_star",
                click.Path(exists=True, path_type=Path),
                "Explicit RELION particle STAR.",
            ),
            o("--motion-star", "motion_star", click.Path(exists=True, path_type=Path), "Explicit RELION trajectories."),
        ]
    elif source == "warp-movie":
        common += [
            o("--pix", "pixel_size_a", float, "Movie image pixel size in Å."),
            o("--image-size", "image_size_px", Grid(2), "Frame width and height."),
            o("--n-frames", "n_frames", int, "Number of sampled frames."),
            o("--fraction-frames", "fraction_frames", float, "Warp runtime fraction of frames."),
        ]
    return common


def target_options(target):
    def o(flag, name, kind=str, help="", is_flag=False):
        return option(flag, "target_" + name, kind, help, is_flag=is_flag)

    if target == "warp":
        return [
            o("--movement-grid", "movement_grid", Grid(2)),
            o("--volume-warp-grid", "volume_warp_grid", Grid(4)),
            o("--lam", "lam", float),
            o("--template-xml", "template_xml", click.Path(exists=True, path_type=Path)),
            o("--global-mode", "global_mode", click.Choice(["fit", "template", "aretomo", "relion"])),
            o("--row-map", "row_map_file", click.Path(exists=True, path_type=Path)),
            o("--global-tol-px", "global_tol_px", float),
        ]
    if target == "aretomo3":
        return [
            o("--patch-grid", "patch_grid", Grid(2)),
            o("--patch-z", "patch_z", click.Choice(["lsq", "zero"])),
            o("--template-aln", "template_aln", click.Path(exists=True, path_type=Path)),
            o("--row-map", "row_map_file", click.Path(exists=True, path_type=Path)),
            o("--global-tol-px", "global_tol_px", float),
        ]
    if target == "relion":
        return [
            o("--no-particles", "no_particles", is_flag=True, help="Export global geometry without trajectories."),
            o("--no-ctf", "no_ctf", is_flag=True, help="Explicit geometry-only CTF placeholders."),
            o("--hand", "hand", click.Choice(["-1", "1"])),
            o("--trajectory-gauge", "trajectory_gauge", click.Choice(["lowest-dose", "ctf-optimal"])),
            o("--tilt-stack", "tilt_stack", click.Path(exists=True, path_type=Path)),
            o("--tilt-image-list", "tilt_image_list", click.Path(exists=True, path_type=Path)),
            o("--voltage", "voltage_kv", float),
            o("--cs", "cs_mm", float),
            o("--amp-contrast", "amplitude_contrast", float),
        ]
    if target == "warp-movie":
        return [
            o("--local-grid", "local_grid", Grid(3)),
            o("--lam", "lam", float),
            o("--template-xml", "template_xml", click.Path(exists=True, path_type=Path)),
            o("--movie-path", "movie_path", str),
        ]
    if target == "mcaln":
        return [o("--patch-grid", "patch_grid", Grid(2)), o("--fm-ref", "fm_ref", int)]
    return [
        o("--movie-name", "movie_name"),
        o("--dose-rate", "dose_rate", float),
        o("--pre-exposure", "pre_exposure", float),
        o("--voltage", "voltage_kv", float),
    ]


def _configuration(path, kwargs):
    data = json.loads(Path(path).read_text()) if path else {}
    if not isinstance(data, dict) or set(data) - {"source", "target", "project"}:
        raise ValueError("config must contain only source, target and project option objects")
    for key in ("source", "target", "project"):
        if not isinstance(data.setdefault(key, {}), dict):
            raise ValueError(f"config.{key} must be an object")
    for key, value in kwargs.items():
        if value is not None and value is not False:
            scope, _, name = key.partition("_")
            data[scope][name] = value
    if "hand" in data["target"]:
        data["target"]["hand"] = int(data["target"]["hand"])
    return data


def _run(mode, source, target, sources, output, config, project, fail_fast, report_bundle=None, **kwargs):
    from cets_nonrigid import api
    from cets_nonrigid.inputs import expand_inputs

    data = _configuration(config, kwargs)
    output = Path(output)
    if mode in {"from-cets", "fit"}:
        if len(sources) != 1:
            raise ValueError("select one CETS document per fit")
        if report_bundle is not None and Path(report_bundle).exists():
            raise FileExistsError(f"{report_bundle} already exists")
        bundle = api.read_bundle(sources[0])
        result = api.fit(bundle, target, **data["target"])
        if (project or not output.suffix) and target in {"aretomo3", "warp"}:
            result = api.prepare_project(bundle, result, **data["project"])
        api.export_native(result, output)
        if report_bundle is not None:
            api.write_bundle(api.with_fit_report(bundle, result), report_bundle)
        click.echo(json.dumps(result.metrics, indent=2, default=str))
        return
    items = expand_inputs(
        source,
        sources,
        cache_root=output.parent / ".cets-native-cache",
        need_stack=bool(data["project"].get("download_stack")),
        tomo_name=data["source"].get("tomo_name"),
        project_root=data["source"].get("project_root"),
    )
    combined_project = mode == "convert" and (
        target == "relion" or target in {"aretomo3", "warp"} and (project or not output.suffix)
    )
    if len(items) > 1 and not combined_project and output.suffix:
        raise ValueError("multiple sources require an output directory, not a single file")
    if len(items) > 1 and not combined_project:
        output.mkdir(parents=True, exist_ok=True)
    failures = []
    prepared = []
    for item in items:
        try:
            source_values = dict(data["source"])
            if item.tomo_name is not None:
                source_values["tomo_name"] = item.tomo_name
            if item.portal:
                source_values["portal_context"] = item.portal
                source_values.setdefault("series_id", item.stem)
            bundle = api.to_cets(source, item.path, **source_values)
            if mode == "to-cets":
                path = output / (item.stem + ".cets.json") if len(items) > 1 or output.is_dir() else output
                api.write_bundle(bundle, path)
            else:
                result = api.fit(bundle, target, **data["target"])
                if combined_project and target in {"aretomo3", "warp"}:
                    result = api.prepare_project(bundle, result, **data["project"])
                if combined_project:
                    prepared.append(result)
                    continue
                if len(items) > 1:
                    path = output / (item.stem if result.layout == "project" else result.primary_file)
                else:
                    path = output
                api.export_native(result, path)
                click.echo(json.dumps(result.metrics, default=str))
            click.echo(f"wrote {path}")
        except (ValueError, OSError, RuntimeError, TypeError) as exc:
            failures.append({"source": str(item.path), "error": str(exc)})
            click.echo(f"{item.path}: {exc}", err=True)
            if fail_fast:
                break
    if prepared:
        result = api.merge_projects(prepared) if len(prepared) > 1 else prepared[0]
        result.metrics["failures"] = failures
        result.files["cets-nonrigid-report.json"] = (json.dumps(result.metrics, indent=2, default=str) + "\n").encode()
        api.export_native(result, output)
        click.echo(f"wrote {output}")
    if failures:
        raise click.ClickException(f"{len(failures)} source(s) failed: " + "; ".join(f["error"] for f in failures))


def command(mode, name, source=None, target=None):
    parameters = [
        click.Argument(["sources"], nargs=-1, required=True),
        click.Option(["-o", "--output"], required=True, type=click.Path(path_type=Path)),
        click.Option(
            ["--config"],
            type=click.Path(exists=True, path_type=Path),
            help="JSON source/target/project options; flags override config.",
        ),
        click.Option(["--project"], is_flag=True, help="Prepare the native project layout and companions."),
        click.Option(["--fail-fast"], is_flag=True, help="Stop batch processing after the first failure."),
    ]
    if mode in {"from-cets", "fit"}:
        parameters.append(
            click.Option(
                ["--report-bundle"],
                type=click.Path(path_type=Path),
                help="Write a new CETS bundle containing fit diagnostics.",
            )
        )
    if source:
        parameters += source_options(source)
    if target:
        parameters += target_options(target)

    def callback(**kwargs):
        try:
            return _run(mode, source, target, **kwargs)
        except (ValueError, OSError, RuntimeError, TypeError) as exc:
            raise click.ClickException(str(exc)) from exc

    description = (
        f"{source} → CETS"
        if target is None
        else f"CETS → {target}"
        if source is None
        else f"{source} → CETS → {target}"
    )
    return click.Command(
        name,
        params=parameters,
        callback=callback,
        help=description + ". A new output is required; existing outputs are never replaced. "
        "Advanced native-reader and fitting options are accepted in the per-format config objects.",
    )


def register(main):
    for name in ("to-cets", "from-cets", "convert", "fit"):
        group = click.Group(name)
        main.add_command(group)
        if name == "to-cets":
            for source in SOURCE_FORMATS:
                group.add_command(command(name, source, source=source))
        elif name in {"from-cets", "fit"}:
            for target in SOURCE_FORMATS:
                group.add_command(command(name, target, target=target))
        else:
            for formats in (TILT_FORMATS, MOVIE_FORMATS):
                for source in formats:
                    for target in formats:
                        if source != target:
                            group.add_command(command(name, f"{source}-to-{target}", source, target))
