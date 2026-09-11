"""Prepare Warp and AreTomo3 projects from fitted CETS alignments."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import numpy as np
import torch

from cets_nonrigid.metadata import get_optics


def prepare_project(bundle, result, *, tilt_stack=None, frames_dir=None, no_stack=False, **options):
    """Use the transferred project writers and retain their executable gates.

    Project files are created in a private workspace. Large image assets are copied
    only during publication; the FitResult owns the workspace until released.
    """
    if options:
        raise ValueError(f"unknown project options: {', '.join(options)}")
    if result.target not in {"warp", "aretomo3"}:
        if result.target == "relion":
            return result
        raise ValueError("project layout is defined for tilt-series targets")
    context = bundle.context
    stem = context.parent.id
    if Path(stem).name != stem or "\\" in stem:
        raise ValueError("project series identity must be a single filename component")
    if tilt_stack is None:
        stack_paths = {r.path.split("@", 1)[1] for r in context.rows if r.path and "@" in r.path}
        if len(stack_paths) == 1:
            tilt_stack = next(iter(stack_paths))
    workspace = tempfile.TemporaryDirectory(prefix="cets-project-")
    root = Path(workspace.name)
    try:
        from cets_nonrigid.api import export_native

        if result.target == "warp":
            from cets_nonrigid.project.warp import WarpProject
            from cets_nonrigid.io.warp_xml import load_warp_tiltseries
            from cets_nonrigid.io.warp_settings import settings_for_project
            from cets_nonrigid.io.frames import frames_from_stack, frames_from_images

            project = WarpProject(root)
            path = project.xml_path(stem)
            export_native(result, path)
            loaded = load_warp_tiltseries(path)
            names = [f"{stem}_{i + 1:03}" for i in range(loaded.ts.n_tilts)]
            frame_root = root / "frames"
            source_map = result.metrics.get("source_to_target_rows", list(range(len(context.rows))))
            reverse = {target_i: source_i for source_i, target_i in enumerate(source_map) if target_i >= 0}
            if tilt_stack is not None:
                if len(reverse) != loaded.ts.n_tilts:
                    raise ValueError(
                        "stack-to-Warp project requires an explicit mapping for every target image, including dark rows"
                    )
                order = [context.rows[reverse[i]].section for i in range(loaded.ts.n_tilts)]
                if any(i is None for i in order):
                    raise ValueError("stack extraction requires CETS image section indices")
                frames_from_stack(
                    tilt_stack, frame_root, names, order=order, pixel_size_a=context.image_frames[0].isotropic_spacing
                )
            else:
                images = []
                for i in range(loaded.ts.n_tilts):
                    source_i = reverse.get(i)
                    row = context.rows[source_i] if source_i is not None else None
                    if frames_dir is not None:
                        candidate = (
                            Path(frames_dir)
                            / "average"
                            / (Path(row.path).stem + ".mrc" if row and row.path else names[i] + ".mrc")
                        )
                    else:
                        candidate = Path(row.path) if row and row.path and "@" not in row.path else None
                    if candidate is None or not candidate.is_file():
                        if no_stack:
                            images = []
                            break
                        raise ValueError(
                            "Warp project requires real images, frames_dir, or tilt_stack; no_stack explicitly prepares metadata only"
                        )
                    images.append(candidate)
                if images:
                    frames_from_images(images, frame_root, names)
            loaded.ts.tilt_movie_paths = [f"../frames/{name}.mrc" for name in names]
            # MoviePath is not alignment-owned; set the explicit project paths in this output.
            from lxml import etree

            xml = etree.fromstring(path.read_bytes())
            node = xml.find("MoviePath")
            if node is None:
                node = etree.SubElement(xml, "MoviePath")
            node.text = "\n".join(loaded.ts.tilt_movie_paths)
            path.write_bytes(etree.tostring(xml, pretty_print=True, xml_declaration=True, encoding="utf-8"))
            exposures = [r.exposure_dose for r in context.rows]
            exposure = (
                exposures[0]
                if exposures and all(v is not None and abs(v - exposures[0]) < 1e-6 for v in exposures)
                else None
            )
            settings = settings_for_project(
                pixel_size_a=context.image_frames[0].isotropic_spacing,
                exposure_per_tilt=exposure,
                tomo_dims_px=context.reference_frame.size_px,
                **get_optics(context),
            )
            project.ensure_settings(settings)
            checks = project.add_series(
                stem,
                loaded.ts,
                xml_path=path,
                settings=settings,
                pixel_size_a=context.image_frames[0].isotropic_spacing,
                frames_dir=frame_root,
            )
            primary = path.relative_to(root).as_posix()
        else:
            from cets_nonrigid.project.aretomo import AretomoDir
            from cets_nonrigid.io.aln import raw_tilts_from_aln
            from cets_nonrigid.io.tlt import tlt_from_aln
            from cets_nonrigid.io.dose import raw_dose_from_pre_exposure
            from cets_nonrigid.io.frames import stack_from_frames
            import mrcfile

            path = root / (stem + ".aln")
            export_native(result, path)
            native = result.native_result
            stack = None
            mapping = native.source_row_map
            reverse = {j: i for i, j in enumerate(mapping) if j >= 0}
            if not no_stack:
                if len(reverse) != native.aln.RawSize[2]:
                    raise ValueError("AreTomo3 project requires image identities for every emitted raw row")
                if tilt_stack is not None:
                    indices = [context.rows[reverse[j]].section for j in range(native.aln.RawSize[2])]
                    if any(i is None for i in indices):
                        raise ValueError("stack extraction requires CETS section indices")
                    stack = root / (stem + ".mrc")
                    with mrcfile.mmap(tilt_stack, mode="r", permissive=True) as source:
                        pixels = source.data[None] if source.data.ndim == 2 else source.data
                        with mrcfile.new(stack) as destination:
                            destination.set_data(np.asarray(pixels[indices]).copy())
                            destination.voxel_size = context.image_frames[0].isotropic_spacing
                else:
                    images = [context.rows[reverse[j]].path for j in range(native.aln.RawSize[2])]
                    if any(not p or "@" in p or not Path(p).is_file() for p in images):
                        raise ValueError(
                            "AreTomo3 project requires real tilt images or tilt_stack; no_stack explicitly prepares metadata only"
                        )
                    stack = root / (stem + ".mrc")
                    stack_from_frames(
                        [Path(p) for p in images], stack, pixel_size_a=context.image_frames[0].isotropic_spacing
                    )
            raw_dose = None
            if native.raw_pre_exposure is not None:
                raw_dose = raw_dose_from_pre_exposure(
                    raw_tilts_from_aln(native.aln), torch.as_tensor(native.raw_pre_exposure)
                )
            tlt = tlt_from_aln(native.aln, raw_dose)
            ctf_path = root / (stem + "_CTF.txt")
            checks = AretomoDir(root).finalize_series(
                stem,
                native.aln,
                aln_path=path,
                aln_check=native.aln_check,
                tlt=tlt,
                ctf_path=ctf_path if ctf_path.exists() else None,
                stack=stack,
                hint_kwargs={
                    "pixel_size_a": context.image_frames[0].isotropic_spacing,
                    "vol_z_px": context.reference_frame.size_px[2],
                    **get_optics(context),
                },
            )
            primary = path.name
        files, assets = {}, {}
        for path in root.rglob("*"):
            if path.is_file():
                name = path.relative_to(root).as_posix()
                if path.suffix.lower() in {".mrc", ".mrcs", ".st", ".eer", ".tif", ".tiff"}:
                    assets[name] = path
                else:
                    files[name] = path.read_bytes()
        metrics = dict(result.metrics)
        metrics["project_gates"] = [gate.model_dump() for gate in checks.gates]
        metrics["command_hint"] = checks.hint.replace(str(root), ".")
        metrics["project_ready"] = all(gate.status == "pass" for gate in checks.gates)
        return replace(
            result,
            files=files,
            primary_file=primary,
            metrics=metrics,
            layout="project",
            assets=assets,
            workspace=workspace,
        )
    except Exception:
        workspace.cleanup()
        raise


def merge_projects(results):
    """Combine prepared series into one native project, checking shared settings."""
    from dataclasses import replace

    if not results:
        raise ValueError("no project results to merge")
    target = results[0].target
    if any(r.target != target or r.layout != "project" for r in results):
        raise ValueError("only projects for one target can be combined")
    if target == "relion":
        from cets_nonrigid.project.relion import RelionProject

        workspace = tempfile.TemporaryDirectory(prefix="cets-relion-batch-")
        root = Path(workspace.name)
        project = RelionProject(root)
        assets = {}
        for result in results:
            native = result.native_result
            project.add_series(
                native.tomo,
                particle_names=native.particle_names,
                centered_coords_a=native.centered_coords_a,
                motion_a=native.motion_a,
                tilt_series_uri=native.tilt_series_uri,
                point_attributes=native.point_attributes,
            )
            for name, source in result.assets.items():
                if name in assets:
                    raise ValueError(f"duplicate image asset {name}")
                assets[name] = source
            # Explicitly sliced images are currently part of a single-series fit.
            for name, data in result.files.items():
                if not name.endswith(".star"):
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.exists():
                        raise ValueError(f"duplicate project artifact {name}")
                    path.write_bytes(data)
        project.flush()
        files = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}
        return replace(
            results[0],
            files=files,
            assets=assets,
            workspace=[workspace, results],
            metrics={"series": [r.metrics for r in results]},
            primary_file="optimisation_set.star",
        )
    files, assets = {}, {}
    for result in results:
        for name, data in result.files.items():
            if name in files:
                if target == "warp" and name.endswith(".settings") and files[name] == data:
                    continue
                if name.lower().startswith("readme"):
                    files[name] += b"\n" + data
                    continue
                raise ValueError(
                    f"project members conflict at {name}; use a separate project or distinct series identities"
                )
            files[name] = data
        for name, source in result.assets.items():
            if name in assets or name in files:
                raise ValueError(f"duplicate native project asset {name}")
            assets[name] = source
    return replace(
        results[0], files=files, assets=assets, workspace=results, metrics={"series": [r.metrics for r in results]}
    )
