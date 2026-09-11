"""Frame-side pipelines to/from RELION's per-micrograph motion star.

rm2w / rm2m: the RELION model (global shifts + third-order polynomial) is
evaluated EXACTLY at IR grid points on valid frames (sentinel frames are
excluded per the backward-fallback/validity rule), then the existing Warp
movie / .mcaln fitters run unchanged (their capacity exceeds the polynomial's,
so recovery is near-exact).

w2rm / m2rm: fit the RELION model with the exact re-gauge (relative
trajectories; the discarded static field is reported, see
fit/relion_motion_fit.py). m2rm requires the .mcaln alignment/original
pixel-size relationship to be the identity (input_to_alignment_scale == 1,
v1) and emits EER labels only when the frame table's source_count is uniform.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from cets_nonrigid.convert import _frame_ir_meta, mcaln_from_motion_model
from cets_nonrigid.fit.relion_motion_fit import RelionMotionFitResult, fit_relion_motion
from cets_nonrigid.io.relion_motion_star import (
    read_micrograph_motion_star,
    write_micrograph_motion_star,
)
from cets_nonrigid.io.store import DeformationStore
from cets_nonrigid.ir.build import build_ir_frame_series

_F64 = torch.float64


@dataclass
class RM2WResult:
    out_xml: Path
    fit: object  # WarpMovieFitResult
    store: Path | None
    template_source: str = "template"
    defaulted_fields: list | None = None


@dataclass
class W2RMResult:
    out_star: Path
    fit: RelionMotionFitResult
    store: Path | None


@dataclass
class RM2MResult:
    out_mcaln: Path
    fit: object  # McAlnFitResult
    store: Path | None


@dataclass
class M2RMResult:
    out_star: Path
    fit: RelionMotionFitResult
    store: Path | None


def _store_frame(store_path, ir, fit_attrs, native=None, target=None):
    if store_path is None:
        return None
    return Path(
        DeformationStore.write(
            store_path, ir,
            native_files=native or {},
            target_files=target or {},
            fit_attrs=fit_attrs,
        )
    )


def relion_motion_to_warp_movie(
    star_path: str | Path,
    template_xml: str | Path | None,
    out_xml: str | Path,
    *,
    fraction_frames: float = 1.0,
    grid_shape: tuple = (11, 11),
    local_grid: tuple = (3, 3, 4),
    lam: float = 1e-3,
    store_path: str | Path | None = None,
    movie_path: str | None = None,
) -> RM2WResult:
    """rm2w: RELION micrograph motion star -> Warp movie XML."""
    from warpylib.movie import Movie

    from cets_nonrigid.fit.movie_fits import fit_warp_movie
    from cets_nonrigid.io.warp_xml import write_movie_alignment_into_template

    motion = read_micrograph_motion_star(star_path)
    model = motion.to_model()
    pix = motion.pixel_size_a
    f_count = motion.n_frames
    image_dims_a = (motion.image_size_px[0] * pix, motion.image_size_px[1] * pix)

    ir = build_ir_frame_series(
        model,
        torch.tensor(image_dims_a),
        meta=_frame_ir_meta(
            Path(star_path).stem, pix, motion.image_size_px, f_count, "relion-motion"
        ),
        grid_shape=tuple(grid_shape),
    )
    synth_report = None
    if template_xml is not None:
        from cets_nonrigid.convert import _reject_synthesis_options

        _reject_synthesis_options(movie_path=movie_path)
        template_movie = Movie(path=str(template_xml))
        template_bytes = Path(template_xml).read_bytes()
    else:
        from cets_nonrigid.io.warp_synth import movie_template_bytes, synthesize_movie

        template_movie, synth_report = synthesize_movie(
            pixel_size_a=pix, voltage_kv=motion.voltage_kv, data_path=movie_path
        )
        template_bytes = movie_template_bytes(template_movie, "rm2w")
        synth_report.warn_once("rm2w")
        from cets_nonrigid.convert import _check_movie_path_stem

        _check_movie_path_stem(movie_path, out_xml)
    fit = fit_warp_movie(
        ir, template_movie,
        n_frames=f_count, image_dims_a=image_dims_a,
        fraction_frames=fraction_frames, local_grid=tuple(local_grid), lam=lam,
    )
    if synth_report is None:
        write_movie_alignment_into_template(template_bytes, fit.movie, out_xml)
    else:
        from cets_nonrigid.io.warp_movie_xml import load_warp_movie_strict
        from cets_nonrigid.io.warp_synth import atomic_write_validated

        atomic_write_validated(
            lambda tmp: write_movie_alignment_into_template(template_bytes, fit.movie, tmp),
            out_xml,
            load_warp_movie_strict,
        )
    store = _store_frame(
        store_path, ir,
        {
            "direction": "rm2w",
            "heldout_status": fit.heldout_status,
            "rms_a_train": fit.rms_a_train,
            "rms_a_heldout": fit.rms_a_heldout,
            "p95_a_heldout": fit.p95_a_heldout,
            "coverage_heldout": fit.coverage_heldout,
            "pixel_size_a": pix,
            "template_source": "template" if synth_report is None else "generated",
            "defaulted_fields": (
                [] if synth_report is None else list(synth_report.defaulted_experimental)
            ),
            **fit.meta,
        },
        native=(
            {"motion_star": Path(star_path).read_bytes()}
            | ({"template_xml": template_bytes} if synth_report is None else {})
        ),
        target={"out_xml": Path(out_xml).read_bytes()},
    )
    return RM2WResult(
        out_xml=Path(out_xml),
        fit=fit,
        store=store,
        template_source="template" if synth_report is None else "generated",
        defaulted_fields=(
            None if synth_report is None else list(synth_report.defaulted_experimental)
        ),
    )


def warp_movie_to_relion_motion(
    xml_path: str | Path,
    out_star: str | Path,
    *,
    image_size_px: tuple,
    pixel_size_a: float,
    n_frames: int,
    fraction_frames: float = 1.0,
    grid_shape: tuple = (11, 11),
    movie_name: str = "",
    dose_rate: float | None = None,
    pre_exposure: float | None = None,
    voltage_kv: float | None = None,
    store_path: str | Path | None = None,
) -> W2RMResult:
    """w2rm: Warp movie XML -> RELION micrograph motion star."""
    from cets_nonrigid.models.warp_movie import WarpMovieModel

    image_dims_a = (image_size_px[0] * pixel_size_a, image_size_px[1] * pixel_size_a)
    model = WarpMovieModel.from_xml(
        xml_path, n_frames=n_frames, image_dims_a=image_dims_a,
        fraction_frames=fraction_frames,
    )
    ir = build_ir_frame_series(
        model,
        torch.tensor(image_dims_a),
        meta=_frame_ir_meta(Path(xml_path).stem, pixel_size_a, image_size_px, n_frames, "warp-movie"),
        grid_shape=tuple(grid_shape),
    )
    fit = fit_relion_motion(
        ir, image_size_px=image_size_px, pixel_size_a=pixel_size_a,
        movie_name=movie_name or Path(xml_path).stem,
        dose_rate=dose_rate, pre_exposure=pre_exposure, voltage_kv=voltage_kv,
    )
    write_micrograph_motion_star(out_star, fit.motion)
    store = _store_frame(
        store_path, ir,
        {
            "direction": "w2rm",
            "heldout_status": fit.heldout_status,
            "rms_px_train": fit.rms_px_train,
            "rms_px_heldout": fit.rms_px_heldout,
            "p95_px_heldout": fit.p95_px_heldout,
            "coverage_heldout": fit.coverage_heldout,
            "static_field_rms_px": fit.static_field_rms_px,
            "static_field_p95_px": fit.static_field_p95_px,
            "static_field_max_px": fit.static_field_max_px,
            "data_rank": fit.data_rank,
            "data_condition": fit.data_condition,
            "pixel_size_a": pixel_size_a,
            **fit.meta,
        },
        target={"out_star": Path(out_star).read_bytes()},
    )
    return W2RMResult(out_star=Path(out_star), fit=fit, store=store)


def relion_motion_to_mcaln(
    star_path: str | Path,
    out_mcaln: str | Path,
    *,
    patch_grid: tuple = (5, 5),
    fm_ref: int = -1,
    grid_shape: tuple = (11, 11),
    store_path: str | Path | None = None,
) -> RM2MResult:
    """rm2m: RELION micrograph motion star -> AreTomo3 .mcaln."""
    from cets_nonrigid.fit.movie_fits import fit_mcaln_shifts

    motion = read_micrograph_motion_star(star_path)
    model = motion.to_model()
    pix = motion.pixel_size_a
    f_count = motion.n_frames
    image_dims_a = (motion.image_size_px[0] * pix, motion.image_size_px[1] * pix)

    ir = build_ir_frame_series(
        model,
        torch.tensor(image_dims_a),
        meta=_frame_ir_meta(
            Path(star_path).stem, pix, motion.image_size_px, f_count, "relion-motion"
        ),
        grid_shape=tuple(grid_shape),
    )
    fit = fit_mcaln_shifts(
        ir, frame_size_px=tuple(motion.image_size_px), pixel_size_a=pix,
        patch_grid=tuple(patch_grid), fm_ref=fm_ref,
    )
    out_mcaln = Path(out_mcaln)
    mcaln = mcaln_from_motion_model(
        fit.model, n_frames=f_count, image_size_px=motion.image_size_px,
        pixel_size_a=pix, patch_grid=tuple(patch_grid), fm_ref=fit.meta["fm_ref"],
    )
    mcaln.to_file(out_mcaln)
    store = _store_frame(
        store_path, ir,
        {
            "direction": "rm2m",
            "heldout_status": fit.heldout_status,
            "rms_a_train": fit.rms_a_train,
            "rms_a_heldout": fit.rms_a_heldout,
            "p95_a_heldout": fit.p95_a_heldout,
            "coverage_heldout": fit.coverage_heldout,
            "pixel_size_a": pix,
            **fit.meta,
        },
        native={"motion_star": Path(star_path).read_bytes()},
        target={"out_mcaln": out_mcaln.read_bytes()},
    )
    return RM2MResult(out_mcaln=out_mcaln, fit=fit, store=store)


def mcaln_to_relion_motion(
    mcaln_path: str | Path,
    out_star: str | Path,
    *,
    grid_shape: tuple = (11, 11),
    movie_name: str = "",
    dose_rate: float | None = None,
    pre_exposure: float | None = None,
    voltage_kv: float | None = None,
    store_path: str | Path | None = None,
) -> M2RMResult:
    """m2rm: AreTomo3 .mcaln -> RELION micrograph motion star.

    v1 requires the alignment-image coordinate space to be the raw movie's
    (input_to_alignment_scale == 1); EER labels are emitted only when the
    frame table's source_count is uniform (else plain frames + a warning)."""
    import warnings

    from cets_nonrigid.io.motion_txt import McAln

    mcaln = McAln.from_file(mcaln_path)
    scale = getattr(mcaln, "input_to_alignment_scale_xy", (1.0, 1.0))
    if any(abs(float(sc) - 1.0) > 1e-6 for sc in scale):
        raise ValueError(
            f"input_to_alignment_scale {scale} != 1: the RELION motion star lives in raw "
            "movie pixels; rescaling alignment-space shifts is not supported in v1"
        )
    model = mcaln.to_model()
    pix = float(mcaln.alignment_pixel_size_a)
    size_px = tuple(mcaln.alignment_image_size_px)
    f_count = int(mcaln.aligned_frame_count)
    image_dims_a = (size_px[0] * pix, size_px[1] * pix)

    ir = build_ir_frame_series(
        model,
        torch.tensor(image_dims_a),
        meta=_frame_ir_meta(Path(mcaln_path).stem, pix, size_px, f_count, "aretomo3-motion"),
        grid_shape=tuple(grid_shape),
    )
    counts = {f.source_count for f in mcaln.frames if f.included}
    eer_grouping = counts.pop() if len(counts) == 1 else None
    if eer_grouping is None:
        warnings.warn(
            "non-uniform frame-table source_count: EER labels omitted from the motion star",
            stacklevel=2,
        )
    fit = fit_relion_motion(
        ir, image_size_px=size_px, pixel_size_a=pix,
        movie_name=movie_name or Path(mcaln_path).stem,
        dose_rate=dose_rate, pre_exposure=pre_exposure, voltage_kv=voltage_kv,
        eer_grouping=eer_grouping if (eer_grouping or 1) > 1 else None,
    )
    write_micrograph_motion_star(out_star, fit.motion)
    store = _store_frame(
        store_path, ir,
        {
            "direction": "m2rm",
            "heldout_status": fit.heldout_status,
            "rms_px_train": fit.rms_px_train,
            "rms_px_heldout": fit.rms_px_heldout,
            "p95_px_heldout": fit.p95_px_heldout,
            "coverage_heldout": fit.coverage_heldout,
            "static_field_rms_px": fit.static_field_rms_px,
            "pixel_size_a": pix,
            **fit.meta,
        },
        native={"mcaln": Path(mcaln_path).read_bytes()},
        target={"out_star": Path(out_star).read_bytes()},
    )
    return M2RMResult(out_star=Path(out_star), fit=fit, store=store)
