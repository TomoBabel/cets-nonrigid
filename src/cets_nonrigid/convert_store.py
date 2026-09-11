"""Store-direct fitting: `cets_nonrigid fit` (plan rev. 5).

The deformation store is pipeline-agnostic — `fit` consumes ONLY the IR and
its `/projection` row table; `source/native_files` is provenance and never an
input. Anything a target needs beyond the store arrives via CLI arguments.

Contracts implemented here:
- strict mode/option validation before any work;
- row identity via ir/rows.py (row-map file > labels > checked angle fallback);
- global fitting per --global-mode with a MANDATORY baseline-consistency gate:
  the achieved target-global projections are compared against the stored
  global baseline at the training points BEFORE locals are fitted and before
  any output is written — external context that disagrees is rejected, locals
  never absorb a global mismatch;
- CTF: never read from the store; template mode preserves the template's CTF
  content, generated mode carries the documented placeholder CTF;
- --store-out writes a NEW store holding the ORIGINAL IR plus target-row
  tables/mappings (dimension "target_projection") and the full context-file
  closure of this fit; the input store is never mutated.
"""

from __future__ import annotations

import hashlib
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from cets_nonrigid.fit.ir_global import (
    fit_aretomo_globals_from_ir,
    fit_warp_globals_from_ir,
)
from cets_nonrigid.io.store import DeformationStore
from cets_nonrigid.ir.core import IRTiltSeries
from cets_nonrigid.ir.rows import RowMatch, TargetRowTable, align_ir_rows, match_rows

_F64 = torch.float64

DEFAULT_GLOBAL_TOL_PX_WARP = 0.5
DEFAULT_GLOBAL_TOL_PX_ARETOMO = 5.0

_WARP_GLOBAL_MODES = ("fit", "template", "aretomo", "relion")


@dataclass
class StoreFitContext:
    """Bookkeeping shared by all targets: context files actually opened by
    THIS fit (the full dependency closure) and the row mapping."""

    contents: object  # StoreContents
    context_files: dict  # name -> bytes
    match: RowMatch | None = None
    target_rows: TargetRowTable | None = None


def _load_tilt_series_store(store_path) -> StoreFitContext:
    from cets_nonrigid.samples import AlignmentBundle
    if isinstance(store_path, AlignmentBundle):
        from cets_nonrigid.runtime import runtime_ir
        from types import SimpleNamespace
        contents = SimpleNamespace(ir=runtime_ir(store_path), native_files={})
    else:
        contents = DeformationStore.read(store_path)
    if contents.ir.meta.kind != "tilt_series":
        raise ValueError(
            f"store kind is {contents.ir.meta.kind!r} - tilt-series targets need a "
            "tilt_series store (frame targets: mcaln, movie-xml, relion-motion)"
        )
    return StoreFitContext(contents=contents, context_files={})


def _load_frame_series_store(store_path, *, compress_movie_rows=False) -> StoreFitContext:
    from cets_nonrigid.samples import AlignmentBundle
    if isinstance(store_path, AlignmentBundle):
        from cets_nonrigid.runtime import runtime_ir
        from types import SimpleNamespace
        contents = SimpleNamespace(ir=runtime_ir(store_path, compress_movie_rows=compress_movie_rows), native_files={})
    else:
        contents = DeformationStore.read(store_path)
    if contents.ir.meta.kind != "frame_series":
        raise ValueError(
            f"store kind is {contents.ir.meta.kind!r} - frame targets need a "
            "frame_series store (tilt-series targets: warp, aretomo)"
        )
    return StoreFitContext(contents=contents, context_files={})


def _add_context_file(ctx: StoreFitContext, name: str, path) -> None:
    if path is not None:
        ctx.context_files[name] = Path(path).read_bytes()


def _validate_global_mode(global_mode, template_xml, aln, optimisation_set, tomograms_star):
    if global_mode not in _WARP_GLOBAL_MODES:
        raise ValueError(f"unknown --global-mode {global_mode!r} (one of {_WARP_GLOBAL_MODES})")
    star_ctx = optimisation_set is not None or tomograms_star is not None
    if global_mode == "fit" and (aln is not None or star_ctx):
        raise ValueError(
            "--global-mode fit takes no source-global context - drop --aln/--optimisation-set/"
            "--tomograms or pick --global-mode aretomo/relion"
        )
    if global_mode == "template":
        if template_xml is None:
            raise ValueError("--global-mode template requires -x/--template")
        if aln is not None or star_ctx:
            raise ValueError("--global-mode template takes no source-global context")
    if global_mode == "aretomo" and (aln is None or star_ctx):
        raise ValueError("--global-mode aretomo requires exactly --aln")
    if global_mode == "relion":
        if aln is not None:
            raise ValueError("--global-mode relion takes no --aln")
        if optimisation_set is not None and tomograms_star is not None:
            raise ValueError("give exactly one of --optimisation-set / --tomograms")
        if not star_ctx:
            raise ValueError("--global-mode relion requires --optimisation-set or --tomograms")


def _global_consistency_gate(ts_target, ir: IRTiltSeries, tol_px: float, what: str) -> float:
    """Achieved target-global projections vs the STORED baseline at the
    training points, active rows, all finite entries — before locals/output."""
    from cets_nonrigid.models.warp_ts import WarpTiltSeriesModel

    model = WarpTiltSeriesModel(ts_target)
    xy, _ = model.project_volume_global(ir.points.to(_F64))
    ref = ir.source_projected_global.to(_F64)
    valid_rows = torch.tensor(
        ir.meta.projection_valid or [True] * ir.n_projections, dtype=torch.bool
    )
    finite = torch.isfinite(ref).all(dim=-1) & valid_rows[:, None]
    diff = (xy.to(_F64) - ref)[finite]
    rms_px = (
        float(diff.pow(2).sum(-1).mean().sqrt()) / ir.meta.pixel_size_image_a
        if diff.numel()
        else 0.0
    )
    if rms_px > tol_px:
        raise ValueError(
            f"{what}: achieved target-global projections disagree with the store's "
            f"global baseline by {rms_px:.3f} px RMS (> {tol_px} px) - the context does "
            "not describe this IR; locals are never allowed to absorb a global mismatch"
        )
    return rms_px


# ---------------------------------------------------------------------------
# fit --to warp
# ---------------------------------------------------------------------------


@dataclass
class StoreFitWarpResult:
    out_xml: Path
    fit: object  # WarpTsFitResult
    global_mode: str
    global_check_rms_px: float
    global_validation_status: str
    global_rms_px_train: float | None
    global_rms_px_heldout: float | None
    template_source: str
    store_out: Path | None
    source_row_map: list[int] | None = None


def _synthesize_fit_warp_template(ir: IRTiltSeries, angles_inverted):
    from cets_nonrigid.io.warp_synth import synthesize_tilt_series, synthesized_template_series

    meta = ir.meta
    t = ir.n_projections
    valid = meta.projection_valid or [True] * t
    doses = [float(d) for d in (meta.projection_dose or [])]
    if len(doses) != t or (t > 1 and max(doses) - min(doses) <= 0):
        from cets_nonrigid.models.warp_ts import EQUAL_DOSE_NUDGE

        raise ValueError(
            "fit --to warp without a template needs the IR's per-tilt dose, but this store carries "
            f"none (all-equal/zero projection dose): {EQUAL_DOSE_NUDGE}. Either pass -x/--template "
            "(its <Dose> is used) or re-dump the store from a source that carries the dose "
            "(a Warp XML, a RELION star, or a2w's discovery from _TLT.txt/mdoc/--dose-per-tilt)"
        )
    ts, report = synthesize_tilt_series(
        angles_deg=meta.projection_angle_deg or [0.0] * t,
        use_tilt=valid,
        axis_angles_deg=torch.zeros(t),
        axis_offset_x_a=torch.zeros(t),
        axis_offset_y_a=torch.zeros(t),
        image_dims_a=tuple(d * meta.pixel_size_image_a for d in meta.image_dims_px),
        volume_dims_a=tuple(d * meta.pixel_size_volume_a for d in meta.volume_dims_px),
        pixel_size_a=meta.pixel_size_image_a,
        dose=doses,
        movie_paths=None,
        angles_inverted=bool(angles_inverted) if angles_inverted is not None else False,
        angles_inverted_known=angles_inverted is not None,
    )
    return synthesized_template_series(ts, "fit --to warp"), report


def _warp_globals_from_context(
    ctx, ir, target_rows, global_mode, aln, optimisation_set, tomograms_star, tomo_name,
    row_map_file, deactivate_unmatched,
):
    """Closed-form globals from external context, transferred onto the target
    rows through the SAME row matcher (context rows -> target rows)."""
    t_tgt = target_rows.n_rows
    level_x = 0.0
    angles = torch.zeros(t_tgt, dtype=_F64)
    axis = torch.zeros(t_tgt, dtype=_F64)
    offx = torch.zeros(t_tgt, dtype=_F64)
    offy = torch.zeros(t_tgt, dtype=_F64)
    covered = torch.zeros(t_tgt, dtype=torch.bool)

    if global_mode == "aretomo":
        from cryoet_alignment.io.aretomo3.aln import AreTomo3ALN

        _add_context_file(ctx, "context_aln", aln)
        aln_obj = AreTomo3ALN.from_file(str(aln))
        ctx_rows = TargetRowTable.from_template_aln(aln_obj)
        ctx_meta = ir.meta.model_copy(
            update={
                "projection_index": list(range(ctx_rows.n_rows)),
                "projection_valid": list(ctx_rows.active),
                "projection_order": list(ctx_rows.order),
                "projection_dose": list(ctx_rows.dose),
                "projection_angle_deg": list(ctx_rows.angle_deg),
                "projection_sec": list(ctx_rows.sec),
                "projection_dark": list(ctx_rows.dark),
                "projection_label": None,
                "projection_angle_kind": list(ctx_rows.angle_kind),
            }
        )
        cmatch = match_rows(
            ctx_meta, target_rows, row_map_file=row_map_file,
            deactivate_unmatched=True,
        )
        pix = ir.meta.pixel_size_image_a
        by_sec = {int(g.sec) - 1: g for g in aln_obj.GlobalAlignments}
        for ctx_row, tgt in enumerate(cmatch.row_map):
            if tgt < 0 or ctx_row not in by_sec:
                continue
            g = by_sec[ctx_row]
            angles[tgt] = -float(g.tilt)
            axis[tgt] = float(g.rot)
            offx[tgt] = float(g.tx) * pix
            offy[tgt] = float(g.ty) * pix
            covered[tgt] = True
    else:  # relion
        from cets_nonrigid.convert_relion import load_relion_source, relion_model_from_data
        from cets_nonrigid.fit.relion_global import _shift_corrections

        for name, path in (
            ("context_optimisation_set", optimisation_set),
            ("context_tomograms_star", tomograms_star),
        ):
            _add_context_file(ctx, name, path)
        # --tomo-name defaults to the store's series name (errors if that does
        # not uniquely select a tomogram in the given star)
        data, model, _pos, _names, _traj, _deform = load_relion_source(
            optimisation_set=optimisation_set, tomograms_star=tomograms_star,
            particles_star=None, motion_star=None,
            tomo_name=tomo_name or ir.meta.series_name or None, image_dims_px=(0, 0),
        )
        pix = data.pixel_size_a
        image_dims_px = tuple(ir.meta.image_dims_px)
        model = relion_model_from_data(data, image_dims_px)
        xt = data.xtilt_deg.to(_F64)
        level_x = float(xt.mean())
        corr = _shift_corrections(model.rotations, tuple(data.tomo_dims_px), image_dims_px, pix)
        t_ctx = data.nominal_stage_angle_deg.shape[0]
        ctx_meta = ir.meta.model_copy(
            update={
                "projection_index": list(range(t_ctx)),
                "projection_valid": [True] * t_ctx,
                "projection_order": list(range(t_ctx)),
                "projection_dose": [float(d) for d in data.pre_exposure],
                "projection_angle_deg": [float(a) for a in data.nominal_stage_angle_deg],
                "projection_sec": [-1] * t_ctx,
                "projection_dark": [False] * t_ctx,
                "projection_label": None,
                "projection_angle_kind": ["nominal"] * t_ctx,
            }
        )
        cmatch = match_rows(
            ctx_meta, target_rows, row_map_file=row_map_file,
            deactivate_unmatched=True,
        )
        for ctx_row, tgt in enumerate(cmatch.row_map):
            if tgt < 0:
                continue
            angles[tgt] = -float(data.ytilt_deg[ctx_row])
            axis[tgt] = float(data.zrot_deg[ctx_row])
            offx[tgt] = float(data.xshift_a[ctx_row]) - float(corr[ctx_row, 0])
            offy[tgt] = float(data.yshift_a[ctx_row]) - float(corr[ctx_row, 1])
            covered[tgt] = True

    return level_x, angles, axis, offx, offy, covered


def fit_store_to_warp(
    store_path,
    out_xml,
    *,
    template_xml=None,
    global_mode: str = "fit",
    aln=None,
    optimisation_set=None,
    tomograms_star=None,
    tomo_name=None,
    global_tol_px: float = DEFAULT_GLOBAL_TOL_PX_WARP,
    movement_grid: tuple = (5, 5),
    lam: float = 1e-3,
    row_map_file=None,
    deactivate_unmatched: bool = False,
    angles_inverted=None,
    max_condition=None,
    min_node_support=None,
    warn_node_support=None,
    volume_warp_grid=None,  # (W, H, D, L | None=T): opt-in GridVolumeWarp fit
    store_out=None,
) -> StoreFitWarpResult:
    import copy as _copy

    from warpylib import CubicGrid, LinearGrid4D

    from cets_nonrigid.io.warp_xml import load_warp_tiltseries, write_alignment_into_template

    _validate_global_mode(global_mode, template_xml, aln, optimisation_set, tomograms_star)
    ctx = _load_tilt_series_store(store_path)
    ir = ctx.contents.ir

    synth_report = None
    if template_xml is not None:
        _add_context_file(ctx, "template_xml", template_xml)
        template = load_warp_tiltseries(template_xml)
        target_rows = TargetRowTable.from_warp_ts(template.ts)
        match = match_rows(
            ir.meta, target_rows, row_map_file=row_map_file,
            deactivate_unmatched=deactivate_unmatched,
        )
    else:
        template, synth_report = _synthesize_fit_warp_template(ir, angles_inverted)
        target_rows = TargetRowTable.identity_of_ir(ir.meta)
        match = RowMatch(
            row_map=list(range(ir.n_projections)),
            target_active=list(target_rows.active),
            method="identity",
        )
    if row_map_file is not None:
        _add_context_file(ctx, "row_map", row_map_file)
    aligned = align_ir_rows(ir, match, target_rows)

    # --- target globals per mode ------------------------------------------
    ts_target = _copy.deepcopy(template.ts)
    ts_target.grid_movement_x = CubicGrid((1, 1, 1))
    ts_target.grid_movement_y = CubicGrid((1, 1, 1))
    ts_target.grid_volume_warp_x = LinearGrid4D((1, 1, 1, 1))
    ts_target.grid_volume_warp_y = LinearGrid4D((1, 1, 1, 1))
    ts_target.grid_volume_warp_z = LinearGrid4D((1, 1, 1, 1))
    active_mask = torch.tensor(match.target_active, dtype=torch.bool)
    ts_target.use_tilt = active_mask.clone()

    global_status = "not_applicable"
    g_train = g_held = None
    if global_mode == "fit":
        wfit, diag = fit_warp_globals_from_ir(aligned)
        global_status = diag.global_validation_status
        g_train, g_held = diag.rms_px_train, diag.rms_px_heldout
        active_idx = [t for t in range(aligned.n_projections) if bool(active_mask[t])]
        ts_target.level_angle_x = wfit.level_angle_x_deg
        ts_target.level_angle_y = 0.0
        angles = ts_target.angles.clone()
        axis = ts_target.tilt_axis_angles.clone()
        offx = ts_target.tilt_axis_offset_x.clone()
        offy = ts_target.tilt_axis_offset_y.clone()
        for k, t in enumerate(active_idx):
            angles[t] = float(wfit.angle_deg[k])
            axis[t] = float(wfit.axis_angle_deg[k])
            offx[t] = float(wfit.axis_offset_x_a[k])
            offy[t] = float(wfit.axis_offset_y_a[k])
        ts_target.angles, ts_target.tilt_axis_angles = angles, axis
        ts_target.tilt_axis_offset_x, ts_target.tilt_axis_offset_y = offx, offy
        if global_status == "not_evaluated":
            warnings.warn(
                "no held-out points in this store: the global fit is NOT independently "
                "validated (global_validation_status=not_evaluated)",
                stacklevel=2,
            )
    elif global_mode == "template":
        pass  # keep the template's own globals, verified by the gate below
    else:
        level_x, angles_c, axis_c, offx_c, offy_c, covered = _warp_globals_from_context(
            ctx, aligned, target_rows, global_mode, aln, optimisation_set, tomograms_star,
            tomo_name, row_map_file, deactivate_unmatched,
        )
        uncovered_active = active_mask & ~covered
        if uncovered_active.any():
            raise ValueError(
                f"context covers no globals for active target rows "
                f"{torch.nonzero(uncovered_active).flatten().tolist()}"
            )
        ts_target.level_angle_x = level_x
        ts_target.level_angle_y = 0.0
        angles = ts_target.angles.clone()
        axis = ts_target.tilt_axis_angles.clone()
        offx = ts_target.tilt_axis_offset_x.clone()
        offy = ts_target.tilt_axis_offset_y.clone()
        for t in range(target_rows.n_rows):
            if covered[t]:
                angles[t] = float(angles_c[t])
                axis[t] = float(axis_c[t])
                offx[t] = float(offx_c[t])
                offy[t] = float(offy_c[t])
        ts_target.angles, ts_target.tilt_axis_angles = angles, axis
        ts_target.tilt_axis_offset_x, ts_target.tilt_axis_offset_y = offx, offy

    # --- MANDATORY baseline gate, before locals and before any output ------
    check_rms_px = _global_consistency_gate(
        ts_target, aligned, global_tol_px, f"--global-mode {global_mode}"
    )

    from cets_nonrigid.fit.coverage import DEFAULT_MAX_CONDITION, DEFAULT_MIN_NODE_SUPPORT, DEFAULT_WARN_NODE_SUPPORT
    from cets_nonrigid.fit.warp_ts_fit import (
        fit_warp_locals,
        resolve_volume_warp_grid,
        volume_warp_fit_arrays,
        volume_warp_fit_attrs,
    )

    vw_grid = resolve_volume_warp_grid(volume_warp_grid, aligned.n_projections)
    fit = fit_warp_locals(
        aligned, ts_target, movement_grid=tuple(movement_grid), lam=lam, volume_warp_grid=vw_grid,
        max_condition=max_condition if max_condition is not None else DEFAULT_MAX_CONDITION,
        min_node_support=min_node_support if min_node_support is not None else DEFAULT_MIN_NODE_SUPPORT,
        warn_node_support=warn_node_support if warn_node_support is not None else DEFAULT_WARN_NODE_SUPPORT,
    )
    _tilt_series_gates(
        aligned, fit, ts_target, movement_grid,
        max_condition=max_condition, min_node_support=min_node_support,
        warn_node_support=warn_node_support,
    )
    vw_arrays, vw_dims = volume_warp_fit_arrays(fit.ts) if fit.volume_warp is not None else ({}, {})

    # --- emission (CTF: template preserved / generated placeholder; the
    #     store's native_files are NEVER consulted) -------------------------
    if synth_report is None:
        write_alignment_into_template(template.xml_bytes, fit.ts, out_xml, with_ctf=False)
    else:
        synth_report.warn_once("fit --to warp")
        from cets_nonrigid.io.warp_synth import atomic_write_validated

        atomic_write_validated(
            lambda tmp: write_alignment_into_template(template.xml_bytes, fit.ts, tmp, with_ctf=False),
            out_xml,
            load_warp_tiltseries,
        )

    store = _write_store_out(
        store_out, ctx, ir, match, target_rows,
        fit_attrs={
            "direction": "fit:warp",
            "global_mode": global_mode,
            "global_check_rms_px": check_rms_px,
            "global_validation_status": global_status,
            "global_rms_px_train": g_train,
            "global_rms_px_heldout": g_held,
            "heldout_status": fit.heldout_status,
            "rms_a_train": fit.rms_a_train,
            "rms_a_heldout": fit.rms_a_heldout,
            "p95_a_heldout": fit.p95_a_heldout,
            "coverage_heldout": fit.coverage_heldout,
            "pixel_size_a": ir.meta.pixel_size_image_a,
            "template_source": "template" if synth_report is None else "generated",
            "ctf_disposition": (
                "template CTF preserved unchanged" if synth_report is None
                else "generated placeholder CTF"
            ),
            **fit.meta,
            **volume_warp_fit_attrs(fit.volume_warp),
        },
        fit_arrays={
            **(
                {"per_target_projection_rms_heldout": torch.as_tensor(fit.per_tilt_rms_a_heldout)}
                if fit.heldout_status == "evaluated"
                else {}
            ),
            **vw_arrays,
        },
        fit_array_dims=vw_dims,
        target_files={"out_xml": Path(out_xml).read_bytes()},
    )
    return StoreFitWarpResult(
        out_xml=Path(out_xml),
        fit=fit,
        global_mode=global_mode,
        source_row_map=list(match.row_map),
        global_check_rms_px=check_rms_px,
        global_validation_status=global_status,
        global_rms_px_train=g_train,
        global_rms_px_heldout=g_held,
        template_source="template" if synth_report is None else "generated",
        store_out=store,
    )


def _tilt_series_gates(
    ir, fit, ts_target, movement_grid, *, max_condition, min_node_support, warn_node_support
):
    """r2w's data-rank/condition + node-support gate block, on any IR.

    Support is measured on the FITTED model (``fit.ts``, which carries the
    fitted volume warp) at the PREMOVEMENT positions — where the movement grid
    is actually sampled — and the reachable-node box is inflated by the fitted
    warp's maximum node norm (corners bound only the undeformed volume).
    ``ts_target`` is kept for its dimensions only.
    """
    from cets_nonrigid.fit.coverage import (
        DEFAULT_MAX_CONDITION,
        DEFAULT_MIN_NODE_SUPPORT,
        DEFAULT_WARN_NODE_SUPPORT,
        evaluate_gates,
        node_support,
        nodes_reachable_by_volume,
    )
    from cets_nonrigid.fit.warp_ts_fit import volume_warp_max_norm_a
    from cets_nonrigid.models.warp_ts import WarpTiltSeriesModel

    gx, gy = tuple(movement_grid)
    img_a = ts_target.image_dimensions_physical.to(_F64)
    nx = torch.linspace(0, float(img_a[0]), max(2, gx), dtype=_F64)
    ny = torch.linspace(0, float(img_a[1]), max(2, gy), dtype=_F64)
    nodes = torch.cartesian_prod(nx, ny)
    spacing = torch.tensor(
        [float(img_a[0]) / max(1, gx - 1), float(img_a[1]) / max(1, gy - 1)], dtype=_F64
    )
    fitted_model = WarpTiltSeriesModel(fit.ts)
    q_pre, q_valid = fitted_model.project_volume_premovement(ir.points)
    active_w = ir.weights.to(_F64) * ir.projection_valid.to(_F64) * q_valid.to(_F64)
    vol_a = ts_target.volume_dimensions_physical.to(_F64)
    corners = torch.tensor(
        [[x, y, z] for x in (0.0, float(vol_a[0]))
         for y in (0.0, float(vol_a[1])) for z in (0.0, float(vol_a[2]))],
        dtype=_F64,
    )
    corners_xy, _ = fitted_model.project_volume_global(corners)
    reachable = nodes_reachable_by_volume(
        nodes, corners_xy.to(_F64), spacing, inflate_a=volume_warp_max_norm_a(fit.ts)
    )
    support = node_support(q_pre.to(_F64), active_w, nodes, spacing, node_active=reachable)
    gate = evaluate_gates(
        support,
        fit.min_data_rank,
        gx * gy,
        fit.max_data_condition,
        max_condition=max_condition if max_condition is not None else DEFAULT_MAX_CONDITION,
        min_node_support=(
            min_node_support if min_node_support is not None else DEFAULT_MIN_NODE_SUPPORT
        ),
        warn_node_support=(
            warn_node_support if warn_node_support is not None else DEFAULT_WARN_NODE_SUPPORT
        ),
    )
    for w in gate.warnings:
        warnings.warn(w, stacklevel=3)
    if gate.failures:
        raise RuntimeError("scattered-fit gates failed: " + "; ".join(gate.failures))
    return gate


# ---------------------------------------------------------------------------
# fit --to aretomo
# ---------------------------------------------------------------------------


@dataclass
class StoreFitAretomoResult:
    out_aln: Path
    global_fit: object
    local_fit: object
    global_validation_status: str
    global_rms_px_train: float | None
    global_rms_px_heldout: float | None
    store_out: Path | None
    aln: object | None = None
    raw_pre_exposure: torch.Tensor | None = None
    aln_check: object | None = None
    source_row_map: list[int] | None = None


def fit_store_to_aretomo(
    store_path,
    out_aln,
    *,
    template_aln=None,
    patch_grid: tuple = (5, 5),
    patch_z: str = "lsq",
    global_tol_px: float = DEFAULT_GLOBAL_TOL_PX_ARETOMO,
    row_map_file=None,
    max_condition=None,
    min_node_support=None,
    warn_node_support=None,
    store_out=None,
) -> StoreFitAretomoResult:
    from cryoet_alignment.io.aretomo3.aln import AreTomo3ALN, DarkFrameInfo

    from cets_nonrigid.fit.aretomo_ts_fit import fit_aretomo_locals
    from cets_nonrigid.models.aretomo_ts import AretomoTsModel

    ctx = _load_tilt_series_store(store_path)
    ir = ctx.contents.ir
    meta = ir.meta
    pix = meta.pixel_size_image_a

    gfit, diag = fit_aretomo_globals_from_ir(ir)
    if diag.rms_px_train > global_tol_px:
        raise ValueError(
            f"IR-driven global fit residual {diag.rms_px_train:.3f} px exceeds "
            f"--global-tol-px {global_tol_px}"
        )
    if diag.global_validation_status == "not_evaluated":
        warnings.warn(
            "no held-out points in this store: the global fit is NOT independently "
            "validated (global_validation_status=not_evaluated)",
            stacklevel=2,
        )

    active = [t for t in range(ir.n_projections) if (meta.projection_valid or [True])[t]]
    order = sorted(range(len(active)), key=lambda k: float(gfit.tilt_deg[k]))
    rows = [active[k] for k in order]

    vol_a = tuple(float(d) * meta.pixel_size_volume_a for d in meta.volume_dims_px)
    model_global = AretomoTsModel(
        rot_deg=gfit.rot_deg[order].clone(),
        tilt_deg=gfit.tilt_deg[order].clone(),
        shifts_px=gfit.shifts_px[order].clone(),
        raw_size_px=tuple(meta.image_dims_px),
        pixel_size_a=pix,
        volume_dims_a=vol_a,
        local=None,
    )
    lfit = fit_aretomo_locals(ir, model_global, rows, patch_grid=tuple(patch_grid), patch_z=patch_z)

    # --- emission: template contract vs dense fitted-TILT order ------------
    if template_aln is not None:
        _add_context_file(ctx, "template_aln", template_aln)
        tmpl = AreTomo3ALN.from_file(str(template_aln))
        target_rows = TargetRowTable.from_template_aln(tmpl)
        fit_meta = meta.model_copy(
            update={
                "projection_index": list(range(len(rows))),
                "projection_valid": [True] * len(rows),
                "projection_order": list(range(len(rows))),
                "projection_dose": [float((meta.projection_dose or [0.0] * ir.n_projections)[r]) for r in rows],
                "projection_angle_deg": [float(gfit.tilt_deg[order[k]]) for k in range(len(rows))],
                "projection_sec": [-1] * len(rows),
                "projection_dark": [False] * len(rows),
                "projection_label": (
                    [meta.projection_label[r] for r in rows] if meta.projection_label else None
                ),
                "projection_angle_kind": ["effective"] * len(rows),
            }
        )
        tmatch = match_rows(fit_meta, target_rows, row_map_file=row_map_file)
        sec_of_fit_row = [target_rows.sec[tmatch.row_map[k]] for k in range(len(rows))]
        # row tables for --store-out: IR source rows -> template rows
        ir_to_target = [-1] * ir.n_projections
        for k, ir_row in enumerate(rows):
            ir_to_target[ir_row] = tmatch.row_map[k]
        store_match = RowMatch(
            row_map=ir_to_target, target_active=tmatch.target_active, method=tmatch.method
        )
        store_target_rows = target_rows
        dark_frames = [
            DarkFrameInfo(section_idx=j, val2=j, angle=float(target_rows.angle_deg[j]))
            for j in range(target_rows.n_rows)
            if target_rows.dark[j]
        ]
        alpha = float(tmpl.AlphaOffset or 0.0)
        beta = float(tmpl.BetaOffset or 0.0)
        raw_z = int(tmpl.RawSize[2])
        thickness = tmpl.Thickness
    else:
        if row_map_file is not None:
            raise ValueError("--row-map without --template-aln has nothing to map against")
        sec_of_fit_row = [k + 1 for k in range(len(rows))]  # dense 1-based, fitted-TILT order
        store_match = store_target_rows = None
        dark_frames = []
        alpha = beta = 0.0
        raw_z = len(rows)
        thickness = int(meta.volume_dims_px[2])

    from cets_nonrigid.io.aln import assemble_aln
    from cets_nonrigid.models.aretomo_ts import AreTomoLocalField

    sorted_rows = sorted(range(len(rows)), key=lambda k: sec_of_fit_row[k])
    perm = torch.tensor(sorted_rows, dtype=torch.long)
    local = lfit.model.local
    aln_out = assemble_aln(
        model_global=AretomoTsModel(
            rot_deg=model_global.rot_deg[perm],
            tilt_deg=model_global.tilt_deg[perm],
            shifts_px=model_global.shifts_px[perm],
            raw_size_px=tuple(meta.image_dims_px),
            pixel_size_a=pix,
            volume_dims_a=vol_a,
            local=None,
        ),
        local=(
            AreTomoLocalField(
                coord_xy=local.coord_xy[perm],
                shift_xy=local.shift_xy[perm],
                good=local.good[perm],
                raw_size_px=local.raw_size_px,
            )
            if local is not None
            else None
        ),
        sec_1b=[sec_of_fit_row[k] for k in sorted_rows],
        raw_size=(int(meta.image_dims_px[0]), int(meta.image_dims_px[1]), raw_z),
        dark_frames=dark_frames,
        alpha_offset=alpha,
        beta_offset=beta,
        thickness=thickness,
    )
    out_aln = Path(out_aln)
    if out_aln.exists():
        raise FileExistsError(f"{out_aln} already exists")
    from cets_nonrigid.io.aln import write_aln

    aln_check = write_aln(out_aln, aln_out, expect_rows=len(rows))
    # NO companion _CTF.txt: fit never reads or converts source-store CTF.
    # per-raw-section pre-exposure from the store's projection table (rows map to
    # raw sections through sec_of_fit_row; dark sections without a row stay unknown)
    raw_pre_exposure = None
    proj_dose = list(getattr(ir.meta, "projection_dose", None) or [])
    if proj_dose and max(rows) < len(proj_dose) and all(v is not None for v in proj_dose):
        pre = torch.full((raw_z,), float("nan"), dtype=torch.float64)
        for k, src_row in enumerate(rows):
            pre[sec_of_fit_row[k] - 1] = float(proj_dose[src_row])
        if torch.isfinite(pre).all():
            raw_pre_exposure = pre

    store = _write_store_out(
        store_out, ctx, ir, store_match, store_target_rows,
        fit_attrs={
            "direction": "fit:aretomo",
            "global_validation_status": diag.global_validation_status,
            "global_rms_px_train": diag.rms_px_train,
            "global_rms_px_heldout": diag.rms_px_heldout,
            "heldout_status": lfit.heldout_status,
            "rms_px_train": lfit.rms_px_train,
            "rms_px_heldout": lfit.rms_px_heldout,
            "p95_px_heldout": lfit.p95_px_heldout,
            "coverage_heldout": lfit.coverage_heldout,
            "pixel_size_a": pix,
            "ctf_disposition": "no CTF emitted (fit never reads source-store CTF)",
            **lfit.meta,
        },
        fit_arrays={},
        target_files={"out_aln": out_aln.read_bytes()},
    )
    return StoreFitAretomoResult(
        out_aln=out_aln,
        global_fit=gfit,
        local_fit=lfit,
        aln=aln_out,
        raw_pre_exposure=raw_pre_exposure,
        aln_check=aln_check,
        source_row_map=[next((sec_of_fit_row[k] - 1 for k, row in enumerate(rows) if row == i), -1) for i in range(ir.n_projections)],
        global_validation_status=diag.global_validation_status,
        global_rms_px_train=diag.rms_px_train,
        global_rms_px_heldout=diag.rms_px_heldout,
        store_out=store,
    )


# ---------------------------------------------------------------------------
# frame targets
# ---------------------------------------------------------------------------


@dataclass
class StoreFitFrameResult:
    out_path: Path
    fit: object
    template_source: str
    store_out: Path | None


def fit_store_to_mcaln(
    store_path, out_mcaln, *, patch_grid=(5, 5), fm_ref=-1, raw_frames_per_aligned=1,
    store_out=None,
) -> StoreFitFrameResult:
    from cets_nonrigid.convert import mcaln_from_motion_model
    from cets_nonrigid.fit.movie_fits import fit_mcaln_shifts

    ctx = _load_frame_series_store(store_path, compress_movie_rows=True)
    ir = ctx.contents.ir
    meta = ir.meta
    fit = fit_mcaln_shifts(
        ir, frame_size_px=tuple(meta.image_dims_px), pixel_size_a=meta.pixel_size_image_a,
        patch_grid=tuple(patch_grid), fm_ref=fm_ref,
    )
    mcaln = mcaln_from_motion_model(
        fit.model, n_frames=ir.n_projections, image_size_px=tuple(meta.image_dims_px),
        pixel_size_a=meta.pixel_size_image_a, patch_grid=tuple(patch_grid),
        fm_ref=fit.meta["fm_ref"], raw_frames_per_aligned=raw_frames_per_aligned,
    )
    out_mcaln = Path(out_mcaln)
    mcaln.to_file(out_mcaln)
    store = _write_store_out(
        store_out, ctx, ir, None, None,
        fit_attrs=_frame_fit_attrs("fit:mcaln", fit, meta),
        fit_arrays={},
        target_files={"out_mcaln": out_mcaln.read_bytes()},
    )
    return StoreFitFrameResult(out_mcaln, fit, "generated", store)


def fit_store_to_movie_xml(
    store_path, out_xml, *, template_xml=None, local_grid=(3, 3, 4), fraction_frames=1.0,
    lam=1e-3, movie_path=None, store_out=None,
) -> StoreFitFrameResult:
    from warpylib.movie import Movie
    from warpylib.movie.io import load_meta

    from cets_nonrigid.convert import _check_movie_path_stem, _reject_synthesis_options
    from cets_nonrigid.fit.movie_fits import fit_warp_movie
    from cets_nonrigid.io.warp_xml import write_movie_alignment_into_template

    ctx = _load_frame_series_store(store_path)
    ir = ctx.contents.ir
    meta = ir.meta
    image_dims_a = tuple(d * meta.pixel_size_image_a for d in meta.image_dims_px)

    synth_report = None
    if template_xml is not None:
        _reject_synthesis_options(movie_path=movie_path)
        _add_context_file(ctx, "template_xml", template_xml)
        template_bytes = Path(template_xml).read_bytes()
        template_movie = Movie()
        load_meta(template_movie, str(template_xml))
    else:
        from cets_nonrigid.io.warp_synth import movie_template_bytes, synthesize_movie

        template_movie, synth_report = synthesize_movie(
            pixel_size_a=meta.pixel_size_image_a, data_path=movie_path
        )
        template_bytes = movie_template_bytes(template_movie, "fit --to movie-xml")
        synth_report.warn_once("fit --to movie-xml")
        _check_movie_path_stem(movie_path, out_xml)

    fit = fit_warp_movie(
        ir, template_movie, n_frames=ir.n_projections, image_dims_a=image_dims_a,
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
    store = _write_store_out(
        store_out, ctx, ir, None, None,
        fit_attrs={
            **_frame_fit_attrs("fit:movie-xml", fit, meta),
            "template_source": "template" if synth_report is None else "generated",
            "ctf_disposition": (
                "template CTF preserved unchanged" if synth_report is None
                else "generated placeholder CTF"
            ),
        },
        fit_arrays={},
        target_files={"out_xml": Path(out_xml).read_bytes()},
    )
    return StoreFitFrameResult(
        Path(out_xml), fit, "template" if synth_report is None else "generated", store
    )


def fit_store_to_relion_motion(
    store_path, out_star, *, movie_name="", dose_rate=None, pre_exposure=None, voltage_kv=None,
    eer_upsampling=None, eer_grouping=None, store_out=None,
) -> StoreFitFrameResult:
    from cets_nonrigid.fit.relion_motion_fit import fit_relion_motion
    from cets_nonrigid.io.relion_motion_star import write_micrograph_motion_star

    ctx = _load_frame_series_store(store_path)
    ir = ctx.contents.ir
    meta = ir.meta
    fit = fit_relion_motion(
        ir, image_size_px=tuple(meta.image_dims_px), pixel_size_a=meta.pixel_size_image_a,
        movie_name=movie_name, dose_rate=dose_rate, pre_exposure=pre_exposure,
        voltage_kv=voltage_kv, eer_upsampling=eer_upsampling, eer_grouping=eer_grouping,
    )
    out_star = Path(out_star)
    write_micrograph_motion_star(out_star, fit.motion)
    store = _write_store_out(
        store_out, ctx, ir, None, None,
        fit_attrs=_frame_fit_attrs("fit:relion-motion", fit, meta),
        fit_arrays={},
        target_files={"out_star": out_star.read_bytes()},
    )
    return StoreFitFrameResult(out_star, fit, "generated", store)


def _frame_fit_attrs(direction, fit, meta) -> dict:
    def metric(name):
        return getattr(fit, f"{name}_a", None) if hasattr(fit, f"{name}_a") else getattr(
            fit, f"{name}_px", None
        )

    return {
        "direction": direction,
        "heldout_status": fit.heldout_status,
        "rms_train": getattr(fit, "rms_a_train", None) or getattr(fit, "rms_px_train", None),
        "rms_heldout": (
            getattr(fit, "rms_a_heldout", None)
            if getattr(fit, "rms_a_heldout", None) is not None
            else getattr(fit, "rms_px_heldout", None)
        ),
        "p95_heldout": (
            getattr(fit, "p95_a_heldout", None)
            if getattr(fit, "p95_a_heldout", None) is not None
            else getattr(fit, "p95_px_heldout", None)
        ),
        "coverage_heldout": getattr(fit, "coverage_heldout", None),
        "pixel_size_a": meta.pixel_size_image_a,
        **fit.meta,
    }


# ---------------------------------------------------------------------------
# --store-out
# ---------------------------------------------------------------------------


def _write_store_out(
    store_out, ctx: StoreFitContext, ir, match, target_rows, *, fit_attrs, fit_arrays,
    target_files, fit_array_dims: dict | None = None,
) -> Path | None:
    """NEW store: the ORIGINAL un-aligned IR + fit/target-row tables + the full
    context-file closure. The input store is never mutated. ``fit_array_dims``
    names the dimensions of arrays that are not per-target-projection (e.g. the
    rank-4 fitted volume-warp grids)."""
    if store_out is None:
        return None
    arrays = dict(fit_arrays)
    dims = {name: ["target_projection"] for name in arrays}
    if fit_array_dims:
        dims.update({k: list(v) for k, v in fit_array_dims.items()})
    if match is not None and target_rows is not None:
        t_tgt = target_rows.n_rows
        ir_to_target = torch.tensor(match.row_map, dtype=torch.int64)
        target_to_ir = torch.full((t_tgt,), -1, dtype=torch.int64)
        for src, tgt in enumerate(match.row_map):
            if tgt >= 0:
                target_to_ir[tgt] = src
        arrays["ir_to_target_row"] = ir_to_target
        arrays["target_to_ir_row"] = target_to_ir
        dims["ir_to_target_row"] = ["projection"]
        dims["target_to_ir_row"] = ["target_projection"]
        arrays["target_projection_angle_deg"] = torch.tensor(
            [float(a) for a in target_rows.angle_deg], dtype=torch.float64
        )
        arrays["target_projection_valid"] = torch.tensor(match.target_active, dtype=torch.bool)
        arrays["target_projection_dark"] = torch.tensor(
            [bool(d) for d in target_rows.dark], dtype=torch.bool
        )
        arrays["target_projection_sec"] = torch.tensor(
            [int(s) for s in target_rows.sec], dtype=torch.int32
        )
        for name in (
            "target_projection_angle_deg", "target_projection_valid",
            "target_projection_dark", "target_projection_sec",
        ):
            dims[name] = ["target_projection"]
        fit_attrs = {
            **fit_attrs,
            "target_projection_identity_angle_kind": list(target_rows.angle_kind),
        }
    context_sha = {
        name: hashlib.sha256(blob).hexdigest() for name, blob in ctx.context_files.items()
    }
    fit_attrs = {**fit_attrs, "context_files_sha256": context_sha}
    return DeformationStore.write(
        store_out,
        ir,
        native_files=dict(ctx.contents.native_files),
        target_files={**target_files, **{f"context_{k}": v for k, v in ctx.context_files.items()}},
        fit_attrs={k: _jsonable(v) for k, v in fit_attrs.items()},
        fit_arrays=arrays,
        fit_array_dims=dims,
        command_line=" ".join(sys.argv),
    )


def _jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


# ---------------------------------------------------------------------------
# dump-ir (source -> IR-only store)
# ---------------------------------------------------------------------------


def _particles_ir(model, positions_a, names, image_dims_a, meta, heldout_fraction, heldout_seed):
    from cets_nonrigid.ir.build import build_ir_tilt_series_from_points

    return build_ir_tilt_series_from_points(
        model, positions_a, image_dims_a,
        meta=meta.model_copy(update={"sampling": "particles"}),
        point_names=names, heldout_fraction=heldout_fraction, heldout_seed=heldout_seed,
    )


def dump_ir_warp(
    xml_path, *, pixel_size_a, grid_shape=(15, 15, 5), positions_a=None, names=None, dims_override=None,
    heldout_fraction=0.2, heldout_seed=20260828,
):
    """Sample a Warp tilt-series model into an IR (w2a's load half)."""
    from cets_nonrigid.convert import _w2a_ir_meta
    from cets_nonrigid.io.warp_xml import load_warp_tiltseries
    from cets_nonrigid.ir.build import build_ir_tilt_series

    template = load_warp_tiltseries(xml_path, dims_override=dims_override)
    meta = _w2a_ir_meta(template, pixel_size_a)
    if positions_a is not None:
        ir = _particles_ir(
            template.model, positions_a, names, template.model.image_dims_a, meta,
            heldout_fraction, heldout_seed,
        )
    else:
        ir = build_ir_tilt_series(
            template.model, template.model.volume_dims_a, template.model.image_dims_a,
            meta=meta, grid_shape=tuple(grid_shape), heldout_seed=heldout_seed,
        )
    return ir, {"source_xml": template.xml_bytes}


def dump_ir_aretomo(
    aln_path, *, template_xml=None, pixel_size_a=None, tomo_size_px=None,
    grid_shape=(15, 15, 5), positions_a=None, names=None,
    heldout_fraction=0.2, heldout_seed=20260828, raw_dose=None,
):
    """Sample an AreTomo3 model into an IR (a2w's load half; the template may
    be a real XML or the Workstream-T synthesized one)."""
    from cryoet_alignment.io.aretomo3.aln import AreTomo3ALN

    from cets_nonrigid.convert import _derive_pixel_size, _ir_meta, _PermutedAlnModel, _synthesize_a2w_template
    from cets_nonrigid.io.aln import load_aln, match_tilts
    from cets_nonrigid.io.warp_xml import load_warp_tiltseries
    from cets_nonrigid.ir.build import build_ir_tilt_series

    aln_probe = AreTomo3ALN.from_file(str(aln_path))
    native = {"aln": Path(aln_path).read_bytes()}
    if template_xml is not None:
        template = load_warp_tiltseries(template_xml)
        pix = _derive_pixel_size(template.model.image_dims_a, aln_probe)
        native["template_xml"] = template.xml_bytes
    else:
        if pixel_size_a is None or tomo_size_px is None:
            raise ValueError("template-free dump-ir --from aretomo requires pixel_size_a and tomo_size_px")
        pix = float(pixel_size_a)
        if raw_dose is None:
            # The synthesized template is a PROBE for sampling the AreTomo3 model; it
            # is never written. A Warp series cannot be built without a dose, so use
            # a placeholder that is stripped from the IR below (projection_dose = 0 =
            # unknown): such a store can feed every target except a template-free
            # Warp synthesis, which refuses with a nudge.
            from cets_nonrigid.io.aln import raw_tilts_from_aln
            from cets_nonrigid.io.dose import raw_dose_assume_stack_order

            probe_dose = raw_dose_assume_stack_order(raw_tilts_from_aln(aln_probe), dose_per_tilt=1.0)
            warnings.warn(
                "dump-ir --from aretomo without a template or dose: the store's projection dose is "
                "unknown (written as 0); a template-free `fit --to warp` from it will be refused",
                stacklevel=2,
            )
        else:
            probe_dose = raw_dose
        template, _report = _synthesize_a2w_template(
            aln_probe, pixel_size_a=pix, tomo_size_px=tomo_size_px, raw_dose=probe_dose,
            tilt_images=None, angles_inverted=None, voltage_kv=None, cs_mm=None,
            amplitude_contrast=None,
        )
    aln_series = load_aln(
        aln_path, pixel_size_a=pix, volume_dims_a=tuple(template.model.volume_dims_a.tolist())
    )
    match = match_tilts(template.ts.angles, template.ts.use_tilt, aln_series)
    perm_model = _PermutedAlnModel(aln_series, match, template.ts.n_tilts)
    meta = _ir_meta(template, aln_series, match, pix)
    if template_xml is None and raw_dose is None:
        meta = meta.model_copy(update={"projection_dose": [0.0] * template.ts.n_tilts})  # unknown, never the probe's
    if positions_a is not None:
        ir = _particles_ir(
            perm_model, positions_a, names, template.model.image_dims_a, meta,
            heldout_fraction, heldout_seed,
        )
    else:
        ir = build_ir_tilt_series(
            perm_model, template.model.volume_dims_a, template.model.image_dims_a,
            meta=meta, grid_shape=tuple(grid_shape), heldout_seed=heldout_seed,
        )
    return ir, native


def dump_ir_relion(
    *, optimisation_set=None, tomograms_star=None, particles_star=None, motion_star=None,
    tomo_name, image_size_px, heldout_fraction=0.2, heldout_seed=20260828, overrides=None, project_root=None,
):
    """Sample a RELION source at its own particles, star row order, identity
    map, no template (row alignment happens at fit time)."""
    from cets_nonrigid.convert_relion import load_relion_source, relion_model_from_data
    from cets_nonrigid.ir.core import IRMeta, row_labels_from_paths
    from cets_nonrigid.models.relion_ts import RelionParticleSetModel

    data, model, positions, names, trajectories, _deform = load_relion_source(
        optimisation_set=optimisation_set, tomograms_star=tomograms_star,
        particles_star=particles_star, motion_star=motion_star,
        tomo_name=tomo_name, image_dims_px=(0, 0), overrides=overrides, project_root=project_root,
    )
    image_dims_px = (int(image_size_px[0]), int(image_size_px[1]))
    model = relion_model_from_data(data, image_dims_px)
    deformation = None
    if data.has_deformations:
        from cets_nonrigid.models.relion_deform import deformation_field

        deformation = deformation_field(
            data.deformation_type, data.deformation_grid, image_dims_px, data.deformation_coeffs
        )
    source = RelionParticleSetModel(
        model, positions, trajectories_a=trajectories, deformation=deformation
    )
    pix = data.pixel_size_a
    t = data.nominal_stage_angle_deg.shape[0]
    meta = IRMeta(
        kind="tilt_series",
        series_name=tomo_name,
        pixel_size_image_a=pix,
        image_dims_px=image_dims_px,
        volume_dims_px=tuple(int(d) for d in data.tomo_dims_px),
        pixel_size_volume_a=pix,
        projection_index=list(range(t)),
        projection_valid=[True] * t,
        projection_order=list(range(t)),
        projection_dose=[float(d) for d in data.pre_exposure],
        projection_angle_deg=[float(a) for a in data.nominal_stage_angle_deg],
        projection_sec=[-1] * t,
        projection_dark=[False] * t,
        source_tool="relion5",
        sampling="particles",
        projection_label=row_labels_from_paths(data.micrograph_names),
        projection_angle_kind=["nominal"] * t,
    )
    ir = _particles_ir(
        source, positions.to(_F64),
        names, torch.tensor([d * pix for d in image_dims_px], dtype=_F64), meta,
        heldout_fraction, heldout_seed,
    )
    ir.native_data = data
    from cets_nonrigid.io.relion_star import read_optimisation_set, resolve_star_ref
    if optimisation_set is not None:
        refs = read_optimisation_set(optimisation_set)
        tomograms_star = tomograms_star or resolve_star_ref(refs["tomograms"], optimisation_set, project_root)
        particles_star = particles_star or (resolve_star_ref(refs["particles"], optimisation_set, project_root) if refs["particles"] else None)
        motion_star = motion_star or (resolve_star_ref(refs["trajectories"], optimisation_set, project_root) if refs["trajectories"] else None)
    if data.micrograph_names:
        def absolute_image(value):
            prefix, sep, filename = str(value).partition("@")
            resolved = resolve_star_ref(filename if sep else prefix, tomograms_star, project_root).resolve()
            return f"{prefix}@{resolved}" if sep else str(resolved)
        data.micrograph_names = [absolute_image(value) for value in data.micrograph_names]
    native = {}
    for name, path in (
        ("optimisation_set", optimisation_set), ("tomograms_star", tomograms_star),
        ("particles_star", particles_star), ("motion_star", motion_star),
    ):
        if path is not None:
            native[name] = Path(path).read_bytes()
    return ir, native


def dump_ir_warp_movie(
    xml_path, *, image_size_px, pixel_size_a, n_frames, fraction_frames=1.0,
    grid_shape=(11, 11), heldout_seed=20260828,
):
    from cets_nonrigid.convert import _frame_ir_meta
    from cets_nonrigid.ir.build import build_ir_frame_series
    from cets_nonrigid.models.warp_movie import WarpMovieModel

    image_dims_a = (image_size_px[0] * pixel_size_a, image_size_px[1] * pixel_size_a)
    model = WarpMovieModel.from_xml(
        xml_path, n_frames=n_frames, image_dims_a=image_dims_a, fraction_frames=fraction_frames
    )
    meta = _frame_ir_meta(Path(xml_path).stem, pixel_size_a, image_size_px, n_frames, "warp")
    ir = build_ir_frame_series(
        model, torch.tensor(image_dims_a), meta=meta, grid_shape=tuple(grid_shape),
        heldout_seed=heldout_seed,
    )
    return ir, {"source_xml": Path(xml_path).read_bytes()}


def dump_ir_mcaln(mcaln_path, *, grid_shape=(11, 11), heldout_seed=20260828):
    from cets_nonrigid.convert import _frame_ir_meta
    from cets_nonrigid.io.motion_txt import McAln
    from cets_nonrigid.ir.build import build_ir_frame_series

    mcaln = McAln.from_file(mcaln_path)
    model = mcaln.to_model()
    pix = mcaln.alignment_pixel_size_a
    size_px = mcaln.alignment_image_size_px
    image_dims_a = (size_px[0] * pix, size_px[1] * pix)
    meta = _frame_ir_meta(
        Path(mcaln_path).stem, pix, size_px, mcaln.aligned_frame_count, "aretomo3-motion"
    )
    ir = build_ir_frame_series(
        model, torch.tensor(image_dims_a), meta=meta, grid_shape=tuple(grid_shape),
        heldout_seed=heldout_seed,
    )
    return ir, {"mcaln": Path(mcaln_path).read_bytes()}


def dump_ir_relion_motion(star_path, *, grid_shape=(11, 11), heldout_seed=20260828):
    from cets_nonrigid.convert import _frame_ir_meta
    from cets_nonrigid.io.relion_motion_star import read_micrograph_motion_star
    from cets_nonrigid.ir.build import build_ir_frame_series

    motion = read_micrograph_motion_star(star_path)
    model = motion.to_model()
    pix = motion.pixel_size_a
    image_dims_a = (motion.image_size_px[0] * pix, motion.image_size_px[1] * pix)
    meta = _frame_ir_meta(
        Path(star_path).stem, pix, motion.image_size_px, motion.n_frames, "relion-motion"
    )
    ir = build_ir_frame_series(
        model, torch.tensor(image_dims_a), meta=meta, grid_shape=tuple(grid_shape),
        heldout_seed=heldout_seed,
    )
    ir.native_data = motion
    return ir, {"motion_star": Path(star_path).read_bytes()}
