"""End-to-end conversion pipelines.

a2w: AreTomo3 .aln -> Warp XML (closed-form globals + linear movement fit).
w2a: Warp XML -> AreTomo3 .aln (fitted globals + per-tilt IDW local fit).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
from cryoet_alignment.io.aretomo3.aln import AreTomo3ALN
from warpylib import CubicGrid, LinearGrid4D

from cets_nonrigid.fit.aretomo_global import GlobalFitResult
from cets_nonrigid.fit.aretomo_ts_fit import AretomoLocalFitResult
from cets_nonrigid.fit.movie_fits import McAlnFitResult, WarpMovieFitResult
from cets_nonrigid.fit.warp_ts_fit import WarpTsFitResult, fit_warp_movement, movement_grid_dims
from cets_nonrigid.io.aln import AlnSeries, TiltMatch, load_aln, match_tilts
from cets_nonrigid.io.store import DeformationStore
from cets_nonrigid.io.warp_xml import DimsOverride, WarpSeries, load_warp_tiltseries, write_alignment_into_template
from cets_nonrigid.ir.build import build_ir_frame_series, build_ir_tilt_series
from cets_nonrigid.ir.core import IRMeta, row_labels_from_paths
from cets_nonrigid.ir.sampling import volume_grid
from cets_nonrigid.models.warp_ts import WarpTiltSeriesModel

#: The closed-form global mapping is exact (zero level angles); anything above
#: this is a wiring error, not alignment disagreement.
GLOBAL_CHECK_TOL_A = 0.5


class _PermutedAlnModel:
    """Presents the .aln model in Warp file order (unmatched tilts invalid)."""

    def __init__(self, aln_series: AlnSeries, match: TiltMatch, n_warp: int):
        self._aln = aln_series
        self._perm = match.aln_to_warp
        self.n_projections = n_warp

    def _expand(self, xy_aln, valid_aln):
        _t_aln, n, _ = xy_aln.shape
        xy = torch.zeros(self.n_projections, n, 2, dtype=xy_aln.dtype)
        valid = torch.zeros(self.n_projections, n, dtype=torch.bool)
        for t, w in enumerate(self._perm):
            xy[w] = xy_aln[t]
            valid[w] = valid_aln[t]
        return xy, valid

    def project_volume(self, points_3d):
        return self._expand(*self._aln.model.project_volume(points_3d))

    def project_volume_global(self, points_3d):
        return self._expand(*self._aln.model.project_volume_global(points_3d))


@dataclass
class A2WResult:
    out_xml: Path
    fit: WarpTsFitResult
    global_check_rms_a: float
    match: TiltMatch
    pixel_size_a: float
    store: Path | None
    ctf_converted: bool = False
    template_source: str = "template"
    defaulted_fields: list | None = None


def _derive_pixel_size(image_dims_a: torch.Tensor, aln: AreTomo3ALN) -> float:
    px = float(image_dims_a[0]) / int(aln.RawSize[0])
    py = float(image_dims_a[1]) / int(aln.RawSize[1])
    if abs(px - py) > 1e-3 * px:
        raise ValueError(
            f"anisotropic pixel size derived from template vs .aln RawSize: "
            f"({px:.5f}, {py:.5f}) A/px - not supported"
        )
    return px


def _build_target_ts(template: WarpSeries, aln_series: AlnSeries, match: TiltMatch):
    """Closed-form globals: Warp TiltSeries with .aln geometry, zero level
    angles, zero local grids, darks disabled."""
    ts = copy.deepcopy(template.ts)
    ts.level_angle_x = 0.0
    ts.level_angle_y = 0.0

    angles = ts.angles.clone()
    axis = ts.tilt_axis_angles.clone()
    offx = ts.tilt_axis_offset_x.clone()
    offy = ts.tilt_axis_offset_y.clone()
    use = torch.zeros(ts.n_tilts, dtype=torch.bool)

    g = aln_series.model
    pix = g.pixel_size_a
    for t_aln, t_warp in enumerate(match.aln_to_warp):
        angles[t_warp] = float(-g.tilt_deg[t_aln])
        axis[t_warp] = float(g.rot_deg[t_aln])
        offx[t_warp] = float(g.shifts_px[t_aln, 0]) * pix
        offy[t_warp] = float(g.shifts_px[t_aln, 1]) * pix
        use[t_warp] = True

    ts.angles = angles
    ts.tilt_axis_angles = axis
    ts.tilt_axis_offset_x = offx
    ts.tilt_axis_offset_y = offy
    ts.use_tilt = use & ts.use_tilt.to(torch.bool)

    ts.grid_movement_x = CubicGrid((1, 1, 1))
    ts.grid_movement_y = CubicGrid((1, 1, 1))
    ts.grid_volume_warp_x = LinearGrid4D((1, 1, 1, 1))
    ts.grid_volume_warp_y = LinearGrid4D((1, 1, 1, 1))
    ts.grid_volume_warp_z = LinearGrid4D((1, 1, 1, 1))
    return ts


def _global_consistency_rms(
    warp_model: WarpTiltSeriesModel, aln_series: AlnSeries, match: TiltMatch
) -> float:
    """The closed-form mapping is exact — verify it at runtime."""
    points = volume_grid(warp_model.volume_dims_a.to(torch.float64), (4, 4, 3))
    w_xy, w_valid = warp_model.project_volume_global(points)
    a_xy, _ = aln_series.model.project_volume_global(points)
    perm = torch.tensor(match.aln_to_warp)
    diff = (w_xy[perm].to(torch.float64) - a_xy.to(torch.float64))[w_valid[perm]]
    return float(diff.pow(2).mean().sqrt()) if diff.numel() else 0.0


def _ir_meta(template: WarpSeries, aln_series: AlnSeries, match: TiltMatch, pix: float) -> IRMeta:
    ts = template.ts
    t = ts.n_tilts
    sec = [-1] * t
    dark = [True] * t
    for t_aln, t_warp in enumerate(match.aln_to_warp):
        sec[t_warp] = int(aln_series.aln.GlobalAlignments[t_aln].sec)
        dark[t_warp] = False
    img_px = (ts.image_dimensions_physical / pix).round().to(torch.int64)
    vol_px = (ts.volume_dimensions_physical / pix).round().to(torch.int64)
    return IRMeta(
        kind="tilt_series",
        series_name=Path(ts.path).stem if ts.path else "",
        pixel_size_image_a=pix,
        image_dims_px=(int(img_px[0]), int(img_px[1])),
        volume_dims_px=(int(vol_px[0]), int(vol_px[1]), int(vol_px[2])),
        pixel_size_volume_a=pix,
        projection_index=list(range(t)),
        projection_valid=[not d for d in dark],
        projection_order=list(range(t)),
        projection_dose=[float(d) for d in ts.dose],
        projection_angle_deg=[float(a) for a in ts.angles],
        projection_sec=sec,
        projection_dark=dark,
        source_tool="aretomo3",
        projection_label=row_labels_from_paths(getattr(ts, "tilt_movie_paths", None)),
    )


def _reject_synthesis_options(**options) -> None:
    """Template mode must never silently ignore synthesis-only inputs."""
    given = [name for name, value in options.items() if value is not None]
    if given:
        raise ValueError(
            f"synthesis-only option(s) {', '.join(given)} are not allowed together with a "
            "template XML - drop the template for a generated output, or drop the option(s)"
        )


def _synthesize_a2w_template(
    aln_probe: AreTomo3ALN,
    *,
    pixel_size_a: float,
    tomo_size_px: tuple[int, int, int],
    raw_dose,
    tilt_images,
    angles_inverted: bool | None,
    voltage_kv: float | None,
    cs_mm: float | None,
    amplitude_contrast: float | None,
):
    """Template-free a2w: synthesize the Warp 'template' from the .aln alone —
    rows in raw-section order (darks included, UseTilt=False), Warp-convention
    angles (Angle = -TILT, level angles zero), geometry from RawSize/--tomo-size."""
    from cets_nonrigid.io.aln import raw_tilts_from_aln
    from cets_nonrigid.io.warp_synth import synthesize_tilt_series, synthesized_template_series

    raw_tilts = raw_tilts_from_aln(aln_probe)
    r = int(aln_probe.RawSize[2])
    # A real Warp template carries NOMINAL stage angles (Warp stage = -Angle);
    # regular .aln rows bake AlphaOffset into TILT, DarkFrame rows record the
    # raw stage angle directly.
    stage = raw_tilts.clone()
    alpha = float(aln_probe.AlphaOffset or 0.0)
    for g in aln_probe.GlobalAlignments:
        stage[int(g.sec) - 1] = float(g.tilt) - alpha
    use = torch.zeros(r, dtype=torch.bool)
    axis = torch.zeros(r, dtype=torch.float64)
    offx = torch.zeros(r, dtype=torch.float64)
    offy = torch.zeros(r, dtype=torch.float64)
    first_rot = float(aln_probe.GlobalAlignments[0].rot) if aln_probe.GlobalAlignments else 0.0
    axis[:] = first_rot  # dark rows: placeholder axis (UseTilt=False, never used)
    for g in aln_probe.GlobalAlignments:
        row = int(g.sec) - 1
        use[row] = True
        axis[row] = float(g.rot)
        offx[row] = float(g.tx) * pixel_size_a
        offy[row] = float(g.ty) * pixel_size_a

    ts, report = synthesize_tilt_series(
        angles_deg=-stage,
        use_tilt=use,
        axis_angles_deg=axis,
        axis_offset_x_a=offx,
        axis_offset_y_a=offy,
        image_dims_a=(
            int(aln_probe.RawSize[0]) * pixel_size_a,
            int(aln_probe.RawSize[1]) * pixel_size_a,
        ),
        volume_dims_a=tuple(float(v) * pixel_size_a for v in tomo_size_px),
        pixel_size_a=pixel_size_a,
        dose=raw_dose.pre_exposure if raw_dose is not None else None,
        movie_paths=tilt_images,
        angles_inverted=bool(angles_inverted) if angles_inverted is not None else False,
        angles_inverted_known=angles_inverted is not None,
        voltage_kv=voltage_kv,
        cs_mm=cs_mm,
        amplitude_contrast=amplitude_contrast,
    )
    return synthesized_template_series(ts, "a2w"), report


def aretomo_to_warp(
    aln_path: str | Path,
    template_xml: str | Path | None,
    out_xml: str | Path,
    *,
    grid_shape: tuple[int, int, int] = (15, 15, 5),
    movement_grid: tuple[int, int] | None = None,
    lam: float = 1e-3,
    store_path: str | Path | None = None,
    ctf_file: str | Path | None = None,
    ctf_voltage_kv: float | None = None,
    ctf_cs_mm: float | None = None,
    ctf_amp_contrast: float | None = None,
    pixel_size_a: float | None = None,
    tomo_size_px: tuple[int, int, int] | None = None,
    raw_dose=None,
    tilt_images=None,
    angles_inverted: bool | None = None,
) -> A2WResult:
    """Convert an AreTomo3 .aln (incl. locals) into a Warp tilt-series XML.

    With a ``template_xml`` the current metadata-overlay behavior is preserved
    byte-for-byte. Without one, a valid Warp model is synthesized directly:
    ``pixel_size_a`` and ``tomo_size_px`` are then required, and dose
    (``raw_dose``), per-raw-section image paths (``tilt_images``) and
    ``angles_inverted`` are optional — every defaulted field is warned about
    once and recorded on the result.

    With ``ctf_file`` (an AreTomo3 ``_CTF.txt``), per-tilt CTF grids are also
    written into the output XML. Voltage/Cs/amplitude-contrast are preserved
    and validated from the template XML when one is given (the ``_CTF.txt``
    cannot supply them); the ``ctf_*`` arguments override the template values
    and are the only source of those scalars in template-free mode.
    """
    aln_probe = AreTomo3ALN.from_file(str(aln_path))

    synth_report = None
    if template_xml is not None:
        _reject_synthesis_options(
            pixel_size_a=pixel_size_a,
            tomo_size_px=tomo_size_px,
            raw_dose=raw_dose,
            tilt_images=tilt_images,
            angles_inverted=angles_inverted,
        )
        template = load_warp_tiltseries(template_xml)
        pix = _derive_pixel_size(template.model.image_dims_a, aln_probe)
    else:
        if pixel_size_a is None or tomo_size_px is None:
            raise ValueError(
                "template-free a2w requires pixel_size_a and tomo_size_px "
                "(physical geometry is never guessed)"
            )
        pix = float(pixel_size_a)
        template, synth_report = _synthesize_a2w_template(
            aln_probe,
            pixel_size_a=pix,
            tomo_size_px=tomo_size_px,
            raw_dose=raw_dose,
            tilt_images=tilt_images,
            angles_inverted=angles_inverted,
            voltage_kv=ctf_voltage_kv,
            cs_mm=ctf_cs_mm,
            amplitude_contrast=ctf_amp_contrast,
        )
    aln_series = load_aln(
        aln_path,
        pixel_size_a=pix,
        volume_dims_a=tuple(template.model.volume_dims_a.tolist()),
    )
    match = match_tilts(template.ts.angles, template.ts.use_tilt, aln_series)

    ts_target = _build_target_ts(template, aln_series, match)
    check = _global_consistency_rms(WarpTiltSeriesModel(ts_target), aln_series, match)
    if check > GLOBAL_CHECK_TOL_A:
        raise RuntimeError(
            f"closed-form global mapping check failed: {check:.3f} A RMS "
            f"(> {GLOBAL_CHECK_TOL_A} A) - conversion aborted"
        )

    if movement_grid is None:
        g = max(2, round(float(aln_probe.NumPatches) ** 0.5))
        movement_grid = movement_grid_dims((g, g))

    perm_model = _PermutedAlnModel(aln_series, match, template.ts.n_tilts)
    ir = build_ir_tilt_series(
        perm_model,
        volume_dims_a=template.model.volume_dims_a,
        image_dims_a=template.model.image_dims_a,
        meta=_ir_meta(template, aln_series, match, pix),
        grid_shape=grid_shape,
    )

    fit = fit_warp_movement(ir, ts_target, movement_grid=movement_grid, lam=lam)

    ctf_converted = False
    ctf_bytes = None
    if ctf_file is not None:
        from cets_nonrigid.ctf import tiltctf_from_aretomo, tiltctf_to_warp
        from cets_nonrigid.io.ctf_aretomo import AreTomoCtfFile

        ctf_bytes = Path(ctf_file).read_bytes()
        parsed = AreTomoCtfFile.from_file(ctf_file)
        ctf = tiltctf_from_aretomo(parsed, template.ts.angles)
        # template supplies voltage/Cs/amplitude; CLI overrides win
        ctf.voltage_kv = ctf_voltage_kv if ctf_voltage_kv is not None else float(template.ts.ctf.voltage)
        ctf.cs_mm = ctf_cs_mm if ctf_cs_mm is not None else float(template.ts.ctf.cs)
        ctf.amplitude_contrast = (
            ctf_amp_contrast if ctf_amp_contrast is not None else float(template.ts.ctf.amplitude)
        )
        tiltctf_to_warp(fit.ts, ctf)
        ctf_converted = True

    if synth_report is not None:
        if ctf_converted and "placeholder per-tilt CTF" in synth_report.defaulted_experimental:
            synth_report.defaulted_experimental.remove("placeholder per-tilt CTF")
        synth_report.warn_once("a2w")

    if synth_report is None:
        write_alignment_into_template(template.xml_bytes, fit.ts, out_xml, with_ctf=ctf_converted)
    else:
        from cets_nonrigid.io.warp_synth import atomic_write_validated

        atomic_write_validated(
            lambda tmp: write_alignment_into_template(
                template.xml_bytes, fit.ts, tmp, with_ctf=ctf_converted
            ),
            out_xml,
            load_warp_tiltseries,
        )

    native_files = {"aln": aln_series.aln_bytes}
    if synth_report is None:
        # only an actually-provided template belongs in the provenance snapshot
        native_files["template_xml"] = template.xml_bytes
    if ctf_bytes is not None:
        native_files["ctf_txt"] = ctf_bytes

    store = None
    if store_path is not None:
        store = DeformationStore.write(
            store_path,
            ir,
            native_files=native_files,
            target_files={"out_xml": Path(out_xml).read_bytes()},
            fit_attrs={
                "direction": "a2w",
                "rms_a_train": fit.rms_a_train,
                "heldout_status": fit.heldout_status,
                "rms_a_heldout": fit.rms_a_heldout,
                "p95_a_heldout": fit.p95_a_heldout,
                "max_a_heldout": fit.max_a_heldout,
                "coverage_heldout": fit.coverage_heldout,
                "min_rank": fit.min_rank,
                "max_condition": fit.max_condition,
                "regularization_norm": fit.regularization_norm,
                "global_check_rms_a": check,
                "pixel_size_a": pix,
                "template_source": "template" if synth_report is None else "generated",
                "defaulted_fields": (
                    [] if synth_report is None else list(synth_report.defaulted_experimental)
                ),
                **fit.meta,
            },
            fit_arrays=(
                {"per_projection_rms_heldout": fit.per_tilt_rms_a_heldout}
                if fit.heldout_status == "evaluated"
                else {}
            ),
        )

    return A2WResult(
        out_xml=Path(out_xml),
        fit=fit,
        global_check_rms_a=check,
        match=match,
        pixel_size_a=pix,
        store=store,
        ctf_converted=ctf_converted,
        template_source="template" if synth_report is None else "generated",
        defaulted_fields=(
            None if synth_report is None else list(synth_report.defaulted_experimental)
        ),
    )


# ---------------------------------------------------------------------------
# w2a
# ---------------------------------------------------------------------------


@dataclass
class W2AResult:
    out_aln: Path
    global_fit: GlobalFitResult
    local_fit: AretomoLocalFitResult
    pixel_size_a: float
    store: Path | None
    ctf_path: Path | None = None
    aln: object | None = None  # the emitted AreTomo3ALN model
    raw_pre_exposure: torch.Tensor | None = None  # per raw row (= XML row) pre-exposure
    aln_check: object | None = None  # io.aln_check.AlnCheck


def warp_to_aretomo(
    xml_path: str | Path,
    out_aln: str | Path,
    *,
    pixel_size_a: float,
    template_aln: str | Path | None = None,
    patch_grid: tuple[int, int] = (5, 5),
    patch_z: str = "lsq",
    grid_shape: tuple[int, int, int] = (15, 15, 5),
    global_tol_px: float = 5.0,
    store_path: str | Path | None = None,
    write_ctf: bool = True,
    max_patch_shift_px: float | None = None,
    dims_override: DimsOverride | None = None,
) -> W2AResult:
    """Convert a Warp tilt-series XML into an AreTomo3 .aln (incl. locals).

    With ``write_ctf`` (default), a companion ``<stem>_CTF.txt`` is emitted
    beside the .aln whenever the source XML carries per-tilt CTF grids.

    Globals are FITTED (LevelAngleX and rounding factors are not closed-form
    representable); the representable remainder is baked into the local
    shifts. ``template_aln`` supplies BetaOffset/dark-frame/SEC bookkeeping.
    ``dims_override`` supplies image/volume dimensions for XMLs Warp re-saved
    with zeroed root attributes (nothing on disk is edited).
    """
    from cryoet_alignment.io.aretomo3.aln import DarkFrameInfo

    from cets_nonrigid.fit.aretomo_global import fit_aretomo_globals
    from cets_nonrigid.fit.aretomo_ts_fit import fit_aretomo_locals
    from cets_nonrigid.models.aretomo_ts import AretomoTsModel

    out_aln = Path(out_aln)
    if out_aln.exists():
        raise FileExistsError(f"{out_aln} already exists")

    template = load_warp_tiltseries(xml_path, dims_override=dims_override)
    ts = template.ts
    pix = float(pixel_size_a)
    img_px = (template.model.image_dims_a.to(torch.float64) / pix).round().to(torch.int64)
    vol_a = tuple(template.model.volume_dims_a.tolist())

    tmpl_aln = AreTomo3ALN.from_file(str(template_aln)) if template_aln else None

    # --- global fit (Warp file order) -------------------------------------
    gfit = fit_aretomo_globals(template.model, pix, grid_shape=(5, 5, 3))
    if gfit.rms_px_heldout > global_tol_px:
        raise RuntimeError(
            f"global-only fit residual {gfit.rms_px_heldout:.2f} px exceeds "
            f"{global_tol_px} px - the Warp global geometry is not adequately "
            "representable by .aln globals"
        )

    # --- .aln row order: used tilts, ascending fitted TILT -----------------
    used = [i for i in range(ts.n_tilts) if bool(ts.use_tilt[i])]
    rows = sorted(used, key=lambda i: float(gfit.tilt_deg[i]))  # aln row -> warp idx

    model_global = AretomoTsModel(
        rot_deg=gfit.rot_deg[rows],
        tilt_deg=gfit.tilt_deg[rows],
        shifts_px=gfit.shifts_px[rows],
        raw_size_px=(int(img_px[0]), int(img_px[1])),
        pixel_size_a=pix,
        volume_dims_a=vol_a,
        local=None,
    )

    # --- IR from the FULL Warp model + local fit ---------------------------
    ir = build_ir_tilt_series(
        template.model,
        volume_dims_a=template.model.volume_dims_a,
        image_dims_a=template.model.image_dims_a,
        meta=_w2a_ir_meta(template, pix),
        grid_shape=grid_shape,
    )
    lfit = fit_aretomo_locals(
        ir, model_global, rows, patch_grid=patch_grid, patch_z=patch_z
    )

    # --- emit .aln (shared assembly, Phase B) -------------------------------
    from cets_nonrigid.io.aln import assemble_aln

    darks = []
    if tmpl_aln is not None and tmpl_aln.DarkFrames:
        darks = list(tmpl_aln.DarkFrames)
    else:
        order = sorted(range(ts.n_tilts), key=lambda i: float(gfit.tilt_deg[i]))
        for i in range(ts.n_tilts):
            if not bool(ts.use_tilt[i]):
                darks.append(
                    DarkFrameInfo(
                        section_idx=order.index(i), val2=i, angle=float(gfit.tilt_deg[i])
                    )
                )

    aln_out = assemble_aln(
        model_global=model_global,
        local=lfit.model.local,
        sec_1b=[r + 1 for r in rows],  # 1-based raw-stack section (Warp file order)
        raw_size=(int(img_px[0]), int(img_px[1]), int(ts.n_tilts)),
        dark_frames=darks,
        alpha_offset=float(-ts.level_angle_y),
        beta_offset=float(tmpl_aln.BetaOffset) if tmpl_aln is not None else 0.0,
        thickness=round(float(template.model.volume_dims_a[2]) / pix),
    )
    from cets_nonrigid.io.aln import write_aln

    aln_check = write_aln(
        out_aln, aln_out,
        source_angles_deg=[float(-(ts.angles[i] + ts.level_angle_y)) for i in rows],
        expect_rows=len(rows), max_patch_shift_px=max_patch_shift_px,
    )

    ctf_path: Path | None = None
    if write_ctf:
        from cets_nonrigid.ctf import tiltctf_from_warp, tiltctf_to_aretomo

        src_ctf = tiltctf_from_warp(ts)
        if src_ctf is not None:
            ctf_path = out_aln.parent / (out_aln.stem + "_CTF.txt")
            tiltctf_to_aretomo(src_ctf, ts.angles).to_file(ctf_path)

    store = None
    if store_path is not None:
        per_proj = torch.full((ts.n_tilts,), float("nan"), dtype=torch.float64)
        for row, warp_i in enumerate(rows):
            if lfit.heldout_status == "evaluated":
                per_proj[warp_i] = float(lfit.per_tilt_rms_px_heldout[row])
        store = DeformationStore.write(
            store_path,
            ir,
            native_files={"source_xml": template.xml_bytes},
            target_files={"out_aln": out_aln.read_bytes()},
            fit_attrs={
                "direction": "w2a",
                "global_rms_px_train": gfit.rms_px_train,
                "heldout_status": lfit.heldout_status,
                "global_rms_px_heldout": gfit.rms_px_heldout,
                "rms_px_train": lfit.rms_px_train,
                "rms_px_heldout": lfit.rms_px_heldout,
                "p95_px_heldout": lfit.p95_px_heldout,
                "max_px_heldout": lfit.max_px_heldout,
                "coverage_heldout": lfit.coverage_heldout,
                "z_stratified_rms_px": lfit.z_stratified_rms_px,
                "pixel_size_a": pix,
                **lfit.meta,
            },
            fit_arrays=(
                {"per_projection_rms_px_heldout": per_proj}
                if lfit.heldout_status == "evaluated"
                else {}
            ),
        )

    return W2AResult(
        out_aln=out_aln,
        global_fit=gfit,
        local_fit=lfit,
        pixel_size_a=pix,
        store=store,
        ctf_path=ctf_path,
        aln=aln_out,
        raw_pre_exposure=ts.dose.detach().to(torch.float64).clone(),
        aln_check=aln_check,
    )


# ---------------------------------------------------------------------------
# Frame series: m2w and w2m
# ---------------------------------------------------------------------------


@dataclass
class M2WResult:
    out_xml: Path
    fit: WarpMovieFitResult
    store: Path | None
    template_source: str = "template"
    defaulted_fields: list | None = None


def _check_movie_path_stem(movie_path, out_xml) -> None:
    """Warp derives the XML path from the raw-movie path: warn when the output
    stem does not match the declared movie."""
    if movie_path is None:
        return
    expected = Path(movie_path).stem
    if Path(out_xml).stem != expected:
        import warnings

        warnings.warn(
            f"output XML stem {Path(out_xml).stem!r} does not match --movie-path stem "
            f"{expected!r} - Warp will not associate it with the movie",
            stacklevel=3,
        )


@dataclass
class W2MResult:
    out_mcaln: Path
    fit: McAlnFitResult
    store: Path | None


def _frame_ir_meta(name: str, pix: float, size_px, f_count: int, tool: str) -> IRMeta:
    return IRMeta(
        kind="frame_series",
        series_name=name,
        pixel_size_image_a=pix,
        image_dims_px=(int(size_px[0]), int(size_px[1])),
        projection_index=list(range(f_count)),
        projection_valid=[True] * f_count,
        projection_order=list(range(f_count)),
        projection_dose=[0.0] * f_count,
        source_tool=tool,
    )


def mcaln_to_warp_movie(
    mcaln_path: str | Path,
    template_xml: str | Path | None,
    out_xml: str | Path,
    *,
    grid_shape: tuple[int, int] = (11, 11),
    local_grid: tuple[int, int, int] = (3, 3, 4),
    store_path: str | Path | None = None,
    movie_path: str | None = None,
) -> M2WResult:
    """m2w: AreTomo3 .mcaln -> Warp movie XML.

    With a template, the current metadata-overlay behavior; without one a
    movie model is synthesized (geometry and frame count come from the
    .mcaln). A generated movie XML is an alignment record only: Warp movie
    XML stores no image dims/frame count/FractionFrames, so it is usable only
    alongside the matching raw movie (``movie_path`` declares its expected
    basename). The written grids are for FractionFrames = 1 (all aligned
    frames loaded).
    """
    from warpylib.movie import Movie
    from warpylib.movie.io import load_meta

    from cets_nonrigid.fit.movie_fits import fit_warp_movie
    from cets_nonrigid.io.motion_txt import McAln
    from cets_nonrigid.io.warp_xml import write_movie_alignment_into_template
    from cets_nonrigid.ir.build import build_ir_frame_series

    mcaln = McAln.from_file(mcaln_path)
    model = mcaln.to_model()
    pix = mcaln.alignment_pixel_size_a
    size_px = mcaln.alignment_image_size_px
    f_count = mcaln.aligned_frame_count
    image_dims_a = (size_px[0] * pix, size_px[1] * pix)

    ir = build_ir_frame_series(
        model,
        torch.tensor(image_dims_a),
        meta=_frame_ir_meta(Path(mcaln_path).stem, pix, size_px, f_count, "aretomo3-motion"),
        grid_shape=grid_shape,
    )

    synth_report = None
    if template_xml is not None:
        _reject_synthesis_options(movie_path=movie_path)
        template_bytes = Path(template_xml).read_bytes()
        template_movie = Movie()
        load_meta(template_movie, str(template_xml))
    else:
        from cets_nonrigid.io.warp_synth import movie_template_bytes, synthesize_movie

        template_movie, synth_report = synthesize_movie(pixel_size_a=pix, data_path=movie_path)
        template_bytes = movie_template_bytes(template_movie, "m2w")
        synth_report.warn_once("m2w")
        _check_movie_path_stem(movie_path, out_xml)

    fit = fit_warp_movie(
        ir,
        template_movie,
        n_frames=f_count,
        image_dims_a=image_dims_a,
        fraction_frames=1.0,
        local_grid=local_grid,
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

    store = None
    if store_path is not None:
        native_files = {"mcaln": Path(mcaln_path).read_bytes()}
        if synth_report is None:
            native_files["template_xml"] = template_bytes
        store = DeformationStore.write(
            store_path,
            ir,
            native_files=native_files,
            target_files={"out_xml": Path(out_xml).read_bytes()},
            fit_attrs={
                "direction": "m2w",
                "rms_a_train": fit.rms_a_train,
                "heldout_status": fit.heldout_status,
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
        )
    return M2WResult(
        out_xml=Path(out_xml),
        fit=fit,
        store=store,
        template_source="template" if synth_report is None else "generated",
        defaulted_fields=(
            None if synth_report is None else list(synth_report.defaulted_experimental)
        ),
    )


def mcaln_from_motion_model(
    m,  # AretomoMotionModel
    *,
    n_frames: int,
    image_size_px,
    pixel_size_a: float,
    patch_grid,
    fm_ref: int,
    raw_frames_per_aligned: int = 1,
):
    """Assemble an McAln from a fitted AretomoMotionModel (shared by w2m and
    rm2m)."""
    from cets_nonrigid.io.motion_txt import McAln, McAlnFrame

    p_count = m.patch_centers_px.shape[0]
    return McAln(
        raw_frame_count=n_frames * raw_frames_per_aligned,
        integrated_frame_count=n_frames,
        aligned_frame_count=n_frames,
        alignment_image_size_px=tuple(image_size_px),
        alignment_pixel_size_a=pixel_size_a,
        patches=tuple(patch_grid),
        fm_ref=fm_ref,
        frames=[
            McAlnFrame(
                integrated_index=i,
                source_start=i * raw_frames_per_aligned,
                source_count=raw_frames_per_aligned,
                included=True,
                aligned_index=i,
            )
            for i in range(n_frames)
        ],
        global_shifts=[
            (f, float(m.global_shifts_px[f, 0]), float(m.global_shifts_px[f, 1]))
            for f in range(n_frames)
        ],
        local_shifts=[
            (
                p,
                [
                    (
                        f,
                        float(m.patch_centers_px[p, 0]),
                        float(m.patch_centers_px[p, 1]),
                        float(m.patch_shifts_px[f, p, 0]),
                        float(m.patch_shifts_px[f, p, 1]),
                        bool(m.patch_valid[f, p]),
                    )
                    for f in range(n_frames)
                ],
            )
            for p in range(p_count)
        ],
    )


def warp_movie_to_mcaln(
    xml_path: str | Path,
    out_mcaln: str | Path,
    *,
    image_size_px: tuple[int, int],
    pixel_size_a: float,
    n_frames: int,
    fraction_frames: float = 1.0,
    raw_frames_per_aligned: int = 1,
    patch_grid: tuple[int, int] = (5, 5),
    fm_ref: int = -1,
    grid_shape: tuple[int, int] = (11, 11),
    store_path: str | Path | None = None,
) -> W2MResult:
    """w2m: Warp movie XML -> .mcaln (the runtime metadata the XML does not
    carry — frame count, physical dims, FractionFrames — must be supplied)."""
    from cets_nonrigid.fit.movie_fits import fit_mcaln_shifts
    from cets_nonrigid.models.warp_movie import WarpMovieModel

    out_mcaln = Path(out_mcaln)
    if out_mcaln.exists():
        raise FileExistsError(f"{out_mcaln} already exists")

    image_dims_a = (image_size_px[0] * pixel_size_a, image_size_px[1] * pixel_size_a)
    model = WarpMovieModel.from_xml(
        xml_path,
        n_frames=n_frames,
        image_dims_a=image_dims_a,
        fraction_frames=fraction_frames,
    )
    ir = build_ir_frame_series(
        model,
        torch.tensor(image_dims_a),
        meta=_frame_ir_meta(Path(xml_path).stem, pixel_size_a, image_size_px, n_frames, "warp-movie"),
        grid_shape=grid_shape,
    )
    fit = fit_mcaln_shifts(
        ir,
        frame_size_px=image_size_px,
        pixel_size_a=pixel_size_a,
        patch_grid=patch_grid,
        fm_ref=fm_ref,
    )

    mcaln = mcaln_from_motion_model(
        fit.model, n_frames=n_frames, image_size_px=image_size_px,
        pixel_size_a=pixel_size_a, patch_grid=patch_grid,
        fm_ref=fit.meta["fm_ref"], raw_frames_per_aligned=raw_frames_per_aligned,
    )
    mcaln.to_file(out_mcaln)

    store = None
    if store_path is not None:
        store = DeformationStore.write(
            store_path,
            ir,
            native_files={"source_xml": Path(xml_path).read_bytes()},
            target_files={"out_mcaln": out_mcaln.read_bytes()},
            fit_attrs={
                "direction": "w2m",
                "rms_a_train": fit.rms_a_train,
                "heldout_status": fit.heldout_status,
                "rms_a_heldout": fit.rms_a_heldout,
                "p95_a_heldout": fit.p95_a_heldout,
                "coverage_heldout": fit.coverage_heldout,
                "pixel_size_a": pixel_size_a,
                **fit.meta,
            },
        )
    return W2MResult(out_mcaln=out_mcaln, fit=fit, store=store)


def _w2a_ir_meta(template: WarpSeries, pix: float) -> IRMeta:
    ts = template.ts
    t = ts.n_tilts
    img_px = (ts.image_dimensions_physical / pix).round().to(torch.int64)
    vol_px = (ts.volume_dimensions_physical / pix).round().to(torch.int64)
    return IRMeta(
        kind="tilt_series",
        series_name=Path(ts.path).stem if ts.path else "",
        pixel_size_image_a=pix,
        image_dims_px=(int(img_px[0]), int(img_px[1])),
        volume_dims_px=(int(vol_px[0]), int(vol_px[1]), int(vol_px[2])),
        pixel_size_volume_a=pix,
        projection_index=list(range(t)),
        projection_valid=[bool(u) for u in ts.use_tilt],
        projection_order=list(range(t)),
        projection_dose=[float(d) for d in ts.dose],
        projection_angle_deg=[float(a) for a in ts.angles],
        projection_sec=[i + 1 for i in range(t)],
        projection_dark=[not bool(u) for u in ts.use_tilt],
        source_tool="warp",
        projection_label=row_labels_from_paths(getattr(ts, "tilt_movie_paths", None)),
    )
