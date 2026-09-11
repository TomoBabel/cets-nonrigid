"""To-RELION tilt-series pipelines.

w2r: Warp XML + picked particles -> RELION bundle (tomograms.star +
tilt_series/<name>.star + particles.star + motion.star + optimisation set)
whose extraction yields "polished" particles embodying the Warp local
alignments. The RELION global is the exact closed-form mapping (runtime
verified); every per-particle local residual is lifted EXACTLY into a 3D
trajectory (min-norm, gauge at the lowest-dose row). Output contains every
input particle in the original order — nothing is withheld.

Dark tilts (UseTilt=false) are DROPPED from the emitted rows by default
(matching RELION's own AreTomo importer); ``keep_darks`` switches to the
whole-stack representation for alignment-only bundles (not validated for
extraction).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from cets_nonrigid.conventions import RELION_TOMO_HAND_FOR_WARP_NOT_INVERTED
from cets_nonrigid.ctf import TiltCtf, tiltctf_from_warp
from cets_nonrigid.fit.relion_global import (
    GLOBAL_EXACT_RMS_PX,
    RelionGlobalResult,
    integer_dims_px,
    relion_global_from_aretomo,
    relion_global_from_warp,
)
from cets_nonrigid.fit.relion_traj import LIFT_EXACT_MAX_PX, TrajectoryLiftResult, lift_particle_trajectories
from cets_nonrigid.io.relion_star import (
    RelionTomogramData,
    write_motion_star,
    write_optimisation_set,
    write_particles_star,
    write_tomograms_star,
)
from cets_nonrigid.io.store import DeformationStore
from cets_nonrigid.io.warp_xml import load_warp_tiltseries
from cets_nonrigid.ir.build import build_ir_tilt_series_from_points
from cets_nonrigid.ir.core import IRMeta

_F64 = torch.float64


@dataclass
class W2RResult:
    out_dir: Path
    tomograms_star: Path
    particles_star: Path | None
    motion_star: Path | None
    optimisation_set: Path
    global_result: RelionGlobalResult
    lift: TrajectoryLiftResult | None
    rows: list  # emitted XML tilt indices
    hand: int
    n_particles: int
    store: Path | None
    tomo: object | None = None  # RelionTomogramData as emitted
    entry: object | None = None  # project.relion.RelionSeriesEntry when written into a project


def _slice_tilt_stack(stack_path: Path, rows: list, out_dir: Path, pixel_size_a: float) -> list:
    """Slice a tilt stack (XML order) into per-tilt MRCs for the emitted rows."""
    import mrcfile
    import numpy as np

    tilt_dir = out_dir / "tilts"
    tilt_dir.mkdir(parents=True, exist_ok=True)
    names = []
    with mrcfile.open(stack_path, permissive=True) as m:
        data = np.asarray(m.data)
    if data.ndim == 2:
        data = data[None]
    if data.shape[0] <= max(rows):
        raise ValueError(
            f"{stack_path}: {data.shape[0]} slices but emitted rows reference XML index {max(rows)}"
        )
    for r in rows:
        path = tilt_dir / f"{stack_path.stem}_{r:03d}.mrc"
        if path.exists():
            raise FileExistsError(f"{path} already exists")
        with mrcfile.new(path) as out:
            out.set_data(data[r])
            out.voxel_size = pixel_size_a
        names.append(str(path))
    return names


def _write_relion_outputs(
    out_dir: Path,
    tomo: RelionTomogramData,
    *,
    particle_names: list | None,
    centered_coords_a: torch.Tensor | None,
    motion_a: torch.Tensor | None,
    project=None,
    tilt_series_uri: str | None = None,
    overwrite: bool = False,
):
    """Standalone bundle (absolute paths, refuses overwrite) or a series added
    to a RelionProject (root-relative paths, appended). Returns
    (tomograms_star, particles_star, motion_star, optimisation_set, entry,
    target_bytes)."""
    if project is not None:
        entry = project.add_series(
            tomo, particle_names=particle_names, centered_coords_a=centered_coords_a,
            motion_a=motion_a, tilt_series_uri=tilt_series_uri, overwrite=overwrite,
        )
        outs = entry.outputs(project.root)
        target = {"tilt_series_star": entry.tilt_star_bytes}
        if entry.particles is not None:
            target["particles_star"] = project.star_bytes({"particles": entry.particles})
        if entry.motion:
            target["motion_star"] = project.star_bytes(
                {"general": {"rlnParticleNumber": len(entry.motion)}, **entry.motion}
            )
        return (
            Path(outs["tomograms_star"]),
            Path(outs["particles_star"]) if outs["particles_star"] else None,
            Path(outs["motion_star"]) if outs["motion_star"] else None,
            Path(outs["optimisation_set"]), entry, target,
        )
    tomograms_star, ts_path = write_tomograms_star(out_dir, tomo)
    particles_star = motion_star = None
    if particle_names:
        particles_star = write_particles_star(
            out_dir / "particles.star",
            tomo_name=tomo.name, particle_names=particle_names, centered_coords_a=centered_coords_a,
            voltage_kv=tomo.voltage_kv, cs_mm=tomo.cs_mm, amplitude_contrast=tomo.amplitude_contrast,
            pixel_size_a=tomo.pixel_size_a,
        )
        if motion_a is not None:
            motion_star = write_motion_star(out_dir / "motion.star", particle_names, motion_a)
    optimisation_set = write_optimisation_set(
        out_dir / "optimisation_set.star",
        particles=str(particles_star) if particles_star else None,
        tomograms=str(tomograms_star),
        trajectories=str(motion_star) if motion_star else None,
    )
    target = {"tomograms_star": tomograms_star.read_bytes(), "tilt_series_star": ts_path.read_bytes()}
    if particles_star:
        target["particles_star"] = particles_star.read_bytes()
    if motion_star:
        target["motion_star"] = motion_star.read_bytes()
    return tomograms_star, particles_star, motion_star, optimisation_set, None, target


def _read_tilt_image_list(list_path: Path, n_rows: int) -> list:
    """Explicit per-emitted-row image list (one path per line, row order) —
    identity by explicit listing, never by glob order."""
    names = [ln.strip() for ln in Path(list_path).read_text().splitlines() if ln.strip()]
    if len(names) != n_rows:
        raise ValueError(f"{list_path}: {len(names)} image paths for {n_rows} emitted rows")
    missing = [n for n in names if not Path(n).exists()]
    if missing:
        raise ValueError(f"{list_path}: missing image files: {missing[:3]}...")
    return names


def warp_to_relion(
    xml_path: str | Path,
    out_dir: str | Path,
    *,
    pixel_size_a: float,
    tomo_name: str,
    positions_eff_a: torch.Tensor | None = None,  # (P, 3) canonical A (io.particles loaders)
    particle_names: list | None = None,
    tilt_stack: str | Path | None = None,
    tilt_image_list: str | Path | None = None,
    keep_darks: bool = False,
    hand: int | None = None,
    global_tol_px: float = GLOBAL_EXACT_RMS_PX,
    no_ctf: bool = False,
    store_path: str | Path | None = None,
    micrograph_names: list | None = None,  # explicit per emitted row (e.g. "N@stack.mrc")
    no_particles: bool = False,
    project=None,  # project.relion.RelionProject: append instead of a standalone bundle
    tilt_series_uri: str | None = None,
    overwrite: bool = False,
    trajectory_gauge: str = "lowest-dose",  # or "ctf-optimal" (fit/relion_traj.py)
) -> W2RResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if no_particles:
        if store_path is not None:
            raise ValueError("a deformation store needs particles (sample points); drop --store with --no-particles")
        positions_eff_a, particle_names = None, []
    elif positions_eff_a is None or particle_names is None:
        raise ValueError("particles are required unless no_particles is set")

    template = load_warp_tiltseries(xml_path)
    ts = template.ts
    pix = float(pixel_size_a)
    tomo_dims_px = integer_dims_px(ts.volume_dimensions_physical, pix, "VolumeDimensionsAngstrom")
    image_dims_px = integer_dims_px(ts.image_dimensions_physical, pix, "ImageDimensionsAngstrom")

    use = ts.use_tilt.to(torch.bool)
    if keep_darks:
        rows = list(range(ts.n_tilts))
    else:
        rows = [i for i in range(ts.n_tilts) if bool(use[i])]
    if not rows:
        raise ValueError("no active tilts to emit")

    # --- global: exact closed form, runtime verified -----------------------
    gres = relion_global_from_warp(
        ts, template.model, rows=rows, pixel_size_a=pix, tolerance_px=global_tol_px
    )

    # --- hand (G4-pinned; CTF-only, prominently reported by the CLI) -------
    inverted = bool(getattr(ts, "are_angles_inverted", False))
    derived_hand = RELION_TOMO_HAND_FOR_WARP_NOT_INVERTED * (-1 if inverted else 1)
    hand = int(hand) if hand is not None else int(derived_hand)

    # --- exact per-particle trajectory lift --------------------------------
    dose_rows = ts.dose.to(_F64)[torch.tensor(rows, dtype=torch.long)]
    lift = None
    if not no_particles:
        row_idx = torch.tensor(rows, dtype=torch.long)
        src_all, _src_valid = template.model.project_volume(positions_eff_a)
        # source 3D displacement at the particles (Warp GridVolumeWarp); None when
        # it is identically zero -> the classic min-norm lift
        disp_all = template.model.displace_volume(positions_eff_a)
        src_disp = disp_all[row_idx].to(_F64) if bool((disp_all != 0).any()) else None
        # the source's own CTF depth (position + displacement, Warp convention) and
        # the EMITTED RELION model with the hand actually written (derived or override)
        src_depth = template.model.ctf_depth(positions_eff_a)[row_idx].to(_F64)
        emitted_model = gres.model.with_ctf_convention(hand=hand)
        lift = lift_particle_trajectories(
            src_all.to(_F64)[row_idx],
            emitted_model,
            positions_eff_a.to(_F64),
            dose_rows,
            source_disp_a=src_disp,
            source_ctf_depth_a=src_depth,
            gauge=trajectory_gauge,
        )
        if lift.max_residual_px > LIFT_EXACT_MAX_PX:
            raise RuntimeError(
                f"trajectory lift residual {lift.max_residual_px:.2e} px exceeds "
                f"{LIFT_EXACT_MAX_PX} px — wiring error, conversion aborted"
            )

    # --- CTF ---------------------------------------------------------------
    idx = torch.tensor(rows, dtype=torch.long)
    src_ctf = tiltctf_from_warp(ts)
    if src_ctf is None and not no_ctf:
        raise ValueError(
            f"{xml_path} carries no per-tilt CTF grids; a normal bundle requires CTF "
            "(rlnDefocusU/V/Angle are mandatory in RELION) — pass no_ctf/--no-ctf for a "
            "geometry-only bundle that must be extracted with relion_tomo_subtomo --no_ctf"
        )
    if no_ctf:
        t = len(rows)
        ctf = TiltCtf(
            defocus_u_a=torch.full((t,), 20000.0, dtype=_F64),
            defocus_v_a=torch.full((t,), 20000.0, dtype=_F64),
            angle_deg=torch.zeros(t, dtype=_F64),
            phase_deg=torch.zeros(t, dtype=_F64),
        )
    else:
        ctf = TiltCtf(
            defocus_u_a=src_ctf.defocus_u_a[idx],
            defocus_v_a=src_ctf.defocus_v_a[idx],
            angle_deg=src_ctf.angle_deg[idx],
            phase_deg=src_ctf.phase_deg[idx],
            voltage_kv=src_ctf.voltage_kv,
            cs_mm=src_ctf.cs_mm,
            amplitude_contrast=src_ctf.amplitude_contrast,
        )

    # --- tilt images -------------------------------------------------------
    if micrograph_names is not None:
        if len(micrograph_names) != len(rows):
            raise ValueError(f"{len(micrograph_names)} micrograph names for {len(rows)} emitted rows")
    elif tilt_image_list is not None:
        micrograph_names = _read_tilt_image_list(Path(tilt_image_list), len(rows))
    elif tilt_stack is not None:
        micrograph_names = _slice_tilt_stack(Path(tilt_stack), rows, out_dir, pix)
    else:
        raise ValueError("need tilt_stack, tilt_image_list or micrograph_names (extraction requires real images)")

    # --- write the bundle --------------------------------------------------
    voltage = ctf.voltage_kv if ctf.voltage_kv is not None else float(ts.ctf.voltage)
    cs = ctf.cs_mm if ctf.cs_mm is not None else float(ts.ctf.cs)
    amp = ctf.amplitude_contrast if ctf.amplitude_contrast is not None else float(ts.ctf.amplitude)
    tomo = RelionTomogramData(
        name=tomo_name,
        voltage_kv=voltage,
        cs_mm=cs,
        amplitude_contrast=amp,
        hand=hand,
        pixel_size_a=pix,
        tomo_dims_px=tomo_dims_px,
        image_dims_px=image_dims_px,
        xtilt_deg=gres.xtilt_deg,
        ytilt_deg=gres.ytilt_deg,
        zrot_deg=gres.zrot_deg,
        xshift_a=gres.xshift_a,
        yshift_a=gres.yshift_a,
        pre_exposure=dose_rows,
        nominal_stage_angle_deg=ts.angles.to(_F64)[idx],
        ctf=ctf,
        micrograph_names=micrograph_names,
    )
    centre_a = torch.tensor([d / 2.0 for d in tomo_dims_px], dtype=_F64) * pix
    tomograms_star, particles_star, motion_star, optimisation_set, entry, target_bytes = _write_relion_outputs(
        out_dir, tomo,
        particle_names=particle_names if not no_particles else None,
        centered_coords_a=(lift.positions_out_a - centre_a) if lift is not None else None,
        motion_a=lift.motion_a if lift is not None else None,
        project=project, tilt_series_uri=tilt_series_uri, overwrite=overwrite,
    )

    # --- provenance store (optional) ---------------------------------------
    store = None
    if store_path is not None:
        meta = IRMeta(
            kind="tilt_series",
            series_name=tomo_name,
            pixel_size_image_a=pix,
            image_dims_px=image_dims_px,
            volume_dims_px=tomo_dims_px,
            pixel_size_volume_a=pix,
            projection_index=rows,
            projection_valid=[True] * len(rows),
            projection_order=list(range(len(rows))),
            projection_dose=[float(d) for d in dose_rows],
            projection_angle_deg=[float(ts.angles[r]) for r in rows],
            projection_sec=[-1] * len(rows),
            projection_dark=[False] * len(rows),
            source_tool="warp",
            sampling="particles",
        )

        class _RowsModel:
            n_projections = len(rows)

            @staticmethod
            def project_volume(points):
                xy, valid = template.model.project_volume(points)
                return xy[idx], valid[idx]

            @staticmethod
            def project_volume_global(points):
                xy, valid = template.model.project_volume_global(points)
                return xy[idx], valid[idx]

            # schema 0.4 intermediates, row-sliced like the projections (the
            # builders would otherwise record them as unavailable)
            @staticmethod
            def displace_volume(points):
                return template.model.displace_volume(points)[idx]

            @staticmethod
            def ctf_depth(points, displacement=None):
                return template.model.ctf_depth(points, displacement)[idx]

        ir = build_ir_tilt_series_from_points(
            _RowsModel, positions_eff_a.to(_F64), ts.image_dimensions_physical,
            meta=meta, point_names=particle_names, heldout_fraction=0.0,
        )
        store = DeformationStore.write(
            store_path,
            ir,
            native_files={"template_xml": template.xml_bytes},
            target_files=target_bytes,
            fit_attrs={
                "direction": "w2r",
                "heldout_status": "not_evaluated",  # exact conversion, no fit
                "global_rms_px": gres.rms_px,
                "global_exact": gres.global_exact,
                "global_used_fallback": gres.used_fallback,
                "lift_max_residual_px": lift.max_residual_px,
                "lift_ref_row": lift.ref_row,
                "hand": hand,
                "pixel_size_a": pix,
                # rev. 4: depth provenance and the RELION-side CTF-depth deviation
                # (over the emitted rows, NOT held-out). RELION files cannot carry these.
                "depth_source": lift.depth_source,
                "trajectory_gauge": lift.gauge,
                "gauge_rank": lift.gauge_rank,
                "n_gauge_fallback": lift.n_gauge_fallback,
                "ctf_depth_deviation_rms_a": lift.ctf_depth_deviation_rms_a,
                "ctf_depth_deviation_max_a": lift.ctf_depth_deviation_max_a,
            },
            fit_arrays={
                "trajectories_a": lift.motion_a.to(torch.float32),
                **(
                    {"ctf_depth_deviation_a": lift.ctf_depth_deviation_a.to(torch.float32)}
                    if lift.ctf_depth_deviation_a is not None
                    else {}
                ),
            },
            fit_array_dims={
                "trajectories_a": ["projection", "point", "xyz"],
                "ctf_depth_deviation_a": ["projection", "point"],
            },
        )

    return W2RResult(
        out_dir=out_dir,
        tomograms_star=tomograms_star,
        particles_star=particles_star,
        motion_star=motion_star,
        optimisation_set=optimisation_set,
        global_result=gres,
        lift=lift,
        rows=rows,
        hand=hand,
        n_particles=int(positions_eff_a.shape[0]) if positions_eff_a is not None else 0,
        store=Path(store) if store else None,
        tomo=tomo,
        entry=entry,
    )


# ---------------------------------------------------------------------------
# a2r
# ---------------------------------------------------------------------------


@dataclass
class A2RResult:
    out_dir: Path
    tomograms_star: Path
    particles_star: Path | None
    motion_star: Path | None
    optimisation_set: Path
    global_result: RelionGlobalResult
    lift: TrajectoryLiftResult | None
    sec_1b: list  # per emitted row: 1-based raw-sorted section
    hand: int
    n_particles: int
    alpha_offset_deg: float  # constant TILT-vs-raw-tilt offset (AlphaOffset)
    store: Path | None
    tomo: object | None = None
    entry: object | None = None


def _raw_tilts_from_aln(aln) -> torch.Tensor:
    from cets_nonrigid.io.aln import raw_tilts_from_aln

    return raw_tilts_from_aln(aln)


def aretomo_to_relion(
    aln_path: str | Path,
    out_dir: str | Path,
    *,
    pixel_size_a: float,
    tomo_name: str,
    tomo_dims_px: tuple,
    voltage_kv: float,
    cs_mm: float,
    amplitude_contrast: float,
    hand: int,  # MANDATORY: .aln carries no handedness
    positions_eff_a: torch.Tensor | None = None,
    particle_names: list | None = None,
    raw_dose=None,  # io.dose.RawDose over the raw-sorted rows (darks included)
    ctf_file: str | Path | None = None,
    no_ctf: bool = False,
    tilt_stack: str | Path | None = None,  # raw-sorted order incl. darks
    tilt_image_list: str | Path | None = None,
    global_tol_px: float = GLOBAL_EXACT_RMS_PX,
    store_path: str | Path | None = None,
    micrograph_names: list | None = None,  # explicit per emitted row (e.g. "N@stack.mrc")
    no_particles: bool = False,
    project=None,  # project.relion.RelionProject: append instead of a standalone bundle
    tilt_series_uri: str | None = None,
    overwrite: bool = False,
) -> A2RResult:
    from cets_nonrigid.io.aln import load_aln

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if raw_dose is None:
        raise ValueError("raw_dose (acquisition order + dose) is required")
    if no_particles:
        if store_path is not None:
            raise ValueError("a deformation store needs particles (sample points); drop --store with --no-particles")
        positions_eff_a, particle_names = None, []
    elif positions_eff_a is None or particle_names is None:
        raise ValueError("particles are required unless no_particles is set")
    pix = float(pixel_size_a)
    vol_a = tuple(float(d) * pix for d in tomo_dims_px)

    aln_series = load_aln(aln_path, pixel_size_a=pix, volume_dims_a=vol_a)
    model = aln_series.model  # rows = .aln rows (dark-removed, ascending TILT)
    sec_1b = [int(g.sec) for g in aln_series.aln.GlobalAlignments]
    raw_tilts = _raw_tilts_from_aln(aln_series.aln)
    r_count = raw_tilts.shape[0]
    if raw_dose.n_raw != r_count:
        raise ValueError(
            f"dose source covers {raw_dose.n_raw} raw tilts but the .aln raw stack has {r_count}"
        )

    # join verification: .aln TILT includes AlphaOffset — the difference to the
    # raw acquisition tilt must be CONSTANT across rows (checked fallback
    # identity; a varying difference means the join is wrong)
    sec_idx = torch.tensor(sec_1b, dtype=torch.long) - 1
    offsets = model.tilt_deg.to(_F64) - raw_dose.tilt_deg[sec_idx]
    alpha_offset = float(offsets.mean())
    if float((offsets - alpha_offset).abs().max()) > 0.5:
        raise ValueError(
            ".aln rows do not join the dose source: TILT minus raw tilt is not a "
            f"constant offset (spread {float((offsets - alpha_offset).abs().max()):.2f} deg)"
        )

    gres = relion_global_from_aretomo(model, model, tolerance_px=global_tol_px)

    dose_rows = raw_dose.pre_exposure[sec_idx]
    lift = None
    if not no_particles:
        src_all, _valid = model.project_volume(positions_eff_a)
        lift = lift_particle_trajectories(
            src_all.to(_F64), gres.model, positions_eff_a.to(_F64), dose_rows
        )
        if lift.max_residual_px > LIFT_EXACT_MAX_PX:
            raise RuntimeError(
                f"trajectory lift residual {lift.max_residual_px:.2e} px exceeds "
                f"{LIFT_EXACT_MAX_PX} px — wiring error, conversion aborted"
            )

    # --- CTF: _CTF.txt rows are raw-sorted ordinals; .aln SEC selects them ---
    t_aln = model.n_projections
    if ctf_file is not None:
        from cets_nonrigid.io.ctf_aretomo import AreTomoCtfFile

        f = AreTomoCtfFile.from_file(ctf_file)
        if f.n_rows != r_count:
            raise ValueError(
                f"{ctf_file}: {f.n_rows} rows but the raw stack has {r_count} tilts"
            )
        rows_sel = [f.rows[i] for i in sec_idx.tolist()]
        ctf = TiltCtf(
            defocus_u_a=torch.tensor([x.df_max_a for x in rows_sel], dtype=_F64),
            defocus_v_a=torch.tensor([x.df_min_a for x in rows_sel], dtype=_F64),
            angle_deg=torch.tensor([x.azimuth_deg for x in rows_sel], dtype=_F64),
            phase_deg=torch.rad2deg(torch.tensor([x.phase_rad for x in rows_sel], dtype=_F64)),
            score=torch.tensor([x.score for x in rows_sel], dtype=_F64),
            res_a=torch.tensor([x.res_a for x in rows_sel], dtype=_F64),
            voltage_kv=voltage_kv, cs_mm=cs_mm, amplitude_contrast=amplitude_contrast,
        )
    elif no_ctf:
        ctf = TiltCtf(
            defocus_u_a=torch.full((t_aln,), 20000.0, dtype=_F64),
            defocus_v_a=torch.full((t_aln,), 20000.0, dtype=_F64),
            angle_deg=torch.zeros(t_aln, dtype=_F64),
            phase_deg=torch.zeros(t_aln, dtype=_F64),
        )
    else:
        raise ValueError(
            "a normal bundle requires --ctf TS_CTF.txt (rlnDefocusU/V/Angle are "
            "mandatory in RELION); pass --no-ctf for a geometry-only bundle that "
            "must be extracted with relion_tomo_subtomo --no_ctf"
        )

    if micrograph_names is not None:
        if len(micrograph_names) != t_aln:
            raise ValueError(f"{len(micrograph_names)} micrograph names for {t_aln} emitted rows")
    elif tilt_image_list is not None:
        micrograph_names = _read_tilt_image_list(Path(tilt_image_list), t_aln)
    elif tilt_stack is not None:
        micrograph_names = _slice_tilt_stack(Path(tilt_stack), sec_idx.tolist(), out_dir, pix)
    else:
        raise ValueError("need tilt_stack, tilt_image_list or micrograph_names (extraction requires real images)")

    tomo = RelionTomogramData(
        name=tomo_name,
        voltage_kv=voltage_kv,
        cs_mm=cs_mm,
        amplitude_contrast=amplitude_contrast,
        hand=int(hand),
        pixel_size_a=pix,
        tomo_dims_px=tuple(int(d) for d in tomo_dims_px),
        image_dims_px=(int(model.raw_size_px[0]), int(model.raw_size_px[1])),
        xtilt_deg=gres.xtilt_deg,
        ytilt_deg=gres.ytilt_deg,
        zrot_deg=gres.zrot_deg,
        xshift_a=gres.xshift_a,
        yshift_a=gres.yshift_a,
        pre_exposure=dose_rows,
        nominal_stage_angle_deg=raw_dose.tilt_deg[sec_idx],
        ctf=ctf,
        micrograph_names=micrograph_names,
    )
    centre_a = torch.tensor([float(d) / 2.0 for d in tomo_dims_px], dtype=_F64) * pix
    tomograms_star, particles_star, motion_star, optimisation_set, entry, target_bytes = _write_relion_outputs(
        out_dir, tomo,
        particle_names=particle_names if not no_particles else None,
        centered_coords_a=(lift.positions_out_a - centre_a) if lift is not None else None,
        motion_a=lift.motion_a if lift is not None else None,
        project=project, tilt_series_uri=tilt_series_uri, overwrite=overwrite,
    )

    store = None
    if store_path is not None:
        meta = IRMeta(
            kind="tilt_series",
            series_name=tomo_name,
            pixel_size_image_a=pix,
            image_dims_px=(int(model.raw_size_px[0]), int(model.raw_size_px[1])),
            volume_dims_px=tuple(int(d) for d in tomo_dims_px),
            pixel_size_volume_a=pix,
            projection_index=list(range(t_aln)),
            projection_valid=[True] * t_aln,
            projection_order=list(range(t_aln)),
            projection_dose=[float(d) for d in dose_rows],
            projection_angle_deg=[float(a) for a in raw_dose.tilt_deg[sec_idx]],
            projection_angle_kind=["effective"] * len(sec_idx),
            projection_sec=sec_1b,
            projection_dark=[False] * t_aln,
            source_tool="aretomo3",
            sampling="particles",
        )
        ir = build_ir_tilt_series_from_points(
            model, positions_eff_a.to(_F64),
            model.raw_size_px.to(_F64) * pix,
            meta=meta, point_names=particle_names, heldout_fraction=0.0,
        )
        store = DeformationStore.write(
            store_path,
            ir,
            native_files={"aln": aln_series.aln_bytes},
            target_files=target_bytes,
            fit_attrs={
                "direction": "a2r",
                "heldout_status": "not_evaluated",
                "global_rms_px": gres.rms_px,
                "global_exact": gres.global_exact,
                "global_used_fallback": gres.used_fallback,
                "lift_max_residual_px": lift.max_residual_px,
                "lift_ref_row": lift.ref_row,
                "alpha_offset_deg": alpha_offset,
                "hand": int(hand),
                "pixel_size_a": pix,
            },
            fit_arrays={"trajectories_a": lift.motion_a.to(torch.float32)},
            fit_array_dims={"trajectories_a": ["projection", "point", "xyz"]},
        )

    return A2RResult(
        out_dir=out_dir,
        tomograms_star=tomograms_star,
        particles_star=particles_star,
        motion_star=motion_star,
        optimisation_set=optimisation_set,
        global_result=gres,
        lift=lift,
        sec_1b=sec_1b,
        hand=int(hand),
        n_particles=int(positions_eff_a.shape[0]) if positions_eff_a is not None else 0,
        alpha_offset_deg=alpha_offset,
        store=Path(store) if store else None,
        tomo=tomo,
        entry=entry,
    )


# ---------------------------------------------------------------------------
# from-RELION: shared readers
# ---------------------------------------------------------------------------


def relion_model_from_data(data, image_dims_px: tuple):
    """RelionTomogramModel honoring RELION's precedence: rlnTomoProj* matrices
    are authoritative when present; valid-but-disagreeing Euler columns only
    warn (malformed matrices are rejected by from_matrices)."""
    import warnings

    from cets_nonrigid.models.relion_ts import RelionTomogramModel

    # CTF-depth convention travels with the geometry (both construction paths):
    # rlnTomoHand is mandatory, rlnTomoDefocusSlope defaults to 1.0 when absent
    # (tomogram_set.cpp:499-503).
    slope = data.defocus_slope if getattr(data, "defocus_slope", None) is not None else 1.0

    if data.matrices is not None:
        model = RelionTomogramModel.from_matrices(
            data.matrices,
            pixel_size_a=data.pixel_size_a,
            tomo_dims_px=tuple(data.tomo_dims_px),
            image_dims_px=tuple(image_dims_px),
            hand=data.hand,
            defocus_slope=slope,
        )
        try:
            euler = RelionTomogramModel(
                xtilt_deg=data.xtilt_deg, ytilt_deg=data.ytilt_deg, zrot_deg=data.zrot_deg,
                xshift_a=data.xshift_a, yshift_a=data.yshift_a,
                tomo_dims_px=tuple(data.tomo_dims_px), image_dims_px=tuple(image_dims_px),
                pixel_size_a=data.pixel_size_a, hand=data.hand, defocus_slope=slope,
            )
            diff = (euler.projection_matrices - model.projection_matrices).abs().max()
            if float(diff) > 1e-3:
                warnings.warn(
                    f"tomogram {data.name}: rlnTomoProj* matrices disagree with the "
                    f"Euler/shift columns (max |dP| = {float(diff):.3g}); the matrices "
                    "are authoritative (RELION precedence)",
                    stacklevel=2,
                )
        except Exception:  # noqa: BLE001, S110 - Euler columns may be absent/degenerate
            pass
        return model
    return RelionTomogramModel(
        xtilt_deg=data.xtilt_deg, ytilt_deg=data.ytilt_deg, zrot_deg=data.zrot_deg,
        xshift_a=data.xshift_a, yshift_a=data.yshift_a,
        tomo_dims_px=tuple(data.tomo_dims_px), image_dims_px=tuple(image_dims_px),
        pixel_size_a=data.pixel_size_a, hand=data.hand, defocus_slope=slope,
    )


@dataclass
class RelionSourceOverrides:
    """Fix-ups applied to a RELION source tomogram at load time (replaces the
    star-rewriting scripts): bin-1 box override (+ even pad per side in X/Y),
    nominal stage angles (from a file or copied from rlnTomoYTilt), and
    per-row micrograph names (what r2w copies into the Warp MoviePath)."""

    tomo_dims_px: tuple | None = None
    pad_px: int = 0
    nominal_angles: list | None = None
    nominal_from_ytilt: bool = False
    micrograph_names: list | None = None

    def is_empty(self) -> bool:
        return (self.tomo_dims_px is None and not self.pad_px and self.nominal_angles is None
                and not self.nominal_from_ytilt and self.micrograph_names is None)


def apply_source_overrides(data: RelionTomogramData, ov: RelionSourceOverrides) -> tuple:
    """-> (data', notes). Gates: pad even (odd dims move the FLOAT centre by
    half a pixel), nominal count == rows and ascending when the rows are."""
    import dataclasses

    notes: list[str] = []
    changes: dict = {}
    if ov.tomo_dims_px is not None or ov.pad_px:
        dims = tuple(int(v) for v in (ov.tomo_dims_px or data.tomo_dims_px))
        if len(dims) != 3:
            raise ValueError("tomo_dims_px override must be XxYxZ")
        if ov.pad_px:
            if ov.pad_px % 2:
                raise ValueError("--pad must be even (odd dims move the float centre by half a pixel)")
            dims = (dims[0] + 2 * ov.pad_px, dims[1] + 2 * ov.pad_px, dims[2])
        changes["tomo_dims_px"] = dims
        notes.append(f"rlnTomoSize {tuple(data.tomo_dims_px)} -> {dims}")
    if ov.nominal_angles is not None:
        if len(ov.nominal_angles) != data.n_tilts:
            raise ValueError(f"{len(ov.nominal_angles)} nominal angles for {data.n_tilts} tilt rows")
        nom = torch.tensor([float(v) for v in ov.nominal_angles], dtype=_F64)
        yt = data.ytilt_deg.to(_F64)
        if bool((yt[1:] >= yt[:-1]).all()) and not bool((nom[1:] >= nom[:-1]).all()):
            raise ValueError("nominal angles are not ascending while rlnTomoYTilt is — row order mismatch")
        dev = yt - nom
        notes.append(f"nominal angles from file: ytilt-nominal mean {float(dev.mean()):+.3f} deg, "
                     f"std {float(dev.std()):.3f} deg")
        if float(dev.std()) > 0.5:
            raise ValueError(f"nominal angles deviate from rlnTomoYTilt with std {float(dev.std()):.2f} deg (> 0.5)")
        changes["nominal_stage_angle_deg"] = nom
    elif ov.nominal_from_ytilt:
        changes["nominal_stage_angle_deg"] = data.ytilt_deg.to(_F64).clone()
        notes.append("rlnTomoNominalStageTiltAngle := rlnTomoYTilt")
    if ov.micrograph_names is not None:
        if len(ov.micrograph_names) != data.n_tilts:
            raise ValueError(f"{len(ov.micrograph_names)} micrograph names for {data.n_tilts} tilt rows")
        changes["micrograph_names"] = list(ov.micrograph_names)
        notes.append("rlnMicrographName rewritten")
    return (dataclasses.replace(data, **changes) if changes else data), notes


def list_relion_tomograms(
    *,
    optimisation_set: str | Path | None = None,
    tomograms_star: str | Path | None = None,
    project_root: str | Path | None = None,
) -> list[str]:
    """rlnTomoName values of a project (for the all-tomograms default)."""
    from cets_nonrigid.io.relion_star import read_optimisation_set, resolve_star_ref

    if tomograms_star is None and optimisation_set is not None:
        refs = read_optimisation_set(optimisation_set)
        tomograms_star = resolve_star_ref(refs["tomograms"], optimisation_set, project_root)
    if tomograms_star is None:
        raise ValueError("need an optimisation set or a tomograms star")
    import pandas as pd
    import starfile

    g = starfile.read(Path(tomograms_star), always_dict=True)["global"]
    if isinstance(g, pd.Series):
        g = g.to_frame().T
    return [str(v) for v in g["rlnTomoName"]]


def load_relion_source(
    *,
    optimisation_set: str | Path | None = None,
    tomograms_star: str | Path | None = None,
    particles_star: str | Path | None = None,
    motion_star: str | Path | None = None,
    tomo_name: str,
    image_dims_px: tuple,
    overrides: RelionSourceOverrides | None = None,
    project_root: str | Path | None = None,
):
    """-> (data, model, positions_eff_a, names, trajectories_a|None, deformation).

    Enforces: pixel-size consistency (v1), unique names, and — read-side —
    rejects deformation-carrying tomograms until the deformation ports land.
    ``overrides`` are applied to the selected tomogram before the model is
    built; ``project_root`` resolves root-relative star references.
    """
    from cets_nonrigid.io.relion_star import (
        read_motion_star,
        read_optimisation_set,
        read_particles_star,
        read_tomograms_star,
        resolve_star_ref,
    )
    from cets_nonrigid.models.relion_ts import effective_positions_a

    if optimisation_set is not None:
        refs = read_optimisation_set(optimisation_set)

        def _resolve(v):
            return None if v is None else resolve_star_ref(v, optimisation_set, project_root)

        tomograms_star = tomograms_star or _resolve(refs["tomograms"])
        particles_star = particles_star or _resolve(refs["particles"])
        motion_star = motion_star or _resolve(refs["trajectories"])
    if tomograms_star is None or particles_star is None:
        raise ValueError("need an optimisation set or explicit tomograms + particles star paths")

    tomos = read_tomograms_star(tomograms_star, project_root)
    if tomo_name not in tomos:
        raise ValueError(f"tomogram {tomo_name!r} not in {tomograms_star} (has {list(tomos)})")
    data = tomos[tomo_name]
    if overrides is not None and not overrides.is_empty():
        data, _notes = apply_source_overrides(data, overrides)
    deformation = None
    if data.has_deformations and int(image_dims_px[0]) > 0:  # probe calls pass (0, 0)
        # read-side evaluation (user decision): the 2D deformations are part
        # of RELION's local alignment and are folded into the IR
        if data.deformation_coeffs is None:
            raise ValueError(
                f"tomogram {tomo_name!r} declares rlnTomoDeformation* but carries no "
                "per-tilt rlnTomoDeformationCoefficients"
            )
        from cets_nonrigid.models.relion_deform import deformation_field

        deformation = deformation_field(
            data.deformation_type, data.deformation_grid, tuple(image_dims_px),
            data.deformation_coeffs,
        )
    model = relion_model_from_data(data, image_dims_px)

    parts = read_particles_star(particles_star)
    sel = [i for i, t in enumerate(parts.tomo_names or [tomo_name] * len(parts.particle_names))
           if t in ("", tomo_name)]
    if not sel:
        raise ValueError(f"no particles of tomogram {tomo_name!r} in {particles_star}")
    data.point_attributes = {key: [values[i] for i in sel] for key, values in parts.point_attributes.items()}
    names = [parts.particle_names[i] for i in sel]
    if len(set(names)) != len(names):
        raise ValueError("duplicate rlnTomoParticleName values")

    for grp, pix in parts.optics_pixel_size_a.items():
        if abs(pix - data.pixel_size_a) > 1e-3 * data.pixel_size_a:
            raise ValueError(
                f"optics group {grp} pixel size {pix} != tomogram pixel size "
                f"{data.pixel_size_a} (v1 requires equality)"
            )

    def sub(t):
        return t[torch.tensor(sel, dtype=torch.long)] if t is not None else None

    positions = effective_positions_a(
        pixel_size_a=data.pixel_size_a,
        tomo_dims_px=tuple(data.tomo_dims_px),
        centered_coords_a=sub(parts.centered_coords_a),
        legacy_coords_px=sub(parts.legacy_coords_px),
        origins_a=sub(parts.origins_a),
        subtomo_angles_deg=sub(parts.subtomo_angles_deg),
    )

    trajectories = None
    if motion_star is not None and Path(motion_star).exists():
        trajectories = read_motion_star(motion_star, names, data.n_tilts)
    return data, model, positions, names, trajectories, deformation


def match_relion_rows_to_template(
    template_ts,
    nominal_deg: torch.Tensor,  # (T_relion,) RELION nominal stage angles
    *,
    tol_deg: float = 0.5,
    deactivate_unmatched: bool = False,
) -> tuple:
    """-> (row_map (T_relion,) template indices, template active mask after
    policy). Identity by nominal stage angle as the CHECKED fallback (v1: star
    bundles carry no template image basenames); bijective, ambiguity fails.
    Any unmatched RELION source row is a hard error; unmatched ACTIVE template
    tilts fail unless deactivate_unmatched."""
    warp_angles = template_ts.angles.to(_F64)
    used = torch.zeros(warp_angles.shape[0], dtype=torch.bool)
    row_map = torch.empty(nominal_deg.shape[0], dtype=torch.long)
    for i, ang in enumerate(nominal_deg.to(_F64)):
        diff = (warp_angles - ang).abs() + used.to(_F64) * 1e6
        j = int(torch.argmin(diff))
        if float(diff[j]) > tol_deg:
            raise ValueError(
                f"RELION source row {i} (nominal {float(ang):.2f} deg) matches no template "
                f"tilt within {tol_deg} deg — source data is never silently dropped"
            )
        second = torch.topk(-diff, 2).values
        if float(-second[1]) - float(diff[j]) < 0.05:
            raise ValueError(f"ambiguous nominal-angle match for RELION row {i}")
        used[j] = True
        row_map[i] = j
    active = template_ts.use_tilt.to(torch.bool).clone()
    unmatched_active = active & ~used
    if unmatched_active.any():
        if not deactivate_unmatched:
            raise ValueError(
                f"template tilts {torch.nonzero(unmatched_active).flatten().tolist()} are "
                "active (UseTilt=True) but matched by no RELION row — pass "
                "--deactivate-unmatched to disable them (zero movement) instead"
            )
        active = active & used
    return row_map, active


@dataclass
class R2WResult:
    out_xml: Path
    fit: object  # WarpTsFitResult
    row_map: list  # RELION row -> template tilt index
    level_angle_x_deg: float
    n_particles: int
    store: Path | None
    template_source: str = "template"
    defaulted_fields: list | None = None


def _synthesize_r2w_template(data, image_dims_px):
    """Template-free r2w: synthesize the Warp 'template' from the star alone —
    RELION row order, all rows active, nominal stage angles (Angle = -nominal),
    geometry/dose/paths/handedness/optics from the star."""
    from cets_nonrigid.io.warp_synth import synthesize_tilt_series, synthesized_template_series

    t = data.nominal_stage_angle_deg.shape[0]
    pix = data.pixel_size_a
    paths = None
    if data.micrograph_names and all(str(p) for p in data.micrograph_names):
        paths = [str(p) for p in data.micrograph_names]
    inverted = int(data.hand) != int(RELION_TOMO_HAND_FOR_WARP_NOT_INVERTED)
    # cets_nonrigid bundles pin rlnTomoNominalStageTiltAngle := Warp Angle verbatim
    # (w2r writes it from ts.angles; match_relion_rows_to_template compares 1:1)
    ts, report = synthesize_tilt_series(
        angles_deg=data.nominal_stage_angle_deg,
        use_tilt=torch.ones(t, dtype=torch.bool),
        axis_angles_deg=torch.zeros(t),
        axis_offset_x_a=torch.zeros(t),
        axis_offset_y_a=torch.zeros(t),
        image_dims_a=tuple(float(d) * pix for d in image_dims_px),
        volume_dims_a=tuple(float(d) * pix for d in data.tomo_dims_px),
        pixel_size_a=pix,
        dose=data.pre_exposure,
        movie_paths=paths,
        angles_inverted=inverted,
        angles_inverted_known=True,
        voltage_kv=data.voltage_kv,
        cs_mm=data.cs_mm,
        amplitude_contrast=data.amplitude_contrast,
    )
    return synthesized_template_series(ts, "r2w"), report


def relion_to_warp(
    template_xml: str | Path | None,
    out_xml: str | Path,
    *,
    tomo_name: str,
    optimisation_set: str | Path | None = None,
    tomograms_star: str | Path | None = None,
    particles_star: str | Path | None = None,
    motion_star: str | Path | None = None,
    movement_grid: tuple = (5, 5),
    lam: float = 1e-3,
    deactivate_unmatched: bool = False,
    heldout_fraction: float = 0.2,
    heldout_seed: int = 20260828,
    write_ctf: bool = True,
    max_condition: float | None = None,
    min_node_support: float | None = None,
    warn_node_support: float | None = None,
    store_path: str | Path | None = None,
    image_size_px: tuple | None = None,
    overrides: RelionSourceOverrides | None = None,
    project_root: str | Path | None = None,
    volume_warp_grid=None,  # (W, H, D, L | None=T): opt-in GridVolumeWarp fit to the trajectories
) -> R2WResult:
    """RELION global + trajectories -> fitted Warp XML (movement grids, and
    optionally GridVolumeWarp fitted to the per-particle 3D trajectories).

    With ``template_xml`` the metadata-overlay behavior is preserved; without
    one, a Warp model is synthesized from the star (``image_size_px`` is then
    required — the star never carries the tilt-image dimensions).

    The Warp global is the exact closed-form inverse of the w2r mapping
    (angle = -ytilt, axis = zrot, LevelAngleX = mean(xtilt), AxisOffset =
    shift - centre-delta corrections) — computed from the star values alone,
    never fitted on particles, so the particle train/held-out split stays
    independent for the movement-grid fit.
    """
    import copy as _copy

    from warpylib import CubicGrid, LinearGrid4D

    from cets_nonrigid.ctf import tiltctf_to_warp
    from cets_nonrigid.fit.relion_global import _shift_corrections
    from cets_nonrigid.fit.warp_ts_fit import (
        fit_warp_locals,
        resolve_volume_warp_grid,
        volume_warp_fit_arrays,
        volume_warp_fit_attrs,
    )
    from cets_nonrigid.io.warp_xml import write_alignment_into_template
    from cets_nonrigid.models.relion_ts import RelionParticleSetModel

    synth_report = None
    if template_xml is not None:
        from cets_nonrigid.convert import _reject_synthesis_options

        _reject_synthesis_options(image_size_px=image_size_px)
        template = load_warp_tiltseries(template_xml)
    else:
        if image_size_px is None:
            raise ValueError(
                "template-free r2w requires image_size_px "
                "(the star never carries the tilt-image dimensions)"
            )
        template = None

    image_dims_px_probe = (0, 0)  # real dims validated once the star's pixel size is known
    data, model, positions, names, trajectories, _deformation_probe = load_relion_source(
        optimisation_set=optimisation_set, tomograms_star=tomograms_star,
        particles_star=particles_star, motion_star=motion_star,
        tomo_name=tomo_name, image_dims_px=image_dims_px_probe,
        overrides=overrides, project_root=project_root,
    )
    pix = data.pixel_size_a
    if template is not None:
        ts = template.ts
        image_dims_px = integer_dims_px(ts.image_dimensions_physical, pix, "ImageDimensionsAngstrom")
    else:
        image_dims_px = (int(image_size_px[0]), int(image_size_px[1]))
        template, synth_report = _synthesize_r2w_template(data, image_dims_px)
        ts = template.ts
    # rebuild the model + deformation with the real image dims
    model = relion_model_from_data(data, image_dims_px)
    deformation = None
    if data.has_deformations:
        from cets_nonrigid.models.relion_deform import deformation_field

        deformation = deformation_field(
            data.deformation_type, data.deformation_grid, tuple(image_dims_px),
            data.deformation_coeffs,
        )

    row_map, active = match_relion_rows_to_template(
        ts, data.nominal_stage_angle_deg, deactivate_unmatched=deactivate_unmatched
    )

    # --- exact closed-form RELION -> Warp global ---------------------------
    xt = data.xtilt_deg.to(_F64)
    level_x = float(xt.mean())
    if float((xt - level_x).abs().max()) > 1e-6:
        import warnings

        warnings.warn(
            f"per-tilt rlnTomoXTilt varies (spread {float((xt - level_x).abs().max()):.3g} deg); "
            "Warp LevelAngleX is a constant — using the mean, remainder absorbed by the "
            "movement grids and visible in the held-out residuals",
            stacklevel=2,
        )
    corr = _shift_corrections(
        model.rotations, tuple(data.tomo_dims_px), image_dims_px, pix
    )  # (T, 2) A — the same delta terms w2r ADDS
    axis_offset_x = data.xshift_a.to(_F64) - corr[:, 0]
    axis_offset_y = data.yshift_a.to(_F64) - corr[:, 1]

    ts_target = _copy.deepcopy(ts)
    ts_target.level_angle_x = level_x
    ts_target.level_angle_y = 0.0
    angles = ts_target.angles.clone()
    axis = ts_target.tilt_axis_angles.clone()
    offx = ts_target.tilt_axis_offset_x.clone()
    offy = ts_target.tilt_axis_offset_y.clone()
    for i in range(row_map.shape[0]):
        w = int(row_map[i])
        angles[w] = float(-data.ytilt_deg[i])
        axis[w] = float(data.zrot_deg[i])
        offx[w] = float(axis_offset_x[i])
        offy[w] = float(axis_offset_y[i])
    ts_target.angles = angles
    ts_target.tilt_axis_angles = axis
    ts_target.tilt_axis_offset_x = offx
    ts_target.tilt_axis_offset_y = offy
    ts_target.use_tilt = active
    ts_target.grid_movement_x = CubicGrid((1, 1, 1))
    ts_target.grid_movement_y = CubicGrid((1, 1, 1))
    ts_target.grid_volume_warp_x = LinearGrid4D((1, 1, 1, 1))
    ts_target.grid_volume_warp_y = LinearGrid4D((1, 1, 1, 1))
    ts_target.grid_volume_warp_z = LinearGrid4D((1, 1, 1, 1))

    # --- IR from the particle-bound source, in template row order ----------
    # Phase B: sample in star row order, then permute the arrays with the
    # shared align_ir_rows (array-equivalent to the retired _RowMappedModel:
    # unmapped template rows invalid, array validity NOT masked by the
    # template's use flags - that distinction lives in the meta, as before).
    source = RelionParticleSetModel(
        model, positions, trajectories_a=trajectories, deformation=deformation
    )
    tomo_dims = tuple(data.tomo_dims_px)
    meta = IRMeta(
        kind="tilt_series",
        series_name=tomo_name,
        pixel_size_image_a=pix,
        image_dims_px=image_dims_px,
        volume_dims_px=tomo_dims,
        pixel_size_volume_a=pix,
        projection_index=list(range(ts.n_tilts)),
        projection_valid=[bool(a) for a in active],
        projection_order=list(range(ts.n_tilts)),
        projection_dose=[float(d) for d in ts.dose],
        projection_angle_deg=[float(a) for a in ts.angles],
        projection_sec=[-1] * ts.n_tilts,
        projection_dark=[not bool(a) for a in active],
        source_tool="relion5",
        sampling="particles",
    )
    t_star = int(data.nominal_stage_angle_deg.shape[0])
    ir_src = build_ir_tilt_series_from_points(
        source, positions.to(_F64), ts.image_dimensions_physical,
        meta=meta.model_copy(update={
            "projection_index": list(range(t_star)),
            "projection_valid": [True] * t_star,
            "projection_order": list(range(t_star)),
            "projection_dose": [float(d) for d in data.pre_exposure],
            "projection_angle_deg": [float(a) for a in data.nominal_stage_angle_deg],
            "projection_sec": [-1] * t_star,
            "projection_dark": [False] * t_star,
        }),
        point_names=names,
        heldout_fraction=heldout_fraction, heldout_seed=heldout_seed,
    )
    from cets_nonrigid.ir.rows import RowMatch as _RowMatch
    from cets_nonrigid.ir.rows import TargetRowTable as _TargetRowTable
    from cets_nonrigid.ir.rows import align_ir_rows as _align_ir_rows

    template_rows = _TargetRowTable(
        angle_deg=list(meta.projection_angle_deg),
        angle_kind=["unknown"] * ts.n_tilts,
        active=[True] * ts.n_tilts,  # array validity comes from the mapping alone
        dark=list(meta.projection_dark),
        sec=list(meta.projection_sec),
        dose=list(meta.projection_dose),
        labels=None,
        order=list(meta.projection_order),
    )
    ir = _align_ir_rows(
        ir_src,
        _RowMatch(
            row_map=[int(m) for m in row_map],
            target_active=[True] * ts.n_tilts,
            method="pipeline",
        ),
        template_rows,
    )
    # exact pipeline meta (behavior-preserving) — but the 0.4 availability flags
    # are the BUILDER's knowledge and must survive this replacement.
    ir.meta = meta.model_copy(
        update={"displacement_3d": ir.meta.displacement_3d, "source_ctf_depth": ir.meta.source_ctf_depth}
    )

    from cets_nonrigid.convert_store import _tilt_series_gates
    from cets_nonrigid.fit.coverage import (
        DEFAULT_MAX_CONDITION,
        DEFAULT_MIN_NODE_SUPPORT,
        DEFAULT_WARN_NODE_SUPPORT,
    )

    vw_grid = resolve_volume_warp_grid(volume_warp_grid, ir.n_projections)
    fit = fit_warp_locals(
        ir, ts_target, movement_grid=tuple(movement_grid), lam=lam, volume_warp_grid=vw_grid,
        max_condition=max_condition if max_condition is not None else DEFAULT_MAX_CONDITION,
        min_node_support=min_node_support if min_node_support is not None else DEFAULT_MIN_NODE_SUPPORT,
        warn_node_support=warn_node_support if warn_node_support is not None else DEFAULT_WARN_NODE_SUPPORT,
    )
    # scattered-fit gates (data-only rank/condition + node support) on the FITTED
    # model at the premovement positions — shared with fit --to warp
    gate = _tilt_series_gates(
        ir, fit, ts_target, movement_grid,
        max_condition=max_condition, min_node_support=min_node_support, warn_node_support=warn_node_support,
    )

    if data.ctf is not None and write_ctf:
        # star rows -> template rows (unmatched template rows keep zeros)
        u = torch.zeros(ts.n_tilts, dtype=_F64)
        v = torch.zeros(ts.n_tilts, dtype=_F64)
        ang = torch.zeros(ts.n_tilts, dtype=_F64)
        ph = torch.zeros(ts.n_tilts, dtype=_F64)
        for i in range(row_map.shape[0]):
            w = int(row_map[i])
            u[w] = data.ctf.defocus_u_a[i]
            v[w] = data.ctf.defocus_v_a[i]
            ang[w] = data.ctf.angle_deg[i]
            ph[w] = data.ctf.phase_deg[i]
        tiltctf_to_warp(fit.ts, TiltCtf(
            defocus_u_a=u, defocus_v_a=v, angle_deg=ang, phase_deg=ph,
            voltage_kv=data.voltage_kv, cs_mm=data.cs_mm,
            amplitude_contrast=data.amplitude_contrast,
        ))
    with_ctf = data.ctf is not None and write_ctf
    if synth_report is not None:
        if with_ctf and "placeholder per-tilt CTF" in synth_report.defaulted_experimental:
            synth_report.defaulted_experimental.remove("placeholder per-tilt CTF")
        synth_report.warn_once("r2w")
        from cets_nonrigid.io.warp_synth import atomic_write_validated

        atomic_write_validated(
            lambda tmp: write_alignment_into_template(
                template.xml_bytes, fit.ts, tmp, with_ctf=with_ctf
            ),
            out_xml,
            load_warp_tiltseries,
        )
    else:
        write_alignment_into_template(template.xml_bytes, fit.ts, out_xml, with_ctf=with_ctf)

    store = None
    if store_path is not None:
        per_proj = (
            {"per_projection_rms_heldout": fit.per_tilt_rms_a_heldout}
            if fit.heldout_status == "evaluated"
            else {}
        )
        vw_arrays, vw_dims = volume_warp_fit_arrays(fit.ts) if fit.volume_warp is not None else ({}, {})
        store = DeformationStore.write(
            store_path,
            ir,
            native_files={
                "tomograms_star": Path(tomograms_star or optimisation_set).read_bytes(),
                **({"template_xml": template.xml_bytes} if synth_report is None else {}),
            },
            target_files={"out_xml": Path(out_xml).read_bytes()},
            fit_attrs={
                "direction": "r2w",
                "heldout_status": fit.heldout_status,
                "rms_a_train": fit.rms_a_train,
                "rms_a_heldout": fit.rms_a_heldout,
                "p95_a_heldout": fit.p95_a_heldout,
                "max_a_heldout": fit.max_a_heldout,
                "coverage_heldout": fit.coverage_heldout,
                "min_rank": fit.min_rank,
                "max_condition": fit.max_condition,
                "min_data_rank": fit.min_data_rank,
                "max_data_condition": fit.max_data_condition,
                "min_node_support": gate.min_node_support,
                "level_angle_x_deg": level_x,
                "pixel_size_a": pix,
                "template_source": "template" if synth_report is None else "generated",
                "defaulted_fields": (
                    [] if synth_report is None else list(synth_report.defaulted_experimental)
                ),
                **fit.meta,
                **volume_warp_fit_attrs(fit.volume_warp),
            },
            fit_arrays={**per_proj, **vw_arrays},
            fit_array_dims=vw_dims,
        )

    return R2WResult(
        out_xml=Path(out_xml),
        fit=fit,
        row_map=[int(w) for w in row_map],
        level_angle_x_deg=level_x,
        n_particles=positions.shape[0],
        store=Path(store) if store else None,
        template_source="template" if synth_report is None else "generated",
        defaulted_fields=(
            None if synth_report is None else list(synth_report.defaulted_experimental)
        ),
    )


# ---------------------------------------------------------------------------
# r2a
# ---------------------------------------------------------------------------


@dataclass
class R2AResult:
    out_aln: Path
    global_fit: object  # GlobalFitResult (fitted over RELION rows)
    local_fit: object  # AretomoLocalFitResult
    ctf_path: Path | None
    n_particles: int
    store: Path | None
    aln: object | None = None
    raw_pre_exposure: torch.Tensor | None = None  # per .aln row (raw order)
    aln_check: object | None = None


def relion_to_aretomo(
    out_aln: str | Path,
    *,
    tomo_name: str,
    pixel_size_a: float | None = None,  # default: the star's tilt-series pixel size
    image_dims_px: tuple,  # tilt-image dims (px) — never stored in the star
    optimisation_set: str | Path | None = None,
    tomograms_star: str | Path | None = None,
    particles_star: str | Path | None = None,
    motion_star: str | Path | None = None,
    patch_grid: tuple = (5, 5),
    patch_z: str = "lsq",
    heldout_fraction: float = 0.2,
    heldout_seed: int = 20260828,
    write_ctf: bool = True,
    max_condition: float | None = None,
    min_node_support: float | None = None,
    warn_node_support: float | None = None,
    store_path: str | Path | None = None,
    max_patch_shift_px: float | None = 100.0,
    overrides: RelionSourceOverrides | None = None,
    project_root: str | Path | None = None,
) -> R2AResult:
    """RELION global + trajectories -> fitted AreTomo3 .aln (globals + IDW
    locals). No-template policy (v1): ALL selected RELION rows are emitted,
    sorted ascending by fitted TILT, with 1-based SEC = row order (matching
    genuine AreTomo3 output) and no DarkFrame records."""
    import warnings as _warnings

    from cets_nonrigid.fit.aretomo_global import fit_aretomo_globals_from_init
    from cets_nonrigid.fit.aretomo_ts_fit import fit_aretomo_locals
    from cets_nonrigid.fit.coverage import (
        DEFAULT_MAX_CONDITION,
        DEFAULT_MIN_NODE_SUPPORT,
        DEFAULT_WARN_NODE_SUPPORT,
        evaluate_gates,
        node_support,
        nodes_reachable_by_volume,
    )
    from cets_nonrigid.fit.relion_global import _shift_corrections
    from cets_nonrigid.models.aretomo_ts import AretomoTsModel
    from cets_nonrigid.models.relion_ts import RelionParticleSetModel

    out_aln = Path(out_aln)
    if out_aln.exists():
        raise FileExistsError(f"{out_aln} already exists")

    data, model, positions, names, trajectories, deformation = load_relion_source(
        optimisation_set=optimisation_set, tomograms_star=tomograms_star,
        particles_star=particles_star, motion_star=motion_star,
        tomo_name=tomo_name, image_dims_px=tuple(image_dims_px),
        overrides=overrides, project_root=project_root,
    )
    pix = float(pixel_size_a) if pixel_size_a is not None else data.pixel_size_a
    t_rel = data.n_tilts

    # --- globals: importer-inverse init, refined against the RELION global --
    corr = _shift_corrections(model.rotations, tuple(data.tomo_dims_px), tuple(image_dims_px), pix)
    rot0 = data.zrot_deg.to(_F64)
    tilt0 = data.ytilt_deg.to(_F64)
    shifts0 = torch.stack(
        [(data.xshift_a.to(_F64) - corr[:, 0]) / pix, (data.yshift_a.to(_F64) - corr[:, 1]) / pix],
        dim=-1,
    )
    gfit = fit_aretomo_globals_from_init(model, pix, rot0, tilt0, shifts0)

    # --- .aln row order: ascending fitted TILT ------------------------------
    order = sorted(range(t_rel), key=lambda i: float(gfit.tilt_deg[i]))
    model_global = AretomoTsModel(
        rot_deg=gfit.rot_deg[order],
        tilt_deg=gfit.tilt_deg[order],
        shifts_px=gfit.shifts_px[order],
        raw_size_px=(int(image_dims_px[0]), int(image_dims_px[1])),
        pixel_size_a=pix,
        volume_dims_a=tuple(float(d) * pix for d in data.tomo_dims_px),
        local=None,
    )

    # --- IR from the particle-bound RELION source, in .aln row order --------
    # Phase B: sample in star row order, then permute the arrays with the
    # shared align_ir_rows (array-equivalent to the retired _RowMappedModel).
    source = RelionParticleSetModel(
        model, positions, trajectories_a=trajectories, deformation=deformation
    )
    row_map_to_aln = [order.index(i) for i in range(t_rel)]
    meta = IRMeta(
        kind="tilt_series",
        series_name=tomo_name,
        pixel_size_image_a=pix,
        image_dims_px=tuple(int(d) for d in image_dims_px),
        volume_dims_px=tuple(int(d) for d in data.tomo_dims_px),
        pixel_size_volume_a=pix,
        projection_index=list(range(t_rel)),
        projection_valid=[True] * t_rel,
        projection_order=order,
        projection_dose=[float(data.pre_exposure[i]) for i in order],
        projection_angle_deg=[float(data.nominal_stage_angle_deg[i]) for i in order],
        projection_angle_kind=["nominal"] * t_rel,
        projection_sec=[i + 1 for i in range(t_rel)],
        projection_dark=[False] * t_rel,
        source_tool="relion5",
        sampling="particles",
    )
    img_a = torch.tensor(
        [image_dims_px[0] * pix, image_dims_px[1] * pix], dtype=_F64
    )
    ir_src = build_ir_tilt_series_from_points(
        source, positions.to(_F64), img_a,
        meta=meta.model_copy(update={
            "projection_order": list(range(t_rel)),
            "projection_dose": [float(d) for d in data.pre_exposure],
            "projection_angle_deg": [float(a) for a in data.nominal_stage_angle_deg],
        }),
        point_names=names,
        heldout_fraction=heldout_fraction, heldout_seed=heldout_seed,
    )
    from cets_nonrigid.ir.rows import RowMatch as _RowMatch
    from cets_nonrigid.ir.rows import TargetRowTable as _TargetRowTable
    from cets_nonrigid.ir.rows import align_ir_rows as _align_ir_rows

    aln_rows = _TargetRowTable(
        angle_deg=list(meta.projection_angle_deg),
        angle_kind=list(meta.projection_angle_kind),
        active=[True] * t_rel,
        dark=[False] * t_rel,
        sec=list(meta.projection_sec),
        dose=list(meta.projection_dose),
        labels=None,
        order=list(meta.projection_order),
    )
    ir = _align_ir_rows(
        ir_src,
        _RowMatch(row_map=row_map_to_aln, target_active=[True] * t_rel, method="pipeline"),
        aln_rows,
    )
    ir.meta = meta.model_copy(  # exact pipeline meta (behavior-preserving) + builder flags
        update={"displacement_3d": ir.meta.displacement_3d, "source_ctf_depth": ir.meta.source_ctf_depth}
    )

    lfit = fit_aretomo_locals(
        ir, model_global, list(range(t_rel)),
        patch_grid=tuple(patch_grid), patch_z=patch_z,
    )

    # --- gates over the patch grid ------------------------------------------
    q_glob, q_valid = model_global.project_volume_global(ir.points)
    active_w = ir.weights.to(_F64) * ir.projection_valid.to(_F64) * q_valid.to(_F64)
    gx, gy = tuple(patch_grid)
    nx = torch.linspace(0.5 / gx, 1 - 0.5 / gx, gx, dtype=_F64) * float(img_a[0])
    ny = torch.linspace(0.5 / gy, 1 - 0.5 / gy, gy, dtype=_F64) * float(img_a[1])
    nodes = torch.cartesian_prod(nx, ny)
    spacing = torch.tensor([float(img_a[0]) / gx, float(img_a[1]) / gy], dtype=_F64)
    vol_a = torch.tensor([float(d) * pix for d in data.tomo_dims_px], dtype=_F64)
    corners = torch.tensor(
        [[x, y, z] for x in (0.0, float(vol_a[0]))
         for y in (0.0, float(vol_a[1])) for z in (0.0, float(vol_a[2]))],
        dtype=_F64,
    )
    corners_xy, _ = model_global.project_volume_global(corners)
    reachable = nodes_reachable_by_volume(nodes, corners_xy.to(_F64), spacing)
    support = node_support(q_glob.to(_F64), active_w, nodes, spacing, node_active=reachable)
    # rank gate via the per-tilt deficit (params vary per tilt for the IDW fit)
    gate = evaluate_gates(
        support,
        lfit.min_data_rank,
        lfit.min_data_rank + lfit.max_rank_deficit,  # deficit > 0 -> failure
        lfit.max_data_condition,
        max_condition=max_condition if max_condition is not None else DEFAULT_MAX_CONDITION,
        min_node_support=(
            min_node_support if min_node_support is not None else DEFAULT_MIN_NODE_SUPPORT
        ),
        warn_node_support=(
            warn_node_support if warn_node_support is not None else DEFAULT_WARN_NODE_SUPPORT
        ),
    )
    for w in gate.warnings:
        _warnings.warn(w, stacklevel=2)
    if gate.failures:
        raise RuntimeError("scattered-fit gates failed: " + "; ".join(gate.failures))

    # --- assemble the .aln (no-template: dense 1-based SEC, no darks) -------
    from cets_nonrigid.io.aln import assemble_aln

    aln_out = assemble_aln(
        model_global=model_global,
        local=lfit.model.local,
        sec_1b=[t + 1 for t in range(t_rel)],
        raw_size=(int(image_dims_px[0]), int(image_dims_px[1]), t_rel),
        dark_frames=[],
        alpha_offset=0.0,
        beta_offset=0.0,
        thickness=int(data.tomo_dims_px[2]),
    )
    from cets_nonrigid.io.aln import write_aln

    aln_check = write_aln(
        out_aln, aln_out,
        source_angles_deg=[float(data.ytilt_deg[order[t]]) for t in range(t_rel)],
        expect_rows=t_rel, max_patch_shift_px=max_patch_shift_px,
    )
    raw_pre_exposure = data.pre_exposure.to(_F64)[torch.tensor(order, dtype=torch.long)].clone()

    ctf_path = None
    if write_ctf and data.ctf is not None:
        from cets_nonrigid.io.ctf_aretomo import AreTomoCtfFile, AreTomoCtfRow

        rows_file = []
        for t in range(t_rel):
            i = order[t]  # RELION row of aln row t
            rows_file.append(
                AreTomoCtfRow(
                    micrograph=t + 1,
                    df_max_a=float(data.ctf.defocus_u_a[i]),
                    df_min_a=float(data.ctf.defocus_v_a[i]),
                    azimuth_deg=float(data.ctf.angle_deg[i]),
                    phase_rad=float(torch.deg2rad(data.ctf.phase_deg[i])),
                    score=float(data.ctf.score[i]) if data.ctf.score is not None else 0.0,
                    res_a=float(data.ctf.res_a[i]) if data.ctf.res_a is not None else 999.99,
                    df_hand=1,
                )
            )
        ctf_path = out_aln.parent / (out_aln.stem + "_CTF.txt")
        AreTomoCtfFile(rows=rows_file).to_file(ctf_path)

    store = None
    if store_path is not None:
        per_proj = (
            {"per_projection_rms_px_heldout": lfit.per_tilt_rms_px_heldout}
            if lfit.heldout_status == "evaluated"
            else {}
        )
        store = DeformationStore.write(
            store_path,
            ir,
            target_files={"out_aln": out_aln.read_bytes()},
            fit_attrs={
                "direction": "r2a",
                "heldout_status": lfit.heldout_status,
                "global_rms_px_heldout": gfit.rms_px_heldout,
                "rms_px_train": lfit.rms_px_train,
                "rms_px_heldout": lfit.rms_px_heldout,
                "p95_px_heldout": lfit.p95_px_heldout,
                "max_px_heldout": lfit.max_px_heldout,
                "coverage_heldout": lfit.coverage_heldout,
                "z_stratified_rms_px": lfit.z_stratified_rms_px,
                "min_data_rank": lfit.min_data_rank,
                "max_data_condition": lfit.max_data_condition,
                "min_node_support": gate.min_node_support,
                "pixel_size_a": pix,
                **lfit.meta,
            },
            fit_arrays=per_proj,
        )

    return R2AResult(
        out_aln=out_aln,
        global_fit=gfit,
        local_fit=lfit,
        ctf_path=ctf_path,
        n_particles=positions.shape[0],
        store=Path(store) if store else None,
        aln=aln_out,
        raw_pre_exposure=raw_pre_exposure,
        aln_check=aln_check,
    )
