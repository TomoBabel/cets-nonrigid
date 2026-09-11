"""RELION 5 tomography star-file IO.

Layout written (relion 5.0.1 conventions, verified in src/jaz/tomography):

  tomograms.star                data_global, one row per tomogram
  tilt_series/<TomoName>.star   data_<TomoName>, one row per emitted tilt
  particles.star                data_general + data_optics + data_particles
  motion.star                   data_general (rlnParticleNumber — the label
                                verified at metadata_label.h:1238, NOT
                                rlnNrOfParticles) + data_<rlnTomoParticleName>
                                per particle
  optimisation_set.star         unnamed list block (OptimisationSet::write
                                emits a default-name table; read takes the
                                first block)

All files are written through the starfile package (a dict value yields the
list-style block, an empty-string key the unnamed ``data_`` block); the
``# version 50001`` comment RELION emits is prepended afterwards —
``readStar`` does not require it, but it keeps the files byte-comparable to
RELION's own output.

Angles are degrees, shifts/coordinates Angstrom. Per-tilt row order in the
per-tomogram table IS the frame index space used by trajectories and
rlnTomoVisibleFrames. rlnCenteredCoordinate*Angst are centre-relative
(FLOAT centre, tomogram_set.cpp:314). Reading honors RELION's projection
precedence: when all four rlnTomoProjX/Y/Z/W columns exist they are
authoritative over the Euler/shift columns (tomogram_set.cpp:358-391).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import starfile
import torch

from cets_nonrigid.ctf import TiltCtf, tiltctf_from_relion_columns, tiltctf_to_relion_columns

_F64 = torch.float64
VERSION_COMMENT = "# version 50001"


# ---------------------------------------------------------------------------
# Tomogram set
# ---------------------------------------------------------------------------


@dataclass
class RelionTomogramData:
    """One tomogram's global + per-tilt alignment metadata (emission order)."""

    name: str
    voltage_kv: float
    cs_mm: float
    amplitude_contrast: float
    hand: int
    pixel_size_a: float  # rlnTomoTiltSeriesPixelSize
    tomo_dims_px: tuple  # rlnTomoSizeX/Y/Z (bin-1)
    image_dims_px: tuple  # from the tilt images (never in the star)
    xtilt_deg: torch.Tensor  # (T,)
    ytilt_deg: torch.Tensor  # (T,)
    zrot_deg: torch.Tensor  # (T,)
    xshift_a: torch.Tensor  # (T,)
    yshift_a: torch.Tensor  # (T,)
    pre_exposure: torch.Tensor  # (T,) cumulative e/A^2, MANDATORY in RELION
    nominal_stage_angle_deg: torch.Tensor  # (T,)
    ctf: TiltCtf | None = None
    original_pixel_size_a: float | None = None  # rlnMicrographOriginalPixelSize
    optics_group_name: str = "opticsGroup1"
    #: rlnTomoDefocusSlope as read from the star (None = absent; RELION then uses
    #: 1.0, tomogram_set.cpp:499-503). Enters Tomogram::getCtf as a factor on the
    #: particle depth (tomogram.cpp:280). Emitted only when not None.
    defocus_slope: float | None = None
    fractional_dose: float | None = None  # rlnTomoImportFractionalDose
    micrograph_names: list | None = None  # per-tilt corrected images (default mode)
    tilt_series_name: str | None = None  # whole-stack mode (--keep-darks)
    frame_count: int | None = None
    matrices: torch.Tensor | None = field(default=None, repr=False)  # (T,4,4) read-side, authoritative
    #: rlnTomoDeformationGridSizeX/Y + rlnTomoDeformationType present (2D image
    #: deformations, applied after rigid projection). Read-side evaluation is
    #: planned (user decision); until the torch ports land, pipelines must
    #: REFUSE such inputs rather than silently ignoring the deformation.
    has_deformations: bool = False
    deformation_type: str | None = None
    deformation_grid: tuple | None = None  # (gx, gy)
    deformation_coeffs: list | None = None  # T coefficient vectors (torch f8)

    @property
    def n_tilts(self) -> int:
        return self.ytilt_deg.shape[0]


def tomogram_global_row(tomo: RelionTomogramData, ts_ref: str) -> dict:
    """One data_global row (dict of scalars) referencing the tilt star ``ts_ref``."""
    if tomo.ctf is None:
        raise ValueError(
            "a normal RELION bundle requires per-tilt CTF (rlnDefocusU/V/Angle are "
            "read with getValueSafely, tomogram_set.cpp:413-415); pass --no-ctf for "
            "a geometry-only bundle with placeholder values"
        )
    if not torch.isfinite(tomo.pre_exposure).all():
        raise ValueError("pre_exposure must be finite for every emitted tilt")
    g = {
        "rlnTomoName": tomo.name,
        "rlnVoltage": tomo.voltage_kv,
        "rlnSphericalAberration": tomo.cs_mm,
        "rlnAmplitudeContrast": tomo.amplitude_contrast,
        "rlnTomoHand": int(tomo.hand),
        "rlnOpticsGroupName": tomo.optics_group_name,
        "rlnTomoTiltSeriesPixelSize": tomo.pixel_size_a,
        "rlnMicrographOriginalPixelSize": (
            tomo.original_pixel_size_a if tomo.original_pixel_size_a is not None else tomo.pixel_size_a
        ),
        "rlnTomoImportFractionalDose": (
            tomo.fractional_dose
            if tomo.fractional_dose is not None
            else _derive_fractional_dose(tomo.pre_exposure)
        ),
        "rlnTomoSizeX": int(tomo.tomo_dims_px[0]),
        "rlnTomoSizeY": int(tomo.tomo_dims_px[1]),
        "rlnTomoSizeZ": int(tomo.tomo_dims_px[2]),
        "rlnTomoTiltSeriesStarFile": ts_ref,
    }
    if tomo.tilt_series_name is not None:
        g["rlnTomoTiltSeriesName"] = tomo.tilt_series_name
        g["rlnTomoFrameCount"] = int(tomo.frame_count if tomo.frame_count is not None else tomo.n_tilts)
    if tomo.defocus_slope is not None:
        g["rlnTomoDefocusSlope"] = float(tomo.defocus_slope)
    return g


def tilt_series_frame(tomo: RelionTomogramData) -> pd.DataFrame:
    """The per-tilt table (data_<TomoName>) in emission order."""
    t = tomo.n_tilts
    if tomo.ctf is None:
        raise ValueError("per-tilt CTF required (see tomogram_global_row)")
    rows = {
        "rlnTomoNominalStageTiltAngle": tomo.nominal_stage_angle_deg.numpy(),
        "rlnTomoXTilt": tomo.xtilt_deg.numpy(),
        "rlnTomoYTilt": tomo.ytilt_deg.numpy(),
        "rlnTomoZRot": tomo.zrot_deg.numpy(),
        "rlnTomoXShiftAngst": tomo.xshift_a.numpy(),
        "rlnTomoYShiftAngst": tomo.yshift_a.numpy(),
        **tiltctf_to_relion_columns(tomo.ctf),
        "rlnMicrographPreExposure": tomo.pre_exposure.numpy(),
    }
    if tomo.micrograph_names is not None:
        if len(tomo.micrograph_names) != t:
            raise ValueError("micrograph_names length mismatch")
        rows["rlnMicrographName"] = list(tomo.micrograph_names)
    elif tomo.tilt_series_name is None:
        raise ValueError("need micrograph_names (per-tilt images) or tilt_series_name (whole stack)")
    return pd.DataFrame(rows)


def write_tomograms_star(out_dir: str | Path, tomo: RelionTomogramData) -> tuple:
    """Write tomograms.star + tilt_series/<name>.star (standalone bundle,
    absolute tilt-star reference). Refuses overwrite."""
    out_dir = Path(out_dir)
    ts_dir = out_dir / "tilt_series"
    ts_dir.mkdir(parents=True, exist_ok=True)
    global_path = out_dir / "tomograms.star"
    ts_path = ts_dir / f"{tomo.name}.star"
    for p in (global_path, ts_path):
        if p.exists():
            raise FileExistsError(f"{p} already exists")
    g = tomogram_global_row(tomo, str(ts_path))
    rows = tilt_series_frame(tomo)
    starfile.write({"global": pd.DataFrame([g])}, global_path)
    starfile.write({tomo.name: rows}, ts_path)
    return global_path, ts_path


def _derive_fractional_dose(pre_exposure: torch.Tensor) -> float:
    su = torch.unique(pre_exposure.to(_F64)).sort().values
    return float(su[1] - su[0]) if su.numel() > 1 else 1.0


def _parse_vector(v) -> list:
    if isinstance(v, str):
        return [float(x) for x in v.strip("[]").split(",")]
    return [float(x) for x in v]


def read_tomograms_star(path: str | Path, project_root: str | Path | None = None) -> dict:
    """tomograms.star -> {name: RelionTomogramData}. Per-tomo star paths are
    resolved by ``resolve_star_ref`` (absolute, star-relative, project-root
    relative, CWD).
    When all four rlnTomoProjX/Y/Z/W columns exist, ``matrices`` is populated
    and is AUTHORITATIVE (RELION's precedence, tomogram_set.cpp:358-391)."""
    path = Path(path)
    glob_df = starfile.read(path, always_dict=True)
    if "global" not in glob_df:
        raise ValueError(f"{path}: no data_global block")
    g = glob_df["global"]
    if isinstance(g, pd.Series):
        g = g.to_frame().T
    out = {}
    for _, row in g.iterrows():
        name = str(row["rlnTomoName"])
        ts_file = resolve_star_ref(str(row["rlnTomoTiltSeriesStarFile"]), path, project_root)
        blocks = starfile.read(ts_file, always_dict=True)
        if name not in blocks:
            raise ValueError(
                f"{ts_file}: expected block data_{name} (found {list(blocks)}) — "
                "RELION errors on mismatched block names"
            )
        df = blocks[name]

        matrices = None
        proj_cols = ["rlnTomoProjX", "rlnTomoProjY", "rlnTomoProjZ", "rlnTomoProjW"]
        if all(c in df.columns for c in proj_cols):
            t = len(df)
            m = torch.empty(t, 4, 4, dtype=_F64)
            for i in range(t):
                for r, c in enumerate(proj_cols):
                    m[i, r, :] = torch.tensor(_parse_vector(df.iloc[i][c]), dtype=_F64)
            matrices = m

        def col(label, default=None, frame=df, src=ts_file):
            if label in frame.columns:
                return torch.tensor(np.asarray(frame[label], dtype=np.float64))
            if default is None:
                raise ValueError(f"{src}: missing mandatory column {label}")
            return torch.full((len(frame),), float(default), dtype=_F64)

        ctf = None
        if "rlnDefocusU" in df.columns:
            ctf = tiltctf_from_relion_columns(
                {k: np.asarray(df[k], dtype=np.float64) for k in df.columns if k.startswith(("rlnDefocus", "rlnPhaseShift", "rlnCtf"))},
                voltage_kv=float(row["rlnVoltage"]),
                cs_mm=float(row["rlnSphericalAberration"]),
                amplitude_contrast=float(row["rlnAmplitudeContrast"]),
            )

        out[name] = RelionTomogramData(
            name=name,
            voltage_kv=float(row["rlnVoltage"]),
            cs_mm=float(row["rlnSphericalAberration"]),
            amplitude_contrast=float(row["rlnAmplitudeContrast"]),
            hand=int(float(row["rlnTomoHand"])),
            pixel_size_a=float(row["rlnTomoTiltSeriesPixelSize"]),
            tomo_dims_px=(
                int(float(row["rlnTomoSizeX"])),
                int(float(row["rlnTomoSizeY"])),
                int(float(row["rlnTomoSizeZ"])),
            ),
            image_dims_px=(0, 0),  # from image headers, never the star
            xtilt_deg=col("rlnTomoXTilt", 0.0),
            ytilt_deg=col("rlnTomoYTilt"),
            zrot_deg=col("rlnTomoZRot"),
            xshift_a=col("rlnTomoXShiftAngst"),
            yshift_a=col("rlnTomoYShiftAngst"),
            pre_exposure=col("rlnMicrographPreExposure"),
            nominal_stage_angle_deg=col("rlnTomoNominalStageTiltAngle", 0.0),
            ctf=ctf,
            original_pixel_size_a=(
                float(row["rlnMicrographOriginalPixelSize"])
                if "rlnMicrographOriginalPixelSize" in g.columns
                else None
            ),
            optics_group_name=str(row.get("rlnOpticsGroupName", "opticsGroup1")),
            defocus_slope=(
                float(row["rlnTomoDefocusSlope"]) if "rlnTomoDefocusSlope" in g.columns else None
            ),
            has_deformations=(
                "rlnTomoDeformationGridSizeX" in g.columns
                and "rlnTomoDeformationGridSizeY" in g.columns
            ),
            deformation_type=(
                str(row["rlnTomoDeformationType"]) if "rlnTomoDeformationType" in g.columns else None
            ),
            deformation_grid=(
                (int(float(row["rlnTomoDeformationGridSizeX"])), int(float(row["rlnTomoDeformationGridSizeY"])))
                if "rlnTomoDeformationGridSizeX" in g.columns
                and "rlnTomoDeformationGridSizeY" in g.columns
                else None
            ),
            deformation_coeffs=(
                [
                    torch.tensor(_parse_vector(v), dtype=_F64)
                    for v in df["rlnTomoDeformationCoefficients"]
                ]
                if "rlnTomoDeformationCoefficients" in df.columns
                else None
            ),
            micrograph_names=(
                [str(x) for x in df["rlnMicrographName"]] if "rlnMicrographName" in df.columns else None
            ),
            tilt_series_name=(
                str(row["rlnTomoTiltSeriesName"]) if "rlnTomoTiltSeriesName" in g.columns else None
            ),
            frame_count=(int(float(row["rlnTomoFrameCount"])) if "rlnTomoFrameCount" in g.columns else None),
            matrices=matrices,
        )
    return out


# ---------------------------------------------------------------------------
# Particles
# ---------------------------------------------------------------------------


def particles_frames(
    *,
    tomo_name: str,
    particle_names: list,
    centered_coords_a: torch.Tensor,  # (P, 3) rlnCenteredCoordinate*Angst
    voltage_kv: float,
    cs_mm: float,
    amplitude_contrast: float,
    pixel_size_a: float,
    optics_group: int = 1,
    optics_group_name: str = "opticsGroup1",
    random_subset: list | None = None,
) -> tuple:
    """(general, optics, particles) DataFrames for one tomogram's particles;
    origins/angles zeroed (the to-RELION gauge lives in the coordinates +
    motion.star); alternating half-sets unless ``random_subset`` is given."""
    p = centered_coords_a.shape[0]
    if len(particle_names) != p:
        raise ValueError("particle_names length mismatch")
    general = pd.DataFrame({"rlnTomoSubTomosAre2DStacks": [0]})
    optics = pd.DataFrame(
        {
            "rlnOpticsGroup": [int(optics_group)],
            "rlnOpticsGroupName": [optics_group_name],
            "rlnVoltage": [voltage_kv],
            "rlnSphericalAberration": [cs_mm],
            "rlnAmplitudeContrast": [amplitude_contrast],
            "rlnTomoTiltSeriesPixelSize": [pixel_size_a],
        }
    )
    zeros = np.zeros(p)
    particles = pd.DataFrame(
        {
            "rlnTomoName": [tomo_name] * p,
            "rlnTomoParticleName": list(particle_names),
            "rlnCenteredCoordinateXAngst": centered_coords_a[:, 0].numpy(),
            "rlnCenteredCoordinateYAngst": centered_coords_a[:, 1].numpy(),
            "rlnCenteredCoordinateZAngst": centered_coords_a[:, 2].numpy(),
            "rlnOriginXAngst": zeros,
            "rlnOriginYAngst": zeros,
            "rlnOriginZAngst": zeros,
            "rlnAngleRot": zeros,
            "rlnAngleTilt": zeros,
            "rlnAnglePsi": zeros,
            "rlnOpticsGroup": np.full(p, int(optics_group), dtype=int),
            # half-sets: harmless for extraction, REQUIRED by relion_tomo_align
            # (getHalfSet, particle_set.cpp:487)
            "rlnRandomSubset": (
                np.asarray(random_subset, dtype=int) if random_subset is not None else (np.arange(p) % 2) + 1
            ),
        }
    )
    return general, optics, particles


def write_particles_star(
    path: str | Path,
    *,
    tomo_name: str,
    particle_names: list,
    centered_coords_a: torch.Tensor,  # (P, 3) rlnCenteredCoordinate*Angst
    voltage_kv: float,
    cs_mm: float,
    amplitude_contrast: float,
    pixel_size_a: float,
    optics_group_name: str = "opticsGroup1",
) -> Path:
    """particles.star with zeroed origins/angles (the to-RELION gauge lives in
    the coordinates + motion.star)."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists")
    general, optics, particles = particles_frames(
        tomo_name=tomo_name, particle_names=particle_names, centered_coords_a=centered_coords_a,
        voltage_kv=voltage_kv, cs_mm=cs_mm, amplitude_contrast=amplitude_contrast,
        pixel_size_a=pixel_size_a, optics_group_name=optics_group_name,
    )
    starfile.write({"general": general, "optics": optics, "particles": particles}, path)
    return path


@dataclass
class RelionParticlesRead:
    tomo_names: list
    particle_names: list
    centered_coords_a: torch.Tensor | None  # (P, 3) or None (legacy file)
    legacy_coords_px: torch.Tensor | None
    origins_a: torch.Tensor | None  # (P, 3)
    subtomo_angles_deg: torch.Tensor | None  # (P, 3)
    optics_group: torch.Tensor | None  # (P,) 1-based
    optics_pixel_size_a: dict  # group number -> rlnTomoTiltSeriesPixelSize
    point_attributes: dict = field(default_factory=dict)


def read_particles_star(path: str | Path) -> RelionParticlesRead:
    path = Path(path)
    blocks = starfile.read(path, always_dict=True)
    if "particles" not in blocks:
        raise ValueError(f"{path}: no data_particles block")
    df = blocks["particles"]

    def tcols(labels):
        if all(c in df.columns for c in labels):
            return torch.tensor(np.stack([np.asarray(df[c], dtype=np.float64) for c in labels], axis=1))
        return None

    centered = tcols(["rlnCenteredCoordinateXAngst", "rlnCenteredCoordinateYAngst", "rlnCenteredCoordinateZAngst"])
    legacy = tcols(["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"])
    if centered is None and legacy is None:
        raise ValueError(f"{path}: no particle coordinates (rlnCenteredCoordinate*Angst or rlnCoordinateX/Y/Z)")
    origins = tcols(["rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"])
    subtomo = tcols(["rlnTomoSubtomogramRot", "rlnTomoSubtomogramTilt", "rlnTomoSubtomogramPsi"])

    tomo_names = [str(x) for x in df["rlnTomoName"]] if "rlnTomoName" in df.columns else []
    if "rlnTomoParticleName" in df.columns:
        names = [str(x) for x in df["rlnTomoParticleName"]]
    else:
        # RELION auto-generates <TomoName>/<n> (1-based), particle_set.cpp:135-159
        names = [f"{t}/{i + 1}" for i, t in enumerate(tomo_names)] if tomo_names else [str(i + 1) for i in range(len(df))]

    optics_pix = {}
    group = None
    if "optics" in blocks:
        odf = blocks["optics"]
        if isinstance(odf, pd.Series):
            odf = odf.to_frame().T
        if "rlnTomoTiltSeriesPixelSize" in odf.columns:
            for _, orow in odf.iterrows():
                optics_pix[int(float(orow.get("rlnOpticsGroup", 1)))] = float(orow["rlnTomoTiltSeriesPixelSize"])
    if "rlnOpticsGroup" in df.columns:
        group = torch.tensor(np.asarray(df["rlnOpticsGroup"], dtype=np.int64))

    return RelionParticlesRead(
        tomo_names=tomo_names,
        particle_names=names,
        centered_coords_a=centered,
        legacy_coords_px=legacy,
        origins_a=origins,
        subtomo_angles_deg=subtomo,
        optics_group=group,
        optics_pixel_size_a=optics_pix,
        point_attributes={name: [int(v) for v in df[column]] for name, column in
            (("half_set", "rlnRandomSubset"), ("class_number", "rlnClassNumber")) if column in df},
    )


# ---------------------------------------------------------------------------
# Trajectories (motion.star) — hand-rolled writer, starfile reader
# ---------------------------------------------------------------------------


def _prepend_version(path: Path) -> None:
    path.write_text(f"{VERSION_COMMENT}\n" + path.read_text())


def motion_blocks(particle_names: list, motion_a: torch.Tensor) -> dict:
    """{particle name: DataFrame(rlnOriginX/Y/ZAngst)} — one block per particle
    with exactly frameCount rows."""
    _t, p, three = motion_a.shape
    if three != 3 or len(particle_names) != p:
        raise ValueError(f"motion_a {tuple(motion_a.shape)} vs {len(particle_names)} names")
    if not torch.isfinite(motion_a).all():
        raise ValueError("motion contains non-finite values")
    m = motion_a.to(_F64).numpy()
    return {
        str(name): pd.DataFrame(
            {"rlnOriginXAngst": m[:, j, 0], "rlnOriginYAngst": m[:, j, 1], "rlnOriginZAngst": m[:, j, 2]}
        )
        for j, name in enumerate(particle_names)
    }


def write_motion_star(
    path: str | Path,
    particle_names: list,
    motion_a: torch.Tensor,  # (T, P, 3) Angstrom, per-tomo-star row order
) -> Path:
    """Pinned native schema: data_general list block FIRST with
    rlnParticleNumber, then exactly one data_<name> block per particle with
    exactly frameCount rows of _rlnOriginX/Y/ZAngst."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists")
    blocks = {"general": {"rlnParticleNumber": len(particle_names)}, **motion_blocks(particle_names, motion_a)}
    starfile.write(blocks, path)
    _prepend_version(path)
    return path


def read_motion_star(path: str | Path, particle_names: list, frame_count: int) -> torch.Tensor:
    """-> (T, P, 3) Angstrom in the given particle order; every particle must
    have exactly one block with frame_count rows (checkTrajectoryLengths)."""
    blocks = starfile.read(Path(path), always_dict=True)
    if "general" not in blocks:
        raise ValueError(f"{path}: no data_general block")
    gen = blocks["general"]
    if isinstance(gen, (dict, pd.Series)):
        n = int(gen["rlnParticleNumber"])
    else:
        n = int(gen["rlnParticleNumber"].iloc[0])
    if n < len(particle_names):
        raise ValueError(f"{path}: {n} trajectories for {len(particle_names)} particles")
    out = torch.empty(frame_count, len(particle_names), 3, dtype=_F64)
    for j, name in enumerate(particle_names):
        if name not in blocks:
            raise ValueError(f"{path}: no trajectory block data_{name}")
        df = blocks[name]
        if len(df) != frame_count:
            raise ValueError(f"{path}: data_{name} has {len(df)} rows, expected {frame_count}")
        for k, c in enumerate(["rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"]):
            out[:, j, k] = torch.tensor(np.asarray(df[c], dtype=np.float64))
    return out


# ---------------------------------------------------------------------------
# Optimisation set — unnamed list block
# ---------------------------------------------------------------------------


def write_optimisation_set(
    path: str | Path,
    *,
    particles: str | None,
    tomograms: str,
    trajectories: str | None = None,
    overwrite: bool = False,
) -> Path:
    """``particles`` may be None for a tomogram-only set
    (relion_tomo_reconstruct_tomogram needs only --t)."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists")
    values = {}
    if particles is not None:
        values["rlnTomoParticlesFile"] = particles
    values["rlnTomoTomogramsFile"] = tomograms
    if trajectories is not None:
        values["rlnTomoTrajectoriesFile"] = trajectories
    starfile.write({"": values}, path, overwrite=True)  # empty key -> RELION's unnamed data_ block
    _prepend_version(path)
    return path


# ---------------------------------------------------------------------------
# Generic table IO + path resolution (project assembly)
# ---------------------------------------------------------------------------


def read_star_tables(path: str | Path) -> dict:
    """{block: DataFrame | dict}; one-row list blocks come back as dicts."""
    blocks = starfile.read(Path(path), always_dict=True)
    out = {}
    for k, v in blocks.items():
        if isinstance(v, pd.Series):
            v = v.to_dict()
        out[k] = v
    return out


def write_star_tables(path: str | Path, blocks: dict, *, version: bool = True, overwrite: bool = False) -> Path:
    """starfile.write + the ``# version 50001`` comment RELION emits on every file."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    starfile.write(blocks, path, overwrite=True)
    if version:
        _prepend_version(path)
    return path


def resolve_star_ref(ref: str | Path, star_path: str | Path, project_root: str | Path | None = None) -> Path:
    """RELION resolves star-internal paths verbatim from the CWD; we try, in
    order: absolute, relative to the referencing star's directory, relative to
    ``project_root`` (default: the parent of the star's directory when the
    reference starts with that directory's name, i.e. the ``input/tilt_series/X``
    layout), then the CWD."""
    ref = Path(ref)
    star_path = Path(star_path)
    if ref.is_absolute():
        return ref
    candidates = [star_path.parent / ref]
    if project_root is not None:
        candidates.append(Path(project_root) / ref)
    elif ref.parts and ref.parts[0] == star_path.parent.name:
        candidates.append(star_path.parent.parent / ref)
    candidates.append(Path.cwd() / ref)
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def read_optimisation_set(path: str | Path) -> dict:
    blocks = starfile.read(Path(path), always_dict=True)
    first = next(iter(blocks.values()))
    row = first if isinstance(first, (dict, pd.Series)) else first.iloc[0]
    return {
        "particles": str(row["rlnTomoParticlesFile"]) if "rlnTomoParticlesFile" in row else None,
        "tomograms": str(row["rlnTomoTomogramsFile"]) if "rlnTomoTomogramsFile" in row else None,
        "trajectories": str(row["rlnTomoTrajectoriesFile"]) if "rlnTomoTrajectoriesFile" in row else None,
    }
